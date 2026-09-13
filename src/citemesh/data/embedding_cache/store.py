"""The :class:`EmbeddingCache` facade and its public read/write API.

Owns namespace construction, the locked-connection and cache-metadata
helpers, and the public surface used by the strategy layer: embedding lookup and
upsert, search entry point, hydration markers, calibration-range persistence,
payload statistics, and namespace clearing. The ingestion pipeline behind
:meth:`EmbeddingCache.get_embeddings`, along with schema/layout,
locking/recovery, and scoring, lives in sibling modules and is mixed in here.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
)

import h5py
import numpy as np
from filelock import FileLock

from citemesh.core.text_batching import (
    l2_normalize_embeddings,
)

from ..cache import (
    format_bytes,
    get_cache_dir,
    path_exists,
)
from ..model_profiles import (
    DEFAULT_EMBEDDING_MODEL_NAME,
)
from .constants import (
    _STORAGE_PRECISIONS,
    BINARY_INDEX_DATASET_NAME,
    CALIBRATION_RANGES_DATASET_NAME,
    CORPUS_METADATA_VERSION,
    CORPUS_METADATA_VERSION_KEY,
    EMBEDDINGS_DATASET_NAME,
    HYDRATION_COMPLETE_KEY,
    HYDRATION_CORPUS_SIZE_KEY,
    HYDRATION_DATASET_SOURCE_KEY,
    HYDRATION_RECONCILED_CACHE_ROWS_KEY,
    HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY,
    HYDRATION_SPLIT_KEY,
    INT8_CLIPPED_VALUE_COUNT_KEY,
    INT8_SATURATION_WARN_RATIO,
    INT8_TOTAL_VALUE_COUNT_KEY,
    MODEL_FINGERPRINT_KEY,
    SQLITE_QUERY_BATCH_SIZE,
    _corpus_size_token,
)
from .ingest import _IngestMixin
from .layout import _H5LayoutMixin
from .models import (
    CacheNamespacePayloadStats,
    CacheSearchResult,
)
from .quantization import (
    _sanitize_ranges,
    validate_compression_filter,
)
from .recovery import _RecoveryMixin
from .search import _SearchMixin

logger = logging.getLogger(__name__)

#: Progress reporter injected by the CLI layer: ``progress(values, description)``
#: returns an iterable over ``values``, optionally rendering a bar as it is consumed.
ProgressWrapper = Callable[[Sequence[Any], str], Iterable[Any]]


class EmbeddingCache(_IngestMixin, _H5LayoutMixin, _RecoveryMixin, _SearchMixin):
    """Persistent cache for paper embeddings and metadata."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        storage_precision: str = "int8",
        binary_prefilter: bool = True,
        calibration_sample_size: int = 2000,
        compression: str = "gzip",
        compression_level: int = 1,
        source_torch_dtype: str = "float32",
        text_formatter_fingerprint: str = "default",
        progress: ProgressWrapper | None = None,
    ):
        """Create a persistent embedding cache for a model variant.

        :param Optional[Path] cache_dir: Cache directory override. Uses global cache when ``None``.
        :param str model_name: Model namespace string used for cache partitioning.
        :param str storage_precision: Persistent embedding precision ``float32``/``int8``.
        :param bool binary_prefilter: Whether to maintain a binary index for int8 search.
        :param int calibration_sample_size: Target sample size for int8 calibration ranges.
        :param str compression: HDF5 compression filter name (``gzip`` or ``lzf``).
        :param int compression_level: Compression level for HDF5 datasets.
        :param str source_torch_dtype: Source inference dtype token, e.g. ``bfloat16``.
        :param str text_formatter_fingerprint: Deterministic metadata-to-text formatter
            fingerprint used for cache invalidation boundaries.
        :param Optional[ProgressWrapper] progress: Reporter wrapping long cache scans as
            ``progress(values, description)``. Storage owns no display policy, so this
            defaults to no reporting and the CLI layer injects its own renderer.
        """
        if storage_precision not in _STORAGE_PRECISIONS:
            expected = ", ".join(sorted(_STORAGE_PRECISIONS))
            raise ValueError(
                f"storage_precision must be one of {{{expected}}}, "
                f"got {storage_precision!r}."
            )
        if calibration_sample_size < 1:
            raise ValueError("calibration_sample_size must be at least 1")

        configured_cache_root = get_cache_dir(create=False)
        if cache_dir is None:
            cache_dir = configured_cache_root / "embeddings"

        self.cache_dir = Path(cache_dir)
        self._managed_cache_root = self._resolve_managed_cache_root(
            self.cache_dir, configured_cache_root
        )

        model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
        self.db_path = self.cache_dir / f"metadata_{model_hash}.db"
        self.h5_path = self.cache_dir / f"embeddings_{model_hash}.h5"
        self.lock_path = self.cache_dir / f"cache_{model_hash}.lock"
        self.hydration_lock_path = self.cache_dir / f"hydration_{model_hash}.lock"
        self._hydration_operation_file_lock = FileLock(str(self.hydration_lock_path))

        self.model_name = model_name
        self.storage_precision = storage_precision
        self.binary_prefilter = bool(binary_prefilter and storage_precision == "int8")
        self.calibration_sample_size = int(calibration_sample_size)
        self.compression = validate_compression_filter(compression)
        self.compression_level = int(compression_level)
        if self.compression_level < 0:
            raise ValueError("compression_level must be non-negative.")
        if self.compression == "lzf" and self.compression_level != 0:
            logger.debug(
                "Ignoring compression_level=%d for compression='lzf'; using 0.",
                self.compression_level,
            )
            self.compression_level = 0
        self._effective_compression = self.compression
        self._effective_compression_level = self.compression_level
        self.source_torch_dtype = str(source_torch_dtype or "float32")
        self.text_formatter_fingerprint = str(text_formatter_fingerprint).strip()
        if not self.text_formatter_fingerprint:
            raise ValueError("text_formatter_fingerprint must be a non-empty string.")
        self.embedding_vector_dtype = "float32"
        self.last_search_used_binary_prefilter: bool | None = None
        self.last_search_total_embeddings: int | None = None
        self.last_search_rescored_embeddings: int | None = None
        self._int8_saturation_warning_emitted = False
        self._progress = progress

        with self._cache_operation_lock():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with self._cache_lock():
                self._init_db()
                self._ensure_h5_layout()

    # ------------------------------------------------------------------
    # Public API

    def get_embeddings(
        self,
        papers: dict[str, dict],
        model: Any,
        batch_size: int = 32,
        show_progress: bool = True,
        text_builder: Callable[[dict[str, object]], str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return embeddings for provided papers, computing only missing ones.

        :param Dict[str, Dict] papers: Mapping of paper ID to metadata payload.
        :param Any model: SentenceTransformer-compatible model exposing ``encode``.
        :param int batch_size: Batch size for model encoding.
        :param bool show_progress: Whether to display progress bars.
        :param Optional[Callable[[Dict[str, object]], str]] text_builder: Optional metadata->text formatter.
        :return Dict[str, np.ndarray]: Mapping of paper IDs to float32 embeddings.
        """
        result = self._process_embeddings(
            papers,
            model,
            batch_size=batch_size,
            show_progress=show_progress,
            text_builder=text_builder,
            return_embeddings=True,
        )
        assert isinstance(result, dict)
        return result

    def upsert_embeddings(
        self,
        papers: dict[str, dict],
        model: Any,
        batch_size: int = 32,
        show_progress: bool = False,
        text_builder: Callable[[dict[str, object]], str] | None = None,
    ) -> None:
        """Persist embeddings for papers without materializing float32 return payloads.

        :param Dict[str, Dict] papers: Mapping of paper ID to metadata payload.
        :param Any model: SentenceTransformer-compatible model exposing ``encode``.
        :param int batch_size: Batch size for model encoding.
        :param bool show_progress: Whether to display progress bars.
        :param Optional[Callable[[Dict[str, object]], str]] text_builder: Optional metadata->text formatter.
        :return None: Persists missing embeddings without materializing them.
        """
        self._process_embeddings(
            papers,
            model,
            batch_size=batch_size,
            show_progress=show_progress,
            text_builder=text_builder,
            return_embeddings=False,
        )

    def embedding_count(self) -> int:
        """Return the number of embeddings persisted in this cache namespace.

        Cheap availability probe (no model load) used to decide whether local
        semantic search has anything to rank.

        :return int: Persisted embedding row count (``0`` for a missing or
            empty cache).
        """
        with self._cache_lock(), self._connect_db() as conn:
            self._recover_pending_replacements_with_connection_locked(conn)
            if not path_exists(self.h5_path):
                return 0
            with h5py.File(self.h5_path, "r") as h5:
                dataset = self._get_embeddings_dataset(h5)
                return int(dataset.shape[0]) if dataset is not None else 0

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        *,
        binary_prefilter: bool,
        binary_rescore_multiplier: int,
    ) -> list[CacheSearchResult]:
        """Search cached embeddings and return ranked metadata-rich candidates.

        :param np.ndarray query_embedding: Query embedding in float32.
        :param int top_k: Number of top documents to return.
        :param bool binary_prefilter: Whether to use binary Hamming prefiltering.
        :param int binary_rescore_multiplier: Candidate oversampling factor for rescoring.
        :return List[CacheSearchResult]: Ranked search results.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self.last_search_used_binary_prefilter = None
        self.last_search_total_embeddings = None
        self.last_search_rescored_embeddings = None
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query_embedding must be 1-dimensional")
        if not np.all(np.isfinite(query)):
            raise ValueError("query_embedding must contain only finite values")
        query = l2_normalize_embeddings(query)
        with (
            self._cache_lock(),
            self._connect_db() as conn,
        ):
            self._recover_pending_replacements_with_connection_locked(conn)
            if not path_exists(self.h5_path):
                return []
            with h5py.File(self.h5_path, "r") as h5:
                embeddings_dataset = self._get_embeddings_dataset(h5)
                if embeddings_dataset is None or embeddings_dataset.shape[0] == 0:
                    self.last_search_total_embeddings = 0
                    self.last_search_rescored_embeddings = 0
                    return []
                self._assert_runtime_cache_consistency(
                    conn=conn,
                    h5_file=h5,
                    embeddings_dataset=embeddings_dataset,
                    fail_mode="runtime",
                )
                embedding_rows = int(embeddings_dataset.shape[0])
                self.last_search_total_embeddings = embedding_rows
                if int(embeddings_dataset.shape[1]) != int(query.shape[0]):
                    raise ValueError(
                        "Query embedding dimension mismatch: "
                        f"{query.shape[0]} != {int(embeddings_dataset.shape[1])}."
                    )

                effective_multiplier = max(int(binary_rescore_multiplier), 1)
                candidate_count = max(top_k * effective_multiplier, top_k)

                use_binary_prefilter = bool(
                    binary_prefilter
                    and self.binary_prefilter
                    and self.storage_precision == "int8"
                    and BINARY_INDEX_DATASET_NAME in h5
                )

                if self.storage_precision == "int8":
                    if use_binary_prefilter:
                        binary_dataset = h5[BINARY_INDEX_DATASET_NAME]
                        if not self._is_binary_dataset_compatible(
                            binary_dataset=binary_dataset,
                            embedding_dim=int(embeddings_dataset.shape[1]),
                            embedding_rows=int(embeddings_dataset.shape[0]),
                        ):
                            logger.warning(
                                "Binary index dataset is incompatible with embedding matrix "
                                "for cache %s; falling back to direct int8 scoring.",
                                self.h5_path,
                            )
                            use_binary_prefilter = False
                    self.last_search_used_binary_prefilter = bool(use_binary_prefilter)
                    if use_binary_prefilter:
                        candidate_rows = self._binary_prefilter_rows(
                            binary_dataset=binary_dataset,
                            query_embedding=query,
                            candidate_count=candidate_count,
                        )
                        self.last_search_rescored_embeddings = int(candidate_rows.size)
                        if candidate_rows.size == 0:
                            return []
                        rows, scores, embeddings = self._score_int8_rows(
                            embeddings_dataset=embeddings_dataset,
                            h5_file=h5,
                            query_embedding=query,
                            top_k=top_k,
                            row_indices=candidate_rows,
                        )
                    else:
                        self.last_search_rescored_embeddings = embedding_rows
                        rows, scores, embeddings = self._score_int8_rows(
                            embeddings_dataset=embeddings_dataset,
                            h5_file=h5,
                            query_embedding=query,
                            top_k=top_k,
                            row_indices=None,
                        )
                else:
                    self.last_search_rescored_embeddings = embedding_rows
                    rows, scores, embeddings = self._score_float_rows(
                        embeddings_dataset=embeddings_dataset,
                        query_embedding=query,
                        top_k=top_k,
                    )
                    self.last_search_used_binary_prefilter = False

                if rows.size == 0:
                    return []

                row_values = [int(row_idx) for row_idx in rows.tolist()]
                metadata_by_row = self._load_metadata_by_rows(conn, row_values)
                missing_rows = [
                    row_idx for row_idx in row_values if row_idx not in metadata_by_row
                ]
                if missing_rows:
                    sampled_rows = ", ".join(str(value) for value in missing_rows[:10])
                    raise RuntimeError(
                        "Embedding cache integrity error: missing metadata rows for "
                        f"{len(missing_rows)} scored embeddings (row_idx={sampled_rows}). "
                        "Rebuild this cache namespace to restore row mapping consistency."
                    )
                results: list[CacheSearchResult] = []
                for idx, row_idx in enumerate(row_values):
                    payload = metadata_by_row[row_idx]
                    result_metadata = {
                        "title": payload.get("title", "Unknown"),
                        "abstract": payload.get("abstract", ""),
                        "year": payload.get("year"),
                        "authors": payload.get("authors", []),
                        "categories": payload.get("categories", []),
                        "venue": payload.get("venue", ""),
                        "arxiv_id": payload.get("arxiv_id", ""),
                        "doi": payload.get("doi", ""),
                    }
                    results.append(
                        CacheSearchResult(
                            paper_id=str(payload["paper_id"]),
                            score=float(scores[idx]),
                            embedding=np.asarray(embeddings[idx], dtype=np.float32),
                            metadata=result_metadata,
                        )
                    )

                results.sort(key=lambda item: (-item.score, str(item.paper_id)))
                return results[:top_k]

    def has_cached_payload(self) -> bool:
        """Return whether namespace contains any cached embedding payload rows.

        :return bool: ``True`` when cache has at least one SQLite/HDF5 embedding row.
        :raises RuntimeError: If storage inspection or pending recovery fails.
        """
        try:
            if not path_exists(self.db_path):
                return False
            with self._cache_lock(), self._connect_db() as conn:
                self._recover_pending_replacements_with_connection_locked(conn)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                paper_rows = int(cursor.fetchone()[0])
                if paper_rows > 0:
                    return True

                if not path_exists(self.h5_path):
                    return False
                with h5py.File(self.h5_path, "r") as h5:
                    dataset = self._get_embeddings_dataset(h5)
                    if dataset is None:
                        return False
                    return int(dataset.shape[0]) > 0
        except (OSError, sqlite3.DatabaseError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                "Failed to inspect cache payload presence "
                f"at {self.db_path} and {self.h5_path}; "
                f"existing cache files were preserved: {exc}"
            ) from exc

    def get_cached_paper_ids(self) -> set[str]:
        """Return all cached paper IDs for this namespace.

        :return Set[str]: Cached paper IDs loaded from SQLite metadata rows.
        """
        if not path_exists(self.db_path):
            return set()

        paper_ids: set[str] = set()
        with self._cache_lock(), self._connect_db() as conn:
            self._recover_pending_replacements_with_connection_locked(conn)
            cursor = conn.cursor()
            cursor.execute("SELECT paper_id FROM papers")
            while True:
                rows = cursor.fetchmany(SQLITE_QUERY_BATCH_SIZE)
                if not rows:
                    break
                for row in rows:
                    raw_paper_id = row[0]
                    if raw_paper_id is None:
                        continue
                    paper_id = str(raw_paper_id).strip()
                    if paper_id:
                        paper_ids.add(paper_id)
        return paper_ids

    def has_calibration_ranges(self) -> bool:
        """Return whether int8 calibration ranges exist in cache.

        :return bool: ``True`` when the calibration range dataset exists.
        """
        if self.storage_precision != "int8":
            return False
        with self._cache_lock(), self._connect_db() as conn:
            self._recover_pending_replacements_with_connection_locked(conn)
            if not path_exists(self.h5_path):
                return False
            with h5py.File(self.h5_path, "r") as h5:
                return CALIBRATION_RANGES_DATASET_NAME in h5

    def set_calibration_ranges(self, ranges: np.ndarray, embedding_dim: int) -> None:
        """Persist int8 calibration ranges for future quantization.

        :param np.ndarray ranges: Range matrix with shape ``(2, embedding_dim)``.
        :param int embedding_dim: Embedding dimension used for validation.
        :return None: Mutates HDF5 state in-place.
        :raises ValueError: If changing ranges would reinterpret existing int8 rows.
        """
        if self.storage_precision != "int8":
            return

        sanitized = _sanitize_ranges(np.asarray(ranges, dtype=np.float32))
        if int(sanitized.shape[1]) != int(embedding_dim):
            raise ValueError(
                "Calibration range dimension mismatch: "
                f"{int(sanitized.shape[1])} != {int(embedding_dim)}."
            )

        with self._cache_lock(), h5py.File(self.h5_path, "a") as h5:
            self._set_h5_attrs(h5)
            dataset = h5.get(CALIBRATION_RANGES_DATASET_NAME)
            if dataset is None:
                h5.create_dataset(
                    CALIBRATION_RANGES_DATASET_NAME,
                    data=sanitized,
                    dtype=np.float32,
                )
            else:
                if dataset.shape != sanitized.shape:
                    raise ValueError(
                        "Existing calibration range shape mismatch: "
                        f"{dataset.shape} != {sanitized.shape}."
                    )
                embeddings = h5.get(EMBEDDINGS_DATASET_NAME)
                if (
                    embeddings is not None
                    and embeddings.shape[0] > 0
                    and not np.array_equal(dataset[:], sanitized)
                ):
                    raise ValueError(
                        "Cannot change int8 calibration ranges after storing embeddings. "
                        "Rebuild the cache namespace to re-encode with new ranges."
                    )
                dataset[...] = sanitized

    def _record_int8_saturation(
        self,
        h5_file: h5py.File,
        *,
        clipped_value_count: int,
        total_value_count: int,
    ) -> None:
        """Persist cumulative int8 clipping stats and warn on notable saturation.

        :param h5py.File h5_file: Open HDF5 cache handle.
        :param int clipped_value_count: Number of values clipped for the current write.
        :param int total_value_count: Total quantized values for the current write.
        :return None: Updates HDF5 attributes and emits best-effort warnings.
        """
        if total_value_count < 1:
            return

        previous_clipped = int(h5_file.attrs.get(INT8_CLIPPED_VALUE_COUNT_KEY, 0))
        previous_total = int(h5_file.attrs.get(INT8_TOTAL_VALUE_COUNT_KEY, 0))
        saturation_ratio = float(clipped_value_count) / float(total_value_count)
        cumulative_clipped = previous_clipped + int(clipped_value_count)
        cumulative_total = previous_total + int(total_value_count)
        h5_file.attrs.modify(INT8_CLIPPED_VALUE_COUNT_KEY, cumulative_clipped)
        h5_file.attrs.modify(INT8_TOTAL_VALUE_COUNT_KEY, cumulative_total)
        cumulative_ratio = float(cumulative_clipped) / float(cumulative_total)
        if (
            saturation_ratio >= INT8_SATURATION_WARN_RATIO
            and not self._int8_saturation_warning_emitted
        ):
            logger.warning(
                "Int8 calibration saturation detected in %s: %.2f%% of values were clipped in this write (cumulative %.2f%% across cached writes). This measures clipped coordinates, not retrieval recall. Resume keeps the saved ranges; changing them requires re-encoding this namespace with --force-rebuild-cache. Further warnings are suppressed for this run.",
                self.h5_path.name,
                saturation_ratio * 100.0,
                cumulative_ratio * 100.0,
            )
            self._int8_saturation_warning_emitted = True

    def is_hydrated(
        self,
        dataset_split: str,
        corpus_size: int | None,
        dataset_source: str | None = None,
    ) -> bool:
        """Return whether cache hydration metadata matches target corpus spec.

        :param str dataset_split: Dataset split token.
        :param Optional[int] corpus_size: Corpus cap or ``None`` for full split.
        :param Optional[str] dataset_source: Expected dataset source token.
        :return bool: ``True`` when hydration metadata matches and HDF5 payload is queryable.
        :raises RuntimeError: If storage inspection fails; existing files are preserved.
        """
        expected_split = str(dataset_split)
        expected_corpus_size = _corpus_size_token(corpus_size)
        expected_source = None if dataset_source is None else str(dataset_source)

        try:
            if not path_exists(self.db_path):
                return False
            with self._cache_lock(), self._connect_db() as conn:
                self._recover_pending_replacements_with_connection_locked(conn)
                if not path_exists(self.h5_path):
                    return False
                with h5py.File(self.h5_path, "r") as h5:
                    metadata = self._load_cache_metadata(conn)
                    metadata_matches = (
                        metadata.get(HYDRATION_COMPLETE_KEY, "0") == "1"
                        and metadata.get(HYDRATION_SPLIT_KEY) == expected_split
                        and metadata.get(HYDRATION_CORPUS_SIZE_KEY)
                        == expected_corpus_size
                    )
                    if not metadata_matches:
                        return False

                    cached_source = (
                        metadata.get(HYDRATION_DATASET_SOURCE_KEY) or ""
                    ).strip()
                    if not cached_source:
                        return False

                    if expected_source is not None and cached_source != expected_source:
                        return False

                    return self._has_queryable_hydrated_payload(conn=conn, h5=h5)
        except (OSError, ValueError, RuntimeError, sqlite3.DatabaseError) as exc:
            raise RuntimeError(
                "Failed to inspect cache hydration state "
                f"at {self.db_path} and {self.h5_path}; "
                f"existing cache files were preserved: {exc}"
            ) from exc

    def _has_queryable_hydrated_payload(
        self, conn: sqlite3.Connection, h5: h5py.File
    ) -> bool:
        """Return whether hydrated HDF5 payload exists and can be queried safely.

        :param sqlite3.Connection conn: Open SQLite connection.
        :param h5py.File h5: Open HDF5 cache handle.
        :return bool: ``True`` when cache has readable embedding + metadata rows.
        """
        embeddings_dataset = self._get_embeddings_dataset(h5)
        if embeddings_dataset is None:
            return False

        self._assert_runtime_cache_consistency(
            conn=conn,
            h5_file=h5,
            embeddings_dataset=embeddings_dataset,
            fail_mode="runtime",
        )
        row_count = int(embeddings_dataset.shape[0])
        if row_count < 1:
            return False

        if (
            self.storage_precision == "int8"
            and CALIBRATION_RANGES_DATASET_NAME not in h5
        ):
            return False
        return True

    def _report_progress(
        self,
        values: Sequence[Any],
        description: str,
        *,
        enabled: bool,
    ) -> Iterable[Any]:
        """Wrap a cache scan in the injected progress reporter.

        :param Sequence[Any] values: Items the cache is about to iterate.
        :param str description: Label handed to the reporter.
        :param bool enabled: Whether this scan is long enough to be worth reporting.
        :return Iterable[Any]: Wrapped iterable, or ``values`` when not reporting.
        """
        if not enabled or self._progress is None:
            return values
        return self._progress(values, description)

    @contextmanager
    def _locked_connection(self) -> Iterator[sqlite3.Connection]:
        """Hold the namespace lock around one short SQLite transaction.

        Every small metadata read/write takes exactly this pairing, so the two
        context managers are acquired in one place instead of at each call site.

        :return Iterator[sqlite3.Connection]: Open connection for the active namespace.
        """
        with self._cache_lock(), self._connect_db() as conn:
            yield conn

    def _read_cache_metadata(self) -> dict[str, str]:
        """Load the persisted metadata table under the namespace lock.

        :return Dict[str, str]: Metadata key/value pairs for the active namespace.
        """
        with self._locked_connection() as conn:
            return self._load_cache_metadata(conn)

    def _write_cache_metadata(self, values: dict[str, object]) -> None:
        """Persist metadata key/value pairs under the namespace lock.

        :param Dict[str, object] values: Metadata keys to upsert.
        :return None: Mutates SQLite metadata in-place.
        """
        with self._locked_connection() as conn:
            self._set_cache_metadata(conn, values)

    def get_hydrated_dataset_source(self) -> str | None:
        """Return dataset source captured for the latest hydrated cache attempt.

        :return Optional[str]: Hydrated dataset source token when set.
        """
        cached_source = self._read_cache_metadata().get(HYDRATION_DATASET_SOURCE_KEY)
        return cached_source if cached_source else None

    def has_current_corpus_metadata(self) -> bool:
        """Return whether corpus years and DOIs use the current source adapter.

        :return bool: Whether a full metadata pass completed with this adapter.
        """
        metadata = self._read_cache_metadata()
        return metadata.get(CORPUS_METADATA_VERSION_KEY) == CORPUS_METADATA_VERSION

    def mark_corpus_metadata_current(self) -> None:
        """Record successful corpus metadata hydration or backfill.

        :return None: Persists the completed metadata adapter version.
        """
        self._write_cache_metadata(
            {CORPUS_METADATA_VERSION_KEY: CORPUS_METADATA_VERSION}
        )

    def update_corpus_metadata(self, papers: Sequence[dict]) -> None:
        """Correct years and DOIs on existing corpus rows without touching vectors.

        :param Sequence[Dict] papers: Source metadata with paper IDs, years and DOIs.
        :return None: Updates only matching SQLite records.
        """
        with self._locked_connection() as conn:
            conn.executemany(
                "UPDATE papers SET year = ?, doi = ? WHERE paper_id = ?",
                [(paper["year"], paper["doi"], paper["paper_id"]) for paper in papers],
            )

    def get_model_fingerprint(self) -> str | None:
        """Return model fingerprint captured for this cache namespace.

        :return Optional[str]: Active model fingerprint or ``None`` when unset.
        """
        metadata = self._read_cache_metadata()
        fingerprint = str(metadata.get(MODEL_FINGERPRINT_KEY, "")).strip()
        return fingerprint or None

    def get_hydration_rowcount_reconciliation(self) -> tuple[int, int] | None:
        """Return persisted full-split reconciliation marker for row-count deltas.

        :return Optional[Tuple[int, int]]: ``(upstream_rows, cached_rows)`` when
            a prior full-split reconciliation confirmed no uncached paper IDs for
            that row-count state; ``None`` when unset/invalid.
        """
        metadata = self._read_cache_metadata()
        raw_upstream = str(
            metadata.get(HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY, "")
        ).strip()
        raw_cached = str(metadata.get(HYDRATION_RECONCILED_CACHE_ROWS_KEY, "")).strip()
        if not raw_upstream or not raw_cached:
            return None
        try:
            upstream_rows = int(raw_upstream)
            cached_rows = int(raw_cached)
        except ValueError:
            return None
        if upstream_rows < 1 or cached_rows < 0:
            return None
        return upstream_rows, cached_rows

    def set_hydration_rowcount_reconciliation(
        self, *, upstream_rows: int, cached_rows: int
    ) -> None:
        """Persist a reconciliation marker for row-count deltas.

        :param int upstream_rows: Upstream split row count observed during reconciliation.
        :param int cached_rows: Local cached row count after reconciliation.
        :return None: Mutates SQLite metadata in-place.
        """
        resolved_upstream = int(upstream_rows)
        resolved_cached = int(cached_rows)
        if resolved_upstream < 1:
            raise ValueError("upstream_rows must be at least 1")
        if resolved_cached < 0:
            raise ValueError("cached_rows must be non-negative")
        self._write_cache_metadata(
            {
                HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY: resolved_upstream,
                HYDRATION_RECONCILED_CACHE_ROWS_KEY: resolved_cached,
            }
        )

    def clear_hydration_rowcount_reconciliation(self) -> None:
        """Clear persisted row-count reconciliation marker metadata.

        :return None: Mutates SQLite metadata in-place.
        """
        self._write_cache_metadata(
            {
                HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY: "",
                HYDRATION_RECONCILED_CACHE_ROWS_KEY: "",
            }
        )

    def payload_stats(self) -> CacheNamespacePayloadStats:
        """Inspect cache payload for hydration and resume decisions.

        :return CacheNamespacePayloadStats: File/row/hydration stats snapshot.
        :raises RuntimeError: If inspection fails; existing files are preserved.
        """
        try:
            with self._cache_lock():
                return self._collect_namespace_payload_stats_locked(strict=True)
        except (OSError, ValueError, RuntimeError, sqlite3.DatabaseError) as exc:
            raise RuntimeError(
                "Failed to inspect cache payload statistics "
                f"at {self.db_path} and {self.h5_path}; "
                f"existing cache files were preserved: {exc}"
            ) from exc

    def set_model_fingerprint(self, fingerprint: str) -> None:
        """Persist model fingerprint for cache invalidation guardrails.

        :param str fingerprint: Deterministic model fingerprint token.
        :return None: Mutates SQLite metadata in-place.
        """
        normalized = str(fingerprint).strip()
        if not normalized:
            raise ValueError("fingerprint must be a non-empty string.")
        self._write_cache_metadata({MODEL_FINGERPRINT_KEY: normalized})

    def mark_hydrated(
        self,
        dataset_source: str,
        dataset_split: str,
        corpus_size: int | None,
        *,
        complete: bool,
    ) -> None:
        """Persist hydration metadata for cache-native retrieval routing.

        :param str dataset_source: Dataset identifier used for hydration.
        :param str dataset_split: Dataset split used for hydration.
        :param Optional[int] corpus_size: Corpus-size cap used for hydration.
        :param bool complete: Whether hydration completed successfully.
        :return None: Mutates SQLite metadata in-place.
        """
        normalized_source = str(dataset_source).strip()
        if complete and not normalized_source:
            raise ValueError(
                "dataset_source must be non-empty when complete=True for hydration."
            )
        self._write_cache_metadata(
            {
                HYDRATION_DATASET_SOURCE_KEY: normalized_source,
                HYDRATION_SPLIT_KEY: dataset_split,
                HYDRATION_CORPUS_SIZE_KEY: _corpus_size_token(corpus_size),
                HYDRATION_COMPLETE_KEY: "1" if complete else "0",
                HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY: "",
                HYDRATION_RECONCILED_CACHE_ROWS_KEY: "",
            }
        )

    def clear(self, reason: str | None = None) -> None:
        """Purge cache artifacts for this cache namespace.

        :param Optional[str] reason: Optional rationale for the clear operation.
        :return None: Removes namespace payload files and reinitializes metadata.
        """
        normalized_reason = str(reason or "").strip() or "unspecified"
        with self._cache_lock():
            stats = self._collect_namespace_payload_stats_locked()
            cached_rows = max(stats.sqlite_rows, stats.embedding_rows)
            if cached_rows > 0:
                logger.warning(
                    "Clearing embedding cache with %d cached paper(s) (%s); "
                    "see debug logs for namespace and reason.",
                    cached_rows,
                    format_bytes(stats.size_bytes),
                )
            if stats.file_count > 0 or cached_rows > 0:
                logger.debug(
                    "Embedding cache clear details: namespace=%s, reason=%s, "
                    "files=%d, size=%s, sqlite_rows=%d, embedding_rows=%d, "
                    "hydrated=%s, cached_split=%s, cached_corpus=%s, cached_source=%s.",
                    self.model_name,
                    normalized_reason,
                    stats.file_count,
                    format_bytes(stats.size_bytes),
                    stats.sqlite_rows,
                    stats.embedding_rows,
                    "yes" if stats.hydration_complete else "no",
                    stats.hydration_split or "unknown",
                    stats.hydration_corpus_size or "unknown",
                    stats.hydration_dataset_source or "unknown",
                )
            else:
                logger.debug(
                    "Embedding cache namespace '%s' is already empty (reason=%s).",
                    self.model_name,
                    normalized_reason,
                )
            self.db_path.unlink(missing_ok=True)
            self.h5_path.unlink(missing_ok=True)
            self._reset_effective_compression()
            self._init_db()
