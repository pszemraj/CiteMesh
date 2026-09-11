"""The :class:`EmbeddingCache` facade and its public read/write API.

Owns namespace construction, the cache-miss/encode/commit pipeline
(:meth:`EmbeddingCache._process_embeddings_locked`), and the public surface used
by the strategy layer: embedding lookup and upsert, search entry point,
hydration markers, calibration-range persistence, payload statistics, and
namespace clearing. Schema/layout, locking/recovery, and scoring live in
sibling modules and are mixed in here.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import h5py
import numpy as np
from filelock import FileLock

from citemesh._runtime import stderr_isatty
from citemesh.progress import progress_iterator
from citemesh.text_batching import (
    encode_texts,
    l2_normalize_embeddings,
)

from ..cache import (
    format_bytes,
    get_cache_dir,
    path_exists,
)
from ..model_profiles import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    compose_title_abstract_text,
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
from .layout import _H5LayoutMixin
from .models import (
    CacheNamespacePayloadStats,
    CacheSearchResult,
    PendingEmbeddingRecord,
)
from .quantization import (
    _count_int8_saturated_values,
    _quantize_int8_embeddings,
    _quantize_ubinary_embeddings,
    _sanitize_ranges,
    _storage_dtype_for_precision,
    validate_compression_filter,
)
from .recovery import _RecoveryMixin
from .search import _SearchMixin

logger = logging.getLogger(__name__)


class EmbeddingCache(_H5LayoutMixin, _RecoveryMixin, _SearchMixin):
    """Persistent cache for paper embeddings and metadata."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        storage_precision: str = "int8",
        binary_prefilter: bool = True,
        calibration_sample_size: int = 2000,
        compression: str = "gzip",
        compression_level: int = 1,
        source_torch_dtype: str = "float32",
        text_formatter_fingerprint: str = "default",
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
        self.last_search_used_binary_prefilter: Optional[bool] = None
        self.last_search_total_embeddings: Optional[int] = None
        self.last_search_rescored_embeddings: Optional[int] = None
        self._int8_saturation_warning_emitted = False

        with self._cache_operation_lock():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with self._cache_lock():
                self._init_db()
                self._ensure_h5_layout()

    # ------------------------------------------------------------------
    # Public API

    def get_embeddings(
        self,
        papers: Dict[str, Dict],
        model: Any,
        batch_size: int = 32,
        show_progress: bool = True,
        text_builder: Optional[Callable[[Dict[str, object]], str]] = None,
    ) -> Dict[str, np.ndarray]:
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
        papers: Dict[str, Dict],
        model: Any,
        batch_size: int = 32,
        show_progress: bool = False,
        text_builder: Optional[Callable[[Dict[str, object]], str]] = None,
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

    def _process_embeddings(
        self,
        papers: Dict[str, Dict],
        model: Any,
        *,
        batch_size: int,
        show_progress: bool,
        text_builder: Optional[Callable[[Dict[str, object]], str]],
        return_embeddings: bool,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Coordinate one complete embedding cache operation with root clearing.

        :param Dict[str, Dict] papers: Mapping of paper ID to metadata payload.
        :param Any model: SentenceTransformer-compatible model exposing ``encode``.
        :param int batch_size: Batch size for model encoding.
        :param bool show_progress: Whether to display progress bars.
        :param Optional[Callable[[Dict[str, object]], str]] text_builder: Optional metadata->text formatter.
        :param bool return_embeddings: Whether to return float32 embedding payloads.
        :return Optional[Dict[str, np.ndarray]]: Embedding map when requested, else ``None``.
        """
        with self._cache_operation_lock():
            return self._process_embeddings_locked(
                papers,
                model,
                batch_size=batch_size,
                show_progress=show_progress,
                text_builder=text_builder,
                return_embeddings=return_embeddings,
            )

    def _process_embeddings_locked(
        self,
        papers: Dict[str, Dict],
        model: Any,
        *,
        batch_size: int,
        show_progress: bool,
        text_builder: Optional[Callable[[Dict[str, object]], str]],
        return_embeddings: bool,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Hydrate cache entries while the root operation lock is held.

        :param Dict[str, Dict] papers: Mapping of paper ID to metadata payload.
        :param Any model: SentenceTransformer-compatible model exposing ``encode``.
        :param int batch_size: Batch size for model encoding.
        :param bool show_progress: Whether to display progress bars.
        :param Optional[Callable[[Dict[str, object]], str]] text_builder: Optional metadata->text formatter.
        :param bool return_embeddings: Whether to return float32 embedding payloads.
        :return Optional[Dict[str, np.ndarray]]: Embedding map when requested, else ``None``.
        """
        if not papers:
            if return_embeddings:
                return {}
            return None

        cached_embeddings: Dict[str, np.ndarray] = {}
        papers_to_embed: List[PendingEmbeddingRecord] = []
        cached_rows: List[Tuple[str, int]] = []
        builder = text_builder or compose_title_abstract_text

        items = list(papers.items())
        progress_enabled = show_progress and stderr_isatty() and len(items) > 50
        iterator: Iterable[Tuple[str, Dict]] = progress_iterator(
            items,
            description="Checking cache",
            unit="papers",
            enabled=progress_enabled,
        )

        calibration_ranges: Optional[np.ndarray] = None
        with (
            self._cache_lock(),
            self._connect_db() as conn,
            h5py.File(self.h5_path, "a") as h5,
        ):
            self._recover_pending_replacements_locked(conn=conn, h5_file=h5)
            cursor = conn.cursor()
            existing_rows = self._load_existing_rows(
                conn, [paper_id for paper_id, _ in items]
            )
            embeddings_dataset = self._get_embeddings_dataset(h5)
            if embeddings_dataset is not None:
                self._assert_runtime_cache_consistency(
                    conn=conn,
                    h5_file=h5,
                    embeddings_dataset=embeddings_dataset,
                    fail_mode="runtime",
                )
            cached_limit = (
                int(embeddings_dataset.shape[0])
                if embeddings_dataset is not None
                else 0
            )

            metadata_updates_on_hit: List[Tuple[Any, ...]] = []

            for paper_id, metadata in iterator:
                text = builder(metadata)
                text_hash = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
                existing_row = existing_rows.get(paper_id)
                row_idx = existing_row["row_idx"] if existing_row is not None else None

                if (
                    existing_row is not None
                    and existing_row["text_hash"] == text_hash
                    and row_idx is not None
                    and embeddings_dataset is not None
                    and 0 <= row_idx < cached_limit
                ):
                    if return_embeddings:
                        cached_rows.append((paper_id, row_idx))
                    if self._metadata_fields_changed(existing_row, metadata):
                        metadata_updates_on_hit.append(
                            self._metadata_refresh_tuple(paper_id, metadata)
                        )
                else:
                    papers_to_embed.append(
                        PendingEmbeddingRecord(
                            paper_id=paper_id,
                            metadata=dict(metadata),
                            text_hash=text_hash,
                            text=text,
                            row_idx=row_idx,
                        )
                    )

            if progress_enabled:
                iterator.close()

            if return_embeddings and cached_rows and embeddings_dataset is not None:
                cached_embeddings = self._load_cached_embeddings(
                    h5,
                    embeddings_dataset,
                    cached_rows,
                )
            if metadata_updates_on_hit:
                cursor.executemany(
                    """
                    UPDATE papers
                    SET title = ?, abstract = ?, year = ?, authors_json = ?, categories_json = ?,
                        venue = ?, arxiv_id = ?, doi = ?
                    WHERE paper_id = ?
                    """,
                    metadata_updates_on_hit,
                )
                conn.commit()

            if not papers_to_embed:
                if return_embeddings:
                    return cached_embeddings
                return None

            if self.storage_precision == "int8":
                calibration_ranges = self._require_calibration_ranges(h5_file=h5)

        texts = [record.text for record in papers_to_embed]
        embeddings_array = encode_texts(
            model,
            texts,
            batch_size=min(int(batch_size), len(texts)),
            show_progress_bar=show_progress,
        )
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        if embeddings_array.ndim != 2:
            raise ValueError(
                "Embedding model must return a 2-dimensional matrix, "
                f"got shape {embeddings_array.shape}."
            )
        if embeddings_array.shape[0] != len(papers_to_embed):
            raise ValueError(
                "Embedding model returned unexpected row count: "
                f"{embeddings_array.shape[0]} for {len(papers_to_embed)} papers."
            )
        if embeddings_array.shape[1] < 1:
            raise ValueError("Embedding model returned vectors with zero dimensions.")
        if not np.all(np.isfinite(embeddings_array)):
            raise ValueError("Embedding model returned non-finite values.")

        embedding_dim = int(embeddings_array.shape[1])
        clipped_value_count = 0
        clipped_total_value_count = 0
        if self.storage_precision == "int8":
            if calibration_ranges is None:
                raise RuntimeError(
                    "Missing persisted int8 calibration ranges for cache writes."
                )
            if int(calibration_ranges.shape[1]) != embedding_dim:
                raise ValueError(
                    "Calibration range dimension mismatch in cache: "
                    f"{int(calibration_ranges.shape[1])} != {embedding_dim}"
                )
            clipped_value_count, clipped_total_value_count = (
                _count_int8_saturated_values(embeddings_array, calibration_ranges)
            )
            storage_embeddings = _quantize_int8_embeddings(
                embeddings_array, calibration_ranges
            )
        else:
            storage_embeddings = np.asarray(
                embeddings_array,
                dtype=_storage_dtype_for_precision(self.storage_precision),
            )
        with (
            self._cache_lock(),
            self._connect_db() as conn,
            h5py.File(self.h5_path, "a") as h5,
        ):
            self._recover_pending_replacements_locked(conn=conn, h5_file=h5)
            cursor = conn.cursor()
            self._set_h5_attrs(h5)
            embeddings_dataset = self._ensure_embeddings_dataset(h5, embedding_dim)
            binary_dataset = self._ensure_binary_dataset(h5, embedding_dim)
            if self.storage_precision == "int8":
                current_ranges = self._require_calibration_ranges(
                    h5_file=h5, embedding_dim=embedding_dim
                )
                if calibration_ranges is None or not np.array_equal(
                    current_ranges, calibration_ranges
                ):
                    raise RuntimeError(
                        "Int8 calibration ranges changed during encode; retry cache write."
                    )
                if clipped_total_value_count > 0:
                    self._record_int8_saturation(
                        h5_file=h5,
                        clipped_value_count=clipped_value_count,
                        total_value_count=clipped_total_value_count,
                    )
            dequantized_embeddings = (
                self._dequantize_int8(h5, storage_embeddings)
                if self.storage_precision == "int8"
                and (self.binary_prefilter or return_embeddings)
                else None
            )
            binary_embeddings = (
                _quantize_ubinary_embeddings(dequantized_embeddings)
                if self.binary_prefilter and dequantized_embeddings is not None
                else None
            )
            # Callers must see the vectors this cache will return forever after,
            # so int8 rows are round-tripped through storage before being handed
            # back; otherwise the encoding run and every later hit disagree.
            returned_embeddings = (
                l2_normalize_embeddings(dequantized_embeddings)
                if return_embeddings and dequantized_embeddings is not None
                else embeddings_array
            )

            existing_row_count = int(embeddings_dataset.shape[0])
            latest_rows = self._load_existing_rows(
                conn, [record.paper_id for record in papers_to_embed]
            )
            new_embeddings: Dict[str, np.ndarray] = {}
            rows_to_upsert: List[Tuple[Any, ...]] = []
            append_embeddings: List[np.ndarray] = []
            append_binary_embeddings: List[np.ndarray] = []
            append_records: List[Tuple[str, Dict[str, object], str]] = []
            replacement_rows: List[
                Tuple[
                    int,
                    np.ndarray,
                    Optional[np.ndarray],
                    np.ndarray,
                    Optional[np.ndarray],
                ]
            ] = []

            for idx, record in enumerate(papers_to_embed):
                storage_embedding = storage_embeddings[idx]
                binary_embedding = (
                    None if binary_embeddings is None else binary_embeddings[idx]
                )
                existing_row = latest_rows.get(record.paper_id)
                latest_row_idx = (
                    existing_row["row_idx"] if existing_row is not None else None
                )

                if (
                    existing_row is not None
                    and existing_row["text_hash"] == record.text_hash
                    and latest_row_idx is not None
                    and 0 <= latest_row_idx < existing_row_count
                ):
                    if return_embeddings:
                        new_embeddings.update(
                            self._load_cached_embeddings(
                                h5,
                                embeddings_dataset,
                                [(record.paper_id, latest_row_idx)],
                            )
                        )
                    rows_to_upsert.append(
                        self._metadata_tuple(
                            paper_id=record.paper_id,
                            metadata=record.metadata,
                            text_hash=record.text_hash,
                            embedding_dim=embedding_dim,
                            row_idx=latest_row_idx,
                        )
                    )
                    continue

                if return_embeddings:
                    new_embeddings[record.paper_id] = np.asarray(
                        returned_embeddings[idx], dtype=np.float32
                    )
                if (
                    latest_row_idx is not None
                    and 0 <= latest_row_idx < existing_row_count
                ):
                    previous_binary = (
                        np.asarray(binary_dataset[latest_row_idx]).copy()
                        if binary_dataset is not None
                        else None
                    )
                    replacement_rows.append(
                        (
                            latest_row_idx,
                            np.asarray(embeddings_dataset[latest_row_idx]).copy(),
                            previous_binary,
                            storage_embedding,
                            binary_embedding,
                        )
                    )
                    rows_to_upsert.append(
                        self._metadata_tuple(
                            paper_id=record.paper_id,
                            metadata=record.metadata,
                            text_hash=record.text_hash,
                            embedding_dim=embedding_dim,
                            row_idx=latest_row_idx,
                        )
                    )
                    continue

                append_embeddings.append(storage_embedding)
                if binary_dataset is not None and binary_embedding is not None:
                    append_binary_embeddings.append(binary_embedding)
                append_records.append(
                    (record.paper_id, record.metadata, record.text_hash)
                )

            if replacement_rows:
                self._persist_replacement_journal(
                    conn=conn,
                    replacements=[
                        (row_idx, previous_embedding, previous_binary)
                        for row_idx, previous_embedding, previous_binary, _, _ in replacement_rows
                    ],
                )
                for (
                    row_idx,
                    _,
                    _,
                    replacement_embedding,
                    replacement_binary,
                ) in replacement_rows:
                    embeddings_dataset[row_idx] = replacement_embedding
                    if binary_dataset is not None and replacement_binary is not None:
                        binary_dataset[row_idx] = replacement_binary

            if append_embeddings:
                append_array = np.vstack(append_embeddings).astype(
                    embeddings_dataset.dtype, copy=False
                )
                start_idx = existing_row_count
                end_idx = start_idx + append_array.shape[0]
                embeddings_dataset.resize((end_idx, embedding_dim))
                embeddings_dataset[start_idx:end_idx] = append_array

                if binary_dataset is not None and append_binary_embeddings:
                    append_binary_array = np.vstack(append_binary_embeddings).astype(
                        np.uint8, copy=False
                    )
                    binary_dataset.resize((end_idx, append_binary_array.shape[1]))
                    binary_dataset[start_idx:end_idx] = append_binary_array

                for offset, (paper_id, metadata, text_hash) in enumerate(
                    append_records
                ):
                    rows_to_upsert.append(
                        self._metadata_tuple(
                            paper_id=paper_id,
                            metadata=metadata,
                            text_hash=text_hash,
                            embedding_dim=embedding_dim,
                            row_idx=start_idx + offset,
                        )
                    )

            if rows_to_upsert:
                # SQLite commits durably, so its row mappings must never become
                # durable ahead of the vectors they point at: an interrupted
                # append would otherwise be recovered by discarding committed rows.
                self._flush_h5_file(h5)
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO papers
                    (paper_id, title, abstract, year, text_hash, embedding_dim, row_idx,
                     authors_json, categories_json, venue, arxiv_id, doi)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows_to_upsert,
                )
                if replacement_rows:
                    cursor.executemany(
                        "DELETE FROM replacement_journal WHERE row_idx = ?",
                        [(row_idx,) for row_idx, *_ in replacement_rows],
                    )

        if return_embeddings:
            return {**cached_embeddings, **new_embeddings}
        return None

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
    ) -> List[CacheSearchResult]:
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
                results: List[CacheSearchResult] = []
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

    def get_cached_paper_ids(self) -> Set[str]:
        """Return all cached paper IDs for this namespace.

        :return Set[str]: Cached paper IDs loaded from SQLite metadata rows.
        """
        if not path_exists(self.db_path):
            return set()

        paper_ids: Set[str] = set()
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
        corpus_size: Optional[int],
        dataset_source: Optional[str] = None,
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

    def get_hydrated_dataset_source(self) -> Optional[str]:
        """Return dataset source captured for the latest hydrated cache attempt.

        :return Optional[str]: Hydrated dataset source token when set.
        """
        with self._cache_lock(), self._connect_db() as conn:
            metadata = self._load_cache_metadata(conn)
        cached_source = metadata.get(HYDRATION_DATASET_SOURCE_KEY)
        return cached_source if cached_source else None

    def has_current_corpus_metadata(self) -> bool:
        """Return whether corpus years and DOIs use the current source adapter.

        :return bool: Whether a full metadata pass completed with this adapter.
        """
        with self._cache_lock(), self._connect_db() as conn:
            metadata = self._load_cache_metadata(conn)
        return metadata.get(CORPUS_METADATA_VERSION_KEY) == CORPUS_METADATA_VERSION

    def mark_corpus_metadata_current(self) -> None:
        """Record successful corpus metadata hydration or backfill.

        :return None: Persists the completed metadata adapter version.
        """
        with self._cache_lock(), self._connect_db() as conn:
            self._set_cache_metadata(
                conn, {CORPUS_METADATA_VERSION_KEY: CORPUS_METADATA_VERSION}
            )

    def update_corpus_metadata(self, papers: Sequence[Dict]) -> None:
        """Correct years and DOIs on existing corpus rows without touching vectors.

        :param Sequence[Dict] papers: Source metadata with paper IDs, years and DOIs.
        :return None: Updates only matching SQLite records.
        """
        with self._cache_lock(), self._connect_db() as conn:
            conn.executemany(
                "UPDATE papers SET year = ?, doi = ? WHERE paper_id = ?",
                [(paper["year"], paper["doi"], paper["paper_id"]) for paper in papers],
            )

    def get_model_fingerprint(self) -> Optional[str]:
        """Return model fingerprint captured for this cache namespace.

        :return Optional[str]: Active model fingerprint or ``None`` when unset.
        """
        with self._cache_lock(), self._connect_db() as conn:
            metadata = self._load_cache_metadata(conn)
        fingerprint = str(metadata.get(MODEL_FINGERPRINT_KEY, "")).strip()
        return fingerprint or None

    def get_hydration_rowcount_reconciliation(self) -> Optional[Tuple[int, int]]:
        """Return persisted full-split reconciliation marker for row-count deltas.

        :return Optional[Tuple[int, int]]: ``(upstream_rows, cached_rows)`` when
            a prior full-split reconciliation confirmed no uncached paper IDs for
            that row-count state; ``None`` when unset/invalid.
        """
        with self._cache_lock(), self._connect_db() as conn:
            metadata = self._load_cache_metadata(conn)
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
        with self._cache_lock(), self._connect_db() as conn:
            self._set_cache_metadata(
                conn,
                {
                    HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY: resolved_upstream,
                    HYDRATION_RECONCILED_CACHE_ROWS_KEY: resolved_cached,
                },
            )

    def clear_hydration_rowcount_reconciliation(self) -> None:
        """Clear persisted row-count reconciliation marker metadata.

        :return None: Mutates SQLite metadata in-place.
        """
        with self._cache_lock(), self._connect_db() as conn:
            self._set_cache_metadata(
                conn,
                {
                    HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY: "",
                    HYDRATION_RECONCILED_CACHE_ROWS_KEY: "",
                },
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
        with self._cache_lock(), self._connect_db() as conn:
            self._set_cache_metadata(conn, {MODEL_FINGERPRINT_KEY: normalized})

    def mark_hydrated(
        self,
        dataset_source: str,
        dataset_split: str,
        corpus_size: Optional[int],
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
        with self._cache_lock(), self._connect_db() as conn:
            self._set_cache_metadata(
                conn,
                {
                    HYDRATION_DATASET_SOURCE_KEY: normalized_source,
                    HYDRATION_SPLIT_KEY: dataset_split,
                    HYDRATION_CORPUS_SIZE_KEY: _corpus_size_token(corpus_size),
                    HYDRATION_COMPLETE_KEY: "1" if complete else "0",
                    HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY: "",
                    HYDRATION_RECONCILED_CACHE_ROWS_KEY: "",
                },
            )

    def clear(self, reason: Optional[str] = None) -> None:
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
