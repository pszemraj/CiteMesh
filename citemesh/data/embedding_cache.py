"""Persistent embedding cache backed by SQLite metadata and HDF5 vectors."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import h5py
import numpy as np
from filelock import FileLock, Timeout
from tqdm.auto import tqdm

from .cache import get_cache_dir

logger = logging.getLogger(__name__)


SQLITE_QUERY_BATCH_SIZE = 900
EMBEDDINGS_DATASET_NAME = "embeddings"
BINARY_INDEX_DATASET_NAME = "binary_index"
CALIBRATION_RANGES_DATASET_NAME = "calibration_ranges"
EMBEDDING_CACHE_SCHEMA_VERSION = 2
EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS = 60.0
H5_LAYOUT_KEY = "h5_layout_version"
H5_LAYOUT_MATRIX_VERSION = "matrix-v2-quantized"
SCHEMA_VERSION_KEY = "schema_version"
STORAGE_PRECISION_KEY = "storage_precision"
SOURCE_TORCH_DTYPE_KEY = "source_torch_dtype"
BINARY_PREFILTER_ENABLED_KEY = "binary_prefilter_enabled"
HYDRATION_DATASET_SOURCE_KEY = "hydration_dataset_source"
HYDRATION_SPLIT_KEY = "hydration_split"
HYDRATION_CORPUS_SIZE_KEY = "hydration_corpus_size"
HYDRATION_COMPLETE_KEY = "hydration_complete"

_STORAGE_PRECISIONS = {"float32", "float16", "int8"}
_POPCOUNT_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
    axis=1
)


def _metadata_table_create_sql() -> str:
    """Return SQL DDL used to create cache metadata table.

    :return str: SQL statement for ``cache_metadata`` table creation.
    """
    return """
    CREATE TABLE IF NOT EXISTS cache_metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """


def _corpus_size_token(corpus_size: Optional[int]) -> str:
    """Convert optional corpus size into stable metadata token.

    :param Optional[int] corpus_size: Optional corpus-size cap.
    :return str: Tokenized corpus-size value.
    """
    return "all" if corpus_size is None else str(int(corpus_size))


def _safe_json_list(value: Any) -> str:
    """Serialize metadata list values as JSON arrays.

    :param Any value: Raw metadata value.
    :return str: JSON array string.
    """
    if isinstance(value, list):
        normalized = [str(item) for item in value if str(item).strip()]
        return json.dumps(normalized)
    return json.dumps([])


def _parse_json_list(value: Optional[str]) -> List[str]:
    """Parse JSON list payload from SQLite metadata rows.

    :param Optional[str] value: Raw JSON string.
    :return List[str]: Parsed list payload.
    """
    if not value:
        return []

    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []

    if not isinstance(decoded, list):
        return []

    return [str(item) for item in decoded if str(item).strip()]


def _storage_dtype_for_precision(storage_precision: str) -> np.dtype:
    """Map storage precision token to NumPy dtype.

    :param str storage_precision: Storage precision token.
    :return np.dtype: Dtype used for HDF5 embedding matrix.
    """
    if storage_precision == "float32":
        return np.dtype(np.float32)
    if storage_precision == "float16":
        return np.dtype(np.float16)
    if storage_precision == "int8":
        return np.dtype(np.int8)
    raise ValueError(f"Unsupported storage precision: {storage_precision}")


def _sanitize_ranges(ranges: np.ndarray) -> np.ndarray:
    """Ensure per-dimension quantization ranges are strictly non-zero.

    :param np.ndarray ranges: Raw ``(2, dim)`` range matrix.
    :return np.ndarray: Sanitized range matrix.
    """
    normalized = np.asarray(ranges, dtype=np.float32)
    if normalized.shape[0] != 2:
        raise ValueError(
            f"Calibration ranges must have shape (2, dim), got {normalized.shape}."
        )

    mins = normalized[0]
    maxs = normalized[1]
    too_small = (maxs - mins) < 1e-6
    if np.any(too_small):
        maxs = maxs.copy()
        maxs[too_small] = mins[too_small] + 1e-6
    return np.vstack((mins, maxs)).astype(np.float32)


@dataclass(frozen=True)
class CacheSearchResult:
    """Search result returned by ``EmbeddingCache.search``."""

    paper_id: str
    score: float
    embedding: np.ndarray
    metadata: Dict[str, Any]


class EmbeddingCache:
    """Persistent cache for paper embeddings and metadata."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        model_name: str = "google/embeddinggemma-300m",
        storage_precision: str = "int8",
        binary_prefilter: bool = True,
        calibration_sample_size: int = 2000,
        compression: str = "gzip",
        compression_level: int = 1,
        source_torch_dtype: str = "float32",
    ):
        """Create a persistent embedding cache for a model variant.

        :param Optional[Path] cache_dir: Cache directory override. Uses global cache when ``None``.
        :param str model_name: Model namespace string used for cache partitioning.
        :param str storage_precision: Persistent embedding precision ``float32``/``float16``/``int8``.
        :param bool binary_prefilter: Whether to maintain a binary index for int8 search.
        :param int calibration_sample_size: Target sample size for int8 calibration ranges.
        :param str compression: HDF5 compression filter name.
        :param int compression_level: Compression level for HDF5 datasets.
        :param str source_torch_dtype: Source inference dtype token, e.g. ``bfloat16``.
        """
        if storage_precision not in _STORAGE_PRECISIONS:
            expected = ", ".join(sorted(_STORAGE_PRECISIONS))
            raise ValueError(
                f"storage_precision must be one of {{{expected}}}, "
                f"got {storage_precision!r}."
            )
        if calibration_sample_size < 1:
            raise ValueError("calibration_sample_size must be at least 1")

        if cache_dir is None:
            cache_dir = get_cache_dir("embeddings")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
        self.db_path = self.cache_dir / f"metadata_{model_hash}.db"
        self.h5_path = self.cache_dir / f"embeddings_{model_hash}.h5"
        self.lock_path = self.cache_dir / f"cache_{model_hash}.lock"

        self.model_name = model_name
        self.storage_precision = storage_precision
        self.binary_prefilter = bool(binary_prefilter and storage_precision == "int8")
        self.calibration_sample_size = int(calibration_sample_size)
        self.compression = compression
        self.compression_level = int(compression_level)
        self.source_torch_dtype = str(source_torch_dtype or "float32")

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
        if not papers:
            return {}

        cached_embeddings: Dict[str, np.ndarray] = {}
        papers_to_embed: List[Tuple[str, Dict, str, str, Optional[int]]] = []
        cached_rows: List[Tuple[str, int]] = []
        builder = text_builder or _build_text

        items = list(papers.items())
        progress_enabled = show_progress and sys.stderr.isatty() and len(items) > 50
        iterator: Iterable[Tuple[str, Dict]] = tqdm(
            items,
            desc="Checking cache",
            unit="papers",
            disable=not progress_enabled,
        )

        with (
            self._cache_lock(),
            sqlite3.connect(self.db_path) as conn,
            h5py.File(self.h5_path, "a") as h5,
        ):
            cursor = conn.cursor()
            existing_rows = self._load_existing_rows(
                conn, [paper_id for paper_id, _ in items]
            )
            embeddings_dataset = self._get_embeddings_dataset(h5)
            cached_limit = (
                int(embeddings_dataset.shape[0])
                if embeddings_dataset is not None
                else 0
            )

            for paper_id, metadata in iterator:
                text = builder(metadata)
                text_hash = self._text_hash(text)
                existing_row = existing_rows.get(paper_id)
                row_idx = existing_row[1] if existing_row is not None else None

                if (
                    existing_row is not None
                    and existing_row[0] == text_hash
                    and row_idx is not None
                    and embeddings_dataset is not None
                    and 0 <= row_idx < cached_limit
                ):
                    cached_rows.append((paper_id, row_idx))
                else:
                    papers_to_embed.append(
                        (paper_id, metadata, text_hash, text, row_idx)
                    )

            if progress_enabled:
                iterator.close()

            if cached_rows and embeddings_dataset is not None:
                cached_embeddings = self._load_cached_embeddings(
                    h5,
                    embeddings_dataset,
                    cached_rows,
                )

            if not papers_to_embed:
                return cached_embeddings

            texts = [text for _, _, _, text, _ in papers_to_embed]
            embeddings_array = np.asarray(
                model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=show_progress,
                ),
                dtype=np.float32,
            )
            if embeddings_array.ndim == 1:
                embeddings_array = embeddings_array.reshape(1, -1)
            if embeddings_array.shape[0] != len(papers_to_embed):
                raise ValueError(
                    "Embedding model returned unexpected row count: "
                    f"{embeddings_array.shape[0]} for {len(papers_to_embed)} papers."
                )

            embedding_dim = int(embeddings_array.shape[1])
            self._set_h5_attrs(h5)
            embeddings_dataset = self._ensure_embeddings_dataset(h5, embedding_dim)
            binary_dataset = self._ensure_binary_dataset(h5, embedding_dim)
            storage_embeddings = self._to_storage_embeddings(h5, embeddings_array)
            binary_embeddings = self._to_binary_embeddings(embeddings_array)

            existing_row_count = int(embeddings_dataset.shape[0])
            new_embeddings: Dict[str, np.ndarray] = {}
            rows_to_upsert: List[Tuple[Any, ...]] = []
            append_embeddings: List[np.ndarray] = []
            append_binary_embeddings: List[np.ndarray] = []
            append_records: List[Tuple[str, Dict, str]] = []

            for idx, (paper_id, metadata, text_hash, _, existing_row_idx) in enumerate(
                papers_to_embed
            ):
                embedding = embeddings_array[idx]
                storage_embedding = storage_embeddings[idx]
                new_embeddings[paper_id] = embedding

                if (
                    existing_row_idx is not None
                    and 0 <= existing_row_idx < existing_row_count
                ):
                    embeddings_dataset[existing_row_idx] = storage_embedding
                    if binary_dataset is not None and binary_embeddings is not None:
                        binary_dataset[existing_row_idx] = binary_embeddings[idx]
                    rows_to_upsert.append(
                        self._metadata_tuple(
                            paper_id=paper_id,
                            metadata=metadata,
                            text_hash=text_hash,
                            embedding_dim=embedding_dim,
                            row_idx=existing_row_idx,
                        )
                    )
                else:
                    append_embeddings.append(storage_embedding)
                    if binary_dataset is not None and binary_embeddings is not None:
                        append_binary_embeddings.append(binary_embeddings[idx])
                    append_records.append((paper_id, metadata, text_hash))

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
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO papers
                    (paper_id, title, abstract, year, text_hash, embedding_dim, row_idx, authors_json, categories_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows_to_upsert,
                )

            conn.commit()

        return {**cached_embeddings, **new_embeddings}

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

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query_embedding must be 1-dimensional")
        if not self.h5_path.exists():
            return []

        with (
            self._cache_lock(),
            sqlite3.connect(self.db_path) as conn,
            h5py.File(self.h5_path, "r") as h5,
        ):
            embeddings_dataset = self._get_embeddings_dataset(h5)
            if embeddings_dataset is None or embeddings_dataset.shape[0] == 0:
                return []
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
                    candidate_rows = self._binary_prefilter_rows(
                        binary_dataset=binary_dataset,
                        query_embedding=query,
                        candidate_count=candidate_count,
                    )
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
                    rows, scores, embeddings = self._score_int8_rows(
                        embeddings_dataset=embeddings_dataset,
                        h5_file=h5,
                        query_embedding=query,
                        top_k=top_k,
                        row_indices=None,
                    )
            else:
                rows, scores, embeddings = self._score_float_rows(
                    embeddings_dataset=embeddings_dataset,
                    query_embedding=query,
                    top_k=top_k,
                )

            if rows.size == 0:
                return []

            metadata_by_row = self._load_metadata_by_rows(conn, rows.tolist())
            results: List[CacheSearchResult] = []
            for idx, row_idx in enumerate(rows.tolist()):
                payload = metadata_by_row.get(row_idx)
                if payload is None:
                    continue
                result_metadata = {
                    "title": payload.get("title", "Unknown"),
                    "abstract": payload.get("abstract", ""),
                    "year": payload.get("year"),
                    "authors": payload.get("authors", []),
                    "categories": payload.get("categories", []),
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

    def get_stats(self) -> Dict[str, Optional[float]]:
        """Return basic cache statistics.

        :return Dict[str, Optional[float]]: Cache size, dimensions, and year range.
        """
        total_h5_size = self.h5_path.stat().st_size if self.h5_path.exists() else 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM papers")
            total_papers = cursor.fetchone()[0]

            cursor.execute(
                "SELECT AVG(embedding_dim), MIN(year), MAX(year) FROM papers WHERE embedding_dim IS NOT NULL"
            )
            avg_dim, min_year, max_year = cursor.fetchone()

        return {
            "total_papers": total_papers,
            "avg_embedding_dim": avg_dim,
            "year_range": (min_year, max_year),
            "cache_size_mb": total_h5_size / (1024 * 1024),
        }

    def has_calibration_ranges(self) -> bool:
        """Return whether int8 calibration ranges exist in cache.

        :return bool: ``True`` when the calibration range dataset exists.
        """
        if self.storage_precision != "int8":
            return False
        if not self.h5_path.exists():
            return False

        with self._cache_lock(), h5py.File(self.h5_path, "r") as h5:
            return CALIBRATION_RANGES_DATASET_NAME in h5

    def set_calibration_ranges(self, ranges: np.ndarray, embedding_dim: int) -> None:
        """Persist int8 calibration ranges for future quantization.

        :param np.ndarray ranges: Range matrix with shape ``(2, embedding_dim)``.
        :param int embedding_dim: Embedding dimension used for validation.
        :return None: Mutates HDF5 state in-place.
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
                dataset[...] = sanitized

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
        """
        expected_split = str(dataset_split)
        expected_corpus_size = _corpus_size_token(corpus_size)
        expected_source = None if dataset_source is None else str(dataset_source)

        with sqlite3.connect(self.db_path) as conn:
            metadata = self._load_cache_metadata(conn)

        if expected_source is not None:
            cached_source = metadata.get(HYDRATION_DATASET_SOURCE_KEY)
            if cached_source != expected_source:
                return False

        metadata_matches = (
            metadata.get(HYDRATION_COMPLETE_KEY, "0") == "1"
            and metadata.get(HYDRATION_SPLIT_KEY) == expected_split
            and metadata.get(HYDRATION_CORPUS_SIZE_KEY) == expected_corpus_size
        )
        if not metadata_matches:
            return False

        return self._has_queryable_hydrated_payload()

    def _has_queryable_hydrated_payload(self) -> bool:
        """Return whether hydrated HDF5 payload exists and can be queried safely.

        :return bool: ``True`` when cache has a readable, non-empty embedding matrix.
        """
        if not self.h5_path.exists():
            return False

        try:
            with self._cache_lock(), h5py.File(self.h5_path, "r") as h5:
                embeddings_dataset = self._get_embeddings_dataset(h5)
                if embeddings_dataset is None:
                    return False
                if int(embeddings_dataset.shape[0]) < 1:
                    return False

                if (
                    self.storage_precision == "int8"
                    and CALIBRATION_RANGES_DATASET_NAME not in h5
                ):
                    return False
        except (OSError, ValueError):
            return False

        return True

    def get_hydrated_dataset_source(self) -> Optional[str]:
        """Return dataset source captured for the latest hydrated cache attempt."""
        with sqlite3.connect(self.db_path) as conn:
            metadata = self._load_cache_metadata(conn)
        cached_source = metadata.get(HYDRATION_DATASET_SOURCE_KEY)
        return cached_source if cached_source else None

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
        with sqlite3.connect(self.db_path) as conn:
            self._set_cache_metadata(conn, HYDRATION_DATASET_SOURCE_KEY, dataset_source)
            self._set_cache_metadata(conn, HYDRATION_SPLIT_KEY, str(dataset_split))
            self._set_cache_metadata(
                conn,
                HYDRATION_CORPUS_SIZE_KEY,
                _corpus_size_token(corpus_size),
            )
            self._set_cache_metadata(
                conn, HYDRATION_COMPLETE_KEY, "1" if complete else "0"
            )
            conn.commit()

    def clear(self) -> None:
        """Purge cache artifacts for this cache namespace."""
        with self._cache_lock():
            self.db_path.unlink(missing_ok=True)
            self.h5_path.unlink(missing_ok=True)
            self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers

    @contextmanager
    def _cache_lock(self) -> Iterator[None]:
        """Serialize cache mutations across processes for this model namespace.

        :return Iterator[None]: Context manager yielding once lock is acquired.
        """
        lock = FileLock(
            str(self.lock_path), timeout=EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS
        )
        try:
            with lock:
                yield
        except Timeout as exc:
            raise TimeoutError(
                "Timed out waiting for embedding cache lock "
                f"at {self.lock_path}. Another process may be holding it."
            ) from exc

    def _init_db(self) -> None:
        """Create and initialize the metadata cache schema when needed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    year INTEGER,
                    text_hash TEXT,
                    embedding_dim INTEGER,
                    row_idx INTEGER,
                    authors_json TEXT,
                    categories_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(papers)").fetchall()
            }
            if "row_idx" not in columns:
                conn.execute("ALTER TABLE papers ADD COLUMN row_idx INTEGER")
            if "authors_json" not in columns:
                conn.execute("ALTER TABLE papers ADD COLUMN authors_json TEXT")
            if "categories_json" not in columns:
                conn.execute("ALTER TABLE papers ADD COLUMN categories_json TEXT")

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_text_hash ON papers(text_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_row_idx ON papers(row_idx)"
            )
            conn.execute(_metadata_table_create_sql())

            self._set_cache_metadata(
                conn, SCHEMA_VERSION_KEY, str(EMBEDDING_CACHE_SCHEMA_VERSION)
            )
            self._set_cache_metadata(conn, H5_LAYOUT_KEY, H5_LAYOUT_MATRIX_VERSION)
            self._set_cache_metadata(
                conn, STORAGE_PRECISION_KEY, self.storage_precision
            )
            self._set_cache_metadata(
                conn, SOURCE_TORCH_DTYPE_KEY, self.source_torch_dtype
            )
            self._set_cache_metadata(
                conn,
                BINARY_PREFILTER_ENABLED_KEY,
                "1" if self.binary_prefilter else "0",
            )
            # Preserve hydration completion across restarts; initialize only once.
            self._set_cache_metadata_default(conn, HYDRATION_COMPLETE_KEY, "0")
            conn.commit()

    @staticmethod
    def _set_cache_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
        """Insert or update a metadata key-value pair.

        :param sqlite3.Connection conn: Open SQLite connection.
        :param str key: Metadata key.
        :param str value: Metadata value.
        :return None: This method mutates DB state in-place.
        """
        conn.execute(
            """
            INSERT INTO cache_metadata (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    @staticmethod
    def _set_cache_metadata_default(
        conn: sqlite3.Connection, key: str, value: str
    ) -> None:
        """Insert metadata key-value pair only when the key is absent.

        :param sqlite3.Connection conn: Open SQLite connection.
        :param str key: Metadata key.
        :param str value: Metadata value.
        :return None: This method mutates DB state in-place.
        """
        conn.execute(
            """
            INSERT OR IGNORE INTO cache_metadata (key, value)
            VALUES (?, ?)
            """,
            (key, value),
        )

    @staticmethod
    def _load_cache_metadata(conn: sqlite3.Connection) -> Dict[str, str]:
        """Load cache metadata table into an in-memory mapping.

        :param sqlite3.Connection conn: Open SQLite connection.
        :return Dict[str, str]: Metadata key-value mapping.
        """
        rows = conn.execute("SELECT key, value FROM cache_metadata").fetchall()
        return {str(key): str(value) for key, value in rows}

    def _reconcile_layout_metadata(self, conn: sqlite3.Connection) -> None:
        """Write metadata keys for active matrix-layout state.

        :param sqlite3.Connection conn: Open SQLite connection.
        :return None: This method mutates DB state in-place.
        """
        self._set_cache_metadata(conn, H5_LAYOUT_KEY, H5_LAYOUT_MATRIX_VERSION)

    def _ensure_h5_layout(self) -> None:
        """Ensure cache file uses matrix-based HDF5 layout."""
        if not self.h5_path.exists():
            with sqlite3.connect(self.db_path) as conn:
                self._reconcile_layout_metadata(conn)
                conn.commit()
                return

        try:
            with h5py.File(self.h5_path, "r") as h5:
                dataset = h5.get(EMBEDDINGS_DATASET_NAME)
                if dataset is None or dataset.ndim != 2:
                    raise ValueError("incompatible embedding cache layout")
        except (OSError, ValueError):
            logger.warning(
                "Embedding cache %s is incompatible with current schema. "
                "Clearing namespace cache and rebuilding.",
                self.h5_path,
            )
            self.h5_path.unlink(missing_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM papers")
                self._set_cache_metadata(conn, HYDRATION_DATASET_SOURCE_KEY, "")
                self._set_cache_metadata(conn, HYDRATION_SPLIT_KEY, "")
                self._set_cache_metadata(conn, HYDRATION_CORPUS_SIZE_KEY, "")
                self._set_cache_metadata(conn, HYDRATION_COMPLETE_KEY, "0")
                self._reconcile_layout_metadata(conn)
                conn.commit()
            return

        with sqlite3.connect(self.db_path) as conn:
            self._reconcile_layout_metadata(conn)
            conn.commit()

    @staticmethod
    def _metadata_tuple(
        paper_id: str,
        metadata: Dict[str, object],
        text_hash: str,
        embedding_dim: int,
        row_idx: int,
    ) -> Tuple[Any, ...]:
        """Build metadata row tuple for SQLite upsert.

        :param str paper_id: Paper identifier.
        :param Dict[str, object] metadata: Paper metadata payload.
        :param str text_hash: Deterministic hash for encoded text.
        :param int embedding_dim: Embedding vector width.
        :param int row_idx: Row index inside matrix dataset.
        :return Tuple[Any, ...]: SQLite upsert tuple matching ``papers`` columns.
        """
        title = str(metadata.get("title", "") or "")
        abstract = str(metadata.get("abstract", "") or "")
        year_raw = metadata.get("year")
        year = None
        if year_raw is not None:
            try:
                year = int(year_raw)
            except (TypeError, ValueError):
                year = None

        return (
            paper_id,
            title,
            abstract,
            year,
            text_hash,
            embedding_dim,
            row_idx,
            _safe_json_list(metadata.get("authors", [])),
            _safe_json_list(metadata.get("categories", [])),
        )

    def _set_h5_attrs(self, h5_file: h5py.File) -> None:
        """Write schema/layout metadata attrs to an open HDF5 file.

        :param h5py.File h5_file: Open cache file handle.
        :return None: Mutates HDF5 attrs in-place.
        """
        h5_file.attrs[SCHEMA_VERSION_KEY] = EMBEDDING_CACHE_SCHEMA_VERSION
        h5_file.attrs[H5_LAYOUT_KEY] = H5_LAYOUT_MATRIX_VERSION
        h5_file.attrs[STORAGE_PRECISION_KEY] = self.storage_precision
        h5_file.attrs[SOURCE_TORCH_DTYPE_KEY] = self.source_torch_dtype
        h5_file.attrs[BINARY_PREFILTER_ENABLED_KEY] = int(self.binary_prefilter)

    def _load_existing_rows(
        self,
        conn: sqlite3.Connection,
        paper_ids: Sequence[str],
    ) -> Dict[str, Tuple[str, Optional[int]]]:
        """Fetch existing metadata rows for target paper IDs.

        :param sqlite3.Connection conn: Open SQLite connection.
        :param Sequence[str] paper_ids: Paper IDs to look up.
        :return Dict[str, Tuple[str, Optional[int]]]: Mapping of paper ID to (text_hash, row_idx).
        """
        if not paper_ids:
            return {}

        existing_rows: Dict[str, Tuple[str, Optional[int]]] = {}
        for id_chunk in _chunked(paper_ids, SQLITE_QUERY_BATCH_SIZE):
            placeholders = ",".join("?" for _ in id_chunk)
            query = (
                "SELECT paper_id, text_hash, row_idx "
                f"FROM papers WHERE paper_id IN ({placeholders})"
            )

            for paper_id, text_hash, row_idx in conn.execute(query, id_chunk):
                normalized_row_idx = int(row_idx) if row_idx is not None else None
                existing_rows[str(paper_id)] = (str(text_hash), normalized_row_idx)

        return existing_rows

    @staticmethod
    def _get_embeddings_dataset(h5_file: h5py.File) -> Optional[h5py.Dataset]:
        """Return matrix embedding dataset when available.

        :param h5py.File h5_file: Open HDF5 cache handle.
        :return Optional[h5py.Dataset]: 2D embedding dataset or ``None``.
        """
        dataset = h5_file.get(EMBEDDINGS_DATASET_NAME)
        if dataset is None:
            return None
        if dataset.ndim != 2:
            raise ValueError(
                f"Embedding dataset '{EMBEDDINGS_DATASET_NAME}' must be 2D."
            )
        return dataset

    def _ensure_embeddings_dataset(
        self, h5_file: h5py.File, embedding_dim: int
    ) -> h5py.Dataset:
        """Create or validate the matrix embedding dataset.

        :param h5py.File h5_file: Open HDF5 file handle.
        :param int embedding_dim: Required embedding width.
        :return h5py.Dataset: Resizable embeddings dataset.
        """
        dataset = self._get_embeddings_dataset(h5_file)
        target_dtype = _storage_dtype_for_precision(self.storage_precision)
        chunk_rows = max(1, min(4096, self.calibration_sample_size))

        if dataset is None:
            return h5_file.create_dataset(
                EMBEDDINGS_DATASET_NAME,
                shape=(0, embedding_dim),
                maxshape=(None, embedding_dim),
                dtype=target_dtype,
                chunks=(chunk_rows, embedding_dim),
                compression=self.compression,
                compression_opts=self.compression_level,
                shuffle=True,
            )

        if int(dataset.shape[1]) != embedding_dim:
            raise ValueError(
                "Embedding dimension mismatch in cache: "
                f"{int(dataset.shape[1])} != {embedding_dim}"
            )
        if np.dtype(dataset.dtype) != np.dtype(target_dtype):
            raise ValueError(
                "Embedding storage dtype mismatch in cache: "
                f"{dataset.dtype} != {target_dtype}"
            )

        return dataset

    def _ensure_binary_dataset(
        self, h5_file: h5py.File, embedding_dim: int
    ) -> Optional[h5py.Dataset]:
        """Create or validate binary-index dataset for int8 cache search.

        :param h5py.File h5_file: Open HDF5 file handle.
        :param int embedding_dim: Embedding width.
        :return Optional[h5py.Dataset]: Binary-index dataset when enabled.
        """
        if not self.binary_prefilter:
            return None

        packed_dim = (int(embedding_dim) + 7) // 8
        chunk_rows = max(1, min(4096, self.calibration_sample_size))
        dataset = h5_file.get(BINARY_INDEX_DATASET_NAME)
        if dataset is None:
            return h5_file.create_dataset(
                BINARY_INDEX_DATASET_NAME,
                shape=(0, packed_dim),
                maxshape=(None, packed_dim),
                dtype=np.uint8,
                chunks=(chunk_rows, packed_dim),
                compression=self.compression,
                compression_opts=self.compression_level,
                shuffle=True,
            )

        if dataset.ndim != 2:
            raise ValueError(
                f"Binary dataset '{BINARY_INDEX_DATASET_NAME}' must be 2D."
            )
        if int(dataset.shape[1]) != packed_dim:
            raise ValueError(
                "Binary embedding dimension mismatch in cache: "
                f"{int(dataset.shape[1])} != {packed_dim}"
            )

        return dataset

    def _ensure_calibration_ranges(
        self,
        h5_file: h5py.File,
        embedding_dim: int,
        embeddings_array: np.ndarray,
    ) -> np.ndarray:
        """Create or validate int8 calibration ranges.

        :param h5py.File h5_file: Open HDF5 cache handle.
        :param int embedding_dim: Expected embedding dimension.
        :param np.ndarray embeddings_array: Current float32 batch.
        :return np.ndarray: Calibration ranges with shape ``(2, dim)``.
        """
        if self.storage_precision != "int8":
            raise RuntimeError("Calibration ranges are only valid for int8 storage")

        existing = h5_file.get(CALIBRATION_RANGES_DATASET_NAME)
        if existing is None:
            if embeddings_array.shape[0] < self.calibration_sample_size:
                logger.warning(
                    "Initializing int8 calibration ranges from %s embeddings; "
                    "consider hydrating with at least %s for better stability.",
                    embeddings_array.shape[0],
                    self.calibration_sample_size,
                )
            ranges = np.vstack(
                (
                    np.min(embeddings_array, axis=0),
                    np.max(embeddings_array, axis=0),
                )
            )
            sanitized = _sanitize_ranges(ranges)
            h5_file.create_dataset(
                CALIBRATION_RANGES_DATASET_NAME,
                data=sanitized,
                dtype=np.float32,
            )
            return sanitized

        if existing.ndim != 2 or existing.shape[0] != 2:
            raise ValueError(
                f"Calibration ranges dataset must have shape (2, dim), got {existing.shape}."
            )
        if int(existing.shape[1]) != embedding_dim:
            raise ValueError(
                "Calibration range dimension mismatch in cache: "
                f"{int(existing.shape[1])} != {embedding_dim}"
            )

        return _sanitize_ranges(np.asarray(existing, dtype=np.float32))

    def _to_storage_embeddings(
        self,
        h5_file: h5py.File,
        embeddings_array: np.ndarray,
    ) -> np.ndarray:
        """Convert float32 embeddings to configured storage precision.

        :param h5py.File h5_file: Open HDF5 cache handle.
        :param np.ndarray embeddings_array: Float32 embeddings.
        :return np.ndarray: Storage-ready embeddings.
        """
        if self.storage_precision == "float32":
            return np.asarray(embeddings_array, dtype=np.float32)
        if self.storage_precision == "float16":
            return np.asarray(embeddings_array, dtype=np.float16)

        ranges = self._ensure_calibration_ranges(
            h5_file,
            embedding_dim=int(embeddings_array.shape[1]),
            embeddings_array=embeddings_array,
        )
        quantize_embeddings = self._get_quantize_embeddings()
        int8_embeddings = quantize_embeddings(
            embeddings_array,
            precision="int8",
            ranges=ranges,
        )
        return np.asarray(int8_embeddings, dtype=np.int8)

    def _to_binary_embeddings(
        self, embeddings_array: np.ndarray
    ) -> Optional[np.ndarray]:
        """Create packed unsigned binary embeddings for Hamming prefiltering.

        :param np.ndarray embeddings_array: Float32 embeddings.
        :return Optional[np.ndarray]: Packed binary embeddings or ``None``.
        """
        if not self.binary_prefilter:
            return None

        return self._quantize_ubinary(embeddings_array)

    @staticmethod
    def _get_quantize_embeddings() -> Callable[..., np.ndarray]:
        """Import and return sentence-transformers quantization helper.

        :return Callable[..., np.ndarray]: ``quantize_embeddings`` function.
        :raises ImportError: If sentence-transformers is unavailable.
        """
        try:
            from sentence_transformers.quantization import quantize_embeddings
        except ImportError as exc:  # pragma: no cover - dependency wiring
            raise ImportError(
                "int8/binary embedding cache requires sentence-transformers. "
                "Install with: pip install citemesh[embeddings]"
            ) from exc

        return quantize_embeddings

    def _quantize_ubinary(self, embeddings_array: np.ndarray) -> np.ndarray:
        """Quantize embeddings to packed unsigned binary rows.

        Uses sentence-transformers quantization first, then falls back to
        ``np.packbits(axis=-1)`` for non-byte-aligned dimensions.

        :param np.ndarray embeddings_array: Float embedding matrix.
        :return np.ndarray: Packed unsigned binary matrix.
        """
        quantize_embeddings = self._get_quantize_embeddings()
        try:
            ubinary = quantize_embeddings(
                embeddings_array,
                precision="ubinary",
            )
            return np.asarray(ubinary, dtype=np.uint8)
        except ValueError:
            packed = np.packbits(np.asarray(embeddings_array) > 0, axis=-1)
            return np.asarray(packed, dtype=np.uint8)

    def _dequantize_int8(
        self, h5_file: h5py.File, int8_embeddings: np.ndarray
    ) -> np.ndarray:
        """Dequantize int8 embeddings to float32 using persisted ranges.

        :param h5py.File h5_file: Open HDF5 file handle.
        :param np.ndarray int8_embeddings: Int8 embeddings.
        :return np.ndarray: Dequantized float32 embeddings.
        """
        ranges_dataset = h5_file.get(CALIBRATION_RANGES_DATASET_NAME)
        if ranges_dataset is None:
            raise ValueError("Missing calibration_ranges dataset for int8 embeddings.")

        ranges = _sanitize_ranges(np.asarray(ranges_dataset, dtype=np.float32))
        starts = ranges[0]
        steps = (ranges[1] - ranges[0]) / 255.0

        float_values = int8_embeddings.astype(np.float32) + 128.0
        return starts + float_values * steps

    def _load_cached_embeddings(
        self,
        h5_file: h5py.File,
        dataset: h5py.Dataset,
        cached_rows: Sequence[Tuple[str, int]],
    ) -> Dict[str, np.ndarray]:
        """Load cached embeddings from matrix dataset in row-index order.

        :param h5py.File h5_file: Open HDF5 cache handle.
        :param h5py.Dataset dataset: Matrix dataset containing all embeddings.
        :param Sequence[Tuple[str, int]] cached_rows: Pairs of paper ID and row index.
        :return Dict[str, np.ndarray]: Mapping of paper IDs to float32 embeddings.
        """
        if not cached_rows:
            return {}

        sorted_rows = sorted(cached_rows, key=lambda item: item[1])
        indices = np.asarray([row_idx for _, row_idx in sorted_rows], dtype=np.int64)
        matrix = np.asarray(dataset[indices])

        if self.storage_precision == "int8":
            matrix_f32 = self._dequantize_int8(
                h5_file,
                matrix.astype(np.int8, copy=False),
            )
        elif self.storage_precision == "float16":
            matrix_f32 = matrix.astype(np.float32, copy=False)
        else:
            matrix_f32 = np.asarray(matrix, dtype=np.float32)

        return {
            paper_id: np.asarray(matrix_f32[idx], dtype=np.float32)
            for idx, (paper_id, _) in enumerate(sorted_rows)
        }

    def _binary_prefilter_rows(
        self,
        binary_dataset: h5py.Dataset,
        query_embedding: np.ndarray,
        candidate_count: int,
    ) -> np.ndarray:
        """Return top candidate row indices via Hamming prefiltering.

        :param h5py.Dataset binary_dataset: Packed binary corpus embeddings.
        :param np.ndarray query_embedding: Float32 query embedding.
        :param int candidate_count: Number of candidate rows to keep.
        :return np.ndarray: Candidate row indices.
        """
        row_count = int(binary_dataset.shape[0])
        if row_count == 0:
            return np.asarray([], dtype=np.int64)

        query_binary = np.asarray(
            self._quantize_ubinary(query_embedding.reshape(1, -1))[0],
            dtype=np.uint8,
        )

        keep_k = min(int(candidate_count), row_count)
        all_rows: List[np.ndarray] = []
        all_dists: List[np.ndarray] = []
        chunk_size = 65536

        for start in range(0, row_count, chunk_size):
            end = min(start + chunk_size, row_count)
            chunk = np.asarray(binary_dataset[start:end], dtype=np.uint8)
            xor = np.bitwise_xor(chunk, query_binary[None, :])
            dists = _POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int32)

            local_k = min(keep_k, dists.shape[0])
            if local_k == dists.shape[0]:
                local_idx = np.arange(dists.shape[0], dtype=np.int64)
            else:
                local_idx = np.argpartition(dists, local_k - 1)[:local_k]

            all_rows.append((start + local_idx).astype(np.int64, copy=False))
            all_dists.append(dists[local_idx])

        candidate_rows = np.concatenate(all_rows, axis=0)
        candidate_dists = np.concatenate(all_dists, axis=0)

        if candidate_rows.shape[0] > keep_k:
            top_idx = np.argpartition(candidate_dists, keep_k - 1)[:keep_k]
            candidate_rows = candidate_rows[top_idx]
            candidate_dists = candidate_dists[top_idx]

        order = np.lexsort((candidate_rows, candidate_dists))
        return candidate_rows[order]

    def _score_int8_rows(
        self,
        embeddings_dataset: h5py.Dataset,
        h5_file: h5py.File,
        query_embedding: np.ndarray,
        top_k: int,
        row_indices: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score int8 embeddings against a float32 query.

        :param h5py.Dataset embeddings_dataset: Int8 matrix dataset.
        :param h5py.File h5_file: Open HDF5 file handle.
        :param np.ndarray query_embedding: Float32 query embedding.
        :param int top_k: Top results to keep.
        :param Optional[np.ndarray] row_indices: Optional candidate subset.
        :return Tuple[np.ndarray, np.ndarray, np.ndarray]: Rows, scores, and embeddings.
        """
        if row_indices is not None:
            rows = np.asarray(row_indices, dtype=np.int64)
            if rows.size == 0:
                return (
                    np.asarray([], dtype=np.int64),
                    np.asarray([], dtype=np.float32),
                    np.empty((0, int(query_embedding.shape[0])), dtype=np.float32),
                )

            int8_matrix = np.asarray(embeddings_dataset[rows], dtype=np.int8)
            matrix = self._dequantize_int8(h5_file, int8_matrix)
            scores = matrix @ query_embedding
            return self._select_top_k(rows, scores, matrix, top_k)

        row_count = int(embeddings_dataset.shape[0])
        chunk_size = 65536
        best_rows = np.asarray([], dtype=np.int64)
        best_scores = np.asarray([], dtype=np.float32)
        best_embeddings = np.empty((0, int(query_embedding.shape[0])), dtype=np.float32)

        for start in range(0, row_count, chunk_size):
            end = min(start + chunk_size, row_count)
            int8_chunk = np.asarray(embeddings_dataset[start:end], dtype=np.int8)
            chunk_matrix = self._dequantize_int8(h5_file, int8_chunk)
            chunk_scores = chunk_matrix @ query_embedding
            chunk_rows = np.arange(start, end, dtype=np.int64)

            rows, scores, embeddings = self._select_top_k(
                chunk_rows,
                chunk_scores,
                chunk_matrix,
                top_k,
            )
            if rows.size == 0:
                continue

            merged_rows = np.concatenate((best_rows, rows), axis=0)
            merged_scores = np.concatenate((best_scores, scores), axis=0)
            merged_embeddings = np.concatenate((best_embeddings, embeddings), axis=0)
            best_rows, best_scores, best_embeddings = self._select_top_k(
                merged_rows,
                merged_scores,
                merged_embeddings,
                top_k,
            )

        return best_rows, best_scores, best_embeddings

    def _score_float_rows(
        self,
        embeddings_dataset: h5py.Dataset,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score float16/float32 embeddings against a float32 query.

        :param h5py.Dataset embeddings_dataset: Float matrix dataset.
        :param np.ndarray query_embedding: Float32 query embedding.
        :param int top_k: Top results to keep.
        :return Tuple[np.ndarray, np.ndarray, np.ndarray]: Rows, scores, and embeddings.
        """
        row_count = int(embeddings_dataset.shape[0])
        chunk_size = 65536

        best_rows = np.asarray([], dtype=np.int64)
        best_scores = np.asarray([], dtype=np.float32)
        best_embeddings = np.empty((0, int(query_embedding.shape[0])), dtype=np.float32)

        for start in range(0, row_count, chunk_size):
            end = min(start + chunk_size, row_count)
            chunk_matrix = np.asarray(
                embeddings_dataset[start:end],
                dtype=np.float32,
            )
            chunk_scores = chunk_matrix @ query_embedding
            chunk_rows = np.arange(start, end, dtype=np.int64)

            rows, scores, embeddings = self._select_top_k(
                chunk_rows,
                chunk_scores,
                chunk_matrix,
                top_k,
            )
            if rows.size == 0:
                continue

            merged_rows = np.concatenate((best_rows, rows), axis=0)
            merged_scores = np.concatenate((best_scores, scores), axis=0)
            merged_embeddings = np.concatenate((best_embeddings, embeddings), axis=0)
            best_rows, best_scores, best_embeddings = self._select_top_k(
                merged_rows,
                merged_scores,
                merged_embeddings,
                top_k,
            )

        return best_rows, best_scores, best_embeddings

    @staticmethod
    def _select_top_k(
        rows: np.ndarray,
        scores: np.ndarray,
        embeddings: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select top-k rows from scored embedding arrays.

        :param np.ndarray rows: Row-index vector.
        :param np.ndarray scores: Score vector.
        :param np.ndarray embeddings: Embedding matrix.
        :param int top_k: Number of rows to keep.
        :return Tuple[np.ndarray, np.ndarray, np.ndarray]: Top rows, scores, embeddings.
        """
        if rows.size == 0:
            return (
                np.asarray([], dtype=np.int64),
                np.asarray([], dtype=np.float32),
                np.empty((0, embeddings.shape[1]), dtype=np.float32),
            )

        keep_k = min(int(top_k), int(rows.size))
        if keep_k == rows.size:
            selected_idx = np.arange(rows.size, dtype=np.int64)
        else:
            selected_idx = np.argpartition(-scores, keep_k - 1)[:keep_k]

        selected_rows = rows[selected_idx]
        selected_scores = scores[selected_idx]
        selected_embeddings = embeddings[selected_idx]

        order = np.lexsort((selected_rows, -selected_scores))
        return (
            selected_rows[order].astype(np.int64, copy=False),
            selected_scores[order].astype(np.float32, copy=False),
            np.asarray(selected_embeddings[order], dtype=np.float32),
        )

    def _load_metadata_by_rows(
        self,
        conn: sqlite3.Connection,
        row_indices: Sequence[int],
    ) -> Dict[int, Dict[str, Any]]:
        """Load metadata rows keyed by embedding matrix row index.

        :param sqlite3.Connection conn: Open SQLite connection.
        :param Sequence[int] row_indices: Matrix row indices.
        :return Dict[int, Dict[str, Any]]: Metadata payloads keyed by row index.
        """
        if not row_indices:
            return {}

        output: Dict[int, Dict[str, Any]] = {}
        for chunk in _chunked(
            [str(idx) for idx in row_indices], SQLITE_QUERY_BATCH_SIZE
        ):
            numeric_chunk = [int(value) for value in chunk]
            placeholders = ",".join("?" for _ in numeric_chunk)
            query = (
                "SELECT paper_id, title, abstract, year, row_idx, authors_json, categories_json "
                f"FROM papers WHERE row_idx IN ({placeholders})"
            )
            for row in conn.execute(query, numeric_chunk):
                (
                    paper_id,
                    title,
                    abstract,
                    year,
                    row_idx,
                    authors_json,
                    categories_json,
                ) = row
                output[int(row_idx)] = {
                    "paper_id": str(paper_id),
                    "title": str(title or ""),
                    "abstract": str(abstract or ""),
                    "year": int(year) if year is not None else None,
                    "authors": _parse_json_list(authors_json),
                    "categories": _parse_json_list(categories_json),
                }

        return output

    @staticmethod
    def _text_hash(text: str) -> str:
        """Compute deterministic SHA-256 hash for text content.

        :param str text: Normalized paper text.
        :return str: Hexadecimal SHA-256 digest.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_text(metadata: Dict) -> str:
    """Compose paper text for embedding computation.

    :param Dict metadata: Paper metadata containing title and abstract.
    :return str: Concatenated title and abstract string.
    """
    title = metadata.get("title", "")
    abstract = metadata.get("abstract", "")
    return f"{title}. {abstract}".strip()


def _chunked(values: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    """Yield fixed-size chunks from a sequence.

    :param Sequence[str] values: Sequence to split into chunks.
    :param int chunk_size: Number of items per yielded chunk.
    :return Iterable[List[str]]: Iterator over chunk lists.
    """
    for start in range(0, len(values), chunk_size):
        yield list(values[start : start + chunk_size])
