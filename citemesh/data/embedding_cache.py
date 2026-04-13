"""Persistent embedding cache backed by SQLite metadata and HDF5 vectors."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
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
    Set,
    Tuple,
)

import h5py
import numpy as np
from filelock import FileLock, Timeout
from tqdm.auto import tqdm

from citemesh._runtime import stderr_isatty
from citemesh.text_batching import (
    encode_texts_in_length_buckets,
    l2_normalize_embeddings,
)

from .cache import format_bytes, get_cache_dir
from .model_profiles import DEFAULT_EMBEDDING_MODEL_NAME, compose_title_abstract_text

logger = logging.getLogger(__name__)


SQLITE_QUERY_BATCH_SIZE = 900
EMBEDDINGS_DATASET_NAME = "embeddings"
BINARY_INDEX_DATASET_NAME = "binary_index"
CALIBRATION_RANGES_DATASET_NAME = "calibration_ranges"
EMBEDDING_CACHE_SCHEMA_VERSION = 2
EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS = 900.0
EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR = "CITEMESH_EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS"
H5_LAYOUT_KEY = "h5_layout_version"
H5_LAYOUT_MATRIX_VERSION = "matrix-v2-quantized"
SCHEMA_VERSION_KEY = "schema_version"
STORAGE_PRECISION_KEY = "storage_precision"
SOURCE_TORCH_DTYPE_KEY = "source_torch_dtype"
EMBEDDING_VECTOR_DTYPE_KEY = "embedding_vector_dtype"
CALIBRATION_SAMPLE_SIZE_KEY = "calibration_sample_size"
BINARY_PREFILTER_ENABLED_KEY = "binary_prefilter_enabled"
COMPRESSION_FILTER_KEY = "compression_filter"
COMPRESSION_LEVEL_KEY = "compression_level"
HYDRATION_DATASET_SOURCE_KEY = "hydration_dataset_source"
HYDRATION_SPLIT_KEY = "hydration_split"
HYDRATION_CORPUS_SIZE_KEY = "hydration_corpus_size"
HYDRATION_COMPLETE_KEY = "hydration_complete"
HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY = "hydration_reconciled_upstream_rows"
HYDRATION_RECONCILED_CACHE_ROWS_KEY = "hydration_reconciled_cache_rows"
MODEL_FINGERPRINT_KEY = "model_fingerprint"
TEXT_FORMATTER_FINGERPRINT_KEY = "text_formatter_fingerprint"
INT8_CLIPPED_VALUE_COUNT_KEY = "int8_clipped_value_count"
INT8_TOTAL_VALUE_COUNT_KEY = "int8_total_value_count"

_STORAGE_PRECISIONS = {"float32", "float16", "int8"}
_COMPRESSION_FILTERS = {"gzip", "lzf"}
_COMPRESSION_FILTER_IDS = {
    "gzip": h5py.h5z.FILTER_DEFLATE,
    "lzf": h5py.h5z.FILTER_LZF,
}
_POPCOUNT_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
    axis=1
)
EMBEDDING_DATASET_CHUNK_ROWS = 2048
INT8_SATURATION_WARN_RATIO = 0.005


def _resolve_cache_lock_timeout_seconds() -> float:
    """Resolve cache lock timeout from env var with safe fallback.

    :return float: Lock timeout in seconds.
    """
    raw_value = os.getenv(EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR)
    if raw_value is None or not str(raw_value).strip():
        return EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS

    try:
        parsed = float(raw_value)
    except ValueError:
        logger.warning(
            "Invalid %s value %r; falling back to default %.1fs.",
            EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR,
            raw_value,
            EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS,
        )
        return EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS

    if not np.isfinite(parsed) or parsed <= 0:
        logger.warning(
            "Non-positive/invalid %s value %r; falling back to default %.1fs.",
            EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR,
            raw_value,
            EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS,
        )
        return EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS
    return parsed


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


def validate_compression_filter(compression: str) -> str:
    """Validate and normalize HDF5 compression filter names.

    :param str compression: Requested HDF5 compression filter token.
    :return str: Normalized lowercase compression token.
    :raises ValueError: If filter name is unsupported or unavailable at runtime.
    """
    normalized = str(compression or "").strip().lower()
    if normalized == "szip":
        raise ValueError(
            "compression='szip' is unsupported; HDF5 szip requires codec-specific "
            "options that are not exposed by current cache settings. Use 'gzip' or "
            "'lzf'."
        )
    if normalized not in _COMPRESSION_FILTERS:
        expected = ", ".join(sorted(_COMPRESSION_FILTERS))
        raise ValueError(
            f"compression must be one of {{{expected}}}, got {compression!r}."
        )

    filter_id = _COMPRESSION_FILTER_IDS[normalized]
    if not bool(h5py.h5z.filter_avail(filter_id)):
        raise ValueError(
            f"compression filter {normalized!r} is unavailable in this h5py runtime."
        )
    return normalized


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


def _count_int8_saturated_values(
    embeddings_array: np.ndarray,
    ranges: np.ndarray,
) -> tuple[int, int]:
    """Count values that fall outside persisted int8 calibration ranges.

    :param np.ndarray embeddings_array: Float32 embedding matrix being quantized.
    :param np.ndarray ranges: Persisted ``(2, dim)`` calibration ranges.
    :return tuple[int, int]: ``(clipped_values, total_values)``.
    """
    normalized = np.asarray(embeddings_array, dtype=np.float32)
    sanitized_ranges = _sanitize_ranges(ranges)
    mins = sanitized_ranges[0][None, :]
    maxs = sanitized_ranges[1][None, :]
    clipped = np.logical_or(normalized < mins, normalized > maxs)
    return int(np.count_nonzero(clipped)), int(normalized.size)


def _as_float32_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Normalize raw embeddings into a 2D float32 matrix.

    :param np.ndarray embeddings: Raw embedding vector or matrix.
    :return np.ndarray: Float32 matrix with shape ``(rows, dim)``.
    :raises ValueError: If embeddings are already quantized or not 1D/2D.
    """
    array = np.asarray(embeddings)
    if array.dtype in (np.int8, np.uint8):
        raise ValueError("Embeddings to quantize must use a floating dtype.")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise ValueError(
            f"Embeddings to quantize must be 1D or 2D, got shape {array.shape}."
        )
    return np.asarray(array, dtype=np.float32)


def _quantize_int8_embeddings(embeddings: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Quantize float embeddings into signed int8 rows with explicit clipping.

    The cache already owns calibration persistence, saturation reporting, and
    dequantization. Keeping the forward quantizer local avoids coupling the
    default embedding/hybrid path to sentence-transformers' internal layout.

    :param np.ndarray embeddings: Float embedding vector or matrix.
    :param np.ndarray ranges: Persisted ``(2, dim)`` calibration ranges.
    :return np.ndarray: Int8 embedding matrix.
    """
    matrix = _as_float32_embedding_matrix(embeddings)
    sanitized_ranges = _sanitize_ranges(ranges)
    starts = sanitized_ranges[0][None, :]
    steps = ((sanitized_ranges[1] - sanitized_ranges[0]) / 255.0)[None, :]
    scaled = (matrix - starts) / steps - 128.0
    return np.clip(scaled, -128.0, 127.0).astype(np.int8)


def _quantize_ubinary_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Pack embedding sign bits into unsigned bytes for Hamming prefiltering.

    :param np.ndarray embeddings: Float embedding vector or matrix.
    :return np.ndarray: Packed unsigned binary embedding matrix.
    """
    matrix = _as_float32_embedding_matrix(embeddings)
    return np.asarray(np.packbits(matrix > 0, axis=-1), dtype=np.uint8)


@dataclass(frozen=True)
class CacheSearchResult:
    """Search result returned by ``EmbeddingCache.search``."""

    paper_id: str
    score: float
    embedding: np.ndarray
    metadata: Dict[str, Any]
    embedding_dtype: str = "float32"
    storage_precision: str = "float32"


@dataclass(frozen=True)
class CacheNamespacePayloadStats:
    """Namespace payload summary used for clear-impact reporting."""

    file_count: int
    size_bytes: int
    sqlite_rows: int
    embedding_rows: int
    hydration_complete: bool
    hydration_split: Optional[str]
    hydration_corpus_size: Optional[str]
    hydration_dataset_source: Optional[str]


@dataclass(frozen=True)
class PendingEmbeddingRecord:
    """Cache-miss record staged across lookup/encode/commit phases."""

    paper_id: str
    metadata: Dict[str, object]
    text_hash: str
    text: str
    row_idx: Optional[int]


@dataclass(frozen=True)
class EmbeddingCacheUpsertStats:
    """Summary of cache-write activity for hydration-only embedding upserts."""

    requested: int
    cache_hits: int
    encoded: int
    race_reused: int


class EmbeddingCache:
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
        :param str storage_precision: Persistent embedding precision ``float32``/``float16``/``int8``.
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
        self.source_torch_dtype = str(source_torch_dtype or "float32")
        self.text_formatter_fingerprint = str(text_formatter_fingerprint).strip()
        if not self.text_formatter_fingerprint:
            raise ValueError("text_formatter_fingerprint must be a non-empty string.")
        self.embedding_vector_dtype = "float32"
        self.last_search_used_binary_prefilter: Optional[bool] = None
        self.last_search_total_embeddings: Optional[int] = None
        self.last_search_rescored_embeddings: Optional[int] = None

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
    ) -> EmbeddingCacheUpsertStats:
        """Persist embeddings for papers without materializing float32 return payloads.

        :param Dict[str, Dict] papers: Mapping of paper ID to metadata payload.
        :param Any model: SentenceTransformer-compatible model exposing ``encode``.
        :param int batch_size: Batch size for model encoding.
        :param bool show_progress: Whether to display progress bars.
        :param Optional[Callable[[Dict[str, object]], str]] text_builder: Optional metadata->text formatter.
        :return EmbeddingCacheUpsertStats: Summary of hit/miss/write activity.
        """
        result = self._process_embeddings(
            papers,
            model,
            batch_size=batch_size,
            show_progress=show_progress,
            text_builder=text_builder,
            return_embeddings=False,
        )
        assert isinstance(result, EmbeddingCacheUpsertStats)
        return result

    def _process_embeddings(
        self,
        papers: Dict[str, Dict],
        model: Any,
        *,
        batch_size: int,
        show_progress: bool,
        text_builder: Optional[Callable[[Dict[str, object]], str]],
        return_embeddings: bool,
    ) -> Dict[str, np.ndarray] | EmbeddingCacheUpsertStats:
        """Hydrate cache entries and optionally materialize float32 embeddings.

        :param Dict[str, Dict] papers: Mapping of paper ID to metadata payload.
        :param Any model: SentenceTransformer-compatible model exposing ``encode``.
        :param int batch_size: Batch size for model encoding.
        :param bool show_progress: Whether to display progress bars.
        :param Optional[Callable[[Dict[str, object]], str]] text_builder: Optional metadata->text formatter.
        :param bool return_embeddings: Whether to return float32 embedding payloads.
        :return Dict[str, np.ndarray] | EmbeddingCacheUpsertStats: Embedding map or write summary.
        """
        if not papers:
            if return_embeddings:
                return {}
            return EmbeddingCacheUpsertStats(
                requested=0,
                cache_hits=0,
                encoded=0,
                race_reused=0,
            )

        cached_embeddings: Dict[str, np.ndarray] = {}
        papers_to_embed: List[PendingEmbeddingRecord] = []
        cached_rows: List[Tuple[str, int]] = []
        cache_hit_count = 0
        builder = text_builder or compose_title_abstract_text

        items = list(papers.items())
        progress_enabled = show_progress and stderr_isatty() and len(items) > 50
        iterator: Iterable[Tuple[str, Dict]] = tqdm(
            items,
            desc="Checking cache",
            unit="papers",
            disable=not progress_enabled,
        )

        calibration_ranges: Optional[np.ndarray] = None
        with (
            self._cache_lock(),
            self._connect_db() as conn,
            h5py.File(self.h5_path, "a") as h5,
        ):
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
                text_hash = self._metadata_hash(metadata, text)
                existing_row = existing_rows.get(paper_id)
                row_idx = existing_row["row_idx"] if existing_row is not None else None

                if (
                    existing_row is not None
                    and existing_row["text_hash"] == text_hash
                    and row_idx is not None
                    and embeddings_dataset is not None
                    and 0 <= row_idx < cached_limit
                ):
                    cache_hit_count += 1
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
                return EmbeddingCacheUpsertStats(
                    requested=len(items),
                    cache_hits=cache_hit_count,
                    encoded=0,
                    race_reused=0,
                )

            if self.storage_precision == "int8":
                calibration_ranges = self._require_calibration_ranges(h5_file=h5)

        texts = [record.text for record in papers_to_embed]
        embeddings_array = encode_texts_in_length_buckets(
            texts,
            batch_size=min(int(batch_size), len(texts)),
            show_progress_bar=show_progress,
            encode_batch=lambda batch_texts, batch_progress: np.asarray(
                model.encode(
                    batch_texts,
                    batch_size=min(int(batch_size), len(batch_texts)),
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=batch_progress,
                ),
                dtype=np.float32,
            ),
        )
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        if embeddings_array.shape[0] != len(papers_to_embed):
            raise ValueError(
                "Embedding model returned unexpected row count: "
                f"{embeddings_array.shape[0]} for {len(papers_to_embed)} papers."
            )

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
            storage_embeddings = self._to_int8_embeddings(
                embeddings_array=embeddings_array,
                ranges=calibration_ranges,
            )
        else:
            storage_embeddings = np.asarray(
                embeddings_array,
                dtype=_storage_dtype_for_precision(self.storage_precision),
            )
        binary_embeddings = self._to_binary_embeddings(embeddings_array)

        race_reused_count = 0
        with (
            self._cache_lock(),
            self._connect_db() as conn,
            h5py.File(self.h5_path, "a") as h5,
        ):
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

            existing_row_count = int(embeddings_dataset.shape[0])
            latest_rows = self._load_existing_rows(
                conn, [record.paper_id for record in papers_to_embed]
            )
            new_embeddings: Dict[str, np.ndarray] = {}
            rows_to_upsert: List[Tuple[Any, ...]] = []
            append_embeddings: List[np.ndarray] = []
            append_binary_embeddings: List[np.ndarray] = []
            append_records: List[Tuple[str, Dict[str, object], str]] = []

            for idx, record in enumerate(papers_to_embed):
                embedding = embeddings_array[idx]
                storage_embedding = storage_embeddings[idx]
                binary_embedding = (
                    None if binary_embeddings is None else binary_embeddings[idx]
                )
                if return_embeddings:
                    new_embeddings[record.paper_id] = embedding
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
                    race_reused_count += 1
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

                if (
                    latest_row_idx is not None
                    and 0 <= latest_row_idx < existing_row_count
                ):
                    embeddings_dataset[latest_row_idx] = storage_embedding
                    if binary_dataset is not None and binary_embedding is not None:
                        binary_dataset[latest_row_idx] = binary_embedding
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
                    (paper_id, title, abstract, year, text_hash, embedding_dim, row_idx,
                     authors_json, categories_json, venue, arxiv_id, doi)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows_to_upsert,
                )

        if return_embeddings:
            return {**cached_embeddings, **new_embeddings}

        return EmbeddingCacheUpsertStats(
            requested=len(items),
            cache_hits=cache_hit_count,
            encoded=len(papers_to_embed),
            race_reused=race_reused_count,
        )

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
        if not self.h5_path.exists():
            return []

        with (
            self._cache_lock(),
            self._connect_db() as conn,
            h5py.File(self.h5_path, "r") as h5,
        ):
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
                        embedding_dtype=self.embedding_vector_dtype,
                        storage_precision=self.storage_precision,
                    )
                )

            results.sort(key=lambda item: (-item.score, str(item.paper_id)))
            return results[:top_k]

    def has_cached_payload(self) -> bool:
        """Return whether namespace contains any cached embedding payload rows.

        :return bool: ``True`` when cache has at least one SQLite/HDF5 embedding row.
        """
        if not self.db_path.exists():
            return False

        try:
            with self._cache_lock(), self._connect_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                paper_rows = int(cursor.fetchone()[0])
                if paper_rows > 0:
                    return True

                if not self.h5_path.exists():
                    return False
                with h5py.File(self.h5_path, "r") as h5:
                    dataset = self._get_embeddings_dataset(h5)
                    if dataset is None:
                        return False
                    return int(dataset.shape[0]) > 0
        except (OSError, sqlite3.DatabaseError, ValueError):
            return False

    def get_cached_paper_ids(self) -> Set[str]:
        """Return all cached paper IDs for this namespace.

        :return Set[str]: Cached paper IDs loaded from SQLite metadata rows.
        """
        if not self.db_path.exists():
            return set()

        paper_ids: Set[str] = set()
        with self._cache_lock(), self._connect_db() as conn:
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
        h5_file.attrs[INT8_CLIPPED_VALUE_COUNT_KEY] = previous_clipped + int(
            clipped_value_count
        )
        h5_file.attrs[INT8_TOTAL_VALUE_COUNT_KEY] = previous_total + int(
            total_value_count
        )

        saturation_ratio = float(clipped_value_count) / float(total_value_count)
        if saturation_ratio >= INT8_SATURATION_WARN_RATIO:
            logger.warning(
                "Int8 calibration saturation detected for %s: %.2f%% of values were clipped in this write. Rebuild calibration ranges if recall degrades.",
                self.model_name,
                saturation_ratio * 100.0,
            )

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

        if not self.h5_path.exists() or not self.db_path.exists():
            return False

        try:
            with (
                self._cache_lock(),
                self._connect_db() as conn,
                h5py.File(self.h5_path, "r") as h5,
            ):
                metadata = self._load_cache_metadata(conn)
                metadata_matches = (
                    metadata.get(HYDRATION_COMPLETE_KEY, "0") == "1"
                    and metadata.get(HYDRATION_SPLIT_KEY) == expected_split
                    and metadata.get(HYDRATION_CORPUS_SIZE_KEY) == expected_corpus_size
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
        except (OSError, ValueError, RuntimeError, sqlite3.DatabaseError):
            return False

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

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM papers")
        total_rows = int(cursor.fetchone()[0])
        if total_rows != row_count:
            return False

        cursor.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT row_idx)
            FROM papers
            WHERE row_idx IS NOT NULL
              AND row_idx >= 0
              AND row_idx < ?
            """,
            (row_count,),
        )
        valid_rows, distinct_rows = cursor.fetchone()
        if int(valid_rows) != row_count or int(distinct_rows) != row_count:
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
                conn, HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY, str(resolved_upstream)
            )
            self._set_cache_metadata(
                conn, HYDRATION_RECONCILED_CACHE_ROWS_KEY, str(resolved_cached)
            )

    def clear_hydration_rowcount_reconciliation(self) -> None:
        """Clear persisted row-count reconciliation marker metadata.

        :return None: Mutates SQLite metadata in-place.
        """
        with self._cache_lock(), self._connect_db() as conn:
            self._set_cache_metadata(conn, HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY, "")
            self._set_cache_metadata(conn, HYDRATION_RECONCILED_CACHE_ROWS_KEY, "")

    def payload_stats(self) -> CacheNamespacePayloadStats:
        """Return a summary of cache payload currently stored for this namespace.

        :return CacheNamespacePayloadStats: File/row/hydration stats snapshot.
        """
        with self._cache_lock():
            return self._collect_namespace_payload_stats_locked()

    def set_model_fingerprint(self, fingerprint: str) -> None:
        """Persist model fingerprint for cache invalidation guardrails.

        :param str fingerprint: Deterministic model fingerprint token.
        :return None: Mutates SQLite metadata in-place.
        """
        normalized = str(fingerprint).strip()
        if not normalized:
            raise ValueError("fingerprint must be a non-empty string.")
        with self._cache_lock(), self._connect_db() as conn:
            self._set_cache_metadata(conn, MODEL_FINGERPRINT_KEY, normalized)

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
                conn, HYDRATION_DATASET_SOURCE_KEY, normalized_source
            )
            self._set_cache_metadata(conn, HYDRATION_SPLIT_KEY, str(dataset_split))
            self._set_cache_metadata(
                conn,
                HYDRATION_CORPUS_SIZE_KEY,
                _corpus_size_token(corpus_size),
            )
            self._set_cache_metadata(
                conn, HYDRATION_COMPLETE_KEY, "1" if complete else "0"
            )
            self._set_cache_metadata(conn, HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY, "")
            self._set_cache_metadata(conn, HYDRATION_RECONCILED_CACHE_ROWS_KEY, "")

    def clear(self, reason: Optional[str] = None) -> None:
        """Purge cache artifacts for this cache namespace.

        :param Optional[str] reason: Optional rationale for the clear operation.
        :return None: Removes namespace payload files and reinitializes metadata.
        """
        normalized_reason = str(reason).strip() or "unspecified"
        with self._cache_lock():
            stats = self._collect_namespace_payload_stats_locked()
            if (
                stats.file_count > 0
                or stats.sqlite_rows > 0
                or stats.embedding_rows > 0
            ):
                logger.warning(
                    "Clearing embedding cache namespace '%s' (reason=%s, files=%d, "
                    "size=%s, sqlite_rows=%d, embedding_rows=%d, hydrated=%s, "
                    "split=%s, corpus=%s, source=%s).",
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
                logger.info(
                    "Embedding cache namespace '%s' is already empty (reason=%s).",
                    self.model_name,
                    normalized_reason,
                )
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
        timeout_seconds = _resolve_cache_lock_timeout_seconds()
        lock = FileLock(str(self.lock_path), timeout=timeout_seconds)
        try:
            with lock:
                yield
        except Timeout as exc:
            raise TimeoutError(
                "Timed out waiting for embedding cache lock "
                f"at {self.lock_path} after {timeout_seconds:.3f}s. "
                "Another process may be holding it. "
                f"Increase {EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR} or set "
                "CITEMESH_CACHE_DIR to an isolated per-run cache root."
            ) from exc

    @contextmanager
    def _connect_db(self) -> Iterator[sqlite3.Connection]:
        """Open a SQLite connection that is closed on context exit.

        ``sqlite3.connect()`` as a context manager only commits/rollbacks —
        it does not close the connection.  On Windows the unclosed handle
        prevents file deletion or replacement, causing ``[WinError 32]``.

        :return Iterator[sqlite3.Connection]: Context manager yielding an open connection.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create and initialize the metadata cache schema when needed."""
        with self._connect_db() as conn:
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
                    venue TEXT,
                    arxiv_id TEXT,
                    doi TEXT,
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
            if "venue" not in columns:
                conn.execute("ALTER TABLE papers ADD COLUMN venue TEXT")
            if "arxiv_id" not in columns:
                conn.execute("ALTER TABLE papers ADD COLUMN arxiv_id TEXT")
            if "doi" not in columns:
                conn.execute("ALTER TABLE papers ADD COLUMN doi TEXT")

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
                conn, EMBEDDING_VECTOR_DTYPE_KEY, self.embedding_vector_dtype
            )
            self._set_cache_metadata(
                conn,
                TEXT_FORMATTER_FINGERPRINT_KEY,
                self.text_formatter_fingerprint,
            )
            self._set_cache_metadata(
                conn, CALIBRATION_SAMPLE_SIZE_KEY, str(self.calibration_sample_size)
            )
            self._set_cache_metadata(
                conn,
                BINARY_PREFILTER_ENABLED_KEY,
                "1" if self.binary_prefilter else "0",
            )
            self._set_cache_metadata(conn, COMPRESSION_FILTER_KEY, self.compression)
            self._set_cache_metadata(
                conn, COMPRESSION_LEVEL_KEY, str(self.compression_level)
            )
            # Preserve hydration completion across restarts; initialize only once.
            self._set_cache_metadata_default(conn, HYDRATION_COMPLETE_KEY, "0")
            self._set_cache_metadata_default(
                conn, HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY, ""
            )
            self._set_cache_metadata_default(
                conn, HYDRATION_RECONCILED_CACHE_ROWS_KEY, ""
            )
            self._set_cache_metadata_default(conn, MODEL_FINGERPRINT_KEY, "")

    def _collect_namespace_payload_stats_locked(self) -> CacheNamespacePayloadStats:
        """Collect namespace payload stats while cache lock is held.

        :return CacheNamespacePayloadStats: Snapshot of files/rows/hydration metadata.
        """
        file_count = 0
        size_bytes = 0
        for payload_path in (self.db_path, self.h5_path):
            if not payload_path.exists() or not payload_path.is_file():
                continue
            file_count += 1
            try:
                size_bytes += int(payload_path.stat().st_size)
            except OSError:
                continue

        sqlite_rows = 0
        hydration_complete = False
        hydration_split: Optional[str] = None
        hydration_corpus_size: Optional[str] = None
        hydration_dataset_source: Optional[str] = None
        if self.db_path.exists():
            try:
                with self._connect_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM papers")
                    sqlite_rows = int(cursor.fetchone()[0])
                    metadata = self._load_cache_metadata(conn)
                    hydration_complete = (
                        metadata.get(HYDRATION_COMPLETE_KEY, "0") == "1"
                    )
                    hydration_split = (
                        str(metadata.get(HYDRATION_SPLIT_KEY, "")).strip() or None
                    )
                    hydration_corpus_size = (
                        str(metadata.get(HYDRATION_CORPUS_SIZE_KEY, "")).strip() or None
                    )
                    hydration_dataset_source = (
                        str(metadata.get(HYDRATION_DATASET_SOURCE_KEY, "")).strip()
                        or None
                    )
            except (OSError, sqlite3.DatabaseError):
                sqlite_rows = 0

        embedding_rows = 0
        if self.h5_path.exists():
            try:
                with h5py.File(self.h5_path, "r") as h5:
                    embeddings = self._get_embeddings_dataset(h5)
                    if embeddings is not None:
                        embedding_rows = int(embeddings.shape[0])
            except (OSError, ValueError):
                embedding_rows = 0

        return CacheNamespacePayloadStats(
            file_count=file_count,
            size_bytes=size_bytes,
            sqlite_rows=sqlite_rows,
            embedding_rows=embedding_rows,
            hydration_complete=hydration_complete,
            hydration_split=hydration_split,
            hydration_corpus_size=hydration_corpus_size,
            hydration_dataset_source=hydration_dataset_source,
        )

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

    @staticmethod
    def _metadata_value_from_h5_attr(value: Any) -> str:
        """Normalize HDF5 attribute values to comparable metadata strings.

        :param Any value: Raw HDF5 attribute payload.
        :return str: Normalized string representation.
        """
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.decode("utf-8", errors="replace")
        if isinstance(value, np.generic):
            value = value.item()
        return str(value)

    def _expected_runtime_metadata(self) -> Dict[str, str]:
        """Return metadata key/value pairs that govern runtime cache semantics.

        :return Dict[str, str]: Expected metadata mapping for this namespace instance.
        """
        expected = {
            SCHEMA_VERSION_KEY: str(EMBEDDING_CACHE_SCHEMA_VERSION),
            H5_LAYOUT_KEY: H5_LAYOUT_MATRIX_VERSION,
            STORAGE_PRECISION_KEY: self.storage_precision,
            SOURCE_TORCH_DTYPE_KEY: self.source_torch_dtype,
            EMBEDDING_VECTOR_DTYPE_KEY: self.embedding_vector_dtype,
            TEXT_FORMATTER_FINGERPRINT_KEY: self.text_formatter_fingerprint,
            BINARY_PREFILTER_ENABLED_KEY: "1" if self.binary_prefilter else "0",
            COMPRESSION_FILTER_KEY: self.compression,
            COMPRESSION_LEVEL_KEY: str(self.compression_level),
        }
        if self.storage_precision == "int8":
            expected[CALIBRATION_SAMPLE_SIZE_KEY] = str(self.calibration_sample_size)
        return expected

    def _assert_runtime_cache_consistency(
        self,
        conn: sqlite3.Connection,
        h5_file: h5py.File,
        embeddings_dataset: h5py.Dataset,
        *,
        fail_mode: str,
    ) -> None:
        """Assert that metadata/attrs/datasets agree on active runtime semantics.

        :param sqlite3.Connection conn: Open SQLite connection for metadata table.
        :param h5py.File h5_file: Open HDF5 cache handle.
        :param h5py.Dataset embeddings_dataset: Open embeddings matrix dataset.
        :param str fail_mode: ``"runtime"`` to fail-closed, ``"repair"`` to signal rebuild.
        :return None: Raises when metadata and payload state diverge.
        :raises RuntimeError: If inconsistency is found and ``fail_mode == "runtime"``.
        :raises ValueError: If inconsistency is found and ``fail_mode == "repair"``.
        """
        expected = self._expected_runtime_metadata()
        metadata = self._load_cache_metadata(conn)

        def _fail(message: str) -> None:
            """Raise consistency error using mode-specific exception semantics."""
            detail = (
                "Embedding cache integrity error: "
                f"{message}. Rebuild this cache namespace to restore consistency."
            )
            if fail_mode == "runtime":
                raise RuntimeError(detail)
            if fail_mode == "repair":
                raise ValueError(detail)
            raise ValueError(f"Unknown fail_mode={fail_mode!r}")

        for key, expected_value in expected.items():
            actual_value = metadata.get(key)
            if actual_value != expected_value:
                _fail(
                    f"metadata key {key!r} mismatch "
                    f"({actual_value!r} != {expected_value!r})"
                )

        h5_schema = self._metadata_value_from_h5_attr(
            h5_file.attrs.get(SCHEMA_VERSION_KEY)
        )
        if h5_schema != expected[SCHEMA_VERSION_KEY]:
            _fail(
                f"HDF5 attr {SCHEMA_VERSION_KEY!r} mismatch "
                f"({h5_schema!r} != {expected[SCHEMA_VERSION_KEY]!r})"
            )

        h5_layout = self._metadata_value_from_h5_attr(h5_file.attrs.get(H5_LAYOUT_KEY))
        if h5_layout != expected[H5_LAYOUT_KEY]:
            _fail(
                f"HDF5 attr {H5_LAYOUT_KEY!r} mismatch "
                f"({h5_layout!r} != {expected[H5_LAYOUT_KEY]!r})"
            )

        h5_precision = self._metadata_value_from_h5_attr(
            h5_file.attrs.get(STORAGE_PRECISION_KEY)
        )
        if h5_precision != expected[STORAGE_PRECISION_KEY]:
            _fail(
                f"HDF5 attr {STORAGE_PRECISION_KEY!r} mismatch "
                f"({h5_precision!r} != {expected[STORAGE_PRECISION_KEY]!r})"
            )

        h5_source_dtype = self._metadata_value_from_h5_attr(
            h5_file.attrs.get(SOURCE_TORCH_DTYPE_KEY)
        )
        if h5_source_dtype != expected[SOURCE_TORCH_DTYPE_KEY]:
            _fail(
                f"HDF5 attr {SOURCE_TORCH_DTYPE_KEY!r} mismatch "
                f"({h5_source_dtype!r} != {expected[SOURCE_TORCH_DTYPE_KEY]!r})"
            )

        h5_embedding_dtype = self._metadata_value_from_h5_attr(
            h5_file.attrs.get(EMBEDDING_VECTOR_DTYPE_KEY)
        )
        if h5_embedding_dtype != expected[EMBEDDING_VECTOR_DTYPE_KEY]:
            _fail(
                f"HDF5 attr {EMBEDDING_VECTOR_DTYPE_KEY!r} mismatch "
                f"({h5_embedding_dtype!r} != {expected[EMBEDDING_VECTOR_DTYPE_KEY]!r})"
            )

        h5_text_formatter_fingerprint = self._metadata_value_from_h5_attr(
            h5_file.attrs.get(TEXT_FORMATTER_FINGERPRINT_KEY)
        )
        if h5_text_formatter_fingerprint != expected[TEXT_FORMATTER_FINGERPRINT_KEY]:
            _fail(
                f"HDF5 attr {TEXT_FORMATTER_FINGERPRINT_KEY!r} mismatch "
                f"({h5_text_formatter_fingerprint!r} != "
                f"{expected[TEXT_FORMATTER_FINGERPRINT_KEY]!r})"
            )

        h5_binary_prefilter = self._metadata_value_from_h5_attr(
            h5_file.attrs.get(BINARY_PREFILTER_ENABLED_KEY)
        )
        if h5_binary_prefilter != expected[BINARY_PREFILTER_ENABLED_KEY]:
            _fail(
                f"HDF5 attr {BINARY_PREFILTER_ENABLED_KEY!r} mismatch "
                f"({h5_binary_prefilter!r} != {expected[BINARY_PREFILTER_ENABLED_KEY]!r})"
            )

        h5_compression_filter = self._metadata_value_from_h5_attr(
            h5_file.attrs.get(COMPRESSION_FILTER_KEY)
        )
        if h5_compression_filter != expected[COMPRESSION_FILTER_KEY]:
            _fail(
                f"HDF5 attr {COMPRESSION_FILTER_KEY!r} mismatch "
                f"({h5_compression_filter!r} != {expected[COMPRESSION_FILTER_KEY]!r})"
            )

        h5_compression_level = self._metadata_value_from_h5_attr(
            h5_file.attrs.get(COMPRESSION_LEVEL_KEY)
        )
        if h5_compression_level != expected[COMPRESSION_LEVEL_KEY]:
            _fail(
                f"HDF5 attr {COMPRESSION_LEVEL_KEY!r} mismatch "
                f"({h5_compression_level!r} != {expected[COMPRESSION_LEVEL_KEY]!r})"
            )

        if CALIBRATION_SAMPLE_SIZE_KEY in expected:
            h5_calibration_sample_size = self._metadata_value_from_h5_attr(
                h5_file.attrs.get(CALIBRATION_SAMPLE_SIZE_KEY)
            )
            if h5_calibration_sample_size != expected[CALIBRATION_SAMPLE_SIZE_KEY]:
                _fail(
                    f"HDF5 attr {CALIBRATION_SAMPLE_SIZE_KEY!r} mismatch "
                    f"({h5_calibration_sample_size!r} != "
                    f"{expected[CALIBRATION_SAMPLE_SIZE_KEY]!r})"
                )

        target_dtype = _storage_dtype_for_precision(self.storage_precision)
        if np.dtype(embeddings_dataset.dtype) != np.dtype(target_dtype):
            _fail(
                f"embeddings dataset dtype mismatch "
                f"({embeddings_dataset.dtype} != {target_dtype})"
            )
        if str(embeddings_dataset.compression or "") != self.compression:
            _fail(
                "embeddings dataset compression mismatch "
                f"({embeddings_dataset.compression!r} != {self.compression!r})"
            )
        if self.compression != "lzf":
            actual_compression_level = embeddings_dataset.compression_opts
            if int(actual_compression_level) != int(self.compression_level):
                _fail(
                    "embeddings dataset compression level mismatch "
                    f"({actual_compression_level!r} != {self.compression_level!r})"
                )

        row_count = int(embeddings_dataset.shape[0])
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM papers")
        paper_rows = int(cursor.fetchone()[0])
        if paper_rows != row_count:
            _fail(
                "embedding row mapping mismatch "
                f"(metadata rows={paper_rows}, embedding rows={row_count})"
            )

        cursor.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT row_idx)
            FROM papers
            WHERE row_idx IS NOT NULL
              AND row_idx >= 0
              AND row_idx < ?
            """,
            (row_count,),
        )
        valid_rows, distinct_rows = cursor.fetchone()
        if int(valid_rows) != row_count or int(distinct_rows) != row_count:
            _fail(
                "embedding row_idx coverage mismatch "
                f"(valid={int(valid_rows)}, distinct={int(distinct_rows)}, expected={row_count})"
            )

    def _reconcile_layout_metadata(self, conn: sqlite3.Connection) -> None:
        """Write metadata keys for active matrix-layout state.

        :param sqlite3.Connection conn: Open SQLite connection.
        :return None: This method mutates DB state in-place.
        """
        self._set_cache_metadata(conn, H5_LAYOUT_KEY, H5_LAYOUT_MATRIX_VERSION)

    def _ensure_h5_layout(self) -> None:
        """Ensure cache file uses matrix-based HDF5 layout."""
        if not self.h5_path.exists():
            with self._connect_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                paper_rows = int(cursor.fetchone()[0])
                metadata = self._load_cache_metadata(conn)
                has_hydration_markers = bool(
                    str(metadata.get(HYDRATION_COMPLETE_KEY, "0")) == "1"
                    or str(metadata.get(HYDRATION_DATASET_SOURCE_KEY, "")).strip()
                    or str(metadata.get(HYDRATION_SPLIT_KEY, "")).strip()
                    or str(metadata.get(HYDRATION_CORPUS_SIZE_KEY, "")).strip()
                )
                if paper_rows > 0 or has_hydration_markers:
                    logger.warning(
                        "Embedding matrix %s is missing; clearing stale SQLite metadata "
                        "and hydration markers.",
                        self.h5_path,
                    )
                    conn.execute("DELETE FROM papers")
                    self._reset_hydration_metadata(conn)
                self._reconcile_layout_metadata(conn)
                return

        try:
            with (
                self._connect_db() as conn,
                h5py.File(self.h5_path, "a") as h5,
            ):
                dataset = h5.get(EMBEDDINGS_DATASET_NAME)
                if dataset is None or dataset.ndim != 2:
                    raise ValueError("incompatible embedding cache layout")
                target_dtype = _storage_dtype_for_precision(self.storage_precision)
                if np.dtype(dataset.dtype) != np.dtype(target_dtype):
                    raise ValueError(
                        "incompatible embedding cache dtype: "
                        f"{dataset.dtype} != {target_dtype}"
                    )

                embedding_dim = int(dataset.shape[1])
                embedding_rows = int(dataset.shape[0])
                if self.storage_precision == "int8":
                    ranges = h5.get(CALIBRATION_RANGES_DATASET_NAME)
                    if ranges is None:
                        raise ValueError(
                            "incompatible int8 embedding cache: missing calibration ranges"
                        )
                    if (
                        ranges.ndim != 2
                        or int(ranges.shape[0]) != 2
                        or int(ranges.shape[1]) != embedding_dim
                    ):
                        raise ValueError(
                            "incompatible int8 embedding cache: calibration range shape "
                            f"{ranges.shape} for embedding dim {embedding_dim}"
                        )

                self._assert_runtime_cache_consistency(
                    conn=conn,
                    h5_file=h5,
                    embeddings_dataset=dataset,
                    fail_mode="repair",
                )

                binary_dataset = h5.get(BINARY_INDEX_DATASET_NAME)
                if (
                    binary_dataset is not None
                    and not self._is_binary_dataset_compatible(
                        binary_dataset=binary_dataset,
                        embedding_dim=embedding_dim,
                        embedding_rows=embedding_rows,
                    )
                ):
                    logger.warning(
                        "Binary index dataset in %s is incompatible with embedding "
                        "matrix shape; dropping stale binary index.",
                        self.h5_path,
                    )
                    del h5[BINARY_INDEX_DATASET_NAME]
                self._set_h5_attrs(h5)
        except (OSError, ValueError, sqlite3.DatabaseError):
            logger.warning(
                "Embedding cache %s is incompatible with current schema. "
                "Clearing namespace cache and rebuilding.",
                self.h5_path,
            )
            self.h5_path.unlink(missing_ok=True)
            with self._connect_db() as conn:
                conn.execute("DELETE FROM papers")
                self._reset_hydration_metadata(conn)
                self._reconcile_layout_metadata(conn)
            return

        with self._connect_db() as conn:
            self._reconcile_layout_metadata(conn)

    @staticmethod
    def _reset_hydration_metadata(conn: sqlite3.Connection) -> None:
        """Reset hydration metadata keys to an incomplete state.

        :param sqlite3.Connection conn: Open SQLite connection.
        :return None: Mutates metadata table in-place.
        """
        EmbeddingCache._set_cache_metadata(conn, HYDRATION_DATASET_SOURCE_KEY, "")
        EmbeddingCache._set_cache_metadata(conn, HYDRATION_SPLIT_KEY, "")
        EmbeddingCache._set_cache_metadata(conn, HYDRATION_CORPUS_SIZE_KEY, "")
        EmbeddingCache._set_cache_metadata(conn, HYDRATION_COMPLETE_KEY, "0")
        EmbeddingCache._set_cache_metadata(
            conn, HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY, ""
        )
        EmbeddingCache._set_cache_metadata(
            conn, HYDRATION_RECONCILED_CACHE_ROWS_KEY, ""
        )

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
        (
            title,
            abstract,
            year,
            authors_json,
            categories_json,
            venue,
            arxiv_id,
            doi,
        ) = EmbeddingCache._normalized_metadata_fields(metadata)

        return (
            paper_id,
            title,
            abstract,
            year,
            text_hash,
            embedding_dim,
            row_idx,
            authors_json,
            categories_json,
            venue,
            arxiv_id,
            doi,
        )

    @staticmethod
    def _normalized_metadata_fields(
        metadata: Dict[str, object],
    ) -> Tuple[str, str, Optional[int], str, str, str, str, str]:
        """Normalize metadata fields to stable cache representations.

        :param Dict[str, object] metadata: Paper metadata payload.
        :return Tuple[str, str, Optional[int], str, str, str, str, str]:
            Normalized title/abstract/year/authors/categories and venue/arXiv/DOI fields.
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

        authors_json = _safe_json_list(metadata.get("authors", []))
        categories_json = _safe_json_list(metadata.get("categories", []))
        venue = str(metadata.get("venue", "") or "").strip()
        arxiv_id = str(metadata.get("arxiv_id", "") or "").strip()
        doi = str(metadata.get("doi", "") or "").strip()
        return (
            title,
            abstract,
            year,
            authors_json,
            categories_json,
            venue,
            arxiv_id,
            doi,
        )

    @staticmethod
    def _metadata_fields_changed(
        existing_row: Dict[str, Any], metadata: Dict[str, object]
    ) -> bool:
        """Return whether cached non-vector metadata differs from incoming payload.

        :param Dict[str, Any] existing_row: Existing cached metadata row.
        :param Dict[str, object] metadata: Incoming metadata payload.
        :return bool: ``True`` when a metadata-only refresh is required.
        """
        current = (
            str(existing_row.get("title", "") or ""),
            str(existing_row.get("abstract", "") or ""),
            (
                int(existing_row["year"])
                if existing_row.get("year") is not None
                else None
            ),
            _safe_json_list(_parse_json_list(existing_row.get("authors_json"))),
            _safe_json_list(_parse_json_list(existing_row.get("categories_json"))),
            str(existing_row.get("venue", "") or "").strip(),
            str(existing_row.get("arxiv_id", "") or "").strip(),
            str(existing_row.get("doi", "") or "").strip(),
        )
        expected = EmbeddingCache._normalized_metadata_fields(metadata)
        return current != expected

    @staticmethod
    def _metadata_refresh_tuple(
        paper_id: str, metadata: Dict[str, object]
    ) -> Tuple[Any, ...]:
        """Build SQL update tuple for metadata-only refresh paths.

        :param str paper_id: Paper identifier.
        :param Dict[str, object] metadata: Incoming metadata payload.
        :return Tuple[Any, ...]: Tuple for metadata UPDATE query.
        """
        (
            title,
            abstract,
            year,
            authors_json,
            categories_json,
            venue,
            arxiv_id,
            doi,
        ) = EmbeddingCache._normalized_metadata_fields(metadata)
        return (
            title,
            abstract,
            year,
            authors_json,
            categories_json,
            venue,
            arxiv_id,
            doi,
            paper_id,
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
        h5_file.attrs[EMBEDDING_VECTOR_DTYPE_KEY] = self.embedding_vector_dtype
        h5_file.attrs[TEXT_FORMATTER_FINGERPRINT_KEY] = self.text_formatter_fingerprint
        h5_file.attrs[CALIBRATION_SAMPLE_SIZE_KEY] = int(self.calibration_sample_size)
        h5_file.attrs[BINARY_PREFILTER_ENABLED_KEY] = int(self.binary_prefilter)
        h5_file.attrs[COMPRESSION_FILTER_KEY] = self.compression
        h5_file.attrs[COMPRESSION_LEVEL_KEY] = int(self.compression_level)

    def _load_existing_rows(
        self,
        conn: sqlite3.Connection,
        paper_ids: Sequence[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch existing metadata rows for target paper IDs.

        :param sqlite3.Connection conn: Open SQLite connection.
        :param Sequence[str] paper_ids: Paper IDs to look up.
        :return Dict[str, Dict[str, Any]]: Mapping of paper ID to cached metadata payload.
        """
        if not paper_ids:
            return {}

        existing_rows: Dict[str, Dict[str, Any]] = {}
        for id_chunk in _chunked(paper_ids, SQLITE_QUERY_BATCH_SIZE):
            placeholders = ",".join("?" for _ in id_chunk)
            query = (
                "SELECT paper_id, text_hash, row_idx, title, abstract, year, "
                "authors_json, categories_json, venue, arxiv_id, doi "
                f"FROM papers WHERE paper_id IN ({placeholders})"
            )

            for row in conn.execute(query, id_chunk):
                (
                    paper_id,
                    text_hash,
                    row_idx,
                    title,
                    abstract,
                    year,
                    authors_json,
                    categories_json,
                    venue,
                    arxiv_id,
                    doi,
                ) = row
                existing_rows[str(paper_id)] = {
                    "text_hash": str(text_hash),
                    "row_idx": int(row_idx) if row_idx is not None else None,
                    "title": str(title or ""),
                    "abstract": str(abstract or ""),
                    "year": int(year) if year is not None else None,
                    "authors_json": str(authors_json or ""),
                    "categories_json": str(categories_json or ""),
                    "venue": str(venue or ""),
                    "arxiv_id": str(arxiv_id or ""),
                    "doi": str(doi or ""),
                }

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
        chunk_rows = EMBEDDING_DATASET_CHUNK_ROWS

        if dataset is None:
            compression_kwargs = self._dataset_compression_kwargs()
            return h5_file.create_dataset(
                EMBEDDINGS_DATASET_NAME,
                shape=(0, embedding_dim),
                maxshape=(None, embedding_dim),
                dtype=target_dtype,
                chunks=(chunk_rows, embedding_dim),
                shuffle=True,
                **compression_kwargs,
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
        chunk_rows = EMBEDDING_DATASET_CHUNK_ROWS
        dataset = h5_file.get(BINARY_INDEX_DATASET_NAME)
        if dataset is None:
            compression_kwargs = self._dataset_compression_kwargs()
            return h5_file.create_dataset(
                BINARY_INDEX_DATASET_NAME,
                shape=(0, packed_dim),
                maxshape=(None, packed_dim),
                dtype=np.uint8,
                chunks=(chunk_rows, packed_dim),
                shuffle=True,
                **compression_kwargs,
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

    @staticmethod
    def _is_binary_dataset_compatible(
        binary_dataset: h5py.Dataset,
        embedding_dim: int,
        embedding_rows: int,
    ) -> bool:
        """Return whether binary dataset shape aligns with embedding matrix.

        :param h5py.Dataset binary_dataset: Binary index dataset.
        :param int embedding_dim: Embedding vector dimension.
        :param int embedding_rows: Number of embedding rows in matrix dataset.
        :return bool: ``True`` when binary index shape is compatible.
        """
        if binary_dataset.ndim != 2:
            return False
        expected_cols = (int(embedding_dim) + 7) // 8
        if int(binary_dataset.shape[1]) != expected_cols:
            return False
        if int(binary_dataset.shape[0]) != int(embedding_rows):
            return False
        return True

    def _dataset_compression_kwargs(self) -> Dict[str, Any]:
        """Build HDF5 dataset compression kwargs for active cache configuration.

        ``lzf`` does not accept ``compression_opts``. Other configured codecs keep
        the numeric level behavior used by existing cache settings.

        :return Dict[str, Any]: Keyword args passed into ``create_dataset``.
        """
        compression = str(self.compression or "").strip()
        if not compression:
            return {}

        kwargs: Dict[str, Any] = {"compression": compression}
        if compression.lower() != "lzf":
            kwargs["compression_opts"] = int(self.compression_level)
        return kwargs

    def _require_calibration_ranges(
        self,
        h5_file: h5py.File,
        embedding_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Load persisted int8 calibration ranges or fail closed.

        Calibration is intentionally explicit at the strategy layer. Bootstrapping
        ranges from whichever request batch happens to arrive first makes the
        namespace path-dependent and can silently skew later quantization quality.

        :param h5py.File h5_file: Open HDF5 cache handle.
        :param Optional[int] embedding_dim: Expected embedding dimension, when known.
        :return np.ndarray: Calibration ranges with shape ``(2, dim)``.
        """
        if self.storage_precision != "int8":
            raise RuntimeError("Calibration ranges are only valid for int8 storage")

        existing = h5_file.get(CALIBRATION_RANGES_DATASET_NAME)
        if existing is None:
            raise RuntimeError(
                "Missing persisted int8 calibration ranges. "
                "Hydrate through EmbeddingGraphBuilder or call "
                "EmbeddingCache.set_calibration_ranges(...) before int8 writes."
            )

        if existing.ndim != 2 or existing.shape[0] != 2:
            raise ValueError(
                f"Calibration ranges dataset must have shape (2, dim), got {existing.shape}."
            )
        if embedding_dim is not None and int(existing.shape[1]) != int(embedding_dim):
            raise ValueError(
                "Calibration range dimension mismatch in cache: "
                f"{int(existing.shape[1])} != {int(embedding_dim)}"
            )

        return _sanitize_ranges(np.asarray(existing, dtype=np.float32))

    def _to_int8_embeddings(
        self, embeddings_array: np.ndarray, ranges: np.ndarray
    ) -> np.ndarray:
        """Quantize float32 embeddings into int8 storage rows.

        :param np.ndarray embeddings_array: Float32 embeddings.
        :param np.ndarray ranges: Persisted calibration ranges.
        :return np.ndarray: Int8 storage matrix.
        """
        return _quantize_int8_embeddings(embeddings_array, ranges)

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

    def _quantize_ubinary(self, embeddings_array: np.ndarray) -> np.ndarray:
        """Quantize embeddings to packed unsigned binary rows.

        :param np.ndarray embeddings_array: Float embedding matrix.
        :return np.ndarray: Packed unsigned binary matrix.
        """
        return _quantize_ubinary_embeddings(embeddings_array)

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
            matrix_f32 = l2_normalize_embeddings(matrix_f32)
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
            # Choose a deterministic top-k candidate set by (distance, row_idx).
            ranked_idx = np.lexsort((candidate_rows, candidate_dists))
            candidate_rows = candidate_rows[ranked_idx[:keep_k]]

        # HDF5 fancy indexing requires monotonically increasing integer indices.
        return np.sort(candidate_rows.astype(np.int64, copy=False))

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
        query = l2_normalize_embeddings(query_embedding)
        if row_indices is not None:
            rows = np.unique(np.asarray(row_indices, dtype=np.int64))
            if rows.size == 0:
                return (
                    np.asarray([], dtype=np.int64),
                    np.asarray([], dtype=np.float32),
                    np.empty((0, int(query_embedding.shape[0])), dtype=np.float32),
                )

            int8_matrix = np.asarray(embeddings_dataset[rows], dtype=np.int8)
            matrix = l2_normalize_embeddings(
                self._dequantize_int8(h5_file, int8_matrix)
            )
            scores = matrix @ query
            return self._select_top_k(rows, scores, matrix, top_k)

        row_count = int(embeddings_dataset.shape[0])
        chunk_size = 65536
        best_rows = np.asarray([], dtype=np.int64)
        best_scores = np.asarray([], dtype=np.float32)
        best_embeddings = np.empty((0, int(query.shape[0])), dtype=np.float32)

        for start in range(0, row_count, chunk_size):
            end = min(start + chunk_size, row_count)
            int8_chunk = np.asarray(embeddings_dataset[start:end], dtype=np.int8)
            chunk_matrix = l2_normalize_embeddings(
                self._dequantize_int8(h5_file, int8_chunk)
            )
            chunk_scores = chunk_matrix @ query
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
                "SELECT paper_id, title, abstract, year, row_idx, "
                "authors_json, categories_json, venue, arxiv_id, doi "
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
                    venue,
                    arxiv_id,
                    doi,
                ) = row
                output[int(row_idx)] = {
                    "paper_id": str(paper_id),
                    "title": str(title or ""),
                    "abstract": str(abstract or ""),
                    "year": int(year) if year is not None else None,
                    "authors": _parse_json_list(authors_json),
                    "categories": _parse_json_list(categories_json),
                    "venue": str(venue or ""),
                    "arxiv_id": str(arxiv_id or ""),
                    "doi": str(doi or ""),
                }

        return output

    @staticmethod
    def _metadata_hash(metadata: Dict[str, object], text: str) -> str:
        """Compute deterministic invalidation hash for embedding model input text.

        :param Dict[str, object] metadata: Paper metadata payload.
        :param str text: Normalized paper text used for embedding.
        :return str: Hexadecimal SHA-256 digest.
        """
        del metadata
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _chunked(values: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    """Yield fixed-size chunks from a sequence.

    :param Sequence[str] values: Sequence to split into chunks.
    :param int chunk_size: Number of items per yielded chunk.
    :return Iterable[List[str]]: Iterator over chunk lists.
    """
    for start in range(0, len(values), chunk_size):
        yield list(values[start : start + chunk_size])
