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
BINARY_INDEX_ENCODING_KEY = "encoding"
BINARY_INDEX_ENCODING = "int8-midpoint-sign-v1"
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
CORPUS_METADATA_VERSION_KEY = "corpus_metadata_version"
CORPUS_METADATA_VERSION = "1"
HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY = "hydration_reconciled_upstream_rows"
HYDRATION_RECONCILED_CACHE_ROWS_KEY = "hydration_reconciled_cache_rows"
MODEL_FINGERPRINT_KEY = "model_fingerprint"
TEXT_FORMATTER_FINGERPRINT_KEY = "text_formatter_fingerprint"
INT8_CLIPPED_VALUE_COUNT_KEY = "int8_clipped_value_count"
INT8_TOTAL_VALUE_COUNT_KEY = "int8_total_value_count"

_STORAGE_PRECISIONS = {"float32", "int8"}
_COMPRESSION_FILTERS = {"gzip", "lzf"}
_COMPRESSION_FILTER_IDS = {
    "gzip": h5py.h5z.FILTER_DEFLATE,
    "lzf": h5py.h5z.FILTER_LZF,
}
_POPCOUNT_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
    axis=1
)
EMBEDDING_DATASET_CHUNK_ROWS = 2048
EMBEDDING_SEARCH_CHUNK_ROWS = 65536
INT8_SATURATION_WARN_RATIO = 0.005
_PAPER_ROW_COLUMNS = (
    "paper_id, text_hash, row_idx, title, abstract, year, authors_json, "
    "categories_json, venue, arxiv_id, doi"
)
_PAPER_ROW_LOOKUP_COLUMNS = {"paper_id", "row_idx"}


class _EmbeddingCacheLayoutError(ValueError):
    """A proven persisted layout mismatch that requires a namespace rebuild."""


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

    Capped tokens carry the slice policy (``newest:N``) so caches hydrated
    under the legacy head-slice policy (bare ``N``) fail ``is_hydrated`` and
    rehydrate instead of silently serving the oldest records.

    :param Optional[int] corpus_size: Optional corpus-size cap.
    :return str: Tokenized corpus-size value.
    """
    return "all" if corpus_size is None else f"newest:{int(corpus_size)}"


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
    if normalized.ndim != 2 or normalized.shape[0] != 2:
        raise ValueError(
            f"Calibration ranges must have shape (2, dim), got {normalized.shape}."
        )
    if normalized.shape[1] < 1:
        raise ValueError("Calibration ranges must cover at least one dimension.")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Calibration ranges must contain only finite values.")

    mins = normalized[0]
    maxs = normalized[1]
    if np.any(maxs < mins):
        raise ValueError(
            "Calibration range maxima must be greater than or equal to minima."
        )
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
    # Select unsigned buckets before the signed offset so every bucket has equal width.
    buckets = np.clip(np.floor((matrix - starts) / steps), 0.0, 255.0)
    return (buckets - 128.0).astype(np.int8)


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
        """Hydrate cache entries and optionally materialize float32 embeddings.

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
            binary_embeddings = (
                _quantize_ubinary_embeddings(
                    self._dequantize_int8(h5, storage_embeddings)
                )
                if self.binary_prefilter
                else None
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
        return None

    def embedding_count(self) -> int:
        """Return the number of embeddings persisted in this cache namespace.

        Cheap availability probe (no model load, no SQLite access) used to
        decide whether local semantic search has anything to rank.

        :return int: Persisted embedding row count (``0`` for a missing or
            empty cache).
        """
        if not self.h5_path.exists():
            return 0
        with self._cache_lock(), h5py.File(self.h5_path, "r") as h5:
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
        normalized_reason = str(reason).strip() or "unspecified"
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

            conn.execute("DROP INDEX IF EXISTS idx_papers_text_hash")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_row_idx ON papers(row_idx)"
            )
            conn.execute(_metadata_table_create_sql())

            runtime_values = self._runtime_contract_values()
            self._set_cache_metadata(
                conn,
                {
                    key: value
                    for key, value in runtime_values.items()
                    if key not in {COMPRESSION_FILTER_KEY, COMPRESSION_LEVEL_KEY}
                },
            )
            # Physical compression and hydration state survive process restarts.
            self._set_cache_metadata(
                conn,
                {
                    COMPRESSION_FILTER_KEY: runtime_values[COMPRESSION_FILTER_KEY],
                    COMPRESSION_LEVEL_KEY: runtime_values[COMPRESSION_LEVEL_KEY],
                    HYDRATION_COMPLETE_KEY: "0",
                    HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY: "",
                    HYDRATION_RECONCILED_CACHE_ROWS_KEY: "",
                    MODEL_FINGERPRINT_KEY: "",
                },
                preserve_existing=True,
            )

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
    def _set_cache_metadata(
        conn: sqlite3.Connection,
        values: Dict[str, object],
        *,
        preserve_existing: bool = False,
    ) -> None:
        """Write a group of cache metadata values with one conflict policy.

        :param sqlite3.Connection conn: Open SQLite connection.
        :param Dict[str, object] values: Metadata values keyed by cache field name.
        :param bool preserve_existing: Insert only missing keys when ``True``.
        :return None: This method mutates DB state in-place.
        """
        if not values:
            return
        statement = (
            "INSERT OR IGNORE INTO cache_metadata (key, value) VALUES (?, ?)"
            if preserve_existing
            else """
                INSERT INTO cache_metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """
        )
        conn.executemany(
            statement,
            ((str(key), str(value)) for key, value in values.items()),
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

    def _runtime_contract_values(self) -> Dict[str, object]:
        """Return the canonical values shared by SQLite and HDF5 metadata.

        :return Dict[str, object]: Runtime cache-contract values keyed by field name.
        """
        return {
            SCHEMA_VERSION_KEY: EMBEDDING_CACHE_SCHEMA_VERSION,
            H5_LAYOUT_KEY: H5_LAYOUT_MATRIX_VERSION,
            STORAGE_PRECISION_KEY: self.storage_precision,
            SOURCE_TORCH_DTYPE_KEY: self.source_torch_dtype,
            EMBEDDING_VECTOR_DTYPE_KEY: self.embedding_vector_dtype,
            TEXT_FORMATTER_FINGERPRINT_KEY: self.text_formatter_fingerprint,
            CALIBRATION_SAMPLE_SIZE_KEY: self.calibration_sample_size,
            BINARY_PREFILTER_ENABLED_KEY: int(self.binary_prefilter),
            COMPRESSION_FILTER_KEY: self._effective_compression,
            COMPRESSION_LEVEL_KEY: self._effective_compression_level,
        }

    def _assert_runtime_cache_consistency(
        self,
        conn: sqlite3.Connection,
        h5_file: h5py.File,
        embeddings_dataset: Optional[h5py.Dataset],
        *,
        fail_mode: str,
    ) -> None:
        """Assert that metadata/attrs/datasets agree on active runtime semantics.

        :param sqlite3.Connection conn: Open SQLite connection for metadata table.
        :param h5py.File h5_file: Open HDF5 cache handle.
        :param Optional[h5py.Dataset] embeddings_dataset: Matrix dataset, or None before the first write.
        :param str fail_mode: ``"runtime"`` to fail-closed, ``"repair"`` to rebuild incompatible layouts.
        :return None: Raises when metadata and payload state diverge.
        :raises RuntimeError: If row mappings are inconsistent or runtime checks fail.
        :raises ValueError: If a layout mismatch is found in repair mode.
        """
        expected = {
            key: self._metadata_value_from_h5_attr(value)
            for key, value in self._runtime_contract_values().items()
        }
        if self.storage_precision != "int8":
            expected.pop(CALIBRATION_SAMPLE_SIZE_KEY)
        if embeddings_dataset is None:
            for key in (
                COMPRESSION_FILTER_KEY,
                COMPRESSION_LEVEL_KEY,
                BINARY_PREFILTER_ENABLED_KEY,
            ):
                expected.pop(key)
        metadata = self._load_cache_metadata(conn)

        def _fail(message: str, *, layout_mismatch: bool = True) -> None:
            """Raise a layout or row-mapping consistency error.

            :param str message: Observed inconsistency.
            :param bool layout_mismatch: Whether the persisted layout needs rebuilding.
            :return None: Always raises the mode-appropriate exception.
            """
            detail = (
                "Embedding cache integrity error: "
                f"{message}. Rebuild this cache namespace to restore consistency."
            )
            if fail_mode == "runtime" or not layout_mismatch:
                raise RuntimeError(detail)
            if fail_mode == "repair":
                raise _EmbeddingCacheLayoutError(detail)
            raise ValueError(f"Unknown fail_mode={fail_mode!r}")

        for key, expected_value in expected.items():
            actual_value = metadata.get(key)
            if actual_value != expected_value:
                _fail(
                    f"metadata key {key!r} mismatch "
                    f"({actual_value!r} != {expected_value!r})"
                )
            h5_value = self._metadata_value_from_h5_attr(h5_file.attrs.get(key))
            if h5_value != expected_value:
                _fail(
                    f"HDF5 attr {key!r} mismatch ({h5_value!r} != {expected_value!r})"
                )

        if embeddings_dataset is None:
            return

        target_dtype = _storage_dtype_for_precision(self.storage_precision)
        if np.dtype(embeddings_dataset.dtype) != np.dtype(target_dtype):
            _fail(
                f"embeddings dataset dtype mismatch "
                f"({embeddings_dataset.dtype} != {target_dtype})"
            )
        if str(embeddings_dataset.compression or "") != self._effective_compression:
            _fail(
                "embeddings dataset compression mismatch "
                f"({embeddings_dataset.compression!r} != "
                f"{self._effective_compression!r})"
            )
        if self._effective_compression != "lzf":
            actual_compression_level = embeddings_dataset.compression_opts
            if int(actual_compression_level) != int(self._effective_compression_level):
                _fail(
                    "embeddings dataset compression level mismatch "
                    f"({actual_compression_level!r} != "
                    f"{self._effective_compression_level!r})"
                )

        row_count = int(embeddings_dataset.shape[0])
        if fail_mode == "runtime":
            # Separate MIN/MAX subqueries each use the row_idx index endpoint.
            # Full coverage is checked once on open, not for every hydration batch.
            first_row, last_row = conn.execute(
                "SELECT (SELECT MIN(row_idx) FROM papers), "
                "(SELECT MAX(row_idx) FROM papers)"
            ).fetchone()
            expected_bounds = (0, row_count - 1) if row_count else (None, None)
            if (first_row, last_row) != expected_bounds:
                _fail(
                    "embedding row mapping mismatch "
                    f"(metadata bounds={(first_row, last_row)}, expected={expected_bounds})",
                    layout_mismatch=False,
                )
            return

        paper_rows = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        if paper_rows != row_count:
            _fail(
                "embedding row mapping mismatch "
                f"(metadata rows={paper_rows}, embedding rows={row_count})",
                layout_mismatch=False,
            )
        valid_rows, distinct_rows = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT row_idx)
            FROM papers
            WHERE row_idx IS NOT NULL
              AND row_idx >= 0
              AND row_idx < ?
            """,
            (row_count,),
        ).fetchone()
        if int(valid_rows) != row_count or int(distinct_rows) != row_count:
            _fail(
                "embedding row_idx coverage mismatch "
                f"(valid={int(valid_rows)}, distinct={int(distinct_rows)}, expected={row_count})",
                layout_mismatch=False,
            )

    def _recover_trailing_rows(
        self,
        conn: sqlite3.Connection,
        h5_file: h5py.File,
        embeddings_dataset: h5py.Dataset,
    ) -> int:
        """Preserve the shared row prefix after an interrupted append.

        :param sqlite3.Connection conn: Open SQLite connection with committed mappings.
        :param h5py.File h5_file: Open HDF5 cache handle.
        :param h5py.Dataset embeddings_dataset: Resizable embeddings matrix dataset.
        :return int: Embedding row count after any recoverable truncation.
        """
        embedding_rows = int(embeddings_dataset.shape[0])
        paper_rows = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        if embedding_rows == paper_rows:
            return embedding_rows

        valid_rows, distinct_rows, minimum_row, maximum_row = conn.execute(
            """
            SELECT COUNT(row_idx), COUNT(DISTINCT row_idx), MIN(row_idx), MAX(row_idx)
            FROM papers
            """,
        ).fetchone()
        committed_prefix_is_complete = bool(
            int(valid_rows) == paper_rows
            and int(distinct_rows) == paper_rows
            and (
                paper_rows == 0
                or (int(minimum_row) == 0 and int(maximum_row) == paper_rows - 1)
            )
        )
        if not committed_prefix_is_complete:
            return embedding_rows

        if paper_rows > embedding_rows:
            logger.warning(
                "Recovering embedding cache %s by removing %d trailing SQLite "
                "mapping(s) without persisted vectors; preserving %d row(s).",
                self.h5_path,
                paper_rows - embedding_rows,
                embedding_rows,
            )
            conn.execute("DELETE FROM papers WHERE row_idx >= ?", (embedding_rows,))
            self._set_cache_metadata(conn, {HYDRATION_COMPLETE_KEY: "0"})
            return embedding_rows

        orphan_rows = embedding_rows - paper_rows
        logger.warning(
            "Recovering embedding cache %s by truncating %d uncommitted trailing "
            "HDF5 row(s); preserving %d committed row(s).",
            self.h5_path,
            orphan_rows,
            paper_rows,
        )
        embeddings_dataset.resize((paper_rows, int(embeddings_dataset.shape[1])))

        binary_dataset = h5_file.get(BINARY_INDEX_DATASET_NAME)
        if (
            binary_dataset is not None
            and binary_dataset.ndim == 2
            and int(binary_dataset.shape[0]) > paper_rows
        ):
            try:
                binary_dataset.resize((paper_rows, int(binary_dataset.shape[1])))
            except (OSError, TypeError, ValueError):
                del h5_file[BINARY_INDEX_DATASET_NAME]

        return paper_rows

    def _reset_effective_compression(self) -> None:
        """Reset physical-layout settings to the originally requested codec.

        :return None: Updates effective compression state in-place.
        """
        self._effective_compression = self.compression
        self._effective_compression_level = self.compression_level

    def _persist_physical_layout_metadata(self, conn: sqlite3.Connection) -> None:
        """Persist effective HDF5 physical-layout settings in SQLite metadata.

        :param sqlite3.Connection conn: Open SQLite connection.
        :return None: Updates compression metadata in-place.
        """
        self._set_cache_metadata(
            conn,
            {
                H5_LAYOUT_KEY: H5_LAYOUT_MATRIX_VERSION,
                COMPRESSION_FILTER_KEY: self._effective_compression,
                COMPRESSION_LEVEL_KEY: self._effective_compression_level,
            },
        )

    def _adopt_existing_dataset_compression(self, dataset: h5py.Dataset) -> None:
        """Adopt immutable compression layout from an existing embedding matrix.

        Compression changes storage layout but not embedding semantics. A requested
        codec therefore applies only when a matrix is first created or rebuilt.

        :param h5py.Dataset dataset: Existing embeddings matrix.
        :return None: Updates effective compression state in-place.
        :raises ValueError: If the persisted physical layout is unsupported.
        """
        compression = str(dataset.compression or "").strip().lower()
        if compression not in _COMPRESSION_FILTERS:
            raise _EmbeddingCacheLayoutError(
                "incompatible embedding cache compression filter: "
                f"{dataset.compression!r}"
            )

        if compression == "lzf":
            if dataset.compression_opts is not None:
                raise _EmbeddingCacheLayoutError(
                    "incompatible lzf embedding cache compression options: "
                    f"{dataset.compression_opts!r}"
                )
            compression_level = 0
        else:
            compression_options = dataset.compression_opts
            if compression_options is None:
                raise _EmbeddingCacheLayoutError(
                    "incompatible gzip embedding cache: missing compression level"
                )
            compression_level = int(compression_options)

        if (
            compression != self.compression
            or compression_level != self.compression_level
        ):
            logger.info(
                "Using existing embedding cache compression %s level %d for %s; "
                "requested %s level %d applies after the cache is rebuilt.",
                compression,
                compression_level,
                self.h5_path,
                self.compression,
                self.compression_level,
            )
        self._effective_compression = compression
        self._effective_compression_level = compression_level

    def _ensure_h5_layout(self) -> None:
        """Recover interrupted appends and rebuild proven incompatible layouts.

        :return None: Preserves valid payloads and propagates IO/open failures.
        """
        if not self.h5_path.exists():
            with self._connect_db() as conn:
                self._reset_effective_compression()
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
                self._persist_physical_layout_metadata(conn)
                return

        try:
            with (
                self._connect_db() as conn,
                h5py.File(self.h5_path, "a") as h5,
            ):
                dataset = self._get_embeddings_dataset(h5)
                if dataset is None:
                    if (
                        self.storage_precision == "int8"
                        and set(h5) == {CALIBRATION_RANGES_DATASET_NAME}
                        and conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
                        == 0
                    ):
                        self._require_calibration_ranges(h5)
                        self._assert_runtime_cache_consistency(
                            conn=conn,
                            h5_file=h5,
                            embeddings_dataset=None,
                            fail_mode="repair",
                        )
                        self._persist_physical_layout_metadata(conn)
                        return
                    raise _EmbeddingCacheLayoutError(
                        "incompatible embedding cache layout"
                    )
                self._adopt_existing_dataset_compression(dataset)

                embedding_dim = int(dataset.shape[1])
                if self.storage_precision == "int8":
                    if CALIBRATION_RANGES_DATASET_NAME not in h5:
                        raise _EmbeddingCacheLayoutError(
                            "missing int8 calibration ranges"
                        )
                    self._require_calibration_ranges(h5, embedding_dim)

                # Prefilter state is auxiliary; toggling it does not change vector rows.
                h5.attrs.modify(
                    BINARY_PREFILTER_ENABLED_KEY, int(self.binary_prefilter)
                )
                embedding_rows = self._recover_trailing_rows(
                    conn=conn,
                    h5_file=h5,
                    embeddings_dataset=dataset,
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

                self._ensure_binary_dataset(h5, embedding_dim)
        except _EmbeddingCacheLayoutError as exc:
            logger.warning(
                "Embedding cache %s is incompatible with current schema. "
                "Clearing namespace cache and rebuilding: %s",
                self.h5_path,
                exc,
            )
            self.h5_path.unlink(missing_ok=True)
            self._reset_effective_compression()
            with self._connect_db() as conn:
                conn.execute("DELETE FROM papers")
                self._reset_hydration_metadata(conn)
                self._persist_physical_layout_metadata(conn)
            return

        with self._connect_db() as conn:
            self._set_cache_metadata(conn, {H5_LAYOUT_KEY: H5_LAYOUT_MATRIX_VERSION})

    @staticmethod
    def _reset_hydration_metadata(conn: sqlite3.Connection) -> None:
        """Reset hydration metadata keys to an incomplete state.

        :param sqlite3.Connection conn: Open SQLite connection.
        :return None: Mutates metadata table in-place.
        """
        EmbeddingCache._set_cache_metadata(
            conn,
            {
                HYDRATION_DATASET_SOURCE_KEY: "",
                HYDRATION_SPLIT_KEY: "",
                HYDRATION_CORPUS_SIZE_KEY: "",
                HYDRATION_COMPLETE_KEY: "0",
                CORPUS_METADATA_VERSION_KEY: "",
                HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY: "",
                HYDRATION_RECONCILED_CACHE_ROWS_KEY: "",
            },
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
        title, abstract, year, *remaining = EmbeddingCache._normalized_metadata_fields(
            metadata
        )
        return (
            paper_id,
            title,
            abstract,
            year,
            text_hash,
            embedding_dim,
            row_idx,
            *remaining,
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
        return (*EmbeddingCache._normalized_metadata_fields(metadata), paper_id)

    def _set_h5_attrs(self, h5_file: h5py.File) -> None:
        """Write schema/layout metadata attrs to an open HDF5 file.

        :param h5py.File h5_file: Open cache file handle.
        :return None: Mutates HDF5 attrs in-place.
        """
        for key, value in self._runtime_contract_values().items():
            if key not in h5_file.attrs:
                h5_file.attrs[key] = value
                continue
            current_value = self._metadata_value_from_h5_attr(h5_file.attrs[key])
            expected_value = self._metadata_value_from_h5_attr(value)
            if current_value != expected_value:
                h5_file.attrs.modify(key, value)

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
        existing_rows: Dict[str, Dict[str, Any]] = {}
        for row in self._query_paper_rows(conn, paper_ids, lookup_column="paper_id"):
            decoded = self._decode_paper_row(row, parse_json_lists=False)
            paper_id = decoded.pop("paper_id")
            existing_rows[str(paper_id)] = decoded

        return existing_rows

    @staticmethod
    def _query_paper_rows(
        conn: sqlite3.Connection,
        lookup_values: Sequence[Any],
        *,
        lookup_column: str,
    ) -> Iterator[Tuple[Any, ...]]:
        """Yield common paper rows for batched SQLite key lookups.

        :param sqlite3.Connection conn: Open SQLite connection.
        :param Sequence[Any] lookup_values: Values for the selected lookup column.
        :param str lookup_column: ``papers`` column used for the ``IN`` lookup.
        :return Iterator[Tuple[Any, ...]]: Rows in the shared paper-column layout.
        :raises ValueError: If the requested lookup column is not supported.
        """
        if lookup_column not in _PAPER_ROW_LOOKUP_COLUMNS:
            raise ValueError(f"Unsupported paper-row lookup column: {lookup_column}")

        for value_chunk in _chunked(lookup_values, SQLITE_QUERY_BATCH_SIZE):
            placeholders = ",".join("?" for _ in value_chunk)
            query = (
                f"SELECT {_PAPER_ROW_COLUMNS}, "
                "(SELECT COUNT(*) FROM papers AS owners "
                "WHERE owners.row_idx = papers.row_idx) FROM papers "
                f"WHERE {lookup_column} IN ({placeholders})"
            )
            for row in conn.execute(query, value_chunk):
                if row[-1] != 1:
                    raise RuntimeError(
                        "Embedding cache integrity error: row_idx coverage mismatch "
                        f"for accessed paper {row[0]!r}."
                    )
                yield row[:-1]

    @staticmethod
    def _decode_paper_row(
        row: Tuple[Any, ...],
        *,
        parse_json_lists: bool,
    ) -> Dict[str, Any]:
        """Normalize one common paper row while retaining JSON shape choice.

        :param Tuple[Any, ...] row: Row returned by ``_query_paper_rows``.
        :param bool parse_json_lists: Decode authors/categories into lists when true.
        :return Dict[str, Any]: Normalized scalar metadata and selected JSON shape.
        """
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
        decoded: Dict[str, Any] = {
            "paper_id": str(paper_id),
            "text_hash": str(text_hash),
            "row_idx": int(row_idx) if row_idx is not None else None,
            "title": str(title or ""),
            "abstract": str(abstract or ""),
            "year": int(year) if year is not None else None,
            "venue": str(venue or ""),
            "arxiv_id": str(arxiv_id or ""),
            "doi": str(doi or ""),
        }
        if parse_json_lists:
            decoded["authors"] = _parse_json_list(authors_json)
            decoded["categories"] = _parse_json_list(categories_json)
        else:
            decoded["authors_json"] = str(authors_json or "")
            decoded["categories_json"] = str(categories_json or "")
        return decoded

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
            raise _EmbeddingCacheLayoutError(
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
        dataset = self._ensure_matrix_dataset(
            h5_file,
            name=EMBEDDINGS_DATASET_NAME,
            width=embedding_dim,
            dtype=_storage_dtype_for_precision(self.storage_precision),
        )
        assert dataset is not None
        return dataset

    def _ensure_binary_dataset(
        self, h5_file: h5py.File, embedding_dim: int
    ) -> Optional[h5py.Dataset]:
        """Create or validate binary-index dataset for int8 cache search.

        :param h5py.File h5_file: Open HDF5 file handle.
        :param int embedding_dim: Embedding width.
        :return Optional[h5py.Dataset]: Binary-index dataset when enabled.
        """
        packed_dim = (int(embedding_dim) + 7) // 8
        binary_dataset = self._ensure_matrix_dataset(
            h5_file,
            name=BINARY_INDEX_DATASET_NAME,
            width=packed_dim,
            dtype=np.uint8,
            enabled=self.binary_prefilter,
        )
        if binary_dataset is None:
            return None

        embeddings_dataset = self._get_embeddings_dataset(h5_file)
        embedding_rows = (
            int(embeddings_dataset.shape[0]) if embeddings_dataset is not None else 0
        )
        if (
            int(binary_dataset.shape[0]) != embedding_rows
            or binary_dataset.attrs.get(BINARY_INDEX_ENCODING_KEY)
            != BINARY_INDEX_ENCODING
        ):
            if embedding_rows:
                logger.warning(
                    "Rebuilding binary index in %s from %d persisted embedding row(s).",
                    self.h5_path,
                    embedding_rows,
                )
            binary_dataset.attrs.modify(BINARY_INDEX_ENCODING_KEY, "")
            if embeddings_dataset is None:
                binary_dataset.resize((0, packed_dim))
            else:
                self._rebuild_binary_dataset(
                    h5_file=h5_file,
                    embeddings_dataset=embeddings_dataset,
                    binary_dataset=binary_dataset,
                )
            binary_dataset.attrs.modify(
                BINARY_INDEX_ENCODING_KEY, BINARY_INDEX_ENCODING
            )
        return binary_dataset

    def _rebuild_binary_dataset(
        self,
        h5_file: h5py.File,
        embeddings_dataset: h5py.Dataset,
        binary_dataset: h5py.Dataset,
    ) -> None:
        """Rebuild the binary prefilter from persisted int8 embeddings.

        :param h5py.File h5_file: Open HDF5 cache handle.
        :param h5py.Dataset embeddings_dataset: Persisted int8 embedding matrix.
        :param h5py.Dataset binary_dataset: Resizable uint8 binary-index matrix.
        :return None: Replaces all binary-index rows in-place.
        """
        if self.storage_precision != "int8":
            raise RuntimeError("Binary prefilter indexes require int8 storage.")

        row_count = int(embeddings_dataset.shape[0])
        packed_dim = (int(embeddings_dataset.shape[1]) + 7) // 8
        binary_dataset.resize((row_count, packed_dim))
        for start in range(0, row_count, EMBEDDING_DATASET_CHUNK_ROWS):
            end = min(start + EMBEDDING_DATASET_CHUNK_ROWS, row_count)
            stored_chunk = np.asarray(
                embeddings_dataset[start:end],
                dtype=np.int8,
            )
            float_chunk = self._dequantize_int8(h5_file, stored_chunk)
            binary_dataset[start:end] = _quantize_ubinary_embeddings(float_chunk)

    def _ensure_matrix_dataset(
        self,
        h5_file: h5py.File,
        *,
        name: str,
        width: int,
        dtype: np.dtype,
        enabled: bool = True,
    ) -> Optional[h5py.Dataset]:
        """Create or validate an enabled resizable two-dimensional HDF5 matrix.

        :param h5py.File h5_file: Open HDF5 file handle.
        :param str name: Dataset name.
        :param int width: Required number of matrix columns.
        :param np.dtype dtype: Required matrix dtype.
        :param bool enabled: Whether the optional matrix is enabled.
        :return Optional[h5py.Dataset]: Valid matrix dataset, or ``None`` when disabled.
        """
        if not enabled:
            return None

        target_dtype = np.dtype(dtype)
        dataset = h5_file.get(name)
        if dataset is None:
            compression_kwargs = self._dataset_compression_kwargs()
            return h5_file.create_dataset(
                name,
                shape=(0, width),
                maxshape=(None, width),
                dtype=target_dtype,
                chunks=(EMBEDDING_DATASET_CHUNK_ROWS, width),
                shuffle=True,
                **compression_kwargs,
            )

        if dataset.ndim != 2:
            raise ValueError(f"Dataset '{name}' must be 2D.")
        if int(dataset.shape[1]) != int(width):
            raise ValueError(
                f"Dataset '{name}' width mismatch in cache: "
                f"{int(dataset.shape[1])} != {int(width)}"
            )
        if np.dtype(dataset.dtype) != target_dtype:
            raise ValueError(
                f"Dataset '{name}' dtype mismatch in cache: "
                f"{dataset.dtype} != {target_dtype}"
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
        if np.dtype(binary_dataset.dtype) != np.dtype(np.uint8):
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
        compression = str(self._effective_compression or "").strip()
        if not compression:
            return {}

        kwargs: Dict[str, Any] = {"compression": compression}
        if compression.lower() != "lzf":
            kwargs["compression_opts"] = int(self._effective_compression_level)
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
            raise _EmbeddingCacheLayoutError(
                f"Calibration ranges dataset must have shape (2, dim), got {existing.shape}."
            )
        if embedding_dim is not None and int(existing.shape[1]) != int(embedding_dim):
            raise _EmbeddingCacheLayoutError(
                "Calibration range dimension mismatch in cache: "
                f"{int(existing.shape[1])} != {int(embedding_dim)}"
            )

        ranges = np.asarray(existing, dtype=np.float32)
        try:
            return _sanitize_ranges(ranges)
        except ValueError as exc:
            raise _EmbeddingCacheLayoutError(str(exc)) from exc

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

        # Reconstruct floor-quantized bucket centres, keeping the final code at max.
        float_values = int8_embeddings.astype(np.float32) + 128.5
        return np.minimum(starts + float_values * steps, ranges[1])

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

        query_binary = _quantize_ubinary_embeddings(query_embedding)[0]

        keep_k = min(max(int(candidate_count), 0), row_count)
        if keep_k == 0:
            return np.asarray([], dtype=np.int64)
        all_rows: List[np.ndarray] = []
        all_dists: List[np.ndarray] = []
        chunk_size = EMBEDDING_SEARCH_CHUNK_ROWS

        for start in range(0, row_count, chunk_size):
            end = min(start + chunk_size, row_count)
            chunk = np.asarray(binary_dataset[start:end], dtype=np.uint8)
            xor = np.bitwise_xor(chunk, query_binary[None, :])
            dists = _POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int32)

            local_k = min(keep_k, dists.shape[0])
            if local_k == dists.shape[0]:
                local_idx = np.arange(dists.shape[0], dtype=np.int64)
            else:
                distance_cutoff = np.partition(dists, local_k - 1)[local_k - 1]
                closer_idx = np.flatnonzero(dists < distance_cutoff)
                remaining = local_k - int(closer_idx.size)
                tied_idx = np.flatnonzero(dists == distance_cutoff)
                local_idx = np.concatenate(
                    (closer_idx, tied_idx[:remaining]),
                    axis=0,
                )

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
            self._require_finite_scores(scores)
            return self._select_top_k(rows, scores, matrix, top_k)

        def load_chunk(start: int, end: int) -> np.ndarray:
            """Load and normalize one int8 cache matrix chunk.

            :param int start: Inclusive row offset.
            :param int end: Exclusive row offset.
            :return np.ndarray: Dequantized, normalized float32 matrix.
            """
            int8_chunk = np.asarray(embeddings_dataset[start:end], dtype=np.int8)
            return l2_normalize_embeddings(self._dequantize_int8(h5_file, int8_chunk))

        return self._score_chunked_rows(embeddings_dataset, query, top_k, load_chunk)

    def _score_float_rows(
        self,
        embeddings_dataset: h5py.Dataset,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score float32 embeddings against a float32 query.

        :param h5py.Dataset embeddings_dataset: Float matrix dataset.
        :param np.ndarray query_embedding: Float32 query embedding.
        :param int top_k: Top results to keep.
        :return Tuple[np.ndarray, np.ndarray, np.ndarray]: Rows, scores, and embeddings.
        """

        def load_chunk(start: int, end: int) -> np.ndarray:
            """Load one float cache matrix chunk as float32.

            :param int start: Inclusive row offset.
            :param int end: Exclusive row offset.
            :return np.ndarray: Float32 matrix chunk.
            """
            return np.asarray(embeddings_dataset[start:end], dtype=np.float32)

        return self._score_chunked_rows(
            embeddings_dataset,
            query_embedding,
            top_k,
            load_chunk,
        )

    def _score_chunked_rows(
        self,
        embeddings_dataset: h5py.Dataset,
        query_embedding: np.ndarray,
        top_k: int,
        load_chunk: Callable[[int, int], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score a matrix through a loader while retaining a bounded global top-k.

        :param h5py.Dataset embeddings_dataset: Matrix dataset whose rows are scored.
        :param np.ndarray query_embedding: Float32 query vector.
        :param int top_k: Number of results to retain.
        :param Callable[[int, int], np.ndarray] load_chunk: Matrix chunk loader.
        :return Tuple[np.ndarray, np.ndarray, np.ndarray]: Rows, scores, and vectors.
        """
        row_count = int(embeddings_dataset.shape[0])
        best_rows = np.asarray([], dtype=np.int64)
        best_scores = np.asarray([], dtype=np.float32)
        best_embeddings = np.empty((0, int(query_embedding.shape[0])), dtype=np.float32)

        for start in range(0, row_count, EMBEDDING_SEARCH_CHUNK_ROWS):
            end = min(start + EMBEDDING_SEARCH_CHUNK_ROWS, row_count)
            chunk_matrix = load_chunk(start, end)
            chunk_scores = chunk_matrix @ query_embedding
            self._require_finite_scores(chunk_scores)
            chunk_rows = np.arange(start, end, dtype=np.int64)
            rows, scores, embeddings = self._select_top_k(
                chunk_rows,
                chunk_scores,
                chunk_matrix,
                top_k,
            )
            if rows.size == 0:
                continue

            best_rows, best_scores, best_embeddings = self._select_top_k(
                np.concatenate((best_rows, rows), axis=0),
                np.concatenate((best_scores, scores), axis=0),
                np.concatenate((best_embeddings, embeddings), axis=0),
                top_k,
            )

        return best_rows, best_scores, best_embeddings

    @staticmethod
    def _require_finite_scores(scores: np.ndarray) -> None:
        """Reject non-finite scores from corrupt cached embedding data.

        Query vectors are validated at the public boundary, so non-finite scores
        here indicate a malformed persisted embedding or a numerical failure.

        :param np.ndarray scores: Similarity scores produced from cached vectors.
        :return None: Raises when a score is not finite.
        :raises RuntimeError: If cached vectors produce a non-finite score.
        """
        if not np.all(np.isfinite(scores)):
            raise RuntimeError(
                "Embedding cache integrity error: non-finite scores encountered "
                "while scoring cached embeddings. Rebuild this cache namespace "
                "to restore valid vectors."
            )

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
            score_cutoff = np.partition(scores, -keep_k)[-keep_k]
            higher_score_idx = np.flatnonzero(scores > score_cutoff)
            remaining = keep_k - int(higher_score_idx.size)
            tied_idx = np.flatnonzero(scores == score_cutoff)
            tied_order = np.argsort(rows[tied_idx], kind="stable")
            selected_idx = np.concatenate(
                (higher_score_idx, tied_idx[tied_order[:remaining]]),
                axis=0,
            )

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
        output: Dict[int, Dict[str, Any]] = {}
        for row in self._query_paper_rows(
            conn,
            [int(idx) for idx in row_indices],
            lookup_column="row_idx",
        ):
            decoded = self._decode_paper_row(row, parse_json_lists=True)
            row_idx = decoded.pop("row_idx")
            decoded.pop("text_hash")
            assert row_idx is not None
            output[row_idx] = decoded

        return output


def _chunked(values: Sequence[Any], chunk_size: int) -> Iterable[List[Any]]:
    """Yield fixed-size chunks from a sequence.

    :param Sequence[Any] values: Sequence to split into chunks.
    :param int chunk_size: Number of items per yielded chunk.
    :return Iterable[List[Any]]: Iterator over chunk lists.
    """
    for start in range(0, len(values), chunk_size):
        yield list(values[start : start + chunk_size])
