"""Persistent embedding cache backed by SQLite metadata and HDF5 vectors.

Exports :class:`EmbeddingCache`, corpus selection helpers, shared constants,
and the storage, quantization, recovery, and search modules.
"""

from __future__ import annotations

from . import (
    constants,
    ingest,
    layout,
    models,
    quantization,
    recovery,
    search,
    sql,
    store,
)
from .constants import (
    BINARY_INDEX_DATASET_NAME,
    BINARY_INDEX_ENCODING,
    BINARY_INDEX_ENCODING_KEY,
    BINARY_PREFILTER_ENABLED_KEY,
    CALIBRATION_RANGES_DATASET_NAME,
    CALIBRATION_SAMPLE_SIZE_KEY,
    COMPRESSION_FILTER_KEY,
    COMPRESSION_LEVEL_KEY,
    CORPUS_METADATA_VERSION,
    CORPUS_METADATA_VERSION_KEY,
    EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR,
    EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS,
    EMBEDDING_CACHE_SCHEMA_VERSION,
    EMBEDDING_DATASET_CHUNK_ROWS,
    EMBEDDING_VECTOR_DTYPE_KEY,
    EMBEDDINGS_DATASET_NAME,
    H5_LAYOUT_KEY,
    H5_LAYOUT_MATRIX_VERSION,
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
    SCHEMA_VERSION_KEY,
    SOURCE_TORCH_DTYPE_KEY,
    SQLITE_QUERY_BATCH_SIZE,
    STORAGE_PRECISION_KEY,
    TEXT_FORMATTER_FINGERPRINT_KEY,
    CorpusCoverage,
    _corpus_size_coverage,
    _corpus_size_from_token,
    _corpus_size_token,  # noqa: F401  (re-exported: white-box test import)
    _resolve_cache_lock_timeout_seconds,  # noqa: F401  (re-exported for tests)
)
from .lifecycle import open_existing_cache, select_existing_cache
from .models import (
    CacheNamespacePayloadStats,
    CacheSearchResult,
    PendingEmbeddingRecord,
)
from .quantization import (
    _quantize_int8_embeddings,  # noqa: F401  (re-exported for tests)
    validate_compression_filter,
)
from .store import EmbeddingCache

# ``EMBEDDING_SEARCH_CHUNK_ROWS`` is deliberately not aliased here: the scoring
# kernel reads it from :mod:`.constants` at call time, so a package-level copy
# would be a silently ineffective monkeypatch target. Patch ``constants``.

__all__ = [
    "BINARY_INDEX_DATASET_NAME",
    "BINARY_INDEX_ENCODING",
    "BINARY_INDEX_ENCODING_KEY",
    "BINARY_PREFILTER_ENABLED_KEY",
    "CALIBRATION_RANGES_DATASET_NAME",
    "CALIBRATION_SAMPLE_SIZE_KEY",
    "COMPRESSION_FILTER_KEY",
    "COMPRESSION_LEVEL_KEY",
    "CORPUS_METADATA_VERSION",
    "CORPUS_METADATA_VERSION_KEY",
    "EMBEDDINGS_DATASET_NAME",
    "EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR",
    "EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS",
    "EMBEDDING_CACHE_SCHEMA_VERSION",
    "EMBEDDING_DATASET_CHUNK_ROWS",
    "EMBEDDING_VECTOR_DTYPE_KEY",
    "H5_LAYOUT_KEY",
    "H5_LAYOUT_MATRIX_VERSION",
    "HYDRATION_COMPLETE_KEY",
    "HYDRATION_CORPUS_SIZE_KEY",
    "HYDRATION_DATASET_SOURCE_KEY",
    "HYDRATION_RECONCILED_CACHE_ROWS_KEY",
    "HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY",
    "HYDRATION_SPLIT_KEY",
    "INT8_CLIPPED_VALUE_COUNT_KEY",
    "INT8_SATURATION_WARN_RATIO",
    "INT8_TOTAL_VALUE_COUNT_KEY",
    "MODEL_FINGERPRINT_KEY",
    "SCHEMA_VERSION_KEY",
    "SOURCE_TORCH_DTYPE_KEY",
    "SQLITE_QUERY_BATCH_SIZE",
    "STORAGE_PRECISION_KEY",
    "TEXT_FORMATTER_FINGERPRINT_KEY",
    "CacheNamespacePayloadStats",
    "CacheSearchResult",
    "CorpusCoverage",
    "EmbeddingCache",
    "PendingEmbeddingRecord",
    "_corpus_size_coverage",
    "_corpus_size_from_token",
    "constants",
    "ingest",
    "layout",
    "models",
    "quantization",
    "recovery",
    "search",
    "sql",
    "store",
    "validate_compression_filter",
    "open_existing_cache",
    "select_existing_cache",
]
