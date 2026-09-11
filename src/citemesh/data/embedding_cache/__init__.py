"""Persistent embedding cache backed by SQLite metadata and HDF5 vectors.

The cache is split across focused modules and re-exported here so
``citemesh.data.embedding_cache`` keeps its historical import surface:

- :mod:`.constants` — dataset names, metadata keys, tunables, lock timeout.
- :mod:`.models` — value dataclasses and the internal layout-error signal.
- :mod:`.quantization` — int8/binary quantizers and compression validation.
- :mod:`.sql` — SQLite statements, JSON column codecs, row decoding.
- :mod:`.layout` — SQLite schema and HDF5 physical-layout management.
- :mod:`.recovery` — locking, connections, and crash recovery.
- :mod:`.search` — the query-time scoring kernel.
- :mod:`.ingest` — the cache-miss/encode/commit ingestion pipeline.
- :mod:`.store` — the :class:`EmbeddingCache` facade and public API.
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
    _corpus_size_token,  # noqa: F401  (re-exported: white-box test import)
    _resolve_cache_lock_timeout_seconds,  # noqa: F401  (re-exported for tests)
)
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
    "EmbeddingCache",
    "PendingEmbeddingRecord",
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
]
