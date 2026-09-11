"""Tunables, dataset names, metadata keys, and lock-timeout resolution.

This module owns every literal the embedding cache persists or dispatches on:
HDF5 dataset names, SQLite metadata keys, schema/layout version tokens, the
compression and storage-precision whitelists, chunk sizes, and the lock-timeout
resolution that reads :data:`EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR`.
"""

from __future__ import annotations

import logging
import os

import h5py
import numpy as np

logger = logging.getLogger(__name__)


SQLITE_QUERY_BATCH_SIZE = 900
EMBEDDINGS_DATASET_NAME = "embeddings"
BINARY_INDEX_DATASET_NAME = "binary_index"
BINARY_INDEX_ENCODING_KEY = "encoding"
BINARY_INDEX_ENCODING = "int8-midpoint-sign-v1"
CALIBRATION_RANGES_DATASET_NAME = "calibration_ranges"
EMBEDDING_CACHE_SCHEMA_VERSION = 3
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
EMBEDDING_DATASET_CHUNK_ROWS = 2048
EMBEDDING_SEARCH_CHUNK_ROWS = 65536
INT8_SATURATION_WARN_RATIO = 0.005
_PAPER_ROW_COLUMNS = (
    "paper_id, text_hash, row_idx, title, abstract, year, authors_json, "
    "categories_json, venue, arxiv_id, doi"
)
_PAPER_ROW_LOOKUP_COLUMNS = {"paper_id", "row_idx"}


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


def _corpus_size_token(corpus_size: int | None) -> str:
    """Convert optional corpus size into stable metadata token.

    Capped tokens carry the slice policy (``newest:N``) so caches hydrated
    under the legacy head-slice policy (bare ``N``) fail ``is_hydrated`` and
    rehydrate instead of silently serving the oldest records.

    :param Optional[int] corpus_size: Optional corpus-size cap.
    :return str: Tokenized corpus-size value.
    """
    return "all" if corpus_size is None else f"newest:{int(corpus_size)}"
