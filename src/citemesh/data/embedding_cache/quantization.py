"""Embedding quantization, dequantization, and compression-filter validation.

Owns the pure numeric kernels shared by the writer and the search path: the
float32 normalization gate, int8 and packed-binary quantizers, calibration-range
sanitization, saturation counting, the storage-dtype mapping, and the HDF5
compression filter validator.
"""

from __future__ import annotations

import h5py
import numpy as np
from numpy.typing import NDArray

from .constants import (
    _COMPRESSION_FILTER_IDS,
    _COMPRESSION_FILTERS,
    CALIBRATION_RANGES_DATASET_NAME,
)

_POPCOUNT_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
    axis=1
)


def _storage_dtype_for_precision(
    storage_precision: str,
) -> np.dtype[np.float32] | np.dtype[np.int8]:
    """Map storage precision token to NumPy dtype.

    :param str storage_precision: Storage precision token.
    :return np.dtype[np.float32] | np.dtype[np.int8]: Dtype used for the HDF5 embedding matrix.
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


def _sanitize_ranges(ranges: NDArray[np.float32]) -> NDArray[np.float32]:
    """Ensure per-dimension quantization ranges are strictly non-zero.

    :param NDArray[np.float32] ranges: Raw ``(2, dim)`` range matrix.
    :return NDArray[np.float32]: Sanitized range matrix.
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
    embeddings_array: NDArray[np.float32],
    ranges: NDArray[np.float32],
) -> tuple[int, int]:
    """Count values that fall outside persisted int8 calibration ranges.

    :param NDArray[np.float32] embeddings_array: Float32 embedding matrix being quantized.
    :param NDArray[np.float32] ranges: Persisted ``(2, dim)`` calibration ranges.
    :return tuple[int, int]: ``(clipped_values, total_values)``.
    """
    normalized = np.asarray(embeddings_array, dtype=np.float32)
    sanitized_ranges = _sanitize_ranges(ranges)
    mins = sanitized_ranges[0][None, :]
    maxs = sanitized_ranges[1][None, :]
    clipped = np.logical_or(normalized < mins, normalized > maxs)
    return int(np.count_nonzero(clipped)), int(normalized.size)


def _as_float32_embedding_matrix(
    embeddings: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Normalize raw embeddings into a 2D float32 matrix.

    :param NDArray[np.float32] embeddings: Raw float32 embedding vector or matrix.
    :return NDArray[np.float32]: Float32 matrix with shape ``(rows, dim)``.
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


def _quantize_int8_embeddings(
    embeddings: NDArray[np.float32], ranges: NDArray[np.float32]
) -> NDArray[np.int8]:
    """Quantize float embeddings into signed int8 rows with explicit clipping.

    :param NDArray[np.float32] embeddings: Float32 embedding vector or matrix.
    :param NDArray[np.float32] ranges: Persisted ``(2, dim)`` calibration ranges.
    :return NDArray[np.int8]: Int8 matrix with values clipped to the calibration ranges.
    """
    matrix = _as_float32_embedding_matrix(embeddings)
    sanitized_ranges = _sanitize_ranges(ranges)
    starts = sanitized_ranges[0][None, :]
    steps = ((sanitized_ranges[1] - sanitized_ranges[0]) / 255.0)[None, :]
    # Select unsigned buckets before the signed offset so every bucket has equal width.
    buckets = np.clip(np.floor((matrix - starts) / steps), 0.0, 255.0)
    return (buckets - 128.0).astype(np.int8)


def _quantize_ubinary_embeddings(
    embeddings: NDArray[np.float32],
) -> NDArray[np.uint8]:
    """Pack embedding sign bits into unsigned bytes for Hamming prefiltering.

    :param NDArray[np.float32] embeddings: Float32 embedding vector or matrix.
    :return NDArray[np.uint8]: Packed unsigned binary embedding matrix.
    """
    matrix = _as_float32_embedding_matrix(embeddings)
    return np.asarray(np.packbits(matrix > 0, axis=-1), dtype=np.uint8)


def _dequantize_int8(
    h5_file: h5py.File, int8_embeddings: NDArray[np.int8]
) -> NDArray[np.float32]:
    """Dequantize int8 embeddings to float32 using persisted ranges.

    :param h5py.File h5_file: Open HDF5 file handle.
    :param NDArray[np.int8] int8_embeddings: Int8 embeddings.
    :return NDArray[np.float32]: Dequantized float32 embeddings.
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
