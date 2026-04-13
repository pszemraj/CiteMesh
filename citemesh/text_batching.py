"""Length-aware text batching helpers for embedding encode workloads."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def estimate_text_length_bucket(text: str) -> int:
    """Estimate relative token length for batching similar text payloads together.

    :param str text: Text payload to estimate.
    :return int: Best-effort length estimate.
    """
    normalized = str(text).strip()
    if not normalized:
        return 0

    whitespace_tokens = len(normalized.split())
    char_tokens = max(len(normalized) // 4, 1)
    return max(whitespace_tokens, char_tokens)


def l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Return float32 embeddings normalized to unit length.

    :param np.ndarray embeddings: Vector or matrix payload to normalize.
    :return np.ndarray: Float32 array with L2-normalized rows.
    """
    normalized = np.asarray(embeddings, dtype=np.float32)
    if normalized.ndim == 1:
        norm = float(np.linalg.norm(normalized))
        return normalized / max(norm, 1e-12)

    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    return normalized / np.clip(norms, 1e-12, None)


def length_bucketed_index_batches(
    texts: Sequence[str],
    batch_size: int,
) -> list[list[int]]:
    """Return input indices grouped into length-similar batches.

    :param Sequence[str] texts: Text payloads to batch.
    :param int batch_size: Maximum rows per batch.
    :return list[list[int]]: Original-text indices grouped by similar length.
    :raises ValueError: If ``batch_size`` is less than 1.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if len(texts) <= batch_size:
        return [list(range(len(texts)))]

    ordered_indices = sorted(
        range(len(texts)),
        key=lambda idx: (
            estimate_text_length_bucket(texts[idx]),
            idx,
        ),
    )
    return [
        ordered_indices[start : start + batch_size]
        for start in range(0, len(ordered_indices), batch_size)
    ]


def encode_texts_in_length_buckets(
    texts: Sequence[str],
    *,
    batch_size: int,
    show_progress_bar: bool,
    encode_batch: Callable[[list[str], bool], np.ndarray],
) -> np.ndarray:
    """Encode texts in length-similar batches while restoring original order.

    :param Sequence[str] texts: Text payloads to encode.
    :param int batch_size: Maximum rows per encode batch.
    :param bool show_progress_bar: Whether the encoder may show progress.
    :param Callable[[list[str], bool], np.ndarray] encode_batch: Batch encoder callback.
    :return np.ndarray: Float32 embeddings in original input order.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    batches = length_bucketed_index_batches(texts, batch_size)
    if len(batches) == 1:
        return np.asarray(
            encode_batch(list(texts), show_progress_bar), dtype=np.float32
        )

    ordered_embeddings: dict[int, np.ndarray] = {}
    for batch_indices in batches:
        batch_embeddings = np.asarray(
            encode_batch([texts[idx] for idx in batch_indices], False),
            dtype=np.float32,
        )
        if batch_embeddings.ndim == 1:
            batch_embeddings = batch_embeddings.reshape(1, -1)
        for position, original_idx in enumerate(batch_indices):
            ordered_embeddings[original_idx] = batch_embeddings[position]

    return np.asarray(
        [ordered_embeddings[idx] for idx in range(len(texts))],
        dtype=np.float32,
    )
