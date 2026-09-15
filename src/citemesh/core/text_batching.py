"""Length-aware text batching helpers for embedding encode workloads.

This lives in ``citemesh.core`` rather than beside the embedding strategy
because ``citemesh.data.embedding_cache`` encodes through it as well, and the
one-way dependency direction forbids ``data`` from importing ``strategies``.
Nothing here performs I/O or touches an optional dependency.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def warn_on_truncated_inputs(model: Any, texts: Sequence[str]) -> None:
    """Report texts that exceed the encoder window, including its default prompt.

    :param Any model: SentenceTransformer encoder or its precision proxy.
    :param Sequence[str] texts: Formatted texts about to be encoded.
    :return None: Logs a warning when tokenization will discard input tokens.
    """
    max_length = getattr(model, "max_seq_length", None)
    if not texts or max_length is None:
        return
    prompt_name = getattr(model, "default_prompt_name", None)
    prompt = model.prompts.get(prompt_name, "") if prompt_name else ""
    lengths = model.tokenizer(
        [prompt + text for text in texts],
        truncation=False,
        padding=False,
        return_length=True,
        verbose=False,
    )["length"]
    truncated_count = sum(length > max_length for length in lengths)
    warn_on_truncated_count(
        logging.getLogger(__name__), truncated_count, len(texts), max_length
    )


def warn_on_truncated_count(
    logger: logging.Logger, truncated_count: int, total: int, max_length: int
) -> None:
    """Report truncation using counts from the caller's tokenization pass.

    :param logging.Logger logger: Logger identifying the encoding path.
    :param int truncated_count: Number of inputs exceeding the token window.
    :param int total: Total number of inputs.
    :param int max_length: Encoder token window, including prompts and special tokens.
    :return None: Logs only when at least one input is truncated.
    """
    if truncated_count:
        logger.warning(
            "Embedding encoder will truncate %d of %d inputs to its %d-token "
            "window (including prompts and special tokens); embeddings will "
            "represent only part of those inputs.",
            truncated_count,
            total,
            max_length,
        )


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


def encode_texts(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Encode text through the model's preferred batching path.

    :param Any model: Encoder or precision proxy exposing ``encode``.
    :param Sequence[str] texts: Text payloads to encode.
    :param int batch_size: Maximum rows per encode batch.
    :param bool show_progress_bar: Whether the encoder may show progress.
    :return np.ndarray: Float32 embeddings in original input order.
    """
    if (
        getattr(model, "prefetch_batches", False) is True
        and len(texts) > batch_size
        and not show_progress_bar
    ):
        return np.asarray(
            model.encode_prefetched(texts, batch_size=batch_size),
            dtype=np.float32,
        )

    warn_on_truncated_inputs(model, texts)
    return encode_texts_in_length_buckets(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
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
