"""Length-aware text batching helpers for embedding encode workloads.

This lives in ``citemesh.core`` rather than beside the embedding strategy
because ``citemesh.data.embedding_cache`` encodes through it as well, and the
one-way dependency direction forbids ``data`` from importing ``strategies``.
Nothing here performs I/O or touches an optional dependency.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class _TextEncoder(Protocol):
    """Encoder interface used by the bounded text helpers."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_tensor: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> ArrayLike: ...


class _PrefetchBatchEncoder(Protocol):
    """Batch encoder available on the prefetching precision wrapper."""

    def __call__(self, texts: Sequence[str], *, batch_size: int) -> ArrayLike: ...


class _TokenizationLengths(Protocol):
    """Tokenizer result exposing the requested token-length field."""

    def __getitem__(self, key: Literal["length"]) -> Sequence[int]: ...


class _LengthTokenizer(Protocol):
    """Tokenizer call that returns each input's untruncated token length."""

    def __call__(
        self,
        texts: list[str],
        *,
        truncation: bool,
        padding: bool,
        return_length: bool,
        verbose: bool,
    ) -> _TokenizationLengths: ...


def warn_on_truncated_inputs(model: _TextEncoder, texts: Sequence[str]) -> None:
    """Report texts that exceed the encoder window, including its default prompt.

    :param _TextEncoder model: Encoder or precision wrapper.
    :param Sequence[str] texts: Formatted texts about to be encoded.
    """
    max_length: int | None = getattr(model, "max_seq_length", None)
    if not texts or max_length is None:
        return
    prompt_name: str | None = getattr(model, "default_prompt_name", None)
    if prompt_name:
        prompts: Mapping[str, str] = getattr(model, "prompts")
        prompt = prompts.get(prompt_name, "")
    else:
        prompt = ""
    tokenizer: _LengthTokenizer = getattr(model, "tokenizer")
    lengths = tokenizer(
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


def l2_normalize_embeddings(embeddings: ArrayLike) -> NDArray[np.float32]:
    """Return float32 embeddings normalized to unit length.

    :param ArrayLike embeddings: Vector or matrix payload to normalize.
    :return NDArray[np.float32]: Float32 array with L2-normalized rows.
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
    encode_batch: Callable[[list[str], bool], ArrayLike],
) -> NDArray[np.float32]:
    """Encode texts in length-similar batches while restoring original order.

    :param Sequence[str] texts: Text payloads to encode.
    :param int batch_size: Maximum rows per encode batch.
    :param bool show_progress_bar: Whether the encoder may show progress.
    :param Callable[[list[str], bool], ArrayLike] encode_batch: Batch encoder callback.
    :return NDArray[np.float32]: Float32 embeddings in original input order.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    batches = length_bucketed_index_batches(texts, batch_size)
    if len(batches) == 1:
        return np.asarray(
            encode_batch(list(texts), show_progress_bar), dtype=np.float32
        )

    ordered_embeddings: dict[int, NDArray[np.float32]] = {}
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
    model: _TextEncoder,
    texts: Sequence[str],
    *,
    batch_size: int,
    show_progress_bar: bool = False,
) -> NDArray[np.float32]:
    """Encode text through the model's preferred batching path.

    :param _TextEncoder model: Encoder or precision wrapper exposing ``encode``.
    :param Sequence[str] texts: Text payloads to encode.
    :param int batch_size: Maximum rows per encode batch.
    :param bool show_progress_bar: Whether the encoder may show progress.
    :return NDArray[np.float32]: Float32 embeddings in original input order.
    """
    prefetch_batches: bool = getattr(model, "prefetch_batches", False)
    if prefetch_batches is True and len(texts) > batch_size and not show_progress_bar:
        encode_prefetched: _PrefetchBatchEncoder = getattr(model, "encode_prefetched")
        return np.asarray(
            encode_prefetched(texts, batch_size=batch_size), dtype=np.float32
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
