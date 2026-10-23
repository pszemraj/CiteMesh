"""Precision-scoped encode proxy wrapping a loaded SentenceTransformer.

Owns the dtype introspection helper and the proxy that runs every encode call
inside the builder's precision context, restores an eager model when a compiled
one fails mid-encode, and optionally prefetches tokenized batches on a worker
thread.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from itertools import chain
from typing import TYPE_CHECKING, Literal, TypedDict, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from citemesh.core.text_batching import l2_normalize_embeddings, warn_on_truncated_count

from . import deps

if TYPE_CHECKING:
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modality_types import PairInput, SingleInput
    from typing_extensions import Unpack

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class _EncodeOptions(TypedDict, total=False):
    """Encoder options used by CiteMesh text batching."""

    batch_size: int
    convert_to_tensor: bool
    normalize_embeddings: bool
    show_progress_bar: bool | None


def _model_floating_dtype_names(model: torch.nn.Module) -> set[str] | None:
    """Return floating-point parameter and buffer dtypes from a loaded model.

    :param torch.nn.Module model: Model whose parameter and buffer tensors are inspected.
    :return Optional[Set[str]]: Normalized dtype names, or ``None`` when live
        tensor inspection is unavailable.
    """
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    buffers = getattr(model, "buffers", None)

    observed: set[str] = set()
    aliases = {
        "float": "float32",
        "float16": "float16",
        "half": "float16",
        "bfloat16": "bfloat16",
        "float32": "float32",
        "double": "float64",
        "float64": "float64",
    }
    try:
        for tensor in chain(parameters(), buffers() if callable(buffers) else ()):
            dtype = getattr(tensor, "dtype", None)
            dtype_name = str(dtype or "").casefold()
            normalized = dtype_name.removeprefix("torch.")
            is_floating = getattr(dtype, "is_floating_point", None)
            if is_floating is False:
                continue
            mapped = aliases.get(normalized)
            if mapped is not None:
                observed.add(mapped)
            elif is_floating is True or normalized.startswith(("float", "bfloat")):
                observed.add(normalized)
    except Exception:
        logger.debug("Could not inspect loaded model tensor dtypes", exc_info=True)
        return None
    return observed


class _PrecisionEncodeProxy:
    """Apply encode-time precision and lazy-compile recovery for every caller."""

    def __init__(
        self,
        model: SentenceTransformer,
        context_factory: Callable[[], AbstractContextManager[None]],
        restore_eager: Callable[[Exception], bool],
        *,
        prefetch_batches: bool = False,
    ) -> None:
        """Create a model proxy for encode-time precision controls.

        :param SentenceTransformer model: Wrapped encoder model.
        :param Callable context_factory: Factory for the encode precision context.
        :param Callable[[Exception], bool] restore_eager: Restore eager execution
            after a compiled-call failure; return whether the call can be retried.
        :param bool prefetch_batches: Whether to overlap CPU preprocessing with
            model execution for CiteMesh text batches.
        """
        self._model = model
        self._context_factory = context_factory
        self._restore_eager = restore_eager
        self.prefetch_batches = bool(prefetch_batches)

    def _call_with_precision(self, operation: Callable[[], _T]) -> _T:
        """Run an operation with precision controls and eager compile recovery.

        :param Callable operation: Encode or forward operation to run.
        :return: Operation result.
        """
        try:
            with self._context_factory():
                return operation()
        except Exception as exc:
            compiled_failure = exc
            if not self._restore_eager(compiled_failure):
                raise
            try:
                with self._context_factory():
                    return operation()
            except Exception as eager_error:
                raise eager_error from compiled_failure

    @overload
    def encode(
        self,
        sentences: list[str],
        /,
        *,
        batch_size: int,
        convert_to_tensor: Literal[False],
        normalize_embeddings: Literal[True],
        show_progress_bar: bool,
    ) -> NDArray[np.float32]: ...

    @overload
    def encode(
        self,
        sentences: list[str],
        /,
        **kwargs: Unpack[_EncodeOptions],
    ) -> ArrayLike | torch.Tensor: ...

    def encode(
        self,
        sentences: list[str],
        /,
        **kwargs: Unpack[_EncodeOptions],
    ) -> ArrayLike | torch.Tensor:
        """Run ``encode`` within the configured context manager.

        :param list[str] sentences: Text inputs passed to the wrapped encoder.
        :param kwargs: Encoder options used by CiteMesh text batching.
        :return ArrayLike | torch.Tensor: Encoded vectors from the wrapped model.
        """
        normalize_output = kwargs.get("normalize_embeddings", False)
        if normalize_output:
            kwargs["normalize_embeddings"] = False
        embeddings = self._call_with_precision(
            lambda: self._model.encode(sentences, **kwargs)
        )
        # Normalize once in FP32; CPU autocast would round ST's division to BF16.
        if normalize_output:
            return l2_normalize_embeddings(embeddings)
        return embeddings

    def _prepare_prefetched_batch(
        self,
        texts: list[str],
        prompt: str | None,
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Preprocess one batch and count inputs that will be truncated.

        :param list[str] texts: Text payloads in encode order.
        :param Optional[str] prompt: Model-resolved prompt prepended by preprocessing.
        :return Tuple[Dict[str, torch.Tensor], int]: Prepared CPU features and truncation count.
        """
        inputs: list[SingleInput | PairInput] = []
        inputs.extend(texts)
        features: dict[str, torch.Tensor] = self._model.preprocess(
            inputs, prompt=prompt
        )
        max_length = getattr(self._model, "max_seq_length", None)
        if max_length is None:
            return features, 0

        post_truncation_lengths: list[int]
        if "cu_seq_lens_q" in features:
            boundaries = features["cu_seq_lens_q"].tolist()
            post_truncation_lengths = [
                int(end) - int(start) for start, end in zip(boundaries, boundaries[1:])
            ]
        elif "attention_mask" in features:
            post_truncation_lengths = [
                int(length) for length in features["attention_mask"].sum(dim=1).tolist()
            ]
        else:
            post_truncation_lengths = [int(max_length)] * len(texts)

        candidates = [
            text
            for text, length in zip(texts, post_truncation_lengths)
            if length >= int(max_length)
        ]
        if not candidates:
            return features, 0

        prompt_prefix = prompt or ""
        candidate_lengths = self._model.tokenizer(
            [prompt_prefix + text for text in candidates],
            truncation=False,
            padding=False,
            return_length=True,
            verbose=False,
        )["length"]
        return features, sum(
            int(length) > int(max_length) for length in candidate_lengths
        )

    def encode_prefetched(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> NDArray[np.float32]:
        """Encode normalized FP32 text vectors while prefetching CPU features.

        :param Sequence[str] texts: Text payloads to encode.
        :param int batch_size: Maximum rows per prepared batch.
        :return NDArray[np.float32]: Normalized FP32 embeddings in original input order.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        from sentence_transformers.util import batch_to_device

        from citemesh.core.text_batching import length_bucketed_index_batches

        model = self._model
        model.eval()
        prompt = model._resolve_prompt(None, None)
        batches = length_bucketed_index_batches(texts, batch_size)
        ordered_embeddings: dict[int, NDArray[np.float32]] = {}
        truncated_count = 0
        torch = deps._import_torch()

        def run_batches() -> None:
            """Run prepared batches on the model in the calling thread.

            :return None: Populates ordered embeddings and truncation count.
            """
            nonlocal truncated_count
            ordered_embeddings.clear()
            truncated_count = 0
            with torch.inference_mode(), ThreadPoolExecutor(max_workers=1) as executor:
                pending = executor.submit(
                    self._prepare_prefetched_batch,
                    [texts[idx] for idx in batches[0]],
                    prompt,
                )
                for batch_number, batch_indices in enumerate(batches):
                    features, batch_truncated_count = pending.result()
                    truncated_count += batch_truncated_count
                    if batch_number + 1 < len(batches):
                        next_indices = batches[batch_number + 1]
                        pending = executor.submit(
                            self._prepare_prefetched_batch,
                            [texts[idx] for idx in next_indices],
                            prompt,
                        )

                    features = batch_to_device(features, model.device)
                    embeddings = model(features)["sentence_embedding"]
                    truncate_dim = getattr(model, "truncate_dim", None)
                    if truncate_dim is not None:
                        embeddings = embeddings[..., : int(truncate_dim)]
                    batch_embeddings = embeddings.float().cpu().numpy()
                    for position, original_idx in enumerate(batch_indices):
                        ordered_embeddings[original_idx] = batch_embeddings[position]

        self._call_with_precision(run_batches)
        max_length = model.max_seq_length
        if truncated_count and max_length is not None:
            warn_on_truncated_count(logger, truncated_count, len(texts), max_length)

        embeddings = np.asarray(
            [ordered_embeddings[idx] for idx in range(len(texts))],
            dtype=np.float32,
        )
        return l2_normalize_embeddings(embeddings)

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attributes to the wrapped model.

        :param str name: Attribute name.
        :return object: Delegated attribute value.
        """
        return getattr(self._model, name)
