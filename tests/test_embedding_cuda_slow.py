"""Opt-in integration tests for the real CUDA embedding runtime."""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from citemesh.data import DEFAULT_EMBEDDING_MODEL_NAME
from citemesh.data.embedding_cache import EMBEDDINGS_DATASET_NAME
from citemesh.strategies.embedding import EmbeddingGraphBuilder

pytestmark = [pytest.mark.slow, pytest.mark.cuda]


def _require_cuda() -> Any:
    """Return a real CUDA-enabled torch runtime or skip this test module.

    :return Any: Imported torch module with an available CUDA device.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    if not torch.cuda.is_available():
        pytest.skip("CUDA backend unavailable in this runtime")
    return torch


def _native_cuda_bf16_supported(torch: Any) -> bool:
    """Return whether torch reports native CUDA bfloat16 execution.

    :param Any torch: Imported torch module.
    :return bool: Whether CUDA bfloat16 is supported without emulation.
    """
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:  # pragma: no cover - compatibility with older torch floors
        return bool(torch.cuda.is_bf16_supported())


def test_real_cuda_embeddinggemma_runtime_contract() -> None:
    """Load the designated model and verify its live CUDA precision contract.

    :return None: The real encoder stack satisfies CiteMesh's runtime policy.
    """
    torch = _require_cuda()
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        device="cuda",
        client=MagicMock(),
    )

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        builder._load_model()
        vectors = builder._encode_texts(
            [
                "task: search result | query: transformer language models",
                "title: Attention Is All You Need | text: Self-attention for translation.",
                "title: Mask R-CNN | text: Instance segmentation for images.",
            ]
        )
    torch.cuda.synchronize()

    assert builder._active_model_name == DEFAULT_EMBEDDING_MODEL_NAME
    assert builder.device == "cuda"
    assert next(builder.model.parameters()).device.type == "cuda"
    assert builder.model[0].auto_model.config.use_bidirectional_attention is True

    native_bf16 = _native_cuda_bf16_supported(torch)
    assert builder._source_dtype_hint == ("bfloat16" if native_bf16 else "float32")
    assert builder._autocast_enabled is native_bf16
    if native_bf16:
        assert builder._autocast_device_type == "cuda"
        assert builder._autocast_dtype == torch.bfloat16
    else:
        assert builder._autocast_device_type is None
        assert builder._autocast_dtype is None

    weight_dtypes = {
        parameter.dtype
        for parameter in builder.model.parameters()
        if parameter.is_floating_point()
    }
    assert torch.float16 not in weight_dtypes
    assert weight_dtypes <= {torch.float32, torch.bfloat16}
    assert vectors.shape == (3, builder.truncate_dim)
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert not [
        warning
        for warning in captured_warnings
        if "torch_dtype" in str(warning.message)
    ]


def test_real_cuda_int8_corpus_cache_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydrate and search a tiny local INT8 corpus with the real CUDA model.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to inject local corpus rows.
    :return None: CUDA embeddings survive persistent quantization and retrieval.
    """
    torch = _require_cuda()
    source = "citemesh/cuda-smoke-corpus"
    records = [
        {
            "id": "2401.00001",
            "title": "Dense Passage Retrieval",
            "abstract": (
                "Dual encoders retrieve relevant passages for open-domain question "
                "answering."
            ),
            "categories": "cs.CL cs.IR",
        },
        {
            "id": "2401.00002",
            "title": "Denoising Diffusion Models",
            "abstract": "A generative image model based on iterative denoising.",
            "categories": "cs.CV cs.LG",
        },
        {
            "id": "2401.00003",
            "title": "Graph Neural Networks",
            "abstract": "Message passing learns representations over graph nodes.",
            "categories": "cs.LG",
        },
        {
            "id": "2401.00004",
            "title": "Protein Structure Prediction",
            "abstract": "A neural model predicts three-dimensional protein folds.",
            "categories": "q-bio.BM",
        },
    ]
    loader_calls: list[tuple[str | None, int | None, int | None, bool]] = []

    def load_local_dataset(
        use_streaming: bool,
        preferred_dataset_source: str | None = None,
        row_limit: int | None = None,
        row_offset: int | None = None,
        allow_source_fallback: bool = True,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return a fresh slice of the network-free CUDA smoke corpus.

        :param bool use_streaming: Requested dataset loading mode.
        :param Optional[str] preferred_dataset_source: Exact source requested by
            calibration or refresh paths.
        :param Optional[int] row_limit: Optional number of records to return.
        :param Optional[int] row_offset: Optional starting record index.
        :param bool allow_source_fallback: Whether source fallback was permitted.
        :return tuple[str, list[dict[str, Any]]]: Source token and copied rows.
        """
        assert use_streaming is False
        if preferred_dataset_source is not None:
            assert preferred_dataset_source == source
        loader_calls.append(
            (
                preferred_dataset_source,
                row_limit,
                row_offset,
                allow_source_fallback,
            )
        )
        start = int(row_offset or 0)
        stop = None if row_limit is None else start + int(row_limit)
        return source, [dict(record) for record in records[start:stop]]

    client = MagicMock()
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        device="cuda",
        semantic_source="arxiv-corpus",
        corpus_size=len(records),
        storage_precision="int8",
        calibration_sample_size=len(records),
        encode_batch_size=len(records),
        client=client,
    )
    monkeypatch.setattr(builder, "_load_dataset_for_hydration", load_local_dataset)

    builder._load_model()
    builder._ensure_cache_hydrated(use_streaming=False)
    torch.cuda.synchronize()

    cache = builder.embedding_cache
    stats = cache.payload_stats()
    assert stats.sqlite_rows == len(records)
    assert stats.embedding_rows == len(records)
    assert stats.hydration_complete is True
    assert cache.embedding_count() == len(records)
    assert cache.has_calibration_ranges() is True
    assert cache.is_hydrated(
        builder.dataset_split,
        builder.corpus_size,
        dataset_source=source,
    )
    with h5py.File(cache.h5_path, "r") as h5_file:
        assert h5_file[EMBEDDINGS_DATASET_NAME].dtype == np.dtype(np.int8)

    assert len(loader_calls) == 2
    blocked_loader = MagicMock(
        side_effect=AssertionError("warm cache unexpectedly reloaded its corpus")
    )
    monkeypatch.setattr(builder, "_load_dataset_for_hydration", blocked_loader)
    builder._ensure_cache_hydrated(use_streaming=False)
    blocked_loader.assert_not_called()

    results = builder.search_local(
        "dual encoder passage retrieval for open-domain question answering",
        top_k=2,
    )
    torch.cuda.synchronize()

    assert any(result.paper_id == "arxiv:2401.00001" for result in results)
    assert all(result.embedding.dtype == np.float32 for result in results)
    assert all(np.isfinite(result.embedding).all() for result in results)
    assert client.method_calls == []
