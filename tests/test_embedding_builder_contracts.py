"""Contract tests for embedding builder precision and hydration behavior."""

from __future__ import annotations

import logging
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from citemesh.data.embedding_cache import CacheSearchResult
from citemesh.strategies.embedding import EmbeddingGraphBuilder, _query_seed_id


def _install_fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Install fake ``sentence_transformers`` module for precision tests."""
    init_log: dict[str, Any] = {}
    encode_log: list[dict[str, Any]] = []

    class _FakeInnerBlock:
        def __init__(self) -> None:
            self.auto_model = object()

    class _FakeSentenceTransformer:
        def __init__(self, model_name_or_path: str, **kwargs: Any):
            init_log["model_name"] = model_name_or_path
            init_log["kwargs"] = kwargs
            self._blocks = [_FakeInnerBlock()]
            init_log["auto_model_before_compile"] = self._blocks[0].auto_model

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            encode_log.append(kwargs)
            return np.ones((len(texts), 2), dtype=np.float32)

        def __getitem__(self, index: int) -> _FakeInnerBlock:
            return self._blocks[index]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return init_log, encode_log


def _disable_embedding_dep_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable optional embedding dependency guard for focused unit tests."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    bf16_supported: bool,
    *,
    compile_behavior: str = "identity",
    capability: tuple[int, int] | None = (8, 0),
    include_tf32_precision_api: bool = True,
    include_tf32_legacy_api: bool = True,
) -> tuple[object, list[tuple[Any, ...]], object]:
    """Install fake ``torch`` module for precision tests."""
    bf16_token = object()
    autocast_log: list[tuple[Any, ...]] = []

    class _FakeAutocast:
        def __init__(self, device_type: str, dtype: object):
            autocast_log.append(("call", device_type, dtype))

        def __enter__(self) -> "_FakeAutocast":
            autocast_log.append(("enter",))
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            del exc_type, exc, tb
            autocast_log.append(("exit",))

    def _autocast(*, device_type: str, dtype: object) -> _FakeAutocast:
        return _FakeAutocast(device_type=device_type, dtype=dtype)

    def _compile(model: object) -> object:
        if compile_behavior == "raise":
            raise RuntimeError("compile failure")
        if compile_behavior == "tagged":
            return ("compiled", model)
        return model

    matmul_backend = types.SimpleNamespace()
    cudnn_backend = types.SimpleNamespace()
    cudnn_conv = types.SimpleNamespace()
    cudnn_backend.conv = cudnn_conv
    if include_tf32_precision_api:
        matmul_backend.fp32_precision = "none"
        cudnn_conv.fp32_precision = "none"
    if include_tf32_legacy_api:
        matmul_backend.allow_tf32 = False
        cudnn_backend.allow_tf32 = False

    cuda_module = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        is_bf16_supported=lambda: bf16_supported,
    )
    if capability is not None:
        cuda_module.get_device_capability = lambda _index=0: capability

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = bf16_token
    fake_torch.autocast = _autocast
    fake_torch.compile = _compile
    fake_torch.cuda = cuda_module
    fake_torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=matmul_backend),
        cudnn=cudnn_backend,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    return bf16_token, autocast_log, fake_torch


class _FakeEncodeModel:
    """Minimal encode model used by hydration tests."""

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        """Return deterministic embeddings for input texts."""
        del kwargs
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_embedding_builder_requires_optional_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding builder should fail with guidance when deps are missing."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setitem(sys.modules, "datasets", None)

    with pytest.raises(
        ImportError,
        match=r"Embedding strategy requires: torch, sentence-transformers, datasets\. "
        r"Install with: pip install citemesh\[embeddings\]",
    ):
        EmbeddingGraphBuilder(
            max_papers=5, model_name="test-model", random_seed=42, client=MagicMock()
        )


def test_embedding_runtime_precision_compile_tf32_and_logging_contracts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Runtime should enforce precision, compile, TF32, and logging policies."""
    _disable_embedding_dep_check(monkeypatch)

    with pytest.raises(
        ValueError,
        match="truncate_dim=300 is not supported for google/embeddinggemma-300m",
    ):
        EmbeddingGraphBuilder(max_papers=1, truncate_dim=300, client=MagicMock())

    precision_cases = [
        (True, True, True),
        (True, False, False),
    ]
    for cuda_available, bf16_supported, expects_bf16 in precision_cases:
        init_log, _ = _install_fake_sentence_transformers(monkeypatch)
        bf16_token, autocast_log, _fake_torch = _install_fake_torch(
            monkeypatch,
            cuda_available=cuda_available,
            bf16_supported=bf16_supported,
        )

        builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
        builder._load_model()
        embeddings = builder._encode_texts(["seed"], show_progress_bar=False)

        assert init_log["model_name"] == "google/embeddinggemma-300m"
        assert init_log["kwargs"]["truncate_dim"] == 256
        assert embeddings.shape == (1, 2)
        if expects_bf16:
            assert init_log["kwargs"]["model_kwargs"]["torch_dtype"] is bf16_token
            assert ("call", "cuda", bf16_token) in autocast_log
            assert ("enter",) in autocast_log and ("exit",) in autocast_log
        else:
            assert "model_kwargs" not in init_log["kwargs"]
            assert autocast_log == []

    compile_cases = [
        ("google/embeddinggemma-300m", "tagged", True),
        ("google/embeddinggemma-300m", "raise", False),
        ("sentence-transformers/all-MiniLM-L6-v2", "tagged", False),
    ]
    for model_name, compile_behavior, expect_compiled in compile_cases:
        init_log, _ = _install_fake_sentence_transformers(monkeypatch)
        _bf16_token, _autocast_log, _fake_torch = _install_fake_torch(
            monkeypatch,
            cuda_available=True,
            bf16_supported=True,
            compile_behavior=compile_behavior,
        )

        builder = EmbeddingGraphBuilder(
            max_papers=1, model_name=model_name, client=MagicMock()
        )
        builder._load_model()

        original = init_log["auto_model_before_compile"]
        assert builder.model is not None
        if expect_compiled:
            assert builder.model[0].auto_model == ("compiled", original)
        else:
            assert builder.model[0].auto_model is original
        assert builder._inner_model_compiled is expect_compiled

    tf32_cases = [
        ((8, 0), "tf32"),
        ((7, 5), "off"),
    ]
    for capability, expected_mode in tf32_cases:
        _install_fake_sentence_transformers(monkeypatch)
        _bf16_token, _autocast_log, fake_torch = _install_fake_torch(
            monkeypatch,
            cuda_available=True,
            bf16_supported=True,
            capability=capability,
        )

        builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
        builder._load_model()

        if expected_mode == "tf32":
            assert fake_torch.backends.cuda.matmul.fp32_precision == "tf32"
            assert fake_torch.backends.cudnn.conv.fp32_precision == "tf32"
        else:
            assert fake_torch.backends.cuda.matmul.fp32_precision == "none"
            assert fake_torch.backends.cudnn.conv.fp32_precision == "none"
        assert builder._tf32_mode == expected_mode

    _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, _autocast_log, _fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=True,
        capability=(8, 0),
        compile_behavior="tagged",
    )

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        builder._load_model()

    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG
    ]

    assert any("runtime: dim=" in message for message in info_messages)
    assert not any(
        "Adds recommended query/document prompts for EmbeddingGemma." in message
        for message in info_messages
    )
    assert any(
        "Adds recommended query/document prompts for EmbeddingGemma." in message
        for message in debug_messages
    )
    assert not any("embedding dimension: using" in message for message in info_messages)
    assert any("embedding dimension: using" in message for message in debug_messages)
    assert any(
        "Enabled torch.compile for google/embeddinggemma-300m inner transformer"
        in message
        for message in debug_messages
    )


def test_embedding_cache_namespace_varies_by_storage_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache namespace should isolate incompatible storage precision settings."""
    _disable_embedding_dep_check(monkeypatch)

    int8_builder = EmbeddingGraphBuilder(
        max_papers=1, storage_precision="int8", client=MagicMock()
    )
    f32_builder = EmbeddingGraphBuilder(
        max_papers=1, storage_precision="float32", client=MagicMock()
    )

    assert (
        int8_builder.embedding_cache.model_name
        != f32_builder.embedding_cache.model_name
    )


def test_metadata_and_streaming_loader_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata parsing and streaming hydration fallback should stay deterministic."""
    _disable_embedding_dep_check(monkeypatch)

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    snapshot = builder._extract_paper_metadata(
        {
            "id": "2301.07041",
            "title": "A snapshot paper",
            "abstract": "An abstract.",
            "authors": "Alice Smith, Bob Jones, Carol White",
            "categories": "cs.LG cs.AI",
            "update_date": "2023-06-15",
        },
        fallback_index=0,
    )
    versioned = builder._extract_paper_metadata(
        {
            "id": "arXiv:1706.03762v5",
            "title": "Versioned",
            "abstract": "An abstract.",
        },
        fallback_index=0,
    )

    assert snapshot["paper_id"] == "arxiv:2301.07041"
    assert snapshot["authors"] == ["Alice Smith", "Bob Jones", "Carol White"]
    assert snapshot["categories"] == ["cs.LG", "cs.AI"]
    assert snapshot["year"] == 2023
    assert versioned["paper_id"] == "arxiv:1706.03762"

    load_calls: list[tuple[str, str, bool]] = []

    def fake_load_dataset(
        dataset_name: str, split: str, streaming: bool = False
    ) -> list[dict[str, Any]]:
        load_calls.append((dataset_name, split, streaming))
        if dataset_name == "librarian-bots/arxiv-metadata-snapshot":
            raise RuntimeError("primary unavailable")
        return [
            {
                "id": "fallback-paper",
                "title": "Fallback Title",
                "abstract": "Fallback abstract.",
                "year": 2020,
            }
        ]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    builder = EmbeddingGraphBuilder(
        max_papers=1, use_streaming=True, random_seed=0, client=MagicMock()
    )
    selected_name, dataset = builder._load_dataset_for_hydration(use_streaming=True)

    assert selected_name == "CShorten/ML-ArXiv-Papers"
    assert [name for name, _, _ in load_calls] == [
        "librarian-bots/arxiv-metadata-snapshot",
        "CShorten/ML-ArXiv-Papers",
    ]
    assert len(list(dataset)) == 1

    with pytest.raises(ValueError, match="does not support sliced dataset splits"):
        EmbeddingGraphBuilder(
            max_papers=1,
            dataset_split="train[:5%]",
            use_streaming=True,
            client=MagicMock(),
        )


def test_collect_papers_query_seed_and_warm_cache_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Query-mode IDs and warm-cache candidate retrieval should be deterministic."""
    _disable_embedding_dep_check(monkeypatch)

    builder = EmbeddingGraphBuilder(
        max_papers=1, use_streaming=False, client=MagicMock()
    )
    builder.client.get_paper = MagicMock(return_value=None)
    monkeypatch.setattr(builder, "_load_model", lambda: None)
    monkeypatch.setattr(
        builder,
        "_encode_texts",
        lambda texts, show_progress_bar=False: np.asarray(
            [[1.0, 0.0] for _ in texts], dtype=np.float32
        ),
    )
    monkeypatch.setattr(builder, "_select_candidates_from_loaded", lambda _: [])
    monkeypatch.setattr(builder, "_update_citation_counts", lambda _: None)

    query = "attention mechanism test query"
    papers = builder.collect_papers(query)
    expected_seed_id = _query_seed_id(query)
    assert list(papers.keys()) == [expected_seed_id]
    assert papers[expected_seed_id].is_seed is True

    fake_datasets = types.ModuleType("datasets")
    fake_load_dataset = MagicMock(
        side_effect=RuntimeError("load_dataset should not be called on warm cache")
    )
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    builder = EmbeddingGraphBuilder(
        max_papers=2, use_streaming=False, random_seed=0, client=MagicMock()
    )
    builder.embedding_cache.is_hydrated = MagicMock(return_value=True)
    builder.embedding_cache.search = MagicMock(
        return_value=[
            CacheSearchResult(
                paper_id="a",
                score=0.9,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                metadata={
                    "title": "A",
                    "abstract": "A",
                    "year": 2020,
                    "authors": ["Alice"],
                    "categories": ["cs.AI"],
                },
            ),
            CacheSearchResult(
                paper_id="b",
                score=0.8,
                embedding=np.asarray([0.5, 0.5], dtype=np.float32),
                metadata={
                    "title": "B",
                    "abstract": "B",
                    "year": 2021,
                    "authors": ["Bob"],
                    "categories": ["cs.LG"],
                },
            ),
        ]
    )
    monkeypatch.setattr(builder, "_get_model_for_encoding", lambda: _FakeEncodeModel())

    candidates = builder._select_candidates_from_loaded(
        np.asarray([1.0, 0.0], dtype=np.float32)
    )
    assert [paper_id for paper_id, _, _ in candidates] == ["a", "b"]
    fake_load_dataset.assert_not_called()


def test_empty_hydration_run_remains_incomplete_and_returns_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Empty hydration should not mark cache complete and should return no candidates."""
    _disable_embedding_dep_check(monkeypatch)
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path / "cache-root"))

    builder = EmbeddingGraphBuilder(
        max_papers=2,
        use_streaming=False,
        client=MagicMock(),
    )
    mark_hydrated_spy = MagicMock(wraps=builder.embedding_cache.mark_hydrated)
    monkeypatch.setattr(builder.embedding_cache, "mark_hydrated", mark_hydrated_spy)
    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        lambda use_streaming: ("empty-snapshot", []),
    )
    monkeypatch.setattr(
        builder,
        "_get_model_for_encoding",
        lambda: (_ for _ in ()).throw(
            AssertionError("encoding should not run for empty hydration")
        ),
    )

    candidates = builder._select_candidates_from_loaded(
        np.asarray([1.0, 0.0], dtype=np.float32)
    )

    assert candidates == []
    complete_flags = [
        call.kwargs["complete"] for call in mark_hydrated_spy.call_args_list
    ]
    assert complete_flags == [False]
    assert not builder.embedding_cache.is_hydrated(
        dataset_split=builder.dataset_split,
        corpus_size=builder.corpus_size,
        dataset_source="empty-snapshot",
    )


def test_embedding_top_k_validation_and_tie_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding top-k should validate bounds and sort cache ties by paper ID."""
    _disable_embedding_dep_check(monkeypatch)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        EmbeddingGraphBuilder(top_k=0, client=MagicMock())

    builder = EmbeddingGraphBuilder(max_papers=2, top_k=2, client=MagicMock())
    builder.embedding_cache.is_hydrated = MagicMock(return_value=True)
    builder.embedding_cache.search = MagicMock(
        return_value=[
            CacheSearchResult(
                paper_id="b",
                score=0.95,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                metadata={"title": "B", "abstract": "B", "authors": []},
            ),
            CacheSearchResult(
                paper_id="a",
                score=0.95,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                metadata={"title": "A", "abstract": "A", "authors": []},
            ),
        ]
    )

    candidates = builder._select_candidates_from_loaded(
        np.asarray([1.0, 0.0], dtype=np.float32)
    )
    assert [paper_id for paper_id, _, _ in candidates] == ["a", "b"]
