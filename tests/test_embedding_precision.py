"""Tests for EmbeddingGemma precision/autocast behavior."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from citemesh.strategies.embedding import EmbeddingGraphBuilder


def _install_fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Install a fake ``sentence_transformers`` module for precision tests.

    :param pytest.MonkeyPatch monkeypatch: Pytest monkeypatch helper.
    :return tuple[dict[str, Any], list[dict[str, Any]]]: Init kwargs and encode kwargs logs.
    """
    init_log: dict[str, Any] = {}
    encode_log: list[dict[str, Any]] = []

    class _FakeInnerBlock:
        """Container exposing ``auto_model`` like SentenceTransformer internals."""

        def __init__(self) -> None:
            """Initialize with a placeholder auto model instance."""
            self.auto_model = object()

    class _FakeSentenceTransformer:
        """Minimal fake SentenceTransformer."""

        def __init__(self, model_name_or_path: str, **kwargs: Any):
            """Capture constructor arguments.

            :param str model_name_or_path: Requested model identifier.
            :param Any kwargs: Extra kwargs passed by caller.
            """
            init_log["model_name"] = model_name_or_path
            init_log["kwargs"] = kwargs
            self._blocks = [_FakeInnerBlock()]
            init_log["auto_model_before_compile"] = self._blocks[0].auto_model

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            """Return deterministic embeddings.

            :param list[str] texts: Text inputs.
            :param Any kwargs: Encode kwargs.
            :return np.ndarray: Float32 embeddings.
            """
            encode_log.append(kwargs)
            return np.ones((len(texts), 2), dtype=np.float32)

        def __getitem__(self, index: int) -> _FakeInnerBlock:
            """Expose internal blocks through subscript access."""
            return self._blocks[index]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return init_log, encode_log


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    bf16_supported: bool,
    *,
    compile_behavior: str = "identity",
) -> tuple[object, list[tuple[Any, ...]]]:
    """Install a fake ``torch`` module for precision tests.

    :param pytest.MonkeyPatch monkeypatch: Pytest monkeypatch helper.
    :param bool cuda_available: Whether fake CUDA is available.
    :param bool bf16_supported: Whether fake CUDA reports BF16 support.
    :param str compile_behavior: ``identity``, ``tagged``, or ``raise``.
    :return tuple[object, list[tuple[Any, ...]]]: BF16 token and autocast event log.
    """
    bf16_token = object()
    autocast_log: list[tuple[Any, ...]] = []

    class _FakeAutocast:
        """Context manager used by fake torch.autocast."""

        def __init__(self, device_type: str, dtype: object):
            """Record autocast constructor call.

            :param str device_type: Device type.
            :param object dtype: Requested dtype.
            """
            autocast_log.append(("call", device_type, dtype))

        def __enter__(self) -> "_FakeAutocast":
            """Record context entry.

            :return _FakeAutocast: Self.
            """
            autocast_log.append(("enter",))
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            """Record context exit.

            :param Any exc_type: Exception type.
            :param Any exc: Exception instance.
            :param Any tb: Traceback object.
            :return None: Always returns ``None``.
            """
            del exc_type, exc, tb
            autocast_log.append(("exit",))

    def _autocast(*, device_type: str, dtype: object) -> _FakeAutocast:
        """Create fake autocast context manager.

        :param str device_type: Device type.
        :param object dtype: Requested dtype.
        :return _FakeAutocast: Context manager.
        """
        return _FakeAutocast(device_type=device_type, dtype=dtype)

    def _compile(model: object) -> object:
        """Mock ``torch.compile`` behavior for tests.

        :param object model: Model object passed to ``torch.compile``.
        :return object: Compiled replacement or original model.
        """
        if compile_behavior == "raise":
            raise RuntimeError("compile failure")
        if compile_behavior == "tagged":
            return ("compiled", model)
        return model

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = bf16_token
    fake_torch.autocast = _autocast
    fake_torch.compile = _compile
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        is_bf16_supported=lambda: bf16_supported,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    return bf16_token, autocast_log


def test_embeddinggemma_uses_bf16_model_kwargs_and_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EmbeddingGemma should use BF16 weights and autocast when CUDA supports BF16."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    bf16_token, autocast_log = _install_fake_torch(
        monkeypatch, cuda_available=True, bf16_supported=True
    )

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    builder._load_model()
    embeddings = builder._encode_texts(["seed"], show_progress_bar=False)

    assert init_log["model_name"] == "google/embeddinggemma-300m"
    assert init_log["kwargs"]["model_kwargs"]["torch_dtype"] is bf16_token
    assert init_log["kwargs"]["truncate_dim"] == 256
    assert builder.truncate_dim == 256
    assert "::truncate_dim=256" in builder.embedding_cache.model_name
    assert "::storage_precision=int8" in builder.embedding_cache.model_name
    assert embeddings.shape == (1, 2)
    assert ("call", "cuda", bf16_token) in autocast_log
    assert ("enter",) in autocast_log
    assert ("exit",) in autocast_log


def test_embeddinggemma_falls_back_when_bf16_not_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EmbeddingGemma should not request BF16/autocast on unsupported CUDA devices."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, autocast_log = _install_fake_torch(
        monkeypatch, cuda_available=True, bf16_supported=False
    )

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    builder._load_model()
    embeddings = builder._encode_texts(["seed"], show_progress_bar=False)

    assert init_log["model_name"] == "google/embeddinggemma-300m"
    assert "model_kwargs" not in init_log["kwargs"]
    assert init_log["kwargs"]["truncate_dim"] == 256
    assert embeddings.shape == (1, 2)
    assert autocast_log == []


def test_embeddinggemma_rejects_unsupported_truncate_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EmbeddingGemma should reject truncate dims outside its MRL-supported values."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )
    with pytest.raises(
        ValueError,
        match="truncate_dim=300 is not supported for google/embeddinggemma-300m",
    ):
        EmbeddingGraphBuilder(max_papers=1, truncate_dim=300, client=MagicMock())


def test_embeddinggemma_compiles_inner_transformer_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EmbeddingGemma should compile only the inner HF model when torch.compile exists."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, _autocast_log = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=True,
        compile_behavior="tagged",
    )

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    builder._load_model()

    original = init_log["auto_model_before_compile"]
    assert builder.model is not None
    assert builder.model[0].auto_model == ("compiled", original)
    assert builder._inner_model_compiled is True


def test_embeddinggemma_compile_failure_falls_back_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile failures should not prevent model loading."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, _autocast_log = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=True,
        compile_behavior="raise",
    )

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    builder._load_model()

    assert builder.model is not None
    assert builder.model[0].auto_model is init_log["auto_model_before_compile"]
    assert builder._inner_model_compiled is False


def test_non_gemma_profile_does_not_compile_inner_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-EmbeddingGemma models should skip the inner-model compile path."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, _autocast_log = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=True,
        compile_behavior="tagged",
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        client=MagicMock(),
    )
    builder._load_model()

    assert builder.model is not None
    assert builder.model[0].auto_model is init_log["auto_model_before_compile"]
    assert builder._inner_model_compiled is False


def test_embedding_cache_namespace_varies_by_storage_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache namespace should isolate incompatible storage precision settings."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    int8_builder = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="int8",
        client=MagicMock(),
    )
    f32_builder = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="float32",
        client=MagicMock(),
    )

    assert (
        int8_builder.embedding_cache.model_name
        != f32_builder.embedding_cache.model_name
    )
