"""Contract tests for embedding builder precision and hydration behavior."""

from __future__ import annotations

import logging
import types
from hashlib import sha256
from typing import Any, Iterable
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from citemesh.core import Paper
from citemesh.data import (
    DEFAULT_EMBEDDING_MODEL_FALLBACKS,
    DEFAULT_EMBEDDING_MODEL_NAME,
)
from citemesh.data.embedding_cache import CacheNamespacePayloadStats, CacheSearchResult
from citemesh.services.semantic_scholar import SemanticScholarUnavailableError
from citemesh.strategies import embedding as embedding_module
from citemesh.strategies.embedding import (
    ENCODE_BATCH_SIZE,
    EmbeddingGraphBuilder,
    _extract_dataset_paper_metadata,
    _query_seed_id,
    resolve_embedding_device,
)
from citemesh.text_batching import estimate_text_length_bucket
from tests._helpers import (
    ConstantEncodeModel,
    disable_embedding_dep_checks,
    raise_import_error,
)

_REAL_DEP_CHECK_TESTS = {
    "test_embedding_builder_requires_optional_deps",
    "test_embedding_builder_requires_modern_torch",
}


@pytest.fixture(autouse=True)
def _disable_embedding_optional_deps(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Bypass optional dependency guards unless a test exercises them directly."""
    if request.node.name in _REAL_DEP_CHECK_TESTS:
        return
    disable_embedding_dep_checks(monkeypatch)


@pytest.mark.parametrize(
    ("transformers_version", "expected_key"),
    [("4.57.1", "torch_dtype"), ("5.0.0", "dtype")],
)
def test_transformers_auto_dtype_key_uses_supported_spelling(
    monkeypatch: pytest.MonkeyPatch,
    transformers_version: str,
    expected_key: str,
) -> None:
    """Automatic model dtype should use the installed Transformers API spelling."""
    fake_transformers = types.SimpleNamespace(__version__=transformers_version)
    monkeypatch.setattr(
        embedding_module.importlib,
        "import_module",
        lambda module_name: fake_transformers,
    )

    assert embedding_module._transformers_auto_dtype_key() == expected_key


def _install_fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_model_names: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Install fake ``sentence_transformers`` module for precision tests."""
    init_log: dict[str, Any] = {}
    encode_log: list[dict[str, Any]] = []
    blocked_models = set(fail_model_names or ())

    class _FakeInnerBlock:
        def __init__(self) -> None:
            self.auto_model = object()

    class _FakeSentenceTransformer:
        def __init__(self, model_name_or_path: str, **kwargs: Any):
            init_log.setdefault("attempts", []).append(model_name_or_path)
            if model_name_or_path in blocked_models:
                raise RuntimeError(f"failed loading {model_name_or_path}")
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
    monkeypatch.setattr(
        embedding_module,
        "_import_sentence_transformer_class",
        lambda: fake_module.SentenceTransformer,
    )
    return init_log, encode_log


def _pin_model_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    builder: EmbeddingGraphBuilder,
    fingerprint: str = "test-fingerprint",
) -> None:
    """Pin deterministic model fingerprint for hydration tests."""
    monkeypatch.setattr(builder, "_resolve_model_fingerprint", lambda: fingerprint)


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    bf16_supported: bool,
    *,
    mps_available: bool = False,
    autocast_behavior: str = "identity",
    compile_behavior: str = "identity",
    capability: tuple[int, int] | None = (8, 0),
    include_tf32_global_api: bool = True,
    torch_version: str = "2.9.0",
) -> tuple[object, list[tuple[Any, ...]], object]:
    """Install fake ``torch`` module for precision tests."""
    bf16_token = object()
    autocast_log: list[tuple[Any, ...]] = []

    class _FakeAutocast:
        def __init__(self, device_type: str, dtype: object):
            autocast_log.append(("call", device_type, dtype))

        def __enter__(self) -> "_FakeAutocast":
            autocast_log.append(("enter",))
            if autocast_behavior == "raise":
                raise RuntimeError("autocast unavailable")
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            del exc_type, exc, tb
            autocast_log.append(("exit",))

    def _autocast(*, device_type: str, dtype: object) -> _FakeAutocast:
        return _FakeAutocast(device_type=device_type, dtype=dtype)

    compile_calls: list[dict[str, object]] = []

    def _compile(model: object, **kwargs: object) -> object:
        compile_calls.append({"model": model, "kwargs": dict(kwargs)})
        if compile_behavior == "raise":
            raise RuntimeError("compile failure")
        if compile_behavior == "tagged":
            return ("compiled", model)
        return model

    matmul_precision_calls: list[str] = []

    def _set_float32_matmul_precision(precision: str) -> None:
        matmul_precision_calls.append(precision)

    matmul_backend = types.SimpleNamespace()
    cudnn_backend = types.SimpleNamespace()
    cudnn_conv = types.SimpleNamespace()
    cudnn_backend.conv = cudnn_conv
    matmul_backend.fp32_precision = "none"
    cudnn_conv.fp32_precision = "none"

    if include_tf32_global_api:

        class _FakeBackends:
            def __init__(self) -> None:
                self.cuda = types.SimpleNamespace(matmul=matmul_backend)
                self.cudnn = cudnn_backend
                self._fp32_precision = "none"

            @property
            def fp32_precision(self) -> str:
                return self._fp32_precision

            @fp32_precision.setter
            def fp32_precision(self, value: str) -> None:
                self._fp32_precision = value
                self.cuda.matmul.fp32_precision = value
                self.cudnn.conv.fp32_precision = value

        fake_backends: object = _FakeBackends()
    else:
        fake_backends = types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=matmul_backend),
            cudnn=cudnn_backend,
        )
    fake_backends.mps = types.SimpleNamespace(
        is_available=lambda: mps_available,
        is_built=lambda: mps_available,
    )

    cuda_module = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        is_bf16_supported=lambda: bf16_supported,
    )
    if capability is not None:
        cuda_module.get_device_capability = lambda _index=0: capability

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = torch_version
    fake_torch.bfloat16 = bf16_token
    fake_torch.autocast = _autocast
    fake_torch.compile = _compile
    fake_torch.set_float32_matmul_precision = _set_float32_matmul_precision
    fake_torch._matmul_precision_calls = matmul_precision_calls
    fake_torch._compile_calls = compile_calls
    fake_torch.cuda = cuda_module
    fake_torch.backends = fake_backends
    monkeypatch.setattr(embedding_module, "_import_torch", lambda: fake_torch)

    return bf16_token, autocast_log, fake_torch


def test_embedding_builder_requires_optional_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding builder should fail with guidance when deps are missing."""
    monkeypatch.setattr(embedding_module, "_import_torch", raise_import_error)
    monkeypatch.setattr(
        embedding_module,
        "_import_sentence_transformer_class",
        raise_import_error,
    )
    monkeypatch.setattr(
        embedding_module,
        "_import_datasets_module",
        raise_import_error,
    )

    with pytest.raises(
        ImportError,
        match=r"Embedding strategy requires: torch, sentence-transformers\. "
        r"Install with: pip install citemesh\[embeddings\]",
    ):
        EmbeddingGraphBuilder(max_papers=5, model_name="test-model", client=MagicMock())

    with pytest.raises(
        ImportError,
        match=r"Embedding strategy requires: torch, sentence-transformers, datasets\. "
        r"Install with: pip install citemesh\[embeddings\]",
    ):
        EmbeddingGraphBuilder(
            max_papers=5,
            model_name="test-model",
            semantic_source="arxiv-corpus",
            client=MagicMock(),
        )


def test_embedding_builder_requires_modern_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding builder should require torch>=2.9 for runtime precision policy."""
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.8.1"
    monkeypatch.setattr(embedding_module, "_import_torch", lambda: fake_torch)
    monkeypatch.setattr(
        embedding_module,
        "_import_sentence_transformer_class",
        lambda: object,
    )
    monkeypatch.setattr(
        embedding_module,
        "_import_datasets_module",
        lambda: types.ModuleType("datasets"),
    )

    with pytest.raises(
        ImportError,
        match=r"Embedding strategy requires torch>=2\.9\.0",
    ):
        EmbeddingGraphBuilder(max_papers=1, client=MagicMock())


def test_embedding_runtime_precision_compile_tf32_and_logging_contracts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Runtime should enforce precision, compile, TF32, and logging policies."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._module_available",
        lambda _module_name: False,
    )

    with pytest.raises(
        ValueError,
        match=(f"truncate_dim=300 is not supported for {DEFAULT_EMBEDDING_MODEL_NAME}"),
    ):
        EmbeddingGraphBuilder(max_papers=1, truncate_dim=300, client=MagicMock())
    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        EmbeddingGraphBuilder(max_papers=1, model_name="   ", client=MagicMock())
    with pytest.raises(ValueError, match="dataset_split must be a non-empty string"):
        EmbeddingGraphBuilder(max_papers=1, dataset_split=" ", client=MagicMock())
    with pytest.raises(
        ValueError, match="corpus_size must be at least 1 when provided"
    ):
        EmbeddingGraphBuilder(max_papers=1, corpus_size=0, client=MagicMock())
    with pytest.raises(ValueError, match="encode_batch_size must be at least 1"):
        EmbeddingGraphBuilder(max_papers=1, encode_batch_size=0, client=MagicMock())

    precision_cases = [
        (True, True, "bfloat16"),
        (True, False, "float32"),
    ]
    for cuda_available, bf16_supported, expected_dtype in precision_cases:
        init_log, _ = _install_fake_sentence_transformers(monkeypatch)
        bf16_token, autocast_log, _fake_torch = _install_fake_torch(
            monkeypatch,
            cuda_available=cuda_available,
            bf16_supported=bf16_supported,
        )

        builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
        builder._load_model()
        embeddings = builder._encode_texts(["seed"], show_progress_bar=False)

        assert init_log["model_name"] == DEFAULT_EMBEDDING_MODEL_NAME
        assert init_log["kwargs"]["truncate_dim"] == 256
        assert embeddings.shape == (1, 2)
        assert init_log["kwargs"]["model_kwargs"]["attn_implementation"] == "sdpa"
        assert (
            init_log["kwargs"]["model_kwargs"].get(
                "dtype", init_log["kwargs"]["model_kwargs"].get("torch_dtype")
            )
            == "auto"
        )
        if expected_dtype == "bfloat16":
            assert ("call", "cuda", bf16_token) in autocast_log
            assert ("enter",) in autocast_log and ("exit",) in autocast_log
        else:
            assert autocast_log == []

    compile_cases = [
        (DEFAULT_EMBEDDING_MODEL_NAME, "tagged", (8, 0), "2.9.0", True),
        (DEFAULT_EMBEDDING_MODEL_NAME, "tagged", (7, 5), "2.9.0", True),
        (DEFAULT_EMBEDDING_MODEL_NAME, "tagged", (7, 5), "2.10.0", True),
        (DEFAULT_EMBEDDING_MODEL_NAME, "raise", (7, 5), "2.10.0", False),
        ("sentence-transformers/all-MiniLM-L6-v2", "tagged", (8, 0), "2.10.0", False),
    ]
    for (
        model_name,
        compile_behavior,
        capability,
        torch_version,
        expect_compiled,
    ) in compile_cases:
        init_log, _ = _install_fake_sentence_transformers(monkeypatch)
        _bf16_token, _autocast_log, _fake_torch = _install_fake_torch(
            monkeypatch,
            cuda_available=True,
            bf16_supported=True,
            compile_behavior=compile_behavior,
            capability=capability,
            torch_version=torch_version,
        )

        builder = EmbeddingGraphBuilder(
            max_papers=1,
            model_name=model_name,
            enable_torch_compile=True,
            client=MagicMock(),
        )
        monkeypatch.setattr(
            builder,
            "_should_defer_compile_for_cache_hydration",
            lambda: False,
        )
        builder._load_model()

        original = init_log["auto_model_before_compile"]
        assert builder.model is not None
        if expect_compiled:
            assert builder.model[0].auto_model == ("compiled", original)
            assert _fake_torch._compile_calls  # type: ignore[attr-defined]
            assert _fake_torch._compile_calls[-1]["kwargs"] == {}  # type: ignore[attr-defined]
        else:
            assert builder.model[0].auto_model is original
            if (
                compile_behavior == "tagged"
                and model_name != DEFAULT_EMBEDDING_MODEL_NAME
            ):
                assert _fake_torch._compile_calls == []  # type: ignore[attr-defined]
        assert builder._inner_model_compiled is expect_compiled

    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, _autocast_log, _fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=True,
        compile_behavior="tagged",
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        enable_torch_compile=False,
        client=MagicMock(),
    )
    builder._load_model()
    original = init_log["auto_model_before_compile"]
    assert builder.model is not None
    assert builder.model[0].auto_model is original
    assert builder._inner_model_compiled is False

    tf32_cases = [
        ((8, 0), True, True, "2.9.0", "tf32-matmul-high", "none", "high"),
        ((8, 0), True, False, "2.9.0", "tf32", "tf32", None),
        ((8, 0), False, True, "2.10.0", "tf32-matmul-high", None, "high"),
        ((7, 5), True, True, "2.10.0", "off", "none", None),
    ]
    for (
        capability,
        include_tf32_global_api,
        enable_torch_compile,
        torch_version,
        expected_mode,
        expected_backend_precision,
        expected_matmul_precision,
    ) in tf32_cases:
        _install_fake_sentence_transformers(monkeypatch)
        _bf16_token, _autocast_log, fake_torch = _install_fake_torch(
            monkeypatch,
            cuda_available=True,
            bf16_supported=True,
            capability=capability,
            include_tf32_global_api=include_tf32_global_api,
            torch_version=torch_version,
        )

        builder = EmbeddingGraphBuilder(
            max_papers=1,
            enable_torch_compile=enable_torch_compile,
            client=MagicMock(),
        )
        builder._load_model()

        if include_tf32_global_api:
            assert fake_torch.backends.fp32_precision == expected_backend_precision
            expected_matmul_backend = expected_backend_precision or "none"
            assert (
                fake_torch.backends.cuda.matmul.fp32_precision
                == expected_matmul_backend
            )
            assert (
                fake_torch.backends.cudnn.conv.fp32_precision == expected_matmul_backend
            )
        else:
            assert not hasattr(fake_torch.backends, "fp32_precision")
            assert fake_torch.backends.cuda.matmul.fp32_precision == "none"
            assert fake_torch.backends.cudnn.conv.fp32_precision == "none"

        if expected_matmul_precision is None:
            assert fake_torch._matmul_precision_calls == []
        else:
            assert fake_torch._matmul_precision_calls == [expected_matmul_precision]
        assert builder._tf32_mode == expected_mode

    _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, _autocast_log, _fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=True,
        capability=(7, 5),
        compile_behavior="tagged",
        torch_version="2.10.0",
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        enable_torch_compile=True,
        client=MagicMock(),
    )
    monkeypatch.setattr(
        builder,
        "_should_defer_compile_for_cache_hydration",
        lambda: False,
    )
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

    assert any("runtime: device=" in message for message in info_messages)
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
        f"Enabled torch.compile for {DEFAULT_EMBEDDING_MODEL_NAME} inner transformer"
        in message
        for message in debug_messages
    )


def test_embedding_bf16_autocast_rejection_falls_back_to_float32(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rejected bf16 autocast context must keep compute in float32."""
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, autocast_log, _fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=True,
        autocast_behavior="raise",
    )

    with caplog.at_level(logging.WARNING):
        builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    builder._load_model()

    assert builder._source_dtype_hint == "float32"
    assert builder._autocast_enabled is False
    assert (
        init_log["kwargs"]["model_kwargs"].get(
            "dtype", init_log["kwargs"]["model_kwargs"].get("torch_dtype")
        )
        == "auto"
    )
    assert autocast_log == [("call", "cuda", _bf16_token), ("enter",)]
    assert any(
        "runtime rejected that context" in record.getMessage()
        for record in caplog.records
    )


def test_embedding_runtime_policy_keeps_fp32_fallback_unmodified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Models without bf16 policy should stay on the default float32 path."""

    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, autocast_log, _fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=False,
    )
    monkeypatch.setattr(
        "citemesh.strategies.embedding._module_available",
        lambda module_name: module_name == "flash_attn",
    )
    accelerator_builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        client=MagicMock(),
    )
    accelerator_builder._load_model()

    assert init_log["kwargs"]["model_kwargs"]["attn_implementation"] == "sdpa"
    assert (
        init_log["kwargs"]["model_kwargs"].get(
            "dtype", init_log["kwargs"]["model_kwargs"].get("torch_dtype")
        )
        == "auto"
    )
    assert accelerator_builder._source_dtype_hint == "float32"
    assert accelerator_builder._attention_implementation_hint == "sdpa"
    assert autocast_log == []

    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
    )
    monkeypatch.setattr(
        "citemesh.strategies.embedding._module_available",
        lambda _module_name: True,
    )
    cpu_builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        client=MagicMock(),
    )
    cpu_builder._load_model()

    assert init_log["kwargs"]["device"] == "cpu"
    assert (
        init_log["kwargs"]["model_kwargs"].get(
            "dtype", init_log["kwargs"]["model_kwargs"].get("torch_dtype")
        )
        == "auto"
    )
    assert cpu_builder._source_dtype_hint == "float32"
    assert cpu_builder._attention_implementation_hint is None


def test_encode_texts_uses_length_bucketed_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encode batching should group similarly sized texts while preserving order."""

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        client=MagicMock(),
    )

    captured_batches: list[list[str]] = []

    class _CaptureEncodeModel:
        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            del kwargs
            captured_batches.append(list(texts))
            return np.asarray(
                [[float(len(text)), float(idx)] for idx, text in enumerate(texts)],
                dtype=np.float32,
            )

    monkeypatch.setattr(
        builder, "_get_model_for_encoding", lambda: _CaptureEncodeModel()
    )

    texts = [
        "x " * 120,
        "tiny",
        "y " * 110,
        "small words",
        "z " * 80,
    ]
    embeddings = builder._encode_texts(texts, batch_size=2, show_progress_bar=False)

    assert [len(batch) for batch in captured_batches] == [2, 2, 1]
    batch_estimates = [
        [estimate_text_length_bucket(text) for text in batch]
        for batch in captured_batches
    ]
    assert batch_estimates == sorted(
        batch_estimates, key=lambda item: (max(item), item)
    )
    assert embeddings.shape == (len(texts), 2)
    assert embeddings[:, 0].tolist() == [float(len(text)) for text in texts]


def test_embedding_compile_is_deferred_when_cache_not_hydrated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold-cache runs should defer compile to avoid hydration slowdowns."""
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, _autocast_log, fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=True,
        compile_behavior="tagged",
        torch_version="2.10.0",
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        enable_torch_compile=True,
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    )
    monkeypatch.setattr(builder, "_cache_hydrated_for_active_spec", lambda: False)
    builder._load_model()

    original = init_log["auto_model_before_compile"]
    assert builder.model is not None
    assert builder.model[0].auto_model is original
    assert builder._inner_model_compiled is False
    assert builder._compile_status_reason is not None
    assert "deferred while hydrating cache" in builder._compile_status_reason
    assert fake_torch._compile_calls == []  # type: ignore[attr-defined]


def test_embedding_compile_does_not_wait_for_candidate_cache_hydration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate mode should compile without corpus hydration metadata."""
    _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, _autocast_log, fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=True,
        compile_behavior="tagged",
        torch_version="2.10.0",
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        enable_torch_compile=True,
        semantic_source="candidates",
        client=MagicMock(),
    )
    hydration_probe = MagicMock(
        side_effect=AssertionError("candidate mode must not inspect corpus hydration")
    )
    monkeypatch.setattr(builder, "_cache_hydrated_for_active_spec", hydration_probe)

    builder._load_model()

    assert builder._inner_model_compiled is True
    assert fake_torch._compile_calls  # type: ignore[attr-defined]
    hydration_probe.assert_not_called()


def test_embedding_default_model_loads_with_fallback_chain(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Default embedding model should fail over to configured fallback checkpoint."""
    fallback_candidates = DEFAULT_EMBEDDING_MODEL_FALLBACKS[
        DEFAULT_EMBEDDING_MODEL_NAME
    ]
    fallback_model = fallback_candidates[0]
    init_log, _ = _install_fake_sentence_transformers(
        monkeypatch,
        fail_model_names={DEFAULT_EMBEDDING_MODEL_NAME},
    )
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
    )

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    caplog.clear()
    with caplog.at_level(logging.INFO):
        builder._load_model()

    assert init_log["attempts"] == [DEFAULT_EMBEDDING_MODEL_NAME, fallback_model]
    assert init_log["model_name"] == fallback_model
    log_messages = [record.getMessage() for record in caplog.records]
    assert any(
        "Using fallback embedding checkpoint:" in message for message in log_messages
    )


def test_embedding_fingerprint_uses_active_fallback_model_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fingerprint checks should bind to active fallback checkpoint identity."""
    fallback_model = DEFAULT_EMBEDDING_MODEL_FALLBACKS[DEFAULT_EMBEDDING_MODEL_NAME][0]
    _install_fake_sentence_transformers(
        monkeypatch,
        fail_model_names={DEFAULT_EMBEDDING_MODEL_NAME},
    )
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
    )

    class _FakeHfApi:
        def model_info(self, repo_id: str, revision: str) -> object:
            assert repo_id == fallback_model
            assert revision == "main"
            return types.SimpleNamespace(sha="0123456789abcdef0123456789abcdef01234567")

    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.HfApi = _FakeHfApi
    monkeypatch.setattr(
        embedding_module,
        "_import_huggingface_hub_module",
        lambda: fake_hf_module,
    )

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    builder._load_model()
    fingerprint = builder._resolve_model_fingerprint()

    assert builder._active_model_name == fallback_model
    assert (
        fingerprint == f"hf::{fallback_model}::0123456789abcdef0123456789abcdef01234567"
    )


def test_embedding_cache_namespace_partition_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Namespace identity should partition precision, source dtype, and calibration."""

    int8_builder = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="int8",
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    )
    f32_builder = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="float32",
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    )
    assert (
        int8_builder.embedding_cache.model_name
        != f32_builder.embedding_cache.model_name
    )

    monkeypatch.setattr(
        EmbeddingGraphBuilder,
        "_resolve_source_dtype_hint",
        lambda self: "float32",
    )
    f32_hint_builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())

    monkeypatch.setattr(
        EmbeddingGraphBuilder,
        "_resolve_source_dtype_hint",
        lambda self: "bfloat16",
    )
    bf16_hint_builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    assert (
        f32_hint_builder.embedding_cache.model_name
        != bf16_hint_builder.embedding_cache.model_name
    )

    monkeypatch.setattr(
        EmbeddingGraphBuilder,
        "_resolve_source_dtype_hint",
        lambda self: "float32",
    )
    int8_small = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="int8",
        calibration_sample_size=32,
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    )
    int8_large = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="int8",
        calibration_sample_size=128,
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    )
    f32_builder = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="float32",
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    )
    assert (
        int8_small.embedding_cache.model_name != int8_large.embedding_cache.model_name
    )
    assert "calibration_sample_size=" in int8_small.embedding_cache.model_name
    assert "calibration_sample_size=" not in f32_builder.embedding_cache.model_name


def test_embedding_cache_namespace_rejects_binary_prefilter_outside_int8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-int8 precision should reject binary-prefilter-specific controls."""

    with pytest.raises(
        ValueError, match="--binary-prefilter requires storage_precision='int8'"
    ):
        EmbeddingGraphBuilder(
            max_papers=1,
            storage_precision="float32",
            binary_prefilter=True,
            client=MagicMock(),
        )

    with pytest.raises(
        ValueError,
        match="--binary-rescore-multiplier requires storage_precision='int8'",
    ):
        EmbeddingGraphBuilder(
            max_papers=1,
            storage_precision="float32",
            binary_rescore_multiplier=8,
            client=MagicMock(),
        )

    with pytest.raises(
        ValueError,
        match="--calibration-sample-size requires storage_precision='int8'",
    ):
        EmbeddingGraphBuilder(
            max_papers=1,
            storage_precision="float32",
            calibration_sample_size=128,
            client=MagicMock(),
        )

    f32_default = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="float32",
        client=MagicMock(),
    )
    int8_prefilter_on = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="int8",
        binary_prefilter=True,
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    )
    int8_prefilter_off = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="int8",
        binary_prefilter=False,
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    )

    assert f32_default.binary_prefilter is False
    assert f32_default.binary_rescore_multiplier == 1
    assert (
        int8_prefilter_on.embedding_cache.model_name
        != int8_prefilter_off.embedding_cache.model_name
    )


def test_embedding_cache_namespace_matches_default_and_explicit_truncate_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default resolved truncate dim should match explicit equivalent namespace."""
    monkeypatch.setattr(
        EmbeddingGraphBuilder,
        "_resolve_source_dtype_hint",
        lambda self: "float32",
    )

    implicit = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="google/embeddinggemma-300m",
        truncate_dim=None,
        client=MagicMock(),
    )
    explicit = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="google/embeddinggemma-300m",
        truncate_dim=256,
        client=MagicMock(),
    )

    assert implicit.embedding_cache.model_name == explicit.embedding_cache.model_name


def test_embedding_model_revision_forwards_to_model_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured model revision should be forwarded to SentenceTransformer."""
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_revision="refs/pr/12",
        client=MagicMock(),
    )
    builder._load_model()

    assert init_log["kwargs"]["revision"] == "refs/pr/12"


def test_embedding_cache_rebuilds_when_model_fingerprint_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydration should clear namespace payload when model fingerprint mismatches."""
    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    _pin_model_fingerprint(monkeypatch, builder, fingerprint="fp-new")

    builder.embedding_cache.get_model_fingerprint = MagicMock(return_value="fp-old")
    builder.embedding_cache.has_cached_payload = MagicMock(return_value=True)
    builder.embedding_cache.clear = MagicMock()
    builder.embedding_cache.set_model_fingerprint = MagicMock()
    builder.embedding_cache.get_hydrated_dataset_source = MagicMock(
        return_value="cached-source"
    )
    builder.embedding_cache.is_hydrated = MagicMock(return_value=True)

    builder._ensure_cache_hydrated(use_streaming=False)

    builder.embedding_cache.clear.assert_called_once()
    builder.embedding_cache.set_model_fingerprint.assert_called_once_with("fp-new")


def test_embedding_cache_offline_fingerprint_lookup_contracts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Offline lookup outcomes should preserve cache safety across identity states."""

    compatible_fp = "hf::org/offline-test::0123456789abcdef0123456789abcdef01234567"
    cases = [
        {
            "label": "compatible cached fingerprint is reused",
            "model_name": "org/offline-test",
            "model_revision": "0123456789abcdef0123456789abcdef01234567",
            "has_cached_payload": True,
            "cached_fingerprint": compatible_fp,
            "expected_resolved": compatible_fp,
            "expect_clear": False,
            "expect_set_fingerprint": None,
            "expected_log_fragment": "Reusing compatible cached fingerprint",
        },
        {
            "label": "sha-only cached fingerprint clears payload",
            "model_name": "org/offline-strict",
            "model_revision": None,
            "has_cached_payload": True,
            "cached_fingerprint": (
                "hf::org/offline-strict::0123456789abcdef0123456789abcdef01234567"
            ),
            "expected_resolved": "hf::org/offline-strict::revision=main::offline-unverified",
            "expect_clear": True,
            "expect_set_fingerprint": "hf::org/offline-strict::revision=main::offline-unverified",
            "expected_log_fragment": "incompatible with requested identity",
        },
        {
            "label": "incompatible cached fingerprint clears payload",
            "model_name": "org/offline-test",
            "model_revision": "refs/pr/12",
            "has_cached_payload": True,
            "cached_fingerprint": compatible_fp,
            "expected_resolved": "hf::org/offline-test::revision=refs/pr/12::offline-unverified",
            "expect_clear": True,
            "expect_set_fingerprint": "hf::org/offline-test::revision=refs/pr/12::offline-unverified",
            "expected_log_fragment": "is incompatible with requested identity",
        },
        {
            "label": "missing cached fingerprint reuses payload with fallback identity",
            "model_name": "org/offline-no-fingerprint",
            "model_revision": "refs/pr/12",
            "has_cached_payload": True,
            "cached_fingerprint": None,
            "expected_resolved": (
                "hf::org/offline-no-fingerprint::revision=refs/pr/12::offline-unverified"
            ),
            "expect_clear": False,
            "expect_set_fingerprint": (
                "hf::org/offline-no-fingerprint::revision=refs/pr/12::offline-unverified"
            ),
            "expected_log_fragment": "Reusing cached payload with fallback identity",
        },
        {
            "label": "offline initialization sets fallback for empty namespace",
            "model_name": "org/offline-init",
            "model_revision": "refs/pr/34",
            "has_cached_payload": False,
            "cached_fingerprint": None,
            "expected_resolved": "hf::org/offline-init::revision=refs/pr/34::offline-unverified",
            "expect_clear": False,
            "expect_set_fingerprint": (
                "hf::org/offline-init::revision=refs/pr/34::offline-unverified"
            ),
            "expected_log_fragment": "offline initialization",
        },
    ]

    for case in cases:
        caplog.clear()
        builder = EmbeddingGraphBuilder(
            max_papers=1,
            model_name=case["model_name"],
            model_revision=case["model_revision"],
            client=MagicMock(),
        )
        builder.embedding_cache.has_cached_payload = MagicMock(
            return_value=bool(case["has_cached_payload"])
        )
        builder.embedding_cache.get_model_fingerprint = MagicMock(
            return_value=case["cached_fingerprint"]
        )
        builder.embedding_cache.clear = MagicMock()
        builder.embedding_cache.set_model_fingerprint = MagicMock()
        builder._resolve_model_fingerprint = MagicMock(
            side_effect=RuntimeError("network unavailable")
        )

        with caplog.at_level(logging.WARNING):
            builder._ensure_cache_model_fingerprint()

        assert builder._resolved_model_fingerprint == case["expected_resolved"], case[
            "label"
        ]
        if case["expect_clear"]:
            builder.embedding_cache.clear.assert_called_once()
        else:
            assert builder.embedding_cache.clear.call_count == 0

        expected_set_fingerprint = case["expect_set_fingerprint"]
        if expected_set_fingerprint is None:
            builder.embedding_cache.set_model_fingerprint.assert_not_called()
        else:
            builder.embedding_cache.set_model_fingerprint.assert_called_once_with(
                expected_set_fingerprint
            )
        assert any(
            case["expected_log_fragment"] in record.getMessage()
            for record in caplog.records
        ), case["label"]


def test_embedding_cache_sets_missing_cached_fingerprint_after_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached payload without fingerprint should be migrated to a resolved fingerprint."""

    builder = EmbeddingGraphBuilder(
        max_papers=1, model_name="org/needs-fingerprint", client=MagicMock()
    )
    builder.embedding_cache.has_cached_payload = MagicMock(return_value=True)
    builder.embedding_cache.get_model_fingerprint = MagicMock(return_value=None)
    builder.embedding_cache.set_model_fingerprint = MagicMock()

    builder._resolve_model_fingerprint = MagicMock(return_value="resolved-fp")
    builder.embedding_cache.clear = MagicMock()

    builder._ensure_cache_model_fingerprint()

    builder.embedding_cache.set_model_fingerprint.assert_called_once_with("resolved-fp")
    assert builder._resolved_model_fingerprint == "resolved-fp"
    assert builder.embedding_cache.clear.call_count == 0


def test_embedding_fingerprint_resolution_contracts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Fingerprint resolution should cover fail-closed, snapshot, and artifact paths."""

    class _FailingHfApi:
        def model_info(self, repo_id: str, revision: str) -> object:
            del repo_id, revision
            raise RuntimeError("network unavailable")

    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.HfApi = _FailingHfApi
    monkeypatch.setattr(
        embedding_module,
        "_import_huggingface_hub_module",
        lambda: fake_hf_module,
    )

    fail_closed_builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/test-model",
        model_revision="main",
        client=MagicMock(),
    )
    with pytest.raises(RuntimeError, match="Could not resolve Hugging Face commit SHA"):
        fail_closed_builder._resolve_model_fingerprint()

    def _snapshot_download_with_sha(
        repo_id: str, revision: str, local_files_only: bool
    ) -> str:
        del repo_id, revision
        assert local_files_only is True
        return "/tmp/models--org--test-model/snapshots/0123456789abcdef0123456789abcdef01234567"

    fake_hf_module.snapshot_download = _snapshot_download_with_sha
    snapshot_builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/test-model",
        model_revision="refs/pr/12",
        client=MagicMock(),
    )
    assert (
        snapshot_builder._resolve_model_fingerprint()
        == "hf::org/test-model::0123456789abcdef0123456789abcdef01234567"
    )

    snapshot_root = tmp_path / "models--org--artifact-model"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    config_bytes = b'{"architectures":["FakeModel"]}\n'
    weights_bytes = b"weights-v1"
    (snapshot_root / "config.json").write_bytes(config_bytes)
    (snapshot_root / "model.safetensors").write_bytes(weights_bytes)

    def _snapshot_download_artifact_only(
        repo_id: str, revision: str, local_files_only: bool
    ) -> str:
        assert repo_id == "org/artifact-model"
        assert revision == "refs/pr/7"
        assert local_files_only is True
        return str(snapshot_root)

    fake_hf_module.snapshot_download = _snapshot_download_artifact_only
    artifact_builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/artifact-model",
        model_revision="refs/pr/7",
        client=MagicMock(),
    )

    expected_config = sha256(config_bytes).hexdigest()
    expected_weights = sha256(weights_bytes).hexdigest()
    assert artifact_builder._resolve_model_fingerprint() == (
        "hf::org/artifact-model::revision=refs/pr/7"
        f"::config={expected_config}::weights={expected_weights}"
    )


def test_metadata_and_streaming_loader_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata parsing and streaming hydration fallback should stay deterministic."""

    snapshot = _extract_dataset_paper_metadata(
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
    versioned = _extract_dataset_paper_metadata(
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
    monkeypatch.setattr(
        embedding_module,
        "_import_datasets_module",
        lambda: fake_datasets,
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1, use_streaming=True, client=MagicMock()
    )
    selected_name, dataset = builder._load_dataset_for_hydration(use_streaming=True)

    assert selected_name == "CShorten/ML-ArXiv-Papers"
    assert [name for name, _, _ in load_calls] == [
        "librarian-bots/arxiv-metadata-snapshot",
        "CShorten/ML-ArXiv-Papers",
    ]
    assert len(list(dataset)) == 1
    assert load_calls[0][1] == "train"

    load_calls.clear()
    builder = EmbeddingGraphBuilder(
        max_papers=1,
        use_streaming=False,
        corpus_size=5,
        client=MagicMock(),
    )
    selected_name, dataset = builder._load_dataset_for_hydration(use_streaming=False)
    assert selected_name == "CShorten/ML-ArXiv-Papers"
    assert [name for name, _, _ in load_calls] == [
        "librarian-bots/arxiv-metadata-snapshot",
        "CShorten/ML-ArXiv-Papers",
    ]
    # Capped hydration loads the full split; newest-N selection happens by
    # arXiv ID chronology after load, not via positional split slicing.
    assert load_calls[0][1] == "train"
    assert len(list(dataset)) == 1

    load_calls.clear()
    selected_name, dataset = builder._load_dataset_for_hydration(
        use_streaming=False,
        row_limit=3,
        row_offset=5,
    )
    assert selected_name == "CShorten/ML-ArXiv-Papers"
    assert [name for name, _, _ in load_calls] == [
        "librarian-bots/arxiv-metadata-snapshot",
        "CShorten/ML-ArXiv-Papers",
    ]
    assert load_calls[0][1] == "train[5:8]"
    assert len(list(dataset)) == 1

    load_calls.clear()
    streaming_builder = EmbeddingGraphBuilder(
        max_papers=1, use_streaming=True, client=MagicMock()
    )
    selected_name, dataset = streaming_builder._load_dataset_for_hydration(
        use_streaming=True,
        row_limit=1,
        row_offset=1,
    )
    assert selected_name == "CShorten/ML-ArXiv-Papers"
    assert [name for name, _, _ in load_calls] == [
        "librarian-bots/arxiv-metadata-snapshot",
        "CShorten/ML-ArXiv-Papers",
    ]
    assert load_calls[0][1] == "train"
    assert len(list(dataset)) == 0

    with pytest.raises(ValueError, match="row_offset must be at least 0"):
        builder._load_dataset_for_hydration(use_streaming=False, row_offset=-1)

    with pytest.raises(ValueError, match="does not support sliced dataset splits"):
        EmbeddingGraphBuilder(
            max_papers=1,
            dataset_split="train[:5%]",
            use_streaming=True,
            semantic_source="arxiv-corpus",
            client=MagicMock(),
        )

    candidate_builder = EmbeddingGraphBuilder(
        max_papers=1,
        dataset_split="train[:5%]",
        use_streaming=True,
        semantic_source="candidates",
        client=MagicMock(),
    )
    assert candidate_builder.semantic_source == "candidates"


def test_arxiv_id_chronology_key_parses_both_styles() -> None:
    """Submission chronology must parse new-style, old-style, and prefixed IDs."""
    key = embedding_module._arxiv_id_chronology_key
    assert key("2508.01234") == (2025, 8, 1234)
    assert key("2508.01234v2") == (2025, 8, 1234)
    assert key("arXiv:1706.03762") == (2017, 6, 3762)
    assert key("0704.0001") == (2007, 4, 1)
    assert key("solv-int/9912015") == (1999, 12, 15)
    assert key("math.GT/0309136") == (2003, 9, 136)
    assert key("hep-th/0504010v3") == (2005, 4, 10)
    assert key("fallback-paper") is None
    assert key("") is None
    assert key(None) is None


@pytest.mark.parametrize(
    ("raw_id", "expected"),
    [
        ("2508.01234v2", "arxiv:2508.01234"),
        ("arXiv:2508.01234v2", "arxiv:2508.01234"),
        ("hep-th/9901001v3", "arxiv:hep-th/9901001"),
        ("arxiv_12", "arxiv_12"),
        ("S2:opaque-id", "S2:opaque-id"),
    ],
)
def test_embedding_dataset_ids_share_arxiv_recognition(
    raw_id: str, expected: str
) -> None:
    """Dataset ID normalization should use shared arXiv recognition rules."""
    assert embedding_module._canonicalize_embedding_paper_id(raw_id) == expected


def test_capped_hydration_selects_newest_rows_by_arxiv_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capped hydration must rank by ID chronology, not dataset row order."""
    # Mirror the real snapshot shape: newest update_date first (including a
    # recently revised OLD paper at row 0), pre-2007 ID block at the tail.
    records = [
        {"id": "1203.0127", "title": "Revised old paper", "abstract": "A."},
        {"id": "2608.01234", "title": "Newest submission", "abstract": "A."},
        {"id": "2607.00042", "title": "Recent submission", "abstract": "A."},
        {"id": "2412.05000", "title": "Late 2024 submission", "abstract": "A."},
        {"id": "hep-th/0504010", "title": "Old-style paper", "abstract": "A."},
        {"id": "solv-int/9912015", "title": "Tail 1999 paper", "abstract": "A."},
    ]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda name, split, streaming=False: iter(records)
    monkeypatch.setattr(
        embedding_module, "_import_datasets_module", lambda: fake_datasets
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1, use_streaming=True, corpus_size=3, client=MagicMock()
    )
    _, dataset = builder._load_dataset_for_hydration(use_streaming=True)
    assert [record["title"] for record in dataset] == [
        "Late 2024 submission",
        "Recent submission",
        "Newest submission",
    ]

    # Sources without parseable arXiv IDs keep the head of the split.
    no_id_records = [
        {"id": f"paper-{idx}", "title": f"Paper {idx}", "abstract": "A."}
        for idx in range(5)
    ]
    fake_datasets.load_dataset = lambda name, split, streaming=False: iter(
        no_id_records
    )
    _, dataset = builder._load_dataset_for_hydration(use_streaming=True)
    assert [record["title"] for record in dataset] == ["Paper 0", "Paper 1", "Paper 2"]


def test_select_newest_corpus_rows_uses_dataset_id_column() -> None:
    """Arrow-style datasets select newest rows via the id column, not full records."""

    class _FakeArrowDataset:
        """Minimal Dataset stand-in exposing column_names/select/__getitem__."""

        column_names = ["id", "title", "abstract"]

        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows
            self.selected_indices: list[int] | None = None

        def __getitem__(self, column: str) -> list[Any]:
            return [row[column] for row in self.rows]

        def select(self, indices: list[int]) -> list[dict[str, Any]]:
            self.selected_indices = list(indices)
            return [self.rows[idx] for idx in indices]

    rows = [
        {"id": "2601.00001", "title": "Jan 2026", "abstract": "A."},
        {"id": "9107.00001", "title": "Century-pivoted to 1991", "abstract": "A."},
        {"id": "astro-ph/9204001", "title": "April 1992", "abstract": "A."},
        {"id": "2603.00001", "title": "Mar 2026", "abstract": "A."},
        {"id": "not-an-id", "title": "Skipped", "abstract": "A."},
    ]
    fake_dataset = _FakeArrowDataset(rows)
    builder = EmbeddingGraphBuilder(
        max_papers=1, use_streaming=False, corpus_size=2, client=MagicMock()
    )
    selected = builder._select_newest_corpus_rows(fake_dataset, "fake/source")
    assert fake_dataset.selected_indices == [0, 3]
    assert [row["title"] for row in selected] == ["Jan 2026", "Mar 2026"]


def test_collect_papers_query_seed_and_warm_cache_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Query-mode IDs and warm-cache candidate retrieval should be deterministic."""

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
    monkeypatch.setattr(
        builder,
        "_select_candidates",
        lambda _seed_embedding, *, use_streaming: [],
    )
    monkeypatch.setattr(builder, "_update_citation_counts", lambda _: None)

    query = "attention mechanism test query"
    papers = builder.collect_papers(query)
    expected_seed_id = _query_seed_id(query)
    assert list(papers.keys()) == [expected_seed_id]
    assert papers[expected_seed_id].is_seed is True

    builder = EmbeddingGraphBuilder(
        max_papers=2, use_streaming=False, client=MagicMock()
    )
    _pin_model_fingerprint(monkeypatch, builder)
    builder.embedding_cache.is_hydrated = MagicMock(return_value=True)
    builder.embedding_cache.get_hydrated_dataset_source = MagicMock(
        return_value="librarian-bots/arxiv-metadata-snapshot"
    )
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
    monkeypatch.setattr(
        builder, "_get_model_for_encoding", lambda: ConstantEncodeModel()
    )
    fake_load_dataset_for_hydration = MagicMock(
        return_value=("librarian-bots/arxiv-metadata-snapshot", [])
    )
    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        fake_load_dataset_for_hydration,
    )

    candidates = builder._select_candidates(
        np.asarray([1.0, 0.0], dtype=np.float32),
        use_streaming=False,
    )
    assert [paper_id for paper_id, _, _ in candidates] == ["a", "b"]
    fake_load_dataset_for_hydration.assert_not_called()


def test_collect_papers_formats_query_and_paper_seeds_in_expected_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed embedding should use query prompts only for free-text query seeds."""

    def _build_builder() -> tuple[
        EmbeddingGraphBuilder, list[str], list[dict[str, str]]
    ]:
        builder = EmbeddingGraphBuilder(
            max_papers=1, use_streaming=False, client=MagicMock()
        )
        query_calls: list[str] = []
        document_calls: list[dict[str, str]] = []
        builder.model_profile = types.SimpleNamespace(
            format_query=lambda text, _metadata: (
                query_calls.append(text) or f"Q::{text}"
            ),
            format_document=lambda metadata: (
                document_calls.append(dict(metadata))
                or f"D::{metadata.get('title', '')}::{metadata.get('abstract', '')}"
            ),
        )
        monkeypatch.setattr(builder, "_load_model", lambda: None)
        monkeypatch.setattr(
            builder,
            "_select_candidates",
            lambda _seed_embedding, *, use_streaming: [],
        )
        monkeypatch.setattr(builder, "_update_citation_counts", lambda _papers: None)
        return builder, query_calls, document_calls

    query_builder, query_calls, query_documents = _build_builder()
    query_texts: list[str] = []
    monkeypatch.setattr(
        query_builder,
        "_encode_texts",
        lambda texts, show_progress_bar=False: (
            query_texts.extend(texts),
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )[1],
    )
    query_builder.client.get_paper = MagicMock(return_value=None)

    query_builder.collect_papers("attention routing")

    query_builder.client.get_paper.assert_called_once_with(
        "attention routing", raise_on_unavailable=True
    )
    assert query_calls == ["attention routing"]
    assert query_documents == []
    assert query_texts == ["Q::attention routing"]

    paper_builder, paper_queries, paper_documents = _build_builder()
    paper_texts: list[str] = []
    monkeypatch.setattr(
        paper_builder,
        "_encode_texts",
        lambda texts, show_progress_bar=False: (
            paper_texts.extend(texts),
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )[1],
    )
    paper_builder.client.get_paper = MagicMock(
        return_value=Paper(
            paper_id="paper-1",
            title="Seed Title",
            abstract="Seed Abstract",
            year=2024,
            is_seed=True,
        )
    )

    paper_builder.collect_papers("paper-1")

    paper_builder.client.get_paper.assert_called_once_with(
        "paper-1", raise_on_unavailable=True
    )
    assert paper_queries == []
    assert paper_documents == [{"title": "Seed Title", "abstract": "Seed Abstract"}]
    assert paper_texts == ["D::Seed Title::Seed Abstract"]

    outage_builder, outage_queries, outage_documents = _build_builder()
    outage_builder.client.get_paper = MagicMock(
        side_effect=SemanticScholarUnavailableError("Semantic Scholar is unavailable")
    )

    with pytest.raises(
        SemanticScholarUnavailableError, match="Semantic Scholar is unavailable"
    ):
        outage_builder.collect_papers("arxiv:1706.03762")

    assert outage_queries == []
    assert outage_documents == []

    prefetched_builder, prefetched_queries, prefetched_documents = _build_builder()
    prefetched_texts: list[str] = []
    monkeypatch.setattr(
        prefetched_builder,
        "_encode_texts",
        lambda texts, show_progress_bar=False: (
            prefetched_texts.extend(texts),
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )[1],
    )
    prefetched_builder.client.get_paper = MagicMock(
        side_effect=AssertionError("seed should be reused from caller")
    )

    prefetched_builder.collect_papers(
        "paper-1",
        seed_paper=Paper(
            paper_id="paper-1",
            title="Seed Title",
            abstract="Seed Abstract",
            year=2024,
            is_seed=True,
        ),
    )

    assert prefetched_queries == []
    assert prefetched_documents == [
        {"title": "Seed Title", "abstract": "Seed Abstract"}
    ]
    assert prefetched_texts == ["D::Seed Title::Seed Abstract"]


def test_collect_papers_dataset_source_revalidation_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Dataset-source mismatches should revalidate or fail closed when unresolved."""
    source = "librarian-bots/arxiv-metadata-snapshot"

    def _build_hydrated_builder(cache_root: str) -> EmbeddingGraphBuilder:
        monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path / cache_root))
        local_builder = EmbeddingGraphBuilder(
            max_papers=2,
            storage_precision="float32",
            use_streaming=False,
            client=MagicMock(),
        )
        _pin_model_fingerprint(monkeypatch, local_builder)
        local_builder.embedding_cache.mark_hydrated(
            dataset_source=source,
            dataset_split=local_builder.dataset_split,
            corpus_size=local_builder.corpus_size,
            complete=True,
        )
        return local_builder

    revalidate_builder = _build_hydrated_builder("cache-root-revalidate")
    mark_hydrated_spy = MagicMock(
        wraps=revalidate_builder.embedding_cache.mark_hydrated
    )
    monkeypatch.setattr(
        revalidate_builder.embedding_cache, "mark_hydrated", mark_hydrated_spy
    )
    loaded_sources: list[tuple[bool, str | None]] = []

    def fake_load_dataset_for_hydration(
        use_streaming: bool, preferred_dataset_source: str | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        loaded_sources.append((use_streaming, preferred_dataset_source))
        return "CShorten/ML-ArXiv-Papers", []

    monkeypatch.setattr(
        revalidate_builder,
        "_load_dataset_for_hydration",
        fake_load_dataset_for_hydration,
    )
    monkeypatch.setattr(
        revalidate_builder,
        "_get_model_for_encoding",
        lambda: ConstantEncodeModel(),
    )

    candidates = revalidate_builder._select_candidates(
        np.asarray([1.0, 0.0], dtype=np.float32),
        use_streaming=False,
    )
    assert candidates == []
    assert loaded_sources == [(False, source)]
    complete_flags = [
        call.kwargs["complete"] for call in mark_hydrated_spy.call_args_list
    ]
    assert complete_flags == [False]

    fail_closed_builder = _build_hydrated_builder("cache-root-fail-closed")
    is_hydrated_spy = MagicMock(wraps=fail_closed_builder.embedding_cache.is_hydrated)
    monkeypatch.setattr(
        fail_closed_builder.embedding_cache,
        "is_hydrated",
        is_hydrated_spy,
    )
    monkeypatch.setattr(
        fail_closed_builder,
        "_load_dataset_for_hydration",
        MagicMock(side_effect=RuntimeError("dataset unavailable")),
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to resolve hydration dataset source",
    ) as exc_info:
        fail_closed_builder._select_candidates(
            np.asarray([1.0, 0.0], dtype=np.float32),
            use_streaming=False,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "dataset unavailable"
    assert is_hydrated_spy.call_count == 1
    assert is_hydrated_spy.call_args.kwargs.get("dataset_source") == source


def test_full_corpus_hydrated_cache_refreshes_incremental_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydrated full-corpus cache should append only upstream row-count deltas."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        corpus_size=None,
        use_streaming=False,
        client=MagicMock(),
    )
    _pin_model_fingerprint(monkeypatch, builder)
    builder.embedding_cache.is_hydrated = MagicMock(return_value=True)
    builder.embedding_cache.get_hydrated_dataset_source = MagicMock(return_value=source)
    builder.embedding_cache.payload_stats = MagicMock(
        side_effect=[
            CacheNamespacePayloadStats(
                file_count=2,
                size_bytes=1024,
                sqlite_rows=100,
                embedding_rows=100,
                hydration_complete=True,
                hydration_split="train",
                hydration_corpus_size="all",
                hydration_dataset_source=source,
            ),
            CacheNamespacePayloadStats(
                file_count=2,
                size_bytes=1024,
                sqlite_rows=110,
                embedding_rows=110,
                hydration_complete=True,
                hydration_split="train",
                hydration_corpus_size="all",
                hydration_dataset_source=source,
            ),
        ]
    )
    builder.embedding_cache.clear = MagicMock()
    builder.embedding_cache.mark_hydrated = MagicMock()
    monkeypatch.setattr(builder, "_resolve_dataset_split_row_count", lambda _: 110)
    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        MagicMock(
            return_value=(
                source,
                [
                    {"id": f"new-{idx}", "title": f"Title {idx}", "abstract": "A"}
                    for idx in range(10)
                ],
            )
        ),
    )
    monkeypatch.setattr(builder, "_cache_metadata_batch", lambda batch: len(batch))

    builder._ensure_cache_hydrated(use_streaming=False)

    builder._load_dataset_for_hydration.assert_called_once_with(
        use_streaming=False,
        preferred_dataset_source=source,
        row_limit=10,
        row_offset=100,
        allow_source_fallback=False,
    )
    assert builder.embedding_cache.clear.call_count == 0
    builder.embedding_cache.mark_hydrated.assert_called_once_with(
        dataset_source=source,
        dataset_split=builder.dataset_split,
        corpus_size=builder.corpus_size,
        complete=True,
    )


def test_incomplete_full_corpus_cache_resumes_from_cached_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incomplete full-corpus hydration should resume from the cached row boundary."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        corpus_size=None,
        use_streaming=False,
        client=MagicMock(),
    )
    _pin_model_fingerprint(monkeypatch, builder)
    builder.embedding_cache.is_hydrated = MagicMock(return_value=False)
    builder.embedding_cache.get_hydrated_dataset_source = MagicMock(return_value=source)
    builder.embedding_cache.payload_stats = MagicMock(
        side_effect=[
            CacheNamespacePayloadStats(
                file_count=2,
                size_bytes=1024,
                sqlite_rows=100,
                embedding_rows=100,
                hydration_complete=False,
                hydration_split="train",
                hydration_corpus_size="all",
                hydration_dataset_source=source,
            ),
            CacheNamespacePayloadStats(
                file_count=2,
                size_bytes=1024,
                sqlite_rows=150,
                embedding_rows=150,
                hydration_complete=False,
                hydration_split="train",
                hydration_corpus_size="all",
                hydration_dataset_source=source,
            ),
        ]
    )
    builder.embedding_cache.mark_hydrated = MagicMock()
    builder.embedding_cache.clear_hydration_rowcount_reconciliation = MagicMock()
    clear_cache_mock = MagicMock()
    monkeypatch.setattr(builder, "_clear_embedding_cache", clear_cache_mock)
    monkeypatch.setattr(builder, "_resolve_dataset_split_row_count", lambda _: 150)
    monkeypatch.setattr(builder, "_ensure_int8_calibration_ranges", lambda **_: None)
    load_mock = MagicMock(
        return_value=(
            source,
            [
                {"id": f"new-{idx}", "title": f"Title {idx}", "abstract": "A"}
                for idx in range(50)
            ],
        )
    )
    monkeypatch.setattr(builder, "_load_dataset_for_hydration", load_mock)
    monkeypatch.setattr(builder, "_cache_metadata_batch", lambda batch: len(batch))

    builder._ensure_cache_hydrated(use_streaming=False)

    load_mock.assert_called_once_with(
        use_streaming=False,
        preferred_dataset_source=source,
        row_limit=50,
        row_offset=100,
        allow_source_fallback=False,
    )
    clear_cache_mock.assert_not_called()
    builder.embedding_cache.mark_hydrated.assert_called_once_with(
        dataset_source=source,
        dataset_split=builder.dataset_split,
        corpus_size=builder.corpus_size,
        complete=True,
    )


def test_incomplete_full_corpus_resume_without_row_count_marks_clean_eof_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean unknown-cardinality EOF should complete the reusable cache."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        corpus_size=None,
        use_streaming=False,
        client=MagicMock(),
    )
    monkeypatch.setattr(builder, "_ensure_cache_model_fingerprint", lambda: None)
    builder.embedding_cache.is_hydrated = MagicMock(return_value=False)
    builder.embedding_cache.get_hydrated_dataset_source = MagicMock(return_value=source)
    builder.embedding_cache.payload_stats = MagicMock(
        return_value=CacheNamespacePayloadStats(
            file_count=2,
            size_bytes=1024,
            sqlite_rows=100,
            embedding_rows=100,
            hydration_complete=False,
            hydration_split="train",
            hydration_corpus_size="all",
            hydration_dataset_source=source,
        )
    )
    builder.embedding_cache.mark_hydrated = MagicMock()
    builder.embedding_cache.clear_hydration_rowcount_reconciliation = MagicMock()
    clear_cache_mock = MagicMock()
    monkeypatch.setattr(builder, "_clear_embedding_cache", clear_cache_mock)
    monkeypatch.setattr(builder, "_resolve_dataset_split_row_count", lambda _: None)
    monkeypatch.setattr(builder, "_ensure_int8_calibration_ranges", lambda **_: None)
    load_mock = MagicMock(return_value=(source, []))
    monkeypatch.setattr(builder, "_load_dataset_for_hydration", load_mock)

    builder._ensure_cache_hydrated(use_streaming=False)

    load_mock.assert_called_once_with(
        use_streaming=False,
        preferred_dataset_source=source,
        row_limit=None,
        row_offset=100,
        allow_source_fallback=False,
    )
    builder.embedding_cache.mark_hydrated.assert_called_once_with(
        dataset_source=source,
        dataset_split=builder.dataset_split,
        corpus_size=builder.corpus_size,
        complete=True,
    )
    builder.embedding_cache.clear_hydration_rowcount_reconciliation.assert_called_once()
    clear_cache_mock.assert_not_called()


def test_incomplete_full_corpus_resume_memoizes_duplicate_id_deficit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fully consumed resume slice should not rebuild for duplicate source IDs."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        corpus_size=None,
        use_streaming=False,
        client=MagicMock(),
    )
    monkeypatch.setattr(builder, "_ensure_cache_model_fingerprint", lambda: None)
    builder.embedding_cache.is_hydrated = MagicMock(return_value=False)
    builder.embedding_cache.get_hydrated_dataset_source = MagicMock(return_value=source)
    builder.embedding_cache.payload_stats = MagicMock(
        side_effect=[
            CacheNamespacePayloadStats(
                file_count=2,
                size_bytes=1024,
                sqlite_rows=2,
                embedding_rows=2,
                hydration_complete=False,
                hydration_split="train",
                hydration_corpus_size="all",
                hydration_dataset_source=source,
            ),
            CacheNamespacePayloadStats(
                file_count=2,
                size_bytes=1024,
                sqlite_rows=4,
                embedding_rows=4,
                hydration_complete=False,
                hydration_split="train",
                hydration_corpus_size="all",
                hydration_dataset_source=source,
            ),
        ]
    )
    builder.embedding_cache.mark_hydrated = MagicMock()
    builder.embedding_cache.set_hydration_rowcount_reconciliation = MagicMock()
    clear_cache_mock = MagicMock()
    monkeypatch.setattr(builder, "_clear_embedding_cache", clear_cache_mock)
    monkeypatch.setattr(builder, "_resolve_dataset_split_row_count", lambda _: 5)
    monkeypatch.setattr(builder, "_ensure_int8_calibration_ranges", lambda **_: None)
    load_mock = MagicMock(
        return_value=(
            source,
            [
                {"id": "p0", "title": "Duplicate", "abstract": "A"},
                {"id": "p2", "title": "Two", "abstract": "A"},
                {"id": "p3", "title": "Three", "abstract": "A"},
            ],
        )
    )
    monkeypatch.setattr(builder, "_load_dataset_for_hydration", load_mock)
    monkeypatch.setattr(builder, "_cache_metadata_batch", lambda batch: len(batch))

    builder._ensure_cache_hydrated(use_streaming=False)

    load_mock.assert_called_once_with(
        use_streaming=False,
        preferred_dataset_source=source,
        row_limit=3,
        row_offset=2,
        allow_source_fallback=False,
    )
    clear_cache_mock.assert_not_called()
    builder.embedding_cache.mark_hydrated.assert_called_once_with(
        dataset_source=source,
        dataset_split=builder.dataset_split,
        corpus_size=builder.corpus_size,
        complete=True,
    )
    builder.embedding_cache.set_hydration_rowcount_reconciliation.assert_called_once_with(
        upstream_rows=5,
        cached_rows=4,
    )


def test_incomplete_full_corpus_resume_does_not_complete_failed_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source exception must propagate before hydration is marked complete."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        corpus_size=None,
        use_streaming=True,
        client=MagicMock(),
    )
    builder.embedding_cache.payload_stats = MagicMock(
        return_value=CacheNamespacePayloadStats(
            file_count=2,
            size_bytes=1024,
            sqlite_rows=2,
            embedding_rows=2,
            hydration_complete=False,
            hydration_split="train",
            hydration_corpus_size="all",
            hydration_dataset_source=source,
        )
    )
    builder.embedding_cache.mark_hydrated = MagicMock()
    monkeypatch.setattr(builder, "_resolve_dataset_split_row_count", lambda _: None)
    monkeypatch.setattr(builder, "_ensure_int8_calibration_ranges", lambda **_: None)

    def failed_source() -> Iterable[dict[str, Any]]:
        """Yield one row before simulating a source iteration failure."""
        yield {"id": "p2", "title": "Two", "abstract": "A"}
        raise RuntimeError("source iteration failed")

    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        MagicMock(return_value=(source, failed_source())),
    )

    with pytest.raises(RuntimeError, match="source iteration failed"):
        builder._resume_incomplete_full_corpus_cache(
            use_streaming=True,
            cached_dataset_source=source,
        )

    builder.embedding_cache.mark_hydrated.assert_not_called()


def test_initial_full_corpus_hydration_memoizes_duplicate_id_row_deficit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initial full hydration should reconcile canonical duplicates across flushes."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        corpus_size=None,
        use_streaming=False,
        client=MagicMock(),
    )
    monkeypatch.setattr(builder, "_ensure_cache_model_fingerprint", lambda: None)
    monkeypatch.setattr("citemesh.strategies.embedding.HYDRATION_FLUSH_SIZE", 2)
    builder.embedding_cache.is_hydrated = MagicMock(return_value=False)
    builder.embedding_cache.get_hydrated_dataset_source = MagicMock(return_value=None)
    builder.embedding_cache.mark_hydrated = MagicMock()
    builder.embedding_cache.set_hydration_rowcount_reconciliation = MagicMock()
    builder.embedding_cache.clear_hydration_rowcount_reconciliation = MagicMock()
    monkeypatch.setattr(builder, "_clear_embedding_cache", MagicMock())
    monkeypatch.setattr(builder, "_resolve_dataset_split_row_count", lambda _: 5)
    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        MagicMock(
            return_value=(
                source,
                [
                    {"id": "p0", "title": "Zero", "abstract": "A"},
                    {"id": "p1", "title": "One", "abstract": "A"},
                    {"id": "p0", "title": "Zero", "abstract": "A"},
                    {"id": "p2", "title": "Two", "abstract": "A"},
                    {"id": "p3", "title": "Three", "abstract": "A"},
                ],
            )
        ),
    )
    cached_ids: set[str] = set()

    def _cache_batch(batch: list[dict[str, Any]]) -> int:
        """Model cache writes that coalesce duplicate canonical IDs."""
        cached_ids.update(str(record["paper_id"]) for record in batch)
        return len(batch)

    monkeypatch.setattr(builder, "_cache_metadata_batch", _cache_batch)
    monkeypatch.setattr(builder, "_cached_payload_row_count", lambda: len(cached_ids))

    builder._ensure_cache_hydrated(use_streaming=False)

    assert cached_ids == {"p0", "p1", "p2", "p3"}
    assert [
        call.kwargs["complete"]
        for call in builder.embedding_cache.mark_hydrated.call_args_list
    ] == [False, True]
    builder.embedding_cache.set_hydration_rowcount_reconciliation.assert_called_once_with(
        upstream_rows=5,
        cached_rows=4,
    )


def test_full_corpus_hydration_finalization_preserves_rowcount_transitions() -> None:
    """Finalization should memoize shortfalls and clear reconciliation on recovery."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        corpus_size=None,
        use_streaming=False,
        client=MagicMock(),
    )
    builder.embedding_cache.mark_hydrated = MagicMock()
    builder.embedding_cache.set_hydration_rowcount_reconciliation = MagicMock()
    builder.embedding_cache.clear_hydration_rowcount_reconciliation = MagicMock()

    assert not builder._finalize_full_corpus_hydration_rows(
        source=source,
        updated_rows=100,
        upstream_rows=110,
        mark_complete=True,
    )
    builder.embedding_cache.mark_hydrated.assert_called_once_with(
        dataset_source=source,
        dataset_split=builder.dataset_split,
        corpus_size=builder.corpus_size,
        complete=True,
    )
    builder.embedding_cache.set_hydration_rowcount_reconciliation.assert_called_once_with(
        upstream_rows=110,
        cached_rows=100,
    )
    builder.embedding_cache.clear_hydration_rowcount_reconciliation.assert_not_called()

    builder.embedding_cache.mark_hydrated.reset_mock()
    builder.embedding_cache.set_hydration_rowcount_reconciliation.reset_mock()
    assert builder._finalize_full_corpus_hydration_rows(
        source=source,
        updated_rows=110,
        upstream_rows=110,
        mark_complete=True,
    )
    builder.embedding_cache.mark_hydrated.assert_called_once_with(
        dataset_source=source,
        dataset_split=builder.dataset_split,
        corpus_size=builder.corpus_size,
        complete=True,
    )
    builder.embedding_cache.set_hydration_rowcount_reconciliation.assert_not_called()
    builder.embedding_cache.clear_hydration_rowcount_reconciliation.assert_called_once()


def test_exact_hydration_slice_fails_closed_on_source_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact-source hydration should not encode data returned by another source."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        corpus_size=None,
        use_streaming=False,
        client=MagicMock(),
    )
    load_mock = MagicMock(return_value=("CShorten/ML-ArXiv-Papers", []))
    hydrate_mock = MagicMock()
    monkeypatch.setattr(builder, "_load_dataset_for_hydration", load_mock)
    monkeypatch.setattr(builder, "_hydrate_dataset_records", hydrate_mock)

    with pytest.raises(RuntimeError, match="Incremental refresh resolved unexpected"):
        builder._hydrate_exact_hydration_source_slice(
            use_streaming=False,
            source=source,
            row_limit=10,
            row_offset=100,
            progress_total=10,
            progress_label=f"Refreshing {source}",
            operation="Incremental refresh",
        )

    load_mock.assert_called_once_with(
        use_streaming=False,
        preferred_dataset_source=source,
        row_limit=10,
        row_offset=100,
        allow_source_fallback=False,
    )
    hydrate_mock.assert_not_called()


def test_full_corpus_hydrated_cache_skips_incremental_refresh_without_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydrated full-corpus cache should skip refresh when upstream rows do not grow."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        corpus_size=None,
        use_streaming=False,
        client=MagicMock(),
    )
    _pin_model_fingerprint(monkeypatch, builder)
    builder.embedding_cache.is_hydrated = MagicMock(return_value=True)
    builder.embedding_cache.get_hydrated_dataset_source = MagicMock(return_value=source)
    builder.embedding_cache.payload_stats = MagicMock(
        return_value=CacheNamespacePayloadStats(
            file_count=2,
            size_bytes=1024,
            sqlite_rows=100,
            embedding_rows=100,
            hydration_complete=True,
            hydration_split="train",
            hydration_corpus_size="all",
            hydration_dataset_source=source,
        )
    )
    monkeypatch.setattr(builder, "_resolve_dataset_split_row_count", lambda _: 100)
    monkeypatch.setattr(builder, "_load_dataset_for_hydration", MagicMock())

    builder._ensure_cache_hydrated(use_streaming=False)

    builder._load_dataset_for_hydration.assert_not_called()


def test_full_corpus_hydrated_cache_revalidates_when_upstream_rows_shrink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydrated full-corpus cache should revalidate when upstream row count shrinks."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        corpus_size=None,
        use_streaming=False,
        client=MagicMock(),
    )
    monkeypatch.setattr(builder, "_ensure_cache_model_fingerprint", lambda: None)
    builder.embedding_cache.is_hydrated = MagicMock(side_effect=[True, False, False])
    builder.embedding_cache.get_hydrated_dataset_source = MagicMock(return_value=source)
    builder.embedding_cache.payload_stats = MagicMock(
        return_value=CacheNamespacePayloadStats(
            file_count=2,
            size_bytes=1024,
            sqlite_rows=120,
            embedding_rows=120,
            hydration_complete=True,
            hydration_split="train",
            hydration_corpus_size="all",
            hydration_dataset_source=source,
        )
    )
    builder.embedding_cache.mark_hydrated = MagicMock()
    builder.embedding_cache.clear_hydration_rowcount_reconciliation = MagicMock()
    monkeypatch.setattr(builder, "_resolve_dataset_split_row_count", lambda _: 100)
    load_mock = MagicMock(
        return_value=(
            source,
            [{"id": "replacement-1", "title": "Replacement", "abstract": "A"}],
        )
    )
    monkeypatch.setattr(builder, "_load_dataset_for_hydration", load_mock)
    clear_cache_mock = MagicMock()
    monkeypatch.setattr(builder, "_clear_embedding_cache", clear_cache_mock)
    monkeypatch.setattr(builder, "_hydrate_dataset_records", MagicMock(return_value=1))

    builder._ensure_cache_hydrated(use_streaming=False)

    load_mock.assert_called_once_with(
        use_streaming=False,
        preferred_dataset_source=source,
    )
    clear_cache_mock.assert_called_once()
    complete_flags = [
        call.kwargs["complete"]
        for call in builder.embedding_cache.mark_hydrated.call_args_list
    ]
    assert complete_flags == [False, False, True]


def test_full_corpus_incremental_refresh_reconciles_missing_ids_when_tail_scan_underfills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incremental refresh should reconcile missing IDs when tail slice is insufficient."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        corpus_size=None,
        use_streaming=False,
        client=MagicMock(),
    )
    _pin_model_fingerprint(monkeypatch, builder)
    builder.embedding_cache.is_hydrated = MagicMock(return_value=True)
    builder.embedding_cache.get_hydrated_dataset_source = MagicMock(return_value=source)
    builder.embedding_cache.get_cached_paper_ids = MagicMock(
        return_value={f"old-{idx}" for idx in range(100)}
    )
    builder.embedding_cache.payload_stats = MagicMock(
        side_effect=[
            CacheNamespacePayloadStats(
                file_count=2,
                size_bytes=1024,
                sqlite_rows=100,
                embedding_rows=100,
                hydration_complete=True,
                hydration_split="train",
                hydration_corpus_size="all",
                hydration_dataset_source=source,
            ),
            CacheNamespacePayloadStats(
                file_count=2,
                size_bytes=1024,
                sqlite_rows=100,
                embedding_rows=100,
                hydration_complete=True,
                hydration_split="train",
                hydration_corpus_size="all",
                hydration_dataset_source=source,
            ),
            CacheNamespacePayloadStats(
                file_count=2,
                size_bytes=1024,
                sqlite_rows=110,
                embedding_rows=110,
                hydration_complete=True,
                hydration_split="train",
                hydration_corpus_size="all",
                hydration_dataset_source=source,
            ),
        ]
    )
    builder.embedding_cache.mark_hydrated = MagicMock()
    monkeypatch.setattr(builder, "_resolve_dataset_split_row_count", lambda _: 110)
    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        MagicMock(
            side_effect=[
                (
                    source,
                    [
                        {"id": f"tail-{idx}", "title": f"Tail {idx}", "abstract": "A"}
                        for idx in range(10)
                    ],
                ),
                (
                    source,
                    [
                        {
                            "id": f"full-{idx}",
                            "title": f"Full {idx}",
                            "abstract": "B",
                        }
                        for idx in range(110)
                    ],
                ),
            ]
        ),
    )
    hydrate_mock = MagicMock(side_effect=[10, 10])
    monkeypatch.setattr(builder, "_hydrate_dataset_records", hydrate_mock)

    builder._ensure_cache_hydrated(use_streaming=False)

    assert builder._load_dataset_for_hydration.call_count == 2
    first_call = builder._load_dataset_for_hydration.call_args_list[0]
    second_call = builder._load_dataset_for_hydration.call_args_list[1]
    assert first_call.kwargs == {
        "use_streaming": False,
        "preferred_dataset_source": source,
        "row_limit": 10,
        "row_offset": 100,
        "allow_source_fallback": False,
    }
    assert second_call.kwargs == {
        "use_streaming": False,
        "preferred_dataset_source": source,
        "row_limit": 10,
        "row_offset": 0,
        "allow_source_fallback": False,
    }
    assert hydrate_mock.call_count == 2
    assert "existing_paper_ids" in hydrate_mock.call_args_list[1].kwargs
    assert hydrate_mock.call_args_list[1].kwargs["max_new_records"] == 10
    builder.embedding_cache.mark_hydrated.assert_called_once_with(
        dataset_source=source,
        dataset_split=builder.dataset_split,
        corpus_size=builder.corpus_size,
        complete=True,
    )


def test_full_corpus_rowcount_delta_memoization_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rowcount reconciliation should memoize duplicate deltas and skip repeat scans."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        corpus_size=None,
        use_streaming=False,
        client=MagicMock(),
    )
    _pin_model_fingerprint(monkeypatch, builder)
    builder.embedding_cache.is_hydrated = MagicMock(return_value=True)
    builder.embedding_cache.get_hydrated_dataset_source = MagicMock(return_value=source)
    builder.embedding_cache.get_cached_paper_ids = MagicMock(
        return_value={f"old-{idx}" for idx in range(100)}
    )
    builder.embedding_cache.get_hydration_rowcount_reconciliation = MagicMock(
        return_value=None
    )
    builder.embedding_cache.set_hydration_rowcount_reconciliation = MagicMock()
    builder.embedding_cache.clear_hydration_rowcount_reconciliation = MagicMock()
    builder.embedding_cache.payload_stats = MagicMock(
        side_effect=[
            CacheNamespacePayloadStats(
                file_count=2,
                size_bytes=1024,
                sqlite_rows=100,
                embedding_rows=100,
                hydration_complete=True,
                hydration_split="train",
                hydration_corpus_size="all",
                hydration_dataset_source=source,
            ),
            CacheNamespacePayloadStats(
                file_count=2,
                size_bytes=1024,
                sqlite_rows=100,
                embedding_rows=100,
                hydration_complete=True,
                hydration_split="train",
                hydration_corpus_size="all",
                hydration_dataset_source=source,
            ),
            CacheNamespacePayloadStats(
                file_count=2,
                size_bytes=1024,
                sqlite_rows=100,
                embedding_rows=100,
                hydration_complete=True,
                hydration_split="train",
                hydration_corpus_size="all",
                hydration_dataset_source=source,
            ),
            CacheNamespacePayloadStats(
                file_count=2,
                size_bytes=1024,
                sqlite_rows=100,
                embedding_rows=100,
                hydration_complete=True,
                hydration_split="train",
                hydration_corpus_size="all",
                hydration_dataset_source=source,
            ),
        ]
    )
    builder.embedding_cache.mark_hydrated = MagicMock()
    monkeypatch.setattr(builder, "_resolve_dataset_split_row_count", lambda _: 110)
    initial_load_mock = MagicMock(
        side_effect=[
            (
                source,
                [
                    {"id": f"tail-{idx}", "title": f"Tail {idx}", "abstract": "A"}
                    for idx in range(10)
                ],
            ),
            (
                source,
                [
                    {"id": f"head-{idx}", "title": f"Head {idx}", "abstract": "B"}
                    for idx in range(10)
                ],
            ),
            (
                source,
                [
                    {"id": f"full-{idx}", "title": f"Full {idx}", "abstract": "C"}
                    for idx in range(110)
                ],
            ),
        ]
    )
    monkeypatch.setattr(builder, "_load_dataset_for_hydration", initial_load_mock)
    hydrate_mock = MagicMock(side_effect=[10, 0, 0])
    monkeypatch.setattr(builder, "_hydrate_dataset_records", hydrate_mock)

    builder._ensure_cache_hydrated(use_streaming=False)
    assert initial_load_mock.call_count == 3
    builder.embedding_cache.set_hydration_rowcount_reconciliation.assert_called_once_with(
        upstream_rows=110,
        cached_rows=100,
    )
    assert (
        builder.embedding_cache.clear_hydration_rowcount_reconciliation.call_count == 0
    )

    builder.embedding_cache.get_hydration_rowcount_reconciliation.return_value = (
        110,
        100,
    )
    builder.embedding_cache.payload_stats = MagicMock(
        return_value=CacheNamespacePayloadStats(
            file_count=2,
            size_bytes=1024,
            sqlite_rows=100,
            embedding_rows=100,
            hydration_complete=True,
            hydration_split="train",
            hydration_corpus_size="all",
            hydration_dataset_source=source,
        )
    )
    repeat_load_mock = MagicMock()
    monkeypatch.setattr(builder, "_load_dataset_for_hydration", repeat_load_mock)

    builder._ensure_cache_hydrated(use_streaming=False)
    repeat_load_mock.assert_not_called()


def test_hydration_reset_restores_model_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Hydration reset should restore resolved model fingerprint metadata."""
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path / "cache-root"))

    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        use_streaming=False,
        corpus_size=1,
        client=MagicMock(),
    )
    _pin_model_fingerprint(monkeypatch, builder, fingerprint="fp-before-clear")
    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        lambda use_streaming, preferred_dataset_source=None: (
            "mini-dataset",
            [{"id": "p1", "title": "Paper 1", "abstract": "A"}],
        ),
    )
    monkeypatch.setattr(builder, "_cache_metadata_batch", lambda batch: len(batch))

    builder._ensure_cache_hydrated(use_streaming=False)
    assert builder.embedding_cache.get_model_fingerprint() == "fp-before-clear"


def test_int8_hydration_calibration_uses_representative_prepass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Int8 hydration should calibrate from a separate representative sample pass."""
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path / "cache-root"))

    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="int8",
        calibration_sample_size=2,
        use_streaming=False,
        corpus_size=5,
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    )
    _pin_model_fingerprint(monkeypatch, builder)

    source = "mini-dataset"
    records = [
        {"id": f"p{idx}", "title": f"Title {idx}", "abstract": f"Abstract {idx}"}
        for idx in range(5)
    ]
    load_mock = MagicMock(
        side_effect=[
            (source, list(records)),
            (source, list(records)),
        ]
    )
    monkeypatch.setattr(builder, "_load_dataset_for_hydration", load_mock)

    captured_texts: list[list[str]] = []

    def _fake_encode_texts(
        texts: list[str],
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del batch_size, show_progress_bar
        captured_texts.append(list(texts))
        return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(builder, "_encode_texts", _fake_encode_texts)

    def _cache_batch(batch: list[dict[str, Any]]) -> int:
        assert builder.embedding_cache.has_calibration_ranges()
        return len(batch)

    monkeypatch.setattr(builder, "_cache_metadata_batch", _cache_batch)

    builder._ensure_cache_hydrated(use_streaming=False)

    expected_calibration_texts = [
        builder.model_profile.format_document(
            {"title": "Title 4", "abstract": "Abstract 4"}
        ),
        builder.model_profile.format_document(
            {"title": "Title 2", "abstract": "Abstract 2"}
        ),
    ]
    assert load_mock.call_count == 2
    assert captured_texts == [expected_calibration_texts]
    assert captured_texts[0] != [
        builder.model_profile.format_document(
            {"title": "Title 0", "abstract": "Abstract 0"}
        ),
        builder.model_profile.format_document(
            {"title": "Title 1", "abstract": "Abstract 1"}
        ),
    ]
    assert builder.embedding_cache.has_calibration_ranges() is True


def test_int8_calibration_uses_percentile_clipping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Calibration ranges should not be dominated by a single extreme outlier."""
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path / "cache-root"))

    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="int8",
        calibration_sample_size=101,
        use_streaming=False,
        corpus_size=101,
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    )
    sample_records = [
        {"paper_id": f"p{idx}", "title": f"Title {idx}", "abstract": f"Abstract {idx}"}
        for idx in range(101)
    ]

    def _fake_encode_texts(
        texts: list[str],
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del texts, batch_size, show_progress_bar
        base = np.zeros((100, 2), dtype=np.float32)
        outlier = np.full((1, 2), 100.0, dtype=np.float32)
        return np.vstack((base, outlier))

    monkeypatch.setattr(builder, "_encode_texts", _fake_encode_texts)
    builder._initialize_calibration_ranges(sample_records)

    with h5py.File(builder.embedding_cache.h5_path, "r") as h5:
        ranges = np.asarray(h5["calibration_ranges"], dtype=np.float32)

    np.testing.assert_allclose(ranges[0], np.zeros(2, dtype=np.float32))
    assert np.all(ranges[1] < 100.0)
    assert np.all(ranges[1] > 0.0)


def test_hydration_flush_size_controls_cache_write_bursting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Hydration should flush metadata batches using configured flush threshold."""
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path / "cache-root"))
    monkeypatch.setattr("citemesh.strategies.embedding.HYDRATION_FLUSH_SIZE", 3)

    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        use_streaming=False,
        corpus_size=7,
        client=MagicMock(),
    )
    _pin_model_fingerprint(monkeypatch, builder, fingerprint="fp-flush-threshold")
    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        lambda use_streaming, preferred_dataset_source=None: (
            "mini-dataset",
            [
                {"id": f"p{i}", "title": f"Paper {i}", "abstract": f"A{i}"}
                for i in range(7)
            ],
        ),
    )

    flushed_batch_sizes: list[int] = []

    def _capture_flush(batch: list[dict[str, Any]]) -> int:
        flushed_batch_sizes.append(len(batch))
        return len(batch)

    monkeypatch.setattr(builder, "_cache_metadata_batch", _capture_flush)

    builder._ensure_cache_hydrated(use_streaming=False)
    assert flushed_batch_sizes == [3, 3, 1]


def test_cache_metadata_batch_caps_model_encode_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache writes should keep encode batch size bounded for stable runtime."""

    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        use_streaming=False,
        corpus_size=1,
        client=MagicMock(),
    )
    monkeypatch.setattr(builder, "_get_model_for_encoding", lambda: object())

    captured: dict[str, int] = {}

    def _capture_upsert_embeddings(
        papers: dict[str, dict[str, Any]],
        model: object,
        batch_size: int = 32,
        show_progress: bool = True,
        text_builder: Any = None,
    ) -> object:
        del model, show_progress, text_builder
        captured["batch_size"] = int(batch_size)
        captured["paper_count"] = int(len(papers))
        return object()

    monkeypatch.setattr(
        builder.embedding_cache, "upsert_embeddings", _capture_upsert_embeddings
    )

    batch = [
        {
            "paper_id": f"paper-{idx}",
            "title": f"Title {idx}",
            "abstract": f"Abstract {idx}",
            "year": 2025,
            "authors": ["A"],
            "categories": ["cs.AI"],
        }
        for idx in range(ENCODE_BATCH_SIZE + 17)
    ]

    routed = builder._cache_metadata_batch(batch)
    assert routed == len(batch)
    assert captured["paper_count"] == len(batch)
    assert captured["batch_size"] == ENCODE_BATCH_SIZE


def test_empty_hydration_run_remains_incomplete_and_returns_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Empty hydration should not mark cache complete and should return no candidates."""
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path / "cache-root"))

    builder = EmbeddingGraphBuilder(
        max_papers=2,
        use_streaming=False,
        client=MagicMock(),
    )
    _pin_model_fingerprint(monkeypatch, builder)
    mark_hydrated_spy = MagicMock(wraps=builder.embedding_cache.mark_hydrated)
    monkeypatch.setattr(builder.embedding_cache, "mark_hydrated", mark_hydrated_spy)
    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        lambda use_streaming, preferred_dataset_source=None, **_kwargs: (
            "empty-snapshot",
            [],
        ),
    )
    monkeypatch.setattr(
        builder,
        "_get_model_for_encoding",
        lambda: (_ for _ in ()).throw(
            AssertionError("encoding should not run for empty hydration")
        ),
    )

    candidates = builder._select_candidates(
        np.asarray([1.0, 0.0], dtype=np.float32),
        use_streaming=False,
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

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        EmbeddingGraphBuilder(top_k=0, client=MagicMock())

    builder = EmbeddingGraphBuilder(max_papers=2, top_k=2, client=MagicMock())
    _pin_model_fingerprint(monkeypatch, builder)
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

    candidates = builder._select_candidates(
        np.asarray([1.0, 0.0], dtype=np.float32),
        use_streaming=False,
    )
    assert [paper_id for paper_id, _, _ in candidates] == ["a", "b"]


def test_embedding_runtime_metadata_tracks_prefilter_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding runtime metadata should reflect effective query-time prefilter use."""

    builder = EmbeddingGraphBuilder(max_papers=2, top_k=2, client=MagicMock())
    _pin_model_fingerprint(monkeypatch, builder)
    builder.embedding_cache.is_hydrated = MagicMock(return_value=True)
    builder.embedding_cache.last_search_used_binary_prefilter = False
    builder.embedding_cache.search = MagicMock(return_value=[])

    candidates = builder._select_candidates(
        np.asarray([1.0, 0.0], dtype=np.float32),
        use_streaming=False,
    )
    assert candidates == []
    assert builder._embedding_runtime_metadata() == {
        "binary_prefilter_used": False,
        "device": builder.device,
        "requested_device": "auto",
        "compute_dtype": builder._source_dtype_hint,
        "autocast": False,
    }


def test_embedding_candidate_search_logs_comparison_counts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Candidate search should log compared/rescored embedding counts."""

    builder = EmbeddingGraphBuilder(max_papers=2, top_k=2, client=MagicMock())
    _pin_model_fingerprint(monkeypatch, builder)
    builder.embedding_cache.is_hydrated = MagicMock(return_value=True)
    builder.embedding_cache.last_search_used_binary_prefilter = True
    builder.embedding_cache.last_search_total_embeddings = 50000
    builder.embedding_cache.last_search_rescored_embeddings = 640
    builder.embedding_cache.search = MagicMock(
        return_value=[
            CacheSearchResult(
                paper_id="a",
                score=0.95,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                metadata={"title": "A", "abstract": "A", "authors": []},
            )
        ]
    )

    caplog.clear()
    with caplog.at_level(logging.INFO):
        candidates = builder._select_candidates(
            np.asarray([1.0, 0.0], dtype=np.float32),
            use_streaming=False,
        )

    assert [paper_id for paper_id, _, _ in candidates] == ["a"]
    log_messages = [record.getMessage() for record in caplog.records]
    assert any(
        "compared against 50,000 embeddings (rescored=640, prefilter=on)" in message
        for message in log_messages
    )


def test_embedding_citation_enrichment_logs_target_count(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Citation enrichment should batch-fetch counts and update paper metadata."""
    client = MagicMock()
    builder = EmbeddingGraphBuilder(max_papers=5, top_k=2, client=client)
    _pin_model_fingerprint(monkeypatch, builder)

    target = Paper(
        paper_id="paper-1",
        title="Paper One",
        year=2024,
        abstract="paper one abstract",
        is_seed=False,
    )
    enriched = Paper(
        paper_id="paper-1",
        title="Paper One",
        year=2024,
        abstract="paper one abstract",
        citation_count=77,
        is_seed=False,
    )
    client.get_papers = MagicMock(return_value={"paper-1": enriched})
    client.get_paper = MagicMock()
    papers = {
        "seed": Paper(
            paper_id="seed",
            title="Seed",
            year=2024,
            abstract="seed abstract",
            is_seed=True,
        ),
        "paper-1": target,
    }

    caplog.clear()
    with caplog.at_level(logging.INFO):
        builder._update_citation_counts(papers)

    assert papers["paper-1"].citation_count == 77
    client.get_papers.assert_called_once_with(["paper-1"])
    client.get_paper.assert_not_called()
    log_messages = [record.getMessage() for record in caplog.records]
    assert any(
        "Fetching citation counts from Semantic Scholar for up to 1 papers..."
        in message
        for message in log_messages
    )


def test_embedding_build_graph_persists_runtime_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding build_graph should propagate runtime metadata to graph attrs."""

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    builder._last_search_used_binary_prefilter = True
    monkeypatch.setattr(
        builder,
        "collect_papers",
        lambda seed_id, **kwargs: {
            "seed": Paper(paper_id="seed", title="Seed", year=2024, is_seed=True)
        },
    )
    monkeypatch.setattr(builder, "compute_similarity", lambda _p1, _p2: 0.0)

    graph, seed_id = builder.build_graph("seed")
    assert seed_id == "seed"
    assert graph.graph["embedding_runtime"] == {
        "binary_prefilter_used": True,
        "device": builder.device,
        "requested_device": "auto",
        "compute_dtype": builder._source_dtype_hint,
        "autocast": False,
    }


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "requested", "expected"),
    [
        (True, True, "auto", "cuda"),
        (False, True, "auto", "mps"),
        (False, False, "auto", "cpu"),
        (True, True, None, "cuda"),
        (True, False, "cuda", "cuda"),
        (False, True, "mps", "mps"),
        (True, True, "cpu", "cpu"),
        (False, False, "cpu", "cpu"),
    ],
)
def test_embedding_device_resolution_matrix(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    requested: str | None,
    expected: str,
) -> None:
    """Device resolution should prefer cuda, then mps, then cpu."""
    _install_fake_torch(
        monkeypatch,
        cuda_available=cuda_available,
        bf16_supported=True,
        mps_available=mps_available,
    )
    assert resolve_embedding_device(requested) == expected


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "requested"),
    [
        (False, True, "cuda"),
        (True, False, "mps"),
        (False, False, "tpu"),
    ],
)
def test_embedding_device_resolution_rejects_unavailable_or_unknown(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    requested: str,
) -> None:
    """Explicit unavailable accelerators and unknown tokens should raise."""
    _install_fake_torch(
        monkeypatch,
        cuda_available=cuda_available,
        bf16_supported=True,
        mps_available=mps_available,
    )
    with pytest.raises(ValueError):
        resolve_embedding_device(requested)


def test_embedding_device_forwarded_to_sentence_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved device should always be passed to SentenceTransformer."""
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
        mps_available=True,
        torch_version="2.13.0",
    )
    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    builder._load_model()

    assert builder.requested_device == "auto"
    assert builder.device == "mps"
    assert init_log["kwargs"]["device"] == "mps"


def test_embedding_mps_precision_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EmbeddingGemma on MPS should use bf16 autocast with auto-loaded weights."""
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    bf16_token, autocast_log, _fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
        mps_available=True,
        torch_version="2.13.0",
    )
    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    builder._load_model()

    assert builder.device == "mps"
    assert builder._source_dtype_hint == "bfloat16"
    assert (
        init_log["kwargs"]["model_kwargs"].get(
            "dtype", init_log["kwargs"]["model_kwargs"].get("torch_dtype")
        )
        == "auto"
    )
    assert init_log["kwargs"]["model_kwargs"]["attn_implementation"] == "sdpa"
    assert builder._autocast_enabled is True
    assert builder._get_model_for_encoding() is not builder.model
    builder._encode_texts(["seed"])
    assert ("call", "mps", bf16_token) in autocast_log
    assert ("enter",) in autocast_log and ("exit",) in autocast_log


def test_embedding_mps_bf16_soft_gate_pre_2_13(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MPS with pre-2.13 torch should decline bf16 with a warning."""
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
        mps_available=True,
        torch_version="2.11.0",
    )
    with caplog.at_level(logging.WARNING):
        builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    builder._load_model()

    assert builder.device == "mps"
    assert builder._source_dtype_hint == "float32"
    assert (
        init_log["kwargs"]["model_kwargs"].get(
            "dtype", init_log["kwargs"]["model_kwargs"].get("torch_dtype")
        )
        == "auto"
    )
    assert any(
        "predates the verified MPS floor" in record.getMessage()
        for record in caplog.records
    )


def test_embedding_mps_default_profile_uses_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profiles without an explicit bf16 policy should stay float32 on MPS."""
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, autocast_log, _fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
        mps_available=True,
        torch_version="2.13.0",
    )
    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        client=MagicMock(),
    )
    builder._load_model()

    assert builder._source_dtype_hint == "float32"
    assert (
        init_log["kwargs"]["model_kwargs"].get(
            "dtype", init_log["kwargs"]["model_kwargs"].get("torch_dtype")
        )
        == "auto"
    )
    assert init_log["kwargs"]["model_kwargs"]["attn_implementation"] == "sdpa"
    assert builder._autocast_enabled is False
    assert autocast_log == []


def test_embedding_mps_never_probes_flash_attn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPS attention policy should not consult flash_attn availability."""
    _install_fake_sentence_transformers(monkeypatch)
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
        mps_available=True,
        torch_version="2.13.0",
    )
    probed: list[str] = []

    def _record_probe(module_name: str) -> bool:
        probed.append(module_name)
        return True

    monkeypatch.setattr(
        "citemesh.strategies.embedding._module_available", _record_probe
    )
    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())

    assert builder._attention_implementation_hint == "sdpa"
    assert "flash_attn" not in probed


def test_embedding_compile_device_gating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch.compile should run on cuda/mps but be declined on cpu."""
    for device_setup, expect_compiled in [
        ({"cuda_available": True, "mps_available": False}, True),
        ({"cuda_available": False, "mps_available": True}, True),
        ({"cuda_available": False, "mps_available": False}, False),
    ]:
        _install_fake_sentence_transformers(monkeypatch)
        _bf16_token, _autocast_log, fake_torch = _install_fake_torch(
            monkeypatch,
            bf16_supported=True,
            torch_version="2.13.0",
            compile_behavior="tagged",
            **device_setup,
        )
        builder = EmbeddingGraphBuilder(
            max_papers=1,
            enable_torch_compile=True,
            client=MagicMock(),
        )
        monkeypatch.setattr(
            builder,
            "_should_defer_compile_for_cache_hydration",
            lambda: False,
        )
        builder._load_model()

        assert builder._inner_model_compiled is expect_compiled
        if not expect_compiled:
            assert builder._compile_status_reason == "compile disabled for device=cpu"
            assert fake_torch._compile_calls == []


def test_embedding_tf32_skipped_for_non_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit non-CUDA device on a CUDA host must not flip global TF32 state."""
    _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, _autocast_log, fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=True,
        torch_version="2.13.0",
    )
    builder = EmbeddingGraphBuilder(max_papers=1, device="cpu", client=MagicMock())
    builder._load_model()

    assert builder.device == "cpu"
    assert builder._tf32_mode == "off"
    assert fake_torch.backends.fp32_precision == "none"


def test_embedding_cache_namespace_stable_across_device_for_same_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bf16 caches must share a namespace across cuda/mps; cpu-fp32 differs."""
    _install_fake_sentence_transformers(monkeypatch)

    _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=True,
        torch_version="2.13.0",
    )
    cuda_builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())

    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
        mps_available=True,
        torch_version="2.13.0",
    )
    mps_builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())

    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
        mps_available=False,
        torch_version="2.13.0",
    )
    cpu_builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())

    assert cuda_builder._source_dtype_hint == "bfloat16"
    assert mps_builder._source_dtype_hint == "bfloat16"
    assert cpu_builder._source_dtype_hint == "float32"
    assert (
        cuda_builder._embedding_cache_namespace()
        == mps_builder._embedding_cache_namespace()
    )
    assert (
        cpu_builder._embedding_cache_namespace()
        != cuda_builder._embedding_cache_namespace()
    )


@pytest.mark.slow
def test_embedding_real_mps_smoke() -> None:
    """Load the real default model on MPS and encode two strings.

    Requires real Metal access: skips on Linux CI and inside sandboxes that
    hide the MPS device. Run escalated on Apple Silicon for a meaningful pass.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend unavailable in this runtime")

    builder = EmbeddingGraphBuilder(max_papers=2, client=MagicMock())
    assert builder.device == "mps"
    builder._load_model()
    vectors = builder._encode_texts(
        ["attention is all you need", "graph neural networks survey"]
    )
    assert vectors.shape[0] == 2
    assert np.isfinite(vectors).all()
