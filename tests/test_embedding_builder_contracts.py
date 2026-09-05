"""Contract tests for embedding builder precision and hydration behavior."""

from __future__ import annotations

import json
import logging
import sys
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable, Iterator
from unittest.mock import MagicMock, call, patch

import h5py
import numpy as np
import pytest

from citemesh.core import Paper
from citemesh.data import (
    DEFAULT_EMBEDDING_MODEL_FALLBACKS,
    DEFAULT_EMBEDDING_MODEL_NAME,
)
from citemesh.data.embedding_cache import (
    EMBEDDING_DATASET_CHUNK_ROWS,
    CacheNamespacePayloadStats,
    CacheSearchResult,
)
from citemesh.services.semantic_scholar import SemanticScholarUnavailableError
from citemesh.strategies import embedding as embedding_module
from citemesh.strategies.embedding import (
    ENCODE_BATCH_SIZE,
    EmbeddingGraphBuilder,
    EmbeddingTask,
    _extract_dataset_paper_metadata,
    _query_seed_id,
    format_paper_for_embedding,
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
    if request.node.name in _REAL_DEP_CHECK_TESTS or request.node.get_closest_marker(
        "slow"
    ):
        return
    disable_embedding_dep_checks(monkeypatch)
    module_available = embedding_module._module_available
    monkeypatch.setattr(
        embedding_module,
        "_module_available",
        lambda name: False if name == "flash_attn" else module_available(name),
    )


@pytest.mark.parametrize(
    ("model_name", "transformers_version", "should_reject"),
    [
        (DEFAULT_EMBEDDING_MODEL_NAME, "5.1.0", True),
        (DEFAULT_EMBEDDING_MODEL_NAME, "5.2.0", False),
        (DEFAULT_EMBEDDING_MODEL_NAME, "5.9.0", False),
        ("org/generic-embedding-model", "4.56.2", False),
    ],
)
def test_model_load_enforces_profile_transformers_floor(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    transformers_version: str,
    should_reject: bool,
) -> None:
    """Model loading should enforce only the active profile's backend floor."""
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
    )
    fake_transformers = types.SimpleNamespace(__version__=transformers_version)
    monkeypatch.setattr(
        embedding_module.importlib,
        "import_module",
        lambda module_name: fake_transformers,
    )
    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name=model_name,
        client=MagicMock(),
    )

    if should_reject:
        with pytest.raises(
            embedding_module.EmbeddingBackendCompatibilityError,
            match=r"requires transformers>=5\.2",
        ):
            builder._load_model()
        assert "attempts" not in init_log
        return

    builder._load_model()
    assert init_log["attempts"] == [model_name]


def test_transformers_version_falls_back_to_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nonstandard module versions should use valid installed package metadata."""
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
    )
    fake_transformers = types.SimpleNamespace(__version__="development-build")
    monkeypatch.setattr(
        embedding_module.importlib,
        "import_module",
        lambda _module_name: fake_transformers,
    )
    monkeypatch.setattr(
        embedding_module.importlib_metadata,
        "version",
        lambda _distribution: "5.14.1",
    )

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    builder._load_model()

    assert init_log["attempts"] == [DEFAULT_EMBEDDING_MODEL_NAME]
    assert "dtype" in init_log["kwargs"]["model_kwargs"]


def test_transformers_unknown_version_fails_with_verification_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unverifiable backend version should fail closed without claiming 0.0."""
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
    )
    fake_transformers = types.SimpleNamespace(__version__="development-build")
    monkeypatch.setattr(
        embedding_module.importlib,
        "import_module",
        lambda _module_name: fake_transformers,
    )
    monkeypatch.setattr(
        embedding_module.importlib_metadata,
        "version",
        lambda _distribution: "unknown",
    )

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    with pytest.raises(
        embedding_module.EmbeddingBackendCompatibilityError,
        match="Could not determine the installed Transformers version",
    ):
        builder._load_model()

    assert "attempts" not in init_log


def _install_fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_model_names: set[str] | None = None,
    active_bidirectional_attention: bool | None = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Install fake ``sentence_transformers`` module for runtime tests.

    :param pytest.MonkeyPatch monkeypatch: Pytest patching fixture.
    :param Optional[set[str]] fail_model_names: Model names whose construction fails.
    :param Optional[bool] active_bidirectional_attention: Live transformer flag, or
        ``None`` to omit it.
    :return tuple[dict[str, Any], list[dict[str, Any]]]: Constructor and encode logs.
    """
    init_log: dict[str, Any] = {}
    encode_log: list[dict[str, Any]] = []
    blocked_models = set(fail_model_names or ())

    class _FakeInnerBlock:
        def __init__(self) -> None:
            config = types.SimpleNamespace()
            if active_bidirectional_attention is not None:
                config.use_bidirectional_attention = active_bidirectional_attention
            self.auto_model = types.SimpleNamespace(config=config)

    class _FakeSentenceTransformer:
        def __init__(self, model_name_or_path: str, **kwargs: Any):
            init_log.setdefault("attempts", []).append(model_name_or_path)
            init_log.setdefault("attempt_kwargs", []).append(dict(kwargs))
            if model_name_or_path in blocked_models:
                raise RuntimeError(f"failed loading {model_name_or_path}")
            init_log["model_name"] = model_name_or_path
            init_log["kwargs"] = kwargs
            self._blocks = [_FakeInnerBlock()]
            init_log["auto_model_before_compile"] = self._blocks[0].auto_model

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            encode_log.append(kwargs)
            return np.ones((len(texts), 2), dtype=np.float32)

        def parameters(self) -> Iterable[object]:
            """Yield one float32 parameter token for live dtype inspection.

            :return Iterable[object]: One fake float32 parameter.
            """
            return iter((types.SimpleNamespace(dtype="torch.float32"),))

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
    builder._resolved_model_fingerprint = fingerprint
    builder._bind_embedding_cache_to_active_model()


def _write_local_sentence_transformer_profile(
    root: Path,
    *,
    embeddinggemma: bool,
    nested_transformer: bool = False,
    include_task_prompts: bool = True,
    include_bidirectional_flag: bool = True,
) -> None:
    """Write minimal local SentenceTransformers metadata for profile tests.

    :param Path root: Checkpoint root to populate.
    :param bool embeddinggemma: Whether metadata should declare EmbeddingGemma.
    :param bool nested_transformer: Whether transformer config lives in a module dir.
    :param bool include_task_prompts: Whether to write task-prompt metadata.
    :param bool include_bidirectional_flag: Whether to write EmbeddingGemma's
        model-specific bidirectional-attention flag.
    :return None: Writes fixture metadata below ``root``.
    """
    root.mkdir(parents=True)
    transformer_root = root / "0_Transformer" if nested_transformer else root
    transformer_root.mkdir(exist_ok=True)
    transformer_config = (
        {
            "model_type": "gemma3_text",
            "architectures": ["Gemma3TextModel"],
            **(
                {"use_bidirectional_attention": True}
                if include_bidirectional_flag
                else {}
            ),
        }
        if embeddinggemma
        else {"model_type": "bert", "architectures": ["BertModel"]}
    )
    (transformer_root / "config.json").write_text(
        json.dumps(transformer_config),
        encoding="utf-8",
    )
    (root / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "0_Transformer" if nested_transformer else "",
                    "type": "sentence_transformers.models.Transformer",
                }
            ]
        ),
        encoding="utf-8",
    )
    prompts = (
        {
            "Retrieval-query": "task: search result | query: ",
            "Retrieval-document": "title: none | text: ",
            "STS": "task: sentence similarity | query: ",
        }
        if embeddinggemma
        else {"query": "", "document": ""}
    )
    if include_task_prompts:
        (root / "config_sentence_transformers.json").write_text(
            json.dumps({"prompts": prompts}),
            encoding="utf-8",
        )


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    bf16_supported: bool,
    *,
    mps_available: bool = False,
    mps_built: bool | None = None,
    bf16_native_supported: bool | None = None,
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
    matmul_precision_state = {"value": "highest"}

    def _set_float32_matmul_precision(precision: str) -> None:
        matmul_precision_calls.append(precision)
        matmul_precision_state["value"] = precision

    def _get_float32_matmul_precision() -> str:
        """Return the fake process-wide matmul precision token.

        :return str: Current fake precision token.
        """
        return matmul_precision_state["value"]

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
    resolved_mps_built = mps_available if mps_built is None else mps_built
    fake_backends.mps = types.SimpleNamespace(
        is_available=lambda: mps_available,
        is_built=lambda: resolved_mps_built,
    )

    bf16_support_calls: list[bool] = []

    def _is_bf16_supported(*, including_emulation: bool = True) -> bool:
        """Return fake native/emulated CUDA bfloat16 support.

        :param bool including_emulation: Whether emulated support is acceptable.
        :return bool: Configured support verdict.
        """
        bf16_support_calls.append(including_emulation)
        if including_emulation or bf16_native_supported is None:
            return bf16_supported
        return bf16_native_supported

    cuda_module = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        is_bf16_supported=_is_bf16_supported,
    )
    if capability is not None:
        cuda_module.get_device_capability = lambda _index=0: capability

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = torch_version
    fake_torch.bfloat16 = bf16_token
    fake_torch.autocast = _autocast
    fake_torch.compile = _compile
    fake_torch.get_float32_matmul_precision = _get_float32_matmul_precision
    fake_torch.set_float32_matmul_precision = _set_float32_matmul_precision
    fake_torch._matmul_precision_calls = matmul_precision_calls
    fake_torch._bf16_support_calls = bf16_support_calls
    fake_torch._compile_calls = compile_calls
    inductor_config = types.SimpleNamespace(
        freezing=False, freezing_discard_parameters=False
    )

    @contextmanager
    def _patch_inductor_config(**updates: bool) -> Iterator[None]:
        """Apply fake scoped compiler settings and restore them on every exit.

        :param bool updates: Temporary compiler configuration values.
        :return Iterator[None]: Scope with the requested settings active.
        """
        previous = {key: getattr(inductor_config, key) for key in updates}
        try:
            for key, value in updates.items():
                setattr(inductor_config, key, value)
            yield
        finally:
            for key, value in previous.items():
                setattr(inductor_config, key, value)

    inductor_config.patch = _patch_inductor_config
    fake_torch._inductor = types.SimpleNamespace(config=inductor_config)
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
        assert init_log["kwargs"]["truncate_dim"] == 512
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
        ("org/generic-embedding-model", "tagged", (8, 0), "2.10.0", False),
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
            assert _fake_torch._compile_calls[-1]["kwargs"] == {"dynamic": True}  # type: ignore[attr-defined]
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
        ((8, 0), True, True, "2.9.0", "tf32-matmul-high"),
        ((8, 0), True, False, "2.9.0", "tf32"),
        ((8, 0), False, True, "2.10.0", "tf32-matmul-high"),
        ((7, 5), True, True, "2.10.0", "off"),
    ]
    for (
        capability,
        include_tf32_global_api,
        enable_torch_compile,
        torch_version,
        expected_mode,
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

        assert fake_torch.backends.cuda.matmul.fp32_precision == "none"
        assert fake_torch.backends.cudnn.conv.fp32_precision == "none"
        if include_tf32_global_api:
            assert fake_torch.backends.fp32_precision == "none"
        else:
            assert not hasattr(fake_torch.backends, "fp32_precision")

        assert builder._tf32_mode == expected_mode
        with builder._tf32_context():
            if expected_mode == "tf32":
                assert fake_torch.backends.cuda.matmul.fp32_precision == "tf32"
                assert fake_torch.backends.cudnn.conv.fp32_precision == "tf32"
            else:
                assert fake_torch.backends.cuda.matmul.fp32_precision == "none"
                assert fake_torch.backends.cudnn.conv.fp32_precision == "none"
            expected_matmul = (
                "high" if expected_mode == "tf32-matmul-high" else "highest"
            )
            assert fake_torch.get_float32_matmul_precision() == expected_matmul

        assert fake_torch.backends.cuda.matmul.fp32_precision == "none"
        assert fake_torch.backends.cudnn.conv.fp32_precision == "none"
        assert fake_torch.get_float32_matmul_precision() == "highest"
        expected_calls = (
            ["high", "highest"] if expected_mode == "tf32-matmul-high" else []
        )
        assert fake_torch._matmul_precision_calls == expected_calls

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
        "Adds recommended retrieval-query, retrieval-document, and symmetric" in message
        for message in info_messages
    )
    assert any(
        "Adds recommended retrieval-query, retrieval-document, and symmetric" in message
        for message in debug_messages
    )
    assert not any("embedding dimension: using" in message for message in info_messages)
    assert any("embedding dimension: using" in message for message in debug_messages)
    assert any(
        f"Enabled torch.compile for {DEFAULT_EMBEDDING_MODEL_NAME} inner transformer"
        in message
        for message in debug_messages
    )


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_embedding_bf16_autocast_rejection_falls_back_to_float32(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    device: str,
) -> None:
    """A rejected bf16 autocast context must keep compute in float32."""
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, autocast_log, _fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        mps_available=True,
        bf16_supported=True,
        autocast_behavior="raise",
        torch_version="2.13.0",
    )
    _fake_torch.cpu = types.SimpleNamespace(get_capabilities=lambda: {"bf16": True})

    with caplog.at_level(logging.WARNING):
        builder = EmbeddingGraphBuilder(max_papers=1, device=device, client=MagicMock())
    builder._load_model()

    assert builder._source_dtype_hint == "float32"
    assert builder._autocast_enabled is False
    assert (
        init_log["kwargs"]["model_kwargs"].get(
            "dtype", init_log["kwargs"]["model_kwargs"].get("torch_dtype")
        )
        == "auto"
    )
    assert autocast_log == [("call", device, _bf16_token), ("enter",)]
    assert any(
        "runtime rejected that context" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize(
    ("capabilities", "legacy", "expected"),
    [
        ({"avx512_bf16": True}, None, True),
        ({"amx_bf16": True}, None, True),
        ({"bf16": True}, None, True),
        ({"sve_bf16": True}, None, True),
        ({"avx2": True}, None, False),
        ({"avx512_f": True, "avx_ne_convert": True}, None, False),
        (None, True, True),
        (None, False, False),
        (None, None, False),
    ],
)
def test_cpu_bf16_autocast_uses_native_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: dict[str, bool] | None,
    legacy: bool | None,
    expected: bool,
) -> None:
    """Enable CPU BF16 only with reported native instructions and valid autocast.

    :param pytest.MonkeyPatch monkeypatch: Isolated runtime patching fixture.
    :param dict[str, bool] | None capabilities: Modern CPU feature mapping.
    :param bool | None legacy: Older torch x86 BF16 probe result, if available.
    :param bool expected: Expected BF16 runtime selection.
    :return None: Checks precision, outputs, automatic weights, and portable attention.
    """
    init_log, encode_log = _install_fake_sentence_transformers(monkeypatch)
    token, autocast_log, torch = _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
    )
    torch.cpu = types.SimpleNamespace()
    if capabilities is not None:
        torch.cpu.get_capabilities = lambda: capabilities
    if legacy is not None:
        torch.cpu._is_avx512_bf16_supported = lambda: legacy
    builder = EmbeddingGraphBuilder(device="cpu", client=MagicMock())
    builder._load_model()
    output = builder._encode_texts(["paper"])
    assert output.dtype == np.float32
    assert builder._source_dtype_hint == ("bfloat16" if expected else "float32")
    assert builder._autocast_enabled is expected
    assert builder._tf32_mode == "off"
    assert init_log["kwargs"]["model_kwargs"] == {"dtype": "auto"}
    assert (("call", "cpu", token) in autocast_log) is expected
    if expected:
        assert encode_log[-1]["normalize_embeddings"] is False
        np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1.0, atol=1e-6)


def test_cuda_bf16_policy_rejects_emulated_only_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA autocast should require native bf16 instead of tensor emulation."""
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, autocast_log, fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=True,
        bf16_native_supported=False,
    )

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    builder._load_model()

    assert builder._source_dtype_hint == "float32"
    assert builder._autocast_enabled is False
    assert fake_torch._bf16_support_calls == [False]
    assert autocast_log == []
    assert (
        init_log["kwargs"]["model_kwargs"].get(
            "dtype", init_log["kwargs"]["model_kwargs"].get("torch_dtype")
        )
        == "auto"
    )


@pytest.mark.parametrize(
    ("weight_dtype", "message"),
    [
        ("torch.float16", "forbids float16"),
        ("torch.bfloat16", "has not verified bfloat16 compute"),
        ("torch.float64", "unsupported automatic tensor dtype.*float64"),
    ],
)
@pytest.mark.parametrize("dtype_location", ["parameter", "buffer"])
def test_automatic_checkpoint_dtype_must_match_verified_runtime_policy(
    monkeypatch: pytest.MonkeyPatch,
    weight_dtype: str,
    message: str,
    dtype_location: str,
) -> None:
    """Live parameter and buffer dtypes must respect the precision policy.

    :param pytest.MonkeyPatch monkeypatch: Isolated runtime patching fixture.
    :param str weight_dtype: Fake floating dtype token to reject.
    :param str message: Expected diagnostic pattern.
    :param str dtype_location: Whether the incompatible tensor is a buffer or parameter.
    :return None: Confirms invalid precision is rejected before inference.
    """

    class _Parameter:
        def __init__(self, dtype: str):
            self.dtype = dtype

    class _DtypeModel:
        def __init__(self, _model_name: str, **_kwargs: Any):
            self._parameters = [
                _Parameter(
                    weight_dtype if dtype_location == "parameter" else "torch.float32"
                )
            ]
            self._buffers = (
                [_Parameter(weight_dtype)] if dtype_location == "buffer" else []
            )

        def parameters(self) -> Iterable[_Parameter]:
            """Iterate fake parameters for live dtype inspection.

            :return Iterable[_Parameter]: One configured fake parameter.
            """
            return iter(self._parameters)

        def buffers(self) -> Iterable[_Parameter]:
            """Iterate fake buffers for live dtype inspection.

            :return Iterable[_Parameter]: Configured fake buffers, if any.
            """
            return iter(self._buffers)

    monkeypatch.setattr(
        embedding_module,
        "_import_sentence_transformer_class",
        lambda: _DtypeModel,
    )
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
    )
    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/custom-embedding-model",
        client=MagicMock(),
    )

    with pytest.raises(
        embedding_module.EmbeddingPrecisionCompatibilityError,
        match=message,
    ):
        builder._load_model()


def test_automatic_checkpoint_dtype_inspection_must_succeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic dtype validation should fail when live parameters are inaccessible."""

    class _OpaqueModel:
        def __init__(self, _model_name: str, **_kwargs: Any):
            """Create a fake model whose parameters cannot be inspected.

            :param str _model_name: Ignored checkpoint name.
            :param Any _kwargs: Ignored model-loading arguments.
            """
            pass

        def parameters(self) -> Iterable[object]:
            """Raise the simulated live-parameter inspection failure.

            :return Iterable[object]: This method never returns.
            :raises RuntimeError: Always, to simulate an opaque model.
            """
            raise RuntimeError("parameters unavailable")

    monkeypatch.setattr(
        embedding_module,
        "_import_sentence_transformer_class",
        lambda: _OpaqueModel,
    )
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
    )
    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/custom-embedding-model",
        client=MagicMock(),
    )

    with pytest.raises(
        embedding_module.EmbeddingPrecisionCompatibilityError,
        match="Could not inspect live parameter and buffer dtypes",
    ):
        builder._load_model()


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
        model_name="org/generic-embedding-model",
        client=MagicMock(),
    )
    accelerator_builder._load_model()

    assert "attn_implementation" not in init_log["kwargs"]["model_kwargs"]
    assert (
        init_log["kwargs"]["model_kwargs"].get(
            "dtype", init_log["kwargs"]["model_kwargs"].get("torch_dtype")
        )
        == "auto"
    )
    assert accelerator_builder._source_dtype_hint == "float32"
    assert accelerator_builder._attention_implementation_hint is None
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
        model_name="org/generic-embedding-model",
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
        model_name="org/generic-embedding-model",
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


@pytest.mark.parametrize(
    ("device", "bf16", "installed", "expected"),
    [
        ("cuda", True, True, "flash_attention_2"),
        ("cuda", True, False, "sdpa"),
        ("cuda", False, True, "sdpa"),
        ("mps", True, True, "sdpa"),
        ("cpu", False, True, None),
    ],
)
def test_attention_selection_uses_fa2_only_on_bf16_cuda(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    bf16: bool,
    installed: bool,
    expected: str | None,
) -> None:
    """Select installed FA2 only on the supported CUDA compute path.

    :param pytest.MonkeyPatch monkeypatch: Isolated runtime patching fixture.
    :param str device: Requested execution device.
    :param bool bf16: Native CUDA BF16 availability.
    :param bool installed: Whether the FA2 module exists.
    :param str | None expected: Expected attention selection.
    :return None: Checks constructor kwargs and effective runtime hint.
    """
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        bf16_supported=bf16,
        mps_available=True,
        torch_version="2.13.0",
    )
    monkeypatch.setattr(embedding_module, "_module_available", lambda _name: installed)
    builder = EmbeddingGraphBuilder(device=device, client=MagicMock())
    builder._load_model()
    assert builder._attention_implementation_hint == expected
    assert init_log["kwargs"]["model_kwargs"].get("attn_implementation") == expected


def test_fa2_load_failure_retries_same_checkpoint_with_sdpa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unusable FA2 installation must retain the chosen checkpoint via SDPA.

    :param pytest.MonkeyPatch monkeypatch: Isolated runtime patching fixture.
    :return None: Checks the fallback backend and automatic checkpoint dtype.
    """
    _install_fake_sentence_transformers(monkeypatch)
    _install_fake_torch(monkeypatch, cuda_available=True, bf16_supported=True)
    original_cls = embedding_module._import_sentence_transformer_class()
    attempts = []

    def load(model_name: str, **kwargs: Any) -> Any:
        """Reject FA2 while permitting the same model through SDPA.

        :param str model_name: Requested checkpoint.
        :param Any kwargs: SentenceTransformer constructor arguments.
        :return Any: Fake successfully loaded model.
        """
        attempts.append((model_name, dict(kwargs["model_kwargs"])))
        if kwargs["model_kwargs"]["attn_implementation"] == "flash_attention_2":
            raise ImportError("flash_attn CUDA extension cannot be loaded")
        return original_cls(model_name, **kwargs)

    monkeypatch.setattr(embedding_module, "_module_available", lambda _name: True)
    monkeypatch.setattr(
        embedding_module, "_import_sentence_transformer_class", lambda: load
    )
    builder = EmbeddingGraphBuilder(device="cuda", client=MagicMock())
    builder._load_model()
    assert attempts == [
        (
            DEFAULT_EMBEDDING_MODEL_NAME,
            {"dtype": "auto", "attn_implementation": "flash_attention_2"},
        ),
        (
            DEFAULT_EMBEDDING_MODEL_NAME,
            {"dtype": "auto", "attn_implementation": "sdpa"},
        ),
    ]
    assert builder._attention_implementation_hint == "sdpa"


def test_verified_fa2_autocast_filters_only_redundant_load_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hide the FP32-weight warning only while verified bf16 FA2 loads.

    :param pytest.MonkeyPatch monkeypatch: Isolated runtime patching fixture.
    :return None: Checks exact filtering and cleanup after a successful load.
    """
    _install_fake_sentence_transformers(monkeypatch)
    _install_fake_torch(monkeypatch, cuda_available=True, bf16_supported=True)
    original_cls = embedding_module._import_sentence_transformer_class()
    transformers_logger = logging.getLogger("transformers.modeling_utils")

    def load(model_name: str, **kwargs: Any) -> Any:
        """Emit the upstream warning plus an unrelated load warning.

        :param str model_name: Requested checkpoint.
        :param Any kwargs: SentenceTransformer constructor arguments.
        :return Any: Fake loaded model.
        """
        transformers_logger.warning(
            "Flash Attention 2 only supports torch.float16 and torch.bfloat16 "
            "dtypes, but the current dype is torch.float32."
        )
        transformers_logger.warning(
            "Flash Attention 2 only supports torch.float16 and torch.bfloat16 "
            "dtypes, but the current dype is torch.float64."
        )
        transformers_logger.warning("independent Transformers load warning")
        return original_cls(model_name, **kwargs)

    monkeypatch.setattr(embedding_module, "_module_available", lambda _name: True)
    monkeypatch.setattr(
        embedding_module, "_import_sentence_transformer_class", lambda: load
    )
    handler = MagicMock(spec=logging.Handler)
    handler.level = logging.NOTSET
    transformers_logger.addHandler(handler)
    try:
        EmbeddingGraphBuilder(device="cuda", client=MagicMock())._load_model()
        transformers_logger.warning(
            "Flash Attention 2 only supports torch.float16 and torch.bfloat16 "
            "dtypes after loading."
        )
    finally:
        transformers_logger.removeHandler(handler)

    messages = [call.args[0].getMessage() for call in handler.handle.call_args_list]
    assert "independent Transformers load warning" in messages
    assert not any("current dype is torch.float32" in message for message in messages)
    assert any("current dype is torch.float64" in message for message in messages)
    assert any("dtypes after loading" in message for message in messages)


def test_fa2_load_warning_filter_disabled_preserves_fp32_warning() -> None:
    """Keep the upstream warning when verified bf16 FA2 is not active.

    :return None: Checks the disabled filter path.
    """
    transformers_logger = logging.getLogger("transformers.modeling_utils")
    handler = MagicMock(spec=logging.Handler)
    handler.level = logging.NOTSET
    transformers_logger.addHandler(handler)

    try:
        with embedding_module._suppress_expected_fa2_load_dtype_warning(enabled=False):
            transformers_logger.warning(
                "Flash Attention 2 only supports torch.float16 and torch.bfloat16 "
                "dtypes, but the current dype is torch.float32."
            )
    finally:
        transformers_logger.removeHandler(handler)

    assert any(
        "current dype is torch.float32" in call.args[0].getMessage()
        for call in handler.handle.call_args_list
    )


def test_fa2_load_warning_filter_is_removed_after_failure() -> None:
    """Restore upstream logging even when model construction raises.

    :return None: Checks filter cleanup on the exceptional path.
    """
    transformers_logger = logging.getLogger("transformers.modeling_utils")
    handler = MagicMock(spec=logging.Handler)
    handler.level = logging.NOTSET
    transformers_logger.addHandler(handler)

    try:
        with pytest.raises(RuntimeError, match="load failed"):
            with embedding_module._suppress_expected_fa2_load_dtype_warning(
                enabled=True
            ):
                raise RuntimeError("load failed")
        transformers_logger.warning(
            "Flash Attention 2 only supports torch.float16 and torch.bfloat16 "
            "dtypes after failed loading."
        )
    finally:
        transformers_logger.removeHandler(handler)

    assert any(
        "after failed loading" in call.args[0].getMessage()
        for call in handler.handle.call_args_list
    )


def test_compile_replaces_and_restores_sentence_transformers_active_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ST6 executes .model even if assignment to the legacy alias is accepted.

    :param pytest.MonkeyPatch monkeypatch: Isolated runtime patching fixture.
    :return None: Checks compilation and eager recovery target the active attribute.
    """
    _install_fake_sentence_transformers(monkeypatch)
    _install_fake_torch(
        monkeypatch, cuda_available=True, bf16_supported=True, compile_behavior="tagged"
    )
    builder = EmbeddingGraphBuilder(device="cuda", client=MagicMock())
    builder._load_model()
    block = builder.model[0]
    original = block.auto_model
    block.model = original
    builder.enable_torch_compile = True
    builder._maybe_compile_inner_transformer()
    assert block.model == ("compiled", original)
    assert block.auto_model is original
    assert builder._inner_model_compiled
    assert builder._restore_eager_model_after_compile_failure(
        RuntimeError("backend failure")
    )
    assert block.model is original
    assert not builder._inner_model_compiled


@pytest.mark.parametrize(
    "logging_option", ["ignore_logging_functions", "reorderable_logging_functions"]
)
def test_fa2_compile_logging_configuration_is_scoped(
    monkeypatch: pytest.MonkeyPatch,
    logging_option: str,
) -> None:
    """Compiler log controls preserve existing settings and restore on failure.

    :param pytest.MonkeyPatch monkeypatch: Isolated runtime patching fixture.
    :param str logging_option: Modern or older supported torch logging control.
    :return None: Verifies restoration after an interrupted encode context.
    """
    _install_fake_torch(monkeypatch, cuda_available=True, bf16_supported=True)
    builder = EmbeddingGraphBuilder(device="cuda", client=MagicMock())
    builder._inner_model_compiled = True
    builder._attention_implementation_hint = "flash_attention_2"
    monkeypatch.setattr(builder, "_autocast_context", nullcontext)
    monkeypatch.setattr(builder, "_tf32_context", nullcontext)
    existing_logger = MagicMock()
    flash_logger = types.SimpleNamespace(warning_once=MagicMock())
    config = types.SimpleNamespace(**{logging_option: {existing_logger}})
    config.patch = lambda **kwargs: patch.object(
        config, logging_option, kwargs[logging_option]
    )
    monkeypatch.setitem(
        sys.modules, "torch._dynamo", types.SimpleNamespace(config=config)
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.modeling_flash_attention_utils",
        types.SimpleNamespace(logger=flash_logger),
    )
    with pytest.raises(RuntimeError, match="encoding failure"):
        with builder._precision_context():
            assert getattr(config, logging_option) == {
                existing_logger,
                flash_logger.warning_once,
            }
            raise RuntimeError("encoding failure")
    assert getattr(config, logging_option) == {existing_logger}


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_embedding_compile_hydration_policy(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """CPU and CUDA compile during hydration; MPS stays deferred.

    :param pytest.MonkeyPatch monkeypatch: Isolated runtime patching fixture.
    :param str device: Accelerator whose hydration policy is checked.
    :return None: Checks active compilation against cold corpus metadata.
    """
    init_log, _ = _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, _autocast_log, fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        mps_available=True,
        bf16_supported=True,
        compile_behavior="tagged",
        torch_version="2.13.0",
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        enable_torch_compile=True,
        semantic_source="arxiv-corpus",
        device=device,
        client=MagicMock(),
    )
    _pin_model_fingerprint(monkeypatch, builder)
    monkeypatch.setattr(builder, "_cache_hydrated_for_active_spec", lambda: False)
    builder._load_model()

    original = init_log["auto_model_before_compile"]
    assert builder.model is not None
    if device in {"cuda", "cpu"}:
        assert builder.model[0].auto_model == ("compiled", original)
        assert builder._inner_model_compiled
        expected_kwargs: dict[str, Any] = {"dynamic": True}
        if device == "cpu":
            expected_kwargs["options"] = {"max_autotune": True}
        assert fake_torch._compile_calls[-1]["kwargs"] == expected_kwargs
        return
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


@pytest.mark.parametrize(
    ("layout", "nested_transformer"),
    [
        ("arbitrary", False),
        ("snapshot", False),
        ("nested", True),
    ],
)
def test_local_embeddinggemma_artifacts_resolve_task_profile(
    tmp_path: Path,
    layout: str,
    nested_transformer: bool,
) -> None:
    """Local and snapshot paths should retain EmbeddingGemma task separation."""
    if layout == "snapshot":
        model_path = (
            tmp_path
            / "models--custom--fine-tune"
            / "snapshots"
            / "0123456789abcdef0123456789abcdef01234567"
        )
    else:
        model_path = tmp_path / f"unrelated-checkpoint-name-{layout}"
    _write_local_sentence_transformer_profile(
        model_path,
        embeddinggemma=True,
        nested_transformer=nested_transformer,
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name=str(model_path),
        client=MagicMock(),
    )
    paper = Paper(
        paper_id="paper",
        title="Attention Models",
        year=2024,
        abstract="An abstract.",
    )
    query = format_paper_for_embedding(
        profile=builder.model_profile,
        paper=paper,
        task=EmbeddingTask.RETRIEVAL_QUERY,
    )
    document = format_paper_for_embedding(
        profile=builder.model_profile,
        paper=paper,
        task=EmbeddingTask.RETRIEVAL_DOCUMENT,
    )
    similarity = format_paper_for_embedding(
        profile=builder.model_profile,
        paper=paper,
        task=EmbeddingTask.GRAPH_SIMILARITY,
    )

    assert builder.model_profile.schema_token == "embeddinggemma-v2"
    assert builder.truncate_dim == 512
    assert query == "task: search result | query: Attention Models. An abstract."
    assert document == "title: Attention Models | text: An abstract."
    assert similarity == (
        "task: sentence similarity | query: Attention Models. An abstract."
    )
    assert len({query, document, similarity}) == 3
    assert "profile=embeddinggemma-v2" in builder.embedding_cache.model_name
    assert "profile=embeddinggemma-v1" not in builder.embedding_cache.model_name
    assert "profile=embeddinggemma-v2" in builder.graph_embedding_cache.model_name


def test_plain_transformers_embeddinggemma_export_resolves_from_model_contract(
    tmp_path: Path,
) -> None:
    """Bidirectional model metadata should identify exports without ST prompts."""
    model_path = tmp_path / "plain-transformers-export"
    _write_local_sentence_transformer_profile(
        model_path,
        embeddinggemma=True,
        include_task_prompts=False,
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name=str(model_path),
        client=MagicMock(),
    )

    assert builder.model_profile.schema_token == "embeddinggemma-v2"
    assert builder.model_profile.minimum_transformers_version == (5, 2)
    assert builder.truncate_dim == 512


def test_ambiguous_local_gemma_checkpoint_warns_before_default_profile(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Architecture alone should not silently claim or discard an embedding profile."""
    model_path = tmp_path / "ambiguous-gemma-export"
    _write_local_sentence_transformer_profile(
        model_path,
        embeddinggemma=True,
        include_task_prompts=False,
        include_bidirectional_flag=False,
    )

    with caplog.at_level(logging.WARNING):
        builder = EmbeddingGraphBuilder(
            max_papers=1,
            model_name=str(model_path),
            client=MagicMock(),
        )

    assert builder.model_profile.schema_token == "default-v1"
    assert any(
        "automatic profile detection cannot prove the embedding contract"
        in record.getMessage()
        for record in caplog.records
    )


def test_local_model_profile_override_and_generic_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local metadata should outrank names while explicit overrides remain available."""
    misleading_path = tmp_path / "google" / "embeddinggemma-custom"
    _write_local_sentence_transformer_profile(
        misleading_path,
        embeddinggemma=False,
    )
    monkeypatch.chdir(tmp_path)

    generic = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="google/embeddinggemma-custom",
        client=MagicMock(),
    )
    forced_embeddinggemma = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="google/embeddinggemma-custom",
        model_profile="embeddinggemma",
        client=MagicMock(),
    )
    recognized_but_forced_default = EmbeddingGraphBuilder(
        max_papers=1,
        model_name=DEFAULT_EMBEDDING_MODEL_NAME,
        model_profile="default",
        client=MagicMock(),
    )

    assert generic.model_profile.schema_token == "default-v1"
    assert generic.truncate_dim is None
    assert forced_embeddinggemma.model_profile.schema_token == "embeddinggemma-v2"
    assert forced_embeddinggemma.truncate_dim == 512
    assert recognized_but_forced_default.model_profile.schema_token == "default-v1"
    assert recognized_but_forced_default.truncate_dim is None
    with pytest.raises(ValueError, match="Unknown model profile"):
        EmbeddingGraphBuilder(
            max_papers=1,
            model_profile="unknown-profile",
            client=MagicMock(),
        )


@pytest.mark.parametrize(
    ("contract_source", "active_bidirectional_attention", "should_fail"),
    [
        pytest.param("prompts", False, True, id="prompts-loaded-false"),
        pytest.param("prompts", None, True, id="prompts-loaded-missing"),
        pytest.param("root-true", False, True, id="root-true-active-false"),
        pytest.param("known-model", True, False, id="loaded-true"),
        pytest.param("explicit", False, True, id="explicit-profile-loaded-false"),
        pytest.param("generic", False, False, id="generic-profile-unaffected"),
    ],
)
def test_loaded_embeddinggemma_requires_bidirectional_attention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_source: str,
    active_bidirectional_attention: bool | None,
    should_fail: bool,
) -> None:
    """The live transformer must satisfy the selected EmbeddingGemma contract."""
    model_name = DEFAULT_EMBEDDING_MODEL_NAME
    model_profile = "auto"
    if contract_source in {"prompts", "root-true"}:
        model_path = tmp_path / contract_source
        _write_local_sentence_transformer_profile(
            model_path,
            embeddinggemma=True,
            include_task_prompts=contract_source == "prompts",
            include_bidirectional_flag=contract_source == "root-true",
        )
        model_name = str(model_path)
    elif contract_source == "explicit":
        model_name = "org/generic-embedding-model"
        model_profile = "embeddinggemma"
    elif contract_source == "generic":
        model_name = "org/generic-embedding-model"

    _install_fake_sentence_transformers(
        monkeypatch,
        active_bidirectional_attention=active_bidirectional_attention,
    )
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
    )
    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name=model_name,
        model_profile=model_profile,
        client=MagicMock(),
    )

    if should_fail:
        with pytest.raises(
            embedding_module.EmbeddingBackendCompatibilityError,
            match="bidirectional attention",
        ):
            builder._load_model()
        assert builder.model is None
        assert builder._embedding_cache is None
        return

    builder._load_model()
    assert builder.model is not None


def test_embedding_fallback_rebinds_profile_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different-family fallback should receive its own complete load contract."""
    requested_model = "org/generic-model"
    fallback_model = tmp_path / "fine-tuned-local-model"
    _write_local_sentence_transformer_profile(
        fallback_model,
        embeddinggemma=True,
    )
    init_log, _ = _install_fake_sentence_transformers(
        monkeypatch,
        fail_model_names={requested_model},
    )
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
    )
    monkeypatch.setattr(
        embedding_module,
        "DEFAULT_EMBEDDING_MODEL_FALLBACKS",
        {requested_model: (str(fallback_model),)},
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name=requested_model,
        client=MagicMock(),
    )
    builder._load_model()

    assert init_log["attempts"] == [requested_model, str(fallback_model)]
    assert "truncate_dim" not in init_log["attempt_kwargs"][0]
    assert init_log["attempt_kwargs"][1]["truncate_dim"] == 512
    assert builder._active_model_name == str(fallback_model)
    assert builder.model_profile.schema_token == "embeddinggemma-v2"
    assert builder.truncate_dim == 512
    assert f"model={fallback_model}" in builder.embedding_cache.model_name
    assert "profile=embeddinggemma-v2" in builder.embedding_cache.model_name


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
    requested_cache_path = builder.embedding_cache.h5_path
    builder._load_model()
    fingerprint = builder._resolve_model_fingerprint()
    builder._ensure_cache_model_fingerprint()

    assert builder._active_model_name == fallback_model
    assert (
        fingerprint == f"hf::{fallback_model}::0123456789abcdef0123456789abcdef01234567"
    )
    assert builder.embedding_cache.h5_path != requested_cache_path
    assert f"model={fallback_model}" in builder.embedding_cache.model_name
    assert f"artifact={fingerprint}" in builder.embedding_cache.model_name


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


@pytest.mark.parametrize("semantic_source", ["candidates", "arxiv-corpus"])
def test_embedding_cache_namespace_matches_default_and_explicit_truncate_dim(
    monkeypatch: pytest.MonkeyPatch,
    semantic_source: str,
) -> None:
    """Default caches should match explicit 512d and stay separate from 256d.

    :param pytest.MonkeyPatch monkeypatch: Isolated runtime patching fixture.
    :param str semantic_source: Retrieval cache sourcing mode.
    :return None: Checks retrieval and graph cache identities after the default change.
    """
    monkeypatch.setattr(
        EmbeddingGraphBuilder,
        "_resolve_source_dtype_hint",
        lambda self: "float32",
    )

    implicit = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="google/embeddinggemma-300m",
        truncate_dim=None,
        semantic_source=semantic_source,
        client=MagicMock(),
    )
    explicit = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="google/embeddinggemma-300m",
        truncate_dim=512,
        semantic_source=semantic_source,
        client=MagicMock(),
    )
    previous_default = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="google/embeddinggemma-300m",
        truncate_dim=256,
        semantic_source=semantic_source,
        client=MagicMock(),
    )

    assert implicit.embedding_cache.model_name == explicit.embedding_cache.model_name
    assert (
        implicit.graph_embedding_cache.model_name
        == explicit.graph_embedding_cache.model_name
    )
    assert previous_default.truncate_dim == 256
    assert previous_default.embedding_cache.h5_path != implicit.embedding_cache.h5_path
    assert (
        previous_default.graph_embedding_cache.h5_path
        != implicit.graph_embedding_cache.h5_path
    )


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinned commits work offline; unresolved mutable identities fail closed."""

    commit_sha = "0123456789abcdef0123456789abcdef01234567"
    pinned = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/offline-test",
        model_revision=commit_sha,
        client=MagicMock(),
    )
    assert pinned._resolve_model_fingerprint() == f"hf::org/offline-test::{commit_sha}"

    for has_payload, cached_fingerprint in [
        (True, f"hf::org/offline-unverified::{commit_sha}"),
        (True, None),
        (False, None),
    ]:
        builder = EmbeddingGraphBuilder(
            max_papers=1,
            model_name="org/offline-unverified",
            model_revision="refs/pr/12",
            client=MagicMock(),
        )
        builder.embedding_cache.has_cached_payload = MagicMock(return_value=has_payload)
        builder.embedding_cache.get_model_fingerprint = MagicMock(
            return_value=cached_fingerprint
        )
        builder.embedding_cache.clear = MagicMock()
        builder.embedding_cache.set_model_fingerprint = MagicMock()
        builder._resolve_model_fingerprint = MagicMock(
            side_effect=RuntimeError("network unavailable")
        )

        with pytest.raises(
            RuntimeError,
            match="refusing persistent cache access",
        ):
            builder._ensure_cache_model_fingerprint()

        builder.embedding_cache.clear.assert_not_called()
        builder.embedding_cache.set_model_fingerprint.assert_not_called()


def test_embedding_cache_clears_payload_missing_fingerprint_after_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unidentified legacy payload must be cleared before recording identity."""

    builder = EmbeddingGraphBuilder(
        max_papers=1, model_name="org/needs-fingerprint", client=MagicMock()
    )
    builder._resolved_model_fingerprint = "resolved-fp"
    builder._bind_embedding_cache_to_active_model()
    builder.embedding_cache.has_cached_payload = MagicMock(return_value=True)
    builder.embedding_cache.get_model_fingerprint = MagicMock(return_value=None)
    builder.embedding_cache.set_model_fingerprint = MagicMock()

    builder._resolve_model_fingerprint = MagicMock(return_value="resolved-fp")
    builder.embedding_cache.clear = MagicMock()

    builder._ensure_cache_model_fingerprint()

    builder.embedding_cache.set_model_fingerprint.assert_called_once_with("resolved-fp")
    assert builder._resolved_model_fingerprint == "resolved-fp"
    builder.embedding_cache.clear.assert_called_once()


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

    expected_artifact = artifact_builder._resolve_inference_artifact_digest(
        snapshot_root
    )
    assert artifact_builder._resolve_model_fingerprint() == (
        f"hf::org/artifact-model::revision=refs/pr/7::artifact={expected_artifact}"
    )


def test_local_model_fingerprint_covers_inference_artifact_manifest(
    tmp_path: Path,
) -> None:
    """Weights, tokenizer, and pooling changes must alter local model identity."""
    model_path = tmp_path / "local-model"
    pooling_path = model_path / "1_Pooling"
    pooling_path.mkdir(parents=True)
    (model_path / "config.json").write_text('{"model_type":"test"}')
    (model_path / "modules.json").write_text(
        '[{"idx":0,"name":"pool","path":"1_Pooling","type":"Pooling"}]'
    )
    (model_path / "model.safetensors").write_bytes(b"weights-a")
    (model_path / "tokenizer.json").write_text('{"version":"a"}')
    (pooling_path / "config.json").write_text('{"pooling_mode_mean_tokens":true}')
    (model_path / "README.md").write_text("documentation a")

    def _identity() -> tuple[str, Path]:
        """Resolve a fresh fingerprint and physical cache for the local fixture."""
        builder = EmbeddingGraphBuilder(
            max_papers=1,
            model_name=str(model_path),
            client=MagicMock(),
        )
        fingerprint = builder._resolve_model_fingerprint()
        builder._ensure_cache_model_fingerprint()
        return fingerprint, builder.embedding_cache.h5_path

    initial, initial_cache_path = _identity()
    (model_path / "README.md").write_text("documentation b")
    assert _identity() == (initial, initial_cache_path)

    (pooling_path / "config.json").write_text('{"pooling_mode_mean_tokens":false}')
    pooling_changed, pooling_cache_path = _identity()
    assert pooling_changed != initial
    assert pooling_cache_path != initial_cache_path

    (model_path / "tokenizer.json").write_text('{"version":"b"}')
    tokenizer_changed, tokenizer_cache_path = _identity()
    assert tokenizer_changed != pooling_changed
    assert tokenizer_cache_path != pooling_cache_path

    (model_path / "model.safetensors").write_bytes(b"weights-b")
    weights_changed, weights_cache_path = _identity()
    assert weights_changed != tokenizer_changed
    assert weights_cache_path != tokenizer_cache_path

    (model_path / "model.safetensors").write_bytes(b"weights-a")
    (model_path / "tokenizer.json").write_text('{"version":"a"}')
    (pooling_path / "config.json").write_text('{"pooling_mode_mean_tokens":true}')
    assert _identity() == (initial, initial_cache_path)


def test_local_model_fingerprint_validates_sharded_weight_indexes(
    tmp_path: Path,
) -> None:
    """Every indexed shard must participate in identity and remain present."""
    model_path = tmp_path / "sharded-model"
    model_path.mkdir()
    first_shard = model_path / "model-00001-of-00002.safetensors"
    second_shard = model_path / "model-00002-of-00002.safetensors"
    first_shard.write_bytes(b"first-a")
    second_shard.write_bytes(b"second-a")
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": first_shard.name,
                    "layer.1": second_shard.name,
                }
            }
        )
    )

    first = EmbeddingGraphBuilder(
        max_papers=1,
        model_name=str(model_path),
        client=MagicMock(),
    )._resolve_model_fingerprint()
    second_shard.write_bytes(b"second-b")
    second = EmbeddingGraphBuilder(
        max_papers=1,
        model_name=str(model_path),
        client=MagicMock(),
    )._resolve_model_fingerprint()
    assert second != first

    first_shard.unlink()
    with pytest.raises(RuntimeError, match="missing shard"):
        EmbeddingGraphBuilder(
            max_papers=1,
            model_name=str(model_path),
            client=MagicMock(),
        )._resolve_model_fingerprint()


def test_embedding_cache_namespace_partitions_revisions_without_thrashing() -> None:
    """Revision A to B to A should select two stable physical cache namespaces."""
    first_a = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/revisioned-model",
        model_revision="revision-a",
        client=MagicMock(),
    )
    revision_b = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/revisioned-model",
        model_revision="revision-b",
        client=MagicMock(),
    )
    second_a = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/revisioned-model",
        model_revision="revision-a",
        client=MagicMock(),
    )

    assert first_a.embedding_cache.h5_path != revision_b.embedding_cache.h5_path
    assert first_a.embedding_cache.h5_path == second_a.embedding_cache.h5_path
    assert "revision=revision-a" in first_a.embedding_cache.model_name
    assert "representation=retrieval-document-v1" in first_a.embedding_cache.model_name


def test_task_specific_embedding_caches_partition_storage_and_formatters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corpus retrieval and graph STS vectors must use distinct physical contracts."""
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        semantic_source="arxiv-corpus",
        storage_precision="int8",
        binary_prefilter=True,
        client=MagicMock(),
    )
    _pin_model_fingerprint(monkeypatch, builder, fingerprint="artifact-a")

    retrieval_cache = builder.embedding_cache
    graph_cache = builder.graph_embedding_cache

    assert retrieval_cache.h5_path != graph_cache.h5_path
    assert retrieval_cache.storage_precision == "int8"
    assert retrieval_cache.binary_prefilter is True
    assert graph_cache.storage_precision == "float32"
    assert graph_cache.binary_prefilter is False
    assert retrieval_cache.text_formatter_fingerprint != (
        graph_cache.text_formatter_fingerprint
    )
    assert builder._document_formatter_fingerprint == "0d40a6b5f7852b1a"
    assert builder._similarity_formatter_fingerprint == "78ac67b7a864666a"
    assert "representation=retrieval-document-v1" in retrieval_cache.model_name
    assert "representation=graph-similarity-v1" in graph_cache.model_name
    assert "storage_precision=float32" in graph_cache.model_name
    assert "mode=" not in graph_cache.model_name


def test_force_rebuild_clears_retrieval_and_graph_task_caches() -> None:
    """An explicit rebuild request should clear both semantic representations."""
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        force_rebuild_cache=True,
        force_rebuild_reason="task contract changed",
        client=MagicMock(),
    )
    builder._resolved_model_fingerprint = "artifact-a"
    retrieval_namespace = builder._embedding_cache_namespace(
        artifact_identity="artifact-a"
    )
    graph_namespace = builder._graph_embedding_cache_namespace()
    retrieval_cache = MagicMock(model_name=retrieval_namespace)
    graph_cache = MagicMock(model_name=graph_namespace)
    builder.embedding_cache = retrieval_cache
    builder.graph_embedding_cache = graph_cache

    builder._bind_embedding_cache_to_active_model()

    expected_reason = (
        "explicit --force-rebuild-cache request; user_reason=task contract changed"
    )
    retrieval_cache.clear.assert_called_once_with(reason=expected_reason)
    graph_cache.clear.assert_called_once_with(reason=expected_reason)
    assert builder._pending_force_rebuild_reason is None


def test_paper_embedding_task_dispatcher_uses_id_fallback() -> None:
    """Every task should retain paper identity when title and abstract are empty."""
    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    paper = Paper(paper_id="paper-without-text", title="", year=None, abstract="")

    query = format_paper_for_embedding(
        profile=builder.model_profile,
        paper=paper,
        task=EmbeddingTask.RETRIEVAL_QUERY,
    )
    document = format_paper_for_embedding(
        profile=builder.model_profile,
        paper=paper,
        task=EmbeddingTask.RETRIEVAL_DOCUMENT,
    )
    graph = format_paper_for_embedding(
        profile=builder.model_profile,
        paper=paper,
        task=EmbeddingTask.GRAPH_SIMILARITY,
    )
    cached_metadata = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "abstract": paper.abstract,
    }

    assert "paper-without-text" in query
    assert "paper-without-text" in document
    assert "paper-without-text" in graph
    assert builder._format_retrieval_document_metadata(cached_metadata) == document
    assert builder._format_graph_similarity_metadata(cached_metadata) == graph


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


def test_collect_papers_excludes_corpus_alias_of_resolved_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corpus arXiv row must not duplicate its S2-resolved seed paper."""
    seed = Paper(
        paper_id="a" * 40,
        title="Recent Seed",
        year=2026,
        abstract="Seed abstract",
        arxiv_id="2608.15411",
    )
    builder = EmbeddingGraphBuilder(
        max_papers=3,
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    )
    builder.client.get_paper.return_value = seed
    monkeypatch.setattr(builder, "_load_model", lambda: None)
    monkeypatch.setattr(
        builder,
        "_encode_texts",
        lambda _texts, show_progress_bar=False: np.asarray(
            [[1.0, 0.0]], dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        builder,
        "_select_candidates",
        lambda _seed_embedding, *, use_streaming: [
            (
                "arxiv:2608.15411",
                {
                    "title": "Recent Seed",
                    "year": 2026,
                    "abstract": "Seed abstract",
                    "arxiv_id": "2608.15411",
                    "authors": [],
                },
                np.asarray([1.0, 0.0], dtype=np.float32),
            ),
            (
                "arxiv:2608.15412",
                {
                    "title": "Neighbor",
                    "year": 2026,
                    "abstract": "Neighbor abstract",
                    "arxiv_id": "2608.15412",
                    "authors": [],
                },
                np.asarray([0.8, 0.2], dtype=np.float32),
            ),
        ],
    )
    monkeypatch.setattr(builder, "_update_citation_counts", lambda _papers: None)

    papers = builder.collect_papers("arxiv:2608.15411")

    assert list(papers) == [seed.paper_id, "arxiv:2608.15412"]
    assert "arxiv:2608.15411" not in builder.retrieval_embeddings


def test_collect_papers_formats_all_seeds_in_retrieval_query_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Free-text and resolved-paper seeds should both use retrieval-query prompts."""

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
    assert paper_queries == ["Seed Title. Seed Abstract"]
    assert paper_documents == []
    assert paper_texts == ["Q::Seed Title. Seed Abstract"]

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

    assert prefetched_queries == ["Seed Title. Seed Abstract"]
    assert prefetched_documents == []
    assert prefetched_texts == ["Q::Seed Title. Seed Abstract"]


def test_graph_similarity_materialization_uses_sts_cache_and_warm_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final graph papers should encode once with STS prompts in their own cache."""
    builder = EmbeddingGraphBuilder(max_papers=2, client=MagicMock())
    _pin_model_fingerprint(monkeypatch, builder, fingerprint="artifact-sts")
    monkeypatch.setattr(builder, "_load_model", lambda: None)
    captured_texts: list[str] = []

    class _CaptureModel:
        def encode(self, texts: list[str], **_kwargs: Any) -> np.ndarray:
            captured_texts.extend(texts)
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)[: len(texts)]

    monkeypatch.setattr(builder, "_get_model_for_encoding", lambda: _CaptureModel())
    papers = {
        "seed": Paper(
            paper_id="seed",
            title="Attention Models",
            year=None,
            abstract="Transformer attention for language.",
            is_seed=True,
        ),
        "peer": Paper(
            paper_id="peer",
            title="Graph Attention",
            year=None,
            abstract="Attention mechanisms for graph neural networks.",
        ),
    }

    vectors = builder.materialize_graph_embeddings(papers)

    assert set(vectors) == set(papers)
    assert set(captured_texts) == {
        "task: sentence similarity | query: Attention Models. "
        "Transformer attention for language.",
        "task: sentence similarity | query: Graph Attention. "
        "Attention mechanisms for graph neural networks.",
    }
    assert builder.graph_embedding_cache.embedding_count() == 2
    assert builder.embedding_cache.embedding_count() == 0
    assert builder.retrieval_embeddings == {}

    class _FailOnEncode:
        def encode(self, _texts: list[str], **_kwargs: Any) -> np.ndarray:
            raise AssertionError("warm graph cache should not re-encode")

    monkeypatch.setattr(builder, "_get_model_for_encoding", lambda: _FailOnEncode())
    warmed = builder.materialize_graph_embeddings(papers)
    assert set(warmed) == set(papers)


def test_graph_similarity_materialization_fails_closed_on_missing_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial graph vector map must fail before pairwise scoring begins."""
    builder = EmbeddingGraphBuilder(max_papers=2, client=MagicMock())
    monkeypatch.setattr(builder, "_load_model", lambda: None)
    monkeypatch.setattr(
        builder,
        "_ensure_cache_model_fingerprint",
        lambda **_kwargs: None,
    )
    builder.graph_embedding_cache = MagicMock()
    builder.graph_embedding_cache.get_embeddings.return_value = {
        "seed": np.asarray([1.0, 0.0], dtype=np.float32)
    }
    monkeypatch.setattr(builder, "_get_model_for_encoding", lambda: MagicMock())
    papers = {
        "seed": Paper(paper_id="seed", title="Seed", year=None, is_seed=True),
        "peer": Paper(paper_id="peer", title="Peer", year=None),
    }

    with pytest.raises(RuntimeError, match="missing 1 vector.*peer"):
        builder.materialize_graph_embeddings(papers)


def test_embedding_build_prepares_graph_vectors_before_edge_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The base build lifecycle should invoke STS preparation before similarities."""
    builder = EmbeddingGraphBuilder(max_papers=2, top_k=1, client=MagicMock())
    papers = {
        "seed": Paper(paper_id="seed", title="Seed", year=2024, is_seed=True),
        "peer": Paper(paper_id="peer", title="Peer", year=2024),
    }
    monkeypatch.setattr(builder, "collect_papers", lambda _seed_id, **_kwargs: papers)
    prepare = MagicMock(
        side_effect=lambda selected: builder.embeddings.update(
            {
                paper_id: np.asarray([1.0, 0.0], dtype=np.float32)
                for paper_id in selected
            }
        )
    )
    monkeypatch.setattr(builder, "materialize_graph_embeddings", prepare)

    graph, seed_id = builder.build_graph("seed")

    assert seed_id == "seed"
    prepare.assert_called_once_with(papers)
    assert graph.has_edge("seed", "peer")


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
    assert builder.embedding_cache.mark_hydrated.call_args_list == [
        call(
            dataset_source=source,
            dataset_split=builder.dataset_split,
            corpus_size=builder.corpus_size,
            complete=False,
        ),
        call(
            dataset_source=source,
            dataset_split=builder.dataset_split,
            corpus_size=builder.corpus_size,
            complete=True,
        ),
    ]


def test_full_corpus_incremental_refresh_failure_stays_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed growth refresh must not remain reusable or trigger a rebuild."""
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
    events: list[tuple[str, bool | None]] = []
    builder.embedding_cache.mark_hydrated = MagicMock(
        side_effect=lambda **kwargs: events.append(("mark", kwargs["complete"]))
    )
    builder.embedding_cache.get_hydration_rowcount_reconciliation = MagicMock(
        return_value=None
    )
    monkeypatch.setattr(builder, "_resolve_dataset_split_row_count", lambda _: 101)
    monkeypatch.setattr(builder, "_ensure_int8_calibration_ranges", lambda **_: None)

    def fail_refresh(**_kwargs: Any) -> None:
        """Record the first write boundary before simulating source failure."""
        events.append(("refresh", None))
        raise RuntimeError("source iteration failed")

    monkeypatch.setattr(builder, "_hydrate_exact_hydration_source_slice", fail_refresh)
    clear_cache_mock = MagicMock()
    load_dataset_mock = MagicMock()
    monkeypatch.setattr(builder, "_clear_embedding_cache", clear_cache_mock)
    monkeypatch.setattr(builder, "_load_dataset_for_hydration", load_dataset_mock)

    with pytest.raises(RuntimeError, match="source iteration failed"):
        builder._ensure_cache_hydrated(use_streaming=False)

    assert events == [("mark", False), ("refresh", None)]
    builder.embedding_cache.is_hydrated.assert_called_once()
    clear_cache_mock.assert_not_called()
    load_dataset_mock.assert_not_called()


def test_incomplete_full_corpus_resume_failure_preserves_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient resume failure must propagate without clearing cached rows."""
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
    resume_mock = MagicMock(side_effect=RuntimeError("resume source failed"))
    clear_cache_mock = MagicMock()
    load_dataset_mock = MagicMock()
    monkeypatch.setattr(builder, "_resume_incomplete_full_corpus_cache", resume_mock)
    monkeypatch.setattr(builder, "_clear_embedding_cache", clear_cache_mock)
    monkeypatch.setattr(builder, "_load_dataset_for_hydration", load_dataset_mock)

    with pytest.raises(RuntimeError, match="resume source failed"):
        builder._ensure_cache_hydrated(use_streaming=False)

    resume_mock.assert_called_once_with(
        use_streaming=False,
        cached_dataset_source=source,
    )
    clear_cache_mock.assert_not_called()
    load_dataset_mock.assert_not_called()


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


def test_exact_hydration_slice_offsets_synthetic_paper_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offset hydration should preserve source positions in synthetic paper IDs."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        storage_precision="float32",
        corpus_size=None,
        use_streaming=False,
        client=MagicMock(),
    )
    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        MagicMock(
            return_value=(
                source,
                [
                    {"title": "First offset row"},
                    {"title": "Second offset row"},
                ],
            )
        ),
    )
    cached_records: list[dict[str, Any]] = []

    def _capture_batch(batch: list[dict[str, Any]]) -> int:
        """Capture hydrated records and report the number written.

        :param list[dict[str, Any]] batch: Hydration batch to capture.
        :return int: Number of captured records.
        """
        cached_records.extend(batch)
        return len(batch)

    monkeypatch.setattr(builder, "_cache_metadata_batch", _capture_batch)

    result = builder._hydrate_exact_hydration_source_slice(
        use_streaming=False,
        source=source,
        row_limit=2,
        row_offset=100,
        progress_total=2,
        progress_label=f"Resuming {source}",
        operation="Incomplete hydration resume",
    )

    assert [record["paper_id"] for record in cached_records] == [
        "arxiv_100",
        "arxiv_101",
    ]
    assert result.hydrated_records == 2
    assert result.source_rows_consumed == 2
    assert result.source_exhausted is True


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
    assert builder.embedding_cache.mark_hydrated.call_args_list == [
        call(
            dataset_source=source,
            dataset_split=builder.dataset_split,
            corpus_size=builder.corpus_size,
            complete=False,
        ),
        call(
            dataset_source=source,
            dataset_split=builder.dataset_split,
            corpus_size=builder.corpus_size,
            complete=True,
        ),
    ]


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


def test_int8_calibration_covers_sample_extrema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Min/max calibration must cover every coordinate in its normalized sample.

    :param pytest.MonkeyPatch monkeypatch: Dependency and cache isolation.
    :param Any tmp_path: Temporary cache directory.
    :return None: Checks persisted extrema and absence of sample clipping.
    """
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
    sample_embeddings = np.tile(np.asarray([[0.6, -0.8]], dtype=np.float32), (101, 1))
    sample_embeddings[-1] = [-1.0, 0.0]

    def _fake_encode_texts(
        texts: list[str],
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Return normalized calibration vectors including one rare extreme.

        :param list[str] texts: Unused formatted sample texts.
        :param int | None batch_size: Unused encoder batch size.
        :param bool show_progress_bar: Unused progress flag.
        :return np.ndarray: Fixed FP32 sample.
        """
        del texts, batch_size, show_progress_bar
        return sample_embeddings

    monkeypatch.setattr(builder, "_encode_texts", _fake_encode_texts)
    builder._initialize_calibration_ranges(sample_records)

    with h5py.File(builder.embedding_cache.h5_path, "r") as h5:
        ranges = np.asarray(h5["calibration_ranges"], dtype=np.float32)

    np.testing.assert_array_equal(ranges[0], sample_embeddings.min(axis=0))
    np.testing.assert_array_equal(ranges[1], sample_embeddings.max(axis=0))
    assert np.all(sample_embeddings >= ranges[0])
    assert np.all(sample_embeddings <= ranges[1])


def test_int8_calibration_reuses_persisted_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resuming a cache must preserve even older percentile-based ranges.

    :param pytest.MonkeyPatch monkeypatch: Encoder isolation.
    :return None: Existing ranges survive without re-encoding calibration texts.
    """
    builder = EmbeddingGraphBuilder(semantic_source="arxiv-corpus", client=MagicMock())
    ranges = np.asarray([[-0.1, -0.2], [0.1, 0.2]], dtype=np.float32)
    builder.embedding_cache.set_calibration_ranges(ranges, embedding_dim=2)
    encode = MagicMock(side_effect=AssertionError("Unexpected recalibration"))
    monkeypatch.setattr(builder, "_encode_texts", encode)

    builder._initialize_calibration_ranges([{"title": "New calibration sample"}])

    encode.assert_not_called()
    with h5py.File(builder.embedding_cache.h5_path, "r") as h5:
        np.testing.assert_array_equal(h5["calibration_ranges"][:], ranges)


def test_hydration_flush_size_controls_cache_write_bursting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Hydration should flush metadata batches using configured flush threshold."""
    assert embedding_module.HYDRATION_FLUSH_SIZE == EMBEDDING_DATASET_CHUNK_ROWS
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
        "retrieval_representation": "retrieval-query/retrieval-document",
        "graph_representation": "graph-similarity",
        "model_profile": "embeddinggemma-v2",
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
    monkeypatch.setattr(builder, "prepare_graph_scoring", lambda _papers: None)

    graph, seed_id = builder.build_graph("seed")
    assert seed_id == "seed"
    assert graph.graph["embedding_runtime"] == {
        "binary_prefilter_used": True,
        "device": builder.device,
        "requested_device": "auto",
        "compute_dtype": builder._source_dtype_hint,
        "autocast": False,
        "retrieval_representation": "retrieval-query/retrieval-document",
        "graph_representation": "graph-similarity",
        "model_profile": "embeddinggemma-v2",
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


@pytest.mark.parametrize(
    ("mps_built", "expected_message"),
    [
        (False, "torch build has no MPS support"),
        (True, "built MPS backend is not available"),
    ],
)
def test_explicit_mps_reports_build_and_runtime_failures_separately(
    monkeypatch: pytest.MonkeyPatch,
    mps_built: bool,
    expected_message: str,
) -> None:
    """Explicit MPS diagnostics should distinguish build and machine support."""
    _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
        mps_available=False,
        mps_built=mps_built,
    )

    with pytest.raises(ValueError, match=expected_message):
        resolve_embedding_device("mps")


def test_mps_availability_implies_build_for_older_torch_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older torch APIs without ``is_built`` should trust live availability."""
    _bf16, _autocast, fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
        mps_available=True,
    )
    del fake_torch.backends.mps.is_built

    assert resolve_embedding_device("mps") == "mps"


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
        model_name="org/generic-embedding-model",
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
    assert "attn_implementation" not in init_log["kwargs"]["model_kwargs"]
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
    """Explicit compilation should run on each supported device."""
    for device_setup in [
        {"cuda_available": True, "mps_available": False},
        {"cuda_available": False, "mps_available": True},
        {"cuda_available": False, "mps_available": False},
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

        assert builder._inner_model_compiled
        assert fake_torch._compile_calls
        expected_kwargs: dict[str, Any] = (
            {} if builder.device == "mps" else {"dynamic": True}
        )
        if builder.device == "cpu":
            expected_kwargs["options"] = {"max_autotune": True}
        assert fake_torch._compile_calls[-1]["kwargs"] == expected_kwargs


@pytest.mark.parametrize("encode_path", ["direct", "retrieval", "graph", "hydration"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_lazy_compile_failure_restores_eager_model_and_retries_batch(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    encode_path: str,
    device: str,
) -> None:
    """Direct and cached encodes must recover from lazy compilation failure.

    :param pytest.MonkeyPatch monkeypatch: Isolated runtime patching fixture.
    :param pytest.LogCaptureFixture caplog: Captured fallback log messages.
    :param str encode_path: Direct or cache-writing encode entry point.
    :param str device: CPU or Metal execution policy.
    :return None: Checks successful eager recovery through each entry point.
    """
    _, _, fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
        mps_available=True,
        torch_version="2.13.0",
        compile_behavior="tagged",
    )
    builder = EmbeddingGraphBuilder(
        max_papers=1,
        device=device,
        enable_torch_compile=True,
        client=MagicMock(),
    )
    _pin_model_fingerprint(monkeypatch, builder)
    compiler_config = fake_torch._inductor.config
    compiler_config.freezing_discard_parameters = True
    compiler_states: list[tuple[bool, bool]] = []
    original_inner = object()

    class _InnerBlock:
        def __init__(self) -> None:
            self.model = original_inner

    class _LazyFailureModel:
        def __init__(self) -> None:
            self.block = _InnerBlock()
            self.encode_attempts = 0

        def __getitem__(self, index: int) -> _InnerBlock:
            """Return the single fake transformer block.

            :param int index: Required zero index.
            :return _InnerBlock: Fake transformer block.
            """
            assert index == 0
            return self.block

        def encode(self, texts: list[str], **_kwargs: Any) -> np.ndarray:
            """Fail while wrapped, then succeed after eager restoration.

            :param list[str] texts: Text batch.
            :param Any _kwargs: Ignored SentenceTransformer encode controls.
            :return np.ndarray: One float32 row per input.
            """
            self.encode_attempts += 1
            compiler_states.append(
                (compiler_config.freezing, compiler_config.freezing_discard_parameters)
            )
            if isinstance(self.block.model, tuple):
                raise RuntimeError("Inductor codegen failed")
            return np.ones((len(texts), 2), dtype=np.float32)

    model = _LazyFailureModel()
    builder.model = model
    builder._maybe_compile_inner_transformer()

    assert builder._inner_model_compiled is True
    assert isinstance(model.block.model, tuple)
    with caplog.at_level(logging.WARNING):
        papers = {"seed": Paper(paper_id="seed", title="Seed", year=2024)}
        if encode_path == "direct":
            embeddings = builder._encode_texts(["seed"])
        elif encode_path == "retrieval":
            embeddings = np.asarray(list(builder.embed_papers(papers).values()))
        elif encode_path == "graph":
            embeddings = np.asarray(
                list(builder.materialize_graph_embeddings(papers).values())
            )
        else:
            builder._ensure_cache_model_fingerprint()
            assert (
                builder._cache_metadata_batch([{"paper_id": "seed", "title": "Seed"}])
                == 1
            )
            embeddings = np.asarray(list(builder.embed_papers(papers).values()))

    assert embeddings.shape == (1, 2)
    assert model.encode_attempts == 2
    assert model.block.model is original_inner
    assert builder._inner_model_compiled is False
    assert compiler_states == [
        (True, False) if device == "cpu" else (False, True),
        (False, True),
    ]
    assert compiler_config.freezing is False
    assert compiler_config.freezing_discard_parameters is True
    assert "restored eager model" in str(builder._compile_status_reason)
    assert any(
        "retrying the affected encode batch" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_embedding_tf32_skipped_for_non_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Explicit non-CUDA devices must not touch global CUDA precision controls.

    :param pytest.MonkeyPatch monkeypatch: Isolated runtime patching fixture.
    :param str device: Explicit non-CUDA device to exercise.
    :return None: Checks TF32 stays off through loading and encoding.
    """
    _install_fake_sentence_transformers(monkeypatch)
    _bf16_token, _autocast_log, fake_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        mps_available=True,
        bf16_supported=True,
        torch_version="2.13.0",
    )
    builder = EmbeddingGraphBuilder(max_papers=1, device=device, client=MagicMock())
    builder._load_model()
    builder._encode_texts(["seed"])

    assert builder.device == device
    assert builder._tf32_mode == "off"
    assert fake_torch.backends.fp32_precision == "none"


@pytest.mark.parametrize("cpu_bf16", [False, True])
def test_embedding_cache_namespace_stable_across_device_for_same_dtype(
    monkeypatch: pytest.MonkeyPatch,
    cpu_bf16: bool,
) -> None:
    """Matching BF16 compute shares a namespace; CPU FP32 remains distinct.

    :param pytest.MonkeyPatch monkeypatch: Isolated runtime patching fixture.
    :param bool cpu_bf16: Whether CPU hardware reports native BF16 support.
    :return None: Checks cross-device cache identity for each CPU compute mode.
    """
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

    _, _, cpu_torch = _install_fake_torch(
        monkeypatch,
        cuda_available=False,
        bf16_supported=False,
        mps_available=False,
        torch_version="2.13.0",
    )
    cpu_torch.cpu = types.SimpleNamespace(
        get_capabilities=lambda: {"avx512_bf16": cpu_bf16}
    )
    cpu_builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())

    assert cuda_builder.truncate_dim == mps_builder.truncate_dim == 512
    assert cpu_builder.truncate_dim == 512
    assert cuda_builder._source_dtype_hint == "bfloat16"
    assert mps_builder._source_dtype_hint == "bfloat16"
    assert cpu_builder._source_dtype_hint == ("bfloat16" if cpu_bf16 else "float32")
    assert (
        cuda_builder._embedding_cache_namespace()
        == mps_builder._embedding_cache_namespace()
    )
    assert (
        cpu_builder._embedding_cache_namespace()
        == cuda_builder._embedding_cache_namespace()
    ) is cpu_bf16


@pytest.mark.slow
@pytest.mark.parametrize("compute_dtype", ["float32", "bfloat16"])
def test_embedding_real_cpu_compile_executes_graphs(
    monkeypatch: pytest.MonkeyPatch,
    compute_dtype: str,
) -> None:
    """Compare FP32 or native BF16 eager and compiled CPU inference.

    :param pytest.MonkeyPatch monkeypatch: Isolated CPU BF16 capability override.
    :param str compute_dtype: Real FP32 fallback or native BF16 execution policy.
    :return None: Checks actual compiled profiler regions and FP32 unit vectors.
    """
    torch = pytest.importorskip("torch")
    from torch._inductor import config as inductor_config

    if compute_dtype == "bfloat16" and not embedding_module._cpu_native_bf16_supported(
        torch
    ):
        pytest.skip("Native CPU BF16 unavailable")
    if compute_dtype == "float32":
        monkeypatch.setattr(
            embedding_module, "_cpu_native_bf16_supported", lambda _: False
        )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        builder = EmbeddingGraphBuilder(device="cpu", client=MagicMock())
        builder._load_model()
        assert builder._autocast_enabled is (compute_dtype == "bfloat16")
        assert builder._source_dtype_hint == compute_dtype
        assert builder._tf32_mode == "off"
        texts = [
            builder.model_profile.format_document(
                {"title": "Research", "abstract": abstract * repetitions}
            )
            for repetitions in range(1, 5)
            for abstract in (
                "Bidirectional attention encodes text for semantic retrieval. ",
                "Stellar spectra constrain the composition of distant galaxies. ",
            )
        ]
        eager = builder._encode_texts(texts, batch_size=8)
        builder.enable_torch_compile = True
        builder._maybe_compile_inner_transformer()
        with inductor_config.patch(freezing=False, freezing_discard_parameters=True):
            builder._encode_texts(texts, batch_size=8)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profile:
                compiled = builder._encode_texts(texts, batch_size=8)
            assert inductor_config.freezing is False
            assert inductor_config.freezing_discard_parameters is True
        assert builder._inner_model_compiled
        assert any(
            "Torch-Compiled Region" in event.key for event in profile.key_averages()
        )
        for embeddings in (eager, compiled):
            assert embeddings.dtype == np.float32
            assert np.isfinite(embeddings).all()
            np.testing.assert_allclose(
                np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6
            )
        assert np.min(np.sum(eager * compiled, axis=1)) > 0.999
        assert builder._restore_eager_model_after_compile_failure(
            RuntimeError("exercise fallback after weight packing")
        )
        restored = builder._encode_texts(texts, batch_size=8)
        np.testing.assert_allclose(restored, eager, atol=1e-6)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.slow
def test_embedding_real_cuda_fa2_compile_executes_graphs() -> None:
    """Exercise the real ST6 compiled forward path and compare eager embeddings.

    :return None: Confirms FA2, compiled profiler events, numerical agreement,
        and restoration of scoped compiler settings on an available CUDA host.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(
        including_emulation=False
    ):
        pytest.skip("Native BF16 CUDA unavailable")
    pytest.importorskip("flash_attn")
    from torch._dynamo import config as dynamo_config
    from transformers import modeling_flash_attention_utils as fa2_utils

    builder = EmbeddingGraphBuilder(device="cuda", client=MagicMock())
    builder._load_model()
    assert builder.model[0].model.config._attn_implementation == "flash_attention_2"
    texts = [
        builder.model_profile.format_document(
            {
                "title": title,
                "abstract": abstract * repetitions,
            }
        )
        for repetitions in range(1, 5)
        for title, abstract in (
            (
                "Language model attention",
                "Bidirectional attention encodes text for semantic retrieval. ",
            ),
            (
                "Stellar evolution",
                "Stellar spectra constrain the chemical composition of distant galaxies. ",
            ),
        )
    ]
    lazy_import_fa2 = fa2_utils.lazy_import_flash_attention
    fa2_kernel_input_dtypes: list[tuple[torch.dtype, torch.dtype, torch.dtype]] = []

    def capture_fa2_kernels(attention_backend: str) -> Any:
        """Wrap both dense and variable-length FA2 kernels for dtype capture.

        :param str attention_backend: Requested attention implementation.
        :return Any: Wrapped kernel tuple and original kwarg processor.
        """
        kernels, process_kwargs = lazy_import_fa2(attention_backend)
        dense_kernel, varlen_kernel, pad_fn, unpad_fn = kernels

        def wrap_kernel(kernel: Any) -> Any:
            """Wrap one FA2 kernel and record its Q/K/V dtypes.

            :param Any kernel: Dense or variable-length FA2 callable.
            :return Any: Dtype-recording kernel wrapper.
            """

            def capture(
                query: Any, key: Any, value: Any, *args: Any, **kwargs: Any
            ) -> Any:
                """Record inputs and invoke the original kernel.

                :param Any query: Query tensor.
                :param Any key: Key tensor.
                :param Any value: Value tensor.
                :param Any args: Additional positional kernel arguments.
                :param Any kwargs: Additional keyword kernel arguments.
                :return Any: Original kernel result.
                """
                fa2_kernel_input_dtypes.append((query.dtype, key.dtype, value.dtype))
                return kernel(query, key, value, *args, **kwargs)

            return capture

        return (
            (wrap_kernel(dense_kernel), wrap_kernel(varlen_kernel), pad_fn, unpad_fn),
            process_kwargs,
        )

    fa2_utils.lazy_import_flash_attention = capture_fa2_kernels
    try:
        eager = builder._encode_texts(texts, batch_size=8)
    finally:
        fa2_utils.lazy_import_flash_attention = lazy_import_fa2
    assert fa2_kernel_input_dtypes
    assert all(
        query_dtype == key_dtype == value_dtype == torch.bfloat16
        for query_dtype, key_dtype, value_dtype in fa2_kernel_input_dtypes
    )
    logging_option = (
        "ignore_logging_functions"
        if hasattr(dynamo_config, "ignore_logging_functions")
        else "reorderable_logging_functions"
    )
    prior_logging = set(getattr(dynamo_config, logging_option))
    builder.enable_torch_compile = True
    builder._maybe_compile_inner_transformer()
    builder._encode_texts(texts, batch_size=8)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        compiled = builder._encode_texts(texts, batch_size=8)
    assert builder._inner_model_compiled
    assert any("Torch-Compiled Region" in event.key for event in profile.key_averages())
    assert compiled.dtype == np.float32
    assert np.isfinite(compiled).all()
    assert np.min(np.sum(eager * compiled, axis=1)) > 0.999
    assert getattr(dynamo_config, logging_option) == prior_logging


@pytest.mark.slow
def test_embedding_real_mps_task_space_quality_smoke() -> None:
    """Validate frozen retrieval and symmetric-task behavior on real MPS.

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

    retrieval_cases = [
        (
            Paper(
                paper_id="transformer-seed",
                title="Transformer language models",
                year=2017,
                abstract=(
                    "A sequence transduction architecture based entirely on "
                    "self-attention for machine translation."
                ),
            ),
            [
                Paper(
                    paper_id="attention",
                    title="Attention Is All You Need",
                    year=2017,
                    abstract=(
                        "The Transformer replaces recurrence with multi-head "
                        "self-attention for sequence modeling."
                    ),
                ),
                Paper(
                    paper_id="bert",
                    title="BERT",
                    year=2018,
                    abstract=(
                        "Bidirectional Transformer pre-training learns deep "
                        "language representations."
                    ),
                ),
                Paper(
                    paper_id="alphazero",
                    title="Mastering Chess and Shogi by Self-Play",
                    year=2017,
                    abstract="A reinforcement learning system for board games.",
                ),
                Paper(
                    paper_id="mask-rcnn",
                    title="Mask R-CNN",
                    year=2017,
                    abstract="An object detection and instance segmentation model.",
                ),
            ],
            {"attention", "bert"},
        ),
        (
            Paper(
                paper_id="rag-seed",
                title="Retrieval-augmented language generation",
                year=2020,
                abstract=(
                    "Dense passage retrieval supplies external documents to a "
                    "neural text generator."
                ),
            ),
            [
                Paper(
                    paper_id="rag",
                    title="Retrieval-Augmented Generation",
                    year=2020,
                    abstract=(
                        "A language model conditions generation on passages from "
                        "a dense neural retriever."
                    ),
                ),
                Paper(
                    paper_id="dpr",
                    title="Dense Passage Retrieval",
                    year=2020,
                    abstract=(
                        "Dual encoders retrieve relevant passages for open-domain "
                        "question answering."
                    ),
                ),
                Paper(
                    paper_id="nerf",
                    title="Neural Radiance Fields",
                    year=2020,
                    abstract="A neural representation for novel view synthesis.",
                ),
                Paper(
                    paper_id="ddpm",
                    title="Denoising Diffusion Probabilistic Models",
                    year=2020,
                    abstract="A generative model based on iterative denoising.",
                ),
            ],
            {"rag", "dpr"},
        ),
    ]

    recalls: list[float] = []
    ndcgs: list[float] = []
    for seed, candidates, relevant_ids in retrieval_cases:
        query_text = format_paper_for_embedding(
            profile=builder.model_profile,
            paper=seed,
            task=EmbeddingTask.RETRIEVAL_QUERY,
        )
        document_texts = [
            format_paper_for_embedding(
                profile=builder.model_profile,
                paper=paper,
                task=EmbeddingTask.RETRIEVAL_DOCUMENT,
            )
            for paper in candidates
        ]
        vectors = builder._encode_texts([query_text, *document_texts])
        scores = vectors[1:] @ vectors[0]
        ranking = np.argsort(-scores)
        ranked_ids = [candidates[int(index)].paper_id for index in ranking]
        recalls.append(len(set(ranked_ids[:2]) & relevant_ids) / len(relevant_ids))
        gains = np.asarray(
            [1.0 if paper_id in relevant_ids else 0.0 for paper_id in ranked_ids]
        )
        discounts = 1.0 / np.log2(np.arange(2, len(ranked_ids) + 2))
        dcg = float(np.sum(gains * discounts))
        ideal_dcg = float(np.sum(np.sort(gains)[::-1] * discounts))
        ndcgs.append(dcg / ideal_dcg)

    assert float(np.mean(recalls)) >= 0.75
    assert float(np.mean(ndcgs)) >= 0.75

    related_pairs = [
        (retrieval_cases[0][1][0], retrieval_cases[0][1][1]),
        (retrieval_cases[1][1][0], retrieval_cases[1][1][1]),
    ]
    unrelated_pairs = [
        (retrieval_cases[0][1][0], retrieval_cases[1][1][2]),
        (retrieval_cases[1][1][0], retrieval_cases[0][1][2]),
    ]
    pair_papers = [
        paper for pair in [*related_pairs, *unrelated_pairs] for paper in pair
    ]
    similarity_texts = [
        format_paper_for_embedding(
            profile=builder.model_profile,
            paper=paper,
            task=EmbeddingTask.GRAPH_SIMILARITY,
        )
        for paper in pair_papers
    ]
    similarity_vectors = builder._encode_texts(similarity_texts)
    pair_scores = np.sum(similarity_vectors[0::2] * similarity_vectors[1::2], axis=1)
    related_mean = float(np.mean(pair_scores[: len(related_pairs)]))
    unrelated_mean = float(np.mean(pair_scores[len(related_pairs) :]))

    assert similarity_vectors.shape == (8, builder.truncate_dim)
    assert np.isfinite(similarity_vectors).all()
    assert related_mean >= unrelated_mean + 0.05
