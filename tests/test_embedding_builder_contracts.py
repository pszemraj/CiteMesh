"""Contract tests for embedding builder precision and hydration behavior."""

from __future__ import annotations

import logging
import sys
import types
from hashlib import sha256
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from citemesh.data.embedding_cache import CacheSearchResult
from citemesh.strategies.embedding import EmbeddingGraphBuilder, _query_seed_id
from tests._helpers import ConstantEncodeModel


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
        EmbeddingGraphBuilder(max_papers=5, model_name="test-model", client=MagicMock())


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


def test_embedding_cache_namespace_ignores_binary_prefilter_outside_int8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-int8 caches should not split namespace by binary prefilter toggles."""
    _disable_embedding_dep_check(monkeypatch)

    f32_prefilter_on = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="float32",
        binary_prefilter=True,
        client=MagicMock(),
    )
    f32_prefilter_off = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="float32",
        binary_prefilter=False,
        client=MagicMock(),
    )
    int8_prefilter_on = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="int8",
        binary_prefilter=True,
        client=MagicMock(),
    )
    int8_prefilter_off = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="int8",
        binary_prefilter=False,
        client=MagicMock(),
    )

    assert (
        f32_prefilter_on.embedding_cache.model_name
        == f32_prefilter_off.embedding_cache.model_name
    )
    assert f32_prefilter_on.binary_prefilter is False
    assert f32_prefilter_off.binary_prefilter is False
    assert f32_prefilter_on.binary_rescore_multiplier == 1
    assert f32_prefilter_off.binary_rescore_multiplier == 1
    assert (
        int8_prefilter_on.embedding_cache.model_name
        != int8_prefilter_off.embedding_cache.model_name
    )


def test_embedding_cache_namespace_varies_by_source_dtype_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source dtype hints should participate in namespace identity."""
    _disable_embedding_dep_check(monkeypatch)

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


def test_embedding_cache_namespace_includes_int8_calibration_sample_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Int8 caches should include calibration sample size in namespace identity."""
    _disable_embedding_dep_check(monkeypatch)
    monkeypatch.setattr(
        EmbeddingGraphBuilder,
        "_resolve_source_dtype_hint",
        lambda self: "float32",
    )

    int8_small = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="int8",
        calibration_sample_size=32,
        client=MagicMock(),
    )
    int8_large = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="int8",
        calibration_sample_size=128,
        client=MagicMock(),
    )
    f32_small = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="float32",
        calibration_sample_size=32,
        client=MagicMock(),
    )
    f32_large = EmbeddingGraphBuilder(
        max_papers=1,
        storage_precision="float32",
        calibration_sample_size=128,
        client=MagicMock(),
    )

    assert (
        int8_small.embedding_cache.model_name != int8_large.embedding_cache.model_name
    )
    assert f32_small.embedding_cache.model_name == f32_large.embedding_cache.model_name


def test_embedding_cache_namespace_matches_default_and_explicit_truncate_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default resolved truncate dim should match explicit equivalent namespace."""
    _disable_embedding_dep_check(monkeypatch)
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
    _disable_embedding_dep_check(monkeypatch)
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
    _disable_embedding_dep_check(monkeypatch)
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


def test_embedding_cache_reuses_cached_fingerprint_when_lookup_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Offline fingerprint lookup failures should not block reuse of a hydrated cache."""
    _disable_embedding_dep_check(monkeypatch)

    builder = EmbeddingGraphBuilder(
        max_papers=1, model_name="org/offline-test", client=MagicMock()
    )
    builder.embedding_cache.has_cached_payload = MagicMock(return_value=True)
    builder.embedding_cache.get_model_fingerprint = MagicMock(
        return_value="hf::org/offline-test::0123456789abcdef0123456789abcdef01234567"
    )
    builder.embedding_cache.clear = MagicMock()
    builder.embedding_cache.set_model_fingerprint = MagicMock()
    builder._resolve_model_fingerprint = MagicMock(
        side_effect=RuntimeError("network unavailable")
    )

    with caplog.at_level(logging.WARNING):
        builder._ensure_cache_model_fingerprint()

    assert (
        builder._resolved_model_fingerprint
        == "hf::org/offline-test::0123456789abcdef0123456789abcdef01234567"
    )
    assert builder.embedding_cache.clear.call_count == 0
    builder.embedding_cache.set_model_fingerprint.assert_not_called()
    assert any(
        "Reusing compatible cached fingerprint" in record.getMessage()
        for record in caplog.records
    )


def test_embedding_cache_clears_stale_cached_fingerprint_when_lookup_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Offline fallback should clear payload when cached fingerprint is incompatible."""
    _disable_embedding_dep_check(monkeypatch)

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/offline-test",
        model_revision="refs/pr/12",
        client=MagicMock(),
    )
    builder.embedding_cache.has_cached_payload = MagicMock(return_value=True)
    builder.embedding_cache.get_model_fingerprint = MagicMock(
        return_value="hf::org/offline-test::0123456789abcdef0123456789abcdef01234567"
    )
    builder.embedding_cache.clear = MagicMock()
    builder.embedding_cache.set_model_fingerprint = MagicMock()
    builder._resolve_model_fingerprint = MagicMock(
        side_effect=RuntimeError("network unavailable")
    )

    with caplog.at_level(logging.WARNING):
        builder._ensure_cache_model_fingerprint()

    assert (
        builder._resolved_model_fingerprint
        == "hf::org/offline-test::revision=refs/pr/12::offline-unverified"
    )
    builder.embedding_cache.clear.assert_called_once()
    builder.embedding_cache.set_model_fingerprint.assert_called_once_with(
        "hf::org/offline-test::revision=refs/pr/12::offline-unverified"
    )
    assert any(
        "is incompatible with requested identity" in record.getMessage()
        for record in caplog.records
    )


def test_embedding_cache_reuses_cached_payload_with_fallback_fingerprint_when_lookup_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Old caches without stored fingerprint should still be reusable offline."""
    _disable_embedding_dep_check(monkeypatch)

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/offline-no-fingerprint",
        model_revision="refs/pr/12",
        client=MagicMock(),
    )
    builder.embedding_cache.has_cached_payload = MagicMock(return_value=True)
    builder.embedding_cache.get_model_fingerprint = MagicMock(return_value=None)
    builder.embedding_cache.clear = MagicMock()
    builder.embedding_cache.set_model_fingerprint = MagicMock()
    builder._resolve_model_fingerprint = MagicMock(
        side_effect=RuntimeError("network unavailable")
    )

    with caplog.at_level(logging.WARNING):
        builder._ensure_cache_model_fingerprint()

    assert (
        builder._resolved_model_fingerprint
        == "hf::org/offline-no-fingerprint::revision=refs/pr/12::offline-unverified"
    )
    assert builder.embedding_cache.clear.call_count == 0
    builder.embedding_cache.set_model_fingerprint.assert_called_once_with(
        "hf::org/offline-no-fingerprint::revision=refs/pr/12::offline-unverified"
    )
    assert any(
        "Reusing cached payload with fallback identity" in record.getMessage()
        for record in caplog.records
    )


def test_embedding_cache_initializes_offline_without_payload(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No-payload namespaces should initialize fingerprint metadata in offline mode."""
    _disable_embedding_dep_check(monkeypatch)

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/offline-init",
        model_revision="refs/pr/34",
        client=MagicMock(),
    )
    builder.embedding_cache.has_cached_payload = MagicMock(return_value=False)
    builder.embedding_cache.get_model_fingerprint = MagicMock(return_value=None)
    builder.embedding_cache.set_model_fingerprint = MagicMock()
    builder._resolve_model_fingerprint = MagicMock(
        side_effect=RuntimeError("network unavailable")
    )

    with caplog.at_level(logging.WARNING):
        builder._ensure_cache_model_fingerprint()

    assert (
        builder._resolved_model_fingerprint
        == "hf::org/offline-init::revision=refs/pr/34::offline-unverified"
    )
    builder.embedding_cache.set_model_fingerprint.assert_called_once_with(
        "hf::org/offline-init::revision=refs/pr/34::offline-unverified"
    )
    assert any(
        "offline initialization" in record.getMessage() for record in caplog.records
    )


def test_embedding_cache_sets_missing_cached_fingerprint_after_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached payload without fingerprint should be migrated to a resolved fingerprint."""
    _disable_embedding_dep_check(monkeypatch)

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


def test_embedding_fingerprint_resolution_fails_closed_for_hf_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF-backed model fingerprints should fail closed when SHA cannot be resolved."""
    _disable_embedding_dep_check(monkeypatch)

    class _FailingHfApi:
        def model_info(self, repo_id: str, revision: str) -> object:
            del repo_id, revision
            raise RuntimeError("network unavailable")

    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.HfApi = _FailingHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/test-model",
        model_revision="main",
        client=MagicMock(),
    )

    with pytest.raises(RuntimeError, match="Could not resolve Hugging Face commit SHA"):
        builder._resolve_model_fingerprint()


def test_embedding_fingerprint_resolution_uses_local_snapshot_sha_when_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF fingerprint resolution should fall back to local snapshot SHA without API."""
    _disable_embedding_dep_check(monkeypatch)

    class _FailingHfApi:
        def model_info(self, repo_id: str, revision: str) -> object:
            del repo_id, revision
            raise RuntimeError("network unavailable")

    def _snapshot_download(repo_id: str, revision: str, local_files_only: bool) -> str:
        del repo_id, revision
        assert local_files_only is True
        return "/tmp/models--org--test-model/snapshots/0123456789abcdef0123456789abcdef01234567"

    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.HfApi = _FailingHfApi
    fake_hf_module.snapshot_download = _snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/test-model",
        model_revision="refs/pr/12",
        client=MagicMock(),
    )

    assert (
        builder._resolve_model_fingerprint()
        == "hf::org/test-model::0123456789abcdef0123456789abcdef01234567"
    )


def test_embedding_fingerprint_resolution_uses_local_artifact_hashes_when_sha_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """HF fingerprint resolution should hash config + weights when SHA is unavailable."""
    _disable_embedding_dep_check(monkeypatch)

    class _FailingHfApi:
        def model_info(self, repo_id: str, revision: str) -> object:
            del repo_id, revision
            raise RuntimeError("network unavailable")

    snapshot_root = tmp_path / "models--org--artifact-model"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    config_bytes = b'{"architectures":["FakeModel"]}\n'
    weights_bytes = b"weights-v1"
    (snapshot_root / "config.json").write_bytes(config_bytes)
    (snapshot_root / "model.safetensors").write_bytes(weights_bytes)

    def _snapshot_download(repo_id: str, revision: str, local_files_only: bool) -> str:
        assert repo_id == "org/artifact-model"
        assert revision == "refs/pr/7"
        assert local_files_only is True
        return str(snapshot_root)

    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.HfApi = _FailingHfApi
    fake_hf_module.snapshot_download = _snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        model_name="org/artifact-model",
        model_revision="refs/pr/7",
        client=MagicMock(),
    )

    expected_config = sha256(config_bytes).hexdigest()
    expected_weights = sha256(weights_bytes).hexdigest()
    assert builder._resolve_model_fingerprint() == (
        "hf::org/artifact-model::revision=refs/pr/7"
        f"::config={expected_config}::weights={expected_weights}"
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
        max_papers=1, use_streaming=True, client=MagicMock()
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

    candidates = builder._select_candidates_from_loaded(
        np.asarray([1.0, 0.0], dtype=np.float32)
    )
    assert [paper_id for paper_id, _, _ in candidates] == ["a", "b"]
    fake_load_dataset_for_hydration.assert_not_called()


def test_collect_papers_revalidates_cache_when_dataset_source_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Cached hydration should be revalidated against the selected dataset source."""
    _disable_embedding_dep_check(monkeypatch)
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path / "cache-root"))

    builder = EmbeddingGraphBuilder(
        max_papers=2, use_streaming=False, client=MagicMock()
    )
    _pin_model_fingerprint(monkeypatch, builder)
    builder.embedding_cache.mark_hydrated(
        dataset_source="librarian-bots/arxiv-metadata-snapshot",
        dataset_split=builder.dataset_split,
        corpus_size=builder.corpus_size,
        complete=True,
    )

    mark_hydrated_spy = MagicMock(wraps=builder.embedding_cache.mark_hydrated)
    monkeypatch.setattr(builder.embedding_cache, "mark_hydrated", mark_hydrated_spy)

    loaded_sources: list[tuple[bool, str | None]] = []

    def fake_load_dataset_for_hydration(
        use_streaming: bool, preferred_dataset_source: str | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        loaded_sources.append((use_streaming, preferred_dataset_source))
        return "CShorten/ML-ArXiv-Papers", []

    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        fake_load_dataset_for_hydration,
    )
    monkeypatch.setattr(
        builder, "_get_model_for_encoding", lambda: ConstantEncodeModel()
    )

    candidates = builder._select_candidates_from_loaded(
        np.asarray([1.0, 0.0], dtype=np.float32)
    )

    assert candidates == []
    assert loaded_sources == [(False, "librarian-bots/arxiv-metadata-snapshot")]

    complete_flags = [
        call.kwargs["complete"] for call in mark_hydrated_spy.call_args_list
    ]
    assert complete_flags == [False]


def test_collect_papers_rejects_source_mismatch_when_dataset_load_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Offline fallback must not reuse cache from a different dataset source."""
    _disable_embedding_dep_check(monkeypatch)
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path / "cache-root"))

    builder = EmbeddingGraphBuilder(
        max_papers=2, use_streaming=False, client=MagicMock()
    )
    _pin_model_fingerprint(monkeypatch, builder)
    builder.embedding_cache.mark_hydrated(
        dataset_source="librarian-bots/arxiv-metadata-snapshot",
        dataset_split=builder.dataset_split,
        corpus_size=builder.corpus_size,
        complete=True,
    )

    is_hydrated_spy = MagicMock(wraps=builder.embedding_cache.is_hydrated)
    monkeypatch.setattr(builder.embedding_cache, "is_hydrated", is_hydrated_spy)
    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        MagicMock(side_effect=RuntimeError("dataset unavailable")),
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to resolve hydration dataset source",
    ) as exc_info:
        builder._select_candidates_from_loaded(np.asarray([1.0, 0.0], dtype=np.float32))

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "dataset unavailable"
    assert is_hydrated_spy.call_count == 1
    assert (
        is_hydrated_spy.call_args.kwargs.get("dataset_source")
        == "librarian-bots/arxiv-metadata-snapshot"
    )


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
    _pin_model_fingerprint(monkeypatch, builder)
    mark_hydrated_spy = MagicMock(wraps=builder.embedding_cache.mark_hydrated)
    monkeypatch.setattr(builder.embedding_cache, "mark_hydrated", mark_hydrated_spy)
    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        lambda use_streaming, preferred_dataset_source=None: ("empty-snapshot", []),
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

    candidates = builder._select_candidates_from_loaded(
        np.asarray([1.0, 0.0], dtype=np.float32)
    )
    assert [paper_id for paper_id, _, _ in candidates] == ["a", "b"]
