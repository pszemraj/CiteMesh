"""Torch/Transformers runtime probes and device policy for embedding builds.

Owns the version floors and capability probes the embedding strategy uses to
decide *where* and *in what precision* a model may run: torch/Transformers
version parsing, accelerator and bf16 capability detection, the requested-device
resolver, the backend/precision compatibility errors, and the two scoped
warning/progress suppressors used around model loading.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import warnings
from contextlib import contextmanager
from importlib import metadata as importlib_metadata
from typing import (
    Any,
    Callable,
    Iterator,
    Optional,
)

from citemesh._runtime import stderr_isatty
from citemesh.data.model_profiles import EmbeddingModelProfile

from . import deps

logger = logging.getLogger(__name__)


_FA2_LOAD_DTYPE_WARNING_PREFIX = (
    "Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes"
)

_FA2_FORWARD_DTYPE_WARNING = (
    "Casting fp32 inputs back to torch.bfloat16 for flash-attn compatibility."
)

_EMBEDDING_MIN_TORCH_VERSION = (2, 9)
# bf16-on-MPS is only enabled on torch releases verified on Apple Silicon; this is
# a policy floor, not a hard technical cliff — lower it once older wheels are vetted.
_MPS_MIN_TORCH_VERSION = (2, 13)
EMBEDDING_DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")
_COMPILE_ELIGIBLE_DEVICES = frozenset({"cuda", "mps", "cpu"})
# Legacy release-branch workaround window where Inductor conflicted with the
# fp32_precision TF32 API; later torch releases use the modern API directly.
_TF32_COMPILE_BRIDGE_TORCH_VERSIONS = frozenset({(2, 9), (2, 10)})
_INFERENCE_ARTIFACT_FILENAMES = frozenset(
    {
        "added_tokens.json",
        "cnn_config.json",
        "config.json",
        "config_sentence_transformers.json",
        "lstm_config.json",
        "merges.txt",
        "phrasetokenizer_config.json",
        "router_config.json",
        "sentence_albert_config.json",
        "sentence_bert_config.json",
        "sentence_camembert_config.json",
        "sentence_distilbert_config.json",
        "sentence_roberta_config.json",
        "sentence_xlm-roberta_config.json",
        "sentence_xlnet_config.json",
        "sentencepiece.bpe.model",
        "sentencepiece.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
        "whitespacetokenizer_config.json",
        "wordembedding_config.json",
    }
)
_TORCH_WEIGHT_LAYOUTS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


class EmbeddingBackendCompatibilityError(RuntimeError):
    """The active embedding model requires a newer inference backend."""


class EmbeddingPrecisionCompatibilityError(RuntimeError):
    """The checkpoint's automatic weight dtype violates runtime precision policy."""


@contextmanager
def _suppress_expected_fa2_load_dtype_warning(*, enabled: bool) -> Iterator[None]:
    """Suppress Transformers' redundant FA2 warning for verified bf16 autocast.

    Transformers validates the checkpoint weight dtype before the first forward
    pass, so automatic FP32 weights trigger a warning even though its FA2 adapter
    uses the active CUDA autocast dtype for attention inputs.

    :param bool enabled: Whether verified CUDA bf16 autocast and FA2 are active.
    :return Iterator[None]: Scoped logging filter context.
    """
    if not enabled:
        yield
        return

    def keep_relevant_warning(record: logging.LogRecord) -> bool:
        """Reject only the expected FP32 checkpoint warning.

        :param logging.LogRecord record: Candidate Transformers log record.
        :return bool: Whether the log record should be emitted.
        """
        message = record.getMessage()
        return not (
            message.startswith(_FA2_LOAD_DTYPE_WARNING_PREFIX)
            and " is torch.float32." in message
        )

    transformers_logger = logging.getLogger("transformers.modeling_utils")
    transformers_logger.addFilter(keep_relevant_warning)
    try:
        yield
    finally:
        transformers_logger.removeFilter(keep_relevant_warning)


@contextmanager
def _suppress_transformers_progress_for_non_tty() -> Iterator[None]:
    """Temporarily hide Transformers progress bars when stderr is redirected.

    :return Iterator[None]: Scoped model-loading output configuration.
    """
    if stderr_isatty():
        yield
        return

    from transformers.utils import logging as transformers_logging

    if not transformers_logging.is_progress_bar_enabled():
        yield
        return

    transformers_logging.disable_progress_bar()
    try:
        yield
    finally:
        transformers_logging.enable_progress_bar()


def _parse_major_minor(
    version: object, *, default: Optional[tuple[int, int]] = None
) -> Optional[tuple[int, int]]:
    """Parse a leading semantic-version major/minor pair.

    Callers that compare against a version floor pass ``default=(0, 0)`` so an
    unparseable version sorts below every floor; callers that must distinguish
    "older" from "unknown" keep the ``None`` default.

    :param object version: Raw package version string.
    :param Optional[tuple[int, int]] default: Value returned on a parse miss.
    :return Optional[tuple[int, int]]: Parsed pair, or ``default`` when unavailable.
    """
    version_match = re.match(r"^(\d+)\.(\d+)", str(version).strip())
    if version_match is None:
        return default
    return int(version_match.group(1)), int(version_match.group(2))


def _installed_transformers_major_minor(transformers: Any) -> Optional[tuple[int, int]]:
    """Resolve the installed Transformers version from module or distribution data.

    Editable and development builds sometimes expose a nonstandard module
    ``__version__`` even though their installed distribution metadata is valid.

    :param Any transformers: Imported Transformers module.
    :return Optional[tuple[int, int]]: Verified major/minor pair when discoverable.
    """
    module_version = _parse_major_minor(getattr(transformers, "__version__", ""))
    if module_version is not None:
        return module_version
    try:
        distribution_version = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError:
        return None
    return _parse_major_minor(distribution_version)


def _require_transformers_compatibility(profile: EmbeddingModelProfile) -> None:
    """Require the Transformers floor declared by an embedding model profile.

    :param EmbeddingModelProfile profile: Runtime-active model contract.
    :return None: The installed backend satisfies the model contract.
    :raises EmbeddingBackendCompatibilityError: If Transformers is too old.
    """
    minimum = profile.minimum_transformers_version
    if minimum is None:
        return
    required = ".".join(str(part) for part in minimum)

    transformers = importlib.import_module("transformers")
    raw_version = str(getattr(transformers, "__version__", "")).strip()
    detected = _installed_transformers_major_minor(transformers)
    if detected is None:
        raise EmbeddingBackendCompatibilityError(
            f"Could not determine the installed Transformers version for {profile.name}; "
            "the bidirectional-attention compatibility floor cannot be verified. "
            f"Reinstall a supported transformers>={required} release."
        )
    if detected < minimum:
        detected_label = (
            raw_version
            if _parse_major_minor(raw_version) is not None
            else ".".join(str(part) for part in detected)
        )
        raise EmbeddingBackendCompatibilityError(
            f"{profile.name} requires transformers>={required} because older "
            "Gemma 3 implementations ignore bidirectional attention. "
            f"Detected transformers=={detected_label}."
        )


def _accelerator_available(torch: Any, backend: str) -> bool:
    """Return whether a torch accelerator backend is usable.

    :param Any torch: Imported ``torch`` module object.
    :param str backend: Accelerator token (``cuda`` or ``mps``).
    :return bool: ``True`` when the backend reports as available.
    """
    if backend == "mps":
        _built, available = _mps_capabilities(torch)
        return available

    owner = torch
    backend_module = getattr(owner, backend, None)
    is_available = getattr(backend_module, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def _mps_capabilities(torch: Any) -> tuple[bool, bool]:
    """Return whether the torch MPS backend is built and currently available.

    :param Any torch: Imported torch module object.
    :return tuple[bool, bool]: ``(is_built, is_available)`` capability flags.
    """
    backends = getattr(torch, "backends", None)
    backend_mps = getattr(backends, "mps", None) if backends is not None else None
    torch_mps = getattr(torch, "mps", None)

    is_built = getattr(backend_mps, "is_built", None)
    try:
        built = bool(is_built()) if callable(is_built) else False
    except Exception:
        built = False

    available = False
    for owner in (torch_mps, backend_mps):
        is_available = getattr(owner, "is_available", None)
        if not callable(is_available):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                available = bool(is_available())
        except Exception:
            available = False
        if available:
            break

    if available:
        built = True
    return built, available


def _native_bf16_supported(
    torch: Any, backend: str, probe: Callable[[Any], bool]
) -> bool:
    """Run one backend's bf16 capability probe, failing closed on any error.

    :param Any torch: Imported torch module object.
    :param str backend: Torch submodule attribute holding the probe (``cuda``/``cpu``).
    :param Callable[[Any], bool] probe: Capability probe for that submodule.
    :return bool: Probe result, or ``False`` when the backend cannot answer.
    """
    backend_module = getattr(torch, backend, None)
    try:
        return bool(probe(backend_module))
    except Exception:
        return False


def _cuda_bf16_probe(cuda_module: Any) -> bool:
    """Ask a live CUDA backend for native (non-emulated) bf16 support.

    :param Any cuda_module: ``torch.cuda`` module object, or ``None``.
    :return bool: Whether CUDA reports native bf16 support.
    """
    is_supported = getattr(cuda_module, "is_bf16_supported", None)
    if not callable(is_supported):
        return False
    try:
        return bool(is_supported(including_emulation=False))
    except TypeError:
        # Older torch releases have no emulation keyword; fall back to the
        # compute capability, since bf16 is native from Ampere (8.x) onward.
        get_capability = getattr(cuda_module, "get_device_capability", None)
        capability = get_capability(0) if callable(get_capability) else None
        return bool(
            isinstance(capability, tuple)
            and capability
            and int(capability[0]) >= 8
            and is_supported()
        )


def _cpu_bf16_probe(cpu_module: Any) -> bool:
    """Check native CPU BF16 instructions using available torch capability APIs.

    :param Any cpu_module: ``torch.cpu`` module object, or ``None``.
    :return bool: Whether x86 or ARM BF16 support can be established.
    """
    capabilities = getattr(cpu_module, "get_capabilities", None)
    if callable(capabilities):
        supported = capabilities()
        return any(
            supported.get(name, False)
            for name in ("avx512_bf16", "amx_bf16", "bf16", "sve_bf16")
        )
    # Older supported torch versions expose only this x86 BF16 probe.
    probe = getattr(cpu_module, "_is_avx512_bf16_supported", None)
    return bool(probe()) if callable(probe) else False


def _cuda_native_bf16_supported(torch: Any) -> bool:
    """Return whether CUDA provides native rather than emulated bfloat16.

    :param Any torch: Imported torch module object.
    :return bool: ``True`` only for a live CUDA backend with native bf16 support.
    """
    return _native_bf16_supported(torch, "cuda", _cuda_bf16_probe)


def _cpu_native_bf16_supported(torch: Any) -> bool:
    """Return whether the CPU exposes native bfloat16 instructions.

    :param Any torch: Imported torch module object.
    :return bool: Whether x86 or ARM BF16 support can be established.
    """
    return _native_bf16_supported(torch, "cpu", _cpu_bf16_probe)


def resolve_embedding_device(requested: Optional[str]) -> str:
    """Resolve a requested device token to a concrete torch device string.

    ``auto`` (or ``None``) prefers ``cuda``, then ``mps``, then ``cpu``. An
    explicit accelerator request fails loudly when that backend is unavailable
    rather than silently downgrading to a slower device.

    :param Optional[str] requested: Requested device token
        (``auto``/``cuda``/``mps``/``cpu`` or ``None``).
    :return str: Resolved device token: ``cuda``, ``mps``, or ``cpu``.
    :raises ValueError: If the token is unknown or names an unavailable backend.
    """
    normalized = str(requested or "auto").strip().lower()
    if normalized not in EMBEDDING_DEVICE_CHOICES:
        formatted = ", ".join(EMBEDDING_DEVICE_CHOICES)
        raise ValueError(f"device must be one of: {formatted} (got {requested!r})")

    try:
        torch = deps._import_torch()
    except ImportError:
        if normalized in {"auto", "cpu"}:
            return "cpu"
        raise ValueError(
            f"device='{normalized}' requires torch; install citemesh[embeddings]."
        ) from None

    if normalized == "auto":
        if _accelerator_available(torch, "cuda"):
            return "cuda"
        if _accelerator_available(torch, "mps"):
            return "mps"
        return "cpu"
    if normalized == "cuda" and not _accelerator_available(torch, "cuda"):
        raise ValueError(
            "device='cuda' was requested but CUDA is not available in this runtime."
        )
    if normalized == "mps":
        mps_built, mps_available = _mps_capabilities(torch)
        if not mps_built:
            raise ValueError(
                "device='mps' was requested but this torch build has no MPS support."
            )
        if not mps_available:
            raise ValueError(
                "device='mps' was requested but the built MPS backend is not "
                "available on this machine."
            )
    return normalized
