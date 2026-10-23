"""Import optional embedding dependencies only when their features are needed."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import TYPE_CHECKING

from citemesh._runtime import stderr_isatty
from citemesh.progress import progress_enabled

from . import runtime

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _check_embedding_deps(require_corpus: bool = True) -> None:
    """Verify embedding dependencies are installed.

    :param bool require_corpus: Whether corpus hydration deps (``datasets``)
        are required. Candidate mode only needs the encoder stack.
    """
    missing: list[str] = []
    torch_module: ModuleType | None = None

    try:
        torch_module = _import_torch()
    except ImportError:
        missing.append("torch")

    try:
        _import_sentence_transformer_class()
    except ImportError:
        missing.append("sentence-transformers")

    if require_corpus:
        try:
            _import_datasets_module()
        except ImportError:
            missing.append("datasets")

    if missing:
        raise ImportError(
            f"Embedding strategy requires: {', '.join(missing)}. "
            f"Install with: pip install citemesh[embeddings]"
        )

    raw_torch_version = str(getattr(torch_module, "__version__", "")).strip()
    torch_version = runtime._parse_major_minor(raw_torch_version, default=(0, 0))
    if torch_version < runtime._EMBEDDING_MIN_TORCH_VERSION:
        raise ImportError(
            "Embedding strategy requires torch>=2.9.0 (runtime precision policy). "
            f"Detected torch=={raw_torch_version or 'unknown'}."
        )


def _module_available(module_name: str) -> bool:
    """Return whether a Python module can be imported in the current runtime.

    :param str module_name: Absolute module name to probe.
    :return bool: ``True`` when the module exists and is importable.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _import_optional(module_name: str) -> ModuleType:
    """Import an optional dependency module.

    Keeping this import behind a function lets the non-embedding package load
    without importing optional dependencies.

    :param str module_name: Absolute module name to import.
    :return ModuleType: Imported optional module.
    :raises ImportError: If the module cannot be imported.
    """
    return importlib.import_module(module_name)


def _import_torch() -> ModuleType:
    """Import and return the ``torch`` module.

    :return ModuleType: Imported ``torch`` module.
    """
    return _import_optional("torch")


def _import_sentence_transformer_class() -> type[SentenceTransformer]:
    """Import and return ``SentenceTransformer``.

    :return type[SentenceTransformer]: Imported ``SentenceTransformer`` class.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def _import_datasets_module() -> ModuleType:
    """Import ``datasets`` and apply CiteMesh's progress-display policy.

    :return ModuleType: Imported ``datasets`` module.
    """
    datasets_module = _import_optional("datasets")
    progress_bars = getattr(datasets_module, "utils", None)
    if not (stderr_isatty() and progress_enabled()):
        disable_progress_bars = getattr(progress_bars, "disable_progress_bars", None)
        if callable(disable_progress_bars):
            disable_progress_bars()
    return datasets_module


def _import_huggingface_hub_module() -> ModuleType:
    """Import and return the ``huggingface_hub`` module.

    :return ModuleType: Imported ``huggingface_hub`` module.
    """
    return _import_optional("huggingface_hub")
