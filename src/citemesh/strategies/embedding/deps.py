"""Optional-dependency probes and import shims for the embedding strategy.

Every optional third-party import used by the embedding builder funnels through
one of the ``_import_*`` helpers here, so tests can inject fakes by patching a
single module attribute and the rest of the package never imports ``torch``,
``sentence_transformers``, ``datasets`` or ``huggingface_hub`` at module scope.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from . import runtime


def _check_embedding_deps(require_corpus: bool = True) -> None:
    """Verify embedding dependencies are installed.

    :param bool require_corpus: Whether corpus hydration deps (``datasets``)
        are required. Candidate mode only needs the encoder stack.
    """
    missing: list[str] = []
    torch_module: Any | None = None

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
    torch_version = runtime._parse_torch_major_minor(raw_torch_version)
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


def _import_torch() -> Any:
    """Import and return the ``torch`` module.

    :return Any: Imported ``torch`` module object.
    """
    import torch

    return torch


def _import_sentence_transformer_class() -> Any:
    """Import and return ``SentenceTransformer``.

    :return Any: Imported ``SentenceTransformer`` class.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def _import_datasets_module() -> Any:
    """Import and return the ``datasets`` module.

    :return Any: Imported ``datasets`` module object.
    """
    import datasets

    return datasets


def _import_huggingface_hub_module() -> Any:
    """Import and return the ``huggingface_hub`` module.

    :return Any: Imported ``huggingface_hub`` module object.
    """
    import huggingface_hub

    return huggingface_hub
