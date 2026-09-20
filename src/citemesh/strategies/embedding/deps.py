"""Optional-dependency probes and import shims for the embedding strategy.

Every optional third-party import used by the embedding builder funnels through
one of the ``_import_*`` helpers here, so tests can inject fakes by patching a
single module attribute and the rest of the package never imports ``torch``,
``sentence_transformers``, ``datasets`` or ``huggingface_hub`` at module scope.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from citemesh._runtime import stderr_isatty
from citemesh.progress import progress_enabled

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


def _import_optional(module_name: str, attribute: str | None = None) -> Any:
    """Import an optional dependency module, or one attribute from it.

    Mirrors ``from <module_name> import <attribute>``: a missing module and a
    missing attribute both surface as ``ImportError``, which is the only
    exception the dependency check and the model loader expect to catch.

    :param str module_name: Absolute module name to import.
    :param Optional[str] attribute: Attribute to read from the imported module.
    :return Any: The imported module, or the named attribute when given.
    :raises ImportError: If the module or the requested attribute is unavailable.
    """
    module = importlib.import_module(module_name)
    if attribute is None:
        return module
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise ImportError(
            f"cannot import name {attribute!r} from {module_name!r}",
            name=module_name,
        ) from error


def _import_torch() -> Any:
    """Import and return the ``torch`` module.

    :return Any: Imported ``torch`` module object.
    """
    return _import_optional("torch")


def _import_sentence_transformer_class() -> Any:
    """Import and return ``SentenceTransformer``.

    :return Any: Imported ``SentenceTransformer`` class.
    """
    return _import_optional("sentence_transformers", "SentenceTransformer")


def _import_datasets_module() -> Any:
    """Import ``datasets`` and apply CiteMesh's progress-display policy.

    :return Any: Imported ``datasets`` module object.
    """
    datasets_module = _import_optional("datasets")
    progress_bars = getattr(datasets_module, "utils", None)
    set_progress_bars = getattr(
        progress_bars,
        "enable_progress_bars"
        if stderr_isatty() and progress_enabled()
        else "disable_progress_bars",
        None,
    )
    if callable(set_progress_bars):
        set_progress_bars()
    return datasets_module


def _import_huggingface_hub_module() -> Any:
    """Import and return the ``huggingface_hub`` module.

    :return Any: Imported ``huggingface_hub`` module object.
    """
    return _import_optional("huggingface_hub")
