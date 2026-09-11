"""CiteMesh package exports."""

from importlib import import_module
from typing import Any

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - fallback for editable/source environments
    __version__ = "0.0.dev0"

__author__ = "CiteMesh Contributors"

from .core import Author, Paper

__all__ = [
    "__version__",
    "Paper",
    "Author",
    "GraphBuilderStrategy",
    "CitationGraphBuilder",
    "RecommendationGraphBuilder",
    "EmbeddingGraphBuilder",
    "HybridGraphBuilder",
]

_LAZY_STRATEGY_EXPORTS = {
    "GraphBuilderStrategy": ("citemesh.strategies.base", "GraphBuilderStrategy"),
    "CitationGraphBuilder": ("citemesh.strategies.citation", "CitationGraphBuilder"),
    "RecommendationGraphBuilder": (
        "citemesh.strategies.recommendation",
        "RecommendationGraphBuilder",
    ),
    "EmbeddingGraphBuilder": (
        "citemesh.strategies.embedding",
        "EmbeddingGraphBuilder",
    ),
    "HybridGraphBuilder": ("citemesh.strategies.hybrid", "HybridGraphBuilder"),
}


def __getattr__(name: str) -> Any:
    """Resolve strategy exports lazily to preserve the historical top-level API.

    :param str name: Requested module attribute.
    :return Any: Lazily imported strategy export.
    :raises AttributeError: If ``name`` is not a supported export.
    """
    target = _LAZY_STRATEGY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return sorted module attribute names for interactive inspection.

    :return list[str]: Sorted module attribute names plus lazy exports.
    """
    return sorted(set(globals()) | set(__all__))
