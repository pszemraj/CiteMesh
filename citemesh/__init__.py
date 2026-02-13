"""
CiteMesh: Citation mesh visualization toolkit.

A unified package for creating academic paper similarity graphs using
multiple strategies: citation networks, semantic embeddings, or hybrid approaches.
"""

from importlib import import_module
from typing import Any

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - fallback for editable/source environments
    __version__ = "0.0.dev0"

__author__ = "CiteMesh Contributors"

from citemesh.core import Author, Paper

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


def __getattr__(name: str) -> Any:
    """Lazily resolve strategy exports to keep optional boundaries lightweight."""
    if name in _LAZY_STRATEGY_EXPORTS:
        module_name, attr_name = _LAZY_STRATEGY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'citemesh' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Expose lazy-exported names to IDEs and runtime introspection."""
    return sorted(set(globals()) | set(__all__))
