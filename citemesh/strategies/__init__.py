"""Graph-building strategy exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import GraphBuilderStrategy
    from .citation import CitationGraphBuilder
    from .embedding import EmbeddingGraphBuilder
    from .hybrid import HybridGraphBuilder
    from .recommendation import RecommendationGraphBuilder

__all__ = [
    "GraphBuilderStrategy",
    "CitationGraphBuilder",
    "RecommendationGraphBuilder",
    "EmbeddingGraphBuilder",
    "HybridGraphBuilder",
]

_STRATEGY_EXPORT_MODULES = {
    "GraphBuilderStrategy": ".base",
    "CitationGraphBuilder": ".citation",
    "RecommendationGraphBuilder": ".recommendation",
    "EmbeddingGraphBuilder": ".embedding",
    "HybridGraphBuilder": ".hybrid",
}


def __getattr__(name: str) -> Any:
    """Resolve strategy exports lazily to avoid eager builder imports.

    :param str name: Requested module attribute.
    :return Any: Export resolved from the defining strategy module.
    :raises AttributeError: If the attribute is not a supported export.
    """
    target_module = _STRATEGY_EXPORT_MODULES.get(name)
    if target_module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(target_module, __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    """Return sorted module attribute names for interactive inspection.

    :return list[str]: Sorted module attribute names plus lazy exports.
    """
    return sorted(set(globals()) | set(__all__))
