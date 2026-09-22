"""Graph-building strategy exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._lazy import install_lazy_exports

if TYPE_CHECKING:
    from .base import GraphBuilderStrategy as GraphBuilderStrategy
    from .citation import CitationGraphBuilder as CitationGraphBuilder
    from .embedding import EmbeddingGraphBuilder as EmbeddingGraphBuilder
    from .hybrid import HybridGraphBuilder as HybridGraphBuilder
    from .recommendation import RecommendationGraphBuilder as RecommendationGraphBuilder

_LAZY_EXPORTS = {
    "GraphBuilderStrategy": (".base", "GraphBuilderStrategy"),
    "CitationGraphBuilder": (".citation", "CitationGraphBuilder"),
    "RecommendationGraphBuilder": (".recommendation", "RecommendationGraphBuilder"),
    "EmbeddingGraphBuilder": (".embedding", "EmbeddingGraphBuilder"),
    "HybridGraphBuilder": (".hybrid", "HybridGraphBuilder"),
}
__all__ = list(_LAZY_EXPORTS)

# Lazy so importing the package does not drag in every builder's dependencies.
__getattr__, __dir__ = install_lazy_exports(
    globals(),
    _LAZY_EXPORTS,
)
