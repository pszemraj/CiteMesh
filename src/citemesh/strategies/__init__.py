"""Graph-building strategy exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._lazy import install_lazy_exports

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

# Lazy so importing the package does not drag in every builder's dependencies.
__getattr__, __dir__ = install_lazy_exports(
    globals(),
    {
        "GraphBuilderStrategy": (".base", "GraphBuilderStrategy"),
        "CitationGraphBuilder": (".citation", "CitationGraphBuilder"),
        "RecommendationGraphBuilder": (".recommendation", "RecommendationGraphBuilder"),
        "EmbeddingGraphBuilder": (".embedding", "EmbeddingGraphBuilder"),
        "HybridGraphBuilder": (".hybrid", "HybridGraphBuilder"),
    },
)
