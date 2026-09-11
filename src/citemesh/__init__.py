"""CiteMesh package exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._lazy import install_lazy_exports

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - fallback for editable/source environments
    __version__ = "0.0.dev0"

__author__ = "CiteMesh Contributors"

from .core import Author, Paper

if TYPE_CHECKING:
    from .strategies.base import GraphBuilderStrategy
    from .strategies.citation import CitationGraphBuilder
    from .strategies.embedding import EmbeddingGraphBuilder
    from .strategies.hybrid import HybridGraphBuilder
    from .strategies.recommendation import RecommendationGraphBuilder

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

# Strategy exports stay lazy so ``import citemesh`` never pulls the embedding or
# visualization stacks.
__getattr__, __dir__ = install_lazy_exports(
    globals(),
    {
        "GraphBuilderStrategy": ("citemesh.strategies.base", "GraphBuilderStrategy"),
        "CitationGraphBuilder": (
            "citemesh.strategies.citation",
            "CitationGraphBuilder",
        ),
        "RecommendationGraphBuilder": (
            "citemesh.strategies.recommendation",
            "RecommendationGraphBuilder",
        ),
        "EmbeddingGraphBuilder": (
            "citemesh.strategies.embedding",
            "EmbeddingGraphBuilder",
        ),
        "HybridGraphBuilder": ("citemesh.strategies.hybrid", "HybridGraphBuilder"),
    },
)
