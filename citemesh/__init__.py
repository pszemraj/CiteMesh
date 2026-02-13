"""
CiteMesh: Citation mesh visualization toolkit.

A unified package for creating academic paper similarity graphs using
multiple strategies: citation networks, semantic embeddings, or hybrid approaches.
"""

from ._version import version as __version__

__author__ = "CiteMesh Contributors"

from citemesh.core import Author, Paper
from citemesh.strategies.base import GraphBuilderStrategy
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import HybridGraphBuilder
from citemesh.strategies.recommendation import RecommendationGraphBuilder

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
