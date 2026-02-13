"""
CiteMesh: Citation mesh visualization toolkit.

A unified package for creating academic paper similarity graphs using
multiple strategies: citation networks, semantic embeddings, or hybrid approaches.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.dev0"

__author__ = "CiteMesh Contributors"

from citemesh.core import Author, Paper
from citemesh.strategies.base import GraphBuilderStrategy
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import HybridGraphBuilder
from citemesh.strategies.recommendation import RecommendationGraphBuilder

__all__ = [
    "Paper",
    "Author",
    "GraphBuilderStrategy",
    "CitationGraphBuilder",
    "RecommendationGraphBuilder",
    "EmbeddingGraphBuilder",
    "HybridGraphBuilder",
]
