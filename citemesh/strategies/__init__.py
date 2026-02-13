"""
Graph building strategies for CiteMesh.

This module provides different approaches to building paper similarity graphs:
- Citation-based: Uses bibliographic coupling and co-citation analysis
- Embedding-based: Uses semantic similarity from sentence transformers
- Hybrid: Combines both approaches intelligently
"""

from citemesh.strategies.base import GraphBuilderStrategy
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import HybridGraphBuilder
from citemesh.strategies.recommendation import RecommendationGraphBuilder

__all__ = [
    "GraphBuilderStrategy",
    "CitationGraphBuilder",
    "RecommendationGraphBuilder",
    "EmbeddingGraphBuilder",
    "HybridGraphBuilder",
]
