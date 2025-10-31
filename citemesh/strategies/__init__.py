"""
Graph building strategies for CiteMesh.

This module provides different approaches to building paper similarity graphs:
- Citation-based: Uses bibliographic coupling and co-citation analysis
- Embedding-based: Uses semantic similarity from sentence transformers
- Hybrid: Combines both approaches intelligently
"""

from citemesh.strategies.base import GraphBuilderStrategy

__all__ = ["GraphBuilderStrategy"]
