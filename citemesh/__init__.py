"""
CiteMesh: Connected Papers-style citation graph visualization.

A unified package for creating academic paper similarity graphs using
multiple strategies: citation networks, semantic embeddings, or hybrid approaches.
"""

__version__ = "2.0.0"
__author__ = "CiteMesh Contributors"

from citemesh.models import Paper, Author
from citemesh.strategies.base import GraphBuilderStrategy

__all__ = ["Paper", "Author", "GraphBuilderStrategy"]
