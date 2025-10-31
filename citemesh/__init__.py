"""
CiteMesh: Connected Papers-style citation graph visualization.

A unified package for creating academic paper similarity graphs using
multiple strategies: citation networks, semantic embeddings, or hybrid approaches.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.dev0"

__author__ = "CiteMesh Contributors"

from citemesh.models import Author, Paper
from citemesh.strategies.base import GraphBuilderStrategy

__all__ = ["Paper", "Author", "GraphBuilderStrategy"]
