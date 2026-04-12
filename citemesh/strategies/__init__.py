"""
Graph building strategies for CiteMesh.

This module provides different approaches to building paper similarity graphs:
- Citation-based: Uses bibliographic coupling and co-citation analysis
- Embedding-based: Uses semantic similarity from sentence transformers
- Hybrid: Combines both approaches intelligently
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GraphBuilderStrategy",
    "CitationGraphBuilder",
    "RecommendationGraphBuilder",
    "EmbeddingGraphBuilder",
    "HybridGraphBuilder",
]

_LAZY_STRATEGY_EXPORTS = {
    "GraphBuilderStrategy": ("citemesh.strategies.base", "GraphBuilderStrategy"),
    "CitationGraphBuilder": ("citemesh.strategies.citation", "CitationGraphBuilder"),
    "RecommendationGraphBuilder": (
        "citemesh.strategies.recommendation",
        "RecommendationGraphBuilder",
    ),
    "EmbeddingGraphBuilder": (
        "citemesh.strategies.embedding",
        "EmbeddingGraphBuilder",
    ),
    "HybridGraphBuilder": ("citemesh.strategies.hybrid", "HybridGraphBuilder"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve strategy exports to keep optional deps isolated."""
    if name not in _LAZY_STRATEGY_EXPORTS:
        raise AttributeError(f"module 'citemesh.strategies' has no attribute {name!r}")

    module_name, attr_name = _LAZY_STRATEGY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy strategy exports to IDEs and runtime introspection."""
    return sorted(set(globals()) | set(__all__))
