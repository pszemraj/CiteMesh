"""Tests for package-level exports."""

from __future__ import annotations


def test_lazy_strategy_exports_resolve() -> None:
    """Top-level strategy exports should be available via lazy loading."""
    import citemesh

    assert citemesh.CitationGraphBuilder.__name__ == "CitationGraphBuilder"
    assert citemesh.RecommendationGraphBuilder.__name__ == "RecommendationGraphBuilder"
    assert citemesh.EmbeddingGraphBuilder.__name__ == "EmbeddingGraphBuilder"
    assert citemesh.HybridGraphBuilder.__name__ == "HybridGraphBuilder"
