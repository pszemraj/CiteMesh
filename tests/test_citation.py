"""Tests for citation strategy edge threshold behavior."""

from unittest.mock import MagicMock

from citemesh.core import CITATION_CONFIG, Paper
from citemesh.strategies.citation import CitationGraphBuilder


def test_should_create_edge_uses_deterministic_thresholds() -> None:
    """Citation edge creation should be deterministic and threshold-based."""
    builder = CitationGraphBuilder(similarity_threshold=0.2, client=MagicMock())

    seed = Paper(paper_id="seed", title="Seed", year=2020, abstract="seed")
    seed.is_seed = True
    related = Paper(paper_id="related", title="Related", year=2021, abstract="related")
    other = Paper(paper_id="other", title="Other", year=2019, abstract="other")

    # Global strategy threshold is always enforced first.
    assert not builder.should_create_edge(seed, related, 0.19)

    # Seed links use a strict > comparison against seed_edge_threshold.
    assert not builder.should_create_edge(
        seed, related, CITATION_CONFIG.seed_edge_threshold
    )
    assert builder.should_create_edge(
        seed, related, CITATION_CONFIG.seed_edge_threshold + 0.01
    )

    # Non-seed links use a strict > comparison against normal_edge_threshold.
    assert not builder.should_create_edge(
        other, related, CITATION_CONFIG.normal_edge_threshold
    )
    assert builder.should_create_edge(
        other, related, CITATION_CONFIG.normal_edge_threshold + 0.01
    )

    # Repeated evaluations should produce identical results (no randomness).
    assert [
        builder.should_create_edge(
            other, related, CITATION_CONFIG.normal_edge_threshold + 0.01
        )
        for _ in range(5)
    ] == [True] * 5
