"""Tests for citation strategy edge threshold behavior."""

from unittest.mock import MagicMock

from citemesh.core import Paper
from citemesh.strategies.citation import CitationGraphBuilder


def test_should_create_edge_uses_single_threshold() -> None:
    """Citation edge creation should use only the user-provided threshold."""
    builder = CitationGraphBuilder(similarity_threshold=0.4, client=MagicMock())

    seed = Paper(paper_id="seed", title="Seed", year=2020, abstract="seed")
    seed.is_seed = True
    related = Paper(paper_id="related", title="Related", year=2021, abstract="related")
    other = Paper(paper_id="other", title="Other", year=2019, abstract="other")

    assert not builder.should_create_edge(seed, related, 0.39)
    assert builder.should_create_edge(seed, related, 0.4)
    assert not builder.should_create_edge(other, related, 0.39)
    assert builder.should_create_edge(other, related, 0.4)
