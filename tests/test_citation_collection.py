"""Additional tests for citation strategy collection and similarity branches."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from citemesh.core import Paper
from citemesh.strategies.citation import CitationGraphBuilder


def _paper(paper_id: str, year: int = 2020, refs: list[str] | None = None) -> Paper:
    """Build a small paper fixture with deterministic text metadata.

    :param str paper_id: Paper ID.
    :param int year: Publication year.
    :param list[str] | None refs: Optional references list.
    :return Paper: Paper fixture.
    """
    return Paper(
        paper_id=paper_id,
        title=f"Title {paper_id}",
        year=year,
        abstract=f"Abstract for {paper_id}",
        references=refs or [],
    )


def test_collect_papers_populates_reference_cache_and_summary() -> None:
    """Collecting papers should populate cache and expose a summary string."""
    seed = _paper("seed", refs=["seed-ref"])
    ref = _paper("ref1")
    cit = _paper("cit1")

    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = [ref]
    client.get_paper_citations.return_value = [cit]
    client.get_reference_ids.side_effect = lambda pid: [f"{pid}-ref"]

    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=True,
        client=client,
    )
    papers = builder.collect_papers("seed")

    assert set(papers) == {"seed", "ref1", "cit1"}
    assert papers["ref1"].references == ["ref1-ref"]
    assert papers["cit1"].references == ["cit1-ref"]
    assert (
        builder.get_collection_summary()
        == "Collected 3 papers (3 with reference lists)"
    )


def test_compute_similarity_uses_reference_and_fallback_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Similarity should use bibliographic branch with refs and citation branch without refs."""
    builder = CitationGraphBuilder(fetch_references=True, client=MagicMock())
    builder._abstract_index = MagicMock()
    builder._abstract_index.similarity.return_value = 0.5

    paper_a = _paper("a", year=2020, refs=["r1", "r2"])
    paper_b = _paper("b", year=2020, refs=["r2", "r3"])
    paper_a.citation_count = 10
    paper_b.citation_count = 20

    with_refs = builder.compute_similarity(paper_a, paper_b)
    expected_with_refs = (
        0.40 * 0.5
        + 0.20 * builder.temporal_similarity(paper_a, paper_b)
        + 0.40 * builder.bibliographic_coupling(paper_a, paper_b)
    )
    assert with_refs == pytest.approx(expected_with_refs)

    paper_b.references = []
    without_refs = builder.compute_similarity(paper_a, paper_b)
    expected_without_refs = (
        0.65 * 0.5
        + 0.20 * builder.temporal_similarity(paper_a, paper_b)
        + 0.15 * builder.citation_similarity(paper_a, paper_b)
    )
    assert without_refs == pytest.approx(expected_without_refs)
