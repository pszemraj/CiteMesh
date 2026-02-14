"""Contract tests for strategy collection and threshold semantics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from citemesh.core import HYBRID_CONFIG, Paper
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import HybridGraphBuilder
from citemesh.strategies.recommendation import RecommendationGraphBuilder


def _paper(paper_id: str, year: int = 2020, refs: list[str] | None = None) -> Paper:
    """Build a deterministic paper fixture.

    :param str paper_id: Paper ID.
    :param int year: Publication year.
    :param list[str] | None refs: Optional references.
    :return Paper: Constructed paper.
    """
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=year,
        abstract=f"Abstract {paper_id}",
        references=refs or [],
        citation_count=10,
    )


def _seed_paper(paper_id: str = "seed") -> Paper:
    """Build deterministic seed paper fixture.

    :param str paper_id: Seed paper ID.
    :return Paper: Seed paper.
    """
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=2024,
        abstract="seed abstract",
        is_seed=True,
    )


def test_citation_and_recommendation_should_create_edge_thresholds() -> None:
    """Citation/recommendation strategies should use configured threshold only."""
    seed = Paper(paper_id="seed", title="Seed", year=2020, abstract="seed")
    seed.is_seed = True
    related = Paper(paper_id="related", title="Related", year=2021, abstract="related")
    other = Paper(paper_id="other", title="Other", year=2019, abstract="other")

    citation = CitationGraphBuilder(similarity_threshold=0.4, client=MagicMock())
    assert not citation.should_create_edge(seed, related, 0.39)
    assert citation.should_create_edge(seed, related, 0.4)
    assert not citation.should_create_edge(other, related, 0.39)
    assert citation.should_create_edge(other, related, 0.4)

    recommendation = RecommendationGraphBuilder(
        similarity_threshold=0.2, client=MagicMock()
    )
    assert recommendation.should_create_edge(seed, related, 0.2)
    assert not recommendation.should_create_edge(seed, related, 0.19)


def test_recommendation_collect_filters_missing_abstract_and_prefers_payload_refs() -> (
    None
):
    """Recommendation collection should drop empty abstracts and keep provided refs."""
    with patch("citemesh.strategies.recommendation.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_paper.return_value = Paper(
            paper_id="seed", title="Seed", year=2020, abstract="seed abstract"
        )
        mock_client.get_recommended_papers.return_value = [
            Paper(
                paper_id="valid",
                title="Valid",
                year=2021,
                abstract="valid abstract",
                references=["r1", "r2"],
            ),
            Paper(
                paper_id="missing",
                title="Missing",
                year=2021,
                abstract="",
            ),
        ]
        mock_get_client.return_value = mock_client

        builder = RecommendationGraphBuilder(max_papers=3, fetch_references=True)
        papers = builder.collect_papers("seed")

    assert "valid" in papers
    assert "missing" not in papers
    assert papers["valid"].references == ["r1", "r2"]
    mock_client.get_reference_ids.assert_not_called()


def test_citation_collect_populates_reference_cache_and_summary() -> None:
    """Citation collection should hydrate references and expose summary."""
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


def test_citation_similarity_uses_reference_and_fallback_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Citation similarity should switch weighting when references are missing."""
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


def test_hybrid_collection_merges_and_tracks_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid collection should merge citation+semantic papers and source labels."""
    monkeypatch.setattr(
        "citemesh.strategies.hybrid._check_embedding_deps", lambda: None
    )
    builder = HybridGraphBuilder(max_papers=5, max_semantic=2, client=MagicMock())

    seed = _paper("seed")
    seed.is_seed = True
    citation_papers = {"seed": seed, "c1": _paper("c1")}
    semantic_papers = {"c1": _paper("c1"), "s1": _paper("s1"), "s2": _paper("s2")}

    builder.citation_builder.collect_papers = MagicMock(return_value=citation_papers)
    assert builder.embedding_builder is not None
    builder.embedding_builder.collect_papers = MagicMock(return_value=semantic_papers)

    papers = builder.collect_papers("seed")

    assert set(papers) == {"seed", "c1", "s1", "s2"}
    assert builder.paper_sources["seed"] == "citation"
    assert builder.paper_sources["s1"] == "semantic"


def test_hybrid_thresholds_and_default_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid should enforce seed/non-seed threshold and semantic budget rules."""
    monkeypatch.setattr(
        "citemesh.strategies.hybrid._check_embedding_deps", lambda: None
    )

    builder = HybridGraphBuilder(max_papers=3, max_semantic=0, client=MagicMock())
    seed = _paper("seed")
    seed.is_seed = True
    other = _paper("other")
    peer = _paper("peer")

    assert not builder.should_create_edge(seed, other, 0.2)
    assert builder.should_create_edge(seed, other, 0.41)
    assert not builder.should_create_edge(other, peer, 0.5)
    assert builder.should_create_edge(other, peer, 0.51)

    with pytest.raises(
        ValueError, match="max_semantic must be between 0 and max_papers - 1"
    ):
        HybridGraphBuilder(max_papers=3, max_semantic=3, client=MagicMock())

    small_builder = HybridGraphBuilder(max_papers=5, client=MagicMock())
    assert small_builder.max_semantic == 4


def test_hybrid_build_graph_skips_pruning_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid build should return parent graph unchanged when pruning disabled."""
    monkeypatch.setattr(
        "citemesh.strategies.hybrid._check_embedding_deps", lambda: None
    )
    monkeypatch.setattr(HYBRID_CONFIG, "max_edges_per_node", 0)

    builder = HybridGraphBuilder(max_papers=3, max_semantic=0, client=MagicMock())
    graph = np.random.default_rng(0)
    monkeypatch.setattr(
        "citemesh.strategies.hybrid.GraphBuilderStrategy.build_graph",
        lambda self, seed_id, **kwargs: (graph, "seed"),
    )

    out_graph, out_seed = builder.build_graph("seed")
    assert out_graph is graph
    assert out_seed == "seed"


@pytest.mark.parametrize(
    "strategy",
    ["recommendation", "citation", "embedding", "hybrid"],
)
def test_max_papers_is_total_node_cap_including_seed(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
) -> None:
    """All strategies should include seed within ``max_papers`` total cap."""
    if strategy == "recommendation":
        client = MagicMock()
        client.get_paper.return_value = _seed_paper()
        client.get_recommended_papers.return_value = [
            _paper(f"r{i}") for i in range(1, 6)
        ]
        papers = RecommendationGraphBuilder(max_papers=3, client=client).collect_papers(
            "seed"
        )
        assert len(papers) == 3
        assert "seed" in papers
        return

    if strategy == "citation":
        client = MagicMock()
        client.get_paper.return_value = _seed_paper()
        client.get_paper_references.return_value = [
            _paper(f"r{i}") for i in range(1, 6)
        ]
        client.get_paper_citations.return_value = [_paper(f"c{i}") for i in range(1, 6)]
        papers = CitationGraphBuilder(
            max_papers=4,
            max_citations=10,
            max_references=10,
            client=client,
        ).collect_papers("seed")
        assert len(papers) == 4
        assert "seed" in papers
        return

    if strategy == "embedding":
        monkeypatch.setattr(
            "citemesh.strategies.embedding._check_embedding_deps", lambda: None
        )
        builder = EmbeddingGraphBuilder(
            max_papers=3,
            model_name="dummy",
            top_k=2,
            corpus_size=10,
            client=MagicMock(),
        )
        builder.client.get_paper.return_value = _seed_paper()
        builder._load_model = lambda: None
        builder._update_citation_counts = lambda _: None
        builder._select_candidates_from_loaded = lambda _: [
            (
                "c1",
                {"title": "Paper c1", "abstract": "A", "authors": [], "categories": []},
                np.array([1.0, 0.0], dtype=np.float32),
            ),
            (
                "c2",
                {"title": "Paper c2", "abstract": "B", "authors": [], "categories": []},
                np.array([0.0, 1.0], dtype=np.float32),
            ),
            (
                "c3",
                {"title": "Paper c3", "abstract": "C", "authors": [], "categories": []},
                np.array([0.5, 0.5], dtype=np.float32),
            ),
        ]
        builder._encode_texts = lambda texts, **kwargs: np.array(
            [[1.0, 0.0] for _ in texts], dtype=np.float32
        )
        papers = builder.collect_papers("seed")
        assert len(papers) == 3
        assert "seed" in papers
        return

    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    class FakeCitationBuilder:
        def __init__(self, max_papers: int, *_args: object, **_kwargs: object) -> None:
            self.max_papers = max_papers

        def collect_papers(self, seed_id: str, **_: object) -> dict[str, Paper]:
            del seed_id
            papers = {"seed": _seed_paper(), "c1": _paper("c1"), "c2": _paper("c2")}
            return dict(list(papers.items())[: self.max_papers])

    class FakeEmbeddingBuilder:
        def __init__(self, max_papers: int, *_args: object, **_kwargs: object) -> None:
            self.max_papers = max_papers

        def collect_papers(self, seed_id: str, **_: object) -> dict[str, Paper]:
            del seed_id
            return {"seed": _seed_paper(), "s1": _paper("s1"), "s2": _paper("s2")}

    monkeypatch.setattr(
        "citemesh.strategies.hybrid.CitationGraphBuilder", FakeCitationBuilder
    )
    monkeypatch.setattr(
        "citemesh.strategies.hybrid.EmbeddingGraphBuilder", FakeEmbeddingBuilder
    )
    monkeypatch.setattr(
        "citemesh.strategies.hybrid._check_embedding_deps", lambda: None
    )

    papers = HybridGraphBuilder(
        max_papers=4, max_semantic=1, client=MagicMock()
    ).collect_papers("seed")
    assert len(papers) == 4
    assert "seed" in papers
