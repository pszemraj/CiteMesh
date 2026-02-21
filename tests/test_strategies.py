"""Consolidated strategy behavior and determinism tests."""

from __future__ import annotations

import logging
from types import MethodType
from typing import Callable
from unittest.mock import MagicMock, call, patch

import networkx as nx
import numpy as np
import pytest

from citemesh.core import HYBRID_CONFIG, Author, Paper
from citemesh.strategies.base import (
    GraphBuilderStrategy,
    deterministic_sort_key,
    select_capped_undirected_edges,
)
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import HybridGraphBuilder
from citemesh.strategies.recommendation import RecommendationGraphBuilder
from tests._helpers import build_top_k_papers


class _Tagged:
    """Hashable identifier with stable string representation."""

    def __init__(self, label: str) -> None:
        """Create tagged identifier wrapper."""
        self.label = label

    def __hash__(self) -> int:
        return hash(self.label)

    def __str__(self) -> str:
        return self.label


def _paper(paper_id: str, year: int = 2020, refs: list[str] | None = None) -> Paper:
    """Build a deterministic paper fixture."""
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=year,
        abstract=f"Abstract {paper_id}",
        references=refs or [],
        citation_count=10,
    )


def _seed_paper(paper_id: str = "seed") -> Paper:
    """Build deterministic seed paper fixture."""
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=2024,
        abstract="seed abstract",
        is_seed=True,
    )


def _disable_embedding_strategy_dep_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disable embedding optional dependency checks for strategy unit tests."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )
    monkeypatch.setattr(
        "citemesh.strategies.hybrid._check_embedding_deps", lambda: None
    )


def _make_constant_similarity_builder(
    builder_factory: Callable[[], object], monkeypatch: pytest.MonkeyPatch
) -> tuple[object, int]:
    """Build strategy with deterministic always-on edge creation for capping tests."""
    builder = builder_factory()
    if isinstance(builder, EmbeddingGraphBuilder):
        cap = builder.top_k
        _disable_embedding_strategy_dep_checks(monkeypatch)
    elif isinstance(builder, HybridGraphBuilder):
        cap = 1
        _disable_embedding_strategy_dep_checks(monkeypatch)
        monkeypatch.setattr(HYBRID_CONFIG, "max_edges_per_node", cap)
    else:
        cap = 1

    papers = build_top_k_papers()

    def fake_collect_papers(self, seed_id: str, **kwargs: object) -> dict[str, Paper]:
        del seed_id
        del kwargs
        return papers

    def always_true(self, paper1: Paper, paper2: Paper, similarity: float) -> bool:
        del paper1, paper2, similarity
        return True

    def constant_similarity(self, paper1: Paper, paper2: Paper) -> float:
        del paper1, paper2
        return 1.0

    builder.collect_papers = MethodType(fake_collect_papers, builder)
    builder.should_create_edge = MethodType(always_true, builder)
    builder.compute_similarity = MethodType(constant_similarity, builder)
    return builder, cap


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
    client.get_reference_ids.side_effect = lambda pid, force_refresh=False: [
        f"{pid}-ref"
    ]

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
    assert client.get_reference_ids.call_args_list == [
        call("ref1", force_refresh=False),
        call("cit1", force_refresh=False),
    ]


def test_citation_build_graph_persists_seed_relation_metadata() -> None:
    """Citation graph export metadata should preserve seed relation classes."""
    seed = _paper("seed", refs=["seed-ref"])
    ref = _paper("ref1")
    cit = _paper("cit1")

    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = [ref]
    client.get_paper_citations.return_value = [cit]

    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=False,
        similarity_threshold=0.0,
        client=client,
    )
    graph, seed_id = builder.build_graph("seed")

    assert seed_id == "seed"
    assert graph.graph["seed_relations"] == {
        "cit1": "cites_seed",
        "ref1": "referenced_by_seed",
        "seed": "seed",
    }


def test_refresh_reference_cache_force_lookup_contracts() -> None:
    """Refresh mode should bypass stale memory entries and force service lookups."""
    citation_client = MagicMock()
    citation_client.get_reference_ids.return_value = ["fresh-ref"]
    citation_builder = CitationGraphBuilder(
        fetch_references=True,
        refresh_reference_cache=True,
        client=citation_client,
    )
    citation_builder.reference_cache["paper-1"] = ["stale-ref"]

    refs = citation_builder._get_references("paper-1")

    assert refs == ["fresh-ref"]
    citation_client.get_reference_ids.assert_called_once_with(
        "paper-1", force_refresh=True
    )
    assert citation_builder.reference_cache["paper-1"] == ["fresh-ref"]

    recommendation_client = MagicMock()
    recommendation_client.get_reference_ids.return_value = ["r2"]
    recommendation_builder = RecommendationGraphBuilder(
        fetch_references=True,
        refresh_reference_cache=True,
        client=recommendation_client,
    )
    paper = Paper(
        paper_id="paper-2",
        title="Paper 2",
        year=2020,
        abstract="paper two",
    )

    recommendation_builder._hydrate_references(paper)

    assert paper.references == ["r2"]
    recommendation_client.get_reference_ids.assert_called_once_with(
        "paper-2", force_refresh=True
    )


def test_citation_collect_clears_in_memory_reference_cache_between_requests() -> None:
    """Citation collection should scope in-memory reference cache to one request."""
    seed = _paper("seed", refs=["seed-ref"])
    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = []
    client.get_paper_citations.return_value = []

    builder = CitationGraphBuilder(
        max_papers=1,
        fetch_references=False,
        client=client,
    )
    builder.reference_cache["stale"] = ["stale-ref"]

    builder.collect_papers("seed")

    assert builder.reference_cache == {}


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
    _disable_embedding_strategy_dep_checks(monkeypatch)
    builder = HybridGraphBuilder(max_papers=5, max_semantic=2, client=MagicMock())

    seed = _paper("seed")
    seed.is_seed = True
    citation_papers = {"seed": seed, "c1": _paper("c1")}
    semantic_papers = {"c1": _paper("c1"), "s1": _paper("s1"), "s2": _paper("s2")}

    builder.citation_builder.collect_papers = MagicMock(return_value=citation_papers)
    builder.citation_builder.seed_relations = {
        "seed": "seed",
        "c1": "referenced_by_seed",
    }
    assert builder.embedding_builder is not None
    builder.embedding_builder.collect_papers = MagicMock(return_value=semantic_papers)

    papers = builder.collect_papers("seed")

    assert set(papers) == {"seed", "c1", "s1", "s2"}
    assert builder.paper_sources["seed"] == "citation"
    assert builder.paper_sources["s1"] == "semantic"
    assert builder.seed_relations["seed"] == "seed"
    assert builder.seed_relations["c1"] == "referenced_by_seed"
    assert builder.seed_relations["s1"] == "semantic_only"


def test_hybrid_collection_fails_closed_on_semantic_enrichment_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid collection should fail when semantic enrichment cannot complete."""
    _disable_embedding_strategy_dep_checks(monkeypatch)
    builder = HybridGraphBuilder(max_papers=5, max_semantic=2, client=MagicMock())

    seed = _paper("seed")
    seed.is_seed = True
    citation_papers = {"seed": seed, "c1": _paper("c1")}

    builder.citation_builder.collect_papers = MagicMock(return_value=citation_papers)
    assert builder.embedding_builder is not None
    builder.embedding_builder.collect_papers = MagicMock(
        side_effect=RuntimeError("semantic backend unavailable")
    )

    with pytest.raises(
        RuntimeError, match="Semantic enrichment failed: semantic backend unavailable"
    ):
        builder.collect_papers("seed")


def test_hybrid_rerank_falls_back_when_seed_embedding_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid rerank should continue when seed embedding encode is unavailable."""
    _disable_embedding_strategy_dep_checks(monkeypatch)
    builder = HybridGraphBuilder(max_papers=4, max_semantic=1, client=MagicMock())
    assert builder.embedding_builder is not None

    seed = _seed_paper("seed")
    candidate = _paper("c1")
    builder.embedding_builder.embeddings = {
        candidate.paper_id: np.asarray([0.2, 0.1, 0.3], dtype=np.float32)
    }
    builder.embedding_builder.model_profile = MagicMock(
        format_query=lambda text, _metadata: text
    )
    builder.embedding_builder._encode_texts = MagicMock(
        side_effect=RuntimeError("temporary seed encode failure")
    )

    seed_embedding = builder._ensure_candidate_embeddings(
        seed, {candidate.paper_id: candidate}
    )

    assert seed_embedding is None
    assert builder._rank_candidates(
        seed,
        {candidate.paper_id: candidate},
        {candidate.paper_id: {"semantic"}},
    ) == [candidate.paper_id]


def test_hybrid_rerank_keeps_candidate_embedding_hydration_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid rerank should not persist citation-candidate embeddings into cache."""
    _disable_embedding_strategy_dep_checks(monkeypatch)
    builder = HybridGraphBuilder(max_papers=4, max_semantic=1, client=MagicMock())
    assert builder.embedding_builder is not None

    seed = _seed_paper("seed")
    candidate = _paper("c1")
    builder.embedding_builder.embeddings = {}
    builder.embedding_builder.model_profile = MagicMock(
        format_query=lambda text, _metadata: text,
        format_document=lambda metadata: (
            f"{metadata.get('title', '')}. {metadata.get('abstract', '')}"
        ),
    )
    builder.embedding_builder._encode_texts = MagicMock(
        side_effect=[
            np.asarray([[0.4, 0.1, 0.2]], dtype=np.float32),
            np.asarray([[0.3, 0.2, 0.1]], dtype=np.float32),
        ]
    )
    builder.embedding_builder.embedding_cache = MagicMock()
    builder.embedding_builder.embedding_cache.get_embeddings = MagicMock(
        side_effect=AssertionError(
            "Hybrid rerank candidate hydration must not write into persistent cache."
        )
    )

    seed_embedding = builder._ensure_candidate_embeddings(
        seed, {candidate.paper_id: candidate}
    )

    np.testing.assert_allclose(
        seed_embedding, np.asarray([0.4, 0.1, 0.2], dtype=np.float32)
    )
    np.testing.assert_allclose(
        builder.embedding_builder.embeddings[candidate.paper_id],
        np.asarray([0.3, 0.2, 0.1], dtype=np.float32),
    )
    builder.embedding_builder.embedding_cache.get_embeddings.assert_not_called()


def test_hybrid_thresholds_and_default_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid should enforce seed/non-seed threshold and semantic budget rules."""
    _disable_embedding_strategy_dep_checks(monkeypatch)

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


def test_hybrid_semantic_branch_collects_full_citation_candidate_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid semantic mode should fetch full reference+citation candidate pools."""
    _disable_embedding_strategy_dep_checks(monkeypatch)
    builder = HybridGraphBuilder(
        max_papers=40,
        max_semantic=None,
        max_references=20,
        max_citations=20,
        client=MagicMock(),
    )

    assert builder.citation_builder.max_papers == 41
    assert builder.citation_builder.max_references == 20
    assert builder.citation_builder.max_citations == 20


def test_hybrid_propagates_refresh_reference_cache_to_citation_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid should forward refresh-reference policy to citation builder."""
    _disable_embedding_strategy_dep_checks(monkeypatch)
    builder = HybridGraphBuilder(
        max_papers=4,
        max_semantic=0,
        refresh_reference_cache=True,
        client=MagicMock(),
    )

    assert builder.citation_builder.refresh_reference_cache is True


def test_hybrid_rerank_enforces_semantic_cap_and_overlap_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid rerank should cap semantic-only additions while preserving overlap papers."""
    _disable_embedding_strategy_dep_checks(monkeypatch)
    builder = HybridGraphBuilder(max_papers=5, max_semantic=1, client=MagicMock())

    seed = _paper("seed")
    seed.is_seed = True
    citation_papers = {
        "seed": seed,
        "c1": _paper("c1"),
        "c2": _paper("c2"),
        "o1": _paper("o1"),
    }
    semantic_papers = {
        "seed": seed,
        "s1": _paper("s1"),
        "s2": _paper("s2"),
        "o1": _paper("o1"),
    }
    builder.citation_builder.collect_papers = MagicMock(return_value=citation_papers)
    assert builder.embedding_builder is not None
    builder.embedding_builder.collect_papers = MagicMock(return_value=semantic_papers)
    monkeypatch.setattr(
        builder,
        "_rank_candidates",
        lambda *_args, **_kwargs: ["o1", "s1", "s2", "c2", "c1"],
    )

    papers = builder.collect_papers("seed")

    assert list(papers.keys()) == ["seed", "o1", "s1", "c2", "c1"]
    assert builder.paper_sources["o1"] == "both"
    assert builder.paper_sources["s1"] == "semantic"
    assert "s2" not in papers


def test_hybrid_collection_dedupes_semantic_seed_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid collection should collapse semantic seed aliases into the citation seed."""
    _disable_embedding_strategy_dep_checks(monkeypatch)
    builder = HybridGraphBuilder(max_papers=5, max_semantic=2, client=MagicMock())

    seed = Paper(
        paper_id="ca997f1a733e53ad0fa29041246ff655243e8c1b",
        title="Polynomial Composition Activations: Unleashing the Dynamics of Large Language Models",
        year=2024,
        abstract=(
            "Transformers have found extensive applications across various domains due "
            "to the powerful fitting capabilities."
        ),
        authors=[Author(name="Zhijian Zhou"), Author(name="Ya Wang")],
        is_seed=True,
    )
    citation_papers = {"seed": seed, "c1": _paper("c1")}
    semantic_seed_alias = Paper(
        paper_id="arXiv:2411.03884v2",
        title=(
            "Polynomial Composition Activations: Unleashing the Dynamics of Large\n"
            "  Language Models"
        ),
        year=2025,
        abstract=seed.abstract,
        authors=[Author(name="Zhijian Zhou"), Author(name="Yitao Zeng")],
    )
    semantic_papers = {
        semantic_seed_alias.paper_id: semantic_seed_alias,
        "s1": _paper("s1"),
    }
    builder.citation_builder.collect_papers = MagicMock(return_value=citation_papers)
    assert builder.embedding_builder is not None
    builder.embedding_builder.collect_papers = MagicMock(return_value=semantic_papers)
    monkeypatch.setattr(
        builder,
        "_rank_candidates",
        lambda *_args, **_kwargs: ["c1", "s1"],
    )

    papers = builder.collect_papers("arXiv:2411.03884")

    assert set(papers) == {seed.paper_id, "c1", "s1"}
    assert "arXiv:2411.03884v2" not in papers
    assert papers[seed.paper_id].year == 2024
    assert builder.paper_sources[seed.paper_id] == "citation"


def test_hybrid_build_graph_skips_pruning_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid build should return parent graph unchanged when pruning disabled."""
    _disable_embedding_strategy_dep_checks(monkeypatch)
    monkeypatch.setattr(HYBRID_CONFIG, "max_edges_per_node", 0)

    builder = HybridGraphBuilder(max_papers=3, max_semantic=0, client=MagicMock())
    builder.paper_sources = {"seed": "citation", "a": "semantic"}
    builder.seed_relations = {"seed": "seed", "a": "semantic_only"}
    graph = nx.Graph()
    graph.add_node("seed", is_seed=True)
    graph.add_node("a", is_seed=False)
    monkeypatch.setattr(
        "citemesh.strategies.hybrid.GraphBuilderStrategy.build_graph",
        lambda self, seed_id, **kwargs: (graph, "seed"),
    )

    out_graph, out_seed = builder.build_graph("seed")
    assert out_graph is graph
    assert out_seed == "seed"
    assert out_graph.graph["paper_sources"] == {"a": "semantic", "seed": "citation"}
    assert out_graph.graph["seed_relations"] == {"a": "semantic_only", "seed": "seed"}


def test_hybrid_build_graph_logs_post_cap_edge_count(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Hybrid pruning should log original and filtered edge counts."""
    _disable_embedding_strategy_dep_checks(monkeypatch)
    monkeypatch.setattr(HYBRID_CONFIG, "max_edges_per_node", 1)

    builder = HybridGraphBuilder(max_papers=4, max_semantic=0, client=MagicMock())
    graph = nx.Graph()
    graph.add_node("seed", is_seed=True)
    graph.add_nodes_from(["a", "b", "c"])
    graph.add_edge("seed", "a", weight=0.9)
    graph.add_edge("seed", "b", weight=0.8)
    graph.add_edge("seed", "c", weight=0.7)
    graph.add_edge("a", "b", weight=0.95)
    graph.add_edge("a", "c", weight=0.85)
    graph.add_edge("b", "c", weight=0.75)
    monkeypatch.setattr(
        "citemesh.strategies.hybrid.GraphBuilderStrategy.build_graph",
        lambda self, seed_id, **kwargs: (graph, "seed"),
    )

    with caplog.at_level(logging.INFO):
        out_graph, out_seed = builder.build_graph("seed")

    assert out_seed == "seed"
    assert out_graph.number_of_edges() < graph.number_of_edges()
    assert any(
        f"Hybrid edge cap applied: {graph.number_of_edges()} -> {out_graph.number_of_edges()} edges"
        in record.getMessage()
        for record in caplog.records
    )


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
        _disable_embedding_strategy_dep_checks(monkeypatch)
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
    _disable_embedding_strategy_dep_checks(monkeypatch)

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

    papers = HybridGraphBuilder(
        max_papers=4, max_semantic=1, client=MagicMock()
    ).collect_papers("seed")
    assert len(papers) == 4
    assert "seed" in papers


@pytest.mark.parametrize(
    "builder_factory",
    [
        lambda: EmbeddingGraphBuilder(max_papers=4, top_k=1, client=MagicMock()),
        lambda: HybridGraphBuilder(max_papers=4, max_semantic=0, client=MagicMock()),
    ],
)
def test_degree_capping_preserves_per_node_limit(
    builder_factory: Callable[[], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pruning should cap node degree deterministically."""
    builder, max_edges_per_node = _make_constant_similarity_builder(
        builder_factory, monkeypatch
    )
    graph, _ = builder.build_graph("seed")

    node_ids = sorted(graph.nodes())
    complete_graph_edges = [
        (node_ids[i], node_ids[j], {"weight": 1.0})
        for i in range(len(node_ids))
        for j in range(i + 1, len(node_ids))
    ]
    expected_edges = {
        (min(u, v), max(u, v))
        for u, v, _ in select_capped_undirected_edges(
            complete_graph_edges, max_edges_per_node
        )
    }

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == len(expected_edges)
    assert all(degree <= max_edges_per_node for _, degree in graph.degree())
    assert {(min(u, v), max(u, v)) for u, v in graph.edges()} == expected_edges


def test_select_capped_undirected_edges_tie_break_and_dedupe() -> None:
    """Equal-weight and duplicated undirected edges should stay deterministic."""
    edges = [
        ("b", "a", {"weight": 1.0}),
        ("a", "b", {"weight": 0.6}),
        ("c", "a", {"weight": 1.0}),
        ("c", "a", {"weight": 0.9}),
        ("b", "c", {"weight": 0.9}),
    ]
    selected = select_capped_undirected_edges(edges, max_edges_per_node=1)
    assert selected == [("a", "b", 1.0)]


def test_select_capped_undirected_edges_is_deterministic_with_mixed_id_types() -> None:
    """Mixed node-id types should still sort using total-order key tuples."""
    edges = [
        (1, 2, {"weight": 0.2}),
        (_Tagged("alpha"), _Tagged("beta"), {"weight": 0.2}),
        ("2", 1, {"weight": 0.2}),
        (_Tagged("01"), "01", {"weight": 0.2}),
    ]

    first = select_capped_undirected_edges(edges, max_edges_per_node=10)
    second = select_capped_undirected_edges(edges, max_edges_per_node=10)
    assert first == second


def test_deterministic_sort_key_is_total_for_secondary_and_stable_fields() -> None:
    """Tuple sort keys should be total across mixed secondary key types."""
    keys = [
        deterministic_sort_key(0.42, 10, secondary_id=_Tagged("z"), stable_index=1),
        deterministic_sort_key(0.42, "10", secondary_id="a", stable_index=0),
        deterministic_sort_key(0.42, _Tagged("10"), secondary_id="a", stable_index=2),
        deterministic_sort_key(0.42, "10", secondary_id=_Tagged("z"), stable_index=1),
    ]

    ordered = sorted(keys)
    assert ordered[0][0] == ordered[1][0] == -0.42
    assert len(ordered) == len(keys)


@pytest.mark.parametrize(
    ("paper1_year", "paper2_year", "expected"),
    [(2020, 2020, 1.0), (2010, 2020, 0.1)],
)
def test_temporal_similarity_extremes(
    paper1_year: int,
    paper2_year: int,
    expected: float,
) -> None:
    """Temporal similarity should be stable for known year deltas."""
    paper1 = Paper(paper_id="p1", title="Test 1", year=paper1_year)
    paper2 = Paper(paper_id="p2", title="Test 2", year=paper2_year)
    assert GraphBuilderStrategy.temporal_similarity(paper1, paper2) == expected


def test_temporal_similarity_unknown_year_defaults() -> None:
    """Unknown years should return neutral fallback for temporal similarity."""
    paper1 = Paper(paper_id="p1", title="Test 1", year=None)
    paper2 = Paper(paper_id="p2", title="Test 2", year=2020)
    assert GraphBuilderStrategy.temporal_similarity(paper1, paper2) == 0.5


def test_citation_similarity_branches() -> None:
    """Citation similarity should handle nearby, distant, and zero-citation branches."""
    near_1 = Paper(paper_id="p1", title="Test 1", year=2020, citation_count=100)
    near_2 = Paper(paper_id="p2", title="Test 2", year=2020, citation_count=105)
    assert GraphBuilderStrategy.citation_similarity(near_1, near_2) > 0.9

    far_1 = Paper(paper_id="p3", title="Test 3", year=2020, citation_count=10)
    far_2 = Paper(paper_id="p4", title="Test 4", year=2020, citation_count=1000)
    assert 0.1 < GraphBuilderStrategy.citation_similarity(far_1, far_2) < 0.7

    zero_1 = Paper(paper_id="p5", title="Test 5", year=2020, citation_count=0)
    zero_2 = Paper(paper_id="p6", title="Test 6", year=2020, citation_count=100)
    assert GraphBuilderStrategy.citation_similarity(zero_1, zero_2) == 0.3


def test_bibliographic_coupling_delegates_to_paper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bibliographic coupling should delegate to ``Paper.reference_overlap``."""
    paper1 = Paper(paper_id="p1", title="Test 1", year=2020)
    paper2 = Paper(paper_id="p2", title="Test 2", year=2020)

    monkeypatch.setattr(Paper, "reference_overlap", lambda self, other: 0.42)
    assert GraphBuilderStrategy.bibliographic_coupling(paper1, paper2) == 0.42
