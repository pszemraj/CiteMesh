"""Additional tests for hybrid collection/similarity behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from citemesh.core import HYBRID_CONFIG, Paper
from citemesh.strategies.hybrid import HybridGraphBuilder


def _paper(paper_id: str, year: int = 2020) -> Paper:
    """Build a minimal paper fixture.

    :param str paper_id: Paper identifier.
    :param int year: Publication year.
    :return Paper: Paper fixture.
    """
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=year,
        abstract=f"Abstract {paper_id}",
        references=["r1", "r2"],
        citation_count=10,
    )


def test_collect_papers_merges_semantic_and_tracks_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid collection should merge citation+semantic papers and track source labels."""
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
    assert builder.paper_sources["s2"] == "semantic"


def test_collect_papers_semantic_failure_falls_back_to_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semantic collection failures should not break citation-only output."""
    monkeypatch.setattr(
        "citemesh.strategies.hybrid._check_embedding_deps", lambda: None
    )
    builder = HybridGraphBuilder(max_papers=4, max_semantic=1, client=MagicMock())

    seed = _paper("seed")
    seed.is_seed = True
    citation_papers = {"seed": seed, "c1": _paper("c1")}
    builder.citation_builder.collect_papers = MagicMock(return_value=citation_papers)
    assert builder.embedding_builder is not None
    builder.embedding_builder.collect_papers = MagicMock(
        side_effect=RuntimeError("boom")
    )

    papers = builder.collect_papers("seed")
    assert set(papers) == {"seed", "c1"}


def test_compute_similarity_uses_source_specific_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid similarity should apply source-mode weights and co-citation boost."""
    monkeypatch.setattr(
        "citemesh.strategies.hybrid._check_embedding_deps", lambda: None
    )
    builder = HybridGraphBuilder(max_papers=4, max_semantic=1, client=MagicMock())

    p_sem_1 = _paper("s1", year=2020)
    p_sem_2 = _paper("s2", year=2021)
    p_cit_1 = _paper("c1", year=2020)
    p_cit_2 = _paper("c2", year=2021)

    assert builder.embedding_builder is not None
    builder.embedding_builder.embeddings = {
        "s1": np.array([1.0, 0.0], dtype=np.float32),
        "s2": np.array([1.0, 0.0], dtype=np.float32),
    }
    builder.paper_sources = {
        "s1": "semantic",
        "s2": "semantic",
        "c1": "citation",
        "c2": "citation",
    }

    semantic_similarity = builder.compute_similarity(p_sem_1, p_sem_2)
    citation_similarity = builder.compute_similarity(p_cit_1, p_cit_2)

    assert semantic_similarity >= citation_similarity
    assert semantic_similarity <= 1.0


def test_hybrid_should_create_edge_thresholds() -> None:
    """Hybrid threshold logic should distinguish seed and non-seed cases."""
    builder = HybridGraphBuilder(max_papers=3, max_semantic=0, client=MagicMock())

    seed = _paper("seed")
    seed.is_seed = True
    other = _paper("other")
    peer = _paper("peer")

    assert not builder.should_create_edge(seed, other, 0.2)
    assert builder.should_create_edge(seed, other, 0.41)
    assert not builder.should_create_edge(other, peer, 0.5)
    assert builder.should_create_edge(other, peer, 0.51)


def test_hybrid_build_graph_skips_pruning_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build should return early when max edge cap is disabled."""
    monkeypatch.setattr(
        "citemesh.strategies.hybrid._check_embedding_deps", lambda: None
    )
    builder = HybridGraphBuilder(max_papers=3, max_semantic=0, client=MagicMock())
    monkeypatch.setattr(HYBRID_CONFIG, "max_edges_per_node", 0)

    graph = np.random.default_rng(0)  # deterministic payload holder
    # Reuse base return path by stubbing parent build_graph.
    monkeypatch.setattr(
        "citemesh.strategies.hybrid.GraphBuilderStrategy.build_graph",
        lambda self, seed_id, **kwargs: (graph, "seed"),
    )

    out_graph, out_seed = builder.build_graph("seed")
    assert out_graph is graph
    assert out_seed == "seed"


def test_hybrid_rejects_invalid_semantic_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid should reject max_semantic values that consume the full paper budget."""
    monkeypatch.setattr(
        "citemesh.strategies.hybrid._check_embedding_deps", lambda: None
    )

    with pytest.raises(
        ValueError, match="max_semantic must be between 0 and max_papers - 1"
    ):
        HybridGraphBuilder(max_papers=3, max_semantic=3, client=MagicMock())
