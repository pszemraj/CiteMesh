"""Tests for embedding top-k pruning behavior."""

from types import MethodType
from unittest.mock import MagicMock

import pytest

from citemesh.core import Paper
from citemesh.strategies.embedding import EmbeddingGraphBuilder


def test_embedding_top_k_enforces_degree_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding graph post-processing should enforce the configured degree cap."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    papers = {
        "seed": Paper(paper_id="seed", title="Seed", year=2024, abstract="seed"),
        "a": Paper(paper_id="a", title="A", year=2024, abstract="alpha"),
        "b": Paper(paper_id="b", title="B", year=2024, abstract="beta"),
        "c": Paper(paper_id="c", title="C", year=2024, abstract="gamma"),
    }
    papers["seed"].is_seed = True

    builder = EmbeddingGraphBuilder(max_papers=4, top_k=1, client=MagicMock())

    def fake_collect_papers(self: EmbeddingGraphBuilder, seed_id: str, **kwargs):
        del seed_id
        del kwargs
        return papers

    def always_true(
        self: EmbeddingGraphBuilder, paper1: Paper, paper2: Paper, similarity: float
    ) -> bool:
        del paper1
        del paper2
        del similarity
        return True

    def constant_similarity(
        self: EmbeddingGraphBuilder, paper1: Paper, paper2: Paper
    ) -> float:
        del paper1
        del paper2
        return 1.0

    builder.collect_papers = MethodType(fake_collect_papers, builder)
    builder.should_create_edge = MethodType(always_true, builder)
    builder.compute_similarity = MethodType(constant_similarity, builder)

    graph, _ = builder.build_graph("seed")

    assert graph.number_of_nodes() == 4
    assert all(degree <= 1 for _, degree in graph.degree())


def test_embedding_top_k_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding builder should reject non-positive ``top_k`` values."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        EmbeddingGraphBuilder(top_k=0, client=MagicMock())
