"""Shared graph-pruning behavior for degree-capped strategies."""

from __future__ import annotations

from types import MethodType
from typing import Callable
from unittest.mock import MagicMock

import pytest

from citemesh.core import HYBRID_CONFIG, Paper
from citemesh.strategies.base import select_capped_undirected_edges
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import HybridGraphBuilder
from tests.conftest import build_top_k_papers


def _make_constant_similarity_builder(
    builder_factory: Callable[[], object], monkeypatch: pytest.MonkeyPatch
):
    builder = builder_factory()
    if isinstance(builder, EmbeddingGraphBuilder):
        cap = builder.top_k
        monkeypatch.setattr(
            "citemesh.strategies.embedding._check_embedding_deps", lambda: None
        )
    elif isinstance(builder, HybridGraphBuilder):
        cap = 1
        monkeypatch.setattr(HYBRID_CONFIG, "max_edges_per_node", cap)
    else:
        cap = 1

    papers = build_top_k_papers()

    def fake_collect_papers(self, seed_id: str, **kwargs):
        del seed_id, kwargs
        return papers

    def always_true(
        self, paper1: Paper, paper2: Paper, similarity: float
    ) -> bool:
        del paper1
        del paper2
        del similarity
        return True

    def constant_similarity(self, paper1: Paper, paper2: Paper) -> float:
        del paper1
        del paper2
        return 1.0

    builder.collect_papers = MethodType(fake_collect_papers, builder)
    builder.should_create_edge = MethodType(always_true, builder)
    builder.compute_similarity = MethodType(constant_similarity, builder)
    return builder, cap


@pytest.mark.parametrize(
    "builder_factory",
    [
        lambda: EmbeddingGraphBuilder(max_papers=4, top_k=1, client=MagicMock()),
        lambda: HybridGraphBuilder(max_papers=4, max_semantic=0, client=MagicMock()),
    ],
)
def test_degree_capping_preserves_per_node_limit(
    builder_factory: Callable[[], object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pruning should cap node degree deterministically for all relevant strategies."""
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
    assert {
        (min(u, v), max(u, v)) for u, v in graph.edges()
    } == expected_edges


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
