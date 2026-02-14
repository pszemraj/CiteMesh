"""Determinism and shared scoring contract tests for strategies."""

from __future__ import annotations

from types import MethodType
from typing import Callable
from unittest.mock import MagicMock

import pytest

from citemesh.core import HYBRID_CONFIG, Paper
from citemesh.strategies.base import (
    GraphBuilderStrategy,
    deterministic_sort_key,
    select_capped_undirected_edges,
)
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import HybridGraphBuilder
from tests._helpers import build_top_k_papers


class _Tagged:
    """Hashable identifier with stable string representation."""

    def __init__(self, label: str) -> None:
        """Create tagged identifier wrapper.

        :param str label: Stable string label.
        :return None: Stores label for hash/str behavior.
        """
        self.label = label

    def __hash__(self) -> int:
        return hash(self.label)

    def __str__(self) -> str:
        return self.label


def _make_constant_similarity_builder(
    builder_factory: Callable[[], object], monkeypatch: pytest.MonkeyPatch
) -> tuple[object, int]:
    """Build strategy instance with deterministic always-on edge creation."""
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


def test_temporal_similarity_and_decay_unknown_year_defaults() -> None:
    """Unknown years should return neutral fallback for temporal functions."""
    paper1 = Paper(paper_id="p1", title="Test 1", year=None)
    paper2 = Paper(paper_id="p2", title="Test 2", year=2020)
    assert GraphBuilderStrategy.temporal_similarity(paper1, paper2) == 0.5
    assert GraphBuilderStrategy.exponential_temporal_decay(paper1, paper2) == 0.5


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
