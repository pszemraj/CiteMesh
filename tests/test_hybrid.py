"""Tests for hybrid graph pruning behavior."""

from types import MethodType

import networkx as nx

from citemesh.core import HYBRID_CONFIG, Paper
from citemesh.strategies.hybrid import HybridGraphBuilder


def test_hybrid_pruning_enforces_node_degree_cap(monkeypatch) -> None:
    """Hybrid post-processing keeps node degree within configured hard cap."""

    papers = {
        "seed": Paper(paper_id="seed", title="Seed", year=2024, abstract="seed"),
        "a": Paper(paper_id="a", title="A", year=2024, abstract="alpha"),
        "b": Paper(paper_id="b", title="B", year=2024, abstract="beta"),
        "c": Paper(paper_id="c", title="C", year=2024, abstract="gamma"),
    }
    papers["seed"].is_seed = True

    builder = HybridGraphBuilder(max_papers=4, max_semantic=0)

    def fake_collect_papers(self: HybridGraphBuilder, seed_id: str, **kwargs):
        del seed_id
        del kwargs
        return papers

    def always_true(
        self: HybridGraphBuilder, paper1: Paper, paper2: Paper, similarity: float
    ) -> bool:
        del paper1
        del paper2
        del similarity
        return True

    def constant_similarity(
        self: HybridGraphBuilder, paper1: Paper, paper2: Paper
    ) -> float:
        del paper1
        del paper2
        return 1.0

    builder.collect_papers = MethodType(fake_collect_papers, builder)
    builder.should_create_edge = MethodType(always_true, builder)
    builder.compute_similarity = MethodType(constant_similarity, builder)

    monkeypatch.setattr(HYBRID_CONFIG, "max_edges_per_node", 1)
    graph, _ = builder.build_graph("seed")

    assert graph.number_of_nodes() == 4
    assert all(degree <= 1 for _, degree in graph.degree())
    assert graph.number_of_edges() == 2


def test_hybrid_pruning_breaks_equal_weight_ties_deterministically(
    monkeypatch,
) -> None:
    """Equal-weight hybrid pruning should use deterministic endpoint ordering."""
    builder = HybridGraphBuilder(max_papers=4, max_semantic=0)

    graph = nx.Graph()
    graph.add_node("seed", is_seed=True)
    graph.add_node("a", is_seed=False)
    graph.add_node("b", is_seed=False)
    graph.add_node("c", is_seed=False)
    graph.add_edge("seed", "b", weight=1.0)
    graph.add_edge("a", "c", weight=1.0)
    graph.add_edge("seed", "a", weight=1.0)
    graph.add_edge("a", "b", weight=1.0)

    monkeypatch.setattr(
        "citemesh.strategies.hybrid.GraphBuilderStrategy.build_graph",
        lambda self, seed_id, **kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(HYBRID_CONFIG, "max_edges_per_node", 1)

    out_graph, _ = builder.build_graph("seed")
    assert set(out_graph.edges()) == {("a", "b")}
