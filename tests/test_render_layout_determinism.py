"""Determinism tests for layout perturbation behavior."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from citemesh.visualization.render import compute_layout, compute_node_sizes


@pytest.mark.parametrize("layout_seed", [None, 123])
def test_compute_layout_perturbation_is_stable_across_node_order(
    monkeypatch: pytest.MonkeyPatch, layout_seed: int | None
) -> None:
    """Node perturbations should map deterministically regardless of insertion order."""

    def fake_kamada_kawai_layout(graph: nx.Graph, **kwargs):
        """Return identical base positions while preserving input iteration order."""
        del kwargs
        return {node: np.array([0.0, 0.0], dtype=np.float64) for node in graph.nodes()}

    monkeypatch.setattr(
        "citemesh.visualization.render.nx.kamada_kawai_layout",
        fake_kamada_kawai_layout,
    )

    graph_1 = nx.Graph()
    graph_1.add_nodes_from(["seed", "a", "b"])
    graph_1.add_edges_from([("seed", "a"), ("a", "b")])

    graph_2 = nx.Graph()
    graph_2.add_nodes_from(["b", "seed", "a"])
    graph_2.add_edges_from([("seed", "a"), ("a", "b")])

    pos_1 = compute_layout(graph_1, iterations=10, layout_seed=layout_seed)
    pos_2 = compute_layout(graph_2, iterations=10, layout_seed=layout_seed)

    for node_id in sorted(graph_1.nodes()):
        assert np.allclose(pos_1[node_id], pos_2[node_id])


def test_compute_layout_is_stable_for_real_kamada_kawai() -> None:
    """Real layout output should be invariant to node insertion order."""
    graph_1 = nx.Graph()
    graph_1.add_nodes_from(["seed", "a", "b", "c"])
    graph_1.add_edge("seed", "a", weight=0.8)
    graph_1.add_edge("a", "b", weight=0.7)
    graph_1.add_edge("b", "c", weight=0.6)
    graph_1.add_edge("c", "seed", weight=0.5)

    graph_2 = nx.Graph()
    graph_2.add_nodes_from(["c", "b", "a", "seed"])
    graph_2.add_edge("b", "c", weight=0.6)
    graph_2.add_edge("a", "b", weight=0.7)
    graph_2.add_edge("seed", "a", weight=0.8)
    graph_2.add_edge("c", "seed", weight=0.5)

    pos_1 = compute_layout(graph_1, iterations=25, layout_seed=77)
    pos_2 = compute_layout(graph_2, iterations=25, layout_seed=77)

    for node_id in sorted(graph_1.nodes()):
        assert np.allclose(pos_1[node_id], pos_2[node_id])


def test_compute_node_sizes_are_stable_for_tied_citation_counts() -> None:
    """Node-size tiers should not depend on graph insertion order under ties."""
    graph_1 = nx.Graph()
    graph_1.add_node("b", title="B", citation_count=0, is_seed=False)
    graph_1.add_node("a", title="A", citation_count=0, is_seed=False)
    graph_1.add_node("seed", title="Seed", citation_count=0, is_seed=True)

    graph_2 = nx.Graph()
    graph_2.add_node("seed", title="Seed", citation_count=0, is_seed=True)
    graph_2.add_node("a", title="A", citation_count=0, is_seed=False)
    graph_2.add_node("b", title="B", citation_count=0, is_seed=False)

    ordered_nodes = sorted(graph_1.nodes(), key=str)
    size_map_1 = dict(zip(ordered_nodes, compute_node_sizes(graph_1)))
    size_map_2 = dict(zip(ordered_nodes, compute_node_sizes(graph_2)))

    assert size_map_1 == size_map_2
    assert size_map_1["a"] >= size_map_1["b"]
