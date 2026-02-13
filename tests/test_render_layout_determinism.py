"""Determinism tests for layout perturbation behavior."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from citemesh.visualization.render import compute_layout


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
