"""Determinism tests for rendering/layout and theme/model profile selection."""

from __future__ import annotations

from typing import Any, Hashable

import networkx as nx
import numpy as np
import pytest

from citemesh.data.model_profiles import get_embedding_model_profile
from citemesh.visualization.render import (
    KK_LAYOUT_DISTANCE_ATTR,
    compute_layout,
    compute_node_colors,
    compute_node_sizes,
)
from citemesh.visualization.themes import get_theme


@pytest.mark.parametrize("layout_seed", [None, 123])
def test_compute_layout_perturbation_is_stable_across_node_order(
    monkeypatch: pytest.MonkeyPatch, layout_seed: int | None
) -> None:
    """Node perturbations should map deterministically regardless of insertion order."""

    def fake_kamada_kawai_layout(
        graph: nx.Graph, **kwargs: Any
    ) -> dict[Hashable, np.ndarray]:
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


def test_compute_layout_uses_distance_weights_for_kamada_kawai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kamada-Kawai should receive inverted similarity distances."""
    captured: dict[str, object] = {}

    def fake_kamada_kawai_layout(
        graph: nx.Graph, **kwargs: Any
    ) -> dict[Hashable, np.ndarray]:
        captured["weight_attr"] = kwargs.get("weight")
        distances = {}
        for left, right, attrs in graph.edges(data=True):
            edge_key = tuple(sorted((str(left), str(right))))
            distances[edge_key] = attrs[KK_LAYOUT_DISTANCE_ATTR]
        captured["distances"] = distances
        return {node: np.array([0.0, 0.0], dtype=np.float64) for node in graph.nodes()}

    monkeypatch.setattr(
        "citemesh.visualization.render.nx.kamada_kawai_layout",
        fake_kamada_kawai_layout,
    )

    graph = nx.Graph()
    graph.add_edge("seed", "high", weight=0.9)
    graph.add_edge("seed", "low", weight=0.1)

    compute_layout(graph, iterations=10, layout_seed=123)

    assert captured["weight_attr"] == KK_LAYOUT_DISTANCE_ATTR
    distances = captured["distances"]
    assert isinstance(distances, dict)
    assert distances[("high", "seed")] < distances[("low", "seed")]


def test_compute_node_colors_and_sizes_are_stable_for_missing_years_and_ties() -> None:
    """Color fallback bounds and size ties should be deterministic."""
    graph_1 = nx.Graph()
    graph_1.add_node("seed", title="Seed", year=None, citation_count=0, is_seed=True)
    graph_1.add_node("b", title="B", year=None, citation_count=0, is_seed=False)
    graph_1.add_node("a", title="A", year=None, citation_count=0, is_seed=False)

    graph_2 = nx.Graph()
    graph_2.add_node("seed", title="Seed", year=None, citation_count=0, is_seed=True)
    graph_2.add_node("a", title="A", year=None, citation_count=0, is_seed=False)
    graph_2.add_node("b", title="B", year=None, citation_count=0, is_seed=False)

    _, min_year, max_year = compute_node_colors(graph_1, "seed", get_theme("light"))
    assert min_year == 2000
    assert max_year == 2001

    ordered_nodes = sorted(graph_1.nodes(), key=str)
    size_map_1 = dict(zip(ordered_nodes, compute_node_sizes(graph_1)))
    size_map_2 = dict(zip(ordered_nodes, compute_node_sizes(graph_2)))
    assert size_map_1 == size_map_2
    assert size_map_1["a"] >= size_map_1["b"]


@pytest.mark.parametrize(
    ("env", "expected_theme"),
    [
        ({"COLORFGBG": "15;0", "DARKMODE": None, "TERM_PROGRAM": None}, "dark"),
        ({"COLORFGBG": "0;15", "DARKMODE": None, "TERM_PROGRAM": None}, "light"),
        ({"COLORFGBG": None, "DARKMODE": "1", "TERM_PROGRAM": None}, "dark"),
    ],
)
def test_get_theme_auto_detection(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str | None],
    expected_theme: str,
) -> None:
    """Auto theme detection should prioritize COLORFGBG then DARKMODE."""
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    assert get_theme("auto").name == expected_theme


def test_get_theme_unknown_defaults_to_light() -> None:
    """Unknown theme keys should default to light palette."""
    assert get_theme("not-a-theme").name == "light"


def test_model_profiles_match_expected_formatters() -> None:
    """Gemma and default profiles should expose expected formatting behavior."""
    gemma = get_embedding_model_profile("google/embeddinggemma-300m")
    assert gemma.name == "google/embeddinggemma"
    assert gemma.float16_supported is False
    assert gemma.preferred_torch_dtype == "bfloat16"
    assert gemma.use_cuda_autocast is True
    assert gemma.compile_inner_transformer is True
    assert gemma.available_truncate_dims == (768, 512, 256, 128)
    assert gemma.recommended_truncate_dim == 256
    assert gemma.format_query("  attention  ").startswith(
        "task: search result | query:"
    )
    assert (
        gemma.format_document({"title": " Title ", "abstract": " Abstract "})
        == "title: Title | text: Abstract"
    )

    default = get_embedding_model_profile("all-MiniLM-L6-v2")
    assert default.name == "default"
    assert default.preferred_torch_dtype is None
    assert default.use_cuda_autocast is False
    assert default.compile_inner_transformer is False
    assert default.available_truncate_dims is None
    assert default.recommended_truncate_dim is None
    assert default.format_query("plain") == "plain"
    assert default.format_document({"title": "T", "abstract": ""}) == "T"
