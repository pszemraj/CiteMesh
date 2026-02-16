"""Consolidated tests for visualization rendering and export contracts."""

from __future__ import annotations

import json
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Hashable

import networkx as nx
import numpy as np
import pytest

from citemesh.core import Author, Paper
from citemesh.data.model_profiles import get_embedding_model_profile
from citemesh.visualization.export import (
    GRAPHML_DETERMINISM_POLICY_STRICT,
    GRAPHML_LAYOUT_METADATA_KEY,
    GRAPHML_LAYOUT_VERSION_KEY,
    GraphExporter,
    _graphml_determinism_policy,
)
from citemesh.visualization.render import (
    KK_LAYOUT_DISTANCE_ATTR,
    _normalize_layout_positions,
    compute_layout,
    compute_node_colors,
    compute_node_sizes,
    visualize_graph,
)
from citemesh.visualization.themes import get_theme


def _install_fake_plotly(
    monkeypatch: pytest.MonkeyPatch,
    *,
    figure_cls: type,
    scatter_factory: Callable[..., dict[str, object]] | None = None,
) -> None:
    """Install a minimal ``plotly`` module with configurable graph_objects types."""
    fake_go = types.SimpleNamespace(
        Scatter=scatter_factory or (lambda **kwargs: {"type": "scatter", **kwargs}),
        Layout=lambda **kwargs: {"type": "layout", **kwargs},
        Figure=figure_cls,
    )
    fake_plotly = types.ModuleType("plotly")
    fake_plotly.graph_objects = fake_go
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)


class _BaseFakeFigure:
    """Reusable minimal Plotly ``Figure`` stand-in for exporter tests."""

    def __init__(self, data: Any, layout: Any) -> None:
        self.data = data
        self.layout = layout

    def write_html(self, path: str, **kwargs: Any) -> None:
        del kwargs
        Path(path).write_text("<html>plotly</html>")


def _canonicalize_graphml(path: Path) -> str:
    """Return a deterministic textual representation for GraphML comparison."""
    document = ET.parse(path)
    root = document.getroot()
    namespace = "{http://graphml.graphdrawing.org/xmlns}"

    def _sorted_children(node: ET.Element) -> None:
        for child in node:
            _sorted_children(child)

        if not list(node):
            return

        if node.tag == f"{namespace}node":
            node[:] = sorted(
                node,
                key=lambda item: item.attrib.get("id", ""),
            )
        elif node.tag == f"{namespace}edge":
            node[:] = sorted(
                node,
                key=lambda item: (
                    item.attrib.get("source", ""),
                    item.attrib.get("target", ""),
                ),
            )
        elif node.tag == f"{namespace}graph":
            node[:] = sorted(
                node,
                key=lambda item: (
                    item.tag,
                    item.attrib.get("id", ""),
                    item.attrib.get("source", ""),
                    item.attrib.get("target", ""),
                    item.attrib.get("for", ""),
                    item.attrib.get("attr.name", ""),
                ),
            )
        else:
            node[:] = sorted(node, key=lambda item: item.tag)

        for child in node:
            child.attrib = dict(sorted(child.attrib.items(), key=lambda item: item[0]))

    _sorted_children(root)
    for element in document.iter():
        element.attrib = dict(sorted(element.attrib.items(), key=lambda item: item[0]))

    return ET.tostring(root, encoding="unicode")


def _build_graph() -> tuple[nx.Graph, str]:
    """Create a small graph with one rich paper node and one fallback node."""
    seed = Paper(
        paper_id="seed",
        title="Seed Paper",
        year=2020,
        authors=[Author(name="Alice Smith")],
        citation_count=42,
        abstract="Seed abstract",
        categories=["cs.AI"],
        is_seed=True,
    )
    related = Paper(
        paper_id="related",
        title="Related Paper",
        year=2021,
        authors=[Author(name="Bob Jones")],
        citation_count=10,
        abstract="Related abstract",
        categories=["cs.LG"],
    )

    graph = nx.Graph()
    graph.add_node(
        seed.paper_id,
        paper=seed,
        title=seed.title,
        year=seed.year,
        authors=[author.name for author in seed.authors],
        citation_count=seed.citation_count,
        is_seed=True,
    )
    graph.add_node(
        related.paper_id,
        title=related.title,
        year=related.year,
        authors=[author.name for author in related.authors],
        citation_count=related.citation_count,
        is_seed=False,
    )
    graph.add_edge(seed.paper_id, related.paper_id, weight=0.7)
    return graph, seed.paper_id


def test_exporter_json_graphml_contracts_and_determinism(tmp_path: Path) -> None:
    """JSON and GraphML exports should preserve fields, metadata, and determinism."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id, metadata={"strategy": "citation"})

    json_path = tmp_path / "graph.json"
    graphml_path = tmp_path / "graph.graphml"
    graphml_again_path = tmp_path / "graph-again.graphml"

    exporter.to_json(json_path)
    exporter.to_graphml(graphml_path)
    exporter.to_graphml(graphml_again_path)

    payload = json.loads(json_path.read_text())
    assert payload["seed_id"] == seed_id
    assert payload["metadata"]["strategy"] == "citation"
    assert len(payload["nodes"]) == 2
    assert payload["edges"][0]["weight"] == pytest.approx(0.7)

    graphml = nx.read_graphml(graphml_path)
    seed_node = graphml.nodes[seed_id]
    assert seed_node["is_seed"] in {"1", 1}
    assert "Alice Smith" in seed_node["authors"]
    assert graphml.graph["citemesh_meta_strategy"] == "citation"

    policy = _graphml_determinism_policy()
    assert graphml.graph[GRAPHML_LAYOUT_METADATA_KEY] == policy
    assert graphml.graph[GRAPHML_LAYOUT_VERSION_KEY] == nx.__version__
    if policy == GRAPHML_DETERMINISM_POLICY_STRICT:
        assert graphml_path.read_text() == graphml_again_path.read_text()
    else:
        assert _canonicalize_graphml(graphml_path) == _canonicalize_graphml(
            graphml_again_path
        )


def test_exporter_interactive_html_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interactive HTML should fail clearly without pyvis and succeed with a stub."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id)

    monkeypatch.setitem(sys.modules, "pyvis", None)
    monkeypatch.setitem(sys.modules, "pyvis.network", None)
    with pytest.raises(RuntimeError, match="pyvis is required"):
        exporter.to_interactive_html(tmp_path / "missing.html")

    class FakeNetwork:
        instances = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.options = None
            self.nodes = []
            self.edges = []
            FakeNetwork.instances.append(self)

        def set_options(self, options: str) -> None:
            self.options = options

        def add_node(self, node_id: str, **kwargs: Any) -> None:
            self.nodes.append((node_id, kwargs))

        def add_edge(self, source: str, target: str, **kwargs: Any) -> None:
            self.edges.append((source, target, kwargs))

        def save_graph(self, path: str) -> None:
            Path(path).write_text("<html>fake</html>")

    fake_pyvis = types.ModuleType("pyvis")
    fake_pyvis_network = types.ModuleType("pyvis.network")
    fake_pyvis_network.Network = FakeNetwork
    fake_pyvis.network = fake_pyvis_network
    monkeypatch.setitem(sys.modules, "pyvis", fake_pyvis)
    monkeypatch.setitem(sys.modules, "pyvis.network", fake_pyvis_network)

    out_path = tmp_path / "graph.html"
    exporter = GraphExporter(graph, seed_id, theme_name="dark")
    exporter.to_interactive_html(out_path, physics=True)

    instance = FakeNetwork.instances[-1]
    assert out_path.exists()
    assert instance.options is not None
    assert len(instance.nodes) == 2
    assert len(instance.edges) == 1
    assert [node_id for node_id, _ in instance.nodes] == ["related", "seed"]


def test_exporter_plotly_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plotly export should cover missing dependency, div-id handling, and labels."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id)

    monkeypatch.setitem(sys.modules, "plotly", None)
    with pytest.raises(RuntimeError, match="plotly is required"):
        exporter.to_plotly_html(tmp_path / "missing.plotly.html")

    captured: dict[str, object] = {}

    class FakeFigure(_BaseFakeFigure):
        def __init__(self, data: Any, layout: Any) -> None:
            super().__init__(data, layout)
            captured["data"] = data
            captured["layout"] = layout

        def write_html(self, path: str, **kwargs: Any) -> None:
            captured["kwargs"] = kwargs
            Path(path).write_text("<html>plotly</html>")

    _install_fake_plotly(monkeypatch, figure_cls=FakeFigure)
    exporter = GraphExporter(
        graph, seed_id, layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)}
    )
    out_path = tmp_path / "graph.plotly.html"
    exporter.to_plotly_html(out_path)

    assert out_path.exists()
    node_trace = captured["data"][1]
    assert list(node_trace["text"]) == ["Related Paper", "Smith, 2020"]
    layout = captured["layout"]
    assert layout["title"] == "CiteMesh: Seed Paper"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["div_id"] == exporter._plotly_div_id()

    class NoDivIdFigure:
        def __init__(self, data: Any, layout: Any) -> None:
            del data
            del layout

        def write_html(self, path: str, **kwargs: Any) -> None:
            del path
            if "div_id" in kwargs:
                raise TypeError("div_id unsupported")

    _install_fake_plotly(monkeypatch, figure_cls=NoDivIdFigure)
    exporter = GraphExporter(
        graph, seed_id, layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)}
    )
    with pytest.raises(RuntimeError, match="Deterministic Plotly export requires"):
        exporter.to_plotly_html(tmp_path / "nodivid.plotly.html")


def test_visualize_graph_uses_full_seed_title_without_ellipsis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Static render title should retain full seed title text."""
    graph = nx.Graph()
    seed_title = "ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents"
    graph.add_node(
        "seed",
        title=seed_title,
        year=2025,
        authors=["Hanyu Lai"],
        citation_count=15,
        is_seed=True,
    )
    graph.add_node(
        "related",
        title="Related Paper",
        year=2024,
        authors=["Example Author"],
        citation_count=3,
        is_seed=False,
    )
    graph.add_edge("seed", "related", weight=0.8)

    captured: dict[str, str] = {}
    import matplotlib.axes

    original_set_title = matplotlib.axes.Axes.set_title

    def capture_title(self: Any, label: str, *args: Any, **kwargs: Any) -> Any:
        captured["title"] = label
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", capture_title)

    visualize_graph(
        graph,
        "seed",
        tmp_path / "graph.png",
        layout={"seed": np.array([0.0, 0.0]), "related": np.array([1.0, 1.0])},
    )

    assert "..." not in captured["title"]
    assert seed_title in captured["title"].replace("\n", " ")


def test_normalize_layout_positions_recenters_and_bounds() -> None:
    """Layout normalization should center and bound coordinates deterministically."""
    raw = {
        "a": np.array([10.0, -2.0]),
        "b": np.array([22.0, 4.0]),
        "c": np.array([16.0, 8.0]),
    }

    normalized = _normalize_layout_positions(raw, padding_ratio=0.1)
    coords = np.array(list(normalized.values()), dtype=float)
    bounds_center = (coords.max(axis=0) + coords.min(axis=0)) * 0.5

    assert np.allclose(bounds_center, np.array([0.0, 0.0]), atol=1e-9)
    assert float(np.max(np.abs(coords[:, 0]))) <= 0.9 + 1e-9
    assert float(np.max(np.abs(coords[:, 1]))) <= 0.9 + 1e-9


def test_visualize_graph_metadata_overlay_is_compact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Static render metadata should exclude large nested debug payloads."""
    graph = nx.Graph()
    graph.add_node(
        "seed",
        title="Seed Paper",
        year=2025,
        authors=["A"],
        citation_count=5,
        is_seed=True,
    )
    graph.add_node(
        "related",
        title="Related Paper",
        year=2024,
        authors=["B"],
        citation_count=3,
        is_seed=False,
    )
    graph.add_edge("seed", "related", weight=0.8)

    captured_text: dict[str, str] = {}
    import matplotlib.axes

    original_text = matplotlib.axes.Axes.text

    def capture_text(self: Any, *args: Any, **kwargs: Any) -> Any:
        if len(args) >= 3:
            captured_text["text"] = str(args[2])
        return original_text(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", capture_text)

    visualize_graph(
        graph,
        "seed",
        tmp_path / "graph.png",
        layout={"seed": np.array([0.0, 0.0]), "related": np.array([1.0, 1.0])},
        metadata={
            "paper_id": "https://arxiv.org/abs/2508.14040",
            "strategy": "hybrid",
            "nodes": 2,
            "edges": 1,
            "theme": "dark",
            "score_contract": {"strategy": "hybrid", "range_hint": "[0,1]"},
            "embedding": {"storage_precision": "int8"},
        },
    )

    text = captured_text["text"]
    assert "Strategy: hybrid" in text
    assert "Nodes: 2" in text
    assert "Edges: 1" in text
    assert "Score Contract" not in text
    assert "Embedding" not in text


def test_exporter_plotly_and_graphml_handle_missing_year_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plotly colors and GraphML year serialization should be stable with null years."""
    captured: dict[str, object] = {}

    def fake_scatter(**kwargs: Any) -> dict[str, object]:
        if kwargs.get("mode") == "markers+text":
            marker = dict(kwargs["marker"])
            marker["color"] = list(marker["color"])
            captured["marker"] = marker
        return {"type": "scatter", **kwargs}

    _install_fake_plotly(
        monkeypatch,
        figure_cls=_BaseFakeFigure,
        scatter_factory=fake_scatter,
    )

    graph = nx.Graph()
    graph.add_node(
        "seed",
        paper=Paper(
            paper_id="seed",
            title="Seed Paper",
            year=None,
            authors=[Author(name="Alice Smith")],
            citation_count=3,
            abstract="Seed abstract",
            categories=["cs.AI"],
            is_seed=True,
        ),
        title="Seed Paper",
        citation_count=3,
        authors=["Alice Smith"],
        is_seed=True,
    )
    graph.add_node("missing-year", title="No Year", citation_count=0, authors=[])
    graph.add_edge("seed", "missing-year")

    exporter = GraphExporter(
        graph,
        "seed",
        layout={"seed": (0.0, 0.0), "missing-year": (1.0, 1.0)},
    )
    plotly_path = tmp_path / "graph.plotly.html"
    graphml_path = tmp_path / "graph.graphml"
    exporter.to_plotly_html(plotly_path)
    exporter.to_graphml(graphml_path)

    assert plotly_path.exists()
    marker = captured["marker"]
    assert isinstance(marker, dict)
    marker_colors = marker["color"]
    assert None not in marker_colors
    assert 0 not in marker_colors
    assert marker["cmin"] == 2000.0
    assert marker["cmax"] == 2001.0

    graphml = nx.read_graphml(graphml_path)
    assert str(graphml.nodes["missing-year"]["year"]) == "0"


def test_export_ordering_is_stable_across_json_graphml_and_plotly_edge_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node and edge ordering should remain deterministic across exporters."""
    captured: dict[str, object] = {}

    class FakeFigure(_BaseFakeFigure):
        def __init__(self, data: Any, layout: Any) -> None:
            super().__init__(data, layout)
            captured["data"] = data

    _install_fake_plotly(monkeypatch, figure_cls=FakeFigure)

    graph = nx.Graph()
    graph.add_node("z", title="Node Z", year=2022, authors=[], citation_count=0)
    graph.add_node("seed", title="Seed", year=2020, authors=[], is_seed=True)
    graph.add_node("a", title="Node A", year=2021, authors=[], citation_count=0)
    graph.add_edge("seed", "z", weight=0.7)
    graph.add_edge("z", "a", weight=0.5)

    exporter = GraphExporter(
        graph,
        "seed",
        layout={"a": (0.0, 0.0), "seed": (1.0, 0.0), "z": (2.0, 0.0)},
    )
    json_path = tmp_path / "ordered.json"
    graphml_path = tmp_path / "ordered.graphml"
    plotly_path = tmp_path / "ordered.plotly.html"
    exporter.to_json(json_path)
    exporter.to_graphml(graphml_path)
    exporter.to_plotly_html(plotly_path)

    payload = json.loads(json_path.read_text())
    assert [node["id"] for node in payload["nodes"]] == ["a", "seed", "z"]
    assert payload["edges"] == [
        {"source": "a", "target": "z", "weight": pytest.approx(0.5)},
        {"source": "seed", "target": "z", "weight": pytest.approx(0.7)},
    ]

    graphml_xml = ET.fromstring(graphml_path.read_text())
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    graph_element = graphml_xml.find("g:graph", ns)
    assert graph_element is not None
    node_ids = [node.attrib["id"] for node in graph_element.findall("g:node", ns)]
    edge_pairs = [
        (edge.attrib["source"], edge.attrib["target"])
        for edge in graph_element.findall("g:edge", ns)
    ]
    assert node_ids == ["a", "seed", "z"]
    assert edge_pairs == [("a", "z"), ("seed", "z")]

    edge_trace = captured["data"][0]
    assert list(edge_trace["x"]) == [0.0, 2.0, None, 1.0, 2.0, None]


def test_exporter_plotly_html_is_byte_stable_with_real_plotly(tmp_path: Path) -> None:
    """Real Plotly exports should be byte-stable for identical graph/layout inputs."""
    pytest.importorskip("plotly")

    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph, seed_id, layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)}
    )
    out_a = tmp_path / "first.plotly.html"
    out_b = tmp_path / "second.plotly.html"

    exporter.to_plotly_html(out_a)
    exporter.to_plotly_html(out_b)

    assert out_a.read_text() == out_b.read_text()


def test_compute_layout_stability_and_distance_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layout perturbations and distance-weight wiring should both be deterministic."""

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

    graph_1 = nx.Graph()
    graph_1.add_nodes_from(["seed", "a", "b"])
    graph_1.add_edges_from([("seed", "a"), ("a", "b")])

    graph_2 = nx.Graph()
    graph_2.add_nodes_from(["b", "seed", "a"])
    graph_2.add_edges_from([("seed", "a"), ("a", "b")])

    for layout_seed in [None, 123]:
        pos_1 = compute_layout(graph_1, iterations=10, layout_seed=layout_seed)
        pos_2 = compute_layout(graph_2, iterations=10, layout_seed=layout_seed)
        for node_id in sorted(graph_1.nodes()):
            assert np.allclose(pos_1[node_id], pos_2[node_id])

    graph_3 = nx.Graph()
    graph_3.add_edge("seed", "high", weight=0.9)
    graph_3.add_edge("seed", "low", weight=0.1)
    compute_layout(graph_3, iterations=10, layout_seed=77)

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


def test_get_theme_auto_detection_and_unknown_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto theme detection should honor env priority and unknown names default light."""
    cases = [
        ({"COLORFGBG": "15;0", "DARKMODE": None, "TERM_PROGRAM": None}, "dark"),
        ({"COLORFGBG": "0;15", "DARKMODE": None, "TERM_PROGRAM": None}, "light"),
        ({"COLORFGBG": None, "DARKMODE": "1", "TERM_PROGRAM": None}, "dark"),
    ]
    for env, expected_theme in cases:
        for key, value in env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        assert get_theme("auto").name == expected_theme

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
    unsloth_gemma = get_embedding_model_profile("unsloth/embeddinggemma-300m")
    assert unsloth_gemma.name == "google/embeddinggemma"
    assert unsloth_gemma.compile_inner_transformer is True
    assert unsloth_gemma.format_query("plain").startswith("task: search result")

    default = get_embedding_model_profile("all-MiniLM-L6-v2")
    assert default.name == "default"
    assert default.preferred_torch_dtype is None
    assert default.use_cuda_autocast is False
    assert default.compile_inner_transformer is False
    assert default.available_truncate_dims is None
    assert default.recommended_truncate_dim is None
    assert default.format_query("plain") == "plain"
    assert default.format_document({"title": "T", "abstract": ""}) == "T"
