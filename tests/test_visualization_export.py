"""Tests for graph exporter output branches."""

from __future__ import annotations

import json
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import networkx as nx
import pytest

from citemesh.core import Author, Paper
from citemesh.visualization.export import GraphExporter


def _build_graph() -> tuple[nx.Graph, str]:
    """Create a small graph with one rich paper node and one fallback node.

    :return tuple[nx.Graph, str]: Graph and seed ID.
    """
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


def test_exporter_json_and_graphml_serialization(tmp_path: Path) -> None:
    """JSON and GraphML exports should serialize expected node/edge fields."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id, metadata={"strategy": "citation"})

    json_path = tmp_path / "graph.json"
    graphml_path = tmp_path / "graph.graphml"
    exporter.to_json(json_path)
    exporter.to_graphml(graphml_path)

    payload = json.loads(json_path.read_text())
    assert payload["seed_id"] == seed_id
    assert payload["metadata"]["strategy"] == "citation"
    assert len(payload["nodes"]) == 2
    assert payload["edges"][0]["weight"] == pytest.approx(0.7)

    graphml = nx.read_graphml(graphml_path)
    seed_node = graphml.nodes[seed_id]
    assert seed_node["is_seed"] in {"1", 1}
    assert "Alice Smith" in seed_node["authors"]


def test_exporter_interactive_html_raises_without_pyvis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interactive HTML export should error cleanly when pyvis is unavailable."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id)

    monkeypatch.setitem(sys.modules, "pyvis", None)
    monkeypatch.setitem(sys.modules, "pyvis.network", None)

    with pytest.raises(RuntimeError, match="pyvis is required"):
        exporter.to_interactive_html(tmp_path / "graph.html")


def test_exporter_interactive_html_with_fake_pyvis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interactive HTML export should write output through a pyvis-compatible API."""

    class FakeNetwork:
        """Minimal pyvis Network stand-in."""

        instances = []

        def __init__(self, **kwargs) -> None:
            """Store creation kwargs for assertions.

            :param kwargs: Constructor arguments.
            """
            self.kwargs = kwargs
            self.options = None
            self.nodes = []
            self.edges = []
            self.saved_path = None
            FakeNetwork.instances.append(self)

        def set_options(self, options: str) -> None:
            """Store physics options string.

            :param str options: JSON options payload.
            """
            self.options = options

        def add_node(self, node_id: str, **kwargs) -> None:
            """Store node payload.

            :param str node_id: Node identifier.
            :param kwargs: Node options.
            """
            self.nodes.append((node_id, kwargs))

        def add_edge(self, source: str, target: str, **kwargs) -> None:
            """Store edge payload.

            :param str source: Source node.
            :param str target: Target node.
            :param kwargs: Edge options.
            """
            self.edges.append((source, target, kwargs))

        def save_graph(self, path: str) -> None:
            """Write a simple marker file.

            :param str path: Output path.
            """
            self.saved_path = path
            Path(path).write_text("<html>fake</html>")

    fake_pyvis = types.ModuleType("pyvis")
    fake_pyvis_network = types.ModuleType("pyvis.network")
    fake_pyvis_network.Network = FakeNetwork
    fake_pyvis.network = fake_pyvis_network

    monkeypatch.setitem(sys.modules, "pyvis", fake_pyvis)
    monkeypatch.setitem(sys.modules, "pyvis.network", fake_pyvis_network)

    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id, theme_name="dark")
    out_path = tmp_path / "graph.html"
    exporter.to_interactive_html(out_path, physics=True)

    instance = FakeNetwork.instances[-1]
    assert out_path.exists()
    assert instance.options is not None
    assert len(instance.nodes) == 2
    assert len(instance.edges) == 1
    assert [node_id for node_id, _ in instance.nodes] == ["related", "seed"]


def test_exporter_plotly_raises_without_plotly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plotly export should error cleanly when plotly is unavailable."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id)

    monkeypatch.setitem(sys.modules, "plotly", None)
    with pytest.raises(RuntimeError, match="plotly is required"):
        exporter.to_plotly_html(tmp_path / "graph.plotly.html")


def test_exporter_plotly_with_fake_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plotly export should write output using a minimal graph_objects API."""

    captured: dict[str, object] = {}

    class FakeFigure:
        """Minimal plotly Figure stand-in."""

        def __init__(self, data, layout) -> None:
            """Store payload for assertions.

            :param data: Figure data traces.
            :param layout: Figure layout spec.
            """
            self.data = data
            self.layout = layout
            captured["data"] = data
            captured["layout"] = layout

        def write_html(self, path: str) -> None:
            """Write a simple marker HTML file.

            :param str path: Output path.
            """
            Path(path).write_text("<html>plotly</html>")

    fake_go = types.SimpleNamespace(
        Scatter=lambda **kwargs: {"type": "scatter", **kwargs},
        Layout=lambda **kwargs: {"type": "layout", **kwargs},
        Figure=FakeFigure,
    )
    fake_plotly = types.ModuleType("plotly")
    fake_plotly.graph_objects = fake_go
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)

    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph, seed_id, layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)}
    )
    out_path = tmp_path / "graph.plotly.html"
    exporter.to_plotly_html(out_path)
    assert out_path.exists()
    node_trace = captured["data"][1]
    assert list(node_trace["text"]) == ["Related Paper", "Smith, 2020"]


def test_exporter_plotly_with_missing_year_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plotly export should avoid None in marker colors when year data is missing."""

    captured: dict[str, list[object]] = {}

    class FakeFigure:
        """Minimal plotly Figure stand-in."""

        def __init__(self, data, layout) -> None:
            """Store payload for assertions."""
            self.data = data
            self.layout = layout

        def write_html(self, path: str) -> None:
            """Write a simple marker HTML file."""
            Path(path).write_text("<html>plotly</html>")

    def fake_scatter(**kwargs) -> dict:
        if kwargs.get("mode") == "markers+text":
            captured["marker"] = list(kwargs["marker"]["color"])
        return {"type": "scatter", **kwargs}

    fake_go = types.SimpleNamespace(
        Scatter=fake_scatter,
        Layout=lambda **kwargs: {"type": "layout", **kwargs},
        Figure=FakeFigure,
    )
    fake_plotly = types.ModuleType("plotly")
    fake_plotly.graph_objects = fake_go
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)

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
    out_path = tmp_path / "graph.plotly.html"
    exporter.to_plotly_html(out_path)

    assert out_path.exists()
    marker_colors = captured["marker"]
    assert None not in marker_colors
    assert 0 in marker_colors


def test_exporter_graphml_with_missing_year(tmp_path: Path) -> None:
    """GraphML export should serialize missing year values as a numeric default."""
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
    graph.add_node(
        "missing-year",
        title="No Year",
        citation_count=0,
        authors=[],
    )
    graph.add_edge("seed", "missing-year")

    exporter = GraphExporter(graph, "seed")
    graphml_path = tmp_path / "graph.graphml"
    exporter.to_graphml(graphml_path)

    graphml = nx.read_graphml(graphml_path)
    assert str(graphml.nodes["missing-year"]["year"]) == "0"


def test_exporter_json_and_graphml_ordering_is_stable(tmp_path: Path) -> None:
    """Serialization should sort nodes/edges regardless of insertion order."""
    graph = nx.Graph()
    graph.add_node("z", title="Node Z", year=2022, authors=[], citation_count=0)
    graph.add_node("seed", title="Seed Paper", year=2020, authors=[], is_seed=True)
    graph.add_node("a", title="Node A", year=2021, authors=[], citation_count=0)
    graph.add_edge("z", "a", weight=0.5)
    graph.add_edge("seed", "z", weight=0.7)

    exporter = GraphExporter(graph, "seed")
    json_path = tmp_path / "ordered.json"
    graphml_path = tmp_path / "ordered.graphml"
    exporter.to_json(json_path)
    exporter.to_graphml(graphml_path)

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


def test_exporter_plotly_edge_order_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plotly edge trace ordering should follow canonicalized edge order."""
    captured: dict[str, object] = {}

    class FakeFigure:
        """Minimal plotly Figure stand-in."""

        def __init__(self, data, layout) -> None:
            """Store payload for assertions."""
            self.data = data
            self.layout = layout
            captured["data"] = data

        def write_html(self, path: str) -> None:
            """Write a simple marker HTML file."""
            Path(path).write_text("<html>plotly</html>")

    fake_go = types.SimpleNamespace(
        Scatter=lambda **kwargs: {"type": "scatter", **kwargs},
        Layout=lambda **kwargs: {"type": "layout", **kwargs},
        Figure=FakeFigure,
    )
    fake_plotly = types.ModuleType("plotly")
    fake_plotly.graph_objects = fake_go
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)

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
    out_path = tmp_path / "ordered.plotly.html"
    exporter.to_plotly_html(out_path)

    assert out_path.exists()
    edge_trace = captured["data"][0]
    assert list(edge_trace["x"]) == [0.0, 2.0, None, 1.0, 2.0, None]
