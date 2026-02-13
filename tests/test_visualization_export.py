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
from citemesh.visualization.export import (
    GRAPHML_DETERMINISM_POLICY_STRICT,
    GRAPHML_LAYOUT_METADATA_KEY,
    GRAPHML_LAYOUT_VERSION_KEY,
    GraphExporter,
    _graphml_determinism_policy,
)


def _install_fake_plotly(
    monkeypatch: pytest.MonkeyPatch,
    *,
    figure_cls: type,
    scatter_factory=None,
) -> None:
    """Install a minimal ``plotly`` module with configurable graph_objects types.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to patch ``sys.modules``.
    :param type figure_cls: Figure class replacement used by exporter tests.
    :param callable scatter_factory: Optional ``Scatter`` constructor.
    :return None: Installs fake plotly modules for the current test.
    """
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

    def __init__(self, data, layout) -> None:
        """Store figure payload for assertion helpers."""
        self.data = data
        self.layout = layout

    def write_html(self, path: str, **kwargs) -> None:
        """Write deterministic marker output and ignore optional kwargs."""
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


def test_graphml_export_is_deterministic_across_runs(tmp_path: Path) -> None:
    """Repeated GraphML exports should be deterministically equivalent."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id, metadata={"strategy": "citation"})

    primary = tmp_path / "graph-a.graphml"
    secondary = tmp_path / "graph-b.graphml"
    exporter.to_graphml(primary)
    exporter.to_graphml(secondary)

    policy = _graphml_determinism_policy()
    if policy == GRAPHML_DETERMINISM_POLICY_STRICT:
        assert primary.read_text() == secondary.read_text()
    else:
        assert _canonicalize_graphml(primary) == _canonicalize_graphml(secondary)


def test_graphml_export_records_determinism_metadata(
    tmp_path: Path,
) -> None:
    """GraphML output should expose deterministic serialization policy metadata."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id, metadata={"strategy": "citation"})

    path = tmp_path / "graph.graphml"
    exporter.to_graphml(path)
    graphml = nx.read_graphml(path)

    policy = _graphml_determinism_policy()
    assert graphml.graph[GRAPHML_LAYOUT_METADATA_KEY] == policy
    assert graphml.graph[GRAPHML_LAYOUT_VERSION_KEY] == nx.__version__


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

    class FakeFigure(_BaseFakeFigure):
        """Minimal plotly Figure stand-in."""

        def __init__(self, data, layout) -> None:
            """Store payload for assertions.

            :param data: Figure data traces.
            :param layout: Figure layout spec.
            """
            super().__init__(data, layout)
            captured["data"] = data
            captured["layout"] = layout

    _install_fake_plotly(monkeypatch, figure_cls=FakeFigure)

    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph, seed_id, layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)}
    )
    out_path = tmp_path / "graph.plotly.html"
    exporter.to_plotly_html(out_path)
    assert out_path.exists()
    node_trace = captured["data"][1]
    assert list(node_trace["text"]) == ["Related Paper", "Smith, 2020"]


def test_exporter_plotly_requires_div_id_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plotly export should fail fast when deterministic div_id is unsupported."""

    class FakeFigure:
        """Minimal plotly Figure stand-in that rejects ``div_id``."""

        def __init__(self, data, layout) -> None:
            del data
            del layout

        def write_html(self, path: str, **kwargs) -> None:
            del path
            if "div_id" in kwargs:
                raise TypeError("div_id unsupported")

    _install_fake_plotly(monkeypatch, figure_cls=FakeFigure)

    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph, seed_id, layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)}
    )

    with pytest.raises(RuntimeError, match="Deterministic Plotly export requires"):
        exporter.to_plotly_html(tmp_path / "graph.plotly.html")


def test_exporter_plotly_uses_deterministic_div_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plotly HTML export should provide a stable div id when supported."""
    captured: dict[str, object] = {}

    class FakeFigure(_BaseFakeFigure):
        """Minimal plotly Figure stand-in."""

        def write_html(self, path: str, **kwargs) -> None:
            """Capture div_id kwargs and write marker output."""
            captured["kwargs"] = kwargs
            Path(path).write_text("<html>plotly</html>")

    _install_fake_plotly(monkeypatch, figure_cls=FakeFigure)

    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph, seed_id, layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)}
    )
    out_path = tmp_path / "graph.plotly.html"
    exporter.to_plotly_html(out_path)

    assert out_path.exists()
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["div_id"] == exporter._plotly_div_id()


def test_exporter_plotly_with_missing_year_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plotly export should avoid None in marker colors when year data is missing."""

    captured: dict[str, object] = {}

    def fake_scatter(**kwargs) -> dict:
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
    out_path = tmp_path / "graph.plotly.html"
    exporter.to_plotly_html(out_path)

    assert out_path.exists()
    marker = captured["marker"]
    assert isinstance(marker, dict)
    marker_colors = marker["color"]
    assert None not in marker_colors
    assert 0 not in marker_colors
    assert marker["cmin"] == 2000.0
    assert marker["cmax"] == 2001.0


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

    class FakeFigure(_BaseFakeFigure):
        """Minimal plotly Figure stand-in."""

        def __init__(self, data, layout) -> None:
            """Store payload for assertions."""
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
    out_path = tmp_path / "ordered.plotly.html"
    exporter.to_plotly_html(out_path)

    assert out_path.exists()
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
