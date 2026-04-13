"""Consolidated tests for visualization rendering and export contracts."""

from __future__ import annotations

import json
import re
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Hashable

import networkx as nx
import numpy as np
import pytest

from citemesh.core import Author, Paper
from citemesh.data.model_profiles import get_embedding_model_profile
from citemesh.visualization import export as export_module
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
from tests._helpers import raise_import_error


def _install_fake_plotly(
    monkeypatch: pytest.MonkeyPatch,
    *,
    figure_cls: type,
    scatter_factory: Callable[..., dict[str, object]] | None = None,
    plotly_js: str = "window.Plotly={};",
) -> None:
    """Install a minimal ``plotly`` module with configurable graph_objects types."""
    fake_go = types.SimpleNamespace(
        Scatter=scatter_factory or (lambda **kwargs: {"type": "scatter", **kwargs}),
        Layout=lambda **kwargs: {"type": "layout", **kwargs},
        Figure=figure_cls,
    )
    monkeypatch.setattr(export_module, "_load_plotly_graph_objects", lambda: fake_go)
    monkeypatch.setattr(
        export_module,
        "_load_plotly_dashboard_runtime",
        lambda: (fake_go, lambda: plotly_js),
    )


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
        venue="TestConf",
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
        venue="Related Journal",
        categories=["cs.LG"],
    )

    graph = nx.Graph()
    graph.graph["strategy"] = "citation"
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
        paper=related,
        title=related.title,
        year=related.year,
        authors=[author.name for author in related.authors],
        citation_count=related.citation_count,
        venue=related.venue,
        is_seed=False,
    )
    graph.add_edge(seed.paper_id, related.paper_id, weight=0.7)
    return graph, seed.paper_id


def test_exporter_serialization_contracts_and_determinism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exporter outputs should preserve metadata, ordering, and determinism."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph,
        seed_id,
        metadata={"strategy": "citation"},
        layout={"seed": (0.0, 0.0), "related": (1.0, 0.0)},
    )

    json_path = tmp_path / "graph.json"
    graphml_path = tmp_path / "graph.graphml"
    graphml_again_path = tmp_path / "graph-again.graphml"

    exporter.to_json(json_path)
    exporter.to_graphml(graphml_path)
    exporter.to_graphml(graphml_again_path)

    payload = json.loads(json_path.read_text())
    assert payload["seed_id"] == seed_id
    assert "metadata" not in payload
    assert payload["summary"] == {"nodes": 2, "edges": 1}
    assert "dashboard" in payload
    assert payload["dashboard"]["meta"]["seed_id"] == seed_id
    assert len(payload["dashboard"]["meta"]["plotly_node_order"]) == 2
    assert len(payload["dashboard"]["meta"]["plotly_positions"]) == 2
    assert len(payload["dashboard"]["meta"]["plotly_node_sizes"]) == 2
    assert len(payload["nodes"]) == 2
    assert payload["edges"][0]["weight"] == pytest.approx(0.7)
    assert payload["edges"][0]["source_title"] == "Related Paper"
    assert payload["edges"][0]["target_title"] == "Seed Paper"

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

    captured: dict[str, object] = {}

    class FakeFigure(_BaseFakeFigure):
        def __init__(self, data: Any, layout: Any) -> None:
            super().__init__(data, layout)
            captured["data"] = data

    _install_fake_plotly(monkeypatch, figure_cls=FakeFigure)

    ordered_graph = nx.Graph()
    ordered_graph.add_node("z", title="Node Z", year=2022, authors=[], citation_count=0)
    ordered_graph.add_node("seed", title="Seed", year=2020, authors=[], is_seed=True)
    ordered_graph.add_node("a", title="Node A", year=2021, authors=[], citation_count=0)
    ordered_graph.add_edge("seed", "z", weight=0.7)
    ordered_graph.add_edge("z", "a", weight=0.5)

    ordered_exporter = GraphExporter(
        ordered_graph,
        "seed",
        layout={"a": (0.0, 0.0), "seed": (1.0, 0.0), "z": (2.0, 0.0)},
    )
    ordered_json_path = tmp_path / "ordered.json"
    ordered_graphml_path = tmp_path / "ordered.graphml"
    ordered_plotly_path = tmp_path / "ordered.plotly.html"
    ordered_exporter.to_json(ordered_json_path)
    ordered_exporter.to_graphml(ordered_graphml_path)
    ordered_exporter.to_plotly_html(ordered_plotly_path)

    ordered_payload = json.loads(ordered_json_path.read_text())
    assert [node["id"] for node in ordered_payload["nodes"]] == ["a", "seed", "z"]
    assert [edge["source"] for edge in ordered_payload["edges"]] == ["a", "seed"]
    assert [edge["target"] for edge in ordered_payload["edges"]] == ["z", "z"]
    assert [edge["weight"] for edge in ordered_payload["edges"]] == [
        pytest.approx(0.5),
        pytest.approx(0.7),
    ]
    assert ordered_payload["edges"][0]["source_title"] == "Node A"
    assert ordered_payload["edges"][0]["target_title"] == "Node Z"

    graphml_xml = ET.fromstring(ordered_graphml_path.read_text())
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
    normalized_layout = ordered_exporter._get_layout()
    edge_x = list(edge_trace["x"])
    assert edge_x[2] is None
    assert edge_x[5] is None
    assert edge_x[0] == pytest.approx(float(normalized_layout["a"][0]))
    assert edge_x[1] == pytest.approx(float(normalized_layout["z"][0]))
    assert edge_x[3] == pytest.approx(float(normalized_layout["seed"][0]))
    assert edge_x[4] == pytest.approx(float(normalized_layout["z"][0]))


def test_json_export_skips_layout_without_precomputed_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSON export should stay data-only unless layout geometry already exists."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id, metadata={"strategy": "citation"})

    def _fail_compute_layout(
        *args: Any, **kwargs: Any
    ) -> dict[str, tuple[float, float]]:
        del args, kwargs
        raise AssertionError("compute_layout should not run for JSON-only export")

    monkeypatch.setattr(export_module, "compute_layout", _fail_compute_layout)

    json_path = tmp_path / "graph.json"
    exporter.to_json(json_path)

    payload = json.loads(json_path.read_text())
    assert payload["meta"]["strategy"] == "citation"
    assert payload["summary"] == {"nodes": 2, "edges": 1}
    assert payload["dashboard"]["meta"]["summary"] == {"nodes": 2, "edges": 1}
    assert "plotly_node_order" not in payload["dashboard"]["meta"]
    assert "plotly_positions" not in payload["dashboard"]["meta"]
    assert "plotly_node_sizes" not in payload["dashboard"]["meta"]


def test_exporter_interactive_html_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interactive HTML should fail clearly without pyvis and succeed with a stub."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id)

    monkeypatch.setattr(export_module, "_load_pyvis_network_class", raise_import_error)
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

    monkeypatch.setattr(export_module, "_load_pyvis_network_class", lambda: FakeNetwork)

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

    monkeypatch.setattr(export_module, "_load_plotly_graph_objects", raise_import_error)
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
    assert list(node_trace["text"]) == ["Jones, 2021", "Smith, 2020"]
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


def _extract_dashboard_script_json(html_text: str, script_id: str) -> dict[str, Any]:
    """Extract embedded JSON payload from a dashboard script tag."""

    match = re.search(
        rf'<script id="{re.escape(script_id)}" type="application/json">(.*?)</script>',
        html_text,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def _extract_inline_script_bodies(html_text: str) -> list[str]:
    """Extract bare inline script bodies in source order."""

    return re.findall(r"<script>(.*?)</script>", html_text, flags=re.DOTALL)


def test_exporter_dashboard_contracts(tmp_path: Path) -> None:
    """Dashboard export should render tri-pane shell and derived payload fields."""
    pytest.importorskip("plotly")

    graph, seed_id = _build_graph()
    graph.graph["paper_sources"] = {"related": "semantic", "seed": "citation"}
    exporter = GraphExporter(
        graph,
        seed_id,
        metadata={
            "strategy": "hybrid",
            "dashboard_collection": {
                "current_result_id": "hybrid:seed",
                "results": [
                    {
                        "result_id": "hybrid:seed",
                        "seed_id": "seed",
                        "title": "Seed Paper",
                        "strategy": "hybrid",
                        "summary": {"nodes": 2, "edges": 1},
                    }
                ],
                "payloads": {
                    "hybrid:seed": {
                        "seed_id": "seed",
                        "meta": {"strategy": "hybrid"},
                        "summary": {"nodes": 2, "edges": 1},
                        "nodes": [],
                        "edges": [],
                        "dashboard": {"meta": {}},
                    }
                },
            },
        },
        layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)},
    )
    out_path = tmp_path / "graph.dashboard.html"
    exporter.to_dashboard_html(out_path, theme="dark")

    assert out_path.exists()
    rendered = out_path.read_text()
    for token in [
        'id="global-nav"',
        'id="filters-toggle"',
        'id="detail-why-lines"',
        'id="dashboard-root"',
        'id="paper-list-pane"',
        'id="graph-pane"',
        'id="detail-pane"',
        'id="citemesh-dashboard-data"',
        'id="citemesh-dashboard-figure"',
        'id="citemesh-dashboard-collection"',
        'id="result-select"',
        'id="dashboard-status"',
        'accept=".json,.html"',
    ]:
        assert token in rendered
    for css_token in [
        "html, body {\n      margin: 0;\n      height: 100%;\n      overflow: hidden;",
        "#dashboard-root {\n      display: grid;\n      gap: 12px;\n      padding: 12px;\n      flex: 1 1 auto;",
        "#paper-list {\n      margin: 0;\n      padding: 0;\n      list-style: none;\n      overflow-y: auto;",
        "#detail-content {\n      padding: 14px 13px 12px;\n      flex: 1;\n      min-height: 0;\n      display: flex;\n      flex-direction: column;\n      gap: 16px;\n      overflow-y: auto;",
        "width: 100%;\n      height: 100%;\n      min-height: 0;",
    ]:
        assert css_token in rendered

    payload = _extract_dashboard_script_json(rendered, "citemesh-dashboard-data")
    assert payload["meta"]["seed_id"] == "seed"
    assert payload["meta"]["strategy"] == "hybrid"
    assert payload["meta"]["summary"] == {"nodes": 2, "edges": 1}
    assert payload["meta"]["plotly_node_order"] == ["related", "seed"]
    assert len(payload["meta"]["plotly_positions"]) == 2
    assert len(payload["meta"]["plotly_node_sizes"]) == 2
    seed_node = next(node for node in payload["nodes"] if node["id"] == "seed")
    related_node = next(node for node in payload["nodes"] if node["id"] == "related")
    assert seed_node["provenance"] == "seed"
    assert seed_node["provenance_base"] == "citation"
    assert "arxiv_id" in seed_node
    assert "doi" in seed_node
    assert seed_node["venue"] == "TestConf"
    assert related_node["provenance"] == "semantic"
    assert related_node["venue"] == "Related Journal"
    assert related_node["seed_relation"] == "semantic_only"
    assert "seed_relevance" in seed_node
    assert isinstance(seed_node["seed_relevance"], float)
    assert seed_node["links"]["semantic_scholar"] is not None
    assert isinstance(seed_node["bibtex"], str)

    collection = _extract_dashboard_script_json(
        rendered, "citemesh-dashboard-collection"
    )
    assert collection["current_result_id"] == "hybrid:seed"
    assert collection["results"][0]["title"] == "Seed Paper"
    assert "hybrid:seed" in collection["payloads"]

    figure = _extract_dashboard_script_json(rendered, "citemesh-dashboard-figure")
    assert len(figure["data"]) == 3
    assert len(figure["layout"].get("shapes", [])) == 1
    assert figure["layout"]["uirevision"] == "citemesh-dashboard-static-layout-v1"
    assert figure["layout"]["xaxis"]["autorange"] is False
    assert figure["layout"]["yaxis"]["autorange"] is False
    assert len(figure["layout"]["xaxis"]["range"]) == 2
    assert len(figure["layout"]["yaxis"]["range"]) == 2
    edge_shape = figure["layout"]["shapes"][0]
    assert edge_shape["type"] == "path"
    assert " Q " in edge_shape["path"]
    halo_trace = next(
        trace for trace in figure["data"] if trace.get("name") == "selection-halo"
    )
    neighborhood_trace = next(
        trace for trace in figure["data"] if trace.get("name") == "neighborhood-edges"
    )
    node_trace = next(trace for trace in figure["data"] if trace.get("name") == "nodes")
    halo_marker = halo_trace["marker"]
    assert halo_trace["mode"] == "markers"
    assert halo_trace["hoverinfo"] == "none"
    assert halo_marker["line"]["width"] == 0
    assert neighborhood_trace["mode"] == "lines"
    assert neighborhood_trace["x"] == []
    assert neighborhood_trace["y"] == []
    marker = node_trace["marker"]
    assert marker["showscale"] is False
    assert marker["sizemode"] == "area"
    assert marker["sizeref"] > 0
    assert max(marker["line"]["width"]) >= 4
    assert min(marker["line"]["width"]) == 0


def test_exporter_dashboard_runtime_script_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dashboard runtime should stay intact and parse imported HTML safely."""

    class FakeFigure(_BaseFakeFigure):
        def to_plotly_json(self) -> dict[str, object]:
            return {"data": self.data, "layout": self.layout}

    _install_fake_plotly(monkeypatch, figure_cls=FakeFigure)

    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph,
        seed_id,
        layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)},
    )
    out_path = tmp_path / "runtime.dashboard.html"
    exporter.to_dashboard_html(out_path)

    runtime_scripts = _extract_inline_script_bodies(out_path.read_text())
    assert len(runtime_scripts) == 2

    runtime_script = runtime_scripts[-1]
    assert "new DOMParser()" in runtime_script
    assert "function parseImportedPayloadFromText" in runtime_script
    assert "function hasCompleteDashboardGeometry" in runtime_script
    assert "escapeRegExp" not in runtime_script


def test_exporter_dashboard_missing_plotly_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dashboard export should fail clearly when plotly dependency is missing."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id)

    monkeypatch.setattr(
        export_module, "_load_plotly_dashboard_runtime", raise_import_error
    )
    with pytest.raises(RuntimeError, match="plotly is required for Dashboard export"):
        exporter.to_dashboard_html(tmp_path / "missing.dashboard.html")


def test_exporter_dashboard_link_derivation_contracts(tmp_path: Path) -> None:
    """Dashboard payload should derive arXiv/DOI/S2 links from IDs and metadata."""
    pytest.importorskip("plotly")

    graph = nx.Graph()
    graph.add_node(
        "arxiv:2411.03884",
        title="Seed",
        year=2024,
        authors=["A"],
        citation_count=10,
        is_seed=True,
    )
    graph.add_node(
        "10.1145/3133956.3134029",
        title="DOI Paper",
        year=2017,
        authors=["B"],
        citation_count=5,
        is_seed=False,
    )
    graph.add_node(
        "abcdef123456",
        title="S2 Paper",
        year=2018,
        authors=["C"],
        citation_count=1,
        is_seed=False,
    )
    graph.add_node(
        "s2-candidate-arxiv",
        title="S2 with arXiv external ID",
        year=2024,
        authors=["D"],
        citation_count=2,
        arxiv_id="2501.00001v3",
        is_seed=False,
    )
    graph.add_node(
        "s2-candidate-doi",
        title="S2 with DOI external ID",
        year=2022,
        authors=["E"],
        citation_count=3,
        doi="10.1109/5.771073",
        is_seed=False,
    )
    graph.add_edge("arxiv:2411.03884", "10.1145/3133956.3134029", weight=0.9)
    graph.add_edge("arxiv:2411.03884", "abcdef123456", weight=0.7)
    graph.add_edge("arxiv:2411.03884", "s2-candidate-arxiv", weight=0.8)
    graph.add_edge("arxiv:2411.03884", "s2-candidate-doi", weight=0.75)

    exporter = GraphExporter(
        graph,
        "arxiv:2411.03884",
        metadata={"strategy": "citation"},
        layout={
            "arxiv:2411.03884": (0.0, 0.0),
            "10.1145/3133956.3134029": (1.0, 0.0),
            "abcdef123456": (0.0, 1.0),
            "s2-candidate-arxiv": (-1.0, 0.0),
            "s2-candidate-doi": (0.0, -1.0),
        },
    )
    out_path = tmp_path / "links.dashboard.html"
    exporter.to_dashboard_html(out_path, theme="light")

    payload = _extract_dashboard_script_json(
        out_path.read_text(), "citemesh-dashboard-data"
    )
    nodes = {node["id"]: node for node in payload["nodes"]}

    arxiv_links = nodes["arxiv:2411.03884"]["links"]
    assert arxiv_links["arxiv_abs"] == "https://arxiv.org/abs/2411.03884"
    assert arxiv_links["arxiv_pdf"] == "https://arxiv.org/pdf/2411.03884.pdf"
    assert arxiv_links["doi"] is None
    assert "semanticscholar.org" in arxiv_links["semantic_scholar"]

    doi_links = nodes["10.1145/3133956.3134029"]["links"]
    assert doi_links["doi"] == "https://doi.org/10.1145/3133956.3134029"
    assert doi_links["arxiv_abs"] is None

    s2_links = nodes["abcdef123456"]["links"]
    assert s2_links["arxiv_abs"] is None
    assert s2_links["doi"] is None
    assert s2_links["semantic_scholar"] == (
        "https://www.semanticscholar.org/paper/abcdef123456"
    )

    s2_arxiv_links = nodes["s2-candidate-arxiv"]["links"]
    assert s2_arxiv_links["arxiv_abs"] == "https://arxiv.org/abs/2501.00001"
    assert s2_arxiv_links["arxiv_pdf"] == "https://arxiv.org/pdf/2501.00001.pdf"
    assert s2_arxiv_links["doi"] is None

    s2_doi_links = nodes["s2-candidate-doi"]["links"]
    assert s2_doi_links["doi"] == "https://doi.org/10.1109/5.771073"
    assert s2_doi_links["arxiv_abs"] is None


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


def test_missing_year_visual_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing-year fallbacks should stay stable across exporters and render helpers."""
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


def test_exporter_html_exports_are_byte_stable_with_real_plotly(
    tmp_path: Path,
) -> None:
    """Plotly and dashboard HTML exports should be byte-stable for identical inputs."""
    pytest.importorskip("plotly")
    layout = {"seed": (0.0, 0.0), "related": (1.0, 1.0)}
    scenarios = [
        ("plotly", "to_plotly_html", None, None),
        (
            "dashboard",
            "to_dashboard_html",
            {"strategy": "hybrid"},
            {"related": "semantic", "seed": "citation"},
        ),
    ]

    for name, method_name, metadata, paper_sources in scenarios:
        graph, seed_id = _build_graph()
        if paper_sources is not None:
            graph.graph["paper_sources"] = paper_sources
        exporter = GraphExporter(graph, seed_id, metadata=metadata, layout=layout)
        out_a = tmp_path / f"first.{name}.html"
        out_b = tmp_path / f"second.{name}.html"
        getattr(exporter, method_name)(out_a)
        getattr(exporter, method_name)(out_b)
        assert out_a.read_text() == out_b.read_text()


def test_layout_positioning_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layout normalization, perturbation stability, and distance weighting should hold."""

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


def test_layout_applies_community_separation_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Community-aware layout should separate detected clusters before normalization."""
    graph = nx.Graph()
    graph.add_edge("a1", "a2", weight=0.9)
    graph.add_edge("b1", "b2", weight=0.9)
    graph.add_edge("a1", "b1", weight=0.01)

    monkeypatch.setattr(
        "citemesh.visualization.render.nx.algorithms.community.greedy_modularity_communities",
        lambda *_args, **_kwargs: [set(["a1", "a2"]), set(["b1", "b2"])],
    )
    monkeypatch.setattr(
        "citemesh.visualization.render.nx.kamada_kawai_layout",
        lambda layout_graph, **_kwargs: {
            node: np.array([0.0, 0.0], dtype=np.float64)
            for node in layout_graph.nodes()
        },
    )

    def fake_spring_layout(
        layout_graph: nx.Graph, **_kwargs: Any
    ) -> dict[int, np.ndarray]:
        if all(isinstance(node, int) for node in layout_graph.nodes()):
            return {
                0: np.array([-1.0, 0.0], dtype=np.float64),
                1: np.array([1.0, 0.0], dtype=np.float64),
            }
        return {
            node: np.array([0.0, 0.0], dtype=np.float64)
            for node in layout_graph.nodes()
        }

    monkeypatch.setattr(
        "citemesh.visualization.render.nx.spring_layout", fake_spring_layout
    )

    pos = compute_layout(graph, iterations=20, layout_seed=99)
    left_center = np.mean([pos["a1"], pos["a2"]], axis=0)
    right_center = np.mean([pos["b1"], pos["b2"]], axis=0)
    assert left_center[0] < right_center[0]


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


def test_exporter_enriched_json_csv_bibtex(tmp_path: Path) -> None:
    """JSON export should include enriched fields; CSV and BibTeX should work."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id, metadata={"strategy": "hybrid"})

    json_path = tmp_path / "enriched.json"
    csv_path = tmp_path / "enriched.csv"
    bib_path = tmp_path / "enriched.bib"

    exporter.to_json(json_path)
    exporter.to_csv(csv_path)
    exporter.to_bibtex(bib_path)

    # --- JSON enrichment ---
    payload = json.loads(json_path.read_text())
    assert "meta" in payload
    assert payload["meta"]["strategy"] == "hybrid"
    assert "year_range" in payload["meta"]
    nodes = payload["nodes"]
    assert len(nodes) == 2
    seed_node = next(n for n in nodes if n["is_seed"])
    assert "provenance" in seed_node
    assert "seed_relevance" in seed_node
    assert isinstance(seed_node["seed_relevance"], float)
    assert "links" in seed_node
    assert "bibtex" in seed_node
    assert seed_node["provenance"] == "seed"
    assert seed_node["seed_relation"] == "seed"

    related_node = next(n for n in nodes if not n["is_seed"])
    assert related_node["provenance"] in {"citation", "semantic", "both"}
    assert "links" in related_node
    assert "semantic_scholar" in related_node["links"]

    # --- CSV ---
    csv_text = csv_path.read_text()
    lines = csv_text.strip().split("\n")
    assert len(lines) == 3  # header + 2 papers
    header = lines[0]
    assert "provenance" in header
    assert "seed_relevance" in header
    assert "arxiv_url" in header

    # --- BibTeX ---
    bib_text = bib_path.read_text()
    assert "@article{" in bib_text
    assert "Seed Paper" in bib_text or "Related Paper" in bib_text


def test_recommendation_export_defaults_to_semantic_provenance(
    tmp_path: Path,
) -> None:
    """Recommendation exports should classify fallback provenance as semantic."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id, metadata={"strategy": "recommendation"})

    json_path = tmp_path / "recommendation.json"
    exporter.to_json(json_path)

    payload = json.loads(json_path.read_text())
    seed_node = next(node for node in payload["nodes"] if node["id"] == seed_id)
    related_node = next(node for node in payload["nodes"] if node["id"] == "related")

    assert seed_node["provenance"] == "seed"
    assert seed_node["provenance_base"] == "semantic"
    assert seed_node["seed_relation"] == "seed"
    assert related_node["provenance"] == "semantic"
    assert related_node["provenance_base"] == "semantic"
    assert related_node["seed_relation"] == "semantic_only"


@pytest.mark.parametrize("strategy", ["recommendation", "embedding"])
def test_semantic_export_uses_graph_strategy_when_metadata_is_omitted(
    tmp_path: Path, strategy: str
) -> None:
    """Semantic exports should stay correct when callers omit exporter metadata."""
    graph, seed_id = _build_graph()
    graph.graph["strategy"] = strategy
    if strategy == "embedding":
        graph.graph["embedding_runtime"] = {"storage_precision": "int8"}

    exporter = GraphExporter(graph, seed_id)
    json_path = tmp_path / f"{strategy}.json"
    exporter.to_json(json_path)

    payload = json.loads(json_path.read_text())
    assert payload["meta"]["strategy"] == strategy

    seed_node = next(node for node in payload["nodes"] if node["id"] == seed_id)
    related_node = next(node for node in payload["nodes"] if node["id"] == "related")

    assert seed_node["provenance"] == "seed"
    assert seed_node["provenance_base"] == "semantic"
    assert seed_node["seed_relation"] == "seed"
    assert related_node["provenance"] == "semantic"
    assert related_node["provenance_base"] == "semantic"
    assert related_node["seed_relation"] == "semantic_only"
