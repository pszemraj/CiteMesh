"""Consolidated tests for visualization rendering and export contracts."""

from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
from collections.abc import Callable, Hashable
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

from citemesh.core import Author, Paper
from citemesh.data import cache as cache_module
from citemesh.data.model_profiles import get_embedding_model_profile
from citemesh.services.semantic_scholar import SemanticScholarClient
from citemesh.visualization import export as export_module
from citemesh.visualization import render as render_module
from citemesh.visualization import themes as themes_module
from citemesh.visualization.dashboard.contracts import (
    DASHBOARD_COLLECTION_KIND,
    DASHBOARD_COLLECTION_SCHEMA_VERSION,
    GRAPH_PAYLOAD_KIND,
    GRAPH_PAYLOAD_SCHEMA_VERSION,
)
from citemesh.visualization.dashboard.package import (
    _validate_dashboard_graph_payload,
    render_dashboard_collection_snapshot,
    update_dashboard_package,
)
from citemesh.visualization.export import (
    DASHBOARD_AXIS_MIN_PADDING,
    DASHBOARD_AXIS_X_PADDING,
    DASHBOARD_FOOTER_MARGIN,
    DASHBOARD_LABEL_CAP,
    DASHBOARD_LABEL_MIN_DISTANCE,
    DASHBOARD_MAX_NODE_DIAMETER,
    GRAPHML_DETERMINISM_POLICY_STRICT,
    GRAPHML_LAYOUT_METADATA_KEY,
    GRAPHML_LAYOUT_VERSION_KEY,
    GraphExporter,
    _graphml_determinism_policy,
    _select_dashboard_label_nodes,
    _stable_curve_direction,
)
from citemesh.visualization.export import geometry as geometry_module
from citemesh.visualization.export import loaders as loaders_module
from citemesh.visualization.export import nodes as nodes_module
from citemesh.visualization.render import (
    KK_LAYOUT_DISTANCE_ATTR,
    MAX_STATIC_NON_SEED_LABELS,
    _layout_viewport_limits,
    _orient_layout_horizontally,
    _pack_disconnected_components,
    _spread_layout_by_communities,
    compute_layout,
    compute_node_colors,
    compute_node_sizes,
    draw_labels,
    normalize_layout_positions,
    visualize_graph,
)
from citemesh.visualization.themes import get_theme
from citemesh.visualization.years import (
    coerce_publication_year,
    publication_year_bounds,
    publication_year_scale,
)
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
    monkeypatch.setattr(loaders_module, "_load_plotly_graph_objects", lambda: fake_go)
    monkeypatch.setattr(
        loaders_module,
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


def _build_hostile_graph(
    *,
    title: str,
    abstract: str = "Seed abstract",
    author: str = "Alice Smith",
) -> tuple[nx.Graph, str]:
    """Create the standard graph with attacker-controlled seed metadata.

    Keeps the exact node attribute shape of :func:`_build_graph` (including the
    ``paper`` object exporters read enriched fields from) so escaping tests
    exercise the real serialization path.

    :param str title: Seed paper title.
    :param str abstract: Seed paper abstract.
    :param str author: Seed paper author name.
    :return tuple[nx.Graph, str]: Graph and seed paper identifier.
    """
    graph, seed_id = _build_graph()
    hostile_paper = replace(
        graph.nodes[seed_id]["paper"],
        title=title,
        abstract=abstract,
        authors=[Author(name=author)],
    )
    graph.nodes[seed_id].update(paper=hostile_paper, title=title, authors=[author])
    return graph, seed_id


def test_exporter_serialization_contracts_and_determinism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exporter outputs should preserve metadata, ordering, and determinism."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph,
        seed_id,
        metadata={
            "strategy": "citation",
            "candidate_source_status": {
                "references": "unavailable",
                "citations": "complete",
            },
        },
        layout={"seed": (0.0, 0.0), "related": (1.0, 0.0)},
    )

    json_path = tmp_path / "graph.json"
    graphml_path = tmp_path / "graph.graphml"
    graphml_again_path = tmp_path / "graph-again.graphml"

    exporter.to_json(json_path)
    exporter.to_graphml(graphml_path)
    exporter.to_graphml(graphml_again_path)

    payload = json.loads(json_path.read_text())
    assert payload["kind"] == "citemesh-graph"
    assert payload["schema_version"] == 1
    assert exporter.graph_payload() == payload
    assert payload["seed_id"] == seed_id
    assert payload["meta"]["candidate_source_status"] == {
        "citations": "complete",
        "references": "unavailable",
    }
    assert "metadata" not in payload
    assert payload["summary"] == {"nodes": 2, "edges": 1}
    assert "dashboard" in payload
    assert payload["dashboard"]["meta"]["seed_id"] == seed_id
    assert payload["dashboard"]["meta"]["candidate_source_status"] == {
        "citations": "complete",
        "references": "unavailable",
    }
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
    assert json.loads(graphml.graph["citemesh_meta_candidate_source_status"]) == {
        "citations": "complete",
        "references": "unavailable",
    }

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


def test_json_export_embeds_geometry_computing_layout_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSON export always embeds dashboard geometry, computing one lazy layout.

    Every ``kind``-stamped payload must be loadable through the dashboard's
    Add Results flow, so a JSON-only export may not omit the geometry arrays.
    """
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id, metadata={"strategy": "citation"})

    layout_calls: list[object] = []

    def _counting_compute_layout(
        graph_arg: Any, *args: Any, **kwargs: Any
    ) -> dict[str, tuple[float, float]]:
        del args, kwargs
        layout_calls.append(graph_arg)
        return {"seed": (0.0, 0.0), "related": (1.0, 1.0)}

    monkeypatch.setattr(nodes_module, "compute_layout", _counting_compute_layout)

    json_path = tmp_path / "graph.json"
    exporter.to_json(json_path)
    exporter.to_json(json_path)

    payload = json.loads(json_path.read_text())
    assert payload["meta"]["strategy"] == "citation"
    assert payload["summary"] == {"nodes": 2, "edges": 1}
    dashboard_meta = payload["dashboard"]["meta"]
    assert dashboard_meta["summary"] == {"nodes": 2, "edges": 1}
    assert sorted(dashboard_meta["plotly_node_order"]) == ["related", "seed"]
    assert len(dashboard_meta["plotly_positions"]) == 2
    assert len(dashboard_meta["plotly_node_sizes"]) == 2
    assert all(size > 0 for size in dashboard_meta["plotly_node_sizes"])
    # The lazily computed layout is memoized across repeated exports.
    assert len(layout_calls) == 1


def test_json_payload_without_layout_passes_dashboard_import_contract() -> None:
    """A layout-free JSON payload must satisfy the strict dashboard import contract.

    ``_validate_dashboard_graph_payload`` mirrors the dashboard's JS importer, so
    running it here locks the JSON -> Load Results round trip for exporters that
    were never handed a run layout.
    """
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id, metadata={"strategy": "citation"})

    payload = exporter.graph_payload()
    validated = _validate_dashboard_graph_payload(
        payload, result_id=f"citation:{seed_id}"
    )

    assert isinstance(validated, dict)
    dashboard_meta = payload["dashboard"]["meta"]
    for geometry_key in (
        "plotly_node_order",
        "plotly_positions",
        "plotly_node_sizes",
    ):
        assert geometry_key in dashboard_meta


def test_exporter_interactive_html_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interactive HTML should fail clearly without pyvis and succeed with a stub."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id)

    monkeypatch.setattr(loaders_module, "_load_pyvis_network_class", raise_import_error)
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

    monkeypatch.setattr(
        loaders_module, "_load_pyvis_network_class", lambda: FakeNetwork
    )

    out_path = tmp_path / "graph.html"
    out_path.write_text("previous", encoding="utf-8")
    out_path.chmod(0o640)
    exporter = GraphExporter(graph, seed_id, theme_name="dark")
    exporter.to_interactive_html(out_path, physics=True)

    instance = FakeNetwork.instances[-1]
    assert out_path.exists()
    assert out_path.stat().st_mode & 0o7777 == 0o640
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

    monkeypatch.setattr(
        loaders_module, "_load_plotly_graph_objects", raise_import_error
    )
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
    out_path.write_text("previous", encoding="utf-8")
    out_path.chmod(0o640)
    exporter.to_plotly_html(out_path)

    assert out_path.exists()
    assert out_path.stat().st_mode & 0o7777 == 0o640
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


def test_plotly_html_survives_empty_and_seedless_graphs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plotly export and auto-naming must not KeyError on empty or seedless graphs."""
    _install_fake_plotly(monkeypatch, figure_cls=_BaseFakeFigure)

    empty_path = tmp_path / "empty.plotly.html"
    GraphExporter(nx.Graph(), "seed").to_plotly_html(empty_path)
    assert empty_path.exists()

    graph, _ = _build_graph()
    seedless_path = tmp_path / "seedless.plotly.html"
    GraphExporter(graph, "absent-seed").to_plotly_html(seedless_path)
    assert seedless_path.exists()

    output_path = render_module.generate_output_path(
        graph, "absent-seed", tmp_path / "out"
    )
    assert output_path.suffix == ".png"
    assert output_path.parent.is_dir()
    for name, candidate, candidate_seed, layout in (
        ("empty", nx.Graph(), "seed", {}),
        ("seedless", graph, "absent-seed", {"seed": (0.0, 0.0), "related": (1.0, 1.0)}),
    ):
        png_path = tmp_path / f"{name}.png"
        visualize_graph(candidate, candidate_seed, png_path, layout=layout, dpi=40)
        assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("existing_label", ["Original Paper Title", "seed"])
def test_output_path_reuses_existing_seed_directory(
    tmp_path: Path, existing_label: str
) -> None:
    """Reuse prior title-based and ID-based directories when metadata changes.

    :param Path tmp_path: Isolated output directory.
    :param str existing_label: Label used by a previous build of the same seed.
    :return None: Checks that the next build preserves the existing artifact path.
    """
    paper_dir = tmp_path / render_module._output_dir_name(existing_label, "seed")
    paper_dir.mkdir()
    artifact = paper_dir / "hybrid.png"
    artifact.write_bytes(b"previous export")
    graph = nx.Graph()
    graph.add_node("seed", title="Corrected Paper Title")

    output_path = render_module.generate_output_path(
        graph, "seed", tmp_path, strategy="hybrid"
    )

    assert output_path == artifact
    assert output_path.read_bytes() == b"previous export"
    assert list(tmp_path.iterdir()) == [paper_dir]


def test_graphml_export_strips_xml_invalid_characters(tmp_path: Path) -> None:
    """GraphML should round-trip nullable metadata and XML-invalid text."""
    graph, seed_id = _build_graph()
    graph.nodes[seed_id]["title"] = "Bad\x0btitle￾"
    graph.nodes[seed_id].pop("paper", None)
    for field in ("venue", "doi", "arxiv_id", "abstract", "authors", "categories"):
        graph.nodes[seed_id][field] = None
    exporter = GraphExporter(graph, seed_id, metadata={"strategy": "citation"})

    graphml_path = tmp_path / "invalid-chars.graphml"
    exporter.to_graphml(graphml_path)

    assert ET.parse(graphml_path) is not None
    assert nx.read_graphml(graphml_path).nodes[seed_id]["title"] == "Badtitle"


@pytest.mark.parametrize("weight", [None, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "method",
    [
        "to_json",
        "to_dashboard_html",
        "to_csv",
        "to_bibtex",
        "to_graphml",
        "to_plotly_html",
        "to_interactive_html",
        "png",
    ],
)
def test_exports_reject_nonfinite_weights_without_replacing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weight: float | None, method: str
) -> None:
    """Invalid weights must not overwrite a usable JSON file or dashboard.

    :param Path tmp_path: Isolated output directory.
    :param pytest.MonkeyPatch monkeypatch: Optional plotting dependency stub.
    :param float | None weight: Null or non-finite edge weight.
    :param str method: Export entry point under test.
    :return None: Checks the failure and preservation of the prior artifact.
    """
    _install_fake_plotly(monkeypatch, figure_cls=_BaseFakeFigure)
    monkeypatch.setattr(
        loaders_module,
        "_load_pyvis_network_class",
        lambda: (
            lambda **kwargs: types.SimpleNamespace(
                set_options=lambda *_args: None, add_node=lambda *_args, **_kwargs: None
            )
        ),
    )
    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph, seed_id, layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)}
    )
    graph.edges["seed", "related"]["weight"] = weight
    path = tmp_path / "existing-output"
    path.write_text("previous valid output")
    with pytest.raises(ValueError, match="non-finite edge weight"):
        if method == "png":
            visualize_graph(graph, seed_id, path)
        else:
            getattr(exporter, method)(path)
    assert path.read_text() == "previous valid output"


def test_json_export_rejects_nonfinite_payload_without_replacing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSON export must reject a non-finite payload before replacing the file.

    :param Path tmp_path: Isolated output directory.
    :param pytest.MonkeyPatch monkeypatch: Supplies a non-finite export payload.
    :return None: Checks strict JSON encoding preserves the previous artifact.
    """
    exporter = GraphExporter(nx.Graph(), "seed")
    monkeypatch.setattr(exporter, "graph_payload", lambda: {"value": float("nan")})
    path = tmp_path / "existing.json"
    path.write_text("previous valid output", encoding="utf-8")

    with pytest.raises(ValueError, match="Out of range float values"):
        exporter.to_json(path)

    assert path.read_text(encoding="utf-8") == "previous valid output"


@pytest.mark.parametrize("node_id", ["", " ", " seed", "seed\t"])
def test_export_rejects_whitespace_ids_before_writing(
    tmp_path: Path, node_id: str
) -> None:
    """Exports must not emit IDs that their own dashboard importer rejects.

    :param Path tmp_path: Isolated artifact directory.
    :param str node_id: Empty or non-canonical graph node identifier.
    :return None: Checks a clear error preserves the prior artifact.
    """
    graph = nx.Graph()
    graph.add_node(node_id, title="Seed", is_seed=True)
    exporter = GraphExporter(graph, node_id, layout={node_id: (0.0, 0.0)})
    path = tmp_path / "existing.json"
    path.write_text("previous valid output")
    with pytest.raises(ValueError, match="non-canonical node ID"):
        exporter.to_json(path)
    assert path.read_text() == "previous valid output"


@pytest.mark.parametrize("for_dashboard", [False, True])
def test_plotly_marker_labels_escape_upstream_markup(
    monkeypatch: pytest.MonkeyPatch, for_dashboard: bool
) -> None:
    """Marker labels must display author/title markup as literal text.

    :param pytest.MonkeyPatch monkeypatch: Installs a capture-only Plotly figure.
    :param bool for_dashboard: Initial standalone or dashboard trace generation.
    :return None: Checks both author-derived and title-only labels.
    """
    _install_fake_plotly(monkeypatch, figure_cls=_BaseFakeFigure)
    graph, seed_id = _build_graph()
    graph.nodes[seed_id]["paper"] = replace(
        graph.nodes[seed_id]["paper"], authors=[Author(name="A <b>Smith</b>")]
    )
    graph.nodes["related"].pop("paper")
    graph.nodes["related"]["title"] = "<i>Title & text</i>"
    exporter = GraphExporter(
        graph, seed_id, layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)}
    )
    figure, _ = exporter._build_plotly_figure(
        go=loaders_module._load_plotly_graph_objects(),
        theme_obj=get_theme("dark"),
        for_dashboard=for_dashboard,
    )
    marker = next(trace for trace in figure.data if trace.get("name") == "nodes")
    assert marker["text"] == [
        html.escape("<i>Title & text</i>"),
        html.escape("<b>Smith</b>, 2020"),
    ]


def test_plotly_figure_title_escapes_upstream_markup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Figure titles must render seed-title markup as literal text.

    :param pytest.MonkeyPatch monkeypatch: Installs a capture-only Plotly figure.
    :return None: Checks the layout title of a markup-carrying seed.
    """
    _install_fake_plotly(monkeypatch, figure_cls=_BaseFakeFigure)
    hostile_title = "<i>Deep</i> & </script> Wide"
    graph, seed_id = _build_hostile_graph(title=hostile_title)
    exporter = GraphExporter(
        graph, seed_id, layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)}
    )

    figure, _ = exporter._build_plotly_figure(
        go=loaders_module._load_plotly_graph_objects(),
        theme_obj=get_theme("light"),
        for_dashboard=False,
    )

    assert figure.layout["title"] == f"CiteMesh: {html.escape(hostile_title)}"


def _extract_dashboard_script_text(html_text: str, script_id: str) -> str:
    """Extract the raw text content of a dashboard JSON script tag."""

    match = re.search(
        rf'<script id="{re.escape(script_id)}" type="application/json">(.*?)</script>',
        html_text,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _extract_dashboard_script_json(html_text: str, script_id: str) -> dict[str, Any]:
    """Extract embedded JSON payload from a dashboard script tag."""

    return json.loads(_extract_dashboard_script_text(html_text, script_id))


def _extract_inline_script_bodies(html_text: str) -> list[str]:
    """Extract bare inline script bodies in source order."""

    return re.findall(r"<script>(.*?)</script>", html_text, flags=re.DOTALL)


def test_inject_darkreader_lock_is_idempotent_and_head_gated(tmp_path: Path) -> None:
    """HTML exports get one Dark Reader lock plus a color-scheme declaration."""
    from citemesh.visualization.export import _inject_darkreader_lock

    page = tmp_path / "page.html"
    page.write_text("<html><head><title>x</title></head></html>", encoding="utf-8")
    _inject_darkreader_lock(page, "dark")
    content = page.read_text(encoding="utf-8")
    assert content.count("darkreader-lock") == 1
    assert content.count('<meta name="color-scheme" content="dark" />') == 1
    _inject_darkreader_lock(page, "dark")
    assert page.read_text(encoding="utf-8").count("darkreader-lock") == 1

    fragment = tmp_path / "fragment.html"
    fragment.write_text("<div>no head</div>", encoding="utf-8")
    _inject_darkreader_lock(fragment)
    assert "darkreader-lock" not in fragment.read_text(encoding="utf-8")

    unknown = tmp_path / "unknown.html"
    unknown.write_text("<html><head></head></html>", encoding="utf-8")
    _inject_darkreader_lock(unknown, "hotdog")
    assert '<meta name="color-scheme" content="light" />' in unknown.read_text(
        encoding="utf-8"
    )

    string_path = tmp_path / "string-path.html"
    string_path.write_text("<html><head></head></html>", encoding="utf-8")
    _inject_darkreader_lock(str(string_path), "dark")
    assert "darkreader-lock" in string_path.read_text(encoding="utf-8")


def test_edge_strength_scale_normalizes_within_graph() -> None:
    """Edge strengths must be min-max scaled so relative weight is visible."""
    from citemesh.visualization.export import _edge_strength_scale

    assert _edge_strength_scale([]) == []
    assert _edge_strength_scale([0.7]) == [0.5]
    assert _edge_strength_scale([0.9, 0.9, 0.9]) == [0.5, 0.5, 0.5]
    scaled = _edge_strength_scale([0.55, 0.75, 0.95])
    assert scaled[0] == pytest.approx(0.0)
    assert scaled[1] == pytest.approx(0.5)
    assert scaled[2] == pytest.approx(1.0)


def test_atomic_dashboard_write_preserves_previous_viewer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed final replace must leave the previous dashboard intact."""
    destination = tmp_path / "dashboard.html"
    destination.write_text("previous viewer", encoding="utf-8")

    def fail_replace(source: object, target: object) -> None:
        """Simulate an operating-system replace failure.

        :param object source: Ignored temporary path.
        :param object target: Ignored destination path.
        :return None: Always raises.
        """
        del source, target
        raise OSError("replace failed")

    monkeypatch.setattr(cache_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        cache_module.atomic_write_text(destination, "new viewer")

    assert destination.read_text(encoding="utf-8") == "previous viewer"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("failure_stage", ["writer", "injection"])
def test_interactive_html_failure_preserves_previous_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_stage: str
) -> None:
    """Pyvis and post-processing failures must not publish partial HTML.

    :param pytest.MonkeyPatch monkeypatch: Installs the failing exporter stage.
    :param Path tmp_path: Isolated output directory.
    :param str failure_stage: Export stage that raises after writing temporary data.
    :return None: Checks that the old destination and directory contents survive.
    """
    graph, seed_id = _build_graph()
    destination = tmp_path / "graph.html"
    destination.write_text("previous export", encoding="utf-8")

    class FakeNetwork:
        """Write a partial Pyvis fixture and optionally fail."""

        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def set_options(self, options: str) -> None:
            del options

        def add_node(self, node_id: str, **kwargs: Any) -> None:
            del node_id, kwargs

        def add_edge(self, source: str, target: str, **kwargs: Any) -> None:
            del source, target, kwargs

        def save_graph(self, path: str) -> None:
            """Write temporary HTML and optionally simulate a Pyvis failure.

            :param str path: Temporary HTML destination.
            :return None: Writes partial data before the selected failure.
            """
            Path(path).write_text("partial export", encoding="utf-8")
            if failure_stage == "writer":
                raise RuntimeError("writer failed")

    def fail_injection(path: Path, scheme: str) -> None:
        """Simulate post-processing failure after the library write.

        :param Path path: Temporary HTML path.
        :param str scheme: Requested color scheme.
        :return None: Always raises for the injection test case.
        """
        del path, scheme
        raise RuntimeError("injection failed")

    monkeypatch.setattr(
        loaders_module, "_load_pyvis_network_class", lambda: FakeNetwork
    )
    if failure_stage == "injection":
        monkeypatch.setattr(geometry_module, "_inject_darkreader_lock", fail_injection)

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        GraphExporter(graph, seed_id).to_interactive_html(destination)

    assert destination.read_text(encoding="utf-8") == "previous export"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("failure_stage", ["writer", "injection"])
def test_plotly_html_failure_preserves_previous_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_stage: str
) -> None:
    """Plotly and post-processing failures must not publish partial HTML.

    :param pytest.MonkeyPatch monkeypatch: Installs the failing exporter stage.
    :param Path tmp_path: Isolated output directory.
    :param str failure_stage: Export stage that raises after writing temporary data.
    :return None: Checks that the old destination and directory contents survive.
    """
    graph, seed_id = _build_graph()
    destination = tmp_path / "graph.plotly.html"
    destination.write_text("previous export", encoding="utf-8")

    class FakeFigure(_BaseFakeFigure):
        """Write a partial Plotly fixture and optionally fail."""

        def write_html(self, path: str, **kwargs: Any) -> None:
            """Write temporary HTML and optionally simulate a Plotly failure.

            :param str path: Temporary HTML destination.
            :param Any kwargs: Plotly serialization options.
            :return None: Writes partial data before the selected failure.
            """
            del kwargs
            Path(path).write_text("partial export", encoding="utf-8")
            if failure_stage == "writer":
                raise RuntimeError("writer failed")

    def fail_injection(path: Path, scheme: str) -> None:
        """Simulate post-processing failure after the library write.

        :param Path path: Temporary HTML path.
        :param str scheme: Requested color scheme.
        :return None: Always raises for the injection test case.
        """
        del path, scheme
        raise RuntimeError("injection failed")

    _install_fake_plotly(monkeypatch, figure_cls=FakeFigure)
    if failure_stage == "injection":
        monkeypatch.setattr(geometry_module, "_inject_darkreader_lock", fail_injection)

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        GraphExporter(
            graph,
            seed_id,
            layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)},
        ).to_plotly_html(destination)

    assert destination.read_text(encoding="utf-8") == "previous export"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("failure_stage", ["writer", "replace"])
def test_png_failure_preserves_previous_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_stage: str
) -> None:
    """Matplotlib and final-replace failures must not publish a partial PNG.

    :param pytest.MonkeyPatch monkeypatch: Installs the failing output operation.
    :param Path tmp_path: Isolated output directory.
    :param str failure_stage: Output operation that raises.
    :return None: Checks that the old destination and directory contents survive.
    """
    graph, seed_id = _build_graph()
    destination = tmp_path / "graph.png"
    destination.write_bytes(b"previous png")

    if failure_stage == "writer":

        def fail_savefig(figure: object, path: Path, **kwargs: Any) -> None:
            """Write partial image data and simulate a Matplotlib failure.

            :param object figure: Matplotlib figure instance.
            :param Path path: Temporary PNG destination.
            :param Any kwargs: Matplotlib serialization options.
            :return None: Writes partial data and raises.
            """
            del figure, kwargs
            path.write_bytes(b"partial png")
            raise RuntimeError("writer failed")

        monkeypatch.setattr(render_module.plt.Figure, "savefig", fail_savefig)
    else:

        def fail_replace(source: object, target: object) -> None:
            """Simulate a failed final atomic replacement.

            :param object source: Temporary PNG path.
            :param object target: Final PNG path.
            :return None: Always raises.
            """
            del source, target
            raise OSError("replace failed")

        monkeypatch.setattr(cache_module.os, "replace", fail_replace)

    error = RuntimeError if failure_stage == "writer" else OSError
    with pytest.raises(error, match=f"{failure_stage} failed"):
        visualize_graph(
            graph,
            seed_id,
            destination,
            layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)},
            dpi=40,
        )

    assert destination.read_bytes() == b"previous png"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("method", ["to_bibtex", "to_graphml"])
def test_bibtex_and_graphml_exports_are_atomic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method: str
) -> None:
    """A failed final replace must preserve existing text-based exports.

    :param pytest.MonkeyPatch monkeypatch: Simulates a failed file replacement.
    :param Path tmp_path: Isolated output directory.
    :param str method: Export method to exercise.
    :return None: Checks that no partial replacement survives the failure.
    """
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id, metadata={"strategy": "citation"})
    destination = tmp_path / f"graph.{method}"
    destination.write_text("previous export", encoding="utf-8")

    def fail_replace(source: object, target: object) -> None:
        """Simulate an operating-system replace failure.

        :param object source: Ignored temporary path.
        :param object target: Ignored destination path.
        :return None: Always raises.
        """
        del source, target
        raise OSError("replace failed")

    monkeypatch.setattr(cache_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        getattr(exporter, method)(destination)

    assert destination.read_text(encoding="utf-8") == "previous export"
    assert list(tmp_path.iterdir()) == [destination]


def test_dashboard_curve_direction_is_endpoint_order_independent() -> None:
    """Python and browser rerenders must curve an undirected edge identically."""
    assert _stable_curve_direction("related", "seed") == 1.0
    assert _stable_curve_direction("seed", "related") == 1.0


def test_plotly_hover_text_wraps_long_titles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Long titles wrap across hover lines instead of one full-width banner."""
    graph, seed_id = _build_graph()
    long_title = (
        "A Remarkably Verbose and Meandering Study of Attention Mechanisms "
        "Across Extremely Wide Tooltip Layouts"
    )
    graph.nodes["related"]["paper"] = Paper(
        paper_id="related",
        title=long_title,
        year=2021,
        authors=[Author(name="Bob Jones")],
        citation_count=10,
        venue="Related Journal",
    )
    graph.nodes["related"]["title"] = long_title

    captured: dict[str, object] = {}

    class FakeFigure(_BaseFakeFigure):
        def __init__(self, data: Any, layout: Any) -> None:
            super().__init__(data, layout)
            captured["data"] = data

        def write_html(self, path: str, **kwargs: Any) -> None:
            del kwargs
            Path(path).write_text("<html>plotly</html>")

    _install_fake_plotly(monkeypatch, figure_cls=FakeFigure)
    exporter = GraphExporter(
        graph, seed_id, layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)}
    )
    exporter.to_plotly_html(tmp_path / "graph.plotly.html")

    node_trace = captured["data"][1]
    related_hover = node_trace["hovertext"][0]
    title_line = related_hover.split("</b>")[0]
    assert title_line.count("<br>") >= 1
    assert all(
        len(chunk) <= 58 for chunk in title_line.replace("<b>", "").split("<br>")
    )


def test_exporter_dashboard_contracts(tmp_path: Path) -> None:
    """Dashboard export should render tri-pane shell and derived payload fields."""
    pytest.importorskip("plotly")

    graph, seed_id = _build_graph()
    graph.graph["paper_sources"] = {"related": "semantic", "seed": "citation"}
    exporter = GraphExporter(
        graph,
        seed_id,
        metadata={"strategy": "hybrid"},
        layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)},
    )
    graph_payload = exporter.graph_payload()
    exporter.metadata["dashboard_collection"] = {
        "kind": "citemesh-dashboard-collection",
        "schema_version": 1,
        "current_result_id": "hybrid:seed",
        "results": [
            {
                "result_id": "hybrid:seed",
                "seed_id": "seed",
                "title": "Seed Paper",
                "strategy": "hybrid",
                "summary": {"nodes": 2, "edges": 1},
                "updated_at": "2026-08-29T12:00:00",
                "payload": graph_payload,
                "build": {"max_papers": 2},
            }
        ],
    }
    out_path = tmp_path / "graph.dashboard.html"
    exporter.to_dashboard_html(out_path, theme="dark")

    assert out_path.exists()
    rendered = out_path.read_text()
    for token in [
        '<meta name="darkreader-lock" />',
        '<meta name="color-scheme" content="',
        "color-scheme: ",
        'id="global-nav"',
        '<header id="dashboard-toolbar" class="collapsed">',
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
        'id="export-collection-btn"',
        'id="add-results-btn"',
        'id="add-results-input"',
        "multiple",
        "Add Results…",
        "Export Collection",
        'id="saved-filter"',
        'id="export-saved-bib-btn"',
        'id="copy-saved-links-btn"',
        "legend-gradient",
        "star-btn",
        "citemesh-saved:",
    ]:
        assert token in rendered
    # The legend must not advertise a provenance encoding the graph does not
    # render (nodes are colored by year, not by source).
    for stale_token in [
        "legend-marker citation",
        "legend-marker semantic",
        "legend-marker both",
    ]:
        assert stale_token not in rendered
    for css_token in [
        "html, body {\n      margin: 0;\n      height: 100%;\n      overflow: hidden;",
        "#dashboard-root {\n      display: grid;\n      gap: 12px;\n      padding: 12px;\n      flex: 1 1 auto;",
        "#paper-list {\n      margin: 0;\n      padding: 0;\n      list-style: none;\n      overflow-y: auto;",
        "#detail-content {\n      padding: 14px 13px 12px;\n      flex: 1;\n      min-height: 0;\n      display: flex;\n      flex-direction: column;\n      gap: 16px;\n      overflow-y: auto;",
        "#graph-pane .pane-header .muted {\n      max-width: 72%;\n      font-size: 12px;",
        "min-height: 160px;\n      flex: 1 0 160px;",
        "#detail-pane { grid-area: detail; min-height: 620px; }",
        ".toolbar-row.secondary { grid-template-columns: 140px 140px 1fr auto; }",
        "@media (max-width: 640px) {\n      #dashboard-toolbar { position: static; }",
        ".toolbar-row.primary,\n      .toolbar-row.secondary {\n        grid-template-columns: minmax(0, 1fr);",
        ".toolbar-row.primary #search-input,\n      #provenance-filters {\n        grid-column: auto;",
        ".js-plotly-plot .modebar-btn path {\n      fill: var(--text-muted) !important;",
        ".js-plotly-plot .modebar-btn:focus-visible {\n      outline: 2px solid var(--accent);",
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
    assert collection["kind"] == "citemesh-dashboard-collection"
    assert collection["schema_version"] == 1
    assert collection["current_result_id"] == "hybrid:seed"
    assert collection["results"][0]["title"] == "Seed Paper"
    assert collection["results"][0]["build"] == {"max_papers": 2}
    assert "payload" in collection["results"][0]
    assert "payloads" not in collection

    figure = _extract_dashboard_script_json(rendered, "citemesh-dashboard-figure")
    assert len(figure["data"]) == 3
    assert len(figure["layout"].get("shapes", [])) == 1
    assert figure["layout"]["uirevision"] == (
        "citemesh-dashboard-static-layout-v1:hybrid:seed"
    )
    assert figure["layout"]["xaxis"]["autorange"] is False
    assert figure["layout"]["yaxis"]["autorange"] is False
    assert figure["layout"]["margin"]["b"] == DASHBOARD_FOOTER_MARGIN
    assert len(figure["layout"]["xaxis"]["range"]) == 2
    assert len(figure["layout"]["yaxis"]["range"]) == 2
    edge_shape = figure["layout"]["shapes"][0]
    assert edge_shape["type"] == "path"
    assert " Q " in edge_shape["path"]
    # Single edge -> tie-normalized strength 0.5 -> midpoint of the visual range.
    assert edge_shape["line"]["width"] == pytest.approx(0.45 + 1.2 * 0.5)
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
    assert neighborhood_trace["line"]["width"] == pytest.approx(2.0)
    marker = node_trace["marker"]
    assert marker["showscale"] is False
    assert marker["sizemode"] == "area"
    assert marker["sizeref"] > 0
    assert marker["sizemin"] == 4
    rendered_max_diameter = (2.0 * max(marker["size"]) / marker["sizeref"]) ** 0.5
    assert rendered_max_diameter == pytest.approx(DASHBOARD_MAX_NODE_DIAMETER)
    assert max(marker["line"]["width"]) >= 4
    assert min(marker["line"]["width"]) == 0
    seed_index = payload["meta"]["plotly_node_order"].index("seed")
    seed_ring_color = marker["line"]["color"][seed_index]
    assert f"--seed-ring: {seed_ring_color};" in rendered
    assert (
        "background: color-mix(in srgb, var(--seed-ring) 28%, transparent);" in rendered
    )
    node_x = node_trace["x"]
    node_y = node_trace["y"]
    x_span = max(node_x) - min(node_x)
    y_span = max(node_y) - min(node_y)
    expected_x_pad = max(DASHBOARD_AXIS_X_PADDING, x_span * 0.1)
    expected_y_pad = max(DASHBOARD_AXIS_MIN_PADDING, y_span * 0.08)
    assert figure["layout"]["xaxis"]["range"] == pytest.approx(
        [min(node_x) - expected_x_pad, max(node_x) + expected_x_pad]
    )
    assert figure["layout"]["yaxis"]["range"] == pytest.approx(
        [min(node_y) - expected_y_pad, max(node_y) + expected_y_pad]
    )
    related_hover = node_trace["hovertext"][0]
    seed_hover = node_trace["hovertext"][1]
    assert "Related Journal" in related_hover
    assert "semantic match" in related_hover
    assert "10 citations" in related_hover
    assert "seed paper" in seed_hover
    # Hover cards use the dashboard panel chrome, not the marker color.
    assert node_trace["hoverlabel"]["bgcolor"] == "#171d25"


def test_exporter_does_not_relabel_legacy_collection_metadata() -> None:
    """Legacy descriptor/payload bundles must remain visibly unversioned."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph,
        seed_id,
        metadata={"strategy": "hybrid"},
        layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)},
    )
    legacy_payload = exporter.graph_payload()
    legacy_payload.pop("kind")
    legacy_payload.pop("schema_version")
    exporter.metadata["dashboard_collection"] = {
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
        "payloads": {"hybrid:seed": legacy_payload},
    }

    bundle = exporter._dashboard_collection_bundle()

    assert "kind" not in bundle
    assert "schema_version" not in bundle
    assert bundle["results"][0]["payload"] == legacy_payload


def _collection_metadata(
    *,
    current_result_id: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build versioned collection metadata for exporter bundle tests.

    :param str current_result_id: Result ID the bundle should open with.
    :param list[dict[str, Any]] results: Raw result descriptors under test.
    :return dict[str, Any]: Collection metadata accepted by the exporter.
    """
    return {
        "kind": DASHBOARD_COLLECTION_KIND,
        "schema_version": DASHBOARD_COLLECTION_SCHEMA_VERSION,
        "current_result_id": current_result_id,
        "results": results,
    }


def test_collection_bundle_never_emits_a_dangling_current_result_id() -> None:
    """Skipping a malformed entry must repoint current_result_id at a real result.

    The viewer rejects a whole bundle whose ``current_result_id`` names no
    included result, so the emitter must never produce one.

    :return None: Checks both a surviving result and an entirely empty bundle.
    """
    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph,
        seed_id,
        metadata={"strategy": "citation"},
        layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)},
    )
    payload = exporter.graph_payload()
    dropped_entry = {
        "result_id": "citation:dropped",
        "seed_id": "dropped",
        "title": "Dropped",
        "strategy": "citation",
        "summary": {"nodes": 0, "edges": 0},
        "updated_at": "2026-09-01T12:00:00",
        "payload": None,
    }
    kept_entry = {
        "result_id": f"citation:{seed_id}",
        "seed_id": seed_id,
        "title": "Seed Paper",
        "strategy": "citation",
        "summary": {"nodes": 2, "edges": 1},
        "updated_at": "2026-09-01T12:00:00",
        "payload": payload,
    }
    exporter.metadata["dashboard_collection"] = _collection_metadata(
        current_result_id="citation:dropped",
        results=[dropped_entry, kept_entry],
    )

    bundle = exporter._dashboard_collection_bundle()

    assert [entry["result_id"] for entry in bundle["results"]] == [
        f"citation:{seed_id}"
    ]
    assert bundle["current_result_id"] == f"citation:{seed_id}"

    exporter.metadata["dashboard_collection"] = _collection_metadata(
        current_result_id="citation:dropped",
        results=[dropped_entry],
    )

    assert exporter._dashboard_collection_bundle() == {
        "kind": DASHBOARD_COLLECTION_KIND,
        "schema_version": DASHBOARD_COLLECTION_SCHEMA_VERSION,
        "current_result_id": None,
        "results": [],
    }


def test_collection_bundle_normalizes_non_dict_build_metadata() -> None:
    """A present-but-null build must normalize to ``{}`` for versioned bundles.

    :return None: Checks the emitted entry keeps the key the viewer requires.
    """
    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph,
        seed_id,
        metadata={"strategy": "citation"},
        layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)},
    )
    exporter.metadata["dashboard_collection"] = _collection_metadata(
        current_result_id=f"citation:{seed_id}",
        results=[
            {
                "result_id": f"citation:{seed_id}",
                "seed_id": seed_id,
                "title": "Seed Paper",
                "strategy": "citation",
                "summary": {"nodes": 2, "edges": 1},
                "updated_at": "2026-09-01T12:00:00",
                "build": None,
                "payload": exporter.graph_payload(),
            }
        ],
    )

    bundle = exporter._dashboard_collection_bundle()

    assert bundle["results"][0]["build"] == {}


class _JsonFakeFigure(_BaseFakeFigure):
    """Fake Plotly figure that can serialize itself for dashboard embedding."""

    def to_plotly_json(self) -> dict[str, object]:
        """Return the captured figure spec.

        :return dict[str, object]: Minimal figure payload for the embedded JSON block.
        """
        return {"data": self.data, "layout": self.layout}


def _dashboard_exporter_with_collection(
    graph: nx.Graph, seed_id: str, *, title: str
) -> GraphExporter:
    """Build a dashboard exporter whose collection bundle embeds the same graph.

    :param nx.Graph graph: Graph to export.
    :param str seed_id: Seed paper identifier.
    :param str title: Result descriptor title.
    :return GraphExporter: Exporter carrying a one-result collection bundle.
    """
    exporter = GraphExporter(
        graph,
        seed_id,
        metadata={"strategy": "citation"},
        layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)},
    )
    result_id = f"citation:{seed_id}"
    exporter.metadata["dashboard_collection"] = {
        "kind": DASHBOARD_COLLECTION_KIND,
        "schema_version": DASHBOARD_COLLECTION_SCHEMA_VERSION,
        "current_result_id": result_id,
        "results": [
            {
                "result_id": result_id,
                "seed_id": seed_id,
                "title": title,
                "strategy": "citation",
                "summary": {"nodes": 2, "edges": 1},
                "updated_at": "2026-09-01T12:00:00",
                "payload": exporter.graph_payload(),
            }
        ],
    }
    return exporter


def test_dashboard_inline_json_contains_no_raw_angle_brackets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inline dashboard JSON must escape every ``<`` so hostile text cannot end a script.

    Upstream text such as ``<!--<script>`` in an abstract would otherwise drive the
    HTML tokenizer into the script-data-double-escaped state, swallow the closing
    ``</script>`` tag, and blank the whole page.
    """
    _install_fake_plotly(monkeypatch, figure_cls=_JsonFakeFigure)

    hostile_title = "Closing </script> tag"
    hostile_abstract = "<!--<script>alert(1)</script>-->"
    graph, seed_id = _build_hostile_graph(
        title=hostile_title, abstract=hostile_abstract
    )
    exporter = _dashboard_exporter_with_collection(graph, seed_id, title=hostile_title)

    out_path = tmp_path / "hostile.dashboard.html"
    exporter.to_dashboard_html(out_path)
    rendered = out_path.read_text()

    data_json = _extract_dashboard_script_text(rendered, "citemesh-dashboard-data")
    figure_json = _extract_dashboard_script_text(rendered, "citemesh-dashboard-figure")
    collection_json = _extract_dashboard_script_text(
        rendered, "citemesh-dashboard-collection"
    )

    assert figure_json
    assert "<" not in data_json
    assert "<" not in collection_json
    assert "\\u003c" in data_json
    assert json.loads(collection_json)["results"][0]["build"] == {}

    seed_node = next(
        node for node in json.loads(data_json)["nodes"] if node["id"] == seed_id
    )
    assert seed_node["abstract"] == hostile_abstract
    assert seed_node["title"] == hostile_title


def test_dashboard_template_tokens_in_metadata_survive_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Template tokens inside paper metadata must survive single-pass rendering.

    Sequential substitution would rescan already-injected values and rewrite a
    ``__PLOTLY_DIV_ID__`` or ``__COLLECTION_JSON__`` token carried by a title.
    """
    _install_fake_plotly(monkeypatch, figure_cls=_JsonFakeFigure)

    hostile_title = "Tokens __PLOTLY_DIV_ID__ and __COLLECTION_JSON__ inline"
    graph, seed_id = _build_hostile_graph(title=hostile_title)
    exporter = _dashboard_exporter_with_collection(graph, seed_id, title=hostile_title)

    out_path = tmp_path / "tokens.dashboard.html"
    exporter.to_dashboard_html(out_path)
    rendered = out_path.read_text()

    payload = _extract_dashboard_script_json(rendered, "citemesh-dashboard-data")
    seed_node = next(node for node in payload["nodes"] if node["id"] == seed_id)
    assert seed_node["title"] == hostile_title

    div_id = exporter._plotly_div_id(prefix="citemesh-dashboard-plotly")
    assert div_id.startswith("citemesh-dashboard-plotly-")
    assert f'<div id="{div_id}"></div>' in rendered


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
    assert f'const GRAPH_PAYLOAD_KIND = "{GRAPH_PAYLOAD_KIND}";' in runtime_script
    assert (
        f"const GRAPH_PAYLOAD_SCHEMA_VERSION = {GRAPH_PAYLOAD_SCHEMA_VERSION};"
        in runtime_script
    )
    assert f'const COLLECTION_KIND = "{DASHBOARD_COLLECTION_KIND}";' in runtime_script
    assert (
        f"const COLLECTION_SCHEMA_VERSION = {DASHBOARD_COLLECTION_SCHEMA_VERSION};"
        in runtime_script
    )
    assert "new DOMParser()" in runtime_script
    assert "function parseImportedResultSetFromText" in runtime_script
    assert "const packageCurrentResultId = String(" in runtime_script
    assert "function normalizeCollectionPackage" in runtime_script
    assert "function collectionEntryFromGraphPayload" in runtime_script
    assert "function isNonNegativeInteger" in runtime_script
    assert (
        "Versioned graph results require canonical top-level seed_id" in runtime_script
    )
    assert (
        "Versioned graph node IDs cannot contain surrounding whitespace"
        in runtime_script
    )
    assert "Versioned graph dashboard metadata must match" in runtime_script
    assert "const positionsAreFinite = positions.every" in runtime_script
    assert "const sizesAreFinite = sizes.every" in runtime_script
    assert "if (isVersionedCollection)" in runtime_script
    assert "function upsertCollectionEntries" in runtime_script
    assert (
        "targetCollection.results = incomingUnique.concat(retained);" in runtime_script
    )
    assert "function portableCollectionPackage" in runtime_script
    assert "loadCollectionResult(collectionResultId)" in runtime_script
    assert "updated_at: entry.updated_at || new Date().toISOString()" in runtime_script
    assert "build: isObjectRecord(entry.build) ? entry.build : {}" in runtime_script
    assert "const duplicateIndex = normalized.results.findIndex" in runtime_script
    assert (
        "Graph result summary does not match its node and edge arrays."
        in runtime_script
    )
    assert "function safeExternalUrl" in runtime_script
    assert (
        'parsed.protocol === "https:" || parsed.protocol === "http:"' in runtime_script
    )
    assert (
        "Unsupported citemesh-dashboard-collection schema version" not in runtime_script
    )
    assert "Unsupported ${COLLECTION_KIND} schema version" in runtime_script
    assert "Collection package must contain a results array." in runtime_script
    assert "Array.from((event.target && event.target.files) || [])" in runtime_script
    assert (
        'unique ${uniqueGraphCount === 1 ? "graph" : "graphs"} in this session'
        in runtime_script
    )
    assert 'document.getElementById("export-collection-btn")' in runtime_script
    assert "alert(" not in runtime_script
    assert "function hasCompleteDashboardGeometry" in runtime_script
    assert "function selectDashboardLabelIds" in runtime_script
    assert "function dashboardNodeLabel" in runtime_script
    assert "function dashboardHoverText" in runtime_script
    assert "hoverTexts.push(dashboardHoverText(node, nodeId));" in runtime_script
    assert "function normalizeDashboardEdgeStrengths" in runtime_script
    assert "const [keyLeft, keyRight]" in runtime_script
    assert "const nextMarkerSizeRef = Math.max" in runtime_script
    assert "sizeref: nextMarkerSizeRef" in runtime_script
    assert "const missingYear = (safeYearMin + safeYearMax) / 2.0;" in runtime_script
    assert ": missingYear;" in runtime_script
    assert (
        "citemesh-dashboard-static-layout-v1:${String(meta.strategy" in runtime_script
    )
    assert (
        f"const xPad = Math.max({DASHBOARD_AXIS_X_PADDING}, xSpan * 0.1);"
        in runtime_script
    )
    assert (
        f"const yPad = Math.max({DASHBOARD_AXIS_MIN_PADDING}, ySpan * 0.08);"
        in runtime_script
    )
    assert "width: 0.45 + (1.2 * strength)" in runtime_script
    assert (
        "color: colorWithAlpha(dashboardEdgeColor, 0.07 + (0.25 * strength))"
        in runtime_script
    )
    assert "return pruneSavedIdsForPayload(persistedSavedIds);" in runtime_script
    assert "`citemesh-saved:${strategy}:${seedId}`" in runtime_script
    assert "`- [${markdownLinkText(title)}](${href})${yearText}`" in runtime_script
    assert '`Year: ${node.year || "n.d."}' not in runtime_script
    assert "setControlsCollapsed(true);" in runtime_script
    assert "escapeRegExp" not in runtime_script

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable for dashboard export validation")
    functions = []
    for name in ("escapeHtml", "dashboardNodeLabel"):
        match = re.search(
            rf"    function {name}\([^\n]*\n.*?\n    }}", runtime_script, re.DOTALL
        )
        assert match is not None
        functions.append(match.group(0))
    for name in ("csvGuard", "csvEscape"):
        match = re.search(rf"        function {name}\(v\).*", runtime_script)
        assert match is not None
        functions.append(match.group(0))
    slug_match = re.search(
        r"      function seedSlug\(\)[^\n]*\n.*?\n      \}", runtime_script, re.DOTALL
    )
    assert slug_match is not None
    functions.append(slug_match.group(0))
    program = (
        "\n".join(functions)
        + r"""
const nodeById = new Map([
  ["ascii", { title: "Deep Learning Survey" }],
  ["nonlatin", { title: "\u4e2d\u6587\u6807\u9898" }],
]);
let payload = { meta: { seed_id: "ascii" } };
const asciiSlug = seedSlug();
payload = { meta: { seed_id: "nonlatin" } };
const nonLatinSlug = seedSlug();
const value = "before\rafter";
process.stdout.write(JSON.stringify({
  csv: csvEscape(value),
  author: dashboardNodeLabel({authors: ["A <b>Smith</b>"], year: 2020}, "seed"),
  title: dashboardNodeLabel({title: "<b>Title</b>"}, "other"),
  asciiSlug,
  nonLatinSlug,
}));
"""
    )
    completed = subprocess.run(
        [node, "-e", program], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert list(csv.reader(io.StringIO(result["csv"], newline=""))) == [
        ["before\rafter"]
    ]
    assert result["author"] == "&lt;b&gt;Smith&lt;/b&gt;, 2020"
    assert result["title"] == "&lt;b&gt;Title&lt;/b&gt;"
    assert result["asciiSlug"] == "deep_learning_survey"
    # A title with no ASCII alphanumerics must still name the download.
    assert result["nonLatinSlug"] == "citemesh"


# Minimal DOM/Plotly stubs shared by every harness that runs a generated
# dashboard runtime. The runtime is read from process.argv[1] and the initial
# localStorage contents from CITEMESH_SAVED_STORE (a JSON array of pairs).
_DASHBOARD_NODE_HARNESS_PRELUDE = r"""
const fs = require("fs");
const html = fs.readFileSync(process.argv[1], "utf8");
function scriptText(id) {
  const pattern = new RegExp(`<script id="${id}"[^>]*>([\\s\\S]*?)<\\/script>`);
  const match = pattern.exec(html);
  if (!match) throw new Error(`missing script ${id}`);
  return match[1];
}
const runtimeScripts = Array.from(html.matchAll(/<script>([\s\S]*?)<\/script>/g));
const runtime = runtimeScripts[runtimeScripts.length - 1][1];
const classes = new Set();
const status = {
  textContent: "",
  classList: {
    toggle(name, enabled) { enabled ? classes.add(name) : classes.delete(name); },
    remove(...names) { names.forEach((name) => classes.delete(name)); },
  },
};
const noOp = () => {};
const genericClasses = {
  toggle() {},
  remove() {},
  add() {},
  contains() { return false; },
};
const fallbackElement = new Proxy({
  textContent: "",
  value: "",
  files: [],
  style: {},
  classList: genericClasses,
  addEventListener: noOp,
  appendChild: noOp,
  click: noOp,
  remove: noOp,
  scrollIntoView: noOp,
  getAttribute() { return null; },
}, {
  get(target, property) {
    return property in target ? target[property] : noOp;
  },
  set(target, property, value) {
    target[property] = value;
    return true;
  },
});
global.document = {
  documentElement: {},
  body: fallbackElement,
  getElementById(id) {
    if (id === "citemesh-dashboard-data"
        || id === "citemesh-dashboard-figure"
        || id === "citemesh-dashboard-collection") {
      return { textContent: scriptText(id) };
    }
    if (id === "dashboard-status") return status;
    return fallbackElement;
  },
  querySelectorAll() { return []; },
  createElement() { return fallbackElement; },
};
const savedStore = new Map(JSON.parse(process.env.CITEMESH_SAVED_STORE || "[]"));
global.window = {
  localStorage: {
    getItem(key) { return savedStore.has(key) ? savedStore.get(key) : null; },
    setItem(key, value) { savedStore.set(key, String(value)); },
  },
  addEventListener: noOp,
  open: noOp,
  setTimeout: noOp,
};
global.ResizeObserver = class {
  observe() {}
};
let plotlyCalled = false;
global.Plotly = {
  react() {
    plotlyCalled = true;
    return new Promise(() => {});
  },
  restyle: noOp,
  Plots: { resize: noOp },
};
global.getComputedStyle = () => ({ getPropertyValue() { return ""; } });
"""


def _execute_dashboard_runtime_in_node(
    path: Path,
    *,
    expected_status: str,
    expect_plotly: bool,
) -> subprocess.CompletedProcess[str]:
    """Execute one generated dashboard runtime against a minimal DOM harness.

    :param Path path: Generated dashboard artifact.
    :param str expected_status: Status-banner substring required after bootstrap.
    :param bool expect_plotly: Whether valid initial graph data should reach Plotly.
    :return subprocess.CompletedProcess[str]: Completed Node.js process.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable for dashboard runtime validation")

    harness = (
        _DASHBOARD_NODE_HARNESS_PRELUDE
        + r"""
const expectedStatus = process.argv[2];
const expectPlotly = process.argv[3] === "true";
eval(runtime);
if (!classes.has("visible")) throw new Error("status banner remained hidden");
if (!status.textContent.includes(expectedStatus)) {
  throw new Error(`unexpected status: ${status.textContent}`);
}
if (plotlyCalled !== expectPlotly) {
  throw new Error(`unexpected Plotly state: ${plotlyCalled}`);
}
process.stdout.write(status.textContent);
"""
    )
    return subprocess.run(
        [node, "-e", harness, str(path), expected_status, str(expect_plotly).lower()],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )


def _probe_dashboard_runtime_in_node(
    path: Path,
    expression: str,
    *,
    saved_store: dict[str, str] | None = None,
) -> Any:
    """Evaluate one expression inside a bootstrapped dashboard runtime scope.

    The runtime's helpers are module-scoped in the generated ``<script>``, so the
    probe is appended to the evaluated source rather than reaching in from
    outside.

    :param Path path: Generated dashboard artifact.
    :param str expression: JavaScript expression evaluated after bootstrap.
    :param dict[str, str] | None saved_store: Initial ``localStorage`` contents.
    :return Any: JSON-decoded expression result.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable for dashboard runtime validation")

    harness = (
        _DASHBOARD_NODE_HARNESS_PRELUDE
        + r"""
eval(runtime + "\n;globalThis.citemeshProbe = (source) => eval(source);");
const probed = globalThis.citemeshProbe(process.argv[2]);
process.stdout.write(JSON.stringify(probed === undefined ? null : probed));
"""
    )
    completed = subprocess.run(
        [node, "-e", harness, str(path), expression],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "CITEMESH_SAVED_STORE": json.dumps(sorted((saved_store or {}).items())),
        },
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_dashboard_invalid_bootstrap_surfaces_status_in_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executable dashboard bootstrap should report invalid embedded graph data.

    Strategy-less graphs now export with the ``unknown`` fallback token, so a
    tampered payload (blanked strategy) exercises the runtime guard instead.
    """

    class FakeFigure(_BaseFakeFigure):
        def to_plotly_json(self) -> dict[str, object]:
            return {"data": self.data, "layout": self.layout}

    _install_fake_plotly(monkeypatch, figure_cls=FakeFigure)
    graph, seed_id = _build_graph()
    graph.graph.clear()
    exporter = GraphExporter(
        graph,
        seed_id,
        layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)},
    )
    out_path = tmp_path / "invalid-bootstrap.dashboard.html"
    exporter.to_dashboard_html(out_path)

    rendered = out_path.read_text(encoding="utf-8")
    assert '"strategy":"unknown"' in rendered
    out_path.write_text(
        rendered.replace('"strategy":"unknown"', '"strategy":""'),
        encoding="utf-8",
    )

    result = _execute_dashboard_runtime_in_node(
        out_path,
        expected_status="Dashboard graph data is invalid",
        expect_plotly=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Graph results must identify the build strategy" in result.stdout


def test_dashboard_invalid_embedded_collection_keeps_current_graph_in_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed embedded collection should warn while rendering the graph."""

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
    out_path = tmp_path / "invalid-collection.dashboard.html"
    exporter.to_dashboard_html(out_path)
    rendered = out_path.read_text(encoding="utf-8")
    rendered = re.sub(
        r'(<script id="citemesh-dashboard-collection" type="application/json">)'
        r".*?(</script>)",
        r'\1{"kind":"unsupported-collection","results":[]}\2',
        rendered,
        count=1,
        flags=re.DOTALL,
    )
    out_path.write_text(rendered, encoding="utf-8")

    result = _execute_dashboard_runtime_in_node(
        out_path,
        expected_status="Embedded graph collection was ignored",
        expect_plotly=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Unsupported collection kind" in result.stdout


def test_dashboard_snapshot_loads_latest_same_result_in_node(tmp_path: Path) -> None:
    """Load the latest saved graph when an earlier build renders the same result ID.

    :param Path tmp_path: Isolated dashboard collection directory.
    :return None: Checks the displayed payload and figure against the latest package.
    """
    pytest.importorskip("plotly")
    first_graph, seed_id = _build_graph()
    first_metadata = {"strategy": "citation"}
    first_exporter = GraphExporter(first_graph, seed_id, metadata=first_metadata)
    package_path = tmp_path / "dashboard.citemesh.json"
    update_dashboard_package(
        package_path,
        graph=first_graph,
        seed_id=seed_id,
        strategy="citation",
        payload=first_exporter.graph_payload(),
        build={},
    )

    latest_graph, _ = _build_graph()
    latest_graph.add_node("new-paper", title="New paper", year=2024)
    latest_exporter = GraphExporter(latest_graph, seed_id)
    update_dashboard_package(
        package_path,
        graph=latest_graph,
        seed_id=seed_id,
        strategy="citation",
        payload=latest_exporter.graph_payload(),
        build={},
    )

    dashboard_path = tmp_path / "dashboard.html"
    render_dashboard_collection_snapshot(
        package_path,
        dashboard_path=dashboard_path,
        exporter=first_exporter,
        metadata=first_metadata,
        theme="dark",
    )
    displayed = _probe_dashboard_runtime_in_node(
        dashboard_path,
        "({"
        " active: payload.nodes.map(node => node.id),"
        " stored: collectionBundle.results[0].payload.nodes.map(node => node.id),"
        " positions: figureSpec.data[nodeTraceIndex].x.length,"
        " resultId: collectionResultId"
        "})",
    )
    assert displayed == {
        "active": ["new-paper", "related", "seed"],
        "stored": ["new-paper", "related", "seed"],
        "positions": 3,
        "resultId": "citation:seed",
    }


def test_saved_reading_list_survives_a_rebuild_that_dropped_a_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving a paper must not erase persisted IDs missing from the current graph.

    The storage key is stable across rebuilds, so a persisted ID whose node was
    dropped has to survive the next save toggle.

    :param Path tmp_path: Isolated output directory.
    :param pytest.MonkeyPatch monkeypatch: Installs the JSON-capable Plotly stub.
    :return None: Checks the persisted superset and the pruned display set.
    """
    _install_fake_plotly(monkeypatch, figure_cls=_JsonFakeFigure)
    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph,
        seed_id,
        layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)},
    )
    out_path = tmp_path / "saved.dashboard.html"
    exporter.to_dashboard_html(out_path)

    probed = _probe_dashboard_runtime_in_node(
        out_path,
        "(() => {"
        ' toggleSaved("related");'
        " return {"
        " stored: JSON.parse(window.localStorage.getItem(savedStorageKey())),"
        " displayed: Array.from(state.savedIds),"
        " };"
        "})()",
        saved_store={
            "citemesh-saved:citation:seed": json.dumps(["seed", "dropped-by-rebuild"])
        },
    )

    assert sorted(probed["stored"]) == ["dropped-by-rebuild", "related", "seed"]
    assert sorted(probed["displayed"]) == ["related", "seed"]


def test_copy_saved_links_escapes_bracketed_titles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Brackets in a title must not terminate the copied Markdown link label.

    :param Path tmp_path: Isolated output directory.
    :param pytest.MonkeyPatch monkeypatch: Installs the JSON-capable Plotly stub.
    :return None: Checks the Markdown label escaper used by Copy Saved Links.
    """
    _install_fake_plotly(monkeypatch, figure_cls=_JsonFakeFigure)
    graph, seed_id = _build_graph()
    exporter = GraphExporter(
        graph,
        seed_id,
        layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)},
    )
    out_path = tmp_path / "markdown.dashboard.html"
    exporter.to_dashboard_html(out_path)

    probed = _probe_dashboard_runtime_in_node(
        out_path,
        '[markdownLinkText("Attention [Is] All You Need"), markdownLinkText("Plain")]',
    )

    assert probed == ["Attention \\[Is\\] All You Need", "Plain"]


def test_dashboard_year_range_is_null_when_no_paper_has_a_year(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A yearless graph must report no range instead of the color-scale sentinel.

    :param Path tmp_path: Isolated output directory.
    :param pytest.MonkeyPatch monkeypatch: Installs the JSON-capable Plotly stub.
    :return None: Checks both exported metadata and the rendered timeline labels.
    """
    _install_fake_plotly(monkeypatch, figure_cls=_JsonFakeFigure)
    graph = nx.Graph()
    graph.graph["strategy"] = "citation"
    graph.add_node("seed", title="Seed Paper", citation_count=3, is_seed=True)
    graph.add_node("related", title="Related Paper", citation_count=1)
    graph.add_edge("seed", "related", weight=0.5)
    exporter = GraphExporter(
        graph,
        "seed",
        layout={"seed": (0.0, 0.0), "related": (1.0, 1.0)},
    )
    out_path = tmp_path / "yearless.dashboard.html"
    exporter.to_dashboard_html(out_path)

    payload = _extract_dashboard_script_json(
        out_path.read_text(), "citemesh-dashboard-data"
    )
    assert payload["meta"]["year_range"] is None
    assert exporter.graph_payload()["meta"]["year_range"] is None

    probed = _probe_dashboard_runtime_in_node(
        out_path,
        "(() => {"
        " renderTimeline();"
        " const roundTrip = portableGraphPayload(normalizeImportedDashboardPayload("
        ' buildPortableJsonPayload(), "round trip", false).payload);'
        " return {"
        " yearRange: yearRange,"
        " exportedYearRange: buildPortableJsonPayload().meta.year_range,"
        " importedYearRange: roundTrip.meta.year_range,"
        " importedDashboardYearRange: roundTrip.dashboard.meta.year_range,"
        " minLabel: controls.timelineYearMin.textContent,"
        " maxLabel: controls.timelineYearMax.textContent,"
        " };"
        "})()",
    )

    assert probed["yearRange"] == {}
    assert probed["exportedYearRange"] is None
    assert probed["importedYearRange"] is None
    assert probed["importedDashboardYearRange"] is None
    assert probed["minLabel"] == "-"
    assert probed["maxLabel"] == "-"


def test_dashboard_highlights_fallback_to_path_order_and_use_theme_styles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Plotly point metadata must retain per-node dashboard styling.

    :param Path tmp_path: Isolated output directory.
    :param pytest.MonkeyPatch monkeypatch: Installs the JSON-capable Plotly stub.
    :return None: Checks graph point classes and halo color in the generated runtime.
    """
    _install_fake_plotly(monkeypatch, figure_cls=_JsonFakeFigure)
    graph = nx.Graph()
    graph.graph["strategy"] = "citation"
    graph.add_node("seed", title="Seed", year=2020, is_seed=True)
    graph.add_node("related", title="Related", year=2021)
    graph.add_node("other", title="Other", year=2022)
    graph.add_edge("seed", "related", weight=0.7)
    exporter = GraphExporter(
        graph,
        "seed",
        layout={"other": (0.0, 0.0), "related": (1.0, 0.0), "seed": (2.0, 0.0)},
    )
    out_path = tmp_path / "highlights.dashboard.html"
    exporter.to_dashboard_html(out_path, theme="light")
    rendered = out_path.read_text(encoding="utf-8")

    for opacity in ("0.74", "0.18", "0.12"):
        assert f"opacity: {opacity} !important;" in rendered
    assert (
        rendered.count(
            "background: color-mix(in srgb, var(--panel-bg) 82%, transparent);"
        )
        == 2
    )
    assert (
        "filter: drop-shadow(0 0 10px color-mix(in srgb, var(--seed-ring) 85%, transparent)) brightness(1.14);"
        in rendered
    )
    assert "rgba(8, 12, 18, 0.72)" not in rendered
    assert "rgba(220, 80, 150, 0.85)" not in rendered

    probed = _probe_dashboard_runtime_in_node(
        out_path,
        "(() => {"
        " const pointPaths = nodeOrder.map(() => {"
        "   const classes = new Set();"
        "   return {"
        "     getAttribute() { return null; },"
        "     classList: { toggle(name, enabled) { enabled ? classes.add(name) : classes.delete(name); } },"
        "     classes,"
        "   };"
        " });"
        " const traceGroups = Array.from({ length: nodeTraceIndex + 1 }, () => null);"
        " traceGroups[nodeTraceIndex] = { querySelectorAll() { return pointPaths; } };"
        " graphDiv.data = Array.from({ length: nodeTraceIndex + 1 }, () => ({}));"
        " graphDiv.querySelectorAll = () => traceGroups;"
        " window.Plotly = Plotly;"
        " globalThis.getComputedStyle = () => ({ getPropertyValue() { return '#123456'; } });"
        " const restyles = [];"
        " Plotly.restyle = (...args) => restyles.push(args);"
        " state.selectedId = 'seed';"
        " state.hoverId = null;"
        " state.visibleIds = new Set(['seed', 'related']);"
        " syncGraphHighlights();"
        " const haloCall = restyles.find((call) => call[2][0] === haloTraceIndex);"
        " return {"
        "   paths: pointPaths.map((path, idx) => ({ id: nodeOrder[idx], classes: Array.from(path.classes).sort() })),"
        "   haloColor: haloCall[1]['marker.color'][0][0],"
        " };"
        "})()",
    )

    classes_by_id = {entry["id"]: entry["classes"] for entry in probed["paths"]}
    assert classes_by_id["other"] == ["is-dimmed", "is-filter-hidden"]
    assert classes_by_id["related"] == ["is-neighbor"]
    assert classes_by_id["seed"] == ["is-glowing"]
    assert probed["haloColor"] == "rgba(18,52,86,0.340)"


def test_dashboard_labels_balance_priority_and_spacing() -> None:
    """Dashboard labels should favor prominent nodes without crowding the seed."""
    graph = nx.Graph()
    graph.add_node("seed", is_seed=True, citation_count=10, year=2020)
    graph.add_node("crowded", citation_count=100, year=2024)
    graph.add_node("far-high", citation_count=90, year=2023)
    graph.add_node("far-low", citation_count=5, year=2022)
    positions = {
        "seed": (0.0, 0.0),
        "crowded": (DASHBOARD_LABEL_MIN_DISTANCE * 0.5, 0.0),
        "far-high": (0.5, 0.0),
        "far-low": (-0.5, 0.0),
    }

    selected = _select_dashboard_label_nodes(
        graph,
        list(graph.nodes),
        positions,
    )

    assert "seed" in selected
    assert "crowded" not in selected
    assert "far-high" in selected
    assert "far-low" in selected
    assert len(selected) <= DASHBOARD_LABEL_CAP


def test_exporter_dashboard_missing_plotly_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dashboard export should fail clearly when plotly dependency is missing."""
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id)

    monkeypatch.setattr(
        loaders_module, "_load_plotly_dashboard_runtime", raise_import_error
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


@pytest.mark.parametrize("area", [None, 100.0, 460.0, 3028.0])
@pytest.mark.parametrize("dpi", [72, 150])
def test_static_labels_clear_marker_bounds(area: float | None, dpi: int) -> None:
    """Labels and their backgrounds must clear actual rendered marker outlines.

    :param float | None area: Explicit marker area, or computed default sizes.
    :param int dpi: Canvas resolution used for text and marker measurement.
    :return None: Checks both seed and non-seed label bounding boxes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Affine2D

    graph, seed_id = _build_graph()
    positions = {"related": (0.75, 0.25), "seed": (0.25, 0.75)}
    sizes = compute_node_sizes(graph) if area is None else [area, area]
    figure, axis = plt.subplots(dpi=dpi)
    try:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        render_module.draw_nodes(
            axis, graph, positions, sizes, [(0.5, 0.5, 0.5)] * 2, get_theme("light")
        )
        render_module.draw_labels(
            axis,
            graph,
            positions,
            seed_id,
            get_theme("light"),
            sizes=None if area is None else sizes,
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        assert len(axis.texts) == 2
        markers = dict(zip(positions.values(), axis.collections))
        for label in axis.texts:
            marker = markers[tuple(label.xy)]
            center = marker.get_offset_transform().transform(marker.get_offsets()[0])
            marker_bounds = (
                marker.get_paths()[0]
                .get_extents(Affine2D(marker.get_transforms()[0]))
                .transformed(Affine2D().translate(*center))
                .padded(marker.get_linewidths()[0] * dpi / 144)
            )
            label_bounds = label.get_bbox_patch().get_window_extent(renderer)
            assert not label_bounds.overlaps(marker_bounds)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("seed_citations", [100_000, 300_000])
def test_node_size_caps_include_citation_bonus(seed_citations: int) -> None:
    """Highly cited nodes must stay within the configured area caps.

    :param int seed_citations: Citation count placing the seed above or below its peer.
    :return None: Checks the final sizes after tier assignment and citation bonuses.
    """
    graph, _ = _build_graph()
    graph.nodes["seed"]["citation_count"] = seed_citations
    graph.nodes["related"]["citation_count"] = 200_000
    related_size, seed_size = compute_node_sizes(graph)
    assert seed_size <= render_module.VIZ_CONFIG.seed_size
    assert related_size <= render_module.VIZ_CONFIG.max_non_seed_size


def test_dashboard_labels_clear_selection_halos_after_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initial and imported graphs must anchor labels outside selection halos.

    :param Path tmp_path: Isolated dashboard output directory.
    :param pytest.MonkeyPatch monkeypatch: Installs the JSON-capable Plotly stub.
    :return None: Checks marker identity, label text and radius-based pixel clearance.
    """
    _install_fake_plotly(monkeypatch, figure_cls=_JsonFakeFigure)
    graph, seed_id = _build_graph()
    graph.nodes["seed"]["citation_count"] = 200_000
    exporter = GraphExporter(
        graph, seed_id, layout={"related": (0.0, 0.0), "seed": (1.0, 1.0)}
    )
    path = tmp_path / "labels.dashboard.html"
    exporter.to_dashboard_html(path)
    initial = _extract_dashboard_script_json(
        path.read_text(encoding="utf-8"), "citemesh-dashboard-figure"
    )
    imported = _probe_dashboard_runtime_in_node(
        path,
        "(() => {"
        " const next = JSON.parse(JSON.stringify(payload));"
        " next.meta.plotly_node_sizes = [6, 5000];"
        " next.nodes.forEach(node => { node.authors = ['A <b>Imported</b>']; });"
        " return buildFigureSpecFromPayload(next);"
        "})()",
    )
    for figure in [initial, imported]:
        trace = next(trace for trace in figure["data"] if trace.get("name") == "nodes")
        assert trace["mode"] == "markers"
        annotations = figure["layout"]["annotations"]
        assert len(annotations) == 2
        assert [label["text"] for label in annotations] == trace["text"]
        for index, label in enumerate(annotations):
            marker = trace["marker"]
            radius = max(
                marker["sizemin"],
                (marker["size"][index] / (2 * marker["sizeref"])) ** 0.5,
            )
            halo_radius = max(
                4, (2.2 * marker["size"][index] / (2 * marker["sizeref"])) ** 0.5
            )
            assert label["x"] == trace["x"][index]
            assert label["y"] == trace["y"][index]
            assert label["xref"] == "x" and label["yref"] == "y"
            assert label["yanchor"] == "bottom"
            assert label["showarrow"] is False
            assert label["yshift"] >= halo_radius + 3
            assert label["yshift"] >= radius + marker["line"]["width"][index] / 2 + 3
    assert "&lt;b&gt;Imported&lt;/b&gt;" in imported["layout"]["annotations"][0]["text"]


def test_static_labels_are_capped_by_citation_priority() -> None:
    """Dense static plots should label only the highest-priority non-seed papers."""
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    graph.add_node(
        "seed",
        title="Seed Paper",
        year=2025,
        authors=["Seed Author"],
        citation_count=0,
        is_seed=True,
    )
    positions: dict[str, np.ndarray] = {"seed": np.array([0.95, 1.08])}
    for index in range(20):
        node_id = f"paper-{index:02d}"
        graph.add_node(
            node_id,
            title=f"Paper {index}",
            year=2000 + index,
            authors=[f"Author {index}"],
            citation_count=index,
            is_seed=False,
        )
        positions[node_id] = np.array(
            [0.1 + 0.27 * float(index % 4), 0.1 + 0.18 * float(index // 4)]
        )

    figure, axis = plt.subplots()
    axis.set_xlim(0.0, 1.1)
    axis.set_ylim(0.0, 1.2)
    draw_labels(
        axis,
        graph,
        positions,
        "seed",
        get_theme("light"),
        sizes=compute_node_sizes(graph),
    )

    labels = {text.get_text() for text in axis.texts}
    assert len(labels) == MAX_STATIC_NON_SEED_LABELS + 1
    assert "Seed Paper" in labels
    assert "19, 2019" in labels
    assert "0, 2000" not in labels
    plt.close(figure)


def test_static_labels_suppress_overlapping_text_bounds() -> None:
    """Static labels should not overlap even when node centers clear the distance gate."""
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    graph.add_node(
        "seed",
        title="Seed Paper",
        year=2025,
        authors=["Seed Author"],
        citation_count=100,
        is_seed=True,
    )
    for node_id, citations in [("left", 20), ("right", 10)]:
        graph.add_node(
            node_id,
            title=node_id.title(),
            year=2024,
            authors=["Extraordinarilylongsurname"],
            citation_count=citations,
            is_seed=False,
        )

    positions = {
        "seed": np.array([0.1, 0.1]),
        "left": np.array([0.45, 0.5]),
        "right": np.array([0.55, 0.5]),
    }
    figure, axis = plt.subplots()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    draw_labels(
        axis,
        graph,
        positions,
        "seed",
        get_theme("light"),
        sizes=[100.0, 100.0, 100.0],
    )

    labels = {text.get_text() for text in axis.texts}
    assert "Seed Paper" in labels
    assert sum(label.startswith("Extraordinarilylongsurname") for label in labels) == 1
    plt.close(figure)


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
            captured["hovertext"] = list(kwargs["hovertext"])
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
    hovertext = captured["hovertext"]
    assert isinstance(hovertext, list)
    seed_hover = next(text for text in hovertext if "Seed Paper" in text)
    assert "n.d. | 3 citations" in seed_hover
    assert "None" not in seed_hover

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


def test_publication_year_coercion_is_shared_across_visual_surfaces() -> None:
    """String, NumPy, and invalid years should resolve consistently everywhere."""
    graph = nx.Graph()
    graph.add_node("seed", title="Seed", year="2024", is_seed=True)
    graph.add_node("older", title="Older", year=np.float64(2020.0))
    graph.add_node("invalid", title="Invalid", year="unknown")
    layout = {"seed": (0.0, 0.0), "older": (1.0, 0.0), "invalid": (0.5, 1.0)}

    _, min_year, max_year = compute_node_colors(graph, "seed", get_theme("light"))
    exporter = GraphExporter(graph, "seed", layout=layout)
    payload = exporter.graph_payload()
    marker_years, scale_min, scale_max = exporter._plotly_year_scale(
        ["invalid", "older", "seed"]
    )

    assert (min_year, max_year) == (2020, 2024)
    assert payload["meta"]["year_range"] == {"min": 2020, "max": 2024}
    assert marker_years == [2022.0, 2020.0, 2024.0]
    assert (scale_min, scale_max) == (2020.0, 2024.0)


@pytest.mark.parametrize(
    ("raw_year", "expected"),
    [
        (2024, 2024),
        (np.int64(2023), 2023),
        (" 2022 ", 2022),
        (True, 0),
        (2021.0, 2021),
        (np.float32(2017.0), 2017),
        (2021.5, 0),
        (float("nan"), 0),
        (float("inf"), 0),
        ("unknown", 0),
        (None, 0),
    ],
)
def test_publication_year_coercion_direct_contract(
    raw_year: object,
    expected: int,
) -> None:
    """The shared year helper should reject ambiguous non-integral values."""
    assert coerce_publication_year(raw_year) == expected


def test_publication_year_bounds_and_scale_direct_contracts() -> None:
    """Missing and singleton year sets should retain deterministic color bounds."""
    assert publication_year_bounds([None, False, "unknown"]) == (2000, 2001)
    assert publication_year_bounds(["2022", np.int64(2019), None]) == (2019, 2022)
    assert publication_year_scale([2024, None]) == (
        [2024.0, 2024.5],
        2024.0,
        2025.0,
    )


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
    normalized = normalize_layout_positions(raw, padding_ratio=0.1)
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


def test_community_anchor_graph_connects_isolated_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Community anchors should scaffold partially disconnected groups."""
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c"])
    graph.add_edge("a", "b", weight=0.8)
    communities = [["a"], ["b"], ["c"]]
    captured: dict[str, bool] = {}

    def capture_spring_layout(
        layout_graph: nx.Graph, **_kwargs: Any
    ) -> dict[int, np.ndarray]:
        """Capture connectivity and return deterministic anchor coordinates."""
        captured["connected"] = nx.is_connected(layout_graph)
        return {
            node: np.array([float(node), 0.0], dtype=np.float64)
            for node in layout_graph.nodes()
        }

    monkeypatch.setattr(
        "citemesh.visualization.render.nx.spring_layout", capture_spring_layout
    )
    _spread_layout_by_communities(
        {node: np.array([0.0, 0.0]) for node in graph.nodes()},
        graph,
        communities,
        layout_seed=42,
    )

    assert captured["connected"] is True


def test_portrait_layout_is_rotated_for_landscape_exports() -> None:
    """Layout orientation should place its longer extent on the horizontal axis."""
    oriented = _orient_layout_horizontally(
        {
            "a": np.array([0.0, -2.0]),
            "b": np.array([0.5, 0.0]),
            "c": np.array([0.0, 2.0]),
        }
    )
    coords = np.array(list(oriented.values()), dtype=float)
    span_x, span_y = np.ptp(coords, axis=0)

    assert span_x > span_y


def test_disconnected_components_are_packed_by_node_count() -> None:
    """Larger components should retain most layout width beside small islands."""
    graph = nx.Graph()
    main_nodes = [f"main-{index}" for index in range(6)]
    graph.add_edges_from(zip(main_nodes, main_nodes[1:]))
    island_nodes = [f"island-{index}" for index in range(6)]
    graph.add_nodes_from(island_nodes)
    positions = {
        **{
            node: np.array([float(index % 3), float(index // 3)])
            for index, node in enumerate(main_nodes)
        },
        **{node: np.array([100.0, 0.0]) for node in island_nodes},
    }

    packed = _pack_disconnected_components(positions, graph)
    main_x = [float(packed[node][0]) for node in main_nodes]
    island_x = [float(packed[node][0]) for node in island_nodes]
    coords = np.array(list(packed.values()), dtype=float)
    span_x, span_y = np.ptp(coords, axis=0)

    assert (max(main_x) - min(main_x)) / span_x > 0.55
    assert np.mean(main_x) < max(island_x)
    assert span_x > span_y


def test_all_singleton_components_are_packed_in_two_dimensions() -> None:
    """Many isolated nodes should occupy rows with a marker-sized footprint."""
    graph = nx.Graph()
    nodes = [f"paper-{index:02d}" for index in range(20)]
    graph.add_nodes_from(nodes)
    positions = {node: np.array([0.0, 0.0]) for node in nodes}

    packed = _pack_disconnected_components(positions, graph)
    coords = np.array([packed[node] for node in nodes], dtype=float)
    pairwise_distances = [
        math.dist(coords[left], coords[right])
        for left in range(len(coords))
        for right in range(left + 1, len(coords))
    ]
    span_x, span_y = np.ptp(coords, axis=0)

    assert span_x > 0.0
    assert span_y > 0.0
    assert min(pairwise_distances) >= 0.45


def test_static_viewport_limits_fill_landscape_canvas() -> None:
    """Static viewport limits should crop whitespace while preserving equal scale."""
    positions = {
        "left": np.array([-0.9, -0.2]),
        "middle": np.array([0.0, 0.1]),
        "right": np.array([0.9, 0.2]),
    }

    x_limits, y_limits = _layout_viewport_limits(positions, viewport_aspect=1.75)

    range_x = x_limits[1] - x_limits[0]
    range_y = y_limits[1] - y_limits[0]
    assert range_x / range_y == pytest.approx(1.75)
    assert range_y < 2.1
    assert all(x_limits[0] < point[0] < x_limits[1] for point in positions.values())
    assert all(y_limits[0] < point[1] < y_limits[1] for point in positions.values())


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


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "expected_theme"),
    [
        (0, "Dark\n", "", "dark"),
        (1, "", "The domain/default pair does not exist", "light"),
    ],
)
def test_get_theme_auto_reads_macos_system_appearance(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    stderr: str,
    expected_theme: str,
) -> None:
    """Auto theme should map the native macOS appearance to CiteMesh themes."""
    for key in ("COLORFGBG", "DARKMODE", "TERM_PROGRAM"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(themes_module.sys, "platform", "darwin")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> object:
        """Capture the native appearance query and return its synthetic result."""
        captured["command"] = command
        captured["kwargs"] = kwargs
        return types.SimpleNamespace(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    monkeypatch.setattr(themes_module.subprocess, "run", fake_run)

    assert get_theme("auto").name == expected_theme
    assert captured["command"] == [
        "defaults",
        "read",
        "-g",
        "AppleInterfaceStyle",
    ]
    assert captured["kwargs"] == {
        "capture_output": True,
        "check": False,
        "text": True,
        "timeout": 1.0,
    }


def test_visualization_api_defaults_to_dark() -> None:
    """Direct exporter users should receive the same dark-first default as the CLI."""
    graph, seed_id = _build_graph()

    assert GraphExporter(graph, seed_id).theme.name == "dark"


def test_model_profiles_match_expected_formatters() -> None:
    """Gemma and default profiles should expose expected formatting behavior."""
    gemma = get_embedding_model_profile("google/embeddinggemma-300m")
    assert gemma.name == "google/embeddinggemma"
    assert gemma.preferred_compute_dtype == "bfloat16"
    assert gemma.autocast_devices == ("cuda", "mps", "cpu")
    assert gemma.compile_inner_transformer is True
    assert gemma.available_truncate_dims == (768, 512, 256, 128)
    assert gemma.recommended_truncate_dim == 512
    assert gemma.format_query("  attention  ").startswith(
        "task: search result | query:"
    )
    assert (
        gemma.format_document({"title": " Title ", "abstract": " Abstract "})
        == "title: Title | text: Abstract"
    )
    assert (
        gemma.format_similarity("  related paper  ")
        == "task: sentence similarity | query: related paper"
    )
    unsloth_gemma = get_embedding_model_profile("unsloth/embeddinggemma-300m")
    assert unsloth_gemma.name == "google/embeddinggemma"
    assert unsloth_gemma.compile_inner_transformer is True
    assert unsloth_gemma.format_query("plain").startswith("task: search result")

    default = get_embedding_model_profile("org/generic-embedding-model")
    assert default.name == "default"
    assert default.preferred_compute_dtype is None
    assert default.autocast_devices == ()
    assert default.compile_inner_transformer is False
    assert default.available_truncate_dims is None
    assert default.recommended_truncate_dim is None
    assert default.format_query("plain") == "plain"
    assert default.format_document({"title": "T", "abstract": ""}) == "T"
    assert default.format_similarity("plain") == "plain"


def test_exporter_enriched_json_csv_bibtex(tmp_path: Path) -> None:
    """Exports must preserve full author lists through API ingestion and disk reload.

    :param Path tmp_path: Temporary directory for exported graphs and bibliography.
    :return None: Checks enriched fields and complete authors in every export format.
    """
    graph, seed_id = _build_graph()
    author_names = ["Alice Alpha", "Bob Beta", "Carol Gamma", "Dana Delta"]
    with SemanticScholarClient(timeout=1) as cold:
        cold._rate_limit = MagicMock()
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "paperId": seed_id,
            "title": "Seed Paper",
            "year": 2020,
            "abstract": "Seed abstract",
            "authors": [
                {"name": name, "authorId": str(index)}
                for index, name in enumerate(author_names)
            ],
        }
        cold._session.get = MagicMock(return_value=response)
        fetched = cold.get_paper(seed_id, raise_on_unavailable=True)
        assert fetched is not None
        assert [author.name for author in fetched.authors] == author_names
        cold._session.get.assert_called_once()

    with SemanticScholarClient(timeout=1) as warm:
        warm._session.get = MagicMock(
            side_effect=AssertionError("persisted paper must reload without the API")
        )
        restored = warm.get_paper(seed_id, raise_on_unavailable=True)
        assert restored is not None and restored is not fetched
        assert restored.authors == fetched.authors
        warm._session.get.assert_not_called()

    graph.nodes[seed_id]["paper"] = restored
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
    assert seed_node["authors"] == author_names
    author_field = "author = {" + " and ".join(author_names) + "}"
    assert author_field in seed_node["bibtex"]

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
    seed_row = next(
        row for row in csv.DictReader(io.StringIO(csv_text)) if row["id"] == seed_id
    )
    assert seed_row["authors"] == "; ".join(author_names)

    # --- BibTeX ---
    bib_text = bib_path.read_text()
    assert "@article{" in bib_text
    assert "Seed Paper" in bib_text or "Related Paper" in bib_text
    assert author_field in bib_text


def test_csv_export_neutralizes_formula_cells_and_uses_lowercase_booleans(
    tmp_path: Path,
) -> None:
    """CSV cells must be formula-inert (CWE-1236) with dashboard-matching booleans."""
    formula_title = "=cmd|'/C calc'!A0"
    formula_author = "+Plus Author"
    formula_abstract = "@formula abstract"
    graph, seed_id = _build_hostile_graph(
        title=formula_title, abstract=formula_abstract, author=formula_author
    )
    exporter = GraphExporter(graph, seed_id, metadata={"strategy": "citation"})

    csv_path = tmp_path / "formula.csv"
    exporter.to_csv(csv_path)

    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    seed_row = next(row for row in rows if row["id"] == seed_id)
    assert seed_row["title"] == f"'{formula_title}"
    assert seed_row["authors"] == f"'{formula_author}"
    assert seed_row["abstract"] == f"'{formula_abstract}"
    assert {row["is_seed"] for row in rows} == {"true", "false"}


def test_csv_export_preserves_single_crlf_row_terminators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CSV rows must keep the writer's CRLF terminators without re-translation.

    These bytes are platform-independent only while the write disables text-mode
    newline translation, so the writer routing is asserted alongside them.

    :param Path tmp_path: Isolated output directory.
    :param pytest.MonkeyPatch monkeypatch: Records the newline policy in use.
    :return None: Checks the raw bytes of a header plus two paper rows.
    """
    newline_modes: list[str | None] = []
    write_text = export_module.atomic_write_text

    def record_newline_mode(
        path: Path, content: str, *, newline: str | None = ""
    ) -> None:
        """Record the newline policy, then delegate to the real atomic writer.

        :param Path path: Target text file path.
        :param str content: Complete text payload.
        :param str | None newline: Text-mode newline translation policy.
        :return None: Writes the target file through the real writer.
        """
        newline_modes.append(newline)
        write_text(path, content, newline=newline)

    monkeypatch.setattr(export_module, "atomic_write_text", record_newline_mode)
    graph, seed_id = _build_graph()
    exporter = GraphExporter(graph, seed_id, metadata={"strategy": "citation"})
    csv_path = tmp_path / "terminators.csv"

    exporter.to_csv(csv_path)

    raw = csv_path.read_bytes()
    assert newline_modes == [""]
    assert b"\r\r\n" not in raw
    assert raw.count(b"\r\n") == 3
    assert raw.count(b"\n") == raw.count(b"\r\n")


def test_empty_csv_preserves_columns_and_bibtex_has_no_entries(tmp_path: Path) -> None:
    """An empty graph still exports a parseable CSV schema and empty bibliography.

    :param Path tmp_path: Isolated output directory.
    :return None: Checks no paper rows or bibliography records are invented.
    """
    exporter = GraphExporter(nx.Graph(), "seed")
    csv_path = tmp_path / "empty.csv"
    bib_path = tmp_path / "empty.bib"
    exporter.to_csv(csv_path)
    exporter.to_bibtex(bib_path)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert reader.fieldnames[0] == "id"
        assert reader.fieldnames[-1] == "abstract"
        assert list(reader) == []
    assert bib_path.read_text().strip() == ""


def test_bibtex_keys_distinguish_ids_with_the_same_slug(tmp_path: Path) -> None:
    """Distinct paper IDs must retain distinct, order-independent citation keys.

    :param Path tmp_path: Isolated bibliography directory.
    :return None: Checks colliding punctuation and letter-case slugs in real exports.
    """
    node_ids = ["a-b", "a_b", "a/b", "A-B"]
    key_sets = []
    for index, ordering in enumerate((node_ids, list(reversed(node_ids)))):
        graph = nx.Graph()
        for node_id in ordering:
            graph.add_node(node_id, title=node_id)
        path = tmp_path / f"collisions-{index}.bib"
        GraphExporter(graph, node_ids[0]).to_bibtex(path)
        keys = re.findall(r"@article\{([^,]+),", path.read_text())
        assert len(keys) == len(node_ids)
        assert len(set(keys)) == len(node_ids)
        key_sets.append(keys)
    assert key_sets[0] == key_sets[1]


def test_bibtex_escapes_latex_specials() -> None:
    """BibTeX fields must escape every LaTeX special, leaving none bare."""
    from citemesh.visualization.export.bibtex import _bibtex_escape

    escaped = _bibtex_escape("100% $x$ & #tag_1 ~ ^ back\\slash {b}")

    assert escaped == (
        "100\\% \\$x\\$ \\& \\#tag\\_1 \\textasciitilde{} "
        "\\textasciicircum{} back\\textbackslash{}slash \\{b\\}"
    )
    for token in (
        "\\%",
        "\\$",
        "\\&",
        "\\#",
        "\\_",
        "\\textasciitilde{}",
        "\\textasciicircum{}",
        "\\textbackslash{}",
        "\\{",
        "\\}",
    ):
        assert token in escaped
    # No unescaped special survives: ~ and ^ become text-mode commands, and the
    # remaining specials are always backslash-prefixed. Braces are excluded here
    # because the emitted \textascii* commands legitimately end in "{}".
    assert "~" not in escaped
    assert "^" not in escaped
    assert re.search(r"(?<!\\)[%$&#_]", escaped) is None


def test_bibtex_identifier_fields_are_not_latex_escaped(tmp_path: Path) -> None:
    """``doi`` and ``url`` must stay verbatim while prose fields keep escaping.

    :param Path tmp_path: Isolated bibliography directory.
    :return None: Checks an underscore DOI and a bracketed Wiley-style DOI.
    """
    underscore_doi = "10.1007/978-3-642-11745-9_11"
    wiley_doi = "10.1002/(SICI)1097-0142(19960101)77:1<138::AID-CNCR23>3.0.CO;2-2"
    graph = nx.Graph()
    graph.add_node(
        "underscore",
        title="Underscore DOI",
        year=2010,
        doi=underscore_doi,
        is_seed=True,
    )
    graph.add_node("wiley", title="Wiley_Style DOI", year=1996, doi=wiley_doi)
    graph.add_edge("underscore", "wiley", weight=0.5)
    exporter = GraphExporter(graph, "underscore", metadata={"strategy": "citation"})
    bib_path = tmp_path / "identifiers.bib"

    exporter.to_bibtex(bib_path)

    rendered = bib_path.read_text()
    assert sorted(re.findall(r"^  doi = \{(.*)\},$", rendered, flags=re.M)) == sorted(
        [underscore_doi, wiley_doi]
    )
    assert sorted(re.findall(r"^  url = \{(.*)\},$", rendered, flags=re.M)) == sorted(
        [
            f"https://doi.org/{underscore_doi}",
            "https://doi.org/10.1002/(SICI)1097-0142(19960101)77:1%3C138"
            "::AID-CNCR23%3E3.0.CO;2-2",
        ]
    )
    # Prose fields keep LaTeX escaping; only identifier fields are verbatim.
    assert "  title = {Wiley\\_Style DOI}," in rendered


@pytest.mark.parametrize(
    ("strategy", "strategy_source"),
    [
        ("recommendation", "exporter"),
        ("recommendation", "graph"),
        ("embedding", "graph"),
    ],
)
def test_semantic_export_provenance(
    tmp_path: Path, strategy: str, strategy_source: str
) -> None:
    """Semantic exports should classify provenance from either strategy source."""
    graph, seed_id = _build_graph()
    metadata = None
    if strategy_source == "exporter":
        metadata = {"strategy": strategy}
    else:
        graph.graph["strategy"] = strategy
    if strategy == "embedding":
        graph.graph["embedding_runtime"] = {"storage_precision": "int8"}

    exporter = GraphExporter(graph, seed_id, metadata=metadata)
    json_path = tmp_path / f"{strategy}-{strategy_source}.json"
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


def test_visualization_import_preserves_programmatically_selected_backend(
    tmp_path: Path,
) -> None:
    """Package import should not replace a backend selected through matplotlib."""
    env = os.environ.copy()
    env.pop("MPLBACKEND", None)
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import matplotlib; matplotlib.use('svg'); "
            "import citemesh.visualization; "
            "assert matplotlib.get_backend().lower() == 'svg'",
        ],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("backend", "suffix"), [("svg", ".svg"), ("pdf", ".pdf"), ("ps", ".ps")]
)
def test_visualize_graph_supports_non_agg_backends(
    backend: str, suffix: str, tmp_path: Path
) -> None:
    """Static labels should render end-to-end on non-Agg canvases."""
    output_path = tmp_path / f"graph{suffix}"
    env = os.environ.copy()
    env["MPLBACKEND"] = backend
    env["MPLCONFIGDIR"] = str(tmp_path / f"matplotlib-{backend}")
    script = """
import sys
from pathlib import Path

import networkx as nx
import numpy as np

from citemesh.visualization import visualize_graph

graph = nx.Graph()
graph.add_node(
    "seed", title="Seed Paper", year=2025, authors=["Seed Author"], is_seed=True
)
graph.add_node(
    "related", title="Related Paper", year=2024, authors=["Other Author"]
)
graph.add_edge("seed", "related", weight=0.8)
visualize_graph(
    graph,
    "seed",
    Path(sys.argv[1]),
    layout={"seed": np.array([0.0, 0.0]), "related": np.array([1.0, 1.0])},
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(output_path)],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.stat().st_size > 0


def test_visualize_graph_uses_local_agg_canvas_for_macosx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Static export should avoid the MacOSX GUI canvas without switching globally."""
    graph, seed_id = _build_graph()
    output_path = tmp_path / "graph.png"

    monkeypatch.setattr(render_module.matplotlib, "get_backend", lambda: "MacOSX")
    monkeypatch.setattr(
        render_module.matplotlib,
        "use",
        lambda *_args, **_kwargs: pytest.fail("Static export must not switch backends"),
    )
    monkeypatch.setattr(
        render_module.plt,
        "subplots",
        lambda **_kwargs: pytest.fail("MacOSX export should use a local Agg canvas"),
    )
    visualize_graph(
        graph,
        seed_id,
        output_path,
        layout={"seed": np.array([0.0, 0.0]), "related": np.array([1.0, 1.0])},
    )

    assert output_path.stat().st_size > 0
