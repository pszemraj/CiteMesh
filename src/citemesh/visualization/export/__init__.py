"""
Graph export utilities for CiteMesh.

Provides a unified interface for exporting graphs to multiple formats,
including interactive visualizations.

This package owns the :class:`GraphExporter` facade and its public writers;
format-specific helpers live in the sibling modules and are re-exported here
so ``citemesh.visualization.export`` stays the stable import path.
"""

from __future__ import annotations

import csv
import html
import io
import json
import logging
from collections.abc import Hashable, Iterable
from pathlib import Path
from typing import Any

import networkx as nx

from citemesh.core import Paper
from citemesh.data.cache import atomic_output_path, atomic_write_text

from ..dashboard.contracts import GRAPH_PAYLOAD_KIND, GRAPH_PAYLOAD_SCHEMA_VERSION
from ..dashboard.payload import DashboardPayloadMixin, _dashboard_template
from ..themes import Theme, get_theme
from ..years import coerce_publication_year
from . import geometry, loaders
from .csv_ import _csv_cell_guard
from .geometry import (
    DASHBOARD_AXIS_MIN_PADDING,
    DASHBOARD_AXIS_X_PADDING,
    DASHBOARD_FOOTER_MARGIN,
    DASHBOARD_LABEL_CAP,
    DASHBOARD_LABEL_MIN_DISTANCE,
    DASHBOARD_MAX_NODE_DIAMETER,
    _edge_strength_scale,
    _inject_darkreader_lock,
    _select_dashboard_label_nodes,
    _stable_curve_direction,
    _theme_color_scheme,
)
from .graphml import (
    GRAPHML_LAYOUT_METADATA_KEY,
    GRAPHML_LAYOUT_VERSION_KEY,
    _graphml_metadata_key,
    _graphml_metadata_value,
    _xml_safe_graph_value,
)
from .links import _safe_script_content, _script_safe_json
from .loaders import (
    GRAPHML_DETERMINISM_POLICY_BEST_EFFORT,
    GRAPHML_DETERMINISM_POLICY_STRICT,
    _graphml_determinism_policy,
)
from .nodes import (
    NodesMixin,
    _node_short_label,
    _node_title,
    _ordered_attrs,
    _serialize_node,
    _sorted_edges,
    _sorted_nodes,
)
from .plotly_figure import PlotlyFigureMixin, _plotly_title_text

logger = logging.getLogger(__name__)

PAGE_TITLE_PREFIX = "CiteMesh"


def _export_page_title(graph: nx.Graph, seed_id: str) -> str:
    """Build the browser tab title shared by CiteMesh HTML exports.

    Reuses the Plotly figure title text so the tab and the on-canvas title name
    the same paper. That helper returns Plotly pseudo-HTML (escaped entities
    joined by ``<br>``), so it is flattened back to plain text here and
    re-escaped by the head injector.

    :param nx.Graph graph: Graph containing the seed node.
    :param str seed_id: Seed paper identifier.
    :return str: Plain-text page title.
    """
    seed_title = html.unescape(_plotly_title_text(graph, seed_id).replace("<br>", " "))
    if not seed_title or seed_title == PAGE_TITLE_PREFIX:
        return PAGE_TITLE_PREFIX
    return f"{PAGE_TITLE_PREFIX}: {seed_title}"


def _export_page_style(theme_obj: Theme) -> str:
    """Build the page chrome CSS injected into standalone HTML exports.

    Both library writers emit a default white body with an 8px margin, which
    frames a themed figure in white; painting the page in the theme background
    with no margin lets the visualization own the viewport.

    :param Theme theme_obj: Active visualization theme.
    :return str: CSS text for the injected head stylesheet.
    """
    return f"html,body{{margin:0;padding:0;background:{theme_obj.background};}}"


__all__ = [
    "DASHBOARD_AXIS_MIN_PADDING",
    "DASHBOARD_AXIS_X_PADDING",
    "DASHBOARD_FOOTER_MARGIN",
    "DASHBOARD_LABEL_CAP",
    "DASHBOARD_LABEL_MIN_DISTANCE",
    "DASHBOARD_MAX_NODE_DIAMETER",
    "GRAPHML_DETERMINISM_POLICY_STRICT",
    "GRAPHML_LAYOUT_METADATA_KEY",
    "GRAPHML_LAYOUT_VERSION_KEY",
    "GraphExporter",
    "_edge_strength_scale",
    "_graphml_determinism_policy",
    "_inject_darkreader_lock",
    "_select_dashboard_label_nodes",
    "_stable_curve_direction",
]


class GraphExporter(NodesMixin, PlotlyFigureMixin, DashboardPayloadMixin):
    """Unified interface for exporting graphs in multiple formats."""

    def __init__(
        self,
        graph: nx.Graph,
        seed_id: str,
        metadata: dict | None = None,
        theme_name: str = "dark",
        layout: dict[Hashable, Iterable[float]] | None = None,
    ):
        """Create exporter bound to a graph and seed paper metadata.

        :param nx.Graph graph: Graph to export.
        :param str seed_id: Seed paper identifier.
        :param Optional[Dict] metadata: Optional metadata to include in outputs. When
            omitted, strategy-sensitive enrichments fall back to graph-level metadata
            when available.
        :param str theme_name: Theme for visual color defaults.
        :param Optional[Dict[Hashable, Iterable[float]]] layout: Optional precomputed
            layout.
        """
        self.graph = graph
        self.seed_id = seed_id
        self.metadata = metadata or {}
        self.theme = get_theme(theme_name)
        self._layout = layout
        self._size_map: dict[Hashable, float] | None = None
        self._color_map_cache: dict[str, dict[Hashable, tuple]] = {}

    # ------------------------------------------------------------------
    # Public export methods

    def graph_payload(self) -> dict[str, Any]:
        """Build the canonical versioned graph payload shared by JSON consumers.

        Dashboard geometry is always embedded (computing a layout on demand when
        the caller did not supply one) so every ``kind``-stamped payload can be
        loaded back through the dashboard's Load Results flow regardless of
        which export formats were requested or in which order exporters ran.

        :return Dict[str, Any]: Portable CiteMesh graph payload with dashboard data.
        """
        enriched = self._enriched_nodes()
        sorted_edges = _sorted_edges(self.graph)
        dashboard_node_ids = [node_id for node_id, _ in _sorted_nodes(self.graph)]
        dashboard_meta = self._dashboard_meta(
            theme_obj=self.theme,
            node_ids=dashboard_node_ids,
            node_payloads=enriched,
            sorted_edges=sorted_edges,
            include_plotly_geometry=True,
        )
        portable_meta: dict[str, Any] = {
            "strategy": dashboard_meta["strategy"],
            "year_range": dashboard_meta["year_range"],
        }
        candidate_source_status = dashboard_meta.get("candidate_source_status")
        if isinstance(candidate_source_status, dict):
            portable_meta["candidate_source_status"] = candidate_source_status
        return {
            "kind": GRAPH_PAYLOAD_KIND,
            "schema_version": GRAPH_PAYLOAD_SCHEMA_VERSION,
            "seed_id": str(self.seed_id),
            "meta": portable_meta,
            "summary": dashboard_meta["summary"],
            "nodes": enriched,
            "dashboard": {
                "meta": dashboard_meta,
            },
            "edges": [
                {
                    "source": str(u),
                    "target": str(v),
                    "source_title": _node_title(self.graph.nodes[u], u),
                    "target_title": _node_title(self.graph.nodes[v], v),
                    "source_label": _node_short_label(self.graph.nodes[u], u),
                    "target_label": _node_short_label(self.graph.nodes[v], v),
                    "weight": float(edge_data.get("weight", 0.0)),
                }
                for u, v, edge_data in sorted_edges
            ],
        }

    def to_json(self, path: Path) -> None:
        """Export the canonical enriched graph payload as atomic UTF-8 JSON.

        :param Path path: Destination JSON path.
        :return None: Writes the graph payload to disk.
        """
        atomic_write_text(
            path,
            json.dumps(self.graph_payload(), sort_keys=True, indent=2, allow_nan=False),
        )

    def to_csv(self, path: Path) -> None:
        """Export flat CSV table with one row per paper.

        Includes all enriched fields for direct import into pandas or
        spreadsheets.

        :param Path path: Destination CSV file.
        :return None: Writes column headers even when the graph is empty.
        """
        enriched = self._enriched_nodes()
        columns = [
            "id",
            "title",
            "year",
            "authors",
            "citation_count",
            "venue",
            "arxiv_id",
            "doi",
            "categories",
            "is_seed",
            "provenance",
            "seed_relation",
            "seed_relevance",
            "arxiv_url",
            "doi_url",
            "semantic_scholar_url",
            "abstract",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for node in enriched:
            links = node.get("links") or {}
            row = {
                "id": _csv_cell_guard(node.get("id", "")),
                "title": _csv_cell_guard(node.get("title", "")),
                "year": node.get("year", ""),
                "authors": _csv_cell_guard("; ".join(node.get("authors", []))),
                "citation_count": node.get("citation_count", 0),
                "venue": _csv_cell_guard(node.get("venue", "")),
                "arxiv_id": _csv_cell_guard(node.get("arxiv_id", "")),
                "doi": _csv_cell_guard(node.get("doi", "")),
                "categories": _csv_cell_guard("; ".join(node.get("categories", []))),
                # Lowercase true/false matches the dashboard's Export CSV button.
                "is_seed": "true" if node.get("is_seed", False) else "false",
                "provenance": _csv_cell_guard(node.get("provenance", "")),
                "seed_relation": _csv_cell_guard(node.get("seed_relation", "")),
                "seed_relevance": f"{node.get('seed_relevance', 0.0):.6f}",
                "arxiv_url": _csv_cell_guard(links.get("arxiv_abs", "")),
                "doi_url": _csv_cell_guard(links.get("doi", "")),
                "semantic_scholar_url": _csv_cell_guard(
                    links.get("semantic_scholar", "")
                ),
                "abstract": _csv_cell_guard(node.get("abstract", "")),
            }
            writer.writerow(row)
        # csv writes RFC 4180 "\r\n" terminators itself; newline="" keeps the
        # text layer from translating them again into "\r\r\n" on Windows.
        atomic_write_text(Path(path), buf.getvalue())

    def to_bibtex(self, path: Path) -> None:
        """Export all papers as a single BibTeX file."""
        enriched = self._enriched_nodes()
        entries = [
            str(node.get("bibtex", "")).strip()
            for node in enriched
            if node.get("bibtex", "").strip()
        ]
        atomic_write_text(path, "\n\n".join(entries) + "\n")

    def to_graphml(self, path: Path) -> None:
        """Export to GraphML for external tools such as Gephi or Cytoscape."""
        determinism_policy = _graphml_determinism_policy()
        if determinism_policy == GRAPHML_DETERMINISM_POLICY_BEST_EFFORT:
            logger.warning(
                "GraphML serialization is deterministic only as best-effort "
                "on this NetworkX version (%s).",
                nx.__version__,
            )

        export_graph = nx.Graph()
        sorted_nodes = _sorted_nodes(self.graph)
        sorted_edges = _sorted_edges(self.graph)
        export_graph.graph[GRAPHML_LAYOUT_METADATA_KEY] = determinism_policy
        export_graph.graph[GRAPHML_LAYOUT_VERSION_KEY] = nx.__version__
        for metadata_key in sorted(self.metadata, key=str):
            graph_key = _graphml_metadata_key(metadata_key)
            export_graph.graph[graph_key] = _xml_safe_graph_value(
                _graphml_metadata_value(self.metadata[metadata_key])
            )

        for node, attrs in sorted_nodes:
            cleaned = _serialize_node(node, attrs)
            cleaned["year"] = coerce_publication_year(cleaned.get("year"))
            if isinstance(cleaned.get("authors"), list):
                cleaned["authors"] = ", ".join(cleaned["authors"])
            if isinstance(cleaned.get("categories"), list):
                cleaned["categories"] = ", ".join(cleaned["categories"])
            cleaned["is_seed"] = int(bool(cleaned.get("is_seed")))
            cleaned = {
                key: _xml_safe_graph_value(value) for key, value in cleaned.items()
            }
            export_graph.add_node(node, **_ordered_attrs(cleaned))

        for u, v, data in sorted_edges:
            export_graph.add_edge(
                u,
                v,
                **_ordered_attrs(
                    {
                        k: float(val) if k == "weight" else _xml_safe_graph_value(val)
                        for k, val in data.items()
                    }
                ),
            )

        buffer = io.BytesIO()
        nx.write_graphml(export_graph, buffer)
        atomic_write_text(path, buffer.getvalue().decode("utf-8"))

    def to_interactive_html(
        self,
        path: Path,
        theme: str | None = None,
        physics: bool = True,
    ) -> None:
        """
        Create interactive HTML visualization with pyvis (vis.js).

        :param Path path: Output HTML path.
        :param Optional[str] theme: Optional override for theme.
        :param bool physics: Whether to enable force-directed physics.
        """
        try:
            network_cls = loaders._load_pyvis_network_class()
        except ImportError as exc:
            raise RuntimeError(
                "pyvis is required for HTML export. Install with: pip install citemesh[viz]."
            ) from exc

        theme_obj = get_theme(theme) if theme else self.theme

        net = network_cls(
            height="900px",
            width="100%",
            bgcolor=theme_obj.background,
            font_color=theme_obj.text_color,
            notebook=False,
            # Pyvis defaults to "local", which references a lib/ folder it
            # writes into the *process* working directory rather than next to
            # the export, and pulls vis-network from a CDN. Inlining keeps the
            # single HTML file openable offline and side-effect free.
            cdn_resources="in_line",
        )

        if physics:
            net.set_options(
                """
                {
                    "physics": {
                        "forceAtlas2Based": {
                            "gravitationalConstant": -50,
                            "centralGravity": 0.01,
                            "springLength": 100,
                            "springConstant": 0.08
                        },
                        "maxVelocity": 50,
                        "solver": "forceAtlas2Based",
                        "timestep": 0.35,
                        "stabilization": {"iterations": 150}
                    }
                }
                """
            )

        for node, attrs in _sorted_nodes(self.graph):
            paper: Paper | None = attrs.get("paper")
            size = self._node_size(node)
            color = self._node_color_hex(node, theme_obj)

            label = paper.label if paper else attrs.get("title", node)

            tooltip_lines = []
            if paper:
                tooltip_lines.append(f"<b>{html.escape(paper.title)}</b>")
                tooltip_lines.append(
                    f"{html.escape(paper.first_author_surname)} et al., {paper.year}"
                )
                tooltip_lines.append(f"Citations: {paper.citation_count}")
                if paper.categories:
                    cats = ", ".join(html.escape(cat) for cat in paper.categories[:3])
                    tooltip_lines.append(f"Categories: {cats}")
            else:
                tooltip_lines.append(html.escape(attrs.get("title", "")))

            net.add_node(
                node,
                label=label,
                title="<br>".join(tooltip_lines),
                size=max(6, size / 30),
                color=color,
                borderWidth=3 if attrs.get("is_seed") else 1,
            )

        for u, v, data in _sorted_edges(self.graph):
            weight = float(data.get("weight", 0.1))
            net.add_edge(u, v, value=max(0.1, weight * 5))

        with atomic_output_path(path) as tmp_path:
            net.save_graph(str(tmp_path))
            geometry._inject_darkreader_lock(
                tmp_path,
                _theme_color_scheme(theme_obj),
                title=_export_page_title(self.graph, self.seed_id),
                style=_export_page_style(theme_obj)
                + "#mynetwork{border:0 !important;}",
                # Pyvis always emits Bootstrap CDN tags (used only by its
                # select/filter menus, which this export does not enable) plus
                # a commented-out node_modules block; both are dropped so the
                # page has no external references left.
                strip_remote_assets=True,
            )

    def to_plotly_html(self, path: Path, theme: str | None = None) -> None:
        """Create Plotly interactive visualization.

        :param Path path: Output HTML path.
        :param Optional[str] theme: Optional theme override.
        """
        try:
            go = loaders._load_plotly_graph_objects()
        except ImportError as exc:
            raise RuntimeError(
                "plotly is required for Plotly export. Install with: pip install citemesh[viz]."
            ) from exc

        theme_obj = get_theme(theme) if theme else self.theme
        fig, _ = self._build_plotly_figure(
            go=go, theme_obj=theme_obj, title_prefix=PAGE_TITLE_PREFIX
        )

        div_id = self._plotly_div_id()
        with atomic_output_path(path) as tmp_path:
            try:
                fig.write_html(
                    str(tmp_path),
                    div_id=div_id,
                    # Plotly sizes the div from the figure layout by default,
                    # leaving a white band under the plot; the graph should own
                    # the viewport instead.
                    default_width="100%",
                    default_height="100vh",
                )
            except TypeError as exc:
                raise RuntimeError(
                    "Deterministic Plotly export requires write_html(div_id=...). "
                    "Upgrade plotly to a version that supports div_id."
                ) from exc
            geometry._inject_darkreader_lock(
                tmp_path,
                _theme_color_scheme(theme_obj),
                title=_export_page_title(self.graph, self.seed_id),
                style=_export_page_style(theme_obj),
            )

    def to_dashboard_html(self, path: Path, theme: str | None = None) -> None:
        """Create a standalone Plotly-backed research dashboard HTML export.

        :param Path path: Output HTML path.
        :param Optional[str] theme: Optional theme override.
        """
        try:
            go, get_plotlyjs = loaders._load_plotly_dashboard_runtime()
        except ImportError as exc:
            raise RuntimeError(
                "plotly is required for Dashboard export. Install with: pip install citemesh[viz]."
            ) from exc

        theme_obj = get_theme(theme) if theme else self.theme
        fig, node_ids = self._build_plotly_figure(
            go=go,
            theme_obj=theme_obj,
            title_prefix=None,
            margin_top=12,
            for_dashboard=True,
        )
        div_id = self._plotly_div_id(prefix="citemesh-dashboard-plotly")
        payload = self._dashboard_payload(theme_obj=theme_obj, node_ids=node_ids)
        payload_json = _script_safe_json(payload)
        figure_json = _script_safe_json(fig.to_plotly_json())
        collection_json = _script_safe_json(self._dashboard_collection_bundle())
        html_output = _dashboard_template(
            theme_obj=theme_obj,
            div_id=div_id,
            plotly_js=_safe_script_content(get_plotlyjs()),
            payload_json=payload_json,
            figure_json=figure_json,
            collection_json=collection_json,
        )
        atomic_write_text(path, html_output)
