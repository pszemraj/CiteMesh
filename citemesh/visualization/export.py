"""
Graph export utilities for CiteMesh.

Provides a unified interface for exporting graphs to multiple formats,
including interactive visualizations.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import logging
import math
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, Optional, Tuple
from urllib.parse import quote

import networkx as nx

from citemesh.core import Paper

from .ordering import ordered_edges_with_data, ordered_nodes
from .render import (
    MISSING_YEAR_FALLBACK_MAX,
    MISSING_YEAR_FALLBACK_MIN,
    _normalize_layout_positions,
    compute_layout,
    compute_node_colors,
    compute_node_sizes,
)
from .themes import Theme, get_theme

logger = logging.getLogger(__name__)

GRAPHML_DETERMINISM_POLICY_STRICT = "strict_sorted_nodes_edges"
GRAPHML_DETERMINISM_POLICY_BEST_EFFORT = "best_effort_sorted_nodes_edges"
_GRAPHML_BEST_EFFORT_MIN_VERSION = (2, 8)
GRAPHML_LAYOUT_METADATA_KEY = "citemesh_graphml_determinism"
GRAPHML_LAYOUT_VERSION_KEY = "citemesh_graphml_writer_version"


def _load_pyvis_network_class() -> Any:
    """Import and return the PyVis ``Network`` class.

    :return Any: Imported ``pyvis.network.Network`` class.
    """
    from pyvis.network import Network

    return Network


def _load_plotly_graph_objects() -> Any:
    """Import and return Plotly graph objects.

    :return Any: Imported ``plotly.graph_objects`` module proxy.
    """
    from plotly import graph_objects as go

    return go


def _load_plotly_dashboard_runtime() -> tuple[Any, Any]:
    """Import and return Plotly graph objects plus inline JS provider.

    :return tuple[Any, Any]: Plotly graph objects and ``get_plotlyjs`` callable.
    """
    from plotly.offline import get_plotlyjs

    return _load_plotly_graph_objects(), get_plotlyjs


def _graphml_determinism_policy() -> str:
    """Return determinism policy name for active NetworkX writer runtime.

    :return str: Determinism policy identifier for current NetworkX version.
    """
    major_minor = tuple(int(part) for part in re.findall(r"\d+", nx.__version__)[:2])
    if len(major_minor) < 2:
        return GRAPHML_DETERMINISM_POLICY_BEST_EFFORT

    major, minor = major_minor[0], major_minor[1]
    if (major, minor) >= _GRAPHML_BEST_EFFORT_MIN_VERSION:
        return GRAPHML_DETERMINISM_POLICY_STRICT
    return GRAPHML_DETERMINISM_POLICY_BEST_EFFORT


def _ordered_attrs(attrs: Dict[str, object]) -> Dict[str, object]:
    """Return a copy of mapping with deterministic key ordering.

    :param Dict[str, object] attrs: Source attribute mapping.
    :return Dict[str, object]: Copy with key order normalized by key string.
    """
    return {key: attrs[key] for key in sorted(attrs, key=str)}


class GraphExporter:
    """Unified interface for exporting graphs in multiple formats."""

    def __init__(
        self,
        graph: nx.Graph,
        seed_id: str,
        metadata: Optional[Dict] = None,
        theme_name: str = "light",
        layout: Optional[Dict[Hashable, Iterable[float]]] = None,
    ):
        """Create exporter bound to a graph and seed paper metadata.

        :param nx.Graph graph: Graph to export.
        :param str seed_id: Seed paper identifier.
        :param Optional[Dict] metadata: Optional metadata to include in outputs.
        :param str theme_name: Theme for visual color defaults.
        :param Optional[Dict[Hashable, Iterable[float]]] layout: Optional precomputed
            layout.
        """
        self.graph = graph
        self.seed_id = seed_id
        self.metadata = metadata or {}
        self.theme = get_theme(theme_name)
        self._layout = layout
        self._size_map: Optional[Dict[Hashable, float]] = None
        self._color_map_cache: Dict[str, Dict[Hashable, tuple]] = {}

    @staticmethod
    def _graphml_metadata_key(raw_key: object) -> str:
        """Normalize metadata key for GraphML graph-level attributes.

        :param object raw_key: Source metadata key.
        :return str: Sanitized GraphML-safe key.
        """
        normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", str(raw_key)).strip("_")
        if not normalized:
            normalized = "metadata"
        return f"citemesh_meta_{normalized}"

    @staticmethod
    def _graphml_metadata_value(raw_value: object) -> str:
        """Normalize metadata value for GraphML graph-level attributes.

        :param object raw_value: Source metadata value.
        :return str: Scalar/serialized value for GraphML export.
        """
        if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
            return str(raw_value)
        return json.dumps(raw_value, sort_keys=True)

    # ------------------------------------------------------------------
    # Public export methods

    def to_json(self, path: Path) -> None:
        """Export enriched graph data JSON with analysis fields.

        Includes provenance, seed relevance scores, external links, and
        BibTeX entries — the same rich fields available in the dashboard.
        """
        enriched = self._enriched_nodes()
        sorted_edges = self._sorted_edges()
        strategy = str(self.metadata.get("strategy") or "").strip().lower()
        dashboard_node_ids = [node_id for node_id, _ in self._sorted_nodes()]
        dashboard_payload = self._dashboard_payload(
            theme_obj=self.theme,
            node_ids=dashboard_node_ids,
        )
        valid_years = [
            int(n.get("year", 0)) for n in enriched if int(n.get("year", 0)) > 0
        ]
        data = {
            "seed_id": str(self.seed_id),
            "meta": {
                "strategy": strategy,
                "year_range": (
                    {"min": min(valid_years), "max": max(valid_years)}
                    if valid_years
                    else {
                        "min": MISSING_YEAR_FALLBACK_MIN,
                        "max": MISSING_YEAR_FALLBACK_MAX,
                    }
                ),
            },
            "summary": {
                "nodes": len(enriched),
                "edges": len(sorted_edges),
            },
            "nodes": enriched,
            "dashboard": {
                "meta": dashboard_payload["meta"],
            },
            "edges": [
                {
                    "source": str(u),
                    "target": str(v),
                    "source_title": self._node_title(self.graph.nodes[u], u),
                    "target_title": self._node_title(self.graph.nodes[v], v),
                    "source_label": self._node_short_label(self.graph.nodes[u], u),
                    "target_label": self._node_short_label(self.graph.nodes[v], v),
                    "weight": float(edge_data.get("weight", 0.0)),
                }
                for u, v, edge_data in sorted_edges
            ],
        }
        Path(path).write_text(json.dumps(data, sort_keys=True, indent=2))

    def to_csv(self, path: Path) -> None:
        """Export flat CSV table with one row per paper.

        Includes all enriched fields for direct import into pandas or
        spreadsheets.
        """
        enriched = self._enriched_nodes()
        if not enriched:
            Path(path).write_text("")
            return

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
                "id": node.get("id", ""),
                "title": node.get("title", ""),
                "year": node.get("year", ""),
                "authors": "; ".join(node.get("authors", [])),
                "citation_count": node.get("citation_count", 0),
                "venue": node.get("venue", ""),
                "arxiv_id": node.get("arxiv_id", ""),
                "doi": node.get("doi", ""),
                "categories": "; ".join(node.get("categories", [])),
                "is_seed": node.get("is_seed", False),
                "provenance": node.get("provenance", ""),
                "seed_relation": node.get("seed_relation", ""),
                "seed_relevance": f"{node.get('seed_relevance', 0.0):.6f}",
                "arxiv_url": links.get("arxiv_abs", ""),
                "doi_url": links.get("doi", ""),
                "semantic_scholar_url": links.get("semantic_scholar", ""),
                "abstract": node.get("abstract", ""),
            }
            writer.writerow(row)
        Path(path).write_text(buf.getvalue(), encoding="utf-8")

    def to_bibtex(self, path: Path) -> None:
        """Export all papers as a single BibTeX file."""
        enriched = self._enriched_nodes()
        entries = [
            str(node.get("bibtex", "")).strip()
            for node in enriched
            if node.get("bibtex", "").strip()
        ]
        Path(path).write_text("\n\n".join(entries) + "\n", encoding="utf-8")

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
        sorted_nodes = self._sorted_nodes()
        sorted_edges = self._sorted_edges()
        export_graph.graph[GRAPHML_LAYOUT_METADATA_KEY] = determinism_policy
        export_graph.graph[GRAPHML_LAYOUT_VERSION_KEY] = nx.__version__
        for metadata_key in sorted(self.metadata, key=str):
            graph_key = self._graphml_metadata_key(metadata_key)
            export_graph.graph[graph_key] = self._graphml_metadata_value(
                self.metadata[metadata_key]
            )

        for node, attrs in sorted_nodes:
            cleaned = self._serialize_node(node, attrs)
            cleaned["year"] = self._coerce_year(cleaned.get("year"))
            if isinstance(cleaned.get("authors"), list):
                cleaned["authors"] = ", ".join(cleaned["authors"])
            if isinstance(cleaned.get("categories"), list):
                cleaned["categories"] = ", ".join(cleaned["categories"])
            cleaned["is_seed"] = int(bool(cleaned.get("is_seed")))
            export_graph.add_node(node, **_ordered_attrs(cleaned))

        for u, v, data in sorted_edges:
            export_graph.add_edge(
                u,
                v,
                **_ordered_attrs(
                    {k: float(val) if k == "weight" else val for k, val in data.items()}
                ),
            )

        nx.write_graphml(export_graph, path)

    def to_interactive_html(
        self,
        path: Path,
        theme: Optional[str] = None,
        physics: bool = True,
    ) -> None:
        """
        Create interactive HTML visualization with pyvis (vis.js).

        :param Path path: Output HTML path.
        :param Optional[str] theme: Optional override for theme.
        :param bool physics: Whether to enable force-directed physics.
        """
        try:
            network_cls = _load_pyvis_network_class()
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

        for node, attrs in self._sorted_nodes():
            paper: Optional[Paper] = attrs.get("paper")
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

        for u, v, data in self._sorted_edges():
            weight = float(data.get("weight", 0.1))
            net.add_edge(u, v, value=max(0.1, weight * 5))

        net.save_graph(str(path))

    def to_plotly_html(self, path: Path, theme: Optional[str] = None) -> None:
        """Create Plotly interactive visualization.

        :param Path path: Output HTML path.
        :param Optional[str] theme: Optional theme override.
        """
        try:
            go = _load_plotly_graph_objects()
        except ImportError as exc:
            raise RuntimeError(
                "plotly is required for Plotly export. Install with: pip install citemesh[viz]."
            ) from exc

        theme_obj = get_theme(theme) if theme else self.theme
        fig, _ = self._build_plotly_figure(go=go, theme_obj=theme_obj)

        div_id = self._plotly_div_id()
        try:
            fig.write_html(str(path), div_id=div_id)
        except TypeError as exc:
            raise RuntimeError(
                "Deterministic Plotly export requires write_html(div_id=...). "
                "Upgrade plotly to a version that supports div_id."
            ) from exc

    def to_dashboard_html(self, path: Path, theme: Optional[str] = None) -> None:
        """Create a standalone Plotly-backed research dashboard HTML export.

        :param Path path: Output HTML path.
        :param Optional[str] theme: Optional theme override.
        """
        try:
            go, get_plotlyjs = _load_plotly_dashboard_runtime()
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
        payload_json = self._safe_script_content(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        figure_json = self._safe_script_content(
            json.dumps(fig.to_plotly_json(), sort_keys=True, separators=(",", ":"))
        )
        collection_json = self._safe_script_content(
            json.dumps(
                self._dashboard_collection_bundle(),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        html_output = self._dashboard_template(
            theme_obj=theme_obj,
            div_id=div_id,
            plotly_js=self._safe_script_content(get_plotlyjs()),
            payload_json=payload_json,
            figure_json=figure_json,
            collection_json=collection_json,
        )
        Path(path).write_text(html_output, encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal helpers

    def _build_plotly_figure(
        self,
        *,
        go: Any,
        theme_obj: Theme,
        title_prefix: Optional[str] = "CiteMesh",
        margin_top: int = 40,
        for_dashboard: bool = False,
    ) -> tuple[Any, list[Hashable]]:
        """Build a deterministic Plotly figure and point-order mapping.

        :param Any go: Plotly graph_objects module.
        :param Theme theme_obj: Active visualization theme.
        :param Optional[str] title_prefix: Optional title prefix. When ``None``,
            the figure omits a title.
        :param int margin_top: Top plot margin.
        :param bool for_dashboard: Whether to apply dashboard-specific styling
            (subtle curved edges, muted colors, and reduced label density).
        :return tuple[Any, list[Hashable]]: Plotly figure and ordered node IDs.
        """
        pos = self._get_layout()
        layout_shapes: list[Dict[str, Any]] = []
        edge_trace: Optional[Any] = None

        if for_dashboard:
            curvature = 0.15
            for u, v, attrs in self._sorted_edges():
                x0f = float(pos[u][0])
                y0f = float(pos[u][1])
                x1f = float(pos[v][0])
                y1f = float(pos[v][1])
                mid_x = (x0f + x1f) / 2.0
                mid_y = (y0f + y1f) / 2.0
                dx = x1f - x0f
                dy = y1f - y0f
                key_left, key_right = sorted((str(u), str(v)))
                direction_digest = hashlib.sha1(
                    f"{key_left}|{key_right}".encode("utf-8")
                ).hexdigest()
                direction = -1.0 if int(direction_digest[:2], 16) % 2 else 1.0
                cx = mid_x - dy * curvature * direction
                cy = mid_y + dx * curvature * direction
                weight = max(float(attrs.get("weight", 0.0)), 0.0)
                alpha = min(0.6, max(0.05, weight))
                layout_shapes.append(
                    {
                        "type": "path",
                        "path": f"M {x0f},{y0f} Q {cx},{cy} {x1f},{y1f}",
                        "line": {
                            "color": _rgb_tuple_to_rgba(theme_obj.edge_color, alpha),
                            "width": max(0.5, weight * 2.0),
                        },
                        "layer": "below",
                    }
                )
        else:
            edge_x: list[float | None] = []
            edge_y: list[float | None] = []
            for u, v, _ in self._sorted_edges():
                x0f = float(pos[u][0])
                y0f = float(pos[u][1])
                x1f = float(pos[v][0])
                y1f = float(pos[v][1])
                edge_x.extend([x0f, x1f, None])
                edge_y.extend([y0f, y1f, None])

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color=_rgb_tuple_to_hex(theme_obj.edge_color)),
                hoverinfo="none",
                mode="lines",
            )

        node_ids = [node_id for node_id, _ in self._sorted_nodes()]
        node_x = [float(pos[node][0]) for node in node_ids]
        node_y = [float(pos[node][1]) for node in node_ids]
        node_sizes = [max(6, self._node_size(node) / 50) for node in node_ids]
        max_node_size = max(node_sizes) if node_sizes else 1.0
        marker_sizeref = max(2.0 * max_node_size / (45.0**2), 1e-6)
        node_years, year_min, year_max = self._plotly_year_scale(node_ids)
        base_labels = [
            self.graph.nodes[node].get("paper").label
            if self.graph.nodes[node].get("paper")
            else self.graph.nodes[node].get("title", node)
            for node in node_ids
        ]
        if for_dashboard:
            ranked_label_nodes = sorted(
                node_ids,
                key=lambda node_id: (
                    0 if bool(self.graph.nodes[node_id].get("is_seed", False)) else 1,
                    -int(self.graph.nodes[node_id].get("citation_count", 0) or 0),
                    self._coerce_year(self.graph.nodes[node_id].get("year")) * -1,
                    str(node_id),
                ),
            )
            label_cap = min(14, len(ranked_label_nodes))
            label_nodes = set(ranked_label_nodes[:label_cap])
            node_labels = [
                str(base_labels[idx]) if node_id in label_nodes else ""
                for idx, node_id in enumerate(node_ids)
            ]
            text_position = "top center"
            # Keep one marker trace so point indices stay stable for hover/click sync;
            # use a muted shared text alpha instead of per-point text styling.
            text_font = dict(
                size=10, color=_rgb_tuple_to_rgba(theme_obj.text_color, 0.62)
            )
            color_scale: object = [
                [0.0, _rgb_tuple_to_hex(theme_obj.node_color_old)],
                [1.0, _rgb_tuple_to_hex(theme_obj.node_color_new)],
            ]
            marker_line_width = [
                4.0 if self.graph.nodes[node].get("is_seed") else 0.0
                for node in node_ids
            ]
            marker_line_color = [
                _rgb_tuple_to_hex(theme_obj.seed_color)
                if self.graph.nodes[node].get("is_seed")
                else _rgb_tuple_to_rgba(theme_obj.background, 0.0)
                for node in node_ids
            ]
            marker_showscale = False
            marker_colorbar: Optional[Dict[str, Any]] = None
        else:
            node_labels = [str(label) for label in base_labels]
            text_position = "bottom center"
            text_font = dict(size=8, color=theme_obj.text_color)
            color_scale = "Plasma" if theme_obj.name == "dark" else "Viridis"
            marker_line_width = 2
            marker_line_color = theme_obj.text_color
            marker_showscale = True
            marker_colorbar = dict(
                thickness=15,
                xanchor="left",
                title=dict(text="Year", side="right"),
            )

        hover_texts = []
        for node in node_ids:
            paper: Optional[Paper] = self.graph.nodes[node].get("paper")
            if paper:
                authors = ", ".join(a.name for a in paper.authors[:3]) or "Unknown"
                hover_texts.append(
                    "<br>".join(
                        [
                            f"<b>{html.escape(paper.title)}</b>",
                            html.escape(authors),
                            f"Year: {paper.year} | Citations: {paper.citation_count}",
                        ]
                    )
                )
            else:
                hover_texts.append(
                    html.escape(self.graph.nodes[node].get("title", node))
                )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            name="nodes",
            mode="markers+text",
            hoverinfo="text",
            text=node_labels,
            textposition=text_position,
            textfont=text_font,
            marker=dict(
                size=node_sizes,
                sizemode="area",
                sizeref=marker_sizeref,
                sizemin=3,
                color=node_years,
                cmin=year_min,
                cmax=year_max,
                colorscale=color_scale,
                line=dict(width=marker_line_width, color=marker_line_color),
                showscale=marker_showscale,
                colorbar=marker_colorbar,
            ),
            hovertext=hover_texts,
        )

        halo_trace: Optional[Any] = None
        neighborhood_trace: Optional[Any] = None
        if for_dashboard:
            seed_index = next(
                (
                    idx
                    for idx, node in enumerate(node_ids)
                    if bool(self.graph.nodes[node].get("is_seed", False))
                ),
                None,
            )
            halo_x: list[float] = []
            halo_y: list[float] = []
            halo_sizes: list[float] = []
            halo_colors: list[str] = []
            if seed_index is not None:
                halo_x = [node_x[seed_index]]
                halo_y = [node_y[seed_index]]
                halo_sizes = [node_sizes[seed_index] * 2.05]
                halo_colors = [_rgb_tuple_to_rgba(theme_obj.seed_color, 0.26)]
            halo_trace = go.Scatter(
                x=halo_x,
                y=halo_y,
                name="selection-halo",
                mode="markers",
                hoverinfo="none",
                showlegend=False,
                marker=dict(
                    size=halo_sizes,
                    color=halo_colors,
                    line=dict(width=0),
                    opacity=0.96,
                    sizemode="area",
                    sizeref=marker_sizeref,
                    sizemin=3,
                ),
            )
            neighborhood_trace = go.Scatter(
                x=[],
                y=[],
                name="neighborhood-edges",
                mode="lines",
                hoverinfo="none",
                showlegend=False,
                line=dict(
                    width=1.4,
                    color=_rgb_tuple_to_rgba(theme_obj.seed_color, 0.54),
                ),
                opacity=0.98,
            )

        layout_kwargs: Dict[str, Any] = {
            "showlegend": False,
            "hovermode": "closest",
            "margin": dict(b=20, l=5, r=5, t=max(0, int(margin_top))),
            "xaxis": dict(showgrid=False, zeroline=False, showticklabels=False),
            "yaxis": dict(showgrid=False, zeroline=False, showticklabels=False),
            "plot_bgcolor": theme_obj.background,
            "paper_bgcolor": theme_obj.background,
            "font": dict(color=theme_obj.text_color),
        }
        if for_dashboard and node_x and node_y:
            x_min = min(node_x)
            x_max = max(node_x)
            y_min = min(node_y)
            y_max = max(node_y)
            x_span = max(x_max - x_min, 1e-6)
            y_span = max(y_max - y_min, 1e-6)
            x_pad = max(0.28, x_span * 0.08)
            y_pad = max(0.28, y_span * 0.08)
            layout_kwargs["xaxis"].update(
                {"autorange": False, "range": [x_min - x_pad, x_max + x_pad]}
            )
            layout_kwargs["yaxis"].update(
                {"autorange": False, "range": [y_min - y_pad, y_max + y_pad]}
            )
            # Keep Plotly restyle updates from re-autoscaling and shifting node positions.
            layout_kwargs["uirevision"] = "citemesh-dashboard-static-layout-v1"
        if for_dashboard and layout_shapes:
            layout_kwargs["shapes"] = layout_shapes
        if title_prefix is not None:
            layout_kwargs["title"] = f"{title_prefix}: {self._plotly_title_text()}"

        if for_dashboard:
            traces = []
            if halo_trace is not None:
                traces.append(halo_trace)
            if neighborhood_trace is not None:
                traces.append(neighborhood_trace)
            traces.append(node_trace)
        else:
            traces = [edge_trace, node_trace]
        fig = go.Figure(data=traces, layout=go.Layout(**layout_kwargs))
        return fig, node_ids

    def _plotly_title_text(self) -> str:
        """Build wrapped seed title text used by Plotly figure titles.

        :return str: Wrapped title string.
        """
        raw_title = " ".join(
            str(self.graph.nodes[self.seed_id].get("title", "CiteMesh")).split()
        )
        title_text = "<br>".join(
            textwrap.wrap(raw_title, width=72, break_long_words=False)
        )
        if not title_text:
            return "CiteMesh"
        return title_text

    def _enriched_nodes(self) -> list[Dict[str, Any]]:
        """Build enriched node payloads with provenance, relevance, links, and BibTeX.

        This is the canonical node enrichment used by JSON export, CSV export,
        BibTeX export, and the dashboard payload.

        :return list[Dict[str, Any]]: Enriched node payloads.
        """
        provenance = self._provenance_map()
        seed_relations = self._seed_relation_map()
        relevance = self._seed_relevance_scores()
        sorted_nodes = self._sorted_nodes()
        strategy = str(self.metadata.get("strategy") or "").strip().lower()

        node_payloads: list[Dict[str, Any]] = []
        for node_id, attrs in sorted_nodes:
            node_str = str(node_id)
            serialized = self._serialize_node(node_id, attrs)
            serialized["id"] = node_str
            serialized["year"] = self._coerce_year(serialized.get("year"))
            serialized["citation_count"] = max(
                int(serialized.get("citation_count") or 0), 0
            )
            serialized["authors"] = [
                str(author).strip()
                for author in serialized.get("authors", [])
                if str(author).strip()
            ]
            serialized["venue"] = str(serialized.get("venue") or "").strip()
            serialized["arxiv_id"] = str(serialized.get("arxiv_id") or "").strip()
            serialized["doi"] = str(serialized.get("doi") or "").strip()
            serialized["categories"] = [
                str(category).strip()
                for category in serialized.get("categories", [])
                if str(category).strip()
            ]
            serialized["abstract"] = str(serialized.get("abstract") or "").strip()
            serialized["is_seed"] = bool(serialized.get("is_seed", False))

            provenance_base = provenance.get(
                node_str, self._default_provenance(strategy=strategy)
            )
            serialized["provenance"] = (
                "seed" if serialized["is_seed"] else provenance_base
            )
            serialized["provenance_base"] = provenance_base
            seed_relation = seed_relations.get(node_str, "")
            if serialized["is_seed"]:
                seed_relation = "seed"
            if not seed_relation and serialized["provenance"] == "semantic":
                seed_relation = "semantic_only"
            serialized["seed_relation"] = seed_relation
            serialized["seed_relevance"] = float(relevance.get(node_str, 0.0))
            links = self._derive_links(node_str, node_payload=serialized)
            serialized["links"] = links
            serialized["bibtex"] = self._node_bibtex(serialized, links=links)
            node_payloads.append(serialized)
        return node_payloads

    def _dashboard_payload(
        self, *, theme_obj: Theme, node_ids: list[Hashable]
    ) -> Dict[str, Any]:
        """Build deterministic dashboard payload from graph metadata.

        :param Theme theme_obj: Active visualization theme.
        :param list[Hashable] node_ids: Node order used by Plotly points.
        :return Dict[str, Any]: JSON payload consumed by dashboard JS.
        """
        node_payloads = self._enriched_nodes()
        sorted_edges = self._sorted_edges()
        strategy = str(self.metadata.get("strategy") or "").strip().lower()
        positions = self._get_layout()
        node_sizes = [max(6.0, self._node_size(node_id) / 50.0) for node_id in node_ids]

        valid_years = [
            int(node.get("year", 0))
            for node in node_payloads
            if int(node.get("year", 0)) > 0
        ]
        if valid_years:
            year_range = {"min": min(valid_years), "max": max(valid_years)}
        else:
            year_range = {
                "min": MISSING_YEAR_FALLBACK_MIN,
                "max": MISSING_YEAR_FALLBACK_MAX,
            }

        payload: Dict[str, Any] = {
            "meta": {
                "seed_id": str(self.seed_id),
                "strategy": strategy,
                "theme": theme_obj.name,
                "summary": {
                    "nodes": len(node_payloads),
                    "edges": len(sorted_edges),
                },
                "year_range": year_range,
                "plotly_node_order": [str(node_id) for node_id in node_ids],
                "plotly_positions": [
                    [float(positions[node_id][0]), float(positions[node_id][1])]
                    for node_id in node_ids
                ],
                "plotly_node_sizes": node_sizes,
            },
            "nodes": node_payloads,
            "edges": [
                {
                    "source": str(left),
                    "target": str(right),
                    "weight": float(data.get("weight", 0.0)),
                }
                for left, right, data in sorted_edges
            ],
        }
        return payload

    def _dashboard_collection_bundle(self) -> Dict[str, Any]:
        """Normalize optional collection metadata for shared dashboard shells.

        :return Dict[str, Any]: Collection result descriptors and embedded payloads.
        """
        raw_bundle = self.metadata.get("dashboard_collection")
        if not isinstance(raw_bundle, dict):
            return {"current_result_id": None, "results": [], "payloads": {}}

        raw_results = raw_bundle.get("results")
        raw_payloads = raw_bundle.get("payloads")
        results = (
            [entry for entry in raw_results if isinstance(entry, dict)]
            if isinstance(raw_results, list)
            else []
        )
        payloads = (
            {
                str(result_id): payload
                for result_id, payload in raw_payloads.items()
                if isinstance(result_id, str) and isinstance(payload, dict)
            }
            if isinstance(raw_payloads, dict)
            else {}
        )
        current_result_id = raw_bundle.get("current_result_id")
        return {
            "current_result_id": (
                str(current_result_id).strip()
                if current_result_id is not None
                else None
            ),
            "results": results,
            "payloads": payloads,
        }

    def _default_provenance(self, *, strategy: str) -> str:
        """Resolve default provenance class for non-hybrid strategies.

        :param str strategy: Strategy metadata token.
        :return str: One of ``citation`` or ``semantic``.
        """
        if strategy in {"embedding", "recommendation"}:
            return "semantic"
        return "citation"

    def _provenance_map(self) -> Dict[str, str]:
        """Resolve normalized per-node provenance map.

        :return Dict[str, str]: Mapping from node ID to provenance class.
        """
        raw_map = self.graph.graph.get("paper_sources")
        if not isinstance(raw_map, dict):
            return {}

        resolved: Dict[str, str] = {}
        for raw_id, raw_value in raw_map.items():
            value = str(raw_value).strip().lower()
            if value not in {"citation", "semantic", "both"}:
                continue
            resolved[str(raw_id)] = value
        return resolved

    def _seed_relation_map(self) -> Dict[str, str]:
        """Resolve normalized relation-to-seed mapping when available.

        :return Dict[str, str]: Node-ID to relation class mapping.
        """
        raw_map = self.graph.graph.get("seed_relations")
        if not isinstance(raw_map, dict):
            return {}

        allowed = {
            "seed",
            "referenced_by_seed",
            "cites_seed",
            "overlap",
            "semantic_only",
            "citation",
        }
        resolved: Dict[str, str] = {}
        for raw_id, raw_value in raw_map.items():
            value = str(raw_value).strip().lower()
            if value in allowed:
                resolved[str(raw_id)] = value
        return resolved

    def _seed_relevance_scores(self) -> Dict[str, float]:
        """Compute seed-centric personalized PageRank scores.

        :return Dict[str, float]: Node-ID keyed relevance scores.
        """
        relevance_graph = nx.Graph()
        for node_id, _ in self._sorted_nodes():
            relevance_graph.add_node(str(node_id))

        for left, right, attrs in self._sorted_edges():
            relevance_graph.add_edge(
                str(left),
                str(right),
                weight=self._normalized_edge_weight(attrs.get("weight", 0.0)),
            )

        if not relevance_graph.nodes:
            return {}

        personalization = {node_id: 0.0 for node_id in relevance_graph.nodes}
        seed_id = str(self.seed_id)
        if seed_id in personalization:
            personalization[seed_id] = 1.0
        else:
            seed_id = next(iter(personalization))
            personalization[seed_id] = 1.0

        try:
            scores = nx.pagerank(
                relevance_graph,
                alpha=0.85,
                personalization=personalization,
                weight="weight",
            )
        except Exception as exc:  # pragma: no cover - highly unlikely fallback
            logger.warning(
                "Failed to compute personalized PageRank relevance; falling back to "
                "uniform relevance scores (%s)",
                exc,
            )
            uniform = 1.0 / float(len(personalization))
            return {node_id: uniform for node_id in personalization}

        return {str(node_id): float(score) for node_id, score in scores.items()}

    @staticmethod
    def _normalized_edge_weight(raw_weight: object) -> float:
        """Normalize edge weights for relevance computation.

        :param object raw_weight: Raw edge weight candidate.
        :return float: Positive finite weight.
        """
        try:
            parsed = float(raw_weight)
        except (TypeError, ValueError):
            parsed = 0.0
        if not math.isfinite(parsed) or parsed <= 0.0:
            return 1e-6
        return parsed

    @staticmethod
    def _safe_script_content(raw: str) -> str:
        """Escape script-closing tokens in inline script payloads.

        :param str raw: Raw script body content.
        :return str: Script-safe content.
        """
        return raw.replace("</", "<\\/")

    def _derive_links(
        self,
        node_id: str,
        *,
        node_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Optional[str]]:
        """Derive external links from canonical node IDs.

        :param str node_id: Canonical graph node identifier.
        :param Optional[Dict[str, Any]] node_payload: Optional node payload carrying
            explicit ``arxiv_id``/``doi`` values.
        :return Dict[str, Optional[str]]: External links dictionary.
        """
        links: Dict[str, Optional[str]] = {
            "arxiv_abs": None,
            "arxiv_pdf": None,
            "doi": None,
            "semantic_scholar": (
                f"https://www.semanticscholar.org/paper/{quote(node_id, safe='')}"
            ),
        }

        arxiv_value = ""
        if isinstance(node_payload, dict):
            arxiv_value = str(node_payload.get("arxiv_id") or "").strip()
        if not arxiv_value:
            arxiv_match = re.match(r"^arxiv:(.+)$", node_id, flags=re.IGNORECASE)
            if arxiv_match:
                arxiv_value = arxiv_match.group(1).strip()
        if arxiv_value:
            arxiv_id = re.sub(r"v\d+$", "", arxiv_value, flags=re.IGNORECASE)
            links["arxiv_abs"] = f"https://arxiv.org/abs/{quote(arxiv_id, safe='')}"
            links["arxiv_pdf"] = f"https://arxiv.org/pdf/{quote(arxiv_id, safe='')}.pdf"

        doi_value = ""
        if isinstance(node_payload, dict):
            doi_value = str(node_payload.get("doi") or "").strip()
        if not doi_value:
            if node_id.lower().startswith("doi:"):
                doi_value = node_id.split(":", 1)[1].strip()
            elif re.match(r"^10\.\d{4,9}/\S+$", node_id):
                doi_value = node_id
        if doi_value:
            links["doi"] = f"https://doi.org/{quote(doi_value, safe='/()[]:._;-')}"
        return links

    @staticmethod
    def _bibtex_entry_key(node_id: str) -> str:
        """Build deterministic BibTeX entry keys from node IDs.

        :param str node_id: Graph node ID.
        :return str: BibTeX entry key.
        """
        normalized = re.sub(r"[^0-9a-zA-Z]+", "_", node_id).strip("_").lower()
        if not normalized:
            normalized = "paper"
        return f"citemesh_{normalized}"

    @staticmethod
    def _bibtex_escape(raw_value: str) -> str:
        """Escape text for conservative BibTeX field rendering.

        :param str raw_value: Raw field value.
        :return str: Escaped value safe for brace-delimited fields.
        """
        collapsed = " ".join(str(raw_value).split())
        collapsed = collapsed.replace("\\", "\\\\")
        collapsed = collapsed.replace("{", "\\{")
        collapsed = collapsed.replace("}", "\\}")
        return collapsed

    def _node_bibtex(
        self, node_payload: Dict[str, Any], *, links: Dict[str, Optional[str]]
    ) -> str:
        """Render a deterministic BibTeX entry for dashboard actions.

        :param Dict[str, Any] node_payload: Node payload.
        :param Dict[str, Optional[str]] links: Derived external links.
        :return str: BibTeX entry string.
        """
        key = self._bibtex_entry_key(str(node_payload.get("id", "")))
        fields: list[tuple[str, str]] = []
        title = str(node_payload.get("title") or "").strip()
        if title:
            fields.append(("title", title))

        authors = node_payload.get("authors", [])
        if isinstance(authors, list):
            author_names = [
                str(author).strip() for author in authors if str(author).strip()
            ]
            if author_names:
                fields.append(("author", " and ".join(author_names)))

        year = self._coerce_year(node_payload.get("year"))
        if year > 0:
            fields.append(("year", str(year)))

        doi_url = links.get("doi")
        if doi_url:
            doi_value = doi_url.replace("https://doi.org/", "", 1)
            fields.append(("doi", doi_value))

        primary_url = (
            links.get("arxiv_abs") or links.get("doi") or links.get("semantic_scholar")
        )
        if primary_url:
            fields.append(("url", primary_url))

        abstract = str(node_payload.get("abstract") or "").strip()
        if abstract:
            fields.append(("abstract", abstract))

        lines = [f"@article{{{key},"]
        for field, value in fields:
            lines.append(f"  {field} = {{{self._bibtex_escape(value)}}},")
        lines.append("}")
        return "\n".join(lines)

    @classmethod
    def _dashboard_template(
        cls,
        *,
        theme_obj: Theme,
        div_id: str,
        plotly_js: str,
        payload_json: str,
        figure_json: str,
        collection_json: str,
    ) -> str:
        """Render standalone dashboard HTML template.

        :param Theme theme_obj: Active theme.
        :param str div_id: Plotly mount div ID.
        :param str plotly_js: Inline Plotly runtime JS.
        :param str payload_json: Serialized dashboard payload JSON.
        :param str figure_json: Serialized Plotly figure JSON.
        :param str collection_json: Serialized collection bundle JSON.
        :return str: Dashboard HTML content.
        """
        is_dark = theme_obj.name in {"dark", "solarized"}
        vars_map = {
            "__BODY_BG__": "#0f1318" if is_dark else "#eef2f7",
            "__PANEL_BG__": "#171d25" if is_dark else "#ffffff",
            "__PANEL_BORDER__": "#2e3948" if is_dark else "#d5dce8",
            "__TEXT_PRIMARY__": "#ecf1f8" if is_dark else "#1b2738",
            "__TEXT_MUTED__": "#9ab0cb" if is_dark else "#5a6a80",
            "__ACCENT__": "#4aa3ff" if is_dark else "#0f67d8",
            "__ACCENT_SOFT__": "rgba(74, 163, 255, 0.2)"
            if is_dark
            else "rgba(15, 103, 216, 0.14)",
            "__GRAPH_BG__": theme_obj.background,
            "__PLOTLY_JS__": plotly_js,
            "__PAYLOAD_JSON__": payload_json,
            "__FIGURE_JSON__": figure_json,
            "__COLLECTION_JSON__": collection_json,
            "__PLOTLY_DIV_ID__": div_id,
        }
        template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CiteMesh Dashboard</title>
  <style>
    :root {
      --body-bg: __BODY_BG__;
      --panel-bg: __PANEL_BG__;
      --panel-border: __PANEL_BORDER__;
      --text-primary: __TEXT_PRIMARY__;
      --text-muted: __TEXT_MUTED__;
      --accent: __ACCENT__;
      --accent-soft: __ACCENT_SOFT__;
      --graph-bg: __GRAPH_BG__;
      --shadow-soft: rgba(0, 0, 0, 0.18);
      --seed-ring: #d66cbf;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      height: 100%;
      overflow: hidden;
      background: radial-gradient(1200px 640px at 18% -12%, rgba(74, 163, 255, 0.12), transparent 58%),
                  radial-gradient(900px 520px at 100% 0%, rgba(214, 108, 191, 0.08), transparent 55%),
                  var(--body-bg);
      color: var(--text-primary);
      font-family: "IBM Plex Sans", "Source Sans 3", "Segoe UI", sans-serif;
    }
    body {
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    input, select, button {
      border: 1px solid var(--panel-border);
      border-radius: 9px;
      background: rgba(255, 255, 255, 0.01);
      color: var(--text-primary);
      font-size: 13px;
      line-height: 1.2;
      padding: 9px 11px;
    }
    input::placeholder { color: var(--text-muted); }
    .visually-hidden {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }
    button {
      cursor: pointer;
      transition: border-color 140ms ease, background-color 140ms ease, transform 140ms ease;
    }
    button:hover {
      border-color: var(--accent);
      background: rgba(255, 255, 255, 0.03);
      transform: translateY(-1px);
    }
    #dashboard-toolbar {
      margin: 12px 12px 0;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--panel-border);
      background: color-mix(in srgb, var(--panel-bg) 90%, transparent);
      backdrop-filter: blur(8px);
      box-shadow: 0 8px 24px var(--shadow-soft);
      display: grid;
      gap: 8px;
    }
    #global-nav {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .nav-group {
      display: inline-flex;
      gap: 7px;
      align-items: center;
      flex-wrap: wrap;
    }
    .nav-btn {
      width: auto;
      padding: 7px 11px;
      border-radius: 8px;
      font-size: 12px;
      color: var(--text-muted);
      background: rgba(255, 255, 255, 0.01);
    }
    .nav-btn.active {
      color: var(--text-primary);
      border-color: color-mix(in srgb, var(--accent) 70%, var(--panel-border));
      background: var(--accent-soft);
    }
    #result-select {
      min-width: 240px;
      max-width: min(46vw, 360px);
    }
    #toolbar-controls {
      display: grid;
      gap: 8px;
    }
    #dashboard-toolbar.collapsed #toolbar-controls {
      display: none;
    }
    #dashboard-status {
      display: none;
      border-radius: 10px;
      padding: 9px 11px;
      font-size: 12px;
      line-height: 1.45;
      border: 1px solid color-mix(in srgb, var(--panel-border) 82%, transparent);
      background: rgba(255, 255, 255, 0.02);
      color: var(--text-muted);
    }
    #dashboard-status.visible {
      display: block;
    }
    #dashboard-status.warning {
      border-color: rgba(221, 166, 94, 0.52);
      background: rgba(108, 74, 20, 0.24);
      color: color-mix(in srgb, var(--text-primary) 88%, #ffe1ad);
    }
    #dashboard-status.info {
      border-color: rgba(101, 162, 221, 0.46);
      background: rgba(27, 61, 104, 0.18);
      color: color-mix(in srgb, var(--text-primary) 90%, #d8eaff);
    }
    .toolbar-row {
      display: grid;
      gap: 8px;
      align-items: center;
    }
    .toolbar-row.primary {
      grid-template-columns: minmax(220px, 1fr) 180px 160px;
    }
    .toolbar-row.secondary {
      grid-template-columns: 140px 140px 1fr;
    }
    #provenance-filters {
      display: inline-flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .chip {
      width: auto;
      border-radius: 999px;
      padding: 6px 11px;
      font-size: 12px;
      letter-spacing: 0.01em;
      color: var(--text-muted);
    }
    .chip.active {
      color: var(--text-primary);
      border-color: color-mix(in srgb, var(--accent) 70%, var(--panel-border));
      background: var(--accent-soft);
    }
    #dashboard-root {
      display: grid;
      gap: 12px;
      padding: 12px;
      flex: 1 1 auto;
      min-height: 0;
      height: 100%;
      overflow: hidden;
      grid-template-columns: minmax(260px, 26vw) minmax(520px, 1fr) minmax(320px, 29vw);
    }
    .pane {
      background: color-mix(in srgb, var(--panel-bg) 94%, transparent);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      overflow: hidden;
      height: 100%;
      min-height: 0;
      display: flex;
      flex-direction: column;
      box-shadow: 0 6px 20px var(--shadow-soft);
    }
    .pane-header {
      padding: 11px 12px;
      border-bottom: 1px solid var(--panel-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .pane-title {
      margin: 0;
      font-size: 15px;
      font-weight: 680;
      letter-spacing: 0.01em;
    }
    .muted { color: var(--text-muted); }
    #paper-list {
      margin: 0;
      padding: 0;
      list-style: none;
      overflow-y: auto;
      overflow-x: hidden;
      flex: 1;
      min-height: 0;
    }
    .paper-row {
      border-bottom: 1px solid color-mix(in srgb, var(--panel-border) 75%, transparent);
      padding: 11px 12px 10px;
      cursor: pointer;
      display: grid;
      gap: 6px;
      transition: background-color 120ms ease, border-left-color 120ms ease;
      border-left: 2px solid transparent;
    }
    .paper-row:hover { background: color-mix(in srgb, var(--accent-soft) 65%, transparent); }
    .paper-row.is-hover { border-left-color: color-mix(in srgb, var(--accent) 70%, transparent); }
    .paper-row.is-selected {
      border-left-color: var(--accent);
      background: color-mix(in srgb, var(--accent-soft) 78%, transparent);
    }
    .paper-row-head {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: baseline;
    }
    .paper-title {
      font-size: 20px;
      font-size: clamp(13.5px, 0.88vw, 15px);
      font-weight: 640;
      line-height: 1.28;
      letter-spacing: 0.003em;
      overflow-wrap: anywhere;
    }
    .paper-year {
      font-size: 12px;
      font-weight: 560;
      color: var(--text-muted);
      white-space: nowrap;
    }
    .paper-subline {
      color: color-mix(in srgb, var(--text-muted) 85%, #c9d8ee);
      font-size: 12px;
      line-height: 1.34;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .paper-meta {
      color: var(--text-muted);
      font-size: 11.5px;
      letter-spacing: 0.01em;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .meta-dot {
      width: 4px;
      height: 4px;
      border-radius: 50%;
      background: color-mix(in srgb, var(--text-muted) 70%, transparent);
      display: inline-block;
    }
    .meta-origin { color: color-mix(in srgb, var(--seed-ring) 82%, #f2d8ea); }
    #graph-pane .pane-header { gap: 10px; }
    #graph-canvas-wrap {
      position: relative;
      flex: 1;
      min-height: 0;
      overflow: hidden;
      background: var(--graph-bg);
    }
    #__PLOTLY_DIV_ID__ {
      width: 100%;
      height: 100%;
      min-height: 0;
    }
    .js-plotly-plot .scatterlayer path.point {
      transition: filter 0.2s ease, opacity 0.2s ease;
    }
    .js-plotly-plot .scatterlayer path.point.is-glowing {
      filter: drop-shadow(0 0 10px rgba(220, 80, 150, 0.85)) brightness(1.14);
    }
    .js-plotly-plot .scatterlayer path.point.is-neighbor {
      opacity: 0.74;
    }
    .js-plotly-plot .scatterlayer path.point.is-dimmed {
      opacity: 0.18;
    }
    .js-plotly-plot .scatterlayer path.point.is-filter-hidden {
      opacity: 0.12;
    }
    #graph-footer {
      position: absolute;
      right: 12px;
      bottom: 10px;
      display: grid;
      gap: 8px;
      align-items: end;
      justify-items: end;
      pointer-events: none;
    }
    #graph-legend {
      background: rgba(8, 12, 18, 0.72);
      border: 1px solid color-mix(in srgb, var(--panel-border) 70%, transparent);
      border-radius: 10px;
      padding: 8px 10px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 11px;
      color: color-mix(in srgb, var(--text-muted) 90%, #d6e2f1);
      backdrop-filter: blur(6px);
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }
    .legend-marker {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      border: 1px solid rgba(255, 255, 255, 0.42);
    }
    .legend-marker.seed {
      border: 2px solid var(--seed-ring);
      background: rgba(214, 108, 191, 0.28);
    }
    .legend-marker.citation { background: #7f8fa3; }
    .legend-marker.semantic { background: #6d9f9b; }
    .legend-marker.both { background: #a196b1; }
    #year-timeline {
      display: inline-grid;
      grid-template-columns: auto minmax(190px, 240px) auto;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      color: color-mix(in srgb, var(--text-muted) 92%, #d8e3f2);
      background: rgba(8, 12, 18, 0.72);
      border: 1px solid color-mix(in srgb, var(--panel-border) 70%, transparent);
      border-radius: 10px;
      padding: 7px 9px;
      backdrop-filter: blur(6px);
    }
    #timeline-bar {
      height: 10px;
      border-radius: 999px;
      border: 1px solid color-mix(in srgb, var(--panel-border) 80%, transparent);
      background: linear-gradient(90deg, #5a4f71 0%, #5e6381 20%, #4b7783 40%, #5b8b8c 60%, #7c9f94 80%, #c8be9f 100%);
    }
    #detail-content {
      padding: 14px 13px 12px;
      flex: 1;
      min-height: 0;
      display: flex;
      flex-direction: column;
      gap: 16px;
      overflow-y: auto;
      overflow-x: hidden;
    }
    #detail-title {
      margin: 0;
      font-size: 32px;
      font-size: clamp(22px, 1.3vw, 28px);
      line-height: 1.24;
      letter-spacing: 0.006em;
      font-weight: 700;
    }
    #detail-subtitle {
      color: color-mix(in srgb, var(--text-muted) 86%, #c9d8ea);
      font-size: 13px;
      line-height: 1.42;
    }
    #detail-metrics {
      display: flex;
      gap: 7px;
      flex-wrap: wrap;
      min-height: 18px;
    }
    .metric-pill {
      border: 1px solid color-mix(in srgb, var(--panel-border) 78%, transparent);
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 11.5px;
      color: var(--text-muted);
      background: rgba(255, 255, 255, 0.01);
    }
    #detail-categories {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      min-height: 17px;
    }
    .category-chip {
      font-size: 11px;
      letter-spacing: 0.01em;
      color: color-mix(in srgb, var(--text-muted) 88%, #c7d7ec);
      border: 1px solid color-mix(in srgb, var(--panel-border) 75%, transparent);
      border-radius: 999px;
      padding: 2px 8px;
    }
    #detail-links {
      display: flex;
      gap: 7px;
      flex-wrap: wrap;
      min-height: 36px;
      align-items: center;
    }
    #detail-links a {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      color: var(--text-primary);
      text-decoration: none;
      border: 1px solid color-mix(in srgb, var(--panel-border) 80%, transparent);
      border-radius: 8px;
      width: 31px;
      height: 31px;
      padding: 0;
      background: rgba(255, 255, 255, 0.012);
      transition: border-color 130ms ease, transform 130ms ease, background-color 130ms ease;
    }
    #detail-links a:hover {
      border-color: color-mix(in srgb, var(--accent) 70%, var(--panel-border));
      background: color-mix(in srgb, var(--accent-soft) 80%, transparent);
      transform: translateY(-1px);
    }
    .icon-link svg {
      width: 16px;
      height: 16px;
      fill: none;
      stroke: color-mix(in srgb, var(--accent) 78%, #dce9fb);
      stroke-width: 1.9;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
    #detail-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      min-height: 36px;
    }
    #detail-actions button {
      width: auto;
      padding: 7px 11px;
      font-size: 12px;
      border-radius: 8px;
    }
    #detail-abstract-card {
      display: flex;
      flex-direction: column;
      gap: 7px;
      border: 1px solid color-mix(in srgb, var(--panel-border) 78%, transparent);
      border-radius: 10px;
      padding: 10px;
      min-height: 0;
      flex: 1;
      background: rgba(255, 255, 255, 0.012);
    }
    #detail-abstract-label {
      margin: 0;
      font-size: 12px;
      color: var(--text-muted);
      letter-spacing: 0.015em;
      text-transform: uppercase;
    }
    #detail-abstract {
      white-space: pre-wrap;
      line-height: 1.56;
      font-size: 14px;
      color: var(--text-primary);
      overflow: auto;
      min-height: 0;
      flex: 1;
      padding-right: 2px;
    }
    #detail-why {
      border: 1px solid color-mix(in srgb, var(--panel-border) 78%, transparent);
      border-radius: 10px;
      padding: 10px;
      display: grid;
      gap: 6px;
      background: rgba(255, 255, 255, 0.01);
      min-height: 84px;
    }
    #detail-why-label {
      margin: 0;
      font-size: 12px;
      color: var(--text-muted);
      letter-spacing: 0.015em;
      text-transform: uppercase;
    }
    #detail-why-lines {
      display: grid;
      gap: 4px;
      font-size: 12px;
      line-height: 1.45;
      color: color-mix(in srgb, var(--text-primary) 90%, #dbe8f8);
    }
    @media (max-width: 1280px) {
      #dashboard-root {
        grid-template-columns: minmax(240px, 30vw) minmax(420px, 1fr) minmax(300px, 34vw);
      }
      .toolbar-row.primary { grid-template-columns: 1fr 168px 152px; }
      .toolbar-row.secondary { grid-template-columns: 128px 128px 1fr; }
    }
    @media (max-width: 1100px) {
      html, body {
        overflow: auto;
      }
      #dashboard-toolbar {
        position: sticky;
        top: 0;
        z-index: 3;
      }
      #global-nav {
        justify-content: flex-start;
      }
      .toolbar-row.primary { grid-template-columns: 1fr 1fr; }
      .toolbar-row.primary #search-input { grid-column: span 2; }
      .toolbar-row.secondary { grid-template-columns: 1fr 1fr; }
      #provenance-filters { justify-content: flex-start; grid-column: span 2; }
      #dashboard-root {
        grid-template-columns: 1fr;
        grid-template-areas:
          "graph"
          "detail"
          "list";
        min-height: auto;
        height: auto;
        overflow: visible;
      }
      #graph-pane { grid-area: graph; min-height: 520px; }
      #detail-pane { grid-area: detail; min-height: 430px; }
      #paper-list-pane { grid-area: list; min-height: 360px; }
      #__PLOTLY_DIV_ID__ { min-height: 500px; }
    }
  </style>
</head>
<body>
  <header id="dashboard-toolbar">
    <div id="global-nav">
      <div id="scope-nav" class="nav-group">
        <button class="nav-btn" data-scope="prior" type="button">Prior works</button>
        <button class="nav-btn" data-scope="derivative" type="button">Derivative works</button>
      </div>
      <div class="nav-group">
        <button id="list-view-btn" class="nav-btn active" type="button">List view</button>
        <button id="filters-toggle" class="nav-btn active" type="button">Filters</button>
        <button id="more-btn" class="nav-btn" type="button">More</button>
      </div>
      <div class="nav-group">
        <button id="export-json-btn" class="nav-btn" type="button">Export JSON</button>
        <button id="export-csv-btn" class="nav-btn" type="button">Export CSV</button>
        <button id="export-bib-btn" class="nav-btn" type="button">All BibTeX</button>
        <button id="load-json-btn" class="nav-btn" type="button">Load Results</button>
        <input id="load-json-input" type="file" accept=".json,.html" style="display:none" />
      </div>
      <div class="nav-group">
        <label class="visually-hidden" for="result-select">Saved result</label>
        <select id="result-select" title="Switch saved result">
          <option value="">Current result</option>
        </select>
      </div>
    </div>
    <div id="dashboard-status" role="status" aria-live="polite"></div>
    <div id="toolbar-controls">
      <div class="toolbar-row primary">
        <input id="search-input" type="search" placeholder="Search title, authors, abstract..." />
        <select id="sort-select" title="Sort papers">
          <option value="relevance">Sort: Relevance</option>
          <option value="year">Sort: Year</option>
          <option value="citation_count">Sort: Citations</option>
          <option value="title">Sort: Title</option>
        </select>
        <button id="clear-selection" type="button">Clear Selection</button>
      </div>
      <div class="toolbar-row secondary">
        <input id="year-min" type="number" placeholder="Year min" />
        <input id="year-max" type="number" placeholder="Year max" />
        <div id="provenance-filters">
          <button class="chip active" data-filter="citation" type="button">citation</button>
          <button class="chip active" data-filter="semantic" type="button">semantic</button>
          <button class="chip active" data-filter="both" type="button">both</button>
        </div>
      </div>
    </div>
  </header>

  <div id="dashboard-root">
    <aside id="paper-list-pane" class="pane">
      <div class="pane-header">
        <h2 class="pane-title">Papers</h2>
        <span id="paper-count" class="muted">0</span>
      </div>
      <ul id="paper-list"></ul>
    </aside>

    <main id="graph-pane" class="pane">
      <div class="pane-header">
        <h2 class="pane-title">Graph</h2>
        <span id="graph-hint" class="muted">Hover to preview, click to lock</span>
      </div>
      <div id="graph-canvas-wrap">
        <div id="__PLOTLY_DIV_ID__"></div>
        <div id="graph-footer">
          <div id="graph-legend">
            <span class="legend-item"><span class="legend-marker seed"></span>seed</span>
            <span class="legend-item"><span class="legend-marker citation"></span>citation</span>
            <span class="legend-item"><span class="legend-marker semantic"></span>semantic</span>
            <span class="legend-item"><span class="legend-marker both"></span>both</span>
          </div>
          <div id="year-timeline">
            <span id="timeline-year-min">-</span>
            <div id="timeline-bar"></div>
            <span id="timeline-year-max">-</span>
          </div>
        </div>
      </div>
    </main>

    <aside id="detail-pane" class="pane">
      <div class="pane-header">
        <h2 class="pane-title">Details</h2>
        <span id="detail-mode" class="muted">No selection</span>
      </div>
      <div id="detail-content">
        <h3 id="detail-title">Select a paper</h3>
        <div id="detail-subtitle"></div>
        <section id="detail-abstract-card">
          <h4 id="detail-abstract-label">Abstract</h4>
          <div id="detail-abstract" class="muted">Hover or click a paper to inspect abstract and metadata.</div>
        </section>
        <div id="detail-metrics"></div>
        <div id="detail-categories"></div>
        <div id="detail-links"></div>
        <section id="detail-why">
          <h4 id="detail-why-label">Why This Paper</h4>
          <div id="detail-why-lines" class="muted">Select a paper to inspect neighborhood evidence.</div>
        </section>
        <div id="detail-actions"></div>
      </div>
    </aside>
  </div>

  <script>__PLOTLY_JS__</script>
  <script id="citemesh-dashboard-data" type="application/json">__PAYLOAD_JSON__</script>
  <script id="citemesh-dashboard-figure" type="application/json">__FIGURE_JSON__</script>
  <script id="citemesh-dashboard-collection" type="application/json">__COLLECTION_JSON__</script>
  <script>
    let payload = JSON.parse(document.getElementById("citemesh-dashboard-data").textContent);
    const baseFigureTemplate = JSON.parse(document.getElementById("citemesh-dashboard-figure").textContent);
    let figureSpec = JSON.parse(document.getElementById("citemesh-dashboard-figure").textContent);
    // Embed the collection bundle directly in the shell so saved-result browsing
    // still works when the dashboard is opened from the local filesystem.
    const collectionBundle = JSON.parse(document.getElementById("citemesh-dashboard-collection").textContent);
    const graphDiv = document.getElementById("__PLOTLY_DIV_ID__");
    const plotConfig = {
      displaylogo: false,
      responsive: true,
    };

    let nodes = [];
    let nodeOrder = [];
    let nodeById = new Map();
    let nodeIndexById = new Map();
    let yearRange = {};
    let seedNode = null;
    let seedYear = null;
    let traceSpecs = [];
    let nodeTraceIndex = 0;
    let haloTraceIndex = -1;
    let neighborhoodTraceIndex = -1;
    let defaultNodeSizes = [];
    let defaultNodeX = [];
    let defaultNodeY = [];
    let adjacency = new Map();
    let collectionResultId = String(collectionBundle.current_result_id || "") || null;
    let runtimeStatusMessage = "";
    let runtimeStatusTone = "warning";

    function normalizeArray(rawValue, length, fallbackValue) {
      if (Array.isArray(rawValue)) {
        if (rawValue.length >= length) {
          return rawValue.slice(0, length).map((value) => Number(value));
        }
        const expanded = rawValue.map((value) => Number(value));
        while (expanded.length < length) {
          expanded.push(fallbackValue);
        }
        return expanded;
      }
      const scalar = Number(rawValue);
      const safe = Number.isFinite(scalar) ? scalar : fallbackValue;
      return Array.from({ length }, () => safe);
    }

    function deepClone(value) {
      return JSON.parse(JSON.stringify(value));
    }

    function escapeRegExp(value) {
      return String(value || "").replace(/[.*+?^${}()|[\\]\\]/g, "\\$&");
    }

    function safeFiniteNumber(value, fallbackValue) {
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : fallbackValue;
    }

    function stableHash(value) {
      const text = String(value || "");
      let digest = 0;
      for (const char of text) {
        digest = ((digest * 33) + char.charCodeAt(0)) >>> 0;
      }
      return digest >>> 0;
    }

    function stableCurveDirection(leftId, rightId) {
      return stableHash(`${String(leftId || "")}|${String(rightId || "")}`) % 2 === 0 ? 1 : -1;
    }

    function currentSeedRingColor() {
      const styles = getComputedStyle(document.documentElement);
      const color = String(styles.getPropertyValue("--seed-ring") || "").trim();
      return color || "#d66cbf";
    }

    function computeImportedYearRange(importedNodes, fallbackRange) {
      const validYears = importedNodes
        .map((node) => Number(node && node.year))
        .filter((year) => Number.isFinite(year) && year > 0);
      if (validYears.length) {
        return {
          min: Math.min(...validYears),
          max: Math.max(...validYears),
        };
      }
      if (fallbackRange && typeof fallbackRange === "object") {
        const minYear = safeFiniteNumber(fallbackRange.min, 2000);
        const maxYear = safeFiniteNumber(fallbackRange.max, minYear);
        return { min: minYear, max: maxYear };
      }
      return { min: 2000, max: 2001 };
    }

    function compareLegacyNodes(leftNode, rightNode) {
      const left = leftNode || {};
      const right = rightNode || {};
      if (!!left.is_seed !== !!right.is_seed) {
        return left.is_seed ? -1 : 1;
      }
      const citationDelta =
        safeFiniteNumber(right.citation_count, 0) - safeFiniteNumber(left.citation_count, 0);
      if (citationDelta !== 0) {
        return citationDelta;
      }
      const leftYear = safeFiniteNumber(left.year, -1);
      const rightYear = safeFiniteNumber(right.year, -1);
      if (leftYear !== rightYear) {
        return rightYear - leftYear;
      }
      return String(left.id || "").localeCompare(String(right.id || ""));
    }

    function computeImportedNodeSizes(importedNodes) {
      const orderedNodes = Array.isArray(importedNodes) ? importedNodes.slice() : [];
      const rankedNodes = orderedNodes.slice().sort(compareLegacyNodes);
      const rankOf = new Map(
        rankedNodes.map((node, index) => [String((node && node.id) || ""), index])
      );
      return orderedNodes.map((node) => {
        const nodeId = String((node && node.id) || "");
        const rank = rankOf.has(nodeId) ? rankOf.get(nodeId) : orderedNodes.length;
        const citationCount = Math.max(0, safeFiniteNumber(node && node.citation_count, 0));
        let size = 100;
        if (node && node.is_seed) {
          size = rank < 3 ? 2500 : 1000;
        } else if (rank === 0) {
          size = 2200;
        } else if (rank < 3) {
          size = 1200 + (3 - rank) * 200;
        } else if (rank < 8) {
          size = 500 + (8 - rank) * 80;
        } else if (rank < 15) {
          size = 250 + (15 - rank) * 30;
        }
        size += Math.log10(citationCount + 1) * 100;
        return Math.max(6.0, size / 50.0);
      });
    }

    function normalizePositionPairs(positionPairs, paddingRatio) {
      const pairs = Array.isArray(positionPairs) ? positionPairs : [];
      if (!pairs.length) {
        return [];
      }
      const safePairs = pairs.map((pair) => {
        const x = Array.isArray(pair) ? safeFiniteNumber(pair[0], 0) : 0;
        const y = Array.isArray(pair) ? safeFiniteNumber(pair[1], 0) : 0;
        return [x, y];
      });
      const xs = safePairs.map((pair) => pair[0]);
      const ys = safePairs.map((pair) => pair[1]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const centerX = (minX + maxX) * 0.5;
      const centerY = (minY + maxY) * 0.5;
      const spanX = maxX - minX;
      const spanY = maxY - minY;
      const maxSpan = Math.max(spanX, spanY);
      const targetHalfExtent = Math.max(1e-6, 1.0 - safeFiniteNumber(paddingRatio, 0.1));
      if (maxSpan <= 1e-9) {
        return safePairs.map(() => [0, 0]);
      }
      const scale = targetHalfExtent / (maxSpan * 0.5);
      return safePairs.map((pair) => [
        (pair[0] - centerX) * scale,
        (pair[1] - centerY) * scale,
      ]);
    }

    function synthesizeLegacyDashboardMeta(importedNodes, importedEdges, baseMeta) {
      const nodes = Array.isArray(importedNodes) ? importedNodes.slice() : [];
      const edges = Array.isArray(importedEdges) ? importedEdges : [];
      const nodeIds = nodes.map((node, index) => {
        const nodeId = String((node && node.id) || "").trim();
        return nodeId || `node-${index}`;
      });
      const seedNode = nodes.find((node) => !!(node && node.is_seed));
      const resolvedSeedId = String(
        (baseMeta && baseMeta.seed_id) || (seedNode && seedNode.id) || nodeIds[0] || ""
      );
      const adjacencyMap = new Map(nodeIds.map((nodeId) => [nodeId, new Set()]));
      edges.forEach((edge) => {
        const leftId = String((edge && edge.source) || "");
        const rightId = String((edge && edge.target) || "");
        if (!leftId || !rightId || leftId === rightId) {
          return;
        }
        if (!adjacencyMap.has(leftId)) {
          adjacencyMap.set(leftId, new Set());
        }
        if (!adjacencyMap.has(rightId)) {
          adjacencyMap.set(rightId, new Set());
        }
        adjacencyMap.get(leftId).add(rightId);
        adjacencyMap.get(rightId).add(leftId);
      });

      const distanceByNode = new Map();
      if (resolvedSeedId && adjacencyMap.has(resolvedSeedId)) {
        const queue = [resolvedSeedId];
        distanceByNode.set(resolvedSeedId, 0);
        while (queue.length) {
          const currentId = queue.shift();
          const currentDistance = distanceByNode.get(currentId) || 0;
          (adjacencyMap.get(currentId) || new Set()).forEach((neighborId) => {
            if (distanceByNode.has(neighborId)) {
              return;
            }
            distanceByNode.set(neighborId, currentDistance + 1);
            queue.push(neighborId);
          });
        }
      }

      let maxDistance = 0;
      distanceByNode.forEach((distance) => {
        maxDistance = Math.max(maxDistance, safeFiniteNumber(distance, 0));
      });

      const legacyNodes = nodes.map((node, index) =>
        Object.assign({ id: nodeIds[index] }, node || {})
      );
      const byLevel = new Map();
      let disconnectedOffset = maxDistance + 1;
      legacyNodes.forEach((node) => {
        const nodeId = String(node.id || "");
        let level = distanceByNode.has(nodeId) ? distanceByNode.get(nodeId) : null;
        if (level === null || level === undefined) {
          level = disconnectedOffset;
          disconnectedOffset += 1;
        }
        if (!byLevel.has(level)) {
          byLevel.set(level, []);
        }
        byLevel.get(level).push(node);
      });

      const positionsById = new Map();
      Array.from(byLevel.keys()).sort((left, right) => left - right).forEach((level) => {
        const levelNodes = byLevel.get(level).slice().sort(compareLegacyNodes);
        if (level === 0 && levelNodes.length) {
          positionsById.set(String(levelNodes[0].id || resolvedSeedId), [0, 0]);
          levelNodes.slice(1).forEach((node, index) => {
            const angle = (2 * Math.PI * index) / Math.max(levelNodes.length - 1, 1);
            positionsById.set(String(node.id || ""), [0.2 * Math.cos(angle), 0.2 * Math.sin(angle)]);
          });
          return;
        }
        const radius = 0.62 + Math.max(0, level - 1) * 0.48;
        const offset = ((stableHash(`${resolvedSeedId}|${level}`) % 360) * Math.PI) / 180;
        levelNodes.forEach((node, index) => {
          const angle = offset + (2 * Math.PI * index) / Math.max(levelNodes.length, 1);
          positionsById.set(String(node.id || ""), [
            radius * Math.cos(angle),
            radius * Math.sin(angle),
          ]);
        });
      });

      const normalizedPositions = normalizePositionPairs(
        nodeIds.map((nodeId) => positionsById.get(nodeId) || [0, 0]),
        0.1
      );
      return {
        seed_id: resolvedSeedId,
        theme: String((baseMeta && baseMeta.theme) || (payload.meta && payload.meta.theme) || "light"),
        summary: (baseMeta && baseMeta.summary) || {
          nodes: nodes.length,
          edges: edges.length,
        },
        year_range: computeImportedYearRange(nodes, baseMeta && baseMeta.year_range),
        plotly_node_order: nodeIds,
        plotly_positions: normalizedPositions,
        plotly_node_sizes: computeImportedNodeSizes(legacyNodes),
      };
    }

    function recoverDashboardMetaFromFigure(importedNodes, baseMeta, figureSpec) {
      if (!figureSpec || !Array.isArray(figureSpec.data)) {
        return null;
      }
      const nodeTrace = figureSpec.data.find(
        (trace) =>
          String((trace && trace.name) || "") === "nodes"
          || String((trace && trace.mode) || "").includes("markers+text")
      );
      if (!nodeTrace) {
        return null;
      }
      const x = Array.isArray(nodeTrace.x) ? nodeTrace.x.map((value) => safeFiniteNumber(value, 0)) : [];
      const y = Array.isArray(nodeTrace.y) ? nodeTrace.y.map((value) => safeFiniteNumber(value, 0)) : [];
      if (!x.length || x.length !== y.length) {
        return null;
      }
      let order = Array.isArray(baseMeta && baseMeta.plotly_node_order)
        ? baseMeta.plotly_node_order.map((nodeId) => String(nodeId || ""))
        : [];
      if (!order.length || order.length !== x.length) {
        const importedOrder = Array.isArray(importedNodes) ? importedNodes : [];
        order = importedOrder.map((node, index) => {
          const nodeId = String((node && node.id) || "").trim();
          return nodeId || `node-${index}`;
        });
      }
      if (!order.length || order.length !== x.length) {
        return null;
      }
      return {
        plotly_node_order: order,
        plotly_positions: x.map((xValue, index) => [xValue, y[index]]),
        plotly_node_sizes: normalizeArray(
          nodeTrace && nodeTrace.marker ? nodeTrace.marker.size : [],
          order.length,
          8
        ),
      };
    }

    function extractEmbeddedScriptJson(text, scriptId) {
      const pattern = new RegExp(
        `<script id="${escapeRegExp(scriptId)}" type="application/json">([\\\\s\\\\S]*?)</script>`
      );
      const match = String(text || "").match(pattern);
      if (!match) {
        throw new Error(`Imported dashboard file is missing ${scriptId}.`);
      }
      return JSON.parse(match[1]);
    }

    function parseImportedPayloadFromText(fileText, filename) {
      const text = String(fileText || "");
      const lowerName = String(filename || "").toLowerCase();
      const looksLikeDashboardHtml =
        lowerName.endsWith(".html") || text.includes('id="citemesh-dashboard-data"');
      if (!looksLikeDashboardHtml) {
        return JSON.parse(text);
      }
      const importedPayload = extractEmbeddedScriptJson(text, "citemesh-dashboard-data");
      const importedFigure = extractEmbeddedScriptJson(text, "citemesh-dashboard-figure");
      return Object.assign({}, importedPayload, {
        __citemesh_dashboard_figure: importedFigure,
      });
    }

    function hasCompleteDashboardGeometry(meta) {
      if (!meta || typeof meta !== "object") {
        return false;
      }
      const order = Array.isArray(meta.plotly_node_order)
        ? meta.plotly_node_order.map((nodeId) => String(nodeId || ""))
        : [];
      const positions = Array.isArray(meta.plotly_positions) ? meta.plotly_positions : [];
      if (!order.length || positions.length !== order.length) {
        return false;
      }
      return true;
    }

    function buildFigureSpecFromPayload(nextPayload) {
      const meta = (nextPayload && nextPayload.meta) || {};
      const order = Array.isArray(meta.plotly_node_order)
        ? meta.plotly_node_order.map((nodeId) => String(nodeId || ""))
        : [];
      const positions = Array.isArray(meta.plotly_positions) ? meta.plotly_positions : [];
      const alignedNodeSizes = normalizeArray(meta.plotly_node_sizes, order.length, 8);
      if (!order.length || positions.length !== order.length) {
        throw new Error(
          "JSON is missing dashboard layout positions. Re-export results with a newer CiteMesh build."
        );
      }

      const template = deepClone(baseFigureTemplate);
      const nextNodes = Array.isArray(nextPayload.nodes) ? nextPayload.nodes : [];
      const nextNodeById = new Map(
        nextNodes.map((node) => [String(node.id || ""), node])
      );
      const nextEdges = Array.isArray(nextPayload.edges) ? nextPayload.edges : [];
      const nextYearRange = meta.year_range || {};
      const nextTraceSpecs = Array.isArray(template.data) ? template.data : [];
      const nextNodeTraceIndex = (() => {
        const namedIdx = nextTraceSpecs.findIndex(
          (trace) => String(trace.name || "") === "nodes"
        );
        if (namedIdx >= 0) {
          return namedIdx;
        }
        const fallbackIdx = nextTraceSpecs.findIndex((trace) =>
          String(trace.mode || "").includes("markers+text")
        );
        return fallbackIdx >= 0 ? fallbackIdx : 0;
      })();
      const nextHaloTraceIndex = nextTraceSpecs.findIndex(
        (trace) => String(trace.name || "") === "selection-halo"
      );
      const nextNeighborhoodTraceIndex = nextTraceSpecs.findIndex(
        (trace) => String(trace.name || "") === "neighborhood-edges"
      );
      const templateNodeTrace = nextTraceSpecs[nextNodeTraceIndex] || {};
      const templateMarker = templateNodeTrace.marker || {};
      const templateLayout = template.layout || {};
      const xPairs = normalizeArray(
        positions.map((position) => Array.isArray(position) ? position[0] : 0),
        order.length,
        0
      );
      const yPairs = normalizeArray(
        positions.map((position) => Array.isArray(position) ? position[1] : 0),
        order.length,
        0
      );
      const yearMin = Number(nextYearRange.min || 0);
      const yearMax = Number(nextYearRange.max || 0);
      const safeYearMin = Number.isFinite(yearMin) ? yearMin : 0;
      const safeYearMax = Number.isFinite(yearMax) ? yearMax : safeYearMin;
      const labelRanking = order
        .map((nodeId) => nextNodeById.get(nodeId))
        .filter(Boolean)
        .sort((leftNode, rightNode) => {
          const left = leftNode || {};
          const right = rightNode || {};
          if (!!left.is_seed !== !!right.is_seed) {
            return left.is_seed ? -1 : 1;
          }
          const citationDelta =
            Number(right.citation_count || 0) - Number(left.citation_count || 0);
          if (citationDelta !== 0) {
            return citationDelta;
          }
          const leftYear = Number.isFinite(Number(left.year)) ? Number(left.year) : -1;
          const rightYear = Number.isFinite(Number(right.year)) ? Number(right.year) : -1;
          if (leftYear !== rightYear) {
            return rightYear - leftYear;
          }
          return String(left.id || "").localeCompare(String(right.id || ""));
        })
        .slice(0, Math.min(14, order.length));
      const labelIds = new Set(labelRanking.map((node) => String(node.id || "")));
      const seedId = String(meta.seed_id || "");
      const seedRingColor = currentSeedRingColor();

      const nodeTexts = [];
      const hoverTexts = [];
      const nodeYears = [];
      const lineWidths = [];
      const lineColors = [];
      for (let idx = 0; idx < order.length; idx += 1) {
        const nodeId = order[idx];
        const node = nextNodeById.get(nodeId) || {};
        const label = labelIds.has(nodeId) ? String(node.title || nodeId) : "";
        nodeTexts.push(label);
        const authors = Array.isArray(node.authors) && node.authors.length
          ? node.authors.slice(0, 3).join(", ")
          : "Unknown";
        const nodeYear = Number.isFinite(Number(node.year)) && Number(node.year) > 0
          ? Number(node.year)
          : safeYearMin;
        nodeYears.push(nodeYear);
        hoverTexts.push([
          `<b>${escapeHtml(node.title || nodeId)}</b>`,
          escapeHtml(authors),
          `Year: ${node.year || "n.d."} | Citations: ${Number(node.citation_count || 0)}`,
        ].join("<br>"));
        if (node.is_seed) {
          lineWidths.push(4.0);
          lineColors.push(seedRingColor);
        } else {
          lineWidths.push(0.0);
          lineColors.push("rgba(0,0,0,0)");
        }
      }

      const xMin = Math.min(...xPairs);
      const xMax = Math.max(...xPairs);
      const yMin = Math.min(...yPairs);
      const yMax = Math.max(...yPairs);
      const xSpan = Math.max(xMax - xMin, 1e-6);
      const ySpan = Math.max(yMax - yMin, 1e-6);
      const xPad = Math.max(0.28, xSpan * 0.08);
      const yPad = Math.max(0.28, ySpan * 0.08);
      const baseShapeColor =
        (((templateLayout.shapes || [])[0] || {}).line || {}).color
        || "rgba(127, 143, 163, 0.24)";
      const edgeShapes = [];
      nextEdges.forEach((edge) => {
        const leftId = String(edge.source || "");
        const rightId = String(edge.target || "");
        const leftIdx = order.indexOf(leftId);
        const rightIdx = order.indexOf(rightId);
        if (leftIdx < 0 || rightIdx < 0) {
          return;
        }
        const x0 = xPairs[leftIdx];
        const y0 = yPairs[leftIdx];
        const x1 = xPairs[rightIdx];
        const y1 = yPairs[rightIdx];
        const midX = (x0 + x1) / 2.0;
        const midY = (y0 + y1) / 2.0;
        const dx = x1 - x0;
        const dy = y1 - y0;
        const direction = stableCurveDirection(leftId, rightId);
        const cx = midX - (dy * 0.15 * direction);
        const cy = midY + (dx * 0.15 * direction);
        edgeShapes.push({
          type: "path",
          path: `M ${x0},${y0} Q ${cx},${cy} ${x1},${y1}`,
          line: {
            color: baseShapeColor,
            width: Math.max(0.5, Number(edge.weight || 0) * 2.0),
          },
          layer: "below",
        });
      });

      const nodeTrace = templateNodeTrace;
      nodeTrace.x = xPairs;
      nodeTrace.y = yPairs;
      nodeTrace.text = nodeTexts;
      nodeTrace.hovertext = hoverTexts;
      nodeTrace.marker = Object.assign({}, templateMarker, {
        size: alignedNodeSizes,
        color: nodeYears,
        cmin: safeYearMin,
        cmax: safeYearMax,
        line: {
          width: lineWidths,
          color: lineColors,
        },
      });
      nextTraceSpecs[nextNodeTraceIndex] = nodeTrace;

      if (nextHaloTraceIndex >= 0) {
        const seedIdx = order.indexOf(seedId);
        const haloTrace = nextTraceSpecs[nextHaloTraceIndex] || {};
        haloTrace.x = seedIdx >= 0 ? [xPairs[seedIdx]] : [];
        haloTrace.y = seedIdx >= 0 ? [yPairs[seedIdx]] : [];
        haloTrace.marker = Object.assign({}, haloTrace.marker || {}, {
          size: seedIdx >= 0 ? [alignedNodeSizes[seedIdx] * 2.05] : [],
          color: seedIdx >= 0 ? ["rgba(214, 108, 191, 0.26)"] : [],
        });
        nextTraceSpecs[nextHaloTraceIndex] = haloTrace;
      }

      if (nextNeighborhoodTraceIndex >= 0) {
        const neighborhoodTrace = nextTraceSpecs[nextNeighborhoodTraceIndex] || {};
        neighborhoodTrace.x = [];
        neighborhoodTrace.y = [];
        nextTraceSpecs[nextNeighborhoodTraceIndex] = neighborhoodTrace;
      }

      const layout = Object.assign({}, templateLayout);
      layout.xaxis = Object.assign({}, layout.xaxis || {}, {
        autorange: false,
        range: [xMin - xPad, xMax + xPad],
      });
      layout.yaxis = Object.assign({}, layout.yaxis || {}, {
        autorange: false,
        range: [yMin - yPad, yMax + yPad],
      });
      layout.shapes = edgeShapes;
      layout.uirevision = "citemesh-dashboard-static-layout-v1";

      template.data = nextTraceSpecs;
      template.layout = layout;
      return template;
    }

    function rebuildDerivedData() {
      nodes = Array.isArray(payload.nodes) ? payload.nodes : [];
      nodeOrder = Array.isArray(payload.meta && payload.meta.plotly_node_order)
        ? payload.meta.plotly_node_order.map((nodeId) => String(nodeId || ""))
        : [];
      nodeById = new Map(nodes.map((node) => [String(node.id || ""), node]));
      nodeIndexById = new Map(nodeOrder.map((nodeId, idx) => [nodeId, idx]));
      yearRange = (payload.meta && payload.meta.year_range) || {};
      seedNode = nodeById.get((payload.meta && payload.meta.seed_id) || "") || null;
      seedYear = seedNode && Number.isFinite(Number(seedNode.year)) && Number(seedNode.year) > 0
        ? Number(seedNode.year)
        : null;

      traceSpecs = figureSpec.data || [];
      nodeTraceIndex = (() => {
        const namedIdx = traceSpecs.findIndex((trace) => String(trace.name || "") === "nodes");
        if (namedIdx >= 0) {
          return namedIdx;
        }
        const fallbackIdx = traceSpecs.findIndex((trace) => String(trace.mode || "").includes("markers+text"));
        return fallbackIdx >= 0 ? fallbackIdx : 0;
      })();
      haloTraceIndex = traceSpecs.findIndex(
        (trace) => String(trace.name || "") === "selection-halo"
      );
      neighborhoodTraceIndex = traceSpecs.findIndex(
        (trace) => String(trace.name || "") === "neighborhood-edges"
      );
      const nodeTraceSource = (traceSpecs[nodeTraceIndex] || {});
      const markerSource = nodeTraceSource.marker || {};
      defaultNodeSizes = normalizeArray(markerSource.size, nodeOrder.length, 8);
      defaultNodeX = normalizeArray(nodeTraceSource.x, nodeOrder.length, 0);
      defaultNodeY = normalizeArray(nodeTraceSource.y, nodeOrder.length, 0);

      adjacency = new Map();
      (payload.edges || []).forEach((edge) => {
        const left = String(edge.source || "");
        const right = String(edge.target || "");
        const weight = Number(edge.weight || 0);
        if (!left || !right) {
          return;
        }
        if (!adjacency.has(left)) {
          adjacency.set(left, []);
        }
        if (!adjacency.has(right)) {
          adjacency.set(right, []);
        }
        adjacency.get(left).push({ id: right, weight });
        adjacency.get(right).push({ id: left, weight });
      });
    }

    rebuildDerivedData();

    const state = {
      selectedId: (payload.meta && payload.meta.seed_id) || null,
      hoverId: null,
      filters: { citation: true, semantic: true, both: true },
      searchText: "",
      sortKey: "relevance",
      scopeMode: "all",
      yearMin: null,
      yearMax: null,
      visibleIds: new Set(nodeOrder),
    };
    const overlayState = {
      neighborhoodKey: "",
      haloKey: "",
    };

    const controls = {
      list: document.getElementById("paper-list"),
      count: document.getElementById("paper-count"),
      search: document.getElementById("search-input"),
      sort: document.getElementById("sort-select"),
      yearMin: document.getElementById("year-min"),
      yearMax: document.getElementById("year-max"),
      clearSelection: document.getElementById("clear-selection"),
      chips: Array.from(document.querySelectorAll("#provenance-filters .chip")),
      scopeButtons: Array.from(document.querySelectorAll("#scope-nav [data-scope]")),
      filtersToggle: document.getElementById("filters-toggle"),
      listViewBtn: document.getElementById("list-view-btn"),
      moreBtn: document.getElementById("more-btn"),
      toolbar: document.getElementById("dashboard-toolbar"),
      detailMode: document.getElementById("detail-mode"),
      detailTitle: document.getElementById("detail-title"),
      detailSubtitle: document.getElementById("detail-subtitle"),
      detailMetrics: document.getElementById("detail-metrics"),
      detailCategories: document.getElementById("detail-categories"),
      detailLinks: document.getElementById("detail-links"),
      detailWhy: document.getElementById("detail-why-lines"),
      detailActions: document.getElementById("detail-actions"),
      detailAbstract: document.getElementById("detail-abstract"),
      graphHint: document.getElementById("graph-hint"),
      timelineYearMin: document.getElementById("timeline-year-min"),
      timelineYearMax: document.getElementById("timeline-year-max"),
      resultSelect: document.getElementById("result-select"),
      statusBanner: document.getElementById("dashboard-status"),
    };

    function clearDashboardStatus() {
      runtimeStatusMessage = "";
      runtimeStatusTone = "warning";
      if (!controls.statusBanner) {
        return;
      }
      controls.statusBanner.textContent = "";
      controls.statusBanner.classList.remove("visible", "warning", "info");
    }

    function setDashboardStatus(message, tone) {
      runtimeStatusMessage = String(message || "").trim();
      runtimeStatusTone = tone === "info" ? "info" : "warning";
      if (!controls.statusBanner) {
        return;
      }
      controls.statusBanner.textContent = runtimeStatusMessage;
      controls.statusBanner.classList.toggle("visible", !!runtimeStatusMessage);
      controls.statusBanner.classList.toggle(
        "warning",
        !!runtimeStatusMessage && runtimeStatusTone === "warning"
      );
      controls.statusBanner.classList.toggle(
        "info",
        !!runtimeStatusMessage && runtimeStatusTone === "info"
      );
    }

    function escapeHtml(value) {
      return String(value || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function hasYear(node) {
      return Number.isFinite(Number(node.year)) && Number(node.year) > 0;
    }

    function nodeFilterClass(node) {
      const base = node.provenance_base || node.provenance || "citation";
      return state.filters[base] === true;
    }

    function nodeMatches(node) {
      if (!nodeFilterClass(node)) {
        return false;
      }
      if (!node.is_seed && seedYear !== null && hasYear(node)) {
        const nodeYear = Number(node.year);
        if (state.scopeMode === "prior" && nodeYear > seedYear) {
          return false;
        }
        if (state.scopeMode === "derivative" && nodeYear < seedYear) {
          return false;
        }
      }
      if (state.yearMin !== null && (!hasYear(node) || Number(node.year) < state.yearMin)) {
        return false;
      }
      if (state.yearMax !== null && (!hasYear(node) || Number(node.year) > state.yearMax)) {
        return false;
      }
      if (!state.searchText) {
        return true;
      }
      const haystack = [
        node.title || "",
        Array.isArray(node.authors) ? node.authors.join(" ") : "",
        node.abstract || "",
      ].join(" ").toLowerCase();
      return haystack.includes(state.searchText);
    }

    function tieBreak(nodeA, nodeB) {
      const citationsA = Number(nodeA.citation_count || 0);
      const citationsB = Number(nodeB.citation_count || 0);
      if (citationsA !== citationsB) {
        return citationsB - citationsA;
      }
      const yearA = hasYear(nodeA) ? Number(nodeA.year) : -1;
      const yearB = hasYear(nodeB) ? Number(nodeB.year) : -1;
      if (yearA !== yearB) {
        return yearB - yearA;
      }
      return String(nodeA.id).localeCompare(String(nodeB.id));
    }

    function compareNodes(nodeA, nodeB) {
      if (!!nodeA.is_seed !== !!nodeB.is_seed) {
        return nodeA.is_seed ? -1 : 1;
      }

      if (state.sortKey === "title") {
        const titleCmp = String(nodeA.title || "").localeCompare(String(nodeB.title || ""));
        return titleCmp || tieBreak(nodeA, nodeB);
      }
      if (state.sortKey === "year") {
        const yearA = hasYear(nodeA) ? Number(nodeA.year) : -1;
        const yearB = hasYear(nodeB) ? Number(nodeB.year) : -1;
        if (yearA !== yearB) {
          return yearB - yearA;
        }
        return tieBreak(nodeA, nodeB);
      }
      if (state.sortKey === "citation_count") {
        const citationsA = Number(nodeA.citation_count || 0);
        const citationsB = Number(nodeB.citation_count || 0);
        if (citationsA !== citationsB) {
          return citationsB - citationsA;
        }
        return tieBreak(nodeA, nodeB);
      }

      const relevanceA = Number(nodeA.seed_relevance || 0);
      const relevanceB = Number(nodeB.seed_relevance || 0);
      if (relevanceA !== relevanceB) {
        return relevanceB - relevanceA;
      }
      return tieBreak(nodeA, nodeB);
    }

    function relationBadgeLabel(node) {
      const relation = String(node.seed_relation || "").trim();
      if (relation === "referenced_by_seed") {
        return "referenced by seed";
      }
      if (relation === "cites_seed") {
        return "cites seed";
      }
      if (relation === "semantic_only") {
        return "semantic-only";
      }
      if (relation === "overlap") {
        return "prior+derivative";
      }
      if (relation === "seed") {
        return "origin";
      }
      return String(node.provenance_base || node.provenance || "citation");
    }

    function filteredNodes() {
      const selected = nodes.filter(nodeMatches);
      selected.sort(compareNodes);
      return selected;
    }

    function detailLinkEntries(links) {
      const entries = [];
      if (links && links.arxiv_pdf) {
        entries.push({ kind: "pdf", title: "Open PDF", href: links.arxiv_pdf });
      }
      if (links && links.arxiv_abs) {
        entries.push({ kind: "arxiv", title: "Open arXiv page", href: links.arxiv_abs });
      }
      if (links && links.doi) {
        entries.push({ kind: "doi", title: "Open DOI", href: links.doi });
      }
      if (links && links.semantic_scholar) {
        entries.push({ kind: "s2", title: "Open Semantic Scholar", href: links.semantic_scholar });
      }
      return entries;
    }

    function linkIconSvg(kind) {
      if (kind === "pdf") {
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z"></path><polyline points="14 2 14 7 19 7"></polyline><line x1="8" y1="12" x2="16" y2="12"></line><line x1="8" y1="16" x2="13" y2="16"></line></svg>';
      }
      if (kind === "arxiv") {
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18L10 6l2 6 2-4 6 10"></path><circle cx="10" cy="6" r="1.2"></circle><circle cx="12" cy="12" r="1.2"></circle><circle cx="14" cy="8" r="1.2"></circle></svg>';
      }
      if (kind === "doi") {
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 7h6"></path><path d="M9 12h6"></path><path d="M9 17h6"></path><circle cx="6.5" cy="7" r="1"></circle><circle cx="6.5" cy="12" r="1"></circle><circle cx="6.5" cy="17" r="1"></circle><path d="M17.5 7a2.5 2.5 0 0 1 0 5"></path><path d="M17.5 12a2.5 2.5 0 0 0 0 5"></path></svg>';
      }
      if (kind === "s2") {
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h12"></path><path d="M6 12h9"></path><path d="M6 19h12"></path><path d="M17 10l2 2-2 2"></path></svg>';
      }
      return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4h16v16H4z"></path><path d="M8 8h8v8H8z"></path></svg>';
    }

    function detailLinksHtml(links) {
      const entries = detailLinkEntries(links);
      if (!entries.length) {
        return "";
      }
      return entries
        .map((entry) => `<a class="icon-link" href="${escapeHtml(entry.href)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(entry.title)}" aria-label="${escapeHtml(entry.title)}">${linkIconSvg(entry.kind)}</a>`)
        .join("");
    }

    function copyText(value) {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        return navigator.clipboard.writeText(value);
      }
      const fallback = document.createElement("textarea");
      fallback.value = value;
      document.body.appendChild(fallback);
      fallback.focus();
      fallback.select();
      try {
        document.execCommand("copy");
      } finally {
        fallback.remove();
      }
      return Promise.resolve();
    }

    function compactNodeLabel(nodeId) {
      const node = nodeById.get(nodeId);
      if (!node) {
        return String(nodeId);
      }
      if (Array.isArray(node.authors) && node.authors.length) {
        const surname = String(node.authors[0]).split(" ").filter(Boolean).slice(-1)[0] || "Unknown";
        const year = hasYear(node) ? String(node.year) : "n.d.";
        return `${surname}, ${year}`;
      }
      return String(node.title || node.id || nodeId);
    }

    function buildPortableJsonPayload() {
      const summary = (payload.meta && payload.meta.summary) || {
        nodes: Array.isArray(payload.nodes) ? payload.nodes.length : 0,
        edges: Array.isArray(payload.edges) ? payload.edges.length : 0,
      };
      const dashboardMeta = Object.assign({}, (payload.meta || {}), {
        summary,
      });
      const edges = (payload.edges || []).map((edge) => {
        const sourceId = String(edge.source || "");
        const targetId = String(edge.target || "");
        const sourceNode = nodeById.get(sourceId);
        const targetNode = nodeById.get(targetId);
        return {
          source: sourceId,
          target: targetId,
          source_title: sourceNode ? String(sourceNode.title || sourceId) : sourceId,
          target_title: targetNode ? String(targetNode.title || targetId) : targetId,
          source_label: compactNodeLabel(sourceId),
          target_label: compactNodeLabel(targetId),
          weight: Number(edge.weight || 0),
        };
      });
      return {
        seed_id: (payload.meta && payload.meta.seed_id) || "",
        meta: {
          strategy: (payload.meta && payload.meta.strategy) || "",
          year_range: (payload.meta && payload.meta.year_range) || {},
        },
        summary,
        dashboard: {
          meta: dashboardMeta,
        },
        nodes: payload.nodes || [],
        edges,
      };
    }

    function currentResultIdForPayload(nextPayload) {
      const meta = (nextPayload && nextPayload.meta) || {};
      const seedId = String(meta.seed_id || "");
      const strategy = String(meta.strategy || "");
      if (!seedId || !strategy) {
        return null;
      }
      return `${strategy}:${seedId}`;
    }

    function collectionEntryLabel(entry) {
      const title = String(entry.title || entry.seed_id || entry.result_id || "Saved result");
      const strategy = String(entry.strategy || "");
      const summary = entry.summary || {};
      const nodeCount = Number(summary.nodes || 0);
      const edgeCount = Number(summary.edges || 0);
      const strategyLabel = strategy ? ` [${strategy}]` : "";
      return `${title}${strategyLabel} • ${nodeCount} papers / ${edgeCount} links`;
    }

    function collectionEntries() {
      return Array.isArray(collectionBundle.results) ? collectionBundle.results : [];
    }

    function populateCollectionSelector() {
      const select = controls.resultSelect;
      if (!select) {
        return;
      }
      const results = collectionEntries();
      const currentId = collectionResultId || currentResultIdForPayload(payload);
      select.innerHTML = "";
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = results.length ? "Saved results" : "Current result only";
      select.appendChild(placeholder);

      results.forEach((entry) => {
        const option = document.createElement("option");
        option.value = String(entry.result_id || "");
        option.textContent = collectionEntryLabel(entry);
        select.appendChild(option);
      });

      if (currentId && results.some((entry) => String(entry.result_id || "") === currentId)) {
        select.value = currentId;
      } else {
        select.value = "";
      }
      select.disabled = results.length <= 1;
    }

    function updateYearPlaceholders() {
      const validYears = nodes
        .map((node) => (hasYear(node) ? Number(node.year) : null))
        .filter((year) => year !== null);
      if (!validYears.length) {
        controls.yearMin.placeholder = "Year min";
        controls.yearMax.placeholder = "Year max";
        return;
      }
      const minYear = Math.min(...validYears);
      const maxYear = Math.max(...validYears);
      controls.yearMin.placeholder = `Year min (${minYear})`;
      controls.yearMax.placeholder = `Year max (${maxYear})`;
    }

    function normalizeImportedDashboardPayload(imported, label) {
      const importedNodes = Array.isArray(imported.nodes) ? imported.nodes : [];
      if (!importedNodes.length) {
        throw new Error("No nodes found in JSON file.");
      }
      const importedDashboardMeta =
        imported && imported.dashboard && imported.dashboard.meta
          ? imported.dashboard.meta
          : {};
      const importedMeta =
        imported && imported.meta && typeof imported.meta === "object"
          ? imported.meta
          : {};
      const baseMeta = {
        seed_id: String(
          imported.seed_id
          || importedDashboardMeta.seed_id
          || importedMeta.seed_id
          || ((importedNodes.find((node) => !!(node && node.is_seed)) || {}).id || "")
        ),
        strategy: String(importedDashboardMeta.strategy || importedMeta.strategy || ""),
        theme: String(
          (payload.meta && payload.meta.theme)
          || importedDashboardMeta.theme
          || importedMeta.theme
          || "light"
        ),
        summary: imported.summary || importedDashboardMeta.summary || importedMeta.summary || {
          nodes: importedNodes.length,
          edges: Array.isArray(imported.edges) ? imported.edges.length : 0,
        },
        year_range: importedDashboardMeta.year_range || importedMeta.year_range || {},
        plotly_node_order:
          importedDashboardMeta.plotly_node_order || importedMeta.plotly_node_order || [],
        plotly_positions:
          importedDashboardMeta.plotly_positions || importedMeta.plotly_positions || [],
        plotly_node_sizes:
          importedDashboardMeta.plotly_node_sizes || importedMeta.plotly_node_sizes || [],
      };
      let nextMeta = Object.assign({}, baseMeta);
      let warningMessage = "";

      if (
        !hasCompleteDashboardGeometry(nextMeta)
        && imported.__citemesh_dashboard_figure
      ) {
        const recoveredMeta = recoverDashboardMetaFromFigure(
          importedNodes,
          nextMeta,
          imported.__citemesh_dashboard_figure
        );
        if (recoveredMeta) {
          nextMeta = Object.assign({}, nextMeta, recoveredMeta);
        }
      }

      if (!hasCompleteDashboardGeometry(nextMeta)) {
        nextMeta = Object.assign(
          {},
          nextMeta,
          synthesizeLegacyDashboardMeta(
            importedNodes,
            Array.isArray(imported.edges) ? imported.edges : [],
            nextMeta
          )
        );
        warningMessage = `Loaded legacy results from ${String(label || "the selected file")} without stored dashboard geometry. CiteMesh reconstructed a deterministic layout, so positions may differ from the original export. Export JSON to save the upgraded payload.`;
      }

      return {
        payload: {
          meta: nextMeta,
          nodes: importedNodes,
          edges: Array.isArray(imported.edges) ? imported.edges : [],
        },
        warningMessage,
        warningTone: "warning",
      };
    }

    function applyImportedPayload(imported, label) {
      const normalizedImport = normalizeImportedDashboardPayload(imported, label);
      const nextPayload = normalizedImport.payload;
      const nextFigureSpec = buildFigureSpecFromPayload(nextPayload);
      payload = nextPayload;
      figureSpec = nextFigureSpec;
      if (normalizedImport.warningMessage) {
        setDashboardStatus(
          normalizedImport.warningMessage,
          normalizedImport.warningTone
        );
        console.warn(normalizedImport.warningMessage);
      } else {
        clearDashboardStatus();
      }
      collectionResultId = currentResultIdForPayload(nextPayload);
      rebuildDerivedData();
      overlayState.neighborhoodKey = "";
      overlayState.haloKey = "";
      state.selectedId = (payload.meta && payload.meta.seed_id) || null;
      state.hoverId = null;
      state.searchText = "";
      state.sortKey = "relevance";
      state.scopeMode = "all";
      state.yearMin = null;
      state.yearMax = null;
      state.visibleIds = new Set(nodeOrder);
      controls.search.value = "";
      controls.sort.value = "relevance";
      controls.yearMin.value = "";
      controls.yearMax.value = "";
      controls.scopeButtons.forEach((entry) => entry.classList.remove("active"));
      controls.chips.forEach((chip) => {
        const filterKey = chip.getAttribute("data-filter");
        if (!filterKey) {
          return;
        }
        state.filters[filterKey] = true;
        chip.classList.add("active");
      });
      populateCollectionSelector();
      updateYearPlaceholders();
      renderTimeline();
      return Plotly.react(graphDiv, figureSpec.data, figureSpec.layout, plotConfig).then(() => {
        setupGraphInteractions();
        if (state.selectedId && nodeById.has(state.selectedId)) {
          renderDetail(state.selectedId, false);
        } else {
          renderDetail(null, false);
        }
        renderList();
        const loadedTitle = nodeById.get(state.selectedId || "");
        controls.graphHint.textContent = "Loaded: " + (loadedTitle ? loadedTitle.title : label);
      });
    }

    function loadCollectionResult(resultId) {
      const normalizedId = String(resultId || "").trim();
      if (!normalizedId) {
        return Promise.resolve();
      }
      const payloads = collectionBundle && collectionBundle.payloads ? collectionBundle.payloads : {};
      const nextPayload = payloads[normalizedId];
      if (!nextPayload) {
        alert("Saved result payload is not embedded in this dashboard shell. Rebuild the collection or use Load Results.");
        return Promise.resolve();
      }
      const entry = collectionEntries().find(
        (candidate) => String(candidate.result_id || "") === normalizedId
      );
      collectionResultId = normalizedId;
      clearDashboardStatus();
      return applyImportedPayload(
        nextPayload,
        entry ? collectionEntryLabel(entry) : normalizedId
      );
    }

    function shortestPathIds(sourceId, targetId) {
      if (!sourceId || !targetId || sourceId === targetId) {
        return sourceId && targetId ? [sourceId] : [];
      }
      const queue = [sourceId];
      const visited = new Set([sourceId]);
      const previous = new Map();

      while (queue.length) {
        const current = queue.shift();
        const neighbors = adjacency.get(current) || [];
        for (const entry of neighbors) {
          const nextId = entry.id;
          if (visited.has(nextId)) {
            continue;
          }
          visited.add(nextId);
          previous.set(nextId, current);
          if (nextId === targetId) {
            const path = [targetId];
            let cursor = targetId;
            while (previous.has(cursor)) {
              cursor = previous.get(cursor);
              path.push(cursor);
              if (cursor === sourceId) {
                break;
              }
            }
            path.reverse();
            return path;
          }
          queue.push(nextId);
        }
      }
      return [];
    }

    function renderWhyLines(nodeId) {
      const node = nodeId ? nodeById.get(nodeId) : null;
      if (!node) {
        controls.detailWhy.textContent = "Select a paper to inspect neighborhood evidence.";
        controls.detailWhy.classList.add("muted");
        return;
      }

      const relationTag = String(node.seed_relation || "");
      const relationLabel = relationTag || (node.provenance || "unknown");
      const neighbors = (adjacency.get(node.id) || [])
        .slice()
        .sort((left, right) => Number(right.weight || 0) - Number(left.weight || 0))
        .slice(0, 3);
      const neighborLine = neighbors.length
        ? `Top links: ${neighbors.map((entry) => `${compactNodeLabel(entry.id)} (w=${Number(entry.weight || 0).toFixed(2)})`).join("; ")}`
        : "Top links: none";
      const seedId = (payload.meta && payload.meta.seed_id) || null;
      const path = seedId ? shortestPathIds(seedId, node.id) : [];
      const pathLine = path.length
        ? `Shortest path to seed: ${path.map((pid) => compactNodeLabel(pid)).join(" -> ")}`
        : "Shortest path to seed: unavailable";
      const relevanceLine = `Seed relevance: ${Number(node.seed_relevance || 0).toFixed(4)} | Relation: ${relationLabel}`;

      controls.detailWhy.classList.remove("muted");
      controls.detailWhy.innerHTML = [
        `<div>${escapeHtml(relevanceLine)}</div>`,
        `<div>${escapeHtml(neighborLine)}</div>`,
        `<div>${escapeHtml(pathLine)}</div>`,
      ].join("");
    }

    function renderDetail(nodeId, previewOnly) {
      const node = nodeId ? nodeById.get(nodeId) : null;
      if (!node) {
        controls.detailMode.textContent = "No selection";
        controls.detailTitle.textContent = "Select a paper";
        controls.detailSubtitle.textContent = "";
        controls.detailMetrics.innerHTML = "";
        controls.detailCategories.innerHTML = "";
        controls.detailLinks.innerHTML = "";
        controls.detailWhy.textContent = "Select a paper to inspect neighborhood evidence.";
        controls.detailWhy.classList.add("muted");
        controls.detailActions.innerHTML = "";
        controls.detailAbstract.textContent = "Hover or click a paper to inspect abstract and metadata.";
        controls.detailAbstract.classList.add("muted");
        controls.graphHint.textContent = "Hover to preview, click to lock";
        return;
      }

      controls.detailMode.textContent = previewOnly ? "Preview" : "Selected";
      if (previewOnly && state.selectedId && state.selectedId !== node.id) {
        controls.graphHint.textContent = "Previewing node (selection locked)";
      } else if (previewOnly) {
        controls.graphHint.textContent = "Previewing node • arcs show strongest direct links";
      } else {
        controls.graphHint.textContent = "Selection locked • red arcs are strongest direct links";
      }
      controls.detailTitle.textContent = node.title || node.id;
      let authorText = "Unknown authors";
      if (Array.isArray(node.authors) && node.authors.length > 0) {
        if (node.authors.length > 2) {
          authorText = `${node.authors[0]} + ${node.authors.length - 1} authors`;
        } else {
          authorText = node.authors.join(", ");
        }
      }
      const yearText = hasYear(node) ? String(node.year) : "n.d.";
      const venueText = node.venue ? `, ${node.venue}` : "";
      controls.detailSubtitle.textContent = `${authorText} | ${yearText}${venueText}`;

      const provenance = node.provenance || "unknown";
      const provenanceLabel = provenance === "seed" ? `seed (${node.provenance_base || "citation"})` : provenance;
      const metrics = [
        `Citations: ${Number(node.citation_count || 0).toLocaleString()}`,
        `Source: ${provenanceLabel}`,
        `Year: ${yearText}`,
      ];
      controls.detailMetrics.innerHTML = metrics
        .map((metric) => `<span class="metric-pill">${escapeHtml(metric)}</span>`)
        .join("");

      const categories = Array.isArray(node.categories) ? node.categories.filter(Boolean).slice(0, 8) : [];
      controls.detailCategories.innerHTML = categories
        .map((category) => `<span class="category-chip">${escapeHtml(category)}</span>`)
        .join("");

      controls.detailLinks.innerHTML = detailLinksHtml(node.links || {});
      renderWhyLines(node.id);
      controls.detailAbstract.textContent = node.abstract || "No abstract available for this record.";
      controls.detailAbstract.classList.toggle("muted", !node.abstract);

      controls.detailActions.innerHTML = "";
      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.textContent = "Copy BibTeX";
      copyBtn.addEventListener("click", () => {
        copyText(node.bibtex || "").then(() => {
          copyBtn.textContent = "Copied";
          window.setTimeout(() => {
            copyBtn.textContent = "Copy BibTeX";
          }, 1000);
        });
      });

      const downloadBtn = document.createElement("button");
      downloadBtn.type = "button";
      downloadBtn.textContent = "Download BibTeX";
      downloadBtn.addEventListener("click", () => {
        const blob = new Blob([node.bibtex || ""], { type: "text/plain;charset=utf-8" });
        const anchor = document.createElement("a");
        anchor.href = URL.createObjectURL(blob);
        anchor.download = `${String(node.id || "paper").replace(/[^a-zA-Z0-9._-]+/g, "_")}.bib`;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(anchor.href);
      });

      controls.detailActions.appendChild(copyBtn);
      controls.detailActions.appendChild(downloadBtn);
    }

    function syncRowHighlights() {
      const rows = Array.from(document.querySelectorAll(".paper-row"));
      rows.forEach((row) => {
        const rowId = row.getAttribute("data-node-id");
        row.classList.toggle("is-hover", !!state.hoverId && rowId === state.hoverId);
        row.classList.toggle("is-selected", !!state.selectedId && rowId === state.selectedId);
      });
    }

    function getNodePointPaths() {
      const traceGroups = graphDiv.querySelectorAll(".scatterlayer .trace");
      if (!traceGroups || !traceGroups.length) {
        return [];
      }
      const traceGroup = traceGroups[nodeTraceIndex] || traceGroups[traceGroups.length - 1];
      return Array.from(traceGroup.querySelectorAll("path.point"));
    }

    function syncGraphHighlights() {
      if (!(window.Plotly && graphDiv && graphDiv.data && graphDiv.data.length > nodeTraceIndex)) {
        return;
      }
      const graphNodePaths = getNodePointPaths();
      const hoverIndex = state.hoverId && nodeIndexById.has(state.hoverId)
        ? nodeIndexById.get(state.hoverId)
        : -1;
      const selectedIndex = state.selectedId && nodeIndexById.has(state.selectedId)
        ? nodeIndexById.get(state.selectedId)
        : -1;
      const hasActiveFocus = hoverIndex !== -1 || selectedIndex !== -1;
      const focusId = state.hoverId || state.selectedId;
      const neighborIds = new Set(
        focusId && adjacency.has(focusId)
          ? (adjacency.get(focusId) || []).map((entry) => entry.id)
          : []
      );

      graphNodePaths.forEach((path, idx) => {
        const rawPointIndex = path.getAttribute("data-point-number");
        const pointIndex = Number(rawPointIndex);
        const stableIdx = Number.isInteger(pointIndex) && pointIndex >= 0 ? pointIndex : idx;
        const nodeId = nodeOrder[stableIdx];
        const isVisible = !!nodeId && state.visibleIds.has(nodeId);
        const isTarget = stableIdx === hoverIndex || stableIdx === selectedIndex;
        const isNeighbor = !!nodeId && neighborIds.has(nodeId) && !isTarget;
        path.classList.toggle("is-filter-hidden", !isVisible);
        path.classList.toggle("is-neighbor", hasActiveFocus && isNeighbor);
        path.classList.toggle("is-dimmed", hasActiveFocus && !isTarget && !isNeighbor);
        path.classList.toggle("is-glowing", isTarget);
      });

      if (neighborhoodTraceIndex >= 0) {
        let neighborhoodX = [];
        let neighborhoodY = [];
        let topNeighbors = [];
        if (focusId && adjacency.has(focusId)) {
          topNeighbors = (adjacency.get(focusId) || [])
            .filter((entry) => state.visibleIds.has(entry.id))
            .sort((left, right) => Number(right.weight || 0) - Number(left.weight || 0))
            .slice(0, 6);
          for (const entry of topNeighbors) {
            if (!nodeIndexById.has(entry.id) || !nodeIndexById.has(focusId)) {
              continue;
            }
            const leftIdx = nodeIndexById.get(focusId);
            const rightIdx = nodeIndexById.get(entry.id);
            neighborhoodX.push(defaultNodeX[leftIdx], defaultNodeX[rightIdx], null);
            neighborhoodY.push(defaultNodeY[leftIdx], defaultNodeY[rightIdx], null);
          }
        }
        const neighborhoodKey = `${focusId || ""}|${topNeighbors.map((entry) => entry.id).join(",")}`;
        if (overlayState.neighborhoodKey !== neighborhoodKey) {
          overlayState.neighborhoodKey = neighborhoodKey;
          Plotly.restyle(
            graphDiv,
            {
              x: [neighborhoodX],
              y: [neighborhoodY],
            },
            [neighborhoodTraceIndex]
          );
        }
      }

      if (haloTraceIndex >= 0) {
        let haloX = [];
        let haloY = [];
        let haloSize = [];
        let haloColor = [];
        if (focusId && nodeIndexById.has(focusId)) {
          const idx = nodeIndexById.get(focusId);
          haloX = [defaultNodeX[idx]];
          haloY = [defaultNodeY[idx]];
          haloSize = [defaultNodeSizes[idx] * (state.selectedId ? 2.2 : 1.88)];
          haloColor = [state.selectedId ? "rgba(238,137,208,0.34)" : "rgba(233,172,245,0.26)"];
        }
        const haloKey = `${focusId || ""}|${state.selectedId ? "selected" : "hover"}`;
        if (overlayState.haloKey !== haloKey) {
          overlayState.haloKey = haloKey;
          Plotly.restyle(
            graphDiv,
            {
              x: [haloX],
              y: [haloY],
              "marker.size": [haloSize],
              "marker.color": [haloColor],
            },
            [haloTraceIndex]
          );
        }
      }
    }

    function syncHighlights() {
      syncRowHighlights();
      syncGraphHighlights();
    }

    function renderList() {
      const listNodes = filteredNodes();
      controls.count.textContent = `${listNodes.length.toLocaleString()} papers`;
      controls.list.innerHTML = "";
      state.visibleIds = new Set(listNodes.map((node) => node.id));
      if (state.selectedId && nodeById.has(state.selectedId)) {
        state.visibleIds.add(state.selectedId);
      }

      if (!listNodes.length) {
        const empty = document.createElement("li");
        empty.className = "paper-row";
        empty.innerHTML = '<div class="paper-title">No papers match current filters.</div>';
        controls.list.appendChild(empty);
        syncHighlights();
        return;
      }

      listNodes.forEach((node) => {
        const row = document.createElement("li");
        row.className = "paper-row";
        row.setAttribute("data-node-id", node.id);

        const yearText = hasYear(node) ? String(node.year) : "n.d.";
        const authors = Array.isArray(node.authors) && node.authors.length
          ? node.authors.slice(0, 4).join(", ")
          : "Unknown authors";
        const provenance = relationBadgeLabel(node);
        const provenanceClass = node.is_seed ? "meta-origin" : "";
        const provenanceLabel = provenance;

        row.innerHTML = `
          <div class="paper-row-head">
            <div class="paper-title">${escapeHtml(node.title || node.id)}</div>
            <div class="paper-year">${escapeHtml(yearText)}</div>
          </div>
          <div class="paper-subline">${escapeHtml(authors)}</div>
          <div class="paper-meta">
            <span>${Number(node.citation_count || 0).toLocaleString()} citations</span>
            <span class="meta-dot"></span>
            <span class="${provenanceClass}">${escapeHtml(provenanceLabel)}</span>
          </div>
        `;

        row.addEventListener("mouseenter", () => {
          state.hoverId = node.id;
          renderDetail(node.id, true);
          syncHighlights();
        });
        row.addEventListener("mouseleave", () => {
          state.hoverId = null;
          if (state.selectedId && nodeById.has(state.selectedId)) {
            renderDetail(state.selectedId, false);
          } else {
            renderDetail(null, false);
          }
          syncHighlights();
        });
        row.addEventListener("click", () => {
          state.selectedId = node.id;
          renderDetail(node.id, false);
          syncHighlights();
          renderList();
        });
        controls.list.appendChild(row);
      });

      syncHighlights();
    }

    function setControlsCollapsed(collapsed) {
      controls.toolbar.classList.toggle("collapsed", collapsed);
      controls.filtersToggle.classList.toggle("active", !collapsed);
    }

    function setupControls() {
      controls.search.addEventListener("input", (event) => {
        state.searchText = String(event.target.value || "").trim().toLowerCase();
        renderList();
      });
      controls.sort.addEventListener("change", (event) => {
        state.sortKey = String(event.target.value || "relevance");
        renderList();
      });
      controls.yearMin.addEventListener("input", (event) => {
        const value = String(event.target.value || "").trim();
        const parsed = Number(value);
        state.yearMin = value === "" || !Number.isFinite(parsed) ? null : parsed;
        renderList();
      });
      controls.yearMax.addEventListener("input", (event) => {
        const value = String(event.target.value || "").trim();
        const parsed = Number(value);
        state.yearMax = value === "" || !Number.isFinite(parsed) ? null : parsed;
        renderList();
      });
      controls.clearSelection.addEventListener("click", () => {
        state.selectedId = null;
        if (state.hoverId && nodeById.has(state.hoverId)) {
          renderDetail(state.hoverId, true);
        } else {
          renderDetail(null, false);
        }
        syncHighlights();
        renderList();
      });

      controls.chips.forEach((chip) => {
        chip.addEventListener("click", () => {
          const filterKey = chip.getAttribute("data-filter");
          if (!filterKey) {
            return;
          }
          state.filters[filterKey] = !state.filters[filterKey];
          chip.classList.toggle("active", state.filters[filterKey]);
          renderList();
        });
      });

      controls.scopeButtons.forEach((button) => {
        button.addEventListener("click", () => {
          const scope = button.getAttribute("data-scope");
          if (!scope) {
            return;
          }
          state.scopeMode = state.scopeMode === scope ? "all" : scope;
          controls.scopeButtons.forEach((entry) => {
            const entryScope = entry.getAttribute("data-scope");
            entry.classList.toggle("active", !!entryScope && entryScope === state.scopeMode);
          });
          renderList();
        });
      });

      controls.filtersToggle.addEventListener("click", () => {
        const collapsed = !controls.toolbar.classList.contains("collapsed");
        setControlsCollapsed(collapsed);
      });
      controls.listViewBtn.addEventListener("click", () => {
        const listPane = document.getElementById("paper-list-pane");
        if (listPane) {
          listPane.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
      controls.moreBtn.addEventListener("click", () => {
        const focusId = state.selectedId || ((payload.meta && payload.meta.seed_id) || null);
        const focusNode = focusId ? nodeById.get(focusId) : null;
        const target = focusNode && focusNode.links && focusNode.links.semantic_scholar
          ? focusNode.links.semantic_scholar
          : null;
        if (target) {
          window.open(target, "_blank", "noopener,noreferrer");
        }
      });

      function downloadBlob(content, filename, mime) {
        const blob = new Blob([content], { type: mime });
        const anchor = document.createElement("a");
        anchor.href = URL.createObjectURL(blob);
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(anchor.href);
      }

      function seedSlug() {
        const seedNode = nodeById.get((payload.meta && payload.meta.seed_id) || "");
        if (seedNode && seedNode.title) {
          return seedNode.title.replace(/[^a-zA-Z0-9]+/g, "_").substring(0, 40).replace(/_+$/, "").toLowerCase();
        }
        return "citemesh";
      }

      document.getElementById("export-json-btn").addEventListener("click", () => {
        const exportPayload = buildPortableJsonPayload();
        downloadBlob(JSON.stringify(exportPayload, null, 2), seedSlug() + ".json", "application/json");
      });

      document.getElementById("export-csv-btn").addEventListener("click", () => {
        const cols = ["id","title","year","authors","citation_count","venue","arxiv_id","doi","categories","is_seed","provenance","seed_relation","seed_relevance","arxiv_url","doi_url","semantic_scholar_url","abstract"];
        function csvEscape(v) { const s = String(v == null ? "" : v); return s.includes(",") || s.includes('"') || s.includes("\\n") ? '"' + s.replace(/"/g, '""') + '"' : s; }
        const rows = [cols.join(",")];
        for (const n of (payload.nodes || [])) {
          const links = n.links || {};
          rows.push([
            n.id, n.title, n.year, (n.authors||[]).join("; "), n.citation_count, n.venue||"", n.arxiv_id||"", n.doi||"",
            (n.categories||[]).join("; "), n.is_seed, n.provenance||"", n.seed_relation||"",
            Number(n.seed_relevance||0).toFixed(6), links.arxiv_abs||"", links.doi||"", links.semantic_scholar||"", n.abstract||""
          ].map(csvEscape).join(","));
        }
        downloadBlob(rows.join("\\n"), seedSlug() + ".csv", "text/csv;charset=utf-8");
      });

      document.getElementById("export-bib-btn").addEventListener("click", () => {
        const entries = (payload.nodes || []).map((n) => (n.bibtex || "").trim()).filter(Boolean);
        downloadBlob(entries.join("\\n\\n") + "\\n", seedSlug() + ".bib", "text/plain;charset=utf-8");
      });

      const loadInput = document.getElementById("load-json-input");
      document.getElementById("load-json-btn").addEventListener("click", () => { loadInput.click(); });
      loadInput.addEventListener("change", (event) => {
        const file = event.target.files && event.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const imported = parseImportedPayloadFromText(e.target.result, file.name);
            applyImportedPayload(imported, file.name).catch((err) => {
              alert("Failed to load graph view from imported results: " + err.message);
            });
          } catch (err) { alert("Failed to parse imported results: " + err.message); }
        };
        reader.readAsText(file);
        loadInput.value = "";
      });
      controls.resultSelect.addEventListener("change", (event) => {
        const resultId = String(event.target.value || "").trim();
        if (!resultId) {
          return;
        }
        loadCollectionResult(resultId).catch((err) => {
          alert("Failed to load saved results: " + err.message);
        });
      });

      setControlsCollapsed(false);
      populateCollectionSelector();
      updateYearPlaceholders();
    }

    function setupGraphInteractions() {
      if (typeof graphDiv.removeAllListeners === "function") {
        graphDiv.removeAllListeners("plotly_hover");
        graphDiv.removeAllListeners("plotly_unhover");
        graphDiv.removeAllListeners("plotly_click");
      }

      graphDiv.on("plotly_hover", (event) => {
        if (!event || !Array.isArray(event.points)) {
          return;
        }
        const nodePoint = event.points.find((point) => point.curveNumber === nodeTraceIndex);
        if (!nodePoint) {
          return;
        }
        const nodeId = nodeOrder[nodePoint.pointIndex];
        if (!nodeId) {
          return;
        }
        state.hoverId = nodeId;
        renderDetail(nodeId, true);
        syncHighlights();
      });

      graphDiv.on("plotly_unhover", () => {
        state.hoverId = null;
        if (state.selectedId && nodeById.has(state.selectedId)) {
          renderDetail(state.selectedId, false);
        } else {
          renderDetail(null, false);
        }
        syncHighlights();
      });

      graphDiv.on("plotly_click", (event) => {
        if (!event || !Array.isArray(event.points)) {
          return;
        }
        const nodePoint = event.points.find((point) => point.curveNumber === nodeTraceIndex);
        if (!nodePoint) {
          return;
        }
        const nodeId = nodeOrder[nodePoint.pointIndex];
        if (!nodeId) {
          return;
        }
        state.selectedId = nodeId;
        renderDetail(nodeId, false);
        syncHighlights();
        renderList();
      });
    }

    function renderTimeline() {
      const minYear = Number(yearRange.min || 0);
      const maxYear = Number(yearRange.max || 0);
      controls.timelineYearMin.textContent = minYear > 0 ? String(minYear) : "-";
      controls.timelineYearMax.textContent = maxYear > 0 ? String(maxYear) : "-";
    }

    function initialize() {
      setupControls();
      renderTimeline();
      Plotly.react(graphDiv, figureSpec.data, figureSpec.layout, plotConfig).then(() => {
        setupGraphInteractions();
        if (state.selectedId && nodeById.has(state.selectedId)) {
          renderDetail(state.selectedId, false);
        } else {
          renderDetail(null, false);
        }
        renderList();
      });
    }

    initialize();
  </script>
</body>
</html>
"""
        rendered = template
        for token, value in vars_map.items():
            rendered = rendered.replace(token, value)
        return rendered

    def _sorted_nodes(self) -> list[tuple[Hashable, Dict[str, Any]]]:
        """Return nodes sorted by ID for deterministic serialization.

        :return list[tuple[Hashable, Dict[str, Any]]]: Sorted ``(node_id, attrs)``
            pairs.
        """
        return [
            (node_id, self.graph.nodes[node_id])
            for node_id in ordered_nodes(self.graph)
        ]

    def _sorted_edges(self) -> list[tuple[Hashable, Hashable, Dict[str, Any]]]:
        """Return undirected edges with canonical endpoints in stable order.

        :return list[tuple[Hashable, Hashable, Dict[str, Any]]]: Sorted edge tuples in
            ``(u, v, attrs)`` form.
        """
        return ordered_edges_with_data(self.graph)

    def _get_layout(self) -> Dict[Hashable, Iterable[float]]:
        """Compute or reuse cached graph layout.

        :return Dict[Hashable, Iterable[float]]: Mapping of node ID to coordinates.
        """
        if self._layout is None:
            self._layout = compute_layout(self.graph)
        self._layout = _normalize_layout_positions(self._layout)
        return self._layout

    def _node_size(self, node: Hashable) -> float:
        """Compute cached node size for a node ID.

        :param Hashable node: Graph node identifier.
        :return float: Cached node size.
        """
        if self._size_map is None:
            sizes = compute_node_sizes(self.graph)
            ordered_nodes = [node_id for node_id, _ in self._sorted_nodes()]
            self._size_map = {
                graph_node: size for graph_node, size in zip(ordered_nodes, sizes)
            }
        return float(self._size_map.get(node, 300.0))

    def _node_color_hex(self, node: Hashable, theme: Theme) -> str:
        """Convert computed node color to hex for export serializers.

        :param Hashable node: Graph node identifier.
        :param Theme theme: Theme to use.
        :return str: Hex color string.
        """
        cache_key = theme.name
        if cache_key not in self._color_map_cache:
            colors, _, _ = compute_node_colors(self.graph, self.seed_id, theme)
            ordered_nodes = [node_id for node_id, _ in self._sorted_nodes()]
            self._color_map_cache[cache_key] = {
                graph_node: color for graph_node, color in zip(ordered_nodes, colors)
            }

        color = self._color_map_cache[cache_key].get(node, theme.node_color_new)
        return _rgb_tuple_to_hex(color)

    def _plotly_div_id(self, prefix: str = "citemesh-plotly") -> str:
        """Build a deterministic Plotly HTML container id.

        :param str prefix: Prefix for resulting ``div_id``.
        :return str: Stable ``div_id`` derived from seed id and sorted graph structure.
        """
        nodes = [str(node_id) for node_id, _ in self._sorted_nodes()]
        edges = [
            (
                str(left),
                str(right),
                round(float(attrs.get("weight", 0.0)), 8),
            )
            for left, right, attrs in self._sorted_edges()
        ]
        digest_payload = json.dumps(
            {"seed_id": str(self.seed_id), "nodes": nodes, "edges": edges},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha1(digest_payload.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}-{digest}"

    def _plotly_year_scale(
        self, node_ids: list[Hashable]
    ) -> Tuple[list[float], float, float]:
        """Build deterministic Plotly marker years and explicit scale bounds.

        :param list[Hashable] node_ids: Sorted node identifiers for the current graph.
        :return Tuple[list[float], float, float]: Marker years, color-scale min, and
            color-scale max.
        """
        raw_years = [
            self._coerce_year(self.graph.nodes[node].get("year")) for node in node_ids
        ]
        valid_years = [year for year in raw_years if year > 0]

        if valid_years:
            year_min = float(min(valid_years))
            year_max = float(max(valid_years))
        else:
            year_min = float(MISSING_YEAR_FALLBACK_MIN)
            year_max = float(MISSING_YEAR_FALLBACK_MAX)

        if year_max <= year_min:
            year_max = year_min + 1.0

        midpoint = (year_min + year_max) / 2.0
        normalized_years = [float(year) if year > 0 else midpoint for year in raw_years]
        return normalized_years, year_min, year_max

    @staticmethod
    def _coerce_year(raw_year: object) -> int:
        """Normalize optional year values for formats that disallow null years.

        :param object raw_year: Raw year value from node metadata.
        :return int: Integer year when valid, otherwise ``0``.
        """
        if isinstance(raw_year, bool):
            return 0

        # Accept native and NumPy integer-like values.
        try:
            import numbers

            if isinstance(raw_year, numbers.Integral):
                return int(raw_year)
        except (TypeError, ValueError):
            pass

        # Some upstream callers may provide year as a numeric string.
        if isinstance(raw_year, str):
            try:
                return int(raw_year)
            except ValueError:
                pass

        return 0

    @staticmethod
    def _serialize_node(node_id: Hashable, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize node attributes into JSON/GraphML friendly dict.

        :param Hashable node_id: Graph node identifier.
        :param Dict[str, Any] attrs: Raw node attributes.
        :return Dict[str, Any]: JSON/GraphML-safe node payload.
        """
        paper: Optional[Paper] = attrs.get("paper")

        node_data = {
            "id": node_id,
            "title": attrs.get("title", ""),
            "year": attrs.get("year"),
            "citation_count": attrs.get("citation_count", 0),
            "venue": attrs.get("venue", ""),
            "arxiv_id": attrs.get("arxiv_id", ""),
            "doi": attrs.get("doi", ""),
            "is_seed": bool(attrs.get("is_seed", False)),
        }

        if paper:
            node_data.update(
                {
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.abstract,
                    "venue": getattr(paper, "venue", "") or attrs.get("venue", ""),
                    "arxiv_id": (
                        getattr(paper, "arxiv_id", "") or attrs.get("arxiv_id", "")
                    ),
                    "doi": getattr(paper, "doi", "") or attrs.get("doi", ""),
                    "categories": paper.categories,
                }
            )
        else:
            node_data.setdefault("authors", attrs.get("authors", []))
            node_data.setdefault("abstract", "")
            node_data.setdefault("categories", [])

        return node_data

    @staticmethod
    def _node_title(attrs: Dict[str, Any], node_id: Hashable) -> str:
        """Return stable node title for edge-sidecar export fields.

        :param Dict[str, Any] attrs: Node attributes map.
        :param Hashable node_id: Node identifier fallback.
        :return str: Human-readable title fallback.
        """
        title = str(attrs.get("title") or "").strip()
        if title:
            return title
        return str(node_id)

    @classmethod
    def _node_short_label(cls, attrs: Dict[str, Any], node_id: Hashable) -> str:
        """Return compact node label for edge export fields.

        :param Dict[str, Any] attrs: Node attributes map.
        :param Hashable node_id: Node identifier fallback.
        :return str: Compact label (author/year or title fallback).
        """
        title = cls._node_title(attrs, node_id)
        raw_authors = attrs.get("authors", [])
        surname = ""
        if isinstance(raw_authors, list) and raw_authors:
            first_author = str(raw_authors[0]).strip()
            surname = first_author.split()[-1] if first_author else ""

        year = cls._coerce_year(attrs.get("year"))
        if surname and year > 0:
            return f"{surname}, {year}"
        if year > 0:
            return f"{title} ({year})"
        return title


def _rgb_tuple_to_hex(color: tuple) -> str:
    """Convert RGB tuple (0-1) to hex string.

    :param tuple color: RGB triple in [0, 1] space.
    :return str: HTML hex color code.
    """
    r, g, b = color
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0, min(1, r)) * 255),
        int(max(0, min(1, g)) * 255),
        int(max(0, min(1, b)) * 255),
    )


def _rgb_tuple_to_rgba(color: object, alpha: float) -> str:
    """Convert theme color payloads to CSS ``rgba()`` string.

    Accepts RGB tuples in ``[0, 1]`` space, RGB tuples in ``[0, 255]`` space,
    hex strings, and ``rgb()/rgba()`` strings.

    :param object color: Color payload.
    :param float alpha: Alpha value in ``[0, 1]``.
    :return str: CSS rgba() color string.
    """
    r: int
    g: int
    b: int
    if isinstance(color, str):
        text = color.strip()
        hex_match = re.fullmatch(r"#([0-9a-fA-F]{6})", text)
        if hex_match:
            hex_value = hex_match.group(1)
            r = int(hex_value[0:2], 16)
            g = int(hex_value[2:4], 16)
            b = int(hex_value[4:6], 16)
        else:
            number_parts = re.findall(r"(\d+(?:\.\d+)?)", text)
            if len(number_parts) >= 3:
                r = int(float(number_parts[0]))
                g = int(float(number_parts[1]))
                b = int(float(number_parts[2]))
            else:
                r, g, b = 255, 255, 255
    else:
        try:
            values = list(color)  # type: ignore[arg-type]
        except TypeError:
            values = [1.0, 1.0, 1.0]
        if len(values) < 3:
            values = [1.0, 1.0, 1.0]
        raw_r = float(values[0])
        raw_g = float(values[1])
        raw_b = float(values[2])
        if max(abs(raw_r), abs(raw_g), abs(raw_b)) <= 1.0:
            r = int(max(0.0, min(1.0, raw_r)) * 255.0)
            g = int(max(0.0, min(1.0, raw_g)) * 255.0)
            b = int(max(0.0, min(1.0, raw_b)) * 255.0)
        else:
            r = int(max(0.0, min(255.0, raw_r)))
            g = int(max(0.0, min(255.0, raw_g)))
            b = int(max(0.0, min(255.0, raw_b)))

    clamped_alpha = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},{b},{clamped_alpha:.3f})"
