"""
Graph export utilities for CiteMesh.

Provides a unified interface for exporting graphs to multiple formats,
including interactive visualizations.
"""

from __future__ import annotations

import hashlib
import html
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
        """Export graph data JSON focused on nodes/edges and readable edge context."""
        sorted_nodes = self._sorted_nodes()
        sorted_edges = self._sorted_edges()
        data = {
            "seed_id": str(self.seed_id),
            "summary": {
                "nodes": len(sorted_nodes),
                "edges": len(sorted_edges),
            },
            "nodes": [
                self._serialize_node(node, attrs) for node, attrs in sorted_nodes
            ],
            "edges": [
                {
                    "source": str(u),
                    "target": str(v),
                    "source_title": self._node_title(self.graph.nodes[u], u),
                    "target_title": self._node_title(self.graph.nodes[v], v),
                    "source_label": self._node_short_label(self.graph.nodes[u], u),
                    "target_label": self._node_short_label(self.graph.nodes[v], v),
                    "weight": float(data.get("weight", 0.0)),
                }
                for u, v, data in sorted_edges
            ],
        }
        Path(path).write_text(json.dumps(data, sort_keys=True, indent=2))

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
            from pyvis.network import Network
        except ImportError as exc:
            raise RuntimeError(
                "pyvis is required for HTML export. Install with: pip install citemesh[viz]."
            ) from exc

        theme_obj = get_theme(theme) if theme else self.theme

        net = Network(
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
            from plotly import graph_objects as go
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
            from plotly import graph_objects as go
            from plotly.offline import get_plotlyjs
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
        html_output = self._dashboard_template(
            theme_obj=theme_obj,
            div_id=div_id,
            plotly_js=self._safe_script_content(get_plotlyjs()),
            payload_json=payload_json,
            figure_json=figure_json,
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

    def _dashboard_payload(
        self, *, theme_obj: Theme, node_ids: list[Hashable]
    ) -> Dict[str, Any]:
        """Build deterministic dashboard payload from graph metadata.

        :param Theme theme_obj: Active visualization theme.
        :param list[Hashable] node_ids: Node order used by Plotly points.
        :return Dict[str, Any]: JSON payload consumed by dashboard JS.
        """
        provenance = self._provenance_map()
        seed_relations = self._seed_relation_map()
        relevance = self._seed_relevance_scores()
        sorted_nodes = self._sorted_nodes()
        sorted_edges = self._sorted_edges()
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
                    "nodes": len(sorted_nodes),
                    "edges": len(sorted_edges),
                },
                "year_range": year_range,
                "plotly_node_order": [str(node_id) for node_id in node_ids],
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

    def _default_provenance(self, *, strategy: str) -> str:
        """Resolve default provenance class for non-hybrid strategies.

        :param str strategy: Strategy metadata token.
        :return str: One of ``citation`` or ``semantic``.
        """
        if strategy == "embedding":
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
    ) -> str:
        """Render standalone dashboard HTML template.

        :param Theme theme_obj: Active theme.
        :param str div_id: Plotly mount div ID.
        :param str plotly_js: Inline Plotly runtime JS.
        :param str payload_json: Serialized dashboard payload JSON.
        :param str figure_json: Serialized Plotly figure JSON.
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
    #toolbar-controls {
      display: grid;
      gap: 8px;
    }
    #dashboard-toolbar.collapsed #toolbar-controls {
      display: none;
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
    .js-plotly-plot .scatterlayer path.point.is-dimmed {
      opacity: 0.24;
    }
    .js-plotly-plot .scatterlayer path.point.is-filter-hidden {
      opacity: 0.16;
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
    </div>
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
  <script>
    const payload = JSON.parse(document.getElementById("citemesh-dashboard-data").textContent);
    const figureSpec = JSON.parse(document.getElementById("citemesh-dashboard-figure").textContent);
    const graphDiv = document.getElementById("__PLOTLY_DIV_ID__");

    const nodes = payload.nodes || [];
    const nodeOrder = (payload.meta && payload.meta.plotly_node_order) || [];
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const nodeIndexById = new Map(nodeOrder.map((nodeId, idx) => [nodeId, idx]));
    const yearRange = (payload.meta && payload.meta.year_range) || {};
    const seedNode = nodeById.get((payload.meta && payload.meta.seed_id) || "");
    const seedYear = seedNode && Number.isFinite(Number(seedNode.year)) && Number(seedNode.year) > 0
      ? Number(seedNode.year)
      : null;

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

    const traceSpecs = figureSpec.data || [];
    const nodeTraceIndex = (() => {
      const namedIdx = traceSpecs.findIndex((trace) => String(trace.name || "") === "nodes");
      if (namedIdx >= 0) {
        return namedIdx;
      }
      const fallbackIdx = traceSpecs.findIndex((trace) => String(trace.mode || "").includes("markers+text"));
      return fallbackIdx >= 0 ? fallbackIdx : 0;
    })();
    const haloTraceIndex = traceSpecs.findIndex(
      (trace) => String(trace.name || "") === "selection-halo"
    );
    const neighborhoodTraceIndex = traceSpecs.findIndex(
      (trace) => String(trace.name || "") === "neighborhood-edges"
    );
    const nodeTraceSource = (traceSpecs[nodeTraceIndex] || {});
    const markerSource = nodeTraceSource.marker || {};
    const defaultNodeSizes = normalizeArray(markerSource.size, nodeOrder.length, 8);
    const defaultNodeX = normalizeArray(nodeTraceSource.x, nodeOrder.length, 0);
    const defaultNodeY = normalizeArray(nodeTraceSource.y, nodeOrder.length, 0);

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

    const adjacency = new Map();
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
    };

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
      controls.graphHint.textContent = previewOnly ? "Previewing node" : "Selection locked";
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
      const focusId = state.selectedId || state.hoverId;

      graphNodePaths.forEach((path, idx) => {
        const rawPointIndex = path.getAttribute("data-point-number");
        const pointIndex = Number(rawPointIndex);
        const stableIdx = Number.isInteger(pointIndex) && pointIndex >= 0 ? pointIndex : idx;
        const nodeId = nodeOrder[stableIdx];
        const isVisible = !!nodeId && state.visibleIds.has(nodeId);
        const isTarget = stableIdx === hoverIndex || stableIdx === selectedIndex;
        path.classList.toggle("is-filter-hidden", !isVisible);
        path.classList.toggle("is-dimmed", hasActiveFocus && !isTarget);
        path.classList.toggle("is-glowing", isTarget);
      });

      if (neighborhoodTraceIndex >= 0) {
        let neighborhoodX = [];
        let neighborhoodY = [];
        if (focusId && adjacency.has(focusId)) {
          for (const entry of adjacency.get(focusId)) {
            if (!nodeIndexById.has(entry.id) || !nodeIndexById.has(focusId)) {
              continue;
            }
            const leftIdx = nodeIndexById.get(focusId);
            const rightIdx = nodeIndexById.get(entry.id);
            neighborhoodX.push(defaultNodeX[leftIdx], defaultNodeX[rightIdx], null);
            neighborhoodY.push(defaultNodeY[leftIdx], defaultNodeY[rightIdx], null);
          }
        }
        Plotly.restyle(
          graphDiv,
          {
            x: [neighborhoodX],
            y: [neighborhoodY],
          },
          [neighborhoodTraceIndex]
        );
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
          if (!state.selectedId) {
            renderDetail(node.id, true);
          }
          syncHighlights();
        });
        row.addEventListener("mouseleave", () => {
          state.hoverId = null;
          if (!state.selectedId) {
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
        renderDetail(null, false);
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
      setControlsCollapsed(false);

      const validYears = nodes
        .map((node) => (hasYear(node) ? Number(node.year) : null))
        .filter((year) => year !== null);
      if (validYears.length) {
        const minYear = Math.min(...validYears);
        const maxYear = Math.max(...validYears);
        controls.yearMin.placeholder = `Year min (${minYear})`;
        controls.yearMax.placeholder = `Year max (${maxYear})`;
      }
    }

    function setupGraphInteractions() {
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
        if (!state.selectedId) {
          renderDetail(nodeId, true);
        }
        syncHighlights();
      });

      graphDiv.on("plotly_unhover", () => {
        state.hoverId = null;
        if (!state.selectedId) {
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
      Plotly.newPlot(graphDiv, figureSpec.data, figureSpec.layout, {
        displaylogo: false,
        responsive: true,
      }).then(() => {
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
