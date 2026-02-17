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
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, Optional, Tuple

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
from .text_limits import clamp_render_text
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

            label = clamp_render_text(
                paper.label if paper else attrs.get("title", node)
            )

            tooltip_lines = []
            if paper:
                tooltip_lines.append(
                    f"<b>{html.escape(clamp_render_text(paper.title))}</b>"
                )
                tooltip_lines.append(
                    f"{html.escape(paper.first_author_surname)} et al., {paper.year}"
                )
                tooltip_lines.append(f"Citations: {paper.citation_count}")
                if paper.categories:
                    cats = clamp_render_text(
                        ", ".join(str(cat) for cat in paper.categories[:3])
                    )
                    tooltip_lines.append(f"Categories: {html.escape(cats)}")
            else:
                tooltip_lines.append(
                    html.escape(clamp_render_text(attrs.get("title", "")))
                )

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
        pos = self._get_layout()

        edge_x, edge_y = [], []
        for u, v, _ in self._sorted_edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color=_rgb_tuple_to_hex(theme_obj.edge_color)),
            hoverinfo="none",
            mode="lines",
        )

        node_ids = [node_id for node_id, _ in self._sorted_nodes()]
        node_x = [pos[node][0] for node in node_ids]
        node_y = [pos[node][1] for node in node_ids]
        node_sizes = [max(6, self._node_size(node) / 50) for node in node_ids]
        node_years, year_min, year_max = self._plotly_year_scale(node_ids)
        node_labels = [
            clamp_render_text(self.graph.nodes[node].get("paper").label)
            if self.graph.nodes[node].get("paper")
            else clamp_render_text(self.graph.nodes[node].get("title", node))
            for node in node_ids
        ]

        hover_texts = []
        for node in node_ids:
            paper: Optional[Paper] = self.graph.nodes[node].get("paper")
            if paper:
                authors = ", ".join(a.name for a in paper.authors[:3]) or "Unknown"
                hover_texts.append(
                    "<br>".join(
                        [
                            f"<b>{html.escape(clamp_render_text(paper.title))}</b>",
                            html.escape(clamp_render_text(authors)),
                            f"Year: {paper.year} | Citations: {paper.citation_count}",
                        ]
                    )
                )
            else:
                hover_texts.append(
                    html.escape(
                        clamp_render_text(self.graph.nodes[node].get("title", node))
                    )
                )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=node_labels,
            textposition="bottom center",
            textfont=dict(size=8, color=theme_obj.text_color),
            marker=dict(
                size=node_sizes,
                color=node_years,
                cmin=year_min,
                cmax=year_max,
                colorscale="Plasma" if theme_obj.name == "dark" else "Viridis",
                line=dict(width=2, color=theme_obj.text_color),
                showscale=True,
                colorbar=dict(
                    thickness=15,
                    xanchor="left",
                    title=dict(text="Year", side="right"),
                ),
            ),
            hovertext=hover_texts,
        )

        raw_title = clamp_render_text(
            " ".join(
                str(self.graph.nodes[self.seed_id].get("title", "CiteMesh")).split()
            )
        )
        title_text = "<br>".join(
            textwrap.wrap(raw_title, width=72, break_long_words=False)
        )
        if not title_text:
            title_text = "CiteMesh"

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"CiteMesh: {title_text}",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor=theme_obj.background,
                paper_bgcolor=theme_obj.background,
                font=dict(color=theme_obj.text_color),
            ),
        )

        div_id = self._plotly_div_id()
        try:
            fig.write_html(str(path), div_id=div_id)
        except TypeError as exc:
            raise RuntimeError(
                "Deterministic Plotly export requires write_html(div_id=...). "
                "Upgrade plotly to a version that supports div_id."
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers

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

    def _plotly_div_id(self) -> str:
        """Build a deterministic Plotly HTML container id.

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
        return f"citemesh-plotly-{digest}"

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
            "is_seed": bool(attrs.get("is_seed", False)),
        }

        if paper:
            node_data.update(
                {
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.abstract,
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
