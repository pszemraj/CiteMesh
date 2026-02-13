"""
Graph export utilities for CiteMesh.

Provides a unified interface for exporting graphs to multiple formats,
including interactive visualizations.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import networkx as nx

from citemesh.core import Paper

from .ordering import ordered_edges_with_data, ordered_nodes
from .render import compute_layout, compute_node_colors, compute_node_sizes
from .themes import Theme, get_theme


class GraphExporter:
    """Unified interface for exporting graphs in multiple formats."""

    def __init__(
        self,
        graph: nx.Graph,
        seed_id: str,
        metadata: Optional[Dict] = None,
        theme_name: str = "light",
        layout: Optional[Dict[str, Iterable[float]]] = None,
    ):
        """Create exporter bound to a graph and seed paper metadata.

        :param nx.Graph graph: Graph to export.
        :param str seed_id: Seed paper identifier.
        :param Optional[Dict] metadata: Optional metadata to include in outputs.
        :param str theme_name: Theme for visual color defaults.
        :param Optional[Dict[str, Iterable[float]]] layout: Optional precomputed layout.
        """
        self.graph = graph
        self.seed_id = seed_id
        self.metadata = metadata or {}
        self.theme = get_theme(theme_name)
        self._layout = layout
        self._size_map: Optional[Dict[str, float]] = None
        self._color_map_cache: Dict[str, Dict[str, tuple]] = {}

    # ------------------------------------------------------------------
    # Public export methods

    def to_json(self, path: Path) -> None:
        """Export full graph with metadata as JSON."""
        sorted_nodes = self._sorted_nodes()
        sorted_edges = self._sorted_edges()
        data = {
            "metadata": self.metadata,
            "seed_id": self.seed_id,
            "nodes": [
                self._serialize_node(node, attrs) for node, attrs in sorted_nodes
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "weight": float(data.get("weight", 0.0)),
                }
                for u, v, data in sorted_edges
            ],
        }
        Path(path).write_text(json.dumps(data, sort_keys=True, indent=2))

    def to_graphml(self, path: Path) -> None:
        """Export to GraphML for external tools such as Gephi or Cytoscape."""
        export_graph = nx.Graph()
        sorted_nodes = self._sorted_nodes()
        sorted_edges = self._sorted_edges()

        for node, attrs in sorted_nodes:
            cleaned = self._serialize_node(node, attrs)
            cleaned["year"] = self._coerce_year(cleaned.get("year"))
            if isinstance(cleaned.get("authors"), list):
                cleaned["authors"] = ", ".join(cleaned["authors"])
            if isinstance(cleaned.get("categories"), list):
                cleaned["categories"] = ", ".join(cleaned["categories"])
            cleaned["is_seed"] = int(bool(cleaned.get("is_seed")))
            export_graph.add_node(node, **cleaned)

        for u, v, data in sorted_edges:
            export_graph.add_edge(
                u,
                v,
                **{k: float(val) if k == "weight" else val for k, val in data.items()},
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
        node_years = [
            self._coerce_year(self.graph.nodes[node].get("year")) for node in node_ids
        ]
        node_labels = [
            self.graph.nodes[node].get("paper").label
            if self.graph.nodes[node].get("paper")
            else self.graph.nodes[node].get("title", node)
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
            mode="markers+text",
            hoverinfo="text",
            text=node_labels,
            textposition="bottom center",
            textfont=dict(size=8, color=theme_obj.text_color),
            marker=dict(
                size=node_sizes,
                color=node_years,
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

        title_text = self.graph.nodes[self.seed_id].get("title", "CiteMesh")[:50]

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"CiteMesh: {title_text}...",
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

        fig.write_html(str(path))

    # ------------------------------------------------------------------
    # Internal helpers

    def _sorted_nodes(self) -> list[tuple[str, Dict]]:
        """Return nodes sorted by ID for deterministic serialization.

        :return list[tuple[str, Dict]]: Sorted ``(node_id, attrs)`` pairs.
        """
        return [
            (node_id, self.graph.nodes[node_id])
            for node_id in ordered_nodes(self.graph)
        ]

    def _sorted_edges(self) -> list[tuple[str, str, Dict]]:
        """Return undirected edges with canonical endpoints in stable order.

        :return list[tuple[str, str, Dict]]: Sorted edge tuples in ``(u, v, attrs)`` form.
        """
        return ordered_edges_with_data(self.graph)

    def _get_layout(self) -> Dict[str, Iterable[float]]:
        """Compute or reuse cached graph layout.

        :return Dict[str, Iterable[float]]: Mapping of node ID to coordinates.
        """
        if self._layout is None:
            self._layout = compute_layout(self.graph)
        return self._layout

    def _node_size(self, node: str) -> float:
        """Compute cached node size for a node ID.

        :param str node: Graph node identifier.
        :return float: Cached node size.
        """
        if self._size_map is None:
            sizes = compute_node_sizes(self.graph)
            ordered_nodes = [node_id for node_id, _ in self._sorted_nodes()]
            self._size_map = {
                graph_node: size for graph_node, size in zip(ordered_nodes, sizes)
            }
        return float(self._size_map.get(node, 300.0))

    def _node_color_hex(self, node: str, theme: Theme) -> str:
        """Convert computed node color to hex for export serializers.

        :param str node: Graph node identifier.
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
    def _serialize_node(node_id: str, attrs: Dict) -> Dict:
        """Serialize node attributes into JSON/GraphML friendly dict.

        :param str node_id: Graph node identifier.
        :param Dict attrs: Raw node attributes.
        :return Dict: JSON/GraphML-safe node payload.
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
