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

from citemesh.models import Paper
from citemesh.themes import Theme, get_theme
from citemesh.visualization import compute_layout, compute_node_colors, compute_node_sizes


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
        data = {
            "metadata": self.metadata,
            "seed_id": self.seed_id,
            "nodes": [self._serialize_node(node, attrs) for node, attrs in self.graph.nodes(data=True)],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "weight": float(data.get("weight", 0.0)),
                }
                for u, v, data in self.graph.edges(data=True)
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def to_graphml(self, path: Path) -> None:
        """Export to GraphML for external tools such as Gephi or Cytoscape."""
        export_graph = nx.Graph()

        for node, attrs in self.graph.nodes(data=True):
            cleaned = self._serialize_node(node, attrs)
            if isinstance(cleaned.get("authors"), list):
                cleaned["authors"] = ", ".join(cleaned["authors"])
            if isinstance(cleaned.get("categories"), list):
                cleaned["categories"] = ", ".join(cleaned["categories"])
            cleaned["is_seed"] = int(bool(cleaned.get("is_seed")))
            export_graph.add_node(node, **cleaned)

        for u, v, data in self.graph.edges(data=True):
            export_graph.add_edge(u, v, **{k: float(v) if k == "weight" else v for k, v in data.items()})

        nx.write_graphml(export_graph, path)

    def to_interactive_html(
        self,
        path: Path,
        theme: Optional[str] = None,
        physics: bool = True,
    ) -> None:
        """
        Create interactive HTML visualization with pyvis (vis.js).

        Args:
            path: Output HTML path.
            theme: Optional override for theme.
            physics: Whether to enable force-directed physics.
        """
        try:
            from pyvis.network import Network
        except ImportError as exc:
            raise RuntimeError(
                "pyvis is required for HTML export. Install the 'pyvis' dependency."
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

        for node in self.graph.nodes():
            paper: Optional[Paper] = self.graph.nodes[node].get("paper")
            size = self._node_size(node)
            color = self._node_color_hex(node, theme_obj)

            label = paper.label if paper else self.graph.nodes[node].get("title", node)

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
                tooltip_lines.append(html.escape(self.graph.nodes[node].get("title", "")))

            net.add_node(
                node,
                label=label,
                title="<br>".join(tooltip_lines),
                size=max(6, size / 30),
                color=color,
                borderWidth=3 if self.graph.nodes[node].get("is_seed") else 1,
            )

        for u, v, data in self.graph.edges(data=True):
            weight = float(data.get("weight", 0.1))
            net.add_edge(u, v, value=max(0.1, weight * 5))

        net.save_graph(str(path))

    def to_plotly_html(self, path: Path, theme: Optional[str] = None) -> None:
        """Create Plotly interactive visualization."""
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise RuntimeError(
                "plotly is required for Plotly export. Install the 'plotly' dependency."
            ) from exc

        theme_obj = get_theme(theme) if theme else self.theme
        pos = self._get_layout()

        edge_x, edge_y = [], []
        for u, v in self.graph.edges():
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

        node_ids = list(self.graph.nodes())
        node_x = [pos[node][0] for node in node_ids]
        node_y = [pos[node][1] for node in node_ids]
        node_sizes = [max(6, self._node_size(node) / 50) for node in node_ids]
        node_years = [self.graph.nodes[node].get("year", 0) for node in node_ids]
        node_labels = [
            self.graph.nodes[node]
            .get("paper")
            .label
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
                hover_texts.append(html.escape(self.graph.nodes[node].get("title", node)))

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

    def _get_layout(self) -> Dict[str, Iterable[float]]:
        if self._layout is None:
            self._layout = compute_layout(self.graph)
        return self._layout

    def _node_size(self, node: str) -> float:
        if self._size_map is None:
            sizes = compute_node_sizes(self.graph)
            self._size_map = {
                graph_node: size for graph_node, size in zip(self.graph.nodes(), sizes)
            }
        return float(self._size_map.get(node, 300.0))

    def _node_color_hex(self, node: str, theme: Theme) -> str:
        cache_key = theme.name
        if cache_key not in self._color_map_cache:
            colors, _, _ = compute_node_colors(self.graph, self.seed_id, theme)
            self._color_map_cache[cache_key] = {
                graph_node: color
                for graph_node, color in zip(self.graph.nodes(), colors)
            }

        color = self._color_map_cache[cache_key].get(node, theme.node_color_new)
        return _rgb_tuple_to_hex(color)

    @staticmethod
    def _serialize_node(node_id: str, attrs: Dict) -> Dict:
        """Serialize node attributes into JSON/GraphML friendly dict."""
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
    """Convert RGB tuple (0-1) to hex string."""
    r, g, b = color
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0, min(1, r)) * 255),
        int(max(0, min(1, g)) * 255),
        int(max(0, min(1, b)) * 255),
    )
