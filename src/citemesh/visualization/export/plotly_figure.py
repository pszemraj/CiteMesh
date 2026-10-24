"""Deterministic Plotly figure assembly: traces, axes, and annotations."""

from __future__ import annotations

import hashlib
import html
import json
import math
import textwrap
from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple, TypedDict

import numpy as np
from numpy.typing import NDArray

from citemesh.core.values import parse_float

from ..themes import Theme
from ..years import coerce_publication_year, publication_year_scale
from ._host import _ExporterHost
from .geometry import (
    _HOVER_RELATION_LABELS,
    DASHBOARD_AXIS_MIN_PADDING,
    DASHBOARD_AXIS_X_PADDING,
    DASHBOARD_FOOTER_MARGIN,
    DASHBOARD_MAX_NODE_DIAMETER,
    DASHBOARD_SELECTION_HALO_SCALE,
    PlotlyHoverLabel,
    _edge_strength_scale,
    _rgb_tuple_to_hex,
    _rgb_tuple_to_rgba,
    _select_dashboard_label_nodes,
    _stable_curve_direction,
    theme_hover_label,
    theme_label_text_alpha,
)
from .loaders import PlotlyGraphObjects
from .nodes import (
    _node_short_label,
    _provenance_map,
    _seed_relation_map,
    _serialize_node,
    _sorted_edges,
    _sorted_nodes,
    _strategy,
)

if TYPE_CHECKING:
    from plotly.graph_objects import Figure, Scatter

    from citemesh.core.models import PaperGraph


class _TextFont(TypedDict):
    """Text font fields supplied to Plotly traces and annotations."""

    size: int
    color: str


class _ColorbarTitle(TypedDict):
    """Title placement for the year colorbar."""

    text: str
    side: Literal["right"]


class _MarkerColorbar(TypedDict):
    """Year colorbar options for static figures."""

    thickness: int
    xanchor: Literal["left"]
    title: _ColorbarTitle


class _PathLine(TypedDict):
    """Stroke options for a curved dashboard edge."""

    color: str
    width: float


class _PathShape(TypedDict):
    """Curved path shape placed beneath dashboard points."""

    type: Literal["path"]
    path: str
    line: _PathLine
    layer: Literal["below"]


class _PlotlyAnnotation(TypedDict):
    """Text label positioned beside one dashboard node."""

    x: float
    y: float
    xref: Literal["x"]
    yref: Literal["y"]
    text: str
    font: _TextFont
    showarrow: Literal[False]
    xanchor: Literal["left", "center", "right"]
    yanchor: Literal["bottom"]
    borderpad: int
    yshift: float


class _AxisOptions(TypedDict, total=False):
    """Axis settings used by static and dashboard figures."""

    showgrid: bool
    zeroline: bool
    showticklabels: bool
    autorange: bool
    range: list[float]


class _PlotlyFont(TypedDict):
    """Figure-wide text color."""

    color: str


class _PlotlyLayoutKwargs(TypedDict, total=False):
    """Layout properties assembled by the graph exporter."""

    showlegend: bool
    hovermode: Literal["closest"]
    margin: dict[str, int]
    xaxis: _AxisOptions
    yaxis: _AxisOptions
    plot_bgcolor: str
    paper_bgcolor: str
    font: _PlotlyFont
    hoverlabel: PlotlyHoverLabel
    annotations: list[dict[str, object]]
    uirevision: str
    shapes: list[_PathShape]
    title: str


def _plotly_title_text(graph: PaperGraph, seed_id: str) -> str:
    """Build wrapped seed title text used by Plotly figure titles.

    :param PaperGraph graph: Graph containing the seed node.
    :param str seed_id: Seed paper identifier.
    :return str: Wrapped title string.
    """
    seed_attrs = graph.nodes[seed_id] if seed_id in graph else {}
    title = (
        _serialize_node(seed_id, seed_attrs).get("title", "CiteMesh")
        if seed_attrs
        else "CiteMesh"
    )
    # Plotly renders titles as pseudo-HTML, so upstream markup must be escaped
    # before wrapping; only the "<br>" joins below stay live markup.
    raw_title = html.escape(" ".join(str(title).split()))
    title_text = "<br>".join(textwrap.wrap(raw_title, width=72, break_long_words=False))
    if not title_text:
        return "CiteMesh"
    return title_text


class _NodeStyle(NamedTuple):
    """Resolved per-figure node label text and marker styling."""

    labels: list[str]
    text_position: str
    text_font: _TextFont
    color_scale: str | list[list[float | str]]
    marker_line_width: int | list[float]
    marker_line_color: str | list[str]
    marker_showscale: bool
    marker_colorbar: _MarkerColorbar | None


def _build_edge_layer(
    *,
    go: PlotlyGraphObjects,
    theme_obj: Theme,
    graph: PaperGraph,
    pos: Mapping[Hashable, NDArray[np.float64]],
    for_dashboard: bool,
) -> tuple[list[_PathShape], Scatter | None]:
    """Build the edge layer as curved dashboard shapes or a straight-line trace.

    :param PlotlyGraphObjects go: Lazily loaded Plotly graph_objects module.
    :param Theme theme_obj: Active visualization theme.
    :param PaperGraph graph: Graph supplying deterministically ordered edges.
    :param Mapping[Hashable, NDArray[np.float64]] pos: Normalized node positions.
    :param bool for_dashboard: Whether to emit curved dashboard shapes.
    :return tuple[list[_PathShape], Scatter | None]: Layout shapes and the
        optional straight-edge scatter trace.
    """
    layout_shapes: list[_PathShape] = []
    edge_trace: Scatter | None = None

    if for_dashboard:
        curvature = 0.15
        edge_records = list(_sorted_edges(graph))
        edge_strengths = _edge_strength_scale(
            [
                max(parse_float(attrs.get("weight", 0.0)), 0.0)
                for _, _, attrs in edge_records
            ]
        )
        for (u, v, attrs), strength in zip(edge_records, edge_strengths):
            x0f = float(pos[u][0])
            y0f = float(pos[u][1])
            x1f = float(pos[v][0])
            y1f = float(pos[v][1])
            mid_x = (x0f + x1f) / 2.0
            mid_y = (y0f + y1f) / 2.0
            dx = x1f - x0f
            dy = y1f - y0f
            direction = _stable_curve_direction(u, v)
            cx = mid_x - dy * curvature * direction
            cy = mid_y + dx * curvature * direction
            layout_shapes.append(
                {
                    "type": "path",
                    "path": f"M {x0f},{y0f} Q {cx},{cy} {x1f},{y1f}",
                    "line": {
                        "color": _rgb_tuple_to_rgba(
                            theme_obj.edge_color, 0.07 + 0.25 * strength
                        ),
                        "width": 0.45 + 1.2 * strength,
                    },
                    "layer": "below",
                }
            )
    else:
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for u, v, _ in _sorted_edges(graph):
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

    return layout_shapes, edge_trace


def _build_node_style(
    *,
    graph: PaperGraph,
    node_ids: list[Hashable],
    pos: Mapping[Hashable, NDArray[np.float64]],
    base_labels: list[object],
    theme_obj: Theme,
    for_dashboard: bool,
) -> _NodeStyle:
    """Resolve node label text and marker styling for the active render mode.

    :param PaperGraph graph: Graph supplying seed and ranking attributes.
    :param list[Hashable] node_ids: Deterministically ordered node identifiers.
    :param Mapping[Hashable, NDArray[np.float64]] pos: Normalized node positions.
    :param list[object] base_labels: Raw per-node label candidates.
    :param Theme theme_obj: Active visualization theme.
    :param bool for_dashboard: Whether to apply dashboard label thinning.
    :return _NodeStyle: Escaped labels plus marker and text styling.
    """
    text_font: _TextFont
    color_scale: str | list[list[float | str]]
    marker_line_width: int | list[float]
    marker_line_color: str | list[str]
    marker_colorbar: _MarkerColorbar | None

    if for_dashboard:
        label_nodes = _select_dashboard_label_nodes(graph, node_ids, pos)
        node_labels = [
            str(base_labels[idx]) if node_id in label_nodes else ""
            for idx, node_id in enumerate(node_ids)
        ]
        text_position = "top center"
        # Keep one marker trace so point indices stay stable for hover/click sync;
        # use a muted shared text alpha instead of per-point text styling.
        text_font = _TextFont(
            size=10,
            color=_rgb_tuple_to_rgba(
                theme_obj.text_color, theme_label_text_alpha(theme_obj)
            ),
        )
        color_scale = [
            [0.0, _rgb_tuple_to_hex(theme_obj.node_color_old)],
            [1.0, _rgb_tuple_to_hex(theme_obj.node_color_new)],
        ]
        marker_line_width = [
            4.0 if graph.nodes[node].get("is_seed") else 0.0 for node in node_ids
        ]
        marker_line_color = [
            _rgb_tuple_to_hex(theme_obj.seed_color)
            if graph.nodes[node].get("is_seed")
            else _rgb_tuple_to_rgba(theme_obj.background, 0.0)
            for node in node_ids
        ]
        marker_showscale = False
        marker_colorbar = None
    else:
        node_labels = [str(label) for label in base_labels]
        text_position = "bottom center"
        text_font = _TextFont(size=8, color=theme_obj.text_color)
        color_scale = "Plasma" if theme_obj.name == "dark" else "Viridis"
        marker_line_width = 2
        marker_line_color = theme_obj.text_color
        marker_showscale = True
        marker_colorbar = _MarkerColorbar(
            thickness=15,
            xanchor="left",
            title=_ColorbarTitle(text="Year", side="right"),
        )
    node_labels = [html.escape(label) for label in node_labels]

    return _NodeStyle(
        labels=node_labels,
        text_position=text_position,
        text_font=text_font,
        color_scale=color_scale,
        marker_line_width=marker_line_width,
        marker_line_color=marker_line_color,
        marker_showscale=marker_showscale,
        marker_colorbar=marker_colorbar,
    )


def _build_hover_texts(*, graph: PaperGraph, node_ids: list[Hashable]) -> list[str]:
    """Build per-node hover tooltips answering "why is this paper here".

    :param PaperGraph graph: Graph supplying node attributes and relation maps.
    :param list[Hashable] node_ids: Deterministically ordered node identifiers.
    :return list[str]: Pseudo-HTML tooltip bodies, one per node.
    """
    seed_relations = _seed_relation_map(graph)
    provenance_map = _provenance_map(graph)
    hover_texts = []
    for node in node_ids:
        attrs = graph.nodes[node]
        serialized = _serialize_node(node, attrs)
        raw_title = " ".join(str(serialized.get("title", node)).split())
        title_html = "<br>".join(
            html.escape(line)
            for line in textwrap.wrap(raw_title, width=58, break_long_words=False)
        ) or html.escape(str(node))
        lines = [f"<b>{title_html}</b>"]
        if attrs.get("paper"):
            serialized_authors = serialized.get("authors", [])
            if not isinstance(serialized_authors, Sequence):
                raise TypeError(
                    f"'{type(serialized_authors).__name__}' object is not subscriptable"
                )
            authors = (
                ", ".join(str(author) for author in serialized_authors[:3]) or "Unknown"
            )
            if len(serialized_authors) > 3:
                authors += f" +{len(serialized_authors) - 3}"
            lines.append(html.escape(authors))
            paper_year = coerce_publication_year(serialized.get("year"))
            # Preserve ``int``'s native failure behavior for raw citation values.
            citation_count = int(serialized.get("citation_count", 0))  # type: ignore[call-overload]
            fact_bits = [
                str(paper_year) if paper_year > 0 else "n.d.",
                f"{citation_count:,} citations",
            ]
            venue = " ".join(str(serialized.get("venue") or "").split())
            if venue:
                fact_bits.append(venue if len(venue) <= 44 else venue[:41] + "...")
            lines.append(html.escape(" | ".join(fact_bits)))
        node_str = str(node)
        relation_label = (
            "seed paper"
            if bool(attrs.get("is_seed", False))
            else _HOVER_RELATION_LABELS.get(
                seed_relations.get(node_str, ""),
                _HOVER_RELATION_LABELS.get(provenance_map.get(node_str, ""), ""),
            )
        )
        if relation_label:
            lines.append(f"<i>{html.escape(relation_label)}</i>")
        hover_texts.append("<br>".join(lines))

    return hover_texts


def _build_dashboard_overlay_traces(
    *,
    go: PlotlyGraphObjects,
    graph: PaperGraph,
    node_ids: list[Hashable],
    node_x: list[float],
    node_y: list[float],
    node_sizes: list[float],
    marker_sizeref: float,
    theme_obj: Theme,
) -> tuple[Scatter, Scatter]:
    """Build the seed selection halo and neighborhood-edge overlay traces.

    :param PlotlyGraphObjects go: Lazily loaded Plotly graph_objects module.
    :param PaperGraph graph: Graph supplying the seed flag.
    :param list[Hashable] node_ids: Deterministically ordered node identifiers.
    :param list[float] node_x: Node x coordinates.
    :param list[float] node_y: Node y coordinates.
    :param list[float] node_sizes: Node marker areas.
    :param float marker_sizeref: Plotly area-mode marker size reference.
    :param Theme theme_obj: Active visualization theme.
    :return tuple[Scatter, Scatter]: Halo trace and empty neighborhood-edge trace.
    """
    seed_index = next(
        (
            idx
            for idx, node in enumerate(node_ids)
            if bool(graph.nodes[node].get("is_seed", False))
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
            sizemin=4,
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
            width=2.0,
            color=_rgb_tuple_to_rgba(theme_obj.seed_color, 0.78),
        ),
        opacity=0.98,
    )

    return halo_trace, neighborhood_trace


def _build_label_annotations(
    *,
    node_x: list[float],
    node_y: list[float],
    node_labels: list[str],
    node_sizes: list[float],
    marker_sizeref: float,
    marker_line_width: int | list[float],
    text_font: _TextFont,
) -> list[_PlotlyAnnotation]:
    """Build labels clear of selection halos and facing into the horizontal span.

    :param list[float] node_x: Node x coordinates.
    :param list[float] node_y: Node y coordinates.
    :param list[str] node_labels: Escaped labels; empty entries are skipped.
    :param list[float] node_sizes: Node marker areas.
    :param float marker_sizeref: Plotly area-mode marker size reference.
    :param int | list[float] marker_line_width: Marker outline widths.
    :param _TextFont text_font: Shared annotation font spec.
    :return list[_PlotlyAnnotation]: Plotly annotation dictionaries.
    """
    x_min = min(node_x, default=0.0)
    x_span = max(node_x, default=0.0) - x_min

    def horizontal_anchor(x: float) -> Literal["left", "center", "right"]:
        # Long title fallbacks near either edge should extend into the graph.
        if x_span > 0:
            if x < x_min + x_span / 3:
                return "left"
            if x > x_min + 2 * x_span / 3:
                return "right"
        return "center"

    return [
        dict(
            x=node_x[idx],
            y=node_y[idx],
            xref="x",
            yref="y",
            text=label,
            font=text_font,
            showarrow=False,
            xanchor=horizontal_anchor(node_x[idx]),
            yanchor="bottom",
            borderpad=0,
            yshift=max(
                4.0,
                math.sqrt(
                    node_sizes[idx]
                    * DASHBOARD_SELECTION_HALO_SCALE
                    / (2.0 * marker_sizeref)
                ),
                max(4.0, math.sqrt(node_sizes[idx] / (2.0 * marker_sizeref)))
                + (
                    marker_line_width[idx]
                    if isinstance(marker_line_width, list)
                    else marker_line_width
                )
                / 2.0,
            )
            + 3.0,
        )
        for idx, label in enumerate(node_labels)
        if label
    ]


def _apply_axis_ranges(
    layout_kwargs: _PlotlyLayoutKwargs,
    node_x: list[float],
    node_y: list[float],
) -> None:
    """Pin explicit padded axis ranges so the dashboard view stays stable.

    :param _PlotlyLayoutKwargs layout_kwargs: Plotly layout kwargs, updated in place.
    :param list[float] node_x: Node x coordinates.
    :param list[float] node_y: Node y coordinates.
    :return None: Mutates ``layout_kwargs``.
    """
    x_min = min(node_x)
    x_max = max(node_x)
    y_min = min(node_y)
    y_max = max(node_y)
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_pad = max(DASHBOARD_AXIS_X_PADDING, x_span * 0.1)
    y_pad = max(DASHBOARD_AXIS_MIN_PADDING, y_span * 0.08)
    layout_kwargs["xaxis"].update(
        {"autorange": False, "range": [x_min - x_pad, x_max + x_pad]}
    )
    layout_kwargs["yaxis"].update(
        {"autorange": False, "range": [y_min - y_pad, y_max + y_pad]}
    )


class PlotlyFigureMixin:
    """Exporter-state Plotly figure construction and deterministic div IDs."""

    def _build_plotly_figure(
        self: _ExporterHost,
        *,
        go: PlotlyGraphObjects,
        theme_obj: Theme,
        title_prefix: str | None = "CiteMesh",
        margin_top: int = 40,
        for_dashboard: bool = False,
    ) -> tuple[Figure, list[Hashable]]:
        """Build a deterministic Plotly figure and point-order mapping.

        :param PlotlyGraphObjects go: Lazily loaded Plotly graph_objects module.
        :param Theme theme_obj: Active visualization theme.
        :param Optional[str] title_prefix: Optional title prefix. When ``None``,
            the figure omits a title.
        :param int margin_top: Top plot margin.
        :param bool for_dashboard: Whether to apply dashboard-specific styling
            (subtle curved edges, muted colors, and reduced label density).
        :return tuple[Figure, list[Hashable]]: Plotly figure and ordered node IDs.
        """
        pos = self._get_layout()
        layout_shapes, edge_trace = _build_edge_layer(
            go=go,
            theme_obj=theme_obj,
            graph=self.graph,
            pos=pos,
            for_dashboard=for_dashboard,
        )

        node_ids = [node_id for node_id, _ in _sorted_nodes(self.graph)]
        metadata_graph = self._serialized_graph()
        node_x = [float(pos[node][0]) for node in node_ids]
        node_y = [float(pos[node][1]) for node in node_ids]
        node_sizes = [max(6, self._node_size(node) / 50) for node in node_ids]
        max_node_size = max(node_sizes) if node_sizes else 1.0
        target_node_diameter = DASHBOARD_MAX_NODE_DIAMETER if for_dashboard else 45.0
        marker_sizeref = max(
            2.0 * max_node_size / (target_node_diameter**2),
            1e-6,
        )
        node_years, year_min, year_max = self._plotly_year_scale(node_ids)
        base_labels = [
            _node_short_label(self.graph.nodes[node], node)
            if self.graph.nodes[node].get("paper")
            else _serialize_node(node, self.graph.nodes[node]).get("title", node)
            for node in node_ids
        ]
        (
            node_labels,
            text_position,
            text_font,
            color_scale,
            marker_line_width,
            marker_line_color,
            marker_showscale,
            marker_colorbar,
        ) = _build_node_style(
            graph=metadata_graph,
            node_ids=node_ids,
            pos=pos,
            base_labels=base_labels,
            theme_obj=theme_obj,
            for_dashboard=for_dashboard,
        )

        hover_texts = _build_hover_texts(graph=metadata_graph, node_ids=node_ids)

        node_hoverlabel = theme_hover_label(theme_obj)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            name="nodes",
            mode="markers" if for_dashboard else "markers+text",
            hoverinfo="text",
            text=node_labels,
            textposition=text_position,
            textfont=text_font,
            marker=dict(
                size=node_sizes,
                sizemode="area",
                sizeref=marker_sizeref,
                sizemin=4 if for_dashboard else 3,
                color=node_years,
                cmin=year_min,
                cmax=year_max,
                colorscale=color_scale,
                line=dict(width=marker_line_width, color=marker_line_color),
                showscale=marker_showscale,
                colorbar=marker_colorbar,
            ),
            hovertext=hover_texts,
            hoverlabel=node_hoverlabel,
        )

        halo_trace: Scatter | None = None
        neighborhood_trace: Scatter | None = None
        if for_dashboard:
            halo_trace, neighborhood_trace = _build_dashboard_overlay_traces(
                go=go,
                graph=metadata_graph,
                node_ids=node_ids,
                node_x=node_x,
                node_y=node_y,
                node_sizes=node_sizes,
                marker_sizeref=marker_sizeref,
                theme_obj=theme_obj,
            )

        layout_kwargs: _PlotlyLayoutKwargs = {
            "showlegend": False,
            "hovermode": "closest",
            "margin": dict(
                b=DASHBOARD_FOOTER_MARGIN if for_dashboard else 20,
                l=5,
                r=5,
                t=max(0, int(margin_top)),
            ),
            "xaxis": dict(showgrid=False, zeroline=False, showticklabels=False),
            "yaxis": dict(showgrid=False, zeroline=False, showticklabels=False),
            "plot_bgcolor": theme_obj.background,
            "paper_bgcolor": theme_obj.background,
            "font": dict(color=theme_obj.text_color),
            # Layout default as well as the per-trace spec: a trace added later
            # (or a hover path Plotly resolves outside the node trace) then
            # still gets the themed card instead of the light default.
            "hoverlabel": node_hoverlabel,
        }
        if for_dashboard:
            # Pixel shifts clear the largest selection halo at every zoom level.
            layout_kwargs["annotations"] = [
                dict(annotation)
                for annotation in _build_label_annotations(
                    node_x=node_x,
                    node_y=node_y,
                    node_labels=node_labels,
                    node_sizes=node_sizes,
                    marker_sizeref=marker_sizeref,
                    marker_line_width=marker_line_width,
                    text_font=text_font,
                )
            ]
        if for_dashboard and node_x and node_y:
            _apply_axis_ranges(layout_kwargs, node_x, node_y)
            # Preserve view state within one result without carrying its viewport
            # into a different seed graph.
            layout_kwargs["uirevision"] = (
                f"citemesh-dashboard-static-layout-v1:{_strategy(self.graph, self.metadata)}:{self.seed_id}"
            )
        if for_dashboard and layout_shapes:
            layout_kwargs["shapes"] = layout_shapes
        if title_prefix is not None:
            layout_kwargs["title"] = (
                f"{title_prefix}: {_plotly_title_text(self.graph, self.seed_id)}"
            )

        if for_dashboard:
            traces: list[Scatter] = []
            if halo_trace is not None:
                traces.append(halo_trace)
            if neighborhood_trace is not None:
                traces.append(neighborhood_trace)
            traces.append(node_trace)
        else:
            traces = [trace for trace in (edge_trace, node_trace) if trace is not None]
        fig = go.Figure(data=traces, layout=go.Layout(**layout_kwargs))
        return fig, node_ids

    def _plotly_div_id(self: _ExporterHost, prefix: str = "citemesh-plotly") -> str:
        """Build a deterministic Plotly HTML container id.

        :param str prefix: Prefix for resulting ``div_id``.
        :return str: Stable ``div_id`` derived from seed id and sorted graph structure.
        """
        nodes = [str(node_id) for node_id, _ in _sorted_nodes(self.graph)]
        edges = [
            (
                str(left),
                str(right),
                round(parse_float(attrs.get("weight", 0.0)), 8),
            )
            for left, right, attrs in _sorted_edges(self.graph)
        ]
        digest_payload = json.dumps(
            {"seed_id": str(self.seed_id), "nodes": nodes, "edges": edges},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha1(digest_payload.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}-{digest}"

    def _plotly_year_scale(
        self: _ExporterHost, node_ids: list[Hashable]
    ) -> tuple[list[float], float, float]:
        """Build deterministic Plotly marker years and explicit scale bounds.

        :param list[Hashable] node_ids: Sorted node identifiers for the current graph.
        :return Tuple[list[float], float, float]: Marker years, color-scale min, and
            color-scale max.
        """
        return publication_year_scale(
            _serialize_node(node, self.graph.nodes[node]).get("year")
            for node in node_ids
        )
