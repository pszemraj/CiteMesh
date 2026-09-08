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
from citemesh.dashboard_contracts import (
    DASHBOARD_COLLECTION_KIND,
    DASHBOARD_COLLECTION_SCHEMA_VERSION,
    GRAPH_PAYLOAD_KIND,
    GRAPH_PAYLOAD_SCHEMA_VERSION,
)
from citemesh.data.cache import atomic_write_text

from .ordering import ordered_edges_with_data, ordered_nodes
from .render import (
    _normalize_layout_positions,
    compute_layout,
    compute_node_colors,
    compute_node_sizes,
)
from .themes import Theme, get_theme
from .years import (
    coerce_publication_year,
    optional_publication_year_bounds,
    publication_year_scale,
)

logger = logging.getLogger(__name__)

GRAPHML_DETERMINISM_POLICY_STRICT = "strict_sorted_nodes_edges"
GRAPHML_DETERMINISM_POLICY_BEST_EFFORT = "best_effort_sorted_nodes_edges"
_GRAPHML_BEST_EFFORT_MIN_VERSION = (2, 8)
GRAPHML_LAYOUT_METADATA_KEY = "citemesh_graphml_determinism"
GRAPHML_LAYOUT_VERSION_KEY = "citemesh_graphml_writer_version"
DASHBOARD_AXIS_MIN_PADDING = 0.14
DASHBOARD_AXIS_X_PADDING = 0.18
DASHBOARD_FOOTER_MARGIN = 78
DASHBOARD_LABEL_CAP = 8
DASHBOARD_LABEL_MIN_DISTANCE = 0.18
DASHBOARD_MAX_NODE_DIAMETER = 58.0
# Identifier fields BibTeX consumers resolve verbatim, so LaTeX escaping them
# would break every machine reader.
_BIBTEX_VERBATIM_FIELDS = frozenset({"doi", "url"})


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


DARKREADER_LOCK_META = '<meta name="darkreader-lock" />'

# Shared UI chrome palette for HTML exports (dashboard vars and Plotly
# hoverlabels must agree so tooltips look native to the page).
_UI_PALETTES: Dict[str, Dict[str, str]] = {
    "dark": {
        "body_bg": "#0f1318",
        "panel_bg": "#171d25",
        "panel_border": "#2e3948",
        "text_primary": "#ecf1f8",
        "text_muted": "#9ab0cb",
        "accent": "#4aa3ff",
        "accent_soft": "rgba(74, 163, 255, 0.2)",
    },
    "light": {
        "body_bg": "#eef2f7",
        "panel_bg": "#ffffff",
        "panel_border": "#d5dce8",
        "text_primary": "#1b2738",
        "text_muted": "#5a6a80",
        "accent": "#0f67d8",
        "accent_soft": "rgba(15, 103, 216, 0.14)",
    },
}


def _theme_color_scheme(theme_obj: Theme) -> str:
    """Resolve the CSS ``color-scheme`` value a theme should declare.

    :param Theme theme_obj: Active visualization theme.
    :return str: ``"dark"`` or ``"light"``.
    """
    return "dark" if theme_obj.name in {"dark", "solarized"} else "light"


# Hover-tooltip relation labels, keyed by seed_relation first and provenance
# as fallback. The tooltip's job is answering "why is this paper here".
_HOVER_RELATION_LABELS: Dict[str, str] = {
    "seed": "seed paper",
    "referenced_by_seed": "referenced by seed",
    "cites_seed": "cites seed",
    "overlap": "prior + derivative work",
    "semantic_only": "semantic match",
    "citation": "citation graph",
    "semantic": "semantic match",
    "both": "citations + semantic match",
}


def _edge_strength_scale(weights: list[float]) -> list[float]:
    """Normalize edge weights to per-graph relative strengths in ``[0, 1]``.

    Raw hybrid edge weights concentrate in a narrow band (typically
    0.55-0.95), so mapping them straight to opacity rendered every edge at a
    visually identical strength. Min-max scaling within the graph makes
    *relative* link strength legible.

    :param list[float] weights: Non-negative raw edge weights.
    :return list[float]: Normalized strengths (all ``0.5`` when weights tie).
    """
    if not weights:
        return []
    w_min = min(weights)
    w_max = max(weights)
    span = w_max - w_min
    if span <= 1e-9:
        return [0.5 for _ in weights]
    return [(weight - w_min) / span for weight in weights]


def _stable_curve_direction(left_id: object, right_id: object) -> float:
    """Return a portable deterministic curve direction for one undirected edge.

    :param object left_id: First edge endpoint.
    :param object right_id: Second edge endpoint.
    :return float: ``1.0`` or ``-1.0`` using the dashboard's 32-bit hash.
    """
    key_left, key_right = sorted((str(left_id), str(right_id)))
    digest = 0
    for character in f"{key_left}|{key_right}":
        digest = ((digest * 33) + ord(character)) & 0xFFFFFFFF
    return 1.0 if digest % 2 == 0 else -1.0


def _select_dashboard_label_nodes(
    graph: nx.Graph,
    node_ids: list[Hashable],
    pos: Dict[Hashable, Iterable[float]],
) -> set[Hashable]:
    """Select prominent dashboard labels without crowding one graph region.

    :param nx.Graph graph: Graph containing node ranking attributes.
    :param list[Hashable] node_ids: Deterministically ordered node identifiers.
    :param Dict[Hashable, Iterable[float]] pos: Normalized node positions.
    :return set[Hashable]: Node identifiers whose labels should remain visible.
    """

    def _rank(node_id: Hashable) -> tuple[int, int, int, str]:
        """Build a stable seed/citation/year priority tuple.

        :param Hashable node_id: Candidate node identifier.
        :return tuple[int, int, int, str]: Sort key for label priority.
        """
        attrs = graph.nodes[node_id]
        try:
            citations = max(int(attrs.get("citation_count", 0) or 0), 0)
        except (TypeError, ValueError):
            citations = 0
        try:
            year = max(int(attrs.get("year", 0) or 0), 0)
        except (TypeError, ValueError):
            year = 0
        return (
            0 if bool(attrs.get("is_seed", False)) else 1,
            -citations,
            -year,
            str(node_id),
        )

    selected: list[Hashable] = []
    for node_id in sorted(node_ids, key=_rank):
        coords = tuple(float(value) for value in pos[node_id])
        if not bool(graph.nodes[node_id].get("is_seed", False)) and any(
            math.dist(coords, tuple(float(value) for value in pos[other]))
            < DASHBOARD_LABEL_MIN_DISTANCE
            for other in selected
        ):
            continue
        selected.append(node_id)
        if len(selected) >= DASHBOARD_LABEL_CAP:
            break
    return set(selected)


# Characters XML 1.0 forbids even when escaped: C0 controls other than
# tab/newline/CR, lone surrogates, and the two non-characters U+FFFE/U+FFFF.
_XML_INVALID_CHARS_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff" + chr(0xFFFE) + chr(0xFFFF) + "]"
)


def _xml_safe_graph_value(value: object) -> object:
    """Normalize nullable attributes and XML-invalid text bound for GraphML.

    Upstream titles/abstracts occasionally carry stray control bytes;
    ``nx.write_graphml`` passes them through and produces a file no XML
    parser will accept.

    :param object value: Raw attribute value.
    :return object: Empty text for nulls, otherwise an XML-safe attribute value.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return _XML_INVALID_CHARS_RE.sub("", value)
    return value


_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_cell_guard(value: object) -> str:
    """Neutralize spreadsheet formula interpretation for one CSV text cell.

    Excel/Sheets execute cells starting with ``=``, ``+``, ``-``, ``@``, tab,
    or CR as formulas (CWE-1236). Prefixing an apostrophe forces text
    rendering; the dashboard's in-page CSV exporter applies the same rule.

    :param object value: Raw text cell value (``None`` renders empty).
    :return str: Cell text, apostrophe-prefixed when formula-leading.
    """
    text = "" if value is None else str(value)
    if text.startswith(_CSV_FORMULA_PREFIXES):
        return f"'{text}"
    return text


def _inject_darkreader_lock(path: Path | str, color_scheme: str = "light") -> None:
    """Insert dark-mode-extension defenses into a written HTML export.

    CiteMesh HTML exports ship their own tuned themes; auto-darkening
    re-theming breaks them outright (Dark Reader paints Plotly's transparent
    overlay SVGs with an opaque background, hiding the entire graph). Two
    signals are injected: the Dark Reader-specific ``darkreader-lock`` opt-out
    meta, and the standards-based ``color-scheme`` meta that Chrome's Auto
    Dark Mode and well-behaved extensions consult before repainting a page.

    :param Path | str path: HTML file to rewrite in place (no-op when it has no
        ``<head>`` tag or already carries the lock).
    :param str color_scheme: Declared scheme for the export, ``dark`` or
        ``light``.
    :return None: Rewrites the file in place.
    """
    resolved_path = Path(path)
    try:
        content = resolved_path.read_text(encoding="utf-8")
    except OSError:
        return
    if "darkreader-lock" in content or "<head>" not in content:
        return
    scheme = color_scheme if color_scheme in {"dark", "light"} else "light"
    scheme_meta = f'<meta name="color-scheme" content="{scheme}" />'
    resolved_path.write_text(
        content.replace("<head>", f"<head>{DARKREADER_LOCK_META}{scheme_meta}", 1),
        encoding="utf-8",
    )


class GraphExporter:
    """Unified interface for exporting graphs in multiple formats."""

    def __init__(
        self,
        graph: nx.Graph,
        seed_id: str,
        metadata: Optional[Dict] = None,
        theme_name: str = "dark",
        layout: Optional[Dict[Hashable, Iterable[float]]] = None,
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

    def graph_payload(self) -> Dict[str, Any]:
        """Build the canonical versioned graph payload shared by JSON consumers.

        Dashboard geometry is always embedded (computing a layout on demand when
        the caller did not supply one) so every ``kind``-stamped payload can be
        loaded back through the dashboard's Load Results flow regardless of
        which export formats were requested or in which order exporters ran.

        :return Dict[str, Any]: Portable CiteMesh graph payload with dashboard data.
        """
        enriched = self._enriched_nodes()
        sorted_edges = self._sorted_edges()
        dashboard_node_ids = [node_id for node_id, _ in self._sorted_nodes()]
        dashboard_meta = self._dashboard_meta(
            theme_obj=self.theme,
            node_ids=dashboard_node_ids,
            node_payloads=enriched,
            sorted_edges=sorted_edges,
            include_plotly_geometry=True,
        )
        portable_meta: Dict[str, Any] = {
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
                    "source_title": self._node_title(self.graph.nodes[u], u),
                    "target_title": self._node_title(self.graph.nodes[v], v),
                    "source_label": self._node_short_label(self.graph.nodes[u], u),
                    "target_label": self._node_short_label(self.graph.nodes[v], v),
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
            json.dumps(self.graph_payload(), sort_keys=True, indent=2),
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
            export_graph.graph[graph_key] = _xml_safe_graph_value(
                self._graphml_metadata_value(self.metadata[metadata_key])
            )

        for node, attrs in sorted_nodes:
            cleaned = self._serialize_node(node, attrs)
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
        _inject_darkreader_lock(path, _theme_color_scheme(theme_obj))

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
        _inject_darkreader_lock(path, _theme_color_scheme(theme_obj))

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
        payload_json = self._script_safe_json(payload)
        figure_json = self._script_safe_json(fig.to_plotly_json())
        collection_json = self._script_safe_json(self._dashboard_collection_bundle())
        html_output = self._dashboard_template(
            theme_obj=theme_obj,
            div_id=div_id,
            plotly_js=self._safe_script_content(get_plotlyjs()),
            payload_json=payload_json,
            figure_json=figure_json,
            collection_json=collection_json,
        )
        atomic_write_text(path, html_output)

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
            edge_records = list(self._sorted_edges())
            edge_strengths = _edge_strength_scale(
                [
                    max(float(attrs.get("weight", 0.0)), 0.0)
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
        target_node_diameter = DASHBOARD_MAX_NODE_DIAMETER if for_dashboard else 45.0
        marker_sizeref = max(
            2.0 * max_node_size / (target_node_diameter**2),
            1e-6,
        )
        node_years, year_min, year_max = self._plotly_year_scale(node_ids)
        base_labels = [
            self.graph.nodes[node].get("paper").label
            if self.graph.nodes[node].get("paper")
            else self.graph.nodes[node].get("title", node)
            for node in node_ids
        ]
        if for_dashboard:
            label_nodes = _select_dashboard_label_nodes(self.graph, node_ids, pos)
            node_labels = [
                str(base_labels[idx]) if node_id in label_nodes else ""
                for idx, node_id in enumerate(node_ids)
            ]
            text_position = "top center"
            # Keep one marker trace so point indices stay stable for hover/click sync;
            # use a muted shared text alpha instead of per-point text styling.
            text_font = dict(
                size=10, color=_rgb_tuple_to_rgba(theme_obj.text_color, 0.72)
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
        node_labels = [html.escape(label) for label in node_labels]

        seed_relations = self._seed_relation_map()
        provenance_map = self._provenance_map()
        hover_texts = []
        for node in node_ids:
            attrs = self.graph.nodes[node]
            paper: Optional[Paper] = attrs.get("paper")
            raw_title = " ".join(
                str(paper.title if paper else attrs.get("title", node)).split()
            )
            title_html = "<br>".join(
                html.escape(line)
                for line in textwrap.wrap(raw_title, width=58, break_long_words=False)
            ) or html.escape(str(node))
            lines = [f"<b>{title_html}</b>"]
            if paper:
                authors = ", ".join(a.name for a in paper.authors[:3]) or "Unknown"
                if len(paper.authors) > 3:
                    authors += f" +{len(paper.authors) - 3}"
                lines.append(html.escape(authors))
                paper_year = coerce_publication_year(paper.year)
                fact_bits = [
                    str(paper_year) if paper_year > 0 else "n.d.",
                    f"{paper.citation_count:,} citations",
                ]
                venue = " ".join(str(attrs.get("venue") or "").split())
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

        hover_palette = _UI_PALETTES[_theme_color_scheme(theme_obj)]
        node_hoverlabel = dict(
            bgcolor=hover_palette["panel_bg"],
            bordercolor=hover_palette["panel_border"],
            font=dict(color=hover_palette["text_primary"], size=12),
            align="left",
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

        layout_kwargs: Dict[str, Any] = {
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
        }
        if for_dashboard and node_x and node_y:
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
            # Preserve view state within one result without carrying its viewport
            # into a different seed graph.
            layout_kwargs["uirevision"] = (
                f"citemesh-dashboard-static-layout-v1:{self._strategy()}:{self.seed_id}"
            )
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
        seed_attrs = (
            self.graph.nodes[self.seed_id] if self.seed_id in self.graph else {}
        )
        # Plotly renders titles as pseudo-HTML, so upstream markup must be escaped
        # before wrapping; only the "<br>" joins below stay live markup.
        raw_title = html.escape(
            " ".join(str(seed_attrs.get("title", "CiteMesh")).split())
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
        strategy = self._strategy()

        node_payloads: list[Dict[str, Any]] = []
        for node_id, attrs in sorted_nodes:
            node_str = str(node_id)
            serialized = self._serialize_node(node_id, attrs)
            serialized["id"] = node_str
            serialized["year"] = coerce_publication_year(serialized.get("year"))
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

    def _dashboard_meta(
        self,
        *,
        theme_obj: Theme,
        node_ids: list[Hashable],
        node_payloads: list[Dict[str, Any]],
        sorted_edges: list[tuple[Hashable, Hashable, Dict[str, Any]]],
        include_plotly_geometry: bool,
    ) -> Dict[str, Any]:
        """Build dashboard metadata shared by dashboard HTML and JSON exports.

        :param Theme theme_obj: Active visualization theme.
        :param list[Hashable] node_ids: Node order used by Plotly points.
        :param list[Dict[str, Any]] node_payloads: Enriched node payloads.
        :param list[tuple[Hashable, Hashable, Dict[str, Any]]] sorted_edges:
            Deterministically ordered edge payloads.
        :param bool include_plotly_geometry: Whether layout-backed Plotly geometry
            should be embedded in the metadata.
        :return Dict[str, Any]: Dashboard metadata payload.
        """
        # The dashboard import contract requires a non-empty strategy token;
        # graphs built outside the CLI/builders may carry none.
        strategy = self._strategy() or "unknown"
        # Null rather than the color-scale sentinel: the timeline renders "-" for
        # a missing range and would otherwise show years no paper carries.
        bounds = optional_publication_year_bounds(
            node.get("year") for node in node_payloads
        )
        year_range = {"min": bounds[0], "max": bounds[1]} if bounds else None

        meta: Dict[str, Any] = {
            "seed_id": str(self.seed_id),
            "strategy": strategy,
            "theme": theme_obj.name,
            "summary": {
                "nodes": len(node_payloads),
                "edges": len(sorted_edges),
            },
            "year_range": year_range,
        }
        raw_source_status = self.metadata.get("candidate_source_status")
        if isinstance(raw_source_status, dict):
            meta["candidate_source_status"] = {
                str(source): str(status)
                for source, status in sorted(
                    raw_source_status.items(), key=lambda item: str(item[0])
                )
            }
        if include_plotly_geometry:
            positions = self._get_layout()
            meta["plotly_node_order"] = [str(node_id) for node_id in node_ids]
            meta["plotly_positions"] = [
                [float(positions[node_id][0]), float(positions[node_id][1])]
                for node_id in node_ids
            ]
            meta["plotly_node_sizes"] = [
                max(6.0, self._node_size(node_id) / 50.0) for node_id in node_ids
            ]
        return meta

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
        payload: Dict[str, Any] = {
            "meta": self._dashboard_meta(
                theme_obj=theme_obj,
                node_ids=node_ids,
                node_payloads=node_payloads,
                sorted_edges=sorted_edges,
                include_plotly_geometry=True,
            ),
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
        empty_bundle: Dict[str, Any] = {
            "kind": DASHBOARD_COLLECTION_KIND,
            "schema_version": DASHBOARD_COLLECTION_SCHEMA_VERSION,
            "current_result_id": None,
            "results": [],
        }
        raw_bundle = self.metadata.get("dashboard_collection")
        if not isinstance(raw_bundle, dict):
            return empty_bundle

        declared_kind = str(raw_bundle.get("kind") or "").strip()
        raw_results = raw_bundle.get("results")
        raw_payloads = raw_bundle.get("payloads")
        legacy_payloads = raw_payloads if isinstance(raw_payloads, dict) else {}
        results: list[Dict[str, Any]] = []
        if isinstance(raw_results, list):
            for raw_entry in raw_results:
                if not isinstance(raw_entry, dict):
                    continue
                result_id = str(raw_entry.get("result_id") or "").strip()
                payload = raw_entry.get("payload")
                if not isinstance(payload, dict):
                    payload = legacy_payloads.get(result_id)
                if not result_id or not isinstance(payload, dict):
                    continue
                entry: Dict[str, Any] = {
                    key: raw_entry[key]
                    for key in (
                        "result_id",
                        "seed_id",
                        "title",
                        "strategy",
                        "summary",
                        "updated_at",
                    )
                    if key in raw_entry
                }
                entry["result_id"] = result_id
                entry["payload"] = payload
                build = raw_entry.get("build")
                if isinstance(build, dict):
                    entry["build"] = build
                elif declared_kind == DASHBOARD_COLLECTION_KIND:
                    # Versioned entries must carry build metadata; the viewer
                    # rejects the whole bundle when the key is missing.
                    entry["build"] = {}
                results.append(entry)
        raw_current_result_id = raw_bundle.get("current_result_id")
        current_result_id = (
            str(raw_current_result_id).strip()
            if raw_current_result_id is not None
            else None
        )
        if current_result_id not in {entry["result_id"] for entry in results}:
            # Malformed entries are dropped above, and the viewer rejects a bundle
            # whose current_result_id names no included result.
            current_result_id = results[0]["result_id"] if results else None
        bundle: Dict[str, Any] = {
            "current_result_id": current_result_id,
            "results": results,
        }
        if declared_kind:
            if (
                declared_kind != DASHBOARD_COLLECTION_KIND
                or raw_bundle.get("schema_version")
                != DASHBOARD_COLLECTION_SCHEMA_VERSION
            ):
                raise ValueError(
                    "Unsupported dashboard collection metadata kind or schema version."
                )
            bundle.update(
                {
                    "kind": DASHBOARD_COLLECTION_KIND,
                    "schema_version": DASHBOARD_COLLECTION_SCHEMA_VERSION,
                }
            )
        return bundle

    def _default_provenance(self, *, strategy: str) -> str:
        """Resolve default provenance class for non-hybrid strategies.

        :param str strategy: Strategy metadata token.
        :return str: One of ``citation`` or ``semantic``.
        """
        if strategy in {"embedding", "recommendation"}:
            return "semantic"
        return "citation"

    @staticmethod
    def _normalize_strategy_token(raw_strategy: object) -> str:
        """Normalize strategy tokens to the exporter-supported vocabulary.

        :param object raw_strategy: Candidate strategy token.
        :return str: Normalized strategy token or an empty string.
        """
        normalized = str(raw_strategy or "").strip().lower()
        if normalized in {"citation", "recommendation", "embedding", "hybrid"}:
            return normalized
        return ""

    def _strategy(self) -> str:
        """Resolve effective strategy token for export metadata and enrichment.

        Explicit exporter metadata wins. When omitted, exporter falls back to graph
        metadata persisted by current builders.

        :return str: Normalized strategy token when available.
        """
        metadata_strategy = self._normalize_strategy_token(
            self.metadata.get("strategy")
        )
        if metadata_strategy:
            return metadata_strategy

        graph_strategy = self._normalize_strategy_token(
            self.graph.graph.get("strategy")
        )
        if graph_strategy:
            return graph_strategy
        return ""

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
        """Escape script-closing tokens in trusted inline script bodies.

        Only suitable for trusted library code (e.g. the bundled plotly.js).
        User-controlled data must go through :meth:`_script_safe_json`, which
        removes every ``<`` so the HTML tokenizer can never enter the
        script-data-escaped states (``<!--`` + ``<script``) that would swallow
        the closing ``</script>`` tag.

        :param str raw: Raw script body content.
        :return str: Script-safe content.
        """
        return raw.replace("</", "<\\/")

    @staticmethod
    def _script_safe_json(payload: Any) -> str:
        """Serialize a payload as JSON that is inert inside an HTML ``<script>``.

        ``json.dumps`` leaves ``<`` unescaped, so upstream text such as
        ``<!--<script>`` in a paper abstract would otherwise drive the HTML
        tokenizer into the script-data-double-escaped state and break the whole
        document. ``<`` can only occur inside JSON string literals, so the
        global ``\\u003c`` rewrite is loss-free for ``JSON.parse``.

        :param Any payload: JSON-serializable payload.
        :return str: Compact deterministic JSON with every ``<`` escaped.
        """
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).replace("<", "\\u003c")

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

        doi_value = self._derive_doi_value(node_id, node_payload=node_payload)
        if doi_value:
            links["doi"] = f"https://doi.org/{quote(doi_value, safe='/()[]:._;-')}"
        return links

    @staticmethod
    def _derive_doi_value(
        node_id: str,
        *,
        node_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Resolve the raw DOI of a node from metadata or its canonical ID.

        Callers that build URLs percent-encode the result themselves; BibTeX and
        other identifier consumers need this unencoded form.

        :param str node_id: Canonical graph node identifier.
        :param Optional[Dict[str, Any]] node_payload: Optional node payload carrying
            an explicit ``doi`` value.
        :return str: Raw DOI without prefix or encoding, empty when unknown.
        """
        doi_value = ""
        if isinstance(node_payload, dict):
            doi_value = str(node_payload.get("doi") or "").strip()
        if not doi_value:
            if node_id.lower().startswith("doi:"):
                doi_value = node_id.split(":", 1)[1].strip()
            elif re.match(r"^10\.\d{4,9}/\S+$", node_id):
                doi_value = node_id
        return doi_value

    @staticmethod
    def _bibtex_entry_key(node_id: str) -> str:
        """Build deterministic BibTeX entry keys from node IDs.

        :param str node_id: Graph node ID.
        :return str: Readable node slug plus a stable identifier-derived suffix.
        """
        normalized = re.sub(r"[^0-9a-zA-Z]+", "_", node_id).strip("_").lower()
        if not normalized:
            normalized = "paper"
        suffix = hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:12]
        return f"citemesh_{normalized}_{suffix}"

    @staticmethod
    def _bibtex_escape(raw_value: str) -> str:
        """Escape text for conservative BibTeX field rendering.

        Handles every LaTeX special: ``{ } % & # $ _`` gain a backslash,
        ``~``/``^`` use their text-mode commands, and a literal backslash
        becomes ``\\textbackslash{}`` (``\\\\`` would typeset a line break).
        Backslashes are staged through a sentinel first so the escapes this
        method itself emits are not re-escaped.

        :param str raw_value: Raw field value.
        :return str: Escaped value safe for brace-delimited fields.
        """
        collapsed = " ".join(str(raw_value).split())
        sentinel = "\x00"
        collapsed = collapsed.replace("\\", sentinel)
        collapsed = collapsed.replace("{", "\\{")
        collapsed = collapsed.replace("}", "\\}")
        for special in ("%", "&", "#", "$", "_"):
            collapsed = collapsed.replace(special, f"\\{special}")
        collapsed = collapsed.replace("~", "\\textasciitilde{}")
        collapsed = collapsed.replace("^", "\\textasciicircum{}")
        return collapsed.replace(sentinel, "\\textbackslash{}")

    @staticmethod
    def _bibtex_verbatim(raw_value: str) -> str:
        """Render an identifier field without LaTeX escaping.

        ``doi`` and ``url`` are consumed by machines, so escaping ``_`` or ``%``
        would corrupt them. They stay literal; only whitespace and the characters
        that would unbalance the surrounding braces are removed.

        :param str raw_value: Raw identifier value.
        :return str: Value safe to place inside a brace-delimited field.
        """
        collapsed = " ".join(str(raw_value).split())
        return re.sub(r"[{}\\]", "", collapsed)

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

        year = coerce_publication_year(node_payload.get("year"))
        if year > 0:
            fields.append(("year", str(year)))

        doi_value = self._derive_doi_value(
            str(node_payload.get("id", "")), node_payload=node_payload
        )
        if doi_value:
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
            rendered = (
                self._bibtex_verbatim(value)
                if field in _BIBTEX_VERBATIM_FIELDS
                else self._bibtex_escape(value)
            )
            lines.append(f"  {field} = {{{rendered}}},")
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
        color_scheme = _theme_color_scheme(theme_obj)
        palette = _UI_PALETTES[color_scheme]
        vars_map = {
            "__COLOR_SCHEME__": color_scheme,
            "__BODY_BG__": palette["body_bg"],
            "__PANEL_BG__": palette["panel_bg"],
            "__PANEL_BORDER__": palette["panel_border"],
            "__TEXT_PRIMARY__": palette["text_primary"],
            "__TEXT_MUTED__": palette["text_muted"],
            "__ACCENT__": palette["accent"],
            "__ACCENT_SOFT__": palette["accent_soft"],
            "__GRAPH_BG__": theme_obj.background,
            "__NODE_COLOR_OLD__": _rgb_tuple_to_hex(theme_obj.node_color_old),
            "__NODE_COLOR_NEW__": _rgb_tuple_to_hex(theme_obj.node_color_new),
            "__SEED_RING__": _rgb_tuple_to_hex(theme_obj.seed_color),
            "__DASHBOARD_EDGE_COLOR__": _rgb_tuple_to_hex(theme_obj.edge_color),
            "__DASHBOARD_AXIS_MIN_PADDING__": str(DASHBOARD_AXIS_MIN_PADDING),
            "__DASHBOARD_AXIS_X_PADDING__": str(DASHBOARD_AXIS_X_PADDING),
            "__DASHBOARD_LABEL_CAP__": str(DASHBOARD_LABEL_CAP),
            "__DASHBOARD_LABEL_MIN_DISTANCE__": str(DASHBOARD_LABEL_MIN_DISTANCE),
            "__DASHBOARD_MAX_NODE_DIAMETER__": str(DASHBOARD_MAX_NODE_DIAMETER),
            "__GRAPH_PAYLOAD_KIND_JSON__": json.dumps(GRAPH_PAYLOAD_KIND),
            "__GRAPH_PAYLOAD_SCHEMA_VERSION__": str(GRAPH_PAYLOAD_SCHEMA_VERSION),
            "__COLLECTION_KIND_JSON__": json.dumps(DASHBOARD_COLLECTION_KIND),
            "__COLLECTION_SCHEMA_VERSION__": str(DASHBOARD_COLLECTION_SCHEMA_VERSION),
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
  <meta name="darkreader-lock" />
  <meta name="color-scheme" content="__COLOR_SCHEME__" />
  <title>CiteMesh Dashboard</title>
  <style>
    :root {
      color-scheme: __COLOR_SCHEME__;
      --body-bg: __BODY_BG__;
      --panel-bg: __PANEL_BG__;
      --panel-border: __PANEL_BORDER__;
      --text-primary: __TEXT_PRIMARY__;
      --text-muted: __TEXT_MUTED__;
      --accent: __ACCENT__;
      --accent-soft: __ACCENT_SOFT__;
      --graph-bg: __GRAPH_BG__;
      --shadow-soft: rgba(0, 0, 0, 0.18);
      --seed-ring: __SEED_RING__;
    }
    * { box-sizing: border-box; }
    /* Plotly overlay SVGs must stay transparent; dark-mode extensions that
       repaint them opaque would otherwise hide the whole graph. */
    .js-plotly-plot svg.main-svg { background: transparent !important; }
    /* Plotly injects low-opacity icon fills that are too dim on dark panels. */
    .js-plotly-plot .modebar-btn path {
      fill: var(--text-muted) !important;
    }
    .js-plotly-plot .modebar-btn:hover path,
    .js-plotly-plot .modebar-btn.active path {
      fill: var(--text-primary) !important;
    }
    .js-plotly-plot .modebar-btn:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 1px;
    }
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
      grid-template-columns: 140px 140px 1fr auto;
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
    #saved-filter {
      justify-self: start;
      width: fit-content;
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
      grid-template-columns: 1fr auto auto;
      gap: 8px;
      align-items: baseline;
    }
    .star-btn {
      background: none;
      border: none;
      padding: 0 2px;
      font-size: 15px;
      line-height: 1;
      color: var(--text-muted);
      cursor: pointer;
    }
    .star-btn:hover {
      background: none;
      border: none;
      transform: none;
      color: #f5c451;
    }
    .star-btn.saved { color: #f5c451; }
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
    #graph-pane .pane-header .muted {
      max-width: 72%;
      font-size: 12px;
      line-height: 1.25;
      text-align: right;
    }
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
      filter: drop-shadow(0 0 10px color-mix(in srgb, var(--seed-ring) 85%, transparent)) brightness(1.14);
    }
    .js-plotly-plot .scatterlayer path.point.is-neighbor {
      opacity: 0.74 !important;
    }
    .js-plotly-plot .scatterlayer path.point.is-dimmed {
      opacity: 0.18 !important;
    }
    .js-plotly-plot .scatterlayer path.point.is-filter-hidden {
      opacity: 0.12 !important;
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
      background: color-mix(in srgb, var(--panel-bg) 82%, transparent);
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
      background: color-mix(in srgb, var(--seed-ring) 28%, transparent);
    }
    .legend-gradient {
      width: 36px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
      border: 1px solid rgba(255, 255, 255, 0.32);
      background: linear-gradient(90deg, __NODE_COLOR_OLD__, __NODE_COLOR_NEW__);
    }
    #year-timeline {
      display: inline-grid;
      grid-template-columns: auto minmax(190px, 240px) auto;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      color: color-mix(in srgb, var(--text-muted) 92%, #d8e3f2);
      background: color-mix(in srgb, var(--panel-bg) 82%, transparent);
      border: 1px solid color-mix(in srgb, var(--panel-border) 70%, transparent);
      border-radius: 10px;
      padding: 7px 9px;
      backdrop-filter: blur(6px);
    }
    #timeline-bar {
      height: 10px;
      border-radius: 999px;
      border: 1px solid color-mix(in srgb, var(--panel-border) 80%, transparent);
      /* Must match the node colorscale so the timeline doubles as the color legend. */
      background: linear-gradient(90deg, __NODE_COLOR_OLD__ 0%, __NODE_COLOR_NEW__ 100%);
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
      min-height: 160px;
      flex: 1 0 160px;
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
      .toolbar-row.secondary { grid-template-columns: 140px 140px 1fr auto; }
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
      #detail-pane { grid-area: detail; min-height: 620px; }
      #paper-list-pane { grid-area: list; min-height: 360px; }
      #__PLOTLY_DIV_ID__ { min-height: 500px; }
    }
    @media (max-width: 640px) {
      #dashboard-toolbar { position: static; }
      .toolbar-row.primary,
      .toolbar-row.secondary {
        grid-template-columns: minmax(0, 1fr);
      }
      .toolbar-row.primary #search-input,
      #provenance-filters {
        grid-column: auto;
      }
    }
  </style>
</head>
<body>
  <header id="dashboard-toolbar" class="collapsed">
    <div id="global-nav">
      <div id="scope-nav" class="nav-group">
        <button class="nav-btn" data-scope="prior" type="button">Prior works</button>
        <button class="nav-btn" data-scope="derivative" type="button">Derivative works</button>
      </div>
      <div class="nav-group">
        <button id="list-view-btn" class="nav-btn active" type="button">List view</button>
        <button id="filters-toggle" class="nav-btn" type="button">Filters</button>
        <button id="more-btn" class="nav-btn" type="button">More</button>
      </div>
      <div class="nav-group">
        <button id="export-json-btn" class="nav-btn" type="button">Export JSON</button>
        <button id="export-csv-btn" class="nav-btn" type="button">Export CSV</button>
        <button id="export-bib-btn" class="nav-btn" type="button">All BibTeX</button>
        <button id="export-saved-bib-btn" class="nav-btn" type="button" style="display:none">Saved BibTeX</button>
        <button id="copy-saved-links-btn" class="nav-btn" type="button" style="display:none" title="Copy a markdown list of saved papers with links">Copy Saved Links</button>
        <button id="export-collection-btn" class="nav-btn" type="button">Export Collection</button>
        <button id="add-results-btn" class="nav-btn" type="button">Add Results…</button>
        <input id="add-results-input" type="file" accept=".json,.html" multiple style="display:none" />
      </div>
      <div class="nav-group">
        <label class="visually-hidden" for="result-select">Graph in result set</label>
        <select id="result-select" title="Switch graph in result set">
          <option value="" disabled>Current graph</option>
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
        <button id="saved-filter" class="chip" type="button" title="Show only papers saved to your reading list">Saved</button>
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
            <span class="legend-item"><span class="legend-gradient"></span>older &#8594; newer</span>
            <span class="legend-item muted">size = citations</span>
          </div>
          <div id="year-timeline" title="Node color encodes publication year">
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
    const GRAPH_PAYLOAD_KIND = __GRAPH_PAYLOAD_KIND_JSON__;
    const GRAPH_PAYLOAD_SCHEMA_VERSION = __GRAPH_PAYLOAD_SCHEMA_VERSION__;
    const COLLECTION_KIND = __COLLECTION_KIND_JSON__;
    const COLLECTION_SCHEMA_VERSION = __COLLECTION_SCHEMA_VERSION__;
    let payload = JSON.parse(document.getElementById("citemesh-dashboard-data").textContent);
    const baseFigureTemplate = JSON.parse(document.getElementById("citemesh-dashboard-figure").textContent);
    let figureSpec = JSON.parse(document.getElementById("citemesh-dashboard-figure").textContent);
    // Embed the collection bundle directly in the shell so saved-result browsing
    // still works when the dashboard is opened from the local filesystem.
    const embeddedCollectionBundle = JSON.parse(
      document.getElementById("citemesh-dashboard-collection").textContent
    );
    let collectionBundle = emptyCollectionPackage();
    let initialEntry = null;
    let bootstrapFailureMessage = "";
    let bootstrapStatusMessage = "";
    try {
      initialEntry = collectionEntryFromGraphPayload(payload, "Current graph", true);
    } catch (err) {
      bootstrapFailureMessage =
        "Dashboard graph data is invalid: " + String((err && err.message) || err);
    }
    if (initialEntry) {
      try {
        collectionBundle = normalizeCollectionPackage(
          embeddedCollectionBundle,
          "this dashboard",
          true
        );
      } catch (err) {
        bootstrapStatusMessage =
          "Embedded graph collection was ignored: " + String((err && err.message) || err);
        collectionBundle = emptyCollectionPackage();
      }
      if (!collectionBundle.results.some((entry) => entry.result_id === initialEntry.result_id)) {
        collectionBundle.results.unshift(initialEntry);
      }
      if (!collectionBundle.current_result_id) {
        collectionBundle.current_result_id = initialEntry.result_id;
      }
    }
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
      const [keyLeft, keyRight] = [String(leftId || ""), String(rightId || "")].sort();
      return stableHash(`${keyLeft}|${keyRight}`) % 2 === 0 ? 1 : -1;
    }

    function selectDashboardLabelIds(order, nodeById, xPairs, yPairs) {
      const rankedIds = order.slice().sort((leftId, rightId) => {
        const left = nodeById.get(leftId) || {};
        const right = nodeById.get(rightId) || {};
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
        return String(leftId).localeCompare(String(rightId));
      });
      const indexById = new Map(order.map((nodeId, idx) => [nodeId, idx]));
      const selectedIds = [];
      for (const nodeId of rankedIds) {
        const node = nodeById.get(nodeId) || {};
        const nodeIdx = indexById.get(nodeId);
        const isCrowded = !node.is_seed && selectedIds.some((selectedId) => {
          const selectedIdx = indexById.get(selectedId);
          return Math.hypot(
            Number(xPairs[nodeIdx] || 0) - Number(xPairs[selectedIdx] || 0),
            Number(yPairs[nodeIdx] || 0) - Number(yPairs[selectedIdx] || 0)
          ) < __DASHBOARD_LABEL_MIN_DISTANCE__;
        });
        if (isCrowded) {
          continue;
        }
        selectedIds.push(nodeId);
        if (selectedIds.length >= __DASHBOARD_LABEL_CAP__) {
          break;
        }
      }
      return new Set(selectedIds);
    }

    function dashboardNodeLabel(node, nodeId) {
      const authors = Array.isArray(node.authors) ? node.authors : [];
      const firstAuthor = String(authors[0] || "").trim();
      if (firstAuthor) {
        const surname = firstAuthor.split(/\\s+/).pop();
        return escapeHtml(`${surname}, ${node.year || "n.d."}`);
      }
      const title = String(node.title || nodeId || "Unknown");
      return escapeHtml(title.length <= 26 ? title : `${title.slice(0, 23)}...`);
    }

    function currentSeedRingColor() {
      const styles = getComputedStyle(document.documentElement);
      const color = String(styles.getPropertyValue("--seed-ring") || "").trim();
      return color || "__SEED_RING__";
    }

    function colorWithAlpha(hexColor, alpha) {
      const match = /^#([0-9a-f]{6})$/i.exec(String(hexColor || "").trim());
      const clampedAlpha = Math.max(0, Math.min(1, Number(alpha) || 0));
      if (!match) {
        return `rgba(255,255,255,${clampedAlpha.toFixed(3)})`;
      }
      const value = match[1];
      const red = parseInt(value.slice(0, 2), 16);
      const green = parseInt(value.slice(2, 4), 16);
      const blue = parseInt(value.slice(4, 6), 16);
      return `rgba(${red},${green},${blue},${clampedAlpha.toFixed(3)})`;
    }

    function normalizeDashboardEdgeStrengths(edges) {
      const weights = edges.map((edge) =>
        Math.max(safeFiniteNumber(edge && edge.weight, 0), 0)
      );
      if (!weights.length) {
        return [];
      }
      const minimum = Math.min(...weights);
      const maximum = Math.max(...weights);
      const span = maximum - minimum;
      if (span <= 1e-9) {
        return weights.map(() => 0.5);
      }
      return weights.map((weight) => (weight - minimum) / span);
    }

    function wrapDashboardHoverTitle(value, width) {
      const words = String(value || "").trim().split(/\\s+/).filter(Boolean);
      const lines = [];
      for (const word of words) {
        const current = lines.length ? lines[lines.length - 1] : "";
        if (!current || current.length + 1 + word.length > width) {
          lines.push(word);
        } else {
          lines[lines.length - 1] = `${current} ${word}`;
        }
      }
      return lines;
    }

    function dashboardHoverText(node, nodeId) {
      const titleLines = wrapDashboardHoverTitle(node.title || nodeId, 58);
      const titleHtml = titleLines.map(escapeHtml).join("<br>") || escapeHtml(nodeId);
      const lines = [`<b>${titleHtml}</b>`];
      const authorNames = Array.isArray(node.authors)
        ? node.authors.map((author) => String(author || "").trim()).filter(Boolean)
        : [];
      let authors = authorNames.slice(0, 3).join(", ") || "Unknown";
      if (authorNames.length > 3) {
        authors += ` +${authorNames.length - 3}`;
      }
      lines.push(escapeHtml(authors));

      const citationCount = Math.max(
        Math.trunc(safeFiniteNumber(node.citation_count, 0)),
        0
      );
      const factBits = [
        hasYear(node) ? String(Number(node.year)) : "n.d.",
        `${citationCount.toLocaleString("en-US")} citations`,
      ];
      const venue = String(node.venue || "").trim().split(/\\s+/).filter(Boolean).join(" ");
      if (venue) {
        factBits.push(venue.length <= 44 ? venue : `${venue.slice(0, 41)}...`);
      }
      lines.push(escapeHtml(factBits.join(" | ")));

      const relationLabels = {
        seed: "seed paper",
        referenced_by_seed: "referenced by seed",
        cites_seed: "cites seed",
        overlap: "prior + derivative work",
        semantic_only: "semantic match",
        citation: "citation graph",
        semantic: "semantic match",
        both: "citations + semantic match",
      };
      const relationKey = node.is_seed
        ? "seed"
        : String(node.seed_relation || node.provenance || "");
      const relationLabel = relationLabels[relationKey] || "";
      if (relationLabel) {
        lines.push(`<i>${escapeHtml(relationLabel)}</i>`);
      }
      return lines.join("<br>");
    }

    function embeddedScriptJson(parsedDocument, scriptId, required) {
      // Parse imported dashboard HTML as a document instead of regex-matching script
      // tags. This avoids brittle parsing and keeps the inline runtime free of raw
      // script-closing sequences that would terminate the surrounding HTML script tag.
      const scriptElement = parsedDocument.getElementById(String(scriptId || ""));
      const isJsonScript =
        scriptElement &&
        String(scriptElement.tagName || "").toLowerCase() === "script" &&
        String(scriptElement.getAttribute("type") || "").toLowerCase() === "application/json";
      if (!isJsonScript) {
        if (!required) {
          return null;
        }
        throw new Error(`Imported dashboard file is missing ${scriptId}.`);
      }
      const rawJson = String(scriptElement.textContent || "").trim();
      if (!rawJson) {
        if (!required) {
          return null;
        }
        throw new Error(`Imported dashboard file is missing ${scriptId}.`);
      }
      return JSON.parse(rawJson);
    }

    function extractEmbeddedScriptJson(text, scriptId) {
      const parsedDocument = new DOMParser().parseFromString(
        String(text || ""),
        "text/html"
      );
      return embeddedScriptJson(parsedDocument, scriptId, true);
    }

    function parseImportedResultSetFromText(fileText, filename) {
      const text = String(fileText || "");
      const lowerName = String(filename || "").toLowerCase();
      const looksLikeDashboardHtml =
        lowerName.endsWith(".html") || text.includes('id="citemesh-dashboard-data"');
      if (looksLikeDashboardHtml) {
        const parsedDocument = new DOMParser().parseFromString(text, "text/html");
        const importedGraph = embeddedScriptJson(
          parsedDocument,
          "citemesh-dashboard-data",
          true
        );
        const importedCollection = embeddedScriptJson(
          parsedDocument,
          "citemesh-dashboard-collection",
          false
        );
        const resultSet = importedCollection
          ? normalizeCollectionPackage(importedCollection, filename, true)
          : emptyCollectionPackage();
        const packageCurrentResultId = String(
          (importedCollection && importedCollection.current_result_id) || ""
        ).trim();
        const currentEntry = collectionEntryFromGraphPayload(
          importedGraph,
          filename,
          true
        );
        if (!resultSet.results.some((entry) => entry.result_id === currentEntry.result_id)) {
          resultSet.results.push(currentEntry);
        }
        if (!packageCurrentResultId) {
          resultSet.current_result_id = currentEntry.result_id;
        }
        return resultSet;
      }

      const imported = JSON.parse(text);
      if (imported && imported.kind === COLLECTION_KIND) {
        return normalizeCollectionPackage(imported, filename, false);
      }
      const entry = collectionEntryFromGraphPayload(imported, filename, true);
      const resultSet = emptyCollectionPackage();
      resultSet.current_result_id = entry.result_id;
      resultSet.results.push(entry);
      return resultSet;
    }

    function hasCompleteDashboardGeometry(meta) {
      if (!meta || typeof meta !== "object") {
        return false;
      }
      const order = Array.isArray(meta.plotly_node_order)
        ? meta.plotly_node_order.map((nodeId) => String(nodeId || ""))
        : [];
      const positions = Array.isArray(meta.plotly_positions) ? meta.plotly_positions : [];
      const sizes = Array.isArray(meta.plotly_node_sizes) ? meta.plotly_node_sizes : [];
      if (
        !order.length
        || order.some((nodeId) => !nodeId)
        || order.some((nodeId) => nodeId !== nodeId.trim())
        || new Set(order).size !== order.length
        || positions.length !== order.length
        || sizes.length !== order.length
      ) {
        return false;
      }
      const positionsAreFinite = positions.every((position) => (
        Array.isArray(position)
        && position.length === 2
        && position.every((coordinate) => Number.isFinite(coordinate))
      ));
      const sizesAreFinite = sizes.every(
        (size) => Number.isFinite(size) && size > 0
      );
      return positionsAreFinite && sizesAreFinite;
    }

    function buildFigureSpecFromPayload(nextPayload) {
      const meta = (nextPayload && nextPayload.meta) || {};
      const order = Array.isArray(meta.plotly_node_order)
        ? meta.plotly_node_order.map((nodeId) => String(nodeId || ""))
        : [];
      const positions = Array.isArray(meta.plotly_positions) ? meta.plotly_positions : [];
      const alignedNodeSizes = normalizeArray(meta.plotly_node_sizes, order.length, 8);
      const maxAlignedNodeSize = alignedNodeSizes.length
        ? Math.max(...alignedNodeSizes)
        : 1.0;
      const nextMarkerSizeRef = Math.max(
        (2.0 * maxAlignedNodeSize) / (__DASHBOARD_MAX_NODE_DIAMETER__ ** 2),
        1e-6
      );
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
      const rawSafeYearMax = Number.isFinite(yearMax) ? yearMax : safeYearMin;
      const safeYearMax = rawSafeYearMax > safeYearMin
        ? rawSafeYearMax
        : safeYearMin + 1.0;
      const missingYear = (safeYearMin + safeYearMax) / 2.0;
      const labelIds = selectDashboardLabelIds(
        order,
        nextNodeById,
        xPairs,
        yPairs
      );
      const seedId = String(meta.seed_id || "");
      const seedRingColor = currentSeedRingColor();
      const edgeStrengths = normalizeDashboardEdgeStrengths(nextEdges);

      const nodeTexts = [];
      const hoverTexts = [];
      const nodeYears = [];
      const lineWidths = [];
      const lineColors = [];
      for (let idx = 0; idx < order.length; idx += 1) {
        const nodeId = order[idx];
        const node = nextNodeById.get(nodeId) || {};
        const label = labelIds.has(nodeId) ? dashboardNodeLabel(node, nodeId) : "";
        nodeTexts.push(label);
        const nodeYear = Number.isFinite(Number(node.year)) && Number(node.year) > 0
          ? Number(node.year)
          : missingYear;
        nodeYears.push(nodeYear);
        hoverTexts.push(dashboardHoverText(node, nodeId));
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
      const xPad = Math.max(__DASHBOARD_AXIS_X_PADDING__, xSpan * 0.1);
      const yPad = Math.max(__DASHBOARD_AXIS_MIN_PADDING__, ySpan * 0.08);
      const dashboardEdgeColor = "__DASHBOARD_EDGE_COLOR__";
      const edgeShapes = [];
      nextEdges.forEach((edge, edgeIndex) => {
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
        const strength = edgeStrengths[edgeIndex];
        edgeShapes.push({
          type: "path",
          path: `M ${x0},${y0} Q ${cx},${cy} ${x1},${y1}`,
          line: {
            color: colorWithAlpha(dashboardEdgeColor, 0.07 + (0.25 * strength)),
            width: 0.45 + (1.2 * strength),
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
        sizeref: nextMarkerSizeRef,
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
          sizeref: nextMarkerSizeRef,
          color: seedIdx >= 0 ? [colorWithAlpha(seedRingColor, 0.26)] : [],
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
      layout.uirevision = `citemesh-dashboard-static-layout-v1:${String(meta.strategy || "")}:${String(meta.seed_id || "")}`;

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

    // Full persisted reading list for the active result key, including IDs the
    // current graph no longer contains: the key survives rebuilds, so persisting
    // only the displayable subset would erase saves whenever a node drops out.
    let persistedSavedIds = new Set();

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
      savedOnly: false,
      savedIds: loadSavedIdSet(),
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
      savedChip: document.getElementById("saved-filter"),
      savedBibBtn: document.getElementById("export-saved-bib-btn"),
      copySavedBtn: document.getElementById("copy-saved-links-btn"),
    };

    function savedStorageKey() {
      const meta = (payload && payload.meta) || {};
      const strategy = String(meta.strategy || "default");
      const seedId = String(meta.seed_id || "default");
      return `citemesh-saved:${strategy}:${seedId}`;
    }

    function pruneSavedIdsForPayload(savedIds) {
      const availableIds = new Set(
        (payload.nodes || []).map((node) => String(node.id || "")).filter(Boolean)
      );
      return new Set(
        Array.from(savedIds).filter((nodeId) => availableIds.has(String(nodeId)))
      );
    }

    function loadPersistedSavedIds() {
      try {
        const raw = window.localStorage.getItem(savedStorageKey());
        const parsed = raw ? JSON.parse(raw) : [];
        return new Set(Array.isArray(parsed) ? parsed.map(String) : []);
      } catch (err) {
        return new Set();
      }
    }

    function loadSavedIdSet() {
      persistedSavedIds = loadPersistedSavedIds();
      return pruneSavedIdsForPayload(persistedSavedIds);
    }

    function persistSavedIds() {
      try {
        window.localStorage.setItem(savedStorageKey(), JSON.stringify(Array.from(persistedSavedIds)));
      } catch (err) {
        // Storage unavailable (strict privacy mode, some file:// contexts):
        // the reading list still works for the current session.
      }
    }

    function isSaved(nodeId) {
      return state.savedIds.has(String(nodeId || ""));
    }

    function savedNodes() {
      return (payload.nodes || []).filter((node) => state.savedIds.has(String(node.id || "")));
    }

    function updateSavedUi() {
      const count = state.savedIds.size;
      if (controls.savedChip) {
        controls.savedChip.textContent = count ? `Saved (${count})` : "Saved";
        controls.savedChip.classList.toggle("active", state.savedOnly);
      }
      const showExports = count > 0;
      if (controls.savedBibBtn) {
        controls.savedBibBtn.style.display = showExports ? "" : "none";
      }
      if (controls.copySavedBtn) {
        controls.copySavedBtn.style.display = showExports ? "" : "none";
      }
    }

    function refreshSavedState() {
      state.savedIds = loadSavedIdSet();
      state.savedOnly = false;
      updateSavedUi();
    }

    function toggleSaved(nodeId) {
      const key = String(nodeId || "");
      if (!key) {
        return;
      }
      if (state.savedIds.has(key)) {
        state.savedIds.delete(key);
        persistedSavedIds.delete(key);
      } else {
        state.savedIds.add(key);
        persistedSavedIds.add(key);
      }
      persistSavedIds();
      if (!state.savedIds.size) {
        state.savedOnly = false;
      }
      updateSavedUi();
      renderList();
      if (state.hoverId === key) {
        renderDetail(key, true);
      } else if (state.selectedId === key && !state.hoverId) {
        renderDetail(key, false);
      }
    }

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
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function markdownLinkText(value) {
      // An unescaped bracket in a title closes the Markdown link label early.
      return String(value ?? "").replace(/([\\[\\]])/g, "\\\\$1");
    }

    function hasYear(node) {
      return Number.isFinite(Number(node.year)) && Number(node.year) > 0;
    }

    function nodeFilterClass(node) {
      const base = node.provenance_base || node.provenance || "citation";
      return state.filters[base] === true;
    }

    function nodeMatches(node) {
      if (state.savedOnly && !state.savedIds.has(String(node.id || ""))) {
        return false;
      }
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

    function safeExternalUrl(value) {
      const candidate = String(value || "").trim();
      if (!candidate) {
        return "";
      }
      try {
        const parsed = new URL(candidate);
        return parsed.protocol === "https:" || parsed.protocol === "http:"
          ? parsed.href
          : "";
      } catch (err) {
        return "";
      }
    }

    function detailLinkEntries(links) {
      const entries = [];
      const pdfUrl = safeExternalUrl(links && links.arxiv_pdf);
      const arxivUrl = safeExternalUrl(links && links.arxiv_abs);
      const doiUrl = safeExternalUrl(links && links.doi);
      const semanticScholarUrl = safeExternalUrl(links && links.semantic_scholar);
      if (pdfUrl) {
        entries.push({ kind: "pdf", title: "Open PDF", href: pdfUrl });
      }
      if (arxivUrl) {
        entries.push({ kind: "arxiv", title: "Open arXiv page", href: arxivUrl });
      }
      if (doiUrl) {
        entries.push({ kind: "doi", title: "Open DOI", href: doiUrl });
      }
      if (semanticScholarUrl) {
        entries.push({ kind: "s2", title: "Open Semantic Scholar", href: semanticScholarUrl });
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

    function portableGraphPayload(nextPayload) {
      const nextNodes = Array.isArray(nextPayload.nodes) ? nextPayload.nodes : [];
      const nextEdges = Array.isArray(nextPayload.edges) ? nextPayload.edges : [];
      const nextMeta = (nextPayload && nextPayload.meta) || {};
      const nextNodeById = new Map(
        nextNodes.map((node) => [String(node.id || ""), node])
      );
      const portableNodeLabel = (nodeId) => {
        const node = nextNodeById.get(String(nodeId || ""));
        if (!node) {
          return String(nodeId || "");
        }
        if (Array.isArray(node.authors) && node.authors.length) {
          const surname = String(node.authors[0]).split(" ").filter(Boolean).slice(-1)[0] || "Unknown";
          const year = hasYear(node) ? String(node.year) : "n.d.";
          return `${surname}, ${year}`;
        }
        return String(node.title || node.id || nodeId);
      };
      const summary = {
        nodes: nextNodes.length,
        edges: nextEdges.length,
      };
      const dashboardMeta = Object.assign({}, nextMeta, {
        summary,
      });
      const edges = nextEdges.map((edge) => {
        const sourceId = String(edge.source || "");
        const targetId = String(edge.target || "");
        const sourceNode = nextNodeById.get(sourceId);
        const targetNode = nextNodeById.get(targetId);
        return {
          source: sourceId,
          target: targetId,
          source_title: sourceNode ? String(sourceNode.title || sourceId) : sourceId,
          target_title: targetNode ? String(targetNode.title || targetId) : targetId,
          source_label: portableNodeLabel(sourceId),
          target_label: portableNodeLabel(targetId),
          weight: Number(edge.weight || 0),
        };
      });
      return {
        kind: GRAPH_PAYLOAD_KIND,
        schema_version: GRAPH_PAYLOAD_SCHEMA_VERSION,
        seed_id: nextMeta.seed_id || "",
        meta: {
          strategy: nextMeta.strategy || "",
          year_range: nextMeta.year_range || {},
          candidate_source_status: nextMeta.candidate_source_status || {},
        },
        summary,
        dashboard: {
          meta: dashboardMeta,
        },
        nodes: nextNodes,
        edges,
      };
    }

    function buildPortableJsonPayload() {
      return portableGraphPayload(payload);
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

    function emptyCollectionPackage() {
      return {
        kind: COLLECTION_KIND,
        schema_version: COLLECTION_SCHEMA_VERSION,
        current_result_id: null,
        results: [],
      };
    }

    function isObjectRecord(value) {
      return !!value && typeof value === "object" && !Array.isArray(value);
    }

    function isNonNegativeInteger(value) {
      return Number.isInteger(value) && value >= 0;
    }

    function normalizeImportedDashboardPayload(imported, label, allowLegacy) {
      if (!isObjectRecord(imported)) {
        throw new Error("Expected a CiteMesh graph object.");
      }
      const declaredKind = String(imported.kind || "");
      if (declaredKind) {
        if (declaredKind !== GRAPH_PAYLOAD_KIND) {
          throw new Error(`Unsupported result kind: ${declaredKind}.`);
        }
        if (imported.schema_version !== GRAPH_PAYLOAD_SCHEMA_VERSION) {
          throw new Error(
            `Unsupported ${GRAPH_PAYLOAD_KIND} schema version: ${String(imported.schema_version)}.`
          );
        }
      } else if (!allowLegacy) {
        throw new Error(`Result payload is missing kind=${GRAPH_PAYLOAD_KIND}.`);
      }

      if (!Array.isArray(imported.nodes) || !imported.nodes.length) {
        throw new Error("No nodes found in graph results.");
      }
      if (!Array.isArray(imported.edges)) {
        throw new Error("Graph results must contain an edges array.");
      }
      const importedNodes = imported.nodes;
      const importedEdges = imported.edges;
      const nodeIds = importedNodes.map((node) => String((node && node.id) || ""));
      if (nodeIds.some((nodeId) => !nodeId) || new Set(nodeIds).size !== nodeIds.length) {
        throw new Error("Graph result nodes must have unique, non-empty IDs.");
      }
      if (declaredKind && nodeIds.some((nodeId) => nodeId !== nodeId.trim())) {
        throw new Error("Versioned graph node IDs cannot contain surrounding whitespace.");
      }
      const nodeIdSet = new Set(nodeIds);
      if (importedEdges.some((edge) => (
        !isObjectRecord(edge)
        || !nodeIdSet.has(String(edge.source || ""))
        || !nodeIdSet.has(String(edge.target || ""))
      ))) {
        throw new Error("Graph result edges must reference included node IDs.");
      }
      if (declaredKind && importedEdges.some((edge) => (
        String(edge.source || "") !== String(edge.source || "").trim()
        || String(edge.target || "") !== String(edge.target || "").trim()
      ))) {
        throw new Error("Versioned graph edge IDs cannot contain surrounding whitespace.");
      }
      if (declaredKind) {
        if (!isObjectRecord(imported.summary)) {
          throw new Error("Versioned graph results must contain a summary object.");
        }
        if (
          !isNonNegativeInteger(imported.summary.nodes)
          || !isNonNegativeInteger(imported.summary.edges)
          || imported.summary.nodes !== importedNodes.length
          || imported.summary.edges !== importedEdges.length
        ) {
          throw new Error("Graph result summary does not match its node and edge arrays.");
        }
      }
      const importedDashboardMeta =
        imported && imported.dashboard && imported.dashboard.meta
          ? imported.dashboard.meta
          : {};
      const importedMeta =
        imported && imported.meta && typeof imported.meta === "object"
          ? imported.meta
          : {};
      const declaredSeedId = String(imported.seed_id || "");
      const declaredStrategy = String(importedMeta.strategy || "");
      if (declaredKind && (
        !declaredSeedId
        || declaredSeedId !== declaredSeedId.trim()
        || !isObjectRecord(imported.meta)
        || !declaredStrategy
        || declaredStrategy !== declaredStrategy.trim()
      )) {
        throw new Error(
          "Versioned graph results require canonical top-level seed_id and meta.strategy."
        );
      }
      if (declaredKind && (
        !isObjectRecord(importedDashboardMeta)
        || String(importedDashboardMeta.seed_id || "") !== declaredSeedId
        || String(importedDashboardMeta.strategy || "") !== declaredStrategy
        || !isObjectRecord(importedDashboardMeta.summary)
        || importedDashboardMeta.summary.nodes !== importedNodes.length
        || importedDashboardMeta.summary.edges !== importedEdges.length
      )) {
        throw new Error(
          "Versioned graph dashboard metadata must match its top-level identity and summary."
        );
      }
      const seedId = String(
        declaredSeedId
        || importedDashboardMeta.seed_id
        || importedMeta.seed_id
        || ((importedNodes.find((node) => !!(node && node.is_seed)) || {}).id || "")
      );
      const strategy = declaredKind
        ? declaredStrategy
        : String(importedDashboardMeta.strategy || importedMeta.strategy || "");
      if (!seedId || !nodeIds.includes(seedId)) {
        throw new Error("Graph results must identify a seed node present in nodes.");
      }
      if (!strategy) {
        throw new Error("Graph results must identify the build strategy.");
      }
      const baseMeta = {
        seed_id: seedId,
        strategy,
        theme: String(
          (payload.meta && payload.meta.theme)
          || importedDashboardMeta.theme
          || importedMeta.theme
          || "light"
        ),
        summary: {
          nodes: importedNodes.length,
          edges: importedEdges.length,
        },
        year_range: importedDashboardMeta.year_range || importedMeta.year_range || {},
        candidate_source_status:
          importedDashboardMeta.candidate_source_status
          || importedMeta.candidate_source_status
          || {},
        plotly_node_order:
          importedDashboardMeta.plotly_node_order || importedMeta.plotly_node_order || [],
        plotly_positions:
          importedDashboardMeta.plotly_positions || importedMeta.plotly_positions || [],
        plotly_node_sizes:
          importedDashboardMeta.plotly_node_sizes || importedMeta.plotly_node_sizes || [],
      };
      if (!hasCompleteDashboardGeometry(baseMeta)) {
        throw new Error(
          `Imported results from ${String(label || "the selected file")} are missing stored dashboard geometry. Export dashboard-compatible CiteMesh results first.`
        );
      }
      const geometryOrder = baseMeta.plotly_node_order.map((nodeId) => String(nodeId || ""));
      if (
        geometryOrder.length !== nodeIds.length
        || new Set(geometryOrder).size !== geometryOrder.length
        || geometryOrder.some((nodeId) => !nodeIds.includes(nodeId))
      ) {
        throw new Error("Dashboard geometry must cover each graph node exactly once.");
      }

      return {
        payload: {
          meta: baseMeta,
          nodes: importedNodes,
          edges: imported.edges,
        },
      };
    }

    function collectionEntryFromGraphPayload(imported, label, allowLegacy) {
      const normalized = normalizeImportedDashboardPayload(imported, label, allowLegacy);
      const normalizedPayload = normalized.payload;
      const resultId = currentResultIdForPayload(normalizedPayload);
      if (!resultId) {
        throw new Error("Graph results do not provide a stable strategy and seed ID.");
      }
      const seedId = String(normalizedPayload.meta.seed_id || "");
      const seedNode = normalizedPayload.nodes.find(
        (node) => String((node && node.id) || "") === seedId
      );
      return {
        result_id: resultId,
        seed_id: seedId,
        title: String((seedNode && seedNode.title) || seedId || label || "Graph result"),
        strategy: String(normalizedPayload.meta.strategy || ""),
        summary: {
          nodes: normalizedPayload.nodes.length,
          edges: normalizedPayload.edges.length,
        },
        payload: normalizedPayload,
        updated_at: new Date().toISOString(),
        build: {},
      };
    }

    function normalizeCollectionPackage(imported, label, allowLegacy) {
      if (!isObjectRecord(imported)) {
        throw new Error("Expected a CiteMesh dashboard collection object.");
      }
      const declaredKind = String(imported.kind || "");
      const isVersionedCollection = declaredKind === COLLECTION_KIND;
      if (declaredKind && !isVersionedCollection) {
        throw new Error(`Unsupported collection kind: ${declaredKind}.`);
      }
      if (isVersionedCollection && imported.schema_version !== COLLECTION_SCHEMA_VERSION) {
        throw new Error(
          `Unsupported ${COLLECTION_KIND} schema version: ${String(imported.schema_version)}.`
        );
      }
      if (!isVersionedCollection && !allowLegacy) {
        throw new Error(`Collection package is missing kind=${COLLECTION_KIND}.`);
      }
      if (!Array.isArray(imported.results)) {
        throw new Error("Collection package must contain a results array.");
      }

      const normalized = emptyCollectionPackage();
      const legacyPayloads = isObjectRecord(imported.payloads) ? imported.payloads : {};
      imported.results.forEach((rawEntry, index) => {
        if (!isObjectRecord(rawEntry)) {
          throw new Error(`Collection result ${index + 1} must be an object.`);
        }
        const declaredResultId = String(rawEntry.result_id || "").trim();
        const rawPayload = isObjectRecord(rawEntry.payload)
          ? rawEntry.payload
          : legacyPayloads[declaredResultId];
        if (!isObjectRecord(rawPayload)) {
          throw new Error(`Collection result ${index + 1} is missing its graph payload.`);
        }
        const normalizedEntry = collectionEntryFromGraphPayload(
          rawPayload,
          `${label || "collection"} result ${index + 1}`,
          allowLegacy
        );
        if (declaredResultId && declaredResultId !== normalizedEntry.result_id) {
          throw new Error(
            `Collection result ${index + 1} ID does not match its graph strategy and seed.`
          );
        }
        if (!declaredResultId && isVersionedCollection) {
          throw new Error(`Collection result ${index + 1} is missing result_id.`);
        }
        if (isVersionedCollection) {
          if (!String(rawEntry.title || "").trim()) {
            throw new Error(`Collection result ${index + 1} is missing title.`);
          }
          if (!String(rawEntry.updated_at || "").trim()) {
            throw new Error(`Collection result ${index + 1} is missing updated_at.`);
          }
          if (!isObjectRecord(rawEntry.summary)) {
            throw new Error(`Collection result ${index + 1} is missing summary.`);
          }
          if (
            !isNonNegativeInteger(rawEntry.summary.nodes)
            || !isNonNegativeInteger(rawEntry.summary.edges)
            || rawEntry.summary.nodes !== normalizedEntry.summary.nodes
            || rawEntry.summary.edges !== normalizedEntry.summary.edges
          ) {
            throw new Error(`Collection result ${index + 1} has an inconsistent summary.`);
          }
          if (!isObjectRecord(rawEntry.build)) {
            throw new Error(`Collection result ${index + 1} is missing build metadata.`);
          }
        }
        for (const field of ["seed_id", "strategy"]) {
          if (
            rawEntry[field] !== undefined
            && String(rawEntry[field]) !== String(normalizedEntry[field])
          ) {
            throw new Error(`Collection result ${index + 1} has inconsistent ${field}.`);
          }
        }
        if (rawEntry.title !== undefined) {
          normalizedEntry.title = String(rawEntry.title || normalizedEntry.title);
        }
        if (rawEntry.updated_at !== undefined) {
          normalizedEntry.updated_at = String(rawEntry.updated_at || "");
        }
        if (rawEntry.build !== undefined) {
          if (!isObjectRecord(rawEntry.build)) {
            throw new Error(`Collection result ${index + 1} build metadata must be an object.`);
          }
          normalizedEntry.build = rawEntry.build;
        }
        const duplicateIndex = normalized.results.findIndex(
          (entry) => entry.result_id === normalizedEntry.result_id
        );
        if (duplicateIndex < 0) {
          normalized.results.push(normalizedEntry);
        }
      });

      const currentId = String(imported.current_result_id || "").trim();
      if (currentId && !normalized.results.some((entry) => entry.result_id === currentId)) {
        throw new Error("Collection current_result_id does not name an included result.");
      }
      normalized.current_result_id = currentId || (
        normalized.results.length ? normalized.results[0].result_id : null
      );
      return normalized;
    }

    function upsertCollectionEntries(targetCollection, incomingEntries) {
      const incomingUnique = [];
      const incomingIds = new Set();
      incomingEntries.forEach((incomingEntry) => {
        const resultId = String(incomingEntry.result_id || "");
        if (resultId && !incomingIds.has(resultId)) {
          incomingIds.add(resultId);
          incomingUnique.push(incomingEntry);
        }
      });
      const retained = targetCollection.results.filter(
        (entry) => !incomingIds.has(String(entry.result_id || ""))
      );
      targetCollection.results = incomingUnique.concat(retained);
    }

    function portableCollectionPackage() {
      return {
        kind: COLLECTION_KIND,
        schema_version: COLLECTION_SCHEMA_VERSION,
        current_result_id: collectionResultId || currentResultIdForPayload(payload),
        results: collectionEntries().map((entry) => {
          const portableEntry = {
            result_id: entry.result_id,
            seed_id: entry.seed_id,
            title: entry.title,
            strategy: entry.strategy,
            summary: entry.summary,
            payload: portableGraphPayload(entry.payload),
            updated_at: entry.updated_at || new Date().toISOString(),
            build: isObjectRecord(entry.build) ? entry.build : {},
          };
          return portableEntry;
        }),
      };
    }

    function collectionEntryLabel(entry) {
      const title = String(entry.title || entry.seed_id || entry.result_id || "Graph result");
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
      placeholder.textContent = "Select a graph";
      placeholder.disabled = true;
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

    function applyImportedPayload(imported, label) {
      const normalizedImport = normalizeImportedDashboardPayload(imported, label, true);
      const nextPayload = normalizedImport.payload;
      const nextFigureSpec = buildFigureSpecFromPayload(nextPayload);
      payload = nextPayload;
      figureSpec = nextFigureSpec;
      clearDashboardStatus();
      collectionResultId = currentResultIdForPayload(nextPayload);
      rebuildDerivedData();
      refreshSavedState();
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
      const entry = collectionEntries().find(
        (candidate) => String(candidate.result_id || "") === normalizedId
      );
      if (!entry || !entry.payload) {
        setDashboardStatus(
          "That graph is unavailable in the current result set. Add its package again.",
          "warning"
        );
        return Promise.resolve();
      }
      collectionResultId = normalizedId;
      clearDashboardStatus();
      return applyImportedPayload(
        entry.payload,
        collectionEntryLabel(entry)
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
      if (node.is_seed) {
        controls.detailWhy.classList.remove("muted");
        controls.detailWhy.innerHTML = [
          `<div>${escapeHtml("Seed paper - every other node in this graph was gathered around it.")}</div>`,
          `<div>${escapeHtml(neighborLine)}</div>`,
        ].join("");
        return;
      }
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
      const saveBtn = document.createElement("button");
      saveBtn.type = "button";
      saveBtn.textContent = isSaved(node.id) ? "★ Saved" : "☆ Save";
      saveBtn.title = isSaved(node.id)
        ? "Remove from reading list"
        : "Save to reading list";
      saveBtn.addEventListener("click", () => {
        toggleSaved(node.id);
      });
      controls.detailActions.appendChild(saveBtn);

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
        const pointIndex = Number.parseInt(rawPointIndex ?? "", 10);
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
          haloColor = [colorWithAlpha(currentSeedRingColor(), state.selectedId ? 0.34 : 0.26)];
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
      controls.count.textContent = `${listNodes.length.toLocaleString()} ${listNodes.length === 1 ? "paper" : "papers"}`;
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

        const saved = isSaved(node.id);
        row.innerHTML = `
          <div class="paper-row-head">
            <div class="paper-title">${escapeHtml(node.title || node.id)}</div>
            <button class="star-btn${saved ? " saved" : ""}" type="button" title="${saved ? "Remove from reading list" : "Save to reading list"}" aria-label="${saved ? "Remove from reading list" : "Save to reading list"}" aria-pressed="${saved ? "true" : "false"}">${saved ? "&#9733;" : "&#9734;"}</button>
            <div class="paper-year">${escapeHtml(yearText)}</div>
          </div>
          <div class="paper-subline">${escapeHtml(authors)}</div>
          <div class="paper-meta">
            <span>${Number(node.citation_count || 0).toLocaleString()} citations</span>
            <span class="meta-dot"></span>
            <span class="${provenanceClass}">${escapeHtml(provenanceLabel)}</span>
          </div>
        `;

        const starBtn = row.querySelector(".star-btn");
        if (starBtn) {
          starBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            toggleSaved(node.id);
          });
        }

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
        const target = safeExternalUrl(
          focusNode && focusNode.links && focusNode.links.semantic_scholar
        );
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
        const slug = seedNode && seedNode.title
          ? seedNode.title.replace(/[^a-zA-Z0-9]+/g, "_").substring(0, 40).replace(/_+$/, "").toLowerCase()
          : "";
        // Titles without ASCII alphanumerics slug to "", which would name the
        // download ".json" and give the browser no stem to disambiguate.
        return slug || "citemesh";
      }

      document.getElementById("export-json-btn").addEventListener("click", () => {
        const exportPayload = buildPortableJsonPayload();
        downloadBlob(JSON.stringify(exportPayload, null, 2), seedSlug() + ".json", "application/json");
      });

      document.getElementById("export-collection-btn").addEventListener("click", () => {
        const exportPayload = portableCollectionPackage();
        downloadBlob(
          JSON.stringify(exportPayload, null, 2),
          "dashboard.citemesh.json",
          "application/json"
        );
        setDashboardStatus(
          `Exported ${exportPayload.results.length} ${exportPayload.results.length === 1 ? "graph" : "graphs"} as one collection package.`,
          "info"
        );
      });

      document.getElementById("export-csv-btn").addEventListener("click", () => {
        const cols = ["id","title","year","authors","citation_count","venue","arxiv_id","doi","categories","is_seed","provenance","seed_relation","seed_relevance","arxiv_url","doi_url","semantic_scholar_url","abstract"];
        // Mirror the CLI CSV writer: neutralize formula-leading cells (CWE-1236).
        function csvGuard(v) { const s = String(v == null ? "" : v); return /^[=+\\-@\\t\\r]/.test(s) ? "'" + s : s; }
        function csvEscape(v) { const s = csvGuard(v); return s.includes(",") || s.includes('"') || s.includes("\\n") || s.includes("\\r") ? '"' + s.replace(/"/g, '""') + '"' : s; }
        const rows = [cols.join(",")];
        for (const n of (payload.nodes || [])) {
          const links = n.links || {};
          rows.push([
            n.id, n.title, n.year, (n.authors||[]).join("; "), n.citation_count, n.venue||"", n.arxiv_id||"", n.doi||"",
            (n.categories||[]).join("; "), (n.is_seed ? "true" : "false"), n.provenance||"", n.seed_relation||"",
            Number(n.seed_relevance||0).toFixed(6), links.arxiv_abs||"", links.doi||"", links.semantic_scholar||"", n.abstract||""
          ].map(csvEscape).join(","));
        }
        downloadBlob(rows.join("\\n"), seedSlug() + ".csv", "text/csv;charset=utf-8");
      });

      document.getElementById("export-bib-btn").addEventListener("click", () => {
        const entries = (payload.nodes || []).map((n) => (n.bibtex || "").trim()).filter(Boolean);
        downloadBlob(entries.join("\\n\\n") + "\\n", seedSlug() + ".bib", "text/plain;charset=utf-8");
      });

      controls.savedBibBtn.addEventListener("click", () => {
        const entries = savedNodes().map((n) => (n.bibtex || "").trim()).filter(Boolean);
        if (!entries.length) {
          return;
        }
        downloadBlob(entries.join("\\n\\n") + "\\n", seedSlug() + "-saved.bib", "text/plain;charset=utf-8");
      });

      controls.copySavedBtn.addEventListener("click", () => {
        const lines = savedNodes().map((n) => {
          const links = n.links || {};
          const href = safeExternalUrl(
            links.arxiv_abs || links.doi || links.semantic_scholar
          );
          const yearText = hasYear(n) ? ` (${n.year})` : "";
          const title = String(n.title || n.id);
          return href
            ? `- [${markdownLinkText(title)}](${href})${yearText}`
            : `- ${title}${yearText}`;
        });
        if (!lines.length) {
          return;
        }
        copyText(lines.join("\\n")).then(() => {
          controls.copySavedBtn.textContent = "Copied";
          window.setTimeout(() => {
            controls.copySavedBtn.textContent = "Copy Saved Links";
          }, 1000);
        });
      });

      controls.savedChip.addEventListener("click", () => {
        state.savedOnly = !state.savedOnly;
        updateSavedUi();
        renderList();
      });

      function readFileText(file) {
        return new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = (event) => resolve(String(event.target.result || ""));
          reader.onerror = () => reject(
            new Error(reader.error ? reader.error.message : "Browser could not read the file.")
          );
          reader.readAsText(file);
        });
      }

      async function addResultFiles(files) {
        let importedCount = 0;
        let desiredResultId = null;
        const failures = [];
        for (const file of files) {
          try {
            const fileText = await readFileText(file);
            const importedCollection = parseImportedResultSetFromText(fileText, file.name);
            upsertCollectionEntries(collectionBundle, importedCollection.results);
            importedCount += importedCollection.results.length;
            desiredResultId = importedCollection.current_result_id || desiredResultId;
          } catch (err) {
            failures.push(`${file.name}: ${err.message}`);
          }
        }

        if (importedCount > 0) {
          const nextResultId = desiredResultId || collectionBundle.results[0].result_id;
          collectionBundle.current_result_id = nextResultId;
          collectionResultId = nextResultId;
          populateCollectionSelector();
          try {
            await loadCollectionResult(nextResultId);
          } catch (err) {
            failures.push(`Display: ${err.message}`);
          }
        }

        const uniqueGraphCount = collectionBundle.results.length;
        const importedMessage = importedCount > 0
          ? `Merged ${importedCount} ${importedCount === 1 ? "graph entry" : "graph entries"}; ${uniqueGraphCount} unique ${uniqueGraphCount === 1 ? "graph" : "graphs"} in this session.`
          : "No graphs were added.";
        const failureMessage = failures.length
          ? ` Skipped ${failures.length} ${failures.length === 1 ? "file" : "files"}: ${failures.join(" | ")}`
          : "";
        setDashboardStatus(
          importedMessage + failureMessage,
          failures.length ? "warning" : "info"
        );
      }

      const addResultsInput = document.getElementById("add-results-input");
      document.getElementById("add-results-btn").addEventListener("click", () => {
        addResultsInput.click();
      });
      addResultsInput.addEventListener("change", (event) => {
        const files = Array.from((event.target && event.target.files) || []);
        addResultsInput.value = "";
        if (!files.length) {
          return;
        }
        addResultFiles(files).catch((err) => {
          setDashboardStatus("Failed to add results: " + err.message, "warning");
        });
      });
      controls.resultSelect.addEventListener("change", (event) => {
        const resultId = String(event.target.value || "").trim();
        if (!resultId) {
          return;
        }
        loadCollectionResult(resultId).catch((err) => {
          setDashboardStatus("Failed to switch graphs: " + err.message, "warning");
        });
      });

      setControlsCollapsed(true);
      populateCollectionSelector();
      updateYearPlaceholders();
      updateSavedUi();
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
      if (bootstrapFailureMessage) {
        setDashboardStatus(bootstrapFailureMessage, "warning");
        return;
      }
      if (bootstrapStatusMessage) {
        setDashboardStatus(bootstrapStatusMessage, "warning");
      }
      setupControls();
      // Toolbar changes resize the pane without triggering a window resize.
      const graphResizeObserver = new ResizeObserver(() => Plotly.Plots.resize(graphDiv));
      graphResizeObserver.observe(graphDiv);
      renderTimeline();
      const initialResultId = currentResultIdForPayload(payload);
      if (collectionResultId && collectionResultId !== initialResultId) {
        loadCollectionResult(collectionResultId).catch((err) => {
          setDashboardStatus("Failed to load the selected graph: " + err.message, "warning");
        });
        return;
      }
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
        # Single-pass substitution over the template only: sequential
        # str.replace would rescan already-injected values, letting a token
        # such as __PLOTLY_DIV_ID__ inside paper metadata get rewritten (or,
        # for the JSON tokens, corrupt the embedded payloads).
        token_pattern = re.compile(
            "|".join(
                re.escape(token) for token in sorted(vars_map, key=len, reverse=True)
            )
        )
        return token_pattern.sub(lambda match: vars_map[match.group(0)], template)

    def _sorted_nodes(self) -> list[tuple[Hashable, Dict[str, Any]]]:
        """Return nodes sorted by ID for deterministic serialization.

        :return list[tuple[Hashable, Dict[str, Any]]]: Sorted ``(node_id, attrs)``
            pairs.
        :raises ValueError: If an ID is empty or has surrounding whitespace.
        """
        nodes = [
            (node_id, self.graph.nodes[node_id])
            for node_id in ordered_nodes(self.graph)
        ]
        for node_id, _ in nodes:
            identifier = str(node_id)
            if not identifier or identifier != identifier.strip():
                raise ValueError(
                    f"Cannot export non-canonical node ID {node_id!r}: "
                    "IDs must be non-empty and have no surrounding whitespace."
                )
        return nodes

    def _sorted_edges(self) -> list[tuple[Hashable, Hashable, Dict[str, Any]]]:
        """Return undirected edges with canonical endpoints in stable order.

        :return list[tuple[Hashable, Hashable, Dict[str, Any]]]: Sorted edge tuples in
            ``(u, v, attrs)`` form.
        :raises ValueError: If an edge weight is null or non-finite.
        """
        edges = ordered_edges_with_data(self.graph)
        for left, right, attrs in edges:
            weight = attrs.get("weight", 0.0)
            if weight is None or not math.isfinite(float(weight)):
                raise ValueError(
                    f"Cannot export null or non-finite edge weight for {left!r} -> {right!r}."
                )
        return edges

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
        return publication_year_scale(
            self.graph.nodes[node].get("year") for node in node_ids
        )

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
            node_data.setdefault("abstract", attrs.get("abstract", ""))
            node_data.setdefault("categories", attrs.get("categories", []))

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

        year = coerce_publication_year(attrs.get("year"))
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
