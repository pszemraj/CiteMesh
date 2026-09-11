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
from functools import lru_cache
from importlib import resources
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
from citemesh.data.cache import atomic_output_path, atomic_write_text

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
DASHBOARD_SELECTION_HALO_SCALE = 2.2
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


_DASHBOARD_ASSET_PACKAGE = "citemesh.visualization.dashboard"
_DASHBOARD_CSS_SLOT = "__DASHBOARD_CSS__\n"
_DASHBOARD_JS_SLOT = "__DASHBOARD_JS__\n"


def _read_dashboard_asset(name: str) -> str:
    """Read a packaged dashboard asset as UTF-8 text.

    :param str name: Asset file name under the dashboard ``assets`` directory.
    :return str: Decoded asset contents.
    """
    asset = resources.files(_DASHBOARD_ASSET_PACKAGE) / "assets" / name
    return asset.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _dashboard_template_source() -> str:
    """Assemble the dashboard HTML template from its packaged assets.

    The CSS and JS slots are filled here rather than through the caller's
    token substitution: both assets themselves contain substitution tokens
    (for example ``__BODY_BG__`` and ``__PLOTLY_DIV_ID__``), which a
    single-pass substitution over the markup alone would never visit.

    :return str: Template markup with the stylesheet and script inlined.
    """
    template = _read_dashboard_asset("template.html")
    template = template.replace(
        _DASHBOARD_CSS_SLOT, _read_dashboard_asset("dashboard.css")
    )
    return template.replace(_DASHBOARD_JS_SLOT, _read_dashboard_asset("dashboard.js"))


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

        buffer = io.BytesIO()
        nx.write_graphml(export_graph, buffer)
        atomic_write_text(path, buffer.getvalue().decode("utf-8"))

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

        with atomic_output_path(path) as tmp_path:
            net.save_graph(str(tmp_path))
            _inject_darkreader_lock(tmp_path, _theme_color_scheme(theme_obj))

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
        with atomic_output_path(path) as tmp_path:
            try:
                fig.write_html(str(tmp_path), div_id=div_id)
            except TypeError as exc:
                raise RuntimeError(
                    "Deterministic Plotly export requires write_html(div_id=...). "
                    "Upgrade plotly to a version that supports div_id."
                ) from exc
            _inject_darkreader_lock(tmp_path, _theme_color_scheme(theme_obj))

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
        if for_dashboard:
            # Pixel shifts clear the largest selection halo at every zoom level.
            layout_kwargs["annotations"] = [
                dict(
                    x=node_x[idx],
                    y=node_y[idx],
                    xref="x",
                    yref="y",
                    text=label,
                    font=text_font,
                    showarrow=False,
                    xanchor="center",
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
                        + marker_line_width[idx] / 2.0,
                    )
                    + 3.0,
                )
                for idx, label in enumerate(node_labels)
                if label
            ]
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
            "__DASHBOARD_SELECTION_HALO_SCALE__": str(DASHBOARD_SELECTION_HALO_SCALE),
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
        template = _dashboard_template_source()
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
