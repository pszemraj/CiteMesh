"""Layout geometry, color conversion, and HTML theming primitives."""

from __future__ import annotations

import html
import math
import re
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

from citemesh.core.values import coerce_citation_count

from ..themes import Theme
from ..years import coerce_publication_year

if TYPE_CHECKING:
    from citemesh.core.models import PaperGraph

DASHBOARD_AXIS_MIN_PADDING = 0.14
DASHBOARD_AXIS_X_PADDING = 0.18
DASHBOARD_FOOTER_MARGIN = 78
DASHBOARD_LABEL_CAP = 8
DASHBOARD_LABEL_MIN_DISTANCE = 0.18
DASHBOARD_MAX_NODE_DIAMETER = 58.0
DASHBOARD_SELECTION_HALO_SCALE = 2.2

DARKREADER_LOCK_META = '<meta name="darkreader-lock" />'


class PlotlyHoverFont(TypedDict):
    """Font fields used by the shared Plotly hover label."""

    color: str
    family: str
    size: int


class PlotlyHoverLabel(TypedDict):
    """The themed Plotly hover label shared by graph traces and the viewer."""

    align: Literal["left"]
    bgcolor: str
    bordercolor: str
    font: PlotlyHoverFont


# Head rewriting patterns. Kept deliberately narrow: they run over the raw
# ``<head>`` text of a library-generated export, which may already carry large
# inlined script and style bundles.
_HEAD_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
_HEAD_REMOTE_ASSET_PATTERN = re.compile(
    r"<link\b[^>]*\bhref\s*=[^>]*>|<script\b[^>]*\bsrc\s*=[^>]*>\s*</script>",
    re.IGNORECASE,
)
_HEAD_TITLE_PATTERN = re.compile(
    r"<title\b[^>]*>.*?</title>", re.DOTALL | re.IGNORECASE
)

# Shared UI chrome palette for HTML exports (dashboard vars and Plotly
# hoverlabels must agree so tooltips look native to the page).
_UI_PALETTES: dict[str, dict[str, str]] = {
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

# One font stack for the dashboard stylesheet and every Plotly hover tooltip.
# Injected into both so the tooltip never falls back to Plotly's Arial default
# while the page around it renders in a different face.
UI_FONT_FAMILY = '"IBM Plex Sans", "Source Sans 3", "Segoe UI", sans-serif'
HOVER_LABEL_FONT_SIZE = 12

# Dashboard labels sit on the plot surface rather than on panel chrome, so they
# are drawn from the theme text color at reduced alpha. Light backgrounds need
# far less fading than dark ones before the text stops reading as a label.
DASHBOARD_LABEL_TEXT_ALPHA: dict[str, float] = {"dark": 0.72, "light": 0.9}


def _theme_color_scheme(theme_obj: Theme) -> str:
    """Resolve the CSS ``color-scheme`` value a theme should declare.

    :param Theme theme_obj: Active visualization theme.
    :return str: ``"dark"`` or ``"light"``.
    """
    return "dark" if theme_obj.name in {"dark", "solarized"} else "light"


def theme_hover_label(theme_obj: Theme) -> PlotlyHoverLabel:
    """Build the Plotly ``hoverlabel`` spec shared by every hoverable trace.

    Tooltips are page chrome, not data, so they take the dashboard panel
    surface, the panel text color, the accent as a border, and the page font
    stack. The dashboard rebuilds its figure in JavaScript from the same
    values, injected as ``__HOVER_LABEL_JSON__``, so the two cannot drift.

    :param Theme theme_obj: Active visualization theme.
    :return PlotlyHoverLabel: The themed Plotly hoverlabel specification.
    """
    palette = _UI_PALETTES[_theme_color_scheme(theme_obj)]
    return {
        "align": "left",
        "bgcolor": palette["panel_bg"],
        "bordercolor": palette["accent"],
        "font": {
            "color": palette["text_primary"],
            "family": UI_FONT_FAMILY,
            "size": HOVER_LABEL_FONT_SIZE,
        },
    }


def theme_label_text_alpha(theme_obj: Theme) -> float:
    """Resolve the dashboard node-label alpha for a theme.

    :param Theme theme_obj: Active visualization theme.
    :return float: Alpha applied to the theme text color for graph labels.
    """
    return DASHBOARD_LABEL_TEXT_ALPHA[_theme_color_scheme(theme_obj)]


# as fallback. The tooltip's job is answering "why is this paper here".
_HOVER_RELATION_LABELS: dict[str, str] = {
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

    This mirrors the dashboard runtime because imported offline payloads rebuild
    in the browser without running Python.

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
    graph: PaperGraph,
    node_ids: list[Hashable],
    pos: Mapping[Hashable, Iterable[float]],
) -> set[Hashable]:
    """Select prominent dashboard labels without crowding one graph region.

    :param PaperGraph graph: Graph containing node ranking attributes.
    :param list[Hashable] node_ids: Deterministically ordered node identifiers.
    :param Mapping[Hashable, Iterable[float]] pos: Normalized node positions.
    :return set[Hashable]: Node identifiers whose labels should remain visible.
    """

    def _rank(node_id: Hashable) -> tuple[int, int, int, str]:
        """Build a stable seed/citation/year priority tuple.

        :param Hashable node_id: Candidate node identifier.
        :return tuple[int, int, int, str]: Sort key for label priority.
        """
        attrs = graph.nodes[node_id]
        citations = coerce_citation_count(attrs.get("citation_count", 0))
        year = max(coerce_publication_year(attrs.get("year", 0)), 0)
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


def _rgb_tuple_to_hex(color: tuple[float, float, float]) -> str:
    """Convert RGB tuple (0-1) to hex string.

    :param tuple color: RGB triple in [0, 1] space.
    :return str: HTML hex color code.
    """
    r, g, b = (int(max(0, min(1, channel)) * 255) for channel in color)
    return f"#{r:02x}{g:02x}{b:02x}"


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
        if not isinstance(color, Iterable):
            values = [1.0, 1.0, 1.0]
        else:
            try:
                values = list(color)
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


def _inject_darkreader_lock(
    path: Path | str,
    color_scheme: str = "light",
    *,
    title: str | None = None,
    style: str | None = None,
    strip_remote_assets: bool = False,
) -> None:
    """Add theme metadata and optional page chrome to an HTML document head.

    The pass is idempotent and leaves the file unchanged when it is unreadable,
    has no ``<head>``, or already contains the Dark Reader lock. All edits stay
    in the head; optional asset removal strips remote stylesheets and scripts.

    :param Path | str path: HTML file to rewrite in place (no-op when it has no
        ``<head>`` tag or already carries the lock).
    :param str color_scheme: Declared scheme for the export, ``dark`` or
        ``light``.
    :param Optional[str] title: Plain-text browser tab title; HTML-escaped here,
        replacing an existing head ``<title>`` when the generator wrote one.
    :param Optional[str] style: CSS injected as the first head stylesheet, so
        generator styles later in the document can still override it.
    :param bool strip_remote_assets: Whether to remove head comments and remote
        stylesheet/script tags, for exports that must open offline.
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

    head_start = content.find("<head>")
    head_end = content.find("</head>", head_start)
    head = content if head_end < 0 else content[:head_end]
    tail = "" if head_end < 0 else content[head_end:]

    if strip_remote_assets:
        head = _HEAD_COMMENT_PATTERN.sub("", head)
        head = _HEAD_REMOTE_ASSET_PATTERN.sub("", head)

    injected = [
        DARKREADER_LOCK_META,
        f'<meta name="color-scheme" content="{scheme}" />',
    ]
    if title is not None:
        title_tag = f"<title>{html.escape(title)}</title>"
        head, replaced = _HEAD_TITLE_PATTERN.subn(lambda _match: title_tag, head, 1)
        if not replaced:
            injected.append(title_tag)
    if style:
        injected.append(f"<style>{style}</style>")

    head = head.replace("<head>", f"<head>{''.join(injected)}", 1)
    resolved_path.write_text(head + tail, encoding="utf-8")
