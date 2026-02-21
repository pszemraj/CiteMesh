"""
Unified graph visualization for CiteMesh.

This module provides a single implementation of the CiteMesh-style
visualization that all strategies can use, eliminating code duplication.
"""

import hashlib
import logging
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, Hashable, List, Mapping, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from citemesh.core import VIZ_CONFIG

from .ordering import (
    canonicalize_graph_for_layout,
    ordered_edges_with_data,
    ordered_nodes,
)
from .themes import Theme, get_theme

logger = logging.getLogger(__name__)
MAX_TITLE_CHARS = 40
MISSING_YEAR_FALLBACK_MIN = 2000
MISSING_YEAR_FALLBACK_MAX = 2001
KK_LAYOUT_DISTANCE_ATTR = "layout_distance"
KK_LAYOUT_DISTANCE_EPSILON = 1e-6
LAYOUT_PADDING_RATIO = 0.1
LABEL_COLLISION_MIN_DISTANCE = 0.075
LABEL_COLLISION_MAX_DISTANCE = 0.14
METADATA_VALUE_MAX_CHARS = 64


def _citation_count(attrs: Mapping[str, Any]) -> int:
    """Normalize citation count values for deterministic ranking.

    :param Mapping[str, Any] attrs: Raw node attributes mapping.
    :return int: Non-negative citation count value.
    """
    raw = attrs.get("citation_count", 0)
    if isinstance(raw, bool) or raw is None:
        return 0
    try:
        return max(int(raw), 0)
    except (TypeError, ValueError):
        return 0


def _filename_safe(text: str, max_chars: int = MAX_TITLE_CHARS) -> str:
    """Create a filesystem-safe slug from input text.

    :param str text: Raw text value.
    :param int max_chars: Maximum slug length.
    :return str: Safe slug using lowercase alnum/hyphen tokens.
    """
    normalized = text.lower()
    normalized = "".join(c if c.isalnum() or c in " -" else "" for c in normalized)
    slug = "-".join(normalized.split())[:max_chars].strip("-")
    return slug or "graph"


def _seed_suffix(seed_id: str, length: int = 8) -> str:
    """Build a short, stable suffix from the seed identifier.

    :param str seed_id: Seed paper identifier.
    :param int length: Number of digest characters to keep.
    :return str: Stable hex suffix used in output directory naming.
    """
    return hashlib.sha256(seed_id.encode("utf-8")).hexdigest()[:length]


def _output_dir_name(title: str, seed_id: str, max_chars: int = MAX_TITLE_CHARS) -> str:
    """Build output directory name with stable seed suffix under truncation.

    :param str title: Seed paper title used for the human-readable slug prefix.
    :param str seed_id: Canonical seed identifier used for stable hash suffix.
    :param int max_chars: Maximum total directory-name length.
    :return str: Filesystem-safe directory name containing title slug and hash suffix.
    """
    suffix = f"-{_seed_suffix(seed_id)}"
    title_budget = max_chars - len(suffix)
    title_budget = max(1, title_budget)
    return f"{_filename_safe(title, max_chars=title_budget)}{suffix}"


def _similarity_to_layout_distance(raw_similarity: object) -> float:
    """Map similarity-style edge weights to positive layout distances for KK.

    :param object raw_similarity: Raw edge similarity value.
    :return float: Strictly positive distance used by Kamada-Kawai.
    """
    try:
        similarity = float(raw_similarity)
    except (TypeError, ValueError):
        similarity = 0.0

    if not np.isfinite(similarity):
        similarity = 0.0

    similarity = max(similarity, 0.0)
    return 1.0 / (KK_LAYOUT_DISTANCE_EPSILON + similarity)


def _choose_metadata_anchor(
    pos: Dict[Hashable, np.ndarray],
) -> Tuple[float, float, str, str]:
    """
    Choose which corner to place the metadata box in based on node density.

    :param Dict[Hashable, np.ndarray] pos: Mapping of node -> position array
    :return Tuple[float, float, str, str]: Tuple of (x, y, horizontal_alignment, vertical_alignment) in axes coords.
    """
    if not pos:
        return 0.02, 0.02, "left", "bottom"

    coords = np.array(list(pos.values()))
    xs = coords[:, 0]
    ys = coords[:, 1]

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    norm_x = (xs - min_x) / span_x
    norm_y = (ys - min_y) / span_y

    corners = {
        "lower_left": ((0.05, 0.05), "left", "bottom"),
        "lower_right": ((0.95, 0.05), "right", "bottom"),
        "upper_left": ((0.05, 0.95), "left", "top"),
        "upper_right": ((0.95, 0.95), "right", "top"),
    }

    window = 0.22  # area around corner to gauge crowding
    scores: Dict[str, float] = {}

    for name, ((cx, cy), ha, va) in corners.items():
        dist_x = np.abs(norm_x - cx)
        dist_y = np.abs(norm_y - cy)
        crowded = np.sum((dist_x < window) & (dist_y < window))
        avg_distance = np.mean(dist_x + dist_y)
        scores[name] = crowded + (1.0 - avg_distance)

    best = min(scores, key=scores.get)
    (x, y), ha, va = corners[best]
    return x, y, ha, va


def _normalize_layout_positions(
    pos: Dict[Hashable, np.ndarray], padding_ratio: float = LAYOUT_PADDING_RATIO
) -> Dict[Hashable, np.ndarray]:
    """Normalize layout positions to a centered square viewport.

    :param Dict[Hashable, np.ndarray] pos: Raw layout map from NetworkX.
    :param float padding_ratio: Fractional interior padding around graph extents.
    :return Dict[Hashable, np.ndarray]: Centered and scaled layout in approximately [-1, 1].
    """
    if not pos:
        return {}

    keys = list(pos.keys())
    coords = np.array([np.asarray(pos[key], dtype=float) for key in keys], dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        return {key: np.asarray(value, dtype=float) for key, value in pos.items()}

    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    center_xy = (min_xy + max_xy) * 0.5
    span_xy = max_xy - min_xy
    max_span = float(np.max(span_xy))
    target_half_extent = max(1e-6, 1.0 - float(padding_ratio))

    if max_span <= 1e-9:
        normalized = np.zeros_like(coords)
    else:
        normalized = (coords - center_xy) / (max_span * 0.5)
        normalized *= target_half_extent

    return {
        key: np.array([float(normalized[idx, 0]), float(normalized[idx, 1])])
        for idx, key in enumerate(keys)
    }


def add_metadata_box(
    ax: plt.Axes,
    metadata: Dict[str, Any],
    pos: Dict[Hashable, np.ndarray],
    theme: Theme,
) -> None:
    """
    Render a small metadata block in the plot corner.

    :param plt.Axes ax: Matplotlib axes
    :param Dict[str, Any] metadata: Dictionary of metadata key/value pairs
    :param Dict[Hashable, np.ndarray] pos: Node position map.
    :param Theme theme: Active theme (used for text colors).
    :return None: Draws metadata box directly to axes.
    """
    lines = []
    label_map = {
        "paper_id": "Query",
        "strategy": "Strategy",
        "nodes": "Nodes",
        "edges": "Edges",
        "theme": "Theme",
        "timestamp": "Generated",
    }

    for key, label in label_map.items():
        value = metadata.get(key)
        if value is not None:
            compact = " ".join(str(value).split())
            if len(compact) > METADATA_VALUE_MAX_CHARS:
                compact = f"{compact[: METADATA_VALUE_MAX_CHARS - 3]}..."
            lines.append(f"{label}: {compact}")

    if not lines:
        return

    text = "\n".join(lines)

    x, y, ha, va = _choose_metadata_anchor(pos)

    facecolor = "#2d3748" if theme.name == "dark" else "#ffffff"
    text_color = "#f0f0f0" if theme.name == "dark" else theme.text_color

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=8,
        ha=ha,
        va=va,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor=facecolor, alpha=0.55, linewidth=0
        ),
        color=text_color,
        zorder=10,
    )


def compute_node_sizes(graph: nx.Graph) -> List[float]:
    """
    Compute node sizes with extreme variation matching CiteMesh style.

    :param nx.Graph graph: NetworkX graph with paper nodes
    :return List[float]: List of sizes (in square pixels) for each node
    """
    nodes = ordered_nodes(graph)
    sizes = []
    seed_nodes = {node for node in nodes if graph.nodes[node].get("is_seed")}
    sorted_nodes = sorted(
        nodes,
        key=lambda node: (-_citation_count(graph.nodes[node]), str(node)),
    )
    rank_of = {node: rank for rank, node in enumerate(sorted_nodes)}

    for node in nodes:
        rank = rank_of.get(node, len(nodes))
        citation_count = _citation_count(graph.nodes[node])

        if node in seed_nodes:
            # Seed paper gets special treatment
            if rank < 3:  # Also top-cited
                size = VIZ_CONFIG.seed_size
            else:
                size = 1000  # Still visible but not dominant
        elif rank == 0 and not graph.nodes[node].get("is_seed"):
            # Highest cited non-seed
            size = VIZ_CONFIG.max_non_seed_size
        elif rank < 3:
            # Top 3 papers
            base, increment = VIZ_CONFIG.size_tiers["top_3"]
            size = base + (3 - rank) * increment
        elif rank < 8:
            # Next 5 papers
            base, increment = VIZ_CONFIG.size_tiers["top_8"]
            size = base + (8 - rank) * increment
        elif rank < 15:
            # Next 7 papers
            base, increment = VIZ_CONFIG.size_tiers["top_15"]
            size = base + (15 - rank) * increment
        else:
            # Remaining papers
            size = VIZ_CONFIG.min_size

        # Add citation bonus (log scale)
        citation_bonus = np.log10(citation_count + 1) * 100
        size += citation_bonus

        sizes.append(size)

    return sizes


def compute_node_colors(
    graph: nx.Graph, seed_id: str, theme: Theme
) -> Tuple[List[Tuple[float, float, float]], int, int]:
    """
    Compute smooth color gradient by publication year.

    :param nx.Graph graph: NetworkX graph with paper nodes
    :param str seed_id: ID of the seed paper (gets special color)
    :param Theme theme: Theme palette used for interpolation.
    :return Tuple[List[Tuple[float, float, float]], int, int]: Tuple of (color_list, min_year, max_year)
    """
    nodes = ordered_nodes(graph)
    years = [graph.nodes[n].get("year") for n in nodes if graph.nodes[n].get("year")]
    if years:
        min_year = min(years)
        max_year = max(years)
    else:
        min_year = MISSING_YEAR_FALLBACK_MIN
        max_year = MISSING_YEAR_FALLBACK_MAX

    colors = []
    for node in nodes:
        # Seed paper gets special color
        if node == seed_id:
            colors.append(theme.seed_color)
            continue

        year = graph.nodes[node].get("year")
        if year is None:
            norm = 0.5
        elif max_year == min_year:
            norm = 0.5
        else:
            norm = (year - min_year) / (max_year - min_year)
        color = theme.interpolate(norm)
        colors.append(color)

    return colors, min_year, max_year


def compute_layout(
    graph: nx.Graph,
    iterations: int = 100,
    layout_seed: Optional[int] = None,
) -> Dict[Hashable, np.ndarray]:
    """
    Compute force-directed layout with organic clustering.

    :param nx.Graph graph: NetworkX graph
    :param int iterations: Number of iterations for spring layout
    :param Optional[int] layout_seed: Optional seed for deterministic layout perturbations/fallback.
    :return Dict[Hashable, np.ndarray]: Dictionary mapping node IDs to (x, y) positions.
    """
    canonical_graph = canonicalize_graph_for_layout(graph)
    for _, _, attrs in canonical_graph.edges(data=True):
        # Kamada-Kawai interprets weights as path lengths (distances), not
        # affinities; convert similarity weights so stronger links are shorter.
        attrs[KK_LAYOUT_DISTANCE_ATTR] = _similarity_to_layout_distance(
            attrs.get("weight", 0.0)
        )

    try:
        pos = nx.kamada_kawai_layout(
            canonical_graph,
            weight=KK_LAYOUT_DISTANCE_ATTR,
            scale=VIZ_CONFIG.layout_scale,
            center=VIZ_CONFIG.layout_center,
        )
    except Exception as e:
        logger.warning(f"Kamada-Kawai failed ({e}), using spring layout")
        k_value = VIZ_CONFIG.spring_k_factor / np.sqrt(
            canonical_graph.number_of_nodes()
        )
        pos = nx.spring_layout(
            canonical_graph,
            k=k_value,
            iterations=iterations,
            seed=42 if layout_seed is None else layout_seed,
            weight="weight",
            scale=VIZ_CONFIG.layout_scale,
            center=VIZ_CONFIG.layout_center,
        )

    # Add small deterministic perturbations for visual separation
    rng = np.random.default_rng(0 if layout_seed is None else layout_seed)
    for node in sorted(pos, key=str):
        pos[node] += rng.normal(0, VIZ_CONFIG.perturbation_std, 2)

    return pos


def draw_edges(
    ax: plt.Axes, graph: nx.Graph, pos: Dict[Hashable, np.ndarray], theme: Theme
) -> None:
    """
    Draw edges with varying thickness and opacity based on weight.

    :param plt.Axes ax: Matplotlib axes
    :param nx.Graph graph: NetworkX graph
    :param Dict[Hashable, np.ndarray] pos: Node positions dictionary.
    :param Theme theme: Theme palette for edge colors.
    :return None: Draws all edges onto the axes.
    """
    for edge_index, (n1, n2, data) in enumerate(ordered_edges_with_data(graph)):
        weight = data.get("weight", 0.1)
        p1 = pos[n1]
        p2 = pos[n2]

        # Style based on weight
        alpha = min(VIZ_CONFIG.edge_alpha_max, weight * VIZ_CONFIG.edge_alpha_max)
        alpha = max(VIZ_CONFIG.edge_alpha_min, alpha)
        width = max(VIZ_CONFIG.edge_width_min, weight * VIZ_CONFIG.edge_width_max)

        direction = -1.0 if edge_index % 2 else 1.0
        curve = mpatches.FancyArrowPatch(
            (float(p1[0]), float(p1[1])),
            (float(p2[0]), float(p2[1])),
            connectionstyle=f"arc3,rad={0.2 * direction}",
            color=theme.edge_color,
            alpha=alpha,
            linewidth=width,
            zorder=1,
            arrowstyle="-",
        )
        ax.add_patch(curve)


def draw_nodes(
    ax: plt.Axes,
    graph: nx.Graph,
    pos: Dict[Hashable, np.ndarray],
    sizes: List[float],
    colors: List[Tuple[float, float, float]],
    theme: Theme,
) -> None:
    """
    Draw nodes with computed sizes and colors.

    :param plt.Axes ax: Matplotlib axes
    :param nx.Graph graph: NetworkX graph
    :param Dict[Hashable, np.ndarray] pos: Node positions dictionary.
    :param List[float] sizes: List of node sizes
    :param List[Tuple[float, float, float]] colors: List of node colors (RGB tuples)
    :param Theme theme: Theme palette for edge outlines.
    :return None: Draws all nodes onto the axes.
    """
    nodes = ordered_nodes(graph)

    for i, node in enumerate(nodes):
        p = pos[node]
        ax.scatter(
            p[0],
            p[1],
            s=sizes[i],
            c=[colors[i]],
            alpha=0.9,
            edgecolors=theme.text_color,
            linewidth=2,
            zorder=2,
        )


def draw_labels(
    ax: plt.Axes,
    graph: nx.Graph,
    pos: Dict[Hashable, np.ndarray],
    seed_id: str,
    theme: Theme,
    sizes: Optional[List[float]] = None,
) -> None:
    """
    Draw paper labels in "Author, Year" format.

    :param plt.Axes ax: Matplotlib axes
    :param nx.Graph graph: NetworkX graph
    :param Dict[Hashable, np.ndarray] pos: Node positions dictionary.
    :param str seed_id: ID of seed paper (gets bold label)
    :param Theme theme: Theme palette for text color.
    :param Optional[List[float]] sizes: Optional node-size list aligned with ``ordered_nodes(graph)``.
    :return None: Draws all node labels.
    """

    def _wrap_title(title: str, width: int = 34) -> str:
        """Wrap long seed labels without dropping any title text.

        :param str title: Full seed paper title.
        :param int width: Approximate character width for line wrapping.
        :return str: Wrapped title label.
        """
        title = " ".join((title or "").split())
        if not title:
            return "Seed paper"
        return textwrap.fill(title, width=width, break_long_words=False)

    ordered = ordered_nodes(graph)
    size_map: Dict[Hashable, float] = {}
    if sizes is not None and len(sizes) == len(ordered):
        size_map = {node: float(sizes[idx]) for idx, node in enumerate(ordered)}

    candidate_nodes = sorted(
        ordered,
        key=lambda node: (
            0 if node == seed_id else 1,
            -_citation_count(graph.nodes[node]),
            str(node),
        ),
    )

    placed: List[np.ndarray] = []
    for node in candidate_nodes:
        p = pos[node]

        # Seed paper gets larger, bold label
        is_seed = node == seed_id
        if is_seed:
            title = graph.nodes[node].get("title", "Seed paper")
            label = _wrap_title(title)
            fontsize = 9
            xytext = (0, 8)
            vertical_alignment = "bottom"
            label_bbox = dict(
                boxstyle="round,pad=0.2",
                facecolor=theme.background,
                alpha=0.75,
                linewidth=0,
            )
        else:
            # Extract author surname
            authors = graph.nodes[node].get("authors", [])
            if authors and authors[0]:
                last_name = authors[0].split()[-1]
            else:
                last_name = "Unknown"

            year = graph.nodes[node].get("year")
            year_label = "n.d." if year is None else str(year)
            label = f"{last_name}, {year_label}"
            fontsize = VIZ_CONFIG.font_size
            xytext = (0, -3)
            vertical_alignment = "top"
            label_bbox = None

        fontweight = "bold" if is_seed else VIZ_CONFIG.font_weight
        if not is_seed:
            node_size = size_map.get(node, float(VIZ_CONFIG.min_size))
            scaled = min(
                max(node_size / max(float(VIZ_CONFIG.seed_size), 1.0), 0.0), 1.0
            )
            min_distance = (
                LABEL_COLLISION_MIN_DISTANCE
                + (LABEL_COLLISION_MAX_DISTANCE - LABEL_COLLISION_MIN_DISTANCE) * scaled
            )
            if any(
                math.dist((float(p[0]), float(p[1])), (float(prev[0]), float(prev[1])))
                < min_distance
                for prev in placed
            ):
                continue

        ax.annotate(
            label,
            xy=p,
            xytext=xytext,
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=fontsize,
            fontweight=fontweight,
            color=theme.text_color,
            bbox=label_bbox,
        )
        placed.append(np.asarray(p, dtype=float))


def visualize_graph(
    graph: nx.Graph,
    seed_id: str,
    output_path: Path,
    iterations: int = 100,
    dpi: int = None,
    metadata: Optional[Dict[str, Any]] = None,
    theme_name: str = "light",
    layout: Optional[Dict[Hashable, np.ndarray]] = None,
    layout_seed: Optional[int] = None,
) -> None:
    """
    Create CiteMesh visualization.

    :param nx.Graph graph: NetworkX graph with paper nodes
    :param str seed_id: ID of the seed paper
    :param Path output_path: Path for output PNG file
    :param int iterations: Iterations used only if spring fallback layout is triggered.
    :param int dpi: Output resolution (defaults to config value).
    :param Optional[Dict[str, Any]] metadata: Optional info to annotate on the figure (auto-positioned).
    :param str theme_name: Name of theme to render.
    :param Optional[Dict[Hashable, np.ndarray]] layout: Optional precomputed layout to
        reuse.
    :param Optional[int] layout_seed: Optional seed used when computing layout internally.
    :return None: Writes output image to the given path.
    """
    if dpi is None:
        dpi = VIZ_CONFIG.dpi

    theme = get_theme(theme_name)

    # Compute layout
    raw_pos = (
        layout
        if layout is not None
        else compute_layout(graph, iterations, layout_seed=layout_seed)
    )
    pos = _normalize_layout_positions(raw_pos)

    # Compute visual properties
    sizes = compute_node_sizes(graph)
    colors, _, _ = compute_node_colors(graph, seed_id, theme)

    # Create figure
    fig, ax = plt.subplots(figsize=VIZ_CONFIG.figure_size, facecolor=theme.background)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(theme.background)
    ax.set_facecolor(theme.background)

    # Draw graph components
    draw_edges(ax, graph, pos, theme)
    draw_nodes(ax, graph, pos, sizes, colors, theme)
    draw_labels(ax, graph, pos, seed_id, theme, sizes=sizes)

    # Add title
    title = graph.nodes[seed_id].get("title", "Unknown")
    wrapped_title = textwrap.fill(
        " ".join(str(title).split()),
        width=72,
        break_long_words=False,
    )
    ax.set_title(
        f"CiteMesh Visualization: {wrapped_title}",
        fontsize=14,
        pad=20,
        color=theme.text_color,
    )

    # Add metadata annotation if requested **after** title so we can reference it
    if metadata:
        add_metadata_box(ax, metadata, pos, theme)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02)

    # Save figure
    plt.savefig(
        output_path,
        dpi=dpi,
        facecolor=theme.background,
    )
    plt.close()


def generate_output_path(
    graph: nx.Graph, seed_id: str, output_dir: Path = Path("out"), strategy: str = ""
) -> Path:
    """
    Generate auto-named output path from paper title.

    :param nx.Graph graph: NetworkX graph
    :param str seed_id: ID of seed paper
    :param Path output_dir: Output directory
    :param str strategy: Optional strategy suffix used in filename.
    :return Path: Path object for output file
    """
    title = graph.nodes[seed_id].get("title", "graph")
    paper_dir = output_dir / _output_dir_name(title=title, seed_id=seed_id)
    paper_dir.mkdir(parents=True, exist_ok=True)

    basename = _filename_safe(strategy, max_chars=32) if strategy else "graph"
    return paper_dir / f"{basename}.png"
