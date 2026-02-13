"""
Unified graph visualization for CiteMesh.

This module provides a single implementation of the CiteMesh-style
visualization that all strategies can use, eliminating code duplication.
"""

import logging
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from citemesh.core import VIZ_CONFIG

from .themes import Theme, get_theme

logger = logging.getLogger(__name__)
MAX_TITLE_CHARS = 50


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
    :return str: Stable hex suffix used in auto-generated output directories.
    """
    return sha1(seed_id.encode("utf-8")).hexdigest()[:length]


def _choose_metadata_anchor(
    pos: Dict[str, np.ndarray],
) -> Tuple[float, float, str, str]:
    """
    Choose which corner to place the metadata box in based on node density.

    :param Dict[str, np.ndarray] pos: Mapping of node -> position array
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


def add_metadata_box(
    ax: plt.Axes, metadata: Dict[str, Any], pos: Dict[str, np.ndarray], theme: Theme
) -> None:
    """
    Render a small metadata block in the plot corner.

    :param plt.Axes ax: Matplotlib axes
    :param Dict[str, Any] metadata: Dictionary of metadata key/value pairs
    :param Dict[str, np.ndarray] pos: Node position map.
    :param Theme theme: Active theme (used for text colors).
    :return None: Draws metadata box directly to axes.
    """
    lines = []
    label_map = {
        "paper_id": "Query",
        "seed_id": "Seed",
        "strategy": "Strategy",
        "timestamp": "Generated",
        "nodes": "Nodes",
        "edges": "Edges",
    }

    for key, label in label_map.items():
        value = metadata.get(key)
        if value is not None:
            lines.append(f"{label}: {value}")

    # Include any extra metadata fields not in the predefined map
    for key, value in metadata.items():
        if key not in label_map and value is not None:
            lines.append(f"{key.replace('_', ' ').title()}: {value}")

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
            boxstyle="round,pad=0.4", facecolor=facecolor, alpha=0.8, linewidth=0
        ),
        color=text_color,
    )


def compute_node_sizes(graph: nx.Graph) -> List[float]:
    """
    Compute node sizes with extreme variation matching CiteMesh style.

    :param nx.Graph graph: NetworkX graph with paper nodes
    :return List[float]: List of sizes (in square pixels) for each node
    """
    nodes = list(graph.nodes())
    sizes = []
    seed_nodes = {node for node in nodes if graph.nodes[node].get("is_seed")}
    sorted_nodes = sorted(
        nodes, key=lambda n: graph.nodes[n].get("citation_count", 0), reverse=True
    )
    rank_of = {node: rank for rank, node in enumerate(sorted_nodes)}

    for node in nodes:
        rank = rank_of.get(node, len(nodes))
        citation_count = graph.nodes[node].get("citation_count", 0)

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
    nodes = list(graph.nodes())
    years = [graph.nodes[n].get("year") for n in nodes if graph.nodes[n].get("year")]
    if years:
        min_year = min(years)
        max_year = max(years)
    else:
        min_year = 2000
        max_year = datetime.now().year

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
) -> Dict[str, np.ndarray]:
    """
    Compute force-directed layout with organic clustering.

    :param nx.Graph graph: NetworkX graph
    :param int iterations: Number of iterations for spring layout
    :param Optional[int] layout_seed: Optional seed for deterministic layout perturbations/fallback.
    :return Dict[str, np.ndarray]: Dictionary mapping node IDs to (x, y) positions.
    """
    try:
        pos = nx.kamada_kawai_layout(
            graph,
            weight="weight",
            scale=VIZ_CONFIG.layout_scale,
            center=VIZ_CONFIG.layout_center,
        )
    except Exception as e:
        logger.warning(f"Kamada-Kawai failed ({e}), using spring layout")
        k_value = VIZ_CONFIG.spring_k_factor / np.sqrt(graph.number_of_nodes())
        pos = nx.spring_layout(
            graph,
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


def draw_edges(ax: plt.Axes, graph: nx.Graph, pos: Dict, theme: Theme) -> None:
    """
    Draw edges with varying thickness and opacity based on weight.

    :param plt.Axes ax: Matplotlib axes
    :param nx.Graph graph: NetworkX graph
    :param Dict pos: Node positions dictionary
    :param Theme theme: Theme palette for edge colors.
    :return None: Draws all edges onto the axes.
    """
    for n1, n2, data in graph.edges(data=True):
        weight = data.get("weight", 0.1)
        p1 = pos[n1]
        p2 = pos[n2]

        # Style based on weight
        alpha = min(VIZ_CONFIG.edge_alpha_max, weight * VIZ_CONFIG.edge_alpha_max)
        alpha = max(VIZ_CONFIG.edge_alpha_min, alpha)
        width = max(VIZ_CONFIG.edge_width_min, weight * VIZ_CONFIG.edge_width_max)

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=theme.edge_color,
            alpha=alpha,
            linewidth=width,
            zorder=1,
        )


def draw_nodes(
    ax: plt.Axes,
    graph: nx.Graph,
    pos: Dict,
    sizes: List[float],
    colors: List[Tuple[float, float, float]],
    theme: Theme,
) -> None:
    """
    Draw nodes with computed sizes and colors.

    :param plt.Axes ax: Matplotlib axes
    :param nx.Graph graph: NetworkX graph
    :param Dict pos: Node positions dictionary
    :param List[float] sizes: List of node sizes
    :param List[Tuple[float, float, float]] colors: List of node colors (RGB tuples)
    :param Theme theme: Theme palette for edge outlines.
    :return None: Draws all nodes onto the axes.
    """
    nodes = list(graph.nodes())

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
    ax: plt.Axes, graph: nx.Graph, pos: Dict, seed_id: str, theme: Theme
) -> None:
    """
    Draw paper labels in "Author, Year" format.

    :param plt.Axes ax: Matplotlib axes
    :param nx.Graph graph: NetworkX graph
    :param Dict pos: Node positions dictionary
    :param str seed_id: ID of seed paper (gets bold label)
    :param Theme theme: Theme palette for text color.
    :return None: Draws all node labels.
    """

    def _shorten_title(title: str, max_chars: int = 34) -> str:
        """
        Shorten long seed labels to keep static plots readable.

        :param str title: Full seed paper title.
        :param int max_chars: Maximum label width before truncation.
        :return str: Label-safe title.
        """
        if len(title) <= max_chars:
            return title
        return f"{title[: max_chars - 3].rstrip()}..."

    for node in graph.nodes():
        p = pos[node]

        # Seed paper gets larger, bold label
        is_seed = node == seed_id
        if is_seed:
            title = graph.nodes[node].get("title", "Seed paper")
            label = _shorten_title(title)
            fontsize = 9
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

        fontweight = "bold" if is_seed else VIZ_CONFIG.font_weight

        ax.annotate(
            label,
            xy=p,
            xytext=(0, -3),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=fontsize,
            fontweight=fontweight,
            color=theme.text_color,
        )


def visualize_graph(
    graph: nx.Graph,
    seed_id: str,
    output_path: Path,
    iterations: int = 100,
    dpi: int = None,
    metadata: Optional[Dict[str, Any]] = None,
    theme_name: str = "light",
    layout: Optional[Dict[str, np.ndarray]] = None,
    layout_seed: Optional[int] = None,
) -> None:
    """
    Create CiteMesh visualization.

    :param nx.Graph graph: NetworkX graph with paper nodes
    :param str seed_id: ID of the seed paper
    :param Path output_path: Path for output PNG file
    :param int iterations: Number of layout iterations (higher = better quality)
    :param int dpi: Output resolution (defaults to config value).
    :param Optional[Dict[str, Any]] metadata: Optional info to annotate on the figure (auto-positioned) Visual Encodings: - Node size: Citation count + importance ranking (80-2500 pixels) - Node color: Smooth gradient by year (light → dark) - Edge thickness: Proportional to similarity weight - Edge opacity: Based on connection strength - Layout: Kamada-Kawai with organic perturbations
    :param str theme_name: Name of theme to render.
    :param Optional[Dict[str, np.ndarray]] layout: Optional precomputed layout to reuse.
    :param Optional[int] layout_seed: Optional seed used when computing layout internally.
    :return None: Writes output image to the given path.
    """
    if dpi is None:
        dpi = VIZ_CONFIG.dpi

    theme = get_theme(theme_name)

    # Compute layout
    pos = (
        layout
        if layout is not None
        else compute_layout(graph, iterations, layout_seed=layout_seed)
    )

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
    draw_labels(ax, graph, pos, seed_id, theme)

    # Add title
    title = graph.nodes[seed_id].get("title", "Unknown")[:60]
    ax.set_title(
        f"CiteMesh Visualization: {title}...",
        fontsize=14,
        pad=20,
        color=theme.text_color,
    )

    # Add metadata annotation if requested **after** title so we can reference it
    if metadata:
        add_metadata_box(ax, metadata, pos, theme)

    # Save figure
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=theme.background,
    )
    plt.close()

    logger.info(f"Visualization saved to {output_path}")


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
    paper_dir = output_dir / f"{_filename_safe(title)}-{_seed_suffix(seed_id)}"
    paper_dir.mkdir(parents=True, exist_ok=True)

    basename = _filename_safe(strategy, max_chars=32) if strategy else "graph"
    return paper_dir / f"{basename}.png"
