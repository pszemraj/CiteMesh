"""
Unified graph visualization for CiteMesh.

This module provides a single implementation of the Connected Papers-style
visualization that all strategies can use, eliminating code duplication.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from citemesh.config import VIZ_CONFIG

logger = logging.getLogger(__name__)


def compute_node_sizes(graph: nx.Graph) -> List[float]:
    """
    Compute node sizes with extreme variation matching Connected Papers style.

    Args:
        graph: NetworkX graph with paper nodes

    Returns:
        List of sizes (in square pixels) for each node
    """
    nodes = list(graph.nodes())
    sizes = []

    # Sort nodes by citation count to identify top papers
    sorted_nodes = sorted(
        nodes, key=lambda n: graph.nodes[n].get("citation_count", 0), reverse=True
    )

    for node in nodes:
        rank = sorted_nodes.index(node)
        citation_count = graph.nodes[node].get("citation_count", 0)

        if graph.nodes[node].get("is_seed"):
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
            size = VIZ_CONFIG.min_size + np.random.randint(0, 100)

        # Add citation bonus (log scale)
        citation_bonus = np.log10(citation_count + 1) * 100
        size += citation_bonus

        sizes.append(size)

    return sizes


def compute_node_colors(
    graph: nx.Graph, seed_id: str
) -> Tuple[List[Tuple[float, float, float]], int, int]:
    """
    Compute smooth color gradient by publication year.

    Args:
        graph: NetworkX graph with paper nodes
        seed_id: ID of the seed paper (gets special color)

    Returns:
        Tuple of (color_list, min_year, max_year)
    """
    nodes = list(graph.nodes())
    years = [graph.nodes[n].get("year", 2020) for n in nodes]
    min_year = min(years)
    max_year = max(years)

    colors = []
    for node in nodes:
        # Seed paper gets special red color
        if node == seed_id:
            colors.append(VIZ_CONFIG.seed_color)
            continue

        year = graph.nodes[node].get("year", 2020)
        color = VIZ_CONFIG.compute_node_color(year, min_year, max_year)
        colors.append(color)

    return colors, min_year, max_year


def compute_layout(graph: nx.Graph, iterations: int = 100) -> Dict[str, np.ndarray]:
    """
    Compute force-directed layout with organic clustering.

    Tries Kamada-Kawai first (better clustering), falls back to spring layout.

    Args:
        graph: NetworkX graph
        iterations: Number of iterations for spring layout

    Returns:
        Dictionary mapping node IDs to (x, y) positions
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
            seed=42,
            weight="weight",
            scale=VIZ_CONFIG.layout_scale,
            center=VIZ_CONFIG.layout_center,
        )

    # Add small random perturbations for organic look
    for node in pos:
        pos[node] += np.random.normal(0, VIZ_CONFIG.perturbation_std, 2)

    return pos


def draw_edges(ax: plt.Axes, graph: nx.Graph, pos: Dict) -> None:
    """
    Draw edges with varying thickness and opacity based on weight.

    Args:
        ax: Matplotlib axes
        graph: NetworkX graph
        pos: Node positions dictionary
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
            color=VIZ_CONFIG.edge_base_color,
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
) -> None:
    """
    Draw nodes with computed sizes and colors.

    Args:
        ax: Matplotlib axes
        graph: NetworkX graph
        pos: Node positions dictionary
        sizes: List of node sizes
        colors: List of node colors (RGB tuples)
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
            edgecolors="white",
            linewidth=2,
            zorder=2,
        )


def draw_labels(ax: plt.Axes, graph: nx.Graph, pos: Dict, seed_id: str) -> None:
    """
    Draw paper labels in "Author, Year" format.

    Args:
        ax: Matplotlib axes
        graph: NetworkX graph
        pos: Node positions dictionary
        seed_id: ID of seed paper (gets bold label)
    """
    for node in graph.nodes():
        p = pos[node]

        # Extract author surname
        authors = graph.nodes[node].get("authors", [])
        if authors and authors[0]:
            last_name = authors[0].split()[-1]
        else:
            last_name = "Unknown"

        year = graph.nodes[node].get("year", "")
        label = f"{last_name}, {year}"

        # Seed paper gets larger, bold label
        is_seed = node == seed_id
        fontsize = 10 if is_seed else VIZ_CONFIG.font_size
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
            color="#2d3748",
        )


def visualize_graph(
    graph: nx.Graph,
    seed_id: str,
    output_path: Path,
    iterations: int = 100,
    dpi: int = None,
) -> None:
    """
    Create Connected Papers-style visualization.

    This is the unified visualization function used by all strategies.

    Args:
        graph: NetworkX graph with paper nodes
        seed_id: ID of the seed paper
        output_path: Path for output PNG file
        iterations: Number of layout iterations (higher = better quality)
        dpi: Output resolution (defaults to config value)

    Visual Encodings:
        - Node size: Citation count + importance ranking (80-2500 pixels)
        - Node color: Smooth gradient by year (light → dark)
        - Edge thickness: Proportional to similarity weight
        - Edge opacity: Based on connection strength
        - Layout: Kamada-Kawai with organic perturbations
    """
    if dpi is None:
        dpi = VIZ_CONFIG.dpi

    # Compute layout
    pos = compute_layout(graph, iterations)

    # Compute visual properties
    sizes = compute_node_sizes(graph)
    colors, min_year, max_year = compute_node_colors(graph, seed_id)

    # Create figure
    fig, ax = plt.subplots(
        figsize=VIZ_CONFIG.figure_size, facecolor=VIZ_CONFIG.background_color
    )
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw graph components
    draw_edges(ax, graph, pos)
    draw_nodes(ax, graph, pos, sizes, colors)
    draw_labels(ax, graph, pos, seed_id)

    # Add title
    title = graph.nodes[seed_id].get("title", "Unknown")[:60]
    ax.set_title(f"Connected Papers Style: {title}...", fontsize=14, pad=20)

    # Save figure
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=VIZ_CONFIG.background_color,
    )
    plt.close()

    logger.info(f"Visualization saved to {output_path}")


def generate_output_path(
    graph: nx.Graph, seed_id: str, output_dir: Path = Path("out")
) -> Path:
    """
    Generate auto-named output path from paper title.

    Args:
        graph: NetworkX graph
        seed_id: ID of seed paper
        output_dir: Output directory

    Returns:
        Path object for output file
    """
    title = graph.nodes[seed_id].get("title", "graph")

    # Clean title for filename
    filename = title.lower()
    filename = "".join(c if c.isalnum() or c in " -" else "" for c in filename)
    filename = "-".join(filename.split())[:50]  # Limit length
    filename = f"{filename}.png"

    output_dir.mkdir(exist_ok=True)
    return output_dir / filename
