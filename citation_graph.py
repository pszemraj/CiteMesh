#!/usr/bin/env python3
"""
Mesh visualization demonstrating reference tool concept
Using simulated similarity for speed
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from semanticscholar import SemanticScholar
from pathlib import Path
import argparse


def build_mesh_graph(
    paper_id: str,
    max_papers: int = 40,
    max_citations: int = 20,
    max_references: int = 20,
    similarity_threshold: float = 0.2,
):
    """Build graph with mesh connections based on simulated similarity."""

    client = SemanticScholar()

    # Get seed
    seed = client.get_paper(paper_id)
    if not seed:
        raise ValueError("Paper not found")

    seed_id = seed.paperId
    print(f"Building mesh graph for: {seed.title[:50]}...")

    graph = nx.Graph()

    # Add seed
    graph.add_node(
        seed_id,
        title=seed.title,
        year=seed.year or 2020,
        authors=[a.name for a in (seed.authors or [])[:3]],
        citation_count=seed.citationCount or 0,
        is_seed=True,
    )

    papers_added = 1
    paper_list = [seed_id]

    # Collect papers
    print("Collecting papers...")

    # Add citations
    try:
        citations = client.get_paper_citations(seed_id, limit=max_citations)
        for cit in citations:
            if papers_added >= max_papers:
                break
            if hasattr(cit, "paper") and cit.paper and hasattr(cit.paper, "paperId"):
                p = cit.paper
                graph.add_node(
                    p.paperId,
                    title=p.title or "Unknown",
                    year=p.year or 2020,
                    authors=[a.name for a in (p.authors or [])[:3]],
                    citation_count=p.citationCount or 0,
                    is_seed=False,
                )
                paper_list.append(p.paperId)
                papers_added += 1
    except (AttributeError, TypeError):
        pass

    # Add references
    try:
        references = client.get_paper_references(seed_id, limit=max_references)
        for ref in references:
            if papers_added >= max_papers:
                break
            if hasattr(ref, "paper") and ref.paper and hasattr(ref.paper, "paperId"):
                p = ref.paper
                graph.add_node(
                    p.paperId,
                    title=p.title or "Unknown",
                    year=p.year or 2020,
                    authors=[a.name for a in (p.authors or [])[:3]],
                    citation_count=p.citationCount or 0,
                    is_seed=False,
                )
                paper_list.append(p.paperId)
                papers_added += 1
    except (AttributeError, TypeError):
        pass

    print(f"Collected {len(paper_list)} papers")

    # Create mesh by simulating similarity
    # Papers close in time and citation count are considered similar
    print("Creating similarity mesh...")

    for i, p1 in enumerate(paper_list):
        for j in range(i + 1, len(paper_list)):
            p2 = paper_list[j]

            # Simulate similarity based on year and citations
            year1 = graph.nodes[p1].get("year", 2020)
            year2 = graph.nodes[p2].get("year", 2020)
            cit1 = graph.nodes[p1].get("citation_count", 0)
            cit2 = graph.nodes[p2].get("citation_count", 0)

            # Year similarity (papers close in time)
            year_diff = abs(year1 - year2)
            year_sim = 1.0 / (1.0 + year_diff / 3.0)

            # Citation similarity (papers with similar impact)
            cit_ratio = min(cit1, cit2) / max(cit1, cit2) if max(cit1, cit2) > 0 else 0

            # Combined similarity
            similarity = 0.6 * year_sim + 0.4 * cit_ratio
            
            # Add variation but reduce overall
            similarity *= np.random.uniform(0.3, 0.8)
            
            # Special case: always connect to seed with some weight
            if p1 == seed_id or p2 == seed_id:
                similarity = max(similarity, similarity_threshold * 0.8)

            # Add edge if similar enough
            if similarity > similarity_threshold:
                graph.add_edge(p1, p2, weight=similarity)

    print(
        f"Graph complete: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )
    return graph, seed_id


def visualize_mesh(
    graph: nx.Graph,
    seed_id: str,
    output_path: Path,
    iterations: int = 100,
    dpi: int = 150,
):
    """Visualize with reference tool-style layout."""

    # Use spring layout with custom parameters
    pos = nx.spring_layout(
        graph, k=1.2, iterations=iterations, seed=42, weight="weight"
    )

    # Ensure seed is more central
    if seed_id in pos:
        current = pos[seed_id]
        center = np.array([0.5, 0.5])
        pos[seed_id] = current * 0.4 + center * 0.6

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#fafafa")
    ax.set_aspect("equal")
    ax.axis("off")

    nodes = list(graph.nodes())

    # Node sizes
    sizes = []
    for node in nodes:
        if graph.nodes[node].get("is_seed"):
            sizes.append(1200)
        else:
            cit = graph.nodes[node].get("citation_count", 0)
            size = 80 + min(400, cit * 3)
            sizes.append(size)

    # Colors by year
    colors = []
    years = [graph.nodes[n].get("year", 2020) for n in nodes]
    min_year, max_year = min(years), max(years)

    for node in nodes:
        year = graph.nodes[node].get("year", 2020)
        year_norm = (year - min_year) / max(max_year - min_year, 1)

        # Color gradient
        if year_norm < 0.33:
            colors.append("#b8d4e3")
        elif year_norm < 0.66:
            colors.append("#6ba3be")
        else:
            colors.append("#457b9d")

    # Draw edges
    for edge in graph.edges(data=True):
        n1, n2, data = edge
        weight = data.get("weight", 0.1)

        p1 = pos[n1]
        p2 = pos[n2]

        # Style based on weight
        alpha = min(0.6, weight)
        width = max(0.5, weight * 2)

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="#94a3b8",
            alpha=alpha,
            linewidth=width,
            zorder=1,
        )

    # Draw nodes
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

    # Labels
    for node in nodes:
        p = pos[node]
        authors = graph.nodes[node].get("authors", [])
        year = graph.nodes[node].get("year", "")

        if authors and authors[0]:
            last_name = authors[0].split()[-1]
        else:
            last_name = "Unknown"
        label = f"{last_name}, {year}"

        fontsize = 10 if graph.nodes[node].get("is_seed") else 8
        fontweight = "bold" if graph.nodes[node].get("is_seed") else "normal"

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

    # Title
    title = graph.nodes[seed_id].get("title", "Unknown")[:60]
    ax.set_title(f"reference tool Style: {title}...", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference tool-style citation graph visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paper_id", help="Paper identifier (DOI, arXiv ID, or Semantic Scholar ID)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG file path (auto-named if not specified)",
    )
    parser.add_argument(
        "-p",
        "--max-papers",
        type=int,
        default=40,
        help="Maximum total papers to include in graph",
    )
    parser.add_argument(
        "-c",
        "--max-citations",
        type=int,
        default=20,
        help="Maximum citations to fetch per paper",
    )
    parser.add_argument(
        "-r",
        "--max-references",
        type=int,
        default=20,
        help="Maximum references to fetch per paper",
    )
    parser.add_argument(
        "-s",
        "--similarity-threshold",
        type=float,
        default=0.2,
        help="Minimum similarity score for edge creation (0-1)",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=100,
        help="Layout algorithm iterations (higher = better quality)",
    )
    parser.add_argument(
        "-d",
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI resolution",
    )
    args = parser.parse_args()

    # Build graph
    graph, seed_id = build_mesh_graph(
        args.paper_id,
        max_papers=args.max_papers,
        max_citations=args.max_citations,
        max_references=args.max_references,
        similarity_threshold=args.similarity_threshold,
    )

    # Auto-generate filename if not specified
    if args.output is None:
        import re

        title = graph.nodes[seed_id].get("title", "unknown")
        # Create safe filename from title
        safe_title = re.sub(r"[^\w\s-]", "", title[:60]).strip()
        safe_title = re.sub(r"[-\s]+", "-", safe_title).lower()
        output_path = Path("out") / f"{safe_title}.png"
        output_path.parent.mkdir(exist_ok=True)
    else:
        output_path = args.output
        output_path.parent.mkdir(exist_ok=True)

    visualize_mesh(
        graph, seed_id, output_path, iterations=args.iterations, dpi=args.dpi
    )


if __name__ == "__main__":
    main()
