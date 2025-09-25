#!/usr/bin/env python3
"""
Direct image visualization using matplotlib and networkx
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from citation_graph import CitationGraphBuilder, GraphConfig, LayoutStyle
import argparse
from pathlib import Path


def visualize_direct(graph, output_path, builder):
    """Create visualization directly as PNG using matplotlib."""
    
    # Get layout positions
    positions = builder.compute_layout(graph, LayoutStyle.SIMILARITY)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#fafafa')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Get node attributes
    nodes = list(graph.nodes())
    years = []
    citations = []
    labels = []
    
    for node in nodes:
        attrs = graph.nodes[node]
        year = attrs.get('year')
        years.append(year if year is not None else 2020)
        citations.append(attrs.get('citation_count', 0))
        
        # Create label
        authors = attrs.get('authors', [])
        first_author = authors[0].split()[-1] if authors else "Unknown"
        year_str = str(year) if year else ''
        label = f"{first_author}, {year_str}" if year_str else first_author
        labels.append(label)
    
    # Normalize years for color
    min_year = min(years) if years else 2020
    max_year = max(years) if years else 2024
    year_range = max(max_year - min_year, 1)
    
    # Normalize citations for size
    max_cit = max(citations) if citations else 1
    sizes = []
    for cit in citations:
        if cit == 0:
            size = 50
        else:
            # Scale between 50 and 1000
            size = 50 + (cit / max_cit) * 950
        sizes.append(size)
    
    # Create colors based on year
    colors = []
    for year in years:
        year_norm = (year - min_year) / year_range
        # Use blue gradient
        lightness = 0.8 - year_norm * 0.5  # From light to dark
        colors.append((0.2, 0.4 + year_norm * 0.3, lightness))
    
    # Draw edges first (so they appear behind nodes)
    if graph.number_of_edges() > 0:
        # Get similarity matrix
        similarity = builder.compute_similarity_matrix(graph)
        
        # Draw edges based on similarity
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], start=i+1):
                sim = similarity[i, j]
                if sim > 0.1:  # Threshold for edge display
                    pos1 = positions[node1]
                    pos2 = positions[node2]
                    
                    # Edge styling based on similarity
                    if sim > 0.3:
                        alpha = 0.7
                        linewidth = 2.0
                        color = '#64748b'
                    elif sim > 0.2:
                        alpha = 0.5
                        linewidth = 1.5
                        color = '#94a3b8'
                    else:
                        alpha = 0.3
                        linewidth = 1.0
                        color = '#94a3b8'
                    
                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                           color=color, alpha=alpha, linewidth=linewidth, zorder=1)
    
    # Draw nodes
    for i, node in enumerate(nodes):
        pos = positions[node]
        ax.scatter(pos[0], pos[1], s=sizes[i], c=[colors[i]], 
                  alpha=0.9, edgecolors='white', linewidth=1, zorder=2)
        
        # Add label
        ax.annotate(labels[i], xy=pos, xytext=(0, -3),
                   textcoords='offset points', ha='center', va='top',
                   fontsize=8, fontweight='bold', color='#2d3748')
    
    # Title
    root_paper = graph.nodes[nodes[0]]
    title = root_paper.get('title', 'Citation Network')[:80]
    ax.set_title(f"Citation Network: {title}...", fontsize=12, pad=20)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_id", help="Paper ID (DOI, arXiv ID, or S2 ID)")
    parser.add_argument("-o", "--output", type=Path, default=Path("out/direct_viz.png"))
    parser.add_argument("--max-citations", type=int, default=10)
    parser.add_argument("--max-references", type=int, default=10)
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()
    
    # Build graph
    config = GraphConfig(layout_style=LayoutStyle.SIMILARITY)
    builder = CitationGraphBuilder(config=config)
    
    graph = builder.build(
        paper_id=args.paper_id,
        depth=args.depth,
        max_citations=args.max_citations,
        max_references=args.max_references,
    )
    
    # Visualize
    visualize_direct(graph, args.output, builder)