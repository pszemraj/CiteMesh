#!/usr/bin/env python3
"""
Simplified reference tool implementation
Focus on getting the core algorithm right
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from semanticscholar import SemanticScholar
from pathlib import Path
import argparse


def build_graph(paper_id: str, max_papers: int = 40):
    """Build a simplified similarity graph."""
    
    client = SemanticScholar()
    
    # Get seed paper
    print(f"Fetching seed paper {paper_id}...")
    seed = client.get_paper(paper_id)
    if not seed:
        raise ValueError(f"Paper not found: {paper_id}")
    
    print(f"Seed: {seed.title[:60]}...")
    
    # Build graph with just direct citations and references
    graph = nx.Graph()
    
    # Add seed
    graph.add_node(seed.paperId,
                  title=seed.title,
                  year=seed.year or 2020,
                  authors=[a.name for a in (seed.authors or [])[:3]],
                  citation_count=seed.citationCount or 0,
                  is_seed=True)
    
    papers_added = 1
    
    # Add some citations
    print("Adding citations...")
    citations = client.get_paper_citations(seed.paperId, limit=20)
    for cit in citations:
        if papers_added >= max_papers:
            break
        try:
            # Access the citing paper through the object (it's .paper not .citingPaper)
            citing_paper = cit.paper
            if citing_paper and hasattr(citing_paper, 'paperId') and citing_paper.paperId:
                graph.add_node(citing_paper.paperId,
                             title=citing_paper.title or "Unknown",
                             year=citing_paper.year or 2020,
                             authors=[a.name for a in (citing_paper.authors or [])[:3]],
                             citation_count=citing_paper.citationCount or 0,
                             is_seed=False)
                # Add edge to seed
                graph.add_edge(seed.paperId, citing_paper.paperId, weight=0.3)
                papers_added += 1
        except:
            pass
    
    # Add some references
    print("Adding references...")
    references = client.get_paper_references(seed.paperId, limit=20)
    for ref in references:
        if papers_added >= max_papers:
            break
        try:
            # Access the cited paper through the object (it's .paper not .citedPaper)
            cited_paper = ref.paper
            if cited_paper and hasattr(cited_paper, 'paperId') and cited_paper.paperId:
                graph.add_node(cited_paper.paperId,
                             title=cited_paper.title or "Unknown",
                             year=cited_paper.year or 2020,
                             authors=[a.name for a in (cited_paper.authors or [])[:3]],
                             citation_count=cited_paper.citationCount or 0,
                             is_seed=False)
                # Add edge to seed
                graph.add_edge(seed.paperId, cited_paper.paperId, weight=0.3)
                papers_added += 1
        except:
            pass
    
    print(f"Graph has {graph.number_of_nodes()} nodes")
    return graph, seed.paperId


def visualize(graph: nx.Graph, seed_id: str, output_path: Path):
    """Create visualization."""
    
    # Simple force-directed layout
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42, center=[0, 0])
    
    # Ensure seed is central
    if seed_id in pos:
        # Move seed closer to center
        current = pos[seed_id]
        pos[seed_id] = current * 0.3
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#f8f8f8')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Node attributes
    nodes = list(graph.nodes())
    
    # Sizes
    sizes = []
    for node in nodes:
        if graph.nodes[node].get('is_seed'):
            sizes.append(1000)
        else:
            cit = graph.nodes[node].get('citation_count', 0)
            size = 100 + min(500, cit * 2)
            sizes.append(size)
    
    # Colors by year
    colors = []
    for node in nodes:
        year = graph.nodes[node].get('year', 2020)
        if year <= 2018:
            colors.append('#b8d4e3')
        elif year <= 2020:
            colors.append('#6ba3be')
        else:
            colors.append('#457b9d')
    
    # Draw edges
    for edge in graph.edges():
        p1 = pos[edge[0]]
        p2 = pos[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
               color='#94a3b8', alpha=0.4, linewidth=1, zorder=1)
    
    # Draw nodes
    for i, node in enumerate(nodes):
        p = pos[node]
        ax.scatter(p[0], p[1], s=sizes[i], c=[colors[i]], 
                  alpha=0.9, edgecolors='white', linewidth=2, zorder=2)
    
    # Labels
    for node in nodes:
        p = pos[node]
        authors = graph.nodes[node].get('authors', [])
        year = graph.nodes[node].get('year', 2020)
        
        if authors:
            last_name = authors[0].split()[-1] if authors[0] else "Unknown"
        else:
            last_name = "Unknown"
        label = f"{last_name}, {year}"
        
        fontsize = 10 if graph.nodes[node].get('is_seed') else 8
        ax.annotate(label, xy=p, xytext=(0, -5),
                   textcoords='offset points', ha='center', va='top',
                   fontsize=fontsize, fontweight='bold', color='#2d3748')
    
    # Title
    seed_title = graph.nodes[seed_id].get('title', 'Unknown')[:60]
    ax.set_title(f"Citation Network: {seed_title}...", fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f8f8f8')
    plt.close()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_id")
    parser.add_argument("-o", "--output", type=Path, default=Path("out/simple_connected.png"))
    args = parser.parse_args()
    
    graph, seed_id = build_graph(args.paper_id, max_papers=40)
    visualize(graph, seed_id, args.output)


if __name__ == "__main__":
    main()