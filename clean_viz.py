#!/usr/bin/env python3
"""
Clean Connected Papers-style visualization with proper node limiting
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from semanticscholar import SemanticScholar
import argparse
from pathlib import Path


def build_limited_graph(paper_id, max_papers=40):
    """Build a limited citation graph similar to Connected Papers."""
    client = SemanticScholar()
    graph = nx.DiGraph()
    
    # Get root paper
    root = client.get_paper(paper_id)
    if not root:
        raise ValueError(f"Paper {paper_id} not found")
    
    # Add root as the central node
    graph.add_node(root.paperId, 
                   title=root.title,
                   year=root.year or 2020,
                   citation_count=root.citationCount or 0,
                   authors=[a.name for a in (root.authors or [])[:3]],
                   is_root=True)
    
    papers_added = 1
    
    # Get limited citations (papers citing the root)
    if papers_added < max_papers:
        citations = list(client.get_paper_citations(root.paperId, limit=min(15, max_papers//2)))
        for cit in citations[:min(15, max_papers//2)]:
            if papers_added >= max_papers:
                break
            citing = cit.get('citingPaper')
            if citing and citing.get('paperId'):
                graph.add_node(citing['paperId'],
                             title=citing.get('title', 'Unknown'),
                             year=citing.get('year', 2020),
                             citation_count=citing.get('citationCount', 0),
                             authors=[a.get('name', '') for a in (citing.get('authors', []))[:3]],
                             is_root=False)
                graph.add_edge(citing['paperId'], root.paperId)
                papers_added += 1
    
    # Get limited references (papers cited by root)
    if papers_added < max_papers:
        references = list(client.get_paper_references(root.paperId, limit=min(15, max_papers//2)))
        for ref in references[:min(15, max_papers//2)]:
            if papers_added >= max_papers:
                break
            cited = ref.get('citedPaper')
            if cited and cited.get('paperId'):
                graph.add_node(cited['paperId'],
                             title=cited.get('title', 'Unknown'),
                             year=cited.get('year', 2020),
                             citation_count=cited.get('citationCount', 0),
                             authors=[a.get('name', '') for a in (cited.get('authors', []))[:3]],
                             is_root=False)
                graph.add_edge(root.paperId, cited['paperId'])
                papers_added += 1
    
    return graph


def compute_connected_papers_layout(graph):
    """Compute layout similar to Connected Papers."""
    nodes = list(graph.nodes())
    n = len(nodes)
    
    # Find root node
    root = None
    for node in nodes:
        if graph.nodes[node].get('is_root'):
            root = node
            break
    
    if not root:
        root = nodes[0]
    
    # Initialize positions
    positions = {}
    
    # Place root at center
    positions[root] = np.array([0, 0])
    
    # Get years for temporal positioning
    years = {node: graph.nodes[node].get('year', 2020) for node in nodes}
    min_year = min(years.values())
    max_year = max(years.values())
    year_range = max(max_year - min_year, 1)
    
    # Position other nodes in circles around root based on year
    angle_offset = 0
    for node in nodes:
        if node == root:
            continue
        
        # Calculate angle based on index
        angle = angle_offset
        angle_offset += 2 * np.pi / (n - 1)
        
        # Distance based on year difference from root
        year_diff = abs(years[node] - years[root])
        radius = 2 + year_diff * 0.3
        
        # Add some randomness for organic look
        radius += np.random.uniform(-0.5, 0.5)
        angle += np.random.uniform(-0.2, 0.2)
        
        positions[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    
    # Apply force-directed refinement
    pos = nx.spring_layout(graph, pos=positions, k=2, iterations=30)
    
    return pos


def visualize_clean(graph, output_path):
    """Create clean visualization."""
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#f5f5f5')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Compute layout
    pos = compute_connected_papers_layout(graph)
    
    # Prepare node attributes
    nodes = list(graph.nodes())
    
    # Node sizes based on citations
    sizes = []
    colors = []
    labels = []
    
    for node in nodes:
        attrs = graph.nodes[node]
        
        # Size
        cit = attrs.get('citation_count', 0)
        is_root = attrs.get('is_root', False)
        
        if is_root:
            size = 2000  # Large central node
        elif cit > 50:
            size = 800
        elif cit > 20:
            size = 400
        elif cit > 10:
            size = 200
        else:
            size = 100
        sizes.append(size)
        
        # Color based on year
        year = attrs.get('year', 2020)
        if year <= 2018:
            colors.append('#b8d4e3')  # Light blue for old
        elif year <= 2020:
            colors.append('#6ba3be')  # Medium blue
        elif year <= 2022:
            colors.append('#457b9d')  # Darker blue
        else:
            colors.append('#1d3557')  # Dark blue for recent
        
        # Label
        authors = attrs.get('authors', [])
        first_author = authors[0].split()[-1] if authors else "Unknown"
        label = f"{first_author}, {year}"
        labels.append(label)
    
    # Draw edges
    for edge in graph.edges():
        p1 = pos[edge[0]]
        p2 = pos[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
               'gray', alpha=0.3, linewidth=1, zorder=1)
    
    # Draw nodes
    for i, node in enumerate(nodes):
        p = pos[node]
        ax.scatter(p[0], p[1], s=sizes[i], c=[colors[i]], 
                  alpha=0.8, edgecolors='white', linewidth=2, zorder=2)
    
    # Add labels
    for i, node in enumerate(nodes):
        p = pos[node]
        ax.annotate(labels[i], xy=p, 
                   xytext=(0, -10), textcoords='offset points',
                   ha='center', va='top', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f5f5f5')
    plt.close()
    print(f"Saved clean visualization to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_id")
    parser.add_argument("-o", "--output", type=Path, default=Path("out/clean_viz.png"))
    args = parser.parse_args()
    
    print("Building limited graph...")
    graph = build_limited_graph(args.paper_id, max_papers=40)
    print(f"Graph has {graph.number_of_nodes()} nodes")
    
    visualize_clean(graph, args.output)