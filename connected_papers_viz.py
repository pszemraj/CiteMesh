#!/usr/bin/env python3
"""
Connected Papers-style visualization implementing the correct algorithm
Based on ARCHITECTURE.md analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from semanticscholar import SemanticScholar
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Set


class ConnectedPapersBuilder:
    """Build a Connected Papers-style similarity graph."""
    
    def __init__(self):
        self.client = SemanticScholar()
        self.seed_paper_id = None
        
    def build_similarity_graph(self, paper_id: str, max_papers: int = 40) -> nx.Graph:
        """
        Build similarity graph following Connected Papers algorithm.
        
        Phase 1: Collect candidate papers
        Phase 2: Select by similarity  
        Phase 3: Build graph with similarity edges
        """
        print(f"Building Connected Papers graph for {paper_id}")
        
        # Phase 1: Get seed paper and candidates
        seed = self.client.get_paper(paper_id)
        if not seed:
            raise ValueError(f"Paper {paper_id} not found")
            
        self.seed_paper_id = seed.paperId
        print(f"Seed paper: {seed.title[:60]}...")
        
        # Collect candidate papers (citations and references of seed only)
        candidates = {}
        
        # Add seed paper
        candidates[seed.paperId] = {
            'id': seed.paperId,
            'title': seed.title,
            'year': seed.year or 2020,
            'authors': [a.name for a in (seed.authors or [])[:3]] if seed.authors else [],
            'citation_count': seed.citationCount or 0,
            'references': [],
            'citations': [],
            'is_seed': True
        }
        
        # Get papers citing the seed (limit to prevent explosion)
        print("Fetching citations...")
        citations = list(self.client.get_paper_citations(seed.paperId, limit=50))
        for cit in citations:
            # Handle both dict and object responses
            if isinstance(cit, dict):
                citing = cit.get('citingPaper')
            else:
                citing = cit.citingPaper if hasattr(cit, 'citingPaper') else None
                
            if citing:
                # Extract paper ID
                paper_id = citing.get('paperId') if isinstance(citing, dict) else citing.paperId
                if paper_id:
                    # Extract other fields handling both dict and object
                    if isinstance(citing, dict):
                        title = citing.get('title', 'Unknown')
                        year = citing.get('year', 2020)
                        authors = [a.get('name', '') for a in (citing.get('authors', []))[:3]]
                        citation_count = citing.get('citationCount', 0)
                    else:
                        title = citing.title or 'Unknown'
                        year = citing.year or 2020
                        authors = [a.name for a in (citing.authors or [])[:3]] if citing.authors else []
                        citation_count = citing.citationCount or 0
                    
                    candidates[paper_id] = {
                        'id': paper_id,
                        'title': title,
                        'year': year,
                        'authors': authors,
                        'citation_count': citation_count,
                        'references': [],
                        'citations': [],
                        'is_seed': False
                    }
                
        # Get papers cited by seed
        print("Fetching references...")  
        references = list(self.client.get_paper_references(seed.paperId, limit=50))
        for ref in references:
            # Handle both dict and object responses
            if isinstance(ref, dict):
                cited = ref.get('citedPaper')
            else:
                cited = ref.citedPaper if hasattr(ref, 'citedPaper') else None
                
            if cited:
                # Extract paper ID
                paper_id = cited.get('paperId') if isinstance(cited, dict) else cited.paperId
                if paper_id:
                    # Extract other fields handling both dict and object
                    if isinstance(cited, dict):
                        title = cited.get('title', 'Unknown')
                        year = cited.get('year', 2020)
                        authors = [a.get('name', '') for a in (cited.get('authors', []))[:3]]
                        citation_count = cited.get('citationCount', 0)
                    else:
                        title = cited.title or 'Unknown'
                        year = cited.year or 2020
                        authors = [a.name for a in (cited.authors or [])[:3]] if cited.authors else []
                        citation_count = cited.citationCount or 0
                    
                    candidates[paper_id] = {
                        'id': paper_id,
                        'title': title,
                        'year': year,
                        'authors': authors,
                        'citation_count': citation_count,
                        'references': [],
                        'citations': [],
                        'is_seed': False
                    }
        
        print(f"Found {len(candidates)} candidate papers")
        
        # Phase 2: Get references and citations for similarity calculation
        # For efficiency, only get these for papers we have
        print("Fetching additional metadata for similarity calculation...")
        
        for paper_id, paper_data in list(candidates.items()):
            if paper_data['is_seed']:
                # We already have seed's references/citations
                continue
                
            try:
                # Get this paper's references (what it cites)
                refs = list(self.client.get_paper_references(paper_id, limit=30))
                ref_ids = []
                for r in refs:
                    if isinstance(r, dict):
                        cited = r.get('citedPaper', {})
                        if cited.get('paperId'):
                            ref_ids.append(cited['paperId'])
                    else:
                        if hasattr(r, 'citedPaper') and r.citedPaper and hasattr(r.citedPaper, 'paperId'):
                            ref_ids.append(r.citedPaper.paperId)
                paper_data['references'] = ref_ids
                
                # Get this paper's citations (what cites it)
                cits = list(self.client.get_paper_citations(paper_id, limit=30))
                cit_ids = []
                for c in cits:
                    if isinstance(c, dict):
                        citing = c.get('citingPaper', {})
                        if citing.get('paperId'):
                            cit_ids.append(citing['paperId'])
                    else:
                        if hasattr(c, 'citingPaper') and c.citingPaper and hasattr(c.citingPaper, 'paperId'):
                            cit_ids.append(c.citingPaper.paperId)
                paper_data['citations'] = cit_ids
            except:
                # If we can't get metadata, use empty lists
                pass
        
        # Store seed's references and citations
        seed_refs = []
        for r in references:
            if isinstance(r, dict):
                cited = r.get('citedPaper', {})
                if cited.get('paperId'):
                    seed_refs.append(cited['paperId'])
            else:
                if hasattr(r, 'citedPaper') and r.citedPaper and hasattr(r.citedPaper, 'paperId'):
                    seed_refs.append(r.citedPaper.paperId)
        
        seed_cits = []
        for c in citations:
            if isinstance(c, dict):
                citing = c.get('citingPaper', {})
                if citing.get('paperId'):
                    seed_cits.append(citing['paperId'])
            else:
                if hasattr(c, 'citingPaper') and c.citingPaper and hasattr(c.citingPaper, 'paperId'):
                    seed_cits.append(c.citingPaper.paperId)
        candidates[self.seed_paper_id]['references'] = seed_refs
        candidates[self.seed_paper_id]['citations'] = seed_cits
        
        # Phase 3: Calculate similarity to seed and select top papers
        print("Calculating similarities...")
        similarities = {}
        
        for paper_id, paper_data in candidates.items():
            if paper_data['is_seed']:
                similarities[paper_id] = 1.0
                continue
                
            # Calculate similarity to seed
            sim = self._calculate_similarity(
                set(paper_data['references']), set(paper_data['citations']),
                set(seed_refs), set(seed_cits)
            )
            similarities[paper_id] = sim
        
        # Select top papers by similarity
        sorted_papers = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        selected = sorted_papers[:max_papers]
        selected_ids = set([p[0] for p in selected])
        
        print(f"Selected {len(selected)} most similar papers")
        
        # Phase 4: Build graph with selected papers
        graph = nx.Graph()
        
        for paper_id, similarity in selected:
            paper = candidates[paper_id]
            graph.add_node(
                paper_id,
                **paper,
                similarity_to_seed=similarity
            )
        
        # Add edges based on pairwise similarity
        nodes = list(selected_ids)
        for i, node1 in enumerate(nodes):
            for j in range(i+1, len(nodes)):
                node2 = nodes[j]
                
                # Calculate pairwise similarity
                data1 = candidates[node1]
                data2 = candidates[node2]
                
                sim = self._calculate_similarity(
                    set(data1['references']), set(data1['citations']),
                    set(data2['references']), set(data2['citations'])
                )
                
                if sim > 0.1:  # Only add edge if significant similarity
                    graph.add_edge(node1, node2, weight=sim)
        
        print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph
    
    def _calculate_similarity(self, refs1: Set, cits1: Set, refs2: Set, cits2: Set) -> float:
        """Calculate similarity between two papers using bibliographic coupling and co-citation."""
        
        # Bibliographic coupling (shared references)
        bc_score = 0
        if refs1 and refs2:
            shared = len(refs1 & refs2)
            union = len(refs1 | refs2)
            bc_score = shared / union if union > 0 else 0
        
        # Co-citation (shared citations) 
        cc_score = 0
        if cits1 and cits2:
            shared = len(cits1 & cits2)
            union = len(cits1 | cits2)
            cc_score = shared / union if union > 0 else 0
        
        # Weight bibliographic coupling more heavily
        return 0.6 * bc_score + 0.4 * cc_score


def compute_connected_papers_layout(graph: nx.Graph, seed_id: str) -> Dict:
    """
    Compute force-directed layout matching Connected Papers style.
    Key: Let clustering emerge naturally from similarity.
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    
    # Initialize positions - seed at center, others in small circle
    pos = {}
    angle = 0
    angle_step = 2 * np.pi / (n - 1) if n > 1 else 0
    
    for node in nodes:
        if node == seed_id:
            pos[node] = np.array([0.0, 0.0])
        else:
            # Small circle around seed
            radius = 50
            pos[node] = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle)
            ])
            angle += angle_step
    
    # Force simulation parameters
    k_repulsion = 500  # Repulsion strength
    k_attraction = 0.5  # Attraction strength
    dt = 0.1  # Time step
    damping = 0.8  # Velocity damping
    iterations = 300
    
    # Node masses based on citations
    masses = {}
    for node in nodes:
        cit = graph.nodes[node]['citation_count']
        masses[node] = 1.0 + np.sqrt(cit / 100.0) if cit > 0 else 1.0
        
        # Seed is heavier to stay central
        if node == seed_id:
            masses[node] *= 3
    
    # Velocities
    vel = {node: np.array([0.0, 0.0]) for node in nodes}
    
    # Run simulation
    for iteration in range(iterations):
        forces = {node: np.array([0.0, 0.0]) for node in nodes}
        
        # Repulsion between all nodes
        for i, node1 in enumerate(nodes):
            for j in range(i+1, len(nodes)):
                node2 = nodes[j]
                
                delta = pos[node2] - pos[node1]
                dist = np.linalg.norm(delta)
                
                if dist < 0.01:
                    dist = 0.01
                    delta = np.random.randn(2) * 0.1
                
                # Coulomb repulsion: F = k * m1 * m2 / r^2
                force_mag = k_repulsion * masses[node1] * masses[node2] / (dist * dist)
                force = delta / dist * force_mag
                
                forces[node1] -= force / masses[node1]
                forces[node2] += force / masses[node2]
        
        # Attraction along edges (based on similarity)
        for edge in graph.edges(data=True):
            node1, node2, data = edge
            weight = data.get('weight', 0.1)
            
            delta = pos[node2] - pos[node1]
            dist = np.linalg.norm(delta)
            
            if dist > 0:
                # Spring force: F = k * weight * (dist - rest_length)
                rest_length = 100 * (1 - weight)  # Higher similarity = shorter rest length
                force_mag = k_attraction * weight * (dist - rest_length)
                force = delta / dist * force_mag
                
                forces[node1] += force / masses[node1]
                forces[node2] -= force / masses[node2]
        
        # Gentle centering force for seed
        if seed_id:
            center_force = -pos[seed_id] * 0.01
            forces[seed_id] += center_force
        
        # Update velocities and positions
        for node in nodes:
            vel[node] = vel[node] * damping + forces[node] * dt
            pos[node] = pos[node] + vel[node] * dt
    
    return pos


def visualize_connected_papers(graph: nx.Graph, output_path: Path, seed_id: str):
    """Create Connected Papers-style visualization."""
    
    # Compute layout
    pos = compute_connected_papers_layout(graph, seed_id)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#f8f8f8')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Prepare visual attributes
    nodes = list(graph.nodes())
    
    # Node sizes
    sizes = []
    for node in nodes:
        if graph.nodes[node]['is_seed']:
            sizes.append(1500)  # Large seed
        else:
            cit = graph.nodes[node]['citation_count']
            if cit > 100:
                size = 500
            elif cit > 50:
                size = 300
            elif cit > 20:
                size = 200
            elif cit > 10:
                size = 150
            else:
                size = 100
            sizes.append(size)
    
    # Node colors based on year
    colors = []
    years = [graph.nodes[node]['year'] for node in nodes]
    min_year = min(years)
    max_year = max(years)
    year_range = max(max_year - min_year, 1)
    
    for node in nodes:
        year = graph.nodes[node]['year']
        year_norm = (year - min_year) / year_range
        
        # Color gradient from light blue (old) to dark blue (new)
        if year_norm < 0.33:
            colors.append('#b8d4e3')
        elif year_norm < 0.66:
            colors.append('#6ba3be')
        else:
            colors.append('#457b9d')
    
    # Draw edges
    for edge in graph.edges(data=True):
        node1, node2 = edge[:2]
        weight = edge[2].get('weight', 0.1)
        
        p1 = pos[node1]
        p2 = pos[node2]
        
        # Edge style based on weight
        alpha = min(0.6, weight * 2)
        width = max(0.5, weight * 3)
        
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
               color='#94a3b8', alpha=alpha, linewidth=width, zorder=1)
    
    # Draw nodes
    for i, node in enumerate(nodes):
        p = pos[node]
        ax.scatter(p[0], p[1], s=sizes[i], c=[colors[i]], 
                  alpha=0.9, edgecolors='white', linewidth=2, zorder=2)
    
    # Add labels
    for node in nodes:
        p = pos[node]
        authors = graph.nodes[node]['authors']
        year = graph.nodes[node]['year']
        
        # Format: "LastName, Year"
        if authors:
            last_name = authors[0].split()[-1] if authors[0] else "Unknown"
        else:
            last_name = "Unknown"
        label = f"{last_name}, {year}"
        
        # Larger font for seed
        fontsize = 10 if graph.nodes[node]['is_seed'] else 8
        fontweight = 'bold' if graph.nodes[node]['is_seed'] else 'normal'
        
        ax.annotate(label, xy=p, xytext=(0, -5), 
                   textcoords='offset points', ha='center', va='top',
                   fontsize=fontsize, fontweight=fontweight, color='#2d3748')
    
    # Title with seed paper name
    seed_title = graph.nodes[seed_id]['title'][:60] + "..."
    ax.set_title(f"Connected Papers: {seed_title}", fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f8f8f8')
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create Connected Papers-style visualization")
    parser.add_argument("paper_id", help="Paper ID (DOI, arXiv ID, or S2 ID)")
    parser.add_argument("-o", "--output", type=Path, default=Path("out/connected_papers.png"))
    parser.add_argument("--max-papers", type=int, default=40, help="Maximum papers to show")
    args = parser.parse_args()
    
    # Build graph
    builder = ConnectedPapersBuilder()
    graph = builder.build_similarity_graph(args.paper_id, max_papers=args.max_papers)
    
    # Visualize
    visualize_connected_papers(graph, args.output, builder.seed_paper_id)


if __name__ == "__main__":
    main()