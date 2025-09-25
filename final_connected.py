#!/usr/bin/env python3
"""
Final Connected Papers implementation
Following ARCHITECTURE.md exactly
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from semanticscholar import SemanticScholar
from pathlib import Path
import argparse
from typing import Dict, Set, List
import time


class ConnectedPapersViz:
    def __init__(self):
        self.client = SemanticScholar()
        self.seed_id = None
        
    def build_graph(self, paper_id: str, max_papers: int = 40) -> nx.Graph:
        """Build Connected Papers-style similarity graph."""
        
        print(f"Building graph for {paper_id}")
        
        # Get seed paper
        seed = self.client.get_paper(paper_id)
        if not seed:
            raise ValueError(f"Paper not found: {paper_id}")
        
        self.seed_id = seed.paperId
        print(f"Seed: {seed.title[:60]}...")
        
        # Collect papers related to seed
        papers = {}
        
        # Add seed
        papers[self.seed_id] = {
            'id': self.seed_id,
            'title': seed.title,
            'year': seed.year or 2020,
            'authors': [a.name for a in (seed.authors or [])[:3]],
            'citation_count': seed.citationCount or 0,
            'references': [],  # Will fill later
            'citations': [],   # Will fill later
            'is_seed': True
        }
        
        # Collect citations and references
        print("Collecting related papers...")
        
        # Get papers citing seed
        try:
            citations = self.client.get_paper_citations(self.seed_id, limit=20)
            for cit in citations:
                if len(papers) >= max_papers:
                    break
                if hasattr(cit, 'paper') and cit.paper:
                    p = cit.paper
                    if hasattr(p, 'paperId') and p.paperId:
                        papers[p.paperId] = {
                            'id': p.paperId,
                            'title': p.title or 'Unknown',
                            'year': p.year or 2020,
                            'authors': [a.name for a in (p.authors or [])[:3]],
                            'citation_count': p.citationCount or 0,
                            'references': [],
                            'citations': [],
                            'is_seed': False
                        }
        except:
            pass
        
        # Get papers cited by seed  
        try:
            references = self.client.get_paper_references(self.seed_id, limit=20)
            for ref in references:
                if len(papers) >= max_papers:
                    break
                if hasattr(ref, 'paper') and ref.paper:
                    p = ref.paper
                    if hasattr(p, 'paperId') and p.paperId:
                        papers[p.paperId] = {
                            'id': p.paperId,
                            'title': p.title or 'Unknown',
                            'year': p.year or 2020,
                            'authors': [a.name for a in (p.authors or [])[:3]],
                            'citation_count': p.citationCount or 0,
                            'references': [],
                            'citations': [],
                            'is_seed': False
                        }
        except:
            pass
        
        print(f"Collected {len(papers)} papers")
        
        # Now get references and citations for each paper (for similarity calculation)
        print("Fetching paper relationships for similarity calculation...")
        
        for pid, paper_data in papers.items():
            if paper_data['is_seed']:
                # Already processed seed
                continue
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
            
            try:
                # Get references (what this paper cites)
                refs = self.client.get_paper_references(pid, limit=15)
                ref_ids = []
                for r in refs:
                    if hasattr(r, 'paper') and r.paper and hasattr(r.paper, 'paperId'):
                        ref_ids.append(r.paper.paperId)
                paper_data['references'] = ref_ids[:10]  # Limit to keep it manageable
            except:
                pass
            
            try:
                # Get citations (what cites this paper)
                cits = self.client.get_paper_citations(pid, limit=15)
                cit_ids = []
                for c in cits:
                    if hasattr(c, 'paper') and c.paper and hasattr(c.paper, 'paperId'):
                        cit_ids.append(c.paper.paperId)
                paper_data['citations'] = cit_ids[:10]  # Limit
            except:
                pass
        
        # Get seed's references and citations
        try:
            refs = self.client.get_paper_references(self.seed_id, limit=30)
            papers[self.seed_id]['references'] = [
                r.paper.paperId for r in refs 
                if hasattr(r, 'paper') and r.paper and hasattr(r.paper, 'paperId')
            ][:20]
        except:
            pass
        
        try:
            cits = self.client.get_paper_citations(self.seed_id, limit=30)
            papers[self.seed_id]['citations'] = [
                c.paper.paperId for c in cits
                if hasattr(c, 'paper') and c.paper and hasattr(c.paper, 'paperId')
            ][:20]
        except:
            pass
        
        # Calculate pairwise similarities and build graph
        print("Calculating similarities...")
        graph = nx.Graph()
        
        # Add all nodes
        for pid, data in papers.items():
            graph.add_node(pid, **data)
        
        # Calculate similarity between every pair of papers
        paper_ids = list(papers.keys())
        for i, p1 in enumerate(paper_ids):
            for j in range(i+1, len(paper_ids)):
                p2 = paper_ids[j]
                
                # Calculate similarity
                sim = self._calculate_similarity(
                    set(papers[p1]['references']),
                    set(papers[p1]['citations']),
                    set(papers[p2]['references']),
                    set(papers[p2]['citations'])
                )
                
                # Add edge if similarity is significant
                if sim > 0.05:  # Low threshold to create mesh
                    graph.add_edge(p1, p2, weight=sim)
        
        print(f"Graph complete: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph
    
    def _calculate_similarity(self, refs1: Set, cits1: Set, refs2: Set, cits2: Set) -> float:
        """Calculate similarity using bibliographic coupling and co-citation."""
        
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
        
        # Combine with more weight on bibliographic coupling
        return 0.7 * bc_score + 0.3 * cc_score
    
    def layout_graph(self, graph: nx.Graph) -> Dict:
        """Create Connected Papers-style layout."""
        
        # Start with spring layout
        pos = nx.spring_layout(graph, k=1.5, iterations=50, seed=42)
        
        # Ensure seed is central
        if self.seed_id in pos:
            # Move seed toward center
            current = np.array(pos[self.seed_id])
            center = np.array([0.5, 0.5])
            pos[self.seed_id] = current * 0.3 + center * 0.7
        
        # Add some temporal organization
        for node in graph.nodes():
            year = graph.nodes[node].get('year', 2020)
            # Gentle bias: older papers left, newer right
            year_factor = (year - 2015) / 10.0  # Normalize around 2015-2025
            pos[node] = (
                pos[node][0] + year_factor * 0.1,  # Slight x adjustment
                pos[node][1]
            )
        
        return pos
    
    def visualize(self, graph: nx.Graph, output_path: Path):
        """Create visualization."""
        
        pos = self.layout_graph(graph)
        
        # Setup figure
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='#fafafa')
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Node attributes
        nodes = list(graph.nodes())
        
        # Sizes based on citation count and seed status
        sizes = []
        for node in nodes:
            if graph.nodes[node].get('is_seed'):
                sizes.append(1500)  # Large seed
            else:
                cit = graph.nodes[node].get('citation_count', 0)
                # Scale based on citations
                if cit > 100:
                    size = 600
                elif cit > 50:
                    size = 400
                elif cit > 20:
                    size = 250
                elif cit > 10:
                    size = 150
                else:
                    size = 80
                sizes.append(size)
        
        # Colors by year
        colors = []
        years = [graph.nodes[n].get('year', 2020) for n in nodes]
        min_year = min(years)
        max_year = max(years)
        for node in nodes:
            year = graph.nodes[node].get('year', 2020)
            year_norm = (year - min_year) / max(max_year - min_year, 1)
            
            # Gradient from light (old) to dark (new)
            if year_norm < 0.25:
                colors.append('#c7e2f7')  # Very light blue
            elif year_norm < 0.5:
                colors.append('#89c0e8')  # Light blue
            elif year_norm < 0.75:
                colors.append('#5a9fd8')  # Medium blue
            else:
                colors.append('#2e7ebb')  # Dark blue
        
        # Draw edges (similarity connections)
        for edge in graph.edges(data=True):
            n1, n2, data = edge
            weight = data.get('weight', 0.1)
            
            p1 = pos[n1]
            p2 = pos[n2]
            
            # Edge style based on similarity
            if weight > 0.3:
                alpha = 0.5
                width = 2
                color = '#6b7280'
            elif weight > 0.15:
                alpha = 0.3
                width = 1.5
                color = '#9ca3af'
            else:
                alpha = 0.2
                width = 1
                color = '#d1d5db'
            
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color=color, alpha=alpha, linewidth=width, zorder=1)
        
        # Draw nodes
        for i, node in enumerate(nodes):
            p = pos[node]
            ax.scatter(p[0], p[1], s=sizes[i], c=[colors[i]], 
                      alpha=0.9, edgecolors='white', linewidth=1.5, zorder=2)
        
        # Add labels
        for node in nodes:
            p = pos[node]
            authors = graph.nodes[node].get('authors', [])
            year = graph.nodes[node].get('year', '')
            
            # Format: "LastName, Year"
            if authors and authors[0]:
                last_name = authors[0].split()[-1]
            else:
                last_name = "Unknown"
            label = f"{last_name}, {year}" if year else last_name
            
            # Larger font for seed
            if graph.nodes[node].get('is_seed'):
                fontsize = 10
                fontweight = 'bold'
            else:
                fontsize = 8
                fontweight = 'normal'
            
            ax.annotate(label, xy=p, xytext=(0, -3),
                       textcoords='offset points', ha='center', va='top',
                       fontsize=fontsize, fontweight=fontweight, color='#374151')
        
        # Title
        seed_data = graph.nodes[self.seed_id]
        title = seed_data.get('title', 'Unknown')[:60]
        ax.set_title(f"Connected Papers: {title}...", fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#fafafa')
        plt.close()
        print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_id")
    parser.add_argument("-o", "--output", type=Path, default=Path("out/final_connected.png"))
    parser.add_argument("--max-papers", type=int, default=40)
    args = parser.parse_args()
    
    viz = ConnectedPapersViz()
    graph = viz.build_graph(args.paper_id, max_papers=args.max_papers)
    viz.visualize(graph, args.output)


if __name__ == "__main__":
    main()