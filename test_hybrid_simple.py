#!/usr/bin/env python3
"""
Simple test of hybrid approach without loading large datasets.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import math
from sentence_transformers import SentenceTransformer
import torch
from semanticscholar import SemanticScholar

def test_hybrid():
    """Test the hybrid approach with a minimal setup."""
    
    print("Initializing...")
    semantic_scholar = SemanticScholar()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Get seed paper
    print("Fetching seed paper...")
    seed = semantic_scholar.get_paper(
        "arxiv:1706.03762",
        fields=['title', 'year', 'authors', 'citationCount', 'abstract', 'paperId']
    )
    
    if not seed:
        print("Failed to fetch seed paper")
        return
    
    print(f"Seed: {seed.title}")
    
    # Get a few citations
    print("Fetching 5 citations...")
    papers = {
        seed.paperId: {
            'title': seed.title,
            'abstract': seed.abstract or '',
            'year': seed.year or 2017,
            'is_seed': True,
        }
    }
    
    try:
        citations = semantic_scholar.get_paper_citations(
            seed.paperId, 
            limit=5,
            fields=['title', 'year', 'abstract', 'paperId']
        )
        
        for cit in citations:
            if hasattr(cit, 'citingPaper') and cit.citingPaper:
                p = cit.citingPaper
                if hasattr(p, 'paperId') and p.paperId:
                    papers[p.paperId] = {
                        'title': p.title or 'Unknown',
                        'abstract': p.abstract or '',
                        'year': p.year or 2020,
                        'is_seed': False,
                    }
    except Exception as e:
        print(f"Citations fetch failed: {e}")
    
    print(f"Got {len(papers)} papers")
    
    # Compute embeddings
    print("Computing embeddings...")
    paper_ids = list(papers.keys())
    texts = [f"{papers[pid]['title']}. {papers[pid]['abstract'][:500]}" for pid in paper_ids]
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    # Build graph
    print("Building graph...")
    graph = nx.Graph()
    
    for i, pid in enumerate(paper_ids):
        paper = papers[pid]
        graph.add_node(pid, 
                      title=paper['title'],
                      year=paper['year'],
                      is_seed=paper['is_seed'])
    
    # Add edges based on similarity
    for i in range(len(paper_ids)):
        for j in range(i + 1, len(paper_ids)):
            # Compute cosine similarity
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            ).item()
            
            if sim > 0.3:
                graph.add_edge(paper_ids[i], paper_ids[j], weight=sim)
    
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Simple visualization
    pos = nx.spring_layout(graph, k=1.5, iterations=50)
    
    plt.figure(figsize=(10, 8))
    
    # Draw edges
    for edge in graph.edges(data=True):
        n1, n2, data = edge
        weight = data.get('weight', 0.5)
        p1, p2 = pos[n1], pos[n2]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                'gray', alpha=weight*0.5, linewidth=weight*2)
    
    # Draw nodes
    for node in graph.nodes():
        p = pos[node]
        color = 'red' if graph.nodes[node]['is_seed'] else 'lightblue'
        size = 500 if graph.nodes[node]['is_seed'] else 200
        plt.scatter(p[0], p[1], s=size, c=color, alpha=0.8)
        
        # Label
        title = graph.nodes[node]['title'][:20] + "..."
        plt.annotate(title, xy=p, fontsize=8, ha='center')
    
    plt.title("Hybrid Test Graph")
    plt.axis('off')
    plt.tight_layout()
    
    output_path = Path("out") / "hybrid_test.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    test_hybrid()