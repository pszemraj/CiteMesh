#!/usr/bin/env python3
"""
Hybrid similarity for reference tool visualization.
Combines citation relationships from Semantic Scholar with
semantic similarity from sentence transformers.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import math
import pickle
from typing import Dict, List, Tuple, Set
from sentence_transformers import SentenceTransformer, util
import torch
from semanticscholar import SemanticScholar
from datasets import load_dataset


class HybridPapersBuilder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Path = Path("cache")):
        """
        Initialize with both Semantic Scholar and sentence transformer.
        
        Args:
            model_name: HuggingFace model name for embeddings
            cache_dir: Directory for caching
        """
        self.semantic_scholar = SemanticScholar()
        self.sentence_model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache files
        self.arxiv_cache = self.cache_dir / "arxiv_papers.pkl"
        self.embeddings_cache = self.cache_dir / f"{model_name.replace('/', '_')}_embeddings.pkl"
        
        # Data storage
        self.arxiv_papers = {}  # Background corpus for finding similar papers
        self.graph_papers = {}  # Papers actually in the graph (from citations)
        self.embeddings = {}    # Paper ID -> embedding
        
    def load_arxiv_corpus(self, max_papers: int = 5000):
        """Load ArXiv corpus for semantic similarity matching."""
        
        if self.arxiv_cache.exists():
            print("Loading ArXiv corpus from cache...")
            with open(self.arxiv_cache, 'rb') as f:
                self.arxiv_papers = pickle.load(f)
            print(f"Loaded {len(self.arxiv_papers)} papers from cache")
            return
        
        print(f"Loading ArXiv corpus (up to {max_papers} papers)...")
        
        try:
            # Load only 0.1% for faster testing
            dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train[:100]")
            print("Using ML-ArXiv-Papers dataset (first 100 papers)")
        except:
            try:
                dataset = load_dataset("gfissore/arxiv-abstracts-2021", split="train[:100]")
                print("Using arxiv-abstracts-2021 dataset (first 100 papers)")
            except:
                print("Warning: Could not load ArXiv dataset, semantic search will be limited")
                return
        
        for i, paper in enumerate(dataset):
            if i >= max_papers:
                break
                
            paper_id = paper.get('id', paper.get('paper_id', f"arxiv_{i}"))
            
            self.arxiv_papers[paper_id] = {
                'title': paper.get('title', 'Unknown'),
                'abstract': paper.get('abstract', paper.get('summary', '')),
                'year': self._extract_year(paper),
                'authors': paper.get('authors', []),
            }
            
            if (i + 1) % 1000 == 0:
                print(f"  Loaded {i + 1} papers...")
        
        print(f"Loaded {len(self.arxiv_papers)} ArXiv papers")
        
        # Cache for next time
        with open(self.arxiv_cache, 'wb') as f:
            pickle.dump(self.arxiv_papers, f)
    
    def _extract_year(self, paper: dict) -> int:
        """Extract year from paper metadata."""
        if 'year' in paper:
            return int(paper['year']) if paper['year'] else 2020
        if 'update_date' in paper:
            try:
                return int(paper['update_date'][:4])
            except:
                pass
        return 2020
    
    def get_citations_and_references(self, paper_id: str, max_citations: int = 20, 
                                    max_references: int = 20) -> str:
        """
        Get citations and references from Semantic Scholar.
        Also fetches abstracts for embedding computation.
        Returns the seed paper ID.
        """
        print(f"Fetching citations and references for seed paper...")
        
        # Get seed paper with abstract
        seed = self.semantic_scholar.get_paper(
            paper_id,
            fields=['title', 'year', 'authors', 'citationCount', 'abstract', 'paperId']
        )
        
        if not seed:
            raise ValueError(f"Paper {paper_id} not found")
        
        seed_id = seed.paperId
        
        # Store seed paper
        self.graph_papers[seed_id] = {
            'title': seed.title,
            'abstract': seed.abstract or '',
            'year': seed.year or 2020,
            'authors': [a.name for a in (seed.authors or [])[:3]],
            'citation_count': seed.citationCount or 0,
            'is_seed': True,
        }
        
        print(f"Seed: {seed.title[:50]}...")
        
        # Fetch citations
        print(f"Fetching up to {max_citations} citations...")
        try:
            citations = self.semantic_scholar.get_paper_citations(
                seed_id, 
                limit=max_citations,
                fields=['title', 'year', 'authors', 'citationCount', 'abstract', 'paperId']
            )
            
            for cit in citations:
                if hasattr(cit, 'citingPaper') and cit.citingPaper:
                    p = cit.citingPaper
                elif hasattr(cit, 'paper') and cit.paper:
                    p = cit.paper
                else:
                    continue
                
                if hasattr(p, 'paperId'):
                    self.graph_papers[p.paperId] = {
                        'title': p.title or 'Unknown',
                        'abstract': p.abstract or '',
                        'year': p.year or 2020,
                        'authors': [a.name for a in (p.authors or [])[:3]],
                        'citation_count': p.citationCount or 0,
                        'is_seed': False,
                        'relationship': 'citation',
                    }
        except Exception as e:
            print(f"  Warning: Citations fetch failed: {e}")
        
        # Fetch references
        print(f"Fetching up to {max_references} references...")
        try:
            references = self.semantic_scholar.get_paper_references(
                seed_id,
                limit=max_references,
                fields=['title', 'year', 'authors', 'citationCount', 'abstract', 'paperId']
            )
            
            for ref in references:
                if hasattr(ref, 'citedPaper') and ref.citedPaper:
                    p = ref.citedPaper
                elif hasattr(ref, 'paper') and ref.paper:
                    p = ref.paper
                else:
                    continue
                
                if hasattr(p, 'paperId'):
                    self.graph_papers[p.paperId] = {
                        'title': p.title or 'Unknown',
                        'abstract': p.abstract or '',
                        'year': p.year or 2020,
                        'authors': [a.name for a in (p.authors or [])[:3]],
                        'citation_count': p.citationCount or 0,
                        'is_seed': False,
                        'relationship': 'reference',
                    }
        except Exception as e:
            print(f"  Warning: References fetch failed: {e}")
        
        print(f"Collected {len(self.graph_papers)} papers from citations/references")
        return seed_id
    
    def compute_embeddings(self):
        """Compute embeddings for all papers (both graph and corpus)."""
        
        if self.embeddings_cache.exists():
            print("Loading embeddings from cache...")
            with open(self.embeddings_cache, 'rb') as f:
                self.embeddings = pickle.load(f)
            return
        
        print("Computing embeddings for all papers...")
        
        # Combine graph papers and ArXiv corpus
        all_papers = {}
        all_papers.update(self.graph_papers)
        all_papers.update(self.arxiv_papers)
        
        # Compute embeddings in batches
        batch_size = 32
        paper_ids = list(all_papers.keys())
        
        for i in range(0, len(paper_ids), batch_size):
            batch_ids = paper_ids[i:i+batch_size]
            batch_texts = []
            
            for pid in batch_ids:
                paper = all_papers[pid]
                text = f"{paper['title']}. {paper.get('abstract', '')[:500]}"
                batch_texts.append(text)
            
            # Compute batch embeddings
            batch_embeddings = self.sentence_model.encode(
                batch_texts, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Store embeddings
            for j, pid in enumerate(batch_ids):
                self.embeddings[pid] = batch_embeddings[j]
            
            if ((i // batch_size) + 1) % 10 == 0:
                print(f"  Computed {min(i + batch_size, len(paper_ids))}/{len(paper_ids)} embeddings...")
        
        print(f"Computed embeddings for {len(self.embeddings)} papers")
        
        # Cache embeddings
        with open(self.embeddings_cache, 'wb') as f:
            pickle.dump(self.embeddings, f)
    
    def find_semantically_similar(self, seed_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find papers from ArXiv corpus that are semantically similar to seed.
        These supplement the citation-based papers.
        """
        if seed_id not in self.embeddings:
            print(f"Warning: No embedding for seed {seed_id}")
            return []
        
        print(f"Finding semantically similar papers from corpus...")
        
        seed_embedding = self.embeddings[seed_id]
        similar_papers = []
        
        # Compare with ArXiv corpus (not papers already in graph)
        for paper_id in self.arxiv_papers:
            if paper_id in self.graph_papers:
                continue  # Skip papers already in graph
            
            if paper_id not in self.embeddings:
                continue
            
            # Compute cosine similarity
            similarity = util.pytorch_cos_sim(
                seed_embedding, 
                self.embeddings[paper_id]
            ).item()
            
            similar_papers.append((paper_id, similarity))
        
        # Sort by similarity and return top-k
        similar_papers.sort(key=lambda x: x[1], reverse=True)
        top_similar = similar_papers[:top_k]
        
        # Add to graph papers
        for paper_id, sim_score in top_similar:
            if paper_id in self.arxiv_papers:
                paper = self.arxiv_papers[paper_id]
                self.graph_papers[paper_id] = {
                    'title': paper['title'],
                    'abstract': paper.get('abstract', ''),
                    'year': paper['year'],
                    'authors': paper.get('authors', []),
                    'citation_count': 0,  # Unknown from ArXiv
                    'is_seed': False,
                    'relationship': 'semantic',
                    'semantic_similarity': sim_score,
                }
        
        print(f"Added {len(top_similar)} semantically similar papers")
        return top_similar
    
    def build_hybrid_graph(self, seed_id: str, similarity_threshold: float = 0.3) -> nx.Graph:
        """
        Build graph with hybrid similarity:
        - Citation relationships (from Semantic Scholar)
        - Semantic similarity (from embeddings)
        """
        graph = nx.Graph()
        
        # Add all nodes
        for paper_id, paper in self.graph_papers.items():
            graph.add_node(
                paper_id,
                title=paper['title'],
                year=paper['year'],
                authors=paper.get('authors', []),
                citation_count=paper.get('citation_count', 0),
                is_seed=paper.get('is_seed', False),
                relationship=paper.get('relationship', 'unknown'),
            )
        
        print("Computing pairwise similarities for graph edges...")
        
        paper_ids = list(self.graph_papers.keys())
        edge_count = 0
        
        # Compute pairwise similarities
        for i, p1_id in enumerate(paper_ids):
            for j in range(i + 1, len(paper_ids)):
                p2_id = paper_ids[j]
                
                # Initialize similarity components
                citation_similarity = 0
                semantic_similarity = 0
                
                # Citation-based similarity
                rel1 = self.graph_papers[p1_id].get('relationship', '')
                rel2 = self.graph_papers[p2_id].get('relationship', '')
                
                # Papers from same source (both citations or both references) are more similar
                if rel1 == rel2 and rel1 in ['citation', 'reference']:
                    citation_similarity = 0.3
                
                # Temporal similarity
                year1 = self.graph_papers[p1_id]['year']
                year2 = self.graph_papers[p2_id]['year']
                year_diff = abs(year1 - year2)
                temporal_similarity = math.exp(-year_diff / 5)
                
                # Semantic similarity from embeddings
                if p1_id in self.embeddings and p2_id in self.embeddings:
                    semantic_similarity = util.pytorch_cos_sim(
                        self.embeddings[p1_id],
                        self.embeddings[p2_id]
                    ).item()
                
                # Combine similarities (hybrid approach)
                # Weight: 40% semantic, 30% citation, 30% temporal
                combined_similarity = (
                    0.4 * semantic_similarity +
                    0.3 * citation_similarity +
                    0.3 * temporal_similarity
                )
                
                # Always connect to seed
                if p1_id == seed_id or p2_id == seed_id:
                    combined_similarity = max(combined_similarity, similarity_threshold * 0.8)
                
                # Add edge if similar enough
                if combined_similarity > similarity_threshold:
                    graph.add_edge(p1_id, p2_id, weight=combined_similarity)
                    edge_count += 1
        
        print(f"Graph complete: {graph.number_of_nodes()} nodes, {edge_count} edges")
        return graph


def visualize_hybrid_graph(graph: nx.Graph, output_path: Path, iterations: int = 200):
    """Visualize the hybrid similarity graph."""
    
    # Spring layout with similarity weights
    pos = nx.spring_layout(graph, k=1.5, iterations=iterations, seed=42, weight="weight")
    
    # Find and center seed node
    seed_nodes = [n for n in graph.nodes() if graph.nodes[n].get('is_seed', False)]
    if seed_nodes:
        seed_id = seed_nodes[0]
        center = np.array([0.5, 0.5])
        current = pos[seed_id]
        pos[seed_id] = current * 0.4 + center * 0.6
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#fafafa")
    ax.set_aspect("equal")
    ax.axis("off")
    
    nodes = list(graph.nodes())
    
    # Node sizes based on citation count
    sizes = []
    for node in nodes:
        if graph.nodes[node].get("is_seed"):
            sizes.append(2000)
        else:
            cit = graph.nodes[node].get("citation_count", 0)
            size = 200 + min(800, np.sqrt(cit) * 30)
            sizes.append(size)
    
    # Colors by relationship type and year
    colors = []
    for node in nodes:
        if graph.nodes[node].get("is_seed"):
            colors.append("#e63946")  # Red for seed
        elif graph.nodes[node].get("relationship") == "semantic":
            colors.append("#f77f00")  # Orange for semantic matches
        elif graph.nodes[node].get("relationship") == "citation":
            colors.append("#06ffa5")  # Green for citations
        elif graph.nodes[node].get("relationship") == "reference":
            colors.append("#0077b6")  # Blue for references
        else:
            colors.append("#94a3b8")  # Gray for unknown
    
    # Draw edges
    for edge in graph.edges(data=True):
        n1, n2, data = edge
        weight = data.get("weight", 0.3)
        
        p1 = pos[n1]
        p2 = pos[n2]
        
        alpha = min(0.6, weight * 1.5)
        width = max(0.5, weight * 4)
        
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
               color="#cbd5e0", alpha=alpha, linewidth=width, zorder=1)
    
    # Draw nodes
    for i, node in enumerate(nodes):
        p = pos[node]
        ax.scatter(p[0], p[1], s=sizes[i], c=[colors[i]],
                  alpha=0.95, edgecolors="white", linewidth=2.5, zorder=2)
    
    # Labels
    for node in nodes:
        p = pos[node]
        paper = graph.nodes[node]
        
        # Create label
        authors = paper.get("authors", [])
        if authors and len(authors) > 0:
            first_author = authors[0] if isinstance(authors[0], str) else "Unknown"
            last_name = first_author.split()[-1] if first_author else "Unknown"
        else:
            last_name = "Unknown"
        
        year = paper.get("year", "")
        label = f"{last_name}, {year}"
        
        fontsize = 10 if paper.get("is_seed") else 8
        fontweight = "bold" if paper.get("is_seed") else "normal"
        
        ax.annotate(label, xy=p, xytext=(0, -5), textcoords="offset points",
                   ha="center", va="top", fontsize=fontsize,
                   fontweight=fontweight, color="#1d3557")
    
    # Legend for relationship types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e63946', label='Seed Paper'),
        Patch(facecolor='#06ffa5', label='Citations'),
        Patch(facecolor='#0077b6', label='References'),
        Patch(facecolor='#f77f00', label='Semantic Similarity'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False)
    
    # Title
    seed_node = seed_nodes[0] if seed_nodes else nodes[0]
    title = graph.nodes[seed_node].get("title", "Unknown")[:60]
    ax.set_title(f"Hybrid reference tool: {title}...", fontsize=16, pad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hybrid citation + semantic similarity graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("paper_id", help="Paper identifier (DOI, arXiv ID, or S2 ID)")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-p", "--max-papers", type=int, default=40)
    parser.add_argument("-c", "--corpus-size", type=int, default=2000,
                       help="Size of ArXiv corpus for semantic search")
    parser.add_argument("-s", "--similarity-threshold", type=float, default=0.35)
    parser.add_argument("-m", "--model", default="all-MiniLM-L6-v2",
                       help="Sentence transformer model")
    parser.add_argument("-i", "--iterations", type=int, default=200)
    parser.add_argument("--max-semantic", type=int, default=10,
                       help="Max papers to add from semantic similarity")
    args = parser.parse_args()
    
    # Initialize builder
    builder = HybridPapersBuilder(model_name=args.model)
    
    # Load ArXiv corpus for semantic matching
    builder.load_arxiv_corpus(max_papers=args.corpus_size)
    
    # Get citations and references from Semantic Scholar
    seed_id = builder.get_citations_and_references(
        args.paper_id, 
        max_citations=args.max_papers // 2,
        max_references=args.max_papers // 2
    )
    
    # Compute embeddings for all papers
    builder.compute_embeddings()
    
    # Find additional semantically similar papers
    builder.find_semantically_similar(seed_id, top_k=args.max_semantic)
    
    # Build hybrid graph
    graph = builder.build_hybrid_graph(seed_id, args.similarity_threshold)
    
    # Generate output path
    if args.output is None:
        import re
        title = builder.graph_papers[seed_id]['title']
        safe_title = re.sub(r"[^\w\s-]", "", title[:60]).strip()
        safe_title = re.sub(r"[-\s]+", "-", safe_title).lower()
        output_path = Path("out") / f"{safe_title}-hybrid.png"
        output_path.parent.mkdir(exist_ok=True)
    else:
        output_path = args.output
    
    # Visualize
    visualize_hybrid_graph(graph, output_path, args.iterations)


if __name__ == "__main__":
    main()