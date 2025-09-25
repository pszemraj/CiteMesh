#!/usr/bin/env python3
"""
Embedding-based similarity for Connected Papers visualization.
Uses sentence transformers on abstracts to find semantically similar papers.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
import pickle
from sentence_transformers import SentenceTransformer, util
import torch
from semanticscholar import SemanticScholar
from datasets import load_dataset


class EmbeddingPapersBuilder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Path = Path("cache")):
        """
        Initialize with sentence transformer model.
        
        Args:
            model_name: HuggingFace model name (could use google/embeddinggemma-300m)
            cache_dir: Directory for caching embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.embeddings_cache = self.cache_dir / f"{model_name.replace('/', '_')}_embeddings.pkl"
        self.papers_cache = self.cache_dir / "papers_metadata.pkl"
        
        self.papers = {}
        self.embeddings = None
        self.paper_ids = []
        
    def load_arxiv_dataset(self, max_papers: int = 10000, categories: List[str] = None):
        """
        Load ArXiv dataset from HuggingFace.
        
        Args:
            max_papers: Maximum number of papers to load
            categories: Filter by ArXiv categories (e.g., ['cs.CL', 'cs.AI'])
        """
        print(f"Loading ArXiv dataset (up to {max_papers} papers)...")
        
        # Try to load from cache first
        if self.papers_cache.exists() and self.embeddings_cache.exists():
            print("Loading from cache...")
            with open(self.papers_cache, 'rb') as f:
                self.papers = pickle.load(f)
            with open(self.embeddings_cache, 'rb') as f:
                cache_data = pickle.load(f)
                self.embeddings = cache_data['embeddings']
                self.paper_ids = cache_data['paper_ids']
            print(f"Loaded {len(self.papers)} papers from cache")
            return
        
        # Load from HuggingFace
        try:
            # Try the ML-focused subset first (smaller, faster)
            # Use split="train[:2%]" for testing
            dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train[:2%]")
            print(f"Loaded ML-ArXiv-Papers dataset")
        except Exception as e:
            print(f"Failed to load ML-ArXiv-Papers: {e}")
            # Fallback to smaller abstracts dataset
            try:
                dataset = load_dataset("gfissore/arxiv-abstracts-2021", split="train[:1%]")
                print(f"Loaded arxiv-abstracts-2021 dataset")
            except Exception as e2:
                print(f"Failed to load arxiv-abstracts-2021: {e2}")
                raise
        
        papers_loaded = 0
        for paper in dataset:
            if papers_loaded >= max_papers:
                break
                
            # Filter by categories if specified
            if categories and 'categories' in paper:
                paper_cats = paper.get('categories', '').split()
                if not any(cat in paper_cats for cat in categories):
                    continue
            
            # Get paper ID - different field names in different datasets
            paper_id = paper.get('id', paper.get('paper_id', f"paper_{papers_loaded}"))
            
            # Store paper metadata
            self.papers[paper_id] = {
                'title': paper.get('title', 'Unknown'),
                'abstract': paper.get('abstract', paper.get('summary', '')),
                'authors': paper.get('authors', paper.get('authors_parsed', [])),
                'year': self._extract_year(paper),
                'categories': paper.get('categories', ''),
            }
            
            papers_loaded += 1
            if papers_loaded % 1000 == 0:
                print(f"  Loaded {papers_loaded} papers...")
        
        print(f"Loaded {len(self.papers)} papers from dataset")
        
        # Save to cache
        with open(self.papers_cache, 'wb') as f:
            pickle.dump(self.papers, f)
    
    def _extract_year(self, paper: dict) -> int:
        """Extract year from paper metadata."""
        if 'year' in paper:
            return paper['year']
        if 'update_date' in paper:
            return int(paper['update_date'][:4])
        if 'id' in paper:
            # ArXiv ID format: YYMM.NNNNN
            try:
                yy = int(paper['id'][:2])
                return 2000 + yy if yy < 50 else 1900 + yy
            except:
                pass
        return 2020  # Default
    
    def compute_embeddings(self):
        """Compute embeddings for all paper abstracts."""
        if self.embeddings is not None and len(self.embeddings) > 0:
            print("Embeddings already computed")
            return
            
        print("Computing embeddings for abstracts...")
        
        # Prepare texts for embedding (title + abstract)
        texts = []
        self.paper_ids = []
        
        for paper_id, paper in self.papers.items():
            text = f"{paper['title']}. {paper['abstract'][:1000]}"  # Limit abstract length
            texts.append(text)
            self.paper_ids.append(paper_id)
        
        # Compute embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
            
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(texts)} papers...")
        
        # Combine all embeddings
        self.embeddings = torch.cat(all_embeddings, dim=0)
        print(f"Computed embeddings for {len(self.embeddings)} papers")
        
        # Save to cache
        with open(self.embeddings_cache, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'paper_ids': self.paper_ids
            }, f)
    
    def find_similar_papers_from_text(self, query_text: str, top_k: int = 40) -> List[Tuple[str, float]]:
        """
        Find similar papers to a query text.
        
        Args:
            query_text: Title and/or abstract to search for
            top_k: Number of similar papers to return
            
        Returns:
            List of (paper_id, similarity_score) tuples
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not computed. Call compute_embeddings() first.")
        
        # Encode query
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        
        # Compute cosine similarities
        similarities = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar papers
        top_results = torch.topk(similarities, min(top_k, len(similarities)))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            paper_id = self.paper_ids[idx]
            results.append((paper_id, float(score)))
        
        return results
    
    def find_similar_papers_from_arxiv(self, arxiv_id: str, top_k: int = 40) -> List[Tuple[str, float]]:
        """
        Find similar papers to an ArXiv paper.
        
        Args:
            arxiv_id: ArXiv ID (e.g., "1706.03762")
            top_k: Number of similar papers to return
        """
        # First check if paper is in our dataset
        clean_id = arxiv_id.replace("arxiv:", "").replace("v1", "").replace("v2", "")
        
        if clean_id in self.papers:
            paper = self.papers[clean_id]
            query_text = f"{paper['title']}. {paper['abstract']}"
        else:
            # Fetch from Semantic Scholar as fallback
            print(f"Paper {arxiv_id} not in dataset, fetching from Semantic Scholar...")
            client = SemanticScholar()
            paper = client.get_paper(f"arxiv:{clean_id}")
            if not paper:
                raise ValueError(f"Paper {arxiv_id} not found")
            query_text = f"{paper.title}. {paper.abstract or ''}"
        
        return self.find_similar_papers_from_text(query_text, top_k)
    
    def build_similarity_graph(self, seed_papers: List[Tuple[str, float]], 
                             similarity_threshold: float = 0.3) -> nx.Graph:
        """
        Build graph from similar papers with pairwise similarity edges.
        
        Args:
            seed_papers: List of (paper_id, similarity_to_seed) tuples
            similarity_threshold: Minimum similarity for edge creation
        """
        graph = nx.Graph()
        
        # Add nodes
        for paper_id, seed_similarity in seed_papers:
            if paper_id not in self.papers:
                continue
                
            paper = self.papers[paper_id]
            graph.add_node(
                paper_id,
                title=paper['title'],
                year=paper['year'],
                authors=paper.get('authors', []),
                seed_similarity=seed_similarity,
                is_seed=(seed_similarity >= 0.99)  # Seed has ~1.0 similarity to itself
            )
        
        # Get embeddings for selected papers
        selected_indices = [self.paper_ids.index(pid) for pid, _ in seed_papers 
                          if pid in self.paper_ids]
        selected_embeddings = self.embeddings[selected_indices]
        
        # Compute pairwise similarities
        print("Computing pairwise similarities...")
        similarities = util.pytorch_cos_sim(selected_embeddings, selected_embeddings)
        
        # Add edges based on similarity
        for i, (paper1_id, _) in enumerate(seed_papers):
            for j, (paper2_id, _) in enumerate(seed_papers):
                if i >= j:  # Skip self and duplicates
                    continue
                    
                sim = float(similarities[i][j])
                if sim > similarity_threshold:
                    graph.add_edge(paper1_id, paper2_id, weight=sim)
        
        print(f"Graph complete: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph


def visualize_embedding_graph(graph: nx.Graph, output_path: Path, iterations: int = 200):
    """Visualize the embedding-based similarity graph."""
    
    # Spring layout with similarity weights
    pos = nx.spring_layout(graph, k=1.5, iterations=iterations, seed=42, weight="weight")
    
    # Find seed node and center it
    seed_nodes = [n for n in graph.nodes() if graph.nodes[n].get('is_seed', False)]
    if seed_nodes:
        seed_id = seed_nodes[0]
        center = np.array([0.5, 0.5])
        current = pos[seed_id]
        pos[seed_id] = current * 0.5 + center * 0.5
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#fafafa")
    ax.set_aspect("equal")
    ax.axis("off")
    
    nodes = list(graph.nodes())
    
    # Node sizes based on similarity to seed
    sizes = []
    for node in nodes:
        if graph.nodes[node].get("is_seed"):
            sizes.append(2000)
        else:
            # Size based on similarity to seed
            sim = graph.nodes[node].get("seed_similarity", 0.5)
            sizes.append(200 + sim * 800)
    
    # Colors by year
    years = [graph.nodes[n].get("year", 2020) for n in nodes]
    min_year = min(years) if years else 2020
    max_year = max(years) if years else 2020
    
    colors = []
    for node in nodes:
        if graph.nodes[node].get("is_seed"):
            colors.append("#e63946")  # Red for seed
        else:
            year = graph.nodes[node].get("year", 2020)
            year_norm = (year - min_year) / max(max_year - min_year, 1) if max_year > min_year else 0.5
            
            # Color gradient
            if year_norm < 0.33:
                colors.append("#caf0f8")
            elif year_norm < 0.66:
                colors.append("#90e0ef")
            else:
                colors.append("#0077b6")
    
    # Draw edges
    for edge in graph.edges(data=True):
        n1, n2, data = edge
        weight = data.get("weight", 0.3)
        
        p1 = pos[n1]
        p2 = pos[n2]
        
        alpha = min(0.6, weight)
        width = max(0.3, weight * 3)
        
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
               color="#94a3b8", alpha=alpha, linewidth=width, zorder=1)
    
    # Draw nodes
    for i, node in enumerate(nodes):
        p = pos[node]
        ax.scatter(p[0], p[1], s=sizes[i], c=[colors[i]],
                  alpha=0.95, edgecolors="white", linewidth=2.5, zorder=2)
    
    # Labels
    for node in nodes:
        p = pos[node]
        title = graph.nodes[node].get("title", "Unknown")
        year = graph.nodes[node].get("year", "")
        
        # Shorten title for label
        label = f"{title[:20]}..., {year}" if len(title) > 20 else f"{title}, {year}"
        
        fontsize = 10 if graph.nodes[node].get("is_seed") else 8
        fontweight = "bold" if graph.nodes[node].get("is_seed") else "normal"
        
        ax.annotate(label, xy=p, xytext=(0, -5), textcoords="offset points",
                   ha="center", va="top", fontsize=fontsize,
                   fontweight=fontweight, color="#1d3557")
    
    ax.set_title("Embedding-Based Paper Similarity Graph", fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper similarity graph using embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("paper_id", help="ArXiv ID or search text")
    parser.add_argument("-o", "--output", type=Path, default=Path("out/embedding_graph.png"))
    parser.add_argument("-p", "--max-papers", type=int, default=40)
    parser.add_argument("-d", "--dataset-size", type=int, default=10000,
                       help="Number of papers to load from dataset")
    parser.add_argument("-m", "--model", default="all-MiniLM-L6-v2",
                       help="Sentence transformer model (or google/embeddinggemma-300m)")
    parser.add_argument("-s", "--similarity-threshold", type=float, default=0.4)
    parser.add_argument("-i", "--iterations", type=int, default=200)
    args = parser.parse_args()
    
    # Build embedding index
    builder = EmbeddingPapersBuilder(model_name=args.model)
    
    # Load dataset
    builder.load_arxiv_dataset(max_papers=args.dataset_size)
    
    # Compute embeddings
    builder.compute_embeddings()
    
    # Find similar papers
    if args.paper_id.startswith("arxiv:") or "." in args.paper_id:
        # Treat as ArXiv ID
        similar_papers = builder.find_similar_papers_from_arxiv(args.paper_id, top_k=args.max_papers)
    else:
        # Treat as search text
        similar_papers = builder.find_similar_papers_from_text(args.paper_id, top_k=args.max_papers)
    
    print(f"\nFound {len(similar_papers)} similar papers")
    for i, (paper_id, score) in enumerate(similar_papers[:5]):
        paper = builder.papers[paper_id]
        print(f"{i+1}. {paper['title'][:60]}... (similarity: {score:.3f})")
    
    # Build graph
    graph = builder.build_similarity_graph(similar_papers, args.similarity_threshold)
    
    # Visualize
    visualize_embedding_graph(graph, args.output, args.iterations)


if __name__ == "__main__":
    main()