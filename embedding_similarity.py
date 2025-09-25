#!/usr/bin/env python3
"""
Embedding-based similarity for Connected Papers visualization.
Uses sentence transformers on abstracts to find semantically similar papers.
Now with proper joblib caching instead of pickle.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer, util
import torch
from semanticscholar import SemanticScholar
from datasets import load_dataset
from joblib import Memory
from tqdm import tqdm

# Set up joblib memory cache
memory = Memory("cache/joblib_cache", verbose=0)


@memory.cache
def load_dataset_cached(
    dataset_split: str, max_papers: Optional[int], categories: Optional[tuple]
) -> Dict:
    """
    Load and cache ArXiv dataset.

    Args:
        dataset_split: HuggingFace dataset split specification
        max_papers: Maximum number of papers to load
        categories: Filter by ArXiv categories

    Returns:
        Dictionary of paper_id -> paper metadata
    """
    papers = {}

    try:
        # Try the ML-focused subset first (smaller, faster)
        dataset = load_dataset("CShorten/ML-ArXiv-Papers", split=dataset_split)
        print(f"Loaded ML-ArXiv-Papers dataset (split: {dataset_split})")
    except Exception as e:
        print(f"Failed to load ML-ArXiv-Papers: {e}")
        # Fallback to smaller abstracts dataset
        try:
            dataset = load_dataset("gfissore/arxiv-abstracts-2021", split=dataset_split)
            print(f"Loaded arxiv-abstracts-2021 dataset (split: {dataset_split})")
        except Exception as e2:
            print(f"Failed to load arxiv-abstracts-2021: {e2}")
            raise

    papers_loaded = 0
    for paper in tqdm(dataset, desc="Loading papers", total=max_papers or len(dataset)):
        if max_papers and papers_loaded >= max_papers:
            break

        # Filter by categories if specified
        if categories and "categories" in paper:
            paper_cats = paper["categories"].split()
            if not any(cat in paper_cats for cat in categories):
                continue

        paper_id = paper.get("id", paper.get("paper_id", f"paper_{papers_loaded}"))

        papers[paper_id] = {
            "title": paper.get("title", "Unknown"),
            "abstract": paper.get("abstract", paper.get("summary", "")),
            "authors": paper.get("authors", paper.get("authors_parsed", [])),
            "year": extract_year(paper),
            "categories": paper.get("categories", ""),
            "num_authors": len(paper.get("authors", paper.get("authors_parsed", []))),
            "citation_count": np.random.randint(0, 500),  # Simulated since dataset lacks this
        }

        papers_loaded += 1

    return papers


def extract_year(paper: dict) -> int:
    """Extract year from paper metadata."""
    if "year" in paper:
        return int(paper["year"]) if paper["year"] else 2020
    if "update_date" in paper:
        return int(paper["update_date"][:4])
    if "id" in paper:
        # ArXiv ID format: YYMM.NNNNN
        try:
            yy = int(paper["id"][:2])
            return 2000 + yy if yy < 50 else 1900 + yy
        except (ValueError, IndexError, KeyError):
            pass
    return 2020  # Default


@memory.cache
def compute_embeddings_cached(texts: tuple, model_name: str) -> torch.Tensor:
    """
    Compute and cache embeddings.

    Args:
        texts: Tuple of text strings to embed
        model_name: Name of the sentence transformer model

    Returns:
        Tensor of embeddings
    """
    print(f"Computing embeddings with {model_name}...")
    model = SentenceTransformer(model_name)

    # Compute embeddings in batches
    batch_size = 32
    all_embeddings = []
    texts = list(texts)  # Convert tuple back to list for processing

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch = texts[i : i + batch_size]
        # Use encode_document for EmbeddingGemma papers
        if hasattr(model, "encode_document"):
            batch_embeddings = model.encode_document(batch, convert_to_numpy=True)
            # L2 normalize for EmbeddingGemma as per requirements
            batch_embeddings = batch_embeddings / np.linalg.norm(
                batch_embeddings, axis=1, keepdims=True
            )
            batch_embeddings = torch.from_numpy(batch_embeddings)
        else:
            batch_embeddings = model.encode(
                batch, convert_to_tensor=True, show_progress_bar=False
            )
        all_embeddings.append(batch_embeddings)

    # Combine all embeddings
    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Computed {len(texts)} embeddings")
    return embeddings


class EmbeddingPapersBuilder:
    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        cache_dir: Path = Path("cache"),
    ):
        """
        Initialize with sentence transformer model.

        Args:
            model_name: HuggingFace model name (default: google/embeddinggemma-300m)
            cache_dir: Directory for caching embeddings (used by joblib)
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.papers = {}
        self.embeddings = None
        self.paper_ids = []

    def load_arxiv_dataset(
        self,
        max_papers: int = None,
        categories: List[str] = None,
        dataset_split: str = "train[:2%]",
    ):
        """
        Load ArXiv dataset from HuggingFace.

        Args:
            max_papers: Maximum number of papers to load (None = no limit)
            categories: Filter by ArXiv categories (e.g., ['cs.CL', 'cs.AI'])
            dataset_split: HuggingFace dataset split specification
        """
        limit_str = f" (up to {max_papers} papers)" if max_papers else ""
        print(f"Loading ArXiv dataset{limit_str}...")

        # Use cached loading (convert categories to tuple for hashing)
        categories_tuple = tuple(categories) if categories else None
        self.papers = load_dataset_cached(dataset_split, max_papers, categories_tuple)
        self.paper_ids = list(self.papers.keys())
        print(f"Loaded {len(self.papers)} papers")

    def compute_embeddings(self):
        """Compute embeddings for all paper abstracts."""
        if self.embeddings is not None and len(self.embeddings) > 0:
            print("Embeddings already computed")
            return

        print("Computing embeddings for abstracts...")

        # Prepare texts for embedding (EmbeddingGemma document format)
        texts = []
        for paper_id in self.paper_ids:
            paper = self.papers[paper_id]
            # Use EmbeddingGemma document format: title | text
            text = f"title: {paper['title']} | text: {paper['abstract']}"
            texts.append(text)

        # Use cached computation (convert to tuple for hashing)
        self.embeddings = compute_embeddings_cached(tuple(texts), self.model_name)

    def find_similar_papers_from_text(
        self, query_text: str, top_k: int = 40
    ) -> List[Tuple[str, float]]:
        """
        Find similar papers to a query text.

        Args:
            query_text: Title and/or abstract to search for
            top_k: Number of similar papers to return

        Returns:
            List of (paper_id, similarity_score) tuples
        """
        if self.embeddings is None:
            raise ValueError(
                "Embeddings not computed. Call compute_embeddings() first."
            )

        # Encode query properly for EmbeddingGemma
        if hasattr(self.model, "encode_query"):
            # Format as task-specific query for retrieval
            query_prompt = f"task: search result | query: {query_text}"
            query_embedding = self.model.encode_query(
                query_prompt, convert_to_numpy=True
            )
            # L2 normalize for EmbeddingGemma
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = torch.from_numpy(query_embedding)
        else:
            query_embedding = self.model.encode(query_text, convert_to_tensor=True)

        # Compute cosine similarities
        similarities = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]

        # Get top-k most similar papers (excluding exact match if present)
        top_results = torch.topk(similarities, min(top_k + 1, len(similarities)))

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            paper_id = self.paper_ids[idx.item()]
            # Skip self-similarity (score > 0.999 indicates same paper)
            if score.item() > 0.999:
                continue
            results.append((paper_id, score.item()))
            if len(results) >= top_k:
                break

        return results

    def find_similar_papers_from_arxiv(
        self, arxiv_id: str, top_k: int = 40
    ) -> List[Tuple[str, float]]:
        """
        Find similar papers to a given ArXiv paper.

        Args:
            arxiv_id: ArXiv ID (e.g., "1706.03762")
            top_k: Number of similar papers to return
        """
        clean_id = arxiv_id.replace("arxiv:", "").replace("v1", "").replace("v2", "")

        if clean_id in self.papers:
            paper = self.papers[clean_id]
            query_text = f"{paper['title']}. {paper['abstract']}"
        else:
            # Fetch from Semantic Scholar
            print(f"Paper {arxiv_id} not in dataset, fetching from Semantic Scholar...")
            client = SemanticScholar()
            paper = client.get_paper(f"arxiv:{clean_id}")
            if not paper:
                raise ValueError(f"Paper {arxiv_id} not found")
            
            # Add to papers dict with real citation data
            self.papers[clean_id] = {
                "title": paper.title,
                "abstract": paper.abstract or "",
                "authors": paper.authors or [],
                "year": paper.year or 2020,
                "categories": "",
                "citation_count": paper.citationCount or 0,
                "num_authors": len(paper.authors or []),
            }
            self.paper_ids.append(clean_id)
            
            # Add embedding for this paper
            query_text = f"title: {paper.title} | text: {paper.abstract or ''}"
            if hasattr(self.model, "encode_document"):
                new_embedding = self.model.encode_document([query_text], convert_to_numpy=True)
                new_embedding = new_embedding / np.linalg.norm(new_embedding, axis=1, keepdims=True)
                new_embedding = torch.from_numpy(new_embedding)
            else:
                new_embedding = self.model.encode([query_text], convert_to_tensor=True)
            
            if self.embeddings is not None:
                self.embeddings = torch.cat([self.embeddings, new_embedding], dim=0)
            else:
                self.embeddings = new_embedding
            
            query_text = f"{paper.title}. {paper.abstract or ''}"

        return self.find_similar_papers_from_text(query_text, top_k)

    def build_similarity_graph(
        self, seed_papers: List[Tuple[str, float]], similarity_threshold: float = 0.3
    ) -> nx.Graph:
        """
        Build graph from similar papers.

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
                title=paper["title"],
                year=paper["year"],
                authors=paper.get("authors", []),
                categories=paper.get("categories", ""),
                citation_count=paper.get("citation_count", 0),
                seed_similarity=seed_similarity,
                is_seed=(seed_similarity >= 0.99),  # Seed has ~1.0 similarity to itself
            )

        # Compute pairwise similarities for edges
        # Computing pairwise similarities with progress bar

        # Get embeddings for papers in graph
        paper_indices = []
        for paper_id, _ in seed_papers:
            if paper_id in self.paper_ids:
                idx = self.paper_ids.index(paper_id)
                paper_indices.append(idx)

        # Add edges using top-k approach for cleaner visualization
        # Each node connects only to its k most similar neighbors
        k_neighbors = 2  # Sparse connections like reference

        for i, (paper1_id, _) in enumerate(tqdm(seed_papers, desc="Computing edges")):
            if paper1_id not in self.paper_ids:
                continue

            idx1 = self.paper_ids.index(paper1_id)
            similarities = []

            # Compute similarities to all other papers
            for j, (paper2_id, _) in enumerate(seed_papers):
                if i >= j or paper2_id not in self.paper_ids:
                    continue

                idx2 = self.paper_ids.index(paper2_id)
                
                # Multi-factor similarity calculation
                # 1. Embedding similarity
                embed_sim = util.pytorch_cos_sim(
                    self.embeddings[idx1], self.embeddings[idx2]
                ).item()
                
                # 2. Year proximity factor (papers close in time are more related)
                year1 = self.papers[paper1_id].get("year", 2020)
                year2 = self.papers[paper2_id].get("year", 2020) 
                year_diff = abs(year1 - year2)
                year_factor = 1.0 / (1.0 + year_diff / 3.0)  # Decay over 3 years
                
                # 3. Category overlap (shared research areas)
                cats1 = set(self.papers[paper1_id].get("categories", "").split())
                cats2 = set(self.papers[paper2_id].get("categories", "").split())
                if cats1 and cats2:
                    category_overlap = len(cats1 & cats2) / len(cats1 | cats2)
                else:
                    category_overlap = 0.3  # Default if no categories
                
                # 4. Author collaboration (shared authors = stronger connection)
                auth1 = set(str(a) for a in self.papers[paper1_id].get("authors", []))
                auth2 = set(str(a) for a in self.papers[paper2_id].get("authors", []))
                if auth1 & auth2:  # Shared authors
                    author_factor = 1.5
                else:
                    author_factor = 1.0
                
                # Combined similarity with weights
                sim = (
                    0.5 * embed_sim +           # Semantic similarity
                    0.2 * year_factor +          # Temporal proximity  
                    0.2 * category_overlap +     # Research area overlap
                    0.1 * author_factor          # Collaboration bonus
                )

                # Adaptive threshold based on year difference
                threshold = 0.5 if year_diff < 2 else 0.55 if year_diff < 5 else 0.6
                if sim > threshold:
                    similarities.append((paper2_id, sim))

            # Add only top-k edges for this node
            similarities.sort(key=lambda x: x[1], reverse=True)
            for paper2_id, sim in similarities[:k_neighbors]:
                graph.add_edge(paper1_id, paper2_id, weight=sim)

        print(
            f"Graph complete: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        return graph


def visualize_graph(graph: nx.Graph, output_path: Path, iterations: int = 300):
    """Visualize the similarity graph."""

    # Use Kamada-Kawai for more organic clustering like reference
    try:
        pos = nx.kamada_kawai_layout(
            graph, weight="weight", scale=0.9, center=[0.5, 0.5]
        )
    except Exception:
        # Fallback to spring if graph structure causes issues
        pos = nx.spring_layout(
            graph,
            k=1.2 / np.sqrt(graph.number_of_nodes()),  # More spacing
            iterations=iterations,
            seed=42,
            weight="weight",
            scale=0.9,
            center=[0.5, 0.5],
        )

    # Add small random perturbations for organic look
    for node in pos:
        pos[node] += np.random.normal(0, 0.015, 2)

    # Find and gently center seed node
    seed_nodes = [n for n in graph.nodes() if graph.nodes[n].get("is_seed", False)]
    if seed_nodes:
        seed_id = seed_nodes[0]
        center = np.array([0.5, 0.5])
        current = pos[seed_id]
        # Only adjust if far from center
        if np.linalg.norm(current - center) > 0.2:
            pos[seed_id] = current * 0.8 + center * 0.2

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#fafafa")
    ax.set_aspect("equal")
    ax.axis("off")

    nodes = list(graph.nodes())

    # Node sizes with extreme variation matching reference
    sizes = []
    colors = []

    # Sort nodes by combined importance (similarity + citations)
    sorted_by_importance = sorted(
        nodes, 
        key=lambda n: (
            graph.nodes[n].get("seed_similarity", 0) * 0.7 +
            min(graph.nodes[n].get("citation_count", 0) / 1000, 1.0) * 0.3
        ),
        reverse=True
    )

    for node in nodes:
        rank = sorted_by_importance.index(node)
        citations = graph.nodes[node].get("citation_count", 0)

        if graph.nodes[node].get("is_seed"):
            sizes.append(2500)  # Seed is always prominent
            colors.append("#e63946")
        else:
            # Size based on rank AND citation count
            base_size = 80
            
            # Rank component
            if rank == 1:  # Most important non-seed
                rank_size = 1200
            elif rank < 4:  # Top 3
                rank_size = 800 + (4 - rank) * 150
            elif rank < 8:  # Next tier
                rank_size = 400 + (8 - rank) * 50
            elif rank < 15:  # Medium tier
                rank_size = 150 + (15 - rank) * 15
            else:  # Lower importance
                rank_size = 50
            
            # Citation component (log scale for better distribution)
            if citations > 0:
                citation_size = np.log10(citations + 1) * 100
            else:
                citation_size = 0
            
            # Combined size with variation
            final_size = base_size + rank_size + citation_size
            sizes.append(min(final_size, 2000))  # Cap at 2000

            # Smooth color gradient by year (matching reference)
            year = graph.nodes[node].get("year", 2020)
            years = [graph.nodes[n].get("year", 2020) for n in nodes]
            min_year = min(years) if years else 2020
            max_year = max(years) if years else 2020

            if max_year > min_year:
                year_norm = (year - min_year) / (max_year - min_year)
                # Smooth RGB gradient from light blue-gray to dark teal
                r = 0.72 - 0.27 * year_norm  # 184 -> 69
                g = 0.83 - 0.19 * year_norm  # 212 -> 123
                b = 0.89 - 0.28 * year_norm  # 227 -> 157
                colors.append((r, g, b))
            else:
                colors.append((0.56, 0.73, 0.82))  # Default medium blue

    # Draw edges
    for edge in graph.edges(data=True):
        n1, n2, data = edge
        weight = data.get("weight", 0.3)

        p1 = pos[n1]
        p2 = pos[n2]

        # Very thin, subtle edges like reference
        alpha = min(0.3, weight * 0.5)  # More transparent
        width = max(0.3, weight * 1.5)  # Thinner lines

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
            alpha=0.95,
            edgecolors="white",
            linewidth=2.5,
            zorder=2,
        )

    # Labels
    for node in nodes:
        p = pos[node]
        paper = graph.nodes[node]

        title = paper.get("title", "Unknown")
        year = paper.get("year", "")

        # Use author last name like reference (e.g., "Köhler, 2019")
        authors = paper.get("authors", [])
        if authors:
            # Extract first author's last name
            if isinstance(authors[0], str):
                # Try to get last name from string
                author_parts = authors[0].split()
                author_name = author_parts[-1] if author_parts else "Unknown"
            else:
                author_name = "Unknown"
        else:
            # Fallback to shortened title if no authors
            author_name = title[:10]

        label = f"{author_name}, {year}"

        fontsize = 10 if paper.get("is_seed") else 8
        fontweight = "bold" if paper.get("is_seed") else "normal"

        ax.annotate(
            label,
            xy=p,
            xytext=(0, -5),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=fontsize,
            fontweight=fontweight,
            color="#1d3557",
        )

    ax.set_title("Embedding-Based Paper Similarity Graph", fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embedding-based similarity graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("paper_id", help="ArXiv ID or search text")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (auto-generates from paper title if not specified)",
    )
    parser.add_argument("-p", "--max-papers", type=int, default=40)
    parser.add_argument(
        "-d",
        "--dataset-size",
        type=int,
        default=None,
        help="Max number of papers to process from dataset",
    )
    parser.add_argument(
        "--dataset-split",
        default="train[:2%]",
        help="Dataset split specification (e.g., 'train', 'train[:10%%]', 'train[:1000]')",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="google/embeddinggemma-300m",
        help="Sentence transformer model",
    )
    parser.add_argument("-s", "--similarity-threshold", type=float, default=0.4)
    parser.add_argument("-i", "--iterations", type=int, default=200)
    args = parser.parse_args()

    # Build embedding index
    builder = EmbeddingPapersBuilder(model_name=args.model)

    # Load dataset
    builder.load_arxiv_dataset(
        max_papers=args.dataset_size, dataset_split=args.dataset_split
    )

    # Compute embeddings
    builder.compute_embeddings()

    # Find similar papers
    if args.paper_id.startswith("arxiv:") or "." in args.paper_id:
        # It's an ArXiv ID
        similar_papers = builder.find_similar_papers_from_arxiv(
            args.paper_id, top_k=args.max_papers
        )
    else:
        # It's a text query
        similar_papers = builder.find_similar_papers_from_text(
            args.paper_id, top_k=args.max_papers
        )

    print(f"\nFound {len(similar_papers)} similar papers")
    for i, (paper_id, score) in enumerate(similar_papers[:5]):
        paper = builder.papers.get(paper_id, {"title": "Unknown"})
        print(f"{i + 1}. {paper['title'][:60]}... (similarity: {score:.3f})")

    # Build and visualize graph
    graph = builder.build_similarity_graph(similar_papers, args.similarity_threshold)

    # Auto-generate safe filename if not specified
    if args.output is None:
        import re

        # Get seed paper title
        if similar_papers:
            seed_id = similar_papers[0][0]  # First paper is usually the seed
            if seed_id in builder.papers:
                title = builder.papers[seed_id]["title"]
            else:
                # Fetch from Semantic Scholar if needed
                clean_id = (
                    args.paper_id.replace("arxiv:", "")
                    .replace("v1", "")
                    .replace("v2", "")
                )
                client = SemanticScholar()
                paper = client.get_paper(f"arxiv:{clean_id}")
                title = paper.title if paper else "unknown"
        else:
            title = "embedding_graph"

        # Create safe filename (max 40 chars from title)
        safe_title = re.sub(r"[^\w\s-]", "", title[:40]).strip()
        safe_title = re.sub(r"[-\s]+", "-", safe_title).lower()
        output_path = Path("out") / f"{safe_title}.png"
    else:
        output_path = args.output

    # Create output directory
    output_path.parent.mkdir(exist_ok=True)
    visualize_graph(graph, output_path, args.iterations)


if __name__ == "__main__":
    main()
