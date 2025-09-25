#!/usr/bin/env python3
"""
Hybrid similarity for Connected Papers visualization.
Combines citation relationships from Semantic Scholar with
semantic similarity from sentence transformers.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import math
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer, util
from semanticscholar import SemanticScholar
from datasets import load_dataset
from joblib import Memory
from tqdm import tqdm

# Set up joblib memory cache
memory = Memory("cache/joblib_cache", verbose=0)


@memory.cache
def load_arxiv_corpus_cached(dataset_split: str, max_papers: Optional[int]) -> Dict:
    """Load and cache ArXiv corpus."""
    papers = {}

    try:
        dataset = load_dataset("CShorten/ML-ArXiv-Papers", split=dataset_split)
        print(f"Using ML-ArXiv-Papers dataset (split: {dataset_split})")
    except Exception:
        try:
            dataset = load_dataset("gfissore/arxiv-abstracts-2021", split=dataset_split)
            print(f"Using arxiv-abstracts-2021 dataset (split: {dataset_split})")
        except Exception:
            print("Warning: Could not load ArXiv dataset")
            return papers

    for i, paper in enumerate(
        tqdm(dataset, desc="Loading papers", total=max_papers or len(dataset))
    ):
        if max_papers and i >= max_papers:
            break

        paper_id = paper.get("id", paper.get("paper_id", f"arxiv_{i}"))

        # Extract year
        year = 2020
        if "year" in paper and paper["year"]:
            year = int(paper["year"])
        elif "update_date" in paper:
            try:
                year = int(paper["update_date"][:4])
            except (ValueError, IndexError):
                pass

        papers[paper_id] = {
            "title": paper.get("title", "Unknown"),
            "abstract": paper.get("abstract", paper.get("summary", "")),
            "year": year,
            "authors": paper.get("authors", []),
        }

    return papers


# Removed embedding caching - not needed for hybrid approach
# ArXiv corpus caching is sufficient


class HybridPapersBuilder:
    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        cache_dir: Path = Path("cache"),
    ):
        """
        Initialize with both Semantic Scholar and sentence transformer.

        Args:
            model_name: HuggingFace model name for embeddings (default: google/embeddinggemma-300m)
            cache_dir: Directory for caching
        """
        self.semantic_scholar = SemanticScholar()
        self.model_name = model_name
        self.sentence_model = None  # Lazy load when needed

        # Data storage
        self.arxiv_papers = {}  # Background corpus for finding similar papers
        self.graph_papers = {}  # Papers actually in the graph (from citations)
        self.embeddings = {}  # Paper ID -> embedding

    def load_arxiv_corpus(
        self, max_papers: int = None, dataset_split: str = "train[:2%]"
    ):
        """Load ArXiv corpus for semantic similarity matching.

        Args:
            max_papers: Maximum number of papers to process
            dataset_split: HuggingFace dataset split specification
        """

        print("Loading ArXiv corpus...")
        self.arxiv_papers = load_arxiv_corpus_cached(dataset_split, max_papers)
        print(f"Loaded {len(self.arxiv_papers)} papers")

    def _cache_embeddings(self):
        """Helper to trigger joblib caching of embeddings."""
        # For hybrid, we don't actually need to cache embeddings separately
        # since they're computed fresh each time from the cached ArXiv corpus
        pass

    def _extract_year(self, paper: dict) -> int:
        """Extract year from paper metadata."""
        if "year" in paper:
            return int(paper["year"]) if paper["year"] else 2020
        if "update_date" in paper:
            try:
                return int(paper["update_date"][:4])
            except (ValueError, IndexError, KeyError):
                pass
        return 2020

    def get_citations_and_references(
        self, paper_id: str, max_citations: int = 20, max_references: int = 20
    ) -> str:
        """
        Get citations and references from Semantic Scholar.
        Also fetches abstracts for embedding computation.
        Returns the seed paper ID.
        """
        print("Fetching citations and references for seed paper...")

        # Get seed paper with more fields for better analysis
        seed = self.semantic_scholar.get_paper(
            paper_id,
            fields=[
                "title",
                "year",
                "authors",
                "citationCount",
                "paperId",
                "fieldsOfStudy",
                "abstract",
            ],
        )

        if not seed:
            raise ValueError(f"Paper {paper_id} not found")

        seed_id = seed.paperId

        # Store seed paper with all metadata
        self.graph_papers[seed_id] = {
            "title": seed.title,
            "abstract": seed.abstract or "",
            "year": seed.year or 2020,
            "authors": [a.name for a in (seed.authors or [])[:5]],  # Keep more authors
            "citation_count": seed.citationCount or 0,
            "fields": seed.fieldsOfStudy or [],
            "is_seed": True,
        }

        print(f"Seed: {seed.title[:50]}...")

        # Fetch citations with intelligent filtering
        print(f"Fetching up to {max_citations} citations...")
        try:
            citations = self.semantic_scholar.get_paper_citations(
                seed_id,
                limit=max_citations * 2,  # Fetch extra to filter
            )
            citation_papers = []
            for cit in citations:
                if (
                    hasattr(cit, "paper")
                    and cit.paper
                    and hasattr(cit.paper, "paperId")
                ):
                    p = cit.paper
                    # Calculate relevance score based on year and citations
                    year_diff = abs((seed.year or 2020) - (p.year or 2020))
                    relevance = (p.citationCount or 0) / (1 + year_diff)
                    citation_papers.append((p, relevance))

            # Sort by relevance and take top citations
            citation_papers.sort(key=lambda x: x[1], reverse=True)
            for p, relevance in citation_papers[:max_citations]:
                self.graph_papers[p.paperId] = {
                    "title": p.title or "Unknown",
                    "abstract": "",  # Will fetch if needed
                    "year": p.year or 2020,
                    "authors": [a.name for a in (p.authors or [])[:3]],
                    "citation_count": p.citationCount or 0,
                    "is_seed": False,
                    "relationship": "citation",
                    "relevance_score": relevance,
                }
        except (AttributeError, TypeError) as e:
            print(f"  Warning: Citations fetch issue: {e}")

        # Fetch references (same as citation_graph.py)
        print(f"Fetching up to {max_references} references...")
        try:
            references = self.semantic_scholar.get_paper_references(
                seed_id, limit=max_references
            )
            for ref in references:
                if (
                    hasattr(ref, "paper")
                    and ref.paper
                    and hasattr(ref.paper, "paperId")
                ):
                    p = ref.paper
                    self.graph_papers[p.paperId] = {
                        "title": p.title or "Unknown",
                        "abstract": "",  # Don't fetch abstract to avoid timeouts
                        "year": p.year or 2020,
                        "authors": [a.name for a in (p.authors or [])[:3]],
                        "citation_count": p.citationCount or 0,
                        "is_seed": False,
                        "relationship": "reference",
                    }
        except (AttributeError, TypeError) as e:
            print(f"  Warning: References fetch issue: {e}")

        print(f"Collected {len(self.graph_papers)} papers from citations/references")
        return seed_id

    def compute_embeddings(self):
        """Compute embeddings for all papers (both graph and corpus)."""

        # Lazy load model
        if self.sentence_model is None:
            print(f"Loading sentence transformer model: {self.model_name}")
            self.sentence_model = SentenceTransformer(self.model_name)

        print("Computing embeddings for all papers...")

        # Combine graph papers and ArXiv corpus
        all_papers = {}
        all_papers.update(self.graph_papers)
        all_papers.update(self.arxiv_papers)

        # Compute embeddings in batches
        batch_size = 32
        paper_ids = list(all_papers.keys())

        for i in range(0, len(paper_ids), batch_size):
            batch_ids = paper_ids[i : i + batch_size]
            batch_texts = []

            for pid in batch_ids:
                paper = all_papers[pid]
                text = f"{paper['title']}. {paper.get('abstract', '')}"
                batch_texts.append(text)

            # Compute batch embeddings (EmbeddingGemma has special encode_document)
            if hasattr(self.sentence_model, "encode_document"):
                batch_embeddings = self.sentence_model.encode_document(
                    batch_texts, convert_to_tensor=True, show_progress_bar=False
                )
            else:
                batch_embeddings = self.sentence_model.encode(
                    batch_texts, convert_to_tensor=True, show_progress_bar=False
                )

            # Store embeddings
            for j, pid in enumerate(batch_ids):
                self.embeddings[pid] = batch_embeddings[j]

            if ((i // batch_size) + 1) % 10 == 0:
                print(
                    f"  Computed {min(i + batch_size, len(paper_ids))}/{len(paper_ids)} embeddings..."
                )

        print(f"Computed embeddings for {len(self.embeddings)} papers")

    def find_semantically_similar(
        self, seed_id: str, top_k: int = 10, fetch_metadata: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Find papers from ArXiv corpus that are semantically similar to seed.
        These supplement the citation-based papers.
        """
        if seed_id not in self.embeddings:
            print(f"Warning: No embedding for seed {seed_id}")
            return []

        print("Finding semantically similar papers from corpus...")

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
                seed_embedding, self.embeddings[paper_id]
            ).item()

            # Skip self-similarity
            if similarity > 0.999:
                continue

            similar_papers.append((paper_id, similarity))

        # Sort by similarity and return top-k
        similar_papers.sort(key=lambda x: x[1], reverse=True)
        top_similar = similar_papers[:top_k]

        # Add to graph papers with optional metadata fetch
        print(f"Fetching metadata for top {len(top_similar)} semantic matches...")
        for i, (paper_id, sim_score) in enumerate(top_similar):
            if paper_id in self.arxiv_papers:
                paper = self.arxiv_papers[paper_id]

                # Try to get citation data from Semantic Scholar if similar enough
                citation_count = 0
                if (
                    fetch_metadata and sim_score > 0.6 and i < 5
                ):  # Only for very similar papers
                    try:
                        # Try to fetch from S2 using arxiv ID
                        s2_paper = self.semantic_scholar.get_paper(
                            f"arxiv:{paper_id}", fields=["citationCount"]
                        )
                        if s2_paper:
                            citation_count = s2_paper.citationCount or 0
                    except Exception:
                        pass  # Use default 0

                self.graph_papers[paper_id] = {
                    "title": paper["title"],
                    "abstract": paper.get("abstract", ""),
                    "year": paper["year"],
                    "authors": paper.get("authors", []),
                    "citation_count": citation_count,
                    "is_seed": False,
                    "relationship": "semantic",
                    "semantic_similarity": sim_score,
                }

        print(f"Added {len(top_similar)} semantically similar papers")
        return top_similar

    def analyze_co_citations(self) -> Dict[str, Dict[str, float]]:
        """Analyze co-citation patterns between papers."""
        co_citations = {}

        # Papers that are both citations or both references likely share themes
        citation_papers = [
            p
            for p, d in self.graph_papers.items()
            if d.get("relationship") == "citation"
        ]
        reference_papers = [
            p
            for p, d in self.graph_papers.items()
            if d.get("relationship") == "reference"
        ]

        # Co-citation strength for papers in same group
        for group in [citation_papers, reference_papers]:
            for p1 in group:
                if p1 not in co_citations:
                    co_citations[p1] = {}
                for p2 in group:
                    if p1 != p2:
                        # Base co-citation score
                        score = 0.3

                        # Boost if similar years (likely same research wave)
                        y1 = self.graph_papers[p1].get("year", 2020)
                        y2 = self.graph_papers[p2].get("year", 2020)
                        if abs(y1 - y2) < 2:
                            score += 0.2

                        co_citations[p1][p2] = score

        return co_citations

    def build_hybrid_graph(
        self, seed_id: str, similarity_threshold: float = 0.3
    ) -> nx.Graph:
        """
        Build graph with hybrid similarity:
        - Citation relationships (from Semantic Scholar)
        - Semantic similarity (from embeddings)
        - Co-citation and bibliographic coupling
        - Multi-factor similarity scoring
        """
        graph = nx.Graph()

        # Analyze co-citation patterns
        co_citations = self.analyze_co_citations()

        # Add all nodes
        for paper_id, paper in self.graph_papers.items():
            graph.add_node(
                paper_id,
                title=paper["title"],
                year=paper["year"],
                authors=paper.get("authors", []),
                citation_count=paper.get("citation_count", 0),
                is_seed=paper.get("is_seed", False),
                relationship=paper.get("relationship", "unknown"),
            )

        print("Computing pairwise similarities for graph edges...")

        paper_ids = list(self.graph_papers.keys())
        edge_count = 0

        # Compute pairwise similarities
        for i, p1_id in enumerate(paper_ids):
            for j in range(i + 1, len(paper_ids)):
                p2_id = paper_ids[j]

                # Initialize similarity components
                paper1 = self.graph_papers[p1_id]
                paper2 = self.graph_papers[p2_id]

                # 1. Semantic similarity from embeddings
                semantic_similarity = 0
                if p1_id in self.embeddings and p2_id in self.embeddings:
                    semantic_similarity = util.pytorch_cos_sim(
                        self.embeddings[p1_id], self.embeddings[p2_id]
                    ).item()

                # 2. Co-citation similarity
                co_citation_similarity = 0
                if p1_id in co_citations and p2_id in co_citations[p1_id]:
                    co_citation_similarity = co_citations[p1_id][p2_id]

                # 3. Temporal similarity with stronger decay
                year1 = paper1["year"]
                year2 = paper2["year"]
                year_diff = abs(year1 - year2)
                if year_diff < 2:
                    temporal_similarity = 1.0
                elif year_diff < 5:
                    temporal_similarity = 0.7 - (year_diff - 2) * 0.15
                else:
                    temporal_similarity = 0.2 * math.exp(-(year_diff - 5) / 3)

                # 4. Author overlap (collaboration indicator)
                auth1 = set(paper1.get("authors", []))
                auth2 = set(paper2.get("authors", []))
                author_overlap = 1.2 if auth1 & auth2 else 1.0

                # 5. Field similarity (if available)
                field_similarity = 0
                fields1 = set(paper1.get("fields", []))
                fields2 = set(paper2.get("fields", []))
                if fields1 and fields2:
                    field_similarity = len(fields1 & fields2) / len(fields1 | fields2)

                # 6. Citation count similarity (papers with similar impact)
                cit1 = paper1.get("citation_count", 0)
                cit2 = paper2.get("citation_count", 0)
                if cit1 > 0 and cit2 > 0:
                    citation_ratio = min(cit1, cit2) / max(cit1, cit2)
                else:
                    citation_ratio = 0.5

                # Intelligent combination based on relationship types
                rel1 = paper1.get("relationship", "")
                rel2 = paper2.get("relationship", "")

                if rel1 == "semantic" and rel2 == "semantic":
                    # Both from embeddings - weight semantic heavily
                    combined_similarity = (
                        0.6 * semantic_similarity
                        + 0.2 * temporal_similarity
                        + 0.1 * field_similarity
                        + 0.1 * citation_ratio
                    )
                elif rel1 in ["citation", "reference"] and rel2 in [
                    "citation",
                    "reference",
                ]:
                    # Both from citations - use co-citation
                    combined_similarity = (
                        0.3 * semantic_similarity
                        + 0.3 * co_citation_similarity
                        + 0.2 * temporal_similarity
                        + 0.1 * field_similarity
                        + 0.1 * citation_ratio
                    )
                else:
                    # Mixed - balanced approach
                    combined_similarity = (
                        0.4 * semantic_similarity
                        + 0.2 * co_citation_similarity
                        + 0.2 * temporal_similarity
                        + 0.1 * field_similarity
                        + 0.1 * citation_ratio
                    )

                # Apply author overlap bonus
                combined_similarity *= author_overlap

                # Special handling for seed connections
                if p1_id == seed_id or p2_id == seed_id:
                    # Always connect highly similar papers to seed
                    if semantic_similarity > 0.7 or co_citation_similarity > 0.4:
                        combined_similarity = max(
                            combined_similarity, similarity_threshold * 1.2
                        )
                    # But be selective - seed shouldn't connect to everything
                    elif combined_similarity < similarity_threshold * 0.7:
                        continue

                # Adaptive threshold based on relationship types
                if rel1 == rel2:  # Same relationship type
                    threshold = similarity_threshold * 0.9
                else:
                    threshold = similarity_threshold

                # Add edge if similar enough
                if combined_similarity > threshold:
                    # Limit edges per node for cleaner graph
                    p1_edges = len([e for e in graph.edges() if p1_id in e])
                    p2_edges = len([e for e in graph.edges() if p2_id in e])

                    if p1_edges < 5 and p2_edges < 5:  # Max 5 connections per node
                        graph.add_edge(p1_id, p2_id, weight=combined_similarity)
                        edge_count += 1

        print(f"Graph complete: {graph.number_of_nodes()} nodes, {edge_count} edges")
        return graph


def visualize_hybrid_graph(graph: nx.Graph, output_path: Path, iterations: int = 300):
    """Visualize the hybrid similarity graph with all improvements."""

    # Use Kamada-Kawai for organic clustering like reference
    try:
        pos = nx.kamada_kawai_layout(
            graph, weight="weight", scale=0.9, center=[0.5, 0.5]
        )
    except Exception:
        # Fallback to spring if graph structure causes issues
        pos = nx.spring_layout(
            graph,
            k=1.2 / np.sqrt(graph.number_of_nodes()),
            iterations=iterations,
            seed=42,
            weight="weight",
            scale=0.9,
            center=[0.5, 0.5],
        )

    # Add small random perturbations for organic look
    for node in pos:
        pos[node] += np.random.normal(0, 0.015, 2)

    # Find and center seed node
    seed_nodes = [n for n in graph.nodes() if graph.nodes[n].get("is_seed", False)]
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

    # Node sizes with extreme variation (citation + relationship importance)
    sizes = []
    # Sort by citation count for ranking
    sorted_by_citations = sorted(
        nodes, key=lambda n: graph.nodes[n].get("citation_count", 0), reverse=True
    )

    for node in nodes:
        if graph.nodes[node].get("is_seed"):
            sizes.append(2500)  # Seed is always largest
        else:
            # Rank-based sizing like reference
            rank = sorted_by_citations.index(node)
            cit = graph.nodes[node].get("citation_count", 0)
            rel = graph.nodes[node].get("relationship", "")

            # Base size on rank
            if rank == 0:  # Highest cited non-seed
                base_size = 1800
            elif rank < 3:
                base_size = 1200 - rank * 150
            elif rank < 8:
                base_size = 600 - rank * 40
            elif rank < 15:
                base_size = 300 - rank * 10
            else:
                base_size = 100

            # Bonus for semantic matches (they're specifically chosen)
            if rel == "semantic":
                base_size *= 1.3

            # Add citation-based variation
            if cit > 0:
                citation_bonus = min(200, np.log10(cit + 1) * 50)
            else:
                citation_bonus = 0

            sizes.append(min(base_size + citation_bonus, 2200))

    # Smooth color gradient by year (like reference) with relationship hints
    colors = []
    years = [graph.nodes[n].get("year", 2020) for n in nodes]
    min_year = min(years) if years else 2020
    max_year = max(years) if years else 2020

    for node in nodes:
        if graph.nodes[node].get("is_seed"):
            colors.append("#e63946")  # Red for seed
        else:
            year = graph.nodes[node].get("year", 2020)
            rel = graph.nodes[node].get("relationship", "")

            # Base color on year gradient
            if max_year > min_year:
                year_norm = (year - min_year) / (max_year - min_year)
            else:
                year_norm = 0.5

            # Different gradients for different relationships
            if rel == "semantic":
                # Orange gradient for semantic matches
                r = 0.95 - 0.15 * year_norm
                g = 0.5 - 0.1 * year_norm
                b = 0.1 + 0.1 * year_norm
            elif rel == "citation":
                # Green gradient for citations
                r = 0.2 + 0.2 * year_norm
                g = 0.85 - 0.15 * year_norm
                b = 0.4 + 0.2 * year_norm
            elif rel == "reference":
                # Blue gradient for references
                r = 0.2 + 0.15 * year_norm
                g = 0.5 + 0.15 * year_norm
                b = 0.85 - 0.15 * year_norm
            else:
                # Gray-blue gradient for others
                r = 0.58 + 0.14 * year_norm
                g = 0.64 + 0.15 * year_norm
                b = 0.72 - 0.1 * year_norm

            colors.append((r, g, b))

    # Draw edges
    for edge in graph.edges(data=True):
        n1, n2, data = edge
        weight = data.get("weight", 0.3)

        p1 = pos[n1]
        p2 = pos[n2]

        # Very thin, subtle edges like reference
        alpha = min(0.3, weight * 0.6)
        width = max(0.3, weight * 2)

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="#cbd5e0",
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

    # Legend for relationship types
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#e63946", label="Seed Paper"),
        Patch(facecolor="#06ffa5", label="Citations"),
        Patch(facecolor="#0077b6", label="References"),
        Patch(facecolor="#f77f00", label="Semantic Similarity"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=False)

    # Title
    seed_node = seed_nodes[0] if seed_nodes else nodes[0]
    title = graph.nodes[seed_node].get("title", "Unknown")[:60]
    ax.set_title(
        f"Hybrid Connected Papers: {title}...", fontsize=16, pad=20, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hybrid citation + semantic similarity graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("paper_id", help="Paper identifier (DOI, arXiv ID, or S2 ID)")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-p", "--max-papers", type=int, default=40)
    parser.add_argument(
        "-c",
        "--corpus-size",
        type=int,
        default=None,
        help="Max papers to process from ArXiv corpus",
    )
    parser.add_argument(
        "--dataset-split",
        default="train[:2%]",
        help="Dataset split specification (e.g., 'train', 'train[:10%%]', 'train[:1000]')",
    )
    parser.add_argument("-s", "--similarity-threshold", type=float, default=0.35)
    parser.add_argument(
        "-m",
        "--model",
        default="google/embeddinggemma-300m",
        help="Sentence transformer model",
    )
    parser.add_argument("-i", "--iterations", type=int, default=200)
    parser.add_argument(
        "--max-semantic",
        type=int,
        default=10,
        help="Max papers to add from semantic similarity",
    )
    args = parser.parse_args()

    # Initialize builder
    builder = HybridPapersBuilder(model_name=args.model)

    # Load ArXiv corpus for semantic matching
    builder.load_arxiv_corpus(
        max_papers=args.corpus_size, dataset_split=args.dataset_split
    )

    # Get citations and references from Semantic Scholar
    seed_id = builder.get_citations_and_references(
        args.paper_id,
        max_citations=args.max_papers // 2,
        max_references=args.max_papers // 2,
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

        title = builder.graph_papers[seed_id]["title"]
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
