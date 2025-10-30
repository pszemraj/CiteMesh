"""
Embedding-based graph building strategy.

This strategy uses semantic similarity from sentence transformers
to find conceptually similar papers without relying on citations.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from joblib import Memory
from tqdm import tqdm

from citemesh.strategies.base import GraphBuilderStrategy
from citemesh.models import Paper, Author
from citemesh.api_client import get_client
from citemesh.config import EMBEDDING_CONFIG
import logging

logger = logging.getLogger(__name__)

# Set up joblib cache
memory = Memory("cache/joblib_cache", verbose=0)


@memory.cache
def load_arxiv_dataset_cached(
    dataset_split: str, max_papers: Optional[int]
) -> Dict[str, Dict]:
    """
    Load and cache ArXiv dataset.

    Args:
        dataset_split: Dataset split (e.g., "train[:2%]")
        max_papers: Maximum papers to load

    Returns:
        Dictionary mapping paper IDs to paper data
    """
    papers = {}

    try:
        dataset = load_dataset("CShorten/ML-ArXiv-Papers", split=dataset_split)
        logger.info(f"Loaded ML-ArXiv-Papers dataset (split: {dataset_split})")
    except Exception:
        try:
            dataset = load_dataset("gfissore/arxiv-abstracts-2021", split=dataset_split)
            logger.info(f"Loaded arxiv-abstracts-2021 dataset (split: {dataset_split})")
        except Exception as e:
            logger.warning(f"Could not load ArXiv dataset: {e}")
            return papers

    for i, paper in enumerate(
        tqdm(dataset, desc="Loading ArXiv papers", total=max_papers or len(dataset))
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

        # Extract authors
        authors_data = paper.get("authors", [])
        if isinstance(authors_data, str):
            authors_data = [authors_data]

        papers[paper_id] = {
            "title": paper.get("title", "Unknown"),
            "abstract": paper.get("abstract", paper.get("summary", "")),
            "year": year,
            "authors": authors_data,
            "categories": paper.get("categories", []),
        }

    return papers


@memory.cache
def compute_embeddings_cached(texts: tuple, model_name: str) -> np.ndarray:
    """
    Compute and cache embeddings for texts.

    Args:
        texts: Tuple of texts (tuple for caching)
        model_name: Model name for sentence transformer

    Returns:
        Numpy array of embeddings
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(texts), convert_to_tensor=False, show_progress_bar=True
    )
    return embeddings


class EmbeddingGraphBuilder(GraphBuilderStrategy):
    """
    Build similarity graphs using semantic embeddings.

    This strategy:
    - Loads ArXiv dataset corpus
    - Computes embeddings for paper abstracts
    - Finds semantically similar papers
    - Combines semantic with temporal/category/author factors
    """

    def __init__(
        self,
        max_papers: int = 40,
        model_name: str = "google/embeddinggemma-300m",
        dataset_split: str = "train[:2%]",
        corpus_size: Optional[int] = None,
        top_k: int = 2,
    ):
        """
        Initialize embedding graph builder.

        Args:
            max_papers: Maximum papers in final graph
            model_name: Sentence transformer model name
            dataset_split: HuggingFace dataset split
            corpus_size: Maximum papers to load from corpus (None = all in split)
            top_k: Number of most similar neighbors per node
        """
        super().__init__(max_papers)
        self.model_name = model_name
        self.dataset_split = dataset_split
        self.corpus_size = corpus_size
        self.top_k = top_k
        self.model: Optional[SentenceTransformer] = None
        self.arxiv_corpus: Dict[str, Dict] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.client = get_client()

    def _load_model(self):
        """Lazy load sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def _load_corpus(self):
        """Load ArXiv corpus if not already loaded."""
        if not self.arxiv_corpus:
            logger.info(f"Loading ArXiv corpus (split: {self.dataset_split})...")
            self.arxiv_corpus = load_arxiv_dataset_cached(
                self.dataset_split, self.corpus_size
            )
            logger.info(f"Corpus loaded: {len(self.arxiv_corpus)} papers")

    def collect_papers(self, seed_id: str, **kwargs) -> Dict[str, Paper]:
        """
        Collect papers via semantic similarity search.

        Args:
            seed_id: Seed paper identifier (ArXiv ID or text query)

        Returns:
            Dictionary of paper_id -> Paper objects
        """
        papers = {}

        # Load corpus and model
        self._load_corpus()
        self._load_model()

        # Try to get seed from Semantic Scholar first
        seed_paper = self.client.get_paper(seed_id)

        if seed_paper:
            # Found via S2 API
            seed_paper.is_seed = True
            papers[seed_paper.paper_id] = seed_paper
            seed_text = f"{seed_paper.title}. {seed_paper.abstract}"
        else:
            # Treat as text query
            logger.info(f"Using '{seed_id}' as text query")
            seed_text = seed_id
            # Create dummy seed paper
            seed_paper = Paper(paper_id="query", title=seed_id, year=2020, is_seed=True)
            papers["query"] = seed_paper

        # Compute seed embedding
        logger.info("Computing seed embedding...")
        seed_embedding = self.model.encode([seed_text], convert_to_tensor=False)[0]

        # Compute corpus embeddings
        logger.info("Computing corpus embeddings...")
        corpus_ids = list(self.arxiv_corpus.keys())
        corpus_texts = [
            f"{self.arxiv_corpus[pid]['title']}. {self.arxiv_corpus[pid]['abstract']}"
            for pid in corpus_ids
        ]

        # Use cached computation
        corpus_embeddings = compute_embeddings_cached(
            tuple(corpus_texts), self.model_name
        )

        # Find most similar papers
        logger.info(f"Finding top {self.max_papers} most similar papers...")
        similarities = util.cos_sim(seed_embedding, corpus_embeddings)[0]
        top_indices = torch.topk(
            similarities, k=min(self.max_papers, len(corpus_ids))
        ).indices

        # Convert to Paper objects
        for idx in top_indices:
            idx = int(idx)
            paper_id = corpus_ids[idx]
            paper_data = self.arxiv_corpus[paper_id]

            # Create Author objects
            authors = [Author(name=name) for name in paper_data["authors"][:3]]

            paper = Paper(
                paper_id=paper_id,
                title=paper_data["title"],
                year=paper_data["year"],
                authors=authors,
                abstract=paper_data["abstract"],
                categories=paper_data.get("categories", []),
                citation_count=0,  # ArXiv data doesn't have citation counts
                is_seed=False,
            )

            papers[paper_id] = paper
            self.embeddings[paper_id] = corpus_embeddings[idx]

        # Store seed embedding
        if seed_paper.paper_id in papers:
            self.embeddings[seed_paper.paper_id] = seed_embedding

        # Fetch citation counts from S2 for top papers
        logger.info("Fetching citation counts from Semantic Scholar...")
        for paper_id in list(papers.keys())[:20]:  # Only top 20 to avoid timeouts
            if paper_id == "query":
                continue
            s2_paper = self.client.get_paper(paper_id)
            if s2_paper:
                papers[paper_id].citation_count = s2_paper.citation_count

        return papers

    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        """
        Compute multi-factor similarity.

        Combines:
        - Semantic embedding similarity (50%)
        - Temporal proximity (20%)
        - Category overlap (20%)
        - Author collaboration (10%)

        Args:
            paper1: First paper
            paper2: Second paper

        Returns:
            Combined similarity score (0.0 to 1.0)
        """
        # Semantic similarity from embeddings
        if paper1.paper_id in self.embeddings and paper2.paper_id in self.embeddings:
            emb1 = self.embeddings[paper1.paper_id]
            emb2 = self.embeddings[paper2.paper_id]
            semantic_sim = float(util.cos_sim(emb1, emb2)[0][0])
        else:
            semantic_sim = 0.0

        # Temporal factor
        year_diff = abs(paper1.year - paper2.year)
        temporal_factor = EMBEDDING_CONFIG.temporal_factor(year_diff)

        # Category overlap
        category_overlap = paper1.category_overlap(paper2)

        # Author collaboration
        author_factor = (
            EMBEDDING_CONFIG.shared_author_bonus
            if paper1.shares_authors_with(paper2)
            else 1.0
        )

        # Combined similarity
        similarity = (
            EMBEDDING_CONFIG.semantic_weight * semantic_sim
            + EMBEDDING_CONFIG.temporal_weight * temporal_factor
            + EMBEDDING_CONFIG.category_weight * category_overlap
        ) * author_factor

        return min(similarity, 1.0)  # Cap at 1.0

    def should_create_edge(
        self, paper1: Paper, paper2: Paper, similarity: float
    ) -> bool:
        """
        Create edges using top-k strategy.

        Each paper connects to its k most similar neighbors.

        Args:
            paper1: First paper
            paper2: Second paper
            similarity: Computed similarity

        Returns:
            True if edge should be created
        """
        # For embedding strategy, we'll compute top-k after all similarities
        # For now, return True for all non-zero similarities
        # The build_graph method will filter to top-k
        return similarity > 0.1

    def build_graph(self, seed_id: str, **kwargs) -> Tuple:
        """
        Build graph with top-k edge selection.

        Overrides base class to implement top-k neighbor selection
        instead of threshold-based edge creation.

        Args:
            seed_id: Seed paper identifier

        Returns:
            Tuple of (NetworkX graph, seed paper ID)
        """
        # Use base class to collect papers and create nodes
        graph, actual_seed_id = super().build_graph(seed_id, **kwargs)

        # Now filter edges to keep only top-k per node
        import networkx as nx

        # Compute all pairwise similarities (already done by base class)
        # Now for each node, keep only top-k edges
        filtered_graph = nx.Graph()
        filtered_graph.add_nodes_from(graph.nodes(data=True))

        for node in graph.nodes():
            # Get all edges for this node
            edges = [
                (node, neighbor, graph[node][neighbor]["weight"])
                for neighbor in graph.neighbors(node)
            ]

            # Sort by weight and keep top-k
            edges.sort(key=lambda x: x[2], reverse=True)
            top_edges = edges[: EMBEDDING_CONFIG.top_k_neighbors]

            for u, v, weight in top_edges:
                filtered_graph.add_edge(u, v, weight=weight)

        logger.info(
            f"Filtered graph: {filtered_graph.number_of_nodes()} nodes, "
            f"{filtered_graph.number_of_edges()} edges (top-{EMBEDDING_CONFIG.top_k_neighbors})"
        )

        return filtered_graph, actual_seed_id
