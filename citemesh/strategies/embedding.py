"""
Embedding-based graph building strategy.

This strategy uses semantic similarity from sentence transformers
to find conceptually similar papers without relying on citations.
"""

import heapq
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from joblib import Memory
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from citemesh.api_client import get_client
from citemesh.config import EMBEDDING_CONFIG
from citemesh.models import Author, Paper
from citemesh.strategies.base import GraphBuilderStrategy

logger = logging.getLogger(__name__)

# Set up joblib cache
memory = Memory("cache/joblib_cache", verbose=0)

STREAMING_BATCH_SIZE = 32
CANDIDATE_MULTIPLIER = 4


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
        list(texts),
        convert_to_tensor=False,
        normalize_embeddings=True,
        show_progress_bar=True,
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
        dataset_split: str = "train",  # Full training set by default (~117k papers)
        corpus_size: Optional[int] = None,
        top_k: int = 2,
        random_seed: int = None,
    ):
        """
        Initialize embedding graph builder.

        Args:
            max_papers: Maximum papers in final graph
            model_name: Sentence transformer model name
            dataset_split: HuggingFace dataset split
            corpus_size: Maximum papers to load from corpus (None = all in split)
            top_k: Number of most similar neighbors per node
            random_seed: Random seed for reproducibility
        """
        super().__init__(max_papers, random_seed)
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
        papers: Dict[str, Paper] = {}
        self.embeddings = {}

        # Load model (corpus loaded lazily depending on mode)
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

        # Compute normalized seed embedding
        logger.info("Computing seed embedding...")
        seed_embedding = self.model.encode(
            [seed_text],
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        self.embeddings[seed_paper.paper_id] = seed_embedding

        # Decide whether to use streaming based on split and corpus_size
        use_streaming = ":" not in self.dataset_split and self.corpus_size is None

        if use_streaming:
            logger.info("Streaming ArXiv corpus for semantic matches...")
            candidates = self._select_candidates_streaming(seed_embedding)
        else:
            self._load_corpus()
            candidates = self._select_candidates_from_loaded(seed_embedding)

        # Convert candidates to Paper objects
        added = 0
        for paper_id, metadata, embedding in candidates:
            if paper_id in papers:
                continue

            authors = [Author(name=name) for name in metadata.get("authors", [])[:3]]

            paper = Paper(
                paper_id=paper_id,
                title=metadata.get("title", "Unknown"),
                year=metadata.get("year", 2020),
                authors=authors,
                abstract=metadata.get("abstract", ""),
                categories=metadata.get("categories", []),
                citation_count=0,  # ArXiv data lacks citation counts
                is_seed=False,
            )

            papers[paper_id] = paper
            self.embeddings[paper_id] = embedding

            added += 1
            if added >= self.max_papers:
                break

        self._update_citation_counts(papers)
        return papers

    def _select_candidates_from_loaded(
        self, seed_embedding: np.ndarray
    ) -> List[Tuple[str, Dict, np.ndarray]]:
        """
        Select top candidates from an in-memory corpus.

        Args:
            seed_embedding: Normalized seed embedding vector

        Returns:
            List of (paper_id, metadata, embedding) tuples sorted by similarity
        """
        if not self.arxiv_corpus:
            return []

        print("Computing corpus embeddings (cached)...")

        corpus_items = list(self.arxiv_corpus.items())
        corpus_texts = [
            f"{metadata['title']}. {metadata['abstract']}"
            for _, metadata in corpus_items
        ]

        embeddings_array = np.asarray(
            compute_embeddings_cached(tuple(corpus_texts), self.model_name)
        )

        similarities = embeddings_array @ seed_embedding
        top_k = min(self.max_papers * CANDIDATE_MULTIPLIER, len(corpus_items))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        candidates: List[Tuple[str, Dict, np.ndarray]] = []
        for idx in top_indices:
            paper_id, metadata = corpus_items[int(idx)]
            candidates.append((paper_id, metadata, embeddings_array[int(idx)]))

        return candidates

    def _select_candidates_streaming(
        self, seed_embedding: np.ndarray
    ) -> List[Tuple[str, Dict, np.ndarray]]:
        """
        Stream dataset and keep top candidates in a bounded heap.

        Args:
            seed_embedding: Normalized seed embedding vector

        Returns:
            List of (paper_id, metadata, embedding) tuples sorted by similarity
        """
        dataset_names = [
            "CShorten/ML-ArXiv-Papers",
            "gfissore/arxiv-abstracts-2021",
        ]

        max_candidates = max(self.max_papers * CANDIDATE_MULTIPLIER, self.max_papers)
        heap: List[Tuple[float, str, Dict, np.ndarray]] = []
        last_exception: Optional[Exception] = None

        for dataset_name in dataset_names:
            try:
                dataset = load_dataset(
                    dataset_name, split=self.dataset_split, streaming=True
                )
            except Exception as exc:
                logger.warning(f"Could not stream dataset {dataset_name}: {exc}")
                last_exception = exc
                continue

            batch: List[Tuple[Dict, str]] = []
            for idx, raw_record in enumerate(dataset):
                if self.corpus_size and idx >= self.corpus_size:
                    break

                metadata = self._extract_paper_metadata(raw_record, idx)
                text = f"{metadata['title']}. {metadata['abstract']}"
                batch.append((metadata, text))

                if len(batch) >= STREAMING_BATCH_SIZE:
                    self._process_stream_batch(
                        batch, seed_embedding, heap, max_candidates
                    )
                    batch = []

            if batch:
                self._process_stream_batch(batch, seed_embedding, heap, max_candidates)

            if heap:
                break  # Successfully collected candidates

        if not heap:
            if last_exception:
                raise last_exception
            return []

        top_candidates = sorted(heap, key=lambda item: item[0], reverse=True)
        limited = top_candidates[: self.max_papers * CANDIDATE_MULTIPLIER]

        return [
            (paper_id, metadata, embedding)
            for _, paper_id, metadata, embedding in limited
        ]

    def _process_stream_batch(
        self,
        batch: List[Tuple[Dict, str]],
        seed_embedding: np.ndarray,
        heap: List[Tuple[float, str, Dict, np.ndarray]],
        max_candidates: int,
    ) -> None:
        """
        Encode a batch of records and push to candidate heap.

        Args:
            batch: List of (metadata, text) tuples
            seed_embedding: Normalized seed embedding vector
            heap: Min-heap storing top candidates
            max_candidates: Maximum heap size
        """
        texts = [text for _, text in batch]
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for (metadata, _), embedding in zip(batch, embeddings):
            similarity = float(np.dot(seed_embedding, embedding))
            candidate = (similarity, metadata["paper_id"], metadata, embedding)

            if len(heap) < max_candidates:
                heapq.heappush(heap, candidate)
            elif similarity > heap[0][0]:
                heapq.heapreplace(heap, candidate)

    def _extract_paper_metadata(self, paper: Dict, fallback_index: int) -> Dict:
        """
        Normalize dataset record into metadata dictionary.

        Args:
            paper: Raw dataset record
            fallback_index: Index used to generate ID if missing

        Returns:
            Dictionary with normalized fields
        """
        paper_id = (
            paper.get("id")
            or paper.get("paper_id")
            or paper.get("paperId")
            or f"arxiv_{fallback_index}"
        )

        year = 2020
        if paper.get("year"):
            try:
                year = int(paper["year"])
            except (TypeError, ValueError):
                pass
        elif paper.get("update_date"):
            try:
                year = int(str(paper["update_date"])[:4])
            except (TypeError, ValueError):
                pass

        authors_data = paper.get("authors", [])
        if isinstance(authors_data, str):
            authors_data = [authors_data]

        categories = paper.get("categories", [])
        if isinstance(categories, str):
            categories = [categories]

        return {
            "paper_id": paper_id,
            "title": paper.get("title", "Unknown"),
            "abstract": paper.get("abstract", paper.get("summary", "")),
            "year": year,
            "authors": authors_data or [],
            "categories": categories or [],
        }

    def _update_citation_counts(self, papers: Dict[str, Paper]) -> None:
        """
        Optionally enrich top papers with citation counts from Semantic Scholar.

        Args:
            papers: Dictionary of collected papers (including seed)
        """
        logger.info("Fetching citation counts from Semantic Scholar (optional)...")

        for paper_id, paper in list(papers.items())[:10]:
            if paper.is_seed or paper_id == "query":
                continue
            if isinstance(paper_id, str) and paper_id.startswith("arxiv_"):
                continue

            try:
                s2_paper = self.client.get_paper(paper_id)
                if s2_paper:
                    paper.citation_count = s2_paper.citation_count
            except Exception as exc:
                logger.warning(f"Could not fetch citation count for {paper_id}: {exc}")

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
            semantic_sim = float(np.clip(np.dot(emb1, emb2), -1.0, 1.0))
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
