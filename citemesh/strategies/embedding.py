"""
Embedding-based graph building strategy.

This strategy uses semantic similarity from sentence transformers
to find conceptually similar papers without relying on citations.
"""

import heapq
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from joblib import Memory
from tqdm.auto import tqdm

from citemesh.core import EMBEDDING_CONFIG, Author, Paper
from citemesh.data import EmbeddingCache, get_cache_dir, get_embedding_model_profile
from citemesh.services import SemanticScholarClient, get_client
from citemesh.strategies.base import GraphBuilderStrategy

logger = logging.getLogger(__name__)

# Set up joblib cache in the user cache directory
_memory: Optional[Memory] = None


def _get_memory() -> Memory:
    """Get or create the joblib cache lazily.

    :return Memory: Shared joblib cache object for expensive dataset operations.
    """
    global _memory
    if _memory is None:
        _memory = Memory(str(get_cache_dir("joblib")), verbose=0)
    return _memory


def _check_embedding_deps() -> None:
    """Verify embedding dependencies are installed."""
    missing: list[str] = []

    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        missing.append("sentence-transformers")

    try:
        import datasets  # noqa: F401
    except ImportError:
        missing.append("datasets")

    if missing:
        raise ImportError(
            f"Embedding strategy requires: {', '.join(missing)}. "
            f"Install with: pip install citemesh[embeddings]"
        )


STREAMING_BATCH_SIZE = 32
CANDIDATE_MULTIPLIER = 4
ARXIV_DATASET_CANDIDATES = (
    "librarian-bots/arxiv-metadata-snapshot",
    "CShorten/ML-ArXiv-Papers",
    "gfissore/arxiv-abstracts-2021",
)


def _parse_year(paper: Dict[str, Any]) -> Optional[int]:
    """Extract publication year from dataset metadata.

    :param Dict[str, Any] paper: Raw dataset record.
    :return Optional[int]: Parsed year or ``None`` if missing/invalid.
    """
    if paper.get("year"):
        try:
            return int(paper["year"])
        except (TypeError, ValueError):
            pass

    update_date = paper.get("update_date")
    if update_date:
        try:
            return int(str(update_date)[:4])
        except (TypeError, ValueError):
            pass

    return None


def _parse_authors(authors_data: Any) -> List[str]:
    """Normalize author metadata to a list of names.

    :param Any authors_data: Raw ``authors`` field from dataset.
    :return List[str]: Author names.
    """
    if isinstance(authors_data, str):
        return [name.strip() for name in authors_data.split(",") if name.strip()]

    if isinstance(authors_data, list):
        author_names: List[str] = []
        for author in authors_data:
            if isinstance(author, str):
                normalized = author.strip()
                if normalized:
                    author_names.append(normalized)
                continue

            if isinstance(author, dict):
                name = author.get("name")
                if isinstance(name, str):
                    normalized = name.strip()
                    if normalized:
                        author_names.append(normalized)
        return author_names

    return []


def _parse_categories(categories_data: Any) -> List[str]:
    """Normalize category metadata to a list of arXiv category codes.

    :param Any categories_data: Raw ``categories`` field from dataset.
    :return List[str]: Category code list.
    """
    if isinstance(categories_data, str):
        normalized = categories_data.replace(",", " ")
        return [category.strip() for category in normalized.split() if category.strip()]

    if isinstance(categories_data, list):
        categories: List[str] = []
        for raw_value in categories_data:
            if isinstance(raw_value, str):
                normalized = raw_value.replace(",", " ")
                categories.extend(
                    [
                        category.strip()
                        for category in normalized.split()
                        if category.strip()
                    ]
                )
        return categories

    return []


def _extract_dataset_paper_metadata(paper: Dict[str, Any], fallback_index: int) -> Dict:
    """Normalize a raw dataset record to embedding metadata fields.

    :param Dict[str, Any] paper: Raw dataset record.
    :param int fallback_index: Index used for synthetic IDs when missing.
    :return Dict: Normalized metadata used by embedding selection.
    """
    paper_id = (
        paper.get("id")
        or paper.get("paper_id")
        or paper.get("paperId")
        or f"arxiv_{fallback_index}"
    )
    title = paper.get("title", "Unknown")
    if not isinstance(title, str) or not title.strip():
        title = "Unknown"
    abstract = paper.get("abstract", paper.get("summary", ""))
    if not isinstance(abstract, str):
        abstract = ""

    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "year": _parse_year(paper),
        "authors": _parse_authors(paper.get("authors", [])),
        "categories": _parse_categories(paper.get("categories", [])),
    }


def load_arxiv_dataset_cached(
    dataset_split: str, max_papers: Optional[int]
) -> Dict[str, Dict]:
    """
    Load and cache ArXiv dataset.

    :param str dataset_split: Dataset split (e.g., "train[:2%]")
    :param Optional[int] max_papers: Maximum papers to load
    :return Dict[str, Dict]: Dictionary mapping paper IDs to paper data
    """
    papers = {}

    from datasets import load_dataset

    dataset = None
    last_error: Optional[Exception] = None
    for dataset_name in ARXIV_DATASET_CANDIDATES:
        try:
            dataset = load_dataset(dataset_name, split=dataset_split)
            logger.info(f"Loaded {dataset_name} dataset (split: {dataset_split})")
            break
        except Exception as exc:  # pragma: no cover - network/source dependent
            last_error = exc
            logger.warning(
                "Could not load dataset %s: %s. Trying fallback.",
                dataset_name,
                exc,
            )

    if dataset is None:
        logger.warning(
            "Could not load any ArXiv dataset for split %s. Returning empty corpus.",
            dataset_split,
        )
        if last_error is not None:
            logger.debug("Last dataset error: %s", last_error)
        return papers

    for i, paper in enumerate(
        tqdm(dataset, desc="Loading ArXiv papers", total=max_papers or len(dataset))
    ):
        if max_papers and i >= max_papers:
            break

        metadata = _extract_dataset_paper_metadata(paper, i)
        paper_id = metadata.pop("paper_id")
        papers[paper_id] = metadata

    return papers


def get_arxiv_dataset_cached(
    dataset_split: str, max_papers: Optional[int]
) -> Dict[str, Dict]:
    """Apply joblib caching to dataset loading.

    :param str dataset_split: HuggingFace split expression (e.g. ``train[:2%]``).
    :param Optional[int] max_papers: Optional paper cap for corpus sampling.
    :return Dict[str, Dict]: Cached dataset mapping paper ID to metadata.
    """
    cache = _get_memory()
    return cache.cache(load_arxiv_dataset_cached)(dataset_split, max_papers)


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
        dataset_split: str = "train",  # Full snapshot split; use corpus_size to bound runtime.
        corpus_size: Optional[int] = None,
        top_k: int = 2,
        random_seed: Optional[int] = None,
        use_streaming: bool = False,
        client: Optional[SemanticScholarClient] = None,
    ):
        """
        Initialize embedding graph builder.

        :param int max_papers: Maximum papers in final graph
        :param str model_name: Sentence transformer model name
        :param str dataset_split: HuggingFace dataset split
        :param Optional[int] corpus_size: Maximum papers to load from corpus (None = all in split)
        :param int top_k: Number of most similar neighbors per node
        :param Optional[int] random_seed: Random seed for reproducibility
        :param bool use_streaming: Whether to stream the HuggingFace dataset instead of loading it
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        """
        _check_embedding_deps()
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        super().__init__(max_papers, random_seed)
        self.model_name = model_name
        self.dataset_split = dataset_split
        self.corpus_size = corpus_size
        self.top_k = top_k
        self.model = None
        self.arxiv_corpus: Dict[str, Dict] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.client = client or get_client()
        self.embedding_cache = EmbeddingCache(model_name=model_name)
        self.use_streaming = use_streaming
        self.model_profile = get_embedding_model_profile(model_name)
        self._profile_logged = False

    def _load_model(self) -> None:
        """Lazy load sentence transformer model.

        :return None: Model is initialized in-place on first access.
        """
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            if not self.model_profile.float16_supported:
                logger.info(
                    f"{self.model_name} does not support float16 activations; defaulting to float32."
                )
            if self.model_profile.notes and not self._profile_logged:
                logger.info(self.model_profile.notes)
                self._profile_logged = True

    def _load_corpus(self) -> None:
        """Load ArXiv corpus if not already loaded.

        :return None: Corpus is populated in-place on first access.
        """
        if not self.arxiv_corpus:
            logger.info(f"Loading ArXiv corpus (split: {self.dataset_split})...")
            self.arxiv_corpus = get_arxiv_dataset_cached(
                self.dataset_split, self.corpus_size
            )
            logger.info(f"Corpus loaded: {len(self.arxiv_corpus)} papers")

    def collect_papers(self, seed_id: str, **kwargs: Any) -> Dict[str, Paper]:
        """
        Collect papers via semantic similarity search.

        :param str seed_id: Seed paper identifier (ArXiv ID or text query)
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Dict[str, Paper]: Dictionary of paper_id -> Paper objects
        """
        papers: Dict[str, Paper] = {}
        self.embeddings = {}

        # Load model (corpus loaded lazily depending on mode)
        self._load_model()

        # Try to get seed from Semantic Scholar first
        seed_paper = self.client.get_paper(seed_id)

        seed_metadata: Dict[str, str] = {}

        if seed_paper:
            # Found via S2 API
            seed_paper.is_seed = True
            papers[seed_paper.paper_id] = seed_paper
            seed_title = (seed_paper.title or "").strip()
            seed_abstract = (seed_paper.abstract or "").strip()
            pieces = [part for part in (seed_title, seed_abstract) if part]
            seed_text = ". ".join(pieces) if pieces else seed_id
            seed_metadata = {
                "title": seed_title,
                "abstract": seed_abstract,
            }
        else:
            # Treat as text query
            logger.info(f"Using '{seed_id}' as text query")
            seed_text = seed_id
            # Create dummy seed paper
            seed_paper = Paper(paper_id="query", title=seed_id, year=None, is_seed=True)
            papers["query"] = seed_paper
            seed_metadata = {"title": seed_id, "abstract": ""}

        # Compute normalized seed embedding
        logger.info("Computing seed embedding...")
        formatted_seed_text = self.model_profile.format_query(seed_text, seed_metadata)
        seed_embedding = self.model.encode(
            [formatted_seed_text],
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        self.embeddings[seed_paper.paper_id] = seed_embedding

        # Decide whether to use streaming based on split and corpus_size
        use_streaming = (
            self.use_streaming
            and ":" not in self.dataset_split
            and self.corpus_size is None
        )

        if use_streaming:
            logger.info(
                "Streaming ArXiv corpus for semantic matches (disables joblib cache)..."
            )
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
                year=metadata.get("year"),
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

        :param np.ndarray seed_embedding: Normalized seed embedding vector
        :return List[Tuple[str, Dict, np.ndarray]]: List of (paper_id, metadata, embedding) tuples sorted by similarity
        """
        if not self.arxiv_corpus:
            return []

        print("Computing corpus embeddings (cache-enabled)...")

        corpus_items = list(self.arxiv_corpus.items())
        metadata_map = {paper_id: metadata for paper_id, metadata in corpus_items}

        embeddings_dict = self.embedding_cache.get_embeddings(
            metadata_map,
            self.model,
            batch_size=STREAMING_BATCH_SIZE,
            show_progress=sys.stderr.isatty(),
            text_builder=self.model_profile.format_document,
        )

        if not embeddings_dict:
            return []

        valid_items: List[Tuple[str, Dict]] = []
        ordered_embeddings: List[np.ndarray] = []

        for paper_id, metadata in corpus_items:
            embedding = embeddings_dict.get(paper_id)
            if embedding is None:
                continue
            valid_items.append((paper_id, metadata))
            ordered_embeddings.append(np.asarray(embedding, dtype=np.float32))

        if not ordered_embeddings:
            return []

        embeddings_array = np.vstack(ordered_embeddings)
        similarities = embeddings_array @ seed_embedding
        top_k = min(self.max_papers * CANDIDATE_MULTIPLIER, len(valid_items))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        candidates: List[Tuple[str, Dict, np.ndarray]] = []
        for idx in top_indices:
            paper_id, metadata = valid_items[int(idx)]
            candidates.append((paper_id, metadata, embeddings_array[int(idx)]))

        return candidates

    def _select_candidates_streaming(
        self, seed_embedding: np.ndarray
    ) -> List[Tuple[str, Dict, np.ndarray]]:
        """
        Stream dataset and keep top candidates in a bounded heap.

        :param np.ndarray seed_embedding: Normalized seed embedding vector
        :return List[Tuple[str, Dict, np.ndarray]]: List of (paper_id, metadata, embedding) tuples sorted by similarity
        """
        max_candidates = max(self.max_papers * CANDIDATE_MULTIPLIER, self.max_papers)
        heap: List[Tuple[float, str, Dict, np.ndarray]] = []

        progress_enabled = sys.stderr.isatty()

        from datasets import load_dataset

        last_exception: Optional[Exception] = None

        for dataset_name in ARXIV_DATASET_CANDIDATES:
            try:
                dataset = load_dataset(
                    dataset_name, split=self.dataset_split, streaming=True
                )
            except Exception as exc:
                logger.warning(
                    "Could not stream dataset %s: %s. Trying fallback.",
                    dataset_name,
                    exc,
                )
                last_exception = exc
                continue

            progress_total = self.corpus_size if self.corpus_size else None
            with tqdm(
                total=progress_total,
                desc=f"Streaming {dataset_name}",
                unit="papers",
                dynamic_ncols=True,
                disable=not progress_enabled,
            ) as progress:
                batch: List[Dict] = []
                for idx, raw_record in enumerate(dataset):
                    if self.corpus_size and idx >= self.corpus_size:
                        break

                    metadata = self._extract_paper_metadata(raw_record, idx)
                    batch.append(metadata)
                    progress.update(1)

                    if len(batch) >= STREAMING_BATCH_SIZE:
                        self._process_stream_batch(
                            batch, seed_embedding, heap, max_candidates
                        )
                        batch = []

                if batch:
                    self._process_stream_batch(
                        batch, seed_embedding, heap, max_candidates
                    )

                if progress_total is None:
                    progress.set_postfix_str(f"processed {progress.n}")

            if heap:
                break

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
        batch: List[Dict],
        seed_embedding: np.ndarray,
        heap: List[Tuple[float, str, Dict, np.ndarray]],
        max_candidates: int,
    ) -> None:
        """
        Encode a batch of records and push to candidate heap.

        :param List[Dict] batch: List of metadata dictionaries
        :param np.ndarray seed_embedding: Normalized seed embedding vector
        :param List[Tuple[float, str, Dict, np.ndarray]] heap: Min-heap storing top candidates
        :param int max_candidates: Maximum heap size
        """
        batch_map = {metadata["paper_id"]: metadata for metadata in batch}
        embeddings = self.embedding_cache.get_embeddings(
            batch_map,
            self.model,
            batch_size=len(batch_map) or STREAMING_BATCH_SIZE,
            show_progress=False,
            text_builder=self.model_profile.format_document,
        )

        for metadata in batch:
            raw_embedding = embeddings.get(metadata["paper_id"])
            if raw_embedding is None:
                continue
            embedding = np.asarray(raw_embedding, dtype=np.float32)

            similarity = float(np.dot(seed_embedding, embedding))
            candidate = (similarity, metadata["paper_id"], metadata, embedding)

            if len(heap) < max_candidates:
                heapq.heappush(heap, candidate)
            elif similarity > heap[0][0]:
                heapq.heapreplace(heap, candidate)

    def _extract_paper_metadata(self, paper: Dict, fallback_index: int) -> Dict:
        """
        Normalize dataset record into metadata dictionary.

        :param Dict paper: Raw dataset record
        :param int fallback_index: Index used to generate ID if missing
        :return Dict: Dictionary with normalized fields
        """
        return _extract_dataset_paper_metadata(paper, fallback_index)

    def _update_citation_counts(self, papers: Dict[str, Paper]) -> None:
        """
        Optionally enrich top papers with citation counts from Semantic Scholar.

        :param Dict[str, Paper] papers: Dictionary of collected papers (including seed)
        """
        logger.info("Fetching citation counts from Semantic Scholar (optional)...")

        targets = [
            (pid, paper)
            for pid, paper in list(papers.items())[:10]
            if not paper.is_seed
            and pid != "query"
            and not (isinstance(pid, str) and pid.startswith("arxiv_"))
        ]

        if not targets:
            return

        progress_enabled = sys.stderr.isatty()
        iterator = (
            tqdm(
                targets,
                desc="Citation metadata",
                unit="papers",
                leave=False,
                dynamic_ncols=True,
            )
            if progress_enabled
            else targets
        )

        for paper_id, paper in iterator:
            try:
                s2_paper = self.client.get_paper(paper_id)
                if s2_paper:
                    paper.citation_count = s2_paper.citation_count
            except Exception as exc:
                logger.warning(f"Could not fetch citation count for {paper_id}: {exc}")

        if progress_enabled:
            iterator.close()

    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        """
        Compute multi-factor similarity.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :return float: Combined similarity score (0.0 to 1.0)
        """
        # Semantic similarity from embeddings
        if paper1.paper_id in self.embeddings and paper2.paper_id in self.embeddings:
            emb1 = self.embeddings[paper1.paper_id]
            emb2 = self.embeddings[paper2.paper_id]
            semantic_sim = float(np.clip(np.dot(emb1, emb2), -1.0, 1.0))
        else:
            semantic_sim = 0.0

        # Temporal factor
        temporal_factor = self.temporal_similarity(paper1, paper2)

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

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :param float similarity: Computed similarity
        :return bool: True if edge should be created
        """
        # For embedding strategy, we'll compute top-k after all similarities
        # For now, return True for all non-zero similarities
        # The build_graph method will filter to top-k
        return similarity > 0.1

    def build_graph(self, seed_id: str, **kwargs: Any) -> Tuple[nx.Graph, str]:
        """
        Build graph with top-k edge selection.

        :param str seed_id: Seed paper identifier
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Tuple[nx.Graph, str]: Tuple of (NetworkX graph, seed paper ID)
        """
        # Use base class to collect papers and create nodes
        graph, actual_seed_id = super().build_graph(seed_id, **kwargs)
        # Enforce a strict per-node top-k cap by greedily keeping strongest edges.
        filtered_graph = nx.Graph()
        filtered_graph.add_nodes_from(graph.nodes(data=True))

        edge_counts = {node: 0 for node in graph.nodes()}
        sorted_edges = sorted(
            graph.edges(data=True),
            key=lambda item: item[2].get("weight", 0.0),
            reverse=True,
        )

        for u, v, data in sorted_edges:
            if edge_counts[u] >= self.top_k or edge_counts[v] >= self.top_k:
                continue
            filtered_graph.add_edge(u, v, weight=data.get("weight", 0.0))
            edge_counts[u] += 1
            edge_counts[v] += 1

        logger.info(
            f"Filtered graph: {filtered_graph.number_of_nodes()} nodes, "
            f"{filtered_graph.number_of_edges()} edges (top-{self.top_k})"
        )

        return filtered_graph, actual_seed_id
