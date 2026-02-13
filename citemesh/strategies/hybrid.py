"""
Hybrid graph building strategy.

Combines citation relationships with semantic similarity for
comprehensive paper discovery.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np

from citemesh.core import HYBRID_CONFIG, Paper
from citemesh.services import SemanticScholarClient, get_client
from citemesh.strategies.base import GraphBuilderStrategy
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import EmbeddingGraphBuilder, _check_embedding_deps

logger = logging.getLogger(__name__)


class HybridGraphBuilder(GraphBuilderStrategy):
    """
    Hybrid strategy combining citations and embeddings.

    This strategy:
    1. Collects papers via citations (ground truth relationships)
    2. Enriches with semantically similar papers from corpus
    3. Uses adaptive similarity computation based on relationship type
    """

    def __init__(
        self,
        max_papers: int = 40,
        max_citations: int = 15,
        max_references: int = 15,
        max_semantic: int = 10,
        model_name: str = "google/embeddinggemma-300m",
        dataset_split: str = "train",  # Full training set by default (~117k papers)
        random_seed: Optional[int] = None,
        client: Optional[SemanticScholarClient] = None,
    ):
        """
        Initialize hybrid graph builder.

        :param int max_papers: Maximum total papers
        :param int max_citations: Maximum citing papers from S2
        :param int max_references: Maximum referenced papers from S2
        :param int max_semantic: Maximum papers from semantic search
        :param str model_name: Embedding model name
        :param str dataset_split: ArXiv dataset split
        :param Optional[int] random_seed: Random seed for reproducibility
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        """
        super().__init__(max_papers, random_seed)
        self.client = client or get_client()
        self.max_semantic = max_semantic

        if self.max_semantic > 0:
            _check_embedding_deps()
            self.embedding_builder = EmbeddingGraphBuilder(
                max_papers=max_semantic,
                model_name=model_name,
                dataset_split=dataset_split,
                random_seed=random_seed,
                client=self.client,
            )
        else:
            self.embedding_builder = None

        # Create citation and embedding builders (with same seed for consistency)
        citation_papers = max_papers - max_semantic
        self.citation_builder = CitationGraphBuilder(
            max_papers=citation_papers,
            max_citations=max_citations,
            max_references=max_references,
            fetch_references=True,  # Enable real bibliographic coupling
            random_seed=random_seed,
            client=self.client,
        )

        # Track paper sources for adaptive similarity
        self.paper_sources: Dict[str, str] = {}  # paper_id -> "citation" or "semantic"

    def collect_papers(self, seed_id: str, **kwargs: Any) -> Dict[str, Paper]:
        """
        Collect papers from both citation and semantic sources.

        :param str seed_id: Seed paper identifier
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Dict[str, Paper]: Combined dictionary of papers
        """
        papers = {}
        self.paper_sources = {}

        # Step 1: Collect from citations
        logger.info("Collecting papers via citations...")
        citation_papers = self.citation_builder.collect_papers(seed_id)

        for paper_id, paper in citation_papers.items():
            papers[paper_id] = paper
            self.paper_sources[paper_id] = "citation"

        # Step 2: Enrich with semantic matches
        if self.max_semantic > 0:
            logger.info(f"Enriching with up to {self.max_semantic} semantic matches...")

            try:
                semantic_papers = self.embedding_builder.collect_papers(seed_id)

                # Add new papers not already in collection
                added = 0
                for paper_id, paper in semantic_papers.items():
                    if paper_id not in papers and added < self.max_semantic:
                        papers[paper_id] = paper
                        self.paper_sources[paper_id] = "semantic"
                        added += 1

                logger.info(f"Added {added} semantic papers")

            except Exception as e:
                logger.warning(f"Semantic enrichment failed: {e}")

        return papers

    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        """
        Compute similarity using adaptive weights.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :return float: Similarity score (0.0 to 1.0)
        """
        source1 = self.paper_sources.get(paper1.paper_id, "citation")
        source2 = self.paper_sources.get(paper2.paper_id, "citation")

        # Get component similarities
        temporal_sim = self.temporal_similarity(paper1, paper2)
        citation_sim = self.citation_similarity(paper1, paper2)
        biblio_coupling = self.bibliographic_coupling(paper1, paper2)

        # Compute embedding similarity if available
        embed_sim = 0.0
        if (
            self.embedding_builder is not None
            and paper1.paper_id in self.embedding_builder.embeddings
            and paper2.paper_id in self.embedding_builder.embeddings
        ):
            emb1 = self.embedding_builder.embeddings[paper1.paper_id]
            emb2 = self.embedding_builder.embeddings[paper2.paper_id]
            embed_sim = float(np.clip(np.dot(emb1, emb2), -1.0, 1.0))

        # Adaptive weighting
        if source1 == "semantic" and source2 == "semantic":
            # Both semantic: emphasize embeddings
            weights = HYBRID_CONFIG.semantic_semantic_weights
            similarity = (
                weights[0] * embed_sim
                + weights[1] * temporal_sim
                + weights[2] * citation_sim
                + weights[3] * biblio_coupling
            )
        elif source1 == "citation" and source2 == "citation":
            # Both citation: emphasize bibliographic coupling
            weights = HYBRID_CONFIG.citation_citation_weights
            similarity = (
                weights[0] * embed_sim
                + weights[1] * temporal_sim
                + weights[2] * citation_sim
                + weights[3] * biblio_coupling
            )
        else:
            # Mixed: balanced approach
            weights = HYBRID_CONFIG.mixed_weights
            similarity = (
                weights[0] * embed_sim
                + weights[1] * temporal_sim
                + weights[2] * citation_sim
                + weights[3] * biblio_coupling
            )

        # Add co-citation boost if papers are from same era
        if (
            paper1.year is not None
            and paper2.year is not None
            and abs(paper1.year - paper2.year) < 2
        ):
            similarity += HYBRID_CONFIG.co_citation_boost

        return min(similarity, 1.0)  # Cap at 1.0

    def build_graph(self, seed_id: str, **kwargs: Any) -> Tuple[nx.Graph, str]:
        """
        Build graph and enforce per-node edge limits for readability.

        :param str seed_id: Seed paper identifier
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Tuple[nx.Graph, str]: Tuple of (NetworkX graph, seed_id).
        """
        graph, actual_seed_id = super().build_graph(seed_id, **kwargs)

        max_edges = HYBRID_CONFIG.max_edges_per_node
        if not max_edges or max_edges <= 0:
            return graph, actual_seed_id

        edge_counts = {node: 0 for node in graph.nodes()}
        # Sort edges by weight descending so strongest connections are kept
        sorted_edges = sorted(
            graph.edges(data=True),
            key=lambda item: item[2].get("weight", 0.0),
            reverse=True,
        )

        edges_to_remove = []
        for u, v, data in sorted_edges:
            # Enforce a strict per-node edge cap; once either endpoint is full, we drop
            # this edge to keep degree bounds predictable for downstream rendering.
            if edge_counts[u] >= max_edges or edge_counts[v] >= max_edges:
                edges_to_remove.append((u, v))
                continue

            edge_counts[u] += 1
            edge_counts[v] += 1

        graph.remove_edges_from(edges_to_remove)
        return graph, actual_seed_id

    def should_create_edge(
        self, paper1: Paper, paper2: Paper, similarity: float
    ) -> bool:
        """
        Create edges with per-node limits.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :param float similarity: Computed similarity
        :return bool: True if edge should be created
        """
        # Basic threshold
        if similarity < 0.2:
            return False

        # Seed paper: always connect if above threshold
        if paper1.is_seed or paper2.is_seed:
            return similarity > 0.4

        # Higher threshold for non-seed
        return similarity > 0.5
