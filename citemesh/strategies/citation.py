"""
Citation-based graph building strategy.

This strategy builds similarity graphs using citation relationships,
bibliographic coupling (shared references), and co-citation analysis.
"""

import logging
import sys
from typing import Any, Dict, Optional

from tqdm.auto import tqdm

from citemesh.core import Paper
from citemesh.services import SemanticScholarClient, get_client
from citemesh.similarity import AbstractSimilarityIndex
from citemesh.strategies.base import GraphBuilderStrategy
from citemesh.strategies.similarity import compute_similarity_features

logger = logging.getLogger(__name__)


class CitationGraphBuilder(GraphBuilderStrategy):
    """
    Build similarity graphs using citation relationships.

    This strategy implements:
    - Real bibliographic coupling (shared references)
    - Temporal proximity scoring
    - Citation impact similarity
    - Sparse edge creation for readability
    """

    def __init__(
        self,
        max_papers: int = 40,
        max_citations: int = 20,
        max_references: int = 20,
        similarity_threshold: float = 0.2,
        fetch_references: bool = True,
        random_seed: Optional[int] = None,
        client: Optional[SemanticScholarClient] = None,
    ):
        """
        Initialize citation graph builder.

        :param int max_papers: Maximum total papers in graph
        :param int max_citations: Maximum citing papers to fetch
        :param int max_references: Maximum referenced papers to fetch
        :param float similarity_threshold: Minimum similarity for edges
        :param bool fetch_references: Whether to fetch reference lists (enables real bibliographic coupling)
        :param Optional[int] random_seed: Random seed for reproducibility
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        """
        super().__init__(max_papers, random_seed)
        self.max_citations = max_citations
        self.max_references = max_references
        self.similarity_threshold = similarity_threshold
        self.fetch_references = fetch_references
        self.client: SemanticScholarClient = client or get_client()
        self.reference_cache: Dict[str, list] = {}  # Cache reference lists
        self._abstract_index = AbstractSimilarityIndex()

    def _get_references(self, paper_id: str) -> list:
        """
        Get reference IDs for a paper with caching.

        :param str paper_id: Paper identifier
        :return list: List of referenced paper IDs
        """
        if paper_id in self.reference_cache:
            return self.reference_cache[paper_id]

        ref_ids = self.client.get_reference_ids(paper_id)
        self.reference_cache[paper_id] = ref_ids
        return ref_ids

    def collect_papers(self, seed_id: str, **kwargs: Any) -> Dict[str, Paper]:
        """
        Collect papers via citations and references.

        :param str seed_id: Seed paper identifier
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Dict[str, Paper]: Dictionary of paper_id -> Paper objects
        """
        papers = {}

        # Step 1: Fetch seed paper
        logger.info(f"Fetching seed paper: {seed_id}")
        seed = self.client.get_paper(seed_id, fetch_references=self.fetch_references)

        if not seed:
            raise ValueError(f"Seed paper not found: {seed_id}")

        seed.is_seed = True
        papers[seed.paper_id] = seed

        # Store seed references in cache
        if self.fetch_references and seed.references:
            self.reference_cache[seed.paper_id] = seed.references

        logger.info(f"Seed: {seed.title[:50]}...")

        # Step 2: Fetch references (older papers)
        logger.info(f"Fetching up to {self.max_references} references...")
        references = self.client.get_paper_references(
            seed.paper_id, limit=self.max_references
        )
        progress_enabled = sys.stderr.isatty()

        if references:
            ref_iterator = (
                tqdm(
                    references,
                    desc="Downloading references",
                    unit="papers",
                    leave=False,
                    dynamic_ncols=True,
                )
                if progress_enabled
                else references
            )
        else:
            ref_iterator = []

        for paper in ref_iterator:
            if len(papers) >= self.max_papers:
                break
            papers[paper.paper_id] = paper

            # Fetch reference lists for bibliographic coupling
            if self.fetch_references and paper.paper_id not in self.reference_cache:
                refs = self._get_references(paper.paper_id)
                paper.references = refs

        if references and progress_enabled:
            ref_iterator.close()

        # Step 3: Fetch citations (newer papers)
        remaining = self.max_papers - len(papers)
        if remaining > 0:
            logger.info(
                f"Fetching up to {min(remaining, self.max_citations)} citations..."
            )
            citations = self.client.get_paper_citations(
                seed.paper_id, limit=min(remaining, self.max_citations)
            )

            if citations:
                cit_iterator = (
                    tqdm(
                        citations,
                        desc="Downloading citations",
                        unit="papers",
                        leave=False,
                        dynamic_ncols=True,
                    )
                    if progress_enabled
                    else citations
                )
            else:
                cit_iterator = []

            for paper in cit_iterator:
                if len(papers) >= self.max_papers:
                    break
                papers[paper.paper_id] = paper

                # Fetch reference lists
                if self.fetch_references and paper.paper_id not in self.reference_cache:
                    refs = self._get_references(paper.paper_id)
                    paper.references = refs

            if citations and progress_enabled:
                cit_iterator.close()

        reference_lists = len(self.reference_cache)
        summary = (
            f"Collected {len(papers)} papers ({reference_lists} with reference lists)"
        )
        logger.info(summary)
        self._abstract_index.build(papers)
        self._set_collection_summary(summary)

        return papers

    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        """
        Compute similarity using temporal, citation, and bibliographic factors.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :return float: Similarity score (0.0 to 1.0)
        """
        features = compute_similarity_features(
            paper1,
            paper2,
            abstract_similarity_fn=lambda a, b: self._abstract_index.similarity(
                a.paper_id, b.paper_id
            ),
            temporal_similarity_fn=self.temporal_similarity,
            citation_similarity_fn=self.citation_similarity,
            bibliographic_coupling_fn=self.bibliographic_coupling,
            use_bibliographic_coupling=bool(
                self.fetch_references and paper1.references and paper2.references
            ),
            with_references_weights=(0.40, 0.20, 0.00, 0.40),
            without_references_weights=(0.65, 0.20, 0.15, 0.00),
        )

        return features.combined_score

    def should_create_edge(
        self, paper1: Paper, paper2: Paper, similarity: float
    ) -> bool:
        """
        Decide whether to create edge based on similarity and sparsity goals.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :param float similarity: Computed similarity score
        :return bool: True if edge should be created
        """
        del paper1
        del paper2
        return similarity >= self.similarity_threshold
