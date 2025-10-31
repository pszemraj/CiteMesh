"""
Citation-based graph building strategy.

This strategy builds similarity graphs using citation relationships,
bibliographic coupling (shared references), and co-citation analysis.
"""

import logging
import sys
from typing import Dict

import numpy as np
from tqdm.auto import tqdm

from citemesh.api_client import SemanticScholarClient, get_client
from citemesh.config import CITATION_CONFIG
from citemesh.models import Paper
from citemesh.strategies.base import GraphBuilderStrategy

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
        random_seed: int = None,
    ):
        """
        Initialize citation graph builder.

        Args:
            max_papers: Maximum total papers in graph
            max_citations: Maximum citing papers to fetch
            max_references: Maximum referenced papers to fetch
            similarity_threshold: Minimum similarity for edges
            fetch_references: Whether to fetch reference lists (enables real bibliographic coupling)
            random_seed: Random seed for reproducibility
        """
        super().__init__(max_papers, random_seed)
        self.max_citations = max_citations
        self.max_references = max_references
        self.similarity_threshold = similarity_threshold
        self.fetch_references = fetch_references
        self.client: SemanticScholarClient = get_client()
        self.reference_cache: Dict[str, list] = {}  # Cache reference lists

    def _get_references(self, paper_id: str) -> list:
        """
        Get reference IDs for a paper with caching.

        Args:
            paper_id: Paper identifier

        Returns:
            List of referenced paper IDs
        """
        if paper_id in self.reference_cache:
            return self.reference_cache[paper_id]

        ref_ids = self.client.get_reference_ids(paper_id)
        self.reference_cache[paper_id] = ref_ids
        return ref_ids

    def collect_papers(self, seed_id: str, **kwargs) -> Dict[str, Paper]:
        """
        Collect papers via citations and references.

        Args:
            seed_id: Seed paper identifier

        Returns:
            Dictionary of paper_id -> Paper objects
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
        self._set_collection_summary(summary)

        return papers

    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        """
        Compute similarity using temporal, citation, and bibliographic factors.

        This implements the bibliography overlap scoring used in citation meshes:
        - Temporal proximity (with strong penalties for distant papers)
        - Citation impact similarity (log scale)
        - Bibliographic coupling (shared references)

        Args:
            paper1: First paper
            paper2: Second paper

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Component 1: Temporal similarity
        temp_sim = self.temporal_similarity(paper1, paper2)

        # Component 2: Citation similarity
        cit_sim = self.citation_similarity(paper1, paper2)

        # Component 3: Bibliographic coupling (real shared references!)
        if self.fetch_references and paper1.references and paper2.references:
            # Use real bibliographic coupling
            bib_coupling = self.bibliographic_coupling(paper1, paper2)
        else:
            # Fallback: estimate based on temporal proximity
            year_diff = abs(paper1.year - paper2.year)
            if year_diff < 3:
                bib_coupling = 0.4  # Likely to share references
            else:
                bib_coupling = 0.1  # Less likely

        # Combined similarity with configured weights
        similarity = (
            CITATION_CONFIG.temporal_weight * temp_sim
            + CITATION_CONFIG.citation_weight * cit_sim
            + CITATION_CONFIG.bibliographic_weight * bib_coupling
        )

        return similarity

    def should_create_edge(
        self, paper1: Paper, paper2: Paper, similarity: float
    ) -> bool:
        """
        Decide whether to create edge based on similarity and sparsity goals.

        Implements selective edge creation to match CiteMesh sparsity:
        - Seed connects to highly similar papers
        - Other papers connect only if very similar and probabilistically

        Args:
            paper1: First paper
            paper2: Second paper
            similarity: Computed similarity score

        Returns:
            True if edge should be created
        """
        # Check minimum threshold
        if similarity < self.similarity_threshold:
            return False

        # Seed paper: lower threshold
        if paper1.is_seed or paper2.is_seed:
            return similarity > CITATION_CONFIG.seed_edge_threshold

        # Non-seed papers: stricter threshold with randomness for sparsity
        if similarity > CITATION_CONFIG.normal_edge_threshold:
            # Add randomness to create sparse mesh
            return np.random.random() > (1.0 - CITATION_CONFIG.random_edge_probability)

        return False
