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
from citemesh.strategies.similarity import compute_indexed_similarity_score

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
        refresh_reference_cache: bool = False,
        client: Optional[SemanticScholarClient] = None,
    ):
        """
        Initialize citation graph builder.

        :param int max_papers: Maximum total papers in graph
        :param int max_citations: Maximum citing papers to fetch
        :param int max_references: Maximum referenced papers to fetch
        :param float similarity_threshold: Minimum similarity for edges
        :param bool fetch_references: Whether to fetch reference lists (enables real bibliographic coupling)
        :param bool refresh_reference_cache: Whether to bypass persisted reference-cache reads.
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        """
        super().__init__(max_papers)
        self.max_citations = max_citations
        self.max_references = max_references
        self.similarity_threshold = similarity_threshold
        self.fetch_references = fetch_references
        self.refresh_reference_cache = bool(refresh_reference_cache)
        self.client: SemanticScholarClient = client or get_client()
        self.reference_cache: Dict[str, list] = {}  # Cache reference lists
        self._abstract_index = AbstractSimilarityIndex()

    def _ingest_relation_batch(
        self,
        papers: Dict[str, Paper],
        relation_records: list[Paper],
        progress_enabled: bool,
        progress_description: str,
    ) -> None:
        """Add related papers and hydrate references while respecting graph limits.

        :param Dict[str, Paper] papers: Collected paper mapping updated in-place.
        :param list[Paper] relation_records: Reference/citation papers from the API.
        :param bool progress_enabled: Whether to wrap records with ``tqdm``.
        :param str progress_description: Progress-bar description label.
        :return None: Mutates ``papers`` and optional per-paper references in place.
        """
        progress_bar = None
        if relation_records:
            # Avoid very short-lived progress bars that can render as blank spacer
            # lines in some terminals when rapidly cleared.
            should_show_progress = progress_enabled and len(relation_records) > 25
            if should_show_progress:
                progress_bar = tqdm(
                    relation_records,
                    desc=progress_description,
                    unit="papers",
                    dynamic_ncols=True,
                )
                relation_iterator = progress_bar
            else:
                relation_iterator = relation_records
        else:
            relation_iterator = []

        for paper in relation_iterator:
            if len(papers) >= self.max_papers:
                break
            papers[paper.paper_id] = paper

            if self.fetch_references and paper.paper_id not in self.reference_cache:
                paper.references = self._get_references(paper.paper_id)

        if progress_bar is not None:
            progress_bar.close()

    def _get_references(self, paper_id: str) -> list:
        """
        Get reference IDs for a paper with caching.

        :param str paper_id: Paper identifier
        :return list: List of referenced paper IDs
        """
        if not self.refresh_reference_cache and paper_id in self.reference_cache:
            return self.reference_cache[paper_id]
        if self.refresh_reference_cache:
            self.reference_cache.pop(paper_id, None)

        ref_ids = self.client.get_reference_ids(
            paper_id,
            force_refresh=self.refresh_reference_cache,
        )
        self.reference_cache[paper_id] = ref_ids
        return ref_ids

    def collect_papers(self, seed_id: str, **kwargs: Any) -> Dict[str, Paper]:
        """
        Collect papers via citations and references.

        :param str seed_id: Seed paper identifier
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Dict[str, Paper]: Dictionary of paper_id -> Paper objects
        """
        # Scope in-memory references to one collection request so stale entries do
        # not leak across caller boundaries when builders are reused.
        self.reference_cache.clear()
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

        logger.info("Seed: %s", seed.title)

        # Step 2: Fetch references (older papers)
        logger.info(f"Fetching up to {self.max_references} references...")
        references = self.client.get_paper_references(
            seed.paper_id, limit=self.max_references
        )
        progress_enabled = sys.stderr.isatty()
        self._ingest_relation_batch(
            papers,
            references,
            progress_enabled=progress_enabled,
            progress_description="Downloading references",
        )

        # Step 3: Fetch citations (newer papers)
        remaining = self.max_papers - len(papers)
        if remaining > 0:
            logger.info(
                f"Fetching up to {min(remaining, self.max_citations)} citations..."
            )
            citations = self.client.get_paper_citations(
                seed.paper_id, limit=min(remaining, self.max_citations)
            )
            self._ingest_relation_batch(
                papers,
                citations,
                progress_enabled=progress_enabled,
                progress_description="Downloading citations",
            )

        reference_lists = len(self.reference_cache)
        summary = (
            f"Collected {len(papers)} papers ({reference_lists} with reference lists)"
        )
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
        return compute_indexed_similarity_score(
            paper1,
            paper2,
            abstract_index=self._abstract_index,
            temporal_similarity_fn=self.temporal_similarity,
            citation_similarity_fn=self.citation_similarity,
            bibliographic_coupling_fn=self.bibliographic_coupling,
            fetch_references=self.fetch_references,
            with_references_weights=(0.40, 0.20, 0.00, 0.40),
            without_references_weights=(0.65, 0.20, 0.15, 0.00),
        )
