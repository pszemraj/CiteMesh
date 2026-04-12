"""Recommendation-based graph builder using Semantic Scholar recommendations API."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from citemesh.core import Paper
from citemesh.services import SemanticScholarClient, get_client
from citemesh.similarity import AbstractSimilarityIndex
from citemesh.strategies.base import GraphBuilderStrategy
from citemesh.strategies.similarity import compute_indexed_similarity_score

logger = logging.getLogger(__name__)


class RecommendationGraphBuilder(GraphBuilderStrategy):
    """
    Build a graph from S2 recommendation neighbors.
    """

    def __init__(
        self,
        max_papers: int = 40,
        fetch_references: bool = True,
        refresh_reference_cache: bool = False,
        similarity_threshold: float = 0.2,
        client: Optional[SemanticScholarClient] = None,
    ):
        """Initialize recommendation graph builder.

        :param int max_papers: Maximum papers to include in graph.
        :param bool fetch_references: Whether to fetch references for seed and recommended papers.
        :param bool refresh_reference_cache: Whether to bypass persisted reference-cache reads.
        :param float similarity_threshold: Threshold for edge creation (default matches CLI).
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        """
        super().__init__(max_papers=max_papers)
        self.fetch_references = fetch_references
        self.refresh_reference_cache = bool(refresh_reference_cache)
        self.similarity_threshold = similarity_threshold
        self.client = client or get_client()
        self._abstract_index = AbstractSimilarityIndex()

    def _hydrate_references(self, paper: Paper) -> None:
        """Populate reference IDs for a paper when strategy settings require it.

        :param Paper paper: Paper record to enrich.
        :return None: Mutates ``paper.references`` in place when successful.
        """
        if not self.fetch_references or paper.references:
            return

        try:
            paper.references = self.client.get_reference_ids(
                paper.paper_id,
                force_refresh=self.refresh_reference_cache,
            )
        except Exception as exc:
            logger.debug(
                "Could not fetch reference IDs for recommendation %s: %s",
                paper.paper_id,
                exc,
            )

    def collect_papers(self, seed_id: str, **kwargs: Any) -> Dict[str, Paper]:
        """Collect recommendations for a seed paper.

        :param str seed_id: Seed paper identifier.
        :param Any kwargs: Strategy-specific arguments (currently unused).
        :return Dict[str, Paper]: Papers included in graph.
        """
        papers: Dict[str, Paper] = {}

        logger.info("Fetching seed paper: %s", seed_id)
        seed = self.client.get_paper(seed_id, fetch_references=self.fetch_references)
        if not seed:
            raise ValueError(f"Seed paper not found: {seed_id}")

        seed.is_seed = True
        papers[seed.paper_id] = seed

        logger.info("Fetching recommendations for %s", seed.paper_id)
        recommendations = self.client.get_recommended_papers(
            seed.paper_id,
            limit=self.max_papers * 2,
            include_references=self.fetch_references,
        )

        for paper in recommendations:
            # Recommendation payloads can include the seed paper itself.
            # Preserve the original seed object so GraphBuilderStrategy can always
            # identify a node with ``is_seed=True``.
            if paper.paper_id == seed.paper_id:
                continue

            if not paper.abstract or not paper.title:
                continue

            existing = papers.get(paper.paper_id)
            if existing is not None:
                if existing.is_seed:
                    continue
                # Keep richer metadata when duplicates are returned.
                if (not existing.abstract and paper.abstract) or (
                    existing.title == "Unknown" and paper.title != "Unknown"
                ):
                    self._hydrate_references(paper)
                    papers[paper.paper_id] = paper
                elif not existing.references:
                    self._hydrate_references(existing)
                continue

            if len(papers) >= self.max_papers:
                break

            self._hydrate_references(paper)
            papers[paper.paper_id] = paper

        self._abstract_index.build(papers)
        self._set_collection_summary(
            f"Collected {len(papers)} papers from recommendations"
        )
        return papers

    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        """
        Compute similarity combining topical and temporal signals.

        60% abstract similarity, 15% temporal (when available), 25% bibliographic.

        :param Paper paper1: First paper.
        :param Paper paper2: Second paper.
        :return float: Similarity score in [0.0, 1.0].
        """
        return compute_indexed_similarity_score(
            paper1,
            paper2,
            abstract_index=self._abstract_index,
            temporal_similarity_fn=self.temporal_similarity,
            citation_similarity_fn=self.citation_similarity,
            bibliographic_coupling_fn=self.bibliographic_coupling,
            fetch_references=self.fetch_references,
            with_references_weights=(0.60, 0.15, 0.00, 0.25),
            without_references_weights=(0.75, 0.25, 0.00, 0.00),
            cap_at_one=True,
        )
