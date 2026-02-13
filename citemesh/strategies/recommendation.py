"""Recommendation-based graph builder using Semantic Scholar recommendations API."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from citemesh.core import Paper
from citemesh.services import SemanticScholarClient, get_client
from citemesh.similarity import AbstractSimilarityIndex
from citemesh.strategies.base import GraphBuilderStrategy

logger = logging.getLogger(__name__)


class RecommendationGraphBuilder(GraphBuilderStrategy):
    """
    Build a graph from S2 recommendation neighbors.
    """

    def __init__(
        self,
        max_papers: int = 40,
        fetch_references: bool = False,
        similarity_threshold: float = 0.15,
        random_seed: Optional[int] = None,
        client: Optional[SemanticScholarClient] = None,
    ):
        """Initialize recommendation graph builder.

        :param int max_papers: Maximum papers to include in graph.
        :param bool fetch_references: Whether to fetch references for seed and recommended papers.
        :param float similarity_threshold: Threshold for edge creation.
        :param Optional[int] random_seed: Seed for reproducibility.
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        """
        super().__init__(max_papers=max_papers, random_seed=random_seed)
        self.fetch_references = fetch_references
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
            paper.references = self.client.get_reference_ids(paper.paper_id)
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

        logger.info("Collected %s papers from recommendations", len(papers))
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
        abstract_sim = self._abstract_index.similarity(paper1.paper_id, paper2.paper_id)
        temporal_sim = self.temporal_similarity(paper1, paper2)

        if paper1.references and paper2.references:
            bib_similarity = self.bibliographic_coupling(paper1, paper2)
            similarity = (
                0.60 * abstract_sim + 0.15 * temporal_sim + 0.25 * bib_similarity
            )
        else:
            similarity = 0.75 * abstract_sim + 0.25 * temporal_sim

        return min(similarity, 1.0)

    def should_create_edge(
        self, paper1: Paper, paper2: Paper, similarity: float
    ) -> bool:
        """Apply threshold logic for recommendation-derived edges.

        :param Paper paper1: First paper.
        :param Paper paper2: Second paper.
        :param float similarity: Computed similarity score.
        :return bool: ``True`` when edge should be kept.
        """
        if similarity < self.similarity_threshold:
            return False
        if paper1.is_seed or paper2.is_seed:
            return similarity >= max(self.similarity_threshold, 0.2)
        return similarity >= self.similarity_threshold * 1.5
