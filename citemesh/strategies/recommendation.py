"""Recommendation-based graph builder using Semantic Scholar recommendations API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import networkx as nx

from citemesh.core import Paper
from citemesh.services import get_client
from citemesh.similarity import AbstractSimilarityIndex
from citemesh.strategies.base import GraphBuilderStrategy
from citemesh.strategies.candidates import (
    IdentityRegistry,
    fetch_candidate_source,
    merge_paper_metadata,
    reconcile_paper_identity,
    register_aliases,
    require_available_candidate_source,
)

if TYPE_CHECKING:
    from citemesh.services.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)


class RecommendationGraphBuilder(GraphBuilderStrategy):
    """
    Build a graph from S2 recommendation neighbors.
    """

    strategy_name = "recommendation"

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
        self.candidate_source_status: Dict[str, str] = {}

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
        self.candidate_source_status = {}

        logger.info("Fetching seed paper: %s", seed_id)
        seed = self.client.get_paper(
            seed_id,
            fetch_references=self.fetch_references,
            raise_on_unavailable=True,
        )
        if not seed:
            raise ValueError(
                f"Seed paper not found: {seed_id} (Semantic Scholar does not "
                "know this identifier; check the DOI/arXiv/S2 ID)."
            )

        seed.is_seed = True
        papers[seed.paper_id] = seed
        identity_aliases = IdentityRegistry()
        register_aliases(identity_aliases, seed.paper_id, seed)

        logger.info("Fetching recommendations for %s", seed.paper_id)
        recommendation_result = fetch_candidate_source(
            "recommendations",
            lambda: self.client.get_recommended_papers(
                seed.paper_id,
                limit=self.max_papers * 2,
                raise_on_unavailable=True,
            ),
        )
        self.candidate_source_status = {
            recommendation_result.source: recommendation_result.state.value
        }
        require_available_candidate_source(
            [recommendation_result],
            context=f"recommendation acquisition for {seed.paper_id}",
        )

        for paper in recommendation_result.papers:
            # Recommendation payloads can include the seed paper itself.
            # Preserve the original seed object so GraphBuilderStrategy can always
            # identify a node with ``is_seed=True``.
            reconciliation = reconcile_paper_identity(
                identity_aliases, seed, papers, paper
            )
            if reconciliation.seed_matched:
                continue

            canonical_id = reconciliation.canonical_id
            if canonical_id is not None:
                existing = papers[canonical_id]
                if not existing.references:
                    self._hydrate_references(paper)
                    merge_paper_metadata(existing, paper)
                continue

            if not paper.abstract or not paper.title:
                continue

            if len(papers) >= self.max_papers:
                continue

            self._hydrate_references(paper)
            papers[paper.paper_id] = paper
            register_aliases(identity_aliases, paper.paper_id, paper)

        self._abstract_index.build(papers)
        self._set_collection_summary(
            f"Collected {len(papers)} papers from recommendations"
        )
        return papers

    def build_graph(self, seed_id: str, **kwargs: Any) -> Tuple[nx.Graph, str]:
        """Build a recommendation graph with candidate-source status metadata.

        :param str seed_id: Seed paper identifier.
        :param Any kwargs: Strategy-specific options forwarded to parent build.
        :return Tuple[nx.Graph, str]: Built graph and canonical seed identifier.
        """
        graph, actual_seed_id = super().build_graph(seed_id, **kwargs)
        graph.graph["candidate_source_status"] = dict(
            sorted(self.candidate_source_status.items())
        )
        return graph, actual_seed_id

    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        """
        Compute similarity combining topical and temporal signals.

        60% abstract similarity, 15% temporal (when available), 25% bibliographic.

        :param Paper paper1: First paper.
        :param Paper paper2: Second paper.
        :return float: Similarity score in [0.0, 1.0].
        """
        return self._compute_indexed_similarity(
            paper1,
            paper2,
            with_references_weights=(0.60, 0.15, 0.00, 0.25),
            without_references_weights=(0.75, 0.25, 0.00, 0.00),
            cap_at_one=True,
        )
