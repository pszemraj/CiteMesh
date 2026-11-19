"""Recommendation-based graph builder using Semantic Scholar recommendations API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from citemesh.core.models import PaperGraph

from citemesh.core import (
    DEFAULT_MAX_PAPERS,
    DEFAULT_RELATIONSHIP_SIMILARITY_THRESHOLD,
    Paper,
)
from citemesh.progress import progress_iterator
from citemesh.services import get_client
from citemesh.services.semantic_scholar.endpoints import RECOMMENDATION_MAX_RESULTS
from citemesh.strategies.base import (
    RELATIONSHIP_DEGREE_CAP,
    GraphBuilderStrategy,
    GraphCollectionOptions,
    build_capped_undirected_graph,
)
from citemesh.strategies.candidates import (
    fetch_candidate_source,
    require_available_candidate_source,
    scope_candidate_collection,
)
from citemesh.strategies.discovery_pool import DiscoveryPool
from citemesh.strategies.reference_hydration import ReferenceHydrator
from citemesh.strategies.similarity import AbstractSimilarityIndex

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
        max_papers: int = DEFAULT_MAX_PAPERS,
        fetch_references: bool = True,
        refresh_reference_cache: bool = False,
        similarity_threshold: float = DEFAULT_RELATIONSHIP_SIMILARITY_THRESHOLD,
        client: SemanticScholarClient | None = None,
    ) -> None:
        """Initialize recommendation graph builder.

        :param int max_papers: Maximum papers to include in graph.
        :param bool fetch_references: Whether to fetch references for focal and recommended papers.
        :param bool refresh_reference_cache: Whether to bypass persisted reference-cache reads.
        :param float similarity_threshold: Threshold for edge creation (default matches CLI).
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        """
        super().__init__(max_papers=max_papers)
        self.similarity_threshold = similarity_threshold
        self.client = client or get_client()
        self.reference_hydrator = ReferenceHydrator(
            self.client,
            fetch_references=fetch_references,
            refresh_reference_cache=refresh_reference_cache,
        )
        self._abstract_index = AbstractSimilarityIndex()
        self.candidate_source_status: dict[str, str] = {}

    @scope_candidate_collection
    def collect_papers(
        self, focal_id: str, **kwargs: Unpack[GraphCollectionOptions]
    ) -> dict[str, Paper]:
        """Collect recommendations for a focal paper.

        :param str focal_id: Focal paper identifier.
        :param kwargs: Extra options are ignored.
        :return Dict[str, Paper]: Papers included in graph.
        """
        papers: dict[str, Paper] = {}
        self.candidate_source_status = {}
        self.reference_hydrator.reset()

        focal = self._resolve_focal_paper(focal_id, client=self.client)
        papers[focal.paper_id] = focal

        logger.info("Collecting recommendations...")
        logger.debug("Recommendation focal paper: %s", focal.paper_id)
        recommendation_result = fetch_candidate_source(
            "recommendations",
            lambda: self.client.get_recommended_papers(
                focal.paper_id,
                limit=min(RECOMMENDATION_MAX_RESULTS, self.max_papers * 2),
                raise_on_unavailable=True,
            ),
        )
        self.candidate_source_status = {
            recommendation_result.source: recommendation_result.state.value
        }
        require_available_candidate_source(
            [recommendation_result],
            context=f"recommendation acquisition for {focal.paper_id}",
        )

        # Discovery is the primary acquisition path. Do it before optionally
        # enriching the focal paper, whose retries share the S2 recovery budget.
        recommendations = recommendation_result.papers
        reference_progress = (
            bool(recommendations) and self.reference_hydrator.fetch_references
        )
        if reference_progress:
            logger.info(
                "Checking reference lists for the focal paper and its recommendations..."
            )
        self.reference_hydrator.ensure(focal)

        # The pool merges records of one work, including records of the focal paper,
        # which enrich the original focal paper object so the graph keeps its focal paper node.
        pool = DiscoveryPool(focal)
        for paper in progress_iterator(
            recommendations,
            description="Checking recommendations",
            unit="papers",
            enabled=None if reference_progress else False,
        ):
            if not str(paper.paper_id).strip():
                continue
            target_id = pool.resolve(paper)
            if target_id is None:
                # A new member needs text for the abstract index and a free slot.
                if not paper.abstract or not paper.title:
                    continue
                if len(pool.papers) + 1 >= self.max_papers:
                    continue
                self.reference_hydrator.ensure(paper)
            elif target_id != focal.paper_id and not pool.papers[target_id].references:
                self.reference_hydrator.ensure(paper)
            pool.add(paper, source="recommendation", relation="semantic_only")
        papers.update(pool.papers)

        self._abstract_index.build(papers)
        self._set_collection_summary(
            f"Collected {len(papers)} papers from recommendations"
        )
        return papers

    def build_graph(
        self, focal_id: str, **kwargs: Unpack[GraphCollectionOptions]
    ) -> tuple[PaperGraph, str]:
        """Build a recommendation graph with candidate-source status metadata.

        :param str focal_id: Focal paper identifier.
        :param kwargs: Extra options are ignored by collection.
        :return tuple[PaperGraph, str]: Built graph and canonical focal paper identifier.
        """
        graph, actual_focal_id = super().build_graph(focal_id, **kwargs)
        graph.graph["candidate_source_status"] = dict(
            sorted(self.candidate_source_status.items())
        )
        filtered_graph = build_capped_undirected_graph(
            graph, RELATIONSHIP_DEGREE_CAP, focal_id=actual_focal_id
        )
        logger.debug(
            "Graph complete: %s nodes, %s edges",
            filtered_graph.number_of_nodes(),
            filtered_graph.number_of_edges(),
        )
        return filtered_graph, actual_focal_id

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
            abstract_index=self._abstract_index,
            reference_coupling=self.reference_hydrator.fetch_references,
            with_references_weights=(0.60, 0.15, 0.00, 0.25),
            without_references_weights=(0.75, 0.25, 0.00, 0.00),
            cap_at_one=True,
        )
