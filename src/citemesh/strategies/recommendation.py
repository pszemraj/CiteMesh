"""Recommendation-based graph builder using Semantic Scholar recommendations API."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import networkx as nx

from citemesh.core import Paper
from citemesh.services import get_client
from citemesh.services.semantic_scholar.endpoints import RECOMMENDATION_MAX_RESULTS
from citemesh.strategies.base import (
    GraphBuilderStrategy,
    build_capped_undirected_graph,
)
from citemesh.strategies.candidates import (
    candidate_records_match,
    fetch_candidate_source,
    merge_paper_metadata,
    require_available_candidate_source,
    scope_candidate_collection,
)
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
        max_papers: int = 40,
        fetch_references: bool = True,
        refresh_reference_cache: bool = False,
        similarity_threshold: float = 0.2,
        client: SemanticScholarClient | None = None,
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
        self.candidate_source_status: dict[str, str] = {}
        self._reference_source_unavailable = False

    def _hydrate_references(self, paper: Paper) -> None:
        """Populate reference IDs for a paper when strategy settings require it.

        :param Paper paper: Paper record to enrich.
        :return None: Mutates ``paper.references`` in place when successful.
        """
        if not self.fetch_references or paper.references:
            return

        if self._reference_source_unavailable:
            if not self.refresh_reference_cache:
                cached_references = self.client.get_cached_reference_ids(paper.paper_id)
                if cached_references is not None:
                    paper.references = list(cached_references)
            return

        from citemesh.services import SemanticScholarUnavailableError

        try:
            paper.references = self.client.get_reference_ids(
                paper.paper_id,
                force_refresh=self.refresh_reference_cache,
            )
        except SemanticScholarUnavailableError as exc:
            self._reference_source_unavailable = True
            logger.warning(
                "Reference IDs unavailable for recommendation %s; continuing "
                "without further reference hydration for this collection: %s",
                paper.paper_id,
                exc,
            )

    @scope_candidate_collection
    def collect_papers(self, seed_id: str, **kwargs: Any) -> dict[str, Paper]:
        """Collect recommendations for a seed paper.

        :param str seed_id: Seed paper identifier.
        :param Any kwargs: Strategy-specific arguments (currently unused).
        :return Dict[str, Paper]: Papers included in graph.
        """
        papers: dict[str, Paper] = {}
        self.candidate_source_status = {}
        self._reference_source_unavailable = False

        logger.info("Fetching seed paper: %s", seed_id)
        seed = self.client.get_paper(
            seed_id,
            raise_on_unavailable=True,
        )
        if not seed:
            raise ValueError(
                f"Seed paper not found: {seed_id} (Semantic Scholar does not "
                "know this identifier; check the DOI/arXiv/S2 ID)."
            )

        # Seed role belongs to this build, not the client-owned metadata record.
        seed = replace(seed, is_seed=True)
        papers[seed.paper_id] = seed

        logger.info("Fetching recommendations for %s", seed.paper_id)
        recommendation_result = fetch_candidate_source(
            "recommendations",
            lambda: self.client.get_recommended_papers(
                seed.paper_id,
                limit=min(RECOMMENDATION_MAX_RESULTS, self.max_papers * 2),
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

        # Discovery is the primary acquisition path. Do it before optional seed
        # enrichment, whose retries share the collection's S2 recovery budget.
        self._hydrate_references(seed)

        for paper in recommendation_result.papers:
            # Recommendation payloads can include the seed paper itself.
            # Preserve the original seed object so GraphBuilderStrategy can always
            # identify a node with ``is_seed=True``.
            paper_id = str(paper.paper_id).strip()
            if not paper_id:
                continue
            if paper_id == seed.paper_id or candidate_records_match(seed, paper):
                merge_paper_metadata(seed, paper)
                continue
            canonical_id = paper_id
            existing = papers.get(canonical_id)
            if existing is None:
                matching_ids = [
                    candidate_id
                    for candidate_id, candidate in papers.items()
                    if candidate_id != seed.paper_id
                    and candidate_records_match(candidate, paper)
                ]
                if len(matching_ids) == 1:
                    canonical_id = matching_ids[0]
                    existing = papers[canonical_id]
            if existing is not None:
                if not existing.references:
                    self._hydrate_references(paper)
                merge_paper_metadata(existing, paper)
                continue

            if not paper.abstract or not paper.title:
                continue

            if len(papers) >= self.max_papers:
                continue

            self._hydrate_references(paper)
            papers[canonical_id] = paper

        self._abstract_index.build(papers)
        self._set_collection_summary(
            f"Collected {len(papers)} papers from recommendations"
        )
        return papers

    def build_graph(self, seed_id: str, **kwargs: Any) -> tuple[nx.Graph, str]:
        """Build a recommendation graph with candidate-source status metadata.

        :param str seed_id: Seed paper identifier.
        :param Any kwargs: Strategy-specific options forwarded to parent build.
        :return Tuple[nx.Graph, str]: Built graph and canonical seed identifier.
        """
        graph, actual_seed_id = super().build_graph(seed_id, **kwargs)
        graph.graph["candidate_source_status"] = dict(
            sorted(self.candidate_source_status.items())
        )
        filtered_graph = build_capped_undirected_graph(graph, 3, seed_id=actual_seed_id)
        logger.info(
            "Graph complete: %s nodes, %s edges",
            filtered_graph.number_of_nodes(),
            filtered_graph.number_of_edges(),
        )
        return filtered_graph, actual_seed_id

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
