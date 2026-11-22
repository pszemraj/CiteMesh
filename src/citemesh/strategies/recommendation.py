"""Recommendation-based graph builder using Semantic Scholar recommendations API."""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from citemesh.core.models import PaperGraph

from citemesh.core import (
    DEFAULT_MAX_PAPERS,
    DEFAULT_RELATIONSHIP_SIMILARITY_THRESHOLD,
    Paper,
)
from citemesh.core.paper_ids import normalize_paper_id
from citemesh.progress import progress_iterator
from citemesh.services import get_client
from citemesh.services.semantic_scholar.endpoints import RECOMMENDATION_MAX_RESULTS
from citemesh.strategies.base import (
    GraphBuilderStrategy,
    GraphCollectionOptions,
    _finalize_relationship_graph,
)
from citemesh.strategies.candidates import (
    fetch_candidate_source,
    provider_lookup_identifier,
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
            self.client,
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

        recommendations = recommendation_result.papers
        fetch_references = self.reference_hydrator.fetch_references
        reference_progress = bool(recommendations) and fetch_references
        if reference_progress:
            logger.info(
                "Checking reference lists for the focal paper and its recommendations..."
            )
        lookups: dict[str, deque[str]] = {}
        reference_prefix_sizes: dict[str, int] = {}
        if (
            fetch_references
            and not focal.references
            and (lookup_id := provider_lookup_identifier(focal.paper_id, focal))
        ):
            lookups[focal.paper_id] = deque([normalize_paper_id(lookup_id)])

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
            # Record aliases that the original order would try if earlier lookups
            # proved empty. Identity matching does not depend on reference lists.
            lookup_id = (
                provider_lookup_identifier(paper.paper_id, paper)
                if fetch_references
                and target_id != focal.paper_id
                and not paper.references
                and (target_id is None or not pool.papers[target_id].references)
                else None
            )
            # A local record taking over a work keeps its own inline references first.
            takes_over = (
                paper.is_local_corpus
                and target_id is not None
                and target_id != paper.paper_id
                and target_id in lookups
            )
            prefix_size = (
                len({value for ref in paper.references if (value := str(ref).strip())})
                if takes_over
                else 0
            )
            canonical_id = pool.add(
                paper, source="recommendation", relation="semantic_only"
            )
            if canonical_id is not None:
                if (
                    target_id is not None
                    and target_id != canonical_id
                    and target_id in lookups
                ):
                    lookups[canonical_id] = lookups.pop(target_id)
                    if prefix_size:
                        reference_prefix_sizes[canonical_id] = prefix_size
                if lookup_id is not None:
                    aliases = lookups.setdefault(canonical_id, deque())
                    normalized_id = normalize_paper_id(lookup_id)
                    if normalized_id not in aliases:
                        aliases.append(normalized_id)
        papers.update(pool.papers)
        self._hydrate_recommendation_references(papers, lookups, reference_prefix_sizes)

        self._abstract_index.build(papers)
        self._set_collection_summary(
            f"Collected {len(papers)} papers from recommendations"
        )
        return papers

    def _hydrate_recommendation_references(
        self,
        papers: dict[str, Paper],
        lookups: dict[str, deque[str]],
        reference_prefix_sizes: dict[str, int],
    ) -> None:
        """Batch the next eligible lookup for each work until it has references."""
        if not lookups:
            return
        seen: dict[str, list[str]] = {}
        with self.client.retry_section("reference hydration"):
            while lookups:
                batch = {paper_id: ids.popleft() for paper_id, ids in lookups.items()}
                missing = list(
                    dict.fromkeys(x for x in batch.values() if x not in seen)
                )
                if missing:
                    fetched = self.reference_hydrator.fetch_many(missing)
                    seen.update(
                        (lookup_id, fetched.get(lookup_id, [])) for lookup_id in missing
                    )
                for paper_id, lookup_id in batch.items():
                    if references := seen[lookup_id]:
                        paper = papers[paper_id]
                        prefix_size = reference_prefix_sizes.get(paper_id, 0)
                        paper.references = list(
                            dict.fromkeys(
                                [
                                    *paper.references[:prefix_size],
                                    *references,
                                    *paper.references[prefix_size:],
                                ]
                            )
                        )
                        del lookups[paper_id]
                    elif not lookups[paper_id]:
                        del lookups[paper_id]

    def build_graph(
        self, focal_id: str, **kwargs: Unpack[GraphCollectionOptions]
    ) -> tuple[PaperGraph, str]:
        """Build a recommendation graph with candidate-source status metadata.

        :param str focal_id: Focal paper identifier.
        :param kwargs: Extra options are ignored by collection.
        :return tuple[PaperGraph, str]: Built graph and canonical focal paper identifier.
        """
        graph, actual_focal_id = super().build_graph(focal_id, **kwargs)
        graph = _finalize_relationship_graph(
            graph, actual_focal_id, self.candidate_source_status
        )
        return graph, actual_focal_id

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
