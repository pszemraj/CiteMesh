"""
Citation-based graph building strategy.

This strategy builds similarity graphs using citation relationships,
bibliographic coupling (shared references), and topical similarity.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from citemesh.core.models import PaperGraph

from citemesh.core import (
    DEFAULT_MAX_PAPERS,
    DEFAULT_RELATIONSHIP_SIMILARITY_THRESHOLD,
    Paper,
)
from citemesh.progress import progress_enabled, progress_iterator
from citemesh.services import get_client
from citemesh.strategies.base import (
    RELATIONSHIP_DEGREE_CAP,
    GraphBuilderStrategy,
    GraphCollectionOptions,
    build_capped_undirected_graph,
)
from citemesh.strategies.candidates import (
    CandidateSourceResult,
    CandidateSourceState,
    fetch_candidate_source,
    fetch_focal_references,
    merge_focal_relation,
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


class CitationGraphBuilder(GraphBuilderStrategy):
    """
    Build similarity graphs using citation relationships.

    This strategy implements:
    - Real bibliographic coupling (shared references)
    - Temporal proximity scoring
    - Citation impact similarity
    - Sparse edge creation for readability
    """

    strategy_name = "citation"

    def __init__(
        self,
        max_papers: int = DEFAULT_MAX_PAPERS,
        max_citations: int = 25,
        max_references: int = 25,
        similarity_threshold: float = DEFAULT_RELATIONSHIP_SIMILARITY_THRESHOLD,
        fetch_references: bool = True,
        refresh_reference_cache: bool = False,
        client: SemanticScholarClient | None = None,
    ) -> None:
        """
        Initialize citation graph builder.

        :param int max_papers: Maximum total papers in graph
        :param int max_citations: Maximum citing papers to fetch (default matches CLI)
        :param int max_references: Maximum referenced papers to fetch (default matches CLI)
        :param float similarity_threshold: Minimum similarity for edges
        :param bool fetch_references: Whether to fetch reference lists (enables real bibliographic coupling)
        :param bool refresh_reference_cache: Whether to bypass persisted reference-cache reads.
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        """
        super().__init__(max_papers)
        self.max_citations = max_citations
        self.max_references = max_references
        self.similarity_threshold = similarity_threshold
        self.client: SemanticScholarClient = client or get_client()
        self.reference_hydrator = ReferenceHydrator(
            self.client,
            fetch_references=fetch_references,
            refresh_reference_cache=refresh_reference_cache,
        )
        self.focal_relations: dict[str, str] = {}
        self.candidate_source_status: dict[str, str] = {}
        self._abstract_index = AbstractSimilarityIndex()

    def _ingest_relation_batch(
        self,
        pool: DiscoveryPool,
        relation_records: list[Paper],
        progress_enabled: bool,
        progress_description: str,
    ) -> list[str]:
        """Add related papers while respecting graph limits.

        Records of the focal paper enrich it, records of a collected paper merge into
        it, and a new paper joins while the graph has a free slot.

        :param DiscoveryPool pool: Collected papers other than the focal paper.
        :param list[Paper] relation_records: Reference/citation papers from the API.
        :param bool progress_enabled: Whether to wrap records with a progress bar.
        :param str progress_description: Progress-bar description label.
        :return list[str]: Processed relation paper IDs in traversal order.
        """
        processed_ids: list[str] = []
        progress_bar: Generator[Paper, None, None] | None = None
        relation_iterator: Iterable[Paper] = relation_records
        if relation_records:
            # Avoid very short-lived progress bars that can render as blank spacer
            # lines in some terminals when rapidly cleared.
            should_show_progress = progress_enabled and len(relation_records) > 25
            if should_show_progress:
                progress_bar = progress_iterator(
                    relation_records,
                    description=progress_description,
                    unit="papers",
                )
                relation_iterator = progress_bar

        for paper in relation_iterator:
            if not str(paper.paper_id).strip():
                continue
            # A new work needs a free slot; the focal paper holds one of max_papers.
            if pool.resolve(paper) is None and len(pool.papers) + 1 >= self.max_papers:
                continue
            member_id = pool.add(paper, source="citation", relation="")
            if member_id is not None:
                processed_ids.append(member_id)

        if progress_bar is not None:
            progress_bar.close()

        return processed_ids

    def _record_focal_relations(self, paper_ids: list[str], relation: str) -> None:
        """Merge labels describing each paper's relation to the focal paper.

        :param list[str] paper_ids: Relation batch paper IDs.
        :param str relation: Relation label.
        :return None: Updates ``self.focal_relations`` in place.
        """
        for paper_id in paper_ids:
            normalized_id = str(paper_id).strip()
            if not normalized_id:
                continue
            merged_relation = merge_focal_relation(
                self.focal_relations.get(normalized_id, ""), relation
            )
            if merged_relation:
                self.focal_relations[normalized_id] = merged_relation

    @scope_candidate_collection
    def collect_papers(
        self,
        focal_id: str,
        *,
        focal_paper: Paper | None = None,
        _corpus_prepared: bool = False,
    ) -> dict[str, Paper]:
        """
        Collect papers via citations and references.

        :param str focal_id: Focal paper identifier
        :param Optional[Paper] focal_paper: Pre-resolved focal paper metadata. When its
            primary identifier is local, only an external alias is sent upstream.
        :param bool _corpus_prepared: Corpus-preparation option accepted by the
            shared interface.
        :return Dict[str, Paper]: Dictionary of paper_id -> Paper objects
        """
        # Scope in-memory references to one collection request so stale entries do
        # not leak across caller boundaries when builders are reused.
        self.reference_hydrator.reset()
        self.focal_relations = {}
        self.candidate_source_status = {}
        papers = {}
        source_results: list[CandidateSourceResult] = []

        focal = self._resolve_focal_paper(
            focal_id, client=self.client, focal_paper=focal_paper
        )
        provider_focal_id = provider_lookup_identifier(focal.paper_id, focal)
        papers[focal.paper_id] = focal
        self.focal_relations[focal.paper_id] = "focal"
        pool = DiscoveryPool(focal)

        if self.max_papers > len(papers) and (
            self.max_references > 0 or self.max_citations > 0
        ):
            logger.info("Collecting related papers...")
            logger.debug(
                "Related-paper limits: references=%d, citations=%d, max_papers=%d.",
                self.max_references,
                self.max_citations,
                self.max_papers,
            )

        # Step 2: Fetch references (older papers)
        show_progress = progress_enabled()
        remaining = self.max_papers - len(papers)
        reference_limit = min(
            remaining,
            self.max_references,
        )
        if reference_limit > 0:
            reference_results = fetch_focal_references(
                self.client,
                focal,
                reference_limit,
                focal_identifier=focal_id,
            )
            source_results.extend(reference_results)
            reference_ids = self._ingest_relation_batch(
                pool,
                [paper for result in reference_results for paper in result.papers],
                progress_enabled=show_progress,
                progress_description="Downloading references",
            )
            self._record_focal_relations(reference_ids, "referenced_by_focal")
            s2_references_empty = any(
                result.source == "references"
                and result.state is CandidateSourceState.EMPTY
                for result in reference_results
            )
            if s2_references_empty:
                # Reuse partial recovery only within this build. Never persist it
                # as the complete S2 bibliography or repeat the empty discovery.
                # Candidate relations remain available, but this capped subset
                # must not drive shared-reference coupling as a full bibliography.
                self.reference_hydrator.withhold_focal_references(focal)

        # Step 3: Fetch citations (newer papers)
        remaining = self.max_papers - 1 - len(pool.papers)
        citation_limit = min(
            remaining,
            self.max_citations,
        )
        if citation_limit > 0 and provider_focal_id is not None:
            logger.debug(f"Fetching up to {citation_limit} citations...")
            citation_result = fetch_candidate_source(
                "citations",
                lambda: self.client.get_paper_citations(
                    provider_focal_id,
                    limit=citation_limit,
                    raise_on_unavailable=True,
                ),
            )
            source_results.append(citation_result)
            citation_ids = self._ingest_relation_batch(
                pool,
                list(citation_result.papers),
                progress_enabled=show_progress,
                progress_description="Downloading citations",
            )
            self._record_focal_relations(citation_ids, "cites_focal")

        papers.update(pool.papers)
        self.candidate_source_status = {
            result.source: result.state.value for result in source_results
        }
        require_available_candidate_source(
            source_results,
            context=f"citation acquisition for {focal.paper_id}",
        )

        # Candidate discovery is required acquisition. Complete every requested
        # source before optional reference enrichment can spend the shared S2
        # recovery budget.
        self.reference_hydrator.hydrate(papers, focal)

        reference_lists = sum(bool(paper.references) for paper in papers.values())
        summary = f"Collected {len(papers)} papers"
        logger.debug(
            "Collected-paper reference hydration: %d of %d papers have reference lists.",
            reference_lists,
            len(papers),
        )
        self._abstract_index.build(papers)
        self._set_collection_summary(summary)

        return papers

    def build_graph(
        self, focal_id: str, **kwargs: Unpack[GraphCollectionOptions]
    ) -> tuple[PaperGraph, str]:
        """Build a citation graph and persist relations to the focal paper.

        :param str focal_id: Focal paper identifier.
        :param kwargs: Collection options forwarded to the parent build.
        :return tuple[PaperGraph, str]: Built graph and canonical focal paper identifier.
        """
        graph, actual_focal_id = super().build_graph(focal_id, **kwargs)
        graph.graph["focal_relations"] = {
            str(node_id): str(relation)
            for node_id, relation in sorted(
                self.focal_relations.items(), key=lambda x: x[0]
            )
            if str(node_id) in graph.nodes
        }
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
        Compute similarity using temporal, citation, and bibliographic factors.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :return float: Similarity score (0.0 to 1.0)
        """
        return self._compute_indexed_similarity(
            paper1,
            paper2,
            abstract_index=self._abstract_index,
            reference_coupling=self.reference_hydrator.fetch_references,
            with_references_weights=(0.40, 0.20, 0.00, 0.40),
            without_references_weights=(0.65, 0.20, 0.15, 0.00),
        )
