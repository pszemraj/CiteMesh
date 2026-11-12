"""
Citation-based graph building strategy.

This strategy builds similarity graphs using citation relationships,
bibliographic coupling (shared references), and topical similarity.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterable
from dataclasses import replace
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
    _CorpusPreparedOption,
    build_capped_undirected_graph,
)
from citemesh.strategies.candidates import (
    CandidateSourceResult,
    CandidateSourceState,
    fetch_candidate_source,
    fetch_seed_references,
    merge_seed_relation,
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
        self.seed_relations: dict[str, str] = {}
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

        Records of the seed enrich it, records of a collected paper merge into
        it, and a new paper joins while the graph has a free slot.

        :param DiscoveryPool pool: Collected papers other than the seed.
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
            # A new work needs a free slot; the seed holds one of max_papers.
            if pool.resolve(paper) is None and len(pool.papers) + 1 >= self.max_papers:
                continue
            member_id = pool.add(paper, source="citation", relation="")
            if member_id is not None:
                processed_ids.append(member_id)

        if progress_bar is not None:
            progress_bar.close()

        return processed_ids

    def _record_seed_relations(self, paper_ids: list[str], relation: str) -> None:
        """Merge relation-to-seed labels for a paper batch.

        :param list[str] paper_ids: Relation batch paper IDs.
        :param str relation: Relation label.
        :return None: Updates ``self.seed_relations`` in place.
        """
        for paper_id in paper_ids:
            normalized_id = str(paper_id).strip()
            if not normalized_id:
                continue
            merged_relation = merge_seed_relation(
                self.seed_relations.get(normalized_id, ""), relation
            )
            if merged_relation:
                self.seed_relations[normalized_id] = merged_relation

    @scope_candidate_collection
    def collect_papers(
        self,
        seed_id: str,
        *,
        seed_paper: Paper | None = None,
        **kwargs: Unpack[_CorpusPreparedOption],
    ) -> dict[str, Paper]:
        """
        Collect papers via citations and references.

        :param str seed_id: Seed paper identifier
        :param Optional[Paper] seed_paper: Pre-resolved seed metadata. When its
            primary identifier is local, only an external alias is sent upstream.
        :param kwargs: Corpus-preparation option accepted by the shared interface.
        :return Dict[str, Paper]: Dictionary of paper_id -> Paper objects
        """
        # Scope in-memory references to one collection request so stale entries do
        # not leak across caller boundaries when builders are reused.
        self.reference_hydrator.reset()
        self.seed_relations = {}
        self.candidate_source_status = {}
        papers = {}
        source_results: list[CandidateSourceResult] = []

        # Step 1: Fetch seed paper
        seed = seed_paper
        if seed is None:
            logger.debug("Fetching seed paper: %s", seed_id)
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
        provider_seed_id = provider_lookup_identifier(seed.paper_id, seed)
        papers[seed.paper_id] = seed
        self.seed_relations[seed.paper_id] = "seed"
        pool = DiscoveryPool(seed)

        logger.debug("Seed: %s", seed.title)

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
            reference_results = fetch_seed_references(
                self.client,
                seed,
                reference_limit,
                seed_identifier=seed_id,
            )
            source_results.extend(reference_results)
            reference_ids = self._ingest_relation_batch(
                pool,
                [paper for result in reference_results for paper in result.papers],
                progress_enabled=show_progress,
                progress_description="Downloading references",
            )
            self._record_seed_relations(reference_ids, "referenced_by_seed")
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
                self.reference_hydrator.withhold_seed_references(seed)

        # Step 3: Fetch citations (newer papers)
        remaining = self.max_papers - 1 - len(pool.papers)
        citation_limit = min(
            remaining,
            self.max_citations,
        )
        if citation_limit > 0 and provider_seed_id is not None:
            logger.debug(f"Fetching up to {citation_limit} citations...")
            citation_result = fetch_candidate_source(
                "citations",
                lambda: self.client.get_paper_citations(
                    provider_seed_id,
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
            self._record_seed_relations(citation_ids, "cites_seed")

        papers.update(pool.papers)
        self.candidate_source_status = {
            result.source: result.state.value for result in source_results
        }
        require_available_candidate_source(
            source_results,
            context=f"citation acquisition for {seed.paper_id}",
        )

        # Candidate discovery is required acquisition. Complete every requested
        # source before optional reference enrichment can spend the shared S2
        # recovery budget.
        self.reference_hydrator.hydrate(papers, seed)

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
        self, seed_id: str, **kwargs: Unpack[GraphCollectionOptions]
    ) -> tuple[PaperGraph, str]:
        """Build citation graph and persist seed-relation metadata.

        :param str seed_id: Seed paper identifier.
        :param kwargs: Collection options forwarded to the parent build.
        :return tuple[PaperGraph, str]: Built graph and canonical seed identifier.
        """
        graph, actual_seed_id = super().build_graph(seed_id, **kwargs)
        graph.graph["seed_relations"] = {
            str(node_id): str(relation)
            for node_id, relation in sorted(
                self.seed_relations.items(), key=lambda x: x[0]
            )
            if str(node_id) in graph.nodes
        }
        graph.graph["candidate_source_status"] = dict(
            sorted(self.candidate_source_status.items())
        )
        filtered_graph = build_capped_undirected_graph(
            graph, RELATIONSHIP_DEGREE_CAP, seed_id=actual_seed_id
        )
        logger.debug(
            "Graph complete: %s nodes, %s edges",
            filtered_graph.number_of_nodes(),
            filtered_graph.number_of_edges(),
        )
        return filtered_graph, actual_seed_id

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
