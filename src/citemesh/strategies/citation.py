"""
Citation-based graph building strategy.

This strategy builds similarity graphs using citation relationships,
bibliographic coupling (shared references), and topical similarity.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import networkx as nx

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
    build_capped_undirected_graph,
)
from citemesh.strategies.candidates import (
    CandidateSourceResult,
    CandidateSourceState,
    ReferenceSelector,
    candidate_records_match,
    fetch_candidate_source,
    fetch_seed_references,
    merge_paper_metadata,
    merge_seed_relation,
    provider_lookup_identifier,
    require_available_candidate_source,
    scope_candidate_collection,
)
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
    ):
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
        self.fetch_references = fetch_references
        self.refresh_reference_cache = bool(refresh_reference_cache)
        self.client: SemanticScholarClient = client or get_client()
        self.reference_cache: dict[str, list] = {}  # Cache reference lists
        self.seed_relations: dict[str, str] = {}
        self.candidate_source_status: dict[str, str] = {}
        self.candidate_source_results: tuple[CandidateSourceResult, ...] = ()
        self._abstract_index = AbstractSimilarityIndex()
        self._reference_source_unavailable = False

    def _ensure_paper_references(
        self, paper: Paper, *, provider_lookup_id: str | None = None
    ) -> None:
        """Hydrate reference IDs for a paper without clobbering existing payload.

        :param Paper paper: Paper record to hydrate in-place.
        :param Optional[str] provider_lookup_id: Provider-safe request identifier;
            defaults to the paper's primary identifier.
        :return None: Mutates ``paper.references`` when needed.
        """
        if not self.fetch_references:
            return

        paper_id = str(paper.paper_id).strip()
        if not paper_id:
            return

        if paper.references:
            self.reference_cache.setdefault(paper_id, list(paper.references))
            return

        cached_refs = self.reference_cache.get(paper_id)
        if cached_refs is not None:
            paper.references = list(cached_refs)
            return
        if self._reference_source_unavailable:
            if not self.refresh_reference_cache:
                cached_refs = self.client.get_cached_reference_ids(
                    provider_lookup_id or paper_id
                )
                if cached_refs is not None:
                    self.reference_cache[paper_id] = cached_refs
                    paper.references = list(cached_refs)
            return

        from citemesh.services import SemanticScholarUnavailableError

        try:
            paper.references = self._get_references(
                paper_id, provider_lookup_id=provider_lookup_id
            )
        except SemanticScholarUnavailableError as exc:
            self._reference_source_unavailable = True
            logger.warning(
                "Reference hydration unavailable; continuing with available "
                "reference data."
            )
            logger.debug(
                "Reference hydration failed for related paper %s: %s",
                paper_id,
                exc,
            )

    def _ingest_relation_batch(
        self,
        papers: dict[str, Paper],
        seed: Paper,
        relation_records: list[Paper],
        progress_enabled: bool,
        progress_description: str,
    ) -> list[str]:
        """Add related papers while respecting graph limits.

        :param Dict[str, Paper] papers: Collected paper mapping updated in-place.
        :param Paper seed: Canonical seed paper in ``papers``.
        :param list[Paper] relation_records: Reference/citation papers from the API.
        :param bool progress_enabled: Whether to wrap records with a progress bar.
        :param str progress_description: Progress-bar description label.
        :return list[str]: Processed relation paper IDs in traversal order.
        """
        processed_ids: list[str] = []
        progress_bar = None
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
            else:
                relation_iterator = relation_records
        else:
            relation_iterator = []

        for paper in relation_iterator:
            raw_paper_id = str(paper.paper_id).strip()
            if not raw_paper_id:
                continue
            if raw_paper_id == seed.paper_id or candidate_records_match(seed, paper):
                merge_paper_metadata(seed, paper)
                continue
            canonical_id = raw_paper_id
            existing = papers.get(canonical_id)
            if existing is None:
                matching_ids = [
                    paper_id
                    for paper_id, candidate in papers.items()
                    if paper_id != seed.paper_id
                    and candidate_records_match(candidate, paper)
                ]
                if len(matching_ids) == 1:
                    canonical_id = matching_ids[0]
                    existing = papers[canonical_id]
            if existing is not None:
                merge_paper_metadata(existing, paper)
                processed_ids.append(canonical_id)
                continue

            if len(papers) >= self.max_papers:
                continue

            papers[canonical_id] = paper
            processed_ids.append(canonical_id)

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

    def hydrate_collected_references(
        self, papers: dict[str, Paper], seed: Paper
    ) -> None:
        """Hydrate optional reference IDs after required candidate acquisition.

        :param Dict[str, Paper] papers: Collected citation papers.
        :param Paper seed: Canonical seed paper in ``papers``.
        :return None: Updates paper records and the in-memory reference cache.
        """
        if self.fetch_references:
            logger.debug("Hydrating reference lists for %d papers...", len(papers))

        provider_seed_id = provider_lookup_identifier(seed.paper_id, seed)
        if seed.references:
            self._ensure_paper_references(seed)
        elif provider_seed_id is not None:
            self._ensure_paper_references(seed, provider_lookup_id=provider_seed_id)

        progress_bar = None
        paper_items = papers.items()
        if self.fetch_references and progress_enabled() and len(papers) > 25:
            progress_bar = progress_iterator(
                paper_items,
                description="Hydrating reference lists",
                unit="papers",
            )
            paper_items = progress_bar

        try:
            for paper_id, paper in paper_items:
                if paper_id != seed.paper_id:
                    self._ensure_paper_references(paper)
        finally:
            if progress_bar is not None:
                progress_bar.close()

    def _get_references(
        self, paper_id: str, *, provider_lookup_id: str | None = None
    ) -> list:
        """
        Get reference IDs for a paper with caching.

        :param str paper_id: Local cache key for the paper.
        :param Optional[str] provider_lookup_id: Provider-safe request identifier;
            defaults to ``paper_id``.
        :return list: List of referenced paper IDs
        """
        if not self.refresh_reference_cache and paper_id in self.reference_cache:
            return self.reference_cache[paper_id]
        if self.refresh_reference_cache:
            self.reference_cache.pop(paper_id, None)

        ref_ids = self.client.get_reference_ids(
            provider_lookup_id or paper_id,
            force_refresh=self.refresh_reference_cache,
        )
        self.reference_cache[paper_id] = ref_ids
        return ref_ids

    @scope_candidate_collection
    def collect_papers(
        self,
        seed_id: str,
        *,
        validate_source_availability: bool = True,
        hydrate_references: bool = True,
        seed_paper: Paper | None = None,
        reference_metadata_lookup: Callable[[list[str]], dict[str, Paper]]
        | None = None,
        reference_selector: ReferenceSelector | None = None,
        **kwargs: Any,
    ) -> dict[str, Paper]:
        """
        Collect papers via citations and references.

        :param str seed_id: Seed paper identifier
        :param bool validate_source_availability: Whether to fail when every
            requested relation source is unavailable.
        :param bool hydrate_references: Whether to perform optional reference-ID
            enrichment before returning.
        :param Optional[Paper] seed_paper: Pre-resolved seed metadata. When its
            primary identifier is local, only an external alias is sent upstream.
        :param Callable | None reference_metadata_lookup: Read metadata from an
            already prepared corpus for HTML reference recovery.
        :param Callable | None reference_selector: Strategy-specific recovered-reference
            selector.
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Dict[str, Paper]: Dictionary of paper_id -> Paper objects
        """
        # Scope in-memory references to one collection request so stale entries do
        # not leak across caller boundaries when builders are reused.
        self.reference_cache.clear()
        self._reference_source_unavailable = False
        self.seed_relations = {}
        self.candidate_source_status = {}
        self.candidate_source_results = ()
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
                local_lookup=reference_metadata_lookup,
                reference_selector=reference_selector,
            )
            source_results.extend(reference_results)
            reference_ids = self._ingest_relation_batch(
                papers,
                seed,
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
                self.reference_cache[seed.paper_id] = []
                seed.references = []

        # Step 3: Fetch citations (newer papers)
        remaining = self.max_papers - len(papers)
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
                papers,
                seed,
                list(citation_result.papers),
                progress_enabled=show_progress,
                progress_description="Downloading citations",
            )
            self._record_seed_relations(citation_ids, "cites_seed")

        self.candidate_source_status = {
            result.source: result.state.value for result in source_results
        }
        self.candidate_source_results = tuple(source_results)
        if validate_source_availability:
            require_available_candidate_source(
                source_results,
                context=f"citation acquisition for {seed.paper_id}",
            )

        # Candidate discovery is required acquisition. Complete every requested
        # source before optional reference enrichment can spend the shared S2
        # recovery budget.
        if hydrate_references:
            self.hydrate_collected_references(papers, seed)

        reference_lists = sum(
            bool(references) for references in self.reference_cache.values()
        )
        summary = f"Collected {len(papers)} papers"
        logger.debug(
            "Collected-paper reference hydration: %d of %d papers have reference lists.",
            reference_lists,
            len(papers),
        )
        self._abstract_index.build(papers)
        self._set_collection_summary(summary)

        return papers

    def build_graph(self, seed_id: str, **kwargs: Any) -> tuple[nx.Graph, str]:
        """Build citation graph and persist seed-relation metadata.

        :param str seed_id: Seed paper identifier.
        :param Any kwargs: Strategy-specific options forwarded to parent build.
        :return Tuple[nx.Graph, str]: Built graph and canonical seed identifier.
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
            with_references_weights=(0.40, 0.20, 0.00, 0.40),
            without_references_weights=(0.65, 0.20, 0.15, 0.00),
        )
