"""
Citation-based graph building strategy.

This strategy builds similarity graphs using citation relationships,
bibliographic coupling (shared references), and topical similarity.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import networkx as nx

from citemesh._runtime import stderr_isatty
from citemesh.core import Paper
from citemesh.progress import progress_iterator
from citemesh.services import get_client
from citemesh.strategies.base import (
    GraphBuilderStrategy,
    build_capped_undirected_graph,
)
from citemesh.strategies.candidates import (
    CandidateSourceResult,
    IdentityRegistry,
    fetch_candidate_source,
    merge_seed_relation,
    reconcile_paper_identity,
    register_aliases,
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
        max_papers: int = 40,
        max_citations: int = 25,
        max_references: int = 25,
        similarity_threshold: float = 0.2,
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
        self._identity_aliases = IdentityRegistry()
        self._abstract_index = AbstractSimilarityIndex()
        self._reference_source_unavailable = False

    def _ensure_paper_references(self, paper: Paper) -> None:
        """Hydrate reference IDs for a paper without clobbering existing payload.

        :param Paper paper: Paper record to hydrate in-place.
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
                cached_refs = self.client.get_cached_reference_ids(paper_id)
                if cached_refs is not None:
                    self.reference_cache[paper_id] = cached_refs
                    paper.references = list(cached_refs)
            return

        from citemesh.services import SemanticScholarUnavailableError

        try:
            paper.references = self._get_references(paper_id)
        except SemanticScholarUnavailableError as exc:
            self._reference_source_unavailable = True
            logger.warning(
                "Reference IDs unavailable for related paper %s; continuing "
                "without further reference hydration for this collection: %s",
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
        """Add related papers and hydrate references while respecting graph limits.

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
            reconciliation = reconcile_paper_identity(
                self._identity_aliases, seed, papers, paper
            )
            if reconciliation.seed_matched:
                collapsed_ids = set(reconciliation.collapsed_ids)
                for paper_id in reconciliation.collapsed_ids:
                    cached_references = self.reference_cache.pop(paper_id, None)
                    if cached_references is not None:
                        self.reference_cache.setdefault(
                            str(seed.paper_id), cached_references
                        )
                    self.seed_relations.pop(paper_id, None)
                processed_ids[:] = [
                    paper_id
                    for paper_id in processed_ids
                    if paper_id not in collapsed_ids
                ]
                continue

            canonical_id = reconciliation.canonical_id
            if canonical_id is not None:
                collapsed_ids = set(reconciliation.collapsed_ids)
                for paper_id in reconciliation.collapsed_ids:
                    cached_references = self.reference_cache.pop(paper_id, None)
                    if cached_references is not None:
                        self.reference_cache.setdefault(canonical_id, cached_references)
                    merged_relation = merge_seed_relation(
                        self.seed_relations.get(canonical_id, ""),
                        self.seed_relations.pop(paper_id, ""),
                    )
                    if merged_relation:
                        self.seed_relations[canonical_id] = merged_relation
                self._ensure_paper_references(papers[canonical_id])
                processed_ids[:] = [
                    canonical_id if paper_id in collapsed_ids else paper_id
                    for paper_id in processed_ids
                ]
                processed_ids.append(canonical_id)
                continue

            if len(papers) >= self.max_papers:
                continue

            papers[raw_paper_id] = paper
            register_aliases(self._identity_aliases, raw_paper_id, paper)
            self._ensure_paper_references(paper)
            processed_ids.append(raw_paper_id)

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

    @scope_candidate_collection
    def collect_papers(
        self,
        seed_id: str,
        *,
        validate_source_availability: bool = True,
        **kwargs: Any,
    ) -> dict[str, Paper]:
        """
        Collect papers via citations and references.

        :param str seed_id: Seed paper identifier
        :param bool validate_source_availability: Whether to fail when every
            requested relation source is unavailable.
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
        self._identity_aliases = IdentityRegistry()
        papers = {}
        source_results: list[CandidateSourceResult] = []

        # Step 1: Fetch seed paper
        logger.info(f"Fetching seed paper: {seed_id}")
        seed = self.client.get_paper(
            seed_id,
            raise_on_unavailable=True,
        )

        if not seed:
            raise ValueError(
                f"Seed paper not found: {seed_id} (Semantic Scholar does not "
                "know this identifier; check the DOI/arXiv/S2 ID)."
            )

        seed.is_seed = True
        papers[seed.paper_id] = seed
        self.seed_relations[seed.paper_id] = "seed"
        register_aliases(self._identity_aliases, seed.paper_id, seed)

        self._ensure_paper_references(seed)

        logger.info("Seed: %s", seed.title)

        # Step 2: Fetch references (older papers)
        progress_enabled = stderr_isatty()
        if self.max_references > 0:
            logger.info(f"Fetching up to {self.max_references} references...")
            reference_result = fetch_candidate_source(
                "references",
                lambda: self.client.get_paper_references(
                    seed.paper_id,
                    limit=self.max_references,
                    raise_on_unavailable=True,
                ),
            )
            source_results.append(reference_result)
            reference_ids = self._ingest_relation_batch(
                papers,
                seed,
                list(reference_result.papers),
                progress_enabled=progress_enabled,
                progress_description="Downloading references",
            )
            self._record_seed_relations(reference_ids, "referenced_by_seed")

        # Step 3: Fetch citations (newer papers)
        remaining = self.max_papers - len(papers)
        if remaining > 0 and self.max_citations > 0:
            logger.info(
                f"Fetching up to {min(remaining, self.max_citations)} citations..."
            )
            citation_result = fetch_candidate_source(
                "citations",
                lambda: self.client.get_paper_citations(
                    seed.paper_id,
                    limit=min(remaining, self.max_citations),
                    raise_on_unavailable=True,
                ),
            )
            source_results.append(citation_result)
            citation_ids = self._ingest_relation_batch(
                papers,
                seed,
                list(citation_result.papers),
                progress_enabled=progress_enabled,
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

        reference_lists = len(self.reference_cache)
        summary = (
            f"Collected {len(papers)} papers ({reference_lists} with reference lists)"
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
        filtered_graph = build_capped_undirected_graph(graph, 3, seed_id=actual_seed_id)
        logger.info(
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
