"""
Citation-based graph building strategy.

This strategy builds similarity graphs using citation relationships,
bibliographic coupling (shared references), and co-citation analysis.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import networkx as nx
from tqdm.auto import tqdm

from citemesh.core import Paper
from citemesh.services import get_client
from citemesh.similarity import AbstractSimilarityIndex
from citemesh.strategies.base import GraphBuilderStrategy
from citemesh.strategies.similarity import compute_indexed_similarity_score

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

    def __init__(
        self,
        max_papers: int = 40,
        max_citations: int = 25,
        max_references: int = 25,
        similarity_threshold: float = 0.2,
        fetch_references: bool = True,
        refresh_reference_cache: bool = False,
        client: Optional[SemanticScholarClient] = None,
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
        self.reference_cache: Dict[str, list] = {}  # Cache reference lists
        self.seed_relations: Dict[str, str] = {}
        self._abstract_index = AbstractSimilarityIndex()

    @staticmethod
    def _merge_relation_paper(existing: Paper, incoming: Paper) -> Paper:
        """Merge supplemental relation payloads without replacing canonical objects.

        :param Paper existing: Existing paper object retained in the collection map.
        :param Paper incoming: Newly observed payload for the same paper ID.
        :return Paper: Mutated ``existing`` paper instance.
        """
        if (
            (not existing.title or existing.title == "Unknown")
            and incoming.title
            and incoming.title != "Unknown"
        ):
            existing.title = incoming.title
        if (not existing.abstract) and incoming.abstract:
            existing.abstract = incoming.abstract
        if existing.year is None and incoming.year is not None:
            existing.year = incoming.year
        if (not existing.authors) and incoming.authors:
            existing.authors = incoming.authors
        if existing.citation_count <= 0 and incoming.citation_count > 0:
            existing.citation_count = incoming.citation_count
        if (not existing.venue) and incoming.venue:
            existing.venue = incoming.venue
        if (not existing.arxiv_id) and incoming.arxiv_id:
            existing.arxiv_id = incoming.arxiv_id
        if (not existing.doi) and incoming.doi:
            existing.doi = incoming.doi
        if (not existing.categories) and incoming.categories:
            existing.categories = incoming.categories

        if incoming.references:
            if not existing.references:
                existing.references = [
                    str(ref_id).strip()
                    for ref_id in incoming.references
                    if str(ref_id).strip()
                ]
            else:
                existing_refs = [ref for ref in existing.references if str(ref).strip()]
                seen = set(existing_refs)
                for ref_id in incoming.references:
                    normalized_ref_id = str(ref_id).strip()
                    if not normalized_ref_id or normalized_ref_id in seen:
                        continue
                    existing_refs.append(normalized_ref_id)
                    seen.add(normalized_ref_id)
                existing.references = existing_refs

        existing.is_seed = bool(existing.is_seed or incoming.is_seed)
        return existing

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

        paper.references = self._get_references(paper_id)

    def _ingest_relation_batch(
        self,
        papers: Dict[str, Paper],
        relation_records: list[Paper],
        progress_enabled: bool,
        progress_description: str,
    ) -> list[str]:
        """Add related papers and hydrate references while respecting graph limits.

        :param Dict[str, Paper] papers: Collected paper mapping updated in-place.
        :param list[Paper] relation_records: Reference/citation papers from the API.
        :param bool progress_enabled: Whether to wrap records with ``tqdm``.
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
                progress_bar = tqdm(
                    relation_records,
                    desc=progress_description,
                    unit="papers",
                    dynamic_ncols=True,
                )
                relation_iterator = progress_bar
            else:
                relation_iterator = relation_records
        else:
            relation_iterator = []

        for paper in relation_iterator:
            normalized_paper_id = str(paper.paper_id).strip()
            if not normalized_paper_id:
                continue
            processed_ids.append(normalized_paper_id)

            existing = papers.get(normalized_paper_id)
            if existing is not None:
                self._merge_relation_paper(existing, paper)
                self._ensure_paper_references(existing)
                continue

            if len(papers) >= self.max_papers:
                continue

            papers[normalized_paper_id] = paper
            self._ensure_paper_references(paper)

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
            existing = self.seed_relations.get(normalized_id)
            if existing is None:
                self.seed_relations[normalized_id] = relation
                continue
            if existing == "seed" or existing == relation or existing == "overlap":
                continue
            self.seed_relations[normalized_id] = "overlap"

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

    def collect_papers(self, seed_id: str, **kwargs: Any) -> Dict[str, Paper]:
        """
        Collect papers via citations and references.

        :param str seed_id: Seed paper identifier
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Dict[str, Paper]: Dictionary of paper_id -> Paper objects
        """
        # Scope in-memory references to one collection request so stale entries do
        # not leak across caller boundaries when builders are reused.
        self.reference_cache.clear()
        self.seed_relations = {}
        papers = {}

        # Step 1: Fetch seed paper
        logger.info(f"Fetching seed paper: {seed_id}")
        seed = self.client.get_paper(seed_id, fetch_references=self.fetch_references)

        if not seed:
            raise ValueError(f"Seed paper not found: {seed_id}")

        seed.is_seed = True
        papers[seed.paper_id] = seed
        self.seed_relations[seed.paper_id] = "seed"

        # Store seed references in cache
        if self.fetch_references and seed.references:
            self.reference_cache[seed.paper_id] = seed.references

        logger.info("Seed: %s", seed.title)

        # Step 2: Fetch references (older papers)
        logger.info(f"Fetching up to {self.max_references} references...")
        references = self.client.get_paper_references(
            seed.paper_id, limit=self.max_references
        )
        progress_enabled = sys.stderr.isatty()
        reference_ids = self._ingest_relation_batch(
            papers,
            references,
            progress_enabled=progress_enabled,
            progress_description="Downloading references",
        )
        self._record_seed_relations(reference_ids, "referenced_by_seed")

        # Step 3: Fetch citations (newer papers)
        remaining = self.max_papers - len(papers)
        if remaining > 0:
            logger.info(
                f"Fetching up to {min(remaining, self.max_citations)} citations..."
            )
            citations = self.client.get_paper_citations(
                seed.paper_id, limit=min(remaining, self.max_citations)
            )
            citation_ids = self._ingest_relation_batch(
                papers,
                citations,
                progress_enabled=progress_enabled,
                progress_description="Downloading citations",
            )
            self._record_seed_relations(citation_ids, "cites_seed")

        reference_lists = len(self.reference_cache)
        summary = (
            f"Collected {len(papers)} papers ({reference_lists} with reference lists)"
        )
        self._abstract_index.build(papers)
        self._set_collection_summary(summary)

        return papers

    def build_graph(self, seed_id: str, **kwargs: Any) -> Tuple[nx.Graph, str]:
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
        return graph, actual_seed_id

    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        """
        Compute similarity using temporal, citation, and bibliographic factors.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :return float: Similarity score (0.0 to 1.0)
        """
        return compute_indexed_similarity_score(
            paper1,
            paper2,
            abstract_index=self._abstract_index,
            temporal_similarity_fn=self.temporal_similarity,
            citation_similarity_fn=self.citation_similarity,
            bibliographic_coupling_fn=self.bibliographic_coupling,
            fetch_references=self.fetch_references,
            with_references_weights=(0.40, 0.20, 0.00, 0.40),
            without_references_weights=(0.65, 0.20, 0.15, 0.00),
        )
