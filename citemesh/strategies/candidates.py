"""
Candidate pool collection and paper-identity helpers.

Provides the corpus-free ("candidates") semantic source: candidates come from
Semantic Scholar neighbors of the seed (references, citations, and
recommendations) instead of a locally hydrated arXiv corpus. The identity
alias/merge helpers here are shared by hybrid and embedding strategies so both
deduplicate equivalent papers identically.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Set

from citemesh.core import Paper
from citemesh.paper_ids import paper_identifier_aliases

if TYPE_CHECKING:
    from citemesh.services.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)

SEMANTIC_SOURCE_CHOICES = ("candidates", "arxiv-corpus")
DEFAULT_CANDIDATE_POOL_SIZE = 400
QUERY_SEED_SEARCH_LIMIT = 20


class CandidateAcquisitionError(RuntimeError):
    """No requested candidate source could be evaluated."""


class CandidateSourceState(str, Enum):
    """Outcome of one attempted candidate source."""

    COMPLETE = "complete"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CandidateSourceResult:
    """Papers and availability state returned by one candidate source."""

    source: str
    state: CandidateSourceState
    papers: tuple[Paper, ...] = ()
    error: str = ""


def fetch_candidate_source(
    source: str,
    fetch: Callable[[], List[Paper]],
) -> CandidateSourceResult:
    """Fetch one source without confusing an outage with valid empty evidence.

    :param str source: Stable source name used in metadata.
    :param Callable[[], List[Paper]] fetch: Strict service fetch operation.
    :return CandidateSourceResult: Tri-state source result.
    :raises Exception: Unexpected programming or payload errors from ``fetch``.
    """
    from citemesh.services import SemanticScholarUnavailableError

    try:
        papers = tuple(fetch())
    except SemanticScholarUnavailableError as exc:
        return CandidateSourceResult(
            source=source,
            state=CandidateSourceState.UNAVAILABLE,
            error=str(exc),
        )
    return CandidateSourceResult(
        source=source,
        state=(CandidateSourceState.COMPLETE if papers else CandidateSourceState.EMPTY),
        papers=papers,
    )


def require_available_candidate_source(
    results: Sequence[CandidateSourceResult],
    *,
    context: str,
) -> None:
    """Require at least one attempted source to have completed, even if empty.

    Partial source outages remain usable but are surfaced once in logs. When every
    attempted source is unavailable, callers must not publish a normal graph.

    :param Sequence[CandidateSourceResult] results: Attempted source results.
    :param str context: Human-readable acquisition context for diagnostics.
    :return None: Returns after validating the result set.
    :raises CandidateAcquisitionError: If every attempted source was unavailable.
    """
    if not results:
        return
    unavailable = [
        result for result in results if result.state is CandidateSourceState.UNAVAILABLE
    ]
    if len(unavailable) == len(results):
        sources = ", ".join(result.source for result in unavailable)
        raise CandidateAcquisitionError(
            f"All requested Semantic Scholar sources were unavailable for {context}: "
            f"{sources}."
        )
    if unavailable:
        details = "; ".join(
            f"{result.source}: {result.error or 'unavailable'}"
            for result in unavailable
        )
        logger.warning(
            "Continuing %s with partial Semantic Scholar evidence (%s).",
            context,
            details,
        )


def normalize_identity_text(raw_text: str) -> str:
    """Normalize free-form text for deterministic paper identity matching.

    :param str raw_text: Raw user/content text.
    :return str: Lowercased alphanumeric text with compact spacing.
    """
    compact = re.sub(r"[^0-9a-z]+", " ", str(raw_text).strip().lower())
    return " ".join(compact.split())


def paper_identity_aliases(paper: Paper) -> List[str]:
    """Return deterministic alias keys used to deduplicate equivalent papers.

    :param Paper paper: Paper candidate to alias.
    :return List[str]: Stable sorted alias keys.
    """
    aliases: Set[str] = set()
    for identifier in paper_identifier_aliases(
        paper_id=paper.paper_id,
        arxiv_id=paper.arxiv_id,
        doi=paper.doi,
    ):
        aliases.add(f"id:{identifier.lower()}")

    normalized_title = normalize_identity_text(paper.title or "")
    if normalized_title:
        year_token = (
            str(int(paper.year))
            if isinstance(paper.year, int) and paper.year > 0
            else "n.d."
        )
        aliases.add(f"meta:{normalized_title}|{year_token}")
        normalized_abstract = normalize_identity_text(paper.abstract or "")
        if normalized_abstract:
            aliases.add(f"meta:{normalized_title}|abs:{normalized_abstract[:256]}")
        author_tokens = [
            normalize_identity_text(author.name)
            for author in paper.authors[:3]
            if getattr(author, "name", None)
        ]
        compact_authors = "|".join(token for token in author_tokens if token)
        if compact_authors:
            aliases.add(f"meta:{normalized_title}|{year_token}|{compact_authors}")

    return sorted(aliases)


def resolve_aliases(aliases: Dict[str, str], paper: Paper) -> List[str]:
    """Resolve every canonical paper ID matched by an incoming payload.

    A record may bridge two previously distinct identifier classes (for
    example, one payload supplies an arXiv ID and a later one supplies both
    that arXiv ID and a DOI). Callers reconcile all returned classes before
    registering the incoming aliases.

    :param Dict[str, str] aliases: Alias-to-canonical map.
    :param Paper paper: Incoming paper payload.
    :return List[str]: Distinct matching canonical IDs in alias-key order.
    """
    matches: List[str] = []
    seen: Set[str] = set()
    for alias in paper_identity_aliases(paper):
        canonical_id = aliases.get(alias)
        if canonical_id is None or canonical_id in seen:
            continue
        seen.add(canonical_id)
        matches.append(canonical_id)
    return matches


def register_aliases(aliases: Dict[str, str], canonical_id: str, paper: Paper) -> None:
    """Register identity aliases for a canonical paper ID.

    :param Dict[str, str] aliases: Alias-to-canonical map to mutate.
    :param str canonical_id: Canonical paper identifier.
    :param Paper paper: Paper payload providing alias candidates.
    :return None: Alias map is mutated in place.
    """
    for alias in paper_identity_aliases(paper):
        aliases.setdefault(alias, canonical_id)


def repoint_aliases(
    aliases: Dict[str, str], canonical_id: str, replaced_ids: Set[str]
) -> None:
    """Point aliases owned by reconciled records at their surviving paper.

    :param Dict[str, str] aliases: Alias-to-canonical map to mutate.
    :param str canonical_id: Surviving canonical paper ID.
    :param Set[str] replaced_ids: Canonical IDs collapsed into the survivor.
    :return None: Alias map is mutated in place.
    """
    for alias, mapped_id in aliases.items():
        if mapped_id in replaced_ids:
            aliases[alias] = canonical_id


def merge_seed_relation(existing: str, incoming: str) -> str:
    """Merge two seed-relation labels conservatively.

    :param str existing: Existing relation label.
    :param str incoming: Incoming relation label.
    :return str: Merged relation label.
    """
    normalized_existing = str(existing or "").strip().lower()
    normalized_incoming = str(incoming or "").strip().lower()
    if not normalized_existing:
        return normalized_incoming
    if (
        not normalized_incoming
        or normalized_existing == normalized_incoming
        or normalized_existing == "seed"
    ):
        return normalized_existing
    if normalized_existing == "overlap" or normalized_incoming == "overlap":
        return "overlap"
    if {
        normalized_existing,
        normalized_incoming,
    } == {"referenced_by_seed", "cites_seed"}:
        return "overlap"
    if normalized_existing == "semantic_only":
        return normalized_incoming
    if normalized_incoming == "semantic_only":
        return normalized_existing
    return normalized_existing


def merge_paper_metadata(preferred: Paper, incoming: Paper) -> Paper:
    """Merge supplemental metadata from an alternate source into ``preferred``.

    :param Paper preferred: Canonical paper record to retain.
    :param Paper incoming: Supplemental paper record to merge.
    :return Paper: ``preferred`` with missing metadata hydrated.
    """
    if (
        (not preferred.title or preferred.title == "Unknown")
        and incoming.title
        and incoming.title != "Unknown"
    ):
        preferred.title = incoming.title
    if not preferred.abstract and incoming.abstract:
        preferred.abstract = incoming.abstract
    if preferred.year is None and incoming.year is not None:
        preferred.year = incoming.year
    if (not preferred.authors) and incoming.authors:
        preferred.authors = incoming.authors
    preferred.citation_count = max(
        preferred.citation_count,
        incoming.citation_count,
    )
    if (not preferred.venue) and incoming.venue:
        preferred.venue = incoming.venue
    if (not preferred.arxiv_id) and incoming.arxiv_id:
        preferred.arxiv_id = incoming.arxiv_id
    if (not preferred.doi) and incoming.doi:
        preferred.doi = incoming.doi
    if (not preferred.categories) and incoming.categories:
        preferred.categories = incoming.categories
    reference_ids: List[str] = []
    seen_references: Set[str] = set()
    for reference_source in (preferred.references, incoming.references):
        for reference_id in reference_source:
            normalized_reference_id = str(reference_id).strip()
            if (
                not normalized_reference_id
                or normalized_reference_id in seen_references
            ):
                continue
            seen_references.add(normalized_reference_id)
            reference_ids.append(normalized_reference_id)
    preferred.references = reference_ids
    preferred.is_seed = bool(preferred.is_seed or incoming.is_seed)
    return preferred


@dataclass(frozen=True)
class IdentityReconciliation:
    """Result of reconciling one paper against known identity classes."""

    canonical_id: Optional[str]
    collapsed_ids: tuple[str, ...] = ()
    seed_matched: bool = False


def reconcile_paper_identity(
    aliases: Dict[str, str],
    seed: Paper,
    papers: Dict[str, Paper],
    incoming: Paper,
) -> IdentityReconciliation:
    """Merge an incoming payload into its seed or candidate identity class.

    The seed always survives. Otherwise the first candidate insertion wins,
    keeping graph order stable. Callers remain responsible for folding any
    sidecar state associated with ``collapsed_ids``.

    :param Dict[str, str] aliases: Alias-to-canonical map to update.
    :param Paper seed: Canonical seed paper.
    :param Dict[str, Paper] papers: Candidate mapping to reconcile in place.
    :param Paper incoming: Newly observed paper payload.
    :return IdentityReconciliation: Survivor and collapsed candidate IDs.
    """
    matched_ids = resolve_aliases(aliases, incoming)
    seed_id = str(seed.paper_id)
    if seed_id in matched_ids:
        collapsed_ids = tuple(
            paper_id
            for paper_id in papers
            if paper_id != seed_id and paper_id in matched_ids
        )
        for paper_id in collapsed_ids:
            merge_paper_metadata(seed, papers.pop(paper_id))
        merge_paper_metadata(seed, incoming)
        repoint_aliases(aliases, seed_id, set(collapsed_ids) | {seed_id})
        register_aliases(aliases, seed_id, incoming)
        return IdentityReconciliation(seed_id, collapsed_ids, True)

    matched_candidates = [
        paper_id
        for paper_id in papers
        if paper_id != seed_id and paper_id in matched_ids
    ]
    if not matched_candidates:
        return IdentityReconciliation(None)

    canonical_id = matched_candidates[0]
    collapsed_ids = tuple(matched_candidates[1:])
    for paper_id in collapsed_ids:
        merge_paper_metadata(papers[canonical_id], papers.pop(paper_id))
    merge_paper_metadata(papers[canonical_id], incoming)
    repoint_aliases(aliases, canonical_id, set(matched_candidates))
    register_aliases(aliases, canonical_id, incoming)
    return IdentityReconciliation(canonical_id, collapsed_ids)


def paper_embedding_metadata(paper: Paper) -> Dict[str, object]:
    """Build embedding-cache metadata payload for a paper.

    :param Paper paper: Paper to normalize.
    :return Dict[str, object]: Metadata payload accepted by embedding cache.
    """
    return {
        "title": paper.title or "",
        "abstract": paper.abstract or "",
        "year": paper.year,
        "authors": [author.name for author in paper.authors],
        "venue": paper.venue or "",
        "arxiv_id": paper.arxiv_id or "",
        "doi": paper.doi or "",
        "categories": list(paper.categories or []),
    }


@dataclass
class CandidatePool:
    """Deduplicated candidate papers fetched from Semantic Scholar."""

    seed: Paper
    papers: Dict[str, Paper] = field(default_factory=dict)
    sources: Dict[str, Set[str]] = field(default_factory=dict)
    seed_relations: Dict[str, str] = field(default_factory=dict)
    source_status: Dict[str, str] = field(default_factory=dict)

    def add(self, paper: Paper, *, source: str, relation: str) -> None:
        """Add a paper to the pool, merging duplicates by identity aliases.

        :param Paper paper: Candidate paper payload.
        :param str source: Provenance tag (``reference``/``citation``/``recommendation``).
        :param str relation: Seed-relation label for this provenance.
        :return None: Pool state is mutated in place.
        """
        if paper.is_seed:
            return
        reconciliation = reconcile_paper_identity(
            self._aliases, self.seed, self.papers, paper
        )
        if reconciliation.seed_matched:
            for paper_id in reconciliation.collapsed_ids:
                self.sources.pop(paper_id, None)
                self.seed_relations.pop(paper_id, None)
            return

        canonical_id = reconciliation.canonical_id
        if canonical_id is not None:
            for paper_id in reconciliation.collapsed_ids:
                self.sources.setdefault(canonical_id, set()).update(
                    self.sources.pop(paper_id, set())
                )
                merged_relation = merge_seed_relation(
                    self.seed_relations.get(canonical_id, ""),
                    self.seed_relations.pop(paper_id, ""),
                )
                if merged_relation:
                    self.seed_relations[canonical_id] = merged_relation
        else:
            canonical_id = str(paper.paper_id)
            self.papers[canonical_id] = paper
        self.sources.setdefault(canonical_id, set()).add(source)
        merged_relation = merge_seed_relation(
            self.seed_relations.get(canonical_id, ""), relation
        )
        if merged_relation:
            self.seed_relations[canonical_id] = merged_relation
        if reconciliation.canonical_id is None:
            register_aliases(self._aliases, canonical_id, paper)

    def __post_init__(self) -> None:
        """Initialize alias map with seed identity."""
        self._aliases: Dict[str, str] = {}
        register_aliases(self._aliases, self.seed.paper_id, self.seed)


def fetch_candidate_pool(
    client: "SemanticScholarClient",
    seed_paper: Paper,
    *,
    max_references: int = 0,
    max_citations: int = 0,
    max_recommendations: int = 0,
) -> CandidatePool:
    """Fetch a deduplicated candidate pool from Semantic Scholar seed neighbors.

    Free-text query seeds (``query:`` IDs) have no S2 neighbors; they are proxied
    through paper search on the query text before recommendation expansion.

    :param SemanticScholarClient client: Semantic Scholar client.
    :param Paper seed_paper: Seed paper (S2-backed or free-text query seed).
    :param int max_references: Maximum seed references to fetch (0 disables).
    :param int max_citations: Maximum citing papers to fetch (0 disables).
    :param int max_recommendations: Maximum recommendations to fetch (0 disables).
    :return CandidatePool: Deduplicated candidate pool with provenance tags.
    """
    pool = CandidatePool(seed=seed_paper)
    source_results: List[CandidateSourceResult] = []
    seed_id = str(seed_paper.paper_id)
    is_query_seed = seed_id.startswith("query:")

    if is_query_seed:
        query_text = (seed_paper.title or "").strip() or seed_id
        search_result = fetch_candidate_source(
            "search",
            lambda: client.search_papers(
                query_text,
                limit=QUERY_SEED_SEARCH_LIMIT,
                raise_on_unavailable=True,
            ),
        )
        source_results.append(search_result)
        for paper in search_result.papers:
            pool.add(paper, source="recommendation", relation="semantic_only")
        anchor_ids = list(pool.papers)[:1]
    else:
        anchor_ids = [seed_id]
        if max_references > 0:
            reference_result = fetch_candidate_source(
                "references",
                lambda: client.get_paper_references(
                    seed_id,
                    limit=max_references,
                    raise_on_unavailable=True,
                ),
            )
            source_results.append(reference_result)
            for paper in reference_result.papers:
                pool.add(paper, source="reference", relation="referenced_by_seed")
        if max_citations > 0:
            citation_result = fetch_candidate_source(
                "citations",
                lambda: client.get_paper_citations(
                    seed_id,
                    limit=max_citations,
                    raise_on_unavailable=True,
                ),
            )
            source_results.append(citation_result)
            for paper in citation_result.papers:
                pool.add(paper, source="citation", relation="cites_seed")

    if max_recommendations > 0 and anchor_ids:
        recommendation_result = fetch_candidate_source(
            "recommendations",
            lambda: client.get_recommended_papers(
                anchor_ids[0],
                limit=max_recommendations,
                raise_on_unavailable=True,
            ),
        )
        source_results.append(recommendation_result)
        for paper in recommendation_result.papers:
            pool.add(paper, source="recommendation", relation="semantic_only")

    pool.source_status = {
        result.source: result.state.value for result in source_results
    }
    require_available_candidate_source(
        source_results,
        context=f"candidate acquisition for {seed_id}",
    )

    logger.debug(
        "Candidate pool for %s: %d papers (refs<=%d cites<=%d recs<=%d).",
        seed_id,
        len(pool.papers),
        max_references,
        max_citations,
        max_recommendations,
    )
    return pool
