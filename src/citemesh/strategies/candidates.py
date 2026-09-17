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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
)

from citemesh.core import Paper
from citemesh.core.choices import SEMANTIC_SOURCE_CHOICES as SEMANTIC_SOURCE_CHOICES
from citemesh.core.paper_ids import (
    external_ids_from_canonical_paper_id,
    is_local_corpus_paper_id,
    normalize_paper_id,
    recognize_arxiv_identifier,
)

if TYPE_CHECKING:
    from citemesh.services.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_POOL_SIZE = 400
QUERY_SEED_SEARCH_LIMIT = 20
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_S2_PATTERN = re.compile(r"^(?:s2:)?[0-9a-f]{40}$", re.IGNORECASE)


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


def scope_candidate_collection(
    collector: Callable[..., Any],
) -> Callable[..., Any]:
    """Scope a strategy collection to capability-specific outage budgets.

    :param Callable[..., Any] collector: Strategy collection method to wrap.
    :return Callable[..., Any]: Collector that shares a client scope with nested
        strategy and candidate-pool calls.
    """

    @wraps(collector)
    def wrapped(builder: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the collector within its client's discovery-operation scope.

        :param Any builder: Strategy instance with a Semantic Scholar client.
        :param Any args: Positional arguments for ``collector``.
        :param Any kwargs: Keyword arguments for ``collector``.
        :return Any: Result returned by ``collector``.
        """
        with builder.client.candidate_operation_scope():
            return collector(builder, *args, **kwargs)

    return wrapped


def _normalized_doi(raw_identifier: object) -> str:
    """Return a normalized DOI token or an empty string.

    :param object raw_identifier: DOI field or primary identifier candidate.
    :return str: Lowercase DOI without a ``doi:`` prefix, or ``""``.
    """
    value = str(raw_identifier or "").strip()
    if not value:
        return ""
    try:
        normalized = normalize_paper_id(value)
    except ValueError:
        return ""
    return normalized.lower() if _DOI_PATTERN.fullmatch(normalized) else ""


def provider_lookup_identifier(paper_id: str, paper: Paper) -> str | None:
    """Return an identifier that is safe to send to Semantic Scholar.

    Locally hydrated corpus rows may use arbitrary source-primary keys. Explicit
    arXiv/DOI metadata remains a valid provider lookup route, while opaque local
    keys and synthetic query IDs must remain inside CiteMesh.

    :param str paper_id: Graph/cache primary identifier.
    :param Paper paper: Paper metadata carrying local provenance and aliases.
    :return Optional[str]: Provider-compatible identifier, or ``None``.
    """
    normalized_id = str(paper_id).strip()
    if normalized_id.startswith("query:"):
        return None
    if not paper.is_local_corpus:
        if is_local_corpus_paper_id(normalized_id):
            return None
        return normalized_id or None

    arxiv_id = recognize_arxiv_identifier(paper.arxiv_id, allow_bare=True)
    if arxiv_id:
        return arxiv_id
    doi = _normalized_doi(paper.doi)
    if doi:
        return doi
    primary_arxiv = recognize_arxiv_identifier(normalized_id, allow_bare=True)
    if primary_arxiv:
        return primary_arxiv
    if _S2_PATTERN.fullmatch(normalized_id):
        return normalized_id
    primary_doi = _normalized_doi(normalized_id)
    return primary_doi or None


def _external_identifiers(paper: Paper) -> dict[str, str]:
    """Return explicit normalized arXiv and DOI identifiers for one paper.

    :param Paper paper: Paper metadata supplying canonical and field identifiers.
    :return dict[str, str]: Valid, internally consistent identifiers by namespace.
    """
    primary_arxiv, primary_doi = external_ids_from_canonical_paper_id(paper.paper_id)
    primary_arxiv_id = recognize_arxiv_identifier(primary_arxiv, allow_bare=True)
    field_arxiv_id = recognize_arxiv_identifier(paper.arxiv_id, allow_bare=True)
    primary_doi_id = _normalized_doi(primary_doi)
    field_doi_id = _normalized_doi(paper.doi)
    if (primary_arxiv_id and field_arxiv_id and primary_arxiv_id != field_arxiv_id) or (
        primary_doi_id and field_doi_id and primary_doi_id != field_doi_id
    ):
        return {}
    arxiv_id = field_arxiv_id or primary_arxiv_id
    doi = field_doi_id or primary_doi_id
    return {
        namespace: identifier
        for namespace, identifier in (("arxiv", arxiv_id), ("doi", doi))
        if identifier
    }


def _candidate_match_namespaces(left: Paper, right: Paper) -> tuple[str, ...]:
    """Return shared, conflict-free identifier namespaces for two records.

    Matching requires a shared normalized arXiv or DOI identifier. Any namespace
    supplied by both records must agree, so partial metadata cannot bridge two
    contradictory records. Two local-corpus rows remain separate because their
    source-primary keys can represent distinct corpus entries.

    :param Paper left: First candidate record.
    :param Paper right: Second candidate record.
    :return tuple[str, ...]: Matching namespaces, or empty when records differ.
    """
    if left.is_local_corpus and right.is_local_corpus:
        return ()
    left_ids = _external_identifiers(left)
    right_ids = _external_identifiers(right)
    shared_namespaces = left_ids.keys() & right_ids.keys()
    if not shared_namespaces or any(
        left_ids[namespace] != right_ids[namespace] for namespace in shared_namespaces
    ):
        return ()
    return tuple(sorted(shared_namespaces))


def candidate_records_match(left: Paper, right: Paper) -> bool:
    """Return whether two candidate records explicitly identify one work.

    Matching requires a shared normalized arXiv or DOI identifier. Any namespace
    supplied by both records must agree, so partial metadata cannot bridge two
    contradictory records. Two local-corpus rows remain separate because their
    source-primary keys can represent distinct corpus entries.

    :param Paper left: First candidate record.
    :param Paper right: Second candidate record.
    :return bool: Whether both records explicitly identify the same work.
    """
    return bool(_candidate_match_namespaces(left, right))


def corpus_matches_s2(corpus_paper: Paper, s2_paper: Paper) -> bool:
    """Return whether a corpus row and S2 paper explicitly identify one work.

    Matching requires a shared normalized arXiv or DOI identifier. Any namespace
    supplied by both records must agree, so partial metadata cannot bridge two
    contradictory records.

    :param Paper corpus_paper: Locally sourced corpus record.
    :param Paper s2_paper: Semantic Scholar record.
    :return bool: Whether both records explicitly identify the same work.
    """
    if not corpus_paper.is_local_corpus or s2_paper.is_local_corpus:
        return False
    return candidate_records_match(corpus_paper, s2_paper)


def fetch_candidate_source(
    source: str,
    fetch: Callable[[], list[Paper]],
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
        details = "; ".join(
            f"{result.source}: {result.error or 'unavailable'}"
            for result in unavailable
        )
        raise CandidateAcquisitionError(
            f"All requested Semantic Scholar sources were unavailable for {context}: "
            f"{sources}. Details: {details}"
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
        preferred.paper_id != incoming.paper_id
        and not preferred.is_local_corpus
        and not incoming.is_local_corpus
    ):
        matching_namespaces = _candidate_match_namespaces(preferred, incoming)
        if matching_namespaces:
            logger.debug(
                "Merged S2 record %s into %s via shared %s identifier%s",
                incoming.paper_id,
                preferred.paper_id,
                ", ".join(matching_namespaces),
                "" if len(matching_namespaces) == 1 else "s",
            )
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
    for record in (preferred, incoming):
        arxiv_id, doi = external_ids_from_canonical_paper_id(record.paper_id)
        if not preferred.arxiv_id:
            preferred.arxiv_id = record.arxiv_id or arxiv_id
        if not preferred.doi:
            preferred.doi = record.doi or doi
    if (not preferred.categories) and incoming.categories:
        preferred.categories = incoming.categories
    reference_ids: list[str] = []
    seen_references: set[str] = set()
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


def paper_embedding_metadata(paper: Paper) -> dict[str, object]:
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
    papers: dict[str, Paper] = field(default_factory=dict)
    sources: dict[str, set[str]] = field(default_factory=dict)
    seed_relations: dict[str, str] = field(default_factory=dict)
    source_status: dict[str, str] = field(default_factory=dict)

    def add(self, paper: Paper, *, source: str, relation: str) -> None:
        """Add a paper, reconciling exact IDs and unambiguous external IDs.

        :param Paper paper: Candidate paper payload.
        :param str source: Provenance tag (``reference``/``citation``/``recommendation``).
        :param str relation: Seed-relation label for this provenance.
        :return None: Pool state is mutated in place.
        """
        paper_id = str(paper.paper_id).strip()
        if paper.is_seed or not paper_id:
            return
        if paper_id == self.seed.paper_id or candidate_records_match(self.seed, paper):
            merge_paper_metadata(self.seed, paper)
            return
        canonical_id = paper_id
        existing = self.papers.get(canonical_id)
        if existing is None:
            matching_ids = [
                candidate_id
                for candidate_id, candidate in self.papers.items()
                if candidate_records_match(candidate, paper)
            ]
            if len(matching_ids) == 1:
                canonical_id = matching_ids[0]
                existing = self.papers[canonical_id]
        if existing is None:
            self.papers[canonical_id] = paper
        else:
            merge_paper_metadata(existing, paper)
        self.sources.setdefault(canonical_id, set()).add(source)
        merged_relation = merge_seed_relation(
            self.seed_relations.get(canonical_id, ""), relation
        )
        if merged_relation:
            self.seed_relations[canonical_id] = merged_relation


def fetch_candidate_pool(
    client: SemanticScholarClient,
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
    with client.candidate_operation_scope():
        return _fetch_candidate_pool(
            client,
            seed_paper,
            max_references=max_references,
            max_citations=max_citations,
            max_recommendations=max_recommendations,
        )


def _fetch_candidate_pool(
    client: SemanticScholarClient,
    seed_paper: Paper,
    *,
    max_references: int = 0,
    max_citations: int = 0,
    max_recommendations: int = 0,
) -> CandidatePool:
    """Fetch candidates while an outer operation scope is active.

    :param SemanticScholarClient client: Semantic Scholar client.
    :param Paper seed_paper: Seed paper (S2-backed or free-text query seed).
    :param int max_references: Maximum seed references to fetch (0 disables).
    :param int max_citations: Maximum citing papers to fetch (0 disables).
    :param int max_recommendations: Maximum recommendations to fetch (0 disables).
    :return CandidatePool: Deduplicated candidate pool with provenance tags.
    """
    pool = CandidatePool(seed=seed_paper)
    source_results: list[CandidateSourceResult] = []
    seed_id = str(seed_paper.paper_id)
    is_query_seed = seed_id.startswith("query:")
    query_candidate_budget = (
        max_references + max_citations + max_recommendations if is_query_seed else 0
    )
    recommendation_limit = max_recommendations

    if is_query_seed:
        query_text = (seed_paper.title or "").strip() or seed_id
        search_limit = min(QUERY_SEED_SEARCH_LIMIT, query_candidate_budget)
        if search_limit > 0:
            search_result = fetch_candidate_source(
                "search",
                lambda: client.search_papers(
                    query_text,
                    limit=search_limit,
                    raise_on_unavailable=True,
                ),
            )
            source_results.append(search_result)
            for paper in search_result.papers:
                pool.add(paper, source="recommendation", relation="semantic_only")
        anchor_ids = list(pool.papers)[:1]
        recommendation_limit = min(
            max_recommendations,
            max(0, query_candidate_budget - len(pool.papers)),
        )
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

    if recommendation_limit > 0 and anchor_ids:
        recommendation_result = fetch_candidate_source(
            "recommendations",
            lambda: client.get_recommended_papers(
                anchor_ids[0],
                limit=recommendation_limit,
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
