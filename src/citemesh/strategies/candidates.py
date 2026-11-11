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
    Concatenate,
    ParamSpec,
    Protocol,
    TypeVar,
)

from citemesh.core import Paper
from citemesh.core.choices import SEMANTIC_SOURCE_CHOICES as SEMANTIC_SOURCE_CHOICES
from citemesh.core.paper_ids import (
    arxiv_identifier_from_doi,
    bibliography_text_key,
    external_ids_from_canonical_paper_id,
    is_local_corpus_paper_id,
    normalize_paper_id,
    paper_identifier_aliases,
    recognize_arxiv_identifier,
)
from citemesh.strategies.similarity import AbstractSimilarityIndex

if TYPE_CHECKING:
    from citemesh.core.paper_ids import BibliographyKey
    from citemesh.data.embedding_cache.models import PaperMetadata
    from citemesh.services.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_POOL_SIZE = 400
QUERY_SEED_SEARCH_LIMIT = 20
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_S2_PATTERN = re.compile(r"^(?:s2:)?[0-9a-f]{40}$", re.IGNORECASE)
ReferenceSelector = Callable[[Paper, Sequence[Paper], int], list[Paper]]
_P = ParamSpec("_P")
_R = TypeVar("_R")
_BuilderT = TypeVar("_BuilderT", bound="_CandidateCollectionOwner")


class _CandidateCollectionOwner(Protocol):
    """Builder interface required by the collection-scope decorator."""

    @property
    def client(self) -> SemanticScholarClient: ...


class CandidateAcquisitionError(RuntimeError):
    """No requested candidate source could be evaluated."""


class CandidateSourceState(str, Enum):
    """Outcome of one attempted candidate source."""

    COMPLETE = "complete"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CandidateSourceResult:
    """Admitted candidates, availability, and reference evidence before admission."""

    source: str
    state: CandidateSourceState
    papers: tuple[Paper, ...] = ()
    error: str | None = ""
    verified_references: tuple[Paper, ...] = ()
    source_ranks: dict[str, int] = field(default_factory=dict)


def scope_candidate_collection(
    collector: Callable[Concatenate[_BuilderT, _P], _R],
) -> Callable[Concatenate[_BuilderT, _P], _R]:
    """Scope a strategy collection to capability-specific outage budgets.

    :param Callable collector: Strategy collection method to wrap.
    :return Callable: Collector that shares a client scope with nested strategy
        and candidate-pool calls.
    """

    @wraps(collector)
    def wrapped(builder: _BuilderT, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Run the collector within its client's discovery-operation scope.

        :param builder: Strategy instance with a Semantic Scholar client.
        :param args: Positional arguments for ``collector``.
        :param kwargs: Keyword arguments for ``collector``.
        :return: Result returned by ``collector``.
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


def _external_identifiers(paper: Paper) -> dict[str, str] | None:
    """Return explicit normalized arXiv and DOI identifiers for one paper.

    An arXiv-issued DOI belongs to the arXiv namespace, independently of
    any publication DOI on the same record.

    :param Paper paper: Paper metadata supplying canonical and field identifiers.
    :return dict[str, str] | None: Identifiers by namespace, or ``None`` on conflict.
    """
    primary_arxiv, primary_doi = external_ids_from_canonical_paper_id(paper.paper_id)
    identifiers: dict[str, str] = {}
    for namespace, identifier in (
        ("arxiv", recognize_arxiv_identifier(primary_arxiv, allow_bare=True)),
        ("arxiv", recognize_arxiv_identifier(paper.arxiv_id, allow_bare=True)),
        ("doi", _normalized_doi(primary_doi)),
        ("doi", _normalized_doi(paper.doi)),
    ):
        if not identifier:
            continue
        identifier = identifier.lower()
        if namespace == "doi" and (
            arxiv_alias := arxiv_identifier_from_doi(identifier)
        ):
            namespace, identifier = "arxiv", arxiv_alias
        if identifiers.setdefault(namespace, identifier) != identifier:
            return None
    return identifiers


def _candidate_match_namespaces(left: Paper, right: Paper) -> tuple[str, ...] | None:
    """Return shared identifier namespaces, distinguishing conflicts from absence.

    Matching requires a shared normalized arXiv or DOI identifier. Any namespace
    supplied by both records must agree, so partial metadata cannot bridge two
    contradictory records. Two local-corpus rows remain separate because their
    source-primary keys can represent distinct corpus entries.

    :param Paper left: First candidate record.
    :param Paper right: Second candidate record.
    :return tuple[str, ...] | None: Matching namespaces, empty without a match,
        or ``None`` when identifiers conflict.
    """
    left_ids = _external_identifiers(left)
    right_ids = _external_identifiers(right)
    if left_ids is None or right_ids is None:
        return None
    shared_namespaces = left_ids.keys() & right_ids.keys()
    if any(
        left_ids[namespace] != right_ids[namespace] for namespace in shared_namespaces
    ):
        return None
    if left.is_local_corpus and right.is_local_corpus:
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


def candidate_identifiers_conflict(left: Paper, right: Paper) -> bool:
    """Detect contradictory arXiv or DOI identifiers, allowing missing aliases.

    :param Paper left: Corpus record being enriched.
    :param Paper right: Provider record returned for its requested identifier.
    :return bool: Whether either record or the pair names different works.
    """
    return _candidate_match_namespaces(left, right) is None


def corpus_publication_matches(left: Paper, right: Paper) -> bool:
    """Match a corpus preprint to a publication with missing cross-identifiers.

    Titles and full first-author names must match after word normalization.
    The publication can be in the preprint's year or the following year.
    Conflicting explicit identifiers remain disqualifying. Callers must require
    a unique candidate match before reconciling records.
    """
    if left.is_local_corpus == right.is_local_corpus:
        return False
    corpus, publication = (left, right) if left.is_local_corpus else (right, left)
    corpus_ids = _external_identifiers(corpus)
    publication_ids = _external_identifiers(publication)
    if (
        not corpus_ids
        or "arxiv" not in corpus_ids
        or not publication_ids
        or "doi" not in publication_ids
        or candidate_identifiers_conflict(corpus, publication)
        or corpus.year is None
        or publication.year not in (corpus.year, corpus.year + 1)
        or not corpus.authors
        or not publication.authors
    ):
        return False
    title = bibliography_text_key(corpus.title)
    author = bibliography_text_key(corpus.authors[0].name)
    return (
        " " in title
        and " " in author
        and len(author.partition(" ")[0]) > 1
        and title == bibliography_text_key(publication.title)
        and author == bibliography_text_key(publication.authors[0].name)
    )


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


@dataclass
class _ReferenceAliasResolver:
    """Request-local bibliography aliases and cross-identifier bindings."""

    aliases: dict[str, Paper] = field(default_factory=dict)

    def remember(self, records: Sequence[Paper]) -> None:
        """Index resolved records by every explicit identifier alias.

        :param Sequence[Paper] records: Resolved metadata.
        :return None: Updates the request-local alias map.
        """
        for paper in records:
            for alias in paper_identifier_aliases(
                paper_id=paper.paper_id, arxiv_id=paper.arxiv_id, doi=paper.doi
            ):
                self.aliases[normalize_paper_id(alias).lower()] = paper

    def resolve(self, entry: Sequence[str]) -> Paper | None:
        """Return the first paper bound to an entry's explicit identifiers.

        :param Sequence[str] entry: Identifier spellings from one bibliography row.
        :return Paper | None: Resolved record, when any spelling is known.
        """
        return next(
            (self.aliases[value] for value in entry if value in self.aliases), None
        )

    def bind_entries(self, entries: Sequence[Sequence[str]]) -> None:
        """Bind every spelling in a resolved bibliography entry to one record.

        :param Sequence[Sequence[str]] entries: Bibliography identifier groups.
        :return None: Adds aliases without replacing earlier evidence.
        """
        for entry in entries:
            paper = self.resolve(entry)
            if paper is not None:
                for value in entry:
                    self.aliases.setdefault(value, paper)


def _enrich_local_reference_records(
    client: SemanticScholarClient,
    recovered: Sequence[Paper],
) -> None:
    """Merge available Semantic Scholar metadata onto local-corpus references.

    :param SemanticScholarClient client: Existing metadata client.
    :param Sequence[Paper] recovered: Materialized bibliography records.
    :return None: Enriches local records in place when matching metadata exists.
    """
    from citemesh.services import (
        SemanticScholarRequestError,
        SemanticScholarUnavailableError,
    )

    local_targets = [
        (paper, lookup_id)
        for paper in recovered
        if paper.is_local_corpus
        and (lookup_id := provider_lookup_identifier(paper.paper_id, paper)) is not None
    ]
    if not local_targets:
        return

    lookup_ids = [lookup_id for _paper, lookup_id in local_targets]
    if client.refresh_paper_cache:
        try:
            s2_records = client.get_papers(lookup_ids, raise_on_unavailable=True)
        except (
            SemanticScholarRequestError,
            SemanticScholarUnavailableError,
        ) as exc:
            logger.debug(
                "S2 metadata enrichment unavailable during arXiv recovery: %s",
                exc,
            )
            s2_records = {}
    else:
        # A local record is already sufficient recovery. Avoid spending the shared
        # retry budget unless the caller explicitly requested refresh.
        s2_records = client.get_cached_papers(lookup_ids)
    for local_paper, lookup_id in local_targets:
        s2_paper = s2_records.get(normalize_paper_id(lookup_id))
        if s2_paper is None:
            s2_paper = next(
                (
                    candidate
                    for candidate in s2_records.values()
                    if candidate_records_match(local_paper, candidate)
                ),
                None,
            )
        if s2_paper is not None and not candidate_identifiers_conflict(
            local_paper, s2_paper
        ):
            merge_paper_metadata(local_paper, s2_paper)


def select_recovered_references_by_score(
    recovered: Sequence[Paper],
    limit: int,
    relevance_score: Callable[[Paper], float],
) -> list[Paper]:
    """Cap recovered references by relevance, impact, then source order.

    :param Sequence[Paper] recovered: Materialized bibliography records.
    :param int limit: Maximum references to retain.
    :param Callable relevance_score: Strategy-appropriate relevance score.
    :return list[Paper]: Ranked, capped reference records.
    """
    ranked = sorted(
        enumerate(recovered),
        key=lambda item: (
            -relevance_score(item[1]),
            -max(item[1].citation_count, 0),
            item[0],
        ),
    )
    return [paper for _position, paper in ranked[:limit]]


def _select_recovered_references_by_text(
    seed: Paper,
    recovered: Sequence[Paper],
    limit: int,
) -> list[Paper]:
    """Select fallback references by relevance instead of bibliography order.

    Title/abstract similarity to the seed is the primary signal. Citation count
    breaks topical ties, and original bibliography order is the stable fallback
    when metadata cannot distinguish records.

    :param Paper seed: Seed paper used as the relevance query.
    :param Sequence[Paper] recovered: Materialized bibliography records.
    :param int limit: Maximum references to retain.
    :return list[Paper]: Ranked, capped reference records.
    """
    if len(recovered) <= limit:
        return list(recovered)

    index = AbstractSimilarityIndex()
    index.build({seed.paper_id: seed, **{paper.paper_id: paper for paper in recovered}})
    return select_recovered_references_by_score(
        recovered,
        limit,
        lambda paper: index.similarity(seed.paper_id, paper.paper_id),
    )


def _add_verified_reference(
    seed: Paper, references: list[Paper], paper: Paper | None
) -> Paper | None:
    """Keep the canonical reference while excluding matches to the seed."""
    if (
        paper is None
        or paper.paper_id == seed.paper_id
        or candidate_records_match(seed, paper)
    ):
        return None
    for existing in references:
        if existing.paper_id == paper.paper_id or candidate_records_match(
            existing, paper
        ):
            return merge_paper_metadata(existing, paper)
    references.append(paper)
    return paper


def fetch_seed_references(
    client: SemanticScholarClient,
    seed: Paper,
    limit: int,
    *,
    seed_identifier: str | None = None,
    local_lookup: Callable[[list[str]], dict[str, Paper]] | None = None,
    local_bibliography_lookup: Callable[
        [Sequence[BibliographyKey]], dict[BibliographyKey, Paper]
    ]
    | None = None,
    reference_selector: ReferenceSelector | None = None,
    s2_reference_selector: ReferenceSelector | None = None,
) -> tuple[CandidateSourceResult, ...]:
    """Collect S2 seed references with an arXiv bibliography fallback.

    :param SemanticScholarClient client: Existing metadata and discovery client.
    :param Paper seed: Resolved seed, supplying external identifiers.
    :param int limit: Maximum usable references; zero disables discovery.
    :param str | None seed_identifier: Original input, retaining arXiv version.
    :param Callable | None local_lookup: Read unambiguous requested aliases from an
        already prepared corpus, keyed by normalized request identifier.
    :param Callable | None local_bibliography_lookup: Resolve identifier-free
        entries by title, first author, and year after an empty S2 response.
    :param Callable | None reference_selector: Strategy-specific selector applied
        after the complete fallback bibliography is materialized.
    :param Callable | None s2_reference_selector: Selector for a wider bounded
        list of provider references, retaining the original admission limit.
    :return tuple[CandidateSourceResult, ...]: S2 outcome and optional fallback.
    """
    from citemesh.services import (
        SemanticScholarRequestError,
        SemanticScholarUnavailableError,
    )
    from citemesh.services.arxiv import ArxivClient, arxiv_identifier

    if limit <= 0:
        return ()
    provider_id = provider_lookup_identifier(seed.paper_id, seed)
    fetch_limit = max(limit, 100) if s2_reference_selector is not None else limit
    original = (
        fetch_candidate_source(
            "references",
            lambda: client.get_paper_references(
                provider_id, limit=fetch_limit, raise_on_unavailable=True
            ),
        )
        if provider_id is not None
        else CandidateSourceResult("references", CandidateSourceState.UNAVAILABLE)
    )
    if original.papers:
        if s2_reference_selector is None:
            return (original,)
        references: list[Paper] = []
        source_ranks: dict[str, int] = {}
        for rank, paper in enumerate(original.papers, start=1):
            canonical = _add_verified_reference(seed, references, paper)
            if canonical is not None:
                source_ranks.setdefault(canonical.paper_id, rank)
        if references:
            return (
                CandidateSourceResult(
                    source=original.source,
                    state=original.state,
                    papers=tuple(s2_reference_selector(seed, references, limit)),
                    verified_references=tuple(references),
                    source_ranks=source_ranks,
                ),
            )
        original = CandidateSourceResult(original.source, CandidateSourceState.EMPTY)
    identifier = next(
        (
            value
            for raw in (seed_identifier or "", seed.arxiv_id, seed.paper_id)
            if (value := arxiv_identifier(raw))
        ),
        None,
    )
    if identifier is None:
        return (original,) if provider_id is not None else ()
    arxiv = ArxivClient()
    entries = arxiv.get_bibliography(identifier)
    if entries is None:
        return (
            original,
            CandidateSourceResult(
                "arxiv_references",
                CandidateSourceState.UNAVAILABLE,
                error="No usable arXiv HTML",
            ),
        )

    # Alias maps are request-local. Partial bibliography evidence is never saved
    # in Semantic Scholar's complete-reference cache.
    resolver = _ReferenceAliasResolver()
    recovered: list[Paper] = []
    attempted: set[str] = set()
    metadata_requested = False
    metadata_completed = False

    # Resolve the full bibliography in bounded batches before applying the cap.
    # Otherwise HTML order (often alphabetical) chooses survivors before useful
    # metadata is available for ranking.
    identified = [entry.identifiers for entry in entries if entry.identifiers]
    for offset in range(0, len(identified), 50):
        batch = identified[offset : offset + 50]
        ids = list(dict.fromkeys(identifier for entry in batch for identifier in entry))

        resolver.bind_entries(batch)
        missing = [
            value
            for value in ids
            if value not in resolver.aliases and value not in attempted
        ]
        if local_lookup is not None:
            local_records = local_lookup(missing)
            for requested_id, paper in local_records.items():
                normalized_id = normalize_paper_id(requested_id).lower()
                if normalized_id in missing:
                    resolver.aliases[normalized_id] = paper
            resolver.bind_entries(batch)
            missing = [value for value in missing if value not in resolver.aliases]
        if missing and not client.refresh_paper_cache:
            cached_records = client.get_cached_papers(missing)
            resolver.remember(list(cached_records.values()))
            resolver.aliases.update(cached_records)
        resolver.bind_entries(batch)
        missing = [value for value in missing if value not in resolver.aliases]
        if missing:
            metadata_requested = True
            try:
                records = client.get_papers(missing, raise_on_unavailable=True)
            except (
                SemanticScholarRequestError,
                SemanticScholarUnavailableError,
            ) as exc:
                logger.debug("S2 metadata unavailable during arXiv recovery: %s", exc)
                records = {}
            else:
                metadata_completed = True
            resolver.remember(list(records.values()))
            for requested_id, paper in records.items():
                resolver.aliases[normalize_paper_id(requested_id).lower()] = paper
            resolver.bind_entries(batch)
            missing_arxiv = [
                value
                for value in missing
                if value.startswith("arxiv:") and value not in resolver.aliases
            ]
            if missing_arxiv:
                arxiv_records = arxiv.get_papers(missing_arxiv)
                if arxiv_records is not None:
                    metadata_completed = True
                    resolver.remember(list(arxiv_records.values()))
            resolver.bind_entries(batch)
        attempted.update(ids)
        for entry in batch:
            _add_verified_reference(seed, recovered, resolver.resolve(entry))
    _enrich_local_reference_records(client, recovered)
    resolved_with_identifiers = len(recovered)

    if (
        original.state is CandidateSourceState.EMPTY
        and local_bibliography_lookup is not None
    ):
        keys = list(
            dict.fromkeys(
                entry.key
                for entry in entries
                if not entry.identifiers and entry.key is not None
            )
        )
        if keys:
            records_by_key = local_bibliography_lookup(keys)
            for key in keys:
                _add_verified_reference(seed, recovered, records_by_key.get(key))

    verified_references = tuple(recovered)
    materialized_count = len(verified_references)
    selector = reference_selector or _select_recovered_references_by_text
    recovered = selector(seed, recovered, limit)
    logger.debug(
        "arXiv bibliography extraction: entries=%d, resolved_with_identifiers=%d, "
        "resolved_by_title=%d, materialized=%d, selected=%d, reference_limit=%d.",
        len(entries),
        resolved_with_identifiers,
        materialized_count - resolved_with_identifiers,
        materialized_count,
        len(recovered),
        limit,
    )
    if recovered:
        if materialized_count > len(recovered):
            logger.info(
                "Ranked %d recovered references from the arXiv bibliography, keeping %d "
                "(reference limit: %d).",
                materialized_count,
                len(recovered),
                limit,
            )
        else:
            logger.info(
                "Recovered %d reference%s from the arXiv bibliography.",
                len(recovered),
                "" if len(recovered) == 1 else "s",
            )
    # COMPLETE describes an evaluated source, not exhaustive bibliography coverage.
    if recovered:
        state = CandidateSourceState.COMPLETE
        error = None
    elif metadata_requested and not metadata_completed:
        state = CandidateSourceState.UNAVAILABLE
        error = "Reference metadata unavailable during arXiv recovery"
    else:
        state = CandidateSourceState.EMPTY
        error = None
    return (
        original,
        CandidateSourceResult(
            "arxiv_references",
            state,
            papers=tuple(recovered),
            error=error,
            verified_references=verified_references,
        ),
    )


def require_available_candidate_source(
    results: Sequence[CandidateSourceResult],
    *,
    context: str,
) -> None:
    """Require an available candidate source, which may be evaluated-empty.

    Partial source outages remain usable but are surfaced once in logs. When every
    requested source is unavailable, callers must not publish a normal graph. An
    empty arXiv fallback has not recovered a candidate source, so it cannot mask
    unavailable Semantic Scholar discovery.

    :param Sequence[CandidateSourceResult] results: Attempted source results.
    :param str context: Human-readable acquisition context for diagnostics.
    :return None: Returns after validating the result set.
    :raises CandidateAcquisitionError: If every attempted source was unavailable.
    """
    if not results:
        return
    recovered_references = any(
        result.source == "arxiv_references" and result.papers for result in results
    )
    candidate_source_available = any(
        result.state is not CandidateSourceState.UNAVAILABLE
        and (result.source != "arxiv_references" or result.papers)
        for result in results
    )
    unavailable = [
        result
        for result in results
        if result.state is CandidateSourceState.UNAVAILABLE
        and not (recovered_references and result.source == "references")
    ]
    if not candidate_source_available:
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
        logger.debug("Unavailable candidate-source details: %s", details)
        logger.warning(
            "Continuing with unavailable candidate sources: %s.",
            ", ".join(result.source for result in unavailable),
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


def annotate_verified_seed_references(
    candidates: dict[str, Paper],
    source_results: Sequence[CandidateSourceResult],
    seed_relations: dict[str, str],
) -> None:
    """Label verified references already in the deduplicated candidate pool."""
    for result in source_results:
        for reference in result.verified_references:
            paper_id = reference.paper_id
            if paper_id not in candidates:
                matches = [
                    candidate_id
                    for candidate_id, candidate in candidates.items()
                    if candidate_records_match(reference, candidate)
                ]
                if len(matches) != 1:
                    continue
                paper_id = matches[0]
            previous = seed_relations.get(paper_id, "")
            seed_relations[paper_id] = merge_seed_relation(
                "" if previous == "citation" else previous, "referenced_by_seed"
            )


def merge_paper_metadata(preferred: Paper, incoming: Paper) -> Paper:
    """Merge supplemental metadata from an alternate source into ``preferred``.

    :param Paper preferred: Canonical paper record to retain.
    :param Paper incoming: Supplemental paper record to merge.
    :return Paper: ``preferred`` with missing metadata hydrated.
    """
    if preferred.paper_id != incoming.paper_id:
        matching_namespaces = _candidate_match_namespaces(preferred, incoming)
        if matching_namespaces:
            logger.debug(
                "Merged record %s into %s via shared %s identifier%s",
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
    preferred_arxiv, preferred_doi = external_ids_from_canonical_paper_id(
        preferred.paper_id
    )
    incoming_arxiv, incoming_doi = external_ids_from_canonical_paper_id(
        incoming.paper_id
    )
    supplemental_doi = incoming.doi or incoming_doi
    doi = preferred.doi or preferred_doi or supplemental_doi
    preprint_alias = arxiv_identifier_from_doi(_normalized_doi(doi))
    if preprint_alias:
        arxiv_suffix = preprint_alias.removeprefix("arxiv:")
        if preferred.doi or preferred_doi:
            preferred_arxiv = preferred_arxiv or arxiv_suffix
        # Keep the arXiv identity alongside compatible publication DOI evidence.
        if not candidate_identifiers_conflict(preferred, incoming):
            incoming_arxiv = incoming_arxiv or arxiv_suffix
            for candidate_doi in (preferred_doi, incoming.doi, incoming_doi):
                normalized_doi = _normalized_doi(candidate_doi)
                if normalized_doi and arxiv_identifier_from_doi(normalized_doi) is None:
                    doi = candidate_doi
                    break
    arxiv_id = (
        preferred.arxiv_id or preferred_arxiv or incoming.arxiv_id or incoming_arxiv
    )
    if not arxiv_id:
        incoming_alias = arxiv_identifier_from_doi(
            _normalized_doi(supplemental_doi)
        ) or arxiv_identifier_from_doi(_normalized_doi(incoming_doi))
        if incoming_alias and not candidate_identifiers_conflict(preferred, incoming):
            arxiv_id = incoming_alias.removeprefix("arxiv:")
    preferred.arxiv_id = arxiv_id
    preferred.doi = doi
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


def paper_embedding_metadata(paper: Paper) -> PaperMetadata:
    """Build embedding-cache metadata payload for a paper.

    :param Paper paper: Paper to normalize.
    :return PaperMetadata: Metadata payload accepted by embedding cache.
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
    seed_identifier: str | None = None,
    reference_selector: ReferenceSelector | None = None,
) -> CandidatePool:
    """Fetch a deduplicated candidate pool from Semantic Scholar seed neighbors.

    Free-text query seeds (``query:`` IDs) have no S2 neighbors; they are proxied
    through paper search on the query text before recommendation expansion.

    :param SemanticScholarClient client: Semantic Scholar client.
    :param Paper seed_paper: Seed paper (S2-backed or free-text query seed).
    :param int max_references: Maximum seed references to fetch (0 disables).
    :param int max_citations: Maximum citing papers to fetch (0 disables).
    :param int max_recommendations: Maximum recommendations to fetch (0 disables).
    :param str | None seed_identifier: Original input, retaining an arXiv version.
    :param Callable | None reference_selector: Strategy-specific recovered-reference
        selector.
    :return CandidatePool: Deduplicated candidate pool with provenance tags.
    """
    with client.candidate_operation_scope():
        return _fetch_candidate_pool(
            client,
            seed_paper,
            max_references=max_references,
            max_citations=max_citations,
            max_recommendations=max_recommendations,
            seed_identifier=seed_identifier,
            reference_selector=reference_selector,
        )


def _fetch_candidate_pool(
    client: SemanticScholarClient,
    seed_paper: Paper,
    *,
    max_references: int = 0,
    max_citations: int = 0,
    max_recommendations: int = 0,
    seed_identifier: str | None = None,
    reference_selector: ReferenceSelector | None = None,
) -> CandidatePool:
    """Fetch candidates while an outer operation scope is active.

    :param SemanticScholarClient client: Semantic Scholar client.
    :param Paper seed_paper: Seed paper (S2-backed or free-text query seed).
    :param int max_references: Maximum seed references to fetch (0 disables).
    :param int max_citations: Maximum citing papers to fetch (0 disables).
    :param int max_recommendations: Maximum recommendations to fetch (0 disables).
    :param str | None seed_identifier: Original input, retaining an arXiv version.
    :param Callable | None reference_selector: Strategy-specific recovered-reference
        selector.
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
            reference_results = fetch_seed_references(
                client,
                seed_paper,
                max_references,
                seed_identifier=seed_identifier,
                reference_selector=reference_selector,
            )
            source_results.extend(reference_results)
            for result in reference_results:
                for paper in result.papers:
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
