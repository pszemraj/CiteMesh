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
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
)

from citemesh.core import Paper
from citemesh.paper_ids import (
    normalize_paper_id,
    paper_identifier_aliases,
    recognize_arxiv_identifier,
)

if TYPE_CHECKING:
    from citemesh.services.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)

SEMANTIC_SOURCE_CHOICES = ("candidates", "arxiv-corpus")
DEFAULT_CANDIDATE_POOL_SIZE = 400
QUERY_SEED_SEARCH_LIMIT = 20
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_S2_PATTERN = re.compile(r"^(?:s2:)?[0-9a-f]{40}$", re.IGNORECASE)
_PLACEHOLDER_TITLES = {
    "n a",
    "na",
    "no title",
    "none",
    "not available",
    "unknown",
    "untitled",
}


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


@dataclass(frozen=True)
class IdentityEvidence:
    """Namespaced strong identifiers and conservative weak metadata evidence."""

    strong_ids: Mapping[str, frozenset[str]]
    weak_keys: frozenset[str]


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


def _strong_identifier_evidence(paper: Paper) -> Dict[str, frozenset[str]]:
    """Build namespaced strong identifier evidence for one paper.

    :param Paper paper: Paper payload to inspect.
    :return Dict[str, frozenset[str]]: Stable identifier sets by namespace.
    """
    identifiers: Dict[str, Set[str]] = {}

    primary = str(paper.paper_id or "").strip()
    primary_arxiv = recognize_arxiv_identifier(primary, allow_bare=True)
    primary_doi = _normalized_doi(primary)
    if primary_arxiv:
        identifiers.setdefault("arxiv", set()).add(primary_arxiv.lower())
    elif primary_doi:
        identifiers.setdefault("doi", set()).add(primary_doi)
    elif primary:
        # Semantic Scholar IDs are normally 40 hexadecimal characters. Treat
        # opaque primary IDs as the same authoritative namespace too: exact
        # equality may reconcile, but metadata must not override disagreement.
        normalized_primary = primary.lower()
        if _S2_PATTERN.fullmatch(normalized_primary):
            normalized_primary = normalized_primary.removeprefix("s2:")
        identifiers.setdefault("s2", set()).add(normalized_primary)

    arxiv_identifier = recognize_arxiv_identifier(paper.arxiv_id, allow_bare=True)
    if arxiv_identifier:
        identifiers.setdefault("arxiv", set()).add(arxiv_identifier.lower())
    doi_identifier = _normalized_doi(paper.doi)
    if doi_identifier:
        identifiers.setdefault("doi", set()).add(doi_identifier)

    return {
        namespace: frozenset(sorted(values))
        for namespace, values in sorted(identifiers.items())
        if values
    }


def _weak_identity_keys(paper: Paper) -> frozenset[str]:
    """Build metadata evidence only when title, year, and authors are meaningful.

    :param Paper paper: Paper payload to inspect.
    :return frozenset[str]: Conservative weak identity keys.
    """
    title = normalize_identity_text(paper.title or "")
    if not title or title in _PLACEHOLDER_TITLES:
        return frozenset()
    if not isinstance(paper.year, int) or paper.year <= 0:
        return frozenset()
    author_tokens = tuple(
        token
        for token in (
            normalize_identity_text(author.name)
            for author in paper.authors[:3]
            if getattr(author, "name", None)
        )
        if token
    )
    if not author_tokens:
        return frozenset()
    base = f"meta:{title}|{paper.year}|{'|'.join(author_tokens)}"
    keys = {base}
    abstract = normalize_identity_text(paper.abstract or "")
    if abstract:
        keys.add(f"{base}|abs:{abstract[:256]}")
    return frozenset(sorted(keys))


def paper_identity_evidence(paper: Paper) -> IdentityEvidence:
    """Return namespaced strong and conservative weak identity evidence.

    :param Paper paper: Paper payload to inspect.
    :return IdentityEvidence: Evidence used by the reconciliation registry.
    """
    return IdentityEvidence(
        strong_ids=_strong_identifier_evidence(paper),
        weak_keys=_weak_identity_keys(paper),
    )


def _merge_identity_evidence(
    left: IdentityEvidence,
    right: IdentityEvidence,
) -> IdentityEvidence:
    """Union compatible identity evidence.

    :param IdentityEvidence left: Existing class evidence.
    :param IdentityEvidence right: Incoming class evidence.
    :return IdentityEvidence: Accumulated evidence.
    """
    namespaces = set(left.strong_ids) | set(right.strong_ids)
    strong_ids = {
        namespace: frozenset(
            set(left.strong_ids.get(namespace, frozenset()))
            | set(right.strong_ids.get(namespace, frozenset()))
        )
        for namespace in sorted(namespaces)
    }
    return IdentityEvidence(
        strong_ids=strong_ids,
        weak_keys=frozenset(set(left.weak_keys) | set(right.weak_keys)),
    )


def has_strong_identifier_conflict(
    left: IdentityEvidence,
    right: IdentityEvidence,
) -> bool:
    """Return whether strong-ID evidence contains an irreconcilable conflict.

    Exact DOI/arXiv agreement identifies one work even when Semantic Scholar
    assigned duplicate opaque records. Conflicting external identifiers remain
    irreconcilable.

    :param IdentityEvidence left: First evidence set.
    :param IdentityEvidence right: Second evidence set.
    :return bool: ``True`` when the evidence cannot describe one work.
    """
    shared_namespaces = set(left.strong_ids) & set(right.strong_ids)
    disagreements = {
        namespace
        for namespace in shared_namespaces
        if set(left.strong_ids[namespace]).isdisjoint(right.strong_ids[namespace])
    }
    if not disagreements:
        return False

    agreeing_external_ids = any(
        namespace in shared_namespaces
        and not set(left.strong_ids[namespace]).isdisjoint(right.strong_ids[namespace])
        for namespace in ("doi", "arxiv")
    )
    if disagreements == {"s2"} and agreeing_external_ids:
        # Semantic Scholar may assign multiple opaque records to one work. A
        # shared DOI/arXiv identifier is authoritative evidence that those S2
        # records describe the same paper; contradictory external IDs remain a
        # hard conflict.
        return False
    return True


class IdentityRegistry:
    """Alias ownership and accumulated evidence for reconciled paper classes."""

    def __init__(self) -> None:
        """Initialize an empty identity registry."""
        self._owners: Dict[str, Set[str]] = {}
        self._evidence: Dict[str, IdentityEvidence] = {}

    def __getitem__(self, alias: str) -> str:
        """Return one unambiguous alias owner.

        :param str alias: Alias key.
        :return str: Sole canonical owner.
        :raises KeyError: If the alias is absent or contested.
        """
        owner = self.get(alias)
        if owner is None:
            raise KeyError(alias)
        return owner

    def get(self, alias: str) -> Optional[str]:
        """Return the sole owner of an alias, ignoring contested aliases.

        :param str alias: Alias key.
        :return Optional[str]: Canonical owner when unambiguous.
        """
        owners = self._owners.get(alias, set())
        if len(owners) != 1:
            return None
        return next(iter(owners))

    def owners(self, alias: str) -> Set[str]:
        """Return every current owner of an alias.

        :param str alias: Alias key.
        :return Set[str]: Copy of canonical owners.
        """
        return set(self._owners.get(alias, set()))

    def evidence(self, canonical_id: str) -> Optional[IdentityEvidence]:
        """Return accumulated evidence for one canonical class.

        :param str canonical_id: Canonical paper ID.
        :return Optional[IdentityEvidence]: Registered evidence, if present.
        """
        return self._evidence.get(canonical_id)

    def canonical_ids(self) -> Iterator[str]:
        """Iterate canonical class IDs in insertion order.

        :return Iterator[str]: Canonical ID iterator.
        """
        return iter(self._evidence)

    def register(self, canonical_id: str, paper: Paper) -> None:
        """Register aliases and evidence for a canonical paper payload.

        :param str canonical_id: Canonical paper ID.
        :param Paper paper: Canonical paper payload.
        :return None: Registry is updated in place.
        """
        evidence = paper_identity_evidence(paper)
        existing = self._evidence.get(canonical_id)
        if existing is not None and has_strong_identifier_conflict(existing, evidence):
            # Exact primary-ID equality remains authoritative, but contradictory
            # secondary IDs are not allowed to contaminate future alias lookups.
            retained_strong = dict(existing.strong_ids)
            for namespace, values in evidence.strong_ids.items():
                if namespace not in retained_strong:
                    retained_strong[namespace] = values
            evidence = IdentityEvidence(
                strong_ids=retained_strong,
                weak_keys=frozenset(set(existing.weak_keys) | set(evidence.weak_keys)),
            )
        elif existing is not None:
            evidence = _merge_identity_evidence(existing, evidence)
        self._evidence[canonical_id] = evidence

        conflicting_refresh = existing is not None and has_strong_identifier_conflict(
            existing, paper_identity_evidence(paper)
        )
        for alias in paper_identity_aliases(paper):
            if conflicting_refresh:
                # Re-register only the retained canonical payload's aliases.
                continue
            self._owners.setdefault(alias, set()).add(canonical_id)

    def repoint(self, canonical_id: str, replaced_ids: Set[str]) -> None:
        """Collapse compatible classes into one survivor.

        :param str canonical_id: Surviving canonical ID.
        :param Set[str] replaced_ids: IDs folded into the survivor.
        :return None: Owners and accumulated evidence are updated.
        """
        accumulated = self._evidence.get(
            canonical_id,
            IdentityEvidence(strong_ids={}, weak_keys=frozenset()),
        )
        for replaced_id in replaced_ids:
            evidence = self._evidence.get(replaced_id)
            if evidence is not None and replaced_id != canonical_id:
                accumulated = _merge_identity_evidence(accumulated, evidence)
        self._evidence[canonical_id] = accumulated
        for replaced_id in replaced_ids:
            if replaced_id != canonical_id:
                self._evidence.pop(replaced_id, None)
        for owners in self._owners.values():
            if owners & replaced_ids:
                owners.difference_update(replaced_ids)
                owners.add(canonical_id)


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
    """Return strong aliases plus conservative metadata corroboration keys.

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

    aliases.update(_weak_identity_keys(paper))

    return sorted(aliases)


def resolve_aliases(aliases: IdentityRegistry, paper: Paper) -> List[str]:
    """Resolve every canonical paper ID matched by an incoming payload.

    A record may bridge two previously distinct identifier classes (for
    example, one payload supplies an arXiv ID and a later one supplies both
    that arXiv ID and a DOI). Callers reconcile all returned classes before
    registering the incoming aliases.

    :param IdentityRegistry aliases: Identity registry.
    :param Paper paper: Incoming paper payload.
    :return List[str]: Distinct matching canonical IDs in alias-key order.
    """
    incoming_evidence = paper_identity_evidence(paper)
    candidates: List[str] = []
    seen: Set[str] = set()
    for alias in paper_identity_aliases(paper):
        for canonical_id in aliases.owners(alias):
            if canonical_id in seen:
                continue
            seen.add(canonical_id)
            candidates.append(canonical_id)

    exact_primary = str(paper.paper_id)
    exact_match = (
        exact_primary if exact_primary in set(aliases.canonical_ids()) else None
    )
    compatible: List[str] = []
    for canonical_id in candidates:
        if canonical_id == exact_match:
            compatible.append(canonical_id)
            continue
        registered = aliases.evidence(canonical_id)
        if registered is None or has_strong_identifier_conflict(
            registered, incoming_evidence
        ):
            continue
        compatible.append(canonical_id)

    if exact_match is not None and exact_match not in compatible:
        compatible.insert(0, exact_match)
    if exact_match is not None:
        compatible.sort(key=lambda item: item != exact_match)

    for idx, left_id in enumerate(compatible):
        left = aliases.evidence(left_id)
        if left is None:
            continue
        for right_id in compatible[idx + 1 :]:
            right = aliases.evidence(right_id)
            if right is not None and has_strong_identifier_conflict(left, right):
                # A contested weak/external alias must not arbitrarily choose a
                # survivor. Exact-primary matches may still update that class,
                # but cannot use the contested alias to absorb the other class.
                return [exact_match] if exact_match is not None else []
    return compatible


def register_aliases(
    aliases: IdentityRegistry,
    canonical_id: str,
    paper: Paper,
) -> None:
    """Register identity aliases for a canonical paper ID.

    :param IdentityRegistry aliases: Identity registry to mutate.
    :param str canonical_id: Canonical paper identifier.
    :param Paper paper: Paper payload providing alias candidates.
    :return None: Alias map is mutated in place.
    """
    aliases.register(canonical_id, paper)


def repoint_aliases(
    aliases: IdentityRegistry,
    canonical_id: str,
    replaced_ids: Set[str],
) -> None:
    """Point aliases owned by reconciled records at their surviving paper.

    :param IdentityRegistry aliases: Identity registry to mutate.
    :param str canonical_id: Surviving canonical paper ID.
    :param Set[str] replaced_ids: Canonical IDs collapsed into the survivor.
    :return None: Alias map is mutated in place.
    """
    aliases.repoint(canonical_id, replaced_ids)


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
    aliases: IdentityRegistry,
    seed: Paper,
    papers: Dict[str, Paper],
    incoming: Paper,
) -> IdentityReconciliation:
    """Merge an incoming payload into its seed or candidate identity class.

    The seed always survives. Otherwise the first candidate insertion wins,
    keeping graph order stable. Callers remain responsible for folding any
    sidecar state associated with ``collapsed_ids``.

    :param IdentityRegistry aliases: Identity registry to update.
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
        register_aliases(aliases, seed_id, seed)
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
    register_aliases(aliases, canonical_id, papers[canonical_id])
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
        self._aliases = IdentityRegistry()
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
