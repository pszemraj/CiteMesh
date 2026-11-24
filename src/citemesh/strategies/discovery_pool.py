"""
Candidate pool with indexed paper identity.

The pool holds one collection's pre-ranking candidates, their source tags,
focal relations, recommendation positions, and the provider-ID aliases that
later lists resolve through.
Hybrid discovery and the embedding strategy's candidates mode both collect
into it. A record joins a member when it carries the member's provider ID or a
recorded alias of it, or when it matches exactly one member under
:func:`~citemesh.strategies.candidates.candidate_records_match`. That rule
holds only for records sharing a normalized arXiv or DOI identifier, so the
pool indexes members by those identifiers and runs the exact check only on
index hits. Ingest therefore costs time in proportion to the members a record
shares an identifier with, not the pool size, and yields the same pool as
comparing every pair.

Member identifiers change only through the pool: every metadata merge
re-indexes the member it enriched.

A publication and its preprint often share no identifier. Before ranking,
:meth:`DiscoveryPool.reconcile_publication_versions` folds such pairs, whether
or not the preprint is a corpus row, under
:func:`~citemesh.strategies.candidates.publication_version_preprint`, which
requires equal normalized titles, so members are grouped by that title and the
rule runs only within a group.
"""

from __future__ import annotations

import logging
from collections.abc import Container, Sequence
from typing import TYPE_CHECKING, TypedDict

from citemesh.core import Paper
from citemesh.core.paper_ids import bibliography_text_key
from citemesh.strategies.candidates import (
    QUERY_FOCAL_SEARCH_LIMIT,
    CandidateIdentityKey,
    CandidateSourceResult,
    ReferenceSelector,
    candidate_identifiers_conflict,
    candidate_identity_keys,
    candidate_records_match,
    fetch_candidate_source,
    fetch_focal_references,
    merge_focal_relation,
    merge_paper_metadata,
    publication_version_preprint,
    require_available_candidate_source,
)
from citemesh.strategies.discovery_ranking import FOCAL_REFERENCE_MIN_CONTEXTS

if TYPE_CHECKING:
    from citemesh.services.semantic_scholar import SemanticScholarClient
    from citemesh.services.semantic_scholar.endpoints import ReferenceEvidence
    from citemesh.strategies.discovery_ranking import RankedCandidate

logger = logging.getLogger(__name__)


class DiscoveryPool:
    """Deduplicated candidates keyed by their surviving paper IDs."""

    def __init__(
        self, focal: Paper, focal_relations: dict[str, str] | None = None
    ) -> None:
        """Start an empty pool for one focal paper.

        :param Paper focal: Canonical focal paper; records matching it enrich it instead.
        :param Optional[Dict[str, str]] focal_relations: Relation labels by paper
            ID, shared with a caller that keeps the focal paper's own label there.
        """
        self.focal = focal
        self.focal_relations = {} if focal_relations is None else focal_relations
        self.papers: dict[str, Paper] = {}
        self.sources: dict[str, set[str]] = {}
        self.recommendation_ranks: dict[str, int] = {}
        self.query_corpus_ranks: dict[str, int] = {}
        self.focal_reference_evidence: dict[str, ReferenceEvidence] = {}
        self.provider_aliases: dict[str, str] = {}
        self.source_status: dict[str, str] = {}
        self._members_by_key: dict[CandidateIdentityKey, dict[str, None]] = {}
        self._keys_by_member: dict[str, tuple[CandidateIdentityKey, ...]] = {}

    def resolve(self, paper: Paper) -> str | None:
        """Name what a record joins, in the order :meth:`add` applies.

        A record of the focal paper (same ID or a matching identifier) joins the focal paper.
        Otherwise an exact member ID, then a compatible recorded alias (for
        provider records), then a unique identifier match selects a member.

        :param Paper paper: Record to reconcile, with a non-empty ID.
        :return Optional[str]: The focal paper's ID for a record of the focal paper, the
            joined member's ID, or ``None`` for a new work.
        """
        paper_id = str(paper.paper_id).strip()
        if paper_id == self.focal.paper_id or candidate_records_match(
            paper, self.focal
        ):
            return self.focal.paper_id
        if paper_id in self.papers:
            return paper_id
        if not paper.is_local_corpus:
            alias_target = self.provider_aliases.get(paper_id)
            if (
                alias_target is not None
                and alias_target in self.papers
                and not candidate_identifiers_conflict(self.papers[alias_target], paper)
            ):
                return alias_target
        return self.unique_match(paper)

    def add(self, incoming: Paper, *, source: str, relation: str) -> str | None:
        """Add one candidate, reconciling it with the focal paper and existing members.

        A record :meth:`resolve` assigns to a member merges into it, and a
        provider record matched only by identifiers is remembered as an alias.
        A local-corpus record matched by identifiers takes over the member's ID
        and evidence instead.

        :param Paper incoming: Newly observed candidate payload.
        :param str source: Candidate source tag.
        :param str relation: Relation-to-focal label.
        :return Optional[str]: Surviving member ID, or ``None`` for a record with
            an empty ID, a record carrying the focal paper role, or one matching the focal paper.
        """
        incoming_id = str(incoming.paper_id).strip()
        if not incoming_id or incoming.is_focal:
            return None
        target_id = self.resolve(incoming)
        if target_id == self.focal.paper_id:
            merge_paper_metadata(self.focal, incoming)
            return None

        canonical_id = incoming_id
        if target_id is None:
            self.papers[canonical_id] = incoming
            self._index(canonical_id)
        elif target_id == incoming_id:
            self.merge(target_id, incoming)
        elif incoming.is_local_corpus:
            # The corpus record takes over; later lists can carry the matched
            # provider ID alone, so absorb keeps it as an alias.
            self.papers[canonical_id] = incoming
            self._index(canonical_id)
            self.absorb(canonical_id, target_id)
        else:
            canonical_id = target_id
            self.merge(target_id, incoming)
            self.provider_aliases[incoming_id] = target_id

        self.sources.setdefault(canonical_id, set()).add(source)
        merged_relation = merge_focal_relation(
            self.focal_relations.get(canonical_id, ""), relation
        )
        if merged_relation:
            self.focal_relations[canonical_id] = merged_relation
        return canonical_id

    def unique_match(self, paper: Paper) -> str | None:
        """Return the only member that explicitly identifies the same work.

        :param Paper paper: Record to reconcile against the pool.
        :return Optional[str]: Matching member ID, or ``None`` when no member or
            more than one member matches.
        """
        hits = dict.fromkeys(
            member_id
            for key in candidate_identity_keys(paper)
            for member_id in self._members_by_key.get(key, ())
        )
        matched_id: str | None = None
        for member_id in hits:
            if candidate_records_match(paper, self.papers[member_id]):
                if matched_id is not None:
                    return None
                matched_id = member_id
        return matched_id

    def merge(self, member_id: str, incoming: Paper) -> Paper:
        """Merge supplemental metadata into one member and re-index it.

        :param str member_id: Member that keeps its ID.
        :param Paper incoming: Record supplying missing metadata.
        :return Paper: The enriched member.
        """
        member = merge_paper_metadata(self.papers[member_id], incoming)
        self._index(member_id)
        return member

    def record_recommendation(self, member_id: str, rank: int) -> None:
        """Keep a member's first position among the provider recommendations.

        :param str member_id: Member ID.
        :param int rank: One-based position in the recommendation list.
        :return None: Leaves an earlier position unchanged.
        """
        self.recommendation_ranks.setdefault(member_id, rank)

    def record_query_corpus_rank(self, member_id: str, rank: int) -> None:
        """Keep the earliest native query retrieval position for this work."""
        self.query_corpus_ranks[member_id] = min(
            rank, self.query_corpus_ranks.get(member_id, rank)
        )

    def record_reference_evidence(
        self, member_id: str, evidence: ReferenceEvidence
    ) -> None:
        """Retain the observed evidence across records of the same work."""
        previous = self.focal_reference_evidence.get(member_id)
        self.focal_reference_evidence[member_id] = (
            previous.combine(evidence) if previous is not None else evidence
        )

    def absorb(self, target_id: str, absorbed_id: str) -> None:
        """Fold one member into another member that records the same work.

        The target keeps its ID and gains the absorbed member's metadata, tags,
        earlier recommendation position and relation. The absorbed ID, and every
        alias that pointed at it, resolve to the target afterwards.

        :param str target_id: Member that survives.
        :param str absorbed_id: Member removed from the pool.
        :return None: Updates papers, tags, ranks, aliases, relations, and index.
        """
        absorbed = self.papers.pop(absorbed_id)
        self._unindex(absorbed_id)
        self.merge(target_id, absorbed)
        self.sources.setdefault(target_id, set()).update(
            self.sources.pop(absorbed_id, set())
        )
        for provider_id, target in self.provider_aliases.items():
            if target == absorbed_id:
                self.provider_aliases[provider_id] = target_id
        self.provider_aliases[absorbed_id] = target_id
        absorbed_rank = self.recommendation_ranks.pop(absorbed_id, None)
        if absorbed_rank is not None:
            self.recommendation_ranks[target_id] = min(
                self.recommendation_ranks.get(target_id, absorbed_rank), absorbed_rank
            )
        absorbed_query_rank = self.query_corpus_ranks.pop(absorbed_id, None)
        if absorbed_query_rank is not None:
            self.record_query_corpus_rank(target_id, absorbed_query_rank)
        if evidence := self.focal_reference_evidence.pop(absorbed_id, None):
            self.record_reference_evidence(target_id, evidence)
        relation = merge_focal_relation(
            self.focal_relations.get(target_id, ""),
            self.focal_relations.pop(absorbed_id, ""),
        )
        if relation:
            self.focal_relations[target_id] = relation

    def reconcile_publication_versions(self) -> int:
        """Fold each publication record into its preprint, corpus row or not.

        A pair is folded only when each side is the other's only match in the
        whole pool, so a competing record leaves both versions in place. The
        preprint survives, a corpus row whenever one of the two is.

        :return int: Number of publication records folded.
        """
        groups: dict[str, list[str]] = {}
        for member_id, paper in self.papers.items():
            title = bibliography_text_key(paper.title)
            # publication_version_preprint requires a multi-word normalized title.
            if " " in title:
                groups.setdefault(title, []).append(member_id)
        matches: dict[str, list[str]] = {}
        for members in groups.values():
            if len(members) < 2:
                continue
            for member_id in members:
                matches[member_id] = [
                    other_id
                    for other_id in members
                    if other_id != member_id
                    and publication_version_preprint(
                        self.papers[member_id], self.papers[other_id]
                    )
                    is not None
                ]
        folds: list[tuple[str, str]] = []
        for member_id, found in matches.items():
            if len(found) != 1 or matches[found[0]] != [member_id]:
                continue
            other_id = found[0]
            preprint = publication_version_preprint(
                self.papers[member_id], self.papers[other_id]
            )
            # Each mutual pair appears twice; fold it from the publication's side.
            if preprint is self.papers[other_id]:
                folds.append((other_id, member_id))
        for preprint_id, publication_id in folds:
            self.absorb(preprint_id, publication_id)
        return len(folds)

    def _index(self, member_id: str) -> None:
        """Register a member under its current identifiers, replacing older keys.

        :param str member_id: Member to (re-)index.
        :return None: Updates both index directions.
        """
        self._unindex(member_id)
        keys = candidate_identity_keys(self.papers[member_id])
        self._keys_by_member[member_id] = keys
        for key in keys:
            self._members_by_key.setdefault(key, {})[member_id] = None

    def _unindex(self, member_id: str) -> None:
        """Remove a member from the identifier index.

        :param str member_id: Member to remove.
        :return None: Drops keys that no longer name any member.
        """
        for key in self._keys_by_member.pop(member_id, ()):
            members = self._members_by_key[key]
            del members[member_id]
            if not members:
                del self._members_by_key[key]


class _DiscoveryLeadRequired(TypedDict):
    """Compact reading-list entry for an omitted candidate."""

    paper_id: str
    title: str
    arxiv_id: str
    doi: str
    rank: int


class DiscoveryLead(_DiscoveryLeadRequired, total=False):
    """Focal references additionally carry their observed passage count."""

    context_count: int


class DiscoveryLeads(TypedDict):
    """Two capped lists of omitted papers with different evidence."""

    references: list[DiscoveryLead]
    corpus: list[DiscoveryLead]


def _discovery_lead(paper: Paper, rank: int) -> DiscoveryLead:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "arxiv_id": paper.arxiv_id,
        "doi": paper.doi,
        "rank": rank,
    }


def build_discovery_leads(
    pool: DiscoveryPool,
    ranked: Sequence[RankedCandidate],
    final_graph_ids: Container[str],
    *,
    limit: int,
) -> DiscoveryLeads | None:
    """Build secondary reading lists from the already ranked candidate pool.

    References require at least two distinct provider contexts and retain the
    fused order of ``ranked``. Corpus leads keep their original native query-role
    retrieval rank and exclude papers acquired through a citation or
    recommendation. Both lists exclude every paper in the final graph.
    """
    if limit <= 0:
        return None
    omitted = {
        item.paper_id: item for item in ranked if item.paper_id not in final_graph_ids
    }
    references: list[DiscoveryLead] = []
    for paper_id, item in omitted.items():
        evidence = pool.focal_reference_evidence.get(paper_id)
        if evidence is None or evidence.context_count < FOCAL_REFERENCE_MIN_CONTEXTS:
            continue
        lead = _discovery_lead(pool.papers[paper_id], item.rank)
        lead["context_count"] = evidence.context_count
        references.append(lead)
        if len(references) == limit:
            break

    corpus_positions = sorted(
        (rank, paper_id)
        for paper_id, rank in pool.query_corpus_ranks.items()
        if paper_id in omitted
        and pool.sources[paper_id] == {"semantic"}
        and paper_id not in pool.recommendation_ranks
    )
    corpus = [
        _discovery_lead(pool.papers[paper_id], rank)
        for rank, paper_id in corpus_positions[:limit]
    ]
    return (
        {"references": references, "corpus": corpus} if references or corpus else None
    )


def fetch_candidate_pool(
    client: SemanticScholarClient,
    focal_paper: Paper,
    *,
    max_references: int = 0,
    max_citations: int = 0,
    max_recommendations: int = 0,
    focal_identifier: str | None = None,
    reference_selector: ReferenceSelector | None = None,
) -> DiscoveryPool:
    """Fetch a deduplicated candidate pool from Semantic Scholar focal paper neighbors.

    Focal paper records generated from free-text queries (``query:`` IDs) have
    no S2 neighbors. Paper search on the query text supplies recommendation input.

    :param SemanticScholarClient client: Semantic Scholar client.
    :param Paper focal_paper: Focal paper backed by S2 or generated from a query.
    :param int max_references: Maximum focal paper references to fetch (0 disables).
    :param int max_citations: Maximum citing papers to fetch (0 disables).
    :param int max_recommendations: Maximum recommendations to fetch (0 disables).
    :param str | None focal_identifier: Original input, retaining an arXiv version.
    :param Callable | None reference_selector: Strategy-specific recovered-reference
        selector.
    :return DiscoveryPool: Deduplicated candidate pool with provenance tags.
    """
    with client.candidate_operation_scope():
        return _fetch_candidate_pool(
            client,
            focal_paper,
            max_references=max_references,
            max_citations=max_citations,
            max_recommendations=max_recommendations,
            focal_identifier=focal_identifier,
            reference_selector=reference_selector,
        )


def _fetch_candidate_pool(
    client: SemanticScholarClient,
    focal_paper: Paper,
    *,
    max_references: int = 0,
    max_citations: int = 0,
    max_recommendations: int = 0,
    focal_identifier: str | None = None,
    reference_selector: ReferenceSelector | None = None,
) -> DiscoveryPool:
    """Fetch candidates while an outer operation scope is active.

    :param SemanticScholarClient client: Semantic Scholar client.
    :param Paper focal_paper: Focal paper backed by S2 or generated from a query.
    :param int max_references: Maximum focal paper references to fetch (0 disables).
    :param int max_citations: Maximum citing papers to fetch (0 disables).
    :param int max_recommendations: Maximum recommendations to fetch (0 disables).
    :param str | None focal_identifier: Original input, retaining an arXiv version.
    :param Callable | None reference_selector: Strategy-specific recovered-reference
        selector.
    :return DiscoveryPool: Deduplicated candidate pool with provenance tags.
    """
    pool = DiscoveryPool(focal_paper)
    source_results: list[CandidateSourceResult] = []
    focal_id = str(focal_paper.paper_id)
    is_query_focal = focal_id.startswith("query:")
    query_candidate_budget = (
        max_references + max_citations + max_recommendations if is_query_focal else 0
    )
    recommendation_limit = max_recommendations

    if is_query_focal:
        query_text = (focal_paper.title or "").strip() or focal_id
        search_limit = min(QUERY_FOCAL_SEARCH_LIMIT, query_candidate_budget)
        if search_limit > 0:
            search_result = fetch_candidate_source(
                client,
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
        anchor_ids = [focal_id]
        if max_references > 0:
            reference_results = fetch_focal_references(
                client,
                focal_paper,
                max_references,
                focal_identifier=focal_identifier,
                reference_selector=reference_selector,
            )
            source_results.extend(reference_results)
            for result in reference_results:
                for paper in result.papers:
                    pool.add(paper, source="reference", relation="referenced_by_focal")
        if max_citations > 0:
            citation_result = fetch_candidate_source(
                client,
                "citations",
                lambda: client.get_paper_citations(
                    focal_id,
                    limit=max_citations,
                    raise_on_unavailable=True,
                ),
            )
            source_results.append(citation_result)
            for paper in citation_result.papers:
                pool.add(paper, source="citation", relation="cites_focal")

    if recommendation_limit > 0 and anchor_ids:
        recommendation_result = fetch_candidate_source(
            client,
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
        context=f"candidate acquisition for {focal_id}",
    )

    logger.debug(
        "Candidate pool for %s: %d papers (refs<=%d cites<=%d recs<=%d).",
        focal_id,
        len(pool.papers),
        max_references,
        max_citations,
        max_recommendations,
    )
    return pool
