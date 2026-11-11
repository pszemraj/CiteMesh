"""
Candidate pool with indexed paper identity.

The pool holds one collection's pre-ranking candidates, their source tags,
seed relations, recommendation positions, and the provider-ID aliases that
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

A publication and its corpus preprint often share no identifier. Before
ranking, :meth:`DiscoveryPool.reconcile_publication_versions` folds such pairs
under :func:`~citemesh.strategies.candidates.corpus_publication_matches`, which
requires equal normalized titles, so members are grouped by that title and the
rule runs only within a group.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from citemesh.core import Paper
from citemesh.core.paper_ids import bibliography_text_key
from citemesh.strategies.candidates import (
    QUERY_SEED_SEARCH_LIMIT,
    CandidateIdentityKey,
    CandidateSourceResult,
    ReferenceSelector,
    candidate_identifiers_conflict,
    candidate_identity_keys,
    candidate_records_match,
    corpus_publication_matches,
    fetch_candidate_source,
    fetch_seed_references,
    merge_paper_metadata,
    merge_seed_relation,
    require_available_candidate_source,
)

if TYPE_CHECKING:
    from citemesh.services.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)


class DiscoveryPool:
    """Deduplicated candidates keyed by their surviving paper IDs."""

    def __init__(
        self, seed: Paper, seed_relations: dict[str, str] | None = None
    ) -> None:
        """Start an empty pool for one seed.

        :param Paper seed: Canonical seed; records matching it enrich it instead.
        :param Optional[Dict[str, str]] seed_relations: Relation labels by paper
            ID, shared with a caller that keeps the seed's own label there.
        """
        self.seed = seed
        self.seed_relations = {} if seed_relations is None else seed_relations
        self.papers: dict[str, Paper] = {}
        self.sources: dict[str, set[str]] = {}
        self.recommendation_ranks: dict[str, int] = {}
        self.provider_aliases: dict[str, str] = {}
        self.source_status: dict[str, str] = {}
        self._members_by_key: dict[CandidateIdentityKey, dict[str, None]] = {}
        self._keys_by_member: dict[str, tuple[CandidateIdentityKey, ...]] = {}

    def resolve(self, paper: Paper) -> str | None:
        """Name what a record joins, in the order :meth:`add` applies.

        A record of the seed (same ID or a matching identifier) joins the seed.
        Otherwise an exact member ID, then a compatible recorded alias (for
        provider records), then a unique identifier match selects a member.

        :param Paper paper: Record to reconcile, with a non-empty ID.
        :return Optional[str]: The seed's ID for a record of the seed, the
            joined member's ID, or ``None`` for a new work.
        """
        paper_id = str(paper.paper_id).strip()
        if paper_id == self.seed.paper_id or candidate_records_match(paper, self.seed):
            return self.seed.paper_id
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
        """Add one candidate, reconciling it with the seed and existing members.

        A record :meth:`resolve` assigns to a member merges into it, and a
        provider record matched only by identifiers is remembered as an alias.
        A local-corpus record matched by identifiers takes over the member's ID
        and evidence instead.

        :param Paper incoming: Newly observed candidate payload.
        :param str source: Candidate source tag.
        :param str relation: Relation-to-seed label.
        :return Optional[str]: Surviving member ID, or ``None`` for a record with
            an empty ID, a seed-role record, or one matching the seed.
        """
        incoming_id = str(incoming.paper_id).strip()
        if not incoming_id or incoming.is_seed:
            return None
        target_id = self.resolve(incoming)
        if target_id == self.seed.paper_id:
            merge_paper_metadata(self.seed, incoming)
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
        merged_relation = merge_seed_relation(
            self.seed_relations.get(canonical_id, ""), relation
        )
        if merged_relation:
            self.seed_relations[canonical_id] = merged_relation
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
        relation = merge_seed_relation(
            self.seed_relations.get(target_id, ""),
            self.seed_relations.pop(absorbed_id, ""),
        )
        if relation:
            self.seed_relations[target_id] = relation

    def reconcile_publication_versions(self) -> int:
        """Fold each publication record into its corpus preprint.

        A pair is folded only when each side is the other's only match in the
        whole pool, so a competing record leaves both versions in place.

        :return int: Number of publication records folded.
        """
        groups: dict[str, list[str]] = {}
        for member_id, paper in self.papers.items():
            title = bibliography_text_key(paper.title)
            # corpus_publication_matches requires a multi-word normalized title.
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
                    if corpus_publication_matches(
                        self.papers[member_id], self.papers[other_id]
                    )
                ]
        folds = [
            (found[0], publication_id)
            for publication_id, found in matches.items()
            if not self.papers[publication_id].is_local_corpus
            and len(found) == 1
            and matches[found[0]] == [publication_id]
        ]
        for corpus_id, publication_id in folds:
            self.absorb(corpus_id, publication_id)
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


def fetch_candidate_pool(
    client: SemanticScholarClient,
    seed_paper: Paper,
    *,
    max_references: int = 0,
    max_citations: int = 0,
    max_recommendations: int = 0,
    seed_identifier: str | None = None,
    reference_selector: ReferenceSelector | None = None,
) -> DiscoveryPool:
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
    :return DiscoveryPool: Deduplicated candidate pool with provenance tags.
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
) -> DiscoveryPool:
    """Fetch candidates while an outer operation scope is active.

    :param SemanticScholarClient client: Semantic Scholar client.
    :param Paper seed_paper: Seed paper (S2-backed or free-text query seed).
    :param int max_references: Maximum seed references to fetch (0 disables).
    :param int max_citations: Maximum citing papers to fetch (0 disables).
    :param int max_recommendations: Maximum recommendations to fetch (0 disables).
    :param str | None seed_identifier: Original input, retaining an arXiv version.
    :param Callable | None reference_selector: Strategy-specific recovered-reference
        selector.
    :return DiscoveryPool: Deduplicated candidate pool with provenance tags.
    """
    pool = DiscoveryPool(seed_paper)
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
