"""
Hybrid discovery pool with indexed candidate identity.

The pool holds one hybrid collection's pre-ranking candidates, their source
tags, per-source provider ranks, seed relations, and the provider-ID aliases
that later lists resolve through. A record joins a member when it carries the
member's provider ID or a recorded alias of it, or when it matches exactly one
member under :func:`~citemesh.strategies.candidates.candidate_records_match`.
That rule holds only for records sharing a normalized arXiv or DOI
identifier, so the pool indexes members by those identifiers and runs the
exact check only on index hits. Ingest therefore costs time in proportion to
the members a record shares an identifier with, not the pool size, and yields
the same pool as comparing every pair.

Member identifiers change only through the pool: every metadata merge
re-indexes the member it enriched.
"""

from __future__ import annotations

from citemesh.core import Paper
from citemesh.strategies.candidates import (
    CandidateIdentityKey,
    candidate_identifiers_conflict,
    candidate_identity_keys,
    candidate_records_match,
    merge_paper_metadata,
    merge_seed_relation,
)


class DiscoveryPool:
    """Deduplicated hybrid candidates keyed by their surviving paper IDs."""

    def __init__(self, seed: Paper, seed_relations: dict[str, str]) -> None:
        """Start an empty pool for one seed.

        :param Paper seed: Canonical seed; records matching it enrich it instead.
        :param Dict[str, str] seed_relations: Relation labels by paper ID, shared
            with the caller, which keeps the seed's own label there.
        """
        self.seed = seed
        self.seed_relations = seed_relations
        self.papers: dict[str, Paper] = {}
        self.sources: dict[str, set[str]] = {}
        self.source_ranks: dict[str, dict[str, int]] = {}
        self.provider_aliases: dict[str, str] = {}
        self._members_by_key: dict[CandidateIdentityKey, dict[str, None]] = {}
        self._keys_by_member: dict[str, tuple[CandidateIdentityKey, ...]] = {}

    def add(self, incoming: Paper, *, source: str, relation: str) -> str | None:
        """Add one candidate, reconciling it with the seed and existing members.

        An exact provider ID or a compatible recorded alias selects a member
        directly. Otherwise a unique identifier match joins that member, and a
        local-corpus record takes over the matched member's ID and evidence.

        :param Paper incoming: Newly observed candidate payload.
        :param str source: Candidate source tag.
        :param str relation: Relation-to-seed label.
        :return Optional[str]: Surviving member ID, or ``None`` for a record with
            an empty ID or one matching the seed.
        """
        incoming_id = str(incoming.paper_id).strip()
        if not incoming_id:
            return None
        if incoming_id == self.seed.paper_id or candidate_records_match(
            incoming, self.seed
        ):
            merge_paper_metadata(self.seed, incoming)
            return None

        canonical_id = incoming_id
        existing = self.papers.get(incoming_id)
        if existing is None and not incoming.is_local_corpus:
            alias_target = self.provider_aliases.get(incoming_id)
            if (
                alias_target is not None
                and alias_target in self.papers
                and not candidate_identifiers_conflict(
                    self.papers[alias_target], incoming
                )
            ):
                canonical_id = alias_target
                existing = self.papers[canonical_id]
        if existing is not None:
            self.merge(canonical_id, incoming)
        else:
            matched_id = self.unique_match(incoming)
            if matched_id is None:
                self.papers[canonical_id] = incoming
                self._index(canonical_id)
            elif incoming.is_local_corpus:
                self._replace_member(matched_id, canonical_id, incoming)
            else:
                canonical_id = matched_id
                self.merge(matched_id, incoming)
                self.provider_aliases[incoming_id] = canonical_id

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

    def record_rank(self, member_id: str, source: str, rank: int) -> None:
        """Keep the first provider rank observed for a member in one source.

        :param str member_id: Member ID.
        :param str source: Ranked provider list.
        :param int rank: One-based position in that list.
        :return None: Leaves an earlier rank for the same source unchanged.
        """
        self.source_ranks.setdefault(member_id, {}).setdefault(source, rank)

    def _replace_member(
        self, matched_id: str, incoming_id: str, incoming: Paper
    ) -> None:
        """Move a matched member's evidence onto an incoming local-corpus record.

        Later lists can carry the matched provider ID without the external IDs
        that established the corpus match, so that ID becomes an alias.

        :param str matched_id: Provider-record member being replaced.
        :param str incoming_id: Local-corpus ID that becomes the member ID.
        :param Paper incoming: Local-corpus record that absorbs the member.
        :return None: Updates papers, tags, ranks, aliases, relations, and index.
        """
        existing = self.papers.pop(matched_id)
        self._unindex(matched_id)
        self.papers[incoming_id] = merge_paper_metadata(incoming, existing)
        self._index(incoming_id)
        self.sources[incoming_id] = self.sources.pop(matched_id, set())
        for provider_id, target in self.provider_aliases.items():
            if target == matched_id:
                self.provider_aliases[provider_id] = incoming_id
        self.provider_aliases[matched_id] = incoming_id
        previous_ranks = self.source_ranks.pop(matched_id, None)
        if previous_ranks is not None:
            self.source_ranks[incoming_id] = previous_ranks
        previous_relation = self.seed_relations.pop(matched_id, "")
        if previous_relation:
            self.seed_relations[incoming_id] = previous_relation

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
