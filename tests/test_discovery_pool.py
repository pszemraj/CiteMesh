"""Indexed discovery-pool identity matches the all-pairs reconciliation it replaced."""

from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import dataclass, field

import pytest

from citemesh.core import Author, Paper
from citemesh.strategies import discovery_pool
from citemesh.strategies.candidates import (
    candidate_identifiers_conflict,
    candidate_identity_keys,
    candidate_records_match,
    merge_paper_metadata,
    merge_seed_relation,
)
from citemesh.strategies.discovery_pool import DiscoveryPool

_S2_IDS = tuple(f"{number:040x}" for number in range(1, 121))
_ARXIV_IDS = (
    *(f"2401.{number:05d}" for number in range(1, 13)),
    "hep-th/9901001",
)
_DOIS = (
    *(f"10.1000/pub-{number}" for number in range(1, 11)),
    "10.1000/PUB-3",
    "https://doi.org/10.1000/pub-1",
    "10.48550/arXiv.2401.00002",
    "10.48550/arXiv.2401.00003v1",
)
_TAGS = (
    ("citation", "cites_seed"),
    ("citation", "referenced_by_seed"),
    ("citation", "citation"),
    ("semantic", "semantic_only"),
)
_RANK_SOURCES = (None, "recommendations", "arxiv_corpus")


@dataclass
class _AllPairsPool:
    """Pre-index hybrid ingest, comparing each record with every member."""

    seed: Paper
    seed_relations: dict[str, str]
    papers: dict[str, Paper] = field(default_factory=dict)
    sources: dict[str, set[str]] = field(default_factory=dict)
    source_ranks: dict[str, dict[str, int]] = field(default_factory=dict)
    provider_aliases: dict[str, str] = field(default_factory=dict)
    branches: Counter[str] = field(default_factory=Counter)

    def add(self, incoming: Paper, *, source: str, relation: str) -> str | None:
        """Reconcile one record exactly as hybrid ingest did before indexing."""
        incoming_id = str(incoming.paper_id).strip()
        if not incoming_id:
            return None
        if incoming_id == self.seed.paper_id or candidate_records_match(
            incoming, self.seed
        ):
            self.branches["seed"] += 1
            merge_paper_metadata(self.seed, incoming)
            return None

        canonical_id = incoming_id
        existing = self.papers.get(incoming_id)
        if existing is not None:
            self.branches["exact"] += 1
        if existing is None and not incoming.is_local_corpus:
            alias_target = self.provider_aliases.get(incoming_id)
            if alias_target is not None and alias_target in self.papers:
                if candidate_identifiers_conflict(self.papers[alias_target], incoming):
                    self.branches["alias_conflict"] += 1
                else:
                    self.branches["alias"] += 1
                    canonical_id = alias_target
                    existing = self.papers[canonical_id]
        if existing is not None:
            merge_paper_metadata(existing, incoming)
        else:
            matching_ids = [
                paper_id
                for paper_id, candidate in self.papers.items()
                if candidate_records_match(incoming, candidate)
            ]
            if len(matching_ids) == 1:
                matched_id = matching_ids[0]
                existing = self.papers[matched_id]
                if incoming.is_local_corpus:
                    self.branches["replace"] += 1
                    canonical_id = incoming_id
                    self.papers.pop(matched_id)
                    self.papers[canonical_id] = merge_paper_metadata(incoming, existing)
                    self.sources[canonical_id] = self.sources.pop(matched_id, set())
                    for provider_id, target in self.provider_aliases.items():
                        if target == matched_id:
                            self.provider_aliases[provider_id] = canonical_id
                    self.provider_aliases[matched_id] = canonical_id
                    previous_ranks = self.source_ranks.pop(matched_id, None)
                    if previous_ranks is not None:
                        self.source_ranks[canonical_id] = previous_ranks
                    previous_relation = self.seed_relations.pop(matched_id, "")
                    if previous_relation:
                        self.seed_relations[canonical_id] = previous_relation
                else:
                    self.branches["match"] += 1
                    canonical_id = matched_id
                    merge_paper_metadata(existing, incoming)
                    self.provider_aliases[incoming_id] = canonical_id
            else:
                self.branches["ambiguous" if matching_ids else "new"] += 1
                self.papers[canonical_id] = incoming

        self.sources.setdefault(canonical_id, set()).add(source)
        merged_relation = merge_seed_relation(
            self.seed_relations.get(canonical_id, ""), relation
        )
        if merged_relation:
            self.seed_relations[canonical_id] = merged_relation
        return canonical_id

    def unique_match(self, paper: Paper) -> str | None:
        """Return the only matching member found by an all-pairs scan."""
        matches = [
            paper_id
            for paper_id, candidate in self.papers.items()
            if candidate_records_match(paper, candidate)
        ]
        return matches[0] if len(matches) == 1 else None

    def record_rank(self, member_id: str, source: str, rank: int) -> None:
        """Keep the first rank per source, as hybrid ingest did."""
        self.source_ranks.setdefault(member_id, {}).setdefault(source, rank)


@dataclass(frozen=True)
class _Scenario:
    """One pseudo-random ingest sequence with interleaved match probes."""

    seed: Paper
    records: list[Paper]
    probes: list[Paper]
    tags: list[tuple[str, str]]
    rank_sources: list[str | None]


def _random_record(rng: random.Random, serial: int) -> Paper:
    """Build a record from small identifier pools so overlaps and conflicts recur."""
    local = rng.random() < 0.3
    if local:
        paper_id = rng.choice(
            [f"arxiv:{arxiv_id}" for arxiv_id in _ARXIV_IDS]
            + [f"local-{number}" for number in range(12)]
        )
    else:
        paper_id = rng.choice(
            [*_S2_IDS]
            + [f"arxiv:{arxiv_id}" for arxiv_id in _ARXIV_IDS]
            + [f"doi:{doi}" for doi in _DOIS[:10]]
        )
    return Paper(
        paper_id=paper_id,
        title=rng.choice(("Unknown", f"Title {serial}")),
        year=rng.choice((None, 2019, 2020, 2021)),
        authors=rng.choice(([], [Author(name=f"Author {serial}")])),
        citation_count=rng.randrange(50),
        abstract=rng.choice(("", f"Abstract {serial}")),
        venue=rng.choice(("", f"Venue {serial}")),
        arxiv_id=rng.choice(("", *_ARXIV_IDS, "2401.00001v2")),
        doi=rng.choice(("", *_DOIS)),
        categories=rng.choice(([], [f"cat-{serial % 3}"])),
        references=rng.choice(([], [f"reference-{serial}"])),
        is_local_corpus=local,
    )


def _scenario(rng_seed: int, count: int) -> _Scenario:
    """Generate records, repeated record objects, tags, ranks, and probes."""
    rng = random.Random(rng_seed)
    records: list[Paper] = []
    for serial in range(count):
        if records and rng.random() < 0.1:
            records.append(rng.choice(records))
        else:
            records.append(_random_record(rng, serial))
    return _Scenario(
        seed=Paper(
            paper_id=_S2_IDS[0],
            title="Seed",
            year=2022,
            arxiv_id="2401.00001",
            is_seed=True,
        ),
        records=records,
        probes=[_random_record(rng, count + serial) for serial in range(count)],
        tags=[rng.choice(_TAGS) for _ in records],
        rank_sources=[rng.choice(_RANK_SOURCES) for _ in records],
    )


def _ingest(
    pool: DiscoveryPool | _AllPairsPool, scenario: _Scenario
) -> list[tuple[str | None, str | None]]:
    """Ingest every record, recording survivors and an interleaved match probe."""
    trace: list[tuple[str | None, str | None]] = []
    for rank, (record, (source, relation), rank_source, probe) in enumerate(
        zip(
            scenario.records,
            scenario.tags,
            scenario.rank_sources,
            scenario.probes,
            strict=True,
        ),
        start=1,
    ):
        member_id = pool.add(record, source=source, relation=relation)
        if member_id is not None and rank_source is not None:
            pool.record_rank(member_id, rank_source, rank)
        trace.append((member_id, pool.unique_match(probe)))
    return trace


def _record_position(records: list[Paper], paper: Paper) -> int:
    """Return which input object survived as a member."""
    return next(index for index, record in enumerate(records) if record is paper)


@pytest.mark.parametrize("rng_seed", [0, 7, 20])
def test_indexed_ingest_matches_all_pairs_ingest(rng_seed: int) -> None:
    """Pools, tags, ranks, aliases, relations, and merges are identical."""
    scenario = _scenario(rng_seed, 300)
    indexed_scenario = copy.deepcopy(scenario)
    brute_scenario = copy.deepcopy(scenario)
    indexed = DiscoveryPool(
        indexed_scenario.seed, {indexed_scenario.seed.paper_id: "seed"}
    )
    brute = _AllPairsPool(brute_scenario.seed, {brute_scenario.seed.paper_id: "seed"})

    assert _ingest(indexed, indexed_scenario) == _ingest(brute, brute_scenario)

    assert list(indexed.papers.items()) == list(brute.papers.items())
    assert [
        _record_position(indexed_scenario.records, paper)
        for paper in indexed.papers.values()
    ] == [
        _record_position(brute_scenario.records, paper)
        for paper in brute.papers.values()
    ]
    assert list(indexed.sources.items()) == list(brute.sources.items())
    assert list(indexed.seed_relations.items()) == list(brute.seed_relations.items())
    assert indexed.provider_aliases == brute.provider_aliases
    assert indexed.source_ranks == brute.source_ranks
    # Every merge mutates records in place, so equal records mean equal merges.
    assert indexed_scenario.seed == brute_scenario.seed
    assert indexed_scenario.records == brute_scenario.records
    exercised = {
        "seed",
        "exact",
        "alias",
        "alias_conflict",
        "match",
        "replace",
        "ambiguous",
        "new",
    }
    assert {branch for branch, count in brute.branches.items() if count} == exercised


def test_exact_check_runs_only_on_index_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new record is checked against the seed and members sharing an identifier."""
    calls: list[tuple[Paper, Paper]] = []

    def counting_match(left: Paper, right: Paper) -> bool:
        calls.append((left, right))
        return candidate_records_match(left, right)

    monkeypatch.setattr(discovery_pool, "candidate_records_match", counting_match)
    seed = Paper(paper_id="seed", title="Seed", year=2024, is_seed=True)
    pool = DiscoveryPool(seed, {})
    members = [
        Paper(
            paper_id=f"{number:040x}",
            title=f"Member {number}",
            year=2020,
            doi=f"10.1000/member-{number}",
        )
        for number in range(200)
    ]
    for member in members:
        pool.add(member, source="citation", relation="cites_seed")
    assert [right for _left, right in calls] == [seed] * len(members)

    calls.clear()
    duplicate = Paper(
        paper_id="f" * 40, title="Duplicate", year=2020, doi="10.1000/MEMBER-57"
    )
    assert pool.add(duplicate, source="semantic", relation="semantic_only") == (
        members[57].paper_id
    )
    assert len(calls) == 2
    assert calls[0][0] is duplicate and calls[0][1] is seed
    assert calls[1][0] is duplicate and calls[1][1] is members[57]
    assert pool.provider_aliases == {duplicate.paper_id: members[57].paper_id}

    calls.clear()
    anonymous = Paper(paper_id="e" * 40, title="Anonymous", year=2021)
    assert pool.add(anonymous, source="semantic", relation="semantic_only") == (
        anonymous.paper_id
    )
    assert len(calls) == 1 and calls[0][1] is seed


def test_random_ingest_checks_only_members_sharing_an_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Across overlapping and conflicting records, every member check is an index hit."""
    scenario = _scenario(17, 300)
    pool = DiscoveryPool(scenario.seed, {scenario.seed.paper_id: "seed"})
    member_checks: list[bool] = []

    def counting_match(left: Paper, right: Paper) -> bool:
        if right is not pool.seed:
            shared = set(candidate_identity_keys(left)) & set(
                candidate_identity_keys(right)
            )
            member_checks.append(bool(shared))
        return candidate_records_match(left, right)

    monkeypatch.setattr(discovery_pool, "candidate_records_match", counting_match)
    _ingest(pool, scenario)

    assert member_checks
    assert all(member_checks)
    # An all-pairs scan of the same sequence compares each record with every member.
    assert len(member_checks) < len(scenario.records) * len(pool.papers) // 4
