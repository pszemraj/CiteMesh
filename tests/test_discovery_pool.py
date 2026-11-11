"""Indexed candidate-pool identity, takeover, and publication reconciliation."""

from __future__ import annotations

import pytest

from citemesh.core import Author, Paper
from citemesh.strategies import discovery_pool
from citemesh.strategies.candidates import candidate_records_match
from citemesh.strategies.discovery_pool import DiscoveryPool


def _seed() -> Paper:
    """Return a seed that shares no identifier with the test records."""
    return Paper(paper_id="seed", title="Seed", year=2024, is_seed=True)


def test_exact_check_runs_only_on_index_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new record is checked against the seed and members sharing an identifier."""
    calls: list[tuple[Paper, Paper]] = []

    def counting_match(left: Paper, right: Paper) -> bool:
        calls.append((left, right))
        return candidate_records_match(left, right)

    monkeypatch.setattr(discovery_pool, "candidate_records_match", counting_match)
    seed = _seed()
    pool = DiscoveryPool(seed)
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


def test_corpus_record_takes_over_its_provider_record() -> None:
    """A matching corpus row becomes the member and keeps the provider ID as alias."""
    pool = DiscoveryPool(_seed())
    provider = Paper(
        paper_id="a" * 40, title="Shared work", year=2023, arxiv_id="2301.00001"
    )
    pool.add(provider, source="citation", relation="cites_seed")
    pool.record_recommendation(provider.paper_id, 4)
    corpus = Paper(
        paper_id="arxiv:2301.00001",
        title="Shared work",
        year=2023,
        abstract="Corpus abstract",
        is_local_corpus=True,
    )

    assert pool.add(corpus, source="semantic", relation="semantic_only") == (
        corpus.paper_id
    )

    assert list(pool.papers) == [corpus.paper_id]
    assert pool.papers[corpus.paper_id] is corpus
    assert pool.sources == {corpus.paper_id: {"citation", "semantic"}}
    assert pool.seed_relations == {corpus.paper_id: "cites_seed"}
    assert pool.recommendation_ranks == {corpus.paper_id: 4}
    assert pool.provider_aliases == {provider.paper_id: corpus.paper_id}
    # A later list carrying only the provider ID resolves through the alias.
    bare = Paper(paper_id=provider.paper_id, title="Shared work", year=2023)
    assert pool.add(bare, source="semantic", relation="semantic_only") == (
        corpus.paper_id
    )


def test_conflicting_alias_and_ambiguous_match_stay_separate() -> None:
    """An alias with contradictory IDs and a two-way match both add new members."""
    pool = DiscoveryPool(_seed())
    first = Paper(paper_id="1" * 40, title="One", year=2023, doi="10.1000/shared")
    second = Paper(paper_id="2" * 40, title="Two", year=2023, arxiv_id="2301.00002")
    pool.add(first, source="citation", relation="cites_seed")
    pool.add(second, source="citation", relation="cites_seed")
    pool.provider_aliases["3" * 40] = first.paper_id
    conflicting = Paper(
        paper_id="3" * 40, title="Three", year=2023, doi="10.1000/different"
    )
    both = Paper(
        paper_id="4" * 40,
        title="Four",
        year=2023,
        doi="10.1000/shared",
        arxiv_id="2301.00002",
    )

    assert pool.add(conflicting, source="semantic", relation="semantic_only") == (
        conflicting.paper_id
    )
    assert pool.add(both, source="semantic", relation="semantic_only") == (
        both.paper_id
    )
    assert len(pool.papers) == 4


def test_seed_records_enrich_the_seed_and_seed_role_records_are_skipped() -> None:
    """Records naming the seed never become members."""
    seed = Paper(paper_id="arxiv:2401.00001", title="Seed", year=2024, is_seed=True)
    pool = DiscoveryPool(seed)
    alias = Paper(
        paper_id="5" * 40,
        title="Seed",
        year=2024,
        arxiv_id="2401.00001",
        abstract="Provider abstract",
    )
    seed_role = Paper(paper_id="6" * 40, title="Other", year=2024, is_seed=True)

    assert pool.add(alias, source="citation", relation="cites_seed") is None
    assert pool.add(seed_role, source="citation", relation="cites_seed") is None
    assert pool.papers == {}
    assert seed.abstract == "Provider abstract"


def _member(paper_id: str, year: int, refs: list[str]) -> Paper:
    """Build one titled pool member with reference IDs.

    :param str paper_id: Member ID.
    :param int year: Publication year.
    :param list[str] refs: Reference IDs carried by the record.
    :return Paper: Record authored by the shared first author.
    """
    return Paper(
        paper_id=paper_id,
        title="Measuring model capabilities",
        year=year,
        authors=[Author(name="Lexin Zhou")],
        references=refs,
    )


@pytest.fixture
def publication_pair() -> tuple[Paper, Paper]:
    """A corpus preprint and its publication without a shared external ID."""
    corpus = _member("arxiv:2503.06378", 2025, ["reference-a"])
    corpus.is_local_corpus = True
    corpus.title = "Measuring Model\nCapabilities"
    corpus.doi = "10.48550/arXiv.2503.06378"
    publication = _member("f" * 40, 2026, ["reference-b"])
    publication.doi = "10.1000/published"
    return corpus, publication


@pytest.mark.parametrize("publication_first", [False, True])
def test_publication_versions_fold_into_the_corpus_preprint(
    publication_pair: tuple[Paper, Paper], publication_first: bool
) -> None:
    """One work keeps one member, with the evidence of both versions."""
    corpus, publication = publication_pair
    assert not candidate_records_match(corpus, publication)
    seed = _seed()
    pool = DiscoveryPool(seed, {seed.paper_id: "seed"})
    order = (publication, corpus) if publication_first else (corpus, publication)
    for paper in order:
        if paper is corpus:
            pool.add(paper, source="semantic", relation="semantic_only")
            pool.record_recommendation(paper.paper_id, 9)
        else:
            pool.add(paper, source="citation", relation="referenced_by_seed")
            pool.record_recommendation(paper.paper_id, 2)
    pool.add(_member("other", 2020, []), source="semantic", relation="semantic_only")
    pool.provider_aliases["older-provider-alias"] = publication.paper_id

    assert pool.reconcile_publication_versions() == 1

    assert set(pool.papers) == {corpus.paper_id, "other"}
    assert corpus.arxiv_id == "2503.06378"
    assert corpus.doi == publication.doi
    assert corpus.references == ["reference-a", "reference-b"]
    assert pool.sources[corpus.paper_id] == {"citation", "semantic"}
    assert pool.seed_relations[corpus.paper_id] == "referenced_by_seed"
    assert publication.paper_id not in pool.seed_relations
    assert pool.recommendation_ranks == {corpus.paper_id: 2}
    assert pool.provider_aliases == {
        "older-provider-alias": corpus.paper_id,
        publication.paper_id: corpus.paper_id,
    }
    # A later record of the publication joins the surviving member.
    again = _member(publication.paper_id, 2026, [])
    assert pool.add(again, source="semantic", relation="semantic_only") == (
        corpus.paper_id
    )


@pytest.mark.parametrize(
    ("target", "field_name", "value"),
    [
        (1, "title", "Measuring different capabilities"),
        (1, "authors", [Author(name="Another Researcher")]),
        (1, "authors", [Author(name="L. Zhou")]),
        (1, "year", 2024),
        (1, "year", 2027),
        (1, "year", None),
        (1, "doi", ""),
        (1, "arxiv_id", "2503.06379"),
        (0, "doi", "10.1000/different-publication"),
        (0, "is_local_corpus", False),
        (1, "is_local_corpus", True),
    ],
)
def test_publication_versions_need_matching_metadata(
    publication_pair: tuple[Paper, Paper],
    target: int,
    field_name: str,
    value: object,
) -> None:
    """Metadata fallback cannot override distinct or insufficient identity evidence."""
    setattr(publication_pair[target], field_name, value)
    pool = DiscoveryPool(Paper(paper_id="seed", title="Seed", year=2024))
    for paper in publication_pair:
        pool.add(paper, source="semantic", relation="semantic_only")

    assert pool.reconcile_publication_versions() == 0
    assert set(pool.papers) == {paper.paper_id for paper in publication_pair}


@pytest.mark.parametrize("competing_publication", [False, True])
def test_ambiguous_publication_versions_stay_separate(
    publication_pair: tuple[Paper, Paper], competing_publication: bool
) -> None:
    """A second matching record anywhere in the pool makes the pair ambiguous."""
    corpus, publication = publication_pair
    other = _member(
        "a" * 40 if competing_publication else "arxiv:2503.06379",
        2026 if competing_publication else 2025,
        [],
    )
    other.is_local_corpus = not competing_publication
    if competing_publication:
        other.doi = "10.1000/other-publication"
    pool = DiscoveryPool(Paper(paper_id="seed", title="Seed", year=2024))
    for paper in (corpus, other, publication):
        pool.add(paper, source="semantic", relation="semantic_only")

    assert pool.reconcile_publication_versions() == 0
    assert len(pool.papers) == 3


def test_publication_reconciliation_checks_only_shared_titles(
    publication_pair: tuple[Paper, Paper], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The version rule runs within normalized-title groups, not over all pairs."""
    calls: list[tuple[str, str]] = []
    original = discovery_pool.corpus_publication_matches

    def counting_match(left: Paper, right: Paper) -> bool:
        calls.append((left.paper_id, right.paper_id))
        return original(left, right)

    monkeypatch.setattr(discovery_pool, "corpus_publication_matches", counting_match)
    pool = DiscoveryPool(Paper(paper_id="seed", title="Seed", year=2024))
    for number in range(2000):
        pool.add(
            Paper(
                paper_id=f"{number:040x}", title=f"Distinct work {number}", year=2024
            ),
            source="semantic",
            relation="semantic_only",
        )
    for paper in publication_pair:
        pool.add(paper, source="semantic", relation="semantic_only")

    assert pool.reconcile_publication_versions() == 1
    assert len(calls) == 4
