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


def test_resolve_names_the_seed_member_alias_or_nothing_in_add_order() -> None:
    """resolve checks the seed, the exact ID, an alias, then identifiers."""
    seed = Paper(paper_id="seed", title="Seed", year=2024, arxiv_id="2401.00001")
    pool = DiscoveryPool(seed)
    member = Paper(paper_id="a", title="A", year=2020, doi="10.1000/d")
    pool.add(member, source="reference", relation="referenced_by_seed")
    pool.add(
        Paper(paper_id="b", title="B", year=2020, doi="10.1000/d"),
        source="reference",
        relation="referenced_by_seed",
    )

    # A record of the seed that also carries the member's DOI joins the seed.
    bridge = Paper(
        paper_id="r", title="R", year=2020, arxiv_id="2401.00001", doi="10.1000/d"
    )
    assert pool.resolve(bridge) == "seed"
    assert pool.resolve(Paper(paper_id="a", title="A", year=2020)) == "a"
    # "b" merged into "a"; without its DOI it still resolves through the alias.
    assert pool.resolve(Paper(paper_id="b", title="B", year=2020)) == "a"
    assert pool.resolve(Paper(paper_id="c", title="C", year=2020, doi="10.1000/D")) == (
        "a"
    )
    assert pool.resolve(Paper(paper_id="n", title="N", year=2020)) is None
    assert pool.add(bridge, source="citation", relation="cites_seed") is None
    assert set(pool.papers) == {"a"}


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
    original = discovery_pool.publication_version_preprint

    def counting_match(left: Paper, right: Paper) -> Paper | None:
        calls.append((left.paper_id, right.paper_id))
        return original(left, right)

    monkeypatch.setattr(discovery_pool, "publication_version_preprint", counting_match)
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
    # Each side against the other, then the fold's orientation from each side.
    assert len(calls) == 4
    assert {frozenset(call) for call in calls} == {
        frozenset(paper.paper_id for paper in publication_pair)
    }


def _nature_pair() -> tuple[Paper, Paper]:
    """The arXiv record and the Nature record of one work, from a live build.

    Neither is a corpus row and they share no identifier. A later author's name
    differs only in how its accent is encoded; the rule reads the first author.
    """
    names = ["Lexin Zhou", "Lorenzo Pacchiardi"]
    preprint = Paper(
        paper_id="cdf2297d1988564f5cb5a9dcd3258683826578c8",
        title="General Scales Unlock AI Evaluation with Explanatory and Predictive Power",
        year=2025,
        arxiv_id="2503.06378",
        doi="10.48550/arXiv.2503.06378",
        venue="arXiv.org",
        authors=[Author(name=name) for name in [*names, "Fernando Mart'inez-Plumed"]],
    )
    publication = Paper(
        paper_id="fac69c9ed2f3f3a59de24146f937f8ff7d5d5861",
        title="General scales unlock AI evaluation with explanatory and predictive power",
        year=2026,
        doi="10.1038/s41586-026-10303-2",
        venue="Nature",
        authors=[Author(name=name) for name in [*names, "Fernando Martínez-Plumed"]],
    )
    return preprint, publication


@pytest.mark.parametrize("publication_first", [False, True])
def test_provider_publication_versions_fold_into_the_provider_preprint(
    publication_first: bool,
) -> None:
    """Without a corpus row, the record with the arXiv ID survives the fold."""
    preprint, publication = _nature_pair()
    assert not candidate_records_match(preprint, publication)
    pool = DiscoveryPool(_seed())
    order = (publication, preprint) if publication_first else (preprint, publication)
    for paper in order:
        if paper is preprint:
            pool.add(paper, source="citation", relation="referenced_by_seed")
        else:
            pool.add(paper, source="recommendation", relation="semantic_only")
            pool.record_recommendation(paper.paper_id, 7)

    assert pool.reconcile_publication_versions() == 1

    assert set(pool.papers) == {preprint.paper_id}
    assert pool.papers[preprint.paper_id].arxiv_id == "2503.06378"
    assert pool.sources[preprint.paper_id] == {"citation", "recommendation"}
    assert pool.seed_relations[preprint.paper_id] == "referenced_by_seed"
    assert pool.recommendation_ranks == {preprint.paper_id: 7}
    assert pool.provider_aliases == {publication.paper_id: preprint.paper_id}


def test_provider_publication_versions_need_the_same_first_author() -> None:
    """Equal titles alone never fold two records."""
    preprint, publication = _nature_pair()
    publication.authors = [Author(name="Another Researcher"), *publication.authors]
    pool = DiscoveryPool(_seed())
    for paper in (preprint, publication):
        pool.add(paper, source="semantic", relation="semantic_only")

    assert pool.reconcile_publication_versions() == 0
    assert len(pool.papers) == 2


def test_two_publications_of_one_provider_preprint_stay_separate() -> None:
    """A preprint matching two publications folds neither."""
    preprint, publication = _nature_pair()
    journal = Paper(
        paper_id="b" * 40,
        title=publication.title,
        year=2026,
        doi="10.1000/another-venue",
        authors=list(publication.authors),
    )
    pool = DiscoveryPool(_seed())
    for paper in (preprint, publication, journal):
        pool.add(paper, source="semantic", relation="semantic_only")

    assert pool.reconcile_publication_versions() == 0
    assert len(pool.papers) == 3
