"""Indexed candidate-pool identity, takeover, and publication reconciliation."""

from __future__ import annotations

import pytest

from citemesh.core import Author, Paper
from citemesh.services.semantic_scholar.endpoints import ReferenceEvidence
from citemesh.strategies import discovery_pool
from citemesh.strategies.candidates import candidate_records_match
from citemesh.strategies.discovery_pool import DiscoveryPool, build_discovery_leads
from citemesh.strategies.discovery_ranking import CandidateFeatures, RankedCandidate
from tests._helpers import build_publication_pair


def _focal() -> Paper:
    """Return a focal paper that shares no identifier with the test records."""
    return Paper(paper_id="focal", title="Focal", year=2024, is_focal=True)


def test_discovery_leads_use_evidence_from_the_omitted_pool() -> None:
    """Cap the two reading lists independently and keep their own rank meanings."""
    pool = DiscoveryPool(_focal())
    reference_counts = {
        "selected-reference": 7,
        "single-context": 1,
        "earlier-tie": 2,
        "later-tie": 2,
        "most-contexts": 4,
    }
    for identifier, count in reference_counts.items():
        pool.add(
            Paper(paper_id=identifier, title=identifier, year=2023),
            source="citation",
            relation="referenced_by_focal",
        )
        pool.record_reference_evidence(
            identifier,
            ReferenceEvidence(tuple(f"Passage {i}" for i in range(count)), True),
        )

    search_positions = {
        "recommended": 1,
        "direct-citer": 2,
        "selected-corpus": 3,
        "arxiv:2401.00001": 4,
        "arxiv:2401.00002": 6,
        "arxiv:2401.00003": 9,
    }
    for identifier, position in search_positions.items():
        pool.add(
            Paper(
                paper_id=identifier, title=identifier, year=2024, is_local_corpus=True
            ),
            source="semantic",
            relation="semantic_only",
        )
        pool.record_query_corpus_rank(identifier, position)
    pool.add(pool.papers["direct-citer"], source="citation", relation="cites_focal")
    pool.record_recommendation("recommended", 1)
    pool.add(
        Paper(paper_id="document-role-only", title="Document role only", year=2024),
        source="semantic",
        relation="semantic_only",
    )
    ranked = [
        RankedCandidate(
            identifier,
            position,
            0.0,
            CandidateFeatures(0.0, 0.0, None, False, None),
            {},
        )
        for position, identifier in enumerate(pool.papers, start=1)
    ]
    final_graph = {"selected-reference", "selected-corpus"}
    leads = build_discovery_leads(pool, ranked, final_graph, limit=2)

    assert leads is not None
    assert [
        (row["paper_id"], row["rank"], row["context_count"])
        for row in leads["references"]
    ] == [("earlier-tie", 3, 2), ("later-tie", 4, 2)]
    assert [(row["paper_id"], row["rank"]) for row in leads["corpus"]] == [
        ("arxiv:2401.00001", 4),
        ("arxiv:2401.00002", 6),
    ]
    assert "context_count" not in leads["corpus"][0]
    assert build_discovery_leads(pool, ranked, set(pool.papers), limit=2) is None
    assert build_discovery_leads(pool, ranked, final_graph, limit=0) is None


def test_query_position_follows_the_canonical_corpus_record() -> None:
    """An explicit alias fold preserves the earliest query retrieval position."""
    pool = DiscoveryPool(_focal())
    first = Paper(
        paper_id="10.1000/shared",
        doi="10.1000/shared",
        title="Shared work",
        year=2024,
        is_local_corpus=True,
    )
    second = Paper(
        paper_id="arxiv:2401.00001",
        doi="10.1000/shared",
        title="Shared work",
        year=2024,
        is_local_corpus=True,
    )
    for rank, paper in ((2, first), (5, second)):
        member_id = pool.add(paper, source="semantic", relation="semantic_only")
        assert member_id is not None
        pool.record_query_corpus_rank(member_id, rank)

    pool.absorb(second.paper_id, first.paper_id)

    assert pool.query_corpus_ranks == {second.paper_id: 2}


def test_exact_check_runs_only_on_index_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new record is checked against the focal paper and members sharing an identifier."""
    calls: list[tuple[Paper, Paper]] = []

    def counting_match(left: Paper, right: Paper) -> bool:
        calls.append((left, right))
        return candidate_records_match(left, right)

    monkeypatch.setattr(discovery_pool, "candidate_records_match", counting_match)
    focal = _focal()
    pool = DiscoveryPool(focal)
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
        pool.add(member, source="citation", relation="cites_focal")
    assert [right for _left, right in calls] == [focal] * len(members)

    calls.clear()
    duplicate = Paper(
        paper_id="f" * 40, title="Duplicate", year=2020, doi="10.1000/MEMBER-57"
    )
    assert pool.add(duplicate, source="semantic", relation="semantic_only") == (
        members[57].paper_id
    )
    assert len(calls) == 2
    assert calls[0][0] is duplicate and calls[0][1] is focal
    assert calls[1][0] is duplicate and calls[1][1] is members[57]
    assert pool.provider_aliases == {duplicate.paper_id: members[57].paper_id}

    calls.clear()
    anonymous = Paper(paper_id="e" * 40, title="Anonymous", year=2021)
    assert pool.add(anonymous, source="semantic", relation="semantic_only") == (
        anonymous.paper_id
    )
    assert len(calls) == 1 and calls[0][1] is focal


def test_corpus_record_takes_over_its_provider_record() -> None:
    """A matching corpus row becomes the member and keeps the provider ID as alias."""
    pool = DiscoveryPool(_focal())
    provider = Paper(
        paper_id="a" * 40, title="Shared work", year=2023, arxiv_id="2301.00001"
    )
    pool.add(provider, source="citation", relation="cites_focal")
    pool.record_recommendation(provider.paper_id, 4)
    evidence = ReferenceEvidence(("A focal reference excerpt",), False)
    pool.record_reference_evidence(provider.paper_id, evidence)
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
    assert pool.focal_relations == {corpus.paper_id: "cites_focal"}
    assert pool.recommendation_ranks == {corpus.paper_id: 4}
    assert pool.focal_reference_evidence == {corpus.paper_id: evidence}
    assert pool.provider_aliases == {provider.paper_id: corpus.paper_id}
    # A later list carrying only the provider ID resolves through the alias.
    bare = Paper(paper_id=provider.paper_id, title="Shared work", year=2023)
    assert pool.add(bare, source="semantic", relation="semantic_only") == (
        corpus.paper_id
    )


def test_conflicting_alias_and_ambiguous_match_stay_separate() -> None:
    """An alias with contradictory IDs and a two-way match both add new members."""
    pool = DiscoveryPool(_focal())
    first = Paper(paper_id="1" * 40, title="One", year=2023, doi="10.1000/shared")
    second = Paper(paper_id="2" * 40, title="Two", year=2023, arxiv_id="2301.00002")
    pool.add(first, source="citation", relation="cites_focal")
    pool.add(second, source="citation", relation="cites_focal")
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


def test_focal_records_enrich_the_focal_and_focal_role_records_are_skipped() -> None:
    """Records naming the focal paper never become members."""
    focal = Paper(paper_id="arxiv:2401.00001", title="Focal", year=2024, is_focal=True)
    pool = DiscoveryPool(focal)
    alias = Paper(
        paper_id="5" * 40,
        title="Focal",
        year=2024,
        arxiv_id="2401.00001",
        abstract="Provider abstract",
    )
    focal_role = Paper(paper_id="6" * 40, title="Other", year=2024, is_focal=True)

    assert pool.add(alias, source="citation", relation="cites_focal") is None
    assert pool.add(focal_role, source="citation", relation="cites_focal") is None
    assert pool.papers == {}
    assert focal.abstract == "Provider abstract"


def test_resolve_names_the_focal_member_alias_or_nothing_in_add_order() -> None:
    """resolve checks the focal paper, the exact ID, an alias, then identifiers."""
    focal = Paper(paper_id="focal", title="Focal", year=2024, arxiv_id="2401.00001")
    pool = DiscoveryPool(focal)
    member = Paper(paper_id="a", title="A", year=2020, doi="10.1000/d")
    pool.add(member, source="reference", relation="referenced_by_focal")
    pool.add(
        Paper(paper_id="b", title="B", year=2020, doi="10.1000/d"),
        source="reference",
        relation="referenced_by_focal",
    )

    # A record of the focal paper that also carries the member's DOI joins the focal paper.
    bridge = Paper(
        paper_id="r", title="R", year=2020, arxiv_id="2401.00001", doi="10.1000/d"
    )
    assert pool.resolve(bridge) == "focal"
    assert pool.resolve(Paper(paper_id="a", title="A", year=2020)) == "a"
    # "b" merged into "a"; without its DOI it still resolves through the alias.
    assert pool.resolve(Paper(paper_id="b", title="B", year=2020)) == "a"
    assert pool.resolve(Paper(paper_id="c", title="C", year=2020, doi="10.1000/D")) == (
        "a"
    )
    assert pool.resolve(Paper(paper_id="n", title="N", year=2020)) is None
    assert pool.add(bridge, source="citation", relation="cites_focal") is None
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
    return build_publication_pair()


@pytest.mark.parametrize("publication_first", [False, True])
def test_publication_versions_fold_into_the_corpus_preprint(
    publication_pair: tuple[Paper, Paper], publication_first: bool
) -> None:
    """One work keeps one member, with the evidence of both versions."""
    corpus, publication = publication_pair
    assert not candidate_records_match(corpus, publication)
    focal = _focal()
    pool = DiscoveryPool(focal, {focal.paper_id: "focal"})
    order = (publication, corpus) if publication_first else (corpus, publication)
    for paper in order:
        if paper is corpus:
            pool.add(paper, source="semantic", relation="semantic_only")
            pool.record_recommendation(paper.paper_id, 9)
        else:
            pool.add(paper, source="citation", relation="referenced_by_focal")
            pool.record_recommendation(paper.paper_id, 2)
    pool.add(_member("other", 2020, []), source="semantic", relation="semantic_only")
    pool.provider_aliases["older-provider-alias"] = publication.paper_id

    assert pool.reconcile_publication_versions() == 1

    assert set(pool.papers) == {corpus.paper_id, "other"}
    assert corpus.arxiv_id == "2503.06378"
    assert corpus.doi == publication.doi
    assert corpus.references == ["reference-a", "reference-b"]
    assert pool.sources[corpus.paper_id] == {"citation", "semantic"}
    assert pool.focal_relations[corpus.paper_id] == "referenced_by_focal"
    assert publication.paper_id not in pool.focal_relations
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
    pool = DiscoveryPool(Paper(paper_id="focal", title="Focal", year=2024))
    for paper in publication_pair:
        pool.add(paper, source="semantic", relation="semantic_only")

    assert pool.reconcile_publication_versions() == 0
    assert set(pool.papers) == {paper.paper_id for paper in publication_pair}


@pytest.mark.parametrize(
    ("preprint_author", "publication_author", "abstract_case", "folds"),
    [
        ("Liunian Harold Li", "L. Li", "same", 1),
        ("Liunian Harold Li", "L. H. Li", "same", 1),
        ("Ludwig van Beethoven", "L. van Beethoven", "same", 1),
        ("John Smith Jr.", "J. Smith Jr.", "same", 1),
        ("Jae O", "J. O", "same", 1),
        ("Jae O Jr.", "J. O Jr.", "same", 1),
        ("John Smith III", "J. Smith III", "same", 1),
        ("Liunian Harold Li", "L. Li", "nineteen", 0),
        ("Liunian Harold Li", "L. H. Li", "twenty", 1),
        ("Liunian Harold Li", "Liunian Harold Li", "nineteen", 1),
        ("Liunian Harold Li", "L. Li", "different", 0),
        ("Liunian Harold Li", "L. Li", "placeholder", 0),
        ("Liunian Harold Li", "H. Li", "same", 0),
        ("Liunian Harold Li", "L. J. Li", "same", 0),
        ("Liunian Harold Li", "H. L. Li", "same", 0),
        ("Liunian Harold Li", "L. H. L. Li", "same", 0),
        ("Liunian Harold Li", "L. Lu", "same", 0),
        ("Ludwig van Beethoven", "L. de Beethoven", "same", 0),
        ("John Smith Jr.", "J. Jr.", "same", 0),
        ("John Smith III", "J. III", "same", 0),
    ],
)
def test_abbreviated_publication_author_needs_matching_abstract(
    preprint_author: str, publication_author: str, abstract_case: str, folds: int
) -> None:
    """Fold an abbreviated publisher's first author with matching content."""
    abstract = (
        "We train a vision-and-language model using independently collected images "
        "and captions, then measure transfer to downstream visual reasoning "
        "and cross-modal retrieval tasks without parallel pretraining data."
    )
    if abstract_case == "placeholder":
        abstract = "No abstract available"
    elif abstract_case in {"nineteen", "twenty"}:
        abstract = " ".join(
            f"word{index}" for index in range(19 if abstract_case == "nineteen" else 20)
        )
    publication_abstract = abstract.replace(
        "vision-and-language", "vision and language"
    )
    if abstract_case == "different":
        publication_abstract = publication_abstract.replace("independently", "jointly")
    title = "Unsupervised Vision-and-Language Pre-training Without Parallel Images and Captions"
    preprint = Paper(
        paper_id="arxiv:2010.12831",
        arxiv_id="2010.12831",
        title=title,
        year=2020,
        authors=[Author(preprint_author)],
        abstract=abstract,
        is_local_corpus=True,
    )
    publication = Paper(
        paper_id="fc123daef8ab12025c2e5e6162a0c94370faa4c5",
        doi="10.18653/v1/2021.naacl-main.420",
        title=title,
        year=2021,
        authors=[Author(publication_author)],
        abstract=publication_abstract,
    )
    pool = DiscoveryPool(_focal())
    pool.add(preprint, source="semantic", relation="semantic_only")
    pool.add(publication, source="citation", relation="referenced_by_focal")
    pool.record_recommendation(publication.paper_id, 4)

    assert pool.reconcile_publication_versions() == folds
    if folds:
        assert list(pool.papers) == [preprint.paper_id]
        assert pool.focal_relations[preprint.paper_id] == "referenced_by_focal"
        assert pool.recommendation_ranks == {preprint.paper_id: 4}
    else:
        assert set(pool.papers) == {preprint.paper_id, publication.paper_id}


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
    pool = DiscoveryPool(Paper(paper_id="focal", title="Focal", year=2024))
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
    pool = DiscoveryPool(Paper(paper_id="focal", title="Focal", year=2024))
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
    pool = DiscoveryPool(_focal())
    order = (publication, preprint) if publication_first else (preprint, publication)
    for paper in order:
        if paper is preprint:
            pool.add(paper, source="citation", relation="referenced_by_focal")
        else:
            pool.add(paper, source="recommendation", relation="semantic_only")
            pool.record_recommendation(paper.paper_id, 7)

    assert pool.reconcile_publication_versions() == 1

    assert set(pool.papers) == {preprint.paper_id}
    assert pool.papers[preprint.paper_id].arxiv_id == "2503.06378"
    assert pool.sources[preprint.paper_id] == {"citation", "recommendation"}
    assert pool.focal_relations[preprint.paper_id] == "referenced_by_focal"
    assert pool.recommendation_ranks == {preprint.paper_id: 7}
    assert pool.provider_aliases == {publication.paper_id: preprint.paper_id}


def test_provider_publication_versions_need_the_same_first_author() -> None:
    """Equal titles alone never fold two records."""
    preprint, publication = _nature_pair()
    publication.authors = [Author(name="Another Researcher"), *publication.authors]
    pool = DiscoveryPool(_focal())
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
    pool = DiscoveryPool(_focal())
    for paper in (preprint, publication, journal):
        pool.add(paper, source="semantic", relation="semantic_only")

    assert pool.reconcile_publication_versions() == 0
    assert len(pool.papers) == 3
