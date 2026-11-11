"""Seed bibliography recovery without live providers or embedding models."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from unittest.mock import MagicMock, call

import pytest

from citemesh.core import Paper
from citemesh.core.paper_ids import BibliographyKey
from citemesh.services import (
    SemanticScholarRequestError,
    SemanticScholarUnavailableError,
)
from citemesh.services.arxiv import ArxivClient, BibliographyEntry
from citemesh.services.semantic_scholar.disk_cache import (
    _load_cached_paper,
    _persist_paper,
    _reference_cache_path,
)
from citemesh.strategies.candidates import (
    CandidateAcquisitionError,
    fetch_seed_references,
    require_available_candidate_source,
)
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.discovery_pool import fetch_candidate_pool
from citemesh.strategies.hybrid import HybridGraphBuilder
from tests._helpers import offline_hybrid_encoder, stub_hybrid_corpus


@pytest.fixture
def providers(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Replace all remote reference and metadata reads.

    :param pytest.MonkeyPatch monkeypatch: Scoped patcher.
    :return tuple: S2 client, bibliography reader, arXiv metadata reader.
    """
    client = MagicMock()
    client.refresh_paper_cache = False
    client.get_paper_references.return_value = []
    client.get_paper_citations.return_value = []
    client.get_papers.return_value = {}
    client.get_reference_ids.return_value = []
    client.get_cached_papers.side_effect = lambda identifiers: {
        identifier: cached
        for identifier in identifiers
        if (cached := _load_cached_paper(identifier)) is not None
    }
    bibliography = MagicMock(return_value=[])
    metadata = MagicMock(return_value={})
    monkeypatch.setattr(ArxivClient, "get_bibliography", bibliography)
    monkeypatch.setattr(ArxivClient, "get_papers", metadata)
    return client, bibliography, metadata


def paper(identifier: str, **kwargs: object) -> Paper:
    """Construct an identified metadata record.

    :param str identifier: Primary paper identifier.
    :param object kwargs: Additional paper fields.
    :return Paper: Usable record.
    """
    return Paper(paper_id=identifier, title=identifier, year=2024, **kwargs)


@pytest.mark.parametrize("limit", [0, 2, 105])
def test_successful_s2_or_disabled_discovery_never_fetches_html(
    providers: tuple, limit: int
) -> None:
    """Usable S2 references and disabled discovery bypass the fallback.

    :param tuple providers: Mock services.
    :param int limit: Disabled or positive source limit.
    :return None: Checks provider calls.
    """
    client, bibliography, metadata = providers
    client.get_paper_references.return_value = [paper("reference")]
    title_lookup = MagicMock()
    selector = MagicMock(side_effect=lambda _seed, refs, cap: refs[:cap])
    results = fetch_seed_references(
        client,
        paper("arxiv:2608.27147"),
        limit,
        local_bibliography_lookup=title_lookup,
        reference_selector=selector,
    )
    title_lookup.assert_not_called()
    bibliography.assert_not_called()
    metadata.assert_not_called()
    selector.assert_not_called()
    if limit:
        assert results[0].papers[0].paper_id == "reference"
        client.get_paper_references.assert_called_once_with(
            "arxiv:2608.27147", limit=limit, raise_on_unavailable=True
        )
    else:
        assert results == ()
        client.get_paper_references.assert_not_called()


@pytest.mark.parametrize("failed", [False, True])
def test_empty_or_unavailable_s2_recovers_arxiv_metadata(
    providers: tuple, failed: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """ArXiv metadata recovery works independently of S2 availability.

    :param tuple providers: Mock services.
    :param bool failed: Simulated S2 outage.
    :param pytest.LogCaptureFixture caplog: Recorded log messages.
    :return None: Checks recovery, version, statuses, and logging.
    """
    client, bibliography, metadata = providers
    if failed:
        client.get_paper_references.side_effect = SemanticScholarUnavailableError(
            "offline"
        )
        client.get_papers.side_effect = SemanticScholarUnavailableError("offline")
    bibliography.return_value = [
        BibliographyEntry(("arxiv:1706.03762",)),
        BibliographyEntry((), BibliographyKey("Title only", "First Author", 2024)),
    ]
    metadata.return_value = {"arxiv:1706.03762": paper("arxiv:1706.03762")}
    title_lookup = MagicMock(return_value={})
    with caplog.at_level(logging.DEBUG):
        results = fetch_seed_references(
            client,
            paper("seed", arxiv_id="2608.27147"),
            2,
            seed_identifier="arxiv:2608.27147v1",
            local_bibliography_lookup=title_lookup,
        )
        require_available_candidate_source(results, context="test")
    assert title_lookup.call_count == int(not failed)
    bibliography.assert_called_once_with("2608.27147v1")
    assert [result.state.value for result in results] == [
        "unavailable" if failed else "empty",
        "complete",
    ]
    assert results[1].papers[0].paper_id == "arxiv:1706.03762"
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
    recovery_logs = [
        record
        for record in caplog.records
        if record.getMessage().startswith("Recovered 1 reference")
    ]
    assert len(recovery_logs) == 1
    assert recovery_logs[0].levelno == logging.INFO
    assert recovery_logs[0].getMessage() == (
        "Recovered 1 reference from the arXiv bibliography."
    )
    extraction_logs = [
        record
        for record in caplog.records
        if record.getMessage().startswith("arXiv bibliography extraction:")
    ]
    assert len(extraction_logs) == 1
    assert extraction_logs[0].levelno == logging.DEBUG


def test_local_metadata_is_enriched_and_disk_metadata_reused(providers: tuple) -> None:
    """Local metadata gains S2 counts while cached S2 metadata is reused.

    :param tuple providers: Mock services.
    :return None: Checks local-first resolution and citation-count enrichment.
    """
    client, bibliography, metadata = providers
    bibliography.return_value = [
        BibliographyEntry(("arxiv:1706.03762", "10.1234/test")),
        BibliographyEntry(("10.1234/test",)),
        BibliographyEntry(("arxiv:2303.08774",)),
    ]
    local = paper(
        "local-record", arxiv_id="1706.03762", doi="10.1234/test", is_local_corpus=True
    )
    overlapping_cache = paper("s2-overlap", arxiv_id="1706.03762", citation_count=99)
    cached = paper("s2-record", arxiv_id="2303.08774", citation_count=42)
    _persist_paper(overlapping_cache, "arxiv:1706.03762")
    _persist_paper(cached, "arxiv:2303.08774")
    lookup = MagicMock(return_value={"arxiv:1706.03762": local})
    results = fetch_seed_references(
        client, paper("arxiv:2608.27147"), 3, local_lookup=lookup
    )
    assert [item.paper_id for item in results[-1].papers] == [
        "local-record",
        "s2-record",
    ]
    assert results[-1].papers[0].citation_count == 99
    assert results[-1].papers[1].citation_count == 42
    client.get_papers.assert_not_called()
    assert [call.args[0] for call in client.get_cached_papers.call_args_list] == [
        ["arxiv:2303.08774"],
        ["arxiv:1706.03762"],
    ]
    metadata.assert_not_called()


def test_local_metadata_does_not_request_live_s2_enrichment(providers: tuple) -> None:
    """Optional enrichment cannot preempt later candidate discovery.

    :param tuple providers: Mock services.
    :return None: Checks local metadata remains usable after a rejected request.
    """
    client, bibliography, metadata = providers
    bibliography.return_value = [BibliographyEntry(("arxiv:1706.03762",))]
    local = paper("local-record", arxiv_id="1706.03762", is_local_corpus=True)
    client.get_papers.side_effect = SemanticScholarRequestError("rejected")
    results = fetch_seed_references(
        client,
        paper("arxiv:2608.27147"),
        1,
        local_lookup=MagicMock(return_value={"arxiv:1706.03762": local}),
    )
    assert results[-1].state.value == "complete"
    assert results[-1].papers == (local,)
    client.get_papers.assert_not_called()
    metadata.assert_not_called()


def test_refresh_enriches_local_metadata_from_live_s2(providers: tuple) -> None:
    """An explicit cache refresh also refreshes recovered local metadata.

    :param tuple providers: Mock services.
    :return None: Checks refresh preserves its documented live-read behavior.
    """
    client, bibliography, metadata = providers
    client.refresh_paper_cache = True
    bibliography.return_value = [BibliographyEntry(("arxiv:1706.03762",))]
    local = paper("local-record", arxiv_id="1706.03762", is_local_corpus=True)
    fresh = paper("s2-record", arxiv_id="1706.03762", citation_count=99)
    client.get_papers.return_value = {"arxiv:1706.03762": fresh}

    results = fetch_seed_references(
        client,
        paper("arxiv:2608.27147"),
        1,
        local_lookup=MagicMock(return_value={"arxiv:1706.03762": local}),
    )

    assert results[-1].papers[0].citation_count == 99
    client.get_papers.assert_called_once_with(
        ["arxiv:1706.03762"], raise_on_unavailable=True
    )
    client.get_cached_papers.assert_not_called()
    metadata.assert_not_called()


def test_title_recovery_appends_local_matches_before_reference_cap(
    providers: tuple,
) -> None:
    """Explicit IDs resolve first, with one cap for both forms of evidence."""
    client, bibliography, metadata = providers
    client.refresh_paper_cache = True
    explicit_id = "arxiv:1706.03762"
    seed = paper("arxiv:2608.27147", is_local_corpus=True)
    explicit = paper(explicit_id, is_local_corpus=True)
    first = paper("arxiv:2303.08774", is_local_corpus=True)
    second = paper("arxiv:2401.12345", is_local_corpus=True)
    first_key = BibliographyKey("First recovered work", "One Author", 2023)
    second_key = BibliographyKey("Second recovered work", "Another Author", 2024)
    duplicate_key = BibliographyKey("Explicit work", "Some Author", 2017)
    seed_key = BibliographyKey("Seed work", "Seed Author", 2026)
    bibliography.return_value = [
        BibliographyEntry((), first_key),
        BibliographyEntry((explicit_id,), duplicate_key),
        BibliographyEntry((), duplicate_key),
        BibliographyEntry((), first_key),
        BibliographyEntry((), seed_key),
        BibliographyEntry((), second_key),
    ]
    client.get_papers.return_value = {
        explicit_id: paper("s2-id", arxiv_id="1706.03762", citation_count=99)
    }

    def lookup(keys: Sequence[BibliographyKey]) -> dict[BibliographyKey, Paper]:
        assert keys == [first_key, duplicate_key, seed_key, second_key]
        client.get_papers.assert_called_once_with(
            [explicit_id], raise_on_unavailable=True
        )
        return {
            second_key: second,
            first_key: first,
            duplicate_key: explicit,
            seed_key: seed,
        }

    selector = MagicMock(return_value=[second])
    results = fetch_seed_references(
        client,
        seed,
        1,
        local_lookup=MagicMock(return_value={explicit_id: explicit}),
        local_bibliography_lookup=lookup,
        reference_selector=selector,
    )

    selector.assert_called_once_with(seed, [explicit, first, second], 1)
    assert explicit.citation_count == 99
    assert results[-1].papers == (second,)
    client.get_papers.assert_called_once_with([explicit_id], raise_on_unavailable=True)
    client.get_cached_papers.assert_not_called()
    metadata.assert_not_called()


def test_title_lookup_does_not_replace_unresolved_explicit_id(providers: tuple) -> None:
    """The title fallback cannot override an entry's explicit identifier."""
    client, bibliography, metadata = providers
    identifier = "arxiv:2401.00001"
    key = BibliographyKey("A Foundational Paper", "First Author", 2023)
    bibliography.return_value = [BibliographyEntry((identifier,), key)]
    title_lookup = MagicMock(return_value={key: paper("arxiv:2210.00001")})

    results = fetch_seed_references(
        client,
        paper("arxiv:2608.27147"),
        1,
        local_bibliography_lookup=title_lookup,
    )

    client.get_papers.assert_called_once_with([identifier], raise_on_unavailable=True)
    metadata.assert_called_once_with([identifier])
    title_lookup.assert_not_called()
    assert results[-1].papers == ()


@pytest.mark.parametrize("semantic_source", ["arxiv-corpus", "both"])
def test_hybrid_title_recovery_uses_selected_corpus_view(
    providers: tuple, monkeypatch: pytest.MonkeyPatch, semantic_source: str
) -> None:
    """Hybrid can admit a reference with no explicit identifier in the bibliography."""
    client, bibliography, metadata = providers
    seed = paper("arxiv:2608.27147", is_local_corpus=True)
    key = BibliographyKey("A Foundational Paper", "First Author", 2023)
    bibliography.return_value = [BibliographyEntry((), key)]
    client.get_recommended_papers.return_value = []
    builder = HybridGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=0,
        fetch_references=False,
        semantic_source=semantic_source,
        corpus_size=20,
        client=client,
    )
    engine = builder.embedding_builder
    stub_hybrid_corpus(builder, seed=seed)
    offline_hybrid_encoder(monkeypatch, builder)
    del engine.cached_bibliography_metadata
    engine._cache_selection = MagicMock()
    engine.embedding_cache = MagicMock()
    title_lookup = engine.embedding_cache.get_paper_metadata_for_bibliography
    reference_id = "arxiv:2210.00001"
    title_lookup.return_value = {
        key: {
            "paper_id": reference_id,
            "title": key.title,
            "year": 2022,
            "authors": [key.first_author],
            "arxiv_id": "2210.00001",
        }
    }
    selected = builder.collect_papers(seed.paper_id)

    title_lookup.assert_called_once_with([key], corpus_size=20)
    assert set(selected) == {seed.paper_id, reference_id}
    assert selected[reference_id].is_local_corpus
    assert builder.seed_relations[reference_id] == "referenced_by_seed"
    assert builder.paper_sources[reference_id] == "citation"
    # Recovery resolved the entry locally; the selected row is enriched afterwards.
    engine._enrich_corpus_papers.assert_called_once_with(
        {reference_id: selected[reference_id]}
    )
    client.get_papers.assert_not_called()
    metadata.assert_not_called()


def test_hydration_uses_provider_aliases_for_recovered_corpus_references(
    providers: tuple,
) -> None:
    """Use arXiv and DOI lookups while keeping graph and cache IDs local."""
    client, _, _ = providers
    seed = paper("content:seed", is_local_corpus=True)
    opaque = paper("content:opaque", is_local_corpus=True)
    local_arxiv = paper("content:arxiv", is_local_corpus=True, arxiv_id="2210.00001")
    local_doi = paper("content:doi", is_local_corpus=True, doi="10.1000/test")
    papers = {item.paper_id: item for item in (seed, opaque, local_arxiv, local_doi)}
    client.get_reference_ids.side_effect = [["arxiv-reference"], ["doi-reference"]]
    builder = CitationGraphBuilder(client=client)

    builder.hydrate_collected_references(papers, seed)

    assert client.get_reference_ids.call_args_list == [
        call("arxiv:2210.00001", force_refresh=False),
        call("10.1000/test", force_refresh=False),
    ]
    assert builder.reference_cache == {
        local_arxiv.paper_id: ["arxiv-reference"],
        local_doi.paper_id: ["doi-reference"],
    }
    assert not opaque.references


def test_local_alias_ambiguity_does_not_cross_bibliography_entries(
    providers: tuple,
) -> None:
    """Resolving one alias must not claim a separately ambiguous DOI entry.

    :param tuple providers: Mock services.
    :return None: Checks local lookup results remain keyed by requested alias.
    """
    client, bibliography, metadata = providers
    bibliography.return_value = [
        BibliographyEntry(("arxiv:1706.03762",)),
        BibliographyEntry(("10.1000/shared",)),
    ]
    local = paper(
        "local-a",
        arxiv_id="1706.03762",
        doi="10.1000/shared",
        is_local_corpus=True,
    )
    remote = paper("remote-doi", doi="10.1000/shared")
    client.get_papers.return_value = {"10.1000/shared": remote}

    results = fetch_seed_references(
        client,
        paper("arxiv:2608.27147"),
        2,
        local_lookup=MagicMock(return_value={"arxiv:1706.03762": local}),
    )

    # DOI identity deduplication keeps one graph record, but the ambiguous DOI
    # still required its own provider resolution instead of borrowing local-a.
    assert [item.paper_id for item in results[-1].papers] == ["local-a"]
    client.get_papers.assert_called_once_with(
        ["10.1000/shared"], raise_on_unavailable=True
    )
    metadata.assert_not_called()


def test_unresolved_entries_do_not_exhaust_admission_limit(
    providers: tuple, caplog: pytest.LogCaptureFixture
) -> None:
    """Recovery ranks all resolvable entries before applying its admission limit.

    :param tuple providers: Mock services.
    :param pytest.LogCaptureFixture caplog: Recorded recovery summary.
    :return None: Checks resolution, ranking, deduplication, and bounded admission.
    """
    client, bibliography, metadata = providers
    bibliography.return_value = [
        BibliographyEntry(("10.1234/missing",)),
        BibliographyEntry(("arxiv:1706.03762",)),
        BibliographyEntry(("arxiv:1706.03762",)),
        BibliographyEntry(("arxiv:2303.08774",)),
        BibliographyEntry(("arxiv:2401.12345",)),
    ]
    records = {
        "arxiv:1706.03762": Paper(
            paper_id="arxiv:1706.03762",
            title="Unrelated optical sensor calibration",
            year=2017,
            abstract="Laboratory measurements of imaging hardware.",
        ),
        "arxiv:2303.08774": Paper(
            paper_id="arxiv:2303.08774",
            title="Continual learning for frontier language models",
            year=2023,
            abstract="Sovereign AI systems retain knowledge across updates.",
            citation_count=1,
        ),
        "arxiv:2401.12345": Paper(
            paper_id="arxiv:2401.12345",
            title="Widely cited unrelated systems paper",
            year=2024,
            abstract="Distributed storage and transaction processing.",
            citation_count=75,
        ),
    }
    metadata.side_effect = lambda ids: {
        value: records[value] for value in ids if value in records
    }
    seed = Paper(
        paper_id="seed",
        title="Thomson: Continual Learning of Frontier Models for SovereignAI",
        year=2026,
        abstract="Continual learning updates sovereign frontier language models.",
        arxiv_id="2608.27147",
    )
    with caplog.at_level(logging.INFO):
        results = fetch_seed_references(client, seed, 2)
    assert [item.paper_id for item in results[-1].papers] == [
        "arxiv:2303.08774",
        "arxiv:2401.12345",
    ]
    recovery_logs = [
        record
        for record in caplog.records
        if record.getMessage().startswith("Ranked 3 recovered references")
    ]
    assert len(recovery_logs) == 1
    assert recovery_logs[0].levelno == logging.INFO
    assert recovery_logs[0].getMessage() == (
        "Ranked 3 recovered references from the arXiv bibliography, "
        "keeping 2 (reference limit: 2)."
    )
    metadata.assert_called_once()


def test_hybrid_recommended_reference_keeps_reference_evidence(
    providers: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recommendation of a recovered reference is one member with both tags."""
    client, bibliography, metadata = providers
    seed = paper("arxiv:2608.27147")
    admitted = paper("arxiv:1706.03762")
    omitted = paper("arxiv:2303.08774")
    rediscovered = paper("recommendation", arxiv_id="2303.08774")
    unrelated = paper("other-recommendation")
    client.get_paper.return_value = seed
    bibliography.return_value = [
        BibliographyEntry((admitted.paper_id,)),
        BibliographyEntry((omitted.paper_id,)),
        BibliographyEntry((), BibliographyKey("Title only", "First Author", 2024)),
    ]
    metadata.return_value = {ref.paper_id: ref for ref in (admitted, omitted)}
    client.get_recommended_papers.return_value = [rediscovered, unrelated]
    builder = HybridGraphBuilder(
        max_papers=3,
        max_citations=0,
        fetch_references=False,
        client=client,
    )
    title_lookup = MagicMock()
    monkeypatch.setattr(
        builder.embedding_builder, "cached_bibliography_metadata", title_lookup
    )
    offline_hybrid_encoder(monkeypatch, builder)

    papers = builder.collect_papers(seed.paper_id)

    pool = builder.discovery_pool
    assert pool is not None
    assert set(pool.papers) == {
        admitted.paper_id,
        omitted.paper_id,
        unrelated.paper_id,
    }
    assert len(papers) == 3
    assert pool.sources[omitted.paper_id] == {"citation", "semantic"}
    assert pool.provider_aliases == {rediscovered.paper_id: omitted.paper_id}
    assert pool.recommendation_ranks == {omitted.paper_id: 1, unrelated.paper_id: 2}
    assert builder.seed_relations[omitted.paper_id] == "referenced_by_seed"
    title_lookup.assert_not_called()


def test_hybrid_pools_every_provider_reference_once(
    providers: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full provider list joins the pool, one member per work."""
    client, bibliography, _ = providers
    seed = paper("arxiv:2608.27147", references=["shared"])
    seed_alias = paper("seed-alias", arxiv_id="2608.27147")
    skipped = paper(
        "early", doi="10.1000/early", citation_count=5, references=["shared"]
    )
    duplicate = paper("duplicate", doi=skipped.doi)
    admitted = paper("later", citation_count=10, references=["different"])
    rediscovered = paper("recommendation", doi=skipped.doi)
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = [
        seed_alias,
        skipped,
        duplicate,
        admitted,
    ]
    client.get_recommended_papers.return_value = [rediscovered]
    builder = HybridGraphBuilder(
        max_papers=3,
        semantic_source="both",
        fetch_references=False,
        client=client,
    )
    stub_hybrid_corpus(builder)
    offline_hybrid_encoder(monkeypatch, builder)

    papers = builder.collect_papers(seed.paper_id)

    assert set(papers) == {seed.paper_id, skipped.paper_id, admitted.paper_id}
    assert papers[admitted.paper_id].references == ["different"]
    assert builder.paper_sources == {
        seed.paper_id: "citation",
        skipped.paper_id: "both",
        admitted.paper_id: "citation",
    }
    assert builder.seed_relations[skipped.paper_id] == "referenced_by_seed"
    assert builder.discovery_pool is not None
    assert set(builder.discovery_pool.papers) == {skipped.paper_id, admitted.paper_id}
    assert builder.discovery_pool.recommendation_ranks == {skipped.paper_id: 1}
    assert builder.candidate_source_status == {
        "references": "complete",
        "citations": "empty",
        "recommendations": "complete",
        "arxiv_corpus": "empty",
    }
    client.get_paper_references.assert_called_once_with(
        seed.paper_id, limit=1000, raise_on_unavailable=True
    )
    client.get_reference_ids.assert_not_called()
    bibliography.assert_not_called()


@pytest.mark.parametrize("html", [None, []])
def test_unusable_arxiv_fallback_does_not_hide_s2_outage(
    providers: tuple, html: object
) -> None:
    """An unusable arXiv fallback cannot make an S2 outage available.

    :param tuple providers: Mock services.
    :param object html: Unavailable or empty parsed HTML.
    :return None: Checks source availability semantics.
    """
    client, bibliography, _ = providers
    bibliography.return_value = html
    client.get_paper_references.side_effect = SemanticScholarUnavailableError("offline")
    results = fetch_seed_references(client, paper("arxiv:2608.27147"), 1)
    assert results[-1].state.value == ("unavailable" if html is None else "empty")
    with pytest.raises(CandidateAcquisitionError, match="references"):
        require_available_candidate_source(results, context="test")


def test_metadata_outage_marks_recovery_unavailable(
    providers: tuple, caplog: pytest.LogCaptureFixture
) -> None:
    """Successful HTML extraction cannot hide metadata-provider outages.

    :param tuple providers: Mock services.
    :param pytest.LogCaptureFixture caplog: Recorded warning.
    :return None: Checks unavailable recovery status and one aggregate warning.
    """
    client, bibliography, metadata = providers
    bibliography.return_value = [BibliographyEntry(("arxiv:1706.03762",))]
    client.get_papers.side_effect = SemanticScholarUnavailableError("offline")
    metadata.return_value = None

    results = fetch_seed_references(client, paper("arxiv:2608.27147"), 1)
    assert [result.state.value for result in results] == ["empty", "unavailable"]
    with caplog.at_level(logging.WARNING):
        require_available_candidate_source(results, context="test")
    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "arxiv_references" in warnings[0].message


def test_citation_reuses_recovery_without_persisting_partial_bibliography(
    providers: tuple,
) -> None:
    """Cached S2 emptiness cannot prevent HTML recovery or trigger rehydration.

    :param tuple providers: Mock services.
    :return None: Checks seed reference reuse, statuses, caps, and cache isolation.
    """
    client, bibliography, metadata = providers
    seed = paper("arxiv:2608.27147")
    client.get_paper.return_value = seed
    client.get_cached_reference_ids.return_value = []
    cache_path = _reference_cache_path(seed.paper_id)
    cache_path.write_text('{"version":1,"paper_id":"arxiv:2608.27147","references":[]}')
    original = cache_path.read_bytes()
    bibliography.return_value = [
        BibliographyEntry(("arxiv:1706.03762",)),
        BibliographyEntry(("arxiv:2303.08774",)),
    ]
    metadata.side_effect = lambda ids: {value: paper(value) for value in ids}
    builder = CitationGraphBuilder(
        max_papers=2, max_references=5, max_citations=0, client=client
    )
    papers = builder.collect_papers(seed.paper_id)
    assert len(papers) == 2
    assert papers[seed.paper_id].references == []
    assert (
        papers[seed.paper_id].reference_overlap(
            paper("other", references=["arxiv:1706.03762"])
        )
        == 0
    )
    assert builder.seed_relations["arxiv:1706.03762"] == "referenced_by_seed"
    assert builder.candidate_source_status == {
        "references": "empty",
        "arxiv_references": "complete",
    }
    assert cache_path.read_bytes() == original
    assert all(
        call.args[0] != seed.paper_id
        for call in client.get_reference_ids.call_args_list
    )
    bibliography.assert_called_once()


@pytest.mark.parametrize("html", [None, []])
def test_citation_hydrates_seed_after_unavailable_arxiv_recovery(
    providers: tuple, html: object
) -> None:
    """A failed arXiv fallback must not suppress cached seed hydration.

    :param tuple providers: Mock services.
    :param object html: Unavailable or empty parsed HTML.
    :return None: Checks fallback failures retain seed bibliography enrichment.
    """
    client, bibliography, _ = providers
    seed = paper("arxiv:2608.27147")
    citation = paper("citation", references=["citation-reference"])
    client.get_paper.return_value = seed
    client.get_paper_references.side_effect = SemanticScholarUnavailableError("offline")
    client.get_paper_citations.return_value = [citation]
    client.get_reference_ids.return_value = ["reference-1", "reference-2"]
    bibliography.return_value = html
    builder = CitationGraphBuilder(
        max_papers=2, max_references=1, max_citations=1, client=client
    )

    papers = builder.collect_papers(seed.paper_id)

    assert papers[seed.paper_id].references == ["reference-1", "reference-2"]
    client.get_reference_ids.assert_called_once_with(seed.paper_id, force_refresh=False)
    assert builder.candidate_source_status == {
        "references": "unavailable",
        "arxiv_references": "unavailable" if html is None else "empty",
        "citations": "complete",
    }


def test_citation_hydrates_cached_seed_after_recovered_arxiv_references(
    providers: tuple,
) -> None:
    """Recovered candidates must not hide the seed's cached full bibliography.

    :param tuple providers: Mock services.
    :return None: Checks S2 outage recovery retains disk-cached seed coupling.
    """
    client, bibliography, metadata = providers
    seed = paper("arxiv:2608.27147")
    client.get_paper.return_value = seed
    client.get_paper_references.side_effect = SemanticScholarUnavailableError("offline")
    client.get_reference_ids.side_effect = lambda paper_id, **_kwargs: (
        ["cached-reference"] if paper_id == seed.paper_id else []
    )
    bibliography.return_value = [BibliographyEntry(("arxiv:1706.03762",))]
    metadata.return_value = {"arxiv:1706.03762": paper("arxiv:1706.03762")}
    builder = CitationGraphBuilder(
        max_papers=2, max_references=1, max_citations=0, client=client
    )

    papers = builder.collect_papers(seed.paper_id)

    assert papers[seed.paper_id].references == ["cached-reference"]
    assert any(
        call.args == (seed.paper_id,) and call.kwargs == {"force_refresh": False}
        for call in client.get_reference_ids.call_args_list
    )
    assert builder.candidate_source_status == {
        "references": "unavailable",
        "arxiv_references": "complete",
    }


def test_embedding_candidate_pool_uses_reference_fallback(providers: tuple) -> None:
    """Candidate-mode embeddings share reference recovery and seed relations.

    :param tuple providers: Mock services.
    :return None: Checks the shared pool path.
    """
    client, bibliography, metadata = providers
    bibliography.return_value = [
        BibliographyEntry(("arxiv:1706.03762",)),
        BibliographyEntry(("arxiv:2303.08774",)),
    ]
    metadata.side_effect = lambda ids: {value: paper(value) for value in ids}
    selector = MagicMock(
        side_effect=lambda _seed, recovered, limit: list(recovered)[-limit:]
    )
    pool = fetch_candidate_pool(
        client,
        paper("arxiv:2608.27147"),
        max_references=1,
        reference_selector=selector,
    )
    assert pool.seed_relations == {"arxiv:2303.08774": "referenced_by_seed"}
    assert pool.source_status == {"references": "empty", "arxiv_references": "complete"}
    assert len(selector.call_args.args[1]) == 2


def test_opaque_local_seed_does_not_attempt_provider_discovery(
    providers: tuple,
) -> None:
    """A local-only seed without external identity has no reference source.

    :param tuple providers: Mock services.
    :return None: Checks no spurious unavailable source.
    """
    client, bibliography, _ = providers
    assert (
        fetch_seed_references(client, paper("content:opaque", is_local_corpus=True), 2)
        == ()
    )
    client.get_paper_references.assert_not_called()
    bibliography.assert_not_called()


def test_refresh_paper_cache_bypasses_persisted_s2_metadata(providers: tuple) -> None:
    """Explicit refresh must not be short-circuited by the fallback cache read.

    :param tuple providers: Mock services.
    :return None: Checks refreshed metadata replaces the persisted value.
    """
    client, bibliography, _ = providers
    client.refresh_paper_cache = True
    stale = paper("s2-record", arxiv_id="1706.03762", citation_count=1)
    fresh = paper("s2-record", arxiv_id="1706.03762", citation_count=99)
    _persist_paper(stale, "arxiv:1706.03762")
    bibliography.return_value = [BibliographyEntry(("arxiv:1706.03762",))]
    client.get_papers.return_value = {"arxiv:1706.03762": fresh}
    results = fetch_seed_references(client, paper("arxiv:2608.27147"), 1)
    client.get_papers.assert_called_once()
    assert results[-1].papers[0].citation_count == 99


def test_small_admission_limit_batches_unresolved_metadata(providers: tuple) -> None:
    """A cap of one should not produce one paced request per bibliography row.

    :param tuple providers: Mock services.
    :return None: Checks batching independent of admission cap.
    """
    client, bibliography, metadata = providers
    identifiers = [f"arxiv:2401.{index:05d}" for index in range(20)]
    bibliography.return_value = [
        BibliographyEntry((identifier,)) for identifier in identifiers
    ]
    metadata.return_value = {identifiers[-1]: paper(identifiers[-1])}
    results = fetch_seed_references(client, paper("arxiv:2608.27147"), 1)
    client.get_papers.assert_called_once()
    metadata.assert_called_once_with(identifiers)
    assert len(results[-1].papers) == 1


def test_candidate_pool_preserves_requested_seed_version(providers: tuple) -> None:
    """The embedding candidate pool passes the original input to arXiv HTML.

    :param tuple providers: Mock services.
    :return None: Checks the versioned fallback request.
    """
    client, bibliography, _ = providers
    fetch_candidate_pool(
        client,
        paper("s2-seed", arxiv_id="2608.27147"),
        max_references=1,
        seed_identifier="arxiv:2608.27147v2",
    )
    bibliography.assert_called_once_with("2608.27147v2")


def test_s2_lookup_alias_preserves_metadata_without_external_ids(
    providers: tuple,
) -> None:
    """The requested identifier still resolves when S2 omits externalIds.

    :param tuple providers: Mock services.
    :return None: Checks that usable S2 metadata and citation counts are retained.
    """
    client, bibliography, metadata = providers
    bibliography.return_value = [BibliographyEntry(("arxiv:1706.03762",))]
    client.get_papers.return_value = {
        "arxiv:1706.03762": paper("s2-id", citation_count=77)
    }
    results = fetch_seed_references(client, paper("arxiv:2608.27147"), 1)
    assert results[-1].papers[0].paper_id == "s2-id"
    assert results[-1].papers[0].citation_count == 77
    assert all(not call.args[0] for call in metadata.call_args_list)
