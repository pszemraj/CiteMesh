"""Seed bibliography recovery without live providers or embedding models."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from citemesh.core import Paper
from citemesh.services import SemanticScholarUnavailableError
from citemesh.services.arxiv import ArxivClient
from citemesh.services.semantic_scholar.disk_cache import (
    _persist_paper,
    _reference_cache_path,
)
from citemesh.strategies.candidates import (
    CandidateAcquisitionError,
    fetch_candidate_pool,
    fetch_seed_references,
    require_available_candidate_source,
)
from citemesh.strategies.citation import CitationGraphBuilder


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


@pytest.mark.parametrize("limit", [0, 2])
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
    results = fetch_seed_references(client, paper("arxiv:2608.27147"), limit)
    bibliography.assert_not_called()
    metadata.assert_not_called()
    if limit:
        assert results[0].papers[0].paper_id == "reference"
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
    bibliography.return_value = [("arxiv:1706.03762",)]
    metadata.return_value = {"arxiv:1706.03762": paper("arxiv:1706.03762")}
    with caplog.at_level(logging.INFO):
        results = fetch_seed_references(
            client,
            paper("seed", arxiv_id="2608.27147"),
            2,
            seed_identifier="arxiv:2608.27147v1",
        )
        require_available_candidate_source(results, context="test")
    bibliography.assert_called_once_with("2608.27147v1")
    assert [result.state.value for result in results] == [
        "unavailable" if failed else "empty",
        "complete",
    ]
    assert results[1].papers[0].paper_id == "arxiv:1706.03762"
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
    assert "Recovered 1 references" in caplog.text


def test_local_and_disk_metadata_avoid_remote_metadata(providers: tuple) -> None:
    """An entry's cached arXiv/DOI aliases do not cause duplicate requests.

    :param tuple providers: Mock services.
    :return None: Checks local-first resolution and known citation count.
    """
    client, bibliography, metadata = providers
    bibliography.return_value = [
        ("arxiv:1706.03762", "10.1234/test"),
        ("10.1234/test",),
        ("arxiv:2303.08774",),
    ]
    local = paper(
        "local-record", arxiv_id="1706.03762", doi="10.1234/test", is_local_corpus=True
    )
    cached = paper("s2-record", arxiv_id="2303.08774", citation_count=42)
    _persist_paper(cached, "arxiv:2303.08774")
    lookup = MagicMock(return_value={"arxiv:1706.03762": local})
    results = fetch_seed_references(
        client, paper("arxiv:2608.27147"), 3, local_lookup=lookup
    )
    assert [item.paper_id for item in results[-1].papers] == [
        "local-record",
        "s2-record",
    ]
    assert results[-1].papers[1].citation_count == 42
    client.get_papers.assert_not_called()
    metadata.assert_not_called()


def test_unresolved_entries_do_not_exhaust_admission_limit(providers: tuple) -> None:
    """Missing DOI records are skipped and later bibliography entries are tried.

    :param tuple providers: Mock services.
    :return None: Checks ordering, deduplication, and bounded graph admission.
    """
    client, bibliography, metadata = providers
    bibliography.return_value = [
        ("10.1234/missing",),
        ("arxiv:1706.03762",),
        ("arxiv:1706.03762",),
        ("arxiv:2303.08774",),
        ("arxiv:2401.12345",),
    ]
    metadata.side_effect = lambda ids: {value: paper(value) for value in ids}
    results = fetch_seed_references(client, paper("arxiv:2608.27147"), 2)
    assert [item.paper_id for item in results[-1].papers] == [
        "arxiv:1706.03762",
        "arxiv:2303.08774",
    ]
    metadata.assert_called_once()


@pytest.mark.parametrize("html", [None, []])
def test_html_failure_preserves_s2_failure_or_empty(
    providers: tuple, html: object
) -> None:
    """Unavailable and evaluated-empty HTML remain distinct outcomes.

    :param tuple providers: Mock services.
    :param object html: Unavailable or empty parsed HTML.
    :return None: Checks source availability semantics.
    """
    client, bibliography, _ = providers
    bibliography.return_value = html
    client.get_paper_references.side_effect = SemanticScholarUnavailableError("offline")
    results = fetch_seed_references(client, paper("arxiv:2608.27147"), 1)
    assert results[-1].state.value == ("unavailable" if html is None else "empty")
    if html is None:
        with pytest.raises(CandidateAcquisitionError):
            require_available_candidate_source(results, context="test")
    else:
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
    bibliography.return_value = [("arxiv:1706.03762",)]
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
    bibliography.return_value = [("arxiv:1706.03762",), ("arxiv:2303.08774",)]
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


def test_embedding_candidate_pool_uses_reference_fallback(providers: tuple) -> None:
    """Candidate-mode embeddings share reference recovery and seed relations.

    :param tuple providers: Mock services.
    :return None: Checks the shared pool path.
    """
    client, bibliography, metadata = providers
    bibliography.return_value = [("arxiv:1706.03762",)]
    metadata.side_effect = lambda ids: {value: paper(value) for value in ids}
    pool = fetch_candidate_pool(client, paper("arxiv:2608.27147"), max_references=1)
    assert pool.seed_relations == {"arxiv:1706.03762": "referenced_by_seed"}
    assert pool.source_status == {"references": "empty", "arxiv_references": "complete"}


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
    bibliography.return_value = [("arxiv:1706.03762",)]
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
    bibliography.return_value = [(identifier,) for identifier in identifiers]
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
    bibliography.return_value = [("arxiv:1706.03762",)]
    client.get_papers.return_value = {
        "arxiv:1706.03762": paper("s2-id", citation_count=77)
    }
    results = fetch_seed_references(client, paper("arxiv:2608.27147"), 1)
    assert results[-1].papers[0].paper_id == "s2-id"
    assert results[-1].papers[0].citation_count == 77
    assert all(not call.args[0] for call in metadata.call_args_list)
