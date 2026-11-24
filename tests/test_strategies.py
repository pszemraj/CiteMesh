"""Consolidated strategy behavior and determinism tests."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest
import requests

from citemesh.core import API_CONFIG, EMBEDDING_CONFIG, HYBRID_CONFIG, Author, Paper
from citemesh.core.paper_ids import arxiv_identifier_from_doi, normalize_paper_id
from citemesh.progress import progress_iterator
from citemesh.services import semantic_scholar as s2
from citemesh.services.semantic_scholar import SemanticScholarClient
from citemesh.services.semantic_scholar.endpoints import ReferenceBatchResult
from citemesh.strategies import hybrid as hybrid_strategy
from citemesh.strategies.base import (
    GraphBuilderStrategy,
    deterministic_sort_key,
    select_capped_undirected_edges,
)
from citemesh.strategies.candidates import (
    CandidateAcquisitionError,
    CandidateSourceResult,
    CandidateSourceState,
    candidate_identifiers_conflict,
    candidate_records_match,
    merge_focal_relation,
)
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.discovery_pool import DiscoveryPool, fetch_candidate_pool
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import EmbeddingInferenceError, HybridGraphBuilder
from citemesh.strategies.recommendation import RecommendationGraphBuilder
from citemesh.strategies.similarity import AbstractSimilarityIndex
from tests._helpers import (
    build_publication_pair,
    disable_embedding_dep_checks,
    http_response,
    offline_hybrid_encoder,
    stub_hybrid_corpus,
    word_hash_vectors,
)

pytestmark = pytest.mark.usefixtures("arxiv_unavailable")


@pytest.fixture(autouse=True)
def _disable_embedding_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch optional embedding dependency guards for strategy tests."""
    disable_embedding_dep_checks(monkeypatch)


class _Tagged:
    """Hashable identifier with stable string representation."""

    def __init__(self, label: str) -> None:
        """Create tagged identifier wrapper."""
        self.label = label

    def __hash__(self) -> int:
        return hash(self.label)

    def __str__(self) -> str:
        return self.label


def _paper(paper_id: str, year: int = 2020, refs: list[str] | None = None) -> Paper:
    """Build a deterministic paper fixture."""
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=year,
        abstract=f"Abstract {paper_id}",
        references=refs or [],
        citation_count=10,
    )


def _focal_paper(paper_id: str = "focal") -> Paper:
    """Build deterministic focal paper fixture."""
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=2024,
        abstract="focal abstract",
        is_focal=True,
    )


def _api_relation_record(paper_id: str) -> SimpleNamespace:
    """Build one SDK-shaped relation record.

    :param str paper_id: Semantic Scholar paper identifier.
    :return SimpleNamespace: Relation wrapper containing paper metadata.
    """
    return SimpleNamespace(
        paper={
            "paperId": paper_id,
            "title": f"Paper {paper_id}",
            "year": 2024,
            "abstract": f"Abstract {paper_id}",
            "citationCount": 10,
            "fieldsOfStudy": [],
            "authors": [],
        }
    )


def _recommendation_payload(paper_id: str) -> dict[str, object]:
    """Build one recommendation response record.

    :param str paper_id: Semantic Scholar paper identifier.
    :return dict[str, object]: REST-shaped paper metadata.
    """
    return {
        "paperId": paper_id,
        "title": f"Paper {paper_id}",
        "year": 2024,
        "abstract": f"Abstract {paper_id}",
        "citationCount": 10,
        "fieldsOfStudy": [],
        "authors": [],
    }


def _relation_response(key: str, paper_id: str) -> MagicMock:
    """Build one REST relation page containing a paper ID."""
    return http_response({"data": [{key: {"paperId": paper_id}}], "next": None})


def _reference_outage_transport(url: str, **_kwargs: object) -> MagicMock:
    """Fail reference requests while serving healthy citations and recommendations."""
    if url.endswith("/references"):
        raise requests.ConnectionError("offline")
    if url.endswith("/citations"):
        return _relation_response("citingPaper", "healthy-citation")
    return http_response(
        {"recommendedPapers": [_recommendation_payload("healthy-recommendation")]}
    )


def _cache_full_papers(*papers: Paper) -> None:
    """Persist complete paper records for real-client discovery tests.

    :param Paper papers: Full metadata records whose IDs will be rechecked upstream.
    :return None: Writes paper-cache entries through CiteMesh's cache helper.
    """
    for paper in papers:
        s2.disk_cache._persist_paper(paper, paper.paper_id)


def _identity_bridge_records() -> tuple[Paper, Paper, Paper]:
    """Build compatible S2/arXiv and DOI classes plus one bridging payload."""
    s2_id = "a" * 40
    arxiv_record = _paper(s2_id)
    arxiv_record.arxiv_id = "2508.12345"
    doi_record = _paper("10.1000/bridge")
    doi_record.doi = "10.1000/bridge"
    bridge = _paper(s2_id)
    bridge.arxiv_id = "2508.12345"
    bridge.doi = "10.1000/bridge"
    return arxiv_record, doi_record, bridge


def _make_constant_similarity_builder(
    builder_factory: Callable[[], object], monkeypatch: pytest.MonkeyPatch
) -> tuple[object, int]:
    """Build strategy with deterministic always-on edge creation for capping tests."""
    builder = builder_factory()
    if isinstance(builder, EmbeddingGraphBuilder):
        cap = builder.top_k
    elif isinstance(builder, HybridGraphBuilder):
        cap = 1
        monkeypatch.setattr(HYBRID_CONFIG, "max_edges_per_node", cap)
    else:
        cap = 3

    papers = {
        "focal": Paper(paper_id="focal", title="Focal", year=2024, abstract="focal"),
        "a": Paper(paper_id="a", title="A", year=2024, abstract="alpha"),
        "b": Paper(paper_id="b", title="B", year=2024, abstract="beta"),
        "c": Paper(paper_id="c", title="C", year=2024, abstract="gamma"),
        "d": Paper(paper_id="d", title="D", year=2024, abstract="delta"),
    }
    papers["focal"].is_focal = True

    def fake_collect_papers(self, focal_id: str, **kwargs: object) -> dict[str, Paper]:
        del focal_id
        del kwargs
        return papers

    def always_true(self, paper1: Paper, paper2: Paper, similarity: float) -> bool:
        del paper1, paper2, similarity
        return True

    def constant_similarity(self, paper1: Paper, paper2: Paper) -> float:
        del paper1, paper2
        return 1.0

    monkeypatch.setattr(builder, "prepare_graph_scoring", lambda _papers: None)

    builder.collect_papers = MethodType(fake_collect_papers, builder)
    builder.should_create_edge = MethodType(always_true, builder)
    builder.compute_similarity = MethodType(constant_similarity, builder)
    return builder, cap


def test_citation_and_recommendation_should_create_edge_thresholds() -> None:
    """Citation/recommendation strategies should use configured threshold only."""
    focal = Paper(paper_id="focal", title="Focal", year=2020, abstract="focal")
    focal.is_focal = True
    related = Paper(paper_id="related", title="Related", year=2021, abstract="related")
    other = Paper(paper_id="other", title="Other", year=2019, abstract="other")

    citation = CitationGraphBuilder(similarity_threshold=0.4, client=MagicMock())
    assert not citation.should_create_edge(focal, related, 0.39)
    assert citation.should_create_edge(focal, related, 0.4)
    assert not citation.should_create_edge(other, related, 0.39)
    assert citation.should_create_edge(other, related, 0.4)

    recommendation = RecommendationGraphBuilder(
        similarity_threshold=0.2, client=MagicMock()
    )
    assert recommendation.should_create_edge(focal, related, 0.2)
    assert not recommendation.should_create_edge(focal, related, 0.19)


@pytest.mark.parametrize(
    "builder_type",
    [
        CitationGraphBuilder,
        RecommendationGraphBuilder,
        EmbeddingGraphBuilder,
        HybridGraphBuilder,
    ],
)
@pytest.mark.parametrize("focal_year", [None, 2020])
def test_repeated_collection_preserves_client_owned_focal_role(
    builder_type: type[GraphBuilderStrategy],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    focal_year: int | None,
) -> None:
    """Reusing a focal paper's metadata as a later candidate must retain that candidate.

    :param type[GraphBuilderStrategy] builder_type: Public builder constructor.
    :param pytest.MonkeyPatch monkeypatch: Offline semantic inference stubs.
    :param pytest.LogCaptureFixture caplog: Captures focal paper notices.
    :param int | None focal_year: Known or unavailable publication year.
    :return None: Both collections have exactly one focal paper and preserve input roles.
    """
    first = replace(_paper("first"), title="Paper\n  first", year=focal_year)
    second = replace(_paper("second"), year=focal_year)
    caplog.set_level(logging.DEBUG)
    client = MagicMock()
    client.get_paper.side_effect = [first, second]

    def candidates(*_args: object, **_kwargs: object) -> list[Paper]:
        current_focal = first if client.get_paper.call_count == 1 else second
        suffix = f" ({focal_year})" if focal_year is not None else ""
        assert f"Focal paper: Paper {current_focal.paper_id}{suffix}" in caplog.messages
        return [first]

    client.get_paper_references.side_effect = candidates
    client.get_paper_citations.return_value = []
    client.get_recommended_papers.side_effect = candidates
    client.get_reference_ids_batch.return_value = ReferenceBatchResult({})
    builder = builder_type(max_papers=2, client=client)
    if isinstance(builder, HybridGraphBuilder):
        offline_hybrid_encoder(monkeypatch, builder)
    elif isinstance(builder, EmbeddingGraphBuilder):
        monkeypatch.setattr(builder, "_load_model", lambda: None)
        monkeypatch.setattr(
            builder,
            "_encode_texts",
            lambda *_args, **_kwargs: np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
        monkeypatch.setattr(
            builder,
            "embed_papers",
            lambda papers: {
                paper_id: np.asarray([1.0, 0.0], dtype=np.float32)
                for paper_id in papers
            },
        )
        monkeypatch.setattr(builder, "prepare_graph_scoring", lambda _papers: None)
        monkeypatch.setattr(builder, "compute_similarity", lambda *_papers: 1.0)

    first_collection = builder.collect_papers("first")
    builder.build_graph("second")
    second_collection = builder.papers

    assert first.is_focal is False
    assert second.is_focal is False
    assert first_collection["first"] is not first
    assert first_collection["first"].is_focal is True
    assert set(second_collection) == {"first", "second"}
    assert [p.paper_id for p in second_collection.values() if p.is_focal] == ["second"]
    suffix = f" ({focal_year})" if focal_year is not None else ""
    assert [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Focal paper:")
    ] == [f"Focal paper: Paper first{suffix}", f"Focal paper: Paper second{suffix}"]


@pytest.mark.parametrize("provided_focal", [False, True])
def test_embedding_focal_resolution_preserves_input_role(provided_focal: bool) -> None:
    """Prefetched and client-returned focal papers are copied for the build.

    :param bool provided_focal: Whether the caller supplies the metadata directly.
    :return None: Focal resolution leaves the supplied metadata reusable.
    """
    paper = _paper("focal")
    client = MagicMock()
    client.get_paper.return_value = paper
    builder = EmbeddingGraphBuilder(client=client)

    resolved = builder._resolve_focal_input("focal", paper if provided_focal else None)

    assert resolved is not paper
    assert resolved.is_focal is True
    assert paper.is_focal is False
    if provided_focal:
        client.get_paper.assert_not_called()


@pytest.mark.parametrize(
    "builder_type",
    [CitationGraphBuilder, RecommendationGraphBuilder, HybridGraphBuilder],
)
@pytest.mark.parametrize("unavailable", [False, True])
def test_required_focal_resolution_stops_before_candidate_acquisition(
    builder_type: type[GraphBuilderStrategy],
    unavailable: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Required focal resolution preserves missing-paper and service errors."""
    client = MagicMock()
    client.get_paper.return_value = None
    if unavailable:
        client.get_paper.side_effect = s2.SemanticScholarUnavailableError(
            "focal outage"
        )
    builder = builder_type(client=client)
    error_type = s2.SemanticScholarUnavailableError if unavailable else ValueError
    with caplog.at_level(logging.INFO), pytest.raises(error_type) as error:
        builder.collect_papers("unknown-focal")

    assert str(error.value) == (
        "focal outage"
        if unavailable
        else "Focal paper not found: unknown-focal (Semantic Scholar does not "
        "know this identifier; check the DOI/arXiv/S2 ID)."
    )
    client.get_paper.assert_called_once_with("unknown-focal", raise_on_unavailable=True)
    client.get_paper_references.assert_not_called()
    client.get_paper_citations.assert_not_called()
    client.get_recommended_papers.assert_not_called()
    assert not any(message.startswith("Focal paper:") for message in caplog.messages)


@pytest.mark.parametrize(
    "builder_type", [CitationGraphBuilder, RecommendationGraphBuilder]
)
@pytest.mark.parametrize("references", [[], ["unshared"]])
def test_indexed_graphs_require_topical_or_shared_reference_evidence(
    builder_type: type[GraphBuilderStrategy],
    references: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same-year popularity must not connect papers without topical evidence.

    :param type[GraphBuilderStrategy] builder_type: Indexed strategy constructor.
    :param list[str] references: Missing or disjoint reference payload.
    :param pytest.LogCaptureFixture caplog: Captured graph warnings.
    :return None: Assertions verify irrelevant edges are absent and supported ones remain.
    """
    focal = Paper(
        paper_id="focal",
        title="Algebra",
        abstract="Polynomial rings theorem",
        year=2024,
        citation_count=100,
        references=["shared"],
    )
    other = Paper(
        paper_id="other",
        title="Biology",
        abstract="Marine coral ecosystem",
        year=2024,
        citation_count=100,
        references=references,
    )
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [other]
    client.get_paper_citations.return_value = []
    client.get_recommended_papers.return_value = [other]
    client.get_reference_ids_batch.return_value = ReferenceBatchResult({})
    builder = builder_type(client=client, similarity_threshold=0.0)

    graph, _ = builder.build_graph("focal")

    assert set(graph) == {"focal", "other"}
    assert graph.number_of_edges() == 0
    assert "Graph contains no edges" in caplog.text
    assert builder.compute_similarity(focal, other) == 0.0
    other.references = ["shared"]
    caplog.clear()
    graph, _ = builder.build_graph("focal")
    assert graph.has_edge("focal", "other")
    assert "Graph contains no edges" not in caplog.text


@pytest.mark.parametrize(
    "builder_type", [CitationGraphBuilder, RecommendationGraphBuilder]
)
def test_reference_outage_stops_hydration_until_next_collection(
    builder_type: type[GraphBuilderStrategy],
) -> None:
    """One exhausted reference call skips remaining hydration and resets next build.

    :param type[GraphBuilderStrategy] builder_type: Indexed strategy constructor.
    :return None: Checks build-local outage state and malformed-response handling.
    """
    client = MagicMock()
    client.get_paper.return_value = _paper("focal", refs=["focal-ref"])
    related = [_paper("first"), _paper("second")]
    client.get_recommended_papers.return_value = related
    client.get_paper_references.return_value = related
    client.get_paper_citations.return_value = []
    client.get_reference_ids_batch.return_value = ReferenceBatchResult(
        {}, unavailable=True
    )
    client.get_cached_reference_ids.return_value = None
    builder = builder_type(client=client)

    papers = builder.collect_papers("focal")

    assert set(papers) == {"focal", "first", "second"}
    client.get_reference_ids_batch.assert_called_once_with(
        ["first", "second"], force_refresh=False
    )
    client.get_reference_ids_batch.return_value = ReferenceBatchResult(
        {"first": ["restored"], "second": ["restored"]}
    )

    papers = builder.collect_papers("focal")

    assert client.get_reference_ids_batch.call_count == 2
    assert papers["first"].references == ["restored"]
    assert papers["second"].references == ["restored"]
    for paper in related:
        paper.references = []
    client.get_reference_ids_batch.side_effect = ValueError("malformed references")
    with pytest.raises(ValueError, match="malformed references"):
        builder.collect_papers("focal")


@pytest.mark.parametrize(
    "builder_type", [CitationGraphBuilder, RecommendationGraphBuilder]
)
@pytest.mark.parametrize(
    ("refresh_reference_cache", "expected_references"),
    [(False, ["shared"]), (True, [])],
)
def test_reference_outage_reuses_persisted_hits_without_new_network_requests(
    builder_type: type[GraphBuilderStrategy],
    refresh_reference_cache: bool,
    expected_references: list[str],
) -> None:
    """A reference outage must retain later disk hits without retrying misses.

    :param type[GraphBuilderStrategy] builder_type: Indexed graph strategy constructor.
    :param bool refresh_reference_cache: Whether persisted reference reads are bypassed.
    :param list[str] expected_references: Expected warm-paper reference IDs.
    :return None: Verifies cache-only recovery after one exhausted request.
    """
    focal = _paper("focal", refs=["focal-reference"])
    cold = _paper("cold")
    warm = _paper("warm")

    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client.get_paper = MagicMock(return_value=focal)
        client._session.post = MagicMock(
            side_effect=requests.ConnectionError("offline")
        )
        client._persist_reference_cache_entry(
            s2.disk_cache._reference_cache_path("warm"), "warm", ["shared"]
        )
        if builder_type is CitationGraphBuilder:
            client.get_paper_references = MagicMock(return_value=[cold, warm])
            client.get_paper_citations = MagicMock(return_value=[])
            builder = CitationGraphBuilder(
                max_papers=3,
                max_references=2,
                max_citations=0,
                refresh_reference_cache=refresh_reference_cache,
                client=client,
            )
        else:
            client.get_recommended_papers = MagicMock(return_value=[cold, warm])
            builder = RecommendationGraphBuilder(
                max_papers=3,
                refresh_reference_cache=refresh_reference_cache,
                client=client,
            )

        with patch("time.sleep"):
            papers = builder.collect_papers("focal")

    assert set(papers) == {"focal", "cold", "warm"}
    assert papers["warm"].references == expected_references
    assert client._session.post.call_count == 1 + API_CONFIG.max_retries


@pytest.mark.parametrize(
    "builder_type", [CitationGraphBuilder, RecommendationGraphBuilder]
)
def test_candidate_scope_keeps_healthy_capabilities_after_reference_outage(
    builder_type: type[GraphBuilderStrategy],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reference outage must not suppress healthy discovery capabilities.

    :param type[GraphBuilderStrategy] builder_type: Indexed strategy constructor.
    :param Path tmp_path: Isolated reference-cache directory.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate cache paths.
    :return None: Checks healthy discovery and warm-cache reuse after an outage.
    """
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", tmp_path)
    focal = _paper("scope-focal")
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client.get_paper = MagicMock(return_value=focal)
        _cache_full_papers(_paper("healthy-citation"), _paper("healthy-recommendation"))

        client._session.get = MagicMock(side_effect=_reference_outage_transport)
        client._session.post = MagicMock(
            side_effect=requests.ConnectionError("offline")
        )
        if builder_type is CitationGraphBuilder:
            builder = CitationGraphBuilder(
                max_papers=3,
                max_references=1,
                max_citations=1,
                client=client,
            )
        else:
            builder = RecommendationGraphBuilder(max_papers=3, client=client)

        client._persist_reference_cache_entry(
            s2.disk_cache._reference_cache_path("warm"), "warm", ["cached-reference"]
        )
        with patch("time.sleep") as sleep_mock:
            for _ in range(2):
                with client.candidate_operation_scope():
                    papers = builder.collect_papers("scope-focal")
                    if builder_type is CitationGraphBuilder:
                        assert set(papers) == {"scope-focal", "healthy-citation"}
                        assert builder.candidate_source_status == {
                            "references": "unavailable",
                            "citations": "complete",
                        }
                    else:
                        assert set(papers) == {
                            "scope-focal",
                            "healthy-recommendation",
                        }
                        assert builder.candidate_source_status == {
                            "recommendations": "complete"
                        }
                    assert client.get_reference_ids("warm") == ["cached-reference"]

        assert sleep_mock.call_count >= 2 * API_CONFIG.max_retries
        assert client._session.post.call_count == 2 * (1 + API_CONFIG.max_retries)
        discovery_ref_requests = sum(
            call.args[0].endswith("/references")
            for call in client._session.get.call_args_list
        )
        assert discovery_ref_requests == (
            2 * (1 + API_CONFIG.max_retries)
            if builder_type is CitationGraphBuilder
            else 0
        )

    if builder_type is CitationGraphBuilder:
        assert (
            len(
                [
                    call
                    for call in client._session.get.call_args_list
                    if call.args[0].endswith("/citations")
                ]
            )
            == 2
        )
    else:
        assert (
            len(
                [
                    call
                    for call in client._session.get.call_args_list
                    if call.args[0].endswith("/citations")
                ]
            )
            == 0
        )
        assert (
            len(
                [
                    call
                    for call in client._session.get.call_args_list
                    if "/recommendations/" in call.args[0]
                ]
            )
            == 2
        )


def test_hybrid_reference_enrichment_gets_independent_retry_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hybrid reference discovery and reference hydration retry independently.

    :param Path tmp_path: Isolated reference-cache directory.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate cache paths.
    :return None: Verifies independent retry sequences and healthy candidate discovery.
    """
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", tmp_path)
    focal = _paper("hybrid-scope-focal")
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client.get_paper = MagicMock(return_value=focal)
        _cache_full_papers(_paper("healthy-citation"), _paper("healthy-recommendation"))

        client._session.get = MagicMock(side_effect=_reference_outage_transport)
        client._session.post = MagicMock(
            side_effect=requests.ConnectionError("offline")
        )
        builder = HybridGraphBuilder(
            max_papers=3,
            max_references=1,
            max_citations=1,
            semantic_source="candidates",
            client=client,
        )
        offline_hybrid_encoder(monkeypatch, builder)

        with patch("time.sleep"):
            papers = builder.collect_papers("hybrid-scope-focal")

        assert (
            len(
                [
                    call
                    for call in client._session.get.call_args_list
                    if call.args[0].endswith("/references")
                ]
            )
            == 1 + API_CONFIG.max_retries
        )
        assert client._session.post.call_count == 1 + API_CONFIG.max_retries
        assert set(papers) == {
            "hybrid-scope-focal",
            "healthy-citation",
            "healthy-recommendation",
        }
        assert builder.candidate_source_status == {
            "references": "unavailable",
            "citations": "complete",
            "recommendations": "complete",
        }

    assert (
        len(
            [
                call
                for call in client._session.get.call_args_list
                if call.args[0].endswith("/citations")
            ]
        )
        == 1
    )
    assert (
        len(
            [
                call
                for call in client._session.get.call_args_list
                if "/recommendations/" in call.args[0]
            ]
        )
        == 1
    )


def test_embedding_candidate_scope_keeps_healthy_later_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Embedding candidates retain citations and recommendations after a ref outage.

    :param Path tmp_path: Isolated reference-cache directory.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate cache paths.
    :return None: Verifies healthy source candidates and accurate status survive.
    """
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", tmp_path)
    focal = _paper("embedding-scope-focal")
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client.get_paper = MagicMock(return_value=focal)
        _cache_full_papers(_paper("healthy-citation"), _paper("healthy-recommendation"))

        client._session.get = MagicMock(side_effect=_reference_outage_transport)
        builder = EmbeddingGraphBuilder(
            max_papers=3,
            semantic_source="candidates",
            client=client,
        )
        monkeypatch.setattr(builder, "_load_model", lambda: None)
        monkeypatch.setattr(
            builder,
            "_encode_texts",
            lambda *_args, **_kwargs: np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
        monkeypatch.setattr(
            builder,
            "embed_papers",
            lambda papers: {
                paper_id: np.asarray([1.0, 0.0], dtype=np.float32)
                for paper_id in papers
            },
        )

        with patch("time.sleep"):
            papers = builder.collect_papers("embedding-scope-focal")

        assert (
            len(
                [
                    call
                    for call in client._session.get.call_args_list
                    if call.args[0].endswith("/references")
                ]
            )
            == 1 + API_CONFIG.max_retries
        )
        assert set(papers) == {
            "embedding-scope-focal",
            "healthy-citation",
            "healthy-recommendation",
        }
        assert builder.candidate_source_status == {
            "references": "unavailable",
            "citations": "complete",
            "recommendations": "complete",
        }

    assert (
        len(
            [
                call
                for call in client._session.get.call_args_list
                if call.args[0].endswith("/citations")
            ]
        )
        == 1
    )
    assert (
        len(
            [
                call
                for call in client._session.get.call_args_list
                if "/recommendations/" in call.args[0]
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    ("reference_papers", "expected_status"),
    [
        ([_paper("early-reference")], "complete"),
        ([], "empty"),
    ],
)
def test_candidate_pool_preserves_earlier_available_evidence_after_outage(
    reference_papers: list[Paper],
    expected_status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone pools retain evidence and continue after one source outage.

    :param list[Paper] reference_papers: Earlier reference-source result.
    :param str expected_status: Expected tri-state status for the references source.
    :param Path tmp_path: Isolated reference-cache directory.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate cache paths.
    :return None: Verifies a citation outage does not suppress recommendations.
    """
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", tmp_path)
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        _cache_full_papers(_paper("healthy-recommendation"))
        client.get_paper_references = MagicMock(return_value=reference_papers)

        def transport(url: str, **_kwargs: object) -> MagicMock:
            """Fail citations while serving the recommendation page."""
            if url.endswith("/citations"):
                raise requests.ConnectionError("offline")
            return http_response(
                {
                    "recommendedPapers": [
                        _recommendation_payload("healthy-recommendation")
                    ]
                }
            )

        client._session.get = MagicMock(side_effect=transport)

        with patch("time.sleep"):
            pool = fetch_candidate_pool(
                client,
                _focal_paper("partial-scope-focal"),
                max_references=1,
                max_citations=1,
                max_recommendations=1,
            )

        assert pool.source_status == {
            "references": expected_status,
            "citations": "unavailable",
            "recommendations": "complete",
        }
        assert set(pool.papers) == {
            *(paper.paper_id for paper in reference_papers),
            "healthy-recommendation",
        }
        assert (
            len(
                [
                    call
                    for call in client._session.get.call_args_list
                    if call.args[0].endswith("/citations")
                ]
            )
            == 1 + API_CONFIG.max_retries
        )

    assert (
        len(
            [
                call
                for call in client._session.get.call_args_list
                if "/recommendations/" in call.args[0]
            ]
        )
        == 1
    )


def test_recommendation_collect_filters_missing_abstract_and_prefers_payload_refs() -> (
    None
):
    """Recommendation collection should drop empty abstracts and keep provided refs."""
    with patch("citemesh.strategies.recommendation.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_paper.return_value = Paper(
            paper_id="focal", title="Focal", year=2020, abstract="focal abstract"
        )
        mock_client.get_recommended_papers.return_value = [
            Paper(
                paper_id="valid",
                title="Valid",
                year=2021,
                abstract="valid abstract",
                references=["r1", "r2"],
            ),
            Paper(
                paper_id="missing",
                title="Missing",
                year=2021,
                abstract="",
            ),
        ]
        mock_get_client.return_value = mock_client
        mock_client.get_reference_ids_batch.return_value = ReferenceBatchResult({})

        builder = RecommendationGraphBuilder(max_papers=3, fetch_references=True)
        papers = builder.collect_papers("focal")

    assert "valid" in papers
    assert "missing" not in papers
    assert papers["valid"].references == ["r1", "r2"]
    mock_client.get_reference_ids_batch.assert_called_once_with(
        ["focal"], force_refresh=False
    )


def test_recommendation_discovery_rechecks_ids_and_reuses_cached_enrichment() -> None:
    """Warm recommendation builds must honor live same-count membership changes.

    Recommendation rows carry their metadata, so no build sends a batch request.

    :return None: Verifies live ID order, inline metadata, and reference reuse.
    """
    focal = _focal_paper()
    first = _paper("first")
    second = _paper("second")
    calls: list[tuple[str, dict[str, object] | str, dict[str, object] | None]] = []
    recommendation_ids = iter([["first", "second"], ["second", "third"]])

    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        _cache_full_papers(focal, first, second)
        for paper_id in ("focal", "first", "second", "third"):
            client._persist_reference_cache_entry(
                s2.disk_cache._reference_cache_path(paper_id), paper_id, []
            )

        def get(
            url: str,
            *,
            params: dict[str, object],
            timeout: float,
        ) -> MagicMock:
            """Serve deterministic recommendation discovery through HTTP transport.

            :param str url: Requested Semantic Scholar endpoint URL.
            :param dict[str, object] params: Query parameters sent to the endpoint.
            :param float timeout: HTTP timeout, unused by this fixture.
            :return MagicMock: Fake endpoint response matching the request.
            """
            del timeout
            if url.endswith("/references"):
                raise AssertionError("warm reference enrichment must stay cached")
            calls.append((url, params, None))
            assert params == {
                "fields": ",".join(s2.payloads.DEFAULT_PAPER_FIELDS),
                "limit": 6,
            }
            return http_response(
                {
                    "recommendedPapers": [
                        _recommendation_payload(paper_id)
                        for paper_id in next(recommendation_ids)
                    ]
                }
            )

        client._session.get = MagicMock(side_effect=get)
        client._session.post = MagicMock(
            side_effect=AssertionError("recommendation metadata arrives inline")
        )
        builder = RecommendationGraphBuilder(max_papers=3, client=client)

        first_run = builder.collect_papers("focal")
        second_run = builder.collect_papers("focal")

    assert list(first_run) == ["focal", "first", "second"]
    assert list(second_run) == ["focal", "second", "third"]
    assert len(calls) == 2


def test_recommendation_fetches_primary_discovery_before_focal_references(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Recommendation acquisition must precede optional focal reference hydration.

    :return None: Verifies reference enrichment batches after candidate discovery.
    """
    client = MagicMock()
    events: list[str] = []
    client.get_paper.return_value = _focal_paper()

    def fetch_recommendations(*_args: object, **_kwargs: object) -> list[Paper]:
        """Record primary discovery before returning one candidate.

        :param object _args: Positional client arguments, unused by this fixture.
        :param object _kwargs: Keyword client arguments, unused by this fixture.
        :return list[Paper]: One recommendation candidate.
        """
        events.append("recommendations")
        return [_paper("candidate")]

    def fetch_references(
        paper_ids: list[str], *, force_refresh: bool
    ) -> ReferenceBatchResult:
        """Record reference hydration without making an upstream request.

        :param list[str] paper_ids: Papers selected for reference enrichment.
        :param bool force_refresh: Reference-cache bypass flag, expected disabled.
        :return ReferenceBatchResult: Empty reference lists for this fixture.
        """
        assert force_refresh is False
        assert "Checking reference lists" in caplog.text
        events.append(f"references:{','.join(paper_ids)}")
        return ReferenceBatchResult({})

    client.get_recommended_papers.side_effect = fetch_recommendations
    client.get_reference_ids_batch.side_effect = fetch_references

    with (
        caplog.at_level(logging.INFO),
        patch(
            "citemesh.strategies.recommendation.progress_iterator",
            wraps=progress_iterator,
        ) as progress,
    ):
        RecommendationGraphBuilder(max_papers=2, client=client).collect_papers("focal")

    assert events == ["recommendations", "references:focal,candidate"]
    assert [paper.paper_id for paper in progress.call_args.args[0]] == ["candidate"]
    assert progress.call_args.kwargs == dict(
        description="Checking recommendations", unit="papers", enabled=None
    )


def test_recommendation_collect_clamps_endpoint_request_limit() -> None:
    """Large recommendation graphs should stay within the endpoint's hard limit.

    :return None: Checks the oversampled request is capped at 500 results.
    """
    client = MagicMock()
    client.get_paper.return_value = _focal_paper()
    client.get_recommended_papers.return_value = []

    RecommendationGraphBuilder(
        max_papers=300,
        fetch_references=False,
        client=client,
    ).collect_papers("focal")

    client.get_recommended_papers.assert_called_once_with(
        "focal",
        limit=500,
        raise_on_unavailable=True,
    )


def test_citation_collect_populates_reference_cache_and_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Citation collection should hydrate references and expose summary."""
    focal = _paper("focal", refs=["focal-ref"])
    ref = _paper("ref1")
    cit = _paper("cit1")

    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [ref]
    client.get_paper_citations.return_value = [cit]
    client.get_reference_ids_batch.side_effect = lambda ids, **_kwargs: (
        ReferenceBatchResult({paper_id: [f"{paper_id}-ref"] for paper_id in ids})
    )

    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=True,
        client=client,
    )
    with caplog.at_level(logging.INFO):
        papers = builder.collect_papers("focal")

    assert set(papers) == {"focal", "ref1", "cit1"}
    assert papers["ref1"].references == ["ref1-ref"]
    assert papers["cit1"].references == ["cit1-ref"]
    assert builder.get_collection_summary() == "Collected 3 papers"
    client.get_reference_ids_batch.assert_called_once_with(
        ["ref1", "cit1"], force_refresh=False
    )
    assert "Hydrating reference lists" not in caplog.text
    assert "Collecting related papers" in caplog.text


def test_citation_hydration_batches_missing_references() -> None:
    """A reference batch contains selected papers missing inline reference lists."""
    focal = _paper("focal", refs=["focal-ref"])
    inline = _paper("inline", refs=["existing-reference"])
    targets = [_paper(f"paper-{index}") for index in range(45)]
    papers = {paper.paper_id: paper for paper in [focal, inline, *targets]}
    client = MagicMock()
    client.get_reference_ids_batch.return_value = ReferenceBatchResult({})
    builder = CitationGraphBuilder(fetch_references=True, client=client)

    builder.reference_hydrator.hydrate(papers, focal)

    client.get_reference_ids_batch.assert_called_once_with(
        [paper.paper_id for paper in targets], force_refresh=False
    )
    assert inline.references == ["existing-reference"]


def test_citation_collect_discovers_all_sources_before_reference_enrichment() -> None:
    """Optional reference hydration must not preempt requested candidate sources."""
    events: list[str] = []
    client = MagicMock()
    client.get_paper.return_value = _focal_paper()

    def references(*_args: object, **_kwargs: object) -> list[Paper]:
        events.append("discover references")
        return [_paper("ref1")]

    def citations(*_args: object, **_kwargs: object) -> list[Paper]:
        events.append("discover citations")
        return [_paper("cit1")]

    def reference_ids(paper_ids: list[str], **_kwargs: object) -> ReferenceBatchResult:
        events.append(f"enrich {','.join(paper_ids)}")
        return ReferenceBatchResult({})

    client.get_paper_references.side_effect = references
    client.get_paper_citations.side_effect = citations
    client.get_reference_ids_batch.side_effect = reference_ids

    papers = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=True,
        client=client,
    ).collect_papers("focal")

    assert set(papers) == {"focal", "ref1", "cit1"}
    assert events[:2] == ["discover references", "discover citations"]
    assert events[2:] == ["enrich focal,ref1,cit1"]


@pytest.mark.parametrize(
    ("external_fields", "provider_id"),
    [
        ({"arxiv_id": "2508.12345"}, "arxiv:2508.12345"),
        ({"doi": "10.1000/LOCAL.FOCAL"}, "10.1000/local.focal"),
    ],
)
def test_citation_collect_uses_external_alias_for_pre_resolved_local_focal(
    external_fields: dict[str, str],
    provider_id: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A local graph ID must remain local while provider calls use its alias.

    :param dict[str, str] external_fields: External identifier field on the focal paper.
    :param str provider_id: Canonical identifier expected by provider methods.
    :return None: Verifies every upstream call and the local reference-cache key.
    """
    focal = Paper(
        paper_id="local-source-key",
        title="Local focal",
        year=2024,
        abstract="Local abstract",
        is_local_corpus=True,
        **external_fields,
    )
    client = MagicMock()
    client.get_reference_ids_batch.return_value = ReferenceBatchResult(
        {provider_id: ["focal-reference"]}
    )
    client.get_paper_references.return_value = [_paper("reference")]
    client.get_paper_citations.return_value = [_paper("citation")]
    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=True,
        client=client,
    )

    with pytest.raises(TypeError, match="seed_paper"):
        builder.collect_papers(focal.paper_id, **{"seed_paper": focal})
    client.get_paper.assert_not_called()

    with caplog.at_level(logging.INFO):
        papers = builder.collect_papers(focal.paper_id, focal_paper=focal)

    assert set(papers) == {focal.paper_id, "reference", "citation"}
    assert papers[focal.paper_id].is_focal is True
    assert papers[focal.paper_id] is not focal
    assert focal.is_focal is False
    assert caplog.messages.count("Focal paper: Local focal (2024)") == 1
    assert builder.reference_hydrator.reference_cache[focal.paper_id] == [
        "focal-reference"
    ]
    client.get_paper.assert_not_called()
    client.get_reference_ids_batch.assert_called_once_with(
        [provider_id, "reference", "citation"], force_refresh=False
    )
    client.get_paper_references.assert_called_once_with(
        provider_id, limit=1, raise_on_unavailable=True
    )
    client.get_paper_citations.assert_called_once_with(
        provider_id, limit=1, raise_on_unavailable=True
    )


def test_citation_collect_keeps_opaque_pre_resolved_local_focal_off_provider() -> None:
    """An opaque corpus focal without an alias must produce a focal paper-only branch."""
    focal = Paper(
        paper_id="local-source-key",
        title="Local focal",
        year=2024,
        abstract="Local abstract",
        is_local_corpus=True,
    )
    client = MagicMock()
    for provider_method in (
        "get_paper",
        "get_reference_ids_batch",
        "get_paper_references",
        "get_paper_citations",
    ):
        getattr(client, provider_method).side_effect = AssertionError(
            "opaque local focal paper must not be sent to Semantic Scholar"
        )
    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=True,
        client=client,
    )

    papers = builder.collect_papers(focal.paper_id, focal_paper=focal)

    assert set(papers) == {focal.paper_id}
    assert papers[focal.paper_id].is_focal is True
    assert builder.reference_hydrator.reference_cache == {}
    assert builder.candidate_source_status == {}


def test_citation_collect_sizes_relation_requests_to_open_graph_slots() -> None:
    """Relation requests should honor capacity and let citations fill duplicate gaps.

    :return None: Checks remaining-slot limits and a full graph after a focal paper alias.
    """
    focal = _focal_paper("focal")
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [
        _paper("focal"),
        _paper("ref1"),
    ]
    client.get_paper_citations.return_value = [_paper("cit1")]

    papers = CitationGraphBuilder(
        max_papers=3,
        max_references=25,
        max_citations=25,
        fetch_references=False,
        client=client,
    ).collect_papers("focal")

    assert set(papers) == {"focal", "ref1", "cit1"}
    client.get_paper_references.assert_called_once_with(
        "focal",
        limit=2,
        raise_on_unavailable=True,
    )
    client.get_paper_citations.assert_called_once_with(
        "focal",
        limit=1,
        raise_on_unavailable=True,
    )

    focal_only_client = MagicMock()
    focal_only_client.get_paper.return_value = _focal_paper("focal-only")
    focal_only_papers = CitationGraphBuilder(
        max_papers=1,
        max_references=25,
        max_citations=25,
        fetch_references=False,
        client=focal_only_client,
    ).collect_papers("focal-only")

    assert set(focal_only_papers) == {"focal-only"}
    focal_only_client.get_paper_references.assert_not_called()
    focal_only_client.get_paper_citations.assert_not_called()


def test_citation_record_reaching_a_member_by_alias_merges_into_a_full_graph() -> None:
    """A row the pool resolves through an alias merges even with no free slot."""
    focal = _focal_paper("focal")
    client = MagicMock()
    client.get_paper.return_value = focal
    # "b" shares a DOI with "a", so it merges into "a" and becomes an alias.
    client.get_paper_references.return_value = [
        replace(_paper("a"), doi="10.1000/d"),
        replace(_paper("b"), doi="10.1000/d"),
    ]
    # "x" fills the last slot; "b" then returns without the DOI.
    client.get_paper_citations.return_value = [_paper("x"), _paper("b")]
    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=25,
        max_citations=25,
        fetch_references=False,
        client=client,
    )

    papers = builder.collect_papers("focal")

    assert set(papers) == {"focal", "a", "x"}
    assert builder.focal_relations["a"] == merge_focal_relation(
        "referenced_by_focal", "cites_focal"
    )


def test_recommendation_record_of_the_focal_merges_before_member_lookup() -> None:
    """A focal paper record that also matches a member enriches the focal paper, unfetched."""
    focal = replace(_focal_paper("focal"), arxiv_id="2001.00001")
    member = replace(_paper("m"), doi="10.1000/d")
    # Same arXiv ID as the focal paper and same DOI as the member.
    bridge = replace(_paper("r"), arxiv_id="2001.00001", doi="10.1000/d")
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_recommended_papers.return_value = [member, bridge]
    client.get_reference_ids_batch.return_value = ReferenceBatchResult(
        {"m": [], "focal": ["ref-of-focal"]}
    )

    papers = RecommendationGraphBuilder(max_papers=10, client=client).collect_papers(
        "focal"
    )

    assert set(papers) == {"focal", "m"}
    client.get_reference_ids_batch.assert_called_once_with(
        ["focal", "m"], force_refresh=False
    )
    assert papers["focal"].references == ["ref-of-focal"]


def test_recommendation_keeps_coupling_from_merged_alias() -> None:
    """An empty canonical list can use a merged record's references for an edge."""
    client = MagicMock()
    client.get_paper.return_value = replace(_focal_paper(), abstract="Protein folding")
    candidate = replace(_paper("candidate"), abstract="Matrix algebra", doi="10.1000/a")
    client.get_recommended_papers.return_value = [
        candidate,
        replace(candidate, paper_id="alias"),
    ]
    client.get_reference_ids_batch.side_effect = [
        ReferenceBatchResult({"focal": ["shared"], "candidate": []}),
        ReferenceBatchResult({"alias": ["shared"]}),
    ]

    graph, focal_id = RecommendationGraphBuilder(
        max_papers=3, client=client
    ).build_graph("focal")

    assert set(graph) == {"focal", "candidate"}
    assert graph.has_edge(focal_id, "candidate")


@pytest.mark.parametrize("local_takeover", [False, True])
def test_recommendation_hydrates_only_admitted_papers_missing_references(
    local_takeover: bool,
) -> None:
    """Batch aliases of admitted members with missing lists in ingestion order."""
    client = MagicMock()
    client.get_paper.return_value = _focal_paper()
    recommendations = [
        replace(_paper("a"), doi="10.1000/a"),
        replace(_paper("rejected"), abstract=""),
        replace(_paper("b"), doi="10.1000/b"),
        replace(_paper("b-alias"), doi="10.1000/b"),
        replace(_paper("b-alias-first-list"), doi="10.1000/b"),
        replace(_paper("b-alias-second-list"), doi="10.1000/b"),
        replace(_paper("inline", refs=["a-inline"]), doi="10.1000/a"),
        _paper("over-cap"),
    ]
    if local_takeover:
        recommendations.append(
            replace(
                _paper("local-a", refs=["a-local"]),
                doi="10.1000/a",
                is_local_corpus=True,
            )
        )
    client.get_recommended_papers.return_value = recommendations
    client.get_reference_ids_batch.side_effect = [
        ReferenceBatchResult({"focal": ["focal-reference"], "b": []}),
        ReferenceBatchResult(
            {
                "b-alias": [],
                "b-alias-first-list": ["first"],
                "b-alias-second-list": ["second"],
            }
        ),
    ]

    papers = RecommendationGraphBuilder(max_papers=3, client=client).collect_papers(
        "focal"
    )

    if local_takeover:
        assert set(papers) == {"focal", "local-a", "b"}
        assert set(papers["local-a"].references) == {"a-local", "a-inline"}
    else:
        assert list(papers) == ["focal", "a", "b"]
        assert papers["a"].references == ["a-inline"]
    assert papers["focal"].references == ["focal-reference"]
    assert papers["b"].references == ["first"]
    assert [
        request.args[0] for request in client.get_reference_ids_batch.call_args_list
    ] == [
        ["focal", "b"],
        ["b-alias", "b-alias-first-list", "b-alias-second-list"],
    ]


def test_citation_collect_preserves_focal_when_relation_reuses_focal_id() -> None:
    """Duplicate relation payloads should not unset the canonical focal paper marker."""
    focal = _paper("focal", refs=["focal-ref"])
    duplicate_focal_record = _paper("focal", year=2021)

    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [duplicate_focal_record]
    client.get_paper_citations.return_value = []

    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=0,
        fetch_references=False,
        similarity_threshold=0.0,
        client=client,
    )

    papers = builder.collect_papers("focal")
    assert papers["focal"].is_focal is True

    graph, focal_id = builder.build_graph("focal")
    assert focal_id == "focal"
    assert graph.nodes["focal"]["is_focal"] is True


def test_citation_collect_preserves_hydrated_references_for_overlap_duplicates() -> (
    None
):
    """Overlap duplicates should retain hydrated references on the canonical object."""
    focal = _paper("focal", refs=["focal-ref"])
    overlap_from_references = _paper("overlap")
    overlap_from_citations = _paper("overlap")

    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [overlap_from_references]
    client.get_paper_citations.return_value = [overlap_from_citations]
    client.get_reference_ids_batch.return_value = ReferenceBatchResult(
        {"overlap": ["overlap-ref"]}
    )

    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=True,
        client=client,
    )
    papers = builder.collect_papers("focal")

    assert papers["overlap"].references == ["overlap-ref"]
    assert builder.focal_relations["overlap"] == "overlap"
    client.get_reference_ids_batch.assert_called_once_with(
        ["overlap"], force_refresh=False
    )


def test_citation_collect_merges_external_primary_aliases() -> None:
    """Citation ingestion collapses external primary aliases into S2 records."""
    focal = _paper("s2-focal")
    focal.arxiv_id = "2508.12345"
    reference = _paper("s2-candidate")
    reference.arxiv_id = "2508.12346"
    citation_alias = _paper("arxiv:2508.12346")
    focal_alias = _paper("arxiv:2508.12345")

    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [focal_alias, reference]
    client.get_paper_citations.return_value = [citation_alias]

    builder = CitationGraphBuilder(
        max_papers=4,
        max_references=2,
        max_citations=1,
        fetch_references=False,
        client=client,
    )
    papers = builder.collect_papers("s2-focal")

    assert set(papers) == {"s2-focal", "s2-candidate"}
    assert papers["s2-focal"].is_focal is True
    assert builder.focal_relations == {
        "s2-focal": "focal",
        "s2-candidate": "overlap",
    }


def test_citation_collect_merges_s2_duplicates_with_shared_external_id() -> None:
    """Citation ingestion collapses S2 records with one agreed external ID."""
    focal = _paper("focal")
    first = _paper("1" * 40)
    first.doi = "10.1000/shared"
    second = _paper("2" * 40)
    second.doi = "10.1000/shared"

    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [first]
    client.get_paper_citations.return_value = [second]

    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=False,
        client=client,
    )
    papers = builder.collect_papers("focal")

    assert set(papers) == {"focal", first.paper_id}
    assert builder.focal_relations == {
        "focal": "focal",
        first.paper_id: "overlap",
    }


def test_citation_collect_keeps_bridge_records_distinct() -> None:
    """A bridge payload cannot merge separate primary IDs."""
    focal = _paper("focal")
    arxiv_record, doi_record, bridge = _identity_bridge_records()

    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [arxiv_record, doi_record, bridge]
    client.get_paper_citations.return_value = []

    builder = CitationGraphBuilder(
        max_papers=4,
        max_references=3,
        max_citations=0,
        fetch_references=False,
        client=client,
    )
    papers = builder.collect_papers("focal")

    assert set(papers) == {"focal", arxiv_record.paper_id, doi_record.paper_id}
    assert builder.focal_relations == {
        "focal": "focal",
        arxiv_record.paper_id: "referenced_by_focal",
        doi_record.paper_id: "referenced_by_focal",
    }


def test_citation_build_graph_persists_focal_relation_metadata() -> None:
    """Citation graph export metadata should preserve focal relation classes."""
    focal = _paper("focal", refs=["focal-ref"])
    ref = _paper("ref1")
    cit = _paper("cit1")

    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [ref]
    client.get_paper_citations.return_value = [cit]

    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=False,
        similarity_threshold=0.0,
        client=client,
    )
    graph, focal_id = builder.build_graph("focal")

    assert focal_id == "focal"
    assert graph.graph["strategy"] == "citation"
    assert graph.graph["focal_relations"] == {
        "cit1": "cites_focal",
        "ref1": "referenced_by_focal",
        "focal": "focal",
    }
    assert graph.graph["candidate_source_status"] == {
        "citations": "complete",
        "references": "complete",
    }


def test_recommendation_build_graph_persists_strategy_metadata() -> None:
    """Recommendation graphs should persist strategy metadata for downstream exports."""
    client = MagicMock()
    client.get_paper.return_value = _focal_paper()
    client.get_recommended_papers.return_value = [
        _paper("rec1", year=2023),
        _paper("rec2", year=2022),
    ]

    builder = RecommendationGraphBuilder(
        max_papers=3,
        fetch_references=False,
        similarity_threshold=0.0,
        client=client,
    )
    graph, focal_id = builder.build_graph("focal")

    assert focal_id == "focal"
    assert graph.graph["strategy"] == "recommendation"
    assert graph.graph["candidate_source_status"] == {"recommendations": "complete"}


def test_candidate_acquisition_distinguishes_empty_partial_and_total_outages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate sources should preserve empty evidence and fail only on total outage.

    :param pytest.MonkeyPatch monkeypatch: Pytest patch helper.
    :return None: Assertions validate standalone and hybrid outage handling.
    """
    from citemesh.services import SemanticScholarUnavailableError

    focal = _focal_paper()
    partial_client = MagicMock()
    partial_client.get_paper_references.side_effect = SemanticScholarUnavailableError(
        "references down"
    )
    partial_client.get_paper_citations.return_value = []
    partial_client.get_recommended_papers.return_value = [_paper("rec1")]

    partial_pool = fetch_candidate_pool(
        partial_client,
        focal,
        max_references=1,
        max_citations=1,
        max_recommendations=1,
    )
    assert set(partial_pool.papers) == {"rec1"}
    assert partial_pool.source_status == {
        "references": "unavailable",
        "citations": "empty",
        "recommendations": "complete",
    }

    empty_client = MagicMock()
    empty_client.get_paper_references.return_value = []
    empty_client.get_paper_citations.return_value = []
    empty_pool = fetch_candidate_pool(
        empty_client,
        focal,
        max_references=1,
        max_citations=1,
    )
    assert empty_pool.papers == {}
    assert empty_pool.source_status == {
        "references": "empty",
        "citations": "empty",
    }

    unavailable_client = MagicMock()
    unavailable_client.get_paper_references.side_effect = (
        SemanticScholarUnavailableError("references down")
    )
    unavailable_client.get_paper_citations.side_effect = (
        SemanticScholarUnavailableError("citations down")
    )
    with pytest.raises(CandidateAcquisitionError, match="references, citations"):
        fetch_candidate_pool(
            unavailable_client,
            focal,
            max_references=1,
            max_citations=1,
        )

    query_client = MagicMock()
    query_client.search_papers.side_effect = SemanticScholarUnavailableError(
        "search down"
    )
    with pytest.raises(CandidateAcquisitionError, match="search"):
        fetch_candidate_pool(
            query_client,
            Paper(
                paper_id="query:topic",
                title="topic",
                year=None,
                is_focal=True,
            ),
            max_recommendations=1,
        )
    query_client.get_recommended_papers.assert_not_called()

    recommendation_client = MagicMock()
    recommendation_client.get_paper.return_value = focal
    recommendation_client.get_recommended_papers.side_effect = (
        SemanticScholarUnavailableError("recommendations down")
    )
    with pytest.raises(CandidateAcquisitionError, match="recommendations"):
        RecommendationGraphBuilder(
            max_papers=3,
            fetch_references=False,
            client=recommendation_client,
        ).collect_papers("focal")

    citation_client = MagicMock()
    citation_client.get_paper.return_value = focal
    citation_client.get_paper_references.side_effect = SemanticScholarUnavailableError(
        "references down"
    )
    citation_client.get_paper_citations.side_effect = SemanticScholarUnavailableError(
        "citations down"
    )
    with pytest.raises(CandidateAcquisitionError, match="references, citations"):
        CitationGraphBuilder(
            max_papers=3,
            max_references=1,
            max_citations=1,
            fetch_references=False,
            client=citation_client,
        ).collect_papers("focal")

    hybrid_client = MagicMock()
    hybrid_client.get_paper.return_value = focal
    hybrid_client.get_paper_references.side_effect = SemanticScholarUnavailableError(
        "references down"
    )
    hybrid_client.get_paper_citations.side_effect = SemanticScholarUnavailableError(
        "citations down"
    )
    hybrid_client.get_recommended_papers.return_value = [_paper("rec1")]
    hybrid_builder = HybridGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=False,
        client=hybrid_client,
    )
    offline_hybrid_encoder(monkeypatch, hybrid_builder)

    hybrid_papers = hybrid_builder.collect_papers("focal")

    assert set(hybrid_papers) == {"focal", "rec1"}
    assert hybrid_builder.candidate_source_status == {
        "references": "unavailable",
        "citations": "unavailable",
        "recommendations": "complete",
    }

    hybrid_client.get_recommended_papers.side_effect = SemanticScholarUnavailableError(
        "recommendations down"
    )
    with pytest.raises(
        CandidateAcquisitionError,
        match="references, citations, recommendations",
    ):
        hybrid_builder.collect_papers("focal")
    assert hybrid_client.get_recommended_papers.call_count == 2
    assert hybrid_builder.candidate_source_status == {
        "references": "unavailable",
        "citations": "unavailable",
        "recommendations": "unavailable",
    }


def test_hybrid_discovers_recommendations_before_reference_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Citation enrichment must not preempt hybrid recommendation discovery."""
    events: list[str] = []
    client = MagicMock()
    client.get_paper.return_value = _focal_paper()

    def unavailable_references(*_args: object, **_kwargs: object) -> list[Paper]:
        events.append("discover references")
        raise s2.SemanticScholarUnavailableError("references down")

    def unavailable_citations(*_args: object, **_kwargs: object) -> list[Paper]:
        events.append("discover citations")
        raise s2.SemanticScholarUnavailableError("citations down")

    def recommendations(*_args: object, **_kwargs: object) -> list[Paper]:
        events.append("discover recommendations")
        return [_paper("rec1")]

    def reference_ids(paper_ids: list[str], **_kwargs: object) -> ReferenceBatchResult:
        events.append(f"enrich {','.join(paper_ids)}")
        return ReferenceBatchResult({})

    client.get_paper_references.side_effect = unavailable_references
    client.get_paper_citations.side_effect = unavailable_citations
    client.get_recommended_papers.side_effect = recommendations
    client.get_reference_ids_batch.side_effect = reference_ids
    builder = HybridGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=True,
        client=client,
    )
    offline_hybrid_encoder(monkeypatch, builder)

    papers = builder.collect_papers("focal")

    assert set(papers) == {"focal", "rec1"}
    # A semantic-only recommendation gets no reference list for coupling.
    assert events == [
        "discover references",
        "discover citations",
        "discover recommendations",
        "enrich focal",
    ]


def test_hybrid_corpus_defers_optional_focal_enrichment(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Corpus focal paper enrichment must follow citation and corpus acquisition."""
    events: list[str] = []
    focal = Paper(
        paper_id="local-focal",
        title="Local focal",
        year=2024,
        abstract="Local abstract",
        arxiv_id="2501.00001",
        is_local_corpus=True,
        is_focal=True,
    )
    semantic = replace(_paper("semantic"), is_local_corpus=True, arxiv_id="2501.00002")
    client = MagicMock()
    client.get_paper_references.side_effect = lambda *_args, **_kwargs: (
        events.append("reference discovery") or []
    )
    client.get_paper_citations.side_effect = lambda *_args, **_kwargs: (
        events.append("citation discovery") or []
    )
    builder = HybridGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        semantic_source="arxiv-corpus",
        dataset_source="example/arxiv",
        fetch_references=True,
        client=client,
    )
    enrich = builder.embedding_builder._enrich_corpus_papers
    search = stub_hybrid_corpus(builder, focal=focal)
    builder.embedding_builder._enrich_corpus_papers = enrich
    search.side_effect = lambda *_args: events.append("corpus discovery") or [semantic]
    offline_hybrid_encoder(monkeypatch, builder)
    client.get_papers.side_effect = lambda ids: (
        events.append("provider metadata enrichment")
        or {
            identifier: replace(_paper(identifier), citation_count=100)
            for identifier in ids
        }
    )
    builder.reference_hydrator.hydrate = MagicMock(
        side_effect=lambda _papers, _focal, **_kwargs: events.append(
            "reference enrichment"
        )
    )

    with caplog.at_level(logging.INFO):
        papers = builder.collect_papers(focal.paper_id)

    assert set(papers) == {focal.paper_id, "semantic"}
    assert papers[focal.paper_id] is not focal
    assert caplog.messages.count("Focal paper: Local focal (2024)") == 1
    assert events == [
        "reference discovery",
        "citation discovery",
        "corpus discovery",
        "corpus discovery",
        "provider metadata enrichment",
        "reference enrichment",
    ]
    client.get_paper.assert_not_called()
    client.get_papers.assert_called_once_with(["arxiv:2501.00001", "arxiv:2501.00002"])
    assert papers[focal.paper_id].citation_count == 100
    assert papers[semantic.paper_id].citation_count == 100


def test_hybrid_hydrates_aliases_for_selected_citation_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid reuses a cited alias's list within citation-sourced hydration."""
    client = MagicMock()
    client.get_paper.return_value = replace(_focal_paper(), references=["shared"])
    cited = replace(_paper("cited"), doi="10.1000/cited")
    semantic = replace(_paper("semantic"), doi="10.1000/semantic")
    client.get_paper_references.return_value = [
        cited,
        replace(cited, paper_id="cited-alias"),
    ]
    client.get_recommended_papers.return_value = [
        semantic,
        replace(semantic, paper_id="semantic-alias"),
    ]
    client.get_reference_ids_batch.side_effect = [
        ReferenceBatchResult({"cited": []}),
        ReferenceBatchResult({"cited-alias": ["shared"]}),
    ]
    builder = HybridGraphBuilder(
        max_papers=3, max_references=2, max_citations=0, client=client
    )
    offline_hybrid_encoder(monkeypatch, builder)

    papers = builder.collect_papers("focal")

    assert set(papers) == {"focal", "cited", "semantic"}
    assert papers["cited"].references == ["shared"]
    assert papers["semantic"].references == []
    assert [
        request.args[0] for request in client.get_reference_ids_batch.call_args_list
    ] == [
        ["cited"],
        ["cited-alias"],
    ]


def test_hybrid_deferred_reference_hydration_failure_keeps_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference hydration outages must not discard already discovered papers."""
    focal = _focal_paper()
    citation_candidate = _paper("citation")
    semantic_candidate = _paper("semantic")
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [citation_candidate]
    client.get_paper_citations.return_value = []
    client.get_reference_ids_batch.return_value = ReferenceBatchResult(
        {}, unavailable=True
    )
    builder = HybridGraphBuilder(
        max_papers=4,
        semantic_source="arxiv-corpus",
        fetch_references=True,
        client=client,
    )
    stub_hybrid_corpus(builder, neighbours=[semantic_candidate])
    offline_hybrid_encoder(monkeypatch, builder)

    papers = builder.collect_papers(focal.paper_id)

    assert set(papers) == {
        focal.paper_id,
        citation_candidate.paper_id,
        semantic_candidate.paper_id,
    }
    client.get_reference_ids_batch.assert_called_once_with(
        ["focal", "citation"], force_refresh=False
    )


def test_query_candidate_bootstrap_shares_the_total_pool_budget() -> None:
    """Free-text bootstrap results must fit inside the candidate pool budget.

    :return None: Checks a one-paper budget cannot admit 20 candidates.
    """
    query_client = MagicMock()
    search_candidates = [_paper(f"search-{index}") for index in range(20)]

    def search_papers(
        query: str,
        *,
        limit: int,
        raise_on_unavailable: bool,
    ) -> list[Paper]:
        """Return as many fake search papers as the request permits.

        :param str query: Search text expected from the synthetic focal paper.
        :param int limit: Requested provider result limit.
        :param bool raise_on_unavailable: Strict availability flag.
        :return list[Paper]: Prefix of the fake search result set.
        """
        assert query == "topic"
        assert raise_on_unavailable is True
        return search_candidates[:limit]

    query_client.search_papers.side_effect = search_papers
    pool = fetch_candidate_pool(
        query_client,
        Paper(
            paper_id="query:topic",
            title="topic",
            year=None,
            is_focal=True,
        ),
        max_recommendations=1,
    )

    assert set(pool.papers) == {"search-0"}
    query_client.search_papers.assert_called_once_with(
        "topic",
        limit=1,
        raise_on_unavailable=True,
    )
    query_client.get_recommended_papers.assert_not_called()


def test_hybrid_corpus_mode_survives_relation_endpoint_outage(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A working local corpus must remain usable when both S2 relations fail.

    :param pytest.MonkeyPatch monkeypatch: Offline collection and ranking stubs.
    :param pytest.LogCaptureFixture caplog: Captured availability warning.
    :return None: Assertions validate retained source status and semantic candidates.
    """
    from citemesh.services import SemanticScholarUnavailableError

    focal = _focal_paper()
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.side_effect = SemanticScholarUnavailableError("offline")
    client.get_paper_citations.side_effect = SemanticScholarUnavailableError("offline")
    client.get_reference_ids_batch.return_value = ReferenceBatchResult(
        {}, unavailable=True
    )
    builder = HybridGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        semantic_source="arxiv-corpus",
        dataset_source="example/arxiv",
        client=client,
    )
    assert builder.embedding_builder.dataset_source == "example/arxiv"
    assert builder.embedding_builder.corpus_size is None
    search = stub_hybrid_corpus(builder, neighbours=[focal, _paper("semantic")])
    offline_hybrid_encoder(monkeypatch, builder)

    with caplog.at_level(logging.WARNING):
        papers = builder.collect_papers("focal")

    assert set(papers) == {"focal", "semantic"}
    assert builder.candidate_source_status == {
        "references": "unavailable",
        "citations": "unavailable",
        "arxiv_corpus": "complete",
    }
    assert search.call_count == 2
    client.get_paper.assert_called_once_with("focal", raise_on_unavailable=True)
    client.get_reference_ids_batch.assert_called_once_with(
        ["focal"], force_refresh=False
    )
    client.get_recommended_papers.assert_not_called()
    assert (
        "Continuing with unavailable candidate sources: references, citations."
        in caplog.text
    )


def test_citation_collect_clears_in_memory_reference_cache_between_requests() -> None:
    """Citation collection should scope in-memory reference cache to one request."""
    focal = _paper("focal", refs=["focal-ref"])
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = []
    client.get_paper_citations.return_value = []

    builder = CitationGraphBuilder(
        max_papers=1,
        fetch_references=False,
        client=client,
    )
    builder.reference_hydrator.reference_cache["stale"] = ["stale-ref"]

    builder.collect_papers("focal")

    assert builder.reference_hydrator.reference_cache == {}


def test_citation_similarity_uses_reference_and_fallback_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Citation similarity should switch weighting when references are missing."""
    builder = CitationGraphBuilder(fetch_references=True, client=MagicMock())
    builder._abstract_index = MagicMock()
    builder._abstract_index.similarity.return_value = 0.5

    paper_a = _paper("a", year=2020, refs=["r1", "r2"])
    paper_b = _paper("b", year=2020, refs=["r2", "r3"])
    paper_a.citation_count = 10
    paper_b.citation_count = 20

    with_refs = builder.compute_similarity(paper_a, paper_b)
    expected_with_refs = (
        0.40 * 0.5
        + 0.20 * builder.temporal_similarity(paper_a, paper_b)
        + 0.40 * builder.bibliographic_coupling(paper_a, paper_b)
    )
    assert with_refs == pytest.approx(expected_with_refs)

    paper_b.references = []
    without_refs = builder.compute_similarity(paper_a, paper_b)
    expected_without_refs = (
        0.65 * 0.5
        + 0.20 * builder.temporal_similarity(paper_a, paper_b)
        + 0.15 * builder.citation_similarity(paper_a, paper_b)
    )
    assert without_refs == pytest.approx(expected_without_refs)


def test_hybrid_collection_merges_and_tracks_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid collection should merge citation+semantic papers and source labels."""
    focal = _paper("focal")
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [_paper("c1")]
    client.get_paper_citations.return_value = []
    builder = HybridGraphBuilder(
        max_papers=5, semantic_source="arxiv-corpus", client=client
    )
    stub_hybrid_corpus(builder, neighbours=[_paper("c1"), _paper("s1"), _paper("s2")])
    offline_hybrid_encoder(monkeypatch, builder)

    papers = builder.collect_papers("focal")

    assert set(papers) == {"focal", "c1", "s1", "s2"}
    assert builder.paper_sources == {
        "focal": "citation",
        "c1": "both",
        "s1": "semantic",
        "s2": "semantic",
    }
    assert builder.focal_relations["focal"] == "focal"
    assert builder.focal_relations["c1"] == "referenced_by_focal"
    assert builder.focal_relations["s1"] == "semantic_only"


@pytest.mark.parametrize("focal_id", ["content:local-paper", "arxiv_7"])
def test_hybrid_candidates_mode_rejects_local_only_focal(focal_id: str) -> None:
    """A corpus-free hybrid build must not send local IDs to the provider.

    :param str focal_id: Current or legacy local-only corpus identifier.
    :return None: Verifies actionable guidance before any provider request.
    """
    client = MagicMock()
    builder = HybridGraphBuilder(max_papers=2, client=client)

    with pytest.raises(ValueError, match="requires a corpus"):
        builder.collect_papers(focal_id)

    client.get_paper.assert_not_called()


def test_embedding_and_hybrid_similarity_normalize_scaled_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semantic branches should score by angle, not by vector magnitude."""
    paper_a = _paper("a", year=2020, refs=["r1"])
    paper_b = _paper("b", year=2020, refs=["r1"])

    embedding_builder = EmbeddingGraphBuilder(max_papers=2, client=MagicMock())
    embedding_builder.embeddings = {
        "a": np.asarray([3.0, 0.0], dtype=np.float32),
        "b": np.asarray([9.0, 0.0], dtype=np.float32),
    }
    expected_embedding_similarity = (
        EMBEDDING_CONFIG.semantic_weight * 1.0
        + EMBEDDING_CONFIG.temporal_weight
        * embedding_builder.temporal_similarity(paper_a, paper_b)
        + EMBEDDING_CONFIG.category_weight * paper_a.category_overlap(paper_b)
    )
    assert embedding_builder.compute_similarity(paper_a, paper_b) == pytest.approx(
        expected_embedding_similarity
    )

    hybrid_builder = HybridGraphBuilder(max_papers=2, client=MagicMock())
    assert hybrid_builder.embedding_builder is not None
    hybrid_builder.paper_sources = {"a": "semantic", "b": "semantic"}
    hybrid_builder.embedding_builder.embeddings = {
        "a": np.asarray([2.0, 0.0], dtype=np.float32),
        "b": np.asarray([6.0, 0.0], dtype=np.float32),
    }
    expected_hybrid_similarity = (
        HYBRID_CONFIG.semantic_semantic_weights[0] * 1.0
        + HYBRID_CONFIG.semantic_semantic_weights[1]
        * hybrid_builder.temporal_similarity(paper_a, paper_b)
        + HYBRID_CONFIG.semantic_semantic_weights[2]
        * hybrid_builder.citation_similarity(paper_a, paper_b)
        + HYBRID_CONFIG.semantic_semantic_weights[3]
        * hybrid_builder.bibliographic_coupling(paper_a, paper_b)
    )
    assert hybrid_builder.compute_similarity(paper_a, paper_b) == pytest.approx(
        min(expected_hybrid_similarity, 1.0)
    )


def test_embedding_similarity_rewards_shared_fourth_author() -> None:
    """A shared author beyond third position should receive the collaboration bonus.

    :return None: Checks that the fourth author contributes to graph similarity.
    """
    paper_a = Paper(
        paper_id="a",
        title="First",
        year=2010,
        authors=[
            Author(name="Alice"),
            Author(name="Bob"),
            Author(name="Carol"),
            Author(name="Dana"),
        ],
    )
    paper_b = Paper(
        paper_id="b",
        title="Second",
        year=2020,
        authors=[
            Author(name="Eve"),
            Author(name="Frank"),
            Author(name="Grace"),
            Author(name="Dana"),
        ],
    )
    builder = EmbeddingGraphBuilder(max_papers=2, client=MagicMock())
    builder.embeddings = {
        "a": np.asarray([1.0, 0.0], dtype=np.float32),
        "b": np.asarray([0.8, 0.6], dtype=np.float32),
    }

    expected = (
        EMBEDDING_CONFIG.semantic_weight * 0.8
        + EMBEDDING_CONFIG.temporal_weight
        * builder.temporal_similarity(paper_a, paper_b)
    ) * EMBEDDING_CONFIG.shared_author_bonus

    assert paper_a.shares_authors_with(paper_b)
    assert builder.compute_similarity(paper_a, paper_b) == pytest.approx(expected)


def test_graph_nodes_mirror_all_paper_authors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph node metadata should retain the complete canonical author list.

    :param pytest.MonkeyPatch monkeypatch: Replaces collection and graph scoring.
    :return None: Checks node-level author metadata remains complete.
    """
    author_names = ["Alice", "Bob", "Carol", "Dana"]
    papers = {
        "focal": Paper(
            paper_id="focal",
            title="Focal",
            year=2024,
            authors=[Author(name=name) for name in author_names],
            is_focal=True,
        )
    }
    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    monkeypatch.setattr(builder, "collect_papers", lambda _focal_id, **_kwargs: papers)
    monkeypatch.setattr(builder, "prepare_graph_scoring", lambda _papers: None)

    graph, _ = builder.build_graph("focal")

    assert graph.nodes["focal"]["authors"] == author_names


@pytest.mark.parametrize(("query_scale", "candidate_scale"), [(1, 1), (10, 1), (1, 10)])
def test_semantic_candidate_ranking_is_invariant_to_vector_scale(
    monkeypatch: pytest.MonkeyPatch,
    query_scale: int,
    candidate_scale: int,
) -> None:
    """Candidate order must follow cosine direction rather than vector magnitude."""
    focal = _focal_paper("focal")
    candidates = {
        "a-less-aligned": _paper("a-less-aligned"),
        "b-more-aligned": _paper("b-more-aligned"),
    }
    query = query_scale * np.asarray([1.0, 0.0], dtype=np.float32)
    embeddings = {
        "a-less-aligned": candidate_scale * np.asarray([0.6, 0.8], dtype=np.float32),
        "b-more-aligned": np.asarray([0.8, 0.6], dtype=np.float32),
    }
    builder = EmbeddingGraphBuilder(semantic_source="candidates", client=MagicMock())
    pool = DiscoveryPool(focal)
    for paper in candidates.values():
        pool.add(paper, source="recommendation", relation="semantic_only")
    monkeypatch.setattr(
        "citemesh.strategies.embedding.builder.fetch_candidate_pool",
        MagicMock(return_value=pool),
    )
    monkeypatch.setattr(builder, "embed_papers", MagicMock(return_value=embeddings))
    ranked = [
        paper_id
        for paper_id, _, _ in builder._select_candidates_from_pool(query, focal)
    ]

    assert ranked == ["b-more-aligned", "a-less-aligned"]


def test_hybrid_edges_use_graph_vectors_not_retrieval_vectors() -> None:
    """Hybrid pairwise topology must not read the retrieval space."""
    builder = HybridGraphBuilder(max_papers=2, client=MagicMock())
    focal = _focal_paper("focal")
    candidate = _paper("candidate", year=focal.year)
    builder.paper_sources = {"focal": "semantic", "candidate": "semantic"}
    builder.embedding_builder.retrieval_embeddings = {
        "focal": np.asarray([1.0, 0.0], dtype=np.float32),
        "candidate": np.asarray([1.0, 0.0], dtype=np.float32),
    }
    builder.embedding_builder.embeddings = {
        "focal": np.asarray([1.0, 0.0], dtype=np.float32),
        "candidate": np.asarray([0.0, 1.0], dtype=np.float32),
    }

    assert builder.compute_similarity(focal, candidate) == 0.0


def test_embedding_strategy_ranks_recovered_references_by_retrieval_cosine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding-enabled recovery must not fall back to lexical ranking.

    :param pytest.MonkeyPatch monkeypatch: Replaces inference with fixed vectors.
    :return None: Checks retrieval cosine wins over order and citation count.
    """
    focal = _focal_paper("focal")
    early = _paper("early")
    early.citation_count = 100
    relevant = _paper("relevant")
    relevant.citation_count = 0
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    embeddings = {
        early.paper_id: np.asarray([0.0, 1.0], dtype=np.float32),
        relevant.paper_id: np.asarray([1.0, 0.0], dtype=np.float32),
    }

    builder = EmbeddingGraphBuilder(
        max_papers=2,
        semantic_source="candidates",
        client=MagicMock(),
    )
    builder.retrieval_embeddings[focal.paper_id] = query
    monkeypatch.setattr(builder, "embed_papers", MagicMock(return_value=embeddings))
    selected = builder._select_recovered_references_by_embedding(
        focal, [early, relevant], 1
    )

    assert selected == [relevant]


def test_embedding_candidate_pool_passes_retrieval_reference_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate acquisition must delegate fallback selection to embeddings."""
    builder = EmbeddingGraphBuilder(
        max_papers=2,
        semantic_source="candidates",
        client=MagicMock(),
    )
    focal = _focal_paper("focal")
    candidate = _paper("candidate")
    pool = DiscoveryPool(focal)
    pool.add(candidate, source="reference", relation="referenced_by_focal")
    fetch = MagicMock(return_value=pool)
    monkeypatch.setattr(
        "citemesh.strategies.embedding.builder.fetch_candidate_pool", fetch
    )
    monkeypatch.setattr(
        builder,
        "embed_papers",
        MagicMock(
            return_value={candidate.paper_id: np.asarray([1.0, 0.0], dtype=np.float32)}
        ),
    )

    builder._select_candidates_from_pool(
        np.asarray([1.0, 0.0], dtype=np.float32), focal
    )

    assert (
        fetch.call_args.kwargs["reference_selector"]
        == builder._select_recovered_references_by_embedding
    )


def test_hybrid_graph_preparation_fails_closed_on_sts_inference() -> None:
    """Hybrid should surface graph-space inference failures before edge creation."""
    builder = HybridGraphBuilder(max_papers=2, client=MagicMock())
    builder.embedding_builder.materialize_graph_embeddings = MagicMock(
        side_effect=RuntimeError("simulated MPS STS failure")
    )

    with pytest.raises(
        EmbeddingInferenceError,
        match="Hybrid graph-similarity embedding failed.*simulated MPS STS failure",
    ):
        builder.prepare_graph_scoring(
            {"focal": _focal_paper("focal"), "peer": _paper("peer")}
        )


@pytest.mark.parametrize("semantic_source", ["arxiv-corpus", "both"])
def test_hybrid_collection_fails_closed_on_semantic_enrichment_errors(
    monkeypatch: pytest.MonkeyPatch, semantic_source: str
) -> None:
    """Hybrid collection should fail when semantic enrichment cannot complete."""
    client = MagicMock()
    client.get_paper.return_value = _paper("focal")
    client.get_paper_references.return_value = [_paper("c1")]
    client.get_paper_citations.return_value = []
    client.get_recommended_papers.return_value = [_paper("rec")]
    builder = HybridGraphBuilder(
        max_papers=5,
        semantic_source=semantic_source,
        client=client,
    )
    search = stub_hybrid_corpus(builder)
    search.side_effect = RuntimeError("semantic backend unavailable")
    offline_hybrid_encoder(monkeypatch, builder)

    with pytest.raises(
        RuntimeError, match="Semantic enrichment failed: semantic backend unavailable"
    ):
        builder.collect_papers("focal")


def test_hybrid_collection_preserves_semantic_scholar_outage_type() -> None:
    """Hybrid API callers should retain the service availability taxonomy."""
    from citemesh.services import SemanticScholarUnavailableError

    client = MagicMock()
    client.get_paper.side_effect = SemanticScholarUnavailableError("focal outage")
    builder = HybridGraphBuilder(
        max_papers=5,
        semantic_source="arxiv-corpus",
        client=client,
    )
    stub_hybrid_corpus(builder)

    with pytest.raises(SemanticScholarUnavailableError, match="focal outage"):
        builder.collect_papers("focal")


def _single_reference_client() -> MagicMock:
    """Return a client whose focal cites one paper and has no other relations."""
    client = MagicMock()
    client.get_paper.return_value = _focal_paper("focal")
    client.get_paper_references.return_value = [_paper("c1")]
    client.get_paper_citations.return_value = []
    client.get_recommended_papers.return_value = []
    client.get_reference_ids_batch.return_value = ReferenceBatchResult({})
    return client


def test_hybrid_collection_fails_when_focal_embedding_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid ranking should surface a focal paper embedding failure."""
    builder = HybridGraphBuilder(max_papers=4, client=_single_reference_client())
    monkeypatch.setattr(builder.embedding_builder, "_load_model", lambda: None)
    builder.embedding_builder._encode_texts = MagicMock(
        side_effect=RuntimeError("temporary focal encode failure")
    )

    with pytest.raises(
        EmbeddingInferenceError,
        match="Hybrid focal paper embedding failed: RuntimeError: temporary focal encode",
    ):
        builder.collect_papers("focal")


@pytest.mark.parametrize(
    ("candidate_vector", "message"),
    [
        (np.asarray([np.nan, 0.0], dtype=np.float32), "non-finite embedding"),
        (np.asarray([0.0, 0.0], dtype=np.float32), "zero embedding"),
        (np.asarray([0.1, 0.2, 0.3], dtype=np.float32), "inconsistent embedding"),
    ],
)
def test_hybrid_ranking_rejects_invalid_candidate_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    candidate_vector: np.ndarray,
    message: str,
) -> None:
    """Hybrid ranking should reject unusable vectors instead of substituting zero."""
    builder = HybridGraphBuilder(max_papers=4, client=_single_reference_client())
    monkeypatch.setattr(builder.embedding_builder, "_load_model", lambda: None)
    focal_vector = np.asarray([1.0, 0.0], dtype=np.float32)
    builder.embedding_builder._encode_texts = MagicMock(
        return_value=focal_vector[np.newaxis, :]
    )
    # The focal paper document, then the one candidate.
    builder.embedding_builder.document_vectors = MagicMock(
        side_effect=[{"focal": focal_vector}, {"c1": candidate_vector}]
    )

    with pytest.raises(EmbeddingInferenceError, match=message):
        builder.collect_papers("focal")


def test_hybrid_candidates_mode_rebuild_encodes_no_candidate_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate document vectors persist in the candidate embedding cache."""
    client = _single_reference_client()
    client.get_recommended_papers.return_value = [_paper("rec")]
    encoded = []
    for _ in range(2):
        builder = HybridGraphBuilder(
            max_papers=4, semantic_source="candidates", client=client
        )
        texts = offline_hybrid_encoder(monkeypatch, builder)
        papers = builder.collect_papers("focal")
        encoded.append(list(texts))

    assert {"c1", "rec"} <= set(papers)
    # Focal query, focal paper document and both candidates; then the focal paper query only.
    assert [len(texts) for texts in encoded] == [4, 1]
    assert encoded[1] == encoded[0][:1]
    # Cached document vectors leave the focal paper's query-role slot alone.
    retrieval = builder.embedding_builder.retrieval_embeddings
    assert list(retrieval) == ["focal"]
    assert np.allclose(retrieval["focal"], word_hash_vectors(encoded[1])[0])


@pytest.mark.parametrize("semantic_source", ["arxiv-corpus", "both"])
def test_hybrid_corpus_modes_encode_provider_rows_in_memory(
    monkeypatch: pytest.MonkeyPatch, semantic_source: str
) -> None:
    """The corpus cache takes no provider rows, so corpus modes encode in memory."""
    client = _single_reference_client()
    client.get_recommended_papers.return_value = [_paper("rec")]
    builder = HybridGraphBuilder(
        max_papers=4, semantic_source=semantic_source, client=client
    )
    stub_hybrid_corpus(builder, neighbours=[_paper("s1")])
    texts = offline_hybrid_encoder(monkeypatch, builder)
    builder.embedding_builder.document_vectors = MagicMock(
        side_effect=AssertionError("provider rows must not enter the corpus cache")
    )

    papers = builder.collect_papers("focal")

    assert "c1" in papers
    builder.embedding_builder.document_vectors.assert_not_called()
    # Focal query and focal paper document, then every candidate once.
    assert len(texts) == 2 + len(builder.discovery_ranking)


def test_hybrid_thresholds_and_fetch_limits() -> None:
    """Hybrid enforces thresholds for focal paper edges and other edges, plus limits."""
    builder = HybridGraphBuilder(max_papers=3, client=MagicMock())
    focal = _paper("focal")
    focal.is_focal = True
    other = _paper("other")
    peer = _paper("peer")

    assert not builder.should_create_edge(focal, other, 0.2)
    assert builder.should_create_edge(focal, other, 0.41)
    assert not builder.should_create_edge(other, peer, 0.5)
    assert builder.should_create_edge(other, peer, 0.51)

    default = HybridGraphBuilder(client=MagicMock())
    assert (default.max_papers, default.max_citations, default.max_references) == (
        45,
        1000,
        1000,
    )
    with pytest.raises(ValueError, match="must be non-negative"):
        HybridGraphBuilder(max_citations=-1, client=MagicMock())


def test_hybrid_propagates_refresh_reference_cache_to_reference_hydration() -> None:
    """Hybrid should forward refresh-reference policy to reference hydration."""
    builder = HybridGraphBuilder(
        max_papers=4,
        refresh_reference_cache=True,
        client=MagicMock(),
    )

    assert builder.reference_hydrator.refresh_reference_cache is True


def test_hybrid_collection_dedupes_semantic_focal_aliases(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Hybrid should collapse and report a corpus alias of the citation focal.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing collection internals.
    :param pytest.LogCaptureFixture caplog: Captured identity reconciliation trace.
    """
    focal = Paper(
        paper_id="ca997f1a733e53ad0fa29041246ff655243e8c1b",
        title="Polynomial Composition Activations: Unleashing the Dynamics of Large Language Models",
        year=2024,
        abstract=(
            "Transformers have found extensive applications across various domains due "
            "to the powerful fitting capabilities."
        ),
        authors=[Author(name="Zhijian Zhou"), Author(name="Ya Wang")],
        arxiv_id="2411.03884",
        is_focal=True,
    )
    semantic_focal_alias = Paper(
        paper_id="arXiv:2411.03884v2",
        title=(
            "Polynomial Composition Activations: Unleashing the Dynamics of Large\n"
            "  Language Models"
        ),
        year=2025,
        abstract=focal.abstract,
        authors=[Author(name="Zhijian Zhou"), Author(name="Yitao Zeng")],
        is_local_corpus=True,
    )
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [_paper("c1")]
    client.get_paper_citations.return_value = []
    builder = HybridGraphBuilder(
        max_papers=5, semantic_source="arxiv-corpus", client=client
    )
    stub_hybrid_corpus(builder, neighbours=[semantic_focal_alias, _paper("s1")])
    offline_hybrid_encoder(monkeypatch, builder)

    with caplog.at_level(logging.DEBUG, logger="citemesh.strategies.candidates"):
        papers = builder.collect_papers("arXiv:2411.03884")

    assert set(papers) == {focal.paper_id, "c1", "s1"}
    assert "arXiv:2411.03884v2" not in papers
    assert papers[focal.paper_id].year == 2024
    assert builder.paper_sources[focal.paper_id] == "citation"
    assert (
        f"Merged record {semantic_focal_alias.paper_id} into {focal.paper_id} "
        "via shared arxiv identifier"
    ) in caplog.messages


def test_hybrid_build_graph_skips_pruning_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid build should return parent graph unchanged when pruning disabled."""
    monkeypatch.setattr(HYBRID_CONFIG, "max_edges_per_node", 0)

    builder = HybridGraphBuilder(max_papers=3, client=MagicMock())
    builder.paper_sources = {"focal": "citation", "a": "semantic"}
    builder.focal_relations = {
        "focal": "focal",
        "a": "semantic_only",
        "pruned": "citation",
    }
    graph = nx.Graph()
    graph.add_node("focal", is_focal=True)
    graph.add_node("a", is_focal=False)
    monkeypatch.setattr(
        hybrid_strategy.GraphBuilderStrategy,
        "build_graph",
        lambda self, focal_id, **kwargs: (graph, "focal"),
    )

    out_graph, out_focal = builder.build_graph("focal")
    assert out_graph is graph
    assert out_focal == "focal"
    assert out_graph.graph["strategy"] == "hybrid"
    assert out_graph.graph["paper_sources"] == {"a": "semantic", "focal": "citation"}
    assert out_graph.graph["focal_relations"] == {
        "a": "semantic_only",
        "focal": "focal",
    }


def test_hybrid_build_graph_logs_post_cap_edge_count(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Hybrid pruning should log original and filtered edge counts."""
    monkeypatch.setattr(HYBRID_CONFIG, "max_edges_per_node", 1)

    builder = HybridGraphBuilder(max_papers=4, client=MagicMock())
    graph = nx.Graph()
    graph.add_node("focal", is_focal=True)
    graph.add_nodes_from(["a", "b", "c"])
    graph.add_edge("focal", "a", weight=0.9)
    graph.add_edge("focal", "b", weight=0.8)
    graph.add_edge("focal", "c", weight=0.7)
    graph.add_edge("a", "b", weight=0.95)
    graph.add_edge("a", "c", weight=0.85)
    graph.add_edge("b", "c", weight=0.75)
    monkeypatch.setattr(
        hybrid_strategy.GraphBuilderStrategy,
        "build_graph",
        lambda self, focal_id, **kwargs: (graph, "focal"),
    )

    with caplog.at_level(logging.DEBUG):
        out_graph, out_focal = builder.build_graph("focal")

    assert out_focal == "focal"
    assert out_graph.number_of_edges() < graph.number_of_edges()
    assert any(
        f"Hybrid edge cap applied: {graph.number_of_edges()} -> {out_graph.number_of_edges()} edges"
        in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "strategy",
    ["recommendation", "citation", "embedding", "hybrid"],
)
def test_max_papers_is_total_node_cap_including_focal(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
) -> None:
    """All strategies should include focal within ``max_papers`` total cap."""
    if strategy == "recommendation":
        client = MagicMock()
        client.get_paper.return_value = _focal_paper()
        client.get_recommended_papers.return_value = [
            _paper(f"r{i}") for i in range(1, 6)
        ]
        papers = RecommendationGraphBuilder(max_papers=3, client=client).collect_papers(
            "focal"
        )
        assert len(papers) == 3
        assert "focal" in papers
        return

    if strategy == "citation":
        client = MagicMock()
        client.get_paper.return_value = _focal_paper()
        client.get_paper_references.return_value = [
            _paper(f"r{i}") for i in range(1, 6)
        ]
        client.get_paper_citations.return_value = [_paper(f"c{i}") for i in range(1, 6)]
        papers = CitationGraphBuilder(
            max_papers=4,
            max_citations=10,
            max_references=10,
            client=client,
        ).collect_papers("focal")
        assert len(papers) == 4
        assert "focal" in papers
        return

    if strategy == "embedding":
        builder = EmbeddingGraphBuilder(
            max_papers=3,
            model_name="dummy",
            top_k=2,
            corpus_size=10,
            semantic_source="arxiv-corpus",
            client=MagicMock(),
        )
        builder.client.get_paper.return_value = _focal_paper()
        builder._load_model = lambda: None
        builder._resolve_model_fingerprint = lambda: "test-fingerprint"
        builder._prepare_corpus_for_build = lambda: None
        builder._enrich_corpus_papers = lambda _: None
        builder._select_candidates = (
            lambda _focal_embedding, *, use_streaming, corpus_prepared=False: [
                (
                    "c1",
                    {
                        "title": "Paper c1",
                        "abstract": "A",
                        "authors": [],
                        "categories": [],
                    },
                    np.array([1.0, 0.0], dtype=np.float32),
                ),
                (
                    "c2",
                    {
                        "title": "Paper c2",
                        "abstract": "B",
                        "authors": [],
                        "categories": [],
                    },
                    np.array([0.0, 1.0], dtype=np.float32),
                ),
                (
                    "c3",
                    {
                        "title": "Paper c3",
                        "abstract": "C",
                        "authors": [],
                        "categories": [],
                    },
                    np.array([0.5, 0.5], dtype=np.float32),
                ),
            ]
        )
        builder._encode_texts = lambda texts, **kwargs: np.array(
            [[1.0, 0.0] for _ in texts], dtype=np.float32
        )
        papers = builder.collect_papers("focal")
        assert len(papers) == 3
        assert "focal" in papers
        return

    client = MagicMock()
    client.get_paper.return_value = _focal_paper()
    client.get_paper_references.return_value = [_paper("c1"), _paper("c2")]
    client.get_paper_citations.return_value = [_paper("c3")]
    hybrid = HybridGraphBuilder(
        max_papers=4, semantic_source="arxiv-corpus", client=client
    )
    stub_hybrid_corpus(hybrid, neighbours=[_paper("s1"), _paper("s2")])
    offline_hybrid_encoder(monkeypatch, hybrid)
    papers = hybrid.collect_papers("focal")
    assert len(papers) == 4
    assert "focal" in papers
    assert len(hybrid.discovery_ranking) == 5


@pytest.mark.parametrize(
    "builder_type",
    [
        CitationGraphBuilder,
        RecommendationGraphBuilder,
        EmbeddingGraphBuilder,
        HybridGraphBuilder,
    ],
)
@pytest.mark.parametrize("eligible_neighbors", [0, 1, 2, 8])
def test_strategy_caps_reserve_existing_focal_neighbors(
    builder_type: type[GraphBuilderStrategy],
    eligible_neighbors: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stronger candidate clusters must not consume the focal paper's available edges.

    :param type[GraphBuilderStrategy] builder_type: Capped strategy constructor.
    :param int eligible_neighbors: Focal neighbors meeting the edge threshold.
    :param pytest.MonkeyPatch monkeypatch: Controlled collection and similarity scores.
    :return None: Asserts the focal paper retains its strongest existing edges within every cap.
    """
    papers = {"focal": _focal_paper(), **{f"p{i}": _paper(f"p{i}") for i in range(8)}}
    builder = builder_type(max_papers=9, client=MagicMock())
    monkeypatch.setattr(builder, "collect_papers", lambda _focal: papers)
    monkeypatch.setattr(builder, "prepare_graph_scoring", lambda _papers: None)
    if isinstance(builder, EmbeddingGraphBuilder):
        cap = builder.top_k
    elif isinstance(builder, HybridGraphBuilder):
        cap = HYBRID_CONFIG.max_edges_per_node
    else:
        cap = 3
    base_score = 0.60 if isinstance(builder, HybridGraphBuilder) else 0.30

    def score(left: Paper, right: Paper) -> float:
        """Score the candidate cluster above all eligible focal edges.

        :param Paper left: First paper.
        :param Paper right: Second paper.
        :return float: Eligible edge weight or zero for an unsupported focal pair.
        """
        if left.is_focal or right.is_focal:
            neighbor = right if left.is_focal else left
            index = int(neighbor.paper_id[1:])
            return base_score + index * 0.001 if index < eligible_neighbors else 0.0
        return base_score + 0.05

    monkeypatch.setattr(builder, "compute_similarity", score)
    graph, focal_id = builder.build_graph("focal")

    expected_focal_neighbors = {
        f"p{i}" for i in range(max(0, eligible_neighbors - cap), eligible_neighbors)
    }
    assert set(graph.neighbors(focal_id)) == expected_focal_neighbors
    assert graph.degree(focal_id) == min(cap, eligible_neighbors)
    assert all(degree <= cap for _, degree in graph.degree())
    assert all(score(papers[left], papers[right]) > 0 for left, right in graph.edges())
    papers = dict(reversed(list(papers.items())))
    reordered, _ = builder.build_graph("focal")
    assert {frozenset(edge) for edge in graph.edges()} == {
        frozenset(edge) for edge in reordered.edges()
    }


@pytest.mark.parametrize(
    ("direct", "expected"),
    [
        ({"d0", "d1", "d2", "d3"}, {"d0", "o0", "d1"}),
        ({"d0"}, {"d0", "o0", "o1"}),
        (set(), {"o0", "o1", "o2"}),
    ],
)
def test_capped_selection_alternates_focal_slots_with_direct_links(
    direct: set[str], expected: set[str]
) -> None:
    """Weaker direct links share the focal paper's slots instead of losing all of them."""
    edges = [("focal", f"o{i}", {"weight": 0.9 - i / 100}) for i in range(4)]
    edges += [("focal", f"d{i}", {"weight": 0.5 - i / 100}) for i in range(4)]

    selected = select_capped_undirected_edges(
        edges, 3, focal_id="focal", priority_focal_neighbors=direct
    )

    assert {u if v == "focal" else v for u, v, _ in selected} == expected


@pytest.mark.parametrize("relation", ["referenced_by_focal", "cites_focal", "overlap"])
def test_hybrid_focal_citation_pairs_face_the_focal_edge_thresholds(
    relation: str,
) -> None:
    """A focal paper's own reference or citer earns an edge only by similarity, like any pair."""
    builder = HybridGraphBuilder(client=MagicMock())
    focal = Paper(paper_id="focal", title="Algebra", abstract="Rings", year=2024)
    focal.is_focal = True
    cited = Paper(paper_id="cited", title="Biology", abstract="Coral", year=2024)
    builder.embedding_builder.embeddings = {
        "focal": np.asarray([1.0, 0.0], dtype=np.float32),
        "cited": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    builder.focal_relations = {"focal": "focal", "cited": relation}

    assert builder.compute_similarity(focal, cited) == 0.0
    assert not builder.should_create_edge(focal, cited, 0.3)
    assert not builder.should_create_edge(cited, focal, 0.3)


def test_corpus_collection_preserves_results_when_citation_counts_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An optional batch rejection must not discard completed semantic retrieval.

    :param pytest.MonkeyPatch monkeypatch: Offline model and corpus retrieval stubs.
    :param pytest.LogCaptureFixture caplog: Captured optional-enrichment warning.
    :return None: Asserts papers and vectors survive with their existing citation counts.
    """
    from citemesh.services import SemanticScholarRequestError

    client = MagicMock()
    focal = _focal_paper()
    focal.citation_count = 7
    client.get_paper.return_value = focal
    client.get_papers.side_effect = SemanticScholarRequestError("HTTP 403")
    builder = EmbeddingGraphBuilder(
        max_papers=2, semantic_source="arxiv-corpus", client=client
    )
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(builder, "_load_model", lambda: None)
    monkeypatch.setattr(
        builder, "_resolve_model_fingerprint", lambda: "test-fingerprint"
    )
    monkeypatch.setattr(
        builder,
        "_prepare_corpus_for_build",
        MagicMock(),
    )
    monkeypatch.setattr(builder, "_encode_texts", lambda *_args, **_kwargs: [vector])
    monkeypatch.setattr(
        builder,
        "_select_candidates",
        lambda *_args, **_kwargs: [
            (
                "arxiv:2501.00001",
                {"title": "Selected corpus paper", "year": 2025},
                vector,
            )
        ],
    )

    papers = builder.collect_papers("focal")

    assert set(papers) == {"focal", "arxiv:2501.00001"}
    assert papers["focal"].citation_count == 7
    assert papers["arxiv:2501.00001"].citation_count == 0
    assert set(builder.retrieval_embeddings) == set(papers)
    assert "Corpus metadata enrichment was rejected" in caplog.text
    assert "HTTP 403" in caplog.text
    client.get_papers.assert_called_once_with(["arxiv:2501.00001"])
    client.get_cached_papers.assert_not_called()
    client.get_paper.assert_called_once_with("focal", raise_on_unavailable=True)
    client.get_papers.side_effect = ValueError("invalid local payload")
    with pytest.raises(ValueError, match="invalid local payload"):
        builder._enrich_corpus_papers(papers)


@pytest.mark.parametrize(
    ("builder_factory", "expected_strategy"),
    [
        (
            lambda: CitationGraphBuilder(max_papers=5, client=MagicMock()),
            "citation",
        ),
        (
            lambda: RecommendationGraphBuilder(max_papers=5, client=MagicMock()),
            "recommendation",
        ),
        (
            lambda: EmbeddingGraphBuilder(max_papers=4, top_k=1, client=MagicMock()),
            "embedding",
        ),
        (
            lambda: HybridGraphBuilder(max_papers=4, client=MagicMock()),
            "hybrid",
        ),
    ],
)
def test_degree_capping_preserves_per_node_limit(
    builder_factory: Callable[[], object],
    expected_strategy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pruning should cap node degree deterministically."""
    builder, max_edges_per_node = _make_constant_similarity_builder(
        builder_factory, monkeypatch
    )
    graph, _ = builder.build_graph("focal")

    node_ids = sorted(graph.nodes())
    complete_graph_edges = [
        (node_ids[i], node_ids[j], {"weight": 1.0})
        for i in range(len(node_ids))
        for j in range(i + 1, len(node_ids))
    ]
    expected_edges = {
        (min(u, v), max(u, v))
        for u, v, _ in select_capped_undirected_edges(
            complete_graph_edges,
            max_edges_per_node,
            focal_id="focal",
        )
    }

    assert graph.number_of_nodes() == 5
    assert graph.graph["strategy"] == expected_strategy
    assert graph.number_of_edges() == len(expected_edges)
    assert all(degree <= max_edges_per_node for _, degree in graph.degree())
    assert {(min(u, v), max(u, v)) for u, v in graph.edges()} == expected_edges


@pytest.mark.parametrize(
    "builder_factory",
    [
        lambda: CitationGraphBuilder(max_papers=5, client=MagicMock()),
        lambda: RecommendationGraphBuilder(max_papers=5, client=MagicMock()),
        lambda: EmbeddingGraphBuilder(max_papers=5, top_k=1, client=MagicMock()),
    ],
)
def test_capped_strategies_log_final_edge_count(
    builder_factory: Callable[[], object],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Capped strategies must reserve the completion count for the final graph.

    :param Callable[[], object] builder_factory: Capped strategy constructor.
    :param pytest.MonkeyPatch monkeypatch: Fixture used for deterministic graph inputs.
    :param pytest.LogCaptureFixture caplog: Captured graph-construction logs.
    :return None: Assertions validate pre-cap and final edge counts are distinct.
    """
    builder, _ = _make_constant_similarity_builder(builder_factory, monkeypatch)

    with caplog.at_level(logging.DEBUG):
        graph, _ = builder.build_graph("focal")

    assert "Graph constructed: 5 nodes, 10 edges" in caplog.text
    assert f"Graph complete: 5 nodes, {graph.number_of_edges()} edges" in caplog.text
    assert "Graph complete: 5 nodes, 10 edges" not in caplog.text
    assert not [
        record
        for record in caplog.records
        if record.levelno >= logging.INFO
        and record.message.startswith("Graph complete:")
    ]


@pytest.mark.parametrize(
    ("edges", "max_edges_per_node", "expected"),
    [
        (
            [
                ("b", "a", {"weight": 1.0}),
                ("a", "b", {"weight": 0.6}),
                ("c", "a", {"weight": 1.0}),
                ("c", "a", {"weight": 0.9}),
                ("b", "c", {"weight": 0.9}),
            ],
            1,
            [("a", "b", 1.0)],
        ),
        (
            [
                (1, 2, {"weight": 0.2}),
                (_Tagged("alpha"), _Tagged("beta"), {"weight": 0.2}),
                ("2", 1, {"weight": 0.2}),
                (_Tagged("01"), "01", {"weight": 0.2}),
            ],
            10,
            None,
        ),
    ],
)
def test_select_capped_undirected_edges_is_deterministic_and_dedupes(
    edges: list[tuple[object, object, dict[str, float]]],
    max_edges_per_node: int,
    expected: list[tuple[object, object, float]] | None,
) -> None:
    """Edge capping should be deterministic across duplicate and mixed-id inputs."""
    first = select_capped_undirected_edges(edges, max_edges_per_node=max_edges_per_node)
    second = select_capped_undirected_edges(
        edges, max_edges_per_node=max_edges_per_node
    )
    assert first == second
    if expected is not None:
        assert first == expected


def test_deterministic_sort_key_is_total_for_secondary_and_stable_fields() -> None:
    """Tuple sort keys should be total across mixed secondary key types."""
    keys = [
        deterministic_sort_key(0.42, 10, secondary_id=_Tagged("z"), stable_index=1),
        deterministic_sort_key(0.42, "10", secondary_id="a", stable_index=0),
        deterministic_sort_key(0.42, _Tagged("10"), secondary_id="a", stable_index=2),
        deterministic_sort_key(0.42, "10", secondary_id=_Tagged("z"), stable_index=1),
    ]

    ordered = sorted(keys)
    assert ordered[0][0] == ordered[1][0] == -0.42
    assert len(ordered) == len(keys)


@pytest.mark.parametrize(
    ("paper1_year", "paper2_year", "expected"),
    [(2020, 2020, 1.0), (2010, 2020, 0.1), (None, 2020, 0.5)],
)
def test_temporal_similarity_extremes(
    paper1_year: int | None,
    paper2_year: int,
    expected: float,
) -> None:
    """Temporal scoring handles known year deltas and missing-year fallback."""
    paper1 = Paper(paper_id="p1", title="Test 1", year=paper1_year)
    paper2 = Paper(paper_id="p2", title="Test 2", year=paper2_year)
    assert GraphBuilderStrategy.temporal_similarity(paper1, paper2) == expected


def test_citation_similarity_branches() -> None:
    """Citation similarity should handle nearby, distant, and zero-citation branches."""
    near_1 = Paper(paper_id="p1", title="Test 1", year=2020, citation_count=100)
    near_2 = Paper(paper_id="p2", title="Test 2", year=2020, citation_count=105)
    assert GraphBuilderStrategy.citation_similarity(near_1, near_2) > 0.9

    far_1 = Paper(paper_id="p3", title="Test 3", year=2020, citation_count=10)
    far_2 = Paper(paper_id="p4", title="Test 4", year=2020, citation_count=1000)
    assert 0.1 < GraphBuilderStrategy.citation_similarity(far_1, far_2) < 0.7

    zero_1 = Paper(paper_id="p5", title="Test 5", year=2020, citation_count=0)
    zero_2 = Paper(paper_id="p6", title="Test 6", year=2020, citation_count=100)
    assert GraphBuilderStrategy.citation_similarity(zero_1, zero_2) == 0.3


@pytest.mark.parametrize(
    ("doi", "expected"),
    [
        ("10.48550/arXiv.2310.03262V3", "arxiv:2310.03262"),
        ("10.48550/arXiv.HEP-TH/9901001v2", "arxiv:hep-th/9901001"),
        ("10.48550/arxiv.2310.03262vX", None),
        ("10.48550/arxiv.arxiv:2310.03262", None),
        ("10.48550/arxiv.2313.03262", None),
        ("10.1000/arxiv.2310.03262", None),
    ],
)
def test_arxiv_issued_doi_identity_is_validated(doi: str, expected: str | None) -> None:
    """Only an arXiv-issued DOI with a valid suffix supplies an arXiv identity."""
    assert arxiv_identifier_from_doi(doi) == expected
    assert normalize_paper_id(f"doi:{doi}") == doi.lower()


def test_identity_arxiv_issued_doi_respects_conflicts_and_local_rows() -> None:
    """DOI-derived aliases obey the same identity boundaries as explicit arXiv IDs."""
    local = _paper("arxiv:2310.03262")
    local.is_local_corpus = True
    local.doi = "10.1000/shared"
    other_local = _paper("local-copy")
    other_local.is_local_corpus = True
    other_local.doi = "10.48550/arxiv.2310.03262"
    contradictory = _paper("doi:10.48550/arxiv.2310.03263")
    contradictory.arxiv_id = "2310.03262"
    contradictory.doi = local.doi
    different = _paper("doi:10.48550/arxiv.2310.03263")
    different.doi = local.doi

    assert not candidate_records_match(local, other_local)
    assert not candidate_identifiers_conflict(local, other_local)
    assert not candidate_records_match(local, contradictory)
    assert candidate_identifiers_conflict(local, contradictory)
    assert not candidate_records_match(local, different)
    assert candidate_identifiers_conflict(local, different)


def test_hybrid_merges_passuntil_doi_alias_and_retains_recommendation_rank() -> None:
    """DOI-only and arXiv-only provider records join the same local work."""
    builder = HybridGraphBuilder(max_papers=4, client=MagicMock())
    focal = _focal_paper()
    pool = DiscoveryPool(focal, builder.focal_relations)
    candidates, sources = pool.papers, pool.sources
    doi_only = _paper("45cbff3a7e803dd949628a91bedef52b28bd63c7")
    doi_only.title = "Unlock Predictable Scaling from Emergent Abilities"
    doi_only.doi = "10.48550/arXiv.2310.03262"
    arxiv_only = _paper("b9fb5262a6c6778f29f54fc8ee502c65bd2d7f65")
    arxiv_only.title = (
        "Predicting Emergent Abilities with Infinite Resolution Evaluation"
    )
    arxiv_only.arxiv_id = "2310.03262"
    local = _paper("arxiv:2310.03262")
    local.is_local_corpus = True
    local.title = arxiv_only.title

    pool.add(arxiv_only, source="citation", relation="cites_focal")
    pool.add(local, source="semantic", relation="semantic_only")
    recommendations = CandidateSourceResult(
        "recommendations",
        CandidateSourceState.COMPLETE,
        papers=(doi_only, arxiv_only),
    )
    builder._ingest(pool, recommendations, {})

    assert set(candidates) == {local.paper_id}
    assert sources[local.paper_id] == {"citation", "semantic"}
    assert builder.focal_relations[local.paper_id] == "cites_focal"
    # Both recommendation records name the corpus work; the first position counts.
    assert pool.recommendation_ranks == {local.paper_id: 1}


@pytest.fixture
def unlinked_publication_pair() -> tuple[Paper, Paper]:
    """A corpus preprint and published record without a shared external ID."""
    return build_publication_pair()


def test_hybrid_collection_preserves_reconciled_publication_reference(
    monkeypatch: pytest.MonkeyPatch,
    unlinked_publication_pair: tuple[Paper, Paper],
) -> None:
    """A cited publication folds into its corpus preprint before ranking."""
    corpus, publication = unlinked_publication_pair
    focal = _focal_paper()
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [publication]
    client.get_paper_citations.return_value = []
    builder = HybridGraphBuilder(
        max_papers=3, semantic_source="arxiv-corpus", client=client
    )
    stub_hybrid_corpus(builder, neighbours=[corpus])
    offline_hybrid_encoder(monkeypatch, builder)
    builder.reference_hydrator.hydrate = MagicMock()

    papers = builder.collect_papers(focal.paper_id)

    # The version pair is one member before ranking.
    assert [item.paper_id for item in builder.discovery_ranking] == [corpus.paper_id]
    assert list(papers) == [focal.paper_id, corpus.paper_id]
    assert papers[corpus.paper_id].doi == publication.doi
    assert builder.focal_relations[corpus.paper_id] == "referenced_by_focal"
    assert builder.paper_sources[corpus.paper_id] == "both"


def test_recommendation_collect_merges_external_primary_aliases() -> None:
    """Recommendation ingestion collapses external primary aliases into S2 records."""
    focal = _focal_paper("s2-focal")
    focal.arxiv_id = "2508.12345"
    candidate = _paper("s2-candidate")
    candidate.arxiv_id = "2508.12346"
    candidate_alias = _paper("arxiv:2508.12346")
    focal_alias = _paper("arxiv:2508.12345")
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_recommended_papers.return_value = [
        focal_alias,
        candidate,
        candidate_alias,
    ]

    papers = RecommendationGraphBuilder(
        max_papers=4, fetch_references=False, client=client
    ).collect_papers("s2-focal")

    assert set(papers) == {"s2-focal", "s2-candidate"}
    assert papers["s2-focal"].is_focal is True


def test_recommendation_collect_merges_s2_focal_duplicate_by_external_id() -> None:
    """A recommended S2 duplicate of the focal paper must not become a graph node."""
    focal = _focal_paper("1" * 40)
    focal.arxiv_id = "2508.12345"
    duplicate = _paper("2" * 40)
    duplicate.arxiv_id = "2508.12345"
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_recommended_papers.return_value = [duplicate]

    papers = RecommendationGraphBuilder(
        max_papers=3, fetch_references=False, client=client
    ).collect_papers(focal.paper_id)

    assert set(papers) == {focal.paper_id}
    assert papers[focal.paper_id].is_focal is True


def test_recommendation_collect_merges_exact_primary_refresh() -> None:
    """A repeated primary ID may supplement its existing record."""
    focal = _focal_paper()
    arxiv_record, doi_record, _bridge = _identity_bridge_records()
    sparse_bridge = Paper(
        paper_id=arxiv_record.paper_id,
        title="Bridge",
        year=2025,
        abstract="",
        arxiv_id="2508.12345",
        doi="10.1000/bridge",
    )
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_recommended_papers.return_value = [
        arxiv_record,
        doi_record,
        sparse_bridge,
    ]

    papers = RecommendationGraphBuilder(
        max_papers=4, fetch_references=False, client=client
    ).collect_papers("focal")

    assert set(papers) == {"focal", arxiv_record.paper_id, doi_record.paper_id}
    assert papers[arxiv_record.paper_id].doi == "10.1000/bridge"


def test_recommendation_collect_capacity_keeps_first_distinct_records() -> None:
    """Capacity applies after exact-primary deduplication."""
    focal = _focal_paper()
    arxiv_record, doi_record, bridge = _identity_bridge_records()
    unrelated = _paper("unrelated")
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_recommended_papers.return_value = [
        arxiv_record,
        doi_record,
        unrelated,
        bridge,
    ]

    papers = RecommendationGraphBuilder(
        max_papers=3, fetch_references=False, client=client
    ).collect_papers("focal")

    assert set(papers) == {"focal", arxiv_record.paper_id, doi_record.paper_id}


def test_merge_paper_metadata_preserves_fields_and_unions_references() -> None:
    """Canonical metadata merging should repair gaps without discarding rich fields."""
    from citemesh.strategies.candidates import merge_paper_metadata

    preferred = Paper(
        paper_id="canonical",
        title="Unknown",
        year=None,
        references=[" existing-ref ", "", "duplicate-ref"],
        is_focal=False,
    )
    incoming = Paper(
        paper_id="alternate",
        title="Recovered Title",
        year=2023,
        authors=[Author(name="Ada Lovelace")],
        citation_count=42,
        abstract="Recovered abstract",
        venue="Recovered venue",
        arxiv_id="2508.12345",
        doi="10.1000/example",
        categories=["cs.AI"],
        references=["duplicate-ref", "new-ref", "  ", "new-ref"],
        is_focal=True,
    )

    merged = merge_paper_metadata(preferred, incoming)

    assert merged is preferred
    assert merged.title == "Recovered Title"
    assert merged.year == 2023
    assert merged.authors == [Author(name="Ada Lovelace")]
    assert merged.citation_count == 42
    assert merged.abstract == "Recovered abstract"
    assert merged.venue == "Recovered venue"
    assert merged.arxiv_id == "2508.12345"
    assert merged.doi == "10.1000/example"
    assert merged.categories == ["cs.AI"]
    assert merged.references == ["existing-ref", "duplicate-ref", "new-ref"]
    assert merged.is_focal is True

    richer = Paper(
        paper_id="richer",
        title="Canonical Title",
        year=2024,
        abstract="Canonical abstract",
        venue="Canonical venue",
        references=["keep"],
    )
    merge_paper_metadata(richer, incoming)
    assert richer.title == "Canonical Title"
    assert richer.abstract == "Canonical abstract"
    assert richer.venue == "Canonical venue"
    assert richer.references == ["keep", "duplicate-ref", "new-ref"]


def test_merge_paper_metadata_keeps_maximum_citation_count_in_either_order() -> None:
    """Citation counts should merge monotonically regardless of arrival order."""
    from citemesh.strategies.candidates import merge_paper_metadata

    low_count = Paper(paper_id="low", title="Paper", year=2024, citation_count=1)
    high_count = Paper(paper_id="high", title="Paper", year=2024, citation_count=100)
    assert merge_paper_metadata(low_count, high_count).citation_count == 100

    low_count = Paper(paper_id="low", title="Paper", year=2024, citation_count=1)
    high_count = Paper(paper_id="high", title="Paper", year=2024, citation_count=100)
    assert merge_paper_metadata(high_count, low_count).citation_count == 100


@pytest.mark.parametrize(
    ("paper_id", "arxiv_id", "doi"),
    [
        ("arxiv:2508.12345v2", "2508.12345", ""),
        ("10.1000/example", "", "10.1000/example"),
    ],
)
@pytest.mark.parametrize("canonical_has_external_id", [True, False])
def test_metadata_merge_preserves_external_ids_from_primary_identifiers(
    paper_id: str, arxiv_id: str, doi: str, canonical_has_external_id: bool
) -> None:
    """Collapsed records must retain primary external identifiers for export.

    :param str paper_id: External primary identifier before merging.
    :param str arxiv_id: Expected preserved arXiv metadata.
    :param str doi: Expected preserved DOI metadata.
    :param bool canonical_has_external_id: Whether the retained record has the external ID.
    :return None: Assertions validate both canonical and discarded identifiers.
    """
    from citemesh.strategies.candidates import merge_paper_metadata

    preferred = _paper(paper_id if canonical_has_external_id else "a" * 40)
    incoming = _paper("a" * 40 if canonical_has_external_id else paper_id)

    merged = merge_paper_metadata(preferred, incoming)

    assert merged is preferred
    assert merged.arxiv_id == arxiv_id
    assert merged.doi == doi


@pytest.mark.parametrize(
    ("pool_size", "expected"),
    [
        (1, (0, 0, 1)),
        (2, (1, 0, 1)),
        (3, (1, 1, 1)),
        (4, (1, 2, 1)),
        (20, (5, 10, 5)),
        (100, (25, 50, 25)),
        (400, (100, 200, 100)),
    ],
)
def test_embedding_candidate_budgets_include_every_source_when_possible(
    pool_size: int, expected: tuple[int, int, int]
) -> None:
    """Small pools must keep relation sources without exceeding the total budget.

    :param int pool_size: Candidate fetch budget.
    :param tuple[int, int, int] expected: Reference, citation, recommendation budgets.
    :return None: Assertions preserve default allocation and source coverage.
    """
    builder = EmbeddingGraphBuilder(candidate_pool_size=pool_size, client=MagicMock())

    budgets = builder._candidate_pool_budgets()

    assert budgets == expected
    assert sum(budgets) == pool_size
    if pool_size >= 3:
        assert all(budget > 0 for budget in budgets)


def test_citation_focal_relations_use_shared_precedence_rules() -> None:
    """Citation relation updates should retain the shared focal-label semantics."""
    builder = CitationGraphBuilder(client=MagicMock())
    builder.focal_relations = {"focal": "focal", "semantic": "semantic_only"}

    builder._record_focal_relations(["focal", "semantic", "shared"], "cites_focal")
    builder._record_focal_relations(["shared"], "referenced_by_focal")

    assert builder.focal_relations == {
        "focal": "focal",
        "semantic": "cites_focal",
        "shared": "overlap",
    }


def test_recommendation_duplicates_merge_without_replacing_richer_record() -> None:
    """Recommendation duplicates should preserve the initial canonical object."""
    focal = _focal_paper()
    existing = Paper(
        paper_id="duplicate",
        title="Canonical Title",
        year=2024,
        authors=[Author(name="Existing Author")],
        citation_count=100,
        abstract="Canonical abstract",
        venue="Canonical venue",
        references=["existing-ref"],
    )
    incoming = Paper(
        paper_id="duplicate",
        title="Incoming Title",
        year=2023,
        citation_count=1,
        abstract="Incoming abstract",
        doi="10.1000/new",
        references=["existing-ref", "incoming-ref"],
    )
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_recommended_papers.return_value = [existing, incoming]

    papers = RecommendationGraphBuilder(
        max_papers=3,
        fetch_references=False,
        client=client,
    ).collect_papers("focal")

    assert papers["duplicate"] is existing
    assert existing.title == "Canonical Title"
    assert existing.abstract == "Canonical abstract"
    assert existing.venue == "Canonical venue"
    assert existing.doi == "10.1000/new"
    assert existing.references == ["existing-ref", "incoming-ref"]


def test_embedding_candidate_mode_skips_corpus_and_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate mode should rank S2 neighbors without touching corpus hydration."""
    from tests._helpers import SeededRandomEncodeModel

    client = MagicMock()
    client.get_paper.return_value = _focal_paper()
    client.get_paper_references.return_value = [_paper("r1"), _paper("r2")]
    client.get_paper_citations.return_value = [_paper("c1")]
    client.get_recommended_papers.return_value = [_paper("rec1"), _paper("rec2")]

    builder = EmbeddingGraphBuilder(max_papers=4, client=client)
    assert builder.semantic_source == "candidates"
    assert builder.truncate_dim == 512
    assert builder.storage_precision == "float32"
    assert "mode=candidates" in builder.embedding_cache.model_name

    def _raise_hydration(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("corpus hydration must not run in candidate mode")

    monkeypatch.setattr(builder, "_ensure_cache_hydrated", _raise_hydration)
    monkeypatch.setattr(builder, "_load_model", lambda: None)
    fingerprint_checks: list[str] = []
    monkeypatch.setattr(
        builder,
        "_ensure_cache_model_fingerprint",
        lambda: fingerprint_checks.append("first"),
    )
    builder.model = SeededRandomEncodeModel(embedding_dim=4)

    papers = builder.collect_papers("focal")

    assert "focal" in papers
    assert len(papers) == 4
    assert client.get_recommended_papers.called
    client.get_papers.assert_not_called()
    client.get_paper.assert_called_once_with("focal", raise_on_unavailable=True)
    non_focal = [paper_id for paper_id in papers if paper_id != "focal"]
    for paper_id in non_focal:
        assert paper_id in builder.retrieval_embeddings
    assert builder.candidate_source_status == {
        "citations": "complete",
        "recommendations": "complete",
        "references": "complete",
    }

    # A fresh builder must hit the persisted candidate cache without encoding.
    class _RaisingModel:
        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            raise AssertionError("cache hit expected; encode must not run")

    second = EmbeddingGraphBuilder(max_papers=4, client=client)
    monkeypatch.setattr(second, "_load_model", lambda: None)
    monkeypatch.setattr(
        second,
        "_ensure_cache_model_fingerprint",
        lambda: fingerprint_checks.append("second"),
    )
    second.model = _RaisingModel()
    cached = second.embed_papers({paper_id: papers[paper_id] for paper_id in non_focal})
    assert sorted(cached) == sorted(non_focal)
    assert fingerprint_checks == ["first", "second"]


def test_embedding_local_search_validates_cache_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local search should validate model identity before reading cached vectors."""
    from tests._helpers import SeededRandomEncodeModel

    builder = EmbeddingGraphBuilder(max_papers=4, client=MagicMock())
    monkeypatch.setattr(builder, "_load_model", lambda: None)
    builder.model = SeededRandomEncodeModel(embedding_dim=4)
    cache_events: list[str] = []
    monkeypatch.setattr(
        builder,
        "_ensure_cache_model_fingerprint",
        lambda: cache_events.append("fingerprint"),
    )
    builder.embedding_cache.search = MagicMock(
        side_effect=lambda **_kwargs: cache_events.append("search") or []
    )
    builder.embedding_cache.hydration_operation_lock = MagicMock()

    assert builder.search_local("cached topic", top_k=2) == []
    assert cache_events == ["fingerprint", "search"]
    builder.embedding_cache.hydration_operation_lock.assert_not_called()


def test_embedding_corpus_local_search_holds_hydration_operation_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corpus local search must hold the operation lock while reading vectors.

    :param pytest.MonkeyPatch monkeypatch: Runtime patch helper.
    :return None: Verifies the cache search runs inside the hydration lock.
    """
    from tests._helpers import SeededRandomEncodeModel

    builder = EmbeddingGraphBuilder(
        max_papers=4,
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    )
    monkeypatch.setattr(builder, "_load_model", lambda: None)
    monkeypatch.setattr(builder, "_ensure_cache_model_fingerprint", lambda: None)
    hydrate = MagicMock(
        side_effect=AssertionError(
            "local search must not mutate corpus lifecycle state"
        )
    )
    monkeypatch.setattr(builder, "_ensure_cache_hydrated", hydrate)
    builder.model = SeededRandomEncodeModel(embedding_dim=4)
    cache = builder.embedding_cache
    lock_held = False

    @contextmanager
    def hydration_operation_lock() -> Iterator[None]:
        """Track the operation-lock lifetime around the cache read.

        :return Iterator[None]: Context manager body executed under the fake lock.
        """
        nonlocal lock_held
        lock_held = True
        try:
            yield
        finally:
            lock_held = False

    def search(**_kwargs: object) -> list[object]:
        """Require the cache read to occur while hydration is excluded.

        :param object _kwargs: Cache search arguments ignored by the assertion.
        :return list[object]: Empty local-search result list.
        """
        assert lock_held is True
        return []

    cache.hydration_operation_lock = MagicMock(side_effect=hydration_operation_lock)
    cache.search = MagicMock(side_effect=search)

    assert builder.search_local("cached topic", top_k=2) == []
    hydrate.assert_not_called()
    cache.hydration_operation_lock.assert_called_once_with()
    assert lock_held is False


def test_hybrid_candidate_mode_uses_recommendations_not_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid candidate mode should source semantics from S2 recommendations."""
    client = MagicMock()
    client.get_paper.return_value = _focal_paper()
    client.get_paper_references.return_value = [_paper("c1"), _paper("c2")]
    client.get_paper_citations.return_value = []
    client.get_recommended_papers.return_value = [_paper("rec1"), _paper("rec2")]
    client.get_reference_ids_batch.return_value = ReferenceBatchResult({})

    builder = HybridGraphBuilder(max_papers=6, client=client)
    assert builder.semantic_source == "candidates"
    assert builder.embedding_builder.truncate_dim == 512

    def _raise_corpus(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("corpus work must not run in candidate mode")

    for name in (
        "_ensure_cache_hydrated",
        "_prepare_corpus_for_build",
        "search_prepared_corpus",
        "cached_corpus_vectors",
    ):
        monkeypatch.setattr(builder.embedding_builder, name, _raise_corpus)
    offline_hybrid_encoder(monkeypatch, builder)

    papers = builder.collect_papers("focal")

    assert set(papers) == {"focal", "c1", "c2", "rec1", "rec2"}
    client.get_recommended_papers.assert_called_once_with(
        "focal",
        limit=200,
        raise_on_unavailable=True,
        pool="all-cs",
    )
    assert builder.candidate_source_status == {
        "references": "complete",
        "citations": "empty",
        "recommendations": "complete",
    }
    for paper_id in ("rec1", "rec2"):
        assert builder.paper_sources[paper_id] == "semantic"
        assert builder.focal_relations[paper_id] == "semantic_only"


@pytest.mark.parametrize("recommendation_id", ["reference", "citation"])
def test_hybrid_both_merges_ordered_sources_and_preserves_citation_evidence(
    monkeypatch: pytest.MonkeyPatch,
    recommendation_id: str,
) -> None:
    """An alias-free recommendation still joins the matching corpus paper."""
    focal = _focal_paper("local-focal")
    focal.is_local_corpus = True
    focal.arxiv_id = "2501.00001"
    reference = _paper("reference")
    reference.doi = "10.1000/shared"
    citation = _paper("citation")
    citation.doi = reference.doi
    corpus_overlap = _paper("corpus-overlap")
    corpus_overlap.is_local_corpus = True
    corpus_overlap.doi = reference.doi
    corpus_semantic = _paper("corpus-semantic")
    corpus_semantic.is_local_corpus = True
    corpus_semantic.doi = "10.1000/semantic"
    semantic_alias = _paper("semantic-alias")
    semantic_alias.doi = corpus_semantic.doi
    recommendation = _paper(recommendation_id)
    recommendation_only = _paper("recommendation-only")
    events: list[str] = []

    def recommendations(*_args: object, **_kwargs: object) -> list[Paper]:
        events.append("recommendations")
        return [semantic_alias, recommendation_only, recommendation]

    def corpus(*_args: object) -> list[Paper]:
        events.append("corpus")
        return [focal, corpus_overlap, corpus_semantic]

    client = MagicMock()
    client.get_paper_references.return_value = [reference]
    client.get_paper_citations.return_value = [citation]
    client.get_recommended_papers.side_effect = recommendations
    builder = HybridGraphBuilder(
        semantic_source="both",
        max_papers=5,
        max_references=1,
        max_citations=1,
        fetch_references=False,
        client=client,
    )
    assert builder.embedding_builder.semantic_source == "arxiv-corpus"
    search = stub_hybrid_corpus(builder, focal=focal)
    search.side_effect = corpus
    offline_hybrid_encoder(monkeypatch, builder)
    builder.embedding_builder._embedding_runtime_metadata = MagicMock(return_value={})
    monkeypatch.setattr(builder, "prepare_graph_scoring", lambda _: None)

    graph, _ = builder.build_graph(focal.paper_id)

    assert events == ["recommendations", "corpus", "corpus"]
    client.get_recommended_papers.assert_called_once_with(
        "arxiv:2501.00001", limit=200, raise_on_unavailable=True, pool="all-cs"
    )
    assert set(graph) == {
        focal.paper_id,
        corpus_overlap.paper_id,
        corpus_semantic.paper_id,
        recommendation_only.paper_id,
    }
    assert builder.discovery_pool is not None
    assert builder.discovery_pool.recommendation_ranks == {
        corpus_overlap.paper_id: 3,
        corpus_semantic.paper_id: 1,
        recommendation_only.paper_id: 2,
    }
    assert builder.paper_sources[corpus_overlap.paper_id] == "both"
    assert builder.focal_relations[corpus_overlap.paper_id] == "overlap"
    assert builder.paper_sources[corpus_semantic.paper_id] == "semantic"
    assert builder.focal_relations[corpus_semantic.paper_id] == "semantic_only"
    assert builder.candidate_source_status == {
        "arxiv_corpus": "complete",
        "citations": "complete",
        "recommendations": "complete",
        "references": "complete",
    }


@pytest.mark.parametrize(
    ("lookup_available", "outage", "recommendation_status"),
    [
        (True, True, "unavailable"),
        (False, False, "unavailable"),
        (True, False, "empty"),
    ],
)
def test_hybrid_both_continues_with_an_empty_or_unavailable_recommendation_source(
    monkeypatch: pytest.MonkeyPatch,
    lookup_available: bool,
    outage: bool,
    recommendation_status: str,
) -> None:
    """Empty results, outages, and opaque focal papers preserve corpus candidates."""
    focal = _focal_paper("local-focal")
    focal.is_local_corpus = True
    focal.arxiv_id = "2501.00001" if lookup_available else ""
    client = MagicMock()
    client.get_recommended_papers.return_value = []
    if outage:
        client.get_recommended_papers.side_effect = s2.SemanticScholarUnavailableError(
            "recommendation outage"
        )
    builder = HybridGraphBuilder(
        semantic_source="both",
        max_papers=2,
        max_references=0,
        max_citations=0,
        client=client,
    )
    stub_hybrid_corpus(builder, focal=focal, neighbours=[focal, _paper("local-result")])
    offline_hybrid_encoder(monkeypatch, builder)

    assert set(builder.collect_papers(focal.paper_id)) == {
        focal.paper_id,
        "local-result",
    }
    assert builder.candidate_source_status == {
        "recommendations": recommendation_status,
        "arxiv_corpus": "complete",
    }
    assert client.get_recommended_papers.call_count == int(lookup_available)


@pytest.mark.parametrize(
    ("sources", "cosine"),
    [
        (("citation", "citation"), 0.0),
        (("semantic", "semantic"), 0.48),
        (("citation", "semantic"), 0.71),
        (("both", "both"), 0.71),
    ],
)
def test_hybrid_edges_require_semantic_or_bibliographic_evidence(
    sources: tuple[str, str],
    cosine: float,
) -> None:
    """Publication era and popularity cannot create hybrid edges alone.

    :param tuple[str, str] sources: Candidate provenance for both papers.
    :param float cosine: Insufficient semantic similarity despite positive scores.
    :return None: Verifies absent unsupported edges and retained bibliographic evidence.
    """
    builder = HybridGraphBuilder(max_papers=2, client=MagicMock())
    builder.paper_sources = dict(zip(("a", "b"), sources))
    builder.embedding_builder.embeddings = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([cosine, np.sqrt(1.0 - cosine**2)], dtype=np.float32),
    }
    a = _paper("a")
    b = _paper("b")
    score = builder.compute_similarity(a, b)
    assert score == 0.0
    assert not builder.should_create_edge(a, b, score)
    a.references = b.references = ["shared"]
    assert builder.compute_similarity(a, b) > 0.0


@pytest.mark.parametrize("prebuilt", [False, True])
def test_text_index_handles_empty_vocabulary(prebuilt: bool) -> None:
    """Stop-word-only titles produce no lexical evidence, including after a rebuild.

    :param bool prebuilt: Whether the index previously contained usable text.
    :return None: Verifies zero similarity without an exception or stale scores.
    """
    index = AbstractSimilarityIndex()
    if prebuilt:
        index.build(
            {
                paper_id: Paper(paper_id, "Quantum field theory", 2024)
                for paper_id in ("a", "b")
            }
        )
        assert index.similarity("a", "b") == pytest.approx(1.0)
    index.build(
        {
            "a": Paper("a", "To be or not to be", 2024),
            "b": Paper("b", "What is it", 2024),
        }
    )
    assert index.similarity("a", "b") == 0.0


def test_citation_build_retains_shared_references_with_empty_vocabulary() -> None:
    """Title-only stop words do not prevent a bibliographically supported graph.

    :return None: Verifies successful graph construction with a shared-reference edge.
    """
    focal = Paper("focal", "To be or not to be", 2024, references=["shared"])
    peer = Paper("peer", "What is it", 2024, references=["shared"])
    client = MagicMock()
    client.get_paper.return_value = focal
    client.get_paper_references.return_value = [peer]
    builder = CitationGraphBuilder(
        max_papers=2, max_references=1, max_citations=0, client=client
    )
    graph, _ = builder.build_graph("focal")
    assert graph.has_edge("focal", "peer")


@pytest.mark.parametrize("cosine", [0.0, -0.1, 0.48, 0.71])
def test_embedding_graph_requires_calibrated_semantic_evidence(
    monkeypatch: pytest.MonkeyPatch, cosine: float
) -> None:
    """Metadata bonuses cannot connect semantically unsupported paper pairs.

    :param pytest.MonkeyPatch monkeypatch: Fixture for replacing acquisition and inference.
    :param float cosine: Unsupported candidate's cosine against the focal paper.
    :return None: Verifies the supported edge survives and the unrelated paper stays isolated.
    """
    papers = {
        paper_id: Paper(
            paper_id,
            title,
            2024,
            authors=[Author(name="Shared Author")],
            categories=["Computer Science"],
            is_focal=paper_id == "focal",
        )
        for paper_id, title in (
            ("focal", "Quantum field theory"),
            ("related", "Quantum field interactions"),
            ("unrelated", "Medieval pottery"),
        )
    }
    builder = EmbeddingGraphBuilder(max_papers=3, client=MagicMock())
    monkeypatch.setattr(builder, "collect_papers", MagicMock(return_value=papers))
    monkeypatch.setattr(builder, "prepare_graph_scoring", MagicMock())
    builder.embeddings = {
        "focal": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "related": np.array([0.8, 0.6, 0.0], dtype=np.float32),
        "unrelated": np.array(
            [cosine, 0.0, np.sqrt(1.0 - cosine**2)], dtype=np.float32
        ),
    }
    graph, _ = builder.build_graph("focal")
    assert graph.has_edge("focal", "related")
    assert graph.degree("unrelated") == 0


@pytest.mark.parametrize("threshold", [0.65, 0.75])
@pytest.mark.parametrize("strategy", [EmbeddingGraphBuilder, HybridGraphBuilder])
def test_semantic_threshold_override_changes_edge_eligibility(
    strategy: type, threshold: float
) -> None:
    """Both scorers use the instance threshold without changing global defaults.

    :param type strategy: Embedding-aware graph builder class.
    :param float threshold: Threshold below or above the fixture cosine.
    :return None: The override controls the actual graph edge decision.
    """
    builder = strategy(client=MagicMock(), min_semantic_similarity=threshold)
    encoder = (
        builder if strategy is EmbeddingGraphBuilder else builder.embedding_builder
    )
    encoder.embeddings = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.7, np.sqrt(1.0 - 0.7**2)], dtype=np.float32),
    }
    a, b = _paper("a"), _paper("b")
    if strategy is HybridGraphBuilder:
        builder.paper_sources = {"a": "citation", "b": "citation"}
    score = builder.compute_similarity(a, b)
    assert bool(builder.should_create_edge(a, b, score)) == (threshold < 0.7)


@pytest.mark.parametrize(
    ("profile", "dimension", "warns"),
    [("auto", 512, False), ("auto", 128, True), ("default", 512, True)],
)
def test_semantic_calibration_warning_matches_profile_and_dimension(
    caplog: pytest.LogCaptureFixture, profile: str, dimension: int, warns: bool
) -> None:
    """Warn once when the active representation lacks calibration.

    :param pytest.LogCaptureFixture caplog: Captures calibration notices.
    :param str profile: Selected formatting profile.
    :param int dimension: Selected truncation dimension.
    :param bool warns: Whether the representation is outside calibration coverage.
    :return None: Checks the warning and its user-facing override guidance.
    """
    builder = EmbeddingGraphBuilder(
        client=MagicMock(), model_profile=profile, truncate_dim=dimension
    )
    builder._log_dimension_policy()
    builder._log_dimension_policy()
    notices = [r.message for r in caplog.records if "is uncalibrated" in r.message]
    assert len(notices) == int(warns)
    if warns:
        assert "--min-semantic-similarity" in notices[0]
