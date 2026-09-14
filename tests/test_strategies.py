"""Consolidated strategy behavior and determinism tests."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import networkx as nx
import numpy as np
import pytest
import requests

from citemesh.core import API_CONFIG, EMBEDDING_CONFIG, HYBRID_CONFIG, Author, Paper
from citemesh.data.model_profiles import compose_title_abstract_text
from citemesh.services import semantic_scholar as s2
from citemesh.services.semantic_scholar import SemanticScholarClient
from citemesh.strategies import hybrid as hybrid_strategy
from citemesh.strategies.base import (
    GraphBuilderStrategy,
    deterministic_sort_key,
    select_capped_undirected_edges,
)
from citemesh.strategies.candidates import (
    CandidateAcquisitionError,
    CandidatePool,
    IdentityRegistry,
    fetch_candidate_pool,
    paper_identity_aliases,
    register_aliases,
)
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import EmbeddingInferenceError, HybridGraphBuilder
from citemesh.strategies.recommendation import RecommendationGraphBuilder
from citemesh.strategies.similarity import AbstractSimilarityIndex
from tests._helpers import disable_embedding_dep_checks


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


def _seed_paper(paper_id: str = "seed") -> Paper:
    """Build deterministic seed paper fixture."""
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=2024,
        abstract="seed abstract",
        is_seed=True,
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
        "seed": Paper(paper_id="seed", title="Seed", year=2024, abstract="seed"),
        "a": Paper(paper_id="a", title="A", year=2024, abstract="alpha"),
        "b": Paper(paper_id="b", title="B", year=2024, abstract="beta"),
        "c": Paper(paper_id="c", title="C", year=2024, abstract="gamma"),
        "d": Paper(paper_id="d", title="D", year=2024, abstract="delta"),
    }
    papers["seed"].is_seed = True

    def fake_collect_papers(self, seed_id: str, **kwargs: object) -> dict[str, Paper]:
        del seed_id
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
    seed = Paper(paper_id="seed", title="Seed", year=2020, abstract="seed")
    seed.is_seed = True
    related = Paper(paper_id="related", title="Related", year=2021, abstract="related")
    other = Paper(paper_id="other", title="Other", year=2019, abstract="other")

    citation = CitationGraphBuilder(similarity_threshold=0.4, client=MagicMock())
    assert not citation.should_create_edge(seed, related, 0.39)
    assert citation.should_create_edge(seed, related, 0.4)
    assert not citation.should_create_edge(other, related, 0.39)
    assert citation.should_create_edge(other, related, 0.4)

    recommendation = RecommendationGraphBuilder(
        similarity_threshold=0.2, client=MagicMock()
    )
    assert recommendation.should_create_edge(seed, related, 0.2)
    assert not recommendation.should_create_edge(seed, related, 0.19)


@pytest.mark.parametrize(
    "builder_type",
    [
        CitationGraphBuilder,
        RecommendationGraphBuilder,
        EmbeddingGraphBuilder,
        HybridGraphBuilder,
    ],
)
def test_repeated_collection_preserves_client_owned_seed_role(
    builder_type: type[GraphBuilderStrategy], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reusing a seed's metadata as a later candidate must retain that candidate.

    :param type[GraphBuilderStrategy] builder_type: Public builder constructor.
    :param pytest.MonkeyPatch monkeypatch: Offline semantic inference stubs.
    :return None: Both collections have exactly one seed and preserve input roles.
    """
    first = _paper("first")
    second = _paper("second")
    client = MagicMock()
    client.get_paper.side_effect = [first, second]
    client.get_paper_references.return_value = [first]
    client.get_paper_citations.return_value = []
    client.get_recommended_papers.return_value = [first]
    client.get_reference_ids.return_value = []
    builder = builder_type(max_papers=2, client=client)
    if isinstance(builder, HybridGraphBuilder):
        monkeypatch.setattr(builder.embedding_builder, "_load_model", lambda: None)
        monkeypatch.setattr(
            builder,
            "_rank_candidates",
            lambda _seed, candidates, _sources: list(candidates),
        )
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

    first_collection = builder.collect_papers("first")
    second_collection = builder.collect_papers("second")

    assert first.is_seed is False
    assert second.is_seed is False
    assert first_collection["first"] is not first
    assert first_collection["first"].is_seed is True
    assert set(second_collection) == {"first", "second"}
    assert [p.paper_id for p in second_collection.values() if p.is_seed] == ["second"]


@pytest.mark.parametrize("provided_seed", [False, True])
def test_embedding_seed_resolution_preserves_input_role(provided_seed: bool) -> None:
    """Prefetched and client-returned seeds are bound to the build by copying.

    :param bool provided_seed: Whether the caller supplies the metadata directly.
    :return None: Seed resolution leaves the supplied metadata reusable.
    """
    paper = _paper("seed")
    client = MagicMock()
    client.get_paper.return_value = paper
    builder = EmbeddingGraphBuilder(client=client)

    resolved = builder._resolve_seed_paper("seed", paper if provided_seed else None)

    assert resolved is not paper
    assert resolved.is_seed is True
    assert paper.is_seed is False
    if provided_seed:
        client.get_paper.assert_not_called()


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
    seed = Paper(
        paper_id="seed",
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
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = [other]
    client.get_paper_citations.return_value = []
    client.get_recommended_papers.return_value = [other]
    client.get_reference_ids.return_value = []
    builder = builder_type(client=client, similarity_threshold=0.0)

    graph, _ = builder.build_graph("seed")

    assert set(graph) == {"seed", "other"}
    assert graph.number_of_edges() == 0
    assert "Graph contains no edges" in caplog.text
    assert builder.compute_similarity(seed, other) == 0.0
    other.references = ["shared"]
    caplog.clear()
    graph, _ = builder.build_graph("seed")
    assert graph.has_edge("seed", "other")
    assert "Graph contains no edges" not in caplog.text


@pytest.mark.parametrize(
    "builder_type", [CitationGraphBuilder, RecommendationGraphBuilder]
)
def test_reference_outage_warns_and_stops_hydration_until_next_collection(
    builder_type: type[GraphBuilderStrategy], caplog: pytest.LogCaptureFixture
) -> None:
    """One exhausted reference call skips remaining hydration and resets next build.

    :param type[GraphBuilderStrategy] builder_type: Indexed strategy constructor.
    :param pytest.LogCaptureFixture caplog: Captured user-visible warnings.
    :return None: Assertions validate outage visibility, retry scope, and contract errors.
    """
    from citemesh.services import SemanticScholarUnavailableError

    client = MagicMock()
    client.get_paper.return_value = _paper("seed", refs=["seed-ref"])
    related = [_paper("first"), _paper("second")]
    client.get_recommended_papers.return_value = related
    client.get_paper_references.return_value = related
    client.get_paper_citations.return_value = []
    client.get_reference_ids.side_effect = SemanticScholarUnavailableError("offline")
    client.get_cached_reference_ids.return_value = None
    builder = builder_type(client=client)

    with caplog.at_level(logging.WARNING):
        papers = builder.collect_papers("seed")

    assert set(papers) == {"seed", "first", "second"}
    client.get_reference_ids.assert_called_once_with("first", force_refresh=False)
    assert len(caplog.records) == 1
    assert "without further reference hydration" in caplog.text
    client.get_reference_ids.reset_mock(side_effect=True)
    client.get_reference_ids.return_value = ["restored"]

    papers = builder.collect_papers("seed")

    assert client.get_reference_ids.call_count == 2
    assert papers["first"].references == ["restored"]
    assert papers["second"].references == ["restored"]
    for paper in related:
        paper.references = []
    client.get_reference_ids.side_effect = ValueError("malformed references")
    with pytest.raises(ValueError, match="malformed references"):
        builder.collect_papers("seed")


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
    seed = _paper("seed", refs=["seed-reference"])
    cold = _paper("cold")
    warm = _paper("warm")

    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client.get_paper = MagicMock(return_value=seed)
        client.client.get_paper_references = MagicMock(
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
            papers = builder.collect_papers("seed")

    assert set(papers) == {"seed", "cold", "warm"}
    assert papers["warm"].references == expected_references
    assert client.client.get_paper_references.call_count == API_CONFIG.max_retries


@pytest.mark.parametrize(
    ("builder_type", "failure_path"),
    [
        (CitationGraphBuilder, "reference_ids"),
        (CitationGraphBuilder, "reference_papers"),
        (RecommendationGraphBuilder, "reference_ids"),
    ],
)
def test_candidate_scope_keeps_healthy_capabilities_after_reference_outage(
    builder_type: type[GraphBuilderStrategy],
    failure_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reference outage must not suppress healthy discovery capabilities.

    :param type[GraphBuilderStrategy] builder_type: Indexed strategy constructor.
    :param str failure_path: Reference-ID or full-paper request that exhausts retries.
    :param Path tmp_path: Isolated reference-cache directory.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate cache paths.
    :return None: Checks domain isolation, cache reuse, and fresh top-level scopes.
    """
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", tmp_path)
    seed = _paper("scope-seed")
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client.get_paper = MagicMock(return_value=seed)

        def fetch_references(paper_id: str, **kwargs: object) -> list[dict]:
            """Allow seed IDs only when full reference metadata is the failure path.

            :param str paper_id: Requested Semantic Scholar paper ID.
            :param object kwargs: SDK request options, including selected fields.
            :return list[dict]: One successful seed reference-ID record.
            """
            if failure_path == "reference_papers" and kwargs["fields"] == ["paperId"]:
                assert paper_id == "scope-seed"
                return [{"paperId": "seed-reference"}]
            raise requests.ConnectionError("offline")

        client.client.get_paper_references = MagicMock(side_effect=fetch_references)
        client.client.get_paper_citations = MagicMock(
            return_value=[_api_relation_record("healthy-citation")]
        )
        client._request_json_once = MagicMock(
            return_value={
                "recommendedPapers": [_recommendation_payload("healthy-recommendation")]
            }
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
            for expected_calls in (API_CONFIG.max_retries, API_CONFIG.max_retries * 2):
                with client.candidate_operation_scope():
                    papers = builder.collect_papers("scope-seed")
                    if builder_type is CitationGraphBuilder:
                        assert set(papers) == {"scope-seed", "healthy-citation"}
                        assert builder.candidate_source_status == {
                            "references": "unavailable",
                            "citations": "complete",
                        }
                    else:
                        assert set(papers) == {
                            "scope-seed",
                            "healthy-recommendation",
                        }
                        assert builder.candidate_source_status == {
                            "recommendations": "complete"
                        }
                    with pytest.raises(
                        s2.SemanticScholarUnavailableError,
                        match="Skipped Semantic Scholar references request",
                    ):
                        client.get_reference_ids("later-citing-paper")
                    assert client.get_reference_ids("warm") == ["cached-reference"]
                    successful_calls = int(failure_path == "reference_papers")
                    assert (
                        client.client.get_paper_references.call_count
                        == expected_calls + successful_calls
                    )

        assert sleep_mock.call_count == 2 * (API_CONFIG.max_retries - 1)

    if builder_type is CitationGraphBuilder:
        assert client.client.get_paper_citations.call_count == 2
        client._request_json_once.assert_not_called()
    else:
        client.client.get_paper_citations.assert_not_called()
        assert client._request_json_once.call_count == 2


def test_hybrid_candidate_scope_is_shared_with_citation_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hybrid nesting shares reference failures without poisoning recommendations.

    :param Path tmp_path: Isolated reference-cache directory.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate cache paths.
    :return None: Verifies reference retry sharing and healthy parent enrichment.
    """
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", tmp_path)
    seed = _paper("hybrid-scope-seed")
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client.get_paper = MagicMock(return_value=seed)
        client.client.get_paper_references = MagicMock(
            side_effect=requests.ConnectionError("offline")
        )
        client.client.get_paper_citations = MagicMock(
            return_value=[_api_relation_record("healthy-citation")]
        )
        client._request_json_once = MagicMock(
            return_value={
                "recommendedPapers": [_recommendation_payload("healthy-recommendation")]
            }
        )
        builder = HybridGraphBuilder(
            max_papers=3,
            max_references=1,
            max_citations=1,
            max_semantic=1,
            semantic_source="candidates",
            client=client,
        )
        assert builder.embedding_builder is not None
        monkeypatch.setattr(builder.embedding_builder, "_load_model", lambda: None)
        monkeypatch.setattr(
            builder,
            "_rank_candidates",
            lambda _seed, candidates, _sources: list(candidates),
        )

        with patch("time.sleep"):
            papers = builder.collect_papers("hybrid-scope-seed")

        assert client.client.get_paper_references.call_count == API_CONFIG.max_retries
        assert set(papers) == {
            "hybrid-scope-seed",
            "healthy-citation",
            "healthy-recommendation",
        }
        assert builder.candidate_source_status == {
            "references": "unavailable",
            "citations": "complete",
            "recommendations": "complete",
        }

    client.client.get_paper_citations.assert_called_once()
    client._request_json_once.assert_called_once()


def test_embedding_candidate_scope_keeps_healthy_later_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Embedding candidates retain citations and recommendations after a ref outage.

    :param Path tmp_path: Isolated reference-cache directory.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate cache paths.
    :return None: Verifies healthy source candidates and accurate status survive.
    """
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", tmp_path)
    seed = _paper("embedding-scope-seed")
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client.get_paper = MagicMock(return_value=seed)
        client.client.get_paper_references = MagicMock(
            side_effect=requests.ConnectionError("offline")
        )
        client.client.get_paper_citations = MagicMock(
            return_value=[_api_relation_record("healthy-citation")]
        )
        client._request_json_once = MagicMock(
            return_value={
                "recommendedPapers": [_recommendation_payload("healthy-recommendation")]
            }
        )
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
            papers = builder.collect_papers("embedding-scope-seed")

        assert client.client.get_paper_references.call_count == API_CONFIG.max_retries
        assert set(papers) == {
            "embedding-scope-seed",
            "healthy-citation",
            "healthy-recommendation",
        }
        assert builder.candidate_source_status == {
            "references": "unavailable",
            "citations": "complete",
            "recommendations": "complete",
        }

    client.client.get_paper_citations.assert_called_once()
    client._request_json_once.assert_called_once()


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
        client.get_paper_references = MagicMock(return_value=reference_papers)
        client.client.get_paper_citations = MagicMock(
            side_effect=requests.ConnectionError("offline")
        )
        client._request_json_once = MagicMock(
            return_value={
                "recommendedPapers": [_recommendation_payload("healthy-recommendation")]
            }
        )

        with patch("time.sleep"):
            pool = fetch_candidate_pool(
                client,
                _seed_paper("partial-scope-seed"),
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
        assert client.client.get_paper_citations.call_count == API_CONFIG.max_retries

    client._request_json_once.assert_called_once()


def test_recommendation_collect_filters_missing_abstract_and_prefers_payload_refs() -> (
    None
):
    """Recommendation collection should drop empty abstracts and keep provided refs."""
    with patch("citemesh.strategies.recommendation.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_paper.return_value = Paper(
            paper_id="seed", title="Seed", year=2020, abstract="seed abstract"
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
        mock_client.get_reference_ids.return_value = []

        builder = RecommendationGraphBuilder(max_papers=3, fetch_references=True)
        papers = builder.collect_papers("seed")

    assert "valid" in papers
    assert "missing" not in papers
    assert papers["valid"].references == ["r1", "r2"]
    mock_client.get_reference_ids.assert_called_once_with("seed", force_refresh=False)


def test_recommendation_collect_clamps_endpoint_request_limit() -> None:
    """Large recommendation graphs should stay within the endpoint's hard limit.

    :return None: Checks the oversampled request is capped at 500 results.
    """
    client = MagicMock()
    client.get_paper.return_value = _seed_paper()
    client.get_recommended_papers.return_value = []

    RecommendationGraphBuilder(
        max_papers=300,
        fetch_references=False,
        client=client,
    ).collect_papers("seed")

    client.get_recommended_papers.assert_called_once_with(
        "seed",
        limit=500,
        raise_on_unavailable=True,
    )


def test_citation_collect_populates_reference_cache_and_summary() -> None:
    """Citation collection should hydrate references and expose summary."""
    seed = _paper("seed", refs=["seed-ref"])
    ref = _paper("ref1")
    cit = _paper("cit1")

    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = [ref]
    client.get_paper_citations.return_value = [cit]
    client.get_reference_ids.side_effect = lambda pid, force_refresh=False: [
        f"{pid}-ref"
    ]

    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=True,
        client=client,
    )
    papers = builder.collect_papers("seed")

    assert set(papers) == {"seed", "ref1", "cit1"}
    assert papers["ref1"].references == ["ref1-ref"]
    assert papers["cit1"].references == ["cit1-ref"]
    assert (
        builder.get_collection_summary()
        == "Collected 3 papers (3 with reference lists)"
    )
    assert client.get_reference_ids.call_args_list == [
        call("ref1", force_refresh=False),
        call("cit1", force_refresh=False),
    ]


@pytest.mark.parametrize(
    ("external_fields", "provider_id"),
    [
        ({"arxiv_id": "2508.12345"}, "arxiv:2508.12345"),
        ({"doi": "10.1000/LOCAL.SEED"}, "10.1000/local.seed"),
    ],
)
def test_citation_collect_uses_external_alias_for_pre_resolved_local_seed(
    external_fields: dict[str, str], provider_id: str
) -> None:
    """A local graph ID must remain local while provider calls use its alias.

    :param dict[str, str] external_fields: External identifier field on the seed.
    :param str provider_id: Canonical identifier expected by provider methods.
    :return None: Verifies every upstream call and the local reference-cache key.
    """
    seed = Paper(
        paper_id="local-source-key",
        title="Local seed",
        year=2024,
        abstract="Local abstract",
        is_local_corpus=True,
        **external_fields,
    )
    client = MagicMock()
    client.get_reference_ids.return_value = ["seed-reference"]
    client.get_paper_references.return_value = [_paper("reference")]
    client.get_paper_citations.return_value = [_paper("citation")]
    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=True,
        client=client,
    )

    papers = builder.collect_papers(seed.paper_id, seed_paper=seed)

    assert set(papers) == {seed.paper_id, "reference", "citation"}
    assert papers[seed.paper_id].is_seed is True
    assert builder.reference_cache[seed.paper_id] == ["seed-reference"]
    client.get_paper.assert_not_called()
    client.get_reference_ids.assert_any_call(provider_id, force_refresh=False)
    client.get_paper_references.assert_called_once_with(
        provider_id, limit=1, raise_on_unavailable=True
    )
    client.get_paper_citations.assert_called_once_with(
        provider_id, limit=1, raise_on_unavailable=True
    )


def test_citation_collect_keeps_opaque_pre_resolved_local_seed_off_provider() -> None:
    """An opaque corpus seed without an alias must produce a seed-only branch."""
    seed = Paper(
        paper_id="local-source-key",
        title="Local seed",
        year=2024,
        abstract="Local abstract",
        is_local_corpus=True,
    )
    client = MagicMock()
    for provider_method in (
        "get_paper",
        "get_reference_ids",
        "get_paper_references",
        "get_paper_citations",
    ):
        getattr(client, provider_method).side_effect = AssertionError(
            "opaque local seed must not be sent to Semantic Scholar"
        )
    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=True,
        client=client,
    )

    papers = builder.collect_papers(seed.paper_id, seed_paper=seed)

    assert set(papers) == {seed.paper_id}
    assert papers[seed.paper_id].is_seed is True
    assert builder.reference_cache == {}
    assert builder.candidate_source_status == {}
    assert builder.candidate_source_results == ()


def test_citation_collect_sizes_relation_requests_to_open_graph_slots() -> None:
    """Relation requests should honor capacity and let citations fill duplicate gaps.

    :return None: Checks remaining-slot limits and a full graph after a seed alias.
    """
    seed = _seed_paper("seed")
    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = [
        _paper("seed"),
        _paper("ref1"),
    ]
    client.get_paper_citations.return_value = [_paper("cit1")]

    papers = CitationGraphBuilder(
        max_papers=3,
        max_references=25,
        max_citations=25,
        fetch_references=False,
        client=client,
    ).collect_papers("seed")

    assert set(papers) == {"seed", "ref1", "cit1"}
    client.get_paper_references.assert_called_once_with(
        "seed",
        limit=2,
        raise_on_unavailable=True,
    )
    client.get_paper_citations.assert_called_once_with(
        "seed",
        limit=1,
        raise_on_unavailable=True,
    )

    seed_only_client = MagicMock()
    seed_only_client.get_paper.return_value = _seed_paper("seed-only")
    seed_only_papers = CitationGraphBuilder(
        max_papers=1,
        max_references=25,
        max_citations=25,
        fetch_references=False,
        client=seed_only_client,
    ).collect_papers("seed-only")

    assert set(seed_only_papers) == {"seed-only"}
    seed_only_client.get_paper_references.assert_not_called()
    seed_only_client.get_paper_citations.assert_not_called()


def test_citation_collect_preserves_seed_when_relation_reuses_seed_id() -> None:
    """Duplicate relation payloads should not unset the canonical seed marker."""
    seed = _paper("seed", refs=["seed-ref"])
    duplicate_seed_record = _paper("seed", year=2021)

    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = [duplicate_seed_record]
    client.get_paper_citations.return_value = []

    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=0,
        fetch_references=False,
        similarity_threshold=0.0,
        client=client,
    )

    papers = builder.collect_papers("seed")
    assert papers["seed"].is_seed is True

    graph, seed_id = builder.build_graph("seed")
    assert seed_id == "seed"
    assert graph.nodes["seed"]["is_seed"] is True


def test_citation_collect_preserves_hydrated_references_for_overlap_duplicates() -> (
    None
):
    """Overlap duplicates should retain hydrated references on the canonical object."""
    seed = _paper("seed", refs=["seed-ref"])
    overlap_from_references = _paper("overlap")
    overlap_from_citations = _paper("overlap")

    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = [overlap_from_references]
    client.get_paper_citations.return_value = [overlap_from_citations]
    client.get_reference_ids.return_value = ["overlap-ref"]

    builder = CitationGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        fetch_references=True,
        client=client,
    )
    papers = builder.collect_papers("seed")

    assert papers["overlap"].references == ["overlap-ref"]
    assert builder.seed_relations["overlap"] == "overlap"
    client.get_reference_ids.assert_called_once_with("overlap", force_refresh=False)


def test_citation_collect_collapses_seed_and_candidate_identifier_aliases() -> None:
    """Citation ingestion should retain canonical IDs for aliased relation records."""
    seed = _paper("s2-seed")
    seed.arxiv_id = "2508.12345"
    reference = _paper("s2-candidate")
    reference.arxiv_id = "2508.12346"
    citation_alias = _paper("arxiv:2508.12346")
    seed_alias = _paper("arxiv:2508.12345")

    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = [seed_alias, reference]
    client.get_paper_citations.return_value = [citation_alias]

    builder = CitationGraphBuilder(
        max_papers=4,
        max_references=2,
        max_citations=1,
        fetch_references=False,
        client=client,
    )
    papers = builder.collect_papers("s2-seed")

    assert set(papers) == {"s2-seed", "s2-candidate"}
    assert papers["s2-seed"].is_seed is True
    assert builder.seed_relations == {
        "s2-seed": "seed",
        "s2-candidate": "overlap",
    }


def test_citation_collect_repoints_relations_for_same_batch_bridge() -> None:
    """Same-batch bridge merges must not leave relations for removed paper IDs."""
    seed = _paper("seed")
    arxiv_record, doi_record, bridge = _identity_bridge_records()

    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = [arxiv_record, doi_record, bridge]
    client.get_paper_citations.return_value = []

    builder = CitationGraphBuilder(
        max_papers=4,
        max_references=3,
        max_citations=0,
        fetch_references=False,
        client=client,
    )
    papers = builder.collect_papers("seed")

    assert set(papers) == {"seed", arxiv_record.paper_id}
    assert builder.seed_relations == {
        "seed": "seed",
        arxiv_record.paper_id: "referenced_by_seed",
    }


def test_citation_build_graph_persists_seed_relation_metadata() -> None:
    """Citation graph export metadata should preserve seed relation classes."""
    seed = _paper("seed", refs=["seed-ref"])
    ref = _paper("ref1")
    cit = _paper("cit1")

    client = MagicMock()
    client.get_paper.return_value = seed
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
    graph, seed_id = builder.build_graph("seed")

    assert seed_id == "seed"
    assert graph.graph["strategy"] == "citation"
    assert graph.graph["seed_relations"] == {
        "cit1": "cites_seed",
        "ref1": "referenced_by_seed",
        "seed": "seed",
    }
    assert graph.graph["candidate_source_status"] == {
        "citations": "complete",
        "references": "complete",
    }


def test_recommendation_build_graph_persists_strategy_metadata() -> None:
    """Recommendation graphs should persist strategy metadata for downstream exports."""
    client = MagicMock()
    client.get_paper.return_value = _seed_paper()
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
    graph, seed_id = builder.build_graph("seed")

    assert seed_id == "seed"
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

    seed = _seed_paper()
    partial_client = MagicMock()
    partial_client.get_paper_references.side_effect = SemanticScholarUnavailableError(
        "references down"
    )
    partial_client.get_paper_citations.return_value = []
    partial_client.get_recommended_papers.return_value = [_paper("rec1")]

    partial_pool = fetch_candidate_pool(
        partial_client,
        seed,
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
        seed,
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
            seed,
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
                is_seed=True,
            ),
            max_recommendations=1,
        )
    query_client.get_recommended_papers.assert_not_called()

    recommendation_client = MagicMock()
    recommendation_client.get_paper.return_value = seed
    recommendation_client.get_recommended_papers.side_effect = (
        SemanticScholarUnavailableError("recommendations down")
    )
    with pytest.raises(CandidateAcquisitionError, match="recommendations"):
        RecommendationGraphBuilder(
            max_papers=3,
            fetch_references=False,
            client=recommendation_client,
        ).collect_papers("seed")

    citation_client = MagicMock()
    citation_client.get_paper.return_value = seed
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
        ).collect_papers("seed")

    hybrid_client = MagicMock()
    hybrid_client.get_paper.return_value = seed
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
        max_semantic=1,
        fetch_references=False,
        client=hybrid_client,
    )
    assert hybrid_builder.embedding_builder is not None
    monkeypatch.setattr(hybrid_builder.embedding_builder, "_load_model", lambda: None)
    monkeypatch.setattr(
        hybrid_builder,
        "_rank_candidates",
        lambda _seed, candidates, _sources: list(candidates),
    )

    hybrid_papers = hybrid_builder.collect_papers("seed")

    assert set(hybrid_papers) == {"seed", "rec1"}
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
        hybrid_builder.collect_papers("seed")
    assert hybrid_client.get_recommended_papers.call_count == 2
    assert hybrid_builder.candidate_source_status == {
        "references": "unavailable",
        "citations": "unavailable",
        "recommendations": "unavailable",
    }


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

        :param str query: Search text expected from the synthetic seed.
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
            is_seed=True,
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


def test_hybrid_candidate_recommendation_limit_stays_within_source_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized budgets must not push the recommendation fetch past S2's cap.

    :param pytest.MonkeyPatch monkeypatch: Offline model and ranking stubs.
    :return None: Assertions pin the per-source recommendation request limit.
    """
    client = MagicMock()
    client.get_paper.return_value = _seed_paper()
    client.get_paper_references.return_value = []
    client.get_paper_citations.return_value = []
    client.get_recommended_papers.return_value = [_paper("rec1")]
    builder = HybridGraphBuilder(
        max_papers=800,
        max_semantic=400,
        candidate_pool_size=600,
        fetch_references=False,
        client=client,
    )
    monkeypatch.setattr(builder.embedding_builder, "_load_model", lambda: None)
    monkeypatch.setattr(
        builder,
        "_rank_candidates",
        lambda _seed, candidates, _sources: list(candidates),
    )

    builder.collect_papers("seed")

    assert client.get_recommended_papers.call_args.kwargs["limit"] == 100


def test_refresh_reference_cache_force_lookup_contracts() -> None:
    """Refresh mode should bypass stale memory entries and force service lookups."""
    citation_client = MagicMock()
    citation_client.get_reference_ids.return_value = ["fresh-ref"]
    citation_builder = CitationGraphBuilder(
        fetch_references=True,
        refresh_reference_cache=True,
        client=citation_client,
    )
    citation_builder.reference_cache["paper-1"] = ["stale-ref"]

    refs = citation_builder._get_references("paper-1")

    assert refs == ["fresh-ref"]
    citation_client.get_reference_ids.assert_called_once_with(
        "paper-1", force_refresh=True
    )
    assert citation_builder.reference_cache["paper-1"] == ["fresh-ref"]

    recommendation_client = MagicMock()
    recommendation_client.get_reference_ids.return_value = ["r2"]
    recommendation_builder = RecommendationGraphBuilder(
        fetch_references=True,
        refresh_reference_cache=True,
        client=recommendation_client,
    )
    paper = Paper(
        paper_id="paper-2",
        title="Paper 2",
        year=2020,
        abstract="paper two",
    )

    recommendation_builder._hydrate_references(paper)

    assert paper.references == ["r2"]
    recommendation_client.get_reference_ids.assert_called_once_with(
        "paper-2", force_refresh=True
    )


def test_hybrid_corpus_mode_survives_relation_endpoint_outage(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A working local corpus must remain usable when both S2 relations fail.

    :param pytest.MonkeyPatch monkeypatch: Offline collection and ranking stubs.
    :param pytest.LogCaptureFixture caplog: Captured availability warning.
    :return None: Assertions validate retained source status and semantic candidates.
    """
    from citemesh.services import SemanticScholarUnavailableError

    seed = _seed_paper()
    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_paper_references.side_effect = SemanticScholarUnavailableError("offline")
    client.get_paper_citations.side_effect = SemanticScholarUnavailableError("offline")
    client.get_reference_ids.side_effect = SemanticScholarUnavailableError("offline")
    builder = HybridGraphBuilder(
        max_papers=3,
        max_references=1,
        max_citations=1,
        max_semantic=1,
        semantic_source="arxiv-corpus",
        dataset_source="example/arxiv",
        client=client,
    )
    assert builder.embedding_builder is not None
    assert builder.embedding_builder.dataset_source == "example/arxiv"
    assert builder.embedding_builder.corpus_size is None
    monkeypatch.setattr(
        builder.embedding_builder, "resolve_cached_corpus_seed", lambda _seed_id: None
    )
    collect_corpus = MagicMock(
        return_value={"seed": seed, "semantic": _paper("semantic")}
    )
    monkeypatch.setattr(builder.embedding_builder, "collect_papers", collect_corpus)
    monkeypatch.setattr(
        builder, "_rank_candidates", lambda _seed, papers, _sources: list(papers)
    )

    with caplog.at_level(logging.WARNING):
        papers = builder.collect_papers("seed")

    assert set(papers) == {"seed", "semantic"}
    assert builder.candidate_source_status == {
        "references": "unavailable",
        "citations": "unavailable",
    }
    collect_corpus.assert_called_once_with("seed", seed_paper=seed)
    client.get_paper.assert_called_once_with("seed", raise_on_unavailable=True)
    client.get_reference_ids.assert_called_once_with("seed", force_refresh=False)
    client.get_recommended_papers.assert_not_called()
    assert (
        "Continuing hybrid corpus acquisition for seed with partial Semantic Scholar "
        "evidence (references: offline; citations: offline)." in caplog.text
    )


def test_citation_related_paper_reference_failure_remains_uncached() -> None:
    """One unavailable related-paper reference list should not abort collection."""
    from citemesh.services import SemanticScholarUnavailableError

    client = MagicMock()
    client.get_reference_ids.side_effect = SemanticScholarUnavailableError(
        "publisher elided references"
    )
    builder = CitationGraphBuilder(fetch_references=True, client=client)
    paper = _paper("related-paper")

    builder._ensure_paper_references(paper)

    assert paper.references == []
    assert paper.paper_id not in builder.reference_cache
    client.get_reference_ids.assert_called_once_with(
        paper.paper_id,
        force_refresh=False,
    )

    client.get_reference_ids.side_effect = RuntimeError("local hydration bug")
    builder = CitationGraphBuilder(fetch_references=True, client=client)
    with pytest.raises(RuntimeError, match="local hydration bug"):
        builder._ensure_paper_references(_paper("broken-paper"))


def test_citation_collect_clears_in_memory_reference_cache_between_requests() -> None:
    """Citation collection should scope in-memory reference cache to one request."""
    seed = _paper("seed", refs=["seed-ref"])
    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = []
    client.get_paper_citations.return_value = []

    builder = CitationGraphBuilder(
        max_papers=1,
        fetch_references=False,
        client=client,
    )
    builder.reference_cache["stale"] = ["stale-ref"]

    builder.collect_papers("seed")

    assert builder.reference_cache == {}


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


def test_hybrid_collection_merges_and_tracks_sources() -> None:
    """Hybrid collection should merge citation+semantic papers and source labels."""
    builder = HybridGraphBuilder(
        max_papers=5, max_semantic=2, semantic_source="arxiv-corpus", client=MagicMock()
    )

    seed = _paper("seed")
    seed.is_seed = True
    citation_papers = {"seed": seed, "c1": _paper("c1")}
    semantic_papers = {"c1": _paper("c1"), "s1": _paper("s1"), "s2": _paper("s2")}

    builder.citation_builder.collect_papers = MagicMock(return_value=citation_papers)
    builder.citation_builder.seed_relations = {
        "seed": "seed",
        "c1": "referenced_by_seed",
    }
    assert builder.embedding_builder is not None
    builder.embedding_builder.resolve_cached_corpus_seed = MagicMock(return_value=None)

    def _collect_semantic(*_args: object, **_kwargs: object) -> dict[str, Paper]:
        builder.embedding_builder.retrieval_embeddings = {
            paper_id: np.asarray([1.0, 0.0], dtype=np.float32)
            for paper_id in {"seed", "c1", "s1", "s2"}
        }
        return semantic_papers

    builder.embedding_builder.collect_papers = MagicMock(side_effect=_collect_semantic)

    papers = builder.collect_papers("seed")

    assert set(papers) == {"seed", "c1", "s1", "s2"}
    builder.embedding_builder.collect_papers.assert_called_once_with(
        "seed",
        seed_paper=seed,
    )
    assert builder.paper_sources["seed"] == "citation"
    assert builder.paper_sources["s1"] == "semantic"
    assert builder.seed_relations["seed"] == "seed"
    assert builder.seed_relations["c1"] == "referenced_by_seed"
    assert builder.seed_relations["s1"] == "semantic_only"


@pytest.mark.parametrize("seed_id", ["content:local-paper", "arxiv_7"])
def test_hybrid_without_semantic_enrichment_rejects_local_only_seed(
    seed_id: str,
) -> None:
    """A citation-only hybrid branch must not send local IDs to the provider.

    :param str seed_id: Current or legacy local-only corpus identifier.
    :return None: Verifies actionable guidance before any provider request.
    """
    client = MagicMock()
    builder = HybridGraphBuilder(max_papers=2, max_semantic=0, client=client)

    with pytest.raises(ValueError, match="--max-semantic above 0"):
        builder.collect_papers(seed_id)

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
        "seed": Paper(
            paper_id="seed",
            title="Seed",
            year=2024,
            authors=[Author(name=name) for name in author_names],
            is_seed=True,
        )
    }
    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    monkeypatch.setattr(builder, "collect_papers", lambda _seed_id, **_kwargs: papers)
    monkeypatch.setattr(builder, "prepare_graph_scoring", lambda _papers: None)

    graph, _ = builder.build_graph("seed")

    assert graph.nodes["seed"]["authors"] == author_names


def test_hybrid_uses_retrieval_vectors_for_rerank_and_graph_vectors_for_edges() -> None:
    """Hybrid seed ranking and pairwise topology must consume different spaces."""
    builder = HybridGraphBuilder(max_papers=2, max_semantic=1, client=MagicMock())
    assert builder.embedding_builder is not None
    seed = _seed_paper("seed")
    candidate = _paper("candidate", year=seed.year)
    builder.paper_sources = {"seed": "semantic", "candidate": "semantic"}
    builder.embedding_builder.retrieval_embeddings = {
        "seed": np.asarray([1.0, 0.0], dtype=np.float32),
        "candidate": np.asarray([1.0, 0.0], dtype=np.float32),
    }
    builder.embedding_builder.embeddings = {
        "seed": np.asarray([1.0, 0.0], dtype=np.float32),
        "candidate": np.asarray([0.0, 1.0], dtype=np.float32),
    }

    seed_vector = builder._ensure_candidate_embeddings(seed, {"candidate": candidate})
    assert seed_vector is not None
    assert float(
        np.dot(seed_vector, builder.embedding_builder.retrieval_embeddings["candidate"])
    ) == pytest.approx(1.0)

    graph_score = builder.compute_similarity(seed, candidate)
    assert graph_score == 0.0


def test_hybrid_max_citation_count_receives_full_rerank_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pool-leading citation count should receive the full citation contribution.

    :param pytest.MonkeyPatch monkeypatch: Isolates the citation score component.
    :return None: Checks citation normalization reaches one at the pool maximum.
    """
    builder = HybridGraphBuilder(max_papers=2, max_semantic=1, client=MagicMock())
    seed = _seed_paper("seed")
    candidate = Paper(
        paper_id="candidate",
        title="Candidate",
        year=seed.year,
        citation_count=7,
    )
    monkeypatch.setattr(builder, "temporal_similarity", lambda _left, _right: 0.0)
    monkeypatch.setattr(builder, "bibliographic_coupling", lambda _left, _right: 0.0)

    score = builder._seed_relevance_score(
        seed_embedding=None,
        seed_paper=seed,
        candidate=candidate,
        source_tags={"semantic"},
        max_citation_count=candidate.citation_count,
    )

    citation_weight = hybrid_strategy.HYBRID_SEED_RERANK_WEIGHTS[2]
    assert score == pytest.approx(citation_weight)


def test_hybrid_citation_weight_breaks_close_semantic_ranking() -> None:
    """A pool-leading citation count should overcome a small semantic deficit.

    :return None: Checks candidate order at the semantic/citation tradeoff boundary.
    """
    builder = HybridGraphBuilder(max_papers=3, max_semantic=2, client=MagicMock())
    assert builder.embedding_builder is not None
    seed = _seed_paper("seed")
    cited = Paper(
        paper_id="cited",
        title="Cited candidate",
        year=seed.year,
        citation_count=1,
    )
    uncited = Paper(
        paper_id="uncited",
        title="Uncited candidate",
        year=seed.year,
        citation_count=0,
    )
    builder.embedding_builder.retrieval_embeddings = {
        "seed": np.asarray([1.0, 0.0], dtype=np.float32),
        "cited": np.asarray([0.0, 1.0], dtype=np.float32),
        "uncited": np.asarray([0.36, np.sqrt(1.0 - 0.36**2)], dtype=np.float32),
    }

    ranked = builder._rank_candidates(
        seed_paper=seed,
        candidates={"cited": cited, "uncited": uncited},
        candidate_sources={"cited": {"semantic"}, "uncited": {"semantic"}},
    )

    assert ranked == ["cited", "uncited"]


def test_hybrid_graph_preparation_fails_closed_on_sts_inference() -> None:
    """Hybrid should surface graph-space inference failures before edge creation."""
    builder = HybridGraphBuilder(max_papers=2, max_semantic=1, client=MagicMock())
    assert builder.embedding_builder is not None
    builder.embedding_builder.materialize_graph_embeddings = MagicMock(
        side_effect=RuntimeError("simulated MPS STS failure")
    )

    with pytest.raises(
        EmbeddingInferenceError,
        match="Hybrid graph-similarity embedding failed.*simulated MPS STS failure",
    ):
        builder.prepare_graph_scoring(
            {"seed": _seed_paper("seed"), "peer": _paper("peer")}
        )


def test_hybrid_collection_fails_closed_on_semantic_enrichment_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid collection should fail when semantic enrichment cannot complete."""
    builder = HybridGraphBuilder(
        max_papers=5, max_semantic=2, semantic_source="arxiv-corpus", client=MagicMock()
    )

    seed = _paper("seed")
    seed.is_seed = True
    citation_papers = {"seed": seed, "c1": _paper("c1")}

    builder.citation_builder.collect_papers = MagicMock(return_value=citation_papers)
    assert builder.embedding_builder is not None
    monkeypatch.setattr(
        builder.embedding_builder, "resolve_cached_corpus_seed", lambda _seed_id: None
    )
    builder.embedding_builder.collect_papers = MagicMock(
        side_effect=RuntimeError("semantic backend unavailable")
    )

    with pytest.raises(
        RuntimeError, match="Semantic enrichment failed: semantic backend unavailable"
    ):
        builder.collect_papers("seed")


def test_hybrid_collection_preserves_semantic_scholar_outage_type() -> None:
    """Hybrid API callers should retain the service availability taxonomy."""
    from citemesh.services import SemanticScholarUnavailableError

    builder = HybridGraphBuilder(
        max_papers=5,
        max_semantic=2,
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    )
    seed = _seed_paper()
    builder.citation_builder.collect_papers = MagicMock(return_value={"seed": seed})
    assert builder.embedding_builder is not None
    builder.embedding_builder.resolve_cached_corpus_seed = MagicMock(return_value=None)
    builder.embedding_builder.collect_papers = MagicMock(
        side_effect=SemanticScholarUnavailableError("semantic service outage")
    )

    with pytest.raises(
        SemanticScholarUnavailableError,
        match="semantic service outage",
    ):
        builder.collect_papers("seed")


def test_hybrid_rerank_fails_when_seed_embedding_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid rerank should surface a batch-wide seed embedding failure."""
    builder = HybridGraphBuilder(max_papers=4, max_semantic=1, client=MagicMock())
    assert builder.embedding_builder is not None

    seed = _seed_paper("seed")
    candidate = _paper("c1")
    builder.embedding_builder.retrieval_embeddings = {
        candidate.paper_id: np.asarray([0.2, 0.1, 0.3], dtype=np.float32)
    }
    builder.embedding_builder.model_profile = MagicMock(
        format_query=lambda text, _metadata: text
    )
    builder.embedding_builder._encode_texts = MagicMock(
        side_effect=RuntimeError("temporary seed encode failure")
    )

    with pytest.raises(
        EmbeddingInferenceError,
        match="Hybrid seed embedding failed during semantic reranking",
    ):
        builder._rank_candidates(
            seed,
            {candidate.paper_id: candidate},
            {candidate.paper_id: {"semantic"}},
        )


@pytest.mark.parametrize(
    ("candidate_vector", "message"),
    [
        (np.asarray([np.nan, 0.0], dtype=np.float32), "non-finite embedding"),
        (np.asarray([0.0, 0.0], dtype=np.float32), "zero embedding"),
        (np.asarray([0.1, 0.2, 0.3], dtype=np.float32), "inconsistent embedding"),
    ],
)
def test_hybrid_rerank_rejects_invalid_candidate_embeddings(
    candidate_vector: np.ndarray,
    message: str,
) -> None:
    """Hybrid rerank should reject unusable vectors instead of substituting zero."""
    builder = HybridGraphBuilder(max_papers=4, max_semantic=1, client=MagicMock())
    assert builder.embedding_builder is not None

    seed = _seed_paper("seed")
    candidate = _paper("c1")
    builder.embedding_builder.retrieval_embeddings = {
        seed.paper_id: np.asarray([1.0, 0.0], dtype=np.float32),
        candidate.paper_id: candidate_vector,
    }

    with pytest.raises(EmbeddingInferenceError, match=message):
        builder._rank_candidates(
            seed,
            {candidate.paper_id: candidate},
            {candidate.paper_id: {"semantic"}},
        )


def test_hybrid_rerank_keeps_candidate_embedding_hydration_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid rerank should not persist citation-candidate embeddings into cache."""
    builder = HybridGraphBuilder(
        max_papers=4, max_semantic=1, semantic_source="arxiv-corpus", client=MagicMock()
    )
    assert builder.embedding_builder is not None

    seed = _seed_paper("seed")
    candidate = _paper("c1")
    builder.embedding_builder.retrieval_embeddings = {}
    builder.embedding_builder.model_profile = MagicMock(
        format_query=lambda text, _metadata: text,
        format_document=compose_title_abstract_text,
    )
    builder.embedding_builder._encode_texts = MagicMock(
        side_effect=[
            np.asarray([[0.4, 0.1, 0.2]], dtype=np.float32),
            np.asarray([[0.3, 0.2, 0.1]], dtype=np.float32),
        ]
    )
    builder.embedding_builder.embedding_cache = MagicMock()
    builder.embedding_builder.embedding_cache.get_embeddings = MagicMock(
        side_effect=AssertionError(
            "Hybrid rerank candidate hydration must not write into persistent cache."
        )
    )

    seed_embedding = builder._ensure_candidate_embeddings(
        seed, {candidate.paper_id: candidate}
    )

    np.testing.assert_allclose(
        seed_embedding, np.asarray([0.4, 0.1, 0.2], dtype=np.float32)
    )
    np.testing.assert_allclose(
        builder.embedding_builder.retrieval_embeddings[candidate.paper_id],
        np.asarray([0.3, 0.2, 0.1], dtype=np.float32),
    )
    builder.embedding_builder.embedding_cache.get_embeddings.assert_not_called()


def test_hybrid_thresholds_and_default_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid should enforce seed/non-seed threshold and semantic budget rules."""

    builder = HybridGraphBuilder(max_papers=3, max_semantic=1, client=MagicMock())
    seed = _paper("seed")
    seed.is_seed = True
    other = _paper("other")
    peer = _paper("peer")

    assert not builder.should_create_edge(seed, other, 0.2)
    assert builder.should_create_edge(seed, other, 0.41)
    assert not builder.should_create_edge(other, peer, 0.5)
    assert builder.should_create_edge(other, peer, 0.51)

    with pytest.raises(
        ValueError, match="max_semantic must be between 0 and max_papers - 1"
    ):
        HybridGraphBuilder(max_papers=3, max_semantic=3, client=MagicMock())

    small_builder = HybridGraphBuilder(max_papers=5, client=MagicMock())
    assert small_builder.max_semantic == 4


def test_hybrid_semantic_branch_collects_full_citation_candidate_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid semantic mode should fetch full reference+citation candidate pools."""
    builder = HybridGraphBuilder(
        max_papers=40,
        max_semantic=None,
        max_references=20,
        max_citations=20,
        client=MagicMock(),
    )

    assert builder.citation_builder.max_papers == 41
    assert builder.citation_builder.max_references == 20
    assert builder.citation_builder.max_citations == 20


def test_hybrid_propagates_refresh_reference_cache_to_citation_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid should forward refresh-reference policy to citation builder."""
    builder = HybridGraphBuilder(
        max_papers=4,
        max_semantic=0,
        refresh_reference_cache=True,
        client=MagicMock(),
    )

    assert builder.citation_builder.refresh_reference_cache is True


def test_hybrid_rerank_enforces_semantic_cap_and_overlap_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid rerank should cap semantic-only additions while preserving overlap papers."""
    builder = HybridGraphBuilder(
        max_papers=5, max_semantic=1, semantic_source="arxiv-corpus", client=MagicMock()
    )

    seed = _paper("seed")
    seed.is_seed = True
    citation_papers = {
        "seed": seed,
        "c1": _paper("c1"),
        "c2": _paper("c2"),
        "o1": _paper("o1"),
    }
    semantic_papers = {
        "seed": seed,
        "s1": _paper("s1"),
        "s2": _paper("s2"),
        "o1": _paper("o1"),
    }
    builder.citation_builder.collect_papers = MagicMock(return_value=citation_papers)
    assert builder.embedding_builder is not None
    monkeypatch.setattr(
        builder.embedding_builder, "resolve_cached_corpus_seed", lambda _seed_id: None
    )
    builder.embedding_builder.collect_papers = MagicMock(return_value=semantic_papers)
    monkeypatch.setattr(
        builder,
        "_rank_candidates",
        lambda *_args, **_kwargs: ["o1", "s1", "s2", "c2", "c1"],
    )

    papers = builder.collect_papers("seed")

    assert list(papers.keys()) == ["seed", "o1", "s1", "c2", "c1"]
    assert builder.paper_sources["o1"] == "both"
    assert builder.paper_sources["s1"] == "semantic"
    assert "s2" not in papers


def test_hybrid_collection_dedupes_semantic_seed_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid collection should collapse semantic seed aliases into the citation seed."""
    builder = HybridGraphBuilder(
        max_papers=5, max_semantic=2, semantic_source="arxiv-corpus", client=MagicMock()
    )

    seed = Paper(
        paper_id="ca997f1a733e53ad0fa29041246ff655243e8c1b",
        title="Polynomial Composition Activations: Unleashing the Dynamics of Large Language Models",
        year=2024,
        abstract=(
            "Transformers have found extensive applications across various domains due "
            "to the powerful fitting capabilities."
        ),
        authors=[Author(name="Zhijian Zhou"), Author(name="Ya Wang")],
        is_seed=True,
    )
    citation_papers = {"seed": seed, "c1": _paper("c1")}
    semantic_seed_alias = Paper(
        paper_id="arXiv:2411.03884v2",
        title=(
            "Polynomial Composition Activations: Unleashing the Dynamics of Large\n"
            "  Language Models"
        ),
        year=2025,
        abstract=seed.abstract,
        authors=[Author(name="Zhijian Zhou"), Author(name="Yitao Zeng")],
    )
    semantic_papers = {
        semantic_seed_alias.paper_id: semantic_seed_alias,
        "s1": _paper("s1"),
    }
    builder.citation_builder.collect_papers = MagicMock(return_value=citation_papers)
    assert builder.embedding_builder is not None
    monkeypatch.setattr(
        builder.embedding_builder, "resolve_cached_corpus_seed", lambda _seed_id: None
    )
    builder.embedding_builder.collect_papers = MagicMock(return_value=semantic_papers)
    monkeypatch.setattr(
        builder,
        "_rank_candidates",
        lambda *_args, **_kwargs: ["c1", "s1"],
    )

    papers = builder.collect_papers("arXiv:2411.03884")

    assert set(papers) == {seed.paper_id, "c1", "s1"}
    assert "arXiv:2411.03884v2" not in papers
    assert papers[seed.paper_id].year == 2024
    assert builder.paper_sources[seed.paper_id] == "citation"


def test_hybrid_build_graph_skips_pruning_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid build should return parent graph unchanged when pruning disabled."""
    monkeypatch.setattr(HYBRID_CONFIG, "max_edges_per_node", 0)

    builder = HybridGraphBuilder(max_papers=3, max_semantic=0, client=MagicMock())
    builder.paper_sources = {"seed": "citation", "a": "semantic"}
    builder.seed_relations = {
        "seed": "seed",
        "a": "semantic_only",
        "pruned": "citation",
    }
    graph = nx.Graph()
    graph.add_node("seed", is_seed=True)
    graph.add_node("a", is_seed=False)
    monkeypatch.setattr(
        hybrid_strategy.GraphBuilderStrategy,
        "build_graph",
        lambda self, seed_id, **kwargs: (graph, "seed"),
    )

    out_graph, out_seed = builder.build_graph("seed")
    assert out_graph is graph
    assert out_seed == "seed"
    assert out_graph.graph["strategy"] == "hybrid"
    assert out_graph.graph["paper_sources"] == {"a": "semantic", "seed": "citation"}
    assert out_graph.graph["seed_relations"] == {"a": "semantic_only", "seed": "seed"}


def test_hybrid_build_graph_logs_post_cap_edge_count(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Hybrid pruning should log original and filtered edge counts."""
    monkeypatch.setattr(HYBRID_CONFIG, "max_edges_per_node", 1)

    builder = HybridGraphBuilder(max_papers=4, max_semantic=0, client=MagicMock())
    graph = nx.Graph()
    graph.add_node("seed", is_seed=True)
    graph.add_nodes_from(["a", "b", "c"])
    graph.add_edge("seed", "a", weight=0.9)
    graph.add_edge("seed", "b", weight=0.8)
    graph.add_edge("seed", "c", weight=0.7)
    graph.add_edge("a", "b", weight=0.95)
    graph.add_edge("a", "c", weight=0.85)
    graph.add_edge("b", "c", weight=0.75)
    monkeypatch.setattr(
        hybrid_strategy.GraphBuilderStrategy,
        "build_graph",
        lambda self, seed_id, **kwargs: (graph, "seed"),
    )

    with caplog.at_level(logging.INFO):
        out_graph, out_seed = builder.build_graph("seed")

    assert out_seed == "seed"
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
def test_max_papers_is_total_node_cap_including_seed(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
) -> None:
    """All strategies should include seed within ``max_papers`` total cap."""
    if strategy == "recommendation":
        client = MagicMock()
        client.get_paper.return_value = _seed_paper()
        client.get_recommended_papers.return_value = [
            _paper(f"r{i}") for i in range(1, 6)
        ]
        papers = RecommendationGraphBuilder(max_papers=3, client=client).collect_papers(
            "seed"
        )
        assert len(papers) == 3
        assert "seed" in papers
        return

    if strategy == "citation":
        client = MagicMock()
        client.get_paper.return_value = _seed_paper()
        client.get_paper_references.return_value = [
            _paper(f"r{i}") for i in range(1, 6)
        ]
        client.get_paper_citations.return_value = [_paper(f"c{i}") for i in range(1, 6)]
        papers = CitationGraphBuilder(
            max_papers=4,
            max_citations=10,
            max_references=10,
            client=client,
        ).collect_papers("seed")
        assert len(papers) == 4
        assert "seed" in papers
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
        builder.client.get_paper.return_value = _seed_paper()
        builder._load_model = lambda: None
        builder._resolve_model_fingerprint = lambda: "test-fingerprint"
        builder._update_citation_counts = lambda _: None
        builder._select_candidates = lambda _seed_embedding, *, use_streaming: [
            (
                "c1",
                {"title": "Paper c1", "abstract": "A", "authors": [], "categories": []},
                np.array([1.0, 0.0], dtype=np.float32),
            ),
            (
                "c2",
                {"title": "Paper c2", "abstract": "B", "authors": [], "categories": []},
                np.array([0.0, 1.0], dtype=np.float32),
            ),
            (
                "c3",
                {"title": "Paper c3", "abstract": "C", "authors": [], "categories": []},
                np.array([0.5, 0.5], dtype=np.float32),
            ),
        ]
        builder._encode_texts = lambda texts, **kwargs: np.array(
            [[1.0, 0.0] for _ in texts], dtype=np.float32
        )
        papers = builder.collect_papers("seed")
        assert len(papers) == 3
        assert "seed" in papers
        return

    class FakeCitationBuilder:
        def __init__(self, max_papers: int, *_args: object, **_kwargs: object) -> None:
            """Initialize the fake citation builder.

            :param int max_papers: Maximum papers returned by the fake collector.
            :param object _args: Positional constructor arguments not used by the fake.
            :param object _kwargs: Keyword constructor arguments not used by the fake.
            """
            self.max_papers = max_papers
            self.candidate_source_results: tuple[object, ...] = ()

        def collect_papers(self, seed_id: str, **_: object) -> dict[str, Paper]:
            del seed_id
            papers = {"seed": _seed_paper(), "c1": _paper("c1"), "c2": _paper("c2")}
            return dict(list(papers.items())[: self.max_papers])

    class FakeEmbeddingBuilder:
        def __init__(self, max_papers: int, *_args: object, **_kwargs: object) -> None:
            self.max_papers = max_papers
            self.retrieval_embeddings: dict[str, np.ndarray] = {}
            self.embeddings: dict[str, np.ndarray] = {}

        def collect_papers(self, seed_id: str, **_: object) -> dict[str, Paper]:
            del seed_id
            self.retrieval_embeddings = {
                paper_id: np.asarray([1.0, 0.0], dtype=np.float32)
                for paper_id in {"seed", "c1", "c2", "s1", "s2"}
            }
            return {"seed": _seed_paper(), "s1": _paper("s1"), "s2": _paper("s2")}

        def resolve_cached_corpus_seed(self, seed_id: str) -> None:
            """Return no pre-resolved seed for this capacity-only fake.

            :param str seed_id: Seed identifier unused by the fake.
            :return None: Forces the citation fake to own seed resolution.
            """
            del seed_id
            return None

    monkeypatch.setattr(hybrid_strategy, "CitationGraphBuilder", FakeCitationBuilder)
    monkeypatch.setattr(hybrid_strategy, "EmbeddingGraphBuilder", FakeEmbeddingBuilder)

    papers = HybridGraphBuilder(
        max_papers=4,
        max_semantic=1,
        semantic_source="arxiv-corpus",
        client=MagicMock(),
    ).collect_papers("seed")
    assert len(papers) == 4
    assert "seed" in papers


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
def test_strategy_caps_reserve_existing_seed_neighbors(
    builder_type: type[GraphBuilderStrategy],
    eligible_neighbors: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stronger candidate clusters must not consume the seed's available edges.

    :param type[GraphBuilderStrategy] builder_type: Capped strategy constructor.
    :param int eligible_neighbors: Seed neighbors meeting the edge threshold.
    :param pytest.MonkeyPatch monkeypatch: Controlled collection and similarity scores.
    :return None: Asserts the seed retains its strongest existing edges within every cap.
    """
    papers = {"seed": _seed_paper(), **{f"p{i}": _paper(f"p{i}") for i in range(8)}}
    builder = builder_type(max_papers=9, client=MagicMock())
    monkeypatch.setattr(builder, "collect_papers", lambda _seed: papers)
    monkeypatch.setattr(builder, "prepare_graph_scoring", lambda _papers: None)
    if isinstance(builder, EmbeddingGraphBuilder):
        cap = builder.top_k
    elif isinstance(builder, HybridGraphBuilder):
        cap = HYBRID_CONFIG.max_edges_per_node
    else:
        cap = 3
    base_score = 0.60 if isinstance(builder, HybridGraphBuilder) else 0.30

    def score(left: Paper, right: Paper) -> float:
        """Score the candidate cluster above all eligible seed edges.

        :param Paper left: First paper.
        :param Paper right: Second paper.
        :return float: Eligible edge weight or zero for an unsupported seed pair.
        """
        if left.is_seed or right.is_seed:
            neighbor = right if left.is_seed else left
            index = int(neighbor.paper_id[1:])
            return base_score + index * 0.001 if index < eligible_neighbors else 0.0
        return base_score + 0.05

    monkeypatch.setattr(builder, "compute_similarity", score)
    graph, seed_id = builder.build_graph("seed")

    expected_seed_neighbors = {
        f"p{i}" for i in range(max(0, eligible_neighbors - cap), eligible_neighbors)
    }
    assert set(graph.neighbors(seed_id)) == expected_seed_neighbors
    assert graph.degree(seed_id) == min(cap, eligible_neighbors)
    assert all(degree <= cap for _, degree in graph.degree())
    assert all(score(papers[left], papers[right]) > 0 for left, right in graph.edges())
    papers = dict(reversed(list(papers.items())))
    reordered, _ = builder.build_graph("seed")
    assert {frozenset(edge) for edge in graph.edges()} == {
        frozenset(edge) for edge in reordered.edges()
    }


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
    seed = _seed_paper()
    seed.citation_count = 7
    client.get_paper.return_value = seed
    client.get_papers.side_effect = SemanticScholarRequestError("HTTP 403")
    builder = EmbeddingGraphBuilder(
        max_papers=2, semantic_source="arxiv-corpus", client=client
    )
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(builder, "_load_model", lambda: None)
    monkeypatch.setattr(
        builder, "_resolve_model_fingerprint", lambda: "test-fingerprint"
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

    papers = builder.collect_papers("seed")

    assert set(papers) == {"seed", "arxiv:2501.00001"}
    assert papers["seed"].citation_count == 7
    assert papers["arxiv:2501.00001"].citation_count == 0
    assert set(builder.retrieval_embeddings) == set(papers)
    assert "Citation-count enrichment was rejected" in caplog.text
    assert "HTTP 403" in caplog.text
    client.get_papers.assert_called_once_with(["arxiv:2501.00001"])
    client.get_paper.assert_called_once_with("seed", raise_on_unavailable=True)
    client.get_papers.side_effect = ValueError("invalid local payload")
    with pytest.raises(ValueError, match="invalid local payload"):
        builder._update_citation_counts(papers)


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
            lambda: HybridGraphBuilder(
                max_papers=4, max_semantic=0, client=MagicMock()
            ),
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
    graph, _ = builder.build_graph("seed")

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
            seed_id="seed",
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

    with caplog.at_level(logging.INFO):
        graph, _ = builder.build_graph("seed")

    assert "Graph constructed: 5 nodes, 10 edges" in caplog.text
    assert f"Graph complete: 5 nodes, {graph.number_of_edges()} edges" in caplog.text
    assert "Graph complete: 5 nodes, 10 edges" not in caplog.text


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
    [(2020, 2020, 1.0), (2010, 2020, 0.1)],
)
def test_temporal_similarity_extremes(
    paper1_year: int,
    paper2_year: int,
    expected: float,
) -> None:
    """Temporal similarity should be stable for known year deltas."""
    paper1 = Paper(paper_id="p1", title="Test 1", year=paper1_year)
    paper2 = Paper(paper_id="p2", title="Test 2", year=paper2_year)
    assert GraphBuilderStrategy.temporal_similarity(paper1, paper2) == expected


def test_temporal_similarity_unknown_year_defaults() -> None:
    """Unknown years should return neutral fallback for temporal similarity."""
    paper1 = Paper(paper_id="p1", title="Test 1", year=None)
    paper2 = Paper(paper_id="p2", title="Test 2", year=2020)
    assert GraphBuilderStrategy.temporal_similarity(paper1, paper2) == 0.5


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


def test_bibliographic_coupling_delegates_to_paper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bibliographic coupling should delegate to ``Paper.reference_overlap``."""
    paper1 = Paper(paper_id="p1", title="Test 1", year=2020)
    paper2 = Paper(paper_id="p2", title="Test 2", year=2020)

    monkeypatch.setattr(Paper, "reference_overlap", lambda self, other: 0.42)
    assert GraphBuilderStrategy.bibliographic_coupling(paper1, paper2) == 0.42


def test_candidate_pool_dedupes_equivalent_papers() -> None:
    """CandidatePool should merge papers with matching identity aliases."""
    from citemesh.strategies.candidates import CandidatePool

    seed = _seed_paper()
    pool = CandidatePool(seed=seed)
    first = Paper(
        paper_id="b" * 40,
        title="Same Paper",
        year=2021,
        abstract="An abstract",
        citation_count=5,
        arxiv_id="2101.00001",
    )
    duplicate = Paper(
        paper_id="arxiv:2101.00001",
        title="Same  Paper",
        year=2021,
        abstract="",
        citation_count=0,
    )
    pool.add(first, source="reference", relation="referenced_by_seed")
    pool.add(duplicate, source="recommendation", relation="semantic_only")

    alias_only_duplicate = Paper(
        paper_id="arxiv:2101.00001",
        title="Payload Without Matching Metadata",
        year=2022,
        abstract="alternate payload",
    )
    pool.add(alias_only_duplicate, source="citation", relation="cites_seed")

    assert list(pool.papers) == [first.paper_id]
    assert pool.sources[first.paper_id] == {
        "reference",
        "recommendation",
        "citation",
    }
    assert pool.seed_relations[first.paper_id] == "overlap"

    seed_duplicate = Paper(
        paper_id=seed.paper_id,
        title=seed.title,
        year=seed.year,
        abstract="richer seed abstract",
    )
    pool.add(seed_duplicate, source="citation", relation="cites_seed")
    assert set(pool.papers) == {first.paper_id}


@pytest.mark.parametrize(
    ("identifier_field", "identifier_value"),
    [("doi", "10.1000/shared"), ("arxiv_id", "2508.12345")],
)
@pytest.mark.parametrize("is_seed", [True, False])
def test_identity_external_id_agreement_overrides_s2_record_disagreement(
    identifier_field: str,
    identifier_value: str,
    is_seed: bool,
) -> None:
    """External-ID matches must retain alternate S2 IDs for later sparse records.

    :param str identifier_field: External identifier shared by the records.
    :param str identifier_value: Shared DOI or arXiv identifier.
    :param bool is_seed: Whether the surviving record is the seed or a candidate.
    :return None: Validates metadata merging and continued identity reconciliation.
    """
    canonical = Paper(
        paper_id="1" * 40,
        title="Canonical Paper",
        year=2025,
        abstract="",
        is_seed=is_seed,
    )
    setattr(canonical, identifier_field, identifier_value)
    duplicate = Paper(
        paper_id="2" * 40,
        title="Duplicate Record",
        year=2025,
        abstract="hydrated abstract",
    )
    setattr(duplicate, identifier_field, identifier_value)
    pool = CandidatePool(seed=canonical if is_seed else _seed_paper())
    if not is_seed:
        pool.add(canonical, source="reference", relation="referenced_by_seed")

    pool.add(duplicate, source="recommendation", relation="semantic_only")
    pool.add(
        Paper(
            paper_id=duplicate.paper_id,
            title="Unknown",
            year=None,
            references=["related-paper"],
        ),
        source="citation",
        relation="cites_seed",
    )

    assert pool.papers == ({} if is_seed else {canonical.paper_id: canonical})
    assert canonical.abstract == "hydrated abstract"
    assert canonical.references == ["related-paper"]
    assert pool._aliases[f"id:{duplicate.paper_id}"] == canonical.paper_id
    assert pool._aliases.evidence(canonical.paper_id).strong_ids["s2"] == frozenset(
        {canonical.paper_id, duplicate.paper_id}
    )


def test_candidate_pool_collapses_identifier_bridge_classes() -> None:
    """A record bridging arXiv and DOI aliases should merge both prior classes."""
    from citemesh.strategies.candidates import CandidatePool

    pool = CandidatePool(seed=_seed_paper())
    arxiv_record, doi_record, bridge = _identity_bridge_records()

    pool.add(arxiv_record, source="reference", relation="referenced_by_seed")
    pool.add(doi_record, source="citation", relation="cites_seed")
    pool.add(bridge, source="recommendation", relation="semantic_only")

    assert list(pool.papers) == [arxiv_record.paper_id]
    assert pool.papers[arxiv_record.paper_id].doi == "10.1000/bridge"
    assert pool.sources[arxiv_record.paper_id] == {
        "reference",
        "citation",
        "recommendation",
    }
    assert pool.seed_relations[arxiv_record.paper_id] == "overlap"
    assert pool._aliases["id:10.1000/bridge"] == arxiv_record.paper_id


@pytest.mark.parametrize("opaque_case_distinction", [False, True])
def test_identity_reconciliation_rejects_conflicting_strong_ids(
    opaque_case_distinction: bool,
) -> None:
    """Matching metadata must not override contradictory source ID or DOI evidence.

    :param bool opaque_case_distinction: Use case-sensitive opaque corpus identifiers.
    :return None: Checks distinct identity classes retain independent provenance.
    """
    authors = [Author(name="Ada Lovelace")]
    first = Paper(
        paper_id="LocalA" if opaque_case_distinction else "1" * 40,
        title="Shared Scientific Title",
        year=2024,
        authors=authors,
        abstract="Shared abstract",
        doi="" if opaque_case_distinction else "10.1000/first",
    )
    second = Paper(
        paper_id="locala" if opaque_case_distinction else "2" * 40,
        title="Shared Scientific Title",
        year=2024,
        authors=authors,
        abstract="Shared abstract",
        doi="" if opaque_case_distinction else "10.1000/second",
    )
    seed = Paper(
        paper_id="3" * 40,
        title="Shared Scientific Title",
        year=2024,
        authors=authors,
        abstract="Shared abstract",
        is_seed=True,
    )
    pool = CandidatePool(seed=seed)

    pool.add(first, source="reference", relation="referenced_by_seed")
    pool.add(second, source="recommendation", relation="semantic_only")

    assert set(pool.papers) == {first.paper_id, second.paper_id}
    assert pool.sources[first.paper_id] == {"reference"}
    assert pool.sources[second.paper_id] == {"recommendation"}

    pool.add(
        Paper(
            paper_id="4" * 40,
            title=seed.title,
            year=seed.year,
            authors=authors,
            abstract=seed.abstract,
        ),
        source="citation",
        relation="cites_seed",
    )
    assert "4" * 40 in pool.papers


@pytest.mark.parametrize("title", ["Unknown", "Untitled", "None", "N/A"])
def test_identity_placeholder_titles_never_create_weak_aliases(title: str) -> None:
    """Placeholder titles must not become global metadata identity keys."""
    paper = Paper(
        paper_id="placeholder",
        title=title,
        year=2024,
        authors=[Author(name="Unknown Author")],
    )
    assert not any(alias.startswith("meta:") for alias in paper_identity_aliases(paper))


def test_identity_same_primary_quarantines_conflicting_secondary_ids() -> None:
    """Same-primary refreshes must not assign a contradictory DOI to that class."""
    primary_a = "a" * 40
    primary_b = "b" * 40
    pool = CandidatePool(seed=_seed_paper())
    canonical = Paper(
        paper_id=primary_a,
        title="Canonical",
        year=2024,
        doi="10.1000/a",
    )
    conflicting_refresh = Paper(
        paper_id=primary_a,
        title="Canonical refreshed",
        year=2024,
        doi="10.1000/b",
    )
    actual_doi_owner = Paper(
        paper_id=primary_b,
        title="Different paper",
        year=2024,
        doi="10.1000/b",
    )

    pool.add(canonical, source="reference", relation="referenced_by_seed")
    pool.add(conflicting_refresh, source="citation", relation="cites_seed")
    pool.add(actual_doi_owner, source="recommendation", relation="semantic_only")

    assert set(pool.papers) == {primary_a, primary_b}
    assert pool.papers[primary_a].doi == "10.1000/a"
    assert pool._aliases["id:10.1000/b"] == primary_b

    # Direct registry callers can refresh an unmerged payload; the conflict
    # branch must also quarantine its contradictory aliases at this boundary.
    registry = IdentityRegistry()
    register_aliases(registry, primary_a, canonical)
    register_aliases(registry, primary_a, conflicting_refresh)
    assert registry["id:10.1000/a"] == primary_a
    assert registry.get("id:10.1000/b") is None
    assert registry.evidence(primary_a).strong_ids["doi"] == frozenset({"10.1000/a"})


def test_identity_transitive_weak_bridge_cannot_collapse_conflicting_classes() -> None:
    """A metadata bridge must not transitively unite contradictory strong IDs."""
    authors = [Author(name="Grace Hopper")]
    pool = CandidatePool(seed=_seed_paper())
    left = Paper(
        paper_id="c" * 40,
        title="Shared",
        year=2023,
        authors=authors,
        doi="10.1000/left",
    )
    middle = Paper(
        paper_id="10.1000/right",
        title="Shared",
        year=2023,
        authors=authors,
        doi="10.1000/right",
    )
    right = Paper(
        paper_id="d" * 40,
        title="Different",
        year=2023,
        authors=authors,
        doi="10.1000/right",
    )

    pool.add(left, source="reference", relation="referenced_by_seed")
    pool.add(middle, source="citation", relation="cites_seed")
    pool.add(right, source="recommendation", relation="semantic_only")

    assert set(pool.papers) == {left.paper_id, middle.paper_id}
    assert pool.sources[left.paper_id] == {"reference"}
    assert pool.sources[middle.paper_id] == {"citation", "recommendation"}


def test_hybrid_identity_conflicts_do_not_manufacture_overlap_provenance() -> None:
    """Hybrid false twins should remain separate source classes without a bonus."""
    builder = HybridGraphBuilder(max_papers=4, max_semantic=0, client=MagicMock())
    seed = _seed_paper()
    aliases = IdentityRegistry()
    register_aliases(aliases, seed.paper_id, seed)
    candidates: dict[str, Paper] = {}
    sources: dict[str, set[str]] = {}
    authors = [Author(name="Katherine Johnson")]
    citation_twin = Paper(
        paper_id="e" * 40,
        title="Same title",
        year=2022,
        authors=authors,
    )
    semantic_twin = Paper(
        paper_id="f" * 40,
        title="Same title",
        year=2022,
        authors=authors,
    )
    builder._ingest_candidate(
        aliases,
        seed,
        candidates,
        sources,
        citation_twin,
        source="citation",
        relation="cites_seed",
    )
    builder._ingest_candidate(
        aliases,
        seed,
        candidates,
        sources,
        semantic_twin,
        source="semantic",
        relation="semantic_only",
    )

    assert set(candidates) == {citation_twin.paper_id, semantic_twin.paper_id}
    assert sources[citation_twin.paper_id] == {"citation"}
    assert sources[semantic_twin.paper_id] == {"semantic"}


def test_recommendation_collect_collapses_seed_and_candidate_identifier_aliases() -> (
    None
):
    """Recommendation ingestion should not retain arXiv aliases as duplicate nodes."""
    seed = _seed_paper("s2-seed")
    seed.arxiv_id = "2508.12345"
    candidate = _paper("s2-candidate")
    candidate.arxiv_id = "2508.12346"
    candidate_alias = _paper("arxiv:2508.12346")
    seed_alias = _paper("arxiv:2508.12345")
    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_recommended_papers.return_value = [
        seed_alias,
        candidate,
        candidate_alias,
    ]

    papers = RecommendationGraphBuilder(
        max_papers=4, fetch_references=False, client=client
    ).collect_papers("s2-seed")

    assert set(papers) == {"s2-seed", "s2-candidate"}
    assert papers["s2-seed"].is_seed is True


def test_recommendation_collect_reconciles_sparse_identifier_bridge() -> None:
    """Sparse known records should still bridge existing identity classes."""
    seed = _seed_paper()
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
    client.get_paper.return_value = seed
    client.get_recommended_papers.return_value = [
        arxiv_record,
        doi_record,
        sparse_bridge,
    ]

    papers = RecommendationGraphBuilder(
        max_papers=4, fetch_references=False, client=client
    ).collect_papers("seed")

    assert set(papers) == {"seed", arxiv_record.paper_id}
    assert papers[arxiv_record.paper_id].doi == "10.1000/bridge"


def test_recommendation_collect_reconciles_bridge_after_capacity() -> None:
    """Capacity filtering should not hide later identity bridge records."""
    seed = _seed_paper()
    arxiv_record, doi_record, bridge = _identity_bridge_records()
    unrelated = _paper("unrelated")
    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_recommended_papers.return_value = [
        arxiv_record,
        doi_record,
        unrelated,
        bridge,
    ]

    papers = RecommendationGraphBuilder(
        max_papers=3, fetch_references=False, client=client
    ).collect_papers("seed")

    assert set(papers) == {"seed", arxiv_record.paper_id}
    assert papers[arxiv_record.paper_id].doi == "10.1000/bridge"


def test_hybrid_collection_collapses_identifier_bridge_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid ingestion should preserve source and relation data across bridge merges."""
    builder = HybridGraphBuilder(
        max_papers=3, max_semantic=1, semantic_source="arxiv-corpus", client=MagicMock()
    )
    seed = _seed_paper()
    arxiv_record, doi_record, bridge = _identity_bridge_records()
    builder.citation_builder.collect_papers = MagicMock(
        return_value={
            seed.paper_id: seed,
            arxiv_record.paper_id: arxiv_record,
            doi_record.paper_id: doi_record,
        }
    )
    builder.citation_builder.seed_relations = {
        seed.paper_id: "seed",
        arxiv_record.paper_id: "referenced_by_seed",
        doi_record.paper_id: "cites_seed",
    }
    assert builder.embedding_builder is not None
    monkeypatch.setattr(
        builder.embedding_builder, "resolve_cached_corpus_seed", lambda _seed_id: None
    )
    builder.embedding_builder.collect_papers = MagicMock(
        return_value={bridge.paper_id: bridge}
    )
    monkeypatch.setattr(
        builder, "_rank_candidates", lambda *_args: [arxiv_record.paper_id]
    )

    papers = builder.collect_papers("seed")

    assert set(papers) == {"seed", arxiv_record.paper_id}
    assert papers[arxiv_record.paper_id].doi == "10.1000/bridge"
    assert builder.paper_sources[arxiv_record.paper_id] == "both"
    assert builder.seed_relations[arxiv_record.paper_id] == "overlap"


def test_merge_paper_metadata_preserves_fields_and_unions_references() -> None:
    """Canonical metadata merging should repair gaps without discarding rich fields."""
    from citemesh.strategies.candidates import merge_paper_metadata

    preferred = Paper(
        paper_id="canonical",
        title="Unknown",
        year=None,
        references=[" existing-ref ", "", "duplicate-ref"],
        is_seed=False,
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
        is_seed=True,
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
    assert merged.is_seed is True

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


def test_citation_seed_relations_use_shared_precedence_rules() -> None:
    """Citation relation updates should retain the shared seed-label semantics."""
    builder = CitationGraphBuilder(client=MagicMock())
    builder.seed_relations = {"seed": "seed", "semantic": "semantic_only"}

    builder._record_seed_relations(["seed", "semantic", "shared"], "cites_seed")
    builder._record_seed_relations(["shared"], "referenced_by_seed")

    assert builder.seed_relations == {
        "seed": "seed",
        "semantic": "cites_seed",
        "shared": "overlap",
    }


def test_recommendation_duplicates_merge_without_replacing_richer_record() -> None:
    """Recommendation duplicates should preserve the initial canonical object."""
    seed = _seed_paper()
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
    client.get_paper.return_value = seed
    client.get_recommended_papers.return_value = [existing, incoming]

    papers = RecommendationGraphBuilder(
        max_papers=3,
        fetch_references=False,
        client=client,
    ).collect_papers("seed")

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
    client.get_paper.return_value = _seed_paper()
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

    papers = builder.collect_papers("seed")

    assert "seed" in papers
    assert len(papers) == 4
    assert client.get_recommended_papers.called
    client.get_papers.assert_not_called()
    client.get_paper.assert_called_once_with("seed", raise_on_unavailable=True)
    non_seed = [paper_id for paper_id in papers if paper_id != "seed"]
    for paper_id in non_seed:
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
    cached = second.embed_papers({paper_id: papers[paper_id] for paper_id in non_seed})
    assert sorted(cached) == sorted(non_seed)
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
    cache.hydration_operation_lock.assert_called_once_with()
    assert lock_held is False


def test_hybrid_candidate_mode_uses_recommendations_not_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid candidate mode should source semantics from S2 recommendations."""
    from tests._helpers import SeededRandomEncodeModel

    client = MagicMock()
    client.get_recommended_papers.return_value = [_paper("rec1"), _paper("rec2")]

    builder = HybridGraphBuilder(
        max_papers=6,
        max_semantic=2,
        candidate_pool_size=1,
        client=client,
    )
    assert builder.semantic_source == "candidates"
    assert builder.embedding_builder is not None
    assert builder.embedding_builder.truncate_dim == 512

    citation_papers = {
        "seed": _seed_paper(),
        "c1": _paper("c1"),
        "c2": _paper("c2"),
    }
    monkeypatch.setattr(
        builder.citation_builder,
        "collect_papers",
        lambda _seed_id, **_kwargs: dict(citation_papers),
    )

    def _raise_hydration(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("corpus hydration must not run in candidate mode")

    monkeypatch.setattr(
        builder.embedding_builder, "_ensure_cache_hydrated", _raise_hydration
    )
    monkeypatch.setattr(
        builder.embedding_builder,
        "collect_papers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("corpus collect_papers must not run in candidate mode")
        ),
    )
    monkeypatch.setattr(builder.embedding_builder, "_load_model", lambda: None)
    monkeypatch.setattr(
        builder.embedding_builder,
        "_resolve_model_fingerprint",
        lambda: "test::immutable-artifact",
    )
    builder.embedding_builder.model = SeededRandomEncodeModel(embedding_dim=4)

    papers = builder.collect_papers("seed")

    assert "seed" in papers
    client.get_recommended_papers.assert_called_once_with(
        "seed",
        limit=1,
        raise_on_unavailable=True,
    )
    assert builder.candidate_source_status == {"recommendations": "complete"}
    semantic_added = [
        paper_id
        for paper_id, source in builder.paper_sources.items()
        if source == "semantic"
    ]
    assert semantic_added
    for paper_id in semantic_added:
        assert builder.seed_relations[paper_id] == "semantic_only"
    # Candidate vectors flow through the persistent cache namespace.
    assert "mode=candidates" in builder.embedding_builder.embedding_cache.model_name


@pytest.mark.parametrize(
    "sources",
    [
        ("citation", "citation"),
        ("semantic", "semantic"),
        ("citation", "semantic"),
        ("both", "both"),
    ],
)
@pytest.mark.parametrize("cosine", [0.0, 0.48, 0.71])
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


def test_hybrid_without_embeddings_keeps_citation_topical_scoring() -> None:
    """Disabling embeddings retains lexical evidence and citation edge eligibility.

    :return None: Verifies a complete build and a bibliographic-only edge.
    """
    seed = Paper("seed", "Quantum field theory", 2024, citation_count=100)
    related = Paper("related", "Quantum field interactions", 2024, citation_count=100)
    unrelated = Paper("unrelated", "Medieval pottery", 2024, citation_count=100)
    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = [related, unrelated]
    builder = HybridGraphBuilder(
        max_papers=3,
        max_semantic=0,
        max_references=2,
        max_citations=0,
        fetch_references=False,
        client=client,
    )
    graph, _ = builder.build_graph("seed")
    assert graph.has_edge("seed", "related")
    assert graph.degree("unrelated") == 0
    assert builder.compute_similarity(seed, related) == pytest.approx(
        builder.citation_builder.compute_similarity(seed, related)
    )
    builder.citation_builder.fetch_references = True
    seed.references = unrelated.references = ["shared"]
    unrelated.year = 2000
    seed.is_seed = False
    score = builder.compute_similarity(seed, unrelated)
    assert 0.2 < score < 0.5
    assert builder.should_create_edge(seed, unrelated, score)


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
    seed = Paper("seed", "To be or not to be", 2024, references=["shared"])
    peer = Paper("peer", "What is it", 2024, references=["shared"])
    client = MagicMock()
    client.get_paper.return_value = seed
    client.get_paper_references.return_value = [peer]
    builder = CitationGraphBuilder(
        max_papers=2, max_references=1, max_citations=0, client=client
    )
    graph, _ = builder.build_graph("seed")
    assert graph.has_edge("seed", "peer")


@pytest.mark.parametrize("cosine", [0.0, -0.1, 0.48, 0.71])
def test_embedding_graph_requires_calibrated_semantic_evidence(
    monkeypatch: pytest.MonkeyPatch, cosine: float
) -> None:
    """Metadata bonuses cannot connect semantically unsupported paper pairs.

    :param pytest.MonkeyPatch monkeypatch: Fixture for replacing acquisition and inference.
    :param float cosine: Unsupported candidate's cosine against the seed.
    :return None: Verifies the supported edge survives and the unrelated paper stays isolated.
    """
    papers = {
        paper_id: Paper(
            paper_id,
            title,
            2024,
            authors=[Author(name="Shared Author")],
            categories=["Computer Science"],
            is_seed=paper_id == "seed",
        )
        for paper_id, title in (
            ("seed", "Quantum field theory"),
            ("related", "Quantum field interactions"),
            ("unrelated", "Medieval pottery"),
        )
    }
    builder = EmbeddingGraphBuilder(max_papers=3, client=MagicMock())
    monkeypatch.setattr(builder, "collect_papers", MagicMock(return_value=papers))
    monkeypatch.setattr(builder, "prepare_graph_scoring", MagicMock())
    builder.embeddings = {
        "seed": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "related": np.array([0.8, 0.6, 0.0], dtype=np.float32),
        "unrelated": np.array(
            [cosine, 0.0, np.sqrt(1.0 - cosine**2)], dtype=np.float32
        ),
    }
    graph, _ = builder.build_graph("seed")
    assert graph.has_edge("seed", "related")
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
