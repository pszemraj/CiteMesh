"""Tests for Semantic Scholar client behavior and reference cache handling."""

from __future__ import annotations

import builtins
import importlib
import json
import logging
import os
import random
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs

import pytest
import requests
from semanticscholar.Reference import Reference

import citemesh.services as services_module
from citemesh.core import API_CONFIG, Author, Paper
from citemesh.core.paper_ids import paper_identifier_aliases
from citemesh.services import semantic_scholar as s2
from citemesh.services import semantic_scholar as semantic_module
from citemesh.services.semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarRequestError,
    SemanticScholarUnavailableError,
    get_client,
    normalize_paper_id,
    reset_client,
)
from tests._helpers import get_paper_id_normalization_cases


def _assert_module_reload_is_lazy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module: Any,
    blocked_prefixes: tuple[str, ...],
    expected_exports: set[str],
) -> None:
    """Assert that reloading a lazy-export package avoids importing blocked modules."""
    original_import = builtins.__import__

    def _guarded_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: object = (),
        level: int = 0,
    ) -> Any:
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in blocked_prefixes
        ):
            raise AssertionError(f"unexpected eager import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    reloaded = importlib.reload(module)
    assert set(reloaded.__all__) == expected_exports


def test_service_and_strategy_package_exports() -> None:
    """Package exports should resolve directly from their defining modules."""
    from citemesh import EmbeddingGraphBuilder as TopLevelEmbeddingGraphBuilder
    from citemesh.strategies import EmbeddingGraphBuilder

    assert services_module.get_client is get_client
    assert services_module.reset_client is reset_client
    assert services_module.SemanticScholarClient is SemanticScholarClient
    assert TopLevelEmbeddingGraphBuilder is EmbeddingGraphBuilder
    assert EmbeddingGraphBuilder.__module__ == "citemesh.strategies.embedding.builder"


def test_candidate_operation_scope_is_thread_local() -> None:
    """A shared client must keep concurrent discovery failures isolated.

    :return None: Verifies each thread retains only its own scoped outage.
    """
    failures: dict[str, SemanticScholarUnavailableError] = {}
    barrier = threading.Barrier(2)
    with SemanticScholarClient(timeout=1) as client:

        def record_failure(label: str) -> None:
            """Record and read one thread-local candidate-operation failure.

            :param str label: Identifier for this worker's synthetic failure.
            :return None: Stores the failure observed within this worker's scope.
            """
            failure = SemanticScholarUnavailableError(label)
            with client.candidate_operation_scope():
                client._record_candidate_operation_failure(
                    s2.errors._FailureDomain.REFERENCES, failure
                )
                barrier.wait()
                observed = client._candidate_operation_failure(
                    s2.errors._FailureDomain.REFERENCES
                )
                assert observed is failure
                assert (
                    client._candidate_operation_failure(
                        s2.errors._FailureDomain.CITATIONS
                    )
                    is None
                )
                failures[label] = observed

        threads = [
            threading.Thread(target=record_failure, args=(label,))
            for label in ("first outage", "second outage")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert (
            client._candidate_operation_failure(s2.errors._FailureDomain.REFERENCES)
            is None
        )

    assert {label: str(failure) for label, failure in failures.items()} == {
        "first outage": "first outage",
        "second outage": "second outage",
    }


def test_candidate_operation_scope_preserves_tolerant_calls(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A tripped domain retains tolerant contracts without blocking another domain.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate cached metadata.
    :param pytest.LogCaptureFixture caplog: Captured skipped-request diagnostics.
    :return None: Verifies cached batches, domain isolation, and same-domain skips.
    """
    cached = Paper(paper_id="cached", title="Cached", year=2024)
    monkeypatch.setattr(
        s2.disk_cache,
        "_load_cached_paper",
        lambda paper_id: cached if paper_id == "cached" else None,
    )
    with SemanticScholarClient(timeout=1) as client:
        client.client.get_paper_citations = MagicMock(return_value=[])
        client._session.post = MagicMock(
            side_effect=AssertionError("tolerant batch call must not reach S2")
        )
        with client.candidate_operation_scope(), caplog.at_level(logging.WARNING):
            client._record_candidate_operation_failure(
                s2.errors._FailureDomain.PAPER_METADATA,
                SemanticScholarUnavailableError("original outage"),
            )
            assert client.get_papers(["cached", "uncached"]) == {"cached": cached}
            assert client.get_paper_citations("uncached") == []
            client._record_candidate_operation_failure(
                s2.errors._FailureDomain.CITATIONS,
                SemanticScholarUnavailableError("citation outage"),
            )
            assert client.get_paper_citations("later") == []

    client.client.get_paper_citations.assert_called_once()
    client._session.post.assert_not_called()
    assert "Skipped batch fetch" in caplog.text
    assert "Skipped citations" in caplog.text


@pytest.mark.parametrize("first_endpoint", ["sdk", "http"])
def test_candidate_operation_scope_records_tolerant_exhaustion(
    first_endpoint: str,
) -> None:
    """A tolerant exhausted request must not stop another capability.

    :param str first_endpoint: SDK or direct-HTTP endpoint that exhausts first.
    :return None: Verifies the failed budget and one healthy cross-domain call.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        if first_endpoint == "sdk":
            s2.disk_cache._persist_paper(
                Paper(paper_id="healthy", title="Healthy", year=None), "healthy"
            )
            client._session.post = MagicMock(
                side_effect=AssertionError("cached metadata")
            )
            client.client.get_paper_citations = MagicMock(
                side_effect=requests.ConnectionError("offline")
            )
            client._session.get = MagicMock(
                return_value=_MockResponse(
                    200,
                    {"recommendedPapers": [_paper_payload(paper_id="healthy")]},
                )
            )
            first_fetch = client.client.get_paper_citations
            healthy_fetch = client._session.get
        else:
            client._session.get = MagicMock(
                side_effect=requests.ConnectionError("offline")
            )
            client.client.get_paper_citations = MagicMock(return_value=[])
            first_fetch = client._session.get
            healthy_fetch = client.client.get_paper_citations

        with (
            client.candidate_operation_scope(),
            patch("time.sleep") as sleep_mock,
        ):
            if first_endpoint == "sdk":
                assert client.get_paper_citations("first") == []
                assert [
                    paper.paper_id
                    for paper in client.get_recommended_papers(
                        "second", raise_on_unavailable=True
                    )
                ] == ["healthy"]
            else:
                assert client.get_recommended_papers("first") == []
                assert (
                    client.get_paper_citations("second", raise_on_unavailable=True)
                    == []
                )

    assert first_fetch.call_count == API_CONFIG.max_retries
    assert healthy_fetch.call_count == 1
    assert sleep_mock.call_count == API_CONFIG.max_retries - 1


@pytest.mark.parametrize(
    ("module", "blocked_prefixes", "expected_exports"),
    [
        (
            services_module,
            ("citemesh.services.semantic_scholar", "semanticscholar"),
            {
                "SemanticScholarClient",
                "SemanticScholarRequestError",
                "SemanticScholarUnavailableError",
                "get_client",
                "reset_client",
            },
        ),
        (
            importlib.import_module("citemesh.strategies"),
            (
                "citemesh.strategies.citation",
                "citemesh.strategies.embedding",
                "citemesh.strategies.hybrid",
                "citemesh.strategies.recommendation",
            ),
            {
                "GraphBuilderStrategy",
                "CitationGraphBuilder",
                "RecommendationGraphBuilder",
                "EmbeddingGraphBuilder",
                "HybridGraphBuilder",
            },
        ),
        (
            importlib.import_module("citemesh"),
            (
                "citemesh.strategies.citation",
                "citemesh.strategies.embedding",
                "citemesh.strategies.hybrid",
                "citemesh.strategies.recommendation",
            ),
            {
                "__version__",
                "Paper",
                "Author",
                "GraphBuilderStrategy",
                "CitationGraphBuilder",
                "RecommendationGraphBuilder",
                "EmbeddingGraphBuilder",
                "HybridGraphBuilder",
            },
        ),
    ],
    ids=["services", "strategies", "top_level"],
)
def test_package_init_exports_remain_lazy(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    blocked_prefixes: tuple[str, ...],
    expected_exports: set[str],
) -> None:
    """Lazy package exports should not import implementation modules eagerly."""
    _assert_module_reload_is_lazy(
        monkeypatch,
        module=module,
        blocked_prefixes=blocked_prefixes,
        expected_exports=expected_exports,
    )


class _MockResponse:
    """Minimal requests response object for client tests."""

    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Create a minimal mock requests response."""
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        """Raise HTTPError when status code indicates failure."""
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> dict:
        """Return mocked JSON body."""
        return self._payload


class _BrokenJsonResponse(_MockResponse):
    """Response object whose JSON body cannot be decoded."""

    def json(self) -> dict:
        """Raise decode error to emulate malformed response body."""
        raise ValueError("Malformed JSON payload")


class _FakeRequestsSession:
    """Minimal request session with close tracking."""

    def __init__(self) -> None:
        """Create in-memory session stub."""
        self.headers: dict[str, str] = {}
        self.closed = False

    def close(self) -> None:
        """Mark fake session as closed."""
        self.closed = True


def _build_fake_semantic_scholar_api() -> type:
    """Create a SemanticScholar API stub class backed by fake sessions."""

    class _FakeApi:
        """Minimal SemanticScholar wrapper stub."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.session = _FakeRequestsSession()
            self._AsyncSemanticScholar = SimpleNamespace(
                _requester=SimpleNamespace(get_data_async=MagicMock())
            )
            self.closed = False

        def close(self) -> None:
            self.closed = True

    return _FakeApi


def _paper_payload(
    *,
    paper_id: str = "p1",
    title: str = "Paper",
    year: int = 2020,
    abstract: str = "Abstract",
) -> dict:
    """Build a minimal paper payload for API tests."""
    return {
        "paperId": paper_id,
        "title": title,
        "year": year,
        "abstract": abstract,
        "citationCount": 10,
        "fieldsOfStudy": [],
        "authors": [],
    }


def _make_reference_record(paper_id: str) -> SimpleNamespace:
    """Create a minimal reference record with ``paper.paperId``."""
    return SimpleNamespace(paper=SimpleNamespace(paperId=paper_id))


def test_retry_and_backoff_contracts() -> None:
    """Retry policy should handle rate limits, transient failures, and exhaustion."""
    relation_cases = [
        ("get_paper_citations", "2", 2.0),
        ("get_paper_references", "3", 3.0),
    ]
    for api_method, retry_after, delay in relation_cases:
        client = SemanticScholarClient(timeout=1)
        client._rate_limit = lambda: None

        response = requests.Response()
        response.status_code = 429
        response.headers = {"Retry-After": retry_after}

        setattr(
            client.client,
            api_method,
            MagicMock(side_effect=[requests.HTTPError(response=response), []]),
        )

        with patch("time.sleep") as sleep_mock:
            if api_method == "get_paper_citations":
                result = client.get_paper_citations("seed", limit=5)
            else:
                result = client.get_paper_references("seed", limit=5)

        assert sleep_mock.call_count == 1
        assert (
            delay
            <= sleep_mock.call_args_list[0].args[0]
            <= max(delay, 2 * API_CONFIG.retry_delay)
        )
        assert result == []

    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock()
    client._session.get.side_effect = [
        _MockResponse(status_code=429, headers={"Retry-After": "2"}),
        _MockResponse(
            status_code=200,
            payload={"data": [_paper_payload(paper_id="x1", title="A", year=2020)]},
        ),
    ]
    with patch("time.sleep") as sleep_mock:
        results = client.search_papers("attention")

    assert sleep_mock.call_count == 1
    # Jittered backoff floored at Retry-After (2s), capped by the rate-limit
    # multiplier for the first retry (2 * retry_delay).
    assert 2.0 <= sleep_mock.call_args_list[0].args[0] <= 2 * API_CONFIG.retry_delay
    assert len(results) == 1
    assert results[0] == Paper(paper_id="x1", title="A", year=2020, abstract="Abstract")

    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock()
    client._session.get.side_effect = [
        _BrokenJsonResponse(status_code=200),
        _MockResponse(
            status_code=200,
            payload={"data": [_paper_payload(paper_id="x2", title="Retry success")]},
        ),
    ]
    with patch("time.sleep") as sleep_mock:
        results = client.search_papers("transformer")

    assert sleep_mock.call_count == 1
    # Full-jitter wait for a non-429 transient failure on the first retry.
    assert 0.0 <= sleep_mock.call_args_list[0].args[0] <= API_CONFIG.retry_delay
    assert len(results) == 1
    assert results[0].paper_id == "x2"

    api_paper = SimpleNamespace(
        paperId="seed",
        title="Seed",
        year=2020,
        authors=[],
        citationCount=1,
        abstract="abstract",
        fieldsOfStudy=[],
    )
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock(
        side_effect=[
            requests.ConnectionError("temporary"),
            _MockResponse(200, vars(api_paper)),
        ]
    )
    with patch("time.sleep") as sleep_mock:
        result = client.get_paper("seed")

    assert isinstance(result, Paper)
    assert result.paper_id == "seed"
    assert sleep_mock.call_count == 1
    # Full-jitter wait for a non-429 transient failure on the first retry.
    assert 0.0 <= sleep_mock.call_args_list[0].args[0] <= API_CONFIG.retry_delay

    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock(side_effect=requests.ConnectionError("down"))
    with patch("time.sleep") as sleep_mock:
        result = client.get_paper("uncached-seed")

    assert result is None
    assert sleep_mock.call_count == API_CONFIG.max_retries - 1

    for api_method, public_method in [
        ("get_paper_citations", "get_paper_citations"),
        ("get_paper_references", "get_paper_references"),
    ]:
        client = SemanticScholarClient(timeout=1)
        client._rate_limit = lambda: None
        setattr(client.client, api_method, MagicMock(side_effect=Exception("down")))
        with (
            patch("time.sleep"),
            pytest.raises(
                SemanticScholarUnavailableError,
                match="Semantic Scholar API unreachable",
            ),
        ):
            getattr(client, public_method)(
                "seed",
                limit=5,
                raise_on_unavailable=True,
            )

        setattr(client.client, api_method, MagicMock(return_value=[]))
        assert (
            getattr(client, public_method)(
                "seed",
                limit=5,
                raise_on_unavailable=True,
            )
            == []
        )


def test_successful_retry_attempts_are_debug_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Recovered SDK and direct HTTP failures should not emit warnings.

    :param pytest.MonkeyPatch monkeypatch: Pytest patch helper.
    :param pytest.LogCaptureFixture caplog: Captured logging fixture.
    :return None: Assertions verify both retry paths are debug-only before exhaustion.
    """
    api_paper = SimpleNamespace(
        paperId="seed",
        title="Seed",
        year=2020,
        authors=[],
        citationCount=1,
        abstract="abstract",
        fieldsOfStudy=[],
    )
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock(
        side_effect=[
            requests.ConnectionError("temporary"),
            _MockResponse(200, vars(api_paper)),
        ]
    )
    monkeypatch.setattr("time.sleep", lambda _: None)

    with caplog.at_level(logging.DEBUG, logger="citemesh.services.semantic_scholar"):
        result = client.get_paper("seed")

    assert result is not None
    retry_records = [
        record for record in caplog.records if "Retrying in" in record.getMessage()
    ]
    assert len(retry_records) == 1
    assert retry_records[0].levelno == logging.DEBUG

    client._session.get = MagicMock(
        side_effect=[
            _MockResponse(status_code=429, headers={"Retry-After": "1"}),
            _MockResponse(
                status_code=200,
                payload={"data": [_paper_payload(paper_id="result")]},
            ),
        ]
    )
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="citemesh.services.semantic_scholar"):
        results = client.search_papers("attention")

    assert [paper.paper_id for paper in results] == ["result"]
    direct_retry_records = [
        record
        for record in caplog.records
        if "Rate limited by Semantic Scholar" in record.getMessage()
    ]
    assert len(direct_retry_records) == 1
    assert direct_retry_records[0].levelno == logging.DEBUG


def test_long_retry_waits_are_visible_at_default_log_level(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A retry sleep at or above the warning threshold must emit a WARNING.

    :param pytest.MonkeyPatch monkeypatch: Pytest patch helper.
    :param pytest.LogCaptureFixture caplog: Captured logging fixture.
    :return None: Checks outage visibility without real waiting.
    """
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock(
        side_effect=[
            _MockResponse(status_code=503, headers={"Retry-After": "45"}),
            _MockResponse(
                status_code=200,
                payload={"data": [_paper_payload(paper_id="result")]},
            ),
        ]
    )
    monkeypatch.setattr("time.sleep", lambda _: None)

    with caplog.at_level(logging.WARNING, logger="citemesh.services.semantic_scholar"):
        results = client.search_papers("attention")

    assert [paper.paper_id for paper in results] == ["result"]
    warning_records = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "waiting 45s before retrying" in record.getMessage()
    ]
    assert len(warning_records) == 1


def test_related_paper_retry_discards_partial_attempt_results() -> None:
    """A failed relation page must not publish or duplicate earlier IDs."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    records = [
        SimpleNamespace(paper=SimpleNamespace(paperId="first")),
        SimpleNamespace(paper=SimpleNamespace(paperId="second")),
    ]
    for paper_id in ("first", "second"):
        s2.disk_cache._persist_paper(
            Paper(paper_id=paper_id, title=paper_id, year=None), paper_id
        )

    def failed_page():
        """Yield one record before a later-page outage.

        :return Iterator[SimpleNamespace]: Partial relation attempt.
        """
        yield records[0]
        raise requests.ConnectionError("temporary page failure")

    client.client.get_paper_references = MagicMock(side_effect=[failed_page(), records])
    client._session.post = MagicMock(side_effect=AssertionError("cached metadata"))
    with patch("time.sleep"):
        papers = client.get_paper_references("seed", limit=2)

    assert [paper.paper_id for paper in papers] == ["first", "second"]
    assert client.client.get_paper_references.call_count == 2


@pytest.mark.parametrize(
    ("status_code", "message"),
    [(400, "request parameters and requested fields"), (401, "S2_API_KEY")],
)
def test_direct_endpoint_non_retryable_4xx_is_actionable(
    status_code: int,
    message: str,
) -> None:
    """Non-429 client errors should fail once without availability wording."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock(return_value=_MockResponse(status_code=status_code))

    with patch("time.sleep") as sleep_mock:
        with pytest.raises(SemanticScholarRequestError, match=message):
            client.search_papers("attention", raise_on_unavailable=True)

    client._session.get.assert_called_once()
    sleep_mock.assert_not_called()


def test_normalization_and_get_paper_id_contracts() -> None:
    """ID normalization should cover URL/DOI edge cases and request canonical IDs."""
    for raw_id, expected in get_paper_id_normalization_cases():
        assert normalize_paper_id(raw_id) == expected

    doi_cases = [
        ("doi:10.1145/3133956.3134029", "10.1145/3133956.3134029"),
        ("https://doi.org:443/10.1145/3133956.3134029", "10.1145/3133956.3134029"),
        ("doi.org/10.1145/3133956.3134029", "10.1145/3133956.3134029"),
        ("dx.doi.org/10.1145/3133956.3134029", "10.1145/3133956.3134029"),
    ]
    for raw_id, expected in doi_cases:
        assert normalize_paper_id(raw_id) == expected

    non_domain_cases = [
        (
            "https://notdoi.org/10.1145/3133956.3134029",
            "https://notdoi.org/10.1145/3133956.3134029",
        ),
        (
            "https://fooarxiv.org/abs/1706.03762",
            "https://fooarxiv.org/abs/1706.03762",
        ),
    ]
    for raw_id, expected in non_domain_cases:
        assert normalize_paper_id(raw_id) == expected

    with pytest.raises(ValueError, match="Invalid paper ID"):
        normalize_paper_id(None)  # type: ignore[arg-type]

    api_paper = SimpleNamespace(
        paperId="seed",
        title="Seed",
        year=2025,
        authors=[],
        citationCount=1,
        abstract="abstract",
        fieldsOfStudy=[],
    )
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._request_json = MagicMock(return_value=api_paper)

    result = client.get_paper("https://arxiv.org/abs/2508.14040")
    assert isinstance(result, Paper)
    assert result.paper_id == "seed"
    assert client._request_json.call_args.args[0].endswith("/arxiv%3A2508.14040")


@pytest.mark.parametrize(
    ("paper_id", "arxiv_id", "doi", "expected_aliases"),
    [
        (
            "2508.12345v2",
            "",
            "",
            {"2508.12345v2", "2508.12345", "arxiv:2508.12345"},
        ),
        (
            "https://arxiv.org/abs/cs.AI/0704123v3",
            "",
            "",
            {
                "https://arxiv.org/abs/cs.AI/0704123v3",
                "arxiv:cs.AI/0704123",
                "cs.AI/0704123",
            },
        ),
        (
            "S2-opaque-id",
            "2508.12345v2",
            "https://doi.org/10.1000/example",
            {
                "S2-opaque-id",
                "2508.12345v2",
                "2508.12345",
                "arxiv:2508.12345",
                "https://doi.org/10.1000/example",
                "10.1000/example",
            },
        ),
        (
            "S2-opaque-id",
            "",
            "10.1109/CVPR.2016.90",
            {
                "S2-opaque-id",
                "10.1109/CVPR.2016.90",
                "10.1109/cvpr.2016.90",
            },
        ),
    ],
)
def test_paper_identifier_aliases_expand_cross_source_identifiers(
    paper_id: str,
    arxiv_id: str,
    doi: str,
    expected_aliases: set[str],
) -> None:
    """Identifier aliases should bridge arXiv, DOI, and opaque S2 payload fields."""
    aliases = paper_identifier_aliases(
        paper_id=paper_id,
        arxiv_id=arxiv_id,
        doi=doi,
    )

    assert expected_aliases <= set(aliases)
    assert normalize_paper_id("2508.12345v2") == "2508.12345v2"


def test_batch_lookup_keys_use_all_paper_identifier_aliases() -> None:
    """Batch matching should resolve API records through field-level aliases."""
    paper = Paper(
        paper_id="s2-opaque",
        title="Aliased",
        year=2025,
        arxiv_id="2508.12345v2",
        doi="10.1000/example",
    )

    lookup_keys = s2.disk_cache._paper_lookup_keys(paper)

    assert {
        "s2-opaque",
        "2508.12345",
        "arxiv:2508.12345",
        "10.1000/example",
    } <= lookup_keys


def test_paper_metadata_survives_client_restart_and_alias_lookup() -> None:
    """Warm seed lookups must avoid the API and preserve all metadata fields.

    :return None: Checks persistent reuse and isolation from caller mutations.
    """
    payload = _paper_payload(paper_id="s2-seed")
    payload.update(
        authors=[
            {"name": "Ada Example", "authorId": "author-1"},
            {"name": "Blaise Example", "authorId": "author-2"},
            {"name": "Chien Example", "authorId": "author-3"},
            {"name": "Daria Example", "authorId": "author-4"},
        ],
        externalIds={"ArXiv": "2404.08801", "DOI": "10.1234/example"},
        venue="Example Conference",
        fieldsOfStudy=["Computer Science"],
    )
    with SemanticScholarClient(timeout=1) as cold:
        cold._rate_limit = MagicMock()
        cold._request_json = MagicMock(return_value=payload)
        first = cold.get_paper("https://arxiv.org/abs/2404.08801v2")
        assert first is not None
        expected = vars(first).copy()
        first.is_seed = True
        first.references = ["graph-only-reference"]
        cold._request_json.assert_called_once()

    with SemanticScholarClient(timeout=1) as warm:
        warm._rate_limit = MagicMock()
        warm._request_json = MagicMock(side_effect=Exception("HTTP 429"))
        for alias in ("arxiv:2404.08801", "s2-seed", "doi:10.1234/example"):
            restored = warm.get_paper(alias, raise_on_unavailable=True)
            assert restored is not None
            assert vars(restored) == expected
            assert [(author.name, author.author_id) for author in restored.authors] == [
                ("Ada Example", "author-1"),
                ("Blaise Example", "author-2"),
                ("Chien Example", "author-3"),
                ("Daria Example", "author-4"),
            ]
            cached = json.loads(
                s2.disk_cache._paper_cache_path(
                    s2.normalize_paper_id(alias)
                ).read_text()
            )
            assert cached["version"] == s2.disk_cache.PAPER_CACHE_VERSION
            assert cached["paper"]["authors"] == [
                {"name": "Ada Example", "author_id": "author-1"},
                {"name": "Blaise Example", "author_id": "author-2"},
                {"name": "Chien Example", "author_id": "author-3"},
                {"name": "Daria Example", "author_id": "author-4"},
            ]
        warm._request_json.assert_not_called()
        warm._rate_limit.assert_not_called()


def test_legacy_paper_metadata_cache_entry_is_refreshed_with_all_authors() -> None:
    """A legacy cached paper must refetch and replace its truncated author list.

    :return None: Checks versioned cache refresh preserves complete author metadata.
    """
    legacy = Paper(
        paper_id="s2-legacy",
        title="Legacy title",
        year=2024,
        authors=[
            Author(name="Ada Example", author_id="author-1"),
            Author(name="Blaise Example", author_id="author-2"),
            Author(name="Chien Example", author_id="author-3"),
        ],
    )
    s2.disk_cache._paper_cache_path("s2-legacy").write_text(json.dumps(asdict(legacy)))
    assert s2.disk_cache._load_cached_paper("s2-legacy") is None

    payload = _paper_payload(paper_id="s2-legacy")
    payload.update(
        authors=[
            {"name": "Ada Example", "authorId": "author-1"},
            {"name": "Blaise Example", "authorId": "author-2"},
            {"name": "Chien Example", "authorId": "author-3"},
            {"name": "Daria Example", "authorId": "author-4"},
        ],
        externalIds={"ArXiv": "2404.08801", "DOI": "10.1234/example"},
    )
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._request_json = MagicMock(return_value=payload)
        paper = client.get_paper("s2-legacy", raise_on_unavailable=True)
        assert paper is not None
        client._request_json.assert_called_once()

    for alias in ("s2-legacy", "arxiv:2404.08801", "doi:10.1234/example"):
        refreshed = s2.disk_cache._load_cached_paper(s2.normalize_paper_id(alias))
        assert refreshed is not None
        assert [(author.name, author.author_id) for author in refreshed.authors] == [
            ("Ada Example", "author-1"),
            ("Blaise Example", "author-2"),
            ("Chien Example", "author-3"),
            ("Daria Example", "author-4"),
        ]


@pytest.mark.parametrize("reference_ids", [[], ["ref-1", "ref-2"]])
def test_paper_metadata_and_reference_caches_are_independent(
    reference_ids: list[str],
) -> None:
    """A metadata-only hit must still fetch missing references once.

    :param list[str] reference_ids: Successful reference response, possibly empty.
    :return None: Checks warm reference enrichment and reuse of successful empties.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._request_json = MagicMock(return_value=_paper_payload())
        client.client.get_paper_references = MagicMock(
            return_value=[
                _make_reference_record(paper_id) for paper_id in reference_ids
            ]
        )
        assert client.get_paper("p1") is not None
        assert client.get_paper("p1", fetch_references=True).references == reference_ids
        assert client.get_paper("p1", fetch_references=True).references == reference_ids
        client._request_json.assert_called_once()
        client.client.get_paper_references.assert_called_once()


@pytest.mark.parametrize("raise_on_unavailable", [False, True])
def test_paper_metadata_is_saved_before_reference_failure(
    raise_on_unavailable: bool,
) -> None:
    """Reference failures must not discard an already successful metadata lookup.

    :param bool raise_on_unavailable: Whether the caller requests strict failures.
    :return None: Checks partial progress survives a failed enrichment request.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._request_json = MagicMock(return_value=_paper_payload())
        client.client.get_paper_references = MagicMock(
            side_effect=Exception("HTTP 429")
        )
        with patch("time.sleep"):
            if raise_on_unavailable:
                with pytest.raises(SemanticScholarUnavailableError):
                    client.get_paper(
                        "p1", fetch_references=True, raise_on_unavailable=True
                    )
            else:
                assert client.get_paper("p1", fetch_references=True) is None
        assert client.get_paper("p1").title == "Paper"
        client._request_json.assert_called_once()


@pytest.mark.parametrize("failure", [None, Exception("HTTP 429"), {"title": "No ID"}])
def test_unsuccessful_paper_lookups_are_not_cached(failure: Any) -> None:
    """An absent, unavailable, or malformed response must not poison a later retry.

    :param Any failure: First lookup response or operational failure.
    :return None: Checks the next successful lookup still reaches the API.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        if isinstance(failure, Exception):
            client._session.get = MagicMock(return_value=_MockResponse(429))
        else:
            client._request_json = MagicMock(return_value=failure)
        with patch("time.sleep"):
            if isinstance(failure, dict):
                with pytest.raises(TypeError):
                    client.get_paper("p1")
            else:
                assert client.get_paper("p1") is None
        assert not s2.disk_cache._paper_cache_path("p1").exists()
        client._request_json = MagicMock(return_value=_paper_payload())
        assert client.get_paper("p1") is not None
        client._request_json.assert_called_once()


def test_batch_metadata_only_fetches_misses_and_warms_single_lookups() -> None:
    """Batch and single lookup paths must share successful paper metadata.

    :return None: Checks partial and fully warm batches across client instances.
    """
    with SemanticScholarClient(timeout=1) as cold:
        cold._rate_limit = MagicMock()
        cold._request_json = MagicMock(return_value=_paper_payload(paper_id="p1"))
        cold.get_paper("p1")
    with SemanticScholarClient(timeout=1) as mixed:
        mixed._rate_limit = MagicMock()
        mixed._session.post = MagicMock(
            return_value=_MockResponse(200, [_paper_payload(paper_id="p2")])
        )
        mixed._request_json = MagicMock(side_effect=AssertionError("Unexpected lookup"))
        assert set(mixed.get_papers(["p1", "p2", "p1"])) == {"p1", "p2"}
        assert mixed._session.post.call_args.kwargs["json"]["ids"] == ["p2"]
        mixed._request_json.assert_not_called()
    with SemanticScholarClient(timeout=1) as warm:
        warm._session.post = MagicMock(side_effect=Exception("HTTP 429"))
        warm._request_json = MagicMock(side_effect=Exception("HTTP 429"))
        assert set(warm.get_papers(["p1", "p2"])) == {"p1", "p2"}
        assert warm.get_paper("p2").paper_id == "p2"
        warm._session.post.assert_not_called()
        warm._request_json.assert_not_called()


@pytest.mark.parametrize("payload", ["broken json", "[]", '{"authors": [null]}'])
def test_unreadable_paper_cache_is_refetched(payload: str) -> None:
    """Unreadable metadata entries follow the existing reference-cache miss policy.

    :param str payload: Unreadable serialized paper entry.
    :return None: Checks successful refetch replaces the broken entry.
    """
    s2.disk_cache._paper_cache_path("p1").write_text(payload, encoding="utf-8")
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._request_json = MagicMock(return_value=_paper_payload())
        assert client.get_paper("p1").title == "Paper"
        assert client.get_paper("p1").title == "Paper"
        client._request_json.assert_called_once()


def test_batch_null_results_skip_single_fetches_and_preserve_requested_aliases() -> (
    None
):
    """Null batch entries are authoritative while successful rows retain request aliases.

    :return None: Checks no per-ID fallback, no negative cache, and positional aliases.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._session.post = MagicMock(
            return_value=_MockResponse(
                200,
                [
                    _paper_payload(paper_id="seed"),
                    None,
                    _paper_payload(paper_id="other"),
                ],
            )
        )
        client.get_paper = MagicMock(
            side_effect=AssertionError("unnecessary single fetch")
        )
        result = client.get_papers(
            ["https://arxiv.org/abs/2508.14040", "missing-id", "10.18653/v1/N18-3011"]
        )
        assert set(result) == {"arxiv:2508.14040", "10.18653/v1/N18-3011"}
        assert result["arxiv:2508.14040"].paper_id == "seed"
        assert result["10.18653/v1/N18-3011"].paper_id == "other"
        assert client._session.post.call_args.kwargs["json"]["ids"] == [
            "arxiv:2508.14040",
            "missing-id",
            "10.18653/v1/N18-3011",
        ]
        client.get_paper.assert_not_called()
        assert not s2.disk_cache._paper_cache_path("missing-id").exists()
        assert s2.disk_cache._load_cached_paper("arxiv:2508.14040").paper_id == "seed"


def test_direct_endpoint_conversion_and_validation_contracts() -> None:
    """Direct endpoint payload conversion and validation should match API contracts."""
    client = SemanticScholarClient(timeout=1)
    payload = {
        "paperId": "p1",
        "title": "",
        "year": None,
        "abstract": "",
        "citationCount": 0,
        "publicationVenue": {"name": "ICLR"},
        "externalIds": {"ArXiv": "2411.03884v2", "DOI": "10.1145/3133956.3134029"},
        "authors": [
            {"name": " "},
            {"name": " Ada Example ", "authorId": "author-1"},
            {},
            {"name": "Blaise Example", "authorId": "author-2"},
            {"name": "Chien Example", "authorId": "author-3"},
            {"name": "Daria Example", "authorId": "author-4"},
        ],
        "fieldsOfStudy": ["cs.AI", "cs.LG"],
    }
    paper = s2.payloads._convert_recommendation(payload)

    assert paper is not None
    assert paper.paper_id == "p1"
    assert paper.title == "Unknown"
    assert paper.year is None
    assert paper.abstract == ""
    assert paper.venue == "ICLR"
    assert paper.arxiv_id == "2411.03884v2"
    assert paper.doi == "10.1145/3133956.3134029"
    assert [(author.name, author.author_id) for author in paper.authors] == [
        ("Ada Example", "author-1"),
        ("Blaise Example", "author-2"),
        ("Chien Example", "author-3"),
        ("Daria Example", "author-4"),
    ]
    assert paper.categories == ["cs.AI", "cs.LG"]

    client = SemanticScholarClient(timeout=1)
    client._request_json = MagicMock(
        side_effect=[
            {
                "recommendedPapers": [
                    {
                        "paperId": "rec1",
                        "title": "Rec One",
                        "year": 2021,
                        "abstract": "A",
                        "citationCount": 3,
                        "authors": [{"name": "Alice"}],
                        "fieldsOfStudy": ["cs.AI"],
                        "references": [
                            {"paperId": "r1"},
                            {"paper": {"paperId": "r2"}},
                            "r3",
                            {"paperId": "r1"},
                        ],
                    }
                ]
            },
            {
                "data": [
                    {
                        "paperId": "search1",
                        "title": "Search One",
                        "year": 2020,
                        "abstract": "B",
                        "citationCount": 5,
                        "authors": [{"name": "Bob"}],
                        "fieldsOfStudy": ["cs.LG"],
                    }
                ]
            },
        ]
    )

    recommendations = client.get_recommended_papers(
        "seed", limit=1, fields=s2.payloads._default_paper_fields()
    )
    search_results = client.search_papers("transformer", limit=1)
    assert [paper.paper_id for paper in recommendations] == ["rec1"]
    assert recommendations[0].references == ["r1", "r2", "r3"]
    assert [paper.paper_id for paper in search_results] == ["search1"]

    client = SemanticScholarClient(timeout=1)
    client._request_json = MagicMock(return_value={"recommendedPapers": []})
    client.get_recommended_papers(
        "seed", limit=1, fields=["paperId", "title", "references"]
    )
    request_params = client._request_json.call_args.args[1]
    assert isinstance(request_params, dict)
    assert "references" not in str(request_params.get("fields", ""))

    client = SemanticScholarClient(timeout=1)
    validation_cases = [
        ("get_recommended_papers", ("seed",), "limit must be at least 1"),
        ("search_papers", ("attention",), "limit must be at least 1"),
    ]
    for method, args, error in validation_cases:
        with pytest.raises(ValueError, match=error):
            getattr(client, method)(*args, limit=0)

    with pytest.raises(ValueError, match="query must not be empty"):
        client.search_papers("   ", limit=1)
    with pytest.raises(ValueError, match="query must be a string"):
        client.search_papers(123, limit=1)  # type: ignore[arg-type]

    client.client.get_paper_citations = MagicMock(return_value=[])
    client.client.get_paper_references = MagicMock(return_value=[])
    assert client.get_paper_citations("seed", limit=0) == []
    assert client.get_paper_references("seed", limit=0) == []
    client.client.get_paper_citations.assert_not_called()
    client.client.get_paper_references.assert_not_called()

    with pytest.raises(ValueError, match="limit must be an integer"):
        client.get_recommended_papers("seed", limit=True)  # type: ignore[arg-type]

    client._request_json = MagicMock(
        side_effect=AssertionError("invalid limits must fail before transport")
    )
    with pytest.raises(ValueError, match="limit must be at most 500"):
        client.get_recommended_papers("seed", limit=501)
    with pytest.raises(ValueError, match="limit must be at most 1000"):
        client.search_papers("attention", limit=1001)
    client._request_json.assert_not_called()

    encoding_cases = [
        ("10.1145/3133956.3134029", "10.1145%2F3133956.3134029"),
        ("arxiv:math/0301234v1", "arxiv%3Amath%2F0301234"),
    ]
    for raw_id, expected_suffix in encoding_cases:
        client = SemanticScholarClient(timeout=1)
        client._request_json = MagicMock(return_value={"recommendedPapers": []})
        client.get_recommended_papers(raw_id, limit=1)
        url = client._request_json.call_args.args[0]
        assert url.endswith(expected_suffix)


def test_search_papers_paginates_above_graph_request_limit() -> None:
    """Search should split a 101-result request into endpoint-safe pages.

    :return None: Checks page sizes, offsets, and ordered result accumulation.
    """
    first_page = [_paper_payload(paper_id=f"p{i}") for i in range(100)]
    second_page = [_paper_payload(paper_id="p100")]
    client = SemanticScholarClient(timeout=1)
    client._request_json = MagicMock(
        side_effect=[
            {"offset": 0, "next": 100, "data": first_page},
            {"offset": 100, "data": second_page},
        ]
    )

    results = client.search_papers("attention", limit=101)

    assert [paper.paper_id for paper in results] == [f"p{i}" for i in range(101)]
    assert client._request_json.call_count == 2
    first_params = client._request_json.call_args_list[0].args[1]
    second_params = client._request_json.call_args_list[1].args[1]
    assert (first_params["limit"], first_params["offset"]) == (100, 0)
    assert (second_params["limit"], second_params["offset"]) == (1, 100)


@pytest.mark.parametrize("recovers", [True, False])
def test_search_papers_retries_later_page_not_found(recovers: bool) -> None:
    """A later-page 404 should use the page retry budget instead of erasing results.

    :param bool recovers: Whether the later page succeeds on its second attempt.
    :return None: Checks per-page recovery and strict exhaustion behavior.
    """
    first_page = [_paper_payload(paper_id=f"p{i}") for i in range(100)]
    final_page = [_paper_payload(paper_id="p100")]
    responses = [
        _MockResponse(200, {"offset": 0, "next": 100, "data": first_page}),
        _MockResponse(404),
    ]
    if recovers:
        responses.append(_MockResponse(200, {"offset": 100, "data": final_page}))
    else:
        responses.extend(_MockResponse(404) for _ in range(API_CONFIG.max_retries - 1))

    client = SemanticScholarClient(timeout=1)
    client._rate_limit = MagicMock()
    client._session.get = MagicMock(side_effect=responses)
    with patch("time.sleep") as sleep_mock:
        if recovers:
            results = client.search_papers(
                "attention", limit=101, raise_on_unavailable=True
            )
            assert [paper.paper_id for paper in results] == [
                *(f"p{i}" for i in range(100)),
                "p100",
            ]
        else:
            with pytest.raises(
                semantic_module.SemanticScholarUnavailableError,
                match="unreachable while searching for 'attention'",
            ):
                client.search_papers("attention", limit=101, raise_on_unavailable=True)

    expected_page_attempts = 2 if recovers else API_CONFIG.max_retries
    assert client._session.get.call_count == 1 + expected_page_attempts
    assert sleep_mock.call_count == expected_page_attempts - 1


@pytest.mark.parametrize(
    ("method", "paper_key"),
    [
        ("get_paper_references", "citedPaper"),
        ("get_paper_citations", "citingPaper"),
    ],
)
def test_related_papers_use_sdk_page_size_without_capping_total(
    method: str, paper_key: str
) -> None:
    """Relation totals above 1,000 should consume the SDK's paginated iterator.

    :param str method: Public relation endpoint using the real SDK.
    :param str paper_key: Provider field containing the related paper.
    :return None: Checks page traversal and the caller's total with HTTP mocked.
    """
    records = [{paper_key: _paper_payload(paper_id=f"r{i}")} for i in range(1002)]
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(
            side_effect=[
                _MockResponse(200, {"offset": 0, "next": 1000, "data": records[:1000]}),
                _MockResponse(200, {"offset": 1000, "data": records[1000:]}),
                AssertionError("the requested total needs only two pages"),
            ]
        )

        def batch_response(_url: str, **kwargs: Any) -> _MockResponse:
            """Serve full metadata only for the requested missing IDs.

            :param str _url: Batch endpoint URL.
            :param Any kwargs: Request payload and options.
            :return _MockResponse: Positional batch response.
            """
            return _MockResponse(
                200, [_paper_payload(paper_id=value) for value in kwargs["json"]["ids"]]
            )

        client._session.post = MagicMock(side_effect=batch_response)
        papers = getattr(client, method)("seed", limit=1001)

        assert [paper.paper_id for paper in papers] == [f"r{i}" for i in range(1001)]
        assert client._session.get.call_count == 2
        for call, offset in zip(client._session.get.call_args_list, (0, 1000)):
            assert call.args[0].endswith(
                f"/paper/seed/{method.removeprefix('get_paper_')}"
            )
            params = parse_qs(call.kwargs["params"])
            assert params["offset"] == [str(offset)]
            assert params["limit"] == ["1000"]
            assert params["fields"] == ["paperId"]
        assert [
            len(call.kwargs["json"]["ids"])
            for call in client._session.post.call_args_list
        ] == [500, 500, 1]


@pytest.mark.parametrize(
    "method",
    [
        "get_paper_citations",
        "get_paper_references",
        "get_recommended_papers",
        "search_papers",
    ],
)
def test_full_relation_recommendation_and_search_records_populate_paper_cache(
    method: str,
) -> None:
    """Complete records from discovery endpoints should satisfy later paper lookups.

    :param str method: Discovery endpoint expected to return complete paper metadata.
    :return None: Checks one live discovery request and a later metadata-cache hit.
    """
    paper_id = f"{method}-paper"
    payload = _paper_payload(paper_id=paper_id, title="Discovered paper")
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._session.post = MagicMock(return_value=_MockResponse(200, [payload]))
        if method in {"get_paper_citations", "get_paper_references"}:
            fetch = MagicMock(
                return_value=[SimpleNamespace(paper=SimpleNamespace(**payload))]
            )
            setattr(client.client, method, fetch)
            results = getattr(client, method)("seed", limit=1)
            fetch.assert_called_once_with(
                "seed",
                fields=["paperId"],
                limit=1,
            )
        elif method == "get_recommended_papers":
            client._request_json = MagicMock(
                return_value={"recommendedPapers": [payload]}
            )
            results = client.get_recommended_papers("seed", limit=1)
        else:
            client._request_json = MagicMock(return_value={"data": [payload]})
            results = client.search_papers("discovered", limit=1)

        assert [paper.paper_id for paper in results] == [paper_id]
        assert client._session.post.call_count == (
            0 if method == "search_papers" else 1
        )
        client._request_json = MagicMock(
            side_effect=AssertionError("unexpected metadata request after cache hit")
        )
        assert client.get_paper(paper_id).title == "Discovered paper"


@pytest.mark.parametrize("method", ["get_recommended_papers", "search_papers"])
def test_partial_discovery_fields_do_not_replace_cached_paper_metadata(
    method: str,
) -> None:
    """Partial discovery responses must leave richer shared metadata untouched.

    :param str method: Endpoint invoked with a partial custom field list.
    :return None: Checks that custom partial responses do not overwrite cache entries.
    """
    paper_id = f"{method}-paper"
    existing = Paper(
        paper_id=paper_id,
        title="Rich cached title",
        abstract="Rich cached abstract",
        year=2024,
        citation_count=7,
    )
    partial_payload = {"paperId": paper_id, "title": "Partial title"}
    with SemanticScholarClient(timeout=1) as client:
        s2.disk_cache._persist_paper(existing, paper_id)
        if method == "get_recommended_papers":
            client._request_json = MagicMock(
                return_value={"recommendedPapers": [partial_payload]}
            )
            results = client.get_recommended_papers("seed", fields=["paperId", "title"])
        else:
            client._request_json = MagicMock(return_value={"data": [partial_payload]})
            results = client.search_papers("partial", fields=["paperId", "title"])

    assert [paper.title for paper in results] == ["Partial title"]
    cached = s2.disk_cache._load_cached_paper(paper_id)
    assert cached is not None
    assert cached.title == "Rich cached title"
    assert cached.abstract == "Rich cached abstract"


def test_external_id_fallback_from_paper_id_contracts() -> None:
    """Paper-ID fallback should populate arXiv/DOI fields when external IDs are absent."""
    arxiv_payload = {
        "paperId": "arxiv:2411.03884v2",
        "title": "ArXiv Paper",
        "year": 2024,
        "abstract": "A",
        "citationCount": 1,
        "authors": [],
        "fieldsOfStudy": [],
    }
    doi_payload = {
        "paperId": "10.1145/3133956.3134029",
        "title": "DOI Paper",
        "year": 2017,
        "abstract": "B",
        "citationCount": 2,
        "authors": [],
        "fieldsOfStudy": [],
    }

    arxiv = s2.payloads._convert_recommendation(arxiv_payload)
    doi = s2.payloads._convert_recommendation(doi_payload)

    assert arxiv is not None
    assert arxiv.arxiv_id == "2411.03884"
    assert arxiv.doi == ""
    assert doi is not None
    assert doi.arxiv_id == ""
    assert doi.doi == "10.1145/3133956.3134029"


def test_reference_cache_hit_corrupt_and_failure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference cache should reuse valid hits and never persist operational failures."""
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", tmp_path)
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None

    normalized = s2.normalize_paper_id("arxiv:1234.5678")
    cache_path = s2.disk_cache._reference_cache_path(normalized)
    cache_path.write_text(
        json.dumps(
            {
                "paper_id": normalized,
                "references": ["r1", "r2"],
                "version": s2.disk_cache.REFERENCE_CACHE_VERSION,
            }
        )
    )
    client.client.get_paper_references = MagicMock(
        side_effect=AssertionError("API should not be called on cache hit")
    )
    assert client.get_cached_reference_ids("arxiv:1234.5678") == ["r1", "r2"]
    assert client.get_cached_reference_ids("missing-cache-entry") is None
    assert client.get_reference_ids("arxiv:1234.5678") == ["r1", "r2"]

    legacy_mixed_paper_id = s2.normalize_paper_id("seed-mixed")
    legacy_mixed_cache_path = s2.disk_cache._reference_cache_path(legacy_mixed_paper_id)
    legacy_mixed_cache_path.write_text(
        json.dumps(
            {
                "paper_id": legacy_mixed_paper_id,
                "references": [
                    "ok-1",
                    None,
                    {"paperId": "ok-2"},
                    {"paper_id": "ok-3"},
                    {"paper": {"paperId": "ok-4"}},
                    {"paper": {"paper_id": "ok-5"}},
                    {"paperId": "   "},
                    123,
                    "ok-1",
                ],
                "version": s2.disk_cache.REFERENCE_CACHE_VERSION,
            }
        )
    )
    client.client.get_paper_references = MagicMock(
        side_effect=AssertionError(
            "API should not be called on compatible legacy cache"
        )
    )
    assert client.get_reference_ids("seed-mixed") == [
        "ok-1",
        "ok-2",
        "ok-3",
        "ok-4",
        "ok-5",
    ]
    assert json.loads(legacy_mixed_cache_path.read_text())["references"] == [
        "ok-1",
        "ok-2",
        "ok-3",
        "ok-4",
        "ok-5",
    ]

    client.client.get_paper_references = MagicMock(
        return_value=[
            _make_reference_record("fresh-1"),
            _make_reference_record("fresh-2"),
            SimpleNamespace(paper=SimpleNamespace(paperId=None)),
        ]
    )
    assert client.get_reference_ids("arxiv:1234.5678", force_refresh=True) == [
        "fresh-1",
        "fresh-2",
    ]
    assert json.loads(cache_path.read_text())["references"] == ["fresh-1", "fresh-2"]

    rebuild_cases = [
        ("seed", "{bad-json", "text", ["a", "b"]),
        ("seed-unicode", b"\xff\xfe", "bytes", ["unicode-fixed"]),
        (
            "seed-non-object",
            json.dumps(["not", "a", "dict"]),
            "text",
            ["non-object-fixed"],
        ),
        (
            "seed-malformed",
            json.dumps(
                {
                    "paper_id": s2.normalize_paper_id("seed-malformed"),
                    "references": {"unexpected": "mapping"},
                    "version": s2.disk_cache.REFERENCE_CACHE_VERSION,
                }
            ),
            "text",
            ["fixed-1", "fixed-2"],
        ),
        (
            "seed-missing-references",
            json.dumps(
                {
                    "paper_id": s2.normalize_paper_id("seed-missing-references"),
                    "version": s2.disk_cache.REFERENCE_CACHE_VERSION,
                }
            ),
            "text",
            ["missing-references-fixed"],
        ),
        (
            "seed-wrong-identity",
            json.dumps(
                {
                    "paper_id": s2.normalize_paper_id("different-paper"),
                    "references": [],
                    "version": s2.disk_cache.REFERENCE_CACHE_VERSION,
                }
            ),
            "text",
            ["wrong-identity-fixed"],
        ),
    ]
    for paper_id, cached_payload, write_mode, rebuilt_ids in rebuild_cases:
        normalized_paper_id = s2.normalize_paper_id(paper_id)
        rebuilt_cache_path = s2.disk_cache._reference_cache_path(normalized_paper_id)
        if write_mode == "bytes":
            rebuilt_cache_path.write_bytes(cached_payload)
        else:
            rebuilt_cache_path.write_text(cached_payload)
        client.client.get_paper_references = MagicMock(
            return_value=[_make_reference_record(ref_id) for ref_id in rebuilt_ids]
        )
        rebuilt_refs = client.get_reference_ids(paper_id)
        assert rebuilt_refs == rebuilt_ids
        assert json.loads(rebuilt_cache_path.read_text())["references"] == rebuilt_ids

    type_error_id = s2.normalize_paper_id("seed-type-error")
    type_error_cache_path = s2.disk_cache._reference_cache_path(type_error_id)
    client.client.get_paper_references = MagicMock(
        side_effect=TypeError("SDK signature changed")
    )
    with patch("time.sleep") as sleep_mock:
        with pytest.raises(TypeError, match="SDK signature changed"):
            client.get_reference_ids("seed-type-error")
    client.client.get_paper_references.assert_called_once()
    sleep_mock.assert_not_called()
    assert not type_error_cache_path.exists()

    client.client.get_paper_references = MagicMock(
        return_value=iter(
            [
                _make_reference_record("real-reference"),
                _make_reference_record("second-reference"),
            ]
        )
    )
    assert client.get_reference_ids("seed-type-error") == [
        "real-reference",
        "second-reference",
    ]
    client.client.get_paper_references.assert_called_once()
    assert json.loads(type_error_cache_path.read_text())["references"] == [
        "real-reference",
        "second-reference",
    ]


def test_sdk_null_relation_pages_are_valid_empty_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDK failures caused by S2 ``data: null`` pages should become empty evidence.

    :param Path tmp_path: Isolated reference-cache directory.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate the cache path.
    :return None: Assertions define the regression contract.
    """
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", tmp_path)
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    null_page_error = TypeError("'NoneType' object is not iterable")
    client.client.get_paper_references = MagicMock(side_effect=null_page_error)

    with patch("time.sleep") as sleep_mock:
        assert (
            client.get_paper_references(
                "empty-related-papers",
                raise_on_unavailable=True,
            )
            == []
        )
    client.client.get_paper_references.assert_called_once()
    sleep_mock.assert_not_called()

    client.client.get_paper_references = MagicMock(
        side_effect=TypeError("'NoneType' object is not iterable")
    )
    with patch("time.sleep") as sleep_mock:
        assert client.get_reference_ids("empty-reference-ids") == []
    client.client.get_paper_references.assert_called_once()
    sleep_mock.assert_not_called()
    cache_path = s2.disk_cache._reference_cache_path(
        s2.normalize_paper_id("empty-reference-ids")
    )
    assert json.loads(cache_path.read_text())["references"] == []


def test_reference_payload_normalization_keeps_cache_and_live_contracts() -> None:
    """Cache parsing should stay strict while live relation parsing stays tolerant."""
    mixed_payload = [
        " r1 ",
        {"paperId": "r2"},
        {"paper": {"paper_id": "r3"}},
        SimpleNamespace(paper=SimpleNamespace(paperId="r4")),
        "r1",
        None,
    ]
    malformed_payload = [{"unexpected": "shape"}]

    assert s2.payloads._coerce_cached_reference_ids(mixed_payload) == [
        "r1",
        "r2",
        "r3",
        "r4",
    ]
    assert s2.payloads._extract_reference_ids(mixed_payload) == [
        "r1",
        "r2",
        "r3",
        "r4",
    ]
    assert s2.payloads._coerce_cached_reference_ids(malformed_payload) is None
    assert s2.payloads._extract_reference_ids(malformed_payload) == []


@pytest.mark.parametrize(
    "malformed_payload",
    [
        [{"unexpected": "shape"}],
        [Reference({"citedPaper": {}})],
        [Reference({"citedPaper": {"title": "Missing ID"}})],
        {"paperId": "mapping-is-not-a-relation-page"},
        "string-is-not-a-relation-page",
    ],
)
def test_malformed_reference_response_is_never_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformed_payload: object,
) -> None:
    """Nonempty malformed relation responses should fail without becoming evidence."""
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", tmp_path)
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client.client.get_paper_references = MagicMock(return_value=malformed_payload)

    with patch("time.sleep") as sleep_mock:
        with pytest.raises(
            TypeError,
            match=(
                "Reference relation response|"
                "Non-empty reference response contained no valid paper IDs"
            ),
        ):
            client.get_reference_ids("malformed-seed")
    client.client.get_paper_references.assert_called_once()
    sleep_mock.assert_not_called()

    cache_path = s2.disk_cache._reference_cache_path(
        s2.normalize_paper_id("malformed-seed")
    )
    assert not cache_path.exists()

    client.client.get_paper_references = MagicMock(
        return_value=iter([_make_reference_record("repaired-reference")])
    )
    assert client.get_reference_ids("malformed-seed") == ["repaired-reference"]
    client.client.get_paper_references.assert_called_once()
    assert json.loads(cache_path.read_text())["references"] == ["repaired-reference"]


def test_reference_cache_resilience_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference cache pathing, empty persistence, failures, and atomicity should hold."""

    cache_root = tmp_path / "runtime-root"
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", None)
    normalized = s2.normalize_paper_id("arxiv:1234.5678")
    path = s2.disk_cache._reference_cache_path(normalized)
    assert path.parent == cache_root / "references"
    assert path.parent.exists()

    cache_dir = tmp_path / "cache-resilience"
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", cache_dir)
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client.client.get_paper_references = MagicMock(return_value=[])
    refs = client.get_reference_ids("seed-empty")
    normalized_empty = s2.normalize_paper_id("seed-empty")
    cache_path_empty = s2.disk_cache._reference_cache_path(normalized_empty)
    assert refs == []
    assert cache_path_empty.exists()
    assert json.loads(cache_path_empty.read_text()) == {
        "paper_id": normalized_empty,
        "references": [],
        "version": s2.disk_cache.REFERENCE_CACHE_VERSION,
    }

    failing_client = SemanticScholarClient(timeout=1)
    failing_client._rate_limit = lambda: None
    failing_client.client.get_paper_references = MagicMock(
        side_effect=RuntimeError("down")
    )
    with patch("time.sleep"):
        with pytest.raises(
            SemanticScholarUnavailableError,
            match="unreachable while fetching reference IDs for seed-failure",
        ):
            failing_client.get_reference_ids("seed-failure")

    monkeypatch.setattr(
        os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("simulated replace failure")
        ),
    )
    atomic_client = SemanticScholarClient(timeout=1)
    atomic_client._rate_limit = lambda: None
    atomic_client.client.get_paper_references = MagicMock(
        side_effect=AssertionError("API should not be called on replace failure")
    )

    normalized = s2.normalize_paper_id("seed")
    cache_path = s2.disk_cache._reference_cache_path(normalized)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prior_payload = {
        "paper_id": normalized,
        "references": ["cached-ref"],
        "version": s2.disk_cache.REFERENCE_CACHE_VERSION,
    }
    cache_path.write_text(json.dumps(prior_payload))

    refs = atomic_client.get_reference_ids("seed")
    assert refs == ["cached-ref"]
    assert json.loads(cache_path.read_text()) == prior_payload


def test_reference_cache_empty_hit_stays_quiet_in_debug_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Empty cached reference lists should not spam one debug line per paper."""
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", tmp_path)
    client = SemanticScholarClient(timeout=1)

    normalized = s2.normalize_paper_id("seed-empty")
    cache_path = s2.disk_cache._reference_cache_path(normalized)
    cache_path.write_text(
        json.dumps(
            {
                "paper_id": normalized,
                "references": [],
                "version": s2.disk_cache.REFERENCE_CACHE_VERSION,
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.DEBUG):
        refs = client.get_reference_ids("seed-empty")

    assert refs == []
    assert not any(
        "Loaded 0 cached references" in record.message for record in caplog.records
    )


def test_service_module_lifecycle_contracts() -> None:
    """Singleton/context lifecycle and reload behavior should preserve global state."""
    first_client = get_client()
    reset_client()
    second_client = get_client()
    assert first_client is not second_client

    previous_session = requests.Session
    previous_client = semantic_module.client.SemanticScholar
    try:
        requests.Session = _FakeRequestsSession
        semantic_module.client.SemanticScholar = _build_fake_semantic_scholar_api()

        reset_client()
        first = get_client()
        reset_client()
        assert first._session.closed is True
        assert first.client.closed is True
        assert first._closed is True

        with semantic_module.SemanticScholarClient(timeout=1) as ctx_client:
            assert ctx_client._closed is False
            api_session = ctx_client.client.session
            request_session = ctx_client._session

        assert ctx_client._closed is True
        assert request_session.closed is True
        assert api_session.closed is True
        assert ctx_client.client.closed is True
    finally:
        requests.Session = previous_session
        semantic_module.client.SemanticScholar = previous_client
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    original_levels = (httpx_logger.level, httpcore_logger.level)
    try:
        httpx_logger.setLevel(logging.ERROR)
        httpcore_logger.setLevel(logging.CRITICAL)
        importlib.reload(semantic_module)
        assert httpx_logger.level == logging.ERROR
        assert httpcore_logger.level == logging.CRITICAL
    finally:
        httpx_logger.setLevel(original_levels[0])
        httpcore_logger.setLevel(original_levels[1])


def test_get_client_replaces_closed_singleton() -> None:
    """Closed shared clients should be invalidated and recreated on demand."""
    previous_session = requests.Session
    previous_client = semantic_module.client.SemanticScholar
    try:
        requests.Session = _FakeRequestsSession
        semantic_module.client.SemanticScholar = _build_fake_semantic_scholar_api()

        reset_client()
        first = get_client()
        first.close()

        second = get_client()
        assert second is not first
        assert first._closed is True
        assert second._closed is False

        with get_client() as shared_client:
            assert shared_client is second

        third = get_client()
        assert third is not second
        assert second._closed is True
        assert third._closed is False
    finally:
        reset_client()
        requests.Session = previous_session
        semantic_module.client.SemanticScholar = previous_client


def test_rate_limit_pace_is_key_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    """Authenticated clients pace at the faster keyed rate; anonymous stays slow."""
    monkeypatch.delenv("S2_API_KEY", raising=False)
    anonymous = SemanticScholarClient(timeout=1)
    assert anonymous.requests_per_second == API_CONFIG.requests_per_second
    assert anonymous.client.retry is False

    monkeypatch.setenv("S2_API_KEY", "test-key")
    keyed = SemanticScholarClient(timeout=1)
    assert keyed.requests_per_second == API_CONFIG.authenticated_requests_per_second
    assert keyed.client.retry is False
    assert keyed._session.headers["x-api-key"] == "test-key"

    # Empty env value means "explicitly anonymous".
    monkeypatch.setenv("S2_API_KEY", "")
    explicit_anonymous = SemanticScholarClient(timeout=1)
    assert explicit_anonymous.requests_per_second == API_CONFIG.requests_per_second
    assert "x-api-key" not in explicit_anonymous._session.headers


def test_anonymous_pool_notice_logged_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The key-less shared-pool notice appears once per process, not per client."""
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.setattr(semantic_module.client, "_anonymous_pool_announced", False)
    with caplog.at_level(logging.INFO, logger="citemesh.services.semantic_scholar"):
        SemanticScholarClient(timeout=1)
        SemanticScholarClient(timeout=1)
    notices = [
        record
        for record in caplog.records
        if "shared anonymous Semantic" in record.getMessage()
    ]
    assert len(notices) == 1
    assert "429" in notices[0].getMessage()


def test_get_paper_raise_on_unavailable_distinguishes_outage() -> None:
    """Exhausted retries raise SemanticScholarUnavailableError in strict mode."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock(
        side_effect=requests.ConnectionError("connection reset")
    )
    with patch("time.sleep"):
        with pytest.raises(
            semantic_module.SemanticScholarUnavailableError,
            match="unreachable while fetching seed",
        ) as outage:
            client.get_paper("seed", raise_on_unavailable=True)
    assert "service availability issue" in str(outage.value)

    rate_limited = SemanticScholarClient(timeout=1)
    rate_limited._rate_limit = lambda: None
    rate_limited._session.get = MagicMock(return_value=_MockResponse(status_code=429))
    with patch("time.sleep"):
        with pytest.raises(
            semantic_module.SemanticScholarUnavailableError,
            match="rate-limited",
        ):
            rate_limited.get_paper("seed", raise_on_unavailable=True)

    sdk_attempt = MagicMock()
    sdk_attempt.exception.return_value = ConnectionRefusedError(
        "HTTP status 429 Too Many Requests."
    )
    wrapped_rate_limit = semantic_module.errors.RetryError(sdk_attempt)
    wrapped = SemanticScholarClient(timeout=1)
    wrapped._rate_limit = lambda: None
    wrapped.client.get_paper_references = MagicMock(side_effect=wrapped_rate_limit)
    with patch("time.sleep") as sleep_mock:
        with pytest.raises(
            semantic_module.SemanticScholarUnavailableError,
            match=r"rate-limited \(HTTP 429\).*HTTP status 429",
        ):
            wrapped.get_paper_references("seed", raise_on_unavailable=True)
    assert wrapped.client.get_paper_references.call_count == API_CONFIG.max_retries
    assert sleep_mock.call_count == API_CONFIG.max_retries - 1


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            semantic_module.endpoints.BadQueryParametersException("unsupported field"),
            id="bad-query",
        ),
        pytest.param(PermissionError("HTTP status 403 Forbidden."), id="forbidden"),
    ],
)
@pytest.mark.parametrize(
    ("sdk_method", "client_method"),
    [
        ("get_paper_citations", "get_paper_citations"),
        ("get_paper_references", "get_paper_references"),
        ("get_paper_references", "get_reference_ids"),
    ],
)
def test_sdk_request_errors_are_not_retried(
    error: Exception, sdk_method: str, client_method: str
) -> None:
    """HTTP 400 and 403 SDK errors should fail once at every API boundary."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    sdk_mock = MagicMock(side_effect=error)
    setattr(client.client, sdk_method, sdk_mock)

    with (
        patch("time.sleep") as sleep_mock,
        pytest.raises(
            semantic_module.SemanticScholarRequestError,
            match="rejected the request",
        ),
    ):
        if client_method == "get_reference_ids":
            client.get_reference_ids("seed", force_refresh=True)
        elif client_method == "get_papers":
            client.get_papers(["seed"])
        else:
            kwargs = {} if client_method == "get_paper" else {"limit": 5}
            getattr(client, client_method)("seed", raise_on_unavailable=True, **kwargs)

    sdk_mock.assert_called_once()
    sleep_mock.assert_not_called()


def test_get_paper_not_found_still_returns_none_in_strict_mode() -> None:
    """Strict mode only changes outage handling; genuine not-found stays None."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock(return_value=_MockResponse(404))
    assert client.get_paper("missing-id", raise_on_unavailable=True) is None


@pytest.mark.parametrize("raise_on_unavailable", [False, True])
def test_malformed_present_paper_payload_is_not_reported_as_missing(
    raise_on_unavailable: bool,
) -> None:
    """A malformed successful seed response should fail once in either mode."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._request_json = MagicMock(return_value={"title": "Missing ID"})

    with (
        patch("time.sleep") as sleep_mock,
        pytest.raises(TypeError, match="malformed paper payload"),
    ):
        client.get_paper("seed", raise_on_unavailable=raise_on_unavailable)

    client._request_json.assert_called_once()
    sleep_mock.assert_not_called()


def test_unconvertible_paper_payload_error_names_the_cause() -> None:
    """A payload that fails Paper conversion must surface the conversion error.

    :return None: Checks the single-paper strict path chains its root cause.
    """
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._request_json = MagicMock(
        return_value={
            "paperId": "Z",
            "title": "T",
            "year": 3000,
            "authors": [],
            "citationCount": 1,
            "abstract": "a",
            "externalIds": {},
        }
    )

    with pytest.raises(TypeError, match="malformed paper payload.*Invalid year"):
        client.get_paper("Z")


def test_recommendations_fall_back_to_all_cs_pool() -> None:
    """Empty default-pool responses retry against the broader all-cs pool."""
    client = SemanticScholarClient(timeout=1)
    s2.disk_cache._persist_paper(
        Paper(paper_id="classic1", title="Classic Result", year=None), "classic1"
    )
    client._session.post = MagicMock(side_effect=AssertionError("cached metadata"))
    client._request_json = MagicMock(
        side_effect=[
            {"recommendedPapers": []},
            {
                "recommendedPapers": [
                    {
                        "paperId": "classic1",
                        "title": "Classic Result",
                        "year": 2017,
                        "abstract": "A",
                        "citationCount": 10,
                        "authors": [],
                        "fieldsOfStudy": [],
                    }
                ]
            },
        ]
    )
    recommendations = client.get_recommended_papers("seed", limit=5)
    assert [paper.paper_id for paper in recommendations] == ["classic1"]
    assert client._request_json.call_count == 2
    first_params = client._request_json.call_args_list[0].args[1]
    second_params = client._request_json.call_args_list[1].args[1]
    assert "from" not in first_params
    assert second_params["from"] == "all-cs"
    assert second_params["limit"] == first_params["limit"]

    # Both pools empty (e.g. non-CS paper): returns [] without error.
    client._request_json = MagicMock(return_value={"recommendedPapers": []})
    assert client.get_recommended_papers("seed", limit=5) == []
    assert client._request_json.call_count == 2

    # An unavailable primary request already exhausted its retry budget; it is
    # not evidence that the recent pool was valid but empty.
    client._request_json = MagicMock(return_value=None)
    assert client.get_recommended_papers("seed", limit=5) == []
    assert client._request_json.call_count == 1

    primary_outage = SemanticScholarUnavailableError("recommendations unavailable")
    client._request_json = MagicMock(side_effect=primary_outage)
    with pytest.raises(
        SemanticScholarUnavailableError, match="recommendations unavailable"
    ):
        client.get_recommended_papers(
            "seed",
            limit=5,
            raise_on_unavailable=True,
        )
    assert client._request_json.call_count == 1

    with SemanticScholarClient(timeout=1, retry_budget_seconds=0) as scoped_client:
        s2.disk_cache._persist_paper(
            Paper(paper_id="later", title="Later", year=None), "later"
        )
        scoped_client._session.post = MagicMock(
            side_effect=AssertionError("cached metadata")
        )
        scoped_client._rate_limit = MagicMock()
        scoped_client._session.get = MagicMock(
            side_effect=[
                _MockResponse(200, {"recommendedPapers": []}),
                *[requests.ConnectionError("all-cs unavailable")]
                * API_CONFIG.max_retries,
                _MockResponse(
                    200,
                    {"recommendedPapers": [_paper_payload(paper_id="later")]},
                ),
            ]
        )
        with (
            scoped_client.candidate_operation_scope(),
            patch("time.sleep") as sleep_mock,
        ):
            with pytest.raises(SemanticScholarUnavailableError, match="all-cs"):
                scoped_client.get_recommended_papers(
                    "seed", limit=5, raise_on_unavailable=True
                )
            with pytest.raises(SemanticScholarUnavailableError, match="Skipped"):
                scoped_client.get_recommended_papers(
                    "later-seed", limit=5, raise_on_unavailable=True
                )

        assert [
            paper.paper_id
            for paper in scoped_client.get_recommended_papers(
                "later-seed", limit=5, raise_on_unavailable=True
            )
        ] == ["later"]

        assert scoped_client._session.get.call_count == API_CONFIG.max_retries + 2
        assert sleep_mock.call_count == API_CONFIG.max_retries - 1


def test_jittered_backoff_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff escalates exponentially with jitter, floored at Retry-After."""
    drawn_bounds: list[tuple[float, float]] = []

    def _record_uniform(low: float, high: float) -> float:
        drawn_bounds.append((low, high))
        return high  # deterministic: always draw the cap

    monkeypatch.setattr(random, "uniform", _record_uniform)

    # Exponential cap growth: retry_delay * 2^(attempt-1), doubled when
    # rate-limited, capped at _MAX_BACKOFF_SECONDS.
    delay = API_CONFIG.retry_delay
    assert semantic_module.retry._jittered_backoff(1) == delay
    assert semantic_module.retry._jittered_backoff(2) == delay * 2
    assert semantic_module.retry._jittered_backoff(1, rate_limited=True) == delay * 2
    assert semantic_module.retry._jittered_backoff(3, rate_limited=True) == delay * 8
    assert (
        semantic_module.retry._jittered_backoff(30, rate_limited=True)
        == semantic_module.retry._MAX_BACKOFF_SECONDS
    )
    assert all(low == 0.0 for low, _high in drawn_bounds)

    # The server's minimum wait takes precedence over the local jitter cap but
    # is itself bounded so one response cannot stall a build for hours.
    monkeypatch.setattr(random, "uniform", lambda low, high: 0.0)
    assert semantic_module.retry._jittered_backoff(1, retry_after=7.5) == 7.5
    assert semantic_module.retry._jittered_backoff(1, retry_after=120.0) == 120.0
    assert (
        semantic_module.retry._jittered_backoff(1, retry_after=999.0)
        == semantic_module.retry._MAX_RETRY_AFTER_SECONDS
    )


@pytest.mark.parametrize("endpoint", ["sdk", "rest"])
def test_long_outage_recovers_on_last_attempt(endpoint: str) -> None:
    """Both retry paths must outlast the old budget and recover on attempt thirty.

    :param str endpoint: SDK reference lookup or direct REST search request.
    :return None: Verifies the default budget without real waiting or network calls.
    """
    assert API_CONFIG.max_retries == 30
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        if endpoint == "sdk":
            fetch = MagicMock(
                side_effect=[
                    ConnectionRefusedError("HTTP status 429 Too Many Requests.")
                ]
                * 29
                + [[_make_reference_record("p1")]]
            )
            client.client.get_paper_references = fetch
        else:
            fetch = MagicMock(
                side_effect=[_MockResponse(status_code=503)] * 29
                + [_MockResponse(status_code=200, payload={"data": [_paper_payload()]})]
            )
            client._session.get = fetch
        with patch("time.sleep") as sleep_mock:
            result = (
                client.get_reference_ids("p1")
                if endpoint == "sdk"
                else client.search_papers("attention", raise_on_unavailable=True)
            )
        assert result
        assert fetch.call_count == 30
        assert sleep_mock.call_count == 29


@pytest.mark.parametrize("endpoint", ["sdk", "rest"])
def test_server_retry_after_can_exceed_local_cap(endpoint: str) -> None:
    """Long server cooldowns must be honored on SDK and direct server failures.

    :param str endpoint: SDK reference lookup or REST search retry path.
    :return None: Checks Retry-After propagation through the actual retry engine.
    """
    response = _MockResponse(status_code=503, headers={"Retry-After": "120"})
    with SemanticScholarClient(timeout=1, retry_budget_seconds=0) as client:
        client._rate_limit = MagicMock()
        client.client.get_paper_references = MagicMock(
            side_effect=[
                requests.HTTPError(response=response),
                [_make_reference_record("p1")],
            ]
        )
        client._session.get = MagicMock(
            side_effect=[response, _MockResponse(status_code=200, payload={"data": []})]
        )
        with patch("time.sleep") as sleep_mock:
            if endpoint == "sdk":
                client.get_reference_ids("p1")
            else:
                client.search_papers("attention", raise_on_unavailable=True)
        sleep_mock.assert_called_once_with(120.0)


@pytest.mark.parametrize("endpoint", ["sdk", "rest"])
def test_retry_wait_can_be_interrupted(endpoint: str) -> None:
    """User cancellation must escape both retry engines immediately.

    :param str endpoint: SDK reference lookup or REST search retry path.
    :return None: Checks KeyboardInterrupt is neither retried nor converted.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client.client.get_paper_references = MagicMock(
            side_effect=ConnectionRefusedError("HTTP status 429 Too Many Requests.")
        )
        client._session.get = MagicMock(return_value=_MockResponse(status_code=429))
        with patch(
            "time.sleep",
            side_effect=KeyboardInterrupt,
        ):
            with pytest.raises(KeyboardInterrupt):
                if endpoint == "sdk":
                    client.get_reference_ids("p1")
                else:
                    client.search_papers("attention", raise_on_unavailable=True)
        fetch = (
            client.client.get_paper_references
            if endpoint == "sdk"
            else client._session.get
        )
        fetch.assert_called_once()


def test_search_raise_on_unavailable_distinguishes_rate_limit() -> None:
    """Exhausted search retries raise in strict mode instead of returning []."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock(return_value=_MockResponse(status_code=429))

    with patch("time.sleep"):
        assert client.search_papers("attention") == []
        with pytest.raises(
            semantic_module.SemanticScholarUnavailableError,
            match="rate-limited .* searching for 'attention'",
        ):
            client.search_papers("attention", raise_on_unavailable=True)


def test_search_raise_on_unavailable_distinguishes_outage() -> None:
    """Connection failures surface as 'unreachable' in strict mode."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock(side_effect=requests.ConnectionError("boom"))

    with patch("time.sleep"):
        with pytest.raises(
            semantic_module.SemanticScholarUnavailableError,
            match="unreachable while searching",
        ):
            client.search_papers("attention", raise_on_unavailable=True)


def test_search_empty_results_stay_empty_in_strict_mode() -> None:
    """Strict mode only changes outage handling; genuinely empty stays []."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock(
        return_value=_MockResponse(status_code=200, payload={"data": []})
    )
    assert client.search_papers("attention", raise_on_unavailable=True) == []

    client._session.get = MagicMock(return_value=_MockResponse(status_code=404))
    assert client.search_papers("attention", raise_on_unavailable=True) == []


@pytest.mark.parametrize("status", [408, 502, 503])
def test_single_paper_uses_http_status_and_recovers(status: int) -> None:
    """Statuses discarded by the SDK must remain retryable on paper lookups.

    :param int status: Transient status unmapped by the SDK.
    :return None: Asserts REST retry recovery and successful metadata persistence.
    """
    from semanticscholar.Paper import Paper as SDKPaper

    assert SDKPaper({})  # The SDK's unknown-status result bypasses truthiness checks.
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client.client.get_paper = MagicMock(return_value=SDKPaper({}))
        client._session.get = MagicMock(
            side_effect=[_MockResponse(status), _MockResponse(200, _paper_payload())]
        )
        with patch("time.sleep") as sleep_mock:
            assert client.get_paper("p1", raise_on_unavailable=True).paper_id == "p1"
        assert client._session.get.call_count == 2
        sleep_mock.assert_called_once()
        client.client.get_paper.assert_not_called()
        assert s2.disk_cache._load_cached_paper("p1").paper_id == "p1"


@pytest.mark.parametrize("status", [400, 401, 403])
def test_single_paper_rejected_request_is_not_retried(status: int) -> None:
    """Revoked credentials and malformed requests must be actionable immediately.

    :param int status: Deterministic client error status.
    :return None: Checks one HTTP attempt and no poisoned paper-cache entry.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(return_value=_MockResponse(status))
        with (
            patch("time.sleep") as sleep_mock,
            pytest.raises(
                semantic_module.SemanticScholarRequestError, match=f"HTTP {status}"
            ),
        ):
            client.get_paper("p1", raise_on_unavailable=True)
        client._session.get.assert_called_once()
        sleep_mock.assert_not_called()
        assert not s2.disk_cache._paper_cache_path("p1").exists()


@pytest.mark.parametrize("strict", [False, True])
def test_exhausted_batch_does_not_restart_retries_per_id(strict: bool) -> None:
    """One exhausted batch must not multiply its retry budget by unresolved IDs.

    :param bool strict: Whether an outage should raise instead of returning cache hits.
    :return None: Checks total calls, aggregate sleeps, and cached data preservation.
    """
    cached = Paper(paper_id="cached", title="Cached", year=2020)
    s2.disk_cache._persist_paper(cached, "cached")
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._session.post = MagicMock(side_effect=requests.ConnectionError("down"))
        with (
            patch("time.sleep") as sleep_mock,
            patch(
                "random.uniform",
                side_effect=lambda low, high: high,
            ),
            client.candidate_operation_scope(),
        ):
            if strict:
                with pytest.raises(semantic_module.SemanticScholarUnavailableError):
                    client.get_papers(["cached", "p1", "p2"], raise_on_unavailable=True)
            else:
                assert client.get_papers(["cached", "p1", "p2"]) == {"cached": cached}
            assert client.get_paper("cached", raise_on_unavailable=True) == cached
            assert client.get_papers(
                ["cached", "still-uncached"], raise_on_unavailable=False
            ) == {"cached": cached}
        assert client._session.post.call_count == API_CONFIG.max_retries
        assert sleep_mock.call_count == API_CONFIG.max_retries - 1
        assert sum(call.args[0] for call in sleep_mock.call_args_list) <= (
            (API_CONFIG.max_retries - 1) * s2.retry._MAX_BACKOFF_SECONDS
        )
        assert s2.disk_cache._load_cached_paper("cached") == cached


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
@pytest.mark.parametrize(
    "method", ["get_papers", "get_paper_references", "get_paper_citations"]
)
def test_deterministic_batch_and_relation_failures_do_not_retry(
    error_type: type[Exception], method: str
) -> None:
    """Local contract failures must bypass every SDK retry budget.

    :param type[Exception] error_type: Deterministic conversion or SDK error class.
    :param str method: Public batch or relation method.
    :return None: Checks one attempt, propagation, and absence of per-ID fallbacks.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        fetch = MagicMock(side_effect=error_type("invalid payload"))
        if method == "get_papers":
            client._session.post = fetch
        else:
            setattr(client.client, method, fetch)
        client.get_paper = MagicMock()
        with (
            patch("time.sleep") as sleep_mock,
            pytest.raises(error_type, match="invalid payload"),
        ):
            getattr(client, method)(["p1"] if method == "get_papers" else "p1")
        fetch.assert_called_once()
        sleep_mock.assert_not_called()
        client.get_paper.assert_not_called()


@pytest.mark.parametrize("mapping", [False, True])
def test_all_unresolved_references_are_empty_and_cached(
    mapping: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Legitimate reference records outside the S2 corpus are successful empty data.

    :param bool mapping: Whether to use a mapping or the installed SDK's record shape.
    :param pytest.LogCaptureFixture caplog: Captured warning records.
    :return None: Checks caching and warning without retry or raw TypeError.
    """
    record = (
        {"paper": {"paperId": None, "title": "Unindexed work"}}
        if mapping
        else Reference({"citedPaper": {"paperId": None, "title": "Unindexed work"}})
    )
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client.client.get_paper_references = MagicMock(return_value=[record])
        with (
            patch("time.sleep") as sleep_mock,
            caplog.at_level(logging.WARNING),
        ):
            assert client.get_reference_ids("p1") == []
            assert client.get_reference_ids("p1") == []
        client.client.get_paper_references.assert_called_once()
        sleep_mock.assert_not_called()
        assert "no reference IDs are available" in caplog.text
        assert (
            json.loads(s2.disk_cache._reference_cache_path("p1").read_text())[
                "references"
            ]
            == []
        )


def test_all_cs_rejection_is_not_a_successful_empty_result() -> None:
    """A failed broader-pool check cannot establish empty discovery.

    :return None: Checks failure propagation and absence of a new snapshot.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._request_json = MagicMock(
            side_effect=[
                {"recommendedPapers": []},
                semantic_module.SemanticScholarRequestError("HTTP 400"),
            ]
        )
        with pytest.raises(SemanticScholarRequestError, match="HTTP 400"):
            client.get_recommended_papers("p1", raise_on_unavailable=True)
        assert client._request_json.call_count == 2
        assert not s2.disk_cache._discovery_cache_path(
            ("recommendations", "p1", 50, "recent")
        ).exists()


def test_rate_limit_detection_uses_status_or_sdk_exception_contract() -> None:
    """Incidental digits in identifiers and messages must not imply rate limiting.

    :return None: Checks HTTP status evidence and the SDK's exact exception shape.
    """
    assert s2.retry._is_rate_limit_error(
        requests.HTTPError(response=_MockResponse(429))
    )
    assert s2.retry._is_rate_limit_error(
        ConnectionRefusedError("HTTP status 429 Too Many Requests.")
    )
    assert s2.retry._is_rate_limit_error(
        semantic_module.errors._RetryableRequestError("HTTP 429", rate_limited=True)
    )
    assert not s2.retry._is_rate_limit_error(
        semantic_module.errors._RetryableRequestError("pagination HTTP 404")
    )
    assert not s2.retry._is_rate_limit_error(
        requests.HTTPError("https://example/paper/42901", response=_MockResponse(503))
    )
    assert not s2.retry._is_rate_limit_error(ValueError("invalid ID 42901"))


@pytest.mark.parametrize("batch", [False, True])
def test_explicit_paper_cache_refresh_updates_metadata_and_preserves_failed_data(
    batch: bool,
) -> None:
    """Metadata refresh must bypass reads and only replace successful cached entries.

    :param bool batch: Whether to exercise batch or single-paper refresh.
    :return None: Checks fresh citation counts and persistence after an outage.
    """
    old = Paper(paper_id="p1", title="Old", year=2020, citation_count=1)
    s2.disk_cache._persist_paper(old, "p1")
    with SemanticScholarClient(timeout=1, refresh_paper_cache=True) as client:
        client._rate_limit = MagicMock()
        payload = {**_paper_payload(paper_id="p1", title="Fresh"), "citationCount": 8}
        client._session.get = MagicMock(return_value=_MockResponse(200, payload))
        client._session.post = MagicMock(return_value=_MockResponse(200, [payload]))
        result = client.get_papers(["p1"])["p1"] if batch else client.get_paper("p1")
        assert result.citation_count == 8
        assert s2.disk_cache._load_cached_paper("p1").title == "Fresh"
        client._session.get = MagicMock(return_value=_MockResponse(503))
        client._session.post = MagicMock(side_effect=requests.ConnectionError("down"))
        with (
            patch("time.sleep"),
            pytest.raises(semantic_module.SemanticScholarUnavailableError),
        ):
            if batch:
                client.get_papers(["p1"], raise_on_unavailable=True)
            else:
                client.get_paper("p1", raise_on_unavailable=True)
        assert s2.disk_cache._load_cached_paper("p1").citation_count == 8


def test_explicit_api_key_does_not_mutate_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config credentials can be injected without exporting them to subprocesses.

    :param pytest.MonkeyPatch monkeypatch: Isolated environment overrides.
    :return None: Checks explicit credentials and explicit anonymous access.
    """
    monkeypatch.setenv("S2_API_KEY", "environment-key")
    with SemanticScholarClient(api_key="configured-key") as client:
        assert client._session.headers["x-api-key"] == "configured-key"
        assert os.environ["S2_API_KEY"] == "environment-key"
    with SemanticScholarClient(api_key="") as client:
        assert "x-api-key" not in client._session.headers
        assert client.requests_per_second == API_CONFIG.requests_per_second


@pytest.mark.parametrize(
    "method", ["get_papers", "get_paper_references", "get_reference_ids"]
)
@pytest.mark.parametrize("recovers", [False, True])
def test_sdk_json_decode_failures_keep_operational_retries(
    method: str, recovers: bool
) -> None:
    """Transient non-JSON SDK error responses must retain the operational retry budget.

    :param str method: Public batch or reference method under test.
    :param bool recovers: Whether the second attempt supplies an empty valid response.
    :return None: Checks recovery or classified exhaustion without masking plain errors.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        failure = json.JSONDecodeError("upstream HTML response", "<html>", 0)
        fetch = MagicMock(side_effect=[failure, []] if recovers else failure)
        sdk_method = "get_paper_references" if method == "get_reference_ids" else method
        if method == "get_papers":
            fetch.side_effect = (
                [failure, _MockResponse(200, [None])] if recovers else failure
            )
            client._session.post = fetch
        else:
            setattr(client.client, sdk_method, fetch)
        args = ["p1"] if method == "get_papers" else "p1"
        kwargs = {} if method == "get_reference_ids" else {"raise_on_unavailable": True}
        with patch("time.sleep") as sleep_mock:
            if recovers:
                if method == "get_papers":
                    client.get_paper = MagicMock(return_value=None)
                assert getattr(client, method)(args, **kwargs) == (
                    {} if method == "get_papers" else []
                )
            else:
                with pytest.raises(semantic_module.SemanticScholarUnavailableError):
                    getattr(client, method)(args, **kwargs)
        assert fetch.call_count == (2 if recovers else API_CONFIG.max_retries)
        assert sleep_mock.call_count == (1 if recovers else API_CONFIG.max_retries - 1)
        if not recovers:
            assert not s2.disk_cache._reference_cache_path("p1").exists()


@pytest.mark.parametrize(
    "method",
    ["get_papers", "get_paper_references", "get_paper_citations", "get_reference_ids"],
)
@pytest.mark.parametrize("recovers", [False, True])
@pytest.mark.parametrize("status", [408, 503])
def test_real_sdk_transport_preserves_outages_and_stops_failed_pagination(
    method: str, recovers: bool, status: int
) -> None:
    """An SDK relation or batch outage must never become empty data or endless paging.

    :param str method: Public SDK-backed API operation.
    :param bool recovers: Whether a successful response follows the first retryable error.
    :param int status: Retryable HTTP status returned by the transport.
    :return None: Checks the real SDK conversion and a single bounded retry budget.
    """
    relation_key = "citingPaper" if method == "get_paper_citations" else "citedPaper"
    payload = (
        [_paper_payload()]
        if method == "get_papers"
        else {
            "offset": 0,
            "data": [{relation_key: _paper_payload()}],
        }
    )
    with SemanticScholarClient(timeout=1, retry_budget_seconds=0) as client:
        client._rate_limit = MagicMock()
        response = _MockResponse(status, headers={"Retry-After": "120"})
        fetch = MagicMock(
            side_effect=[response, _MockResponse(200, payload)] if recovers else None,
            return_value=response,
        )
        if method == "get_papers":
            client._session.post = fetch
            client._session.get = MagicMock(
                side_effect=AssertionError("unexpected per-ID fallback")
            )
        else:
            client._session.get = fetch
            client._session.post = MagicMock(
                side_effect=AssertionError("unexpected metadata request")
            )
            if recovers and method in {"get_paper_references", "get_paper_citations"}:
                s2.disk_cache._persist_paper(
                    s2.payloads._convert_api_paper(_paper_payload()), "p1"
                )
        args = ["p1"] if method == "get_papers" else "p1"
        kwargs = {} if method == "get_reference_ids" else {"raise_on_unavailable": True}
        with patch("time.sleep") as sleep_mock:
            if recovers:
                result = getattr(client, method)(args, **kwargs)
                assert (
                    list(result) == ["p1"]
                    if method in {"get_papers", "get_reference_ids"}
                    else [paper.paper_id for paper in result] == ["p1"]
                )
            else:
                with pytest.raises(semantic_module.SemanticScholarUnavailableError):
                    getattr(client, method)(args, **kwargs)
        assert fetch.call_count == (2 if recovers else API_CONFIG.max_retries)
        assert sleep_mock.call_count == (1 if recovers else API_CONFIG.max_retries - 1)
        assert all(call.args == (120.0,) for call in sleep_mock.call_args_list)
        if not recovers:
            assert not s2.disk_cache._reference_cache_path("p1").exists()
            assert not s2.disk_cache._paper_cache_path("p1").exists()


def test_mid_pagination_404_is_an_outage_not_an_empty_reference_list() -> None:
    """A 404 on a later relation page must not persist an empty reference list.

    :return None: Checks the real SDK pagination path with a two-page response.
    """
    first_page = {
        "offset": 0,
        "next": 100,
        "data": [{"citedPaper": _paper_payload()} for _ in range(100)],
    }

    def _respond(url: str, **kwargs: Any) -> _MockResponse:
        """Serve a full first page, then HTTP 404 for every later page.

        :param str url: Requested endpoint URL.
        :param Any kwargs: Request keyword arguments carrying the params string.
        :return _MockResponse: Page payload or a pagination 404.
        """
        params = str(kwargs.get("params", ""))
        if "offset=0" in params or "offset=" not in params:
            return _MockResponse(200, first_page)
        return _MockResponse(404)

    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(side_effect=_respond)
        with patch("time.sleep") as sleep_mock:
            with pytest.raises(
                semantic_module.SemanticScholarUnavailableError,
                match="unreachable",
            ) as outage:
                client.get_reference_ids("p1")
        assert "rate-limited (HTTP 429)" not in str(outage.value)
        assert sleep_mock.call_count == API_CONFIG.max_retries - 1
        # Each attempt refetches page one and then hits the pagination 404.
        assert client._session.get.call_count == API_CONFIG.max_retries * 2
        assert not s2.disk_cache._reference_cache_path("p1").exists()


def test_first_page_404_is_not_persisted_as_empty_reference_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing paper must not become a permanent cached empty reference list.

    :param Path tmp_path: Isolated reference-cache directory.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate the cache path.
    :return None: Assertions validate a later successful lookup remains possible.
    """
    monkeypatch.setattr(s2.disk_cache, "REFERENCE_CACHE_DIR", tmp_path)
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client.client.get_paper_references = MagicMock(
        side_effect=semantic_module.client.ObjectNotFoundException("missing paper")
    )

    assert client.get_reference_ids("p1") == []
    cache_path = s2.disk_cache._reference_cache_path("p1")
    assert not cache_path.exists()

    client.client.get_paper_references = MagicMock(
        return_value=[_make_reference_record("recovered-reference")]
    )
    assert client.get_reference_ids("p1") == ["recovered-reference"]
    assert json.loads(cache_path.read_text())["references"] == ["recovered-reference"]


@pytest.mark.parametrize(
    "method",
    ["get_papers", "get_paper_references", "get_paper_citations", "get_reference_ids"],
)
def test_real_sdk_transport_auth_rejection_fails_once(method: str) -> None:
    """SDK-backed methods must propagate the transport's actionable HTTP 401 error.

    :param str method: Public SDK-backed API operation.
    :return None: Checks one HTTP attempt and no sleeping or cached failure.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        fetch = MagicMock(return_value=_MockResponse(401))
        client._session.get = fetch
        client._session.post = fetch
        with (
            patch("time.sleep") as sleep_mock,
            pytest.raises(
                semantic_module.SemanticScholarRequestError, match="HTTP 401"
            ),
        ):
            getattr(client, method)(["p1"] if method == "get_papers" else "p1")
        fetch.assert_called_once()
        sleep_mock.assert_not_called()
        assert not s2.disk_cache._reference_cache_path("p1").exists()
        assert not s2.disk_cache._paper_cache_path("p1").exists()


@pytest.mark.parametrize("api_key, interval", [("", 2.0), ("test-key", 1.0)])
def test_real_sdk_transport_keeps_relation_pagination(
    monkeypatch: pytest.MonkeyPatch, api_key: str, interval: float
) -> None:
    """Pace every page and subsequent direct requests using the same request budget.

    :param pytest.MonkeyPatch monkeypatch: Fixture for replacing the clock.
    :param str api_key: Anonymous or authenticated request configuration.
    :param float interval: Minimum spacing between HTTP requests.
    :return None: Verifies complete pagination and exactly one pacing interval per send.
    """
    clock = [100.0]
    request_times = []

    def sleep(seconds: float) -> None:
        """Advance simulated time without delaying the test.

        :param float seconds: Requested wait duration.
        :return None: Updates the simulated clock.
        """
        clock[0] += seconds

    monkeypatch.setattr(
        semantic_module.client,
        "time",
        SimpleNamespace(time=lambda: clock[0], sleep=sleep),
    )
    responses = iter(
        [
            _MockResponse(
                200,
                {
                    "offset": 0,
                    "next": 100,
                    "data": [{"citedPaper": {"paperId": f"r{i}"}} for i in range(100)],
                },
            ),
            _MockResponse(
                200, {"offset": 100, "data": [{"citedPaper": {"paperId": "r100"}}]}
            ),
            _MockResponse(200, _paper_payload()),
            _MockResponse(200, [_paper_payload(paper_id="p2")]),
        ]
    )

    def send(*args: object, **kwargs: object) -> _MockResponse:
        """Record a request start and simulate a short response latency.

        :param object args: Transport positional arguments.
        :param object kwargs: Transport keyword arguments.
        :return _MockResponse: Next successful response.
        """
        request_times.append(clock[0])
        clock[0] += 0.01
        return next(responses)

    with SemanticScholarClient(timeout=1, api_key=api_key) as client:
        client._session.get = MagicMock(side_effect=send)
        client._session.post = MagicMock(side_effect=send)
        assert client.get_reference_ids("p1") == [f"r{i}" for i in range(101)]
        assert client._session.get.call_count == 2
        assert "offset=0" in client._session.get.call_args_list[0].kwargs["params"]
        assert "offset=100" in client._session.get.call_args_list[1].kwargs["params"]
        assert client.get_paper("p1") is not None
        assert client.get_papers(["p2"])
        assert request_times == pytest.approx([100.0 + i * interval for i in range(4)])


@pytest.mark.parametrize("batch", [False, True])
def test_real_sdk_transport_keeps_authentication_fields_and_post_body(
    batch: bool,
) -> None:
    """Changing SDK transport must preserve prepared request headers and batch IDs.

    :param bool batch: Whether to inspect a batch POST or reference GET request.
    :return None: Checks real requests preparation with all sending stubbed locally.
    """
    with SemanticScholarClient(timeout=1, api_key="transport-test-key") as client:
        client._rate_limit = MagicMock()
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            [_paper_payload()]
            if batch
            else {"offset": 0, "data": [{"citedPaper": {"paperId": "r1"}}]}
        ).encode()
        client._session.send = MagicMock(return_value=response)
        if batch:
            assert list(client.get_papers(["p1"])) == ["p1"]
        else:
            assert client.get_reference_ids("p1") == ["r1"]
        client._session.send.assert_called_once()
        prepared = client._session.send.call_args.args[0]
        assert prepared.headers["x-api-key"] == "transport-test-key"
        assert "fields=paperId" in prepared.url
        assert prepared.method == ("POST" if batch else "GET")
        assert client._session.send.call_args.kwargs["timeout"] == 1
        if batch:
            assert json.loads(prepared.body) == {"ids": ["p1"]}
        else:
            assert prepared.body is None


@pytest.mark.parametrize("payload", [{}, {"error": "upstream failure"}])
@pytest.mark.parametrize(
    "method", ["get_reference_ids", "get_paper_references", "get_paper_citations"]
)
def test_real_sdk_transport_malformed_relation_cannot_repeat_empty_page(
    payload: dict[str, str], method: str
) -> None:
    """Missing page data must fail before SDK pagination can request it forever.

    :param dict[str, str] payload: Malformed successful response without relation data.
    :param str method: Public SDK relation operation.
    :return None: Checks bounded failure without persisting an apparent empty result.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(
            side_effect=[
                _MockResponse(200, payload),
                AssertionError("malformed page must not be fetched twice"),
            ]
        )
        with (
            patch("time.sleep") as sleep_mock,
            pytest.raises(TypeError, match="malformed relation payload"),
        ):
            getattr(client, method)("p1")
        client._session.get.assert_called_once()
        sleep_mock.assert_not_called()
        assert not s2.disk_cache._reference_cache_path("p1").exists()


def test_sdk_transport_does_not_keep_owning_client_alive() -> None:
    """Retained SDK transport callbacks must not create an owning-client cycle.

    :return None: Checks prompt session closure without forcing garbage collection.
    """
    import weakref

    client = SemanticScholarClient(timeout=1)
    owner = weakref.ref(client)
    session = client._session
    session.close = MagicMock()
    callback = client.client._AsyncSemanticScholar._requester.get_data_async
    del client
    assert owner() is None
    session.close.assert_called_once()
    assert callable(callback)


@pytest.mark.parametrize(
    "requester", [None, SimpleNamespace(), SimpleNamespace(get_data_async=None)]
)
def test_missing_sdk_transport_hook_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch, requester: object
) -> None:
    """An incompatible SDK must fail clearly before allocating an HTTP session.

    :param pytest.MonkeyPatch monkeypatch: Supplies an SDK with changed internals.
    :param object requester: Missing requester or absent/non-callable transport hook.
    :return None: Checks the supported version guidance and no HTTP session leak.
    """
    sdk = SimpleNamespace(_AsyncSemanticScholar=SimpleNamespace(_requester=requester))
    monkeypatch.setattr(
        semantic_module.client, "SemanticScholar", MagicMock(return_value=sdk)
    )
    session_factory = MagicMock()
    monkeypatch.setattr(requests, "Session", session_factory)
    with pytest.raises(
        RuntimeError, match=r"Unsupported Semantic Scholar SDK.*>=0\.8\.0,<0\.13"
    ):
        SemanticScholarClient(timeout=1)
    session_factory.assert_not_called()


@pytest.mark.parametrize(
    "method", ["get_reference_ids", "get_paper_references", "get_papers"]
)
@pytest.mark.parametrize("recovers", [False, True])
def test_requests_json_decoder_with_nonstdlib_base_keeps_retries(
    monkeypatch: pytest.MonkeyPatch, method: str, recovers: bool
) -> None:
    """The requests JSONDecodeError contract also covers a simplejson-style base.

    :param pytest.MonkeyPatch monkeypatch: Replaces the decoder error base contract.
    :param str method: Public reference or batch API operation.
    :param bool recovers: Whether the second request returns valid empty data.
    :return None: Checks operational retries without requiring optional simplejson.
    """

    class AlternateJSONDecodeError(requests.RequestException, ValueError):
        """A requests decoder error that does not inherit stdlib JSONDecodeError."""

    monkeypatch.setattr(
        requests.exceptions, "JSONDecodeError", AlternateJSONDecodeError
    )
    failure = AlternateJSONDecodeError("upstream HTML response")
    assert not isinstance(failure, json.JSONDecodeError)
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        payload = [None] if method == "get_papers" else {"offset": 0, "data": []}
        fetch = MagicMock(
            side_effect=[failure, _MockResponse(200, payload)] if recovers else failure
        )
        client._session.post = fetch
        client._session.get = fetch
        args = ["p1"] if method == "get_papers" else "p1"
        kwargs = {} if method == "get_reference_ids" else {"raise_on_unavailable": True}
        with patch("time.sleep") as sleep_mock:
            if recovers:
                assert getattr(client, method)(args, **kwargs) == (
                    {} if method == "get_papers" else []
                )
            else:
                with pytest.raises(semantic_module.SemanticScholarUnavailableError):
                    getattr(client, method)(args, **kwargs)
        assert fetch.call_count == (2 if recovers else API_CONFIG.max_retries)
        assert sleep_mock.call_count == (1 if recovers else API_CONFIG.max_retries - 1)


def test_real_requests_json_decode_failure_is_retryable() -> None:
    """The installed requests JSON backend must retry an actual non-JSON response.

    :return None: Exercises requests.Response.json with either stdlib or simplejson.
    """
    invalid_response = requests.Response()
    invalid_response.status_code = 200
    invalid_response._content = b"<html>temporarily unavailable</html>"
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(
            side_effect=[
                invalid_response,
                _MockResponse(200, {"offset": 0, "data": []}),
            ]
        )
        with patch("time.sleep") as sleep_mock:
            assert client.get_reference_ids("p1") == []
        assert client._session.get.call_count == 2
        sleep_mock.assert_called_once()


@pytest.mark.parametrize(
    ("paper_id", "expected_path"),
    [
        ("10.18653/v1/N18-3011", "10.18653/v1/N18-3011"),
        ("DOI:10.18653/v1/N18-3011", "10.18653/v1/n18-3011"),
        ("arxiv:math/0301234v1", "arxiv%3Amath/0301234"),
        ("10.1234/a b?c#d", "10.1234/a%20b%3Fc%23d"),
    ],
)
def test_paper_lookup_preserves_slashes_in_prepared_path(
    paper_id: str, expected_path: str
) -> None:
    """Graph IDs retain slashes while URL delimiters remain escaped.

    :param str paper_id: DOI or legacy arXiv identifier to request.
    :param str expected_path: Escaped identifier expected on the wire.
    :return None: Checks the exact requested URL without using the network.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(_paper_payload()).encode()
        client._session.send = MagicMock(return_value=response)
        assert client.get_paper(paper_id).paper_id == "p1"
        client._session.send.assert_called_once()
        prepared = client._session.send.call_args.args[0]
        assert (
            prepared.url.split("?")[0]
            == f"{s2.endpoints.PAPER_BASE_URL}/{expected_path}"
        )


def test_all_null_batch_omits_missing_ids_without_negative_cache() -> None:
    """An authoritative all-null batch needs no individual lookup or persisted miss.

    :return None: Checks one POST, zero GETs, and no cached missing paper records.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._session.post = MagicMock(return_value=_MockResponse(200, [None, None]))
        client._session.get = MagicMock(
            side_effect=AssertionError("unexpected individual lookup")
        )
        assert client.get_papers(["missing-one", "missing-two"]) == {}
        client._session.post.assert_called_once()
        client._session.get.assert_not_called()
        assert not s2.disk_cache._paper_cache_path("missing-one").exists()
        assert not s2.disk_cache._paper_cache_path("missing-two").exists()


@pytest.mark.parametrize("payload", [{}, [], [_paper_payload(), None]])
def test_malformed_batch_responses_fail_without_individual_fallback(
    payload: object,
) -> None:
    """Batch positions cannot be guessed when response cardinality is invalid.

    :param object payload: Malformed array for a one-ID batch request.
    :return None: Checks one deterministic failure and no single-paper retries.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._session.post = MagicMock(return_value=_MockResponse(200, payload))
        client._session.get = MagicMock()
        with (
            patch("time.sleep") as sleep_mock,
            pytest.raises(TypeError, match="batch response"),
        ):
            client.get_papers(["p1"])
        client._session.post.assert_called_once()
        client._session.get.assert_not_called()
        sleep_mock.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        [{"title": "No ID"}],
        [{**_paper_payload(), "citationCount": -1}],
    ],
)
def test_invalid_batch_rows_warn_and_are_skipped(
    payload: object, caplog: pytest.LogCaptureFixture
) -> None:
    """Invalid batch rows should not invalidate valid positional batch results.

    :param object payload: One invalid paper row returned for the requested ID.
    :param pytest.LogCaptureFixture caplog: Captured row-conversion warning.
    :return None: Checks the invalid row is omitted without individual fallback.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._session.post = MagicMock(return_value=_MockResponse(200, payload))
        client._session.get = MagicMock()
        with caplog.at_level(
            logging.WARNING, logger="citemesh.services.semantic_scholar"
        ):
            assert client.get_papers(["p1"]) == {}
        client._session.post.assert_called_once()
        client._session.get.assert_not_called()
    assert "Skipping malformed batch paper for p1." in caplog.text


def test_client_finalization_under_client_lock_does_not_deadlock() -> None:
    """Finalizing an unclosed client while ``_client_lock`` is held must not hang.

    ``get_client()`` constructs the singleton under ``_client_lock``. That
    allocation can trigger a cyclic-GC pass, which runs ``__del__`` (and thus
    ``close()``, which takes the same lock) for any unclosed client caught in a
    reference cycle, on the same thread. A non-reentrant lock deadlocks there,
    so re-entrancy is asserted first: with a plain lock this test would hang
    instead of failing.
    """
    import gc
    import weakref

    lock = semantic_module.client._client_lock
    assert lock.acquire(timeout=5)
    try:
        assert lock.acquire(blocking=False), "_client_lock must be re-entrant"
        lock.release()
    finally:
        lock.release()

    client = SemanticScholarClient(timeout=1)
    client._cycle = client  # reachable only through the cyclic collector
    finalized = weakref.ref(client)
    del client
    with lock:
        gc.collect()
    assert finalized() is None


@pytest.mark.parametrize("endpoint", ["recommendations", "references", "citations"])
def test_discovery_rechecks_ordered_ids_and_fetches_only_missing_metadata(
    endpoint: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Cold, immediate and later collections recheck IDs without repeating metadata.

    :param str endpoint: Ordered discovery endpoint under test.
    :param pytest.LogCaptureFixture caplog: Discovery comparison messages.
    :return None: Verifies additions, removals, order and same-count replacement.
    """
    sequences = [["a", "b"], ["a", "b"], ["b", "a"], ["c", "a"], ["a"]]
    batches: list[list[str]] = []

    def batch(_url: str, **kwargs: Any) -> _MockResponse:
        """Return complete metadata only for explicitly requested IDs.

        :param str _url: Batch endpoint URL.
        :param Any kwargs: Transport options including the requested IDs.
        :return _MockResponse: Positional full metadata response.
        """
        ids = kwargs["json"]["ids"]
        batches.append(ids)
        return _MockResponse(200, [_paper_payload(paper_id=value) for value in ids])

    with caplog.at_level(logging.INFO):
        for ids in sequences:
            with SemanticScholarClient(api_key="") as client:
                client._rate_limit = MagicMock()
                client._session.post = MagicMock(side_effect=batch)
                if endpoint == "recommendations":
                    fetch = MagicMock(
                        return_value=_MockResponse(
                            200,
                            {
                                "recommendedPapers": [
                                    {"paperId": value} for value in ids
                                ]
                            },
                        )
                    )
                    client._session.get = fetch
                    result = client.get_recommended_papers("seed", limit=5)
                    assert fetch.call_args.kwargs["params"]["fields"] == "paperId"
                else:
                    fetch = MagicMock(
                        return_value=[_make_reference_record(value) for value in ids]
                    )
                    setattr(client.client, f"get_paper_{endpoint}", fetch)
                    result = getattr(client, f"get_paper_{endpoint}")("seed", limit=5)
                    assert fetch.call_args.kwargs["fields"] == ["paperId"]
                fetch.assert_called_once()
                assert [paper.paper_id for paper in result] == ids
                key = (
                    endpoint,
                    "seed",
                    5,
                    "recent" if endpoint == "recommendations" else "",
                )
                snapshot = json.loads(
                    s2.disk_cache._discovery_cache_path(key).read_text()
                )
                assert snapshot["paper_ids"] == ids
                assert snapshot["checked_at"]
    assert batches == [["a", "b"], ["c"]]
    assert s2.disk_cache._load_cached_paper("b") is not None
    assert "membership and order unchanged" in caplog.text
    assert "discovery upstream" in caplog.text
    assert "reused 2 cached paper records; fetched 0 missing records" in caplog.text


def test_discovery_scope_limit_and_enrichment_caches_remain_distinct() -> None:
    """In-scope reuse respects limits without truncating full references.

    :return None: Checks normalized seed keys, scope lifetime and refresh controls.
    """
    seed = "arxiv:1706.03762"
    s2.disk_cache._persist_paper(Paper(paper_id="a", title="Cached", year=2020), "a")
    with SemanticScholarClient(api_key="") as client:
        client._rate_limit = MagicMock()
        client._persist_reference_cache_entry(
            s2.disk_cache._reference_cache_path(seed), seed, ["a", "b", "c"]
        )
        references_before = s2.disk_cache._reference_cache_path(seed).read_bytes()
        fetch = MagicMock(return_value=[_make_reference_record("a")])
        client.client.get_paper_references = fetch
        client._session.post = MagicMock(
            side_effect=AssertionError("metadata already cached")
        )
        with client.candidate_operation_scope():
            client.get_paper_references("https://arxiv.org/abs/1706.03762", limit=1)
            client.get_paper_references(seed, limit=1)
            assert fetch.call_count == 1
            client.get_paper_references(seed, limit=2)
            assert fetch.call_count == 2
            assert client.get_reference_ids(seed) == ["a", "b", "c"]
            assert fetch.call_count == 2
            assert (
                s2.disk_cache._reference_cache_path(seed).read_bytes()
                == references_before
            )
        client.get_paper_references(seed, limit=1)
        assert fetch.call_count == 3
        for limit in (1, 2):
            assert s2.disk_cache._discovery_cache_path(
                ("references", seed, limit, "")
            ).exists()
        assert client.get_reference_ids(seed, force_refresh=True) == ["a"]
        assert fetch.call_count == 4


@pytest.mark.parametrize(
    "failure",
    ["check", "fallback", "batch", "malformed", "malformed_batch", "missing_metadata"],
)
def test_failed_discovery_preserves_previous_successful_snapshots(failure: str) -> None:
    """No acquisition failure may publish a new empty or partial discovery snapshot.

    :param str failure: Acquisition stage that fails during the current check.
    :return None: Verifies old snapshot bytes survive and stale IDs are not returned.
    """
    keys = [("recommendations", "seed", 5, pool) for pool in ("recent", "all-cs")]
    for key in keys:
        s2.disk_cache._persist_discovery(key, ["old"])
    previous = {
        key: s2.disk_cache._discovery_cache_path(key).read_bytes() for key in keys
    }
    unavailable = _MockResponse(503, headers={"Retry-After": "120"})
    first = _MockResponse(200, {"recommendedPapers": [{"paperId": "new"}]})
    if failure == "check":
        responses = [unavailable]
    elif failure == "fallback":
        responses = [_MockResponse(200, {"recommendedPapers": []}), unavailable]
    elif failure == "malformed":
        responses = [
            _MockResponse(200, {"recommendedPapers": [{"paperId": "new"}, {}]})
        ]
    else:
        responses = [first]
    with SemanticScholarClient(api_key="", retry_budget_seconds=5) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(side_effect=responses)
        client._session.post = MagicMock(
            return_value=(
                _MockResponse(200, [None])
                if failure == "missing_metadata"
                else _MockResponse(200, [{"title": "Missing ID"}])
                if failure == "malformed_batch"
                else unavailable
            )
        )
        with (
            patch("time.sleep") as sleep_mock,
            pytest.raises((SemanticScholarUnavailableError, TypeError)),
        ):
            client.get_recommended_papers("seed", limit=5, raise_on_unavailable=True)
        sleep_mock.assert_not_called()
    for key in keys:
        assert s2.disk_cache._discovery_cache_path(key).read_bytes() == previous[key]


def test_successful_empty_recommendations_check_both_pools_every_collection() -> None:
    """Only successful recent and all-cs checks establish empty recommendations.

    :return None: Checks successful empty snapshots and collection-local reuse.
    """
    with SemanticScholarClient(api_key="") as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(
            return_value=_MockResponse(200, {"recommendedPapers": []})
        )
        client._session.post = MagicMock(
            side_effect=AssertionError("no metadata needed")
        )
        with client.candidate_operation_scope():
            assert client.get_recommended_papers("seed", limit=5) == []
            assert client.get_recommended_papers("seed", limit=5) == []
            assert client._session.get.call_count == 2
        assert client.get_recommended_papers("seed", limit=5) == []
        assert client._session.get.call_count == 4
        assert [
            call.kwargs["params"].get("from", "recent")
            for call in client._session.get.call_args_list
        ] == ["recent", "all-cs", "recent", "all-cs"]
        for pool in ("recent", "all-cs"):
            snapshot = json.loads(
                s2.disk_cache._discovery_cache_path(
                    ("recommendations", "seed", 5, pool)
                ).read_text()
            )
            assert snapshot["paper_ids"] == []


def test_failed_discovery_page_preserves_snapshot_and_enrichment() -> None:
    """An outage after a successful relation page cannot replace either cache.

    :return None: Verifies incomplete IDs never reach metadata hydration.
    """
    key = ("references", "seed", 5, "")
    s2.disk_cache._persist_discovery(key, ["old"])
    previous = s2.disk_cache._discovery_cache_path(key).read_bytes()

    def pages(*_args: Any, **_kwargs: Any) -> Any:
        """Yield one page, then a retryable outage exceeding the recovery allowance.

        :param Any _args: SDK positional arguments.
        :param Any _kwargs: SDK keyword arguments.
        :return Any: Generator matching the SDK pagination interface.
        """
        yield _make_reference_record("new")
        raise s2.errors._RetryableRequestError("HTTP 503 page two", retry_after=120)

    with SemanticScholarClient(api_key="", retry_budget_seconds=5) as client:
        client.client.get_paper_references = MagicMock(side_effect=pages)
        client._session.post = MagicMock(
            side_effect=AssertionError("partial IDs must not hydrate")
        )
        with (
            patch("time.sleep") as sleep_mock,
            pytest.raises(SemanticScholarUnavailableError),
        ):
            client.get_paper_references("seed", limit=5, raise_on_unavailable=True)
        sleep_mock.assert_not_called()
        client._session.post.assert_not_called()
    assert s2.disk_cache._discovery_cache_path(key).read_bytes() == previous
    assert not s2.disk_cache._reference_cache_path("seed").exists()


@pytest.mark.parametrize("relation", ["references", "citations"])
def test_missing_relation_seed_is_not_successful_empty_discovery(relation: str) -> None:
    """Strict revalidation cannot accept a 404 as evidence of no related papers.

    :param str relation: Seeded relation endpoint to check.
    :return None: Verifies the old snapshot is retained on a missing-seed response.
    """
    key = (relation, "seed", 5, "")
    s2.disk_cache._persist_discovery(key, ["old"])
    previous = s2.disk_cache._discovery_cache_path(key).read_bytes()
    with SemanticScholarClient(api_key="") as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(return_value=_MockResponse(404))
        with pytest.raises(SemanticScholarUnavailableError, match="current.*HTTP 404"):
            getattr(client, f"get_paper_{relation}")(
                "seed", limit=5, raise_on_unavailable=True
            )
        client._session.get.assert_called_once()
    assert s2.disk_cache._discovery_cache_path(key).read_bytes() == previous


def test_refresh_paper_cache_also_refreshes_discovery_metadata() -> None:
    """Explicit metadata refresh bypasses records behind freshly checked IDs.

    :return None: Checks the metadata batch and updated persisted record.
    """
    s2.disk_cache._persist_paper(Paper(paper_id="a", title="Old", year=2020), "a")
    with SemanticScholarClient(api_key="", refresh_paper_cache=True) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(
            return_value=_MockResponse(200, {"recommendedPapers": [{"paperId": "a"}]})
        )
        client._session.post = MagicMock(
            return_value=_MockResponse(
                200, [_paper_payload(paper_id="a", title="Updated")]
            )
        )
        result = client.get_recommended_papers("seed")
        assert result[0].title == "Updated"
        client._session.post.assert_called_once()
        assert s2.disk_cache._load_cached_paper("a").title == "Updated"


def test_anonymous_recovery_budget_stops_before_a_long_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anonymous 429 recovery must not sleep beyond the shared budget."""
    clock = [0.0]
    monkeypatch.setattr(semantic_module.client, "monotonic", lambda: clock[0])
    with SemanticScholarClient(timeout=30, retry_budget_seconds=5) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(
            return_value=_MockResponse(429, headers={"Retry-After": "120"})
        )
        with (
            patch("time.sleep") as sleep_mock,
            pytest.raises(
                SemanticScholarUnavailableError,
                match=r"after 1 attempts and 0\.0s recovery time \(next wait exceeds remaining recovery budget",
            ),
        ):
            client.get_paper("budgeted", raise_on_unavailable=True)
    client._session.get.assert_called_once()
    sleep_mock.assert_not_called()


def test_recovery_budget_is_shared_and_resets_for_a_new_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spent collection budget blocks later domains but not a new collection."""
    clock = [0.0]
    monkeypatch.setattr(semantic_module.client, "monotonic", lambda: clock[0])

    def sleep(seconds: float) -> None:
        """Advance the monotonic test clock by a retry wait."""
        clock[0] += seconds

    call_timeouts: list[float] = []

    def unavailable(*_args: object, **kwargs: Any) -> _MockResponse:
        """Return a failed response after consuming request time."""
        call_timeouts.append(kwargs["timeout"])
        clock[0] += 2.0
        return _MockResponse(503)

    with SemanticScholarClient(timeout=30, retry_budget_seconds=5) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(side_effect=unavailable)
        with (
            patch("time.sleep", side_effect=sleep) as sleep_mock,
            patch.object(semantic_module.retry.random, "uniform", return_value=2.0),
            client.candidate_operation_scope(),
        ):
            assert (
                client._request_json(
                    "https://example.test/first",
                    {},
                    failure_domain=semantic_module.errors._FailureDomain.SEARCH,
                )
                is None
            )
            with pytest.raises(semantic_module.errors._CandidateOperationSkippedError):
                client._request_json(
                    "https://example.test/second",
                    {},
                    failure_domain=semantic_module.errors._FailureDomain.REFERENCES,
                    raise_on_unavailable=True,
                )
        assert client._session.get.call_count == 2
        assert call_timeouts == [30, 1]
        sleep_mock.assert_called_once_with(2.0)

        client._session.get = MagicMock(
            return_value=_MockResponse(200, _paper_payload(paper_id="fresh"))
        )
        assert client.get_paper("fresh", raise_on_unavailable=True).paper_id == "fresh"
        client._session.get.assert_called_once()


def test_recovery_attempt_success_is_charged_but_slow_healthy_work_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery time covers retried requests while a slow initial success remains valid."""
    clock = [0.0]
    monkeypatch.setattr(semantic_module.client, "monotonic", lambda: clock[0])

    def sleep(seconds: float) -> None:
        """Advance the monotonic test clock by the requested retry delay."""
        clock[0] += seconds

    response_timeouts: list[float] = []

    def response(*_args: object, **kwargs: Any) -> _MockResponse:
        """Fail once, then succeed while recording the shortened retry timeout."""
        response_timeouts.append(kwargs["timeout"])
        clock[0] += 1.0
        if len(response_timeouts) == 1:
            return _MockResponse(503)
        return _MockResponse(200, _paper_payload(paper_id="recovered"))

    with SemanticScholarClient(timeout=30, retry_budget_seconds=5) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(side_effect=response)
        with (
            patch("time.sleep", side_effect=sleep),
            patch.object(semantic_module.retry.random, "uniform", return_value=2.0),
        ):
            assert (
                client.get_paper("recovered", raise_on_unavailable=True).paper_id
                == "recovered"
            )
        assert response_timeouts == [30, 2]

    def slow_success(*_args: object, **kwargs: Any) -> _MockResponse:
        """Simulate a healthy first response that takes longer than the budget."""
        clock[0] += 120.0
        return _MockResponse(200, _paper_payload(paper_id="slow"))

    with SemanticScholarClient(timeout=30, retry_budget_seconds=5) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(side_effect=slow_success)
        assert client.get_paper("slow", raise_on_unavailable=True).paper_id == "slow"
        assert client._session.get.call_args.kwargs["timeout"] == 30


def test_retry_budget_defaults_and_zero_opt_out() -> None:
    """Anonymous access defaults to 90 seconds; keys and explicit zero are uncapped."""
    assert SemanticScholarClient(api_key="").retry_budget_seconds == 90.0
    assert SemanticScholarClient(api_key="test-key").retry_budget_seconds == 0.0
    assert (
        SemanticScholarClient(api_key="", retry_budget_seconds=0).retry_budget_seconds
        == 0.0
    )
    with pytest.raises(ValueError, match="non-negative"):
        SemanticScholarClient(api_key="", retry_budget_seconds=-1)


def test_malformed_json_request_time_consumes_recovery_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An HTTP 200 with broken JSON is failed work, not an excluded healthy request.

    :param pytest.MonkeyPatch monkeypatch: Fixture installing a monotonic clock.
    :return None: Verifies failed response time stops recovery before another request.
    """
    clock = [0.0]
    monkeypatch.setattr(semantic_module.client, "monotonic", lambda: clock[0])

    def response(*_args: Any, **_kwargs: Any) -> _BrokenJsonResponse:
        """Spend the allowance receiving a body that cannot be decoded.

        :param Any _args: Ignored transport arguments.
        :param Any _kwargs: Ignored transport keyword arguments.
        :return _BrokenJsonResponse: HTTP-success response with invalid JSON.
        """
        clock[0] += 5
        return _BrokenJsonResponse(200)

    with SemanticScholarClient(api_key="", retry_budget_seconds=5) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(side_effect=response)
        with (
            patch("time.sleep") as sleep_mock,
            pytest.raises(SemanticScholarUnavailableError, match="5.0s recovery time"),
        ):
            client.get_paper("seed", raise_on_unavailable=True)
        client._session.get.assert_called_once()
        sleep_mock.assert_not_called()


def test_recovery_budget_checks_each_sdk_transport_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paginated SDK retry cannot issue another request after its budget expires."""
    clock = [5.0]
    monkeypatch.setattr(semantic_module.client, "monotonic", lambda: clock[0])
    with SemanticScholarClient(timeout=30, retry_budget_seconds=5) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock()
        with client.candidate_operation_scope():
            state = client._candidate_operation.state
            state.active_recovery_request = True
            state.current_attempt = 2
            state.recovery_seconds = 5.0
            state.last_recovery_error = requests.ConnectionError("offline")
            with pytest.raises(semantic_module.errors._RetryExhaustedError):
                client._request_json_once(
                    "https://example.test/page",
                    {},
                    context="testing pagination",
                )
        client._session.get.assert_not_called()


def test_permanent_http_error_stays_immediate_after_a_recovery_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later 400 must remain a request error rather than an outage summary."""
    clock = [0.0]
    monkeypatch.setattr(semantic_module.client, "monotonic", lambda: clock[0])

    def sleep(seconds: float) -> None:
        """Advance the recovery clock without pausing the test."""
        clock[0] += seconds

    with SemanticScholarClient(timeout=30, retry_budget_seconds=5) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(
            side_effect=[_MockResponse(503), _MockResponse(400)]
        )
        with (
            patch("time.sleep", side_effect=sleep),
            patch.object(semantic_module.retry.random, "uniform", return_value=1.0),
            pytest.raises(SemanticScholarRequestError, match="HTTP 400"),
        ):
            client.get_paper("permanent", raise_on_unavailable=True)
        assert client._session.get.call_count == 2


def test_sdk_retry_recomputes_timeout_for_each_pagination_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Later SDK pages receive only the recovery time remaining at their request."""
    clock = [0.0]
    monkeypatch.setattr(semantic_module.client, "monotonic", lambda: clock[0])
    timeouts: list[float] = []
    responses = iter(
        [
            _MockResponse(503),
            _MockResponse(
                200,
                {
                    "offset": 0,
                    "next": 100,
                    "data": [{"citedPaper": {"paperId": "page-one"}}],
                },
            ),
            _MockResponse(
                200,
                {
                    "offset": 100,
                    "data": [{"citedPaper": {"paperId": "page-two"}}],
                },
            ),
        ]
    )

    def respond(*_args: object, **kwargs: Any) -> _MockResponse:
        """Advance the test clock for one failed or paginated SDK request."""
        timeouts.append(kwargs["timeout"])
        clock[0] += (1.0, 2.0, 1.0)[len(timeouts) - 1]
        return next(responses)

    def sleep(seconds: float) -> None:
        """Advance the test clock for the retry delay."""
        clock[0] += seconds

    with SemanticScholarClient(timeout=30, retry_budget_seconds=5) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(side_effect=respond)
        with (
            patch("time.sleep", side_effect=sleep),
            patch.object(semantic_module.retry.random, "uniform", return_value=1.0),
        ):
            assert client.get_reference_ids("retry-pages") == ["page-one", "page-two"]
    assert timeouts == [30, 3, 1]


def test_spent_recovery_budget_blocks_later_network_but_not_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful retry that spends the budget stops later uncached requests."""
    clock = [0.0]
    monkeypatch.setattr(semantic_module.client, "monotonic", lambda: clock[0])
    responses = iter(
        [
            _MockResponse(503),
            _MockResponse(200, _paper_payload(paper_id="budget-recovered")),
        ]
    )

    def respond(*_args: object, **_kwargs: Any) -> _MockResponse:
        """Spend one second on each response."""
        clock[0] += 1.0
        return next(responses)

    def sleep(seconds: float) -> None:
        """Spend one second on the deterministic retry wait."""
        clock[0] += seconds

    with SemanticScholarClient(timeout=30, retry_budget_seconds=3) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(side_effect=respond)
        with (
            patch("time.sleep", side_effect=sleep),
            patch.object(semantic_module.retry.random, "uniform", return_value=1.0),
            client.candidate_operation_scope(),
        ):
            assert (
                client.get_paper("budget-recovered", raise_on_unavailable=True).paper_id
                == "budget-recovered"
            )
            client._session.get = MagicMock(
                side_effect=AssertionError("budget-spent request must not run")
            )
            assert client.get_paper("budget-recovered") is not None
            with pytest.raises(
                semantic_module.errors._CandidateOperationSkippedError,
                match="earlier Semantic Scholar outage",
            ):
                client.get_paper("budget-later", raise_on_unavailable=True)
        client._session.get.assert_not_called()


def test_oversized_retry_after_reports_its_wait_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server cooldown above the allowance identifies the next wait as the stop."""
    clock = [0.0]
    monkeypatch.setattr(semantic_module.client, "monotonic", lambda: clock[0])
    with SemanticScholarClient(timeout=30, retry_budget_seconds=5) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(
            return_value=_MockResponse(429, headers={"Retry-After": "120"})
        )
        with pytest.raises(
            SemanticScholarUnavailableError,
            match="next wait exceeds remaining recovery budget",
        ):
            client.get_paper("oversized-wait", raise_on_unavailable=True)
        client._session.get.assert_called_once()


@pytest.mark.parametrize("budget", [float("inf"), float("-inf"), float("nan")])
def test_retry_budget_rejects_nonfinite_values(budget: float) -> None:
    """A recovery budget must be a finite non-negative duration."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        SemanticScholarClient(api_key="", retry_budget_seconds=budget)


def test_oversized_retry_after_stops_every_discovery_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized cooldown blocks later uncached calls in the same collection.

    :param pytest.MonkeyPatch monkeypatch: Replaces the monotonic clock.
    :return None: Checks the global recovery breaker after an oversized wait.
    """
    clock = [0.0]
    monkeypatch.setattr(semantic_module.client, "monotonic", lambda: clock[0])
    with SemanticScholarClient(timeout=30, retry_budget_seconds=5) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(
            return_value=_MockResponse(429, headers={"Retry-After": "120"})
        )
        with client.candidate_operation_scope():
            assert (
                client._request_json(
                    "https://example.test/first",
                    {},
                    failure_domain=semantic_module.errors._FailureDomain.SEARCH,
                )
                is None
            )
            with pytest.raises(
                semantic_module.errors._CandidateOperationSkippedError,
                match="earlier Semantic Scholar outage",
            ):
                client._request_json(
                    "https://example.test/second",
                    {},
                    failure_domain=semantic_module.errors._FailureDomain.REFERENCES,
                    raise_on_unavailable=True,
                )
        client._session.get.assert_called_once()


def test_budget_preflight_does_not_count_an_unstarted_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry blocked before its request reports only requests that started.

    :param pytest.MonkeyPatch monkeypatch: Replaces the monotonic clock.
    :return None: Checks request counts and the no-network retry preflight.
    """
    clock = [0.0]
    monkeypatch.setattr(semantic_module.client, "monotonic", lambda: clock[0])

    def respond(*_args: object, **_kwargs: Any) -> _MockResponse:
        """Consume one second before returning a retryable response.

        :param object _args: Transport positional arguments.
        :param Any _kwargs: Transport keyword arguments.
        :return _MockResponse: Retryable response with a two-second cooldown.
        """
        clock[0] += 1.0
        return _MockResponse(503, headers={"Retry-After": "2"})

    def sleep(seconds: float) -> None:
        """Advance the fake clock by the scheduled backoff.

        :param float seconds: Backoff duration.
        :return None: Updates the fake clock.
        """
        clock[0] += seconds

    with SemanticScholarClient(timeout=30, retry_budget_seconds=3) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(side_effect=respond)
        with (
            patch("time.sleep", side_effect=sleep),
            patch.object(semantic_module.retry.random, "uniform", return_value=0.0),
            pytest.raises(
                SemanticScholarUnavailableError,
                match=r"after 1 attempts and 3\.0s recovery time",
            ),
        ):
            client.get_paper("no-unstarted-attempt", raise_on_unavailable=True)
        client._session.get.assert_called_once()


def test_sdk_budget_stop_is_terminal_and_reports_active_recovery_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spent retry page stops without a new SDK retry and reports its elapsed time.

    :param pytest.MonkeyPatch monkeypatch: Replaces the monotonic clock.
    :return None: Checks terminal handling and active-retry diagnostics.
    """
    clock = [0.0]
    monkeypatch.setattr(semantic_module.client, "monotonic", lambda: clock[0])
    responses = iter(
        [
            _MockResponse(503, headers={"Retry-After": "1"}),
            _MockResponse(
                200,
                {
                    "offset": 0,
                    "next": 100,
                    "data": [{"citedPaper": {"paperId": "first-page"}}],
                },
            ),
        ]
    )

    def respond(*_args: object, **_kwargs: Any) -> _MockResponse:
        """Spend one second on the first call and four on the retry page.

        :param object _args: Transport positional arguments.
        :param Any _kwargs: Transport keyword arguments.
        :return _MockResponse: Next mocked transport response.
        """
        clock[0] += 1.0 if clock[0] == 0.0 else 4.0
        return next(responses)

    def sleep(seconds: float) -> None:
        """Advance the fake clock by the retry delay.

        :param float seconds: Backoff duration.
        :return None: Updates the fake clock.
        """
        clock[0] += seconds

    with SemanticScholarClient(timeout=30, retry_budget_seconds=5) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(side_effect=respond)
        with (
            patch("time.sleep", side_effect=sleep),
            patch.object(semantic_module.retry.random, "uniform", return_value=0.0),
            pytest.raises(
                SemanticScholarUnavailableError,
                match=r"after 2 attempts and 6\.0s recovery time",
            ),
        ):
            client.get_reference_ids("terminal-sdk-budget")
        assert client._session.get.call_count == 2


def test_first_attempt_successful_pages_do_not_consume_recovery_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the failed initial request is charged before a retry begins.

    :param pytest.MonkeyPatch monkeypatch: Replaces the monotonic clock.
    :return None: Checks initial healthy pages are excluded and retry pages are charged.
    """
    clock = [0.0]
    monkeypatch.setattr(semantic_module.client, "monotonic", lambda: clock[0])
    responses = iter(
        [
            _MockResponse(
                200,
                {
                    "offset": 0,
                    "next": 100,
                    "data": [{"citedPaper": {"paperId": "initial-page"}}],
                },
            ),
            _MockResponse(503, headers={"Retry-After": "1"}),
            _MockResponse(
                200,
                {
                    "offset": 0,
                    "next": 100,
                    "data": [{"citedPaper": {"paperId": "retry-page"}}],
                },
            ),
            _MockResponse(503, headers={"Retry-After": "120"}),
        ]
    )

    def respond(*_args: object, **_kwargs: Any) -> _MockResponse:
        """Spend 120 seconds on each healthy page and one on each failure.

        :param object _args: Transport positional arguments.
        :param Any _kwargs: Transport keyword arguments.
        :return _MockResponse: Next mocked transport response.
        """
        response = next(responses)
        clock[0] += 120.0 if response.status_code == 200 else 1.0
        return response

    def sleep(seconds: float) -> None:
        """Advance the fake clock by the requested retry delay.

        :param float seconds: Backoff duration.
        :return None: Updates the fake clock.
        """
        clock[0] += seconds

    with SemanticScholarClient(timeout=30, retry_budget_seconds=150) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(side_effect=respond)
        with (
            patch("time.sleep", side_effect=sleep),
            patch.object(semantic_module.retry.random, "uniform", return_value=0.0),
            pytest.raises(
                SemanticScholarUnavailableError,
                match=r"after 2 attempts and 123\.0s recovery time",
            ),
        ):
            client.get_reference_ids("initial-pages")
        assert client._session.get.call_count == 4
