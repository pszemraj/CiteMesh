"""Tests for the Semantic Scholar API client."""

import importlib
import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

from citemesh.core import API_CONFIG, Paper
from citemesh.services import semantic_scholar as semantic_module
from citemesh.services.semantic_scholar import (
    SemanticScholarClient,
    get_client,
    normalize_paper_id,
    reset_client,
)
from tests.conftest import get_paper_id_normalization_cases


class _MockResponse:
    """Minimal requests response object for client tests."""

    def __init__(
        self,
        status_code: int,
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Create a minimal mock requests response.

        :param int status_code: HTTP status code.
        :param Optional[Dict[str, Any]] payload: Optional JSON payload for this response.
        :param Optional[Dict[str, str]] headers: Optional headers mapping.
        """
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        """Raise HTTPError when payload status_code indicates failure."""
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> dict:
        """Return mocked JSON body.

        :return dict: Configured mock payload.
        """
        return self._payload


class _BrokenJsonResponse(_MockResponse):
    """Response object whose JSON body cannot be decoded."""

    def json(self) -> dict:
        """Raise decode error to emulate malformed upstream response body."""
        raise ValueError("Malformed JSON payload")


class _FakeRequestsSession:
    """Minimal request session with close tracking."""

    def __init__(self) -> None:
        """Create an in-memory session stub."""
        self.headers: dict[str, str] = {}
        self.closed = False

    def close(self) -> None:
        """Mark the fake session as closed."""
        self.closed = True


def _build_fake_semantic_scholar_api(
    created: Optional[list[object]] = None,
) -> type:
    """Create a SemanticScholar API stub class backed by fake sessions.

    :param Optional[list[object]] created: Optional sink receiving each created API
        instance.
    :return type: Fake SemanticScholar-like class for monkeypatching.
    """

    class _FakeApi:
        """Minimal SemanticScholar wrapper stub."""

        def __init__(self, *_, **__) -> None:
            """Create fake API instance and track it when requested."""
            self.session = _FakeRequestsSession()
            self.closed = False
            if created is not None:
                created.append(self)

        def close(self) -> None:
            """Mark fake API instance as closed."""
            self.closed = True

    return _FakeApi


def _paper_payload(
    *,
    paper_id: str = "p1",
    title: str = "Paper",
    year: int = 2020,
    abstract: str = "Abstract",
) -> dict:
    """Build a minimal paper payload for API tests.

    :param str paper_id: Paper identifier.
    :param str title: Paper title.
    :param int year: Publication year.
    :param str abstract: Paper abstract.
    :return dict: Minimal paper payload.
    """
    return {
        "paperId": paper_id,
        "title": title,
        "year": year,
        "abstract": abstract,
        "citationCount": 10,
        "fieldsOfStudy": [],
        "authors": [],
    }


def test_retries_on_rate_limit_from_direct_endpoint() -> None:
    """Search should retry after Retry-After for 429 responses."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock()
    client._session.get.side_effect = [
        _MockResponse(
            status_code=429,
            headers={"Retry-After": "2"},
        ),
        _MockResponse(
            status_code=200,
            payload={"data": [_paper_payload(paper_id="x1", title="A", year=2020)]},
        ),
    ]

    with patch("citemesh.services.semantic_scholar.time.sleep") as sleep_mock:
        results = client.search_papers("attention")

    assert sleep_mock.call_count == 1
    assert sleep_mock.call_args_list[0].args[0] == 2.0
    assert len(results) == 1
    assert results[0] == Paper(paper_id="x1", title="A", year=2020, abstract="Abstract")


def test_retries_on_transient_api_failure() -> None:
    """get_paper should retry failed requests and eventually return paper."""
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
    client.client.get_paper = MagicMock(side_effect=[Exception("temporary"), api_paper])

    with patch("citemesh.services.semantic_scholar.time.sleep") as sleep_mock:
        result = client.get_paper("seed")

    assert isinstance(result, Paper)
    assert result.paper_id == "seed"
    assert sleep_mock.call_count == 1
    assert sleep_mock.call_args_list[0].args[0] == API_CONFIG.retry_delay


def test_retries_when_direct_endpoint_returns_malformed_json() -> None:
    """Direct endpoint helper should retry when response body is not valid JSON."""
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

    with patch("citemesh.services.semantic_scholar.time.sleep") as sleep_mock:
        results = client.search_papers("transformer")

    assert sleep_mock.call_count == 1
    assert sleep_mock.call_args_list[0].args[0] == API_CONFIG.retry_delay
    assert len(results) == 1
    assert results[0].paper_id == "x2"


def test_get_paper_returns_none_after_retries() -> None:
    """get_paper should return None after all retry attempts are exhausted."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client.client.get_paper = MagicMock(side_effect=Exception("down"))

    with patch("citemesh.services.semantic_scholar.time.sleep") as sleep_mock:
        result = client.get_paper("seed")

    assert result is None
    assert sleep_mock.call_count == API_CONFIG.max_retries - 1


def test_retries_on_rate_limit_for_citations() -> None:
    """citation lookup should back off using Retry-After on 429 responses."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None

    response = requests.Response()
    response.status_code = 429
    response.headers = {"Retry-After": "2"}

    client.client.get_paper_citations = MagicMock(
        side_effect=[requests.HTTPError(response=response), []]
    )

    with patch("citemesh.services.semantic_scholar.time.sleep") as sleep_mock:
        result = client.get_paper_citations("seed", limit=5)

    assert sleep_mock.call_count == 1
    assert sleep_mock.call_args_list[0].args[0] == 2.0
    assert result == []


def test_retries_on_rate_limit_for_references() -> None:
    """reference lookup should back off using Retry-After on 429 responses."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None

    response = requests.Response()
    response.status_code = 429
    response.headers = {"Retry-After": "3"}

    client.client.get_paper_references = MagicMock(
        side_effect=[requests.HTTPError(response=response), []]
    )

    with patch("citemesh.services.semantic_scholar.time.sleep") as sleep_mock:
        result = client.get_paper_references("seed", limit=5)

    assert sleep_mock.call_count == 1
    assert sleep_mock.call_args_list[0].args[0] == 3.0
    assert result == []


@pytest.mark.parametrize("raw_id, expected", get_paper_id_normalization_cases())
def test_normalize_paper_id_urls(raw_id: str, expected: str) -> None:
    """URL and prefixed identifiers should normalize to API-friendly IDs.

    :param str raw_id: Raw identifier input.
    :param str expected: Expected canonicalized output.
    """
    assert normalize_paper_id(raw_id) == expected


@pytest.mark.parametrize(
    ("raw_id", "expected"),
    [
        ("doi:10.1145/3133956.3134029", "10.1145/3133956.3134029"),
        ("https://doi.org:443/10.1145/3133956.3134029", "10.1145/3133956.3134029"),
        ("doi.org/10.1145/3133956.3134029", "10.1145/3133956.3134029"),
        ("dx.doi.org/10.1145/3133956.3134029", "10.1145/3133956.3134029"),
    ],
)
def test_normalize_paper_id_handles_doi_prefixes_and_ports(
    raw_id: str, expected: str
) -> None:
    """DOI forms with prefixes/ports should normalize to bare DOI IDs."""
    assert normalize_paper_id(raw_id) == expected


@pytest.mark.parametrize(
    ("raw_id", "expected"),
    [
        (
            "https://notdoi.org/10.1145/3133956.3134029",
            "https://notdoi.org/10.1145/3133956.3134029",
        ),
        ("https://fooarxiv.org/abs/1706.03762", "https://fooarxiv.org/abs/1706.03762"),
    ],
)
def test_normalize_paper_id_does_not_match_non_domains(
    raw_id: str, expected: str
) -> None:
    """Host matching should only accept exact domains or proper subdomains."""
    assert normalize_paper_id(raw_id) == expected


def test_normalize_paper_id_rejects_non_string_input() -> None:
    """normalize_paper_id should fail with a clear error for non-string input."""
    with pytest.raises(ValueError, match="Invalid paper ID"):
        normalize_paper_id(None)  # type: ignore[arg-type]


def test_get_paper_normalizes_arxiv_url_before_api_call() -> None:
    """get_paper should transform arXiv URLs before querying Semantic Scholar."""
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
    client.client.get_paper = MagicMock(return_value=api_paper)

    result = client.get_paper("https://arxiv.org/abs/2508.14040")

    assert isinstance(result, Paper)
    assert result.paper_id == "seed"
    assert client.client.get_paper.call_args.args[0] == "arxiv:2508.14040"


def test_convert_recommendation_with_missing_fields_is_robust() -> None:
    """_convert_recommendation should tolerate sparse/missing payload keys."""
    client = SemanticScholarClient(timeout=1)
    payload = {
        "paperId": "p1",
        "title": "",
        "year": None,
        "abstract": "",
        "citationCount": 0,
        "authors": [{"name": ""}, {}],
        "fieldsOfStudy": ["cs.AI", "cs.LG"],
    }

    paper = client._convert_recommendation(payload)

    assert paper is not None
    assert paper.paper_id == "p1"
    assert paper.title == "Unknown"
    assert paper.year is None
    assert paper.abstract == ""
    assert paper.authors == []
    assert paper.categories == ["cs.AI", "cs.LG"]


def test_reset_client_recreates_singleton() -> None:
    """reset_client should clear and recreate the shared Semantic Scholar client."""
    first_client = get_client()
    reset_client()
    second_client = get_client()

    assert first_client is not second_client


def test_semantic_module_reload_does_not_mutate_http_logger_levels() -> None:
    """Importing/reloading service module should not change global HTTP logger levels."""
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


def test_close_and_reset_client_close_prior_session() -> None:
    """`reset_client()` should close both request session and API wrapper."""
    created: list[object] = []

    previous_session = semantic_module.requests.Session
    previous_client = semantic_module.SemanticScholar
    try:
        semantic_module.requests.Session = _FakeRequestsSession
        semantic_module.SemanticScholar = _build_fake_semantic_scholar_api(created)

        reset_client()
        first = get_client()
        reset_client()

        assert first._session.closed is True
        assert first.client.closed is True
        assert first._closed is True
        assert len(created) == 1

        second = get_client()
        assert second is not first
        assert len(created) == 2
    finally:
        semantic_module.requests.Session = previous_session
        semantic_module.SemanticScholar = previous_client


def test_client_context_manager_closes_sessions() -> None:
    """Client context manager should close all owned sessions."""

    previous_session = semantic_module.requests.Session
    previous_client = semantic_module.SemanticScholar
    try:
        semantic_module.requests.Session = _FakeRequestsSession
        semantic_module.SemanticScholar = _build_fake_semantic_scholar_api()

        with semantic_module.SemanticScholarClient(timeout=1) as client:
            assert client._closed is False
            api_session = client.client.session
            request_session = client._session

        assert client._closed is True
        assert request_session.closed is True
        assert api_session.closed is True
        assert client.client.closed is True
    finally:
        semantic_module.requests.Session = previous_session
        semantic_module.SemanticScholar = previous_client


def test_get_recommended_papers_include_references_adds_field() -> None:
    """Recommendation requests should include references field when requested."""
    client = SemanticScholarClient(timeout=1)
    client._request_json = MagicMock(return_value={"recommendedPapers": []})

    client.get_recommended_papers("seed", limit=5, include_references=True)

    params = client._request_json.call_args.args[1]
    fields = params["fields"].split(",")
    assert "references" in fields


def test_recommendation_and_search_limits_validate_positive() -> None:
    """Direct endpoint helpers should reject non-positive limits."""
    client = SemanticScholarClient(timeout=1)

    with pytest.raises(ValueError, match="limit must be at least 1"):
        client.get_recommended_papers("seed", limit=0)

    with pytest.raises(ValueError, match="limit must be at least 1"):
        client.search_papers("attention", limit=0)


def test_search_rejects_blank_query() -> None:
    """Search should reject empty or whitespace-only query strings."""
    client = SemanticScholarClient(timeout=1)

    with pytest.raises(ValueError, match="query must not be empty"):
        client.search_papers("   ", limit=1)


def test_search_rejects_non_string_query() -> None:
    """Search should reject non-string query inputs with a clear message."""
    client = SemanticScholarClient(timeout=1)

    with pytest.raises(ValueError, match="query must be a string"):
        client.search_papers(123, limit=1)  # type: ignore[arg-type]


def test_relation_limit_validation_for_citations_and_references() -> None:
    """Citation/reference helpers should allow zero to disable fetches."""
    client = SemanticScholarClient(timeout=1)
    client.client.get_paper_citations = MagicMock(return_value=[])
    client.client.get_paper_references = MagicMock(return_value=[])

    assert client.get_paper_citations("seed", limit=0) == []
    assert client.get_paper_references("seed", limit=0) == []
    client.client.get_paper_citations.assert_not_called()
    client.client.get_paper_references.assert_not_called()


def test_direct_endpoint_limit_validation_rejects_non_integer() -> None:
    """Direct endpoint helpers should reject non-integer limit values."""
    client = SemanticScholarClient(timeout=1)

    with pytest.raises(ValueError, match="limit must be an integer"):
        client.get_recommended_papers("seed", limit=True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw_id", "expected_suffix"),
    [
        ("10.1145/3133956.3134029", "10.1145%2F3133956.3134029"),
        ("arxiv:math/0301234v1", "arxiv%3Amath%2F0301234"),
    ],
)
def test_get_recommended_papers_url_encodes_paper_id_path_segment(
    raw_id: str, expected_suffix: str
) -> None:
    """Recommendation URL should treat paper_id as one encoded path token."""
    client = SemanticScholarClient(timeout=1)
    client._request_json = MagicMock(return_value={"recommendedPapers": []})

    client.get_recommended_papers(raw_id, limit=1)

    url = client._request_json.call_args.args[0]
    assert url.endswith(expected_suffix)
