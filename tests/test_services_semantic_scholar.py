"""Tests for the Semantic Scholar API client."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from citemesh.core import API_CONFIG
from citemesh.models import Paper
from citemesh.services.semantic_scholar import SemanticScholarClient


class _MockResponse:
    """Minimal requests response object for client tests."""

    def __init__(self, status_code: int, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._payload


def _paper_payload(*, paper_id="p1", title="Paper", year=2020, abstract="Abstract"):
    return {
        "paperId": paper_id,
        "title": title,
        "year": year,
        "abstract": abstract,
        "citationCount": 10,
        "fieldsOfStudy": [],
        "authors": [],
    }


def test_retries_on_rate_limit_from_direct_endpoint():
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


def test_retries_on_transient_api_failure():
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


def test_get_paper_returns_none_after_retries():
    """get_paper should return None after all retry attempts are exhausted."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client.client.get_paper = MagicMock(side_effect=Exception("down"))

    with patch("citemesh.services.semantic_scholar.time.sleep") as sleep_mock:
        result = client.get_paper("seed")

    assert result is None
    assert sleep_mock.call_count == API_CONFIG.max_retries - 1


def test_retries_on_rate_limit_for_citations():
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


def test_retries_on_rate_limit_for_references():
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
