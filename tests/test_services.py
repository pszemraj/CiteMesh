"""Tests for Semantic Scholar client behavior and reference cache handling."""

from __future__ import annotations

import builtins
import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

import citemesh.services as services_module
from citemesh.core import API_CONFIG, Paper
from citemesh.services import semantic_scholar as s2
from citemesh.services import semantic_scholar as semantic_module
from citemesh.services.semantic_scholar import (
    SemanticScholarClient,
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
    assert EmbeddingGraphBuilder.__module__ == "citemesh.strategies.embedding"


@pytest.mark.parametrize(
    ("module", "blocked_prefixes", "expected_exports"),
    [
        (
            services_module,
            ("citemesh.services.semantic_scholar", "semanticscholar"),
            {
                "SemanticScholarClient",
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
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
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

        with patch("citemesh.services.semantic_scholar.time.sleep") as sleep_mock:
            if api_method == "get_paper_citations":
                result = client.get_paper_citations("seed", limit=5)
            else:
                result = client.get_paper_references("seed", limit=5)

        assert sleep_mock.call_count == 1
        assert sleep_mock.call_args_list[0].args[0] == delay
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
    with patch("citemesh.services.semantic_scholar.time.sleep") as sleep_mock:
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
    with patch("citemesh.services.semantic_scholar.time.sleep") as sleep_mock:
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
    client.client.get_paper = MagicMock(side_effect=[Exception("temporary"), api_paper])
    with patch("citemesh.services.semantic_scholar.time.sleep") as sleep_mock:
        result = client.get_paper("seed")

    assert isinstance(result, Paper)
    assert result.paper_id == "seed"
    assert sleep_mock.call_count == 1
    # Full-jitter wait for a non-429 transient failure on the first retry.
    assert 0.0 <= sleep_mock.call_args_list[0].args[0] <= API_CONFIG.retry_delay

    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client.client.get_paper = MagicMock(side_effect=Exception("down"))
    with patch("citemesh.services.semantic_scholar.time.sleep") as sleep_mock:
        result = client.get_paper("seed")

    assert result is None
    assert sleep_mock.call_count == API_CONFIG.max_retries - 1


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
    client.client.get_paper = MagicMock(return_value=api_paper)

    result = client.get_paper("https://arxiv.org/abs/2508.14040")
    assert isinstance(result, Paper)
    assert result.paper_id == "seed"
    assert client.client.get_paper.call_args.args[0] == "arxiv:2508.14040"


def test_get_papers_batches_and_falls_back_for_unmatched_ids() -> None:
    """Batch paper fetches should map results back to requested IDs and retry misses."""
    batch_paper = SimpleNamespace(
        paperId="seed",
        title="Seed",
        year=2025,
        authors=[],
        citationCount=11,
        abstract="abstract",
        fieldsOfStudy=[],
        externalIds={"ArXiv": "2508.14040"},
    )
    fallback_paper = Paper(
        paper_id="fallback",
        title="Fallback",
        year=2024,
        abstract="fallback abstract",
        citation_count=22,
    )

    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client.client.get_papers = MagicMock(return_value=([batch_paper], ["missing-id"]))
    client.get_paper = MagicMock(return_value=fallback_paper)

    result = client.get_papers(["https://arxiv.org/abs/2508.14040", "missing-id"])

    assert set(result) == {"arxiv:2508.14040", "missing-id"}
    assert result["arxiv:2508.14040"].paper_id == "seed"
    assert result["missing-id"].paper_id == "fallback"
    assert client.client.get_papers.call_args.args[0] == [
        "arxiv:2508.14040",
        "missing-id",
    ]
    assert client.get_paper.call_args.args[0] == "missing-id"


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
        "authors": [{"name": ""}, {}],
        "fieldsOfStudy": ["cs.AI", "cs.LG"],
    }
    paper = client._convert_recommendation(payload)

    assert paper is not None
    assert paper.paper_id == "p1"
    assert paper.title == "Unknown"
    assert paper.year is None
    assert paper.abstract == ""
    assert paper.venue == "ICLR"
    assert paper.arxiv_id == "2411.03884v2"
    assert paper.doi == "10.1145/3133956.3134029"
    assert paper.authors == []
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

    recommendations = client.get_recommended_papers("seed", limit=1)
    search_results = client.search_papers("transformer", limit=1)
    assert [paper.paper_id for paper in recommendations] == ["rec1"]
    assert recommendations[0].references == ["r1", "r2", "r3"]
    assert [paper.paper_id for paper in search_results] == ["search1"]

    client = SemanticScholarClient(timeout=1)
    client._request_json = MagicMock(return_value={"recommendedPapers": []})
    client.get_recommended_papers("seed", limit=1, include_references=True)
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


def test_external_id_fallback_from_paper_id_contracts() -> None:
    """Paper-ID fallback should populate arXiv/DOI fields when external IDs are absent."""
    client = SemanticScholarClient(timeout=1)

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

    arxiv = client._convert_recommendation(arxiv_payload)
    doi = client._convert_recommendation(doi_payload)

    assert arxiv is not None
    assert arxiv.arxiv_id == "2411.03884"
    assert arxiv.doi == ""
    assert doi is not None
    assert doi.arxiv_id == ""
    assert doi.doi == "10.1145/3133956.3134029"


def test_reference_cache_hit_corrupt_and_type_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference cache should reuse valid/legacy hits and rebuild invalid payloads."""
    monkeypatch.setattr(s2, "REFERENCE_CACHE_DIR", tmp_path)
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None

    normalized = s2.normalize_paper_id("arxiv:1234.5678")
    cache_path = s2._reference_cache_path(normalized)
    cache_path.write_text(
        json.dumps(
            {
                "paper_id": normalized,
                "references": ["r1", "r2"],
                "version": s2.REFERENCE_CACHE_VERSION,
            }
        )
    )
    client.client.get_paper_references = MagicMock(
        side_effect=AssertionError("API should not be called on cache hit")
    )
    assert client.get_reference_ids("arxiv:1234.5678") == ["r1", "r2"]

    legacy_mixed_paper_id = s2.normalize_paper_id("seed-mixed")
    legacy_mixed_cache_path = s2._reference_cache_path(legacy_mixed_paper_id)
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
                "version": s2.REFERENCE_CACHE_VERSION,
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
                    "version": s2.REFERENCE_CACHE_VERSION,
                }
            ),
            "text",
            ["fixed-1", "fixed-2"],
        ),
    ]
    for paper_id, cached_payload, write_mode, rebuilt_ids in rebuild_cases:
        normalized_paper_id = s2.normalize_paper_id(paper_id)
        rebuilt_cache_path = s2._reference_cache_path(normalized_paper_id)
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

    client.client.get_paper_references = MagicMock(side_effect=TypeError("missing"))
    assert client.get_reference_ids("seed-type-error") == []


def test_reference_cache_resilience_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference cache pathing, empty persistence, failures, and atomicity should hold."""

    cache_root = tmp_path / "runtime-root"
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(s2, "REFERENCE_CACHE_DIR", None)
    normalized = s2.normalize_paper_id("arxiv:1234.5678")
    path = s2._reference_cache_path(normalized)
    assert path.parent == cache_root / "references"
    assert path.parent.exists()

    cache_dir = tmp_path / "cache-resilience"
    monkeypatch.setattr(s2, "REFERENCE_CACHE_DIR", cache_dir)
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client.client.get_paper_references = MagicMock(return_value=[])
    refs = client.get_reference_ids("seed-empty")
    normalized_empty = s2.normalize_paper_id("seed-empty")
    cache_path_empty = s2._reference_cache_path(normalized_empty)
    assert refs == []
    assert cache_path_empty.exists()
    assert json.loads(cache_path_empty.read_text()) == {
        "paper_id": normalized_empty,
        "references": [],
        "version": s2.REFERENCE_CACHE_VERSION,
    }

    failing_client = SemanticScholarClient(timeout=1)
    failing_client._rate_limit = lambda: None
    failing_client.client.get_paper_references = MagicMock(
        side_effect=RuntimeError("down")
    )
    with patch("citemesh.services.semantic_scholar.time.sleep"):
        with pytest.raises(
            RuntimeError,
            match=r"Failed to fetch reference IDs after retries for seed-failure\.",
        ):
            failing_client.get_reference_ids("seed-failure")

    monkeypatch.setattr(
        s2.os,
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
    cache_path = s2._reference_cache_path(normalized)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prior_payload = {
        "paper_id": normalized,
        "references": ["cached-ref"],
        "version": s2.REFERENCE_CACHE_VERSION,
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
    monkeypatch.setattr(s2, "REFERENCE_CACHE_DIR", tmp_path)
    client = SemanticScholarClient(timeout=1)

    normalized = s2.normalize_paper_id("seed-empty")
    cache_path = s2._reference_cache_path(normalized)
    cache_path.write_text(
        json.dumps(
            {
                "paper_id": normalized,
                "references": [],
                "version": s2.REFERENCE_CACHE_VERSION,
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

    previous_session = semantic_module.requests.Session
    previous_client = semantic_module.SemanticScholar
    try:
        semantic_module.requests.Session = _FakeRequestsSession
        semantic_module.SemanticScholar = _build_fake_semantic_scholar_api()

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
        semantic_module.requests.Session = previous_session
        semantic_module.SemanticScholar = previous_client
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
    previous_session = semantic_module.requests.Session
    previous_client = semantic_module.SemanticScholar
    try:
        semantic_module.requests.Session = _FakeRequestsSession
        semantic_module.SemanticScholar = _build_fake_semantic_scholar_api()

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
        semantic_module.requests.Session = previous_session
        semantic_module.SemanticScholar = previous_client


def test_rate_limit_pace_is_key_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    """Authenticated clients pace at the faster keyed rate; anonymous stays slow."""
    monkeypatch.delenv("S2_API_KEY", raising=False)
    anonymous = SemanticScholarClient(timeout=1)
    assert anonymous.requests_per_second == API_CONFIG.requests_per_second

    monkeypatch.setenv("S2_API_KEY", "test-key")
    keyed = SemanticScholarClient(timeout=1)
    assert keyed.requests_per_second == API_CONFIG.authenticated_requests_per_second
    assert keyed._session.headers["x-api-key"] == "test-key"

    # Empty env value means "explicitly anonymous" (CI convention).
    monkeypatch.setenv("S2_API_KEY", "")
    explicit_anonymous = SemanticScholarClient(timeout=1)
    assert explicit_anonymous.requests_per_second == API_CONFIG.requests_per_second
    assert "x-api-key" not in explicit_anonymous._session.headers


def test_anonymous_pool_notice_logged_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The key-less shared-pool notice appears once per process, not per client."""
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.setattr(semantic_module, "_anonymous_pool_announced", False)
    with caplog.at_level(logging.INFO, logger="citemesh.services.semantic_scholar"):
        SemanticScholarClient(timeout=1)
        SemanticScholarClient(timeout=1)
    notices = [
        record
        for record in caplog.records
        if "shared anonymous Semantic" in record.getMessage()
    ]
    assert len(notices) == 1
    assert semantic_module.S2_API_KEY_SIGNUP_URL in notices[0].getMessage()


def test_get_paper_raise_on_unavailable_distinguishes_outage() -> None:
    """Exhausted retries raise SemanticScholarUnavailableError in strict mode."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client.client.get_paper = MagicMock(side_effect=Exception("connection reset"))
    with patch("citemesh.services.semantic_scholar.time.sleep"):
        with pytest.raises(
            semantic_module.SemanticScholarUnavailableError,
            match="unreachable while fetching seed",
        ):
            client.get_paper("seed", raise_on_unavailable=True)

    rate_limited = SemanticScholarClient(timeout=1)
    rate_limited._rate_limit = lambda: None
    rate_limited.client.get_paper = MagicMock(side_effect=Exception("HTTP 429"))
    with patch("citemesh.services.semantic_scholar.time.sleep"):
        with pytest.raises(
            semantic_module.SemanticScholarUnavailableError,
            match="rate-limited",
        ):
            rate_limited.get_paper("seed", raise_on_unavailable=True)


def test_get_paper_not_found_still_returns_none_in_strict_mode() -> None:
    """Strict mode only changes outage handling; genuine not-found stays None."""
    from semanticscholar.SemanticScholarException import ObjectNotFoundException

    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client.client.get_paper = MagicMock(side_effect=ObjectNotFoundException("missing"))
    assert client.get_paper("missing-id", raise_on_unavailable=True) is None


def test_recommendations_fall_back_to_all_cs_pool() -> None:
    """Empty default-pool responses retry against the broader all-cs pool."""
    client = SemanticScholarClient(timeout=1)
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


def test_jittered_backoff_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff escalates exponentially with jitter, floored at Retry-After."""
    drawn_bounds: list[tuple[float, float]] = []

    def _record_uniform(low: float, high: float) -> float:
        drawn_bounds.append((low, high))
        return high  # deterministic: always draw the cap

    monkeypatch.setattr(semantic_module.random, "uniform", _record_uniform)

    # Exponential cap growth: retry_delay * 2^(attempt-1), doubled when
    # rate-limited, capped at _MAX_BACKOFF_SECONDS.
    delay = API_CONFIG.retry_delay
    assert semantic_module._jittered_backoff(1) == delay
    assert semantic_module._jittered_backoff(2) == delay * 2
    assert semantic_module._jittered_backoff(1, rate_limited=True) == delay * 2
    assert semantic_module._jittered_backoff(3, rate_limited=True) == delay * 8
    assert (
        semantic_module._jittered_backoff(30, rate_limited=True)
        == semantic_module._MAX_BACKOFF_SECONDS
    )
    assert all(low == 0.0 for low, _high in drawn_bounds)

    # Retry-After floors the wait but never exceeds the global cap.
    monkeypatch.setattr(semantic_module.random, "uniform", lambda low, high: 0.0)
    assert semantic_module._jittered_backoff(1, retry_after=7.5) == 7.5
    assert (
        semantic_module._jittered_backoff(1, retry_after=999.0)
        == semantic_module._MAX_BACKOFF_SECONDS
    )


def test_search_raise_on_unavailable_distinguishes_rate_limit() -> None:
    """Exhausted search retries raise in strict mode instead of returning []."""
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client._session.get = MagicMock(return_value=_MockResponse(status_code=429))

    with patch("citemesh.services.semantic_scholar.time.sleep"):
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

    with patch("citemesh.services.semantic_scholar.time.sleep"):
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
