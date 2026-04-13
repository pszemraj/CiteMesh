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


def test_services_package_init_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing ``citemesh.services`` should not eagerly import service clients."""
    _assert_module_reload_is_lazy(
        monkeypatch,
        module=services_module,
        blocked_prefixes=("citemesh.services.semantic_scholar", "semanticscholar"),
        expected_exports={"SemanticScholarClient", "get_client", "reset_client"},
    )


def test_strategies_package_init_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing ``citemesh.strategies`` should not eagerly import strategy modules."""
    import citemesh.strategies as strategies_module

    _assert_module_reload_is_lazy(
        monkeypatch,
        module=strategies_module,
        blocked_prefixes=(
            "citemesh.strategies.citation",
            "citemesh.strategies.embedding",
            "citemesh.strategies.hybrid",
            "citemesh.strategies.recommendation",
        ),
        expected_exports={
            "GraphBuilderStrategy",
            "CitationGraphBuilder",
            "RecommendationGraphBuilder",
            "EmbeddingGraphBuilder",
            "HybridGraphBuilder",
        },
    )


def test_citemesh_package_init_preserves_lazy_top_level_strategy_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing ``citemesh`` should keep top-level strategy exports available lazily."""
    import citemesh as citemesh_module

    _assert_module_reload_is_lazy(
        monkeypatch,
        module=citemesh_module,
        blocked_prefixes=(
            "citemesh.strategies.citation",
            "citemesh.strategies.embedding",
            "citemesh.strategies.hybrid",
            "citemesh.strategies.recommendation",
        ),
        expected_exports={
            "__version__",
            "Paper",
            "Author",
            "GraphBuilderStrategy",
            "CitationGraphBuilder",
            "RecommendationGraphBuilder",
            "EmbeddingGraphBuilder",
            "HybridGraphBuilder",
        },
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
    assert sleep_mock.call_args_list[0].args[0] == 2.0
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
    assert sleep_mock.call_args_list[0].args[0] == API_CONFIG.retry_delay
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
    assert sleep_mock.call_args_list[0].args[0] == API_CONFIG.retry_delay

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
    """Reference cache should support hit, refresh, corrupt-rebuild, and type-error fallback."""
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

    normalized_seed = s2.normalize_paper_id("seed")
    seed_cache_path = s2._reference_cache_path(normalized_seed)
    seed_cache_path.write_text("{bad-json")
    client.client.get_paper_references = MagicMock(
        return_value=[_make_reference_record("a"), _make_reference_record("b")]
    )
    refs = client.get_reference_ids("seed")
    assert refs == ["a", "b"]
    assert json.loads(seed_cache_path.read_text())["references"] == ["a", "b"]

    unicode_seed = s2.normalize_paper_id("seed-unicode")
    unicode_cache_path = s2._reference_cache_path(unicode_seed)
    unicode_cache_path.write_bytes(b"\xff\xfe")
    client.client.get_paper_references = MagicMock(
        return_value=[_make_reference_record("unicode-fixed")]
    )
    unicode_refs = client.get_reference_ids("seed-unicode")
    assert unicode_refs == ["unicode-fixed"]
    assert json.loads(unicode_cache_path.read_text())["references"] == ["unicode-fixed"]

    non_object_seed = s2.normalize_paper_id("seed-non-object")
    non_object_cache_path = s2._reference_cache_path(non_object_seed)
    non_object_cache_path.write_text(
        json.dumps(
            [
                "not",
                "a",
                "dict",
            ]
        )
    )
    client.client.get_paper_references = MagicMock(
        return_value=[_make_reference_record("non-object-fixed")]
    )
    rebuilt_non_object_refs = client.get_reference_ids("seed-non-object")
    assert rebuilt_non_object_refs == ["non-object-fixed"]
    assert json.loads(non_object_cache_path.read_text())["references"] == [
        "non-object-fixed"
    ]

    malformed_seed = s2.normalize_paper_id("seed-malformed")
    malformed_cache_path = s2._reference_cache_path(malformed_seed)
    malformed_cache_path.write_text(
        json.dumps(
            {
                "paper_id": malformed_seed,
                "references": {"unexpected": "mapping"},
                "version": s2.REFERENCE_CACHE_VERSION,
            }
        )
    )
    client.client.get_paper_references = MagicMock(
        return_value=[
            _make_reference_record("fixed-1"),
            _make_reference_record("fixed-2"),
        ]
    )
    rebuilt_refs = client.get_reference_ids("seed-malformed")
    assert rebuilt_refs == ["fixed-1", "fixed-2"]
    assert json.loads(malformed_cache_path.read_text())["references"] == [
        "fixed-1",
        "fixed-2",
    ]

    mixed_seed = s2.normalize_paper_id("seed-mixed")
    mixed_cache_path = s2._reference_cache_path(mixed_seed)
    mixed_cache_path.write_text(
        json.dumps(
            {
                "paper_id": mixed_seed,
                "references": [
                    "ok-1",
                    None,
                    {"paperId": "ok-2"},
                    {"paper_id": "ok-3"},
                    {"paper": {"paperId": "ok-4"}},
                    {"paperId": "   "},
                    123,
                    "ok-1",
                ],
                "version": s2.REFERENCE_CACHE_VERSION,
            }
        )
    )
    client.client.get_paper_references = MagicMock(
        side_effect=AssertionError("API should not be called for mixed cache payload")
    )
    mixed_refs = client.get_reference_ids("seed-mixed")
    assert mixed_refs == ["ok-1", "ok-2", "ok-3", "ok-4"]
    assert json.loads(mixed_cache_path.read_text())["references"] == [
        "ok-1",
        "ok-2",
        "ok-3",
        "ok-4",
    ]

    invalid_only_seed = s2.normalize_paper_id("seed-invalid-only")
    invalid_only_cache_path = s2._reference_cache_path(invalid_only_seed)
    invalid_only_cache_path.write_text(
        json.dumps(
            {
                "paper_id": invalid_only_seed,
                "references": [None, {"paperId": "   "}, {"paper": {}}, 123],
                "version": s2.REFERENCE_CACHE_VERSION,
            }
        )
    )
    client.client.get_paper_references = MagicMock(
        return_value=[_make_reference_record("rebuilt-1")]
    )
    rebuilt_invalid_only_refs = client.get_reference_ids("seed-invalid-only")
    assert rebuilt_invalid_only_refs == ["rebuilt-1"]
    assert json.loads(invalid_only_cache_path.read_text())["references"] == [
        "rebuilt-1"
    ]

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
