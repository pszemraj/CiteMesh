"""Additional service tests for reference cache and recommendation/search paths."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from citemesh.services import semantic_scholar as s2
from citemesh.services.semantic_scholar import SemanticScholarClient


def _make_reference_record(paper_id: str) -> SimpleNamespace:
    """Create a minimal reference record with ``paper.paperId``.

    :param str paper_id: Reference paper ID.
    :return SimpleNamespace: Record compatible with parser expectations.
    """
    return SimpleNamespace(paper=SimpleNamespace(paperId=paper_id))


def test_get_reference_ids_uses_valid_cache(tmp_path: Path, monkeypatch) -> None:
    """Reference ID lookup should return cached payload without API call."""
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


def test_get_reference_ids_recovers_from_corrupt_cache(
    tmp_path: Path, monkeypatch
) -> None:
    """Corrupt cache files should be discarded and rebuilt from API response."""
    monkeypatch.setattr(s2, "REFERENCE_CACHE_DIR", tmp_path)
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None

    normalized = s2.normalize_paper_id("seed")
    cache_path = s2._reference_cache_path(normalized)
    cache_path.write_text("{bad-json")
    client.client.get_paper_references = MagicMock(
        return_value=[_make_reference_record("a"), _make_reference_record("b")]
    )

    refs = client.get_reference_ids("seed")
    assert refs == ["a", "b"]

    rewritten = json.loads(cache_path.read_text())
    assert rewritten["references"] == ["a", "b"]
    assert rewritten["version"] == s2.REFERENCE_CACHE_VERSION


def test_get_reference_ids_handles_type_error_as_empty(
    tmp_path: Path, monkeypatch
) -> None:
    """Missing reference payloads should return an empty list."""
    monkeypatch.setattr(s2, "REFERENCE_CACHE_DIR", tmp_path)
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client.client.get_paper_references = MagicMock(side_effect=TypeError("missing"))

    assert client.get_reference_ids("seed") == []


def test_recommendation_and_search_payload_conversion() -> None:
    """Recommendation/search wrappers should convert direct endpoint payloads."""
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


def test_atomic_reference_cache_write_preserves_existing_file_on_replace_error(
    tmp_path: Path, monkeypatch
) -> None:
    """Atomic writes must keep valid cached data when filesystem rename fails."""
    monkeypatch.setattr(s2, "REFERENCE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        s2.os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("simulated replace failure")
        ),
    )
    client = SemanticScholarClient(timeout=1)
    client._rate_limit = lambda: None
    client.client.get_paper_references = MagicMock(
        side_effect=AssertionError(
            "API should not be called on replace failure with valid cache"
        )
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
    refs = client.get_reference_ids("seed")
    assert refs == ["cached-ref"]
    assert json.loads(cache_path.read_text()) == prior_payload
