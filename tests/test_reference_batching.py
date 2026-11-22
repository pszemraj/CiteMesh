"""Reference hydration uses batch requests without losing complete lists."""

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from citemesh.core import Paper
from citemesh.services import SemanticScholarRequestError
from citemesh.services.semantic_scholar import SemanticScholarClient, endpoints
from citemesh.strategies.reference_hydration import ReferenceHydrator
from tests._helpers import http_response


@pytest.fixture
def client() -> Iterator[SemanticScholarClient]:
    """Use the real client with a fake transport and a short recovery ceiling."""
    with SemanticScholarClient(api_key="", retry_budget_seconds=1) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock()
        client._session.post = MagicMock()
        yield client


def _response(status: int, payload: object = None) -> MagicMock:
    response = http_response(payload)
    response.status_code = status
    return response


def _row(paper_id: str, *refs: str | None) -> dict[str, object]:
    return {
        "paperId": paper_id,
        "referenceCount": len(refs),
        "references": [{"paperId": ref} for ref in refs],
    }


def _requested_batches(post: MagicMock) -> list[list[str]]:
    return [request.kwargs["json"]["ids"] for request in post.call_args_list]


def test_hydrates_cold_selected_ids_and_reuses_complete_lists(
    client: SemanticScholarClient,
) -> None:
    """Hydrate safe aliases while retaining supplied and withheld references."""
    focal = Paper("focal", "Focal", 2020)
    local = Paper(
        "content:local", "Local", 2021, arxiv_id="2101.00001", is_local_corpus=True
    )
    opaque = Paper("content:opaque", "Opaque", 2021, is_local_corpus=True)
    existing = Paper("existing", "Existing", 2021, references=["prior"])
    empty = Paper("empty", "Empty", 2021)
    missing = Paper("missing", "Missing", 2021)
    selected = {p.paper_id: p for p in (focal, local, opaque, existing, empty, missing)}
    client._session.post.return_value = http_response(
        [_row("provider-id", " r1 ", None, "r1"), _row("empty"), None]
    )

    hydrator = ReferenceHydrator(client)
    hydrator.withhold_focal_references(focal)
    hydrator.hydrate(selected, focal)

    client._session.post.assert_called_once()
    request = client._session.post.call_args.kwargs
    assert request["json"] == {"ids": ["arxiv:2101.00001", "empty", "missing"]}
    assert request["params"] == {"fields": "referenceCount,references.paperId"}
    client._session.get.assert_not_called()
    assert local.references == ["r1"]
    assert existing.references == ["prior"]
    assert not focal.references
    assert not opaque.references
    assert client.get_cached_reference_ids("arxiv:2101.00001") == ["r1"]
    assert client.get_cached_reference_ids("empty") == []
    assert client.get_cached_reference_ids("missing") is None

    warm = client.get_reference_ids_batch(
        ["arXiv:2101.00001v2", "arxiv:2101.00001", "empty"]
    )
    assert warm.reference_ids == {"arxiv:2101.00001": ["r1"], "empty": []}
    client._session.post.assert_called_once()

    client._session.post.return_value = http_response([_row("provider-id", "r2")])
    fresh = client.get_reference_ids_batch(["arxiv:2101.00001"], force_refresh=True)
    assert fresh.reference_ids == {"arxiv:2101.00001": ["r2"]}
    assert client.get_cached_reference_ids("arxiv:2101.00001") == ["r2"]


def test_reference_batch_falls_back_only_for_incomplete_rows(
    client: SemanticScholarClient,
) -> None:
    """Truncation uses complete pagination while missing rows stay uncached."""
    incomplete = _row("truncated", "r1")
    incomplete["referenceCount"] = 2
    client._session.post.return_value = http_response(
        [_row("complete", "other"), incomplete, None]
    )
    client._session.get.side_effect = [
        http_response({"data": [{"citedPaper": {"paperId": "r1"}}], "next": 1}),
        http_response({"data": [{"citedPaper": {"paperId": "r2"}}]}),
    ]

    batch = client.get_reference_ids_batch(["complete", "truncated", "missing"])

    assert batch.reference_ids == {"complete": ["other"], "truncated": ["r1", "r2"]}
    assert [request.args[0] for request in client._session.get.call_args_list] == [
        f"{endpoints.PAPER_BASE_URL}/truncated/references",
        f"{endpoints.PAPER_BASE_URL}/truncated/references",
    ]
    assert client.get_cached_reference_ids("truncated") == ["r1", "r2"]
    assert client.get_cached_reference_ids("missing") is None


def test_reference_batch_splits_explicit_size_rejections(
    client: SemanticScholarClient,
) -> None:
    """Retry smaller payloads, with pagination when even a singleton is too large."""
    client._session.post.side_effect = [
        _response(400, {"error": "Response would exceed maximum size of 10 MB"}),
        http_response([_row("small", "small-ref")]),
        _response(413),
    ]
    client._session.get.return_value = http_response(
        {"data": [{"citedPaper": {"paperId": "large-ref"}}]}
    )

    batch = client.get_reference_ids_batch(["small", "large"])

    assert batch.reference_ids == {"small": ["small-ref"], "large": ["large-ref"]}
    assert _requested_batches(client._session.post) == [
        ["small", "large"],
        ["small"],
        ["large"],
    ]
    assert client._session.get.call_args.args[0].endswith("/large/references")


def test_reference_batch_propagates_generic_request_errors(
    client: SemanticScholarClient,
) -> None:
    """Unrecognized fields do not cause a smaller retry or per-paper fallback."""
    client._session.post.return_value = _response(
        400, {"error": "Unrecognized or unsupported fields"}
    )

    with pytest.raises(SemanticScholarRequestError, match="HTTP 400"):
        client.get_reference_ids_batch(["a", "b"])

    client._session.post.assert_called_once()
    client._session.get.assert_not_called()


@pytest.mark.parametrize("status", [429, 503])
def test_reference_batch_outage_keeps_warm_and_completed_lists(
    client: SemanticScholarClient, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """An exhausted later chunk does not start cold or paginated fan-out."""
    monkeypatch.setattr(endpoints, "PAPER_BATCH_SIZE", 2)
    incomplete = _row("incomplete")
    incomplete["referenceCount"] = 2
    client._session.post.side_effect = [
        http_response([_row("done", "ref"), incomplete]),
        _response(status),
    ]
    client._persist_reference_cache_entry(
        endpoints.disk_cache._reference_cache_path("warm"), "warm", []
    )
    hydrator = ReferenceHydrator(client)

    assert hydrator.fetch_many(["done", "incomplete", "failed", "warm"]) == {
        "warm": [],
        "done": ["ref"],
    }
    assert hydrator.fetch_many(["warm", "done", "incomplete", "other"]) == {
        "warm": [],
        "done": ["ref"],
    }
    assert _requested_batches(client._session.post) == [
        ["done", "incomplete"],
        ["failed"],
    ]
    client._session.get.assert_not_called()
    assert client.get_cached_reference_ids("incomplete") is None
    assert client.get_cached_reference_ids("failed") is None
