"""Semantic Scholar direct-HTTP service and cache contracts."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from citemesh.core import API_CONFIG, Author, Paper
from citemesh.data.cache import JSONValue
from citemesh.services import semantic_scholar as s2
from citemesh.services.semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarRequestError,
    SemanticScholarUnavailableError,
    get_client,
    normalize_paper_id,
    reset_client,
)
from tests._helpers import get_paper_id_normalization_cases


class _MockResponse:
    """Small requests response stub."""

    def __init__(
        self,
        status_code: int,
        payload: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Store response status, payload, and headers.

        :param int status_code: HTTP status code.
        :param Any payload: JSON response value.
        :param dict[str, str] | None headers: Response headers.
        :return None: Initializes the response.
        """
        self.status_code = status_code
        self._payload = {} if payload is None else payload
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        """Raise the requests error used by real responses.

        :return None: Returns for a successful status.
        """
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> Any:
        """Return the configured JSON value.

        :return Any: Configured response payload.
        """
        return self._payload


class _BrokenJsonResponse(_MockResponse):
    """Response whose JSON body cannot be decoded."""

    def json(self) -> Any:
        """Raise the decoder error emitted by Requests.

        :raises requests.exceptions.JSONDecodeError: Always.
        :return Any: Does not return successfully.
        """
        raise requests.exceptions.JSONDecodeError("malformed JSON", "invalid", 0)


class _Clock:
    """Deterministic monotonic clock for recovery-budget tests."""

    def __init__(self) -> None:
        """Start at monotonic time zero.

        :return None: Initializes the clock.
        """
        self.now = 0.0

    def monotonic(self) -> float:
        """Return current synthetic time.

        :return float: Current synthetic time.
        """
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance by a requested sleep.

        :param float seconds: Elapsed seconds.
        :return None: Advances the clock.
        """
        self.now += seconds


def _paper_payload(
    paper_id: str = "p1",
    *,
    title: str = "Paper",
    year: int = 2020,
    doi: str = "",
    arxiv_id: str = "",
) -> dict[str, Any]:
    """Build a complete Semantic Scholar paper payload.

    :param str paper_id: Paper identifier.
    :param str title: Paper title.
    :param int year: Publication year.
    :param str doi: Optional DOI.
    :param str arxiv_id: Optional arXiv identifier.
    :return dict[str, Any]: Complete response row.
    """
    external_ids: dict[str, str] = {}
    if doi:
        external_ids["DOI"] = doi
    if arxiv_id:
        external_ids["ArXiv"] = arxiv_id
    return {
        "paperId": paper_id,
        "title": title,
        "year": year,
        "authors": [{"authorId": "a1", "name": "Ada"}],
        "citationCount": 3,
        "abstract": "Abstract",
        "fieldsOfStudy": ["Computer Science"],
        "externalIds": external_ids,
        "venue": "Venue",
    }


def _relation_page(
    records: list[dict[str, Any]], nested_key: str, *, next_offset: int | None = None
) -> dict[str, Any]:
    """Build one relation page from nested paper records.

    :param list[dict[str, Any]] records: Nested related-paper payloads.
    :param str nested_key: Relation paper key.
    :param int | None next_offset: Optional next-page offset.
    :return dict[str, Any]: Relation response payload.
    """
    payload: dict[str, Any] = {"data": [{nested_key: record} for record in records]}
    if next_offset is not None:
        payload["next"] = next_offset
    return payload


def _relation_payload(
    ids: list[str | None], nested_key: str, *, next_offset: int | None = None
) -> dict[str, Any]:
    """Build one relation page whose rows carry full paper metadata.

    :param list[str | None] ids: Related paper identifiers; ``None`` is an
        unresolved row.
    :param str nested_key: Relation paper key.
    :param int | None next_offset: Optional next-page offset.
    :return dict[str, Any]: Relation response payload.
    """
    return _relation_page(
        [
            {"paperId": None} if paper_id is None else _paper_payload(paper_id)
            for paper_id in ids
        ],
        nested_key,
        next_offset=next_offset,
    )


def _disable_pacing(client: SemanticScholarClient) -> None:
    """Disable request pacing in tests unrelated to timing.

    :param SemanticScholarClient client: Client to modify.
    :return None: Replaces its pacing method.
    """
    client._rate_limit = MagicMock()


@pytest.mark.parametrize(("raw_id", "expected"), get_paper_id_normalization_cases())
def test_identifier_normalization_contract(raw_id: str, expected: str) -> None:
    """Public normalization keeps cross-source identifiers canonical.

    :param str raw_id: Input identifier.
    :param str expected: Expected canonical identifier.
    :return None: Verifies normalization.
    """
    assert normalize_paper_id(raw_id) == expected


def test_payload_conversion_preserves_metadata_quirks() -> None:
    """Dict payload conversion keeps authors, venue, categories, and external IDs.

    :return None: Verifies normalized fields.
    """
    paper = s2.payloads._convert_recommendation(
        {
            **_paper_payload("arxiv:1234.5678", doi="10.1/ABC"),
            "venue": "",
            "publicationVenue": {"name": "Conference"},
            "references": [{"paperId": "r1"}, {"paper": {"paperId": "r2"}}],
        }
    )
    assert paper is not None
    assert paper.authors == [Author(name="Ada", author_id="a1")]
    assert paper.venue == "Conference"
    assert paper.categories == ["Computer Science"]
    assert paper.arxiv_id == "1234.5678"
    assert paper.doi == "10.1/ABC"
    assert paper.references == ["r1", "r2"]


def test_arxiv_journal_identity_matches_corpus_and_survives_cache() -> None:
    """C-Pack's journal identifier joins its otherwise unlinked S2 record."""
    from citemesh.strategies.candidates import candidate_records_match

    paper = s2.payloads._convert_api_paper(
        {
            **_paper_payload("2ff694e20f492a7acf7fd0646c5e1576f0b3c901"),
            "title": "C-Pack: Packaged Resources To Advance General Chinese Embedding",
            "externalIds": {"DBLP": "journals/corr/abs-2309-07597"},
            "journal": {"name": "ArXiv", "volume": "abs/2309.07597"},
        }
    )
    assert paper is not None
    assert paper.arxiv_id == "2309.07597"
    assert candidate_records_match(
        paper, Paper(paper_id="arxiv:2309.07597", title="C-Pack", year=2023)
    )
    s2.disk_cache._persist_paper(paper, paper.paper_id)
    cached = s2.disk_cache._load_cached_paper("arxiv:2309.07597")
    assert cached is not None
    assert cached.paper_id == paper.paper_id
    assert cached.arxiv_id == "2309.07597"


@pytest.mark.parametrize(
    ("journal", "expected"),
    [
        ({"name": " arXiv ", "volume": "abs/cs/0101001v2"}, "cs/0101001"),
        ({"name": "Other journal", "volume": "abs/2309.07597"}, ""),
        ({"name": "ArXiv", "volume": "42"}, ""),
        ({"name": "ArXiv", "volume": "abs/not-an-id"}, ""),
        ({"name": "ArXiv", "volume": "abs/arxiv:not-an-id"}, ""),
        ({"name": "ArXiv"}, ""),
    ],
)
def test_journal_identity_requires_an_explicit_arxiv_volume(
    journal: dict[str, str], expected: str
) -> None:
    """Only a recognized identifier in an arXiv volume adds an alias."""
    paper = s2.payloads._convert_recommendation(
        {**_paper_payload(), "journal": journal}
    )
    assert paper is not None
    assert paper.arxiv_id == expected


@pytest.mark.parametrize(
    ("paper_id", "external_ids", "expected"),
    [
        ("s2-id", {"ArXiv": "2401.00001"}, "2401.00001"),
        ("arxiv:2401.00001", {}, "2401.00001"),
        ("s2-id", {"DOI": "10.48550/arXiv.2401.00001"}, ""),
        ("s2-id", {"DOI": "https://doi.org/10.48550/arXiv.2401.00001"}, ""),
        ("s2-id", {"DOI": "doi:10.48550/arXiv.2401.00001"}, ""),
    ],
)
def test_journal_identity_does_not_override_existing_arxiv_identity(
    paper_id: str, external_ids: dict[str, str], expected: str
) -> None:
    """A journal fallback cannot contradict an explicit arXiv ID or DOI."""
    paper = s2.payloads._convert_api_paper(
        {
            **_paper_payload(paper_id),
            "externalIds": external_ids,
            "journal": {"name": "ArXiv", "volume": "abs/2309.07597"},
        }
    )
    assert paper is not None
    assert paper.arxiv_id == expected


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"title": 123},
        {"year": "2024"},
        {"citationCount": "3"},
        {"abstract": 456},
        {"authors": [{"name": "Ada", "authorId": 123}]},
    ],
)
def test_payload_conversion_rejects_invalid_scalar_types(
    invalid_fields: dict[str, object],
) -> None:
    """Malformed API fields follow the existing conversion failure path.

    :param invalid_fields: Raw fields that violate the paper model's types.
    """
    payload = {**_paper_payload(), **invalid_fields}
    assert s2.payloads._convert_recommendation(payload) is None
    assert s2.payloads._convert_api_paper(payload) is None


def test_cache_writes_one_canonical_record_and_alias_stubs() -> None:
    """Paper metadata is stored once while DOI/arXiv/request aliases stay small.

    :return None: Verifies on-disk payload shapes.
    """
    paper = Paper(
        paper_id="S2-CANON",
        title="Canonical",
        year=2024,
        doi="10.1/example",
        arxiv_id="2401.00001",
        references=["ignored"],
        is_seed=True,
    )
    s2.disk_cache._persist_paper(paper, "requested")

    canonical = json.loads(
        s2.disk_cache._paper_cache_path("S2-CANON").read_text(encoding="utf-8")
    )
    assert canonical["paper"]["title"] == "Canonical"
    assert canonical["paper"]["references"] == []
    assert canonical["paper"]["is_seed"] is False
    for alias in ("requested", "10.1/example", "arxiv:2401.00001"):
        assert json.loads(
            s2.disk_cache._paper_cache_path(alias).read_text(encoding="utf-8")
        ) == {"version": 2, "canonical_paper_id": "S2-CANON"}


def test_alias_reads_follow_canonical_refresh_and_return_fresh_instances() -> None:
    """Alias reads always resolve the current canonical record through one hop.

    :return None: Verifies fresh canonical reads.
    """
    first = Paper(paper_id="canonical", title="Old", year=2020, doi="10.1/x")
    second = Paper(paper_id="canonical", title="New", year=2025, doi="10.1/x")
    s2.disk_cache._persist_paper(first, "10.1/x")
    s2.disk_cache._persist_paper(second, "canonical")
    left = s2.disk_cache._load_cached_paper("10.1/x")
    right = s2.disk_cache._load_cached_paper("10.1/x")
    assert left is not None and right is not None
    assert left.title == right.title == "New"
    assert left is not right


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"year": "2024"},
        {"citation_count": None},
        {"abstract": []},
        {"categories": [1]},
        {"references": "other"},
        {"authors": [{"name": 1}]},
    ],
)
def test_paper_cache_misses_on_invalid_typed_fields(
    invalid_fields: dict[str, JSONValue],
) -> None:
    """Reject cached fields that cannot form typed paper metadata.

    :param dict[str, JSONValue] invalid_fields: Malformed cached paper fields.
    :return None: Checks that invalid metadata produces a cache miss.
    """
    paper: dict[str, JSONValue] = {
        "paper_id": "canonical",
        "title": "Cached",
        "year": 2024,
        **invalid_fields,
    }
    s2.disk_cache._paper_cache_path("canonical").write_text(
        json.dumps({"version": 2, "paper": paper}), encoding="utf-8"
    )
    assert s2.disk_cache._load_cached_paper("canonical") is None


def test_paper_cache_accepts_omitted_optional_fields() -> None:
    """Load minimal cached paper data with the dataclass defaults."""
    s2.disk_cache._paper_cache_path("canonical").write_text(
        json.dumps(
            {
                "version": 2,
                "paper": {"paper_id": "canonical", "title": "Cached", "year": None},
            }
        ),
        encoding="utf-8",
    )
    cached = s2.disk_cache._load_cached_paper("canonical")
    assert cached is not None
    assert cached.title == "Cached" and cached.year is None
    assert cached.authors == [] and cached.citation_count == 0


def test_legacy_full_alias_uses_canonical_and_misses_without_it() -> None:
    """Legacy duplicated alias payloads cannot return stale embedded metadata.

    :return None: Verifies migration-compatible lookup.
    """
    canonical = Paper(paper_id="canonical", title="Fresh", year=2025)
    legacy = Paper(paper_id="canonical", title="Stale", year=2020)
    s2.disk_cache._persist_paper(canonical, "canonical")
    s2.disk_cache._paper_cache_path("alias").write_text(
        json.dumps({"version": 2, "paper": asdict(legacy)}), encoding="utf-8"
    )
    assert s2.disk_cache._load_cached_paper("alias").title == "Fresh"
    s2.disk_cache._paper_cache_path("canonical").unlink()
    assert s2.disk_cache._load_cached_paper("alias") is None


def test_reassigned_alias_is_not_reclaimed_by_later_old_canonical_refresh() -> None:
    """Refreshing an old paper never traverses history to reclaim a reassigned DOI.

    :return None: Verifies current alias ownership.
    """
    s2.disk_cache._persist_paper(
        Paper(paper_id="old", title="Old", year=2020, doi="10.1/shared"), "old"
    )
    s2.disk_cache._persist_paper(
        Paper(paper_id="new", title="New", year=2024, doi="10.1/shared"), "10.1/shared"
    )
    s2.disk_cache._persist_paper(
        Paper(paper_id="old", title="Refreshed", year=2025, doi="10.1/other"),
        "old",
    )
    assert s2.disk_cache._load_cached_paper("10.1/shared").paper_id == "new"


def test_failed_canonical_write_preserves_existing_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed metadata write must not redirect aliases to a missing record.

    :param pytest.MonkeyPatch monkeypatch: Injects a canonical cache write failure.
    :return None: Verifies the previously cached paper remains reachable.
    """
    alias = "10.1000/shared"
    old = Paper(paper_id="old", title="Old", year=2020, doi=alias)
    new = Paper(paper_id="new", title="New", year=2024, doi=alias)
    s2.disk_cache._persist_paper(old, alias)
    new_path = s2.disk_cache._paper_cache_path(new.paper_id)
    write_json = s2.disk_cache.atomic_write_json

    def fail_canonical_write(path: Path, payload: dict[str, Any]) -> None:
        """Fail the new record write while allowing alias writes.

        :param Path path: Destination cache file.
        :param dict[str, Any] payload: Cache payload to write.
        :return None: Writes all files except the new canonical record.
        """
        if path == new_path:
            raise OSError("canonical cache write failed")
        write_json(path, payload)

    monkeypatch.setattr(s2.disk_cache, "atomic_write_json", fail_canonical_write)
    s2.disk_cache._persist_paper(new, alias)

    assert not new_path.exists()
    assert s2.disk_cache._load_cached_paper(alias) == old


def test_get_paper_cache_not_found_refresh_and_reference_enrichment() -> None:
    """Single lookup honors cache, 404, explicit refresh, and relation enrichment.

    :return None: Verifies single-paper lookup behavior.
    """
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            return_value=_MockResponse(200, _paper_payload())
        )
        assert client.get_paper("p1").title == "Paper"
        client._session.get.assert_called_once()
        assert client.get_paper("p1").title == "Paper"
        client._session.get.assert_called_once()

    with SemanticScholarClient(api_key="", refresh_paper_cache=True) as refresh:
        _disable_pacing(refresh)
        refresh._session.get = MagicMock(
            side_effect=[
                _MockResponse(200, _paper_payload(title="Fresh")),
                _MockResponse(404),
            ]
        )
        assert refresh.get_paper("p1").title == "Fresh"
        assert refresh.get_paper("missing", raise_on_unavailable=True) is None


def test_get_cached_papers_reads_only_persisted_records() -> None:
    """Cache-only bulk lookup never performs a provider request.

    :return None: Verifies normalized cache hits and refresh bypass behavior.
    """
    cached = Paper(paper_id="warm", title="Warm", year=2020)
    s2.disk_cache._persist_paper(cached, "warm")
    with SemanticScholarClient(api_key="") as client:
        client._session.post = MagicMock()
        assert client.get_cached_papers(["warm", "missing", "warm"]) == {"warm": cached}
        client._session.post.assert_not_called()

    with SemanticScholarClient(api_key="", refresh_paper_cache=True) as refresh:
        assert refresh.get_cached_papers(["warm"]) == {}


def test_get_papers_fetches_only_misses_and_preserves_request_order() -> None:
    """Bulk lookup owns cache reads and batches only missing identifiers.

    :return None: Verifies mixed warm/cold batching.
    """
    s2.disk_cache._persist_paper(
        Paper(paper_id="warm", title="Warm", year=2020), "warm"
    )
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.post = MagicMock(
            return_value=_MockResponse(200, [_paper_payload("cold")])
        )
        papers = client.get_papers(["cold", "warm", "cold"])
    assert list(papers) == ["cold", "warm"]
    assert client._session.post.call_args.kwargs["json"] == {"ids": ["cold"]}


def test_get_papers_splits_arbitrary_missing_count_at_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public bulk call splits missing IDs into valid provider batches.

    :param pytest.MonkeyPatch monkeypatch: Skips unrelated disk persistence.
    :return None: Verifies 500-record chunking.
    """
    ids = [f"p{i}" for i in range(1001)]
    persist_paper = MagicMock()
    monkeypatch.setattr(s2.disk_cache, "_persist_paper", persist_paper)

    def respond(_url: str, **kwargs: Any) -> _MockResponse:
        """Return positional rows for the submitted batch.

        :param str _url: Ignored batch URL.
        :param Any kwargs: Submitted request arguments.
        :return _MockResponse: Positional batch response.
        """
        return _MockResponse(
            200, [_paper_payload(paper_id) for paper_id in kwargs["json"]["ids"]]
        )

    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.post = MagicMock(side_effect=respond)
        papers = client.get_papers(ids, raise_on_unavailable=True)
    assert list(papers) == ids
    assert [
        len(call.kwargs["json"]["ids"]) for call in client._session.post.call_args_list
    ] == [500, 500, 1]
    assert [call.args[1] for call in persist_paper.call_args_list] == ids


def test_tolerant_get_papers_warns_after_unavailable_batch(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed metadata batch emits one actionable availability warning.

    :param pytest.MonkeyPatch monkeypatch: Makes retry waits unaffordable.
    :param pytest.LogCaptureFixture caplog: Captured warning log.
    :return None: Verifies partial metadata results remain attributable to an outage.
    """
    ids = [f"p{index}" for index in range(1001)]
    persist_paper = MagicMock()
    monkeypatch.setattr(s2.disk_cache, "_persist_paper", persist_paper)
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 120.0)
    with SemanticScholarClient(api_key="", retry_budget_seconds=5) as client:
        _disable_pacing(client)
        client._session.post = MagicMock(
            side_effect=[
                _MockResponse(
                    200, [_paper_payload(paper_id) for paper_id in ids[:500]]
                ),
                _MockResponse(503),
            ]
        )
        with caplog.at_level(logging.WARNING):
            papers = client.get_papers(ids)

    assert list(papers) == ids[:500]
    assert [
        len(call.kwargs["json"]["ids"]) for call in client._session.post.call_args_list
    ] == [500, 500]
    assert [call.args[1] for call in persist_paper.call_args_list] == ids[:500]
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "batch fetching 500 papers" in warnings[0]


@pytest.mark.parametrize("refresh_paper_cache", [False, True])
def test_batch_mixed_null_malformed_all_unknown_and_bad_request(
    caplog: pytest.LogCaptureFixture,
    refresh_paper_cache: bool,
) -> None:
    """Batch misses skip and only the exact unknown-400 normalizes.

    :param pytest.LogCaptureFixture caplog: Captured warning log.
    :param bool refresh_paper_cache: Whether previous cached rows are bypassed.
    :return None: Verifies batch rows and partial results on a rejected request.
    """
    with SemanticScholarClient(
        api_key="", refresh_paper_cache=refresh_paper_cache
    ) as client:
        _disable_pacing(client)
        client._session.post = MagicMock(
            side_effect=[
                _MockResponse(200, [_paper_payload("ok"), None, {"title": "bad"}]),
                _MockResponse(400, {"error": "No valid paper ids given"}),
                _MockResponse(400, {"error": "Bad fields"}),
            ]
        )
        with caplog.at_level(logging.DEBUG):
            assert list(client.get_papers(["ok", "missing", "bad"])) == ["ok"]
        assert client.get_papers(["unknown"]) == {}
        with pytest.raises(SemanticScholarRequestError, match="HTTP 400") as caught:
            client.get_papers(["ok", "rejected"])
        assert list(caught.value.partial_results) == (
            [] if refresh_paper_cache else ["ok"]
        )
    assert "Skipping malformed batch paper" in caplog.text
    assert "Paper metadata skipped 1 malformed records." in caplog.text


def test_get_papers_aggregates_malformed_rows_into_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Metadata batches summarize malformed rows while preserving debug details.

    :param pytest.LogCaptureFixture caplog: Captured provider logs.
    :return None: Verifies one user-facing warning represents the degraded batch.
    """
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.post = MagicMock(
            return_value=_MockResponse(
                200, [{"title": "bad one"}, {"title": "bad two"}]
            )
        )
        with caplog.at_level(logging.DEBUG):
            assert client.get_papers(["one", "two"]) == {}

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG
    ]
    assert warnings == ["Paper metadata skipped 2 malformed records."]
    assert debug_messages.count("Skipping malformed batch paper for one.") == 1
    assert debug_messages.count("Skipping malformed batch paper for two.") == 1


@pytest.mark.parametrize(
    ("paper_id", "url_id"),
    [
        ("seed", "seed"),
        ("DOI:10.1000/example", "10.1000/example"),
        ("arXiv:hep-th/9901001", "arxiv%3Ahep-th/9901001"),
    ],
)
@pytest.mark.parametrize(
    ("method", "relation", "nested"),
    [
        ("get_paper_references", "references", "citedPaper"),
        ("get_paper_citations", "citations", "citingPaper"),
    ],
)
def test_relation_discovery_pages_inline_metadata_without_batch(
    method: str, relation: str, nested: str, paper_id: str, url_id: str
) -> None:
    """Relation discovery pages and deduplicates rows that carry full metadata.

    :param str method: Public relation method.
    :param str relation: Relation endpoint segment.
    :param str nested: Nested response paper key.
    :param str paper_id: Seed identifier, including slash-bearing DOI/arXiv IDs.
    :param str url_id: Expected normalized and quoted identifier in the URL.
    :return None: Verifies paging, inline fields, and the absent batch request.
    """
    pages = [
        _MockResponse(200, _relation_payload(["a", "a"], nested, next_offset=2)),
        _MockResponse(200, _relation_payload(["b", None], nested)),
    ]
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(side_effect=pages)
        client._session.post = MagicMock()
        papers = getattr(client, method)(paper_id, limit=2, raise_on_unavailable=True)
    assert papers == [
        s2.payloads._convert_api_paper(_paper_payload(related_id))
        for related_id in ("a", "b")
    ]
    assert client._session.get.call_count == 2
    for call in client._session.get.call_args_list:
        assert call.args[0] == f"{s2.endpoints.PAPER_BASE_URL}/{url_id}/{relation}"
        assert call.kwargs["params"]["fields"] == ",".join(
            s2.payloads.DEFAULT_PAPER_FIELDS
        )
    client._session.post.assert_not_called()
    assert client._session.get.call_args_list[1].kwargs["params"]["offset"] == 2


def test_relation_retries_only_failed_page_and_every_invocation_is_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed page retries at its offset and no memo suppresses a later call.

    :param pytest.MonkeyPatch monkeypatch: Patches retry delay.
    :return None: Verifies page-local retries and fresh discovery.
    """
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_args, **_kwargs: 0.0)
    responses = [
        _MockResponse(200, _relation_payload(["a"], "citedPaper", next_offset=1)),
        _MockResponse(503),
        _MockResponse(200, _relation_payload(["b"], "citedPaper")),
        _MockResponse(200, _relation_payload(["a"], "citedPaper")),
    ]
    with SemanticScholarClient(api_key="", retry_budget_seconds=0) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(side_effect=responses)
        assert [p.paper_id for p in client.get_paper_references("seed", limit=2)] == [
            "a",
            "b",
        ]
        assert [p.paper_id for p in client.get_paper_references("seed", limit=1)] == [
            "a"
        ]
    offsets = [
        call.kwargs["params"]["offset"] for call in client._session.get.call_args_list
    ]
    assert offsets == [0, 1, 1, 0]


def test_relation_null_missing_and_malformed_contracts() -> None:
    """Null pages are empty while malformed rows never cache.

    :return None: Verifies relation payload distinctions.
    """
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            side_effect=[
                _MockResponse(200, {"data": None}),
                _MockResponse(404),
                _MockResponse(404),
                _MockResponse(200, {"data": [{"unexpected": "shape"}]}),
            ]
        )
        assert client.get_reference_ids("empty", force_refresh=True) == []
        assert client.get_reference_ids("missing", force_refresh=True) == []
        with pytest.raises(SemanticScholarUnavailableError, match="HTTP 404"):
            client.get_paper_references("missing", raise_on_unavailable=True)
        with pytest.raises(TypeError, match="malformed references"):
            client.get_reference_ids("malformed", force_refresh=True)
    assert not s2.disk_cache._reference_cache_path("malformed").exists()


def test_reference_cache_empty_legacy_normalization_corruption_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference cache repairs legacy rows and never caches outages.

    :param pytest.MonkeyPatch monkeypatch: Patches retry delay.
    :return None: Verifies reference cache resilience.
    """
    path = s2.disk_cache._reference_cache_path("seed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "paper_id": "seed",
                "references": ["a", {"paper": {"paperId": "b"}}, "a"],
            }
        ),
        encoding="utf-8",
    )
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        assert client.get_reference_ids("seed") == ["a", "b"]
        assert json.loads(path.read_text())["references"] == ["a", "b"]
        path.write_text("not-json", encoding="utf-8")
        client._session.get = MagicMock(
            return_value=_MockResponse(200, _relation_payload([], "citedPaper"))
        )
        assert client.get_reference_ids("seed") == []
        assert client._session.get.call_args.kwargs["params"]["fields"] == "paperId"

    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 120.0)
    failed = s2.disk_cache._reference_cache_path("failed")
    with SemanticScholarClient(api_key="", retry_budget_seconds=5) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(return_value=_MockResponse(503))
        with pytest.raises(SemanticScholarUnavailableError):
            client.get_reference_ids("failed", force_refresh=True)
    assert not failed.exists()


def test_get_paper_with_references_raises_when_reference_fetch_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference hydration cannot represent an unavailable page as an empty list.

    :param pytest.MonkeyPatch monkeypatch: Makes retry waits unaffordable.
    :return None: Verifies the tolerant metadata default does not hide an outage.
    """
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 120.0)
    with SemanticScholarClient(
        api_key="", refresh_paper_cache=True, retry_budget_seconds=5
    ) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            side_effect=[
                _MockResponse(200, _paper_payload("seed-with-unavailable-references")),
                _MockResponse(503),
            ]
        )
        with pytest.raises(
            SemanticScholarUnavailableError, match="fetching references"
        ):
            client.get_paper("seed-with-unavailable-references", fetch_references=True)


@pytest.mark.parametrize(
    "recent_rows",
    [[], [{**_paper_payload("unconvertible"), "year": "unknown"}, {"paperId": None}]],
)
def test_recommendations_fallback_when_recent_returns_no_papers(
    recent_rows: list[dict[str, object]],
) -> None:
    """A recent pool without usable rows falls back to all-cs in provider order.

    :param list[dict[str, object]] recent_rows: Recent-pool rows yielding no paper.
    :return None: Verifies recommendation fallback without a batch request.
    """
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            side_effect=[
                _MockResponse(200, {"recommendedPapers": recent_rows}),
                _MockResponse(
                    200,
                    {"recommendedPapers": [_paper_payload("b"), _paper_payload("a")]},
                ),
            ]
        )
        client._session.post = MagicMock()
        papers = client.get_recommended_papers("seed", raise_on_unavailable=True)
    assert [paper.paper_id for paper in papers] == ["b", "a"]
    assert "from" not in client._session.get.call_args_list[0].kwargs["params"]
    assert client._session.get.call_args_list[1].kwargs["params"]["from"] == "all-cs"
    client._session.post.assert_not_called()


def test_recommendations_parse_inline_metadata_without_batch_or_cache() -> None:
    """Default recommendations carry their metadata and leave the paper cache alone.

    :return None: Verifies requested fields, order, deduplication and caching.
    """
    rows = [
        _paper_payload("b", title="Second"),
        _paper_payload("a", title="First"),
        _paper_payload("b", title="Repeated"),
    ]
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        fetch = MagicMock(return_value=_MockResponse(200, {"recommendedPapers": rows}))
        client._session.get = fetch
        client._session.post = MagicMock()
        papers = client.get_recommended_papers(
            "seed", limit=200, raise_on_unavailable=True, pool="all-cs"
        )

    assert [(paper.paper_id, paper.title) for paper in papers] == [
        ("b", "Second"),
        ("a", "First"),
    ]
    assert fetch.call_args.kwargs["params"]["fields"] == ",".join(
        s2.payloads.DEFAULT_PAPER_FIELDS
    )
    client._session.post.assert_not_called()
    assert s2.disk_cache._load_cached_paper("a") is None


@pytest.mark.parametrize(
    ("status", "paper_ids", "expected_error"),
    [
        (200, ["b", "a"], None),
        (200, [], None),
        (404, [], SemanticScholarUnavailableError),
        (403, [], SemanticScholarRequestError),
    ],
)
def test_recommendations_can_query_all_cs_directly(
    status: int, paper_ids: list[str], expected_error: type[Exception] | None
) -> None:
    """Direct all-cs discovery keeps provider order and does not try another pool."""
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            return_value=_MockResponse(
                status,
                {"recommendedPapers": [_paper_payload(pid) for pid in paper_ids]},
            )
        )
        client._session.post = MagicMock()
        if expected_error is None:
            papers = client.get_recommended_papers(
                "seed", limit=2, raise_on_unavailable=True, pool="all-cs"
            )
            assert [paper.paper_id for paper in papers] == paper_ids
        else:
            with pytest.raises(expected_error, match=f"HTTP {status}"):
                client.get_recommended_papers(
                    "seed", limit=2, raise_on_unavailable=True, pool="all-cs"
                )

    client._session.get.assert_called_once()
    assert client._session.get.call_args.kwargs["params"] == {
        "fields": ",".join(s2.payloads.DEFAULT_PAPER_FIELDS),
        "limit": 2,
        "from": "all-cs",
    }
    client._session.post.assert_not_called()


def test_custom_projection_adds_paper_id_and_never_overwrites_complete_cache() -> None:
    """Partial discovery rows convert but leave complete metadata intact.

    :return None: Verifies projection and cache behavior.
    """
    complete = Paper(paper_id="p1", title="Complete", year=2020)
    s2.disk_cache._persist_paper(complete, "p1")
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            side_effect=[
                _MockResponse(
                    200,
                    {
                        "recommendedPapers": [
                            None,
                            "malformed",
                            {"paperId": "p1", "title": "Partial"},
                        ]
                    },
                ),
                _MockResponse(200, {"data": [{"paperId": "p1", "title": "Search"}]}),
            ]
        )
        assert (
            client.get_recommended_papers("seed", fields=["title"])[0].title
            == "Partial"
        )
        assert client.search_papers("query", fields=["title"])[0].title == "Search"
    assert (
        client._session.get.call_args_list[0].kwargs["params"]["fields"]
        == "title,paperId"
    )
    assert (
        client._session.get.call_args_list[1].kwargs["params"]["fields"]
        == "title,paperId"
    )
    assert s2.disk_cache._load_cached_paper("p1").title == "Complete"


def test_complete_explicit_projection_is_cacheable_and_recommendations_filter_references() -> (
    None
):
    """Complete explicit records cache while unsupported references are omitted.

    :return None: Verifies complete projection handling.
    """
    fields = [*s2.payloads.DEFAULT_PAPER_FIELDS, "references"]
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            return_value=_MockResponse(
                200, {"recommendedPapers": [_paper_payload("p1")]}
            )
        )
        assert client.get_recommended_papers("seed", fields=fields)[0].paper_id == "p1"
    sent = client._session.get.call_args.kwargs["params"]["fields"].split(",")
    assert "references" not in sent
    assert s2.disk_cache._load_cached_paper("p1") is not None


@pytest.mark.parametrize("complete_projection", [False, True])
def test_custom_projection_recommendations_keep_first_row_per_paper(
    complete_projection: bool,
) -> None:
    """Projected recommendations keep, and cache, only a paper's first row.

    :param bool complete_projection: Whether the projection is cacheable.
    :return None: Verifies order, deduplication and the cached row.
    """
    fields = (
        list(s2.payloads.DEFAULT_PAPER_FIELDS) if complete_projection else ["title"]
    )
    rows = [
        _paper_payload("b", title="Second"),
        _paper_payload("a", title="First"),
        _paper_payload("b", title="Repeated"),
    ]
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            return_value=_MockResponse(200, {"recommendedPapers": rows})
        )
        papers = client.get_recommended_papers("seed", fields=fields, pool="all-cs")

    assert [(paper.paper_id, paper.title) for paper in papers] == [
        ("b", "Second"),
        ("a", "First"),
    ]
    cached = s2.disk_cache._load_cached_paper("b")
    assert (cached.title if cached else None) == (
        "Second" if complete_projection else None
    )


def test_search_paginates_deduplicates_and_validates_query_offsets() -> None:
    """Search follows offsets, honors limits, and rejects invalid inputs.

    :return None: Verifies search pagination and validation.
    """
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            side_effect=[
                _MockResponse(200, {"data": [_paper_payload("a")], "next": 10}),
                _MockResponse(
                    200, {"data": [_paper_payload("a"), _paper_payload("b")]}
                ),
            ]
        )
        papers = client.search_papers(" graph ", limit=2, raise_on_unavailable=True)
        assert [paper.paper_id for paper in papers] == ["a", "b"]
        assert (
            client._session.get.call_args_list[0].kwargs["params"]["query"] == "graph"
        )
        assert client._session.get.call_args_list[1].kwargs["params"]["offset"] == 10
        with pytest.raises(ValueError):
            client.search_papers(" ")
        with pytest.raises(ValueError):
            client.search_papers("x", limit=1001)


def test_strict_and_tolerant_discovery_404_contracts() -> None:
    """Required discovery 404 is empty only for tolerant callers.

    :return None: Verifies caller-selected 404 behavior.
    """
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(return_value=_MockResponse(404))
        assert client.get_paper_references("missing") == []
        with pytest.raises(SemanticScholarUnavailableError, match="HTTP 404"):
            client.get_paper_citations("missing", raise_on_unavailable=True)
        assert client.get_recommended_papers("missing") == []
        with pytest.raises(SemanticScholarUnavailableError, match="HTTP 404"):
            client.search_papers("missing", raise_on_unavailable=True)


def test_retry_attempt_count_status_and_nonretryable_4xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One HTTP request gets 30 sends and actionable diagnostics.

    :param pytest.MonkeyPatch monkeypatch: Removes retry waits.
    :return None: Verifies attempts and deterministic 4xx handling.
    """
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 0.0)
    with SemanticScholarClient(api_key="test", retry_budget_seconds=0) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(return_value=_MockResponse(503))
        with pytest.raises(
            SemanticScholarUnavailableError,
            match=r"fetching p1.*30 attempts.*maximum retry attempts reached.*HTTP 503",
        ):
            client.get_paper("p1", raise_on_unavailable=True)
        assert client._session.get.call_count == 30

        client._session.get.reset_mock()
        client._session.get.return_value = _MockResponse(403)
        with pytest.raises(SemanticScholarRequestError, match="S2_API_KEY"):
            client.get_paper("forbidden", raise_on_unavailable=True)
        client._session.get.assert_called_once()


@pytest.mark.parametrize("error_type", [ValueError, requests.exceptions.InvalidHeader])
@pytest.mark.parametrize("strict", [False, True])
def test_request_configuration_errors_fail_without_retry(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception], strict: bool
) -> None:
    """Invalid request settings surface once for strict and tolerant callers.

    :param pytest.MonkeyPatch monkeypatch: Removes retry waits to expose replay.
    :param type[Exception] error_type: Local timeout or header validation error.
    :param bool strict: Whether service unavailability would normally raise.
    :return None: Verifies the original error escapes without repeated requests.
    """
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 0.0)
    error = error_type("invalid request configuration")
    with SemanticScholarClient(api_key="test", retry_budget_seconds=0) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(side_effect=error)

        with pytest.raises(error_type) as caught:
            client.get_paper("p1", raise_on_unavailable=strict)

        assert caught.value is error
        client._session.get.assert_called_once()


@pytest.mark.parametrize(
    "error_type",
    [
        requests.Timeout,
        requests.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ContentDecodingError,
    ],
)
def test_transient_transport_errors_retry(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    """Transient sends and response transfer failures can recover on retry.

    :param pytest.MonkeyPatch monkeypatch: Removes retry waits.
    :param type[Exception] error_type: Retryable transport or response error.
    :return None: Verifies one retry returns the successful paper.
    """
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 0.0)
    with SemanticScholarClient(api_key="test", retry_budget_seconds=0) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            side_effect=[
                error_type("transient transport failure"),
                _MockResponse(200, _paper_payload("p1")),
            ]
        )

        assert client.get_paper("p1", raise_on_unavailable=True).paper_id == "p1"
        assert client._session.get.call_count == 2


@pytest.mark.parametrize(
    ("delay", "failed_attempts"),
    [(30.0, 2), (15.0, 3)],
    ids=["single-wait", "cumulative-waits"],
)
def test_retry_waits_warn_once_after_thirty_cumulative_seconds(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    delay: float,
    failed_attempts: int,
) -> None:
    """Retry delays disclose one actionable warning after a long accumulated wait.

    :param pytest.MonkeyPatch monkeypatch: Replaces retry timing and sleeping.
    :param pytest.LogCaptureFixture caplog: Captured retry disclosure.
    :param float delay: Fixed delay after every failed attempt.
    :param int failed_attempts: Number of transient failures before success.
    :return None: Verifies quiet retry logs do not hide prolonged recovery.
    """
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: delay)
    monkeypatch.setattr(s2.client.time, "sleep", lambda _seconds: None)
    with SemanticScholarClient(api_key="test", retry_budget_seconds=0) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            side_effect=[_MockResponse(503)] * failed_attempts
            + [_MockResponse(200, _paper_payload("p1"))]
        )
        with caplog.at_level(logging.WARNING):
            assert client.get_paper("p1", raise_on_unavailable=True).paper_id == "p1"

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert f"waiting {delay:.0f}s (30s total retry delay)" in warnings[0]


def test_retry_after_cap_and_nonfinite_values() -> None:
    """Retry-After parsing is finite and capped at 300 seconds.

    :return None: Verifies delay parsing.
    """
    assert (
        s2.retry.retry_after_seconds(_MockResponse(429, headers={"Retry-After": "999"}))
        == 300
    )
    assert (
        s2.retry.retry_after_seconds(
            _MockResponse(429, headers={"Retry-After": "inf"}), default=2
        )
        == 2
    )
    assert (
        s2.retry.retry_after_seconds(
            _MockResponse(429, headers={"Retry-After": "bad"}), default=2
        )
        == 2
    )


def test_recovery_budget_counts_failed_request_wait_retry_and_excludes_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only recovery work consumes the shared outer allowance.

    :param pytest.MonkeyPatch monkeypatch: Installs a deterministic clock.
    :return None: Verifies recovery accounting.
    """
    clock = _Clock()
    monkeypatch.setattr(s2.client, "monotonic", clock.monotonic)
    monkeypatch.setattr(s2.client.time, "sleep", clock.sleep)
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 2.0)

    responses = iter(
        [_MockResponse(503), _MockResponse(200, _paper_payload("recovered"))]
    )

    def timed_get(*_args: Any, **_kwargs: Any) -> _MockResponse:
        """Spend one second in each HTTP send.

        :return _MockResponse: Next configured response.
        """
        clock.now += 1.0
        return next(responses)

    with SemanticScholarClient(api_key="", retry_budget_seconds=10) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(side_effect=timed_get)
        with client.candidate_operation_scope():
            assert client.get_paper("recovered", raise_on_unavailable=True) is not None
            assert client._candidate_operation.state.recovery_seconds == pytest.approx(
                4.0
            )

            before = client._candidate_operation.state.recovery_seconds
            client._session.get = MagicMock(
                side_effect=lambda *_a, **_k: (
                    setattr(clock, "now", clock.now + 50.0)
                    or _MockResponse(200, _paper_payload("healthy"))
                )
            )
            assert client.get_paper("healthy", raise_on_unavailable=True) is not None
            assert client._candidate_operation.state.recovery_seconds == before


def test_spent_budget_blocks_network_but_cached_reads_still_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global recovery exhaustion stops uncached sends but allows cache hits.

    :param pytest.MonkeyPatch monkeypatch: Installs a deterministic clock.
    :return None: Verifies shared-budget exhaustion.
    """
    clock = _Clock()
    monkeypatch.setattr(s2.client, "monotonic", clock.monotonic)
    monkeypatch.setattr(s2.client.time, "sleep", clock.sleep)
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 2.0)
    cached = Paper(paper_id="cached", title="Cached", year=2020)
    s2.disk_cache._persist_paper(cached, "cached")

    def fail(*_args: Any, **_kwargs: Any) -> _MockResponse:
        """Spend one second then return a transient response.

        :return _MockResponse: HTTP 503 response.
        """
        clock.now += 1.0
        return _MockResponse(503)

    with SemanticScholarClient(api_key="", retry_budget_seconds=3) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(side_effect=fail)
        with client.candidate_operation_scope():
            with pytest.raises(
                SemanticScholarUnavailableError, match="budget exhausted"
            ):
                client.get_paper("first", raise_on_unavailable=True)
            assert (
                client.get_paper("cached", raise_on_unavailable=True).title == "Cached"
            )
            with pytest.raises(SemanticScholarUnavailableError, match="0 attempts"):
                client.get_paper("second", raise_on_unavailable=True)
        assert client._session.get.call_count == 1


def test_oversized_retry_after_stops_only_request_without_spending_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unaffordable cooldown leaves time for another healthy request.

    :param pytest.MonkeyPatch monkeypatch: Installs a deterministic clock.
    :return None: Verifies oversized wait handling.
    """
    clock = _Clock()
    monkeypatch.setattr(s2.client, "monotonic", clock.monotonic)
    monkeypatch.setattr(s2.client.time, "sleep", clock.sleep)

    def first_then_healthy(*_args: Any, **_kwargs: Any) -> _MockResponse:
        """Return one rate limit followed by a healthy response.

        :return _MockResponse: Next response.
        """
        clock.now += 1.0
        if clock.now == 1.0:
            return _MockResponse(429, headers={"Retry-After": "120"})
        return _MockResponse(200, _paper_payload("healthy"))

    with SemanticScholarClient(api_key="", retry_budget_seconds=5) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(side_effect=first_then_healthy)
        with client.candidate_operation_scope():
            with pytest.raises(
                SemanticScholarUnavailableError, match="next wait exceeds"
            ):
                client.get_paper("limited", raise_on_unavailable=True)
            assert client._candidate_operation.state.recovery_seconds == 1.0
            assert client.get_paper("healthy", raise_on_unavailable=True) is not None
        assert client._session.get.call_count == 2


def test_singleton_credentials_lifecycle_and_retry_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Singleton replacement and retry defaults follow credentials.

    :param pytest.MonkeyPatch monkeypatch: Changes the environment API key.
    :return None: Verifies lifecycle and defaults.
    """
    reset_client()
    monkeypatch.delenv("S2_API_KEY", raising=False)
    anonymous = get_client()
    assert anonymous.retry_budget_seconds == 90.0
    assert anonymous.requests_per_second == API_CONFIG.requests_per_second
    monkeypatch.setenv("S2_API_KEY", "key")
    keyed = get_client()
    assert keyed is not anonymous
    assert keyed.retry_budget_seconds == 0.0
    assert keyed._session.headers["x-api-key"] == "key"
    keyed.close()
    replacement = get_client()
    assert replacement is not keyed
    reset_client()


@pytest.mark.parametrize("budget", [-1.0, float("inf"), float("nan")])
def test_retry_budget_rejects_invalid_values(budget: float) -> None:
    """Recovery budgets must be finite and non-negative.

    :param float budget: Invalid budget value.
    :return None: Verifies validation.
    """
    with pytest.raises(ValueError):
        SemanticScholarClient(api_key="", retry_budget_seconds=budget)


def test_malformed_json_retries_and_batch_404_is_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode failures and batch 404 retry without restarting batches.

    :param pytest.MonkeyPatch monkeypatch: Removes retry waits.
    :return None: Verifies retryable response cases.
    """
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 0.0)
    with SemanticScholarClient(api_key="test", retry_budget_seconds=0) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            side_effect=[
                _BrokenJsonResponse(200),
                _MockResponse(200, _paper_payload("p1")),
            ]
        )
        assert client.get_paper("p1", raise_on_unavailable=True).paper_id == "p1"
        assert client._session.get.call_count == 2

        client._session.post = MagicMock(
            side_effect=[_MockResponse(404), _MockResponse(200, [_paper_payload("p2")])]
        )
        assert (
            client.get_papers(["p2"], raise_on_unavailable=True)["p2"].title == "Paper"
        )
        assert client._session.post.call_count == 2


def test_batch_404_retries_are_bounded_for_keyed_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistent transient batch 404 stops before the general retry limit.

    :param pytest.MonkeyPatch monkeypatch: Removes retry waits.
    :return None: Verifies an API key cannot turn a persistent 404 into 30 retries.
    """
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 0.0)
    with SemanticScholarClient(api_key="test", retry_budget_seconds=0) as client:
        _disable_pacing(client)
        client._session.post = MagicMock(return_value=_MockResponse(404))
        with pytest.raises(
            SemanticScholarUnavailableError,
            match=(
                r"batch fetching 1 papers.*after 3 attempts.*"
                r"maximum transient not-found attempts reached.*HTTP 404"
            ),
        ):
            client.get_papers(["p2"], raise_on_unavailable=True)

    assert client._session.post.call_count == 3


def test_batch_404_does_not_restart_retry_budget_after_other_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late 404 terminates recovery instead of starting 30 further retries.

    :param pytest.MonkeyPatch monkeypatch: Removes retry waits.
    :return None: Verifies the 404 ceiling still applies after other failures.
    """
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 0.0)
    with SemanticScholarClient(api_key="test", retry_budget_seconds=0) as client:
        _disable_pacing(client)
        client._session.post = MagicMock(
            side_effect=[
                _MockResponse(503),
                _MockResponse(503),
                _MockResponse(503),
                _MockResponse(404),
            ]
        )
        with pytest.raises(
            SemanticScholarUnavailableError,
            match=(
                r"batch fetching 1 papers.*after 4 attempts.*"
                r"maximum transient not-found attempts reached.*HTTP 404"
            ),
        ):
            client.get_papers(["p2"], raise_on_unavailable=True)

    assert client._session.post.call_count == 4


def test_relation_records_bypass_and_never_write_paper_cache() -> None:
    """Inline relation rows replace warm cache reads and leave the paper cache alone.

    :return None: Verifies inline metadata wins and nothing is persisted.
    """
    s2.disk_cache._persist_paper(
        Paper(paper_id="warm", title="Warm", year=2020), "warm"
    )
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            return_value=_MockResponse(
                200, _relation_payload(["warm", "cold"], "citedPaper")
            )
        )
        client._session.post = MagicMock()
        papers = client.get_paper_references("seed")
    client._session.post.assert_not_called()
    assert [(paper.paper_id, paper.title) for paper in papers] == [
        ("warm", "Paper"),
        ("cold", "Paper"),
    ]
    cached = s2.disk_cache._load_cached_paper("warm")
    assert cached is not None
    assert cached.title == "Warm"
    assert s2.disk_cache._load_cached_paper("cold") is None


def test_cache_operation_scope_uses_outer_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested endpoint work acquires only the outer cache operation lock.

    :param pytest.MonkeyPatch monkeypatch: Replaces the cache lock.
    :return None: Verifies lock nesting.
    """
    entered: list[str] = []

    class _Lock:
        """Record lock entry and exit."""

        def __enter__(self) -> None:
            """Record entry.

            :return None: Enters the lock.
            """
            entered.append("enter")

        def __exit__(self, *_args: Any) -> None:
            """Record exit.

            :return None: Exits the lock.
            """
            entered.append("exit")

    monkeypatch.setattr(s2.client, "cache_operation_lock", lambda _path: _Lock())
    with SemanticScholarClient(api_key="") as client:
        with client.candidate_operation_scope():
            with client.candidate_operation_scope():
                pass
    assert entered == ["enter", "exit"]


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("initial_pacing", [0.0, 150.0])
def test_recovery_charges_retry_pacing_from_zero_but_not_initial_pacing(
    monkeypatch: pytest.MonkeyPatch, status: int, initial_pacing: float
) -> None:
    """Retry pacing is charged even when its monotonic start is exactly zero.

    :param pytest.MonkeyPatch monkeypatch: Replaces time and backoff.
    :param int status: Transient response returned by the first two requests.
    :param float initial_pacing: Healthy first-attempt wait, excluded from recovery.
    :return None: Checks actual paced elapsed time and the shared recovery charge.
    """
    clock = _Clock()
    monkeypatch.setattr(s2.client, "monotonic", clock.monotonic)
    monkeypatch.setattr(s2.client.time, "sleep", clock.sleep)
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 0.0)
    with SemanticScholarClient(api_key="", retry_budget_seconds=10) as client:
        client.requests_per_second = 0.5
        client._next_request_time = initial_pacing
        client._session.get = MagicMock(
            side_effect=[
                _MockResponse(status),
                _MockResponse(status),
                _MockResponse(200, _paper_payload("paced")),
            ]
        )
        with client.candidate_operation_scope():
            assert client.get_paper("paced", raise_on_unavailable=True) is not None
            assert client._candidate_operation.state.recovery_seconds == 4.0
        assert client._session.get.call_count == 3
    assert clock.now == initial_pacing + 4.0


def test_recovery_timeout_has_positive_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tiny remaining allowance never passes a zero timeout to Requests.

    :param pytest.MonkeyPatch monkeypatch: Replaces time and backoff.
    :return None: Checks the initial transport timeout and positive recovery floor.
    """
    clock = _Clock()
    monkeypatch.setattr(s2.client, "monotonic", clock.monotonic)
    monkeypatch.setattr(s2.client.time, "sleep", clock.sleep)
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 0.0)
    with SemanticScholarClient(
        api_key="", timeout=30, retry_budget_seconds=0.0005
    ) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            side_effect=[_MockResponse(503), _MockResponse(200, _paper_payload())]
        )
        assert client.get_paper("p1", raise_on_unavailable=True) is not None
        assert [
            call.kwargs["timeout"] for call in client._session.get.call_args_list
        ] == [
            30,
            0.001,
        ]


@pytest.mark.parametrize("api_key", ["", "configured-key"])
def test_nested_recovery_budget_excludes_healthy_work_and_resets(
    monkeypatch: pytest.MonkeyPatch, api_key: str
) -> None:
    """Nested calls share overrides, while healthy work and later scopes remain independent.

    :param pytest.MonkeyPatch monkeypatch: Replaces time, backoff, and HTTP responses.
    :param str api_key: Anonymous or authenticated access with an explicit budget.
    :return None: Checks sharing, timeout capping, healthy work, and scope reset.
    """
    clock = _Clock()
    monkeypatch.setattr(s2.client, "monotonic", clock.monotonic)
    monkeypatch.setattr(s2.client.time, "sleep", clock.sleep)
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 1.0)
    responses = iter(
        [
            (1.0, _MockResponse(503)),
            (1.0, _MockResponse(200, _paper_payload("recovered"))),
            (150.0, _MockResponse(200, _paper_payload("healthy"))),
            (1.0, _MockResponse(503)),
            (1.0, _MockResponse(200, _paper_payload("fresh"))),
        ]
    )

    def respond(*_args: Any, **_kwargs: Any) -> _MockResponse:
        """Advance by each request's elapsed time before returning its response.

        :param Any _args: Ignored transport positional arguments.
        :param Any _kwargs: Ignored transport keyword arguments.
        :return _MockResponse: Next timed HTTP response.
        """
        elapsed, response = next(responses)
        clock.now += elapsed
        return response

    with SemanticScholarClient(api_key=api_key, retry_budget_seconds=4) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(side_effect=respond)
        with client.candidate_operation_scope():
            assert client.get_paper("recovered", raise_on_unavailable=True) is not None
            assert client._candidate_operation.state.recovery_seconds == 3.0
            assert client.get_paper("healthy", raise_on_unavailable=True) is not None
            assert client._candidate_operation.state.recovery_seconds == 3.0
            with client.candidate_operation_scope():
                with pytest.raises(
                    SemanticScholarUnavailableError, match="budget exhausted"
                ):
                    client.get_paper("failed", raise_on_unavailable=True)
            with pytest.raises(SemanticScholarUnavailableError, match="0 attempts"):
                client.get_paper("blocked", raise_on_unavailable=True)
        assert not hasattr(client._candidate_operation, "state")
        assert client.get_paper("fresh", raise_on_unavailable=True) is not None
        assert [
            call.kwargs["timeout"] for call in client._session.get.call_args_list
        ] == [
            client.timeout,
            2.0,
            client.timeout,
            client.timeout,
            client.timeout,
        ]


def test_interrupting_recovery_releases_collection_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted retry propagates and leaves the next collection usable.

    :param pytest.MonkeyPatch monkeypatch: Replaces retry sleeps and backoff.
    :return None: Checks interruption, no extra send, and fresh subsequent work.
    """
    monkeypatch.setattr(s2.retry, "_jittered_backoff", lambda *_a, **_k: 1.0)
    interrupted_sleep = MagicMock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr(s2.client.time, "sleep", interrupted_sleep)
    with SemanticScholarClient(api_key="", retry_budget_seconds=5) as client:
        _disable_pacing(client)
        client._session.get = MagicMock(return_value=_MockResponse(503))
        with pytest.raises(KeyboardInterrupt):
            client.get_paper("interrupted", raise_on_unavailable=True)
        client._session.get.assert_called_once()
        interrupted_sleep.assert_called_once_with(1.0)
        assert not hasattr(client._candidate_operation, "state")
        client._session.get = MagicMock(
            return_value=_MockResponse(200, _paper_payload())
        )
        assert client.get_paper("p1", raise_on_unavailable=True) is not None


@pytest.mark.parametrize(
    ("method", "nested_key"),
    [
        ("get_paper_references", "citedPaper"),
        ("get_paper_citations", "citingPaper"),
    ],
)
def test_repeated_relation_discovery_refetches_pages_only(
    method: str, nested_key: str
) -> None:
    """Every relation call reads fresh pages and never hydrates through a batch.

    :param str method: Public relation method.
    :param str nested_key: Relation payload key.
    :return None: Verifies one GET per call and no batch or paper-cache write.
    """
    raw_id = "a" * 40
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            return_value=_MockResponse(200, _relation_payload([raw_id], nested_key))
        )
        client._session.post = MagicMock()
        for _ in range(2):
            assert [
                paper.paper_id
                for paper in getattr(client, method)(
                    "seed", limit=1, raise_on_unavailable=True
                )
            ] == [raw_id]

    assert client._session.get.call_count == 2
    client._session.post.assert_not_called()
    assert s2.disk_cache._load_cached_paper(raw_id) is None


@pytest.mark.parametrize(
    ("method", "nested_key"),
    [
        ("get_paper_references", "citedPaper"),
        ("get_paper_citations", "citingPaper"),
    ],
)
def test_relation_records_equal_batch_records_for_the_same_payload(
    method: str, nested_key: str
) -> None:
    """Inline relation rows convert exactly as batch hydration of the same records.

    :param str method: Public relation method.
    :param str nested_key: Relation payload key.
    :return None: Verifies identical records, order, and malformed-row skipping.
    """
    records: list[dict[str, Any]] = [
        _paper_payload("a", doi="10.1000/A", arxiv_id="2401.00001v2"),
        {
            **_paper_payload("b", doi="10.48550/arXiv.2401.00002"),
            "journal": {"name": "Nature", "volume": "1"},
        },
        {
            **_paper_payload("c"),
            "journal": {"name": "arXiv", "volume": "abs/2401.00003"},
            "publicationVenue": {"name": "NeurIPS"},
            "venue": "",
        },
        {
            **_paper_payload("d"),
            "title": None,
            "abstract": None,
            "year": None,
            "authors": [{"name": "Grace"}],
            "citationCount": None,
            "fieldsOfStudy": None,
        },
        {**_paper_payload("e"), "year": "2020"},
    ]
    ids = ["a", "b", "c", "d", "e"]
    pages = [
        _MockResponse(200, _relation_page(records[:3], nested_key, next_offset=3)),
        _MockResponse(
            200,
            _relation_page(
                [
                    _paper_payload("a", title="Duplicate row"),
                    *records[3:],
                    {"paperId": None},
                ],
                nested_key,
            ),
        ),
    ]
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(side_effect=pages)
        client._session.post = MagicMock()
        papers = getattr(client, method)("seed", limit=10, raise_on_unavailable=True)
    client._session.post.assert_not_called()
    assert all(s2.disk_cache._load_cached_paper(paper_id) is None for paper_id in ids)

    with SemanticScholarClient(api_key="", refresh_paper_cache=True) as client:
        _disable_pacing(client)
        client._session.post = MagicMock(return_value=_MockResponse(200, records))
        batch = client.get_papers(ids)

    assert [paper.paper_id for paper in papers] == ["a", "b", "c", "d"]
    assert papers == [batch[paper_id] for paper_id in ids if paper_id in batch]


@pytest.mark.parametrize(
    ("relation", "limit", "pages", "complete"),
    [
        ("references", 10, [(["a", "b"], 2), (["c", None, "a"], None)], True),
        ("references", 10, [([], None)], True),
        ("references", 3, [(["a", "b"], 2), (["c"], None)], False),
        ("references", 2, [(["a", "b"], 2), (["c"], None)], False),
        ("citations", 10, [(["a", "b"], None)], False),
    ],
)
def test_complete_reference_listing_populates_reference_cache(
    relation: str,
    limit: int,
    pages: list[tuple[list[str | None], int | None]],
    complete: bool,
) -> None:
    """Only an exhausted reference listing stores what a reference-ID fetch would.

    A listing that fills its limit may have stopped before the provider's last
    row, so it leaves the cache untouched, as citations always do.

    :param str relation: Relation endpoint under test.
    :param int limit: Discovery limit.
    :param list pages: Page IDs and next offsets returned by the provider.
    :param bool complete: Whether the listing is known to be exhaustive.
    :return None: Verifies the persisted list and the absent follow-up request.
    """
    nested_key = {"references": "citedPaper", "citations": "citingPaper"}[relation]

    def responses(key: str) -> list[_MockResponse]:
        return [
            _MockResponse(200, _relation_payload(ids, key, next_offset=next_offset))
            for ids, next_offset in pages
        ]

    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(side_effect=responses("citedPaper"))
        reference_ids = client.get_reference_ids("seed", force_refresh=True)
        s2.disk_cache._reference_cache_path("seed").unlink()

        client._session.get = MagicMock(side_effect=responses(nested_key))
        getattr(client, f"get_paper_{relation}")(
            "seed", limit=limit, raise_on_unavailable=True
        )
        assert client.get_cached_reference_ids("seed") == (
            reference_ids if complete else None
        )
        if complete:
            client._session.get = MagicMock()
            assert client.get_reference_ids("seed") == reference_ids
            client._session.get.assert_not_called()


@pytest.mark.parametrize(("malformed", "level"), [(1, "DEBUG"), (3, "WARNING")])
def test_unconvertible_relation_rows_warn_only_when_most_are_dropped(
    caplog: pytest.LogCaptureFixture, malformed: int, level: str
) -> None:
    """A few unconvertible rows in a listing are routine; most of them are not.

    :param pytest.LogCaptureFixture caplog: Captured client logs.
    :param int malformed: Rows of five whose metadata cannot be converted.
    :param str level: Expected level of the skip message.
    :return None: Verifies the kept papers and the message level.
    """
    rows = [
        {**_paper_payload(f"p{index}"), "year": "unknown"}
        if index < malformed
        else _paper_payload(f"p{index}")
        for index in range(5)
    ]
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            return_value=_MockResponse(200, _relation_page(rows, "citingPaper"))
        )
        with caplog.at_level(logging.DEBUG):
            papers = client.get_paper_citations("seed", limit=10)

    assert len(papers) == 5 - malformed
    [record] = [
        record for record in caplog.records if "without resolvable" in record.message
    ]
    assert record.levelname == level
    assert f"skipping {malformed} of 5 rows" in record.message


@pytest.mark.parametrize(("malformed", "level"), [(1, "DEBUG"), (2, "WARNING")])
def test_unusable_recommendation_rows_warn_only_when_most_are_dropped(
    caplog: pytest.LogCaptureFixture, malformed: int, level: str
) -> None:
    """Recommendations apply the relation listings' skip-warning rule.

    :param pytest.LogCaptureFixture caplog: Captured client logs.
    :param int malformed: Rows of three without a usable record.
    :param str level: Expected level of the skip message.
    :return None: Verifies the kept papers and the message level.
    """
    unusable: list[dict[str, object]] = [
        {**_paper_payload("p0"), "year": "unknown"},
        {"paperId": None},
    ]
    rows = [*unusable[:malformed], _paper_payload("p2")]
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._session.get = MagicMock(
            return_value=_MockResponse(200, {"recommendedPapers": rows})
        )
        with caplog.at_level(logging.DEBUG):
            papers = client.get_recommended_papers("seed", pool="all-cs")

    assert [paper.paper_id for paper in papers] == ["p2"]
    [record] = [
        record for record in caplog.records if "without resolvable" in record.message
    ]
    assert record.levelname == level
    assert f"skipping {malformed} of {malformed + 1} rows" in record.message


def test_complete_reference_cache_does_not_cap_fresh_limited_discovery() -> None:
    """A complete reference cache must not suppress a bounded current discovery call.

    :return None: Verifies reference enrichment remains independent from discovery.
    """
    seed = "arxiv:1706.03762"
    with SemanticScholarClient(api_key="") as client:
        _disable_pacing(client)
        client._persist_reference_cache_entry(
            s2.disk_cache._reference_cache_path(seed), seed, ["old-a", "old-b", "old-c"]
        )
        client._session.get = MagicMock(
            return_value=_MockResponse(
                200, _relation_payload(["current"], "citedPaper")
            )
        )
        client._session.post = MagicMock()

        assert client.get_reference_ids(seed) == ["old-a", "old-b", "old-c"]
        client._session.get.assert_not_called()
        assert [
            paper.paper_id
            for paper in client.get_paper_references(
                seed, limit=1, raise_on_unavailable=True
            )
        ] == ["current"]
        assert client._session.get.call_args.kwargs["params"]["limit"] == 1

    client._session.post.assert_not_called()
    assert client.get_cached_reference_ids(seed) == ["old-a", "old-b", "old-c"]


@pytest.mark.parametrize(
    ("failure", "raise_on_unavailable", "expected_ids"),
    [
        ("fallback", False, []),
        ("fallback", True, None),
        ("later_page", False, []),
        ("later_page", True, None),
    ],
)
def test_failed_required_discovery_preserves_existing_persistent_records(
    failure: str, raise_on_unavailable: bool, expected_ids: list[str] | None
) -> None:
    """Required acquisition failures cannot replace previously successful persistent data.

    :param str failure: Failed recommendation fallback or later relation page.
    :param bool raise_on_unavailable: Whether the caller requires an exception.
    :param list[str] | None expected_ids: Tolerant materialized IDs, if any.
    :return None: Verifies strict exceptions, tolerant empties, and cache preservation.
    """
    with SemanticScholarClient(api_key="", retry_budget_seconds=5) as client:
        _disable_pacing(client)
        s2.disk_cache._persist_paper(
            Paper(paper_id="warm", title="Warm metadata", year=2020), "warm"
        )
        client._persist_reference_cache_entry(
            s2.disk_cache._reference_cache_path("seed"), "seed", ["old-reference"]
        )

        if failure == "fallback":
            client._session.get = MagicMock(
                side_effect=[
                    _MockResponse(200, {"recommendedPapers": []}),
                    _MockResponse(503, headers={"Retry-After": "120"}),
                ]
            )
            client._session.post = MagicMock(
                side_effect=AssertionError("failed fallback cannot hydrate metadata")
            )

            def discover() -> list[Paper]:
                """Fetch recommendations whose all-cs fallback cannot recover.

                :return list[Paper]: Empty recommendations for tolerant callers.
                """
                return client.get_recommended_papers(
                    "seed", limit=1, raise_on_unavailable=raise_on_unavailable
                )

        else:
            client._session.get = MagicMock(
                side_effect=[
                    _MockResponse(
                        200,
                        _relation_payload(["cold"], "citedPaper", next_offset=1),
                    ),
                    _MockResponse(503, headers={"Retry-After": "120"}),
                ]
            )
            client._session.post = MagicMock(
                side_effect=AssertionError(
                    "failed page cannot hydrate partial metadata"
                )
            )

            def discover() -> list[Paper]:
                """Fetch a relation whose second discovery page cannot recover.

                :return list[Paper]: Empty relation records for tolerant callers.
                """
                return client.get_paper_references(
                    "seed", limit=2, raise_on_unavailable=raise_on_unavailable
                )

        if raise_on_unavailable:
            with pytest.raises(SemanticScholarUnavailableError):
                discover()
        else:
            assert [paper.paper_id for paper in discover()] == expected_ids

        cached = s2.disk_cache._load_cached_paper("warm")
        assert cached is not None
        assert cached.title == "Warm metadata"
        assert client.get_cached_reference_ids("seed") == ["old-reference"]

    client._session.post.assert_not_called()
