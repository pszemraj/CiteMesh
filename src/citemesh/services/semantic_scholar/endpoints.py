"""Semantic Scholar endpoint methods, as a mixin over the transport client.

Owns the public API surface -- paper lookup, batch fetch, citation/reference
relations, reference IDs, recommendations, and search -- together with the
endpoint URLs they address and the per-endpoint failure contracts expressed in
their nested callbacks. Everything here is request *composition*: pacing,
retrying, and the shared HTTP/SDK plumbing stay on the host client.

:class:`_EndpointsMixin` is not usable on its own. Its host must provide:

- ``client``: the Semantic Scholar SDK handle used for relation pagination.
- ``refresh_paper_cache``: whether persisted paper metadata is bypassed on read.
- ``_request_json`` / ``_request_json_once``: REST transport, with and without
  its own retry budget.
- ``_call_with_retries``: SDK transport retry orchestration.
- ``_is_rate_limit_error``: HTTP 429 classification for failure messages.
- ``_convert_api_paper`` / ``_convert_recommendation``: payload conversion.
- ``_persist_reference_cache_entry``: reference-cache writes.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

from semanticscholar.SemanticScholarException import (
    BadQueryParametersException,
    ObjectNotFoundException,
)

from citemesh.core import API_CONFIG, Paper
from citemesh.core.paper_ids import normalize_paper_id

from . import disk_cache, payloads
from .errors import (
    SemanticScholarRequestError,
    SemanticScholarUnavailableError,
    _CandidateOperationSkippedError,
    _FailureDomain,
    _raise_request_error,
    _SemanticScholarResponseContractError,
)

logger = logging.getLogger(__name__)

RECOMMENDATION_BASE_URL = (
    "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
)
PAPER_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
SEARCH_BASE_URL = f"{PAPER_BASE_URL}/search"
GRAPH_RELATION_PAGE_SIZE = 1000
RECOMMENDATION_MAX_RESULTS = 500
SEARCH_MAX_RESULTS = 1000
SEARCH_PAGE_SIZE = 100


class _EndpointsMixin:
    """Semantic Scholar endpoint methods shared by the transport client."""

    def get_paper(
        self,
        paper_id: str,
        fetch_references: bool = False,
        *,
        raise_on_unavailable: bool = False,
    ) -> Paper | None:
        """
        Fetch a paper by ID, reusing persisted metadata before calling the API.

        :param str paper_id: Paper identifier (DOI, arXiv ID, or S2 ID)
        :param bool fetch_references: Whether to fetch reference list (slower)
        :param bool raise_on_unavailable: When ``True``, exhausted retries raise
            :class:`SemanticScholarUnavailableError` instead of returning
            ``None``, so callers can distinguish "not found" from "API down".
        :return Paper | None: Paper object or None if not found
        :raises TypeError: If a present API payload cannot be converted to a paper.
        :raises SemanticScholarRequestError: If Semantic Scholar rejects the
            request because its parameters or credentials are invalid.
        """
        if not paper_id or not isinstance(paper_id, str):
            raise ValueError(f"Invalid paper ID: {paper_id}")

        paper_id = normalize_paper_id(paper_id)
        if fetch_references:
            paper = self.get_paper(paper_id, raise_on_unavailable=raise_on_unavailable)
            if paper is not None:
                try:
                    paper.references = self.get_reference_ids(paper_id)
                except SemanticScholarUnavailableError as exc:
                    if raise_on_unavailable:
                        raise
                    logger.error("Failed to fetch references for %s: %s", paper_id, exc)
                    return None
            return paper

        if not self.refresh_paper_cache:
            cached_paper = disk_cache._load_cached_paper(paper_id)
            if cached_paper is not None:
                return cached_paper
        # The SDK silently maps unrecognized HTTP statuses to an empty paper.
        # The graph endpoint requires literal slashes in DOI and legacy arXiv IDs.
        api_paper = self._request_json(
            f"{PAPER_BASE_URL}/{quote(paper_id, safe='/')}",
            {"fields": ",".join(payloads._default_paper_fields())},
            failure_domain=_FailureDomain.PAPER_METADATA,
            raise_on_unavailable=raise_on_unavailable,
            context=f"fetching {paper_id}",
        )
        if api_paper is None:
            return None
        try:
            paper = payloads._convert_payload_paper(
                api_paper,
                category_keys=("fields", "fieldsOfStudy"),
            )
        except Exception as exc:
            raise _SemanticScholarResponseContractError(
                f"Semantic Scholar returned a malformed paper payload for "
                f"{paper_id}: {exc}"
            ) from exc
        if paper is None:
            raise _SemanticScholarResponseContractError(
                f"Semantic Scholar returned a malformed paper payload without a "
                f"paper ID for {paper_id}."
            )
        disk_cache._persist_paper(paper, paper_id)
        return paper

    def get_papers(
        self, paper_ids: Sequence[str], *, raise_on_unavailable: bool = False
    ) -> dict[str, Paper]:
        """Reuse cached metadata and fetch missing papers through the batch endpoint.

        :param Sequence[str] paper_ids: Paper identifiers (DOI, arXiv ID, or S2 IDs).
        :param bool raise_on_unavailable: Whether exhausted retries raise instead
            of returning cached results without further per-paper requests.
        :return dict[str, Paper]: Mapping of normalized requested IDs to fetched papers.
        :raises ValueError: If an ID is missing/not a string or over 500 uncached
            IDs would require one batch request.
        :raises SemanticScholarRequestError: If Semantic Scholar rejects the
            batch request itself (non-429 HTTP 4xx), regardless of
            ``raise_on_unavailable`` — a rejected request is a caller error,
            not an availability failure.
        """
        normalized_ids: list[str] = []
        for raw_paper_id in paper_ids:
            if not raw_paper_id or not isinstance(raw_paper_id, str):
                raise ValueError(f"Invalid paper ID: {raw_paper_id}")
            normalized_ids.append(normalize_paper_id(raw_paper_id))

        normalized_ids = list(dict.fromkeys(normalized_ids))
        if not normalized_ids:
            return {}

        cached = {
            paper_id: paper
            for paper_id in normalized_ids
            if not self.refresh_paper_cache
            and (paper := disk_cache._load_cached_paper(paper_id)) is not None
        }
        normalized_ids = [
            paper_id for paper_id in normalized_ids if paper_id not in cached
        ]
        if not normalized_ids:
            return cached

        if len(normalized_ids) > 500:
            raise ValueError("A batch request supports at most 500 paper IDs.")

        def _operation() -> dict[str, Paper]:
            """Fetch positional batch results, preserving authoritative null entries.

            :return dict[str, Paper]: Successful results keyed by requested ID.
            """
            api_papers = self._request_json_once(
                f"{PAPER_BASE_URL}/batch",
                {"fields": ",".join(payloads._default_paper_fields())},
                context="batch fetching papers",
                payload={"ids": normalized_ids},
            )
            if api_papers is None:
                return {}
            if not isinstance(api_papers, list) or len(api_papers) != len(
                normalized_ids
            ):
                raise _SemanticScholarResponseContractError(
                    "Semantic Scholar batch response must contain one result per requested ID."
                )
            matched: dict[str, Paper] = {}
            for requested_id, api_paper in zip(normalized_ids, api_papers):
                if api_paper is None:
                    continue
                paper = self._convert_api_paper(api_paper)
                if paper is None:
                    logger.warning(
                        "Skipping malformed batch paper for %s.", requested_id
                    )
                    continue
                matched[requested_id] = paper
                disk_cache._persist_paper(paper, requested_id)
            return matched

        def _final_failure(exc: Exception) -> None:
            """Stop the batch after exhaustion without restarting retries per ID.

            :param Exception exc: Last operational failure.
            :raises SemanticScholarUnavailableError: In strict mode.
            :return None: Signals an unavailable batch in tolerant mode.
            """
            if isinstance(exc, _CandidateOperationSkippedError):
                if raise_on_unavailable:
                    raise exc
                logger.warning(
                    "Skipped batch fetch for %s papers after an earlier paper-metadata "
                    "outage: %s",
                    len(normalized_ids),
                    exc,
                )
                return None
            if raise_on_unavailable:
                raise payloads._unavailable_error(
                    "batch fetching papers",
                    f": {exc}",
                    rate_limited=self._is_rate_limit_error(exc),
                ) from exc
            logger.warning(
                "Failed to batch fetch %s papers after %s attempts: %s",
                len(normalized_ids),
                API_CONFIG.max_retries,
                exc,
            )
            return None

        matched = self._call_with_retries(
            _operation,
            failure_domain=_FailureDomain.PAPER_METADATA,
            failure_context="batch fetching papers",
            on_retry=lambda attempt, wait_time, exc: logger.debug(
                "Failed to batch fetch papers (attempt %s). Retrying in %.1fs: %s",
                attempt,
                wait_time,
                exc,
            ),
            on_final_failure=_final_failure,
        )

        if matched is None:
            return cached

        return {**cached, **matched}

    def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 20,
        *,
        raise_on_unavailable: bool = False,
    ) -> list[Paper]:
        """
        Fetch papers that cite the given paper.

        :param str paper_id: Paper identifier
        :param int limit: Maximum number of citations to fetch
        :param bool raise_on_unavailable: Whether exhausted operational retries
            raise instead of returning an empty list.
        :return list[Paper]: Citation Papers (may be empty).
        :raises SemanticScholarRequestError: If Semantic Scholar rejects the request.
        """
        return self._get_related_papers(
            paper_id=paper_id,
            limit=limit,
            fetch_method=self.client.get_paper_citations,
            relation_label="citations",
            failure_domain=_FailureDomain.CITATIONS,
            raise_on_unavailable=raise_on_unavailable,
        )

    def get_paper_references(
        self,
        paper_id: str,
        limit: int = 20,
        *,
        raise_on_unavailable: bool = False,
    ) -> list[Paper]:
        """
        Fetch papers referenced by the given paper.

        :param str paper_id: Paper identifier
        :param int limit: Maximum number of references to fetch
        :param bool raise_on_unavailable: Whether exhausted operational retries
            raise instead of returning an empty list.
        :return list[Paper]: List of Paper objects (may be shorter than limit)
        :raises SemanticScholarRequestError: If Semantic Scholar rejects the request.
        """
        return self._get_related_papers(
            paper_id=paper_id,
            limit=limit,
            fetch_method=self.client.get_paper_references,
            relation_label="references",
            failure_domain=_FailureDomain.REFERENCES,
            raise_on_unavailable=raise_on_unavailable,
        )

    def _get_related_papers(
        self,
        paper_id: str,
        limit: int,
        fetch_method: Callable[..., Any],
        relation_label: str,
        failure_domain: _FailureDomain,
        raise_on_unavailable: bool,
    ) -> list[Paper]:
        """Fetch and convert citation-like relation payloads with shared retry logic.

        :param str paper_id: Raw paper identifier.
        :param int limit: Maximum number of relation records to fetch.
        :param Callable[..., Any] fetch_method: Semantic Scholar relation fetch method.
        :param str relation_label: Human-readable label used in logs.
        :param _FailureDomain failure_domain: Capability sharing this retry budget.
        :param bool raise_on_unavailable: Whether exhausted operational retries raise.
        :return list[Paper]: Converted relation papers, empty when ``limit`` is zero.
        :raises ValueError: If ``limit`` is not a non-negative integer.
        """
        limit = payloads._validate_integer_limit(limit, "limit", allow_zero=True)
        if limit == 0:
            return []

        normalized_paper_id = normalize_paper_id(paper_id)

        def _operation() -> list[Paper]:
            """Fetch and convert citation/reference relation records.

            :return list[Paper]: Converted relation papers for this attempt.
            """
            attempt_papers: list[Paper] = []
            try:
                relation_records = fetch_method(
                    normalized_paper_id,
                    fields=payloads._default_paper_fields(),
                    limit=min(limit, GRAPH_RELATION_PAGE_SIZE),
                )
            except TypeError as exc:
                if not payloads._is_sdk_null_relation_page(exc):
                    raise
                logger.debug(
                    "Semantic Scholar returned an empty %s page for %s.",
                    relation_label,
                    normalized_paper_id,
                )
                return attempt_papers
            if not relation_records:
                return attempt_papers

            for record in relation_records:
                paper = self._convert_api_paper(getattr(record, "paper", None))
                if paper:
                    attempt_papers.append(paper)
                    disk_cache._persist_paper(paper, paper.paper_id)

                if len(attempt_papers) >= limit:
                    break

            return attempt_papers

        def _final_failure(exc: Exception) -> list[Paper]:
            """Apply the caller-selected failure contract after retry exhaustion.

            :param Exception exc: Final operational failure.
            :return list[Paper]: Empty list in tolerant mode.
            :raises SemanticScholarUnavailableError: In strict mode.
            """
            if isinstance(exc, _CandidateOperationSkippedError):
                if raise_on_unavailable:
                    raise exc
                logger.warning(
                    "Skipped %s for %s after an earlier %s outage: %s",
                    relation_label,
                    normalized_paper_id,
                    relation_label,
                    exc,
                )
                return []
            if raise_on_unavailable:
                raise payloads._unavailable_error(
                    f"fetching {relation_label} for {normalized_paper_id}",
                    f": {exc}",
                    rate_limited=self._is_rate_limit_error(exc),
                ) from exc
            logger.warning(
                "Failed to fetch %s for %s after %s attempts: %s",
                relation_label,
                normalized_paper_id,
                API_CONFIG.max_retries,
                exc,
            )
            return []

        return self._call_with_retries(
            _operation,
            failure_domain=failure_domain,
            failure_context=f"fetching {relation_label} for {normalized_paper_id}",
            on_retry=lambda attempt, wait_time, exc: logger.debug(
                "Failed to fetch %s for %s (attempt %s). Retrying in %.1fs",
                relation_label,
                normalized_paper_id,
                attempt,
                wait_time,
            ),
            on_final_failure=_final_failure,
            handled_exceptions=(
                (
                    ObjectNotFoundException,
                    lambda _exc: (
                        logger.warning(
                            "Paper not found for %s: %s",
                            relation_label,
                            normalized_paper_id,
                        )
                        or []
                    ),
                ),
                (
                    BadQueryParametersException,
                    lambda exc: _raise_request_error(
                        exc, f"fetching {relation_label} for {normalized_paper_id}"
                    ),
                ),
                (
                    PermissionError,
                    lambda exc: _raise_request_error(
                        exc, f"fetching {relation_label} for {normalized_paper_id}"
                    ),
                ),
            ),
        )

    def get_cached_reference_ids(self, paper_id: str) -> list[str] | None:
        """Read a validated persisted reference entry without making an API request.

        :param str paper_id: Paper identifier whose cached references are requested.
        :return list[str] | None: Cached reference IDs, including an authoritative
            empty list, or ``None`` when no valid cache entry exists.
        """
        normalized_paper_id = normalize_paper_id(paper_id)
        cache_path = disk_cache._reference_cache_path(normalized_paper_id)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("reference cache payload must be a JSON object")
            if data.get("version") != disk_cache.REFERENCE_CACHE_VERSION:
                return None
            if data.get("paper_id") != normalized_paper_id:
                raise ValueError("reference cache paper ID does not match its key")
            if "references" not in data:
                raise ValueError("reference cache payload is missing references")
            cached_references = data["references"]
            refs = payloads._coerce_cached_reference_ids(cached_references)
            if refs is None:
                logger.warning(
                    "Invalid reference cache payload for %s; rebuilding entry.",
                    normalized_paper_id,
                )
                with contextlib.suppress(OSError):
                    cache_path.unlink(missing_ok=True)
                return None
            if cached_references != refs:
                self._persist_reference_cache_entry(
                    cache_path,
                    normalized_paper_id,
                    refs,
                )
            if refs:
                logger.debug(
                    "Loaded %d cached references for %s",
                    len(refs),
                    normalized_paper_id,
                )
            return refs
        except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
            with contextlib.suppress(OSError):
                cache_path.unlink(missing_ok=True)
            return None

    def get_reference_ids(
        self, paper_id: str, *, force_refresh: bool = False
    ) -> list[str]:
        """
        Fetch only the reference IDs for a paper (faster than full references).

        :param str paper_id: Paper identifier
        :param bool force_refresh: Whether to bypass cache reads and fetch fresh IDs.
        :return list[str]: List of referenced paper IDs
        :raises TypeError: If the SDK or response payload violates the relation contract.
        :raises SemanticScholarRequestError: If Semantic Scholar rejects the request.
        :raises SemanticScholarUnavailableError: If operational retries are exhausted.
        """
        normalized_paper_id = normalize_paper_id(paper_id)
        cache_path = disk_cache._reference_cache_path(normalized_paper_id)
        if force_refresh:
            logger.debug(
                "Bypassing reference cache for %s due to force_refresh.",
                normalized_paper_id,
            )
        if not force_refresh:
            cached_references = self.get_cached_reference_ids(normalized_paper_id)
            if cached_references is not None:
                return cached_references

        def _persist_empty() -> list[str]:
            """Persist and return an empty cached reference-ID list.

            :return list[str]: Empty reference-ID list.
            """
            self._persist_reference_cache_entry(
                cache_path,
                normalized_paper_id,
                [],
            )
            return []

        def _operation() -> list[Any]:
            """Fetch one materialized reference-relation response.

            :return list[Any]: Raw relation records for later validation.
            """
            try:
                raw_references = self.client.get_paper_references(
                    normalized_paper_id,
                    fields=["paperId"],
                )
            except TypeError as exc:
                if not payloads._is_sdk_null_relation_page(exc):
                    raise
                logger.debug(
                    "Semantic Scholar returned an empty reference-ID page for %s.",
                    normalized_paper_id,
                )
                return []
            if isinstance(raw_references, (str, bytes, Mapping)):
                raise TypeError(
                    "Reference relation response must be an iterable of records."
                )
            return list(raw_references)

        def _raise_contract_failure(exc: Exception) -> list[Any]:
            """Surface local SDK/payload contract errors without retrying.

            :param Exception exc: Local contract failure.
            :raises Exception: Always re-raises ``exc``.
            :return list[Any]: This function does not return successfully.
            """
            raise exc

        def _raise_failure(exc: Exception) -> list[Any]:
            """Raise an availability error after operational retry exhaustion.

            :param Exception exc: Final exception raised by the API client.
            :raises SemanticScholarUnavailableError: Always with failure context.
            :return list[Any]: This function does not return successfully.
            """
            if isinstance(exc, _CandidateOperationSkippedError):
                raise exc
            raise payloads._unavailable_error(
                f"fetching reference IDs for {normalized_paper_id}",
                f": {exc}",
                rate_limited=self._is_rate_limit_error(exc),
            ) from exc

        paper_not_found = False

        def _handle_not_found(_exc: Exception) -> list[Any]:
            """Return missing-paper references without treating them as successful empties.

            :param Exception _exc: SDK exception indicating the relation endpoint's
                first page was not found.
            :return list[Any]: Empty relation list for the absent paper.
            """
            nonlocal paper_not_found
            paper_not_found = True
            logger.warning(
                "Paper not found for reference IDs: %s",
                normalized_paper_id,
            )
            return []

        references = self._call_with_retries(
            _operation,
            failure_domain=_FailureDomain.REFERENCES,
            failure_context=f"fetching reference IDs for {normalized_paper_id}",
            on_retry=lambda attempt, wait_time, exc: logger.debug(
                "Failed to fetch reference IDs for %s (attempt %s). Retrying in %.1fs",
                normalized_paper_id,
                attempt,
                wait_time,
            ),
            on_final_failure=_raise_failure,
            handled_exceptions=(
                (TypeError, _raise_contract_failure),
                (ValueError, _raise_contract_failure),
                (
                    ObjectNotFoundException,
                    _handle_not_found,
                ),
                (
                    BadQueryParametersException,
                    lambda exc: _raise_request_error(
                        exc, f"fetching reference IDs for {normalized_paper_id}"
                    ),
                ),
                (
                    PermissionError,
                    lambda exc: _raise_request_error(
                        exc, f"fetching reference IDs for {normalized_paper_id}"
                    ),
                ),
            ),
        )
        if paper_not_found:
            return []
        if not references:
            return _persist_empty()

        normalized_ref_ids = payloads._coerce_cached_reference_ids(references)
        if normalized_ref_ids is None:
            if all(payloads._is_unresolved_reference(record) for record in references):
                logger.warning(
                    "All references for %s are outside Semantic Scholar's resolved "
                    "paper corpus; no reference IDs are available.",
                    normalized_paper_id,
                )
                return _persist_empty()
            raise TypeError(
                "Non-empty reference response contained no valid paper IDs."
            )
        self._persist_reference_cache_entry(
            cache_path,
            normalized_paper_id,
            normalized_ref_ids,
        )
        return normalized_ref_ids

    @staticmethod
    def _resolve_paper_fields(fields: list[str] | None) -> tuple[list[str], bool]:
        """Default the requested field set and decide whether results are cacheable.

        Persisting a paper is only safe when the response carries every field the
        metadata cache stores, so a caller-narrowed field list disables caching.

        :param list[str] | None fields: Caller-supplied fields, or ``None`` for the
            default paper field set.
        :return tuple[list[str], bool]: Fields to request, and whether the responses
            carry full metadata worth persisting.
        """
        default_fields = payloads._default_paper_fields()
        if fields is None:
            fields = default_fields
        return fields, set(default_fields).issubset(fields)

    def _papers_from_records(
        self, records: Iterable[Any], *, cache_full_metadata: bool
    ) -> list[Paper]:
        """Convert recommendation/search records, persisting full-metadata results.

        :param Iterable[Any] records: Raw records from a search-like response.
        :param bool cache_full_metadata: Whether converted papers may be persisted.
        :return list[Paper]: Converted papers, skipping malformed records.
        """
        papers: list[Paper] = []
        for record in records:
            paper = self._convert_recommendation(record)
            if paper:
                papers.append(paper)
                if cache_full_metadata:
                    disk_cache._persist_paper(paper, paper.paper_id)
        return papers

    def get_recommended_papers(
        self,
        paper_id: str,
        limit: int = 50,
        fields: list[str] | None = None,
        *,
        raise_on_unavailable: bool = False,
    ) -> list[Paper]:
        """
        Get semantically related papers using S2 recommendations.

        :param str paper_id: S2 paper ID
        :param int limit: Maximum recommendations
        :param list[str] | None fields: API fields to return.
        :param bool raise_on_unavailable: Whether exhausted operational retries
            for the primary request raise instead of returning an empty list. An
            unavailable optional ``all-cs`` widening request preserves a
            successful empty primary result.
        :return list[Paper]: Ranked recommendation papers.
        """
        fields, cache_full_metadata = self._resolve_paper_fields(fields)
        # Semantic Scholar's recommendations endpoint does not currently support
        # requesting ``references`` in field lists (returns HTTP 400 with
        # unsupported nested-reference field tokens). Keep the request field set
        # endpoint-compatible and let callers hydrate references via dedicated
        # reference-ID methods when needed.
        if "references" in fields:
            fields = [field for field in fields if field != "references"]
        parsed_limit = payloads._validate_integer_limit(
            limit,
            "limit",
            maximum=RECOMMENDATION_MAX_RESULTS,
        )

        normalized_paper_id = normalize_paper_id(paper_id)
        encoded_paper_id = quote(normalized_paper_id, safe="")
        base_params = {"fields": ",".join(fields), "limit": parsed_limit}
        payload = self._request_json(
            f"{RECOMMENDATION_BASE_URL}/{encoded_paper_id}",
            base_params,
            failure_domain=_FailureDomain.RECOMMENDATIONS,
            raise_on_unavailable=raise_on_unavailable,
            context=f"fetching recommendations for {normalized_paper_id}",
        )
        if payload is None:
            return []
        raw_recommendations = payload.get("recommendedPapers", [])
        if not raw_recommendations:
            # The default candidate pool ("recent") only covers recent papers
            # and returns nothing for classic seeds (e.g. 2017 landmark
            # papers). Fall back to the broader CS pool before giving up.
            try:
                fallback_payload = self._request_json(
                    f"{RECOMMENDATION_BASE_URL}/{encoded_paper_id}",
                    {**base_params, "from": "all-cs"},
                    failure_domain=_FailureDomain.RECOMMENDATIONS,
                    raise_on_unavailable=raise_on_unavailable,
                    record_domain_failure=False,
                    context=(
                        f"fetching all-cs recommendations for {normalized_paper_id}"
                    ),
                )
            except (
                SemanticScholarUnavailableError,
                SemanticScholarRequestError,
            ) as exc:
                logger.warning(
                    "The optional all-cs recommendation fallback failed "
                    "for %s; preserving the successful empty recent-pool result: %s",
                    normalized_paper_id,
                    exc,
                )
                fallback_payload = None
            raw_recommendations = (fallback_payload or {}).get("recommendedPapers", [])
            if raw_recommendations:
                logger.debug(
                    "Recommendations for %s came from the all-cs pool "
                    "(recent pool was empty).",
                    normalized_paper_id,
                )

        return self._papers_from_records(
            raw_recommendations, cache_full_metadata=cache_full_metadata
        )

    def search_papers(
        self,
        query: str,
        limit: int = 10,
        fields: list[str] | None = None,
        *,
        raise_on_unavailable: bool = False,
    ) -> list[Paper]:
        """Search papers by title or keyword.

        :param str query: Search query string.
        :param int limit: Maximum number of results.
        :param list[str] | None fields: Optional fields list for API payload.
        :param bool raise_on_unavailable: When ``True``, exhausted retries raise
            :class:`SemanticScholarUnavailableError` instead of returning an
            empty list, so callers can distinguish "no matches" from "API down".
        :return list[Paper]: Search results.
        """
        parsed_limit = payloads._validate_integer_limit(
            limit,
            "limit",
            maximum=SEARCH_MAX_RESULTS,
        )
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        fields, cache_full_metadata = self._resolve_paper_fields(fields)

        raw_results: list[Any] = []
        offset = 0
        while len(raw_results) < parsed_limit:
            page_limit = min(SEARCH_PAGE_SIZE, parsed_limit - len(raw_results))
            payload = self._request_json(
                SEARCH_BASE_URL,
                {
                    "query": normalized_query,
                    "fields": ",".join(fields),
                    "limit": page_limit,
                    "offset": offset,
                },
                failure_domain=_FailureDomain.SEARCH,
                raise_on_unavailable=raise_on_unavailable,
                context=f"searching for {normalized_query!r}",
            )
            if not payload:
                return []

            page_records = payload.get("data", [])
            if not page_records:
                break
            raw_results.extend(page_records)

            next_offset = payload.get("next")
            if (
                isinstance(next_offset, bool)
                or not isinstance(next_offset, int)
                or next_offset <= offset
            ):
                break
            offset = next_offset

        return self._papers_from_records(
            raw_results[:parsed_limit], cache_full_metadata=cache_full_metadata
        )
