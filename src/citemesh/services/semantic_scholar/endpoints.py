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
- ``_persist_reference_cache_entry``: reference-cache writes.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import wraps
from typing import Any
from urllib.parse import quote

from semanticscholar.SemanticScholarException import (
    BadQueryParametersException,
    ObjectNotFoundException,
)

from citemesh.core import Paper
from citemesh.core.paper_ids import normalize_paper_id

from . import disk_cache, payloads
from .errors import (
    SemanticScholarUnavailableError,
    _CandidateOperationSkippedError,
    _FailureDomain,
    _raise_request_error,
    _RetryableRequestError,
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


def _scoped_endpoint(method: Callable[..., Any]) -> Callable[..., Any]:
    """Share discovery checks and recovery time across an endpoint's subrequests.

    :param Callable[..., Any] method: Client endpoint or compound endpoint helper.
    :return Callable[..., Any]: Method sharing its enclosing collection scope.
    """

    @wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Run one endpoint inside a collection scope.

        :param Any self: Semantic Scholar client.
        :param Any args: Endpoint positional arguments.
        :param Any kwargs: Endpoint keyword arguments.
        :return Any: Endpoint result.
        """
        with self.candidate_operation_scope():
            return method(self, *args, **kwargs)

    return wrapped


class _EndpointsMixin:
    """Semantic Scholar endpoint methods shared by the transport client."""

    @_scoped_endpoint
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
                logger.debug("Reused cached paper metadata for %s.", paper_id)
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

    @_scoped_endpoint
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
            """Fetch positional batch results, omitting unavailable records.

            :return dict[str, Paper]: Successful results keyed by requested ID.
            """
            api_papers = self._request_json_once(
                f"{PAPER_BASE_URL}/batch",
                {"fields": ",".join(payloads._default_paper_fields())},
                context="batch fetching papers",
                payload={"ids": normalized_ids},
            )
            if api_papers is None:
                # Missing individual papers are positional null rows. A 404 for
                # the fixed batch endpoint is therefore an availability failure,
                # not an authoritative result for every requested ID.
                raise _RetryableRequestError(f"HTTP 404 from {PAPER_BASE_URL}/batch")
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
                paper = payloads._convert_api_paper(api_paper)
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
                    "Skipped batch fetch for %s papers after an earlier Semantic Scholar "
                    "outage: %s",
                    len(normalized_ids),
                    exc,
                )
                return None
            if raise_on_unavailable:
                raise self._unavailable_error("batch fetching papers", exc) from exc
            logger.warning("%s", self._unavailable_error("batch fetching papers", exc))
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

    def _materialize_discovery(
        self, paper_ids: list[str], *, context: str
    ) -> list[Paper]:
        """Load checked candidates from disk and batch-fetch only missing records.

        :param list[str] paper_ids: Freshly checked IDs in provider order. Updated
            in place to omit IDs without resolvable metadata.
        :param str context: Discovery operation used in logs and failure messages.
        :return list[Paper]: Resolvable papers in the checked discovery order.
        :raises SemanticScholarUnavailableError: If a required metadata batch fails.
        """
        unique_ids = list(dict.fromkeys(paper_ids))
        papers = {
            paper_id: paper
            for paper_id in unique_ids
            if not self.refresh_paper_cache
            and (paper := disk_cache._load_cached_paper(paper_id)) is not None
        }
        reused_count = len(papers)
        missing = [paper_id for paper_id in unique_ids if paper_id not in papers]
        try:
            for offset in range(0, len(missing), 500):
                papers.update(
                    self.get_papers(
                        missing[offset : offset + 500], raise_on_unavailable=True
                    )
                )
        except SemanticScholarUnavailableError as exc:
            raise SemanticScholarUnavailableError(
                f"Could not complete current {context}; required paper metadata "
                f"was unavailable. Previous discovery snapshot retained. {exc}"
            ) from exc
        unresolved = [paper_id for paper_id in unique_ids if paper_id not in papers]
        if unresolved:
            logger.warning(
                "%s: skipping %d checked IDs without resolvable paper metadata.",
                context,
                len(unresolved),
            )
            # Keep the successful snapshot and same-scope reuse aligned with the
            # candidates this metadata service can actually materialize.
            paper_ids[:] = [paper_id for paper_id in paper_ids if paper_id in papers]
        logger.info(
            "%s: reused %d cached paper records; fetched %d missing records.",
            context,
            reused_count,
            len(papers) - reused_count,
        )
        return [papers[paper_id] for paper_id in paper_ids]

    def _save_discovery(
        self, key: tuple[str, str, int, str], paper_ids: list[str]
    ) -> None:
        """Publish a successful discovery snapshot and reuse it within this scope.

        :param tuple[str, str, int, str] key: Endpoint, seed, limit and pool.
        :param list[str] paper_ids: Successfully checked, ordered candidate IDs.
        :return None: Saves only after required acquisition has completed.
        """
        self._candidate_operation.state.discovery_ids[key] = list(paper_ids)
        disk_cache._persist_discovery(key, paper_ids)

    @_scoped_endpoint
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
        key = (relation_label, normalized_paper_id, limit, "")
        context = f"{relation_label} discovery for {normalized_paper_id}"
        checked_ids = self._candidate_operation.state.discovery_ids.get(key)
        if checked_ids is not None:
            return self._materialize_discovery(checked_ids, context=context)
        paper_not_found = False

        def _operation() -> list[str]:
            """Check citation/reference IDs without downloading known paper records.

            :return list[str]: Ordered IDs for this completed attempt.
            """
            attempt_ids: list[str] = []
            seen_ids: set[str] = set()
            try:
                # SDK limit is a page size (at most 1,000), not the total.
                # Its iterator fetches later pages; our loop bounds the result.
                relation_records = fetch_method(
                    normalized_paper_id,
                    fields=["paperId"],
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
                return attempt_ids
            if not relation_records:
                return attempt_ids

            for record in relation_records:
                paper = getattr(record, "paper", None)
                paper_id = payloads._payload_get(paper, "paperId")
                if isinstance(paper_id, str) and paper_id.strip():
                    normalized_id = normalize_paper_id(paper_id)
                    if normalized_id not in seen_ids:
                        seen_ids.add(normalized_id)
                        attempt_ids.append(normalized_id)
                elif not payloads._is_unresolved_reference(record):
                    raise _SemanticScholarResponseContractError(
                        f"Semantic Scholar returned malformed {relation_label} discovery."
                    )
                if len(attempt_ids) >= limit:
                    break

            return attempt_ids

        def _final_failure(exc: Exception) -> None:
            """Apply the caller-selected failure contract after retry exhaustion.

            :param Exception exc: Final operational failure.
            :return None: Unavailable marker in tolerant mode, never a cached empty.
            :raises SemanticScholarUnavailableError: In strict mode.
            """
            if isinstance(exc, _CandidateOperationSkippedError):
                if raise_on_unavailable:
                    raise exc
                logger.warning(
                    "Skipped %s for %s after an earlier Semantic Scholar outage: %s",
                    relation_label,
                    normalized_paper_id,
                    exc,
                )
                return None
            if raise_on_unavailable:
                raise self._unavailable_error(
                    f"checking current {context}", exc
                ) from exc
            logger.warning(
                "%s", self._unavailable_error(f"checking current {context}", exc)
            )
            return None

        def _not_found(_exc: Exception) -> list[str]:
            """Return a missing relation without replacing a successful snapshot.

            :param Exception _exc: First-page not-found response.
            :return list[str]: Empty result that is not persisted.
            """
            nonlocal paper_not_found
            paper_not_found = True
            if raise_on_unavailable:
                raise SemanticScholarUnavailableError(
                    f"Could not check current {context}: HTTP 404 (paper not found). "
                    "Previous discovery snapshot retained. Retry later or check the seed ID."
                ) from _exc
            logger.warning(
                "Paper not found for %s: %s", relation_label, normalized_paper_id
            )
            return []

        checked_ids = self._call_with_retries(
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
                    _not_found,
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
        if checked_ids is None or paper_not_found:
            return []
        try:
            papers = self._materialize_discovery(checked_ids, context=context)
        except SemanticScholarUnavailableError:
            if raise_on_unavailable:
                raise
            logger.warning("Could not complete current %s; snapshot retained.", context)
            return []
        self._save_discovery(key, checked_ids)
        return papers

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

    @_scoped_endpoint
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
                self._candidate_operation.state.reference_cache_hits += 1
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
            raise self._unavailable_error(
                f"fetching reference IDs for {normalized_paper_id}", exc
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
            paper = payloads._convert_recommendation(record)
            if paper:
                papers.append(paper)
                if cache_full_metadata:
                    disk_cache._persist_paper(paper, paper.paper_id)
        return papers

    @_scoped_endpoint
    def get_recommended_papers(
        self,
        paper_id: str,
        limit: int = 50,
        fields: list[str] | None = None,
        *,
        raise_on_unavailable: bool = False,
    ) -> list[Paper]:
        """Check recommendation IDs and reuse full paper records already on disk.

        Explicit field projections retain their direct-response behavior. Empty
        recent-pool discovery requires a successful all-cs check before it can
        be reported or persisted as empty.

        :param str paper_id: S2 paper identifier.
        :param int limit: Maximum recommendations, up to 500.
        :param list[str] | None fields: Explicit API fields, or cached full records.
        :param bool raise_on_unavailable: Raise on failed discovery/materialization
            instead of returning an uncached empty result.
        :return list[Paper]: Papers in the freshly checked recommendation order.
        """
        custom_fields = fields is not None
        fields, cache_full_metadata = self._resolve_paper_fields(fields)
        # Recommendations do not accept nested reference fields. Enrichment
        # continues to use the independent, complete reference-ID cache.
        fields = [field for field in fields if field != "references"]
        parsed_limit = payloads._validate_integer_limit(
            limit, "limit", maximum=RECOMMENDATION_MAX_RESULTS
        )
        normalized_paper_id = normalize_paper_id(paper_id)
        encoded_paper_id = quote(normalized_paper_id, safe="")
        context = f"recommendation discovery for {normalized_paper_id}"
        snapshots: list[tuple[tuple[str, str, int, str], list[str]]] = []
        checked_ids: list[str] = []
        raw_recommendations: list[Any] = []
        for pool in ("recent", "all-cs"):
            key = ("recommendations", normalized_paper_id, parsed_limit, pool)
            scoped_ids = (
                None
                if custom_fields
                else self._candidate_operation.state.discovery_ids.get(key)
            )
            if scoped_ids is not None:
                checked_ids = scoped_ids
            else:
                params: dict[str, Any] = {
                    "fields": ",".join(fields) if custom_fields else "paperId",
                    "limit": parsed_limit,
                }
                if pool != "recent":
                    params["from"] = pool
                response = self._request_json(
                    f"{RECOMMENDATION_BASE_URL}/{encoded_paper_id}",
                    params,
                    failure_domain=_FailureDomain.RECOMMENDATIONS,
                    raise_on_unavailable=raise_on_unavailable,
                    context=f"checking current {pool} recommendations for {normalized_paper_id}",
                )
                if response is None:
                    if raise_on_unavailable:
                        raise SemanticScholarUnavailableError(
                            f"Could not check current {pool} recommendations for "
                            f"{normalized_paper_id}: HTTP 404 (paper not found). "
                            "Previous discovery snapshot retained. Retry later or check the seed ID."
                        )
                    return []
                if not isinstance(response, dict) or not isinstance(
                    response.get("recommendedPapers"), list
                ):
                    raise _SemanticScholarResponseContractError(
                        "Semantic Scholar returned malformed recommendation discovery."
                    )
                raw_recommendations = response["recommendedPapers"]
                if custom_fields:
                    if raw_recommendations:
                        break
                    continue
                ids = [
                    payloads._payload_get(record, "paperId")
                    for record in raw_recommendations
                ]
                if any(
                    not isinstance(value, str) or not value.strip() for value in ids
                ):
                    raise _SemanticScholarResponseContractError(
                        "Semantic Scholar returned recommendation records without paper IDs."
                    )
                checked_ids = list(
                    dict.fromkeys(normalize_paper_id(value) for value in ids)
                )
                snapshots.append((key, checked_ids))
            if checked_ids:
                break

        if custom_fields:
            return self._papers_from_records(
                raw_recommendations, cache_full_metadata=cache_full_metadata
            )
        try:
            papers = self._materialize_discovery(checked_ids, context=context)
        except SemanticScholarUnavailableError:
            if raise_on_unavailable:
                raise
            logger.warning("Could not complete current %s; snapshot retained.", context)
            return []
        for key, ids in snapshots:
            self._save_discovery(key, ids)
        return papers

    @_scoped_endpoint
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
                retry_not_found=offset > 0,
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
