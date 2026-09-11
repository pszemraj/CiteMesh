"""Semantic Scholar transport: the API client and its process-wide accessors.

Owns :class:`SemanticScholarClient` -- session lifecycle, request pacing, the
retry orchestration wrapped around the SDK and REST endpoints, the endpoint
methods themselves, and the shared singleton returned by :func:`get_client`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from semanticscholar import SemanticScholar
from semanticscholar.SemanticScholarException import (
    BadQueryParametersException,
    ObjectNotFoundException,
)
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
)
from tenacity.retry import retry_base

from citemesh.core import API_CONFIG, Paper
from citemesh.data.cache import atomic_write_json
from citemesh.paper_ids import normalize_paper_id

from . import disk_cache, payloads, retry
from .errors import (
    SemanticScholarRequestError,
    SemanticScholarUnavailableError,
    _CandidateOperationSkippedError,
    _CandidateOperationState,
    _FailureDomain,
    _raise_request_error,
    _RetryableRequestError,
    _SemanticScholarResponseContractError,
    _unwrap_sdk_retry_error,
)

logger = logging.getLogger(__name__)


RECOMMENDATION_BASE_URL = (
    "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
)
PAPER_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
SEARCH_BASE_URL = f"{PAPER_BASE_URL}/search"
_anonymous_pool_announced = False


class SemanticScholarClient:
    """
    Wrapper for Semantic Scholar API with retry logic and caching.

    This client provides:
    - Automatic retry on transient failures
    - Rate limiting
    - Reference list cache
    - Direct recommendation and search endpoint support
    """

    def __init__(
        self,
        timeout: float = API_CONFIG.default_timeout,
        *,
        api_key: str | None = None,
        refresh_paper_cache: bool = False,
    ):
        """
        Initialize the API client.

        :param float timeout: Request timeout in seconds.
        :param str | None api_key: Explicit API key, or ``None`` to read S2_API_KEY.
            An empty string explicitly selects anonymous access.
        :param bool refresh_paper_cache: Bypass persisted paper metadata on reads.
        """
        if api_key is None:
            api_key = os.getenv("S2_API_KEY") or None

        self.client = SemanticScholar(timeout=timeout, api_key=api_key, retry=False)
        try:
            requester = self.client._AsyncSemanticScholar._requester
            if not callable(requester.get_data_async):
                raise AttributeError("get_data_async is not callable")
        except AttributeError as exc:
            raise RuntimeError(
                "Unsupported Semantic Scholar SDK transport. "
                "Install semanticscholar>=0.8.0,<0.13 with CiteMesh's dependencies."
            ) from exc
        self.refresh_paper_cache = refresh_paper_cache
        self.timeout = timeout
        self.last_request_time = 0.0
        self._candidate_operation = threading.local()
        self._session = requests.Session()
        # SDK 0.8-0.12 has no public transport hook and discards unrecognized HTTP
        # statuses. Replace only this client's requester; retain SDK pagination.
        # A weak owner prevents a requester/client cycle from delaying session close.
        requester.get_data_async = partial(
            type(self)._request_sdk_json, weakref.proxy(self)
        )
        self._closed = False
        self.requests_per_second = (
            API_CONFIG.authenticated_requests_per_second
            if api_key
            else API_CONFIG.requests_per_second
        )

        if api_key:
            self._session.headers["x-api-key"] = api_key
            logger.info("Using Semantic Scholar API key")
        else:
            global _anonymous_pool_announced
            if not _anonymous_pool_announced:
                _anonymous_pool_announced = True
                logger.info(
                    "No S2_API_KEY set; using the shared anonymous Semantic "
                    "Scholar pool (slower rate limit, higher 429 likelihood). "
                    "Free keys: %s",
                    payloads.S2_API_KEY_SIGNUP_URL,
                )

    @contextlib.contextmanager
    def candidate_operation_scope(self) -> Iterator[None]:
        """Share capability-specific outages across one discovery operation.

        Nested strategy calls share the outer scope. A later independent collection
        receives a new scope and may retry normally.

        :return Iterator[None]: Candidate-discovery scope.
        """
        state = getattr(self._candidate_operation, "state", None)
        owns_state = state is None
        if owns_state:
            state = _CandidateOperationState()
            self._candidate_operation.state = state
        state.depth += 1
        try:
            yield
        finally:
            state.depth -= 1
            if owns_state and state.depth == 0:
                del self._candidate_operation.state

    def _candidate_operation_failure(
        self, failure_domain: _FailureDomain
    ) -> SemanticScholarUnavailableError | None:
        """Return an exhausted capability failure from the active scope.

        :param _FailureDomain failure_domain: Capability whose state is requested.
        :return SemanticScholarUnavailableError | None: The recorded failure, or
            ``None`` outside a scope or before that capability exhausts a request.
        """
        state = getattr(self._candidate_operation, "state", None)
        if state is None:
            return None
        return state.failures.get(failure_domain)

    def _record_candidate_operation_failure(
        self,
        failure_domain: _FailureDomain,
        error: SemanticScholarUnavailableError,
    ) -> None:
        """Remember the first exhausted request for one scoped capability.

        :param _FailureDomain failure_domain: Capability whose retry budget exhausted.
        :param SemanticScholarUnavailableError error: Exhausted service failure.
        :return None: Updates the active scope when one exists.
        """
        state = getattr(self._candidate_operation, "state", None)
        if state is not None:
            state.failures.setdefault(failure_domain, error)

    @staticmethod
    def _skipped_candidate_operation_error(
        failure_domain: _FailureDomain,
        failure: SemanticScholarUnavailableError,
    ) -> _CandidateOperationSkippedError:
        """Describe a request skipped after the same capability failed.

        :param _FailureDomain failure_domain: Capability skipped by the breaker.
        :param SemanticScholarUnavailableError failure: Original exhausted failure.
        :return _CandidateOperationSkippedError: Skipped-request failure retaining the
            original error text.
        """
        return _CandidateOperationSkippedError(
            f"Skipped Semantic Scholar {failure_domain.value} request after an "
            f"earlier {failure_domain.value} outage: {failure}"
        )

    def __enter__(self) -> SemanticScholarClient:
        """Return this client for context-manager use."""
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> None:
        """Close the client session on context-manager exit."""
        self.close()

    def close(self) -> None:
        """Close the HTTP session and underlying API client handles."""
        if self._closed:
            return
        self._closed = True
        global _client_instance
        with _client_lock:
            if _client_instance is self:
                _client_instance = None
        with contextlib.suppress(Exception):
            self._session.close()
        client_session = getattr(self.client, "session", None)
        if hasattr(client_session, "close"):
            with contextlib.suppress(Exception):
                client_session.close()
        with contextlib.suppress(Exception):
            close_api_client = getattr(self.client, "close", None)
            if callable(close_api_client):
                close_api_client()

    def __del__(self) -> None:
        """Attempt to close sessions on object finalization."""
        with contextlib.suppress(Exception):
            self.close()

    def _persist_reference_cache_entry(
        self, cache_path: Path, paper_id: str, reference_ids: list[str]
    ) -> None:
        """Persist normalized reference IDs to cache with best-effort durability.

        :param Path cache_path: Target cache file path.
        :param str paper_id: Normalized paper ID for payload metadata.
        :param list[str] reference_ids: Reference IDs to persist.
        :return None: Writes cache payload when filesystem operations succeed.
        """
        normalized_reference_ids = payloads._coerce_cached_reference_ids(reference_ids)
        if normalized_reference_ids is None:
            normalized_reference_ids = []

        try:
            atomic_write_json(
                cache_path,
                {
                    "paper_id": paper_id,
                    "references": normalized_reference_ids,
                    "version": disk_cache.REFERENCE_CACHE_VERSION,
                },
            )
        except OSError as exc:
            logger.debug(
                "Failed to persist reference cache for %s: %s",
                paper_id,
                exc,
            )

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests (key-aware pace)."""
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """Detect rate-limit exceptions.

        :param Exception error: Exception from request/client layer.
        :return bool: ``True`` when the error indicates HTTP 429.
        """
        return retry._is_rate_limit_error(error)

    def _run_with_retries(
        self,
        operation: Callable[[], Any],
        *,
        failure_domain: _FailureDomain,
        predicate: retry_base,
        on_skip: Callable[
            [_CandidateOperationSkippedError, SemanticScholarUnavailableError], Any
        ],
        on_retry: Callable[[RetryCallState, Exception | None, float], None],
        on_exhausted: Callable[[Exception], Any],
        caught: tuple[type[Exception], ...] = (Exception,),
    ) -> Any:
        """Drive one API operation through the shared retry schedule.

        Both transports share the per-capability circuit breaker, the attempt
        budget, the jittered backoff, and the outage-scale wait warning. They
        differ only in which failures are retryable and in what a skipped or
        exhausted budget means, which the caller supplies as callbacks.

        :param Callable[[], Any] operation: Zero-argument API operation to execute.
        :param _FailureDomain failure_domain: Capability sharing this retry budget.
        :param retry_base predicate: Tenacity predicate selecting retryable failures.
        :param Callable[..., Any] on_skip: Policy applied when this capability already
            failed in the active scope, receiving the skip error and that outage.
        :param Callable[[RetryCallState, Exception | None, float], None] on_retry:
            Callback invoked before each sleep with the retry state, the triggering
            exception, and the upcoming wait in seconds.
        :param Callable[[Exception], Any] on_exhausted: Policy applied to the final
            failure; it decides when to record the outage and whether to raise.
        :param tuple[type[Exception], ...] caught: Exception types routed to
            ``on_exhausted``; anything else propagates to the caller unchanged.
        :return Any: Result of ``operation``, or of one of the policy callbacks.
        """
        scope_failure = self._candidate_operation_failure(failure_domain)
        if scope_failure is not None:
            return on_skip(
                self._skipped_candidate_operation_error(failure_domain, scope_failure),
                scope_failure,
            )

        def _before_sleep(retry_state: RetryCallState) -> None:
            """Warn on outage-scale waits, then run the transport's own callback.

            :param RetryCallState retry_state: Failed attempt with its next action.
            :return None: Invokes ``on_retry``.
            """
            retry._warn_on_long_wait(retry_state)
            on_retry(
                retry_state,
                retry_state.outcome.exception() if retry_state.outcome else None,
                retry_state.next_action.sleep if retry_state.next_action else 0.0,
            )

        retryer = Retrying(
            stop=stop_after_attempt(API_CONFIG.max_retries),
            wait=retry._S2BackoffWait(),
            retry=predicate,
            before_sleep=_before_sleep,
            sleep=lambda seconds: time.sleep(seconds),
            reraise=True,
        )
        try:
            return retryer(operation)
        except caught as exc:
            return on_exhausted(exc)

    def _record_unavailable(
        self,
        context: str,
        exc: Exception,
        *,
        failure_domain: _FailureDomain,
        omit_rate_limited_detail: bool = False,
        record_domain_failure: bool = True,
    ) -> SemanticScholarUnavailableError:
        """Build the exhausted-budget error and remember it for the active scope.

        :param str context: Human-readable request context.
        :param Exception exc: Final operational failure.
        :param _FailureDomain failure_domain: Capability whose retry budget exhausted.
        :param bool omit_rate_limited_detail: Whether a rate-limited failure drops the
            trailing exception text, as the REST endpoints' message shape does.
        :param bool record_domain_failure: Whether the active scope should skip later
            calls to the same capability.
        :return SemanticScholarUnavailableError: Availability error to raise or log.
        """
        rate_limited = self._is_rate_limit_error(exc)
        detail = "" if rate_limited and omit_rate_limited_detail else f": {exc}"
        unavailable = payloads._unavailable_error(
            context, detail, rate_limited=rate_limited
        )
        if record_domain_failure:
            self._record_candidate_operation_failure(failure_domain, unavailable)
        return unavailable

    def _call_with_retries(
        self,
        operation: Callable[[], Any],
        *,
        failure_domain: _FailureDomain,
        failure_context: str,
        on_retry: Callable[[int, float, Exception], None],
        on_final_failure: Callable[[Exception], Any],
        handled_exceptions: tuple[
            tuple[type[Exception], Callable[[Exception], Any]], ...
        ] = (),
    ) -> Any:
        """Run an API operation with shared retry/backoff behavior.

        :param Callable[[], Any] operation: Zero-argument API operation to execute.
        :param _FailureDomain failure_domain: Capability sharing this retry budget.
        :param str failure_context: Description used if operational retries exhaust.
        :param Callable[[int, float, Exception], None] on_retry: Callback invoked before
            each retry with 1-based attempt count, sleep delay, and triggering exception.
        :param Callable[[Exception], Any] on_final_failure: Callback used to produce the
            final return value or raise when retries are exhausted.
        :param tuple[tuple[type[Exception], Callable[[Exception], Any]], ...] handled_exceptions:
            Exception-specific handlers that short-circuit normal retry handling.
        :return Any: Result produced by ``operation`` or one of the failure handlers.
        """
        local_failures = (TypeError, ValueError, SemanticScholarRequestError)
        decode_failures = (json.JSONDecodeError, requests.exceptions.JSONDecodeError)

        def _attempt() -> Any:
            """Run the operation, surfacing the cause behind the SDK's retry wrapper.

            :return Any: Result produced by ``operation``.
            """
            try:
                return operation()
            except Exception as raw_exc:
                raise _unwrap_sdk_retry_error(raw_exc)

        def _on_exhausted(exc: Exception) -> Any:
            """Apply the SDK contract to the failure that ended the retry budget.

            :param Exception exc: Final failure raised by the SDK operation.
            :return Any: Value produced by a handler or by ``on_final_failure``.
            :raises Exception: Local contract failures propagate unchanged.
            """
            # SDK JSON decoding can fail on transient HTML/plain-text error bodies.
            if not isinstance(exc, decode_failures):
                for error_type, handler in handled_exceptions:
                    if isinstance(exc, error_type):
                        return handler(exc)
                if isinstance(exc, local_failures):
                    raise exc
            self._record_unavailable(
                failure_context, exc, failure_domain=failure_domain
            )
            return on_final_failure(exc)

        return self._run_with_retries(
            _attempt,
            failure_domain=failure_domain,
            predicate=(
                retry_if_exception_type(Exception)
                & retry_if_not_exception_type(
                    local_failures
                    + tuple(kind for kind, _handler in handled_exceptions)
                )
            )
            | retry_if_exception_type(decode_failures),
            on_skip=lambda skipped, _scope_failure: on_final_failure(skipped),
            on_retry=lambda retry_state, exc, wait_seconds: on_retry(
                retry_state.attempt_number, wait_seconds, exc
            ),
            on_exhausted=_on_exhausted,
        )

    @staticmethod
    def _convert_api_paper(api_paper: Any) -> Paper | None:
        """
        Convert Semantic Scholar API response to Paper model.

        :param Any api_paper: Raw paper object from S2 API
        :return Paper | None: Paper object or None if conversion fails
        """
        return payloads._convert_api_paper(api_paper)

    @staticmethod
    def _convert_recommendation(rec: dict[str, Any]) -> Paper | None:
        """Convert recommendation/search record dict to a Paper model.

        :param dict[str, Any] rec: Record returned by recommendation/search APIs.
        :return Paper | None: Parsed Paper model or ``None`` on malformed payload.
        """
        return payloads._convert_recommendation(rec)

    @staticmethod
    def _extract_reference_ids(raw_references: Any) -> list[str]:
        """Extract reference IDs from recommendation/search payload shapes.

        :param Any raw_references: Raw ``references`` payload from API response.
        :return list[str]: Parsed reference ID list (order-preserving, deduplicated).
        """
        return payloads._extract_reference_ids(raw_references)

    async def _request_sdk_json(
        self,
        url: str,
        parameters: str,
        headers: dict[str, str] | None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        """Supply status-preserving responses to the SDK's synchronous wrapper.

        :param str url: SDK endpoint URL.
        :param str parameters: SDK-encoded query parameters.
        :param dict[str, str] | None headers: SDK authentication headers.
        :param dict[str, Any] | None payload: Batch POST body, or no body for GET.
        :return Any: Decoded response consumed by SDK pagination/conversion.
        :raises ObjectNotFoundException: When the first page returns HTTP 404.
        :raises _RetryableRequestError: When a later pagination page returns
            HTTP 404, so partially fetched relations are never misread as a
            missing paper.
        """
        data = self._request_json_once(
            url,
            parameters.lstrip("&"),
            context=f"requesting {url}",
            headers=headers,
            payload=payload,
        )
        if data is None:
            offset_match = re.search(r"(?:^|&)offset=(\d+)", parameters)
            if offset_match is not None and int(offset_match.group(1)) > 0:
                # The paper was served moments ago on an earlier page, so a
                # 404 here is a transient service inconsistency. Mapping it to
                # not-found would discard the fetched records and persist an
                # empty relation list for a paper that has relations.
                raise _RetryableRequestError(
                    f"HTTP 404 at pagination offset {offset_match.group(1)} from {url}"
                )
            raise ObjectNotFoundException(f"Paper not found: {url}")
        if url.endswith(("/references", "/citations")) and (
            not isinstance(data, dict) or "data" not in data
        ):
            raise _SemanticScholarResponseContractError(
                "Semantic Scholar returned a malformed relation payload without data."
            )
        return data

    def _request_json_once(
        self,
        url: str,
        params: dict[str, Any] | str,
        *,
        context: str,
        headers: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        """Issue one paced HTTP request without adding another retry budget.

        :param str url: Semantic Scholar endpoint URL.
        :param dict[str, Any] | str params: Mapping or pre-encoded query parameters.
        :param str context: Description included in request errors.
        :param dict[str, str] | None headers: Optional SDK authentication headers.
        :param dict[str, Any] | None payload: POST body, or ``None`` for GET.
        :return Any: Decoded JSON, or ``None`` for HTTP 404.
        """
        kwargs: dict[str, Any] = {"params": params, "timeout": self.timeout}
        if headers is not None:
            kwargs["headers"] = headers
        self._rate_limit()
        response = (
            self._session.get(url, **kwargs)
            if payload is None
            else self._session.post(url, json=payload, **kwargs)
        )
        if response.status_code == 404:
            return None
        if response.status_code == 429:
            raise _RetryableRequestError(
                f"HTTP 429 from {url}",
                retry_after=retry.retry_after_seconds(
                    response, default=API_CONFIG.retry_delay
                ),
                rate_limited=True,
            )
        if 400 <= response.status_code < 500 and response.status_code != 408:
            if response.status_code in {401, 403}:
                remediation = "Check S2_API_KEY credentials and access permissions."
            else:
                remediation = "Check request parameters and requested fields."
            raise SemanticScholarRequestError(
                f"Semantic Scholar rejected the request while {context} "
                f"(HTTP {response.status_code}). {remediation}"
            )
        response.raise_for_status()
        return response.json()

    def _request_json(
        self,
        url: str,
        params: dict[str, Any],
        *,
        failure_domain: _FailureDomain,
        raise_on_unavailable: bool = False,
        record_domain_failure: bool = True,
        context: str = "requesting data",
    ) -> dict[str, Any] | None:
        """Request JSON payload from direct Semantic Scholar REST endpoints.

        :param str url: Endpoint URL.
        :param dict[str, Any] params: Query parameters.
        :param _FailureDomain failure_domain: Capability sharing this retry budget.
        :param bool raise_on_unavailable: When ``True``, exhausted retries raise
            :class:`SemanticScholarUnavailableError` instead of returning
            ``None``, so callers can distinguish "no data" from "API down".
            HTTP 404 still returns ``None`` (genuinely absent resource).
        :param bool record_domain_failure: Whether an exhausted optional request
            should suppress later calls to the same capability in this collection.
        :param str context: Request description used in availability errors.
        :return dict[str, Any] | None: Parsed JSON payload or ``None`` on failure.
        :raises SemanticScholarRequestError: If a non-408/429 HTTP 4xx response
            rejects the request.
        """
        retryable = (_RetryableRequestError, requests.RequestException, ValueError)

        def _attempt() -> dict[str, Any] | None:
            """Issue one paced request, raising on retryable failures.

            :return dict[str, Any] | None: Parsed payload or ``None`` on 404.
            """
            return self._request_json_once(url, params, context=context)

        def _on_skip(
            skipped: _CandidateOperationSkippedError,
            scope_failure: SemanticScholarUnavailableError,
        ) -> None:
            """Skip this request after an earlier outage in the same capability.

            :param _CandidateOperationSkippedError skipped: Breaker error for this call.
            :param SemanticScholarUnavailableError scope_failure: Original outage.
            :return None: Reports the skip and yields no payload in tolerant mode.
            :raises _CandidateOperationSkippedError: In strict mode.
            """
            if raise_on_unavailable:
                raise skipped from scope_failure
            logger.warning(
                "Skipped %s after an earlier Semantic Scholar %s outage: %s",
                url,
                failure_domain.value,
                scope_failure,
            )
            return None

        def _log_retry(
            retry_state: RetryCallState, exc: Exception | None, wait_seconds: float
        ) -> None:
            """Log the upcoming retry with its computed wait.

            :param RetryCallState retry_state: Tenacity retry state.
            :param Exception | None exc: Failure that triggered this retry.
            :param float wait_seconds: Upcoming sleep duration in seconds.
            :return None: Emits one DEBUG line.
            """
            if exc is not None and self._is_rate_limit_error(exc):
                logger.debug(
                    "Rate limited by Semantic Scholar. Waiting %.1fs before retry.",
                    wait_seconds,
                )
            else:
                logger.debug(
                    "Request failed (attempt %s) for %s: %s. Retrying in %.1fs",
                    retry_state.attempt_number,
                    url,
                    exc,
                    wait_seconds,
                )

        def _on_exhausted(exc: Exception) -> None:
            """Apply the REST contract to an exhausted retry budget.

            :param Exception exc: Final operational failure.
            :return None: Reports the outage and yields no payload in tolerant mode.
            :raises SemanticScholarUnavailableError: In strict mode.
            """
            unavailable = self._record_unavailable(
                context,
                exc,
                failure_domain=failure_domain,
                omit_rate_limited_detail=True,
                record_domain_failure=record_domain_failure,
            )
            if raise_on_unavailable:
                raise unavailable from exc
            logger.error(
                "Failed to call %s after %s attempts: %s",
                url,
                API_CONFIG.max_retries,
                exc,
            )
            return None

        return self._run_with_retries(
            _attempt,
            failure_domain=failure_domain,
            predicate=retry_if_exception_type(retryable),
            on_skip=_on_skip,
            on_retry=_log_retry,
            on_exhausted=_on_exhausted,
            caught=retryable,
        )

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
                    limit=limit,
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
        parsed_limit = payloads._validate_integer_limit(limit, "limit")

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
        parsed_limit = payloads._validate_integer_limit(limit, "limit")
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        fields, cache_full_metadata = self._resolve_paper_fields(fields)

        payload = self._request_json(
            SEARCH_BASE_URL,
            {
                "query": normalized_query,
                "fields": ",".join(fields),
                "limit": parsed_limit,
            },
            failure_domain=_FailureDomain.SEARCH,
            raise_on_unavailable=raise_on_unavailable,
            context=f"searching for {normalized_query!r}",
        )
        if not payload:
            return []

        return self._papers_from_records(
            payload.get("data", []), cache_full_metadata=cache_full_metadata
        )


_client_instance: SemanticScholarClient | None = None
_client_lock = threading.Lock()


def get_client() -> SemanticScholarClient:
    """Get or create the shared Semantic Scholar API client instance.

    :return SemanticScholarClient: Process-wide singleton client.
    """
    global _client_instance
    client = _client_instance
    if client is None or client._closed:
        with _client_lock:
            if _client_instance is None or _client_instance._closed:
                _client_instance = SemanticScholarClient()
            # Return the instance observed under the lock: re-reading the
            # global outside it could observe a concurrent reset_client().
            client = _client_instance
    return client


def reset_client() -> None:
    """Reset cached client instance (for testing)."""
    global _client_instance
    client_to_close: SemanticScholarClient | None = None
    with _client_lock:
        client_to_close = _client_instance
        _client_instance = None
    if client_to_close is not None:
        client_to_close.close()
