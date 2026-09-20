"""Semantic Scholar HTTP transport and process-wide client lifecycle."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from time import monotonic
from typing import Any

import requests

from citemesh.core import API_CONFIG
from citemesh.data.cache import atomic_write_json, cache_operation_lock, get_cache_dir

from . import disk_cache, payloads, retry
from .endpoints import PAPER_BASE_URL, _EndpointsMixin
from .errors import (
    SemanticScholarRequestError,
    SemanticScholarUnavailableError,
    _CandidateOperationState,
    _RetryableRequestError,
    _RetryDiagnostics,
    _RetryExhaustedError,
)

logger = logging.getLogger(__name__)

_anonymous_pool_announced = False
_MIN_REQUEST_TIMEOUT_SECONDS = 0.001
_MAX_TRANSIENT_NOT_FOUND_ATTEMPTS = 3


class SemanticScholarClient(_EndpointsMixin):
    """Semantic Scholar client with cache-aware endpoints and bounded retries."""

    def __init__(
        self,
        timeout: float = API_CONFIG.default_timeout,
        *,
        api_key: str | None = None,
        refresh_paper_cache: bool = False,
        retry_budget_seconds: float | None = None,
    ) -> None:
        """Initialize the HTTP client.

        :param float timeout: Per-request timeout in seconds.
        :param str | None api_key: Explicit key, or ``None`` to read ``S2_API_KEY``.
        :param bool refresh_paper_cache: Bypass persisted paper records on reads.
        :param float | None retry_budget_seconds: Shared recovery-time allowance.
            ``None`` uses 90 seconds for anonymous access and no cap for keyed access;
            ``0`` explicitly disables the cap.
        :return None: Initializes the client.
        """
        if api_key is None:
            api_key = os.getenv("S2_API_KEY") or None
        if retry_budget_seconds is not None:
            retry_budget_seconds = float(retry_budget_seconds)
            if not math.isfinite(retry_budget_seconds) or retry_budget_seconds < 0:
                raise ValueError("retry_budget_seconds must be finite and non-negative")

        self.refresh_paper_cache = refresh_paper_cache
        self._api_key = api_key or None
        self.timeout = float(timeout)
        self.retry_budget_seconds = (
            API_CONFIG.anonymous_retry_budget_seconds
            if retry_budget_seconds is None and not api_key
            else 0.0
            if retry_budget_seconds is None
            else float(retry_budget_seconds)
        )
        self.requests_per_second = (
            API_CONFIG.authenticated_requests_per_second
            if api_key
            else API_CONFIG.requests_per_second
        )
        self._next_request_time = 0.0
        self._rate_limit_lock = threading.Lock()
        self._candidate_operation = threading.local()
        self._session = requests.Session()
        self._closed = False

        if api_key:
            self._session.headers["x-api-key"] = api_key
            logger.debug("Using Semantic Scholar API key")
        else:
            global _anonymous_pool_announced
            if not _anonymous_pool_announced:
                _anonymous_pool_announced = True
                logger.debug(
                    "No S2_API_KEY set; using the shared anonymous Semantic Scholar pool."
                )

    @contextlib.contextmanager
    def candidate_operation_scope(self) -> Iterator[None]:
        """Share one recovery allowance across a complete candidate collection.

        The outer scope also holds the cache operation lock so cache clearing cannot
        interleave with paper/reference reads and writes.

        :return Iterator[None]: Candidate-collection context.
        """
        state = getattr(self._candidate_operation, "state", None)
        owns_state = state is None
        operation_lock = (
            cache_operation_lock(get_cache_dir())
            if owns_state
            else contextlib.nullcontext()
        )
        with operation_lock:
            if owns_state:
                state = _CandidateOperationState(self.retry_budget_seconds)
                self._candidate_operation.state = state
            state.depth += 1
            try:
                yield
            finally:
                state.depth -= 1
                if owns_state and state.depth == 0:
                    if state.reference_cache_hits:
                        logger.debug(
                            "Reused cached reference enrichment for %d lookups.",
                            state.reference_cache_hits,
                        )
                    del self._candidate_operation.state

    def __enter__(self) -> SemanticScholarClient:
        """Return the client for context-manager use.

        :return SemanticScholarClient: This client.
        """
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> None:
        """Close the HTTP session when leaving a context manager.

        :param type | None exc_type: Exception type leaving the context.
        :param BaseException | None exc: Exception leaving the context.
        :param Any | None tb: Associated traceback.
        :return None: Closes the client.
        """
        self.close()

    def close(self) -> None:
        """Close the HTTP session and detach the singleton when applicable.

        :return None: Closes the client once.
        """
        if self._closed:
            return
        self._closed = True
        global _client_instance
        with _client_lock:
            if _client_instance is self:
                _client_instance = None
        with contextlib.suppress(Exception):
            self._session.close()

    def __del__(self) -> None:
        """Best-effort session cleanup during finalization.

        :return None: Closes any remaining session.
        """
        with contextlib.suppress(Exception):
            self.close()

    def _persist_reference_cache_entry(
        self, cache_path: Path, paper_id: str, reference_ids: list[str]
    ) -> None:
        """Persist normalized reference IDs.

        :param Path cache_path: Target JSON path.
        :param str paper_id: Normalized seed identifier.
        :param list[str] reference_ids: Ordered reference identifiers.
        :return None: Writes best-effort cache data.
        """
        normalized = payloads._coerce_cached_reference_ids(reference_ids) or []
        try:
            atomic_write_json(
                cache_path,
                {
                    "paper_id": paper_id,
                    "references": normalized,
                    "version": disk_cache.REFERENCE_CACHE_VERSION,
                },
            )
        except OSError as exc:
            logger.debug("Failed to persist reference cache for %s: %s", paper_id, exc)

    @staticmethod
    def _remaining_recovery_budget(
        state: _CandidateOperationState,
    ) -> float | None:
        """Return remaining shared recovery seconds, or ``None`` when uncapped.

        :param _CandidateOperationState state: Active collection state.
        :return float | None: Remaining seconds.
        """
        if state.retry_budget_seconds == 0:
            return None
        return max(0.0, state.retry_budget_seconds - state.recovery_seconds)

    @staticmethod
    def _charge_recovery(state: _CandidateOperationState, started_at: float) -> None:
        """Charge actual elapsed recovery work to the shared allowance.

        :param _CandidateOperationState state: Active collection state.
        :param float started_at: Monotonic start time.
        :return None: Updates the state.
        """
        state.recovery_seconds += max(0.0, monotonic() - started_at)

    def _retry_exhausted(
        self,
        state: _CandidateOperationState,
        *,
        operation: str,
        attempts: int,
        reason: str,
        cause: Exception | None = None,
    ) -> _RetryExhaustedError:
        """Build a terminal request error with current shared-budget usage.

        :param _CandidateOperationState state: Active collection state.
        :param str operation: Human-readable request operation.
        :param int attempts: Requests started for this HTTP operation.
        :param str reason: Retry stop reason.
        :param Exception | None cause: Last transport failure.
        :return _RetryExhaustedError: Diagnostic terminal error.
        """
        cause = cause or state.last_recovery_error or RuntimeError(reason)
        status_code = getattr(cause, "status_code", None)
        if status_code is None:
            status_code = getattr(getattr(cause, "response", None), "status_code", None)
        return _RetryExhaustedError(
            cause,
            _RetryDiagnostics(
                operation=operation,
                attempts=attempts,
                recovery_seconds=state.recovery_seconds,
                stop_reason=reason,
                status_code=status_code,
            ),
        )

    def _rate_limit(
        self,
        state: _CandidateOperationState | None = None,
        *,
        recovery_started_at: float | None = None,
        operation: str = "requesting data",
        attempts: int = 0,
    ) -> None:
        """Reserve one globally paced request slot.

        :param _CandidateOperationState | None state: Active recovery state.
        :param float | None recovery_started_at: Start of a budgeted attempt.
        :param str operation: Request operation for a budget error.
        :param int attempts: Attempts already started for diagnostics.
        :return None: Sleeps when the shared request pace requires it.
        :raises _RetryExhaustedError: If pacing cannot fit in the remaining budget.
        """
        min_interval = 1.0 / self.requests_per_second
        with self._rate_limit_lock:
            wait_seconds = max(0.0, self._next_request_time - monotonic())
            if state is not None and recovery_started_at is not None:
                remaining = self._remaining_recovery_budget(state)
                if remaining is not None:
                    remaining -= max(0.0, monotonic() - recovery_started_at)
                    if wait_seconds > max(0.0, remaining):
                        raise self._retry_exhausted(
                            state,
                            operation=operation,
                            attempts=attempts,
                            reason="request pacing exceeds remaining recovery budget",
                        )
            if wait_seconds:
                time.sleep(wait_seconds)
            self._next_request_time = monotonic() + min_interval

    def _request_json_once(
        self,
        url: str,
        params: dict[str, Any],
        *,
        context: str,
        timeout: float,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        """Issue one HTTP request and classify its response.

        :param str url: Semantic Scholar endpoint.
        :param dict[str, Any] params: Query parameters.
        :param str context: Request description used in errors.
        :param float timeout: Timeout for this attempt.
        :param dict[str, Any] | None payload: Optional POST body.
        :return Any: Decoded JSON, or ``None`` for HTTP 404.
        """
        kwargs = {"params": params, "timeout": timeout}
        response = (
            self._session.get(url, **kwargs)
            if payload is None
            else self._session.post(url, json=payload, **kwargs)
        )
        if response.status_code == 404:
            return None
        if response.status_code == 400 and url == f"{PAPER_BASE_URL}/batch":
            with contextlib.suppress(ValueError):
                error_payload = response.json()
                if (
                    error_payload == {"error": "No valid paper ids given"}
                    and payload is not None
                    and isinstance(payload.get("ids"), list)
                ):
                    return [None] * len(payload["ids"])
        if response.status_code in {408, 429} or response.status_code >= 500:
            raise _RetryableRequestError(
                f"HTTP {response.status_code} from {url}",
                retry_after=retry.retry_after_seconds(response),
                status_code=response.status_code,
                rate_limited=response.status_code == 429,
            )
        if 400 <= response.status_code < 500:
            remediation = (
                "Check S2_API_KEY credentials and access permissions."
                if response.status_code in {401, 403}
                else "Check request parameters and requested fields."
            )
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
        context: str,
        payload: dict[str, Any] | None = None,
        raise_on_unavailable: bool = False,
        retry_not_found: bool = False,
    ) -> Any:
        """Run one HTTP request with per-request retries and shared recovery time.

        :param str url: Semantic Scholar endpoint.
        :param dict[str, Any] params: Query parameters.
        :param str context: Human-readable operation.
        :param dict[str, Any] | None payload: Optional POST body.
        :param bool raise_on_unavailable: Raise after recovery stops.
        :param bool retry_not_found: Treat HTTP 404 as transient only during the
            first three request attempts.
        :return Any: Decoded payload or ``None`` for a tolerated failure/not-found.
        """
        if getattr(self._candidate_operation, "state", None) is None:
            with self.candidate_operation_scope():
                return self._request_json(
                    url,
                    params,
                    context=context,
                    payload=payload,
                    raise_on_unavailable=raise_on_unavailable,
                    retry_not_found=retry_not_found,
                )

        state: _CandidateOperationState = self._candidate_operation.state
        remaining = self._remaining_recovery_budget(state)
        if remaining is not None and remaining <= 0:
            exhausted = self._retry_exhausted(
                state,
                operation=context,
                attempts=0,
                reason="shared recovery budget exhausted",
            )
            return self._handle_unavailable(exhausted, raise_on_unavailable)

        last_error: Exception | None = None
        for attempt in range(1, API_CONFIG.max_retries + 1):
            recovery_started_at = monotonic() if attempt > 1 else None
            request_started_at: float | None = None
            try:
                self._rate_limit(
                    state,
                    recovery_started_at=recovery_started_at,
                    operation=context,
                    attempts=attempt - 1,
                )
                request_started_at = monotonic()
                timeout = self.timeout
                if recovery_started_at is not None:
                    remaining = self._remaining_recovery_budget(state)
                    if remaining is not None:
                        remaining -= max(0.0, monotonic() - recovery_started_at)
                        if remaining <= 0:
                            raise self._retry_exhausted(
                                state,
                                operation=context,
                                attempts=attempt - 1,
                                reason="shared recovery budget exhausted",
                            )
                        timeout = max(
                            _MIN_REQUEST_TIMEOUT_SECONDS, min(timeout, remaining)
                        )
                result = self._request_json_once(
                    url,
                    params,
                    context=context,
                    timeout=timeout,
                    payload=payload,
                )
                if result is None and retry_not_found:
                    raise _RetryableRequestError(
                        f"HTTP 404 from {url}", status_code=404
                    )
            except _RetryExhaustedError as exhausted:
                if recovery_started_at is not None:
                    self._charge_recovery(state, recovery_started_at)
                    exhausted = self._retry_exhausted(
                        state,
                        operation=context,
                        attempts=attempt - 1,
                        reason=exhausted.retry_diagnostics.stop_reason,
                    )
                return self._handle_unavailable(exhausted, raise_on_unavailable)
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ContentDecodingError,
                requests.exceptions.JSONDecodeError,
                _RetryableRequestError,
            ) as exc:
                charged_from = (
                    recovery_started_at
                    if recovery_started_at is not None
                    else request_started_at
                )
                if charged_from is not None:
                    self._charge_recovery(state, charged_from)
                state.last_recovery_error = exc
                last_error = exc
            except (
                SemanticScholarRequestError,
                requests.RequestException,
                TypeError,
                ValueError,
            ):
                if recovery_started_at is not None:
                    self._charge_recovery(state, recovery_started_at)
                raise
            else:
                if recovery_started_at is not None:
                    self._charge_recovery(state, recovery_started_at)
                return result

            if (
                isinstance(last_error, _RetryableRequestError)
                and last_error.status_code == 404
                and attempt >= _MAX_TRANSIENT_NOT_FOUND_ATTEMPTS
            ):
                exhausted = self._retry_exhausted(
                    state,
                    operation=context,
                    attempts=attempt,
                    reason="maximum transient not-found attempts reached",
                    cause=last_error,
                )
                return self._handle_unavailable(exhausted, raise_on_unavailable)

            if attempt == API_CONFIG.max_retries:
                exhausted = self._retry_exhausted(
                    state,
                    operation=context,
                    attempts=attempt,
                    reason="maximum retry attempts reached",
                    cause=last_error,
                )
                return self._handle_unavailable(exhausted, raise_on_unavailable)

            remaining = self._remaining_recovery_budget(state)
            if remaining is not None and remaining <= 0:
                exhausted = self._retry_exhausted(
                    state,
                    operation=context,
                    attempts=attempt,
                    reason="shared recovery budget exhausted",
                    cause=last_error,
                )
                return self._handle_unavailable(exhausted, raise_on_unavailable)
            delay = retry._jittered_backoff(
                attempt,
                retry_after=getattr(last_error, "retry_after", None),
                rate_limited=last_error is not None
                and retry._is_rate_limit_error(last_error),
            )
            if remaining is not None and delay > remaining:
                exhausted = self._retry_exhausted(
                    state,
                    operation=context,
                    attempts=attempt,
                    reason="next wait exceeds remaining recovery budget",
                    cause=last_error,
                )
                return self._handle_unavailable(exhausted, raise_on_unavailable)
            logger.debug(
                "Semantic Scholar request failed while %s (attempt %d/%d); "
                "retrying in %.1fs: %s",
                context,
                attempt,
                API_CONFIG.max_retries,
                delay,
                last_error,
            )
            sleep_started_at = monotonic()
            try:
                time.sleep(delay)
            finally:
                self._charge_recovery(state, sleep_started_at)

        raise AssertionError("retry loop terminated without a result")

    def _handle_unavailable(
        self, exhausted: _RetryExhaustedError, raise_on_unavailable: bool
    ) -> None:
        """Raise or log a request exhaustion according to endpoint policy.

        :param _RetryExhaustedError exhausted: Terminal retry error.
        :param bool raise_on_unavailable: Whether to raise the public error.
        :return None: Tolerated requests yield no payload.
        :raises SemanticScholarUnavailableError: In strict mode.
        """
        unavailable = self._unavailable_error(exhausted)
        if raise_on_unavailable:
            raise unavailable from exhausted
        logger.warning("%s", unavailable)
        return None

    @staticmethod
    def _unavailable_error(
        exhausted: _RetryExhaustedError,
    ) -> SemanticScholarUnavailableError:
        """Convert internal retry exhaustion to the public actionable error.

        :param _RetryExhaustedError exhausted: Terminal request error.
        :return SemanticScholarUnavailableError: Public availability error.
        """
        diagnostics = exhausted.retry_diagnostics
        status = (
            f", HTTP {diagnostics.status_code}"
            if diagnostics.status_code is not None
            else ""
        )
        flavor = (
            "rate-limited" if retry._is_rate_limit_error(exhausted) else "unavailable"
        )
        return SemanticScholarUnavailableError(
            f"Semantic Scholar API {flavor} while {diagnostics.operation} "
            f"after {diagnostics.attempts} attempts and "
            f"{diagnostics.recovery_seconds:.1f}s recovery time "
            f"({diagnostics.stop_reason}{status}): {exhausted.cause}. "
            "Retry shortly, set S2_API_KEY for a dedicated rate limit, or use the "
            "local-corpus semantic source (requires embeddings extras and a "
            "downloaded, embedded corpus)."
        )


_client_instance: SemanticScholarClient | None = None
_client_lock = threading.RLock()


def get_client() -> SemanticScholarClient:
    """Return the process-wide client for the current environment API key.

    :return SemanticScholarClient: Shared client instance.
    """
    global _client_instance
    desired_api_key = os.getenv("S2_API_KEY") or None
    client = _client_instance
    if client is None or client._closed or client._api_key != desired_api_key:
        with _client_lock:
            desired_api_key = os.getenv("S2_API_KEY") or None
            if (
                _client_instance is None
                or _client_instance._closed
                or _client_instance._api_key != desired_api_key
            ):
                _client_instance = SemanticScholarClient(
                    api_key=desired_api_key if desired_api_key is not None else ""
                )
            client = _client_instance
    return client


def reset_client() -> None:
    """Close and clear the process-wide client.

    :return None: Resets singleton state.
    """
    global _client_instance
    with _client_lock:
        client = _client_instance
        _client_instance = None
    if client is not None:
        client.close()
