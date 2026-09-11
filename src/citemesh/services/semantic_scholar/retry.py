"""Retry policy for Semantic Scholar requests.

Owns the jittered-backoff schedule, Retry-After parsing, rate-limit detection,
and the tenacity wait strategy shared by the SDK and REST transports.
"""

from __future__ import annotations

import logging
import random
import re

import requests
from tenacity import RetryCallState
from tenacity.wait import wait_base

from citemesh.core import API_CONFIG

from .errors import _RetryableRequestError, _unwrap_sdk_retry_error

logger = logging.getLogger(__name__)


_MAX_BACKOFF_SECONDS = 60.0
# Servers under sustained saturation have answered with hour-scale cooldowns;
# a bounded honor window keeps a single sleep from silently stalling a build.
_MAX_RETRY_AFTER_SECONDS = 300.0
_LONG_RETRY_WARNING_SECONDS = 30.0


def _jittered_backoff(
    attempt_number: int,
    *,
    retry_after: float | None = None,
    rate_limited: bool = False,
) -> float:
    """Compute full-jitter exponential backoff floored at the server's Retry-After.

    Repeated 429s escalate beyond a flat server hint (S2 keeps answering
    ``Retry-After: 2`` while its shared pool stays saturated), while jitter
    de-synchronizes concurrent clients.

    :param int attempt_number: 1-based retry attempt number.
    :param float | None retry_after: Server-provided Retry-After seconds.
    :param bool rate_limited: Whether the failure was an HTTP 429.
    :return float: Capped jitter delay, or the server's longer requested wait
        bounded by ``_MAX_RETRY_AFTER_SECONDS``.
    """
    multiplier = API_CONFIG.retry_delay * (2.0 if rate_limited else 1.0)
    cap = min(multiplier * (2.0 ** (attempt_number - 1)), _MAX_BACKOFF_SECONDS)
    wait = random.uniform(0.0, cap)
    if retry_after is not None:
        wait = max(min(retry_after, _MAX_RETRY_AFTER_SECONDS), wait)
    return wait


def _is_rate_limit_error(error: Exception) -> bool:
    """Detect rate-limit exceptions.

    :param Exception error: Exception from request/client layer.
    :return bool: ``True`` when the error indicates HTTP 429.
    """
    error = _unwrap_sdk_retry_error(error)
    if isinstance(error, _RetryableRequestError):
        return error.rate_limited
    response = getattr(error, "response", None)
    if response is not None:
        return getattr(response, "status_code", None) == 429
    # The SDK discards response objects and uses this built-in exception.
    return isinstance(error, ConnectionRefusedError) and bool(
        re.search(r"\bHTTP(?: status)? 429\b", str(error))
    )


def _get_retry_after(error: Exception) -> float | None:
    """Extract Retry-After from library or HTTP errors.

    :param Exception error: Exception instance captured from request.
    :return float | None: Parsed Retry-After value in seconds, if available.
    """
    if isinstance(error, requests.RequestException) and error.response is not None:
        header = error.response.headers.get("Retry-After")
        if header:
            try:
                return float(header)
            except ValueError:
                pass
        return None

    for candidate in ("response",):
        response = getattr(error, candidate, None)
        if response is not None and hasattr(response, "headers"):
            header = response.headers.get("Retry-After")  # type: ignore[attr-defined]
            if header:
                try:
                    return float(header)
                except ValueError:
                    pass
    return None


def _safe_retry_after(response: requests.Response) -> float:
    """Extract Retry-After from direct HTTP responses safely.

    :param requests.Response response: HTTP response to inspect.
    :return float: Retry delay in seconds (header value or default delay).
    """
    header = response.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            logger.debug(
                "Ignoring invalid Retry-After value %s; using default delay",
                header,
            )
    return API_CONFIG.retry_delay


def _warn_on_long_wait(retry_state: RetryCallState) -> None:
    """Announce outage-scale retry sleeps at the default log level.

    Per-endpoint retry callbacks stay debug-only for quick blips; a wait at or
    above the warning threshold means the API is effectively down and silence
    would read as a hang.

    :param RetryCallState retry_state: Failed attempt with its next sleep action.
    :return None: Emits one WARNING when the upcoming sleep is long.
    """
    sleep_seconds = retry_state.next_action.sleep if retry_state.next_action else 0.0
    if sleep_seconds >= _LONG_RETRY_WARNING_SECONDS:
        logger.warning(
            "Semantic Scholar unavailable (attempt %s/%s); waiting %.0fs "
            "before retrying: %s",
            retry_state.attempt_number,
            API_CONFIG.max_retries,
            sleep_seconds,
            retry_state.outcome.exception() if retry_state.outcome else None,
        )


class _S2BackoffWait(wait_base):
    """Tenacity wait strategy applying the shared jittered-backoff policy."""

    def __call__(self, retry_state: RetryCallState) -> float:
        """Compute the wait for the given retry state.

        :param RetryCallState retry_state: Tenacity retry state.
        :return float: Wait duration in seconds.
        """
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is None and exc is not None:
            retry_after = _get_retry_after(exc)
        return _jittered_backoff(
            retry_state.attempt_number,
            retry_after=retry_after,
            rate_limited=exc is not None and _is_rate_limit_error(exc),
        )
