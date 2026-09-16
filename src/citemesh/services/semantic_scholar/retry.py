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

from .errors import (
    _RetryableRequestError,
    _RetryExhaustedError,
    _unwrap_sdk_retry_error,
)

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
    if isinstance(error, _RetryExhaustedError):
        error = error.cause
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


def retry_after_seconds(
    source: BaseException | requests.Response, *, default: float | None = None
) -> float | None:
    """Read the server's Retry-After hint from a response or from a failed request.

    Accepts either side of the transport split: a :class:`requests.Response` held
    by the REST path, or an exception carrying one on ``response`` as the SDK
    path sees it.

    :param BaseException | requests.Response source: Response to inspect, or an
        exception carrying one.
    :param float | None default: Value returned when no usable header is present.
    :return float | None: Retry-After seconds, or ``default`` when no response is
        available and when its header is missing or unparseable.
    """
    response = (
        source if hasattr(source, "headers") else getattr(source, "response", None)
    )
    headers = getattr(response, "headers", None)
    header = headers.get("Retry-After") if headers is not None else None
    if not header:
        return default
    try:
        return float(header)
    except (TypeError, ValueError):
        logger.debug(
            "Ignoring invalid Retry-After value %s; using default delay", header
        )
    return default


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
            retry_after = retry_after_seconds(exc)
        return _jittered_backoff(
            retry_state.attempt_number,
            retry_after=retry_after,
            rate_limited=exc is not None and _is_rate_limit_error(exc),
        )
