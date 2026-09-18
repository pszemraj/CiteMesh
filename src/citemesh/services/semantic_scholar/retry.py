"""Retry delay helpers for Semantic Scholar HTTP requests."""

from __future__ import annotations

import logging
import math
import random

import requests

from citemesh.core import API_CONFIG

from .errors import _RetryableRequestError, _RetryExhaustedError

logger = logging.getLogger(__name__)

_MAX_BACKOFF_SECONDS = 60.0
_MAX_RETRY_AFTER_SECONDS = 300.0
_LONG_RETRY_WARNING_SECONDS = 30.0


def _jittered_backoff(
    attempt_number: int,
    *,
    retry_after: float | None = None,
    rate_limited: bool = False,
) -> float:
    """Compute full-jitter exponential backoff with a Retry-After floor.

    :param int attempt_number: 1-based number of the failed request.
    :param float | None retry_after: Server-requested delay.
    :param bool rate_limited: Whether the failure was HTTP 429.
    :return float: Delay before the next request attempt.
    """
    multiplier = API_CONFIG.retry_delay * (2.0 if rate_limited else 1.0)
    cap = min(multiplier * (2.0 ** (attempt_number - 1)), _MAX_BACKOFF_SECONDS)
    delay = random.uniform(0.0, cap)
    if retry_after is not None:
        delay = max(0.0, retry_after, delay)
    return delay


def _is_rate_limit_error(error: Exception) -> bool:
    """Return whether an error represents HTTP 429.

    :param Exception error: Request or exhausted-retry error.
    :return bool: Whether the final response was rate limited.
    """
    if isinstance(error, _RetryExhaustedError):
        error = error.cause
    if isinstance(error, _RetryableRequestError):
        return error.rate_limited
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == 429


def retry_after_seconds(
    source: BaseException | requests.Response, *, default: float | None = None
) -> float | None:
    """Read a numeric Retry-After header.

    :param BaseException | requests.Response source: Response or response-bearing error.
    :param float | None default: Fallback for a missing or invalid header.
    :return float | None: Parsed non-negative seconds or ``default``.
    """
    response = (
        source if hasattr(source, "headers") else getattr(source, "response", None)
    )
    headers = getattr(response, "headers", None)
    header = headers.get("Retry-After") if headers is not None else None
    if not header:
        return default
    try:
        seconds = float(header)
        if not math.isfinite(seconds):
            raise ValueError("Retry-After must be finite")
        return min(_MAX_RETRY_AFTER_SECONDS, max(0.0, seconds))
    except (TypeError, ValueError):
        logger.debug("Ignoring invalid Retry-After value %s", header)
        return default
