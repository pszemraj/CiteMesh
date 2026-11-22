"""Retry delay helpers for Semantic Scholar HTTP requests."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from citemesh.core import API_CONFIG

from .errors import _RetryableRequestError, _RetryExhaustedError

logger = logging.getLogger(__name__)

_MAX_BACKOFF_SECONDS = 300.0


def _retry_delay(
    attempt_number: int,
    *,
    retry_after: float | None = None,
) -> float:
    """Compute the section's exponential backoff with a Retry-After floor.

    :param int attempt_number: 1-based number of the failed request.
    :param float | None retry_after: Server-requested delay.
    :return float: Delay before the next request attempt.
    """
    delay = min(
        API_CONFIG.retry_delay * (2.0 ** (attempt_number - 1)), _MAX_BACKOFF_SECONDS
    )
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
    """Read Retry-After seconds or an HTTP date without shortening its wait.

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
            return default
    except (TypeError, ValueError):
        try:
            deadline = parsedate_to_datetime(header)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            seconds = (deadline - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            logger.debug("Ignoring invalid Retry-After header.")
            return default
    return max(0.0, seconds)
