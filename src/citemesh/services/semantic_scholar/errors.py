"""Errors and request-local retry state for Semantic Scholar."""

from __future__ import annotations

from dataclasses import dataclass


class SemanticScholarUnavailableError(RuntimeError):
    """Raised when the Semantic Scholar API stays unavailable after retries."""


class SemanticScholarRequestError(RuntimeError):
    """Raised when Semantic Scholar rejects a non-retryable request."""


class _SemanticScholarResponseContractError(TypeError):
    """Raised when a successful response has an unusable shape."""


@dataclass
class _CandidateOperationState:
    """Recovery budget shared by one outer candidate collection."""

    retry_budget_seconds: float
    depth: int = 0
    recovery_seconds: float = 0.0
    reference_cache_hits: int = 0
    last_recovery_error: Exception | None = None


@dataclass(frozen=True)
class _RetryDiagnostics:
    """Observed work and stop condition for one failed HTTP request."""

    operation: str
    attempts: int
    recovery_seconds: float
    stop_reason: str
    status_code: int | None = None


class _RetryExhaustedError(RuntimeError):
    """Attach retry diagnostics to the final transport failure."""

    def __init__(self, cause: Exception, diagnostics: _RetryDiagnostics) -> None:
        """Store the final failure and request-local diagnostics.

        :param Exception cause: Last retryable transport failure.
        :param _RetryDiagnostics diagnostics: Request retry details.
        :return None: Initializes the error.
        """
        super().__init__(str(cause))
        self.cause = cause
        self.retry_diagnostics = diagnostics


class _RetryableRequestError(RuntimeError):
    """Internal marker for a transient HTTP response."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        status_code: int | None = None,
        rate_limited: bool = False,
    ) -> None:
        """Create a retryable response error.

        :param str message: Failure description.
        :param float | None retry_after: Server-requested wait.
        :param int | None status_code: HTTP response status when available.
        :param bool rate_limited: Whether the response was HTTP 429.
        :return None: Initializes the error.
        """
        super().__init__(message)
        self.retry_after = retry_after
        self.status_code = status_code
        self.rate_limited = rate_limited
