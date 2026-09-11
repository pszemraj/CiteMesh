"""Exception taxonomy for the Semantic Scholar client.

Owns the public failure types, the internal contract/retry markers, and the
capability-scoped failure bookkeeping shared by a single discovery operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tenacity import RetryError


class SemanticScholarUnavailableError(RuntimeError):
    """Raised when the Semantic Scholar API stays unreachable after retries."""


class _FailureDomain(str, Enum):
    """Semantic Scholar capabilities with independent collection retry budgets."""

    REFERENCES = "references"
    CITATIONS = "citations"
    RECOMMENDATIONS = "recommendations"
    SEARCH = "search"
    PAPER_METADATA = "paper_metadata"


@dataclass
class _CandidateOperationState:
    """Failures shared by nested calls in one thread-local collection."""

    depth: int = 0
    failures: dict[_FailureDomain, SemanticScholarUnavailableError] = field(
        default_factory=dict
    )


class _CandidateOperationSkippedError(SemanticScholarUnavailableError):
    """Raised when a scoped operation skips a request after an earlier outage."""


class SemanticScholarRequestError(RuntimeError):
    """Raised when Semantic Scholar rejects a non-retryable client request."""


class _SemanticScholarResponseContractError(TypeError):
    """Raised when a successful SDK response cannot satisfy CiteMesh's schema."""


def _unwrap_sdk_retry_error(exc: Exception) -> Exception:
    """Recover the original exception from the SDK's one-attempt retry wrapper.

    :param Exception exc: Exception raised by the Semantic Scholar SDK.
    :return Exception: Wrapped attempt failure when available, otherwise ``exc``.
    """
    if not isinstance(exc, RetryError):
        return exc
    wrapped = exc.last_attempt.exception()
    return wrapped if isinstance(wrapped, Exception) else exc


def _raise_request_error(exc: Exception, context: str) -> None:
    """Surface a deterministic Semantic Scholar SDK request failure.

    :param Exception exc: SDK exception caused by an HTTP 400 or 403.
    :param str context: Human-readable request operation.
    :raises SemanticScholarRequestError: Always.
    """
    if isinstance(exc, PermissionError):
        remediation = "Check S2_API_KEY credentials and access permissions."
    else:
        remediation = "Check the paper ID and requested fields."
    raise SemanticScholarRequestError(
        f"Semantic Scholar rejected the request while {context}: {exc}. {remediation}"
    ) from exc


class _RetryableRequestError(RuntimeError):
    """Internal marker for transient request failures worth retrying."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        *,
        rate_limited: bool = False,
    ) -> None:
        """Create a retryable request error.

        :param str message: Failure description.
        :param float | None retry_after: Parsed Retry-After header seconds.
        :param bool rate_limited: Whether the failure was an HTTP 429.
        """
        super().__init__(message)
        self.retry_after = retry_after
        self.rate_limited = rate_limited
