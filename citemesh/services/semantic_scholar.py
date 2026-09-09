"""Semantic Scholar API client with error handling and caching."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import numbers
import os
import random
import re
import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import quote

import requests
from semanticscholar import SemanticScholar
from semanticscholar.SemanticScholarException import (
    BadQueryParametersException,
    ObjectNotFoundException,
)
from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
)
from tenacity.wait import wait_base

from citemesh.core import API_CONFIG, Author, Paper
from citemesh.data import get_cache_dir
from citemesh.data.cache import atomic_write_json
from citemesh.paper_ids import (
    external_ids_from_canonical_paper_id,
    normalize_paper_id,
    paper_identifier_aliases,
)

logger = logging.getLogger(__name__)

# Optional runtime override for tests and one-off callers.
# When unset, reference cache paths are resolved from ``get_cache_dir`` per call.
REFERENCE_CACHE_DIR: Optional[Path] = None
REFERENCE_CACHE_VERSION = 1
RECOMMENDATION_BASE_URL = (
    "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
)
PAPER_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
SEARCH_BASE_URL = f"{PAPER_BASE_URL}/search"
# semanticscholar 0.11 raises this when S2 encodes an empty relation page as
# ``{"data": null}`` instead of ``{"data": []}``.
_SDK_NULL_RELATION_PAGE_ERROR = "'NoneType' object is not iterable"
DEFAULT_PAPER_FIELDS = (
    "paperId",
    "title",
    "year",
    "authors",
    "citationCount",
    "abstract",
    "fieldsOfStudy",
    "externalIds",
    "venue",
    "publicationVenue",
    "journal",
)


S2_API_KEY_SIGNUP_URL = "https://www.semanticscholar.org/product/api"
_anonymous_pool_announced = False


class SemanticScholarUnavailableError(RuntimeError):
    """Raised when the Semantic Scholar API stays unreachable after retries."""


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


_MAX_BACKOFF_SECONDS = 60.0
# Servers under sustained saturation have answered with hour-scale cooldowns;
# a bounded honor window keeps a single sleep from silently stalling a build.
_MAX_RETRY_AFTER_SECONDS = 300.0
_LONG_RETRY_WARNING_SECONDS = 30.0


def _jittered_backoff(
    attempt_number: int,
    *,
    retry_after: Optional[float] = None,
    rate_limited: bool = False,
) -> float:
    """Compute full-jitter exponential backoff floored at the server's Retry-After.

    Repeated 429s escalate beyond a flat server hint (S2 keeps answering
    ``Retry-After: 2`` while its shared pool stays saturated), while jitter
    de-synchronizes concurrent clients.

    :param int attempt_number: 1-based retry attempt number.
    :param Optional[float] retry_after: Server-provided Retry-After seconds.
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


class _RetryableRequestError(RuntimeError):
    """Internal marker for transient request failures worth retrying."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        *,
        rate_limited: bool = False,
    ) -> None:
        """Create a retryable request error.

        :param str message: Failure description.
        :param Optional[float] retry_after: Parsed Retry-After header seconds.
        :param bool rate_limited: Whether the failure was an HTTP 429.
        """
        super().__init__(message)
        self.retry_after = retry_after
        self.rate_limited = rate_limited


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
            retry_after = SemanticScholarClient._get_retry_after(exc)
        return _jittered_backoff(
            retry_state.attempt_number,
            retry_after=retry_after,
            rate_limited=exc is not None
            and SemanticScholarClient._is_rate_limit_error(exc),
        )


def _reference_cache_dir() -> Path:
    """Resolve reference-cache directory at call time.

    :return Path: Directory where reference cache JSON files are stored.
    """
    if REFERENCE_CACHE_DIR is None:
        return get_cache_dir("references")

    resolved = Path(REFERENCE_CACHE_DIR)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _is_sdk_null_relation_page(error: TypeError) -> bool:
    """Identify the SDK failure used for a valid empty relation page.

    :param TypeError error: Exception raised while the SDK decodes a relation page.
    :return bool: Whether the exception represents an S2 ``data: null`` response.
    """
    return str(error) == _SDK_NULL_RELATION_PAGE_ERROR


def _default_paper_fields() -> List[str]:
    """Return a mutable default field list for paper-like API endpoints.

    :return List[str]: Default paper fields for search/recommendation/get operations.
    """
    return list(DEFAULT_PAPER_FIELDS)


def _paper_lookup_keys(paper: Paper) -> set[str]:
    """Build normalized aliases for matching batch responses to requested IDs.

    :param Paper paper: Converted paper payload from Semantic Scholar.
    :return set[str]: Normalized identifier aliases for the paper.
    """
    return set(
        paper_identifier_aliases(
            paper_id=paper.paper_id,
            arxiv_id=paper.arxiv_id,
            doi=paper.doi,
        )
    )


def _reference_cache_path(paper_id: str) -> Path:
    """Build cache file path for a normalized paper identifier.

    :param str paper_id: Normalized paper identifier.
    :return Path: JSON cache path for the paper's reference IDs.
    """
    digest = hashlib.sha1(paper_id.encode("utf-8")).hexdigest()
    return _reference_cache_dir() / f"{digest}.json"


def _paper_cache_path(paper_id: str) -> Path:
    """Locate persisted metadata for a normalized paper identifier.

    :param str paper_id: Normalized requested ID or paper alias.
    :return Path: Paper metadata JSON path.
    """
    digest = hashlib.sha1(paper_id.encode("utf-8")).hexdigest()
    return get_cache_dir("papers") / f"{digest}.json"


def _load_cached_paper(paper_id: str) -> Optional[Paper]:
    """Read paper metadata, treating unreadable entries as cache misses.

    :param str paper_id: Normalized requested paper identifier.
    :return Optional[Paper]: Fresh paper instance, or ``None`` on a cache miss.
    """
    try:
        data = json.loads(_paper_cache_path(paper_id).read_text(encoding="utf-8"))
        data["authors"] = [Author(**author) for author in data["authors"]]
        return Paper(**data)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError):
        return None


def _persist_paper(paper: Paper, requested_id: str) -> None:
    """Save successful metadata under the requested ID and known aliases.

    :param Paper paper: Converted Semantic Scholar paper metadata.
    :param str requested_id: Normalized identifier used for the request.
    :return None: Writes metadata independently of embeddings and references.
    """
    data = asdict(paper)
    data["references"] = []
    data["is_seed"] = False
    for alias in _paper_lookup_keys(paper) | {requested_id}:
        try:
            atomic_write_json(_paper_cache_path(alias), data)
        except OSError as exc:
            logger.debug("Failed to persist paper cache for %s: %s", alias, exc)


def _reference_id_candidate(raw_value: Any) -> Optional[str]:
    """Extract a paper ID from accepted reference payload shapes.

    Supports current cache entries, legacy mixed-format cache entries, and
    Semantic Scholar relation objects returned by the SDK.

    :param Any raw_value: Raw reference-like entry.
    :return Optional[str]: Candidate paper ID string, or ``None`` when absent.
    """
    if isinstance(raw_value, str):
        return raw_value

    if isinstance(raw_value, dict):
        for key in ("paperId", "paper_id"):
            candidate = raw_value.get(key)
            if isinstance(candidate, str):
                return candidate

        nested_paper = raw_value.get("paper")
        if isinstance(nested_paper, dict):
            for key in ("paperId", "paper_id"):
                candidate = nested_paper.get(key)
                if isinstance(candidate, str):
                    return candidate
        return None

    for attr in ("paperId", "paper_id"):
        candidate = getattr(raw_value, attr, None)
        if isinstance(candidate, str):
            return candidate

    nested_paper = getattr(raw_value, "paper", None)
    for attr in ("paperId", "paper_id"):
        candidate = getattr(nested_paper, attr, None)
        if isinstance(candidate, str):
            return candidate

    return None


def _is_unresolved_reference(record: Any) -> bool:
    """Recognize a valid relation record with an explicitly null paper identifier.

    :param Any record: SDK relation record or equivalent mapping.
    :return bool: Whether a paper ID is present and null on the relation's paper.
    """
    paper = (
        record.get("paper", record)
        if isinstance(record, dict)
        else getattr(record, "paper", record)
    )
    # SDK properties default to None even when the response omits the field.
    paper = getattr(paper, "raw_data", paper)
    if isinstance(paper, dict):
        return "paperId" in paper and paper["paperId"] is None
    return hasattr(paper, "paperId") and paper.paperId is None


def _normalize_reference_ids(payload: Any, *, strict: bool) -> Optional[List[str]]:
    """Normalize reference payloads under cache or live-response rules.

    Both callers accept current list-of-string payloads and legacy mixed
    relation-shaped entries. Strict cache parsing marks malformed non-empty
    lists invalid so they can be rebuilt; live API parsing tolerates them.

    :param Any payload: Raw reference payload.
    :param bool strict: Whether malformed payloads return ``None``.
    :return Optional[List[str]]: Normalized IDs, or ``None`` for invalid strict data.
    """
    if not isinstance(payload, list):
        return None if strict else []

    normalized: List[str] = []
    seen: set[str] = set()

    for raw_value in payload:
        candidate = _reference_id_candidate(raw_value)
        if candidate is None:
            continue
        paper_id = candidate.strip()
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        normalized.append(paper_id)

    if payload and not normalized:
        return None if strict else []
    return normalized


def _coerce_cached_reference_ids(payload: Any) -> Optional[List[str]]:
    """Validate and normalize cached reference ID payloads.

    :param Any payload: Cached ``references`` field from JSON payload.
    :return Optional[List[str]]: Normalized ID list, or ``None`` when invalid.
    """
    return _normalize_reference_ids(payload, strict=True)


def _validate_integer_limit(
    limit: int, field_name: str, allow_zero: bool = False
) -> int:
    """Validate API limit argument values and return normalized int.

    :param int limit: Raw limit value supplied by caller.
    :param str field_name: Parameter name used in error messages.
    :param bool allow_zero: Whether zero is accepted as a disable switch.
    :return int: Parsed integer limit value.
    :raises ValueError: If value is non-integer or below the accepted minimum.
    """
    if isinstance(limit, bool) or not isinstance(limit, numbers.Integral):
        raise ValueError(f"{field_name} must be an integer, got {limit!r}")

    parsed_limit = int(limit)
    minimum = 0 if allow_zero else 1
    if parsed_limit < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}, got {parsed_limit}")
    return parsed_limit


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
        api_key: Optional[str] = None,
        refresh_paper_cache: bool = False,
    ):
        """
        Initialize the API client.

        :param float timeout: Request timeout in seconds.
        :param Optional[str] api_key: Explicit API key, or ``None`` to read S2_API_KEY.
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
                    S2_API_KEY_SIGNUP_URL,
                )

    def __enter__(self) -> "SemanticScholarClient":
        """Return this client for context-manager use."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc: Optional[BaseException],
        tb: Optional[Any],
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
        self, cache_path: Path, paper_id: str, reference_ids: List[str]
    ) -> None:
        """Persist normalized reference IDs to cache with best-effort durability.

        :param Path cache_path: Target cache file path.
        :param str paper_id: Normalized paper ID for payload metadata.
        :param List[str] reference_ids: Reference IDs to persist.
        :return None: Writes cache payload when filesystem operations succeed.
        """
        normalized_reference_ids = _coerce_cached_reference_ids(reference_ids)
        if normalized_reference_ids is None:
            normalized_reference_ids = []

        try:
            atomic_write_json(
                cache_path,
                {
                    "paper_id": paper_id,
                    "references": normalized_reference_ids,
                    "version": REFERENCE_CACHE_VERSION,
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

    @staticmethod
    def _get_retry_after(error: Exception) -> Optional[float]:
        """Extract Retry-After from library or HTTP errors.

        :param Exception error: Exception instance captured from request.
        :return Optional[float]: Parsed Retry-After value in seconds, if available.
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

    @staticmethod
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

    def _call_with_retries(
        self,
        operation: Callable[[], Any],
        *,
        on_retry: Callable[[int, float, Exception], None],
        on_final_failure: Callable[[Exception], Any],
        handled_exceptions: tuple[
            tuple[type[Exception], Callable[[Exception], Any]], ...
        ] = (),
    ) -> Any:
        """Run an API operation with shared retry/backoff behavior.

        :param Callable[[], Any] operation: Zero-argument API operation to execute.
        :param Callable[[int, float, Exception], None] on_retry: Callback invoked before
            each retry with 1-based attempt count, sleep delay, and triggering exception.
        :param Callable[[Exception], Any] on_final_failure: Callback used to produce the
            final return value or raise when retries are exhausted.
        :param tuple[tuple[type[Exception], Callable[[Exception], Any]], ...] handled_exceptions:
            Exception-specific handlers that short-circuit normal retry handling.
        :return Any: Result produced by ``operation`` or one of the failure handlers.
        """

        def _before_sleep(state: RetryCallState) -> None:
            """Report the scheduled retry using the endpoint's existing logger.

            :param RetryCallState state: Failed attempt with its next sleep action.
            :return None: Invokes the endpoint's retry callback.
            """
            _warn_on_long_wait(state)
            on_retry(
                state.attempt_number, state.next_action.sleep, state.outcome.exception()
            )

        retryer = Retrying(
            stop=stop_after_attempt(API_CONFIG.max_retries),
            wait=_S2BackoffWait(),
            retry=(
                retry_if_exception_type(Exception)
                & retry_if_not_exception_type(
                    (TypeError, ValueError, SemanticScholarRequestError)
                    + tuple(kind for kind, _handler in handled_exceptions)
                )
            )
            | retry_if_exception_type(
                (json.JSONDecodeError, requests.exceptions.JSONDecodeError)
            ),
            before_sleep=_before_sleep,
            sleep=lambda seconds: time.sleep(seconds),
            reraise=True,
        )
        try:
            for attempt in retryer:
                with attempt:
                    try:
                        return operation()
                    except Exception as raw_exc:
                        raise _unwrap_sdk_retry_error(raw_exc)
        except Exception as exc:
            # SDK JSON decoding can fail on transient HTML/plain-text error bodies.
            if not isinstance(
                exc, (json.JSONDecodeError, requests.exceptions.JSONDecodeError)
            ):
                for error_type, handler in handled_exceptions:
                    if isinstance(exc, error_type):
                        return handler(exc)
                if isinstance(
                    exc, (TypeError, ValueError, SemanticScholarRequestError)
                ):
                    raise
            return on_final_failure(exc)

        raise AssertionError("retry loop exhausted without returning")

    @staticmethod
    def _extract_venue_name(value: object) -> str:
        """Normalize venue-like payload values into a display string.

        :param object value: Raw venue payload value.
        :return str: Normalized venue string (empty when unavailable).
        """
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            value = value.get("name")
        else:
            value = getattr(value, "name", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return ""

    @classmethod
    def _extract_venue(cls, *candidates: object) -> str:
        """Extract the first non-empty venue label from candidate payloads.

        :param object candidates: Venue candidate payloads.
        :return str: Normalized venue string (empty when unavailable).
        """
        for candidate in candidates:
            venue = cls._extract_venue_name(candidate)
            if venue:
                return venue
        return ""

    @staticmethod
    def _normalize_external_id(raw_value: object) -> str:
        """Normalize optional external-id strings.

        :param object raw_value: Raw external ID payload.
        :return str: Normalized external ID string (empty when unavailable).
        """
        if not isinstance(raw_value, str):
            return ""
        return raw_value.strip()

    @classmethod
    def _extract_external_ids_from_mapping(
        cls, mapping: Dict[str, Any]
    ) -> tuple[str, str]:
        """Extract arXiv and DOI IDs from external-id style mappings.

        :param Dict[str, Any] mapping: External IDs map.
        :return tuple[str, str]: ``(arxiv_id, doi)`` normalized identifiers.
        """
        normalized = {str(key).lower(): value for key, value in mapping.items()}
        arxiv_id = cls._normalize_external_id(normalized.get("arxiv"))
        doi = cls._normalize_external_id(normalized.get("doi"))
        return arxiv_id, doi

    @classmethod
    def _extract_external_ids(cls, external_ids: object) -> tuple[str, str]:
        """Extract arXiv and DOI values from raw external-id payloads.

        :param object external_ids: Raw external ID payload.
        :return tuple[str, str]: ``(arxiv_id, doi)`` normalized identifiers.
        """
        if isinstance(external_ids, dict):
            return cls._extract_external_ids_from_mapping(external_ids)
        return "", ""

    @classmethod
    def _resolve_external_ids(
        cls, external_ids: object, paper_id: object
    ) -> tuple[str, str]:
        """Resolve external IDs from payload data with canonical-ID fallback.

        :param object external_ids: Raw external ID payload.
        :param object paper_id: Canonical or near-canonical paper ID fallback.
        :return tuple[str, str]: ``(arxiv_id, doi)`` pair.
        """
        arxiv_id, doi = cls._extract_external_ids(external_ids)
        fallback_arxiv_id, fallback_doi = external_ids_from_canonical_paper_id(
            str(paper_id)
        )
        return arxiv_id or fallback_arxiv_id, doi or fallback_doi

    @staticmethod
    def _payload_get(payload: object, key: str, default: Any = None) -> Any:
        """Read a field from dict-like or object-like API payloads.

        :param object payload: Raw API payload object or mapping.
        :param str key: Field name to read.
        :param Any default: Value returned when the field is absent.
        :return Any: Extracted field value or ``default`` when unavailable.
        """
        if isinstance(payload, dict):
            return payload.get(key, default)
        return getattr(payload, key, default)

    @classmethod
    def _extract_authors(cls, raw_authors: object) -> list[Author]:
        """Extract up to three authors from raw API payload shapes.

        :param object raw_authors: Raw authors payload from Semantic Scholar.
        :return list[Author]: Up to three normalized author records.
        """
        if not isinstance(raw_authors, list):
            return []

        authors: list[Author] = []
        for raw_author in raw_authors[:3]:
            name = cls._payload_get(raw_author, "name")
            if not isinstance(name, str) or not name:
                continue
            authors.append(
                Author(
                    name=name,
                    author_id=cls._payload_get(raw_author, "authorId"),
                )
            )
        return authors

    @staticmethod
    def _extract_categories(*raw_candidates: object) -> list[str]:
        """Return the first usable category list from candidate payload fields.

        :param object raw_candidates: Candidate category payload values.
        :return list[str]: First normalized non-empty category list.
        """
        for raw_categories in raw_candidates:
            if isinstance(raw_categories, str):
                return [raw_categories]
            if isinstance(raw_categories, list):
                return [category for category in raw_categories if category]
        return []

    def _convert_payload_paper(
        self,
        payload: object,
        *,
        category_keys: tuple[str, ...],
        references: Optional[list[str]] = None,
    ) -> Optional[Paper]:
        """Convert a dict-like or object-like paper payload into a ``Paper`` model.

        :param object payload: Raw Semantic Scholar payload object or mapping.
        :param tuple[str, ...] category_keys: Category field names checked in order.
        :param Optional[list[str]] references: Optional normalized reference IDs.
        :return Optional[Paper]: Converted paper or ``None`` when no usable paper ID exists.
        """
        paper_id = self._payload_get(payload, "paperId")
        if not isinstance(paper_id, str) or not paper_id:
            return None

        arxiv_id, doi = self._resolve_external_ids(
            self._payload_get(payload, "externalIds"),
            paper_id,
        )
        return Paper(
            paper_id=paper_id,
            title=self._payload_get(payload, "title") or "Unknown",
            year=self._payload_get(payload, "year"),
            authors=self._extract_authors(self._payload_get(payload, "authors")),
            citation_count=self._payload_get(payload, "citationCount", 0) or 0,
            abstract=self._payload_get(payload, "abstract") or "",
            venue=self._extract_venue(
                self._payload_get(payload, "venue"),
                self._payload_get(payload, "publicationVenue"),
                self._payload_get(payload, "journal"),
            ),
            arxiv_id=arxiv_id,
            doi=doi,
            categories=self._extract_categories(
                *(self._payload_get(payload, key) for key in category_keys)
            ),
            references=references or [],
            is_seed=False,
        )

    def _convert_api_paper(self, api_paper: Any) -> Optional[Paper]:
        """
        Convert Semantic Scholar API response to Paper model.

        :param Any api_paper: Raw paper object from S2 API
        :return Optional[Paper]: Paper object or None if conversion fails
        """
        try:
            return self._convert_payload_paper(
                api_paper,
                category_keys=("fields", "fieldsOfStudy"),
            )
        except Exception as exc:
            logger.warning("Failed to convert API paper: %s", exc)
            return None

    def _convert_recommendation(self, rec: Dict[str, Any]) -> Optional[Paper]:
        """Convert recommendation/search record dict to a Paper model.

        :param Dict[str, Any] rec: Record returned by recommendation/search APIs.
        :return Optional[Paper]: Parsed Paper model or ``None`` on malformed payload.
        """
        try:
            return self._convert_payload_paper(
                rec,
                category_keys=("fieldsOfStudy", "fields"),
                references=self._extract_reference_ids(rec.get("references")),
            )
        except (TypeError, ValueError) as exc:
            logger.debug("Skipping malformed recommendation record: %s", exc)
            return None

    @staticmethod
    def _extract_reference_ids(raw_references: Any) -> List[str]:
        """Extract reference IDs from recommendation/search payload shapes.

        :param Any raw_references: Raw ``references`` payload from API response.
        :return List[str]: Parsed reference ID list (order-preserving, deduplicated).
        """
        return _normalize_reference_ids(raw_references, strict=False) or []

    @staticmethod
    def _unavailable_error(
        context: str,
        detail: str,
        *,
        rate_limited: bool,
        issue_hint: str = "This is a service availability issue",
    ) -> "SemanticScholarUnavailableError":
        """Build the availability error raised when retries are exhausted.

        :param str context: Human-readable request context (e.g. ``"searching for 'x'"``).
        :param str detail: Trailing detail appended after the attempt count.
        :param bool rate_limited: Whether the final failure was an HTTP 429.
        :param str issue_hint: Explanation placed before retry guidance.
        :return SemanticScholarUnavailableError: Flavored availability error.
        """
        flavor = "rate-limited (HTTP 429)" if rate_limited else "unreachable"
        return SemanticScholarUnavailableError(
            f"Semantic Scholar API {flavor} while {context} "
            f"(after {API_CONFIG.max_retries} attempts{detail}). "
            f"{issue_hint} - retry shortly, or set "
            f"S2_API_KEY for a dedicated rate limit (free keys: {S2_API_KEY_SIGNUP_URL})."
        )

    async def _request_sdk_json(
        self,
        url: str,
        parameters: str,
        headers: Optional[Dict[str, str]],
        payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Supply status-preserving responses to the SDK's synchronous wrapper.

        :param str url: SDK endpoint URL.
        :param str parameters: SDK-encoded query parameters.
        :param Optional[Dict[str, str]] headers: SDK authentication headers.
        :param Optional[Dict[str, Any]] payload: Batch POST body, or no body for GET.
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
        params: Dict[str, Any] | str,
        *,
        context: str,
        headers: Optional[Dict[str, str]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Issue one paced HTTP request without adding another retry budget.

        :param str url: Semantic Scholar endpoint URL.
        :param Dict[str, Any] | str params: Mapping or pre-encoded query parameters.
        :param str context: Description included in request errors.
        :param Optional[Dict[str, str]] headers: Optional SDK authentication headers.
        :param Optional[Dict[str, Any]] payload: POST body, or ``None`` for GET.
        :return Any: Decoded JSON, or ``None`` for HTTP 404.
        """
        kwargs: Dict[str, Any] = {"params": params, "timeout": self.timeout}
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
                retry_after=self._safe_retry_after(response),
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
        params: Dict[str, Any],
        *,
        raise_on_unavailable: bool = False,
        context: str = "requesting data",
    ) -> Optional[Dict[str, Any]]:
        """Request JSON payload from direct Semantic Scholar REST endpoints.

        :param str url: Endpoint URL.
        :param Dict[str, Any] params: Query parameters.
        :param bool raise_on_unavailable: When ``True``, exhausted retries raise
            :class:`SemanticScholarUnavailableError` instead of returning
            ``None``, so callers can distinguish "no data" from "API down".
            HTTP 404 still returns ``None`` (genuinely absent resource).
        :param str context: Request description used in availability errors.
        :return Optional[Dict[str, Any]]: Parsed JSON payload or ``None`` on failure.
        :raises SemanticScholarRequestError: If a non-408/429 HTTP 4xx response
            rejects the request.
        """

        def _attempt() -> Optional[Dict[str, Any]]:
            """Issue one paced request, raising on retryable failures.

            :return Optional[Dict[str, Any]]: Parsed payload or ``None`` on 404.
            """
            return self._request_json_once(url, params, context=context)

        def _log_before_sleep(retry_state: RetryCallState) -> None:
            """Log the upcoming retry with its computed wait.

            :param RetryCallState retry_state: Tenacity retry state.
            """
            _warn_on_long_wait(retry_state)
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            wait_seconds = (
                retry_state.next_action.sleep if retry_state.next_action else 0.0
            )
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

        retryer = Retrying(
            stop=stop_after_attempt(API_CONFIG.max_retries),
            wait=_S2BackoffWait(),
            retry=retry_if_exception_type(
                (_RetryableRequestError, requests.RequestException, ValueError)
            ),
            before_sleep=_log_before_sleep,
            sleep=lambda seconds: time.sleep(seconds),
            reraise=True,
        )

        try:
            return retryer(_attempt)
        except (_RetryableRequestError, requests.RequestException, ValueError) as exc:
            rate_limited = self._is_rate_limit_error(exc)
            if raise_on_unavailable:
                detail = "" if rate_limited else f": {exc}"
                raise self._unavailable_error(
                    context, detail, rate_limited=rate_limited
                ) from exc
            logger.error(
                "Failed to call %s after %s attempts: %s",
                url,
                API_CONFIG.max_retries,
                exc,
            )
            return None

    def get_paper(
        self,
        paper_id: str,
        fetch_references: bool = False,
        *,
        raise_on_unavailable: bool = False,
    ) -> Optional[Paper]:
        """
        Fetch a paper by ID, reusing persisted metadata before calling the API.

        :param str paper_id: Paper identifier (DOI, arXiv ID, or S2 ID)
        :param bool fetch_references: Whether to fetch reference list (slower)
        :param bool raise_on_unavailable: When ``True``, exhausted retries raise
            :class:`SemanticScholarUnavailableError` instead of returning
            ``None``, so callers can distinguish "not found" from "API down".
        :return Optional[Paper]: Paper object or None if not found
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
            cached_paper = _load_cached_paper(paper_id)
            if cached_paper is not None:
                return cached_paper
        # The SDK silently maps unrecognized HTTP statuses to an empty paper.
        # The graph endpoint requires literal slashes in DOI and legacy arXiv IDs.
        api_paper = self._request_json(
            f"{PAPER_BASE_URL}/{quote(paper_id, safe='/')}",
            {"fields": ",".join(_default_paper_fields())},
            raise_on_unavailable=raise_on_unavailable,
            context=f"fetching {paper_id}",
        )
        if api_paper is None:
            return None
        try:
            paper = self._convert_payload_paper(
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
        _persist_paper(paper, paper_id)
        return paper

    def get_papers(
        self, paper_ids: Sequence[str], *, raise_on_unavailable: bool = False
    ) -> Dict[str, Paper]:
        """Reuse cached metadata and fetch missing papers through the batch endpoint.

        :param Sequence[str] paper_ids: Paper identifiers (DOI, arXiv ID, or S2 IDs).
        :param bool raise_on_unavailable: Whether exhausted retries raise instead
            of returning cached results without further per-paper requests.
        :return Dict[str, Paper]: Mapping of normalized requested IDs to fetched papers.
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
            and (paper := _load_cached_paper(paper_id)) is not None
        }
        normalized_ids = [
            paper_id for paper_id in normalized_ids if paper_id not in cached
        ]
        if not normalized_ids:
            return cached

        if len(normalized_ids) > 500:
            raise ValueError("A batch request supports at most 500 paper IDs.")

        def _operation() -> Dict[str, Paper]:
            """Fetch positional batch results, preserving authoritative null entries.

            :return Dict[str, Paper]: Successful results keyed by requested ID.
            """
            api_papers = self._request_json_once(
                f"{PAPER_BASE_URL}/batch",
                {"fields": ",".join(_default_paper_fields())},
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
            matched: Dict[str, Paper] = {}
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
                _persist_paper(paper, requested_id)
            return matched

        def _final_failure(exc: Exception) -> None:
            """Stop the batch after exhaustion without restarting retries per ID.

            :param Exception exc: Last operational failure.
            :raises SemanticScholarUnavailableError: In strict mode.
            :return None: Signals an unavailable batch in tolerant mode.
            """
            if raise_on_unavailable:
                raise self._unavailable_error(
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
    ) -> List[Paper]:
        """
        Fetch papers that cite the given paper.

        :param str paper_id: Paper identifier
        :param int limit: Maximum number of citations to fetch
        :param bool raise_on_unavailable: Whether exhausted operational retries
            raise instead of returning an empty list.
        :return List[Paper]: Citation Papers (may be empty).
        :raises SemanticScholarRequestError: If Semantic Scholar rejects the request.
        """
        parsed_limit = _validate_integer_limit(limit, "limit", allow_zero=True)
        if parsed_limit == 0:
            return []

        return self._get_related_papers(
            paper_id=paper_id,
            limit=parsed_limit,
            fetch_method=self.client.get_paper_citations,
            relation_label="citations",
            raise_on_unavailable=raise_on_unavailable,
        )

    def get_paper_references(
        self,
        paper_id: str,
        limit: int = 20,
        *,
        raise_on_unavailable: bool = False,
    ) -> List[Paper]:
        """
        Fetch papers referenced by the given paper.

        :param str paper_id: Paper identifier
        :param int limit: Maximum number of references to fetch
        :param bool raise_on_unavailable: Whether exhausted operational retries
            raise instead of returning an empty list.
        :return List[Paper]: List of Paper objects (may be shorter than limit)
        :raises SemanticScholarRequestError: If Semantic Scholar rejects the request.
        """
        parsed_limit = _validate_integer_limit(limit, "limit", allow_zero=True)
        if parsed_limit == 0:
            return []

        return self._get_related_papers(
            paper_id=paper_id,
            limit=parsed_limit,
            fetch_method=self.client.get_paper_references,
            relation_label="references",
            raise_on_unavailable=raise_on_unavailable,
        )

    def _get_related_papers(
        self,
        paper_id: str,
        limit: int,
        fetch_method: Callable[..., Any],
        relation_label: str,
        raise_on_unavailable: bool,
    ) -> List[Paper]:
        """Fetch and convert citation-like relation payloads with shared retry logic.

        :param str paper_id: Raw paper identifier.
        :param int limit: Maximum number of relation records to fetch.
        :param Callable[..., Any] fetch_method: Semantic Scholar relation fetch method.
        :param str relation_label: Human-readable label used in logs.
        :param bool raise_on_unavailable: Whether exhausted operational retries raise.
        :return List[Paper]: Converted relation papers.
        """
        normalized_paper_id = normalize_paper_id(paper_id)

        def _operation() -> List[Paper]:
            """Fetch and convert citation/reference relation records.

            :return List[Paper]: Converted relation papers for this attempt.
            """
            attempt_papers: List[Paper] = []
            try:
                relation_records = fetch_method(normalized_paper_id, limit=limit)
            except TypeError as exc:
                if not _is_sdk_null_relation_page(exc):
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

                if len(attempt_papers) >= limit:
                    break

            return attempt_papers

        def _final_failure(exc: Exception) -> List[Paper]:
            """Apply the caller-selected failure contract after retry exhaustion.

            :param Exception exc: Final operational failure.
            :return List[Paper]: Empty list in tolerant mode.
            :raises SemanticScholarUnavailableError: In strict mode.
            """
            if raise_on_unavailable:
                raise self._unavailable_error(
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

    def get_reference_ids(
        self, paper_id: str, *, force_refresh: bool = False
    ) -> List[str]:
        """
        Fetch only the reference IDs for a paper (faster than full references).

        :param str paper_id: Paper identifier
        :param bool force_refresh: Whether to bypass cache reads and fetch fresh IDs.
        :return List[str]: List of referenced paper IDs
        :raises TypeError: If the SDK or response payload violates the relation contract.
        :raises SemanticScholarRequestError: If Semantic Scholar rejects the request.
        :raises SemanticScholarUnavailableError: If operational retries are exhausted.
        """
        normalized_paper_id = normalize_paper_id(paper_id)
        cache_path = _reference_cache_path(normalized_paper_id)
        if force_refresh:
            logger.debug(
                "Bypassing reference cache for %s due to force_refresh.",
                normalized_paper_id,
            )
        if not force_refresh and cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("reference cache payload must be a JSON object")
                if data.get("version") == REFERENCE_CACHE_VERSION:
                    if data.get("paper_id") != normalized_paper_id:
                        raise ValueError(
                            "reference cache paper ID does not match its key"
                        )
                    if "references" not in data:
                        raise ValueError(
                            "reference cache payload is missing references"
                        )
                    cached_references = data["references"]
                    refs = _coerce_cached_reference_ids(cached_references)
                    if refs is None:
                        logger.warning(
                            "Invalid reference cache payload for %s; rebuilding entry.",
                            normalized_paper_id,
                        )
                        with contextlib.suppress(OSError):
                            cache_path.unlink(missing_ok=True)
                    else:
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

        def _persist_empty() -> List[str]:
            """Persist and return an empty cached reference-ID list.

            :return List[str]: Empty reference-ID list.
            """
            self._persist_reference_cache_entry(
                cache_path,
                normalized_paper_id,
                [],
            )
            return []

        def _operation() -> List[Any]:
            """Fetch one materialized reference-relation response.

            :return List[Any]: Raw relation records for later validation.
            """
            try:
                raw_references = self.client.get_paper_references(
                    normalized_paper_id,
                    fields=["paperId"],
                )
            except TypeError as exc:
                if not _is_sdk_null_relation_page(exc):
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

        def _raise_contract_failure(exc: Exception) -> List[Any]:
            """Surface local SDK/payload contract errors without retrying.

            :param Exception exc: Local contract failure.
            :raises Exception: Always re-raises ``exc``.
            :return List[Any]: This function does not return successfully.
            """
            raise exc

        def _raise_failure(exc: Exception) -> List[Any]:
            """Raise an availability error after operational retry exhaustion.

            :param Exception exc: Final exception raised by the API client.
            :raises SemanticScholarUnavailableError: Always with failure context.
            :return List[Any]: This function does not return successfully.
            """
            raise self._unavailable_error(
                f"fetching reference IDs for {normalized_paper_id}",
                f": {exc}",
                rate_limited=self._is_rate_limit_error(exc),
            ) from exc

        paper_not_found = False

        def _handle_not_found(_exc: Exception) -> List[Any]:
            """Return missing-paper references without treating them as successful empties.

            :param Exception _exc: SDK exception indicating the relation endpoint's
                first page was not found.
            :return List[Any]: Empty relation list for the absent paper.
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

        normalized_ref_ids = _normalize_reference_ids(references, strict=True)
        if normalized_ref_ids is None:
            if all(_is_unresolved_reference(record) for record in references):
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

    def get_recommended_papers(
        self,
        paper_id: str,
        limit: int = 50,
        fields: Optional[List[str]] = None,
        *,
        raise_on_unavailable: bool = False,
    ) -> List[Paper]:
        """
        Get semantically related papers using S2 recommendations.

        :param str paper_id: S2 paper ID
        :param int limit: Maximum recommendations
        :param Optional[List[str]] fields: API fields to return.
        :param bool raise_on_unavailable: Whether exhausted operational retries
            for the primary request raise instead of returning an empty list. An
            unavailable optional ``all-cs`` widening request preserves a
            successful empty primary result.
        :return List[Paper]: Ranked recommendation papers.
        """
        if fields is None:
            fields = _default_paper_fields()
        # Semantic Scholar's recommendations endpoint does not currently support
        # requesting ``references`` in field lists (returns HTTP 400 with
        # unsupported nested-reference field tokens). Keep the request field set
        # endpoint-compatible and let callers hydrate references via dedicated
        # reference-ID methods when needed.
        if "references" in fields:
            fields = [field for field in fields if field != "references"]
        parsed_limit = _validate_integer_limit(limit, "limit")

        normalized_paper_id = normalize_paper_id(paper_id)
        encoded_paper_id = quote(normalized_paper_id, safe="")
        base_params = {"fields": ",".join(fields), "limit": parsed_limit}
        payload = self._request_json(
            f"{RECOMMENDATION_BASE_URL}/{encoded_paper_id}",
            base_params,
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
                    raise_on_unavailable=raise_on_unavailable,
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

        papers = []
        for rec in raw_recommendations:
            paper = self._convert_recommendation(rec)
            if paper:
                papers.append(paper)
        return papers

    def search_papers(
        self,
        query: str,
        limit: int = 10,
        fields: Optional[List[str]] = None,
        *,
        raise_on_unavailable: bool = False,
    ) -> List[Paper]:
        """Search papers by title or keyword.

        :param str query: Search query string.
        :param int limit: Maximum number of results.
        :param Optional[List[str]] fields: Optional fields list for API payload.
        :param bool raise_on_unavailable: When ``True``, exhausted retries raise
            :class:`SemanticScholarUnavailableError` instead of returning an
            empty list, so callers can distinguish "no matches" from "API down".
        :return List[Paper]: Search results.
        """
        parsed_limit = _validate_integer_limit(limit, "limit")
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        if fields is None:
            fields = _default_paper_fields()

        payload = self._request_json(
            SEARCH_BASE_URL,
            {
                "query": normalized_query,
                "fields": ",".join(fields),
                "limit": parsed_limit,
            },
            raise_on_unavailable=raise_on_unavailable,
            context=f"searching for {normalized_query!r}",
        )
        if not payload:
            return []

        papers = []
        for rec in payload.get("data", []):
            paper = self._convert_recommendation(rec)
            if paper:
                papers.append(paper)
        return papers


_client_instance: Optional[SemanticScholarClient] = None
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
    client_to_close: Optional[SemanticScholarClient] = None
    with _client_lock:
        client_to_close = _client_instance
        _client_instance = None
    if client_to_close is not None:
        client_to_close.close()
