"""Semantic Scholar payload parsing, validation, and conversion to core models.

Owns the default field set, the reference-ID normalization rules shared by the
cache and the live API, the argument validators, and the payload-to-``Paper``
conversions used by every endpoint.
"""

from __future__ import annotations

import logging
import numbers
from typing import Any

from citemesh.core import API_CONFIG, Author, Paper
from citemesh.paper_ids import external_ids_from_canonical_paper_id

from .errors import SemanticScholarUnavailableError

logger = logging.getLogger(__name__)


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


def _is_sdk_null_relation_page(error: TypeError) -> bool:
    """Identify the SDK failure used for a valid empty relation page.

    :param TypeError error: Exception raised while the SDK decodes a relation page.
    :return bool: Whether the exception represents an S2 ``data: null`` response.
    """
    return str(error) == _SDK_NULL_RELATION_PAGE_ERROR


def _default_paper_fields() -> list[str]:
    """Return a mutable default field list for paper-like API endpoints.

    :return list[str]: Default paper fields for search/recommendation/get operations.
    """
    return list(DEFAULT_PAPER_FIELDS)


def _reference_id_candidate(raw_value: Any) -> str | None:
    """Extract a paper ID from accepted reference payload shapes.

    Supports current cache entries, legacy mixed-format cache entries, and
    Semantic Scholar relation objects returned by the SDK.

    :param Any raw_value: Raw reference-like entry.
    :return str | None: Candidate paper ID string, or ``None`` when absent.
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


def _normalize_reference_ids(payload: Any, *, strict: bool) -> list[str] | None:
    """Normalize reference payloads under cache or live-response rules.

    Both callers accept current list-of-string payloads and legacy mixed
    relation-shaped entries. Strict cache parsing marks malformed non-empty
    lists invalid so they can be rebuilt; live API parsing tolerates them.

    :param Any payload: Raw reference payload.
    :param bool strict: Whether malformed payloads return ``None``.
    :return list[str] | None: Normalized IDs, or ``None`` for invalid strict data.
    """
    if not isinstance(payload, list):
        return None if strict else []

    normalized: list[str] = []
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


def _coerce_cached_reference_ids(payload: Any) -> list[str] | None:
    """Validate and normalize cached reference ID payloads.

    :param Any payload: Cached ``references`` field from JSON payload.
    :return list[str] | None: Normalized ID list, or ``None`` when invalid.
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


def _extract_venue(*candidates: object) -> str:
    """Extract the first non-empty venue label from candidate payloads.

    :param object candidates: Venue candidate payloads.
    :return str: Normalized venue string (empty when unavailable).
    """
    for candidate in candidates:
        venue = _extract_venue_name(candidate)
        if venue:
            return venue
    return ""


def _normalize_external_id(raw_value: object) -> str:
    """Normalize optional external-id strings.

    :param object raw_value: Raw external ID payload.
    :return str: Normalized external ID string (empty when unavailable).
    """
    if not isinstance(raw_value, str):
        return ""
    return raw_value.strip()


def _extract_external_ids_from_mapping(mapping: dict[str, Any]) -> tuple[str, str]:
    """Extract arXiv and DOI IDs from external-id style mappings.

    :param dict[str, Any] mapping: External IDs map.
    :return tuple[str, str]: ``(arxiv_id, doi)`` normalized identifiers.
    """
    normalized = {str(key).lower(): value for key, value in mapping.items()}
    arxiv_id = _normalize_external_id(normalized.get("arxiv"))
    doi = _normalize_external_id(normalized.get("doi"))
    return arxiv_id, doi


def _extract_external_ids(external_ids: object) -> tuple[str, str]:
    """Extract arXiv and DOI values from raw external-id payloads.

    :param object external_ids: Raw external ID payload.
    :return tuple[str, str]: ``(arxiv_id, doi)`` normalized identifiers.
    """
    if isinstance(external_ids, dict):
        return _extract_external_ids_from_mapping(external_ids)
    return "", ""


def _resolve_external_ids(external_ids: object, paper_id: object) -> tuple[str, str]:
    """Resolve external IDs from payload data with canonical-ID fallback.

    :param object external_ids: Raw external ID payload.
    :param object paper_id: Canonical or near-canonical paper ID fallback.
    :return tuple[str, str]: ``(arxiv_id, doi)`` pair.
    """
    arxiv_id, doi = _extract_external_ids(external_ids)
    fallback_arxiv_id, fallback_doi = external_ids_from_canonical_paper_id(
        str(paper_id)
    )
    return arxiv_id or fallback_arxiv_id, doi or fallback_doi


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


def _extract_authors(raw_authors: object) -> list[Author]:
    """Extract all authors from raw API payload shapes.

    :param object raw_authors: Raw authors payload from Semantic Scholar.
    :return list[Author]: All normalized author records in source order.
    """
    if not isinstance(raw_authors, list):
        return []

    authors: list[Author] = []
    for raw_author in raw_authors:
        name = _payload_get(raw_author, "name")
        if not isinstance(name, str) or not name.strip():
            continue
        authors.append(
            Author(
                name=name.strip(),
                author_id=_payload_get(raw_author, "authorId"),
            )
        )
    return authors


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
    payload: object,
    *,
    category_keys: tuple[str, ...],
    references: list[str] | None = None,
) -> Paper | None:
    """Convert a dict-like or object-like paper payload into a ``Paper`` model.

    :param object payload: Raw Semantic Scholar payload object or mapping.
    :param tuple[str, ...] category_keys: Category field names checked in order.
    :param list[str] | None references: Optional normalized reference IDs.
    :return Paper | None: Converted paper or ``None`` when no usable paper ID exists.
    """
    paper_id = _payload_get(payload, "paperId")
    if not isinstance(paper_id, str) or not paper_id:
        return None

    arxiv_id, doi = _resolve_external_ids(
        _payload_get(payload, "externalIds"),
        paper_id,
    )
    return Paper(
        paper_id=paper_id,
        title=_payload_get(payload, "title") or "Unknown",
        year=_payload_get(payload, "year"),
        authors=_extract_authors(_payload_get(payload, "authors")),
        citation_count=_payload_get(payload, "citationCount", 0) or 0,
        abstract=_payload_get(payload, "abstract") or "",
        venue=_extract_venue(
            _payload_get(payload, "venue"),
            _payload_get(payload, "publicationVenue"),
            _payload_get(payload, "journal"),
        ),
        arxiv_id=arxiv_id,
        doi=doi,
        categories=_extract_categories(
            *(_payload_get(payload, key) for key in category_keys)
        ),
        references=references or [],
        is_seed=False,
    )


def _convert_api_paper(api_paper: Any) -> Paper | None:
    """
    Convert Semantic Scholar API response to Paper model.

    :param Any api_paper: Raw paper object from S2 API
    :return Paper | None: Paper object or None if conversion fails
    """
    try:
        return _convert_payload_paper(
            api_paper,
            category_keys=("fields", "fieldsOfStudy"),
        )
    except Exception as exc:
        logger.warning("Failed to convert API paper: %s", exc)
        return None


def _convert_recommendation(rec: dict[str, Any]) -> Paper | None:
    """Convert recommendation/search record dict to a Paper model.

    :param dict[str, Any] rec: Record returned by recommendation/search APIs.
    :return Paper | None: Parsed Paper model or ``None`` on malformed payload.
    """
    try:
        return _convert_payload_paper(
            rec,
            category_keys=("fieldsOfStudy", "fields"),
            references=_extract_reference_ids(rec.get("references")),
        )
    except (TypeError, ValueError) as exc:
        logger.debug("Skipping malformed recommendation record: %s", exc)
        return None


def _extract_reference_ids(raw_references: Any) -> list[str]:
    """Extract reference IDs from recommendation/search payload shapes.

    :param Any raw_references: Raw ``references`` payload from API response.
    :return list[str]: Parsed reference ID list (order-preserving, deduplicated).
    """
    return _normalize_reference_ids(raw_references, strict=False) or []


def _unavailable_error(
    context: str,
    detail: str,
    *,
    rate_limited: bool,
    issue_hint: str = "This is a service availability issue",
) -> SemanticScholarUnavailableError:
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
