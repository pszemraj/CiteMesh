"""Semantic Scholar payload parsing, validation, and conversion to core models.

Owns the default field set, the reference-ID normalization rules shared by the
cache and the live API, the argument validators, and the payload-to-``Paper``
conversions used by every endpoint.
"""

from __future__ import annotations

import logging
import numbers
from typing import Any

from citemesh.core import Author, Paper
from citemesh.core.paper_fields import (
    coerce_author_name,
    coerce_categories,
    coerce_venue,
)
from citemesh.core.paper_ids import external_ids_from_canonical_paper_id

logger = logging.getLogger(__name__)


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


def _default_paper_fields() -> list[str]:
    """Return a mutable default field list for paper-like API endpoints.

    :return list[str]: Default paper fields for search/recommendation/get operations.
    """
    return list(DEFAULT_PAPER_FIELDS)


def _reference_id_candidate(raw_value: Any) -> str | None:
    """Extract a paper ID from accepted reference payload shapes.

    Supports current cache entries and legacy mixed-format cache entries.

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


def _coerce_cached_reference_ids(payload: Any) -> list[str] | None:
    """Normalize a reference payload, rejecting one that is malformed.

    Accepts current list-of-string payloads and legacy mixed relation-shaped
    entries, from either a cache entry or a live relation response. A non-list
    payload, or a non-empty one holding no usable paper ID, is invalid: the
    caller rebuilds the cache entry or reports a contract failure.

    :param Any payload: Cached ``references`` field or live relation records.
    :return list[str] | None: Normalized IDs, or ``None`` when the payload is invalid.
    """
    if not isinstance(payload, list):
        return None

    normalized: list[str] = []
    seen: set[str] = set()

    for raw_value in payload:
        candidate = _reference_id_candidate(raw_value)
        paper_id = candidate.strip() if candidate is not None else ""
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        normalized.append(paper_id)

    if payload and not normalized:
        return None
    return normalized


def _extract_reference_ids(raw_references: Any) -> list[str]:
    """Extract reference IDs from recommendation/search payload shapes.

    Tolerant counterpart of :func:`_coerce_cached_reference_ids`: a malformed
    payload yields no references instead of signalling invalid data.

    :param Any raw_references: Raw ``references`` payload from API response.
    :return list[str]: Parsed reference ID list (order-preserving, deduplicated).
    """
    return _coerce_cached_reference_ids(raw_references) or []


def _validate_integer_limit(
    limit: int,
    field_name: str,
    allow_zero: bool = False,
    maximum: int | None = None,
) -> int:
    """Validate API limit argument values and return normalized int.

    :param int limit: Raw limit value supplied by caller.
    :param str field_name: Parameter name used in error messages.
    :param bool allow_zero: Whether zero is accepted as a disable switch.
    :param int | None maximum: Optional inclusive upper bound.
    :return int: Parsed integer limit value.
    :raises ValueError: If value is non-integer or outside the accepted range.
    """
    if isinstance(limit, bool) or not isinstance(limit, numbers.Integral):
        raise ValueError(f"{field_name} must be an integer, got {limit!r}")

    parsed_limit = int(limit)
    minimum = 0 if allow_zero else 1
    if parsed_limit < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}, got {parsed_limit}")
    if maximum is not None and parsed_limit > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}, got {parsed_limit}")
    return parsed_limit


def _extract_venue(*candidates: object) -> str:
    """Extract the first non-empty venue label from candidate payloads.

    :param object candidates: Venue candidate payloads.
    :return str: Normalized venue string (empty when unavailable).
    """
    for candidate in candidates:
        venue = coerce_venue(candidate)
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
        name = coerce_author_name(raw_author)
        if not name:
            continue
        authors.append(
            Author(name=name, author_id=_payload_get(raw_author, "authorId"))
        )
    return authors


def _extract_categories(*raw_candidates: object) -> list[str]:
    """Return the first usable category list from candidate payload fields.

    The first candidate of a usable type wins, even when it normalizes to an
    empty list. Semantic Scholar's ``fieldsOfStudy`` labels contain spaces, so
    the shared coercer must keep each label whole.

    :param object raw_candidates: Candidate category payload values.
    :return list[str]: First normalized category list.
    """
    for raw_categories in raw_candidates:
        if isinstance(raw_categories, (str, list)):
            return coerce_categories(raw_categories)
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
            references=_extract_reference_ids(_payload_get(rec, "references")),
        )
    except (TypeError, ValueError) as exc:
        logger.debug("Skipping malformed recommendation record: %s", exc)
        return None
