"""Semantic Scholar payload parsing, validation, and conversion to core models.

Owns the default field set, the reference-ID normalization rules shared by the
cache and the live API, the argument validators, and the payload-to-``Paper``
conversions used by every endpoint.
"""

from __future__ import annotations

import logging
import numbers
from collections.abc import Mapping
from typing import TypedDict

from citemesh.core import Author, Paper
from citemesh.core.paper_fields import (
    coerce_author_name,
    coerce_categories,
    coerce_venue,
)
from citemesh.core.paper_ids import external_ids_from_canonical_paper_id
from citemesh.data.cache import JSONValue

logger = logging.getLogger(__name__)


class PaperBatchRequest(TypedDict):
    """Body sent to the Semantic Scholar paper batch endpoint."""

    ids: list[str]


class _RelationPageData(TypedDict):
    """Envelope returned by citation and reference pagination."""

    data: JSONValue


class RelationPage(_RelationPageData, total=False):
    """Optional pagination cursor returned with a relation page."""

    next: JSONValue


class RecommendationResponse(TypedDict):
    """Envelope returned by the recommendation endpoint."""

    recommendedPapers: list[JSONValue]


class _SearchResponseData(TypedDict):
    """Envelope returned by the paper search endpoint."""

    data: list[JSONValue]


class SearchResponse(_SearchResponseData, total=False):
    """Optional pagination cursor returned with search results."""

    next: JSONValue


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


def _reference_id_candidate(raw_value: object) -> str | None:
    """Extract a paper ID from accepted reference payload shapes.

    Supports current cache entries and legacy mixed-format cache entries.

    :param object raw_value: Raw reference-like entry.
    :return str | None: Candidate paper ID string, or ``None`` when absent.
    """
    if isinstance(raw_value, str):
        return raw_value

    if isinstance(raw_value, dict):
        record: dict[object, object] = raw_value
        for key in ("paperId", "paper_id"):
            candidate = record.get(key)
            if isinstance(candidate, str):
                return candidate

        nested_paper = record.get("paper")
        if isinstance(nested_paper, dict):
            nested_record: dict[object, object] = nested_paper
            for key in ("paperId", "paper_id"):
                candidate = nested_record.get(key)
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


def _coerce_cached_reference_ids(payload: object) -> list[str] | None:
    """Normalize a reference payload, rejecting one that is malformed.

    Accepts current list-of-string payloads and legacy mixed relation-shaped
    entries, from either a cache entry or a live relation response. A non-list
    payload, or a non-empty one holding no usable paper ID, is invalid: the
    caller rebuilds the cache entry or reports a contract failure.

    :param object payload: Cached ``references`` field or live relation records.
    :return list[str] | None: Normalized IDs, or ``None`` when the payload is invalid.
    """
    if not isinstance(payload, list):
        return None

    normalized: list[str] = []
    seen: set[str] = set()

    for untyped_value in payload:
        raw_value: object = untyped_value
        candidate = _reference_id_candidate(raw_value)
        paper_id = candidate.strip() if candidate is not None else ""
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        normalized.append(paper_id)

    if payload and not normalized:
        return None
    return normalized


def _extract_reference_ids(raw_references: object) -> list[str]:
    """Extract reference IDs from recommendation/search payload shapes.

    Tolerant counterpart of :func:`_coerce_cached_reference_ids`: a malformed
    payload yields no references instead of signalling invalid data.

    :param object raw_references: Raw ``references`` payload from API response.
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


def _extract_external_ids_from_mapping(
    mapping: Mapping[object, object],
) -> tuple[str, str]:
    """Extract arXiv and DOI IDs from external-id style mappings.

    :param Mapping[object, object] mapping: External IDs map.
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
        mapping: Mapping[object, object] = external_ids
        return _extract_external_ids_from_mapping(mapping)
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


def _payload_get(payload: object, key: str, default: object = None) -> object:
    """Read a field from dict-like or object-like API payloads.

    :param object payload: Raw API payload object or mapping.
    :param str key: Field name to read.
    :param object default: Value returned when the field is absent.
    :return object: Raw field value or ``default`` when unavailable.
    """
    if isinstance(payload, dict):
        mapping: dict[object, object] = payload
        return mapping.get(key, default)
    value: object = getattr(payload, key, default)
    return value


def _extract_authors(raw_authors: object) -> list[Author]:
    """Extract all authors from raw API payload shapes.

    :param object raw_authors: Raw authors payload from Semantic Scholar.
    :return list[Author]: All normalized author records in source order.
    :raises TypeError: If an author's ID is neither a string nor ``None``.
    """
    if not isinstance(raw_authors, list):
        return []

    authors: list[Author] = []
    for untyped_author in raw_authors:
        raw_author: object = untyped_author
        name = coerce_author_name(raw_author)
        if not name:
            continue
        author_id = _payload_get(raw_author, "authorId")
        if author_id is not None and not isinstance(author_id, str):
            raise TypeError("authorId must be a string or None")
        authors.append(Author(name=name, author_id=author_id))
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
    :raises TypeError: If a non-empty title or abstract, year, or citation count
        has an unsupported type, or an author's ID is not a string.
    """
    paper_id = _payload_get(payload, "paperId")
    if not isinstance(paper_id, str) or not paper_id:
        return None

    title = _payload_get(payload, "title") or "Unknown"
    if not isinstance(title, str):
        raise TypeError("title must be a string")

    raw_year = _payload_get(payload, "year")
    if raw_year is not None and (
        isinstance(raw_year, bool) or not isinstance(raw_year, int)
    ):
        raise TypeError("year must be an integer or None")

    citation_count = _payload_get(payload, "citationCount", 0) or 0
    if isinstance(citation_count, bool) or not isinstance(citation_count, int):
        raise TypeError("citationCount must be an integer")

    abstract = _payload_get(payload, "abstract") or ""
    if not isinstance(abstract, str):
        raise TypeError("abstract must be a string")

    arxiv_id, doi = _resolve_external_ids(
        _payload_get(payload, "externalIds"),
        paper_id,
    )
    return Paper(
        paper_id=paper_id,
        title=title,
        year=raw_year,
        authors=_extract_authors(_payload_get(payload, "authors")),
        citation_count=citation_count,
        abstract=abstract,
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


def _convert_api_paper(api_paper: object) -> Paper | None:
    """
    Convert Semantic Scholar API response to Paper model.

    :param object api_paper: Raw paper object from S2 API
    :return Paper | None: Paper object or None if conversion fails
    """
    try:
        return _convert_payload_paper(
            api_paper,
            category_keys=("fields", "fieldsOfStudy"),
        )
    except Exception as exc:
        logger.debug("Failed to convert API paper: %s", exc)
        return None


def _convert_recommendation(rec: object) -> Paper | None:
    """Convert recommendation/search record dict to a Paper model.

    :param object rec: Record returned by recommendation/search APIs.
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
