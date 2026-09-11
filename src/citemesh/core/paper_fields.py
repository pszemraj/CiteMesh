"""Tolerant coercers for paper metadata fields shared across ingestion sources.

Semantic Scholar payloads and HuggingFace arXiv records describe the same three
fields in different shapes: a venue arrives as a string, a ``{"name": ...}``
mapping, or an object exposing ``.name``; authors arrive as records, as plain
names, or as one delimited string; categories arrive as a list of labels or as
one packed string. These functions accept every shape and return a normalized
value, so each source layer only decides *which* raw field to hand over.

Where the two sources genuinely disagree the difference is a documented keyword
rather than a hidden default: arXiv packs several category codes into one
whitespace-separated string, while Semantic Scholar's ``fieldsOfStudy`` labels
contain spaces and must survive whole.

This module is deliberately dependency-free: ``citemesh.core`` must not import
from ``citemesh.data``, ``citemesh.services``, or ``citemesh.strategies``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "coerce_author_name",
    "coerce_authors",
    "coerce_categories",
    "coerce_venue",
]


def _display_name(raw: Any) -> str:
    """Read a display string from a bare string, a ``name`` key, or a ``name`` attribute.

    :param Any raw: Raw field value in any of the supported shapes.
    :return str: Stripped display string, empty when no usable name is present.
    """
    if isinstance(raw, str):
        return raw.strip()
    name = raw.get("name") if isinstance(raw, Mapping) else getattr(raw, "name", None)
    return name.strip() if isinstance(name, str) else ""


def coerce_venue(raw: Any) -> str:
    """Normalize a venue-like value into a display string.

    Accepts a plain string, a ``{"name": ...}`` mapping (Semantic Scholar's
    ``publicationVenue`` and ``journal``), an object exposing ``.name``, or
    ``None``.

    :param Any raw: Raw venue payload value.
    :return str: Stripped venue name, empty when unavailable.
    """
    return _display_name(raw)


def coerce_author_name(raw: Any) -> str:
    """Normalize one author entry into a display name.

    Accepts a plain name, a ``{"name": ...}`` mapping, an object exposing
    ``.name``, or ``None``. Callers that also need the source record's other
    fields (an author ID, say) read those from the entry themselves.

    :param Any raw: Raw author entry.
    :return str: Stripped author name, empty when unavailable.
    """
    return _display_name(raw)


def coerce_authors(raw: Any, *, split_string: bool = True) -> list[str]:
    """Normalize an authors field into a list of display names.

    Accepts a list of names, a list of ``{"name": ...}`` mappings or objects, a
    single delimited string, or ``None``. Blank and unusable entries are
    dropped; source order and duplicates are preserved, because two authors may
    legitimately share a display name.

    :param Any raw: Raw authors payload.
    :param bool split_string: Whether a single string is split on commas into
        several names, as HuggingFace arXiv records pack them. Semantic Scholar
        always sends a list, so its callers can pass ``False`` to treat a bare
        string as unusable.
    :return list[str]: Author display names in source order.
    """
    if isinstance(raw, str):
        if not split_string:
            return []
        return [name.strip() for name in raw.split(",") if name.strip()]
    if not isinstance(raw, list):
        return []
    return [name for name in map(coerce_author_name, raw) if name]


def coerce_categories(raw: Any, *, split_whitespace: bool = False) -> list[str]:
    """Normalize a categories field into a deduplicated list of labels.

    Accepts a list of labels, a single string, or ``None``. Non-string entries,
    blanks, and repeats are dropped; the first occurrence keeps its position.

    :param Any raw: Raw categories payload.
    :param bool split_whitespace: Whether each string is split into several
        labels on commas and whitespace. arXiv packs category codes that way
        (``"cs.CL cs.LG"``), while Semantic Scholar's ``fieldsOfStudy`` labels
        contain spaces (``"Computer Science"``) and must be kept whole.
    :return list[str]: Labels in source order, stripped and deduplicated.
    """
    if isinstance(raw, str):
        values: list[Any] = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        return []

    categories: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        parts = value.replace(",", " ").split() if split_whitespace else [value.strip()]
        for part in parts:
            if part and part not in seen:
                seen.add(part)
                categories.append(part)
    return categories
