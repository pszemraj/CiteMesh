"""Shared scalar coercion helpers for untrusted metadata values.

Upstream payloads (API responses, user-authored graph attributes, cached
records) routinely carry ``None``, empty strings, or text where a number is
expected. These helpers centralize the "parse or fall back" pattern so the
scoring, layout, and export surfaces agree on what a malformed value means.
"""

from __future__ import annotations


def coerce_float(raw: object, default: float) -> float:
    """Parse a float, returning ``default`` for null or non-numeric input.

    Booleans are *not* special-cased: they parse as ``1.0``/``0.0`` the way
    every call site already treated them. Non-finite text (``"nan"``,
    ``"inf"``) parses to the matching float, so callers that care must check
    ``math.isfinite`` themselves.

    :param object raw: Raw value of unknown type.
    :param float default: Value returned when ``raw`` is not parseable.
    :return float: Parsed float, otherwise ``default``.
    """
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def coerce_citation_count(raw: object) -> int:
    """Normalize a citation count to a non-negative integer.

    ``True``/``False`` are metadata noise rather than counts of one and zero,
    so booleans and ``None`` both collapse to ``0``. Negative values clamp to
    ``0``; floats and numeric strings truncate toward zero.

    :param object raw: Raw citation count value.
    :return int: Non-negative citation count, ``0`` when unusable.
    """
    if isinstance(raw, bool) or raw is None:
        return 0
    try:
        return max(int(raw), 0)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0
