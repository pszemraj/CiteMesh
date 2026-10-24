"""Parse raw metadata numbers with shared fallback behavior."""

from __future__ import annotations


def parse_float(raw: object) -> float:
    """Convert a value with Python's built-in ``float`` semantics.

    :param object raw: Value accepted or rejected by ``float``.
    :return float: Converted value.
    :raises TypeError: If ``raw`` is not float-convertible.
    :raises ValueError: If ``raw`` cannot be parsed as a float.
    :raises OverflowError: If numeric conversion exceeds the float range.
    """
    return float(raw)  # type: ignore[arg-type]


def coerce_float(raw: object, default: float) -> float:
    """Parse a float, returning ``default`` for null or non-numeric input.

    Booleans and non-finite values follow Python's ``float`` semantics.

    :param object raw: Raw value of unknown type.
    :param float default: Value returned when ``raw`` is not parseable.
    :return float: Parsed float, otherwise ``default``.
    """
    try:
        return parse_float(raw)
    except (TypeError, ValueError):
        return default


def coerce_citation_count(raw: object) -> int:
    """Normalize a citation count to a non-negative integer.

    Booleans, ``None``, and invalid integer text return ``0``. Negative values
    clamp to ``0``; finite numeric values truncate toward zero.

    :param object raw: Raw citation count value.
    :return int: Non-negative citation count, ``0`` when unusable.
    :raises OverflowError: If ``raw`` is an infinite numeric value.
    """
    if isinstance(raw, bool) or raw is None:
        return 0
    try:
        return max(int(raw), 0)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0
