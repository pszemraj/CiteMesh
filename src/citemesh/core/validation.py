"""Shared scalar validation shared by the CLI parser and the user config.

Owns the parsing rules -- not the phrasing -- for the two scalar shapes both
layers accept: a lower-bounded integer and a float on the unit interval. Each
failure carries a machine-readable :attr:`ValueValidationError.reason` so the
CLI can raise ``argparse`` messages and the config loader can raise its own
wording from one implementation.

Stdlib only: this module must stay importable from every layer.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "ValueValidationError",
    "parse_bounded_int",
    "parse_unit_interval_float",
]


class ValueValidationError(ValueError):
    """Scalar validation failure tagged with the rule that rejected the value.

    :param str reason: Stable token identifying the failed rule.
    :param str message: Default human-readable description of the failure.
    """

    def __init__(self, reason: str, message: str) -> None:
        """Store the failure reason alongside the default message.

        :param str reason: Stable token identifying the failed rule.
        :param str message: Default human-readable description of the failure.
        :return None: Initializes the exception.
        """
        super().__init__(message)
        self.reason = reason


def parse_bounded_int(value: Any, *, minimum: int) -> int:
    """Parse an integer constrained by an inclusive lower bound.

    Booleans are rejected outright because ``bool`` is an ``int`` subclass and
    ``--max-papers true`` is never intended. Strings are trimmed before parsing;
    any other non-integer type is rejected rather than truncated.

    :param Any value: Raw string, integer, or unsupported object.
    :param int minimum: Inclusive lower bound for accepted values.
    :return int: Parsed integer greater than or equal to ``minimum``.
    :raises ValueValidationError: With reason ``not_integer`` if the value is not
        an integer, or ``below_minimum`` if it is below ``minimum``.
    """
    if isinstance(value, bool):
        raise ValueValidationError("not_integer", "expected an integer, got a boolean")
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError as exc:
            raise ValueValidationError("not_integer", "expected an integer") from exc
    elif isinstance(value, int):
        parsed = value
    else:
        raise ValueValidationError("not_integer", "expected an integer")
    if parsed < minimum:
        raise ValueValidationError("below_minimum", f"expected an integer >= {minimum}")
    return parsed


def parse_unit_interval_float(value: Any) -> float:
    """Parse a finite float constrained to the inclusive range ``[0.0, 1.0]``.

    :param Any value: Raw string, number, or unsupported object.
    :return float: Parsed float between zero and one, inclusive.
    :raises ValueValidationError: With reason ``not_number`` if the value is not
        numeric, ``not_finite`` for NaN/infinity, or ``out_of_range`` if the
        value falls outside ``[0.0, 1.0]``.
    """
    if isinstance(value, bool):
        raise ValueValidationError("not_number", "expected a float, got a boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueValidationError("not_number", "expected a float") from exc
    if not math.isfinite(parsed):
        raise ValueValidationError("not_finite", "expected a finite float")
    if parsed < 0.0 or parsed > 1.0:
        raise ValueValidationError(
            "out_of_range", "expected a float between 0.0 and 1.0"
        )
    return parsed
