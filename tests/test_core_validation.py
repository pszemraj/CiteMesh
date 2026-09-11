"""Contracts for the shared scalar validators in :mod:`citemesh.core.validation`.

Covers the boundary, NaN/infinity, non-numeric, and boolean cases both the CLI
parser and the user-config casters depend on, plus the wording each wrapper
layer derives from :attr:`ValueValidationError.reason`.
"""

from __future__ import annotations

import argparse
import math

import pytest

from citemesh.cli.parser import (
    _bounded_int,
    _non_negative_int,
    _positive_int,
    _threshold_float,
)
from citemesh.core.validation import (
    ValueValidationError,
    parse_bounded_int,
    parse_unit_interval_float,
)
from citemesh.data.user_config import ConfigValueError, _cast_similarity, _int_caster


@pytest.mark.parametrize(
    ("value", "minimum", "expected"),
    [
        ("0", 0, 0),
        ("1", 1, 1),
        ("  7  ", 0, 7),
        ("-3", -5, -3),
        (12, 0, 12),
        ("+4", 0, 4),
    ],
)
def test_parse_bounded_int_accepts_in_range_values(
    value: object, minimum: int, expected: int
) -> None:
    """Integers at or above the bound should parse from strings and ints alike.

    :param object value: Raw value handed to the validator.
    :param int minimum: Inclusive lower bound.
    :param int expected: Expected parsed integer.
    :return None: Asserts the parsed value.
    """
    assert parse_bounded_int(value, minimum=minimum) == expected


@pytest.mark.parametrize(
    ("value", "minimum", "reason"),
    [
        ("-1", 0, "below_minimum"),
        ("0", 1, "below_minimum"),
        (-1, 0, "below_minimum"),
        ("1.5", 0, "not_integer"),
        ("", 0, "not_integer"),
        ("   ", 0, "not_integer"),
        ("abc", 0, "not_integer"),
        ("nan", 0, "not_integer"),
        (1.5, 0, "not_integer"),
        (2.0, 0, "not_integer"),
        (None, 0, "not_integer"),
        (True, 0, "not_integer"),
        (False, 0, "not_integer"),
    ],
)
def test_parse_bounded_int_rejects_invalid_values(
    value: object, minimum: int, reason: str
) -> None:
    """Non-integers, booleans, and under-bound values should carry a reason.

    :param object value: Raw value handed to the validator.
    :param int minimum: Inclusive lower bound.
    :param str reason: Expected failure reason token.
    :return None: Asserts the raised reason.
    """
    with pytest.raises(ValueValidationError) as excinfo:
        parse_bounded_int(value, minimum=minimum)

    assert excinfo.value.reason == reason
    assert isinstance(excinfo.value, ValueError)


def test_parse_bounded_int_rejects_booleans_before_int_subclassing() -> None:
    """``True`` must not slip through as ``1`` even when the bound allows it."""
    with pytest.raises(ValueValidationError) as excinfo:
        parse_bounded_int(True, minimum=1)

    assert excinfo.value.reason == "not_integer"
    assert str(excinfo.value) == "expected an integer, got a boolean"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", 0.0), ("1", 1.0), ("0.5", 0.5), (0.0, 0.0), (1.0, 1.0), (" 0.25 ", 0.25)],
)
def test_parse_unit_interval_float_accepts_inclusive_bounds(
    value: object, expected: float
) -> None:
    """Both endpoints of ``[0.0, 1.0]`` are valid thresholds.

    :param object value: Raw value handed to the validator.
    :param float expected: Expected parsed float.
    :return None: Asserts the parsed value.
    """
    assert parse_unit_interval_float(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("nan", "not_finite"),
        ("inf", "not_finite"),
        ("-inf", "not_finite"),
        (math.nan, "not_finite"),
        (math.inf, "not_finite"),
        (-math.inf, "not_finite"),
        ("1.0001", "out_of_range"),
        ("-0.0001", "out_of_range"),
        (2.0, "out_of_range"),
        (-1.0, "out_of_range"),
        ("abc", "not_number"),
        ("", "not_number"),
        (None, "not_number"),
        (True, "not_number"),
        (False, "not_number"),
    ],
)
def test_parse_unit_interval_float_rejects_invalid_values(
    value: object, reason: str
) -> None:
    """NaN, infinity, out-of-range, and non-numeric values each get a reason.

    :param object value: Raw value handed to the validator.
    :param str reason: Expected failure reason token.
    :return None: Asserts the raised reason.
    """
    with pytest.raises(ValueValidationError) as excinfo:
        parse_unit_interval_float(value)

    assert excinfo.value.reason == reason


@pytest.mark.parametrize(
    ("caster", "value", "message"),
    [
        (_positive_int, "0", "must be at least 1"),
        (_positive_int, "abc", "must be an integer"),
        (_positive_int, "1.5", "must be an integer"),
        (_non_negative_int, "-1", "must be at least 0"),
        (_threshold_float, "nan", "must be a finite float"),
        (_threshold_float, "inf", "must be a finite float"),
        (_threshold_float, "1.5", "must be between 0.0 and 1.0"),
        (_threshold_float, "-0.5", "must be between 0.0 and 1.0"),
        (_threshold_float, "abc", "must be a float"),
    ],
)
def test_parser_wrappers_keep_argparse_wording(
    caster: object, value: str, message: str
) -> None:
    """CLI wrappers must keep their argparse phrasing on top of shared rules.

    :param object caster: Argparse ``type`` callable under test.
    :param str value: Raw argparse value.
    :param str message: Expected ``ArgumentTypeError`` text.
    :return None: Asserts the argparse message.
    """
    with pytest.raises(argparse.ArgumentTypeError, match=f"^{message}$"):
        caster(value)  # type: ignore[operator]


def test_parser_bounded_int_reports_its_own_minimum() -> None:
    """``_bounded_int`` should echo the caller's bound, not the shared wording."""
    with pytest.raises(argparse.ArgumentTypeError, match="^must be at least 4$"):
        _bounded_int("3", minimum=4)

    assert _bounded_int("4", minimum=4) == 4


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (0, "expected an integer >= 1"),
        ("0", "expected an integer >= 1"),
        (True, "expected an integer, got a boolean"),
        ("abc", "expected an integer"),
        (1.5, "expected an integer"),
        (2.0, "expected an integer"),
    ],
)
def test_config_int_caster_keeps_config_wording(value: object, message: str) -> None:
    """Config casters must keep their ``expected ...`` phrasing.

    :param object value: Raw TOML value.
    :param str message: Expected :class:`ConfigValueError` text.
    :return None: Asserts the config message.
    """
    with pytest.raises(ConfigValueError, match=f"^{message}$"):
        _int_caster(1)(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "expected a float, got a boolean"),
        ("abc", "expected a float"),
        (None, "expected a float"),
        (math.nan, "expected a finite float between 0.0 and 1.0"),
        (math.inf, "expected a finite float between 0.0 and 1.0"),
        (2.0, "expected a finite float between 0.0 and 1.0"),
        (-0.5, "expected a finite float between 0.0 and 1.0"),
    ],
)
def test_config_similarity_caster_keeps_config_wording(
    value: object, message: str
) -> None:
    """``_cast_similarity`` collapses non-finite and out-of-range into one message.

    :param object value: Raw TOML value.
    :param str message: Expected :class:`ConfigValueError` text.
    :return None: Asserts the config message.
    """
    with pytest.raises(ConfigValueError, match=f"^{message}$"):
        _cast_similarity(value)


def test_config_and_parser_agree_on_accepted_values() -> None:
    """Both layers accept the same values even though they phrase failures apart."""
    assert _int_caster(1)("  5 ") == _positive_int("  5 ") == 5
    assert _cast_similarity("0.5") == _threshold_float("0.5") == pytest.approx(0.5)
    assert _cast_similarity(1.0) == _threshold_float("1.0") == pytest.approx(1.0)
