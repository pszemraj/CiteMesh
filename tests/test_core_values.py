"""Tests for the shared scalar coercion helpers."""

from __future__ import annotations

import math

import pytest

from citemesh.core.values import coerce_citation_count, coerce_float


@pytest.mark.parametrize(
    ("raw", "default", "expected"),
    [
        (1.5, 0.0, 1.5),
        (3, 0.0, 3.0),
        ("2.25", 0.0, 2.25),
        ("  4 ", 0.0, 4.0),
        ("-1.5", 0.0, -1.5),
        (None, 0.0, 0.0),
        ("", 0.0, 0.0),
        ("abc", 0.0, 0.0),
        ([1.0], 0.0, 0.0),
        ({"a": 1}, 0.0, 0.0),
        (object(), 7.5, 7.5),
        ("nope", -2.0, -2.0),
    ],
)
def test_coerce_float_parses_or_falls_back(
    raw: object, default: float, expected: float
) -> None:
    """Numeric-looking input parses; anything else returns the default."""
    assert coerce_float(raw, default) == expected


def test_coerce_float_treats_booleans_as_numbers() -> None:
    """Booleans keep the numeric reading every call site already relied on."""
    assert coerce_float(True, 9.0) == 1.0
    assert coerce_float(False, 9.0) == 0.0


def test_coerce_float_passes_through_non_finite_text() -> None:
    """``nan``/``inf`` text parses; finiteness checks stay with the caller."""
    assert math.isnan(coerce_float("nan", 0.0))
    assert coerce_float("inf", 0.0) == math.inf
    assert coerce_float("-inf", 0.0) == -math.inf


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (12, 12),
        (1, 1),
        (0, 0),
        (-5, 0),
        (7.9, 7),
        (-7.9, 0),
        ("42", 42),
        ("-42", 0),
        (None, 0),
        (True, 0),
        (False, 0),
        ("", 0),
        ("12.5", 0),
        ("twelve", 0),
        (float("nan"), 0),
        ("nan", 0),
        ("inf", 0),
        ([3], 0),
    ],
)
def test_coerce_citation_count(raw: object, expected: int) -> None:
    """Citation counts normalize to non-negative integers or zero."""
    assert coerce_citation_count(raw) == expected
