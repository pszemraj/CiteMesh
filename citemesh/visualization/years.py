"""Shared publication-year normalization for visualization and export surfaces."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable

MISSING_YEAR_FALLBACK_MIN = 2000
MISSING_YEAR_FALLBACK_MAX = 2001


def coerce_publication_year(raw_year: object) -> int:
    """Normalize an optional year value to an integer or ``0``.

    :param object raw_year: Raw year value from paper or node metadata.
    :return int: Integer year when valid, otherwise ``0``.
    """
    if isinstance(raw_year, bool):
        return 0
    if isinstance(raw_year, numbers.Integral):
        return int(raw_year)
    if isinstance(raw_year, numbers.Real):
        if math.isfinite(raw_year) and float(raw_year).is_integer():
            return int(raw_year)
        return 0
    if isinstance(raw_year, str):
        try:
            return int(raw_year)
        except ValueError:
            pass
    return 0


def publication_year_bounds(raw_years: Iterable[object]) -> tuple[int, int]:
    """Return valid publication-year bounds with a deterministic fallback.

    :param Iterable[object] raw_years: Raw year values to inspect.
    :return tuple[int, int]: Minimum and maximum usable years.
    """
    valid_years = [
        year
        for year in (coerce_publication_year(value) for value in raw_years)
        if year > 0
    ]
    if valid_years:
        return min(valid_years), max(valid_years)
    return MISSING_YEAR_FALLBACK_MIN, MISSING_YEAR_FALLBACK_MAX


def publication_year_scale(
    raw_years: Iterable[object],
) -> tuple[list[float], float, float]:
    """Return Plotly-compatible year values and nondegenerate scale bounds.

    :param Iterable[object] raw_years: Raw year values in node order.
    :return tuple[list[float], float, float]: Normalized years, minimum, and maximum.
    """
    years = [coerce_publication_year(value) for value in raw_years]
    year_min_raw, year_max_raw = publication_year_bounds(years)
    year_min = float(year_min_raw)
    year_max = float(year_max_raw)
    if year_max <= year_min:
        year_max = year_min + 1.0
    midpoint = (year_min + year_max) / 2.0
    normalized_years = [float(year) if year > 0 else midpoint for year in years]
    return normalized_years, year_min, year_max
