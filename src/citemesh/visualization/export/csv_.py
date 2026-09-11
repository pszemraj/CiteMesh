"""CSV cell sanitization for spreadsheet formula injection defense."""

from __future__ import annotations

_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_cell_guard(value: object) -> str:
    """Neutralize spreadsheet formula interpretation for one CSV text cell.

    Excel/Sheets execute cells starting with ``=``, ``+``, ``-``, ``@``, tab,
    or CR as formulas (CWE-1236). Prefixing an apostrophe forces text
    rendering; the dashboard's in-page CSV exporter applies the same rule.

    :param object value: Raw text cell value (``None`` renders empty).
    :return str: Cell text, apostrophe-prefixed when formula-leading.
    """
    text = "" if value is None else str(value)
    if text.startswith(_CSV_FORMULA_PREFIXES):
        return f"'{text}"
    return text
