"""Text-length guardrails for visualization rendering surfaces."""

from __future__ import annotations

from typing import Any

MAX_RENDER_TEXT_CHARS = 10_000
TRUNCATION_LABEL_TEMPLATE = " ...[truncated +{overflow} chars]"


def clamp_render_text(value: Any, max_chars: int = MAX_RENDER_TEXT_CHARS) -> str:
    """Clamp render-bound text length and append a truncation indicator when needed.

    :param Any value: Raw text-like value destined for render output.
    :param int max_chars: Maximum number of rendered characters.
    :return str: Render-safe text with truncation marker on overflow.
    :raises ValueError: If ``max_chars`` is less than ``1``.
    """
    if int(max_chars) < 1:
        raise ValueError("max_chars must be at least 1")

    text = "" if value is None else str(value)
    if len(text) <= int(max_chars):
        return text

    overflow = len(text) - int(max_chars)
    marker = TRUNCATION_LABEL_TEMPLATE.format(overflow=overflow)
    keep = max(0, int(max_chars) - len(marker))
    return f"{text[:keep]}{marker}"
