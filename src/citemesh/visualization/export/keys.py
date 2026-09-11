"""Deterministic identifier slugging shared by the export formats."""

from __future__ import annotations

import hashlib
import re

_SLUG_ALNUM_RE = re.compile(r"[^0-9a-zA-Z]+")
_SLUG_ALNUM_UNDERSCORE_RE = re.compile(r"[^0-9a-zA-Z_]+")


def slug_key(
    text: object,
    *,
    prefix: str,
    fallback: str,
    keep_underscores: bool = False,
    lowercase: bool = False,
    digest_length: int = 0,
) -> str:
    """Build a stable, format-safe key from arbitrary text.

    Runs of disallowed characters collapse to a single underscore, leading and
    trailing underscores are dropped, and an all-disallowed input becomes
    ``fallback``. When ``digest_length`` is positive, a SHA-256 prefix of the
    *original* text is appended so keys stay unique after slugging collapses
    distinct inputs.

    :param object text: Source text (stringified before slugging).
    :param str prefix: Literal prefix placed in front of the slug.
    :param str fallback: Slug used when nothing survives normalization.
    :param bool keep_underscores: Treat ``_`` as an allowed character.
    :param bool lowercase: Lowercase the slug body.
    :param int digest_length: Hex digest characters to append, ``0`` for none.
    :return str: Prefixed slug, optionally suffixed with a stable digest.
    """
    source = str(text)
    pattern = _SLUG_ALNUM_UNDERSCORE_RE if keep_underscores else _SLUG_ALNUM_RE
    normalized = pattern.sub("_", source).strip("_")
    if lowercase:
        normalized = normalized.lower()
    if not normalized:
        normalized = fallback
    if digest_length <= 0:
        return f"{prefix}{normalized}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:digest_length]
    return f"{prefix}{normalized}_{digest}"
