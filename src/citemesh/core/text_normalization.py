"""Unicode-safe text normalization for embedding inputs and exported text."""

from __future__ import annotations

import rehuman

# Keep every script, emoji and math symbol; drop invisible and control characters;
# fold typographic quotes, dashes and spaces; NFC only, since NFKC would rewrite
# superscripts and other compatibility characters that carry meaning in abstracts.
_OPTIONS = rehuman.Options(
    remove_hidden=True,
    remove_trailing_whitespace=True,
    normalize_spaces=True,
    normalize_dashes=True,
    normalize_quotes=True,
    normalize_other=True,
    keyboard_only=False,
    keep_emoji=True,
    preserve_joiners=True,
    remove_control_chars=True,
    collapse_whitespace=True,
    line_endings="lf",
    unicode_normalization="nfc",
)
_CLEANER = rehuman.Cleaner(_OPTIONS)


def normalize_text(text: str) -> str:
    """Return ``text`` cleaned by rehuman and folded onto one line.

    rehuman collapses runs of horizontal whitespace but keeps line breaks, and
    metadata sources such as arXiv hard-wrap titles and abstracts, so the
    remaining line breaks are folded into single spaces.

    :param str text: Raw title, abstract or query text.
    :return str: Single-spaced text with no leading or trailing whitespace.
    """
    return " ".join(_CLEANER.clean(text).text.split())
