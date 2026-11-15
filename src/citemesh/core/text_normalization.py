"""Unicode-safe text normalization for embedding inputs and exported text."""

from __future__ import annotations

import rehuman

# Keep every script, emoji and math symbol; drop invisible and control characters;
# fold typographic quotes, dashes and spaces; NFC only, since NFKC would rewrite
# superscripts and other compatibility characters that carry meaning in abstracts.
# Dash folding also turns minus signs, superscript and subscript minus included,
# into "-", so 10⁻³ becomes 10-³. Any change here changes the embedded text: the
# formatter fingerprint probes in strategies/embedding/text.py must change with it.
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
    remaining line breaks are folded into single spaces. Lone surrogates, which
    a JSON payload or an undecodable command-line argument can carry, become
    ``?`` first, since rehuman cannot encode them; valid text is unaffected.

    :param str text: Raw title, abstract or query text.
    :return str: Single-spaced text with no leading or trailing whitespace.
    """
    text = text.encode("utf-8", "replace").decode("utf-8")
    return " ".join(_CLEANER.clean(text).text.split())
