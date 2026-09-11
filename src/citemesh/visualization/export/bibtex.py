"""Deterministic BibTeX entry rendering and LaTeX escaping."""

from __future__ import annotations

import re
from typing import Any

from ..years import coerce_publication_year
from .keys import slug_key
from .links import _derive_doi_value

# Identifier fields BibTeX consumers resolve verbatim, so LaTeX escaping them
# would break every machine reader.
_BIBTEX_VERBATIM_FIELDS = frozenset({"doi", "url"})


def _bibtex_entry_key(node_id: str) -> str:
    """Build deterministic BibTeX entry keys from node IDs.

    :param str node_id: Graph node ID.
    :return str: Readable node slug plus a stable identifier-derived suffix.
    """
    return slug_key(
        node_id,
        prefix="citemesh_",
        fallback="paper",
        lowercase=True,
        digest_length=12,
    )


def _bibtex_escape(raw_value: str) -> str:
    """Escape text for conservative BibTeX field rendering.

    Handles every LaTeX special: ``{ } % & # $ _`` gain a backslash,
    ``~``/``^`` use their text-mode commands, and a literal backslash
    becomes ``\\textbackslash{}`` (``\\\\`` would typeset a line break).
    Backslashes are staged through a sentinel first so the escapes this
    method itself emits are not re-escaped.

    :param str raw_value: Raw field value.
    :return str: Escaped value safe for brace-delimited fields.
    """
    collapsed = " ".join(str(raw_value).split())
    sentinel = "\x00"
    collapsed = collapsed.replace("\\", sentinel)
    collapsed = collapsed.replace("{", "\\{")
    collapsed = collapsed.replace("}", "\\}")
    for special in ("%", "&", "#", "$", "_"):
        collapsed = collapsed.replace(special, f"\\{special}")
    collapsed = collapsed.replace("~", "\\textasciitilde{}")
    collapsed = collapsed.replace("^", "\\textasciicircum{}")
    return collapsed.replace(sentinel, "\\textbackslash{}")


def _bibtex_verbatim(raw_value: str) -> str:
    """Render an identifier field without LaTeX escaping.

    ``doi`` and ``url`` are consumed by machines, so escaping ``_`` or ``%``
    would corrupt them. They stay literal; only whitespace and the characters
    that would unbalance the surrounding braces are removed.

    :param str raw_value: Raw identifier value.
    :return str: Value safe to place inside a brace-delimited field.
    """
    collapsed = " ".join(str(raw_value).split())
    return re.sub(r"[{}\\]", "", collapsed)


def _node_bibtex(node_payload: dict[str, Any], *, links: dict[str, str | None]) -> str:
    """Render a deterministic BibTeX entry for dashboard actions.

    :param Dict[str, Any] node_payload: Node payload.
    :param Dict[str, Optional[str]] links: Derived external links.
    :return str: BibTeX entry string.
    """
    key = _bibtex_entry_key(str(node_payload.get("id", "")))
    fields: list[tuple[str, str]] = []
    title = str(node_payload.get("title") or "").strip()
    if title:
        fields.append(("title", title))

    authors = node_payload.get("authors", [])
    if isinstance(authors, list):
        author_names = [
            str(author).strip() for author in authors if str(author).strip()
        ]
        if author_names:
            fields.append(("author", " and ".join(author_names)))

    year = coerce_publication_year(node_payload.get("year"))
    if year > 0:
        fields.append(("year", str(year)))

    doi_value = _derive_doi_value(
        str(node_payload.get("id", "")), node_payload=node_payload
    )
    if doi_value:
        fields.append(("doi", doi_value))

    primary_url = (
        links.get("arxiv_abs") or links.get("doi") or links.get("semantic_scholar")
    )
    if primary_url:
        fields.append(("url", primary_url))

    abstract = str(node_payload.get("abstract") or "").strip()
    if abstract:
        fields.append(("abstract", abstract))

    lines = [f"@article{{{key},"]
    for field, value in fields:
        rendered = (
            _bibtex_verbatim(value)
            if field in _BIBTEX_VERBATIM_FIELDS
            else _bibtex_escape(value)
        )
        lines.append(f"  {field} = {{{rendered}}},")
    lines.append("}")
    return "\n".join(lines)
