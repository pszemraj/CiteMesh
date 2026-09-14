"""External link derivation and inline-script-safe serialization."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote

from citemesh.core.paper_ids import (
    canonicalize_or_none,
    external_ids_from_canonical_paper_id,
    is_local_corpus_paper_id,
    recognize_arxiv_identifier,
)


def _semantic_scholar_resolves_id(node_id: str) -> bool:
    """Return whether a corpus primary ID is a supported external paper identity.

    :param str node_id: Graph node identifier.
    :return bool: Whether the ID is an arXiv, DOI, or opaque S2 paper ID.
    """
    canonical_id = canonicalize_or_none(node_id) or node_id
    arxiv_id, doi = external_ids_from_canonical_paper_id(canonical_id)
    return bool(
        arxiv_id
        or doi
        or recognize_arxiv_identifier(canonical_id, allow_bare=True)
        or re.fullmatch(r"(?:s2:)?[0-9a-f]{40}", canonical_id, flags=re.IGNORECASE)
    )


def _safe_script_content(raw: str) -> str:
    """Escape script-closing tokens in trusted inline script bodies.

    Only suitable for trusted library code (e.g. the bundled plotly.js).
    User-controlled data must go through :meth:`_script_safe_json`, which
    removes every ``<`` so the HTML tokenizer can never enter the
    script-data-escaped states (``<!--`` + ``<script``) that would swallow
    the closing ``</script>`` tag.

    :param str raw: Raw script body content.
    :return str: Script-safe content.
    """
    return raw.replace("</", "<\\/")


def _script_safe_json(payload: Any) -> str:
    """Serialize a payload as JSON that is inert inside an HTML ``<script>``.

    ``json.dumps`` leaves ``<`` unescaped, so upstream text such as
    ``<!--<script>`` in a paper abstract would otherwise drive the HTML
    tokenizer into the script-data-double-escaped state and break the whole
    document. ``<`` can only occur inside JSON string literals, so the
    global ``\\u003c`` rewrite is loss-free for ``JSON.parse``.

    :param Any payload: JSON-serializable payload.
    :return str: Compact deterministic JSON with every ``<`` escaped.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).replace("<", "\\u003c")


def _derive_doi_value(
    node_id: str,
    *,
    node_payload: dict[str, Any] | None = None,
) -> str:
    """Resolve the raw DOI of a node from metadata or its canonical ID.

    Callers that build URLs percent-encode the result themselves; BibTeX and
    other identifier consumers need this unencoded form.

    :param str node_id: Canonical graph node identifier.
    :param Optional[Dict[str, Any]] node_payload: Optional node payload carrying
        an explicit ``doi`` value.
    :return str: Raw DOI without prefix or encoding, empty when unknown.
    """
    doi_value = ""
    if isinstance(node_payload, dict):
        doi_value = str(node_payload.get("doi") or "").strip()
    if not doi_value:
        if node_id.lower().startswith("doi:"):
            doi_value = node_id.split(":", 1)[1].strip()
        elif re.match(r"^10\.\d{4,9}/\S+$", node_id):
            doi_value = node_id
    return doi_value


def _derive_links(
    node_id: str,
    *,
    node_payload: dict[str, Any] | None = None,
) -> dict[str, str | None]:
    """Derive external links from canonical node IDs.

    :param str node_id: Canonical graph node identifier.
    :param Optional[Dict[str, Any]] node_payload: Optional node payload carrying
        explicit ``arxiv_id``/``doi`` values.
    :return Dict[str, Optional[str]]: External links dictionary.
    """
    local_corpus_record = bool(
        isinstance(node_payload, dict) and node_payload.get("is_local_corpus", False)
    )
    links: dict[str, str | None] = {
        "arxiv_abs": None,
        "arxiv_pdf": None,
        "doi": None,
        "semantic_scholar": (
            None
            if is_local_corpus_paper_id(node_id)
            or node_id.startswith("query:")
            or (local_corpus_record and not _semantic_scholar_resolves_id(node_id))
            else f"https://www.semanticscholar.org/paper/{quote(node_id, safe='')}"
        ),
    }

    arxiv_value = ""
    if isinstance(node_payload, dict):
        arxiv_value = str(node_payload.get("arxiv_id") or "").strip()
    if not arxiv_value:
        arxiv_match = re.match(r"^arxiv:(.+)$", node_id, flags=re.IGNORECASE)
        if arxiv_match:
            arxiv_value = arxiv_match.group(1).strip()
    if arxiv_value:
        arxiv_id = re.sub(r"v\d+$", "", arxiv_value, flags=re.IGNORECASE)
        links["arxiv_abs"] = f"https://arxiv.org/abs/{quote(arxiv_id, safe='')}"
        links["arxiv_pdf"] = f"https://arxiv.org/pdf/{quote(arxiv_id, safe='')}.pdf"

    doi_value = _derive_doi_value(node_id, node_payload=node_payload)
    if doi_value:
        links["doi"] = f"https://doi.org/{quote(doi_value, safe='/()[]:._;-')}"
    return links
