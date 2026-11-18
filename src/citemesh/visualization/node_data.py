"""Canonical effective metadata for graph nodes across visualization outputs."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping

from citemesh.core import Paper
from citemesh.core.text_normalization import normalize_text

from .dashboard.contracts import EffectiveNodeMetadata


def validate_canonical_node_ids(node_ids: Iterable[Hashable]) -> None:
    """Validate node IDs shared by static and structured graph exports.

    :param Iterable[Hashable] node_ids: Node identifiers to validate.
    :return None: Returns after every identifier passes validation.
    :raises ValueError: If an ID is empty, has surrounding whitespace, or collides
        with another ID after string conversion.
    """
    serialized_owners: dict[str, Hashable] = {}
    for node_id in node_ids:
        identifier = str(node_id)
        if not identifier or identifier != identifier.strip():
            raise ValueError(
                f"Cannot export non-canonical node ID {node_id!r}: "
                "IDs must be non-empty and have no surrounding whitespace."
            )
        if identifier in serialized_owners:
            existing = serialized_owners[identifier]
            raise ValueError(
                f"Cannot export node IDs {existing!r} and {node_id!r}: "
                f"both serialize as {identifier!r}."
            )
        serialized_owners[identifier] = node_id


def _display_text(value: object) -> object:
    """Normalize a title or abstract for export, leaving non-strings unchanged.

    arXiv metadata hard-wraps titles and abstracts and other sources carry
    invisible or typographic characters; exports show the same normalized text
    that embedding inputs use.

    :param object value: Raw title or abstract field.
    :return object: :func:`normalize_text` of ``value`` when it is a string,
        otherwise ``value`` itself.
    """
    return normalize_text(value) if isinstance(value, str) else value


def effective_node_metadata(
    node_id: Hashable, attrs: Mapping[str, object]
) -> EffectiveNodeMetadata:
    """Return the metadata every renderer and exporter should expose for a node.

    An attached :class:`~citemesh.core.models.Paper` owns bibliographic metadata.
    The graph's explicit ``is_focal`` attribute remains authoritative because focal
    role belongs to one graph build rather than to the reusable paper record.
    Titles and abstracts are normalized here exactly as embedding inputs are.

    :param Hashable node_id: Graph node identifier.
    :param Mapping[str, object] attrs: Raw graph node attributes.
    :return EffectiveNodeMetadata: Selected metadata without mutating ``attrs``.
    """
    paper = attrs.get("paper")
    if isinstance(paper, Paper):
        return {
            "id": node_id,
            "title": _display_text(paper.title),
            "year": paper.year,
            "authors": [author.name for author in paper.authors],
            "citation_count": paper.citation_count,
            "abstract": _display_text(paper.abstract),
            "venue": paper.venue,
            "arxiv_id": paper.arxiv_id,
            "doi": paper.doi,
            "categories": paper.categories,
            "is_focal": bool(attrs.get("is_focal", paper.is_focal)),
            "is_local_corpus": bool(paper.is_local_corpus),
        }

    return {
        "id": node_id,
        "title": _display_text(attrs.get("title", "")),
        "year": attrs.get("year"),
        "authors": attrs.get("authors") or [],
        "citation_count": attrs.get("citation_count", 0),
        "abstract": _display_text(attrs.get("abstract", "")),
        "venue": attrs.get("venue", ""),
        "arxiv_id": attrs.get("arxiv_id", ""),
        "doi": attrs.get("doi", ""),
        "categories": attrs.get("categories") or [],
        "is_focal": bool(attrs.get("is_focal", False)),
        "is_local_corpus": bool(attrs.get("is_local_corpus", False)),
    }
