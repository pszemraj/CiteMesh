"""Canonical effective metadata for graph nodes across visualization outputs."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Any

from citemesh.core import Paper


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


def effective_node_metadata(
    node_id: Hashable, attrs: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the metadata every renderer and exporter should expose for a node.

    An attached :class:`~citemesh.core.models.Paper` owns bibliographic metadata.
    The graph's explicit ``is_seed`` attribute remains authoritative because seed
    role belongs to one graph build rather than to the reusable paper record.

    :param Hashable node_id: Graph node identifier.
    :param Mapping[str, Any] attrs: Raw graph node attributes.
    :return Dict[str, Any]: Effective node metadata without mutating ``attrs``.
    """
    paper = attrs.get("paper")
    if isinstance(paper, Paper):
        return {
            "id": node_id,
            "title": paper.title,
            "year": paper.year,
            "authors": [author.name for author in paper.authors],
            "citation_count": paper.citation_count,
            "abstract": paper.abstract,
            "venue": paper.venue,
            "arxiv_id": paper.arxiv_id,
            "doi": paper.doi,
            "categories": paper.categories,
            "is_seed": bool(attrs.get("is_seed", paper.is_seed)),
            "is_local_corpus": bool(paper.is_local_corpus),
        }

    return {
        "id": node_id,
        "title": attrs.get("title", ""),
        "year": attrs.get("year"),
        "authors": attrs.get("authors") or [],
        "citation_count": attrs.get("citation_count", 0),
        "abstract": attrs.get("abstract", ""),
        "venue": attrs.get("venue", ""),
        "arxiv_id": attrs.get("arxiv_id", ""),
        "doi": attrs.get("doi", ""),
        "categories": attrs.get("categories") or [],
        "is_seed": bool(attrs.get("is_seed", False)),
        "is_local_corpus": bool(attrs.get("is_local_corpus", False)),
    }
