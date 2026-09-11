"""Prompt-conditioned text formatting for embedding inputs.

Owns the embedding task roles and the single place where a paper's title and
abstract become model input text, so retrieval documents, graph-similarity
vectors and cached corpus rows are all formatted identically.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import (
    Any,
)

from citemesh.core import Paper
from citemesh.data.model_profiles import compose_title_abstract_text

_RETRIEVAL_DOCUMENT_REPRESENTATION = "retrieval-document-v1"
_GRAPH_SIMILARITY_REPRESENTATION = "graph-similarity-v1"
_PLACEHOLDER_EMBEDDING_TITLES = frozenset(
    {"", "n/a", "na", "none", "unknown", "untitled"}
)
_FORMATTER_FINGERPRINT_PROBES = (
    {"title": "Alpha", "abstract": "Beta"},
    {"title": "Alpha", "abstract": ""},
    {"title": "", "abstract": "Beta"},
    {"title": "  Alpha  ", "abstract": "  Beta  "},
)


class EmbeddingTask(str, Enum):
    """Prompt-conditioned embedding roles used by CiteMesh."""

    RETRIEVAL_QUERY = "retrieval-query"
    RETRIEVAL_DOCUMENT = "retrieval-document"
    GRAPH_SIMILARITY = "graph-similarity"


def _embedding_text_metadata(title: object, abstract: object) -> dict[str, str]:
    """Normalize title/abstract fields without treating placeholders as content.

    :param object title: Raw paper title.
    :param object abstract: Raw paper abstract.
    :return Dict[str, str]: Clean text metadata for prompt formatting.
    """
    normalized_title = str(title or "").strip()
    if normalized_title.casefold() in _PLACEHOLDER_EMBEDDING_TITLES:
        normalized_title = ""
    return {
        "title": normalized_title,
        "abstract": str(abstract or "").strip(),
    }


def format_paper_for_embedding(
    *, profile: Any, paper: Paper, task: EmbeddingTask
) -> str:
    """Format one paper for a specific retrieval or graph task.

    :param Any profile: Active embedding model profile.
    :param Paper paper: Paper whose title/abstract should be formatted.
    :param EmbeddingTask task: Required prompt-conditioned vector role.
    :return str: Model input text, falling back to the paper ID when needed.
    :raises ValueError: If ``task`` is unsupported.
    """
    return format_embedding_metadata(
        profile=profile,
        metadata={"title": paper.title, "abstract": paper.abstract},
        paper_id=paper.paper_id,
        task=task,
    )


def format_embedding_metadata(
    *,
    profile: Any,
    metadata: Mapping[str, object],
    task: EmbeddingTask,
    paper_id: object = "",
) -> str:
    """Format paper metadata for one prompt-conditioned embedding role.

    :param Any profile: Active embedding model profile.
    :param Mapping[str, object] metadata: Paper metadata containing text fields.
    :param EmbeddingTask task: Required prompt-conditioned vector role.
    :param object paper_id: Identity fallback when title and abstract are empty.
    :return str: Profile-formatted input with a stable identity fallback.
    :raises ValueError: If ``task`` is unsupported.
    """
    text_metadata = _embedding_text_metadata(
        metadata.get("title"), metadata.get("abstract")
    )
    fallback_id = str(paper_id or metadata.get("paper_id") or "unknown-paper")
    content = compose_title_abstract_text(text_metadata) or fallback_id
    if task is EmbeddingTask.RETRIEVAL_QUERY:
        return str(profile.format_query(content, text_metadata))
    if task is EmbeddingTask.RETRIEVAL_DOCUMENT:
        document_metadata = dict(text_metadata)
        if not compose_title_abstract_text(document_metadata):
            document_metadata["title"] = fallback_id
        return str(profile.format_document(document_metadata)) or content
    if task is EmbeddingTask.GRAPH_SIMILARITY:
        return str(profile.format_similarity(content, text_metadata)) or content
    raise ValueError(f"Unsupported embedding task: {task}")
