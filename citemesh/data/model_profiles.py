"""
Embedding model profiles with optional formatting/runtime hints.

This allows CiteMesh to adapt prompts and other behaviours for specific
embedding checkpoints without hard-coding logic in the strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

QueryFormatter = Callable[[str, Optional[Dict[str, str]]], str]
DocumentFormatter = Callable[[Dict[str, str]], str]


def _identity_query_formatter(text: str, _: Optional[Dict[str, str]]) -> str:
    return text


def _identity_document_formatter(metadata: Dict[str, str]) -> str:
    title = metadata.get("title", "").strip()
    abstract = metadata.get("abstract", "").strip()
    if title and abstract:
        return f"{title}. {abstract}"
    if title:
        return title
    return abstract


@dataclass(frozen=True)
class EmbeddingModelProfile:
    """Per-model hints used by embedding strategies."""

    name: str
    query_formatter: QueryFormatter = _identity_query_formatter
    document_formatter: DocumentFormatter = _identity_document_formatter
    float16_supported: bool = True
    notes: Optional[str] = None

    def format_query(self, text: str, metadata: Optional[Dict[str, str]] = None) -> str:
        return self.query_formatter(text, metadata)

    def format_document(self, metadata: Dict[str, str]) -> str:
        return self.document_formatter(metadata)


def _gemma_query_formatter(text: str, _: Optional[Dict[str, str]]) -> str:
    text = text.strip()
    return f"task: search result | query: {text}"


def _gemma_document_formatter(metadata: Dict[str, str]) -> str:
    title = metadata.get("title") or "none"
    abstract = metadata.get("abstract") or ""
    title = title.strip() or "none"
    abstract = abstract.strip()
    return f"title: {title} | text: {abstract}"


DEFAULT_PROFILE = EmbeddingModelProfile(name="default")

EMBEDDING_MODEL_PROFILES = (
    EmbeddingModelProfile(
        name="google/embeddinggemma",
        query_formatter=_gemma_query_formatter,
        document_formatter=_gemma_document_formatter,
        float16_supported=False,
        notes="Adds recommended query/document prompts for EmbeddingGemma.",
    ),
)


def get_embedding_model_profile(model_name: str) -> EmbeddingModelProfile:
    """Return the best matching profile for a model name."""
    normalized = model_name.lower()
    for profile in EMBEDDING_MODEL_PROFILES:
        if normalized.startswith(profile.name):
            return profile
    return DEFAULT_PROFILE
