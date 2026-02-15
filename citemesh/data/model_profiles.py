"""
Embedding model profiles with optional formatting/runtime hints.

This allows CiteMesh to adapt prompts and other behaviours for specific
embedding checkpoints without hard-coding logic in the strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple

QueryFormatter = Callable[[str, Optional[Dict[str, str]]], str]
DocumentFormatter = Callable[[Dict[str, str]], str]

DEFAULT_EMBEDDING_MODEL_NAME = "unsloth/embeddinggemma-300m"
DEFAULT_EMBEDDING_MODEL_FALLBACKS: Mapping[str, Tuple[str, ...]] = {
    "unsloth/embeddinggemma-300m": ("google/embeddinggemma-300m",),
}


def compose_title_abstract_text(metadata: Mapping[str, object]) -> str:
    """Compose a stable document string from title/abstract metadata.

    :param Mapping[str, object] metadata: Paper metadata payload.
    :return str: Best-effort ``"title. abstract"`` representation.
    """
    raw_title = metadata.get("title", "")
    raw_abstract = metadata.get("abstract", "")
    title = str(raw_title).strip() if raw_title is not None else ""
    abstract = str(raw_abstract).strip() if raw_abstract is not None else ""
    if title and abstract:
        return f"{title}. {abstract}"
    if title:
        return title
    return abstract


def _identity_query_formatter(text: str, _: Optional[Dict[str, str]]) -> str:
    """Return input text unchanged for query formatting.

    :param str text: Original query text.
    :param Optional[Dict[str, str]] _: Unused metadata context.
    :return str: Unmodified query text.
    """
    return text


def _identity_document_formatter(metadata: Dict[str, str]) -> str:
    """Compose a minimal document string from title and abstract.

    :param Dict[str, str] metadata: Paper metadata payload.
    :return str: Best-effort ``\"title. abstract\"`` representation.
    """
    return compose_title_abstract_text(metadata)


@dataclass(frozen=True)
class EmbeddingModelProfile:
    """Per-model hints used by embedding strategies."""

    name: str
    aliases: Tuple[str, ...] = ()
    query_formatter: QueryFormatter = _identity_query_formatter
    document_formatter: DocumentFormatter = _identity_document_formatter
    float16_supported: bool = True
    preferred_torch_dtype: Optional[str] = None
    use_cuda_autocast: bool = False
    compile_inner_transformer: bool = False
    available_truncate_dims: Optional[Tuple[int, ...]] = None
    recommended_truncate_dim: Optional[int] = None
    notes: Optional[str] = None

    def format_query(self, text: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """Format a query using the profile-specific rule.

        :param str text: Raw query string.
        :param Optional[Dict[str, str]] metadata: Optional metadata context.
        :return str: Profile-formatted query string.
        """
        return self.query_formatter(text, metadata)

    def format_document(self, metadata: Dict[str, str]) -> str:
        """Format paper metadata into embedding model input text.

        :param Dict[str, str] metadata: Paper metadata payload.
        :return str: Profile-formatted document text.
        """
        return self.document_formatter(metadata)

    def matches(self, model_name: str) -> bool:
        """Return whether model identifier maps to this profile.

        :param str model_name: Lowercased model identifier.
        :return bool: ``True`` when identifier matches profile name or alias.
        """
        if model_name.startswith(self.name):
            return True
        return any(model_name.startswith(alias) for alias in self.aliases)


def _gemma_query_formatter(text: str, _: Optional[Dict[str, str]]) -> str:
    """Format a query with Gemma-style task prompt.

    :param str text: Raw query text.
    :param Optional[Dict[str, str]] _: Unused metadata context.
    :return str: Prompt-prefixed query string.
    """
    text = text.strip()
    return f"task: search result | query: {text}"


def _gemma_document_formatter(metadata: Dict[str, str]) -> str:
    """Format paper metadata with explicit fields for Gemma models.

    :param Dict[str, str] metadata: Paper metadata payload.
    :return str: Gemma-friendly text representation.
    """
    title = metadata.get("title") or "none"
    abstract = metadata.get("abstract") or ""
    title = title.strip() or "none"
    abstract = abstract.strip()
    return f"title: {title} | text: {abstract}"


DEFAULT_PROFILE = EmbeddingModelProfile(name="default")

EMBEDDING_MODEL_PROFILES = (
    EmbeddingModelProfile(
        name="google/embeddinggemma",
        aliases=("unsloth/embeddinggemma",),
        query_formatter=_gemma_query_formatter,
        document_formatter=_gemma_document_formatter,
        float16_supported=False,
        preferred_torch_dtype="bfloat16",
        use_cuda_autocast=True,
        compile_inner_transformer=True,
        available_truncate_dims=(768, 512, 256, 128),
        recommended_truncate_dim=256,
        notes="Adds recommended query/document prompts for EmbeddingGemma.",
    ),
)


def get_embedding_model_profile(model_name: str) -> EmbeddingModelProfile:
    """Return the best matching profile for a model name.

    :param str model_name: Model identifier.
    :return EmbeddingModelProfile: Selected profile, falling back to default.
    """
    normalized = model_name.lower()
    for profile in EMBEDDING_MODEL_PROFILES:
        if profile.matches(normalized):
            return profile
    return DEFAULT_PROFILE
