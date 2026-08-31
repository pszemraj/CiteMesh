"""
Embedding model profiles with optional formatting/runtime hints.

This allows CiteMesh to adapt prompts and other behaviours for specific
embedding checkpoints without hard-coding logic in the strategies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple

QueryFormatter = Callable[[str, Optional[Dict[str, str]]], str]
DocumentFormatter = Callable[[Dict[str, str]], str]
SimilarityFormatter = Callable[[str, Optional[Dict[str, str]]], str]

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL_NAME = "unsloth/embeddinggemma-300m"
DEFAULT_EMBEDDING_MODEL_FALLBACKS: Mapping[str, Tuple[str, ...]] = {
    "unsloth/embeddinggemma-300m": ("google/embeddinggemma-300m",),
}
EMBEDDING_MODEL_PROFILE_CHOICES: Tuple[str, ...] = (
    "auto",
    "default",
    "embeddinggemma",
)


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


def _identity_similarity_formatter(text: str, _: Optional[Dict[str, str]]) -> str:
    """Return symmetric-similarity input text unchanged.

    :param str text: Composed paper text.
    :param Optional[Dict[str, str]] _: Unused metadata context.
    :return str: Unmodified similarity text.
    """
    return text


@dataclass(frozen=True)
class EmbeddingModelProfile:
    """Per-model hints used by embedding strategies."""

    name: str
    schema_token: str
    aliases: Tuple[str, ...] = ()
    minimum_transformers_version: Optional[Tuple[int, int]] = None
    query_formatter: QueryFormatter = _identity_query_formatter
    document_formatter: DocumentFormatter = _identity_document_formatter
    similarity_formatter: SimilarityFormatter = _identity_similarity_formatter
    preferred_compute_dtype: Optional[str] = None
    autocast_devices: Tuple[str, ...] = ()
    preferred_attention_implementation: Optional[str] = None
    requires_bidirectional_attention: bool = False
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

    def format_similarity(
        self, text: str, metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Format paper text for symmetric semantic-similarity scoring.

        :param str text: Composed paper text.
        :param Optional[Dict[str, str]] metadata: Optional paper metadata context.
        :return str: Profile-formatted symmetric-similarity input.
        """
        return self.similarity_formatter(text, metadata)

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


def _gemma_similarity_formatter(text: str, _: Optional[Dict[str, str]]) -> str:
    """Format paper text for EmbeddingGemma's symmetric STS task.

    :param str text: Composed title and abstract text.
    :param Optional[Dict[str, str]] _: Unused metadata context.
    :return str: EmbeddingGemma sentence-similarity prompt.
    """
    return f"task: sentence similarity | query: {text.strip()}"


DEFAULT_PROFILE = EmbeddingModelProfile(
    name="default",
    schema_token="default-v1",
)

EMBEDDING_MODEL_PROFILES = (
    EmbeddingModelProfile(
        name="google/embeddinggemma",
        schema_token="embeddinggemma-v2",
        aliases=("unsloth/embeddinggemma",),
        minimum_transformers_version=(4, 57),
        query_formatter=_gemma_query_formatter,
        document_formatter=_gemma_document_formatter,
        similarity_formatter=_gemma_similarity_formatter,
        preferred_compute_dtype="bfloat16",
        autocast_devices=("cuda", "mps"),
        preferred_attention_implementation="sdpa",
        requires_bidirectional_attention=True,
        compile_inner_transformer=True,
        available_truncate_dims=(768, 512, 256, 128),
        recommended_truncate_dim=256,
        notes=(
            "Adds recommended retrieval-query, retrieval-document, and symmetric "
            "sentence-similarity prompts for EmbeddingGemma. "
            "Runs bf16 through autocast on supported CUDA and MPS runtimes; "
            "otherwise uses float32."
        ),
    ),
)

_PROFILE_BY_KEY: Mapping[str, EmbeddingModelProfile] = {
    "default": DEFAULT_PROFILE,
    "embeddinggemma": EMBEDDING_MODEL_PROFILES[0],
}


def _read_json_object(path: Path) -> Dict[str, object]:
    """Read a JSON object, returning an empty mapping for absent/invalid files.

    :param Path path: JSON file to inspect.
    :return Dict[str, object]: Parsed object or an empty mapping.
    """
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _local_transformer_configs(root: Path) -> Tuple[Dict[str, object], ...]:
    """Read transformer configs from root and SentenceTransformers modules.

    :param Path root: Local model directory.
    :return Tuple[Dict[str, object], ...]: Non-empty transformer config objects.
    """
    config_paths = [root / "config.json"]
    try:
        modules = json.loads((root / "modules.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        modules = []
    raw_modules = modules if isinstance(modules, list) else []

    for module in raw_modules:
        if not isinstance(module, dict):
            continue
        module_type = str(module.get("type", "")).casefold()
        module_path = module.get("path")
        if "transformer" not in module_type or not isinstance(module_path, str):
            continue
        config_paths.append(root / module_path / "config.json")

    configs = tuple(_read_json_object(path) for path in dict.fromkeys(config_paths))
    return tuple(config for config in configs if config)


def _local_embeddinggemma_evidence(root: Path) -> Tuple[bool, bool, bool]:
    """Inspect local metadata for EmbeddingGemma contract evidence.

    :param Path root: Local model directory.
    :return Tuple[bool, bool, bool]: Architecture, bidirectional-attention, and
        SentenceTransformers-task evidence flags.
    """
    has_gemma_architecture = False
    has_bidirectional_attention = False
    for transformer_config in _local_transformer_configs(root):
        architectures = transformer_config.get("architectures", [])
        normalized_architectures = (
            {value.casefold() for value in architectures if isinstance(value, str)}
            if isinstance(architectures, list)
            else set()
        )
        model_type = str(transformer_config.get("model_type", "")).casefold()
        if model_type == "gemma3_text" or "gemma3textmodel" in normalized_architectures:
            has_gemma_architecture = True
            has_bidirectional_attention = has_bidirectional_attention or (
                transformer_config.get("use_bidirectional_attention") is True
            )

    sentence_transformer_config = _read_json_object(
        root / "config_sentence_transformers.json"
    )
    raw_prompts = sentence_transformer_config.get("prompts", {})
    prompt_names = (
        {str(name).casefold() for name in raw_prompts}
        if isinstance(raw_prompts, dict)
        else set()
    )
    has_embedding_tasks = {
        "retrieval-query",
        "retrieval-document",
        "sts",
    }.issubset(prompt_names)
    return (
        has_gemma_architecture,
        has_bidirectional_attention,
        has_embedding_tasks,
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


def resolve_embedding_model_profile(
    model_name_or_path: str,
    requested_profile: str = "auto",
) -> EmbeddingModelProfile:
    """Resolve a model profile from an override, local artifact, or Hub alias.

    :param str model_name_or_path: Hub identifier or local checkpoint directory.
    :param str requested_profile: ``auto`` or an explicit profile key.
    :return EmbeddingModelProfile: Resolved task/runtime contract.
    :raises ValueError: If ``requested_profile`` is unknown.
    """
    normalized_request = str(requested_profile).strip().casefold()
    if normalized_request != "auto":
        try:
            return _PROFILE_BY_KEY[normalized_request]
        except KeyError as exc:
            choices = ", ".join(EMBEDDING_MODEL_PROFILE_CHOICES)
            raise ValueError(
                f"Unknown model profile {requested_profile!r}; expected one of: {choices}."
            ) from exc

    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        architecture, bidirectional_attention, embedding_tasks = (
            _local_embeddinggemma_evidence(local_path)
        )
        if architecture and (bidirectional_attention or embedding_tasks):
            return _PROFILE_BY_KEY["embeddinggemma"]
        if architecture and not (bidirectional_attention or embedding_tasks):
            logger.warning(
                "Local Gemma 3 checkpoint %s lacks both "
                "use_bidirectional_attention=true and EmbeddingGemma task prompts; "
                "automatic profile detection cannot prove the embedding contract. "
                "Using the default profile; pass --model-profile embeddinggemma only "
                "if this artifact is an EmbeddingGemma export.",
                local_path,
            )
        return DEFAULT_PROFILE

    return get_embedding_model_profile(model_name_or_path)
