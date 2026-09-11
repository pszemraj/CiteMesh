"""Caching and model profile utilities for CiteMesh."""

from .cache import format_bytes, get_cache_dir
from .embedding_cache import EmbeddingCache, validate_compression_filter
from .model_profiles import (
    DEFAULT_EMBEDDING_MODEL_FALLBACKS,
    DEFAULT_EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PROFILE_CHOICES,
    get_embedding_model_profile,
    resolve_embedding_model_profile,
)

__all__ = [
    "get_cache_dir",
    "format_bytes",
    "EmbeddingCache",
    "validate_compression_filter",
    "DEFAULT_EMBEDDING_MODEL_NAME",
    "DEFAULT_EMBEDDING_MODEL_FALLBACKS",
    "EMBEDDING_MODEL_PROFILE_CHOICES",
    "get_embedding_model_profile",
    "resolve_embedding_model_profile",
]
