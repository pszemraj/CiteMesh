"""Caching and model profile utilities for CiteMesh."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._lazy import install_lazy_exports

if TYPE_CHECKING:
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

# Lazy so ``citemesh.data.cache`` users never pay for h5py/numpy via the package.
__getattr__, __dir__ = install_lazy_exports(
    globals(),
    {
        "get_cache_dir": (".cache", "get_cache_dir"),
        "format_bytes": (".cache", "format_bytes"),
        "EmbeddingCache": (".embedding_cache", "EmbeddingCache"),
        "validate_compression_filter": (
            ".embedding_cache",
            "validate_compression_filter",
        ),
        "DEFAULT_EMBEDDING_MODEL_NAME": (
            ".model_profiles",
            "DEFAULT_EMBEDDING_MODEL_NAME",
        ),
        "DEFAULT_EMBEDDING_MODEL_FALLBACKS": (
            ".model_profiles",
            "DEFAULT_EMBEDDING_MODEL_FALLBACKS",
        ),
        "EMBEDDING_MODEL_PROFILE_CHOICES": (
            ".model_profiles",
            "EMBEDDING_MODEL_PROFILE_CHOICES",
        ),
        "get_embedding_model_profile": (
            ".model_profiles",
            "get_embedding_model_profile",
        ),
        "resolve_embedding_model_profile": (
            ".model_profiles",
            "resolve_embedding_model_profile",
        ),
    },
)
