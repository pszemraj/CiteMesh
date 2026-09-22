"""Core data structures and configuration for CiteMesh."""

from __future__ import annotations

from .config import (
    API_CONFIG,
    DEFAULT_MAX_PAPERS,
    DEFAULT_RELATIONSHIP_SIMILARITY_THRESHOLD,
    EMBEDDING_CONFIG,
    EMBEDDING_STORAGE_CONFIG,
    HYBRID_CONFIG,
    TEMPORAL_CONFIG,
    VIZ_CONFIG,
)
from .models import Author, Paper

__all__ = [
    "Author",
    "Paper",
    "API_CONFIG",
    "DEFAULT_MAX_PAPERS",
    "DEFAULT_RELATIONSHIP_SIMILARITY_THRESHOLD",
    "EMBEDDING_CONFIG",
    "EMBEDDING_STORAGE_CONFIG",
    "HYBRID_CONFIG",
    "TEMPORAL_CONFIG",
    "VIZ_CONFIG",
]
