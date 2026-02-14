"""Core data structures and configuration for CiteMesh."""

from .config import (
    API_CONFIG,
    EMBEDDING_CONFIG,
    HYBRID_CONFIG,
    TEMPORAL_CONFIG,
    VIZ_CONFIG,
)
from .models import Author, Paper

__all__ = [
    "Author",
    "Paper",
    "API_CONFIG",
    "EMBEDDING_CONFIG",
    "HYBRID_CONFIG",
    "TEMPORAL_CONFIG",
    "VIZ_CONFIG",
]
