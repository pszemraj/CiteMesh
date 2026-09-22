"""
Configuration constants for CiteMesh.

This module centralizes all magic numbers, thresholds, and weights
to make the codebase self-documenting and enable easy experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .choices import STORAGE_PRECISION_CHOICES

DEFAULT_MAX_PAPERS = 40
DEFAULT_RELATIONSHIP_SIMILARITY_THRESHOLD = 0.2


@dataclass
class TemporalConfig:
    """Configuration for temporal similarity calculations."""

    # Papers more than this many years apart rarely connect
    max_year_diff_threshold: int = 5

    def year_similarity(self, year_diff: int) -> float:
        """
        Compute temporal similarity based on year difference.

        :param int year_diff: Absolute difference in publication years
        :return float: Similarity score from 0.0 (very distant) to 1.0 (same year)
        """
        if year_diff > self.max_year_diff_threshold:
            return 0.1  # Strong penalty for distant papers
        return 1.0 - (year_diff / self.max_year_diff_threshold) * 0.8


@dataclass
class EmbeddingSimilarityConfig:
    """Configuration for embedding-based similarity."""

    # Real-abstract development grid winner; see the semantic quality fixtures.
    min_semantic_similarity: float = 0.74

    # Multi-factor similarity weights.
    semantic_weight: float = 0.5
    temporal_weight: float = 0.2
    category_weight: float = 0.2

    # Author collaboration bonus multiplier
    shared_author_bonus: float = 1.5

    def validate(self) -> None:
        """Ensure component weights are valid."""
        total = self.semantic_weight + self.temporal_weight + self.category_weight
        if total <= 0 or total > 1.0:
            raise ValueError(
                f"Embedding similarity weights must total within (0.0, 1.0], got {total:.3f}"
            )


@dataclass
class EmbeddingStorageConfig:
    """Configuration for embedding cache storage/retrieval behavior."""

    storage_precision: str = "int8"
    binary_prefilter: bool = True
    binary_rescore_multiplier: int = 8
    calibration_sample_size: int = 2000
    compression: str = "gzip"
    compression_level: int = 1

    def validate(self) -> None:
        """Ensure storage settings are valid."""
        if self.storage_precision not in STORAGE_PRECISION_CHOICES:
            raise ValueError(
                "Embedding storage_precision must be one of {'float32', 'int8'}"
            )
        if self.binary_rescore_multiplier < 1:
            raise ValueError("binary_rescore_multiplier must be at least 1")
        if self.calibration_sample_size < 1:
            raise ValueError("calibration_sample_size must be at least 1")
        if self.compression_level < 0:
            raise ValueError("compression_level must be non-negative")


@dataclass
class HybridSimilarityConfig:
    """Configuration for hybrid similarity approach."""

    # Adaptive weights for different relationship types
    semantic_semantic_weights: tuple[float, float, float, float] = (0.6, 0.2, 0.1, 0.1)
    citation_citation_weights: tuple[float, float, float, float] = (0.3, 0.3, 0.2, 0.2)
    mixed_weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)

    # Edge limiting
    max_edges_per_node: int = 5


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization."""

    # Figure settings
    figure_size: tuple[int, int] = (14, 8)
    dpi: int = 150

    # Node area caps and minimum (in square points for static scatter plots)
    seed_size: int = 2500
    max_non_seed_size: int = 2200
    min_size: int = 100

    # Node size tiers (rank: (base, increment))
    size_tiers: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "top_3": (1200, 200),
            "top_8": (500, 80),
            "top_15": (250, 30),
        }
    )

    # Edge rendering
    edge_alpha_min: float = 0.3
    edge_alpha_max: float = 0.6
    edge_width_min: float = 0.3
    edge_width_max: float = 1.5

    # Layout parameters
    spring_iterations: int = 100
    layout_scale: float = 0.9
    layout_center: tuple[float, float] = (0.5, 0.5)
    perturbation_std: float = 0.02  # Random perturbation for organic look
    spring_k_factor: float = 0.8  # Spring layout k parameter divisor

    # Font settings
    font_size: int = 8
    font_weight: str = "normal"


@dataclass
class APIConfig:
    """Configuration for API interactions."""

    # Timeout settings (seconds)
    default_timeout: float = 30.0

    # Unauthenticated builds use a bounded recovery window after the first failed
    # request. An explicit API key keeps the established attempt-only policy.
    max_retries: int = 30
    retry_delay: float = 2.0  # Initial delay, increases exponentially
    anonymous_retry_budget_seconds: float = 90.0

    # Rate limiting: anonymous requests share a small public pool; authenticated
    # keys are granted 1 request/second by Semantic Scholar.
    requests_per_second: float = 0.5
    authenticated_requests_per_second: float = 1.0


# Global config instances (can be overridden)
TEMPORAL_CONFIG = TemporalConfig()
EMBEDDING_CONFIG = EmbeddingSimilarityConfig()
EMBEDDING_STORAGE_CONFIG = EmbeddingStorageConfig()
HYBRID_CONFIG = HybridSimilarityConfig()
VIZ_CONFIG = VisualizationConfig()
API_CONFIG = APIConfig()

# Validate all configs on import
EMBEDDING_CONFIG.validate()
EMBEDDING_STORAGE_CONFIG.validate()
