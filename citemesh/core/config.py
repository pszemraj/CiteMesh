"""
Configuration constants for CiteMesh.

This module centralizes all magic numbers, thresholds, and weights
to make the codebase self-documenting and enable easy experimentation.
"""

from dataclasses import dataclass
from typing import Tuple


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
class CitationSimilarityConfig:
    """Configuration for citation-based similarity thresholds."""

    # Edge creation thresholds
    seed_edge_threshold: float = 0.45  # Lower threshold for seed connections
    normal_edge_threshold: float = 0.65  # Standard threshold for other papers


@dataclass
class EmbeddingSimilarityConfig:
    """Configuration for embedding-based similarity."""

    # Multi-factor similarity weights.
    semantic_weight: float = 0.5
    temporal_weight: float = 0.2
    category_weight: float = 0.2

    # Author collaboration bonus multiplier
    shared_author_bonus: float = 1.5

    def validate(self) -> None:
        """Ensure component weights are valid."""
        total = (
            self.semantic_weight
            + self.temporal_weight
            + self.category_weight
        )
        if total <= 0 or total > 1.0:
            raise ValueError(
                f"Embedding similarity weights must total within (0.0, 1.0], got {total:.3f}"
            )


@dataclass
class HybridSimilarityConfig:
    """Configuration for hybrid similarity approach."""

    # Adaptive weights for different relationship types
    semantic_semantic_weights: Tuple[float, float, float, float] = (0.6, 0.2, 0.1, 0.1)
    citation_citation_weights: Tuple[float, float, float, float] = (0.3, 0.3, 0.2, 0.2)
    mixed_weights: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)

    # Co-citation boost
    co_citation_boost: float = 0.2

    # Edge limiting
    max_edges_per_node: int = 5


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization."""

    # Figure settings
    figure_size: Tuple[int, int] = (12, 10)
    dpi: int = 150

    # Node size parameters (in square pixels)
    seed_size: int = 2500
    max_non_seed_size: int = 2200
    min_size: int = 100

    # Node size tiers (rank: base_size)
    size_tiers = {
        "top_3": (1200, 200),  # (base, increment)
        "top_8": (500, 80),
        "top_15": (250, 30),
    }

    # Edge rendering
    edge_alpha_min: float = 0.3
    edge_alpha_max: float = 0.6
    edge_width_min: float = 0.3
    edge_width_max: float = 1.5

    # Layout parameters
    layout_scale: float = 0.9
    layout_center: Tuple[float, float] = (0.5, 0.5)
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
    long_timeout: float = 60.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0  # Initial delay, increases exponentially

    # Rate limiting
    requests_per_second: float = 0.5  # Conservative rate limit


# Global config instances (can be overridden)
TEMPORAL_CONFIG = TemporalConfig()
CITATION_CONFIG = CitationSimilarityConfig()
EMBEDDING_CONFIG = EmbeddingSimilarityConfig()
HYBRID_CONFIG = HybridSimilarityConfig()
VIZ_CONFIG = VisualizationConfig()
API_CONFIG = APIConfig()

# Validate all configs on import
EMBEDDING_CONFIG.validate()
