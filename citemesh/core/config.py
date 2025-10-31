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

    # Exponential decay factor for year differences
    year_decay_factor: float = 8.0

    def year_similarity(self, year_diff: int) -> float:
        """
        Compute temporal similarity based on year difference.

        Args:
            year_diff: Absolute difference in publication years

        Returns:
            Similarity score from 0.0 (very distant) to 1.0 (same year)
        """
        if year_diff > self.max_year_diff_threshold:
            return 0.1  # Strong penalty for distant papers
        return 1.0 - (year_diff / self.max_year_diff_threshold) * 0.8


@dataclass
class CitationSimilarityConfig:
    """Configuration for citation-based similarity."""

    # Similarity weights (must sum to 1.0)
    temporal_weight: float = 0.3
    citation_weight: float = 0.3
    bibliographic_weight: float = 0.4

    # Edge creation thresholds
    seed_edge_threshold: float = 0.45  # Lower threshold for seed connections
    normal_edge_threshold: float = 0.65  # Standard threshold for other papers
    random_edge_probability: float = 0.7  # Probability to create edge above threshold

    def validate(self) -> None:
        """Ensure weights sum to 1.0."""
        total = self.temporal_weight + self.citation_weight + self.bibliographic_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Citation similarity weights must sum to 1.0, got {total:.3f}"
            )


@dataclass
class EmbeddingSimilarityConfig:
    """Configuration for embedding-based similarity."""

    # Multi-factor similarity weights (must sum to 1.0)
    semantic_weight: float = 0.5
    temporal_weight: float = 0.2
    category_weight: float = 0.2
    author_weight: float = 0.1

    # Author collaboration bonus multiplier
    shared_author_bonus: float = 1.5

    # Temporal factor scaling
    temporal_scale: float = 3.0  # Divisor for year difference

    # Top-k neighbors per node
    top_k_neighbors: int = 2

    def validate(self) -> None:
        """Ensure weights sum to 1.0."""
        total = (
            self.semantic_weight
            + self.temporal_weight
            + self.category_weight
            + self.author_weight
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Embedding similarity weights must sum to 1.0, got {total:.3f}"
            )

    def temporal_factor(self, year_diff: int) -> float:
        """Compute temporal proximity factor for embeddings."""
        return 1.0 / (1.0 + year_diff / self.temporal_scale)


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

    # Relevance scoring for citations
    def relevance_score(self, citation_count: int, year_diff: int) -> float:
        """Compute relevance score for filtering citations."""
        return citation_count / (1 + year_diff)


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization."""

    # Figure settings
    # Default output size: 1440px tall at 200 DPI with golden-ratio width.
    figure_size: Tuple[float, float] = (11.65, 7.2)
    dpi: int = 200
    background_color: str = "#fafafa"

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

    # Color gradient (RGB normalized to 0-1)
    # Light blue/gray (old) → dark teal (recent)
    color_start: Tuple[float, float, float] = (0.72, 0.83, 0.89)  # Light
    color_end: Tuple[float, float, float] = (0.45, 0.64, 0.61)  # Dark

    # Seed paper color
    seed_color: Tuple[float, float, float] = (0.87, 0.27, 0.27)  # Red

    # Edge rendering
    edge_alpha_min: float = 0.3
    edge_alpha_max: float = 0.6
    edge_width_min: float = 0.3
    edge_width_max: float = 1.5
    edge_base_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    # Layout parameters
    layout_scale: float = 0.9
    layout_center: Tuple[float, float] = (0.5, 0.5)
    perturbation_std: float = 0.02  # Random perturbation for organic look
    spring_k_factor: float = 0.8  # Spring layout k parameter divisor

    # Font settings
    font_size: int = 8
    font_weight: str = "normal"

    def compute_node_color(
        self, year: int, min_year: int, max_year: int
    ) -> Tuple[float, float, float]:
        """
        Compute smooth RGB gradient color based on year.

        Args:
            year: Paper's publication year
            min_year: Earliest year in graph
            max_year: Latest year in graph

        Returns:
            RGB tuple (normalized 0-1)
        """
        if max_year == min_year:
            year_norm = 0.5
        else:
            year_norm = (year - min_year) / (max_year - min_year)

        r = self.color_start[0] - (self.color_start[0] - self.color_end[0]) * year_norm
        g = self.color_start[1] - (self.color_start[1] - self.color_end[1]) * year_norm
        b = self.color_start[2] - (self.color_start[2] - self.color_end[2]) * year_norm

        return (r, g, b)


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
CITATION_CONFIG.validate()
EMBEDDING_CONFIG.validate()
