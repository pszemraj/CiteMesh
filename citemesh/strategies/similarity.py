"""Shared similarity utilities for strategy implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

from citemesh.core import Paper


@dataclass(frozen=True)
class SimilarityFeatures:
    """Computed feature bundle and weighted score for a node pair."""

    abstract_similarity: float
    temporal_similarity: float
    citation_similarity: float
    bibliographic_coupling: float
    has_bibliographic_coupling: bool
    combined_score: float


def compute_similarity_features(
    paper1: Paper,
    paper2: Paper,
    *,
    abstract_similarity_fn: Callable[[Paper, Paper], float],
    temporal_similarity_fn: Callable[[Paper, Paper], float],
    citation_similarity_fn: Callable[[Paper, Paper], float],
    bibliographic_coupling_fn: Callable[[Paper, Paper], float],
    with_references_weights: Tuple[float, float, float, float],
    without_references_weights: Tuple[float, float, float, float],
    use_bibliographic_coupling: bool = False,
) -> SimilarityFeatures:
    """Compute reusable similarity components and weighted aggregate.

    :param Paper paper1: First paper node.
    :param Paper paper2: Second paper node.
    :param Callable[[Paper, Paper], float] abstract_similarity_fn: Abstract similarity
        function.
    :param Callable[[Paper, Paper], float] temporal_similarity_fn: Temporal similarity
        function.
    :param Callable[[Paper, Paper], float] citation_similarity_fn: Citation similarity
        function.
    :param Callable[[Paper, Paper], float] bibliographic_coupling_fn: Bibliographic
        coupling function.
    :param Tuple[float, float, float, float] with_references_weights: Weights for
        (abstract, temporal, citation, bibliographic_coupling) when both papers have
        references.
    :param Tuple[float, float, float, float] without_references_weights: Weights for
        (abstract, temporal, citation, bibliographic_coupling) when bibliographic coupling
        should not be used.
    :param bool use_bibliographic_coupling: Enable reference-based branch when both papers
        expose references.
    :return SimilarityFeatures: Structured component-level output and weighted score.
    """
    abstract_similarity = abstract_similarity_fn(paper1, paper2)
    temporal_similarity = temporal_similarity_fn(paper1, paper2)
    citation_similarity = citation_similarity_fn(paper1, paper2)

    has_bibliographic_coupling = (
        use_bibliographic_coupling
        and bool(paper1.references)
        and bool(paper2.references)
    )
    bibliographic_coupling = (
        bibliographic_coupling_fn(paper1, paper2)
        if has_bibliographic_coupling
        else 0.0
    )

    if has_bibliographic_coupling:
        weights = with_references_weights
    else:
        weights = without_references_weights

    abstract_weight, temporal_weight, citation_weight, bibliographic_weight = weights
    combined_score = (
        abstract_weight * abstract_similarity
        + temporal_weight * temporal_similarity
        + citation_weight * citation_similarity
        + bibliographic_weight * bibliographic_coupling
    )

    return SimilarityFeatures(
        abstract_similarity=abstract_similarity,
        temporal_similarity=temporal_similarity,
        citation_similarity=citation_similarity,
        bibliographic_coupling=bibliographic_coupling,
        has_bibliographic_coupling=has_bibliographic_coupling,
        combined_score=combined_score,
    )
