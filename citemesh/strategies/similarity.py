"""Shared similarity utilities for strategy implementations."""

from __future__ import annotations

from typing import Callable, Protocol, Tuple

from citemesh.core import Paper


class _AbstractIndexProtocol(Protocol):
    """Protocol for abstract-similarity indexes used by strategies."""

    def similarity(self, paper_id_a: str, paper_id_b: str) -> float:
        """Return similarity score between two paper IDs.

        :param str paper_id_a: First paper identifier.
        :param str paper_id_b: Second paper identifier.
        :return float: Similarity score in ``[0, 1]``.
        """
        ...


def compute_indexed_similarity_score(
    paper1: Paper,
    paper2: Paper,
    *,
    abstract_index: _AbstractIndexProtocol,
    temporal_similarity_fn: Callable[[Paper, Paper], float],
    citation_similarity_fn: Callable[[Paper, Paper], float],
    bibliographic_coupling_fn: Callable[[Paper, Paper], float],
    fetch_references: bool,
    with_references_weights: Tuple[float, float, float, float],
    without_references_weights: Tuple[float, float, float, float],
    cap_at_one: bool = False,
) -> float:
    """Compute weighted similarity score using a shared abstract index.

    :param Paper paper1: First paper node.
    :param Paper paper2: Second paper node.
    :param _AbstractIndexProtocol abstract_index: Abstract similarity index.
    :param Callable[[Paper, Paper], float] temporal_similarity_fn: Temporal component.
    :param Callable[[Paper, Paper], float] citation_similarity_fn: Citation component.
    :param Callable[[Paper, Paper], float] bibliographic_coupling_fn: Bibliographic component.
    :param bool fetch_references: Whether reference-based scoring is enabled.
    :param Tuple[float, float, float, float] with_references_weights: Weights with references.
    :param Tuple[float, float, float, float] without_references_weights: Weights without references.
    :param bool cap_at_one: Whether to clamp result to 1.0.
    :return float: Combined similarity score.
    """
    abstract_similarity = abstract_index.similarity(paper1.paper_id, paper2.paper_id)
    temporal_similarity = temporal_similarity_fn(paper1, paper2)
    citation_similarity = citation_similarity_fn(paper1, paper2)
    has_bibliographic_coupling = bool(
        fetch_references and paper1.references and paper2.references
    )
    bibliographic_coupling = (
        bibliographic_coupling_fn(paper1, paper2) if has_bibliographic_coupling else 0.0
    )
    weights = (
        with_references_weights
        if has_bibliographic_coupling
        else without_references_weights
    )
    abstract_weight, temporal_weight, citation_weight, bibliographic_weight = weights
    score = (
        abstract_weight * abstract_similarity
        + temporal_weight * temporal_similarity
        + citation_weight * citation_similarity
        + bibliographic_weight * bibliographic_coupling
    )
    if cap_at_one:
        return min(score, 1.0)
    return score
