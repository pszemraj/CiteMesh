"""
Seed-relevance ranking of a hybrid discovery pool.

Every candidate carries one retrieval-document vector. Ranked lists over the
pool are fused by reciprocal rank fusion with ``k = 60``:

- ``semantic``: every candidate by the cosine between the seed's and the
  candidate's document vectors (``cos_dd``); ties by the cosine with the seed's
  query-role vector (``cos_q``), then paper ID.
- ``reference_context``: every candidate with at least five seed references
  other than itself, by the mean of its five largest cosines to those
  references (``ctx_ref_top5``); ties as ``semantic``.
- ``direct``: candidates the seed cites or that cite the seed, by ``cos_dd``;
  ties by paper ID.
- ``recommended``: the provider's recommendations in provider order, each
  paper at its first appearance; ties by paper ID. Without recommendations the
  list is empty and the other three decide.

A candidate scores the sum of ``1 / (k + rank)`` over the lists that contain
it, with 1-based ranks, and the final order is by score descending, then paper
ID. Nothing is fitted and no source has a quota. Cosines are computed between
unit-normalized vectors, so the ranking ignores vector scale, and candidates
are processed in paper-ID order, so it ignores input order.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

RRF_K = 60
REFERENCE_CONTEXT_SIZE = 5

_SortKey = tuple[float | str, ...]
Vector = NDArray[np.float32] | NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class DiscoveryCandidate:
    """One pool member presented to the ranker.

    :param str paper_id: Unique pool ID, also the final tie-breaker.
    :param NDArray vector: Retrieval-document vector of any positive scale.
    :param bool is_reference: Whether the seed cites this paper.
    :param bool cites_seed: Whether this paper cites the seed.
    :param int | None recommendation_rank: Provider position of the paper's
        first appearance among the recommendations, or ``None`` when it was
        not recommended.
    """

    paper_id: str
    vector: Vector
    is_reference: bool = False
    cites_seed: bool = False
    recommendation_rank: int | None = None


@dataclass(frozen=True)
class CandidateFeatures:
    """Seed-relative features of one candidate.

    :param float cos_dd: Cosine with the seed's document vector.
    :param float cos_q: Cosine with the seed's query-role vector.
    :param float | None ctx_ref_top5: Mean of the five largest cosines with
        seed references other than the candidate, or ``None`` when fewer than
        five such references exist.
    :param bool direct: Whether the seed cites the candidate or the candidate
        cites the seed.
    :param int | None recommendation_rank: Provider recommendation position.
    """

    cos_dd: float
    cos_q: float
    ctx_ref_top5: float | None
    direct: bool
    recommendation_rank: int | None


@dataclass(frozen=True)
class RankedCandidate:
    """One candidate's fused score, final rank and evidence.

    :param str paper_id: Pool ID.
    :param int rank: 1-based position in the fused order.
    :param float score: Sum of ``1 / (k + rank)`` over the containing lists.
    :param CandidateFeatures features: Values the lists were ordered by.
    :param Mapping[str, int] list_ranks: 1-based rank in each containing list.
    """

    paper_id: str
    rank: int
    score: float
    features: CandidateFeatures
    list_ranks: Mapping[str, int]


def _semantic_key(paper_id: str, features: CandidateFeatures) -> _SortKey:
    """Order the whole pool by document cosine, then query cosine and ID."""
    return (-features.cos_dd, -features.cos_q, paper_id)


def _reference_context_key(
    paper_id: str, features: CandidateFeatures
) -> _SortKey | None:
    """Order candidates with a reference context by it, then as ``semantic``."""
    if features.ctx_ref_top5 is None:
        return None
    return (-features.ctx_ref_top5, -features.cos_q, paper_id)


def _direct_key(paper_id: str, features: CandidateFeatures) -> _SortKey | None:
    """Order direct citation links by document cosine, then ID."""
    return (-features.cos_dd, paper_id) if features.direct else None


def _recommended_key(paper_id: str, features: CandidateFeatures) -> _SortKey | None:
    """Order recommendations by provider position, then ID."""
    if features.recommendation_rank is None:
        return None
    return (float(features.recommendation_rank), paper_id)


# The fused lists, in summation order. A key returns ``None`` for a candidate
# the list does not contain.
FUSED_LISTS: tuple[
    tuple[str, Callable[[str, CandidateFeatures], _SortKey | None]], ...
] = (
    ("semantic", _semantic_key),
    ("reference_context", _reference_context_key),
    ("direct", _direct_key),
    ("recommended", _recommended_key),
)


def _unit_rows(vectors: Sequence[Vector]) -> NDArray[np.float64]:
    """Stack vectors into unit-length float64 rows.

    :param Sequence[NDArray] vectors: Finite, nonzero vectors of one length.
    :return NDArray[np.float64]: ``(len(vectors), dimension)`` unit rows.
    """
    matrix = np.vstack([np.asarray(vector, dtype=np.float64) for vector in vectors])
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def compute_features(
    candidates: Sequence[DiscoveryCandidate],
    *,
    seed_document: Vector,
    seed_query: Vector,
) -> dict[str, CandidateFeatures]:
    """Compute every candidate's seed-relative features.

    The reference set for ``ctx_ref_top5`` is the candidates flagged as
    references; each candidate is compared with the references other than
    itself. Callers validate the vectors: finite, nonzero and of one length.

    :param Sequence[DiscoveryCandidate] candidates: Pool members with unique IDs.
    :param NDArray seed_document: Seed's retrieval-document vector.
    :param NDArray seed_query: Seed's retrieval-query vector.
    :return dict[str, CandidateFeatures]: Features keyed by paper ID, in ID order.
    """
    ordered = sorted(candidates, key=lambda candidate: candidate.paper_id)
    if not ordered:
        return {}
    seed_vectors = _unit_rows([seed_document, seed_query])
    vectors = _unit_rows([candidate.vector for candidate in ordered])
    cos_dd = vectors @ seed_vectors[0]
    cos_q = vectors @ seed_vectors[1]

    reference_rows = [
        row for row, candidate in enumerate(ordered) if candidate.is_reference
    ]
    context: list[float | None] = [None] * len(ordered)
    if len(reference_rows) >= REFERENCE_CONTEXT_SIZE:
        similarities = vectors @ vectors[reference_rows].T
        for column, row in enumerate(reference_rows):
            similarities[row, column] = -np.inf
        available = len(reference_rows) - np.asarray(
            [candidate.is_reference for candidate in ordered], dtype=np.int64
        )
        top = np.partition(
            similarities, len(reference_rows) - REFERENCE_CONTEXT_SIZE, axis=1
        )[:, -REFERENCE_CONTEXT_SIZE:]
        means = np.sort(top, axis=1).mean(axis=1)
        context = [
            float(means[row]) if available[row] >= REFERENCE_CONTEXT_SIZE else None
            for row in range(len(ordered))
        ]

    return {
        candidate.paper_id: CandidateFeatures(
            cos_dd=float(cos_dd[row]),
            cos_q=float(cos_q[row]),
            ctx_ref_top5=context[row],
            direct=candidate.is_reference or candidate.cites_seed,
            recommendation_rank=candidate.recommendation_rank,
        )
        for row, candidate in enumerate(ordered)
    }


def fused_lists(features: Mapping[str, CandidateFeatures]) -> dict[str, list[str]]:
    """Order the pool into each fused list.

    :param Mapping[str, CandidateFeatures] features: Features keyed by paper ID.
    :return dict[str, list[str]]: Paper IDs per list name, best first.
    """
    lists: dict[str, list[str]] = {}
    for name, sort_key in FUSED_LISTS:
        keyed = [
            (key, paper_id)
            for paper_id, values in features.items()
            if (key := sort_key(paper_id, values)) is not None
        ]
        lists[name] = [paper_id for _key, paper_id in sorted(keyed)]
    return lists


def reciprocal_rank_fusion(
    lists: Mapping[str, Sequence[str]],
) -> list[tuple[str, float]]:
    """Fuse ranked lists by summing ``1 / (RRF_K + rank)`` per containing list.

    Each score is a correctly rounded sum (:func:`math.fsum`), so papers with
    the same ranks in any lists, in any list order, score identical floats and
    the tie-break by paper ID decides between them.

    :param Mapping[str, Sequence[str]] lists: Ranked paper IDs per list, best
        first, each ID at most once per list.
    :return list[tuple[str, float]]: ``(paper_id, score)`` by score descending,
        then paper ID.
    """
    terms: dict[str, list[float]] = {}
    for ranked in lists.values():
        for rank, paper_id in enumerate(ranked, start=1):
            terms.setdefault(paper_id, []).append(1.0 / (RRF_K + rank))
    scores = [(paper_id, math.fsum(values)) for paper_id, values in terms.items()]
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def rank_candidates(
    candidates: Sequence[DiscoveryCandidate],
    *,
    seed_document: Vector,
    seed_query: Vector,
) -> list[RankedCandidate]:
    """Rank a discovery pool by reciprocal rank fusion of :data:`FUSED_LISTS`.

    :param Sequence[DiscoveryCandidate] candidates: Pool members with unique IDs
        and validated vectors.
    :param NDArray seed_document: Seed's retrieval-document vector.
    :param NDArray seed_query: Seed's retrieval-query vector.
    :return list[RankedCandidate]: Every candidate, best first.
    """
    features = compute_features(
        candidates, seed_document=seed_document, seed_query=seed_query
    )
    lists = fused_lists(features)
    list_ranks: dict[str, dict[str, int]] = {paper_id: {} for paper_id in features}
    for name, ranked in lists.items():
        for rank, paper_id in enumerate(ranked, start=1):
            list_ranks[paper_id][name] = rank
    return [
        RankedCandidate(
            paper_id=paper_id,
            rank=rank,
            score=score,
            features=features[paper_id],
            list_ranks=list_ranks[paper_id],
        )
        for rank, (paper_id, score) in enumerate(reciprocal_rank_fusion(lists), start=1)
    ]
