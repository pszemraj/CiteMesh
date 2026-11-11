"""Hand-computed checks of hybrid discovery ranking features and fusion."""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from citemesh.strategies.discovery_ranking import (
    FUSED_LISTS,
    DiscoveryCandidate,
    compute_features,
    fused_lists,
    rank_candidates,
    reciprocal_rank_fusion,
)

SEED_DOCUMENT = np.array([1.0, 0.0, 0.0])
SEED_QUERY = np.array([0.0, 1.0, 0.0])


def _unit(*values: float) -> np.ndarray:
    """Return a unit vector with the given direction.

    :param float values: Unnormalized components.
    :return np.ndarray: Unit-length float64 vector.
    """
    vector = np.asarray(values, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _rrf(*ranks: int) -> float:
    """Sum reciprocal ranks in list order with the shipped offset.

    :param int ranks: 1-based ranks in the containing lists.
    :return float: Fused score accumulated in the given order.
    """
    score = 0.0
    for rank in ranks:
        score += 1.0 / (60 + rank)
    return score


def test_fused_lists_are_the_four_evaluated_lists_in_summation_order() -> None:
    """The ranker fuses exactly the evaluated lists, in a fixed order."""
    assert [name for name, _key in FUSED_LISTS] == [
        "semantic",
        "reference_context",
        "direct",
        "recommended",
    ]


def test_reciprocal_rank_fusion_sums_reciprocal_ranks_and_breaks_ties_by_id() -> None:
    """Scores are 1/(k + rank) sums; equal scores order by paper ID."""
    fused = reciprocal_rank_fusion({"a": ["y", "x", "z"], "b": ["x", "y"]})

    assert fused == [("x", _rrf(2, 1)), ("y", _rrf(1, 2)), ("z", _rrf(3))]
    assert fused[0][1] == fused[1][1]
    assert reciprocal_rank_fusion({"a": ["p", "q"]}) == [
        ("p", 1.0 / 61),
        ("q", 1.0 / 62),
    ]
    assert reciprocal_rank_fusion({}) == []


@pytest.mark.parametrize("order", list(itertools.permutations(range(4))))
def test_equal_rank_multisets_tie_exactly_in_any_list_order(
    order: tuple[int, ...],
) -> None:
    """Float rounding cannot decide a tie; the paper ID does."""
    # ``a`` sits at ranks (1, 1, 2, 3) and ``z`` at (2, 3, 1, 1).
    positions = [(1, 2), (1, 3), (2, 1), (3, 1)]
    lists = {}
    for index in order:
        rank_a, rank_z = positions[index]
        ranked = [f"filler-{index}-{slot}" for slot in range(max(rank_a, rank_z))]
        ranked[rank_a - 1], ranked[rank_z - 1] = "a", "z"
        lists[f"list-{index}"] = ranked

    scores = dict(reciprocal_rank_fusion(lists))
    fused = [paper_id for paper_id, _ in reciprocal_rank_fusion(lists)]

    assert scores["a"] == scores["z"]
    assert fused.index("a") < fused.index("z")


def test_features_are_cosines_to_both_seed_roles() -> None:
    """cos_dd uses the seed document vector and cos_q its query vector."""
    features = compute_features(
        [
            DiscoveryCandidate("a", _unit(1.0, 1.0, 0.0), is_reference=True),
            DiscoveryCandidate("b", np.array([0.0, 0.0, 2.0]), cites_seed=True),
            DiscoveryCandidate("c", np.array([3.0, 0.0, 4.0]), recommendation_rank=7),
        ],
        seed_document=SEED_DOCUMENT,
        seed_query=SEED_QUERY,
    )

    assert list(features) == ["a", "b", "c"]
    assert features["a"].cos_dd == pytest.approx(math.sqrt(0.5))
    assert features["a"].cos_q == pytest.approx(math.sqrt(0.5))
    assert features["b"].cos_dd == pytest.approx(0.0)
    assert features["c"].cos_dd == pytest.approx(0.6)
    assert features["c"].cos_q == pytest.approx(0.0)
    assert [features[key].direct for key in "abc"] == [True, True, False]
    assert features["c"].recommendation_rank == 7
    assert features["a"].recommendation_rank is None


def test_reference_context_excludes_the_candidate_and_needs_five_references() -> None:
    """ctx_ref_top5 averages the five closest other references, else is absent."""
    references = [
        DiscoveryCandidate(
            f"r{index}", _unit(1.0, float(index), 0.0), is_reference=True
        )
        for index in range(5)
    ]
    outsider = DiscoveryCandidate("x", _unit(1.0, 0.0, 1.0))

    five = compute_features(
        [*references, outsider], seed_document=SEED_DOCUMENT, seed_query=SEED_QUERY
    )
    # With exactly five references, a reference sees only four others.
    assert all(five[f"r{index}"].ctx_ref_top5 is None for index in range(5))
    expected = np.mean([_unit(1.0, 0.0, 1.0) @ ref.vector for ref in references])
    assert five["x"].ctx_ref_top5 == pytest.approx(expected)

    sixth = DiscoveryCandidate("r5", _unit(1.0, 5.0, 0.0), is_reference=True)
    six = compute_features(
        [*references, sixth, outsider],
        seed_document=SEED_DOCUMENT,
        seed_query=SEED_QUERY,
    )
    # r0 compares with r1..r5, never with its own cosine of 1.0.
    others = [ref.vector for ref in [*references[1:], sixth]]
    assert six["r0"].ctx_ref_top5 == pytest.approx(
        np.mean([references[0].vector @ vector for vector in others])
    )
    top_five = sorted(
        (_unit(1.0, 0.0, 1.0) @ ref.vector for ref in [*references, sixth]),
        reverse=True,
    )[:5]
    assert six["x"].ctx_ref_top5 == pytest.approx(np.mean(top_five))


def test_fewer_than_five_references_leave_the_context_list_empty() -> None:
    """Below five references nobody enters the reference-context list."""
    candidates = [
        DiscoveryCandidate(
            f"r{index}", _unit(1.0, float(index), 0.0), is_reference=True
        )
        for index in range(4)
    ] + [DiscoveryCandidate("x", _unit(1.0, 0.0, 1.0))]
    lists = fused_lists(
        compute_features(candidates, seed_document=SEED_DOCUMENT, seed_query=SEED_QUERY)
    )
    assert lists["reference_context"] == []
    assert lists["direct"] == ["r0", "r1", "r2", "r3"]


def test_rank_candidates_matches_hand_computed_scores() -> None:
    """Each score is the ordered sum over the lists holding the candidate."""
    candidates = [
        DiscoveryCandidate("cited", _unit(0.9, 0.1, 0.4), is_reference=True),
        DiscoveryCandidate("citer", _unit(0.6, 0.0, 0.8), cites_seed=True),
        DiscoveryCandidate("close", _unit(0.95, 0.0, 0.3)),
        DiscoveryCandidate("recommended", _unit(0.1, 0.0, 1.0), recommendation_rank=3),
    ]

    ranked = rank_candidates(
        candidates, seed_document=SEED_DOCUMENT, seed_query=SEED_QUERY
    )

    # semantic: close (0.95), cited (0.9), citer (0.6), recommended (0.1)
    # direct: cited, citer; recommended: recommended; reference_context: empty
    expected = {
        "close": _rrf(1),
        "cited": _rrf(2, 1),
        "citer": _rrf(3, 2),
        "recommended": _rrf(4, 1),
    }
    assert [item.paper_id for item in ranked] == [
        "cited",
        "recommended",
        "citer",
        "close",
    ]
    assert {item.paper_id: item.score for item in ranked} == expected
    assert [item.rank for item in ranked] == [1, 2, 3, 4]
    by_id = {item.paper_id: item for item in ranked}
    assert by_id["cited"].list_ranks == {"semantic": 2, "direct": 1}
    assert by_id["recommended"].list_ranks == {"semantic": 4, "recommended": 1}
    assert by_id["close"].list_ranks == {"semantic": 1}


def test_ties_break_by_query_cosine_then_paper_id() -> None:
    """semantic ties use cos_q then ID; direct and fused ties use ID only."""
    candidates = [
        DiscoveryCandidate("b", _unit(1.0, 0.5, 0.0), is_reference=True),
        DiscoveryCandidate("a", _unit(1.0, 0.0, 0.5), is_reference=True),
        DiscoveryCandidate("c", _unit(1.0, 0.5, 0.0)),
    ]
    lists = fused_lists(
        compute_features(candidates, seed_document=SEED_DOCUMENT, seed_query=SEED_QUERY)
    )
    # Equal cos_dd for all three: b and c beat a on cos_q; b beats c on ID.
    assert lists["semantic"] == ["b", "c", "a"]
    # The direct list ignores cos_q.
    assert lists["direct"] == ["a", "b"]

    twins = rank_candidates(
        [
            DiscoveryCandidate("y", SEED_DOCUMENT),
            DiscoveryCandidate("x", SEED_DOCUMENT),
        ],
        seed_document=SEED_DOCUMENT,
        seed_query=SEED_QUERY,
    )
    assert [item.paper_id for item in twins] == ["x", "y"]


def test_a_recommended_paper_outranks_an_identical_unrecommended_one() -> None:
    """Provider recommendation is evidence the embedding lists cannot see."""
    vector = _unit(0.8, 0.2, 0.3)
    ranked = rank_candidates(
        [
            DiscoveryCandidate("a-not-recommended", vector),
            DiscoveryCandidate("b-recommended", vector, recommendation_rank=40),
        ],
        seed_document=SEED_DOCUMENT,
        seed_query=SEED_QUERY,
    )
    assert [item.paper_id for item in ranked] == [
        "b-recommended",
        "a-not-recommended",
    ]
    assert ranked[0].score == _rrf(2, 1)
    assert ranked[1].score == _rrf(1)


def test_without_recommendations_the_fusion_is_the_three_other_lists() -> None:
    """arxiv-corpus pools carry no recommendation ranks, so only three lists fuse."""
    rng = np.random.default_rng(7)
    candidates = [
        DiscoveryCandidate(
            f"p{index:02d}",
            rng.normal(size=3),
            is_reference=index % 3 == 0,
            cites_seed=index % 5 == 0,
        )
        for index in range(30)
    ]
    features = compute_features(
        candidates, seed_document=SEED_DOCUMENT, seed_query=SEED_QUERY
    )
    lists = fused_lists(features)
    assert lists["recommended"] == []
    three = {name: lists[name] for name in ("semantic", "reference_context", "direct")}
    ranked = rank_candidates(
        candidates, seed_document=SEED_DOCUMENT, seed_query=SEED_QUERY
    )
    assert [(item.paper_id, item.score) for item in ranked] == (
        reciprocal_rank_fusion(three)
    )


def test_ranking_ignores_input_order_and_vector_scale() -> None:
    """Shuffled inputs give identical floats; rescaled vectors the same order."""
    rng = np.random.default_rng(11)
    candidates = [
        DiscoveryCandidate(
            f"p{index:03d}",
            rng.normal(size=8),
            is_reference=index % 4 == 0,
            cites_seed=index % 7 == 0,
            recommendation_rank=index if index % 3 == 0 else None,
        )
        for index in range(120)
    ]
    seed_document = rng.normal(size=8)
    seed_query = rng.normal(size=8)
    baseline = rank_candidates(
        candidates, seed_document=seed_document, seed_query=seed_query
    )
    assert baseline[0].features.ctx_ref_top5 is not None

    shuffled = list(candidates)
    random.Random(3).shuffle(shuffled)
    assert (
        rank_candidates(shuffled, seed_document=seed_document, seed_query=seed_query)
        == baseline
    )

    rescaled = [
        DiscoveryCandidate(
            item.paper_id,
            item.vector * (0.01 + index),
            is_reference=item.is_reference,
            cites_seed=item.cites_seed,
            recommendation_rank=item.recommendation_rank,
        )
        for index, item in enumerate(candidates)
    ]
    scaled = rank_candidates(
        rescaled, seed_document=seed_document * 9.0, seed_query=seed_query * 0.5
    )
    assert [item.paper_id for item in scaled] == [item.paper_id for item in baseline]
    assert [item.features.cos_dd for item in scaled] == pytest.approx(
        [item.features.cos_dd for item in baseline]
    )


def test_an_empty_pool_ranks_nothing() -> None:
    """No candidates means no ranked output."""
    assert rank_candidates([], seed_document=SEED_DOCUMENT, seed_query=SEED_QUERY) == []
