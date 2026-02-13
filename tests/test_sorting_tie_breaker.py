"""Regression tests for deterministic ordering and total comparisons."""

from __future__ import annotations

from citemesh.strategies.base import deterministic_sort_key, select_capped_undirected_edges


class _Tagged:
    """Hashable identifier with stable string representation."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __hash__(self) -> int:
        return hash(self.label)

    def __str__(self) -> str:
        return self.label


def test_select_capped_undirected_edges_is_deterministic_with_mixed_id_types() -> None:
    """Mixed node-id types should still sort using total-order key tuples."""
    edges = [
        (1, 2, {"weight": 0.2}),
        (_Tagged("alpha"), _Tagged("beta"), {"weight": 0.2}),
        ("2", 1, {"weight": 0.2}),
        (_Tagged("01"), "01", {"weight": 0.2}),
    ]

    first = select_capped_undirected_edges(edges, max_edges_per_node=10)
    second = select_capped_undirected_edges(edges, max_edges_per_node=10)
    assert first == second


def test_deterministic_sort_key_is_total_for_secondary_and_stable_fields() -> None:
    """Tuple sort keys should be total across mixed secondary key types."""
    keys = [
        deterministic_sort_key(0.42, 10, secondary_id=_Tagged("z"), stable_index=1),
        deterministic_sort_key(0.42, "10", secondary_id="a", stable_index=0),
        deterministic_sort_key(0.42, _Tagged("10"), secondary_id="a", stable_index=2),
        deterministic_sort_key(0.42, "10", secondary_id=_Tagged("z"), stable_index=1),
    ]

    ordered = sorted(keys)
    assert ordered[0][0] == ordered[1][0] == -0.42
    assert len(ordered) == len(keys)
