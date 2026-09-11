"""Deterministic ordering helpers shared by visualization/export code."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import networkx as nx


def ordered_nodes(graph: nx.Graph) -> list[Hashable]:
    """Return graph node IDs in canonical deterministic order.

    :param nx.Graph graph: Graph whose node ordering should be canonicalized.
    :return List[Hashable]: Sorted node IDs.
    """
    return sorted(graph.nodes(), key=str)


def ordered_edges_with_data(
    graph: nx.Graph,
) -> list[tuple[Hashable, Hashable, dict[str, Any]]]:
    """Return canonicalized edge tuples with deterministic endpoint ordering.

    :param nx.Graph graph: Graph whose edges should be canonicalized.
    :return List[Tuple[Hashable, Hashable, Dict[str, Any]]]: Sorted
        ``(u, v, attrs)`` tuples.
    """
    canonicalized: list[tuple[Hashable, Hashable, dict[str, Any]]] = []
    for left, right, attrs in graph.edges(data=True):
        edge_left, edge_right = (
            (left, right) if str(left) <= str(right) else (right, left)
        )
        canonicalized.append((edge_left, edge_right, dict(attrs)))

    return sorted(canonicalized, key=lambda item: (str(item[0]), str(item[1])))


def canonicalize_graph_for_layout(graph: nx.Graph) -> nx.Graph:
    """Create a graph copy with deterministic node/edge insertion ordering.

    :param nx.Graph graph: Source graph.
    :return nx.Graph: Canonicalized copy used as layout input.
    """
    canonical_graph = nx.Graph()

    for node in ordered_nodes(graph):
        canonical_graph.add_node(node, **dict(graph.nodes[node]))

    for left, right, attrs in ordered_edges_with_data(graph):
        canonical_graph.add_edge(left, right, **attrs)

    return canonical_graph
