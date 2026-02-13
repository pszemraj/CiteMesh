"""Deterministic ordering helpers shared by visualization/export code."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx


def ordered_nodes(graph: nx.Graph) -> List[str]:
    """Return graph node IDs in canonical deterministic order.

    :param nx.Graph graph: Graph whose node ordering should be canonicalized.
    :return List[str]: Sorted node IDs.
    """
    return sorted(graph.nodes(), key=str)


def ordered_edges_with_data(graph: nx.Graph) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return canonicalized edge tuples with deterministic endpoint ordering.

    :param nx.Graph graph: Graph whose edges should be canonicalized.
    :return List[Tuple[str, str, Dict[str, Any]]]: Sorted ``(u, v, attrs)`` tuples.
    """
    canonicalized = []
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
