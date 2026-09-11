"""Lazy third-party imports and NetworkX writer capability detection."""

from __future__ import annotations

import re
from typing import Any

import networkx as nx

GRAPHML_DETERMINISM_POLICY_STRICT = "strict_sorted_nodes_edges"
GRAPHML_DETERMINISM_POLICY_BEST_EFFORT = "best_effort_sorted_nodes_edges"
_GRAPHML_BEST_EFFORT_MIN_VERSION = (2, 8)


def _load_pyvis_network_class() -> Any:
    """Import and return the PyVis ``Network`` class.

    :return Any: Imported ``pyvis.network.Network`` class.
    """
    from pyvis.network import Network

    return Network


def _load_plotly_graph_objects() -> Any:
    """Import and return Plotly graph objects.

    :return Any: Imported ``plotly.graph_objects`` module proxy.
    """
    from plotly import graph_objects as go

    return go


def _load_plotly_dashboard_runtime() -> tuple[Any, Any]:
    """Import and return Plotly graph objects plus inline JS provider.

    :return tuple[Any, Any]: Plotly graph objects and ``get_plotlyjs`` callable.
    """
    from plotly.offline import get_plotlyjs

    return _load_plotly_graph_objects(), get_plotlyjs


def _graphml_determinism_policy() -> str:
    """Return determinism policy name for active NetworkX writer runtime.

    :return str: Determinism policy identifier for current NetworkX version.
    """
    major_minor = tuple(int(part) for part in re.findall(r"\d+", nx.__version__)[:2])
    if len(major_minor) < 2:
        return GRAPHML_DETERMINISM_POLICY_BEST_EFFORT

    major, minor = major_minor[0], major_minor[1]
    if (major, minor) >= _GRAPHML_BEST_EFFORT_MIN_VERSION:
        return GRAPHML_DETERMINISM_POLICY_STRICT
    return GRAPHML_DETERMINISM_POLICY_BEST_EFFORT
