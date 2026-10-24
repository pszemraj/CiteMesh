"""Lazy third-party imports and NetworkX writer capability detection."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import networkx as nx

if TYPE_CHECKING:
    from plotly.graph_objects import Figure, Layout, Scatter
    from pyvis.network import Network


class PlotlyGraphObjects(Protocol):
    """Constructors used by the lazily imported Plotly graph_objects module."""

    Figure: type[Figure]
    Layout: type[Layout]
    Scatter: type[Scatter]


GRAPHML_DETERMINISM_POLICY_STRICT = "strict_sorted_nodes_edges"
GRAPHML_DETERMINISM_POLICY_BEST_EFFORT = "best_effort_sorted_nodes_edges"
_GRAPHML_BEST_EFFORT_MIN_VERSION = (2, 8)


def _load_pyvis_network_class() -> type[Network]:
    """Import and return the PyVis ``Network`` class.

    :return type[Network]: Imported ``pyvis.network.Network`` class.
    """
    from pyvis.network import Network

    return Network


def _load_plotly_graph_objects() -> PlotlyGraphObjects:
    """Import and return Plotly graph objects.

    :return PlotlyGraphObjects: Imported ``plotly.graph_objects`` module.
    """
    from plotly import graph_objects as go

    return go


def _load_plotly_dashboard_runtime() -> tuple[PlotlyGraphObjects, Callable[[], str]]:
    """Import and return Plotly graph objects plus inline JS provider.

    :return tuple[PlotlyGraphObjects, Callable[[], str]]: Figure constructors and
        the Plotly.js source provider.
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
