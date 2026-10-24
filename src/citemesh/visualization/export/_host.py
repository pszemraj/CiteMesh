"""Structural state and helpers required by exporter mixins."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from citemesh.core.models import PaperGraph

    from ..dashboard.contracts import DashboardMetadata, EnrichedNodePayload
    from ..themes import Theme


class _ExporterHost(Protocol):
    """Exporter state and helpers consumed across mixin boundaries."""

    graph: PaperGraph
    seed_id: str
    metadata: Mapping[str, object]
    _layout: Mapping[Hashable, Iterable[float]] | None
    _size_map: dict[Hashable, float] | None
    _color_map_cache: dict[str, dict[Hashable, tuple[float, float, float]]]

    def _serialized_graph(self) -> PaperGraph: ...

    def _enriched_nodes(self) -> list[EnrichedNodePayload]: ...

    def _seed_relevance_scores(self) -> dict[str, float]: ...

    def _get_layout(self) -> dict[Hashable, NDArray[np.float64]]: ...

    def _node_size(self, node: Hashable) -> float: ...

    def _node_color_hex(self, node: Hashable, theme: Theme) -> str: ...

    def _plotly_year_scale(
        self, node_ids: list[Hashable]
    ) -> tuple[list[float], float, float]: ...

    def _dashboard_meta(
        self,
        *,
        theme_obj: Theme,
        node_ids: list[Hashable],
        node_payloads: list[EnrichedNodePayload],
        sorted_edges: list[tuple[Hashable, Hashable, dict[str, object]]],
        include_plotly_geometry: bool,
    ) -> DashboardMetadata: ...
