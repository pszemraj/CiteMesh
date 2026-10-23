"""Node ordering, enrichment, sizing, and coloring for every export format."""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable, Iterable, Mapping
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from citemesh.core.values import coerce_citation_count, coerce_float, parse_float

from ..dashboard.contracts import EnrichedNodePayload
from ..node_data import effective_node_metadata as _serialize_node
from ..node_data import validate_canonical_node_ids
from ..ordering import ordered_edges_with_data, ordered_nodes
from ..render import (
    compute_layout,
    compute_node_colors,
    compute_node_sizes,
    normalize_layout_positions,
)
from ..themes import Theme
from ..years import coerce_publication_year
from ._host import _ExporterHost
from .bibtex import _node_bibtex
from .geometry import _rgb_tuple_to_hex
from .links import _derive_links

if TYPE_CHECKING:
    from citemesh.core.models import PaperGraph

logger = logging.getLogger(__name__)


def _ordered_attrs(attrs: Mapping[str, object]) -> dict[str, object]:
    """Return a copy of mapping with deterministic key ordering.

    :param Mapping[str, object] attrs: Source attribute mapping.
    :return Dict[str, object]: Copy with key order normalized by key string.
    """
    return {key: attrs[key] for key in sorted(attrs, key=str)}


def _normalize_strategy_token(raw_strategy: object) -> str:
    """Normalize strategy tokens to the exporter-supported vocabulary.

    :param object raw_strategy: Candidate strategy token.
    :return str: Normalized strategy token or an empty string.
    """
    normalized = str(raw_strategy or "").strip().lower()
    if normalized in {"citation", "recommendation", "embedding", "hybrid"}:
        return normalized
    return ""


def _normalized_edge_weight(raw_weight: object) -> float:
    """Normalize edge weights for relevance computation.

    :param object raw_weight: Raw edge weight candidate.
    :return float: Positive finite weight.
    """
    parsed = coerce_float(raw_weight, 0.0)
    if not math.isfinite(parsed) or parsed <= 0.0:
        return 1e-6
    return parsed


def _normalized_text_values(raw_values: object) -> list[str]:
    """Normalize non-empty items from a graph attribute iterable.

    :param object raw_values: Raw author or category values.
    :return list[str]: Stripped, non-empty string values.
    :raises TypeError: If a truthy value is not iterable.
    """
    if not raw_values:
        return []
    if not isinstance(raw_values, Iterable):
        raise TypeError(f"'{type(raw_values).__name__}' object is not iterable")
    return [str(value).strip() for value in raw_values if str(value).strip()]


def _node_title(attrs: Mapping[str, object], node_id: Hashable) -> str:
    """Return stable node title for edge-sidecar export fields.

    :param Mapping[str, object] attrs: Node attributes map.
    :param Hashable node_id: Node identifier fallback.
    :return str: Human-readable title fallback.
    """
    serialized = _serialize_node(node_id, attrs)
    title = str(serialized.get("title") or "").strip()
    if title:
        return title
    return str(node_id)


def _node_short_label(attrs: Mapping[str, object], node_id: Hashable) -> str:
    """Return compact node label for edge export fields.

    :param Mapping[str, object] attrs: Node attributes map.
    :param Hashable node_id: Node identifier fallback.
    :return str: Compact label (author/year or title fallback).
    """
    serialized = _serialize_node(node_id, attrs)
    title = str(serialized.get("title") or "").strip() or str(node_id)
    raw_authors = serialized.get("authors", [])
    surname = ""
    if isinstance(raw_authors, list) and raw_authors:
        first_author = str(raw_authors[0]).strip()
        surname = first_author.split()[-1] if first_author else ""

    year = coerce_publication_year(serialized.get("year"))
    if surname and year > 0:
        return f"{surname}, {year}"
    if year > 0:
        return f"{title} ({year})"
    return title


def _sorted_nodes(
    graph: PaperGraph,
) -> list[tuple[Hashable, dict[str, object]]]:
    """Return nodes sorted by ID for deterministic serialization.

    :param PaperGraph graph: Graph whose nodes should be ordered.
    :return list[tuple[Hashable, dict[str, object]]]: Sorted ``(node_id, attrs)``
        pairs.
    :raises ValueError: If an ID is empty, has surrounding whitespace, or collides
        with another ID after string conversion.
    """
    nodes = [(node_id, graph.nodes[node_id]) for node_id in ordered_nodes(graph)]
    validate_canonical_node_ids(node_id for node_id, _ in nodes)
    return nodes


def _sorted_edges(
    graph: PaperGraph,
) -> list[tuple[Hashable, Hashable, dict[str, object]]]:
    """Return undirected edges with canonical endpoints in stable order.

    :param PaperGraph graph: Graph whose edges should be ordered.
    :return list[tuple[Hashable, Hashable, dict[str, object]]]: Sorted edge tuples in
        ``(u, v, attrs)`` form.
    :raises ValueError: If an edge weight is null or non-finite.
    """
    edges = ordered_edges_with_data(graph)
    for left, right, attrs in edges:
        weight = attrs.get("weight", 0.0)
        if weight is None or not math.isfinite(parse_float(weight)):
            raise ValueError(
                f"Cannot export null or non-finite edge weight for {left!r} -> {right!r}."
            )
    return edges


def _default_provenance(*, strategy: str) -> str:
    """Resolve default provenance class for non-hybrid strategies.

    :param str strategy: Strategy metadata token.
    :return str: One of ``citation`` or ``semantic``.
    """
    if strategy in {"embedding", "recommendation"}:
        return "semantic"
    return "citation"


def _strategy(graph: PaperGraph, metadata: Mapping[str, object]) -> str:
    """Resolve effective strategy token for export metadata and enrichment.

    Explicit exporter metadata wins. When omitted, exporter falls back to graph
    metadata persisted by current builders.

    :param PaperGraph graph: Graph carrying builder-persisted metadata.
    :param Mapping[str, object] metadata: Exporter-level metadata mapping.
    :return str: Normalized strategy token when available.
    """
    metadata_strategy = _normalize_strategy_token(metadata.get("strategy"))
    if metadata_strategy:
        return metadata_strategy

    graph_strategy = _normalize_strategy_token(graph.graph.get("strategy"))
    if graph_strategy:
        return graph_strategy
    return ""


_PROVENANCE_CLASSES = frozenset({"citation", "semantic", "both"})
_SEED_RELATION_CLASSES = frozenset(
    {
        "seed",
        "referenced_by_seed",
        "cites_seed",
        "overlap",
        "semantic_only",
        "citation",
    }
)


def _graph_class_map(
    graph: PaperGraph, attribute: str, allowed: frozenset[str]
) -> dict[str, str]:
    """Read a graph-level node classification map, keeping known classes only.

    :param PaperGraph graph: Graph carrying the classification mapping.
    :param str attribute: Graph attribute holding the raw mapping.
    :param frozenset[str] allowed: Normalized class tokens to retain.
    :return Dict[str, str]: Node ID to normalized class, unknown values dropped.
    """
    raw_map = graph.graph.get(attribute)
    if not isinstance(raw_map, dict):
        return {}

    resolved: dict[str, str] = {}
    for raw_id, raw_value in raw_map.items():
        value = str(raw_value).strip().lower()
        if value in allowed:
            resolved[str(raw_id)] = value
    return resolved


def _provenance_map(graph: PaperGraph) -> dict[str, str]:
    """Resolve normalized per-node provenance map.

    :param PaperGraph graph: Graph carrying a ``paper_sources`` mapping.
    :return Dict[str, str]: Mapping from node ID to provenance class.
    """
    return _graph_class_map(graph, "paper_sources", _PROVENANCE_CLASSES)


def _seed_relation_map(graph: PaperGraph) -> dict[str, str]:
    """Resolve normalized relation-to-seed mapping when available.

    :param PaperGraph graph: Graph carrying a ``seed_relations`` mapping.
    :return Dict[str, str]: Node-ID to relation class mapping.
    """
    return _graph_class_map(graph, "seed_relations", _SEED_RELATION_CLASSES)


class NodesMixin:
    """Exporter-state node helpers: enrichment, layout, size, and color caches."""

    _layout: Mapping[Hashable, Iterable[float]] | None
    _size_map: dict[Hashable, float] | None
    _color_map_cache: dict[str, dict[Hashable, tuple[float, float, float]]]

    def _serialized_graph(self: _ExporterHost) -> PaperGraph:
        """Copy the graph with effective exported metadata on every node.

        :return PaperGraph: Shallow graph copy whose scalar node fields follow
            :func:`_serialize_node` without mutating the caller's graph.
        """
        serialized_graph = self.graph.copy()
        for node_id, attrs in _sorted_nodes(self.graph):
            serialized_graph.nodes[node_id].update(_serialize_node(node_id, attrs))
        return serialized_graph

    def _enriched_nodes(self: _ExporterHost) -> list[EnrichedNodePayload]:
        """Build enriched node payloads with provenance, relevance, links, and BibTeX.

        This is the canonical node enrichment used by JSON export, CSV export,
        BibTeX export, and the dashboard payload.

        :return list[EnrichedNodePayload]: Enriched node payloads.
        """
        provenance = _provenance_map(self.graph)
        seed_relations = _seed_relation_map(self.graph)
        relevance = self._seed_relevance_scores()
        sorted_nodes = _sorted_nodes(self.graph)
        strategy = _strategy(self.graph, self.metadata)

        node_payloads: list[EnrichedNodePayload] = []
        for node_id, attrs in sorted_nodes:
            node_str = str(node_id)
            serialized = _serialize_node(node_id, attrs)
            title = serialized.get("title", "")
            year = coerce_publication_year(serialized.get("year"))
            citation_count = coerce_citation_count(serialized.get("citation_count"))
            authors = _normalized_text_values(serialized.get("authors", []))
            venue = str(serialized.get("venue") or "").strip()
            arxiv_id = str(serialized.get("arxiv_id") or "").strip()
            doi = str(serialized.get("doi") or "").strip()
            categories = _normalized_text_values(serialized.get("categories", []))
            abstract = str(serialized.get("abstract") or "").strip()
            is_seed = bool(serialized.get("is_seed", False))
            is_local_corpus = bool(serialized.get("is_local_corpus", False))

            provenance_base = provenance.get(
                node_str, _default_provenance(strategy=strategy)
            )
            node_provenance = "seed" if is_seed else provenance_base
            seed_relation = seed_relations.get(node_str, "")
            if is_seed:
                seed_relation = "seed"
            if not seed_relation and node_provenance == "semantic":
                seed_relation = "semantic_only"
            node_payload: EnrichedNodePayload = {
                "id": node_str,
                "title": title,
                "year": year,
                "authors": authors,
                "citation_count": citation_count,
                "abstract": abstract,
                "venue": venue,
                "arxiv_id": arxiv_id,
                "doi": doi,
                "categories": categories,
                "is_seed": is_seed,
                "is_local_corpus": is_local_corpus,
                "provenance": node_provenance,
                "provenance_base": provenance_base,
                "seed_relation": seed_relation,
                "seed_relevance": float(relevance.get(node_str, 0.0)),
                "links": {},
                "bibtex": "",
            }
            links = _derive_links(node_str, node_payload=node_payload)
            node_payload["links"] = links
            node_payload["bibtex"] = _node_bibtex(node_payload, links=links)
            node_payloads.append(node_payload)
        return node_payloads

    def _seed_relevance_scores(self: _ExporterHost) -> dict[str, float]:
        """Compute seed-centric personalized PageRank scores.

        :return Dict[str, float]: Node-ID keyed relevance scores.
        """
        relevance_graph: nx.Graph[str, dict[str, object], dict[str, object]] = (
            nx.Graph()
        )
        for node_id, _ in _sorted_nodes(self.graph):
            relevance_graph.add_node(str(node_id))

        for left, right, attrs in _sorted_edges(self.graph):
            relevance_graph.add_edge(
                str(left),
                str(right),
                weight=_normalized_edge_weight(attrs.get("weight", 0.0)),
            )

        if not relevance_graph.nodes:
            return {}

        personalization: dict[str, float] = {
            str(node_id): 0.0 for node_id in relevance_graph.nodes
        }
        seed_id = str(self.seed_id)
        if seed_id in personalization:
            personalization[seed_id] = 1.0
        else:
            seed_id = next(iter(personalization))
            personalization[seed_id] = 1.0

        try:
            scores = nx.pagerank(
                relevance_graph,
                alpha=0.85,
                personalization=personalization,
                weight="weight",
            )
        except Exception as exc:  # pragma: no cover - highly unlikely fallback
            logger.warning(
                "Failed to compute personalized PageRank relevance; falling back to "
                "uniform relevance scores (%s)",
                exc,
            )
            uniform = 1.0 / float(len(personalization))
            return {node_id: uniform for node_id in personalization}

        return {str(node_id): float(score) for node_id, score in scores.items()}

    def _get_layout(self: _ExporterHost) -> dict[Hashable, NDArray[np.float64]]:
        """Compute or reuse cached graph layout.

        :return dict[Hashable, NDArray[np.float64]]: Node IDs mapped to coordinates.
        """
        layout = self._layout
        if layout is None:
            layout = compute_layout(self.graph)
        normalized = normalize_layout_positions(layout)
        self._layout = normalized
        return normalized

    def _node_size(self: _ExporterHost, node: Hashable) -> float:
        """Compute cached node size for a node ID.

        :param Hashable node: Graph node identifier.
        :return float: Cached node size.
        """
        size_map = self._size_map
        if size_map is None:
            sizes = compute_node_sizes(self._serialized_graph())
            ordered_nodes = [node_id for node_id, _ in _sorted_nodes(self.graph)]
            size_map = {
                graph_node: size for graph_node, size in zip(ordered_nodes, sizes)
            }
            self._size_map = size_map
        return float(size_map.get(node, 300.0))

    def _node_color_hex(self: _ExporterHost, node: Hashable, theme: Theme) -> str:
        """Convert computed node color to hex for export serializers.

        :param Hashable node: Graph node identifier.
        :param Theme theme: Theme to use.
        :return str: Hex color string.
        """
        cache_key = theme.name
        if cache_key not in self._color_map_cache:
            colors, _, _ = compute_node_colors(
                self._serialized_graph(), self.seed_id, theme
            )
            ordered_nodes = [node_id for node_id, _ in _sorted_nodes(self.graph)]
            self._color_map_cache[cache_key] = {
                graph_node: color for graph_node, color in zip(ordered_nodes, colors)
            }

        color = self._color_map_cache[cache_key].get(node, theme.node_color_new)
        return _rgb_tuple_to_hex(color)
