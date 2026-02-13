"""Shared test helpers for CiteMesh test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import networkx as nx

from citemesh.core import Paper


def build_seed_graph(seed_id: str = "seed") -> nx.Graph:
    """Create a minimal one-node graph for CLI strategy tests."""
    graph = nx.Graph()
    graph.add_node(
        seed_id,
        title="Seed",
        year=2020,
        authors=[],
        citation_count=0,
        is_seed=True,
    )
    return graph


def build_top_k_papers() -> Dict[str, Paper]:
    """Create a four-paper fixture used by degree capping tests."""
    papers = {
        "seed": Paper(paper_id="seed", title="Seed", year=2024, abstract="seed"),
        "a": Paper(paper_id="a", title="A", year=2024, abstract="alpha"),
        "b": Paper(paper_id="b", title="B", year=2024, abstract="beta"),
        "c": Paper(paper_id="c", title="C", year=2024, abstract="gamma"),
    }
    papers["seed"].is_seed = True
    return papers


def build_fake_strategy_builder_factory(
    captured_kwargs: Dict[str, Any],
    *,
    graph: nx.Graph | None = None,
    seed_id: str = "seed",
) -> type:
    """Build a fake strategy class that captures ctor kwargs."""

    base_graph = graph if graph is not None else build_seed_graph(seed_id)

    class _FakeStrategyBuilder:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        def build_graph(self, _: str) -> tuple[nx.Graph, str]:
            return base_graph, seed_id

    return _FakeStrategyBuilder


def build_fake_exporter_factory(
    captured_data: Dict[str, Any], *, methods: Iterable[str] | None = None
) -> type:
    """Build a fake exporter class that captures metadata/layout for assertions."""

    requested_methods = set(methods or [])

    class _FakeExporter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured_data["kwargs"] = kwargs
            captured_data["metadata"] = kwargs.get("metadata")
            captured_data["layout"] = kwargs.get("layout")

        def to_json(self, path: Path) -> None:
            if "to_json" in requested_methods or not requested_methods:
                path.write_text("{}")

        def to_interactive_html(self, path: Path, *_, **__) -> None:
            if "to_interactive_html" in requested_methods or not requested_methods:
                path.write_text("<html/>")

        def to_plotly_html(self, path: Path, *_, **__) -> None:
            if "to_plotly_html" in requested_methods or not requested_methods:
                path.write_text("<html/>")

        def to_graphml(self, path: Path) -> None:
            if "to_graphml" in requested_methods or not requested_methods:
                path.write_text("<graphml/>")

    return _FakeExporter


def get_paper_id_normalization_cases() -> List[tuple[str, str]]:
    """Canonicalization scenarios reused across CLI and service tests."""
    return [
        ("https://arxiv.org/abs/2508.14040", "arxiv:2508.14040"),
        ("https://arxiv.org/pdf/2508.14040.pdf", "arxiv:2508.14040"),
        ("arXiv:2508.14040", "arxiv:2508.14040"),
        ("arXiv:1706.03762v5", "arxiv:1706.03762"),
        ("https://arxiv.org/abs/1706.03762v5", "arxiv:1706.03762"),
        ("https://arxiv.org/pdf/1706.03762v5.pdf", "arxiv:1706.03762"),
        ("https://doi.org/10.1145/3133956.3134029", "10.1145/3133956.3134029"),
    ]
