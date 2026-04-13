"""Shared helpers for consolidated CiteMesh tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import networkx as nx
import numpy as np
import pytest


def build_seed_graph(seed_id: str = "seed") -> nx.Graph:
    """Create a minimal one-node graph for CLI strategy tests.

    :param str seed_id: Seed-node identifier.
    :return nx.Graph: Graph containing only the seed node.
    """
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


class ConstantEncodeModel:
    """Deterministic encode model that repeats one embedding vector."""

    def __init__(self, vector: Sequence[float] = (1.0, 0.0)) -> None:
        """Initialize constant encode model with a fixed output vector.

        :param Sequence[float] vector: Embedding vector repeated for each text.
        """
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        """Return the configured embedding vector for each input text.

        :param list[str] texts: Input texts.
        :param Any kwargs: Ignored keyword arguments.
        :return np.ndarray: Repeated float32 embedding matrix.
        """
        del kwargs
        if not texts:
            return np.empty((0, int(self.vector.shape[0])), dtype=np.float32)
        return np.repeat(self.vector[np.newaxis, :], len(texts), axis=0)


class SeededRandomEncodeModel:
    """Deterministic pseudo-random encode model keyed by batch size."""

    def __init__(self, embedding_dim: int = 2) -> None:
        """Initialize seeded-random encode model.

        :param int embedding_dim: Number of embedding dimensions.
        """
        self.embedding_dim = embedding_dim
        self.encode_calls = 0

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        """Generate deterministic pseudo-random embeddings for a batch.

        :param list[str] texts: Input texts for encoding.
        :param Any kwargs: Ignored keyword arguments.
        :return np.ndarray: Deterministic embeddings.
        """
        del kwargs
        self.encode_calls += 1
        np.random.seed(len(texts))
        return np.random.rand(len(texts), self.embedding_dim)


class LookupEncodeModel:
    """Deterministic encode model backed by a text-to-embedding lookup."""

    def __init__(self, lookup: Dict[str, np.ndarray]) -> None:
        """Store lookup table used by encode calls.

        :param Dict[str, np.ndarray] lookup: Text-to-embedding lookup table.
        """
        self.lookup = lookup

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        """Return embeddings for texts in input order.

        :param list[str] texts: Input texts.
        :param Any kwargs: Ignored keyword arguments.
        :return np.ndarray: Float32 embedding matrix.
        """
        del kwargs
        return np.asarray([self.lookup[text] for text in texts], dtype=np.float32)


def build_fake_strategy_builder_factory(
    captured_kwargs: Dict[str, Any],
    *,
    graph: nx.Graph | None = None,
    seed_id: str = "seed",
) -> type:
    """Build fake strategy class that captures constructor kwargs.

    :param Dict[str, Any] captured_kwargs: Sink for constructor kwargs.
    :param nx.Graph | None graph: Optional graph returned by fake builder.
    :param str seed_id: Seed ID returned alongside graph.
    :return type: Fake strategy builder class.
    """

    base_graph = graph if graph is not None else build_seed_graph(seed_id)

    class _FakeStrategyBuilder:
        """Fake strategy builder used by CLI tests."""

        def __init__(self, **kwargs: Any) -> None:
            """Capture provided constructor kwargs.

            :param Any kwargs: Strategy constructor keyword arguments.
            :return None: This initializer stores kwargs for assertions.
            """
            captured_kwargs.update(kwargs)

        def build_graph(self, _: str) -> tuple[nx.Graph, str]:
            """Return configured graph and seed ID.

            :param str _: Ignored input seed identifier.
            :return tuple[nx.Graph, str]: Graph and normalized seed ID.
            """
            return base_graph, seed_id

    return _FakeStrategyBuilder


def build_fake_exporter_factory(
    captured_data: Dict[str, Any], *, methods: Iterable[str] | None = None
) -> type:
    """Build fake exporter class that captures metadata/layout for assertions.

    :param Dict[str, Any] captured_data: Sink for exporter constructor payload.
    :param Iterable[str] | None methods: Method names to enable on fake exporter.
    :return type: Fake exporter class.
    """

    from citemesh.cli import _EXPORTER_METHOD

    requested_methods = set(methods or _EXPORTER_METHOD.values())
    payloads = {
        "to_json": "{}",
        "to_interactive_html": "<html/>",
        "to_plotly_html": "<html/>",
        "to_dashboard_html": "<html/>",
        "to_graphml": "<graphml/>",
        "to_csv": "id,title\n",
        "to_bibtex": "@article{test,}\n",
    }

    class _FakeExporter:
        """Fake graph exporter used by CLI tests."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Capture exporter constructor metadata for assertions.

            :param Any args: Positional constructor arguments.
            :param Any kwargs: Keyword constructor arguments.
            :return None: This initializer stores metadata only.
            """
            del args
            captured_data["kwargs"] = kwargs
            captured_data["metadata"] = kwargs.get("metadata")
            captured_data["layout"] = kwargs.get("layout")

    def _write_payload(
        self: object, path: Path, method_name: str, *_args: object, **_kwargs: object
    ) -> None:
        """Write the minimal artifact body for the requested fake exporter method."""
        del self, _args, _kwargs
        if method_name in requested_methods:
            path.write_text(payloads[method_name])

    for method_name in payloads:
        setattr(
            _FakeExporter,
            method_name,
            lambda self, path, *args, _method=method_name, **kwargs: _write_payload(
                self,
                path,
                _method,
                *args,
                **kwargs,
            ),
        )

    return _FakeExporter


def get_paper_id_normalization_cases() -> List[tuple[str, str]]:
    """Return shared paper-id normalization fixtures.

    :return List[tuple[str, str]]: Pairs of raw ID and expected normalized ID.
    """
    return [
        ("https://arxiv.org/abs/2508.14040", "arxiv:2508.14040"),
        ("https://arxiv.org/pdf/2508.14040.pdf", "arxiv:2508.14040"),
        ("arXiv:2508.14040", "arxiv:2508.14040"),
        ("arXiv:1706.03762v5", "arxiv:1706.03762"),
        ("https://arxiv.org/abs/1706.03762v5", "arxiv:1706.03762"),
        ("https://arxiv.org/abs/arXiv:1706.03762v5", "arxiv:1706.03762"),
        ("https://arxiv.org/pdf/1706.03762v5.pdf", "arxiv:1706.03762"),
        ("https://doi.org/10.1145/3133956.3134029", "10.1145/3133956.3134029"),
    ]


def disable_embedding_dep_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable embedding optional dependency checks for strategy tests.

    :param pytest.MonkeyPatch monkeypatch: Monkeypatch fixture.
    :return None: Patches dependency guards in embedding/hybrid strategy modules.
    """
    from citemesh.strategies import embedding as embedding_strategy
    from citemesh.strategies import hybrid as hybrid_strategy

    monkeypatch.setattr(embedding_strategy, "_check_embedding_deps", lambda: None)
    monkeypatch.setattr(hybrid_strategy, "_check_embedding_deps", lambda: None)


def raise_import_error(*_args: object, **_kwargs: object) -> Any:
    """Raise ``ImportError`` for optional dependency contract tests."""
    raise ImportError("optional dependency unavailable")
