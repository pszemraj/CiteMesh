"""Shared helpers for consolidated CiteMesh tests."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

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

    monkeypatch.setattr(
        embedding_strategy, "_check_embedding_deps", lambda *a, **k: None
    )
    monkeypatch.setattr(hybrid_strategy, "_check_embedding_deps", lambda *a, **k: None)


def raise_import_error(*_args: object, **_kwargs: object) -> Any:
    """Raise ``ImportError`` for optional dependency contract tests."""
    raise ImportError("optional dependency unavailable")
