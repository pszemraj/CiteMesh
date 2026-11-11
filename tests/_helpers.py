"""Shared helpers for CiteMesh tests."""

from __future__ import annotations

import hashlib
import io
import os
import re
import threading
from collections.abc import Callable, Sequence
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from citemesh.core import Paper
    from citemesh.strategies.hybrid import HybridGraphBuilder


def fail_stat_for(
    target_path: Path, error: OSError
) -> Callable[[Path], os.stat_result]:
    """Return a ``Path.stat`` wrapper that fails for one path only.

    :param Path target_path: Path whose metadata lookup should fail.
    :param OSError error: Exception raised for the selected path.
    :return Callable[[Path], os.stat_result]: Wrapper preserving other stat calls.
    """
    original_stat = Path.stat

    def fail_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        """Fail for the selected path and forward other stat requests.

        :param Path path: Path whose metadata is requested.
        :param bool follow_symlinks: Whether metadata follows symbolic links.
        :return os.stat_result: Metadata for paths outside the fault scope.
        """
        if path == target_path:
            raise error
        return original_stat(path, follow_symlinks=follow_symlinks)

    return fail_stat


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


class LengthRecordingEncodeModel:
    """Record batches and encode text length and batch-local index."""

    def __init__(self) -> None:
        """Initialize an empty batch log."""
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str], **kwargs: object) -> NDArray[np.float32]:
        """Log texts and return two-column FP32 rows; ignore encoding options."""
        del kwargs
        self.calls.append(list(texts))
        return np.asarray(
            [[float(len(text)), float(idx)] for idx, text in enumerate(texts)],
            dtype=np.float32,
        )


class BlockingEncodeModel:
    """Pause an encode call before returning fixed-width FP32 vectors."""

    def __init__(
        self,
        entered: threading.Event,
        release: threading.Event,
        dimension: int,
    ) -> None:
        """Store the synchronization events and output dimension.

        :param threading.Event entered: Event set when encoding begins.
        :param threading.Event release: Event that releases the paused encode.
        :param int dimension: Width of each returned embedding vector.
        """
        self.entered = entered
        self.release = release
        self.dimension = dimension

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        """Signal entry, wait for release, and return deterministic vectors.

        :param list[str] texts: Input texts.
        :param Any kwargs: Ignored encoder keyword arguments.
        :return np.ndarray: One all-ones FP32 vector per text.
        """
        del kwargs
        self.entered.set()
        assert self.release.wait(timeout=5), "encode was never released"
        return np.ones((len(texts), self.dimension), dtype=np.float32)


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

    def __init__(self, lookup: dict[str, np.ndarray]) -> None:
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


def word_hash_vectors(texts: Sequence[str], dimension: int = 16) -> NDArray[np.float32]:
    """Embed texts as sums of hashed word directions, one unit row per text.

    Texts sharing words get close vectors, independently of batch composition.

    :param Sequence[str] texts: Input texts.
    :param int dimension: Vector width.
    :return NDArray[np.float32]: ``(len(texts), dimension)`` unit rows.
    """
    rows = np.zeros((len(texts), dimension), dtype=np.float64)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9]+", text.lower()) or ["empty"]:
            seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
            rows[row] += np.random.default_rng(seed).normal(size=dimension)
        rows[row] /= np.linalg.norm(rows[row])
    return rows.astype(np.float32)


class WordHashEncodeModel:
    """Encode model built on :func:`word_hash_vectors`, logging every text."""

    def __init__(self) -> None:
        """Start with an empty text log."""
        self.texts: list[str] = []

    def encode(self, texts: Sequence[str], **kwargs: object) -> NDArray[np.float32]:
        """Embed texts; ignore encoding options.

        :param Sequence[str] texts: Formatted model inputs.
        :param object kwargs: Ignored encoder options.
        :return NDArray[np.float32]: One unit row per text.
        """
        del kwargs
        self.texts.extend(texts)
        return word_hash_vectors(texts)


def stub_hybrid_corpus_operation_lock(builder: HybridGraphBuilder) -> None:
    """Bind a no-I/O corpus lock for hybrid orchestration unit tests.

    :param HybridGraphBuilder builder: Hybrid builder whose embedding child is active.
    :return None: Replaces artifact binding with a reentrant no-op lock fixture.
    """
    cache = MagicMock()
    cache.hydration_operation_lock.return_value = nullcontext()
    builder.embedding_builder.prepare_embedding_cache = MagicMock(return_value=cache)


def stub_hybrid_corpus(
    builder: HybridGraphBuilder,
    *,
    seed: Paper | None = None,
    neighbours: list[Paper] | None = None,
) -> MagicMock:
    """Serve a hybrid builder's prepared corpus from memory.

    Provider records match no corpus row and every member is encoded in memory.

    :param HybridGraphBuilder builder: Corpus-mode hybrid builder.
    :param Paper | None seed: Corpus seed returned by seed resolution.
    :param list[Paper] | None neighbours: Rows returned by each corpus search.
    :return MagicMock: The corpus search stub.
    """
    stub_hybrid_corpus_operation_lock(builder)
    embedding = builder.embedding_builder
    embedding._prepare_corpus_for_build = MagicMock()
    embedding.resolve_cached_corpus_seed = MagicMock(return_value=seed)
    embedding.cached_reference_metadata = MagicMock(return_value={})
    embedding.cached_bibliography_metadata = MagicMock(return_value={})
    embedding.cached_corpus_vectors = MagicMock(return_value={})
    embedding.search_prepared_corpus = MagicMock(return_value=list(neighbours or []))
    embedding._enrich_corpus_papers = MagicMock()
    embedding.enrich_cached_corpus_seed = MagicMock()
    return embedding.search_prepared_corpus


def offline_hybrid_encoder(
    monkeypatch: pytest.MonkeyPatch, builder: HybridGraphBuilder
) -> list[str]:
    """Encode a hybrid builder's texts offline with :func:`word_hash_vectors`.

    :param pytest.MonkeyPatch monkeypatch: Runtime boundary patch fixture.
    :param HybridGraphBuilder builder: Builder whose embedding child is patched.
    :return list[str]: Log of every encoded text, in call order.
    """
    texts: list[str] = []

    def encode(
        batch: list[str], batch_size: int | None = None, show_progress_bar: bool = False
    ) -> NDArray[np.float32]:
        del batch_size, show_progress_bar
        texts.extend(batch)
        return word_hash_vectors(batch)

    monkeypatch.setattr(builder.embedding_builder, "_load_model", lambda: None)
    monkeypatch.setattr(builder.embedding_builder, "_encode_texts", encode)
    return texts


def get_paper_id_normalization_cases() -> list[tuple[str, str]]:
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
        ("doi:10.1109/CVPR.2016.90", "10.1109/cvpr.2016.90"),
        (
            "https://doi.org/10.1007/978-3-642-11745-9_11",
            "10.1007/978-3-642-11745-9_11",
        ),
        ("DOI:10.18653/V1/N18-3011", "10.18653/v1/n18-3011"),
        ("https://doi.org/10.1002/example;2-A", "10.1002/example;2-a"),
        ("doi.org/10.1002/example;2-A", "10.1002/example;2-a"),
        ("https://doi.org/10.1002/example%3B2-A", "10.1002/example;2-a"),
        ("https://doi.org/10.1002/example/", "10.1002/example"),
        ("dx.doi.org/10.1002/example/", "10.1002/example"),
        ("https://doi.org/10.1002/example%2F", "10.1002/example"),
        (
            "https://doi.org/10.48550/arXiv.1706.03762/",
            "10.48550/arxiv.1706.03762",
        ),
    ]


def disable_embedding_dep_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable embedding optional dependency checks for strategy tests.

    :param pytest.MonkeyPatch monkeypatch: Monkeypatch fixture.
    :return None: Patches the embedding ``deps`` guard and hybrid's own import.
    """
    from citemesh.strategies import hybrid as hybrid_strategy
    from citemesh.strategies.embedding import deps as embedding_deps

    monkeypatch.setattr(embedding_deps, "_check_embedding_deps", lambda *a, **k: None)
    monkeypatch.setattr(hybrid_strategy, "_check_embedding_deps", lambda *a, **k: None)


def raise_import_error(*_args: object, **_kwargs: object) -> Any:
    """Raise ``ImportError`` for optional dependency contract tests."""
    raise ImportError("optional dependency unavailable")


def run_captured_cli(entrypoint: Callable[[], int | None]) -> SimpleNamespace:
    """Run a CLI entrypoint and capture stdout/stderr.

    :param Callable[[], int | None] entrypoint: In-process command invocation.
    :return SimpleNamespace: Exit status and captured output streams.
    """
    stdout = io.StringIO()
    stderr = io.StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            returncode = entrypoint()
        except SystemExit as exc:
            code = exc.code
            if isinstance(code, int):
                returncode = code
            elif code is None:
                returncode = 0
            else:
                returncode = 1

    return SimpleNamespace(
        returncode=returncode,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


class PausedFirstWrite:
    """Coordinate overlapping callers while retaining the real writer behavior."""

    def __init__(self, writer: Callable[..., None]) -> None:
        """Wrap a writer and expose events used by concurrent-update tests.

        :param Callable[..., None] writer: Real file writer to call after release.
        :return None: Initializes independent coordination state.
        """
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.second_started = threading.Event()
        self._lock = threading.Lock()
        self._calls = 0
        self._writer = writer

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Pause the first write and signal any later write attempts.

        :param Any args: Original writer arguments.
        :param Any kwargs: Original writer keyword arguments.
        :return None: Calls the real writer when allowed to proceed.
        """
        with self._lock:
            self._calls += 1
            call_number = self._calls
        if call_number == 1:
            self.first_started.set()
            assert self.release_first.wait(timeout=5), "first write never released"
        else:
            self.second_started.set()
        self._writer(*args, **kwargs)
