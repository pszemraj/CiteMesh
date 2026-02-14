"""Tests for embedding cache hit/miss behavior."""

import multiprocessing as mp
import sqlite3
import tempfile
from pathlib import Path
from queue import Empty
from typing import Any

import h5py
import numpy as np

from citemesh.data.embedding_cache import EmbeddingCache


class _MockModel:
    """Deterministic embedding model mock."""

    def __init__(self) -> None:
        """Initialize deterministic mock model."""
        self.encode_calls = 0

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        """Generate deterministic pseudo-embeddings.

        :param list[str] texts: Input texts for encoding.
        :param kwargs: Extra arguments ignored by the mock.
        :return np.ndarray: Deterministic embeddings shaped ``(len(texts), 2)``.
        """
        self.encode_calls += 1
        np.random.seed(len(texts))
        return np.random.rand(len(texts), 2)


def _multiprocess_cache_worker(
    cache_dir: str, worker_idx: int, queue: mp.Queue
) -> None:
    """Write embeddings in subprocess and report success/failure via queue.

    :param str cache_dir: Cache directory shared by workers.
    :param int worker_idx: Worker index used to create unique paper IDs.
    :param mp.Queue queue: Multiprocessing queue receiving status tuples.
    :return None: Worker reports status via queue.
    """
    try:
        cache = EmbeddingCache(cache_dir=cache_dir, model_name="process-lock-test")
        model = _MockModel()
        papers = {
            f"p{worker_idx}_{offset}": {
                "title": f"Title {offset}",
                "abstract": f"Abstract {offset}",
            }
            for offset in range(100)
        }
        cache.get_embeddings(papers, model, show_progress=False)
        queue.put(("ok", worker_idx))
    except Exception as exc:  # pragma: no cover - subprocess path
        queue.put(("err", worker_idx, repr(exc)))


def test_embedding_cache_returns_cached_vectors() -> None:
    """Repeated lookup should avoid re-encoding unchanged records."""
    model = _MockModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="test-model")
        papers = {
            "p1": {"title": "Paper One", "abstract": "Abstract one"},
            "p2": {"title": "Paper Two", "abstract": "Abstract two"},
        }

        first = cache.get_embeddings(papers, model)
        second = cache.get_embeddings(papers, model)

    assert model.encode_calls == 1
    assert first["p1"].shape == second["p1"].shape


def test_embedding_cache_recomputes_on_text_change() -> None:
    """Cache key changes when text changes."""
    model = _MockModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="test-model")
        papers_v1 = {"p1": {"title": "Original", "abstract": "Abstract"}}
        papers_v2 = {"p1": {"title": "Updated", "abstract": "Abstract"}}

        cache.get_embeddings(papers_v1, model)
        cache.get_embeddings(papers_v2, model)

    assert model.encode_calls == 2


def test_embedding_cache_uses_matrix_dataset_layout() -> None:
    """Embeddings are stored in a single matrix dataset with row indices."""
    model = _MockModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="test-model")
        papers = {
            "p1": {"title": "Paper One", "abstract": "Abstract one"},
            "p2": {"title": "Paper Two", "abstract": "Abstract two"},
        }
        cache.get_embeddings(papers, model)

        with h5py.File(cache.h5_path, "r") as h5:
            assert list(h5.keys()) == ["embeddings"]
            assert h5["embeddings"].shape == (2, 2)

        with sqlite3.connect(cache.db_path) as conn:
            rows = conn.execute(
                "SELECT paper_id, row_idx FROM papers ORDER BY row_idx"
            ).fetchall()

    assert rows == [("p1", 0), ("p2", 1)]


def test_embedding_cache_reuses_row_for_text_updates() -> None:
    """Text updates overwrite existing rows instead of appending duplicates."""
    model = _MockModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="test-model")
        papers_v1 = {"p1": {"title": "Original", "abstract": "Abstract"}}
        papers_v2 = {"p1": {"title": "Updated", "abstract": "Abstract"}}

        cache.get_embeddings(papers_v1, model)
        with sqlite3.connect(cache.db_path) as conn:
            row_idx_before = conn.execute(
                "SELECT row_idx FROM papers WHERE paper_id = 'p1'"
            ).fetchone()[0]

        cache.get_embeddings(papers_v2, model)
        with sqlite3.connect(cache.db_path) as conn:
            row_idx_after = conn.execute(
                "SELECT row_idx FROM papers WHERE paper_id = 'p1'"
            ).fetchone()[0]

        with h5py.File(cache.h5_path, "r") as h5:
            assert h5["embeddings"].shape[0] == 1

    assert row_idx_before == row_idx_after
    assert model.encode_calls == 2


def test_embedding_cache_serializes_multiprocess_writes(tmp_path: Path) -> None:
    """Concurrent processes should serialize writes without HDF5 lock failures."""
    queue: mp.Queue = mp.Queue()
    processes = [
        mp.Process(
            target=_multiprocess_cache_worker,
            args=(str(tmp_path), idx, queue),
        )
        for idx in range(4)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join()

    results = []
    for _ in processes:
        try:
            results.append(queue.get(timeout=5))
        except Empty:
            results.append(("err", "missing", "worker did not report result"))
    errors = [result for result in results if result[0] == "err"]
    assert not errors, f"Concurrent cache writes failed: {errors}"
