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


class _LookupModel:
    """Deterministic model backed by a text->embedding lookup table."""

    def __init__(self, lookup: dict[str, np.ndarray]) -> None:
        """Store lookup table used for ``encode`` calls.

        :param dict[str, np.ndarray] lookup: Text->embedding table.
        """
        self.lookup = lookup

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        """Return lookup embeddings in request order.

        :param list[str] texts: Input texts.
        :param Any kwargs: Ignored keyword arguments.
        :return np.ndarray: Embedding matrix.
        """
        del kwargs
        return np.asarray([self.lookup[text] for text in texts], dtype=np.float32)


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
    """Embeddings are stored as int8 matrix + binary index + calibration ranges."""
    model = _MockModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="test-model")
        papers = {
            "p1": {"title": "Paper One", "abstract": "Abstract one"},
            "p2": {"title": "Paper Two", "abstract": "Abstract two"},
        }
        cache.get_embeddings(papers, model)

        with h5py.File(cache.h5_path, "r") as h5:
            assert set(h5.keys()) == {
                "embeddings",
                "binary_index",
                "calibration_ranges",
            }
            assert h5["embeddings"].shape == (2, 2)
            assert h5["embeddings"].dtype == np.int8
            assert h5["binary_index"].shape == (2, 1)
            assert h5["binary_index"].dtype == np.uint8
            assert h5["calibration_ranges"].shape == (2, 2)
            assert h5["calibration_ranges"].dtype == np.float32

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


def test_embedding_cache_search_returns_metadata_and_float_embeddings() -> None:
    """Cache search should return metadata payload and float32 embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="search-metadata")
        papers = {
            "p1": {
                "title": "Alpha",
                "abstract": "First",
                "year": 2020,
                "authors": ["Alice", "Bob"],
                "categories": ["cs.AI"],
            },
            "p2": {
                "title": "Beta",
                "abstract": "Second",
                "year": 2021,
                "authors": ["Carol"],
                "categories": ["cs.LG"],
            },
        }
        model = _LookupModel(
            {
                "Alpha. First": np.array([1.0, 0.0], dtype=np.float32),
                "Beta. Second": np.array([0.0, 1.0], dtype=np.float32),
            }
        )

        _ = cache.get_embeddings(papers, model, show_progress=False)
        results = cache.search(
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=2,
            binary_prefilter=True,
            binary_rescore_multiplier=4,
        )

    assert [result.paper_id for result in results] == ["p1", "p2"]
    assert results[0].metadata["authors"] == ["Alice", "Bob"]
    assert results[0].metadata["categories"] == ["cs.AI"]
    assert results[0].metadata["year"] == 2020
    assert results[0].embedding.dtype == np.float32


def test_embedding_cache_reuses_calibration_ranges_for_same_namespace() -> None:
    """Calibration ranges should persist and be reused within one namespace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="calibration-reuse")
        first_model = _LookupModel(
            {
                "One. A": np.array([0.1, 0.9], dtype=np.float32),
                "Two. B": np.array([0.9, 0.1], dtype=np.float32),
            }
        )
        second_model = _LookupModel(
            {
                "Three. C": np.array([0.3, 0.7], dtype=np.float32),
            }
        )

        _ = cache.get_embeddings(
            {
                "p1": {"title": "One", "abstract": "A"},
                "p2": {"title": "Two", "abstract": "B"},
            },
            first_model,
            show_progress=False,
        )
        with h5py.File(cache.h5_path, "r") as h5:
            ranges_before = np.asarray(h5["calibration_ranges"], dtype=np.float32)

        _ = cache.get_embeddings(
            {"p3": {"title": "Three", "abstract": "C"}},
            second_model,
            show_progress=False,
        )
        with h5py.File(cache.h5_path, "r") as h5:
            ranges_after = np.asarray(h5["calibration_ranges"], dtype=np.float32)

    np.testing.assert_allclose(ranges_before, ranges_after)
