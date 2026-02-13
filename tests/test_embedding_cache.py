"""Tests for embedding cache hit/miss behavior."""

import sqlite3
import tempfile
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
