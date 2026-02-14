"""Regression tests for embedding cache reset behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import h5py
import numpy as np

from citemesh.data.embedding_cache import EmbeddingCache


class _MockModel:
    """Minimal embedding model stub with deterministic encode output."""

    def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
        """Return deterministic embeddings for provided text batch.

        :param list[str] texts: Input texts.
        :param object _kwargs: Ignored keyword arguments.
        :return np.ndarray: Deterministic embedding matrix.
        """
        return np.array([[float(len(texts)), 1.0] for _ in texts], dtype=np.float32)


def test_legacy_h5_layout_is_dropped_and_rebuilt(tmp_path: Path) -> None:
    """Legacy cache files should be dropped and rebuilt under current schema."""
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="legacy-recovery")
    model = _MockModel()

    cache.h5_path.unlink(missing_ok=True)
    with h5py.File(cache.h5_path, "w") as h5:
        h5.create_dataset("legacy_payload", data=np.array([1, 2, 3], dtype=np.float32))

    with sqlite3.connect(cache.db_path) as conn:
        conn.execute(
            """
            INSERT INTO papers (paper_id, title, abstract, year, text_hash, embedding_dim, row_idx)
            VALUES ('seed', 'seed', '', NULL, 'hash', 3, 2)
            """
        )
        conn.commit()

    reloaded = EmbeddingCache(cache_dir=tmp_path, model_name="legacy-recovery")

    with sqlite3.connect(reloaded.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0

    reloaded.get_embeddings(
        {"seed": {"title": "Seed", "abstract": "x", "year": None}}, model
    )

    with h5py.File(reloaded.h5_path, "r") as h5:
        assert "embeddings" in h5
        assert h5["embeddings"].shape[0] == 1


def test_clear_removes_cache_files_and_recreates_schema(tmp_path: Path) -> None:
    """`clear()` should remove namespace cache files and recreate DB schema."""
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="clear-recovery")
    model = _MockModel()
    cache.get_embeddings(
        {"seed": {"title": "Seed", "abstract": "x", "year": 2020}}, model
    )

    cache.clear()

    assert cache.db_path.exists()
    assert not cache.h5_path.exists()
