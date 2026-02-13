"""Regression tests for non-destructive embedding cache migration."""

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


def test_legacy_h5_layout_is_backed_up_and_preserved(tmp_path: Path) -> None:
    """Legacy cache files should be renamed for recovery instead of deleted."""
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
    backups = sorted(cache.h5_path.parent.glob(f"{cache.h5_path.name}.bak.*"))
    assert backups, "Expected a backup path for legacy HDF5 layout"

    with sqlite3.connect(reloaded.db_path) as conn:
        assert (
            conn.execute(
                "SELECT row_idx FROM papers WHERE paper_id = 'seed'"
            ).fetchone()[0]
            is None
        )

    reloaded.get_embeddings(
        {"seed": {"title": "Seed", "abstract": "x", "year": None}}, model
    )

    with h5py.File(reloaded.h5_path, "r") as h5:
        assert "embeddings" in h5
        assert h5["embeddings"].shape[0] == 1

    assert any(backup.exists() for backup in backups)


def test_clear_moves_cache_files_to_backups(tmp_path: Path) -> None:
    """`clear()` should move cache files and rebuild a fresh empty schema."""
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="clear-recovery")
    model = _MockModel()
    cache.get_embeddings(
        {"seed": {"title": "Seed", "abstract": "x", "year": 2020}}, model
    )

    cache.clear()

    db_backups = sorted(cache.db_path.parent.glob(f"{cache.db_path.name}.bak.*"))
    h5_backups = sorted(cache.h5_path.parent.glob(f"{cache.h5_path.name}.bak.*"))
    assert db_backups
    assert h5_backups
    assert cache.db_path.exists()
    assert not cache.h5_path.exists()
