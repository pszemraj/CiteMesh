"""
Persistent embedding cache backed by SQLite metadata and HDF5 vectors.
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
from tqdm.auto import tqdm

from citemesh.cache_utils import get_cache_dir


class EmbeddingCache:
    """
    Persistent cache for paper embeddings.

    SQLite stores metadata (including text hashes) while the actual embedding
    vectors live inside an HDF5 file for efficient random access.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        model_name: str = "google/embeddinggemma-300m",
    ):
        if cache_dir is None:
            cache_dir = get_cache_dir("embeddings")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
        self.db_path = self.cache_dir / f"metadata_{model_hash}.db"
        self.h5_path = self.cache_dir / f"embeddings_{model_hash}.h5"
        self.model_name = model_name

        self._init_db()

    # ------------------------------------------------------------------
    # Public API

    def get_embeddings(
        self,
        papers: Dict[str, Dict],
        model,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Return embeddings for provided papers, computing only the missing ones.

        Args:
            papers: Mapping of paper_id -> metadata dict containing title/abstract/year.
            model: SentenceTransformer-compatible model providing encode().
            batch_size: Batch size for model encoding.
            show_progress: Whether to display tqdm progress bars.
        """
        if not papers:
            return {}

        cached_embeddings: Dict[str, np.ndarray] = {}
        papers_to_embed: list[Tuple[str, Dict, str]] = []

        items = list(papers.items())
        progress_enabled = show_progress and sys.stderr.isatty() and len(items) > 50
        iterator: Iterable[Tuple[str, Dict]] = tqdm(
            items, desc="Checking cache", unit="papers", disable=not progress_enabled
        )

        with sqlite3.connect(self.db_path) as conn, h5py.File(self.h5_path, "a") as h5:
            cursor = conn.cursor()

            for paper_id, metadata in iterator:
                text = _build_text(metadata)
                text_hash = self._text_hash(text)

                row = cursor.execute(
                    "SELECT text_hash FROM papers WHERE paper_id = ?",
                    (paper_id,),
                ).fetchone()

                if row and row[0] == text_hash and paper_id in h5:
                    cached_embeddings[paper_id] = h5[paper_id][:]
                else:
                    papers_to_embed.append((paper_id, metadata, text_hash))

            if progress_enabled:
                iterator.close()

            if not papers_to_embed:
                return cached_embeddings

            texts = [_build_text(meta) for _, meta, _ in papers_to_embed]
            embeddings_array = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )

            new_embeddings: Dict[str, np.ndarray] = {}
            for idx, (paper_id, metadata, text_hash) in enumerate(papers_to_embed):
                embedding = np.asarray(embeddings_array[idx], dtype=np.float32)
                new_embeddings[paper_id] = embedding

                if paper_id in h5:
                    del h5[paper_id]
                h5.create_dataset(paper_id, data=embedding)

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO papers
                    (paper_id, title, abstract, year, text_hash, embedding_dim)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id,
                        metadata.get("title", ""),
                        metadata.get("abstract", ""),
                        metadata.get("year", 0),
                        text_hash,
                        int(embedding.shape[0]),
                    ),
                )

            conn.commit()

        return {**cached_embeddings, **new_embeddings}

    def get_stats(self) -> Dict[str, Optional[float]]:
        """Return basic cache statistics."""
        total_h5_size = self.h5_path.stat().st_size if self.h5_path.exists() else 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM papers")
            total_papers = cursor.fetchone()[0]

            cursor.execute(
                "SELECT AVG(embedding_dim), MIN(year), MAX(year) FROM papers WHERE embedding_dim IS NOT NULL"
            )
            avg_dim, min_year, max_year = cursor.fetchone()

        return {
            "total_papers": total_papers,
            "avg_embedding_dim": avg_dim,
            "year_range": (min_year, max_year),
            "cache_size_mb": total_h5_size / (1024 * 1024),
        }

    def clear(self) -> None:
        """Remove cached metadata and embeddings."""
        if self.db_path.exists():
            self.db_path.unlink()
        if self.h5_path.exists():
            self.h5_path.unlink()
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    year INTEGER,
                    text_hash TEXT,
                    embedding_dim INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_text_hash ON papers(text_hash)"
            )
            conn.commit()

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_text(metadata: Dict) -> str:
    """Compose text used for embedding computation."""
    title = metadata.get("title", "")
    abstract = metadata.get("abstract", "")
    return f"{title}. {abstract}".strip()
