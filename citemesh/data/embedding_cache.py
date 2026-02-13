"""
Persistent embedding cache backed by SQLite metadata and HDF5 vectors.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import h5py
import numpy as np
from filelock import FileLock, Timeout
from tqdm.auto import tqdm

from .cache import get_cache_dir

logger = logging.getLogger(__name__)


SQLITE_QUERY_BATCH_SIZE = 900
EMBEDDINGS_DATASET_NAME = "embeddings"
EMBEDDING_CACHE_SCHEMA_VERSION = 1
EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS = 60.0
H5_LAYOUT_KEY = "h5_layout_version"
H5_LAYOUT_MATRIX_VERSION = "matrix-v1"
SCHEMA_VERSION_KEY = "schema_version"
MIGRATION_STATE_KEY = "migration_state"
MIGRATION_STATE_ACTIVE = "active"
MIGRATION_STATE_LEGACY_BACKUP = "legacy_h5_backed_up"
MIGRATION_STATE_CLEAR = "explicit_clear"


def _metadata_table_create_sql() -> str:
    return """
    CREATE TABLE IF NOT EXISTS cache_metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """


def _now_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")


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
        """Create a persistent embedding cache for a model variant.

        :param Optional[Path] cache_dir: Cache directory override. Uses global cache when ``None``.
        :param str model_name: Model name used to namespace cached embeddings.
        """
        if cache_dir is None:
            cache_dir = get_cache_dir("embeddings")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
        self.db_path = self.cache_dir / f"metadata_{model_hash}.db"
        self.h5_path = self.cache_dir / f"embeddings_{model_hash}.h5"
        self.lock_path = self.cache_dir / f"cache_{model_hash}.lock"
        self.model_name = model_name

        with self._cache_lock():
            self._init_db()
            self._ensure_h5_layout()

    # ------------------------------------------------------------------
    # Public API

    def get_embeddings(
        self,
        papers: Dict[str, Dict],
        model: Any,
        batch_size: int = 32,
        show_progress: bool = True,
        text_builder: Optional[Callable[[Dict[str, object]], str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Return embeddings for provided papers, computing only the missing ones.

        :param Dict[str, Dict] papers: Mapping of paper_id -> metadata dict containing title/abstract/year.
        :param Any model: SentenceTransformer-compatible model exposing ``encode``.
        :param int batch_size: Batch size for model encoding.
        :param bool show_progress: Whether to display tqdm progress bars.
        :param Optional[Callable[[Dict[str, object]], str]] text_builder: Optional text builder
            for each paper metadata record. Defaults to internal helper.
        :return Dict[str, np.ndarray]: Paper embeddings for requested records.
        """
        if not papers:
            return {}

        cached_embeddings: Dict[str, np.ndarray] = {}
        papers_to_embed: List[Tuple[str, Dict, str, str, Optional[int]]] = []
        cached_rows: List[Tuple[str, int]] = []
        builder = text_builder or _build_text

        items = list(papers.items())
        progress_enabled = show_progress and sys.stderr.isatty() and len(items) > 50
        iterator: Iterable[Tuple[str, Dict]] = tqdm(
            items, desc="Checking cache", unit="papers", disable=not progress_enabled
        )

        with (
            self._cache_lock(),
            sqlite3.connect(self.db_path) as conn,
            h5py.File(self.h5_path, "a") as h5,
        ):
            cursor = conn.cursor()
            existing_rows = self._load_existing_rows(
                conn, [paper_id for paper_id, _ in items]
            )
            embeddings_dataset = self._get_embeddings_dataset(h5)
            cached_limit = (
                int(embeddings_dataset.shape[0])
                if embeddings_dataset is not None
                else 0
            )

            for paper_id, metadata in iterator:
                text = builder(metadata)
                text_hash = self._text_hash(text)
                existing_row = existing_rows.get(paper_id)
                row_idx = existing_row[1] if existing_row is not None else None

                if (
                    existing_row is not None
                    and existing_row[0] == text_hash
                    and row_idx is not None
                    and embeddings_dataset is not None
                    and 0 <= row_idx < cached_limit
                ):
                    cached_rows.append((paper_id, row_idx))
                else:
                    papers_to_embed.append(
                        (paper_id, metadata, text_hash, text, row_idx)
                    )

            if progress_enabled:
                iterator.close()

            if cached_rows and embeddings_dataset is not None:
                cached_embeddings = self._load_cached_embeddings(
                    embeddings_dataset,
                    cached_rows,
                )

            if not papers_to_embed:
                return cached_embeddings

            texts = [text for _, _, _, text, _ in papers_to_embed]
            embeddings_array = np.asarray(
                model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=show_progress,
                ),
                dtype=np.float32,
            )
            if embeddings_array.ndim == 1:
                embeddings_array = embeddings_array.reshape(1, -1)
            if embeddings_array.shape[0] != len(papers_to_embed):
                raise ValueError(
                    "Embedding model returned unexpected row count: "
                    f"{embeddings_array.shape[0]} for {len(papers_to_embed)} papers."
                )

            embedding_dim = int(embeddings_array.shape[1])
            embeddings_dataset = self._ensure_embeddings_dataset(h5, embedding_dim)
            existing_row_count = int(embeddings_dataset.shape[0])

            new_embeddings: Dict[str, np.ndarray] = {}
            rows_to_upsert: List[Tuple[Any, ...]] = []
            append_embeddings: List[np.ndarray] = []
            append_records: List[Tuple[str, Dict, str]] = []

            for idx, (paper_id, metadata, text_hash, _, existing_row_idx) in enumerate(
                papers_to_embed
            ):
                embedding = embeddings_array[idx]
                new_embeddings[paper_id] = embedding

                if (
                    existing_row_idx is not None
                    and 0 <= existing_row_idx < existing_row_count
                ):
                    embeddings_dataset[existing_row_idx] = embedding
                    rows_to_upsert.append(
                        self._metadata_tuple(
                            paper_id=paper_id,
                            metadata=metadata,
                            text_hash=text_hash,
                            embedding_dim=embedding_dim,
                            row_idx=existing_row_idx,
                        )
                    )
                else:
                    append_embeddings.append(embedding)
                    append_records.append((paper_id, metadata, text_hash))

            if append_embeddings:
                append_array = np.vstack(append_embeddings).astype(
                    np.float32, copy=False
                )
                start_idx = existing_row_count
                end_idx = start_idx + append_array.shape[0]
                embeddings_dataset.resize((end_idx, embedding_dim))
                embeddings_dataset[start_idx:end_idx] = append_array

                for offset, (paper_id, metadata, text_hash) in enumerate(
                    append_records
                ):
                    rows_to_upsert.append(
                        self._metadata_tuple(
                            paper_id=paper_id,
                            metadata=metadata,
                            text_hash=text_hash,
                            embedding_dim=embedding_dim,
                            row_idx=start_idx + offset,
                        )
                    )

            if rows_to_upsert:
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO papers
                    (paper_id, title, abstract, year, text_hash, embedding_dim, row_idx)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows_to_upsert,
                )

            conn.commit()

        return {**cached_embeddings, **new_embeddings}

    def get_stats(self) -> Dict[str, Optional[float]]:
        """Return basic cache statistics.

        :return Dict[str, Optional[float]]: Cache size, embedding dimensions, and year range.
        """
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
        """Purge cache artifacts while preserving old files as backups."""
        with self._cache_lock():
            if self.db_path.exists():
                self._backup_cache_file(self.db_path, MIGRATION_STATE_CLEAR)

            if self.h5_path.exists():
                self._backup_cache_file(self.h5_path, MIGRATION_STATE_CLEAR)

            self._init_db()

    def _make_backup_path(self, target_path: Path, suffix: str) -> Path:
        """Build a timestamped backup path for cache artifacts."""
        return target_path.with_suffix(f"{target_path.suffix}.{suffix}")

    def _backup_cache_file(self, target_path: Path, suffix: str) -> Optional[Path]:
        """Move an existing cache artifact to a timestamped backup path."""
        if not target_path.exists():
            return None

        backup_suffix = f"bak.{suffix}.{_now_timestamp()}"
        backup_path = self._make_backup_path(target_path, backup_suffix)
        counter = 0
        while backup_path.exists():
            counter += 1
            backup_path = self._make_backup_path(
                target_path, f"bak.{suffix}.{_now_timestamp()}.{counter}"
            )

        target_path.rename(backup_path)
        return backup_path

    # ------------------------------------------------------------------
    # Internal helpers

    @contextmanager
    def _cache_lock(self) -> Iterator[None]:
        """Serialize cache mutations across processes for this model namespace."""
        lock = FileLock(
            str(self.lock_path), timeout=EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS
        )
        try:
            with lock:
                yield
        except Timeout as exc:
            raise TimeoutError(
                "Timed out waiting for embedding cache lock "
                f"at {self.lock_path}. Another process may be holding it."
            ) from exc

    def _init_db(self) -> None:
        """Create and initialize the metadata cache schema when needed."""
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
                    row_idx INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Upgrade old metadata databases created before row_idx existed.
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(papers)").fetchall()
            }
            if "row_idx" not in columns:
                conn.execute("ALTER TABLE papers ADD COLUMN row_idx INTEGER")

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_text_hash ON papers(text_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_row_idx ON papers(row_idx)"
            )
            conn.execute(_metadata_table_create_sql())
            conn.execute(
                """
                INSERT INTO cache_metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (SCHEMA_VERSION_KEY, str(EMBEDDING_CACHE_SCHEMA_VERSION)),
            )
            conn.execute(
                """
                INSERT INTO cache_metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (H5_LAYOUT_KEY, H5_LAYOUT_MATRIX_VERSION),
            )
            conn.execute(
                """
                INSERT INTO cache_metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (MIGRATION_STATE_KEY, MIGRATION_STATE_ACTIVE),
            )
            conn.commit()

    @staticmethod
    def _set_cache_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            """
            INSERT INTO cache_metadata (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _invalidate_cached_row_indices(self, conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE papers SET row_idx = NULL")

    def _backup_incompatible_h5(self) -> Path:
        return self._backup_cache_file(
            self.h5_path, MIGRATION_STATE_LEGACY_BACKUP
        ) or self._make_backup_path(
            self.h5_path, f"bak.{MIGRATION_STATE_LEGACY_BACKUP}.{_now_timestamp()}"
        )

    def _reconcile_layout_metadata(self, conn: sqlite3.Connection) -> None:
        self._set_cache_metadata(conn, H5_LAYOUT_KEY, H5_LAYOUT_MATRIX_VERSION)
        self._set_cache_metadata(conn, MIGRATION_STATE_KEY, MIGRATION_STATE_ACTIVE)

    def _ensure_h5_layout(self) -> None:
        """Ensure cache file uses matrix-based HDF5 layout."""
        if not self.h5_path.exists():
            with sqlite3.connect(self.db_path) as conn:
                self._reconcile_layout_metadata(conn)
                conn.commit()
                return

        try:
            with (
                h5py.File(self.h5_path, "r") as h5,
                sqlite3.connect(self.db_path) as conn,
            ):
                if EMBEDDINGS_DATASET_NAME in h5:
                    self._reconcile_layout_metadata(conn)
                    conn.commit()
                    return

                first_key = next(iter(h5.keys()), None)
        except OSError:
            logger.warning(
                "Embedding cache file at %s is unreadable. Resetting cache.",
                self.h5_path,
            )
            with sqlite3.connect(self.db_path) as conn:
                self._invalidate_cached_row_indices(conn)
                try:
                    backup_path = self._backup_incompatible_h5()
                    logger.warning(
                        "Moved unreadable embedding cache %s to %s.",
                        self.h5_path,
                        backup_path,
                    )
                except OSError as exc:
                    logger.warning(
                        "Could not back up unreadable embedding cache %s: %s",
                        self.h5_path,
                        exc,
                    )
                    self._reconcile_layout_metadata(conn)
                    conn.commit()
                return

        if first_key is not None:
            logger.warning(
                "Detected legacy per-paper embedding cache layout at %s. "
                "Moving cache and rebuilding matrix storage on first use.",
                self.h5_path,
            )
            with sqlite3.connect(self.db_path) as conn:
                self._invalidate_cached_row_indices(conn)
                try:
                    backup_path = self._backup_incompatible_h5()
                    logger.warning(
                        "Moved legacy per-paper embedding cache %s to %s.",
                        self.h5_path,
                        backup_path,
                    )
                except OSError as exc:
                    logger.warning(
                        "Could not back up legacy embedding cache %s: %s",
                        self.h5_path,
                        exc,
                    )
                self._reconcile_layout_metadata(conn)
                conn.commit()

    @staticmethod
    def _metadata_tuple(
        paper_id: str,
        metadata: Dict[str, object],
        text_hash: str,
        embedding_dim: int,
        row_idx: int,
    ) -> Tuple[Any, ...]:
        """Build metadata row tuple for SQLite upsert."""
        return (
            paper_id,
            metadata.get("title", ""),
            metadata.get("abstract", ""),
            metadata.get("year", 0),
            text_hash,
            embedding_dim,
            row_idx,
        )

    def _load_existing_rows(
        self,
        conn: sqlite3.Connection,
        paper_ids: Sequence[str],
    ) -> Dict[str, Tuple[str, Optional[int]]]:
        """Fetch existing metadata rows for target paper IDs.

        :param sqlite3.Connection conn: Open SQLite connection.
        :param Sequence[str] paper_ids: Paper IDs to look up.
        :return Dict[str, Tuple[str, Optional[int]]]: Mapping of paper ID to (text_hash, row_idx).
        """
        if not paper_ids:
            return {}

        existing_rows: Dict[str, Tuple[str, Optional[int]]] = {}
        for id_chunk in _chunked(paper_ids, SQLITE_QUERY_BATCH_SIZE):
            placeholders = ",".join("?" for _ in id_chunk)
            query = (
                "SELECT paper_id, text_hash, row_idx "
                f"FROM papers WHERE paper_id IN ({placeholders})"
            )

            for paper_id, text_hash, row_idx in conn.execute(query, id_chunk):
                normalized_row_idx = int(row_idx) if row_idx is not None else None
                existing_rows[str(paper_id)] = (str(text_hash), normalized_row_idx)

        return existing_rows

    @staticmethod
    def _get_embeddings_dataset(h5_file: h5py.File) -> Optional[h5py.Dataset]:
        """Return matrix embedding dataset when available."""
        dataset = h5_file.get(EMBEDDINGS_DATASET_NAME)
        if dataset is None:
            return None
        if dataset.ndim != 2:
            raise ValueError(
                f"Embedding dataset '{EMBEDDINGS_DATASET_NAME}' must be 2D."
            )
        return dataset

    def _ensure_embeddings_dataset(
        self, h5_file: h5py.File, embedding_dim: int
    ) -> h5py.Dataset:
        """Create or validate the matrix embedding dataset.

        :param h5py.File h5_file: Open HDF5 file handle.
        :param int embedding_dim: Required embedding width.
        :return h5py.Dataset: Resizable embeddings dataset.
        """
        dataset = self._get_embeddings_dataset(h5_file)
        if dataset is None:
            return h5_file.create_dataset(
                EMBEDDINGS_DATASET_NAME,
                shape=(0, embedding_dim),
                maxshape=(None, embedding_dim),
                dtype=np.float32,
            )

        if int(dataset.shape[1]) != embedding_dim:
            raise ValueError(
                "Embedding dimension mismatch in cache: "
                f"{int(dataset.shape[1])} != {embedding_dim}"
            )

        return dataset

    @staticmethod
    def _load_cached_embeddings(
        dataset: h5py.Dataset,
        cached_rows: Sequence[Tuple[str, int]],
    ) -> Dict[str, np.ndarray]:
        """Load cached embeddings from matrix dataset in row-index order.

        :param h5py.Dataset dataset: Matrix dataset containing all embeddings.
        :param Sequence[Tuple[str, int]] cached_rows: Pairs of paper_id and row index.
        :return Dict[str, np.ndarray]: Mapping of paper IDs to embeddings.
        """
        if not cached_rows:
            return {}

        sorted_rows = sorted(cached_rows, key=lambda item: item[1])
        indices = np.asarray([row_idx for _, row_idx in sorted_rows], dtype=np.int64)
        matrix = np.asarray(dataset[indices], dtype=np.float32)

        return {paper_id: matrix[idx] for idx, (paper_id, _) in enumerate(sorted_rows)}

    @staticmethod
    def _text_hash(text: str) -> str:
        """Compute deterministic SHA-256 hash for text content.

        :param str text: Normalized paper text.
        :return str: Hexadecimal SHA-256 digest.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_text(metadata: Dict) -> str:
    """Compose paper text for embedding computation.

    :param Dict metadata: Paper metadata containing title and abstract.
    :return str: Concatenated title and abstract string.
    """
    title = metadata.get("title", "")
    abstract = metadata.get("abstract", "")
    return f"{title}. {abstract}".strip()


def _chunked(values: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    """Yield fixed-size chunks from a sequence."""
    for start in range(0, len(values), chunk_size):
        yield list(values[start : start + chunk_size])
