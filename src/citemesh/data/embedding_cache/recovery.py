"""Lock acquisition, SQLite connections, and crash-recovery for the cache.

Owns the coordination and durability surface: the reentrant hydration operation
lock, the short-lived namespace lock, managed cache-root resolution, the SQLite
connection context manager, the replacement journal that makes vector rewrites
crash-safe, its replay paths, trailing-row truncation, and the HDF5 flush/fsync
barrier.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import h5py
import numpy as np
from filelock import FileLock, Timeout

from ..cache import (
    CACHE_COORDINATION_DIRNAME,
    cache_operation_lock,
    path_exists,
)
from .constants import (
    BINARY_INDEX_DATASET_NAME,
    EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR,
    HYDRATION_COMPLETE_KEY,
    _resolve_cache_lock_timeout_seconds,
)
from .models import _EmbeddingCacheLayoutError

logger = logging.getLogger(__name__)


class _RecoveryMixin:
    """Locking, connection, and crash-recovery behaviour for the cache."""

    @contextmanager
    def hydration_operation_lock(self) -> Iterator[None]:
        """Serialize a complete corpus hydration and its consuming search.

        The same lock object is intentionally retained on the cache instance so
        nested acquisition from the builder remains reentrant. Short SQLite/HDF5
        mutations continue to use :meth:`_cache_lock` independently.

        :return Iterator[None]: Context manager yielding once the operation lock is acquired.
        """
        timeout_seconds = _resolve_cache_lock_timeout_seconds()
        try:
            with self._cache_operation_lock():
                with self._hydration_operation_file_lock.acquire(
                    timeout=timeout_seconds
                ):
                    yield
        except Timeout as exc:
            raise TimeoutError(
                "Timed out waiting for embedding cache hydration operation lock "
                f"at {self.hydration_lock_path} after {timeout_seconds:.3f}s. "
                "Another process may be hydrating or searching this namespace. "
                f"Increase {EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR} or set "
                "CITEMESH_CACHE_DIR to an isolated per-run cache root."
            ) from exc

    @contextmanager
    def _cache_lock(self) -> Iterator[None]:
        """Serialize cache mutations across processes for this model namespace.

        :return Iterator[None]: Context manager yielding once lock is acquired.
        """
        timeout_seconds = _resolve_cache_lock_timeout_seconds()
        with self._cache_operation_lock():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            lock = FileLock(str(self.lock_path), timeout=timeout_seconds)
            try:
                with lock:
                    if not path_exists(self.db_path):
                        self._init_db()
                    yield
            except Timeout as exc:
                raise TimeoutError(
                    "Timed out waiting for embedding cache lock "
                    f"at {self.lock_path} after {timeout_seconds:.3f}s. "
                    "Another process may be holding it. "
                    f"Increase {EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR} or set "
                    "CITEMESH_CACHE_DIR to an isolated per-run cache root."
                ) from exc

    @staticmethod
    def _resolve_managed_cache_root(
        cache_dir: Path, configured_cache_root: Path
    ) -> Optional[Path]:
        """Return the configured root when ``cache_dir`` is cleared by the CLI.

        :param Path cache_dir: Embedding namespace directory.
        :param Path configured_cache_root: Active CiteMesh cache root.
        :return Optional[Path]: Root coordinated with ``cache clear``, if applicable.
        """
        resolved_root = configured_cache_root.expanduser().resolve()
        try:
            cache_dir.expanduser().resolve().relative_to(resolved_root)
        except ValueError:
            return None
        return resolved_root

    @contextmanager
    def _cache_operation_lock(self) -> Iterator[None]:
        """Hold a shared root lock when this namespace belongs to CiteMesh's root.

        :return Iterator[None]: Context manager protecting a live cache operation.
        """
        if self._managed_cache_root is None:
            yield
            return
        timeout_seconds = _resolve_cache_lock_timeout_seconds()
        try:
            with cache_operation_lock(
                self._managed_cache_root, timeout=timeout_seconds
            ):
                yield
        except Timeout as exc:
            raise TimeoutError(
                "Timed out waiting for cache-root operation lock "
                f"at {self._managed_cache_root / CACHE_COORDINATION_DIRNAME} "
                f"after {timeout_seconds:.3f}s. Another process may be clearing "
                "or using this cache root. "
                f"Increase {EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR} or set "
                "CITEMESH_CACHE_DIR to an isolated per-run cache root."
            ) from exc

    @contextmanager
    def _connect_db(self) -> Iterator[sqlite3.Connection]:
        """Open a SQLite connection that is closed on context exit.

        ``sqlite3.connect()`` as a context manager only commits/rollbacks —
        it does not close the connection.  On Windows the unclosed handle
        prevents file deletion or replacement, causing ``[WinError 32]``.

        :return Iterator[sqlite3.Connection]: Context manager yielding an open connection.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _persist_replacement_journal(
        self,
        conn: sqlite3.Connection,
        replacements: Sequence[Tuple[int, np.ndarray, Optional[np.ndarray]]],
    ) -> None:
        """Commit prior replacement rows before mutating HDF5 storage.

        :param sqlite3.Connection conn: Open SQLite connection for the active namespace.
        :param Sequence[Tuple[int, np.ndarray, Optional[np.ndarray]]] replacements:
            ``(row_idx, embedding, binary_embedding)`` rows to preserve.
        :return None: Inserts and commits durable undo records.
        """
        if not replacements:
            return

        journal_rows = []
        for row_idx, embedding, binary_embedding in replacements:
            old_embedding = np.ascontiguousarray(embedding)
            old_binary = (
                None
                if binary_embedding is None
                else np.ascontiguousarray(binary_embedding)
            )
            journal_rows.append(
                (
                    int(row_idx),
                    sqlite3.Binary(old_embedding.tobytes()),
                    int(old_embedding.size),
                    None
                    if old_binary is None
                    else sqlite3.Binary(old_binary.tobytes()),
                    None if old_binary is None else int(old_binary.size),
                )
            )
        conn.executemany(
            """
            INSERT INTO replacement_journal
                (row_idx, embedding, embedding_width, binary_embedding, binary_width)
            VALUES (?, ?, ?, ?, ?)
            """,
            journal_rows,
        )
        conn.commit()

    def _recover_pending_replacements_with_connection_locked(
        self, conn: sqlite3.Connection
    ) -> None:
        """Open HDF5 for durable replacement and trailing-row recovery.

        Callers must already hold this namespace's cache lock and own ``conn``.

        :param sqlite3.Connection conn: Open SQLite connection for the active namespace.
        :return None: Restores any pending replacement rows before subsequent reads.
        :raises RuntimeError: If pending replacement rows cannot be restored safely.
        """
        pending = conn.execute("SELECT 1 FROM replacement_journal LIMIT 1").fetchone()
        if not path_exists(self.h5_path):
            if pending is not None:
                raise RuntimeError(
                    "Embedding cache recovery error: replacement journal exists but the "
                    "embedding matrix is missing. Existing cache files were preserved."
                )
            return
        if pending is None:
            with h5py.File(self.h5_path, "r") as h5_file:
                embeddings_dataset = self._get_embeddings_dataset(h5_file)
                if embeddings_dataset is None:
                    return
                embedding_rows = int(embeddings_dataset.shape[0])
            paper_rows = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
            if embedding_rows == paper_rows:
                return
        with h5py.File(self.h5_path, "a") as h5_file:
            self._recover_pending_replacements_locked(conn=conn, h5_file=h5_file)

    def _recover_pending_replacements_locked(
        self,
        conn: sqlite3.Connection,
        h5_file: h5py.File,
        *,
        validate_runtime_contract: bool = True,
    ) -> None:
        """Restore journaled rows and discard uncommitted trailing rows.

        :param sqlite3.Connection conn: Open SQLite connection for the active namespace.
        :param h5py.File h5_file: Writable HDF5 handle for the active namespace.
        :param bool validate_runtime_contract: Whether to validate payload identity
            before runtime recovery mutates HDF5.
        :return None: Reinstates durable prior rows and clears their journal entries.
        :raises RuntimeError: If a journaled row cannot be restored safely.
        """
        journal_rows = conn.execute(
            """
            SELECT row_idx, embedding, embedding_width, binary_embedding, binary_width
            FROM replacement_journal
            ORDER BY row_idx
            """
        ).fetchall()

        try:
            embeddings_dataset = self._get_embeddings_dataset(h5_file)
        except _EmbeddingCacheLayoutError as exc:
            if not journal_rows:
                raise
            raise RuntimeError(
                "Embedding cache recovery error: replacement journal cannot be "
                "applied to the embedding matrix. Existing cache files were "
                "preserved."
            ) from exc
        if embeddings_dataset is None:
            if journal_rows:
                raise RuntimeError(
                    "Embedding cache recovery error: replacement journal exists without "
                    "an embedding matrix. Existing cache files were preserved."
                )
            return

        embedding_rows = int(embeddings_dataset.shape[0])
        paper_rows = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        if not journal_rows and embedding_rows == paper_rows:
            return
        if validate_runtime_contract:
            self._assert_runtime_cache_consistency(
                conn=conn,
                h5_file=h5_file,
                embeddings_dataset=embeddings_dataset,
                fail_mode="runtime",
                check_row_mapping=False,
            )
        embedding_width = int(embeddings_dataset.shape[1])
        binary_dataset = h5_file.get(BINARY_INDEX_DATASET_NAME)
        for (
            row_idx,
            embedding_payload,
            stored_embedding_width,
            binary_payload,
            stored_binary_width,
        ) in journal_rows:
            resolved_row_idx = int(row_idx)
            if not 0 <= resolved_row_idx < embedding_rows:
                raise RuntimeError(
                    "Embedding cache recovery error: replacement journal row index "
                    f"{resolved_row_idx} is outside the embedding matrix. Existing "
                    "cache files were preserved."
                )
            if int(stored_embedding_width) != embedding_width:
                raise RuntimeError(
                    "Embedding cache recovery error: replacement journal embedding "
                    "width does not match the embedding matrix. Existing cache files "
                    "were preserved."
                )

            old_embedding = np.frombuffer(
                bytes(embedding_payload), dtype=embeddings_dataset.dtype
            )
            if old_embedding.size != embedding_width:
                raise RuntimeError(
                    "Embedding cache recovery error: replacement journal embedding "
                    "payload has an invalid size. Existing cache files were preserved."
                )
            embeddings_dataset[resolved_row_idx] = old_embedding

            if binary_payload is None:
                continue
            if binary_dataset is None or binary_dataset.ndim != 2:
                raise RuntimeError(
                    "Embedding cache recovery error: replacement journal requires a "
                    "binary index that is unavailable. Existing cache files were "
                    "preserved."
                )
            binary_width = int(binary_dataset.shape[1])
            if (
                not 0 <= resolved_row_idx < int(binary_dataset.shape[0])
                or int(stored_binary_width) != binary_width
            ):
                raise RuntimeError(
                    "Embedding cache recovery error: replacement journal binary row "
                    "does not match the binary index. Existing cache files were "
                    "preserved."
                )
            old_binary = np.frombuffer(bytes(binary_payload), dtype=np.uint8)
            if old_binary.size != binary_width:
                raise RuntimeError(
                    "Embedding cache recovery error: replacement journal binary "
                    "payload has an invalid size. Existing cache files were preserved."
                )
            binary_dataset[resolved_row_idx] = old_binary

        recovered_embedding_rows = self._recover_trailing_rows(
            conn=conn,
            h5_file=h5_file,
            embeddings_dataset=embeddings_dataset,
        )
        if journal_rows or recovered_embedding_rows != embedding_rows:
            self._flush_h5_file(h5_file)
        if journal_rows:
            conn.execute("DELETE FROM replacement_journal")

    @staticmethod
    def _flush_h5_file(h5_file: h5py.File) -> None:
        """Flush HDF5 buffers and sync the namespace file descriptor.

        :param h5py.File h5_file: Writable HDF5 handle whose mutations must persist.
        :return None: Flushes HDF5 metadata/raw data and synchronizes the file.
        :raises RuntimeError: If the namespace file cannot be synchronized.
        """
        h5_file.flush()
        try:
            with open(h5_file.filename, "r+b", buffering=0) as sync_file:
                os.fsync(sync_file.fileno())
        except OSError as exc:
            raise RuntimeError(
                "Embedding cache recovery error: failed to durably flush HDF5 "
                "replacement rows. Existing cache files were preserved."
            ) from exc

    def _recover_trailing_rows(
        self,
        conn: sqlite3.Connection,
        h5_file: h5py.File,
        embeddings_dataset: h5py.Dataset,
    ) -> int:
        """Preserve the shared row prefix after an interrupted append.

        :param sqlite3.Connection conn: Open SQLite connection with committed mappings.
        :param h5py.File h5_file: Open HDF5 cache handle.
        :param h5py.Dataset embeddings_dataset: Resizable embeddings matrix dataset.
        :return int: Embedding row count after any recoverable truncation.
        """
        embedding_rows = int(embeddings_dataset.shape[0])
        paper_rows = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        if embedding_rows == paper_rows:
            return embedding_rows

        valid_rows, distinct_rows, minimum_row, maximum_row = conn.execute(
            """
            SELECT COUNT(row_idx), COUNT(DISTINCT row_idx), MIN(row_idx), MAX(row_idx)
            FROM papers
            """,
        ).fetchone()
        committed_prefix_is_complete = bool(
            int(valid_rows) == paper_rows
            and int(distinct_rows) == paper_rows
            and (
                paper_rows == 0
                or (int(minimum_row) == 0 and int(maximum_row) == paper_rows - 1)
            )
        )
        if not committed_prefix_is_complete:
            return embedding_rows

        if paper_rows > embedding_rows:
            logger.warning(
                "Recovering embedding cache %s by removing %d trailing SQLite "
                "mapping(s) without persisted vectors; preserving %d row(s).",
                self.h5_path,
                paper_rows - embedding_rows,
                embedding_rows,
            )
            conn.execute("DELETE FROM papers WHERE row_idx >= ?", (embedding_rows,))
            self._set_cache_metadata(conn, {HYDRATION_COMPLETE_KEY: "0"})
            return embedding_rows

        orphan_rows = embedding_rows - paper_rows
        logger.warning(
            "Recovering embedding cache %s by truncating %d uncommitted trailing "
            "HDF5 row(s); preserving %d committed row(s).",
            self.h5_path,
            orphan_rows,
            paper_rows,
        )
        embeddings_dataset.resize((paper_rows, int(embeddings_dataset.shape[1])))

        binary_dataset = h5_file.get(BINARY_INDEX_DATASET_NAME)
        if (
            binary_dataset is not None
            and binary_dataset.ndim == 2
            and int(binary_dataset.shape[0]) > paper_rows
        ):
            try:
                binary_dataset.resize((paper_rows, int(binary_dataset.shape[1])))
            except (OSError, TypeError, ValueError):
                del h5_file[BINARY_INDEX_DATASET_NAME]

        return paper_rows
