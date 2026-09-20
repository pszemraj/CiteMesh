"""SQLite schema and HDF5 physical-layout management mixed into the cache.

Owns everything that shapes persisted namespace state: schema creation,
cache-metadata read/write, runtime-contract stamping and consistency
enforcement, compression adoption, HDF5 dataset creation/resizing for the
matrix and packed-binary indexes, hydration-metadata resets, and the SQLite
paper-row read/write helpers.
"""

from __future__ import annotations

import logging
import sqlite3
import stat
from collections.abc import Iterator, Sequence
from typing import Any

import h5py
import numpy as np

from citemesh.core.paper_ids import encode_arxiv_id_chronology_key

from ..cache import path_exists
from .constants import (
    _COMPRESSION_FILTERS,
    _PAPER_ROW_COLUMNS,
    _PAPER_ROW_LOOKUP_COLUMNS,
    BINARY_INDEX_DATASET_NAME,
    BINARY_INDEX_ENCODING,
    BINARY_INDEX_ENCODING_KEY,
    BINARY_PREFILTER_ENABLED_KEY,
    CALIBRATION_RANGES_DATASET_NAME,
    CALIBRATION_SAMPLE_SIZE_KEY,
    COMPRESSION_FILTER_KEY,
    COMPRESSION_LEVEL_KEY,
    CORPUS_METADATA_VERSION_KEY,
    EMBEDDING_CACHE_SCHEMA_VERSION,
    EMBEDDING_DATASET_CHUNK_ROWS,
    EMBEDDING_VECTOR_DTYPE_KEY,
    EMBEDDINGS_DATASET_NAME,
    H5_LAYOUT_KEY,
    H5_LAYOUT_MATRIX_VERSION,
    HYDRATION_COMPLETE_KEY,
    HYDRATION_CORPUS_SIZE_KEY,
    HYDRATION_DATASET_SOURCE_KEY,
    HYDRATION_RECONCILED_CACHE_ROWS_KEY,
    HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY,
    HYDRATION_SPLIT_KEY,
    MODEL_FINGERPRINT_KEY,
    SCHEMA_VERSION_KEY,
    SOURCE_TORCH_DTYPE_KEY,
    SQLITE_QUERY_BATCH_SIZE,
    STORAGE_PRECISION_KEY,
    TEXT_FORMATTER_FINGERPRINT_KEY,
)
from .models import CacheNamespacePayloadStats, _EmbeddingCacheLayoutError
from .quantization import (
    _quantize_ubinary_embeddings,
    _storage_dtype_for_precision,
)
from .sql import (
    PAPER_ROW_QUERY_SQL_TEMPLATE,
    PAPERS_TABLE_CREATE_SQL,
    REPLACEMENT_JOURNAL_TABLE_CREATE_SQL,
    _chunked,
    _decode_paper_row,
    _metadata_table_create_sql,
    _parse_json_list,
    _safe_json_list,
)

logger = logging.getLogger(__name__)


class _H5LayoutMixin:
    """SQLite schema and HDF5 layout management for :class:`EmbeddingCache`."""

    def _init_db(self) -> None:
        """Create and initialize the metadata cache schema when needed."""
        with self._connect_db() as conn:
            conn.execute(PAPERS_TABLE_CREATE_SQL)

            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(papers)").fetchall()
            }
            if "row_idx" not in columns:
                conn.execute("ALTER TABLE papers ADD COLUMN row_idx INTEGER")
            if "authors_json" not in columns:
                conn.execute("ALTER TABLE papers ADD COLUMN authors_json TEXT")
            if "categories_json" not in columns:
                conn.execute("ALTER TABLE papers ADD COLUMN categories_json TEXT")
            if "venue" not in columns:
                conn.execute("ALTER TABLE papers ADD COLUMN venue TEXT")
            if "arxiv_id" not in columns:
                conn.execute("ALTER TABLE papers ADD COLUMN arxiv_id TEXT")
            if "doi" not in columns:
                conn.execute("ALTER TABLE papers ADD COLUMN doi TEXT")
            if "chronology_key" not in columns:
                # Layout repair retains this SQLite table while replacing only
                # vector payloads, so columns must be current before it stamps
                # the repaired namespace with the current schema version.
                conn.execute("ALTER TABLE papers ADD COLUMN chronology_key INTEGER")

            conn.execute("DROP INDEX IF EXISTS idx_papers_text_hash")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_row_idx ON papers(row_idx)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_chronology_key "
                "ON papers(chronology_key)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_paper_id_nocase "
                "ON papers(paper_id COLLATE NOCASE)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id_nocase "
                "ON papers(arxiv_id COLLATE NOCASE)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_doi_nocase "
                "ON papers(doi COLLATE NOCASE)"
            )
            conn.execute(_metadata_table_create_sql())
            conn.execute(REPLACEMENT_JOURNAL_TABLE_CREATE_SQL)

            # Contract values are seeded, never re-stamped, here: overwriting them
            # would make _assert_runtime_cache_consistency compare the runtime
            # against itself. Absent keys seed as a fresh namespace would; stored
            # ones survive to be compared, and are restamped once validated or
            # repaired. Physical compression and hydration state likewise survive
            # process restarts.
            self._set_cache_metadata(
                conn,
                {
                    **self._runtime_contract_values(),
                    HYDRATION_COMPLETE_KEY: "0",
                    HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY: "",
                    HYDRATION_RECONCILED_CACHE_ROWS_KEY: "",
                    MODEL_FINGERPRINT_KEY: "",
                },
                preserve_existing=True,
            )

    def _collect_namespace_payload_stats_locked(
        self, *, strict: bool = False
    ) -> CacheNamespacePayloadStats:
        """Collect namespace payload stats while cache lock is held.

        Best-effort counts are only for logging an already requested clear.
        Hydration decisions must use strict inspection through ``payload_stats``.

        :param bool strict: Propagate storage errors instead of substituting zeroes.
        :return CacheNamespacePayloadStats: Snapshot of files/rows/hydration metadata.
        """
        file_count = 0
        size_bytes = 0
        existing_paths = set()
        for payload_path in (self.db_path, self.h5_path):
            try:
                payload_stat = payload_path.stat()
            except FileNotFoundError:
                continue
            except OSError:
                if strict:
                    raise
                continue
            existing_paths.add(payload_path)
            if not stat.S_ISREG(payload_stat.st_mode):
                continue
            file_count += 1
            size_bytes += int(payload_stat.st_size)

        sqlite_rows = 0
        hydration_complete = False
        hydration_split: str | None = None
        hydration_corpus_size: str | None = None
        hydration_dataset_source: str | None = None
        if self.db_path in existing_paths:
            try:
                with self._connect_db() as conn:
                    if strict:
                        self._recover_pending_replacements_with_connection_locked(conn)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM papers")
                    sqlite_rows = int(cursor.fetchone()[0])
                    metadata = self._load_cache_metadata(conn)
                    hydration_complete = (
                        metadata.get(HYDRATION_COMPLETE_KEY, "0") == "1"
                    )
                    hydration_split = (
                        str(metadata.get(HYDRATION_SPLIT_KEY, "")).strip() or None
                    )
                    hydration_corpus_size = (
                        str(metadata.get(HYDRATION_CORPUS_SIZE_KEY, "")).strip() or None
                    )
                    hydration_dataset_source = (
                        str(metadata.get(HYDRATION_DATASET_SOURCE_KEY, "")).strip()
                        or None
                    )
            except (OSError, sqlite3.DatabaseError):
                if strict:
                    raise
                sqlite_rows = 0

        embedding_rows = 0
        if self.h5_path in existing_paths:
            try:
                with h5py.File(self.h5_path, "r") as h5:
                    embeddings = self._get_embeddings_dataset(h5)
                    if embeddings is not None:
                        embedding_rows = int(embeddings.shape[0])
            except (OSError, ValueError):
                if strict:
                    raise
                embedding_rows = 0

        return CacheNamespacePayloadStats(
            file_count=file_count,
            size_bytes=size_bytes,
            sqlite_rows=sqlite_rows,
            embedding_rows=embedding_rows,
            hydration_complete=hydration_complete,
            hydration_split=hydration_split,
            hydration_corpus_size=hydration_corpus_size,
            hydration_dataset_source=hydration_dataset_source,
        )

    @staticmethod
    def _set_cache_metadata(
        conn: sqlite3.Connection,
        values: dict[str, object],
        *,
        preserve_existing: bool = False,
    ) -> None:
        """Write a group of cache metadata values with one conflict policy.

        :param sqlite3.Connection conn: Open SQLite connection.
        :param Dict[str, object] values: Metadata values keyed by cache field name.
        :param bool preserve_existing: Insert only missing keys when ``True``.
        :return None: This method mutates DB state in-place.
        """
        if not values:
            return
        statement = (
            "INSERT OR IGNORE INTO cache_metadata (key, value) VALUES (?, ?)"
            if preserve_existing
            else """
                INSERT INTO cache_metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """
        )
        conn.executemany(
            statement,
            ((str(key), str(value)) for key, value in values.items()),
        )

    @staticmethod
    def _load_cache_metadata(conn: sqlite3.Connection) -> dict[str, str]:
        """Load cache metadata table into an in-memory mapping.

        :param sqlite3.Connection conn: Open SQLite connection.
        :return Dict[str, str]: Metadata key-value mapping.
        """
        rows = conn.execute("SELECT key, value FROM cache_metadata").fetchall()
        return {str(key): str(value) for key, value in rows}

    @staticmethod
    def _metadata_value_from_h5_attr(value: Any) -> str:
        """Normalize HDF5 attribute values to comparable metadata strings.

        :param Any value: Raw HDF5 attribute payload.
        :return str: Normalized string representation.
        """
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.decode("utf-8", errors="replace")
        if isinstance(value, np.generic):
            value = value.item()
        return str(value)

    def _runtime_contract_values(self) -> dict[str, object]:
        """Return the canonical values shared by SQLite and HDF5 metadata.

        :return Dict[str, object]: Runtime cache-contract values keyed by field name.
        """
        return {
            SCHEMA_VERSION_KEY: EMBEDDING_CACHE_SCHEMA_VERSION,
            H5_LAYOUT_KEY: H5_LAYOUT_MATRIX_VERSION,
            STORAGE_PRECISION_KEY: self.storage_precision,
            SOURCE_TORCH_DTYPE_KEY: self.source_torch_dtype,
            EMBEDDING_VECTOR_DTYPE_KEY: self.embedding_vector_dtype,
            TEXT_FORMATTER_FINGERPRINT_KEY: self.text_formatter_fingerprint,
            CALIBRATION_SAMPLE_SIZE_KEY: self.calibration_sample_size,
            BINARY_PREFILTER_ENABLED_KEY: int(self.binary_prefilter),
            COMPRESSION_FILTER_KEY: self._effective_compression,
            COMPRESSION_LEVEL_KEY: self._effective_compression_level,
        }

    def _assert_runtime_cache_consistency(
        self,
        conn: sqlite3.Connection,
        h5_file: h5py.File,
        embeddings_dataset: h5py.Dataset | None,
        *,
        fail_mode: str,
        check_row_mapping: bool = True,
    ) -> None:
        """Assert that metadata/attrs/datasets agree on active runtime semantics.

        :param sqlite3.Connection conn: Open SQLite connection for metadata table.
        :param h5py.File h5_file: Open HDF5 cache handle.
        :param Optional[h5py.Dataset] embeddings_dataset: Matrix dataset, or None before the first write.
        :param str fail_mode: ``"runtime"`` to fail-closed, ``"repair"`` to rebuild incompatible layouts.
        :param bool check_row_mapping: Whether to validate SQLite/HDF5 row coverage.
        :return None: Raises when metadata and payload state diverge.
        :raises RuntimeError: If row mappings are inconsistent or runtime checks fail.
        :raises ValueError: If a layout mismatch is found in repair mode.
        """
        expected = {
            key: self._metadata_value_from_h5_attr(value)
            for key, value in self._runtime_contract_values().items()
        }
        # Other live cache objects may enable or remove this derived index.
        # Its presence/encoding is checked separately from primary vector rows.
        expected.pop(BINARY_PREFILTER_ENABLED_KEY)
        # Compute dtype is provenance, not identity, and is deliberately absent
        # from the namespace so hosts resolving different dtypes share one cache.
        # Comparing it here would make each host wipe the other's vectors on open.
        # The stored value survives as a record of what created the namespace.
        expected.pop(SOURCE_TORCH_DTYPE_KEY)
        if self.storage_precision != "int8":
            expected.pop(CALIBRATION_SAMPLE_SIZE_KEY)
        if embeddings_dataset is None:
            for key in (
                COMPRESSION_FILTER_KEY,
                COMPRESSION_LEVEL_KEY,
            ):
                expected.pop(key)
        metadata = self._load_cache_metadata(conn)

        def _fail(message: str, *, layout_mismatch: bool = True) -> None:
            """Raise a layout or row-mapping consistency error.

            :param str message: Observed inconsistency.
            :param bool layout_mismatch: Whether the persisted layout needs rebuilding.
            :return None: Always raises the mode-appropriate exception.
            """
            detail = f"Embedding cache integrity error: {message}."
            if fail_mode == "runtime" or not layout_mismatch:
                raise RuntimeError(
                    f"{detail} Rebuild this cache namespace to restore consistency."
                )
            if fail_mode == "repair":
                raise _EmbeddingCacheLayoutError(detail)
            raise ValueError(f"Unknown fail_mode={fail_mode!r}")

        for key, expected_value in expected.items():
            actual_value = metadata.get(key)
            if actual_value != expected_value:
                _fail(
                    f"metadata key {key!r} mismatch "
                    f"({actual_value!r} != {expected_value!r})"
                )
            h5_value = self._metadata_value_from_h5_attr(h5_file.attrs.get(key))
            if h5_value != expected_value:
                _fail(
                    f"HDF5 attr {key!r} mismatch ({h5_value!r} != {expected_value!r})"
                )

        if embeddings_dataset is None:
            return

        target_dtype = _storage_dtype_for_precision(self.storage_precision)
        if np.dtype(embeddings_dataset.dtype) != np.dtype(target_dtype):
            _fail(
                f"embeddings dataset dtype mismatch "
                f"({embeddings_dataset.dtype} != {target_dtype})"
            )
        if str(embeddings_dataset.compression or "") != self._effective_compression:
            _fail(
                "embeddings dataset compression mismatch "
                f"({embeddings_dataset.compression!r} != "
                f"{self._effective_compression!r})"
            )
        if self._effective_compression != "lzf":
            actual_compression_level = embeddings_dataset.compression_opts
            if int(actual_compression_level) != int(self._effective_compression_level):
                _fail(
                    "embeddings dataset compression level mismatch "
                    f"({actual_compression_level!r} != "
                    f"{self._effective_compression_level!r})"
                )

        if not check_row_mapping:
            return

        row_count = int(embeddings_dataset.shape[0])
        if fail_mode == "runtime":
            # Separate MIN/MAX subqueries each use the row_idx index endpoint.
            # Full coverage is checked once on open, not for every hydration batch.
            first_row, last_row = conn.execute(
                "SELECT (SELECT MIN(row_idx) FROM papers), "
                "(SELECT MAX(row_idx) FROM papers)"
            ).fetchone()
            expected_bounds = (0, row_count - 1) if row_count else (None, None)
            if (first_row, last_row) != expected_bounds:
                _fail(
                    "embedding row mapping mismatch "
                    f"(metadata bounds={(first_row, last_row)}, expected={expected_bounds})",
                    layout_mismatch=False,
                )
            return

        paper_rows = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        if paper_rows != row_count:
            _fail(
                "embedding row mapping mismatch "
                f"(metadata rows={paper_rows}, embedding rows={row_count})",
                layout_mismatch=False,
            )
        valid_rows, distinct_rows = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT row_idx)
            FROM papers
            WHERE row_idx IS NOT NULL
              AND row_idx >= 0
              AND row_idx < ?
            """,
            (row_count,),
        ).fetchone()
        if int(valid_rows) != row_count or int(distinct_rows) != row_count:
            _fail(
                "embedding row_idx coverage mismatch "
                f"(valid={int(valid_rows)}, distinct={int(distinct_rows)}, expected={row_count})",
                layout_mismatch=False,
            )

    def _reset_effective_compression(self) -> None:
        """Reset physical-layout settings to the originally requested codec.

        :return None: Updates effective compression state in-place.
        """
        self._effective_compression = self.compression
        self._effective_compression_level = self.compression_level

    @classmethod
    def _keeps_recorded_source_dtype(cls, stored_value: Any) -> bool:
        """Return whether an already-recorded source dtype must be left alone.

        The source compute dtype is provenance rather than identity, so it is
        never compared on open (see :meth:`_assert_runtime_cache_consistency`).
        That only makes it useful if it keeps naming the runtime whose vectors
        are actually stored: restamping it would let any later opener — one that
        reads nothing and exits included — claim authorship of another dtype's
        vectors. Every non-blank recorded value therefore outranks the opening
        runtime, in both the SQLite and HDF5 witnesses; the paths that discard
        the vectors clear it so the rebuilding runtime records its own.

        :param Any stored_value: Value already recorded in metadata or HDF5 attrs.
        :return bool: ``True`` when the stored value must not be overwritten.
        """
        return bool(cls._metadata_value_from_h5_attr(stored_value).strip())

    @staticmethod
    def _clear_recorded_source_dtype(conn: sqlite3.Connection) -> None:
        """Forget which runtime created a namespace that no longer holds vectors.

        Called only where the persisted vectors are gone, so the next runtime to
        write any is recorded as the creator instead of a discarded build.

        :param sqlite3.Connection conn: Open SQLite connection.
        :return None: Blanks the recorded source dtype in-place.
        """
        _H5LayoutMixin._set_cache_metadata(conn, {SOURCE_TORCH_DTYPE_KEY: ""})

    def _persist_runtime_contract_metadata(self, conn: sqlite3.Connection) -> None:
        """Stamp the active runtime cache contract into SQLite metadata.

        Only legitimate once the persisted namespace is known to match this
        runtime: either the consistency check passed, the namespace was rebuilt,
        or it holds no vectors yet. The source compute dtype is the one
        exception and is preserved per :meth:`_keeps_recorded_source_dtype`.

        :param sqlite3.Connection conn: Open SQLite connection.
        :return None: Updates runtime contract metadata in-place.
        """
        values = self._runtime_contract_values()
        metadata = self._load_cache_metadata(conn)
        if self._keeps_recorded_source_dtype(metadata.get(SOURCE_TORCH_DTYPE_KEY)):
            values.pop(SOURCE_TORCH_DTYPE_KEY)
        self._set_cache_metadata(conn, values)

    def _adopt_existing_dataset_compression(self, dataset: h5py.Dataset) -> None:
        """Adopt immutable compression layout from an existing embedding matrix.

        Compression changes storage layout but not embedding semantics. A requested
        codec therefore applies only when a matrix is first created or rebuilt.

        :param h5py.Dataset dataset: Existing embeddings matrix.
        :return None: Updates effective compression state in-place.
        :raises ValueError: If the persisted physical layout is unsupported.
        """
        compression = str(dataset.compression or "").strip().lower()
        if compression not in _COMPRESSION_FILTERS:
            raise _EmbeddingCacheLayoutError(
                "incompatible embedding cache compression filter: "
                f"{dataset.compression!r}"
            )

        if compression == "lzf":
            if dataset.compression_opts is not None:
                raise _EmbeddingCacheLayoutError(
                    "incompatible lzf embedding cache compression options: "
                    f"{dataset.compression_opts!r}"
                )
            compression_level = 0
        else:
            compression_options = dataset.compression_opts
            if compression_options is None:
                raise _EmbeddingCacheLayoutError(
                    "incompatible gzip embedding cache: missing compression level"
                )
            compression_level = int(compression_options)

        if (
            compression != self.compression
            or compression_level != self.compression_level
        ):
            logger.debug(
                "Using existing embedding cache compression %s level %d for %s; "
                "requested %s level %d applies after the cache is rebuilt.",
                compression,
                compression_level,
                self.h5_path,
                self.compression,
                self.compression_level,
            )
        self._effective_compression = compression
        self._effective_compression_level = compression_level

    def _ensure_h5_layout(self) -> None:
        """Recover interrupted appends and rebuild proven incompatible layouts.

        :return None: Preserves valid payloads and propagates IO/open failures.
        """
        if not path_exists(self.h5_path):
            with self._connect_db() as conn:
                pending_replacement = conn.execute(
                    "SELECT 1 FROM replacement_journal LIMIT 1"
                ).fetchone()
                if pending_replacement is not None:
                    raise RuntimeError(
                        "Embedding cache recovery error: replacement journal exists "
                        "but the embedding matrix is missing. Existing cache files "
                        "were preserved."
                    )
                self._reset_effective_compression()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                paper_rows = int(cursor.fetchone()[0])
                metadata = self._load_cache_metadata(conn)
                has_hydration_markers = bool(
                    str(metadata.get(HYDRATION_COMPLETE_KEY, "0")) == "1"
                    or str(metadata.get(HYDRATION_DATASET_SOURCE_KEY, "")).strip()
                    or str(metadata.get(HYDRATION_SPLIT_KEY, "")).strip()
                    or str(metadata.get(HYDRATION_CORPUS_SIZE_KEY, "")).strip()
                )
                if paper_rows > 0 or has_hydration_markers:
                    logger.warning(
                        "Embedding matrix %s is missing; clearing stale SQLite metadata "
                        "and hydration markers.",
                        self.h5_path,
                    )
                    conn.execute("DELETE FROM papers")
                    self._reset_hydration_metadata(conn)
                # A missing matrix means the namespace holds no vectors at all,
                # so whichever runtime writes the first ones creates it — not
                # whoever happened to create this SQLite file.
                self._clear_recorded_source_dtype(conn)
                self._persist_runtime_contract_metadata(conn)
                return

        try:
            with (
                self._connect_db() as conn,
                h5py.File(self.h5_path, "a") as h5,
            ):
                self._recover_pending_replacements_locked(
                    conn=conn,
                    h5_file=h5,
                    validate_runtime_contract=False,
                )
                dataset = self._get_embeddings_dataset(h5)
                if dataset is None:
                    if len(h5) == 0 and len(h5.attrs) == 0:
                        # An interrupted first encode leaves the file this
                        # namespace creates before writing anything: no datasets
                        # and no schema attrs. That is an empty cache to fill in,
                        # not a layout this runtime can prove incompatible.
                        self._discard_vectorless_row_mappings(conn)
                        # Nothing in this file was ever written, so the run that
                        # fills it is the creator on record.
                        self._clear_recorded_source_dtype(conn)
                        self._persist_runtime_contract_metadata(conn)
                        return
                    if (
                        self.storage_precision == "int8"
                        and set(h5) == {CALIBRATION_RANGES_DATASET_NAME}
                        and conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
                        == 0
                    ):
                        self._require_calibration_ranges(h5)
                        self._assert_runtime_cache_consistency(
                            conn=conn,
                            h5_file=h5,
                            embeddings_dataset=None,
                            fail_mode="repair",
                        )
                        self._persist_runtime_contract_metadata(conn)
                        return
                    raise _EmbeddingCacheLayoutError(
                        "incompatible embedding cache layout"
                    )
                self._adopt_existing_dataset_compression(dataset)

                embedding_dim = int(dataset.shape[1])
                if self.storage_precision == "int8":
                    if CALIBRATION_RANGES_DATASET_NAME not in h5:
                        raise _EmbeddingCacheLayoutError(
                            "missing int8 calibration ranges"
                        )
                    self._require_calibration_ranges(h5, embedding_dim)

                # Prefilter state is auxiliary; toggling it does not change vector
                # rows, so both witnesses adopt it instead of failing the check.
                h5.attrs.modify(
                    BINARY_PREFILTER_ENABLED_KEY, int(self.binary_prefilter)
                )
                self._set_cache_metadata(
                    conn,
                    {BINARY_PREFILTER_ENABLED_KEY: int(self.binary_prefilter)},
                )
                embedding_rows = self._recover_trailing_rows(
                    conn=conn,
                    h5_file=h5,
                    embeddings_dataset=dataset,
                )
                self._assert_runtime_cache_consistency(
                    conn=conn,
                    h5_file=h5,
                    embeddings_dataset=dataset,
                    fail_mode="repair",
                )

                binary_dataset = h5.get(BINARY_INDEX_DATASET_NAME)
                if (
                    binary_dataset is not None
                    and not self._is_binary_dataset_compatible(
                        binary_dataset=binary_dataset,
                        embedding_dim=embedding_dim,
                        embedding_rows=embedding_rows,
                    )
                ):
                    logger.warning(
                        "Binary index dataset in %s is incompatible with embedding "
                        "matrix shape; dropping stale binary index.",
                        self.h5_path,
                    )
                    del h5[BINARY_INDEX_DATASET_NAME]

                self._ensure_binary_dataset(h5, embedding_dim)
        except _EmbeddingCacheLayoutError as exc:
            logger.warning(
                "REBUILDING EMBEDDING CACHE — please hang tight. "
                "Embeddings will be regenerated automatically; this may take a while. "
                "No action is needed.\n"
                "Cache %s is incompatible with current schema. Reason: %s",
                self.h5_path,
                exc,
            )
            self.h5_path.unlink(missing_ok=True)
            self._reset_effective_compression()
            with self._connect_db() as conn:
                conn.execute("DELETE FROM papers")
                self._reset_hydration_metadata(conn)
                # The rebuilt namespace keeps none of the discarded vectors, so
                # this runtime — not the one that created them — is its creator.
                self._clear_recorded_source_dtype(conn)
                self._persist_runtime_contract_metadata(conn)
            return

        with self._connect_db() as conn:
            self._persist_runtime_contract_metadata(conn)

    @staticmethod
    def _discard_vectorless_row_mappings(conn: sqlite3.Connection) -> None:
        """Drop row mappings left behind by a namespace holding no vectors.

        :param sqlite3.Connection conn: Open SQLite connection for the namespace.
        :return None: Deletes unusable mappings and marks hydration incomplete.
        """
        paper_rows = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        if paper_rows == 0:
            return
        logger.warning(
            "Recovering embedding cache by removing %d SQLite mapping(s) whose "
            "vectors were never persisted; the embedding matrix is empty.",
            paper_rows,
        )
        conn.execute("DELETE FROM papers")
        _H5LayoutMixin._set_cache_metadata(conn, {HYDRATION_COMPLETE_KEY: "0"})

    @staticmethod
    def _reset_hydration_metadata(conn: sqlite3.Connection) -> None:
        """Reset hydration metadata keys to an incomplete state.

        :param sqlite3.Connection conn: Open SQLite connection.
        :return None: Mutates metadata table in-place.
        """
        _H5LayoutMixin._set_cache_metadata(
            conn,
            {
                HYDRATION_DATASET_SOURCE_KEY: "",
                HYDRATION_SPLIT_KEY: "",
                HYDRATION_CORPUS_SIZE_KEY: "",
                HYDRATION_COMPLETE_KEY: "0",
                CORPUS_METADATA_VERSION_KEY: "",
                HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY: "",
                HYDRATION_RECONCILED_CACHE_ROWS_KEY: "",
            },
        )

    @staticmethod
    def _metadata_tuple(
        paper_id: str,
        metadata: dict[str, object],
        text_hash: str,
        embedding_dim: int,
        row_idx: int,
    ) -> tuple[Any, ...]:
        """Build metadata row tuple for SQLite upsert.

        :param str paper_id: Paper identifier.
        :param Dict[str, object] metadata: Paper metadata payload.
        :param str text_hash: Deterministic hash for encoded text.
        :param int embedding_dim: Embedding vector width.
        :param int row_idx: Row index inside matrix dataset.
        :return Tuple[Any, ...]: SQLite upsert tuple matching ``papers`` columns.
        """
        title, abstract, year, *remaining = _H5LayoutMixin._normalized_metadata_fields(
            metadata
        )
        return (
            paper_id,
            title,
            abstract,
            year,
            text_hash,
            embedding_dim,
            row_idx,
            *remaining,
            # Derived from the primary key rather than the arxiv_id column, which
            # is empty for rows whose paper_id is still a parseable arXiv ID.
            encode_arxiv_id_chronology_key(paper_id),
        )

    @staticmethod
    def _normalized_metadata_fields(
        metadata: dict[str, object],
    ) -> tuple[str, str, int | None, str, str, str, str, str]:
        """Normalize metadata fields to stable cache representations.

        :param Dict[str, object] metadata: Paper metadata payload.
        :return Tuple[str, str, Optional[int], str, str, str, str, str]:
            Normalized title/abstract/year/authors/categories and venue/arXiv/DOI fields.
        """
        title = str(metadata.get("title", "") or "")
        abstract = str(metadata.get("abstract", "") or "")
        year_raw = metadata.get("year")
        year = None
        if year_raw is not None:
            try:
                year = int(year_raw)
            except (TypeError, ValueError):
                year = None

        authors_json = _safe_json_list(metadata.get("authors", []))
        categories_json = _safe_json_list(metadata.get("categories", []))
        venue = str(metadata.get("venue", "") or "").strip()
        arxiv_id = str(metadata.get("arxiv_id", "") or "").strip()
        doi = str(metadata.get("doi", "") or "").strip()
        return (
            title,
            abstract,
            year,
            authors_json,
            categories_json,
            venue,
            arxiv_id,
            doi,
        )

    @staticmethod
    def _metadata_fields_changed(
        existing_row: dict[str, Any], metadata: dict[str, object]
    ) -> bool:
        """Return whether cached non-vector metadata differs from incoming payload.

        :param Dict[str, Any] existing_row: Existing cached metadata row.
        :param Dict[str, object] metadata: Incoming metadata payload.
        :return bool: ``True`` when a metadata-only refresh is required.
        """
        current = (
            str(existing_row.get("title", "") or ""),
            str(existing_row.get("abstract", "") or ""),
            (
                int(existing_row["year"])
                if existing_row.get("year") is not None
                else None
            ),
            _safe_json_list(_parse_json_list(existing_row.get("authors_json"))),
            _safe_json_list(_parse_json_list(existing_row.get("categories_json"))),
            str(existing_row.get("venue", "") or "").strip(),
            str(existing_row.get("arxiv_id", "") or "").strip(),
            str(existing_row.get("doi", "") or "").strip(),
        )
        expected = _H5LayoutMixin._normalized_metadata_fields(metadata)
        return current != expected

    @staticmethod
    def _metadata_refresh_tuple(
        paper_id: str, metadata: dict[str, object]
    ) -> tuple[Any, ...]:
        """Build SQL update tuple for metadata-only refresh paths.

        :param str paper_id: Paper identifier.
        :param Dict[str, object] metadata: Incoming metadata payload.
        :return Tuple[Any, ...]: Tuple for metadata UPDATE query.
        """
        # The chronology key rides along so both write paths agree on it; it is
        # keyed off the immutable primary key, so rewriting it is idempotent.
        return (
            *_H5LayoutMixin._normalized_metadata_fields(metadata),
            encode_arxiv_id_chronology_key(paper_id),
            paper_id,
        )

    def _set_h5_attrs(self, h5_file: h5py.File) -> None:
        """Write schema/layout metadata attrs to an open HDF5 file.

        :param h5py.File h5_file: Open cache file handle.
        :return None: Mutates HDF5 attrs in-place.
        """
        for key, value in self._runtime_contract_values().items():
            if key not in h5_file.attrs:
                h5_file.attrs[key] = value
                continue
            current_value = self._metadata_value_from_h5_attr(h5_file.attrs[key])
            if key == SOURCE_TORCH_DTYPE_KEY and self._keeps_recorded_source_dtype(
                current_value
            ):
                continue
            expected_value = self._metadata_value_from_h5_attr(value)
            if current_value != expected_value:
                h5_file.attrs.modify(key, value)

    def _load_existing_rows(
        self,
        conn: sqlite3.Connection,
        paper_ids: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """Fetch existing metadata rows for target paper IDs.

        :param sqlite3.Connection conn: Open SQLite connection.
        :param Sequence[str] paper_ids: Paper IDs to look up.
        :return Dict[str, Dict[str, Any]]: Mapping of paper ID to cached metadata payload.
        """
        existing_rows: dict[str, dict[str, Any]] = {}
        for row in self._query_paper_rows(conn, paper_ids, lookup_column="paper_id"):
            decoded = _decode_paper_row(row, parse_json_lists=False)
            paper_id = decoded.pop("paper_id")
            existing_rows[str(paper_id)] = decoded

        return existing_rows

    @staticmethod
    def _query_paper_rows(
        conn: sqlite3.Connection,
        lookup_values: Sequence[Any],
        *,
        lookup_column: str,
        case_insensitive: bool = False,
    ) -> Iterator[tuple[Any, ...]]:
        """Yield common paper rows for batched SQLite key lookups.

        :param sqlite3.Connection conn: Open SQLite connection.
        :param Sequence[Any] lookup_values: Values for the selected lookup column.
        :param str lookup_column: ``papers`` column used for the ``IN`` lookup.
        :param bool case_insensitive: Compare text values with SQLite ``NOCASE``.
        :return Iterator[Tuple[Any, ...]]: Rows in the shared paper-column layout.
        :raises ValueError: If the requested lookup column is not supported.
        """
        if lookup_column not in _PAPER_ROW_LOOKUP_COLUMNS:
            raise ValueError(f"Unsupported paper-row lookup column: {lookup_column}")

        for value_chunk in _chunked(lookup_values, SQLITE_QUERY_BATCH_SIZE):
            placeholders = ",".join("?" for _ in value_chunk)
            lookup_expression = (
                f"{lookup_column} COLLATE NOCASE" if case_insensitive else lookup_column
            )
            query = PAPER_ROW_QUERY_SQL_TEMPLATE.format(
                columns=_PAPER_ROW_COLUMNS,
                lookup_column=lookup_expression,
                placeholders=placeholders,
            )
            for row in conn.execute(query, value_chunk):
                if row[-1] != 1:
                    raise RuntimeError(
                        "Embedding cache integrity error: row_idx coverage mismatch "
                        f"for accessed paper {row[0]!r}."
                    )
                yield row[:-1]

    @staticmethod
    def _get_embeddings_dataset(h5_file: h5py.File) -> h5py.Dataset | None:
        """Return matrix embedding dataset when available.

        :param h5py.File h5_file: Open HDF5 cache handle.
        :return Optional[h5py.Dataset]: 2D embedding dataset or ``None``.
        """
        dataset = h5_file.get(EMBEDDINGS_DATASET_NAME)
        if dataset is None:
            return None
        if dataset.ndim != 2:
            raise _EmbeddingCacheLayoutError(
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
        dataset = self._ensure_matrix_dataset(
            h5_file,
            name=EMBEDDINGS_DATASET_NAME,
            width=embedding_dim,
            dtype=_storage_dtype_for_precision(self.storage_precision),
        )
        assert dataset is not None
        return dataset

    def _ensure_binary_dataset(
        self, h5_file: h5py.File, embedding_dim: int
    ) -> h5py.Dataset | None:
        """Create or validate binary-index dataset for int8 cache search.

        :param h5py.File h5_file: Open HDF5 file handle.
        :param int embedding_dim: Embedding width.
        :return Optional[h5py.Dataset]: Binary-index dataset when enabled.
        """
        if not self.binary_prefilter:
            if BINARY_INDEX_DATASET_NAME in h5_file:
                del h5_file[BINARY_INDEX_DATASET_NAME]
            return None

        packed_dim = (int(embedding_dim) + 7) // 8
        binary_dataset = self._ensure_matrix_dataset(
            h5_file,
            name=BINARY_INDEX_DATASET_NAME,
            width=packed_dim,
            dtype=np.uint8,
            enabled=self.binary_prefilter,
        )
        if binary_dataset is None:
            return None

        embeddings_dataset = self._get_embeddings_dataset(h5_file)
        embedding_rows = (
            int(embeddings_dataset.shape[0]) if embeddings_dataset is not None else 0
        )
        if (
            int(binary_dataset.shape[0]) != embedding_rows
            or binary_dataset.attrs.get(BINARY_INDEX_ENCODING_KEY)
            != BINARY_INDEX_ENCODING
        ):
            if embedding_rows:
                logger.warning(
                    "Rebuilding binary index in %s from %d persisted embedding row(s).",
                    self.h5_path,
                    embedding_rows,
                )
            binary_dataset.attrs.modify(BINARY_INDEX_ENCODING_KEY, "")
            if embeddings_dataset is None:
                binary_dataset.resize((0, packed_dim))
            else:
                self._rebuild_binary_dataset(
                    h5_file=h5_file,
                    embeddings_dataset=embeddings_dataset,
                    binary_dataset=binary_dataset,
                )
            binary_dataset.attrs.modify(
                BINARY_INDEX_ENCODING_KEY, BINARY_INDEX_ENCODING
            )
        return binary_dataset

    def _rebuild_binary_dataset(
        self,
        h5_file: h5py.File,
        embeddings_dataset: h5py.Dataset,
        binary_dataset: h5py.Dataset,
    ) -> None:
        """Rebuild the binary prefilter from persisted int8 embeddings.

        :param h5py.File h5_file: Open HDF5 cache handle.
        :param h5py.Dataset embeddings_dataset: Persisted int8 embedding matrix.
        :param h5py.Dataset binary_dataset: Resizable uint8 binary-index matrix.
        :return None: Replaces all binary-index rows in-place.
        """
        if self.storage_precision != "int8":
            raise RuntimeError("Binary prefilter indexes require int8 storage.")

        row_count = int(embeddings_dataset.shape[0])
        packed_dim = (int(embeddings_dataset.shape[1]) + 7) // 8
        binary_dataset.resize((row_count, packed_dim))
        for start in range(0, row_count, EMBEDDING_DATASET_CHUNK_ROWS):
            end = min(start + EMBEDDING_DATASET_CHUNK_ROWS, row_count)
            stored_chunk = np.asarray(
                embeddings_dataset[start:end],
                dtype=np.int8,
            )
            float_chunk = self._dequantize_int8(h5_file, stored_chunk)
            binary_dataset[start:end] = _quantize_ubinary_embeddings(float_chunk)

    def _ensure_matrix_dataset(
        self,
        h5_file: h5py.File,
        *,
        name: str,
        width: int,
        dtype: np.dtype,
        enabled: bool = True,
    ) -> h5py.Dataset | None:
        """Create or validate an enabled resizable two-dimensional HDF5 matrix.

        :param h5py.File h5_file: Open HDF5 file handle.
        :param str name: Dataset name.
        :param int width: Required number of matrix columns.
        :param np.dtype dtype: Required matrix dtype.
        :param bool enabled: Whether the optional matrix is enabled.
        :return Optional[h5py.Dataset]: Valid matrix dataset, or ``None`` when disabled.
        """
        if not enabled:
            return None

        target_dtype = np.dtype(dtype)
        dataset = h5_file.get(name)
        if dataset is None:
            compression_kwargs = self._dataset_compression_kwargs()
            return h5_file.create_dataset(
                name,
                shape=(0, width),
                maxshape=(None, width),
                dtype=target_dtype,
                chunks=(EMBEDDING_DATASET_CHUNK_ROWS, width),
                shuffle=True,
                **compression_kwargs,
            )

        if dataset.ndim != 2:
            raise ValueError(f"Dataset '{name}' must be 2D.")
        if int(dataset.shape[1]) != int(width):
            raise ValueError(
                f"Dataset '{name}' width mismatch in cache: "
                f"{int(dataset.shape[1])} != {int(width)}"
            )
        if np.dtype(dataset.dtype) != target_dtype:
            raise ValueError(
                f"Dataset '{name}' dtype mismatch in cache: "
                f"{dataset.dtype} != {target_dtype}"
            )
        return dataset

    @staticmethod
    def _is_binary_dataset_compatible(
        binary_dataset: h5py.Dataset,
        embedding_dim: int,
        embedding_rows: int,
    ) -> bool:
        """Return whether the binary index has a complete encoding and valid shape.

        :param h5py.Dataset binary_dataset: Binary index dataset.
        :param int embedding_dim: Embedding vector dimension.
        :param int embedding_rows: Number of embedding rows in matrix dataset.
        :return bool: ``True`` when binary index encoding and shape are compatible.
        """
        if binary_dataset.attrs.get(BINARY_INDEX_ENCODING_KEY) != BINARY_INDEX_ENCODING:
            return False
        if binary_dataset.ndim != 2:
            return False
        if np.dtype(binary_dataset.dtype) != np.dtype(np.uint8):
            return False
        expected_cols = (int(embedding_dim) + 7) // 8
        if int(binary_dataset.shape[1]) != expected_cols:
            return False
        if int(binary_dataset.shape[0]) != int(embedding_rows):
            return False
        return True

    def _dataset_compression_kwargs(self) -> dict[str, Any]:
        """Build HDF5 dataset compression kwargs for active cache configuration.

        ``lzf`` does not accept ``compression_opts``. Other configured codecs keep
        the numeric level behavior used by existing cache settings.

        :return Dict[str, Any]: Keyword args passed into ``create_dataset``.
        """
        compression = str(self._effective_compression or "").strip()
        if not compression:
            return {}

        kwargs: dict[str, Any] = {"compression": compression}
        if compression.lower() != "lzf":
            kwargs["compression_opts"] = int(self._effective_compression_level)
        return kwargs
