"""The cache-miss/encode/commit ingestion pipeline for :class:`EmbeddingCache`.

Owns everything between "a caller asked for these papers" and "their vectors are
durable": the locked pre-encode scan that separates hits from misses, the
unlocked encode of the misses, quantization and saturation accounting, and the
second locked pass that journals replacements, appends new rows and commits the
SQLite mappings that point at them.

:class:`_IngestMixin` is mixed into :class:`~.store.EmbeddingCache` and reads the
following from its host: the ``storage_precision``, ``binary_prefilter`` and
``h5_path`` attributes; the locking and recovery surface
(``_cache_operation_lock``, ``_cache_lock``, ``_connect_db``,
``_recover_pending_replacements_locked``, ``_persist_replacement_journal``,
``_flush_h5_file``) from :mod:`.recovery`; the schema and dataset surface
(``_load_existing_rows``, ``_get_embeddings_dataset``,
``_ensure_embeddings_dataset``, ``_ensure_binary_dataset``, ``_set_h5_attrs``,
``_assert_runtime_cache_consistency``, ``_metadata_tuple``,
``_metadata_refresh_tuple``, ``_metadata_fields_changed``) from :mod:`.layout`;
``_require_calibration_ranges``, ``_dequantize_int8`` and
``_load_cached_embeddings`` from :mod:`.search`; and ``_report_progress`` plus
``_record_int8_saturation`` from :mod:`.store` itself.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from citemesh.text_batching import encode_texts, l2_normalize_embeddings

from ..model_profiles import compose_title_abstract_text
from .models import (
    PendingEmbeddingRecord,
    _CacheLookupPlan,
    _Int8Saturation,
    _VectorWritePlan,
)
from .quantization import (
    _count_int8_saturated_values,
    _quantize_int8_embeddings,
    _quantize_ubinary_embeddings,
    _storage_dtype_for_precision,
)
from .sql import PAPER_METADATA_REFRESH_SQL, PAPER_ROW_UPSERT_SQL


def _close_progress(iterator: Iterable[Any]) -> None:
    """Tear down a progress reporter that was consumed to completion.

    Reporters are generators whose display must be closed; a plain passthrough
    sequence has nothing to close.

    :param Iterable[Any] iterator: Value previously returned by the reporter.
    :return None: Closes the reporter when it supports closing.
    """
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


class _IngestMixin:
    """Cache-miss lookup, encoding, and durable commit for :class:`EmbeddingCache`."""

    def _process_embeddings(
        self,
        papers: Dict[str, Dict],
        model: Any,
        *,
        batch_size: int,
        show_progress: bool,
        text_builder: Optional[Callable[[Dict[str, object]], str]],
        return_embeddings: bool,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Coordinate one complete embedding cache operation with root clearing.

        :param Dict[str, Dict] papers: Mapping of paper ID to metadata payload.
        :param Any model: SentenceTransformer-compatible model exposing ``encode``.
        :param int batch_size: Batch size for model encoding.
        :param bool show_progress: Whether to display progress bars.
        :param Optional[Callable[[Dict[str, object]], str]] text_builder: Optional metadata->text formatter.
        :param bool return_embeddings: Whether to return float32 embedding payloads.
        :return Optional[Dict[str, np.ndarray]]: Embedding map when requested, else ``None``.
        """
        with self._cache_operation_lock():
            return self._process_embeddings_locked(
                papers,
                model,
                batch_size=batch_size,
                show_progress=show_progress,
                text_builder=text_builder,
                return_embeddings=return_embeddings,
            )

    def _process_embeddings_locked(
        self,
        papers: Dict[str, Dict],
        model: Any,
        *,
        batch_size: int,
        show_progress: bool,
        text_builder: Optional[Callable[[Dict[str, object]], str]],
        return_embeddings: bool,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Hydrate cache entries while the root operation lock is held.

        Runs three phases in order: a locked scan that separates hits from
        misses, an unlocked encode of the misses, and a second locked pass that
        quantizes, journals and commits the new rows. The lock, connection and
        HDF5 handle are opened here and handed to each step, so no step widens
        or shortens the window a caller sees.

        :param Dict[str, Dict] papers: Mapping of paper ID to metadata payload.
        :param Any model: SentenceTransformer-compatible model exposing ``encode``.
        :param int batch_size: Batch size for model encoding.
        :param bool show_progress: Whether to display progress bars.
        :param Optional[Callable[[Dict[str, object]], str]] text_builder: Optional metadata->text formatter.
        :param bool return_embeddings: Whether to return float32 embedding payloads.
        :return Optional[Dict[str, np.ndarray]]: Embedding map when requested, else ``None``.
        """
        if not papers:
            if return_embeddings:
                return {}
            return None

        builder = text_builder or compose_title_abstract_text
        items = list(papers.items())

        calibration_ranges: Optional[np.ndarray] = None
        with (
            self._cache_lock(),
            self._connect_db() as conn,
            h5py.File(self.h5_path, "a") as h5,
        ):
            lookup = self._scan_cache_for_hits(
                conn=conn,
                h5_file=h5,
                items=items,
                builder=builder,
                return_embeddings=return_embeddings,
                show_progress=show_progress,
            )
            if not lookup.papers_to_embed:
                if return_embeddings:
                    return lookup.cached_embeddings
                return None

            if self.storage_precision == "int8":
                calibration_ranges = self._require_calibration_ranges(h5_file=h5)

        embeddings_array = self._encode_pending_texts(
            model,
            lookup.papers_to_embed,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        embedding_dim = int(embeddings_array.shape[1])
        storage_embeddings, saturation = self._prepare_storage_embeddings(
            embeddings_array,
            calibration_ranges=calibration_ranges,
            embedding_dim=embedding_dim,
        )

        with (
            self._cache_lock(),
            self._connect_db() as conn,
            h5py.File(self.h5_path, "a") as h5,
        ):
            new_embeddings = self._commit_encoded_embeddings(
                conn=conn,
                h5_file=h5,
                papers_to_embed=lookup.papers_to_embed,
                embeddings_array=embeddings_array,
                storage_embeddings=storage_embeddings,
                calibration_ranges=calibration_ranges,
                saturation=saturation,
                embedding_dim=embedding_dim,
                return_embeddings=return_embeddings,
            )

        if return_embeddings:
            return {**lookup.cached_embeddings, **new_embeddings}
        return None

    def _scan_cache_for_hits(
        self,
        *,
        conn: sqlite3.Connection,
        h5_file: h5py.File,
        items: Sequence[Tuple[str, Dict]],
        builder: Callable[[Dict[str, object]], str],
        return_embeddings: bool,
        show_progress: bool,
    ) -> _CacheLookupPlan:
        """Separate cache hits from misses for one request, under the cache lock.

        Recovers any interrupted prior write first, then diffs each incoming
        payload against its stored row: a matching text hash pointing at a live
        matrix row is a hit, everything else becomes pending encode work. Hits
        whose non-vector metadata drifted are refreshed here, since that needs
        no encode pass.

        :param sqlite3.Connection conn: Open SQLite connection for the active namespace.
        :param h5py.File h5_file: Writable HDF5 handle for the active namespace.
        :param Sequence[Tuple[str, Dict]] items: Requested ``(paper_id, metadata)`` pairs.
        :param Callable[[Dict[str, object]], str] builder: Metadata-to-text formatter.
        :param bool return_embeddings: Whether cached vectors must be materialized.
        :param bool show_progress: Whether the scan may report progress.
        :return _CacheLookupPlan: Cached vectors plus the records still to encode.
        """
        self._recover_pending_replacements_locked(conn=conn, h5_file=h5_file)
        cursor = conn.cursor()
        existing_rows = self._load_existing_rows(
            conn, [paper_id for paper_id, _ in items]
        )
        embeddings_dataset = self._get_embeddings_dataset(h5_file)
        if embeddings_dataset is not None:
            self._assert_runtime_cache_consistency(
                conn=conn,
                h5_file=h5_file,
                embeddings_dataset=embeddings_dataset,
                fail_mode="runtime",
            )
        cached_limit = (
            int(embeddings_dataset.shape[0]) if embeddings_dataset is not None else 0
        )

        cached_rows: List[Tuple[str, int]] = []
        papers_to_embed: List[PendingEmbeddingRecord] = []
        metadata_updates_on_hit: List[Tuple[Any, ...]] = []

        iterator: Iterable[Tuple[str, Dict]] = self._report_progress(
            items, "Checking cache", enabled=show_progress and len(items) > 50
        )
        for paper_id, metadata in iterator:
            text = builder(metadata)
            text_hash = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
            existing_row = existing_rows.get(paper_id)
            row_idx = existing_row["row_idx"] if existing_row is not None else None

            if (
                existing_row is not None
                and existing_row["text_hash"] == text_hash
                and row_idx is not None
                and embeddings_dataset is not None
                and 0 <= row_idx < cached_limit
            ):
                if return_embeddings:
                    cached_rows.append((paper_id, row_idx))
                if self._metadata_fields_changed(existing_row, metadata):
                    metadata_updates_on_hit.append(
                        self._metadata_refresh_tuple(paper_id, metadata)
                    )
            else:
                papers_to_embed.append(
                    PendingEmbeddingRecord(
                        paper_id=paper_id,
                        metadata=dict(metadata),
                        text_hash=text_hash,
                        text=text,
                        row_idx=row_idx,
                    )
                )

        _close_progress(iterator)

        cached_embeddings: Dict[str, np.ndarray] = {}
        if return_embeddings and cached_rows and embeddings_dataset is not None:
            cached_embeddings = self._load_cached_embeddings(
                h5_file,
                embeddings_dataset,
                cached_rows,
            )
        self._refresh_hit_metadata(conn, cursor, metadata_updates_on_hit)
        return _CacheLookupPlan(
            cached_embeddings=cached_embeddings,
            papers_to_embed=papers_to_embed,
        )

    @staticmethod
    def _refresh_hit_metadata(
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        updates: Sequence[Tuple[Any, ...]],
    ) -> None:
        """Rewrite non-vector metadata for cache hits whose payload drifted.

        :param sqlite3.Connection conn: Open SQLite connection for the active namespace.
        :param sqlite3.Cursor cursor: Cursor owned by ``conn``.
        :param Sequence[Tuple[Any, ...]] updates: Refresh tuples for the UPDATE query.
        :return None: Commits the metadata-only refresh when there is work to do.
        """
        if not updates:
            return
        cursor.executemany(PAPER_METADATA_REFRESH_SQL, updates)
        conn.commit()

    @staticmethod
    def _encode_pending_texts(
        model: Any,
        papers_to_embed: Sequence[PendingEmbeddingRecord],
        *,
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Encode pending texts and reject any malformed model output.

        Runs with no lock held: encoding is the slow phase and must not block
        other processes reading this namespace.

        :param Any model: SentenceTransformer-compatible model exposing ``encode``.
        :param Sequence[PendingEmbeddingRecord] papers_to_embed: Records to encode.
        :param int batch_size: Batch size for model encoding.
        :param bool show_progress: Whether to display the encoder progress bar.
        :return np.ndarray: Validated ``(rows, dim)`` float embedding matrix.
        :raises ValueError: If the model returns a malformed or non-finite matrix.
        """
        texts = [record.text for record in papers_to_embed]
        embeddings_array = encode_texts(
            model,
            texts,
            batch_size=min(int(batch_size), len(texts)),
            show_progress_bar=show_progress,
        )
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        if embeddings_array.ndim != 2:
            raise ValueError(
                "Embedding model must return a 2-dimensional matrix, "
                f"got shape {embeddings_array.shape}."
            )
        if embeddings_array.shape[0] != len(papers_to_embed):
            raise ValueError(
                "Embedding model returned unexpected row count: "
                f"{embeddings_array.shape[0]} for {len(papers_to_embed)} papers."
            )
        if embeddings_array.shape[1] < 1:
            raise ValueError("Embedding model returned vectors with zero dimensions.")
        if not np.all(np.isfinite(embeddings_array)):
            raise ValueError("Embedding model returned non-finite values.")
        return embeddings_array

    def _prepare_storage_embeddings(
        self,
        embeddings_array: np.ndarray,
        *,
        calibration_ranges: Optional[np.ndarray],
        embedding_dim: int,
    ) -> Tuple[np.ndarray, _Int8Saturation]:
        """Convert encoded float vectors into this namespace's storage dtype.

        :param np.ndarray embeddings_array: Validated float embedding matrix.
        :param Optional[np.ndarray] calibration_ranges: Ranges captured before encoding.
        :param int embedding_dim: Embedding vector width.
        :return Tuple[np.ndarray, _Int8Saturation]: Storage matrix and clipping counts.
        :raises RuntimeError: If int8 storage has no persisted calibration ranges.
        :raises ValueError: If persisted ranges do not match the encoded width.
        """
        if self.storage_precision != "int8":
            storage_embeddings = np.asarray(
                embeddings_array,
                dtype=_storage_dtype_for_precision(self.storage_precision),
            )
            return storage_embeddings, _Int8Saturation(clipped_values=0, total_values=0)

        if calibration_ranges is None:
            raise RuntimeError(
                "Missing persisted int8 calibration ranges for cache writes."
            )
        if int(calibration_ranges.shape[1]) != embedding_dim:
            raise ValueError(
                "Calibration range dimension mismatch in cache: "
                f"{int(calibration_ranges.shape[1])} != {embedding_dim}"
            )
        clipped_value_count, clipped_total_value_count = _count_int8_saturated_values(
            embeddings_array, calibration_ranges
        )
        storage_embeddings = _quantize_int8_embeddings(
            embeddings_array, calibration_ranges
        )
        return storage_embeddings, _Int8Saturation(
            clipped_values=clipped_value_count,
            total_values=clipped_total_value_count,
        )

    def _commit_encoded_embeddings(
        self,
        *,
        conn: sqlite3.Connection,
        h5_file: h5py.File,
        papers_to_embed: Sequence[PendingEmbeddingRecord],
        embeddings_array: np.ndarray,
        storage_embeddings: np.ndarray,
        calibration_ranges: Optional[np.ndarray],
        saturation: _Int8Saturation,
        embedding_dim: int,
        return_embeddings: bool,
    ) -> Dict[str, np.ndarray]:
        """Persist one encoded batch while the cache lock is held.

        :param sqlite3.Connection conn: Open SQLite connection for the active namespace.
        :param h5py.File h5_file: Writable HDF5 handle for the active namespace.
        :param Sequence[PendingEmbeddingRecord] papers_to_embed: Encoded records.
        :param np.ndarray embeddings_array: Float matrix returned by the model.
        :param np.ndarray storage_embeddings: Matrix in this namespace's storage dtype.
        :param Optional[np.ndarray] calibration_ranges: Ranges captured before encoding.
        :param _Int8Saturation saturation: Clipping counts from quantization.
        :param int embedding_dim: Embedding vector width.
        :param bool return_embeddings: Whether float32 payloads must be returned.
        :return Dict[str, np.ndarray]: Embeddings for the newly written papers.
        """
        self._recover_pending_replacements_locked(conn=conn, h5_file=h5_file)
        cursor = conn.cursor()
        self._set_h5_attrs(h5_file)
        embeddings_dataset = self._ensure_embeddings_dataset(h5_file, embedding_dim)
        binary_dataset = self._ensure_binary_dataset(h5_file, embedding_dim)
        self._verify_calibration_and_record_saturation(
            h5_file,
            calibration_ranges=calibration_ranges,
            saturation=saturation,
            embedding_dim=embedding_dim,
        )
        binary_embeddings, returned_embeddings = self._derive_write_vectors(
            h5_file,
            storage_embeddings=storage_embeddings,
            embeddings_array=embeddings_array,
            return_embeddings=return_embeddings,
        )
        plan = self._plan_vector_writes(
            conn=conn,
            h5_file=h5_file,
            embeddings_dataset=embeddings_dataset,
            binary_dataset=binary_dataset,
            papers_to_embed=papers_to_embed,
            storage_embeddings=storage_embeddings,
            binary_embeddings=binary_embeddings,
            returned_embeddings=returned_embeddings,
            embedding_dim=embedding_dim,
            return_embeddings=return_embeddings,
        )
        self._apply_replacement_rows(
            conn=conn,
            embeddings_dataset=embeddings_dataset,
            binary_dataset=binary_dataset,
            replacement_rows=plan.replacement_rows,
        )
        self._append_new_rows(
            embeddings_dataset=embeddings_dataset,
            binary_dataset=binary_dataset,
            plan=plan,
            embedding_dim=embedding_dim,
        )
        self._commit_row_mappings(
            cursor=cursor,
            h5_file=h5_file,
            rows_to_upsert=plan.rows_to_upsert,
            replacement_rows=plan.replacement_rows,
        )
        return plan.new_embeddings

    def _verify_calibration_and_record_saturation(
        self,
        h5_file: h5py.File,
        *,
        calibration_ranges: Optional[np.ndarray],
        saturation: _Int8Saturation,
        embedding_dim: int,
    ) -> None:
        """Reject writes calibrated against ranges that changed during encoding.

        Quantization happened outside the lock, so the persisted ranges are
        re-read here and compared before any row is written.

        :param h5py.File h5_file: Writable HDF5 handle for the active namespace.
        :param Optional[np.ndarray] calibration_ranges: Ranges captured before encoding.
        :param _Int8Saturation saturation: Clipping counts from quantization.
        :param int embedding_dim: Embedding vector width.
        :return None: Records saturation telemetry when this batch clipped values.
        :raises RuntimeError: If persisted calibration ranges changed during encoding.
        """
        if self.storage_precision != "int8":
            return
        current_ranges = self._require_calibration_ranges(
            h5_file=h5_file, embedding_dim=embedding_dim
        )
        if calibration_ranges is None or not np.array_equal(
            current_ranges, calibration_ranges
        ):
            raise RuntimeError(
                "Int8 calibration ranges changed during encode; retry cache write."
            )
        if saturation.total_values > 0:
            self._record_int8_saturation(
                h5_file=h5_file,
                clipped_value_count=saturation.clipped_values,
                total_value_count=saturation.total_values,
            )

    def _derive_write_vectors(
        self,
        h5_file: h5py.File,
        *,
        storage_embeddings: np.ndarray,
        embeddings_array: np.ndarray,
        return_embeddings: bool,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Derive the binary index rows and the vectors handed back to callers.

        Callers must see the vectors this cache will return forever after, so
        int8 rows are round-tripped through storage before being handed back;
        otherwise the encoding run and every later hit disagree.

        :param h5py.File h5_file: Writable HDF5 handle for the active namespace.
        :param np.ndarray storage_embeddings: Matrix in this namespace's storage dtype.
        :param np.ndarray embeddings_array: Float matrix returned by the model.
        :param bool return_embeddings: Whether float32 payloads must be returned.
        :return Tuple[Optional[np.ndarray], np.ndarray]: Packed binary rows, when the
            prefilter is enabled, and the float32 vectors to return.
        """
        dequantized_embeddings = (
            self._dequantize_int8(h5_file, storage_embeddings)
            if self.storage_precision == "int8"
            and (self.binary_prefilter or return_embeddings)
            else None
        )
        binary_embeddings = (
            _quantize_ubinary_embeddings(dequantized_embeddings)
            if self.binary_prefilter and dequantized_embeddings is not None
            else None
        )
        returned_embeddings = (
            l2_normalize_embeddings(dequantized_embeddings)
            if return_embeddings and dequantized_embeddings is not None
            else embeddings_array
        )
        return binary_embeddings, returned_embeddings

    def _plan_vector_writes(
        self,
        *,
        conn: sqlite3.Connection,
        h5_file: h5py.File,
        embeddings_dataset: h5py.Dataset,
        binary_dataset: Optional[h5py.Dataset],
        papers_to_embed: Sequence[PendingEmbeddingRecord],
        storage_embeddings: np.ndarray,
        binary_embeddings: Optional[np.ndarray],
        returned_embeddings: np.ndarray,
        embedding_dim: int,
        return_embeddings: bool,
    ) -> _VectorWritePlan:
        """Sort each encoded record into a no-op, an in-place replacement or an append.

        Row mappings are re-read under this second lock because another process
        may have committed the same papers while this one was encoding: a record
        whose stored text hash now matches is already durable and only needs its
        metadata restamped.

        :param sqlite3.Connection conn: Open SQLite connection for the active namespace.
        :param h5py.File h5_file: Writable HDF5 handle for the active namespace.
        :param h5py.Dataset embeddings_dataset: Resizable embeddings matrix dataset.
        :param Optional[h5py.Dataset] binary_dataset: Packed binary index, when enabled.
        :param Sequence[PendingEmbeddingRecord] papers_to_embed: Encoded records.
        :param np.ndarray storage_embeddings: Matrix in this namespace's storage dtype.
        :param Optional[np.ndarray] binary_embeddings: Packed binary rows, when enabled.
        :param np.ndarray returned_embeddings: Float32 vectors to return to callers.
        :param int embedding_dim: Embedding vector width.
        :param bool return_embeddings: Whether float32 payloads must be returned.
        :return _VectorWritePlan: Replacement, append and metadata work for this batch.
        """
        existing_row_count = int(embeddings_dataset.shape[0])
        latest_rows = self._load_existing_rows(
            conn, [record.paper_id for record in papers_to_embed]
        )
        plan = _VectorWritePlan(
            existing_row_count=existing_row_count,
            new_embeddings={},
            rows_to_upsert=[],
            append_embeddings=[],
            append_binary_embeddings=[],
            append_records=[],
            replacement_rows=[],
        )

        for idx, record in enumerate(papers_to_embed):
            storage_embedding = storage_embeddings[idx]
            binary_embedding = (
                None if binary_embeddings is None else binary_embeddings[idx]
            )
            existing_row = latest_rows.get(record.paper_id)
            latest_row_idx = (
                existing_row["row_idx"] if existing_row is not None else None
            )

            if (
                existing_row is not None
                and existing_row["text_hash"] == record.text_hash
                and latest_row_idx is not None
                and 0 <= latest_row_idx < existing_row_count
            ):
                if return_embeddings:
                    plan.new_embeddings.update(
                        self._load_cached_embeddings(
                            h5_file,
                            embeddings_dataset,
                            [(record.paper_id, latest_row_idx)],
                        )
                    )
                plan.rows_to_upsert.append(
                    self._metadata_tuple(
                        paper_id=record.paper_id,
                        metadata=record.metadata,
                        text_hash=record.text_hash,
                        embedding_dim=embedding_dim,
                        row_idx=latest_row_idx,
                    )
                )
                continue

            if return_embeddings:
                plan.new_embeddings[record.paper_id] = np.asarray(
                    returned_embeddings[idx], dtype=np.float32
                )
            if latest_row_idx is not None and 0 <= latest_row_idx < existing_row_count:
                previous_binary = (
                    np.asarray(binary_dataset[latest_row_idx]).copy()
                    if binary_dataset is not None
                    else None
                )
                plan.replacement_rows.append(
                    (
                        latest_row_idx,
                        np.asarray(embeddings_dataset[latest_row_idx]).copy(),
                        previous_binary,
                        storage_embedding,
                        binary_embedding,
                    )
                )
                plan.rows_to_upsert.append(
                    self._metadata_tuple(
                        paper_id=record.paper_id,
                        metadata=record.metadata,
                        text_hash=record.text_hash,
                        embedding_dim=embedding_dim,
                        row_idx=latest_row_idx,
                    )
                )
                continue

            plan.append_embeddings.append(storage_embedding)
            if binary_dataset is not None and binary_embedding is not None:
                plan.append_binary_embeddings.append(binary_embedding)
            plan.append_records.append(
                (record.paper_id, record.metadata, record.text_hash)
            )

        return plan

    def _apply_replacement_rows(
        self,
        *,
        conn: sqlite3.Connection,
        embeddings_dataset: h5py.Dataset,
        binary_dataset: Optional[h5py.Dataset],
        replacement_rows: Sequence[
            Tuple[
                int, np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]
            ]
        ],
    ) -> None:
        """Journal the rows about to be overwritten, then overwrite them.

        The journal is committed before the first HDF5 write so an interruption
        mid-replacement is recoverable from durable prior vectors.

        :param sqlite3.Connection conn: Open SQLite connection for the active namespace.
        :param h5py.Dataset embeddings_dataset: Resizable embeddings matrix dataset.
        :param Optional[h5py.Dataset] binary_dataset: Packed binary index, when enabled.
        :param Sequence[Tuple[int, np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]] replacement_rows:
            Rows to overwrite, carrying both their prior and replacement vectors.
        :return None: Mutates the matrix and binary datasets in-place.
        """
        if not replacement_rows:
            return

        self._persist_replacement_journal(
            conn=conn,
            replacements=[
                (row_idx, previous_embedding, previous_binary)
                for row_idx, previous_embedding, previous_binary, _, _ in replacement_rows
            ],
        )
        for (
            row_idx,
            _,
            _,
            replacement_embedding,
            replacement_binary,
        ) in replacement_rows:
            embeddings_dataset[row_idx] = replacement_embedding
            if binary_dataset is not None and replacement_binary is not None:
                binary_dataset[row_idx] = replacement_binary

    def _append_new_rows(
        self,
        *,
        embeddings_dataset: h5py.Dataset,
        binary_dataset: Optional[h5py.Dataset],
        plan: _VectorWritePlan,
        embedding_dim: int,
    ) -> None:
        """Grow the matrix and binary index with this batch's unseen papers.

        :param h5py.Dataset embeddings_dataset: Resizable embeddings matrix dataset.
        :param Optional[h5py.Dataset] binary_dataset: Packed binary index, when enabled.
        :param _VectorWritePlan plan: Write plan whose append rows are persisted.
        :param int embedding_dim: Embedding vector width.
        :return None: Resizes the datasets and extends ``plan.rows_to_upsert``.
        """
        if not plan.append_embeddings:
            return

        append_array = np.vstack(plan.append_embeddings).astype(
            embeddings_dataset.dtype, copy=False
        )
        start_idx = plan.existing_row_count
        end_idx = start_idx + append_array.shape[0]
        embeddings_dataset.resize((end_idx, embedding_dim))
        embeddings_dataset[start_idx:end_idx] = append_array

        if binary_dataset is not None and plan.append_binary_embeddings:
            append_binary_array = np.vstack(plan.append_binary_embeddings).astype(
                np.uint8, copy=False
            )
            binary_dataset.resize((end_idx, append_binary_array.shape[1]))
            binary_dataset[start_idx:end_idx] = append_binary_array

        for offset, (paper_id, metadata, text_hash) in enumerate(plan.append_records):
            plan.rows_to_upsert.append(
                self._metadata_tuple(
                    paper_id=paper_id,
                    metadata=metadata,
                    text_hash=text_hash,
                    embedding_dim=embedding_dim,
                    row_idx=start_idx + offset,
                )
            )

    def _commit_row_mappings(
        self,
        *,
        cursor: sqlite3.Cursor,
        h5_file: h5py.File,
        rows_to_upsert: Sequence[Tuple[Any, ...]],
        replacement_rows: Sequence[
            Tuple[
                int, np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]
            ]
        ],
    ) -> None:
        """Flush vectors, then commit the SQLite rows that point at them.

        SQLite commits durably, so its row mappings must never become durable
        ahead of the vectors they point at: an interrupted append would
        otherwise be recovered by discarding committed rows.

        :param sqlite3.Cursor cursor: Cursor owned by the active connection.
        :param h5py.File h5_file: Writable HDF5 handle for the active namespace.
        :param Sequence[Tuple[Any, ...]] rows_to_upsert: Paper rows to insert or replace.
        :param Sequence[Tuple[int, np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]] replacement_rows:
            Replacement rows whose journal entries are now redundant.
        :return None: Mutates HDF5 durability state and SQLite rows in-place.
        """
        if not rows_to_upsert:
            return

        self._flush_h5_file(h5_file)
        cursor.executemany(PAPER_ROW_UPSERT_SQL, rows_to_upsert)
        if replacement_rows:
            cursor.executemany(
                "DELETE FROM replacement_journal WHERE row_idx = ?",
                [(row_idx,) for row_idx, *_ in replacement_rows],
            )
