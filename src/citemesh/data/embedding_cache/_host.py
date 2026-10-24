"""Structural interface for operations shared across cache mixins."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np
from filelock import FileLock
from numpy.typing import NDArray

from citemesh.data.embedding_contract import EmbeddingContract

from .lifecycle import CorpusState
from .models import (
    CacheMetadataValues,
    NormalizedMetadataFields,
    PaperRequest,
    PaperRowValues,
    StoredEmbedding,
)
from .sql import (
    PaperMetadataRefreshValues,
    SerializedPaperMetadata,
    SQLitePaperRow,
)


class _EmbeddingCacheHost(Protocol):
    """Cache members consumed by the operation and layout mixins."""

    _effective_compression: str
    _effective_compression_level: int
    _hydration_operation_file_lock: FileLock
    _managed_cache_root: Path | None
    _operation_owner: tuple[int, str] | None
    binary_prefilter: bool
    cache_dir: Path
    calibration_sample_size: int
    compression: str
    compression_level: int
    db_path: Path
    embedding_vector_dtype: str
    h5_path: Path
    hydration_lock_path: Path
    lock_path: Path
    model_name: str
    source_torch_dtype: str
    storage_precision: str
    text_formatter_fingerprint: str

    def hydration_operation_lock(self) -> AbstractContextManager[None]: ...

    def namespace_lock_only(self) -> AbstractContextManager[None]: ...

    def _cache_lock(self) -> AbstractContextManager[None]: ...

    def _cache_operation_lock(self) -> AbstractContextManager[None]: ...

    def _connect_db(self) -> AbstractContextManager[sqlite3.Connection]: ...

    def _inspect_db(self) -> AbstractContextManager[sqlite3.Connection]: ...

    def _init_db(self) -> None: ...

    def _inspect_existing_locked(
        self,
        *,
        contract: EmbeddingContract | None = None,
        allow_legacy: bool = False,
    ) -> CorpusState: ...

    def _assert_runtime_cache_consistency(
        self,
        conn: sqlite3.Connection,
        h5_file: h5py.File,
        embeddings_dataset: h5py.Dataset | None,
        *,
        fail_mode: str,
        check_row_mapping: bool = True,
    ) -> None: ...

    def _adopt_existing_dataset_compression(self, dataset: h5py.Dataset) -> None: ...

    def _get_embeddings_dataset(self, h5_file: h5py.File) -> h5py.Dataset | None: ...

    def _ensure_embeddings_dataset(
        self, h5_file: h5py.File, embedding_dim: int
    ) -> h5py.Dataset: ...

    def _ensure_binary_dataset(
        self, h5_file: h5py.File, embedding_dim: int
    ) -> h5py.Dataset | None: ...

    def _load_cache_metadata(self, conn: sqlite3.Connection) -> dict[str, str]: ...

    def _set_cache_metadata(
        self,
        conn: sqlite3.Connection,
        values: Mapping[str, str | int],
        *,
        preserve_existing: bool = False,
    ) -> None: ...

    def _set_h5_attrs(self, h5_file: h5py.File) -> None: ...

    def _recover_pending_replacements_with_connection_locked(
        self, conn: sqlite3.Connection
    ) -> None: ...

    def _recover_pending_replacements_locked(
        self,
        conn: sqlite3.Connection,
        h5_file: h5py.File,
        *,
        validate_runtime_contract: bool = True,
    ) -> None: ...

    def _recover_trailing_rows(
        self,
        conn: sqlite3.Connection,
        h5_file: h5py.File,
        embeddings_dataset: h5py.Dataset,
    ) -> int: ...

    def _flush_h5_file(self, h5_file: h5py.File) -> None: ...

    def _persist_replacement_journal(
        self,
        conn: sqlite3.Connection,
        replacements: Sequence[tuple[int, StoredEmbedding, NDArray[np.uint8] | None]],
    ) -> None: ...

    def _require_calibration_ranges(
        self, h5_file: h5py.File, embedding_dim: int | None = None
    ) -> NDArray[np.float32]: ...

    def _dequantize_int8(
        self, h5_file: h5py.File, int8_embeddings: NDArray[np.int8]
    ) -> NDArray[np.float32]: ...

    def _load_cached_embeddings(
        self,
        h5_file: h5py.File,
        dataset: h5py.Dataset,
        cached_rows: Sequence[tuple[str, int]],
    ) -> dict[str, NDArray[np.float32]]: ...

    def _load_existing_rows(
        self, conn: sqlite3.Connection, paper_ids: Sequence[str]
    ) -> dict[str, SerializedPaperMetadata]: ...

    def _query_paper_rows(
        self,
        conn: sqlite3.Connection,
        lookup_values: Sequence[str | int],
        *,
        lookup_column: str,
        case_insensitive: bool = False,
    ) -> Iterator[SQLitePaperRow]: ...

    def _metadata_tuple(
        self,
        paper_id: str,
        metadata: CacheMetadataValues,
        text_hash: str,
        embedding_dim: int,
        row_idx: int,
    ) -> PaperRowValues: ...

    def _metadata_fields_changed(
        self, existing_row: SerializedPaperMetadata, metadata: CacheMetadataValues
    ) -> bool: ...

    def _metadata_refresh_tuple(
        self, paper_id: str, metadata: CacheMetadataValues
    ) -> PaperMetadataRefreshValues: ...

    def _normalized_metadata_fields(
        self, metadata: CacheMetadataValues
    ) -> NormalizedMetadataFields: ...

    def _report_progress(
        self, values: Sequence[PaperRequest], description: str, *, enabled: bool
    ) -> Iterable[PaperRequest]: ...

    def _record_int8_saturation(
        self,
        h5_file: h5py.File,
        *,
        clipped_value_count: int,
        total_value_count: int,
    ) -> None: ...

    def _require_write_owned_locked(self, conn: sqlite3.Connection) -> None: ...
