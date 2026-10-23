"""Native corpus certification, strict selection and owned import/refresh sessions."""

from __future__ import annotations

import json
import shlex
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from filelock import FileLock

from citemesh.data.cache import cache_operation_lock, get_cache_dir
from citemesh.data.embedding_contract import (
    CorpusSource,
    EmbeddingContract,
    ExistingCacheSelection,
    require_namespace_contract,
)

from . import constants
from .sql import (
    CORPUS_MEMBERSHIP_ORDINAL_INDEX_SQL,
    CORPUS_MEMBERSHIP_TABLE_CREATE_SQL,
    PAPER_ROW_UPSERT_SQL,
    _parse_json_list,
)

CORPUS_STATE_KEY = "corpus_state"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _read_state(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        "SELECT value FROM cache_metadata WHERE key = ?", (CORPUS_STATE_KEY,)
    ).fetchone()
    return json.loads(row[0]) if row else {}


def _write_state(conn: sqlite3.Connection, state: dict[str, Any]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO cache_metadata(key,value) VALUES (?,?)",
        (CORPUS_STATE_KEY, _json(state)),
    )


class _LifecycleMixin:
    """Operations that preserve a corpus's fixed contract and certified membership."""

    @contextmanager
    def _read_only_db(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            yield conn
        finally:
            conn.close()

    def _has_corpus_state_read_only(self) -> bool:
        if not self.db_path.exists():
            return False
        with self._read_only_db() as conn:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='cache_metadata'"
            ).fetchone():
                return False
            return bool(_read_state(conn))

    def _inspect_existing_locked(
        self, *, contract: EmbeddingContract | None = None, allow_legacy: bool = False
    ):
        """Inspect both files without repairs, schema changes, or vector writes."""
        if not self.db_path.is_file() or not self.h5_path.is_file():
            raise RuntimeError(
                "Existing embedding cache requires both SQLite and HDF5 files; "
                "the existing files were preserved"
            )
        with self._read_only_db() as conn, h5py.File(self.h5_path, "r") as h5:
            state = _read_state(conn)
            if state:
                require_namespace_contract(
                    EmbeddingContract(**state["contract"]),
                    state["namespace_selectors"],
                    state["namespace_serialization"],
                )
            if contract is not None and state.get("contract") != asdict(contract):
                metadata = self._load_cache_metadata(conn)
                if not (
                    allow_legacy
                    and not state
                    and metadata.get(constants.MODEL_FINGERPRINT_KEY)
                    == contract.model_artifact_identity
                ):
                    raise ValueError("Selected cache embedding contract differs")
            dataset = self._get_embeddings_dataset(h5)
            count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            if dataset is None:
                if count:
                    raise RuntimeError(
                        "Existing cache has mappings but no embedding matrix"
                    )
            else:
                self._adopt_existing_dataset_compression(dataset)
                self._assert_runtime_cache_consistency(
                    conn=conn,
                    h5_file=h5,
                    embeddings_dataset=dataset,
                    fail_mode="runtime",
                    check_row_mapping=not bool(state.get("pending")),
                )
                if contract is not None and (
                    dataset.shape[1] != contract.dimension
                    or dataset.dtype != np.dtype(contract.stored_dtype)
                ):
                    raise ValueError(
                        "Existing cache matrix differs from embedding contract"
                    )
                if not state.get("pending"):
                    bounds = conn.execute(
                        "SELECT COUNT(DISTINCT row_idx), MIN(row_idx),MAX(row_idx) FROM papers"
                    ).fetchone()
                    if (
                        int(dataset.shape[0]) != count
                        or bounds[0] != count
                        or (count and bounds[1:] != (0, count - 1))
                    ):
                        raise RuntimeError(
                            "Existing cache row coverage is inconsistent"
                        )
                if self.storage_precision == "int8":
                    self._require_calibration_ranges(h5, int(dataset.shape[1]))
            return state

    def corpus_state(self) -> dict[str, Any]:
        """Read source and operation state without model construction or repair."""
        with self.namespace_lock_only(), self._read_only_db() as conn:
            return _read_state(conn)

    def _require_ready_locked(self, conn: sqlite3.Connection) -> dict[str, Any]:
        state = _read_state(conn)
        if state.get("pending"):
            raise RuntimeError(
                f"Corpus cache has an incomplete {state['pending']['kind']} operation; "
                "resume that operation before using the corpus"
            )
        if state and not state.get("membership_complete"):
            raise RuntimeError(
                "Corpus membership is not certified; run explicit refresh"
            )
        return state

    def require_complete_certified_corpus(self) -> dict[str, Any]:
        with self.namespace_lock_only(), self._read_only_db() as conn:
            state = self._require_ready_locked(conn)
            if not state.get("membership_complete"):
                raise RuntimeError(
                    "Export requires explicitly reconciled corpus membership"
                )
            return state

    def _require_write_owned_locked(self, conn: sqlite3.Connection) -> None:
        state = _read_state(conn)
        if not state:
            return
        owner = getattr(self, "_operation_owner", None)
        pending = state.get("pending")
        if pending and owner == (threading.get_ident(), _json(pending["identity"])):
            return
        if pending or state.get("update_policy") == "manual":
            raise RuntimeError(
                "This corpus is managed manually; writes require an owned import/refresh session"
            )

    @property
    def namespace_id(self) -> str:
        return self.db_path.stem.removeprefix("metadata_")

    def _eligible_rows_locked(
        self,
        conn: sqlite3.Connection,
        row_count: int,
        corpus_size: int | None = None,
        eligible_rows: np.ndarray | None = None,
    ) -> np.ndarray | None:
        state = self._require_ready_locked(conn)
        mask = None
        if state.get("membership_complete"):
            source = state["source"]
            if source["selection"] == "newest" and (
                corpus_size is None or corpus_size > source["limit"]
            ):
                if corpus_size is not None:
                    raise ValueError("Requested corpus view exceeds installed coverage")
            mask = np.zeros(row_count, dtype=bool)
            sql = (
                "SELECT p.row_idx FROM corpus_membership m "
                "JOIN papers p ON p.paper_id=m.paper_id"
            )
            if corpus_size is not None:
                if corpus_size < 1:
                    raise ValueError("corpus_size must be positive")
                sql += (
                    " ORDER BY (p.chronology_key IS NULL), p.chronology_key DESC, "
                    "CASE WHEN p.chronology_key IS NOT NULL THEN m.source_ordinal END DESC, "
                    "m.source_ordinal ASC LIMIT ?"
                )
                cursor = conn.execute(sql, (corpus_size,))
            else:
                cursor = conn.execute(sql)
            for rows in iter(lambda: cursor.fetchmany(8192), []):
                indices = np.asarray([row[0] for row in rows], dtype=np.int64)
                if np.any(indices < 0) or np.any(indices >= row_count):
                    raise RuntimeError(
                        "Corpus membership refers to missing vector rows"
                    )
                mask[indices] = True
        elif corpus_size is not None:
            raise RuntimeError("Exact corpus views require certified source membership")
        if eligible_rows is not None:
            requested = np.asarray(eligible_rows)
            if requested.dtype != np.bool_ or requested.shape != (row_count,):
                raise ValueError(
                    "eligible_rows must be a boolean mask of physical vector rows"
                )
            mask = requested if mask is None else mask & requested
        return mask

    def get_cached_embeddings(self, paper_ids: Sequence[str]) -> dict[str, np.ndarray]:
        """Read existing vectors without updating metadata or invoking an encoder."""
        with self._cache_lock(), self._connect_db() as conn:
            self._require_ready_locked(conn)
            with h5py.File(self.h5_path, "r") as h5:
                dataset = self._get_embeddings_dataset(h5)
                if dataset is None:
                    return {}
                rows = self._load_existing_rows(conn, list(paper_ids))
                return self._load_cached_embeddings(
                    h5,
                    dataset,
                    [(paper_id, row["row_idx"]) for paper_id, row in rows.items()],
                )

    @contextmanager
    def operation_session(
        self,
        *,
        kind: str,
        identity: dict[str, Any],
        target_source: CorpusSource,
        contract: EmbeddingContract,
        namespace_selectors: dict[str, Any],
        origin: dict[str, Any] | None = None,
        producer_provenance: dict[str, Any] | None = None,
    ) -> Iterator[CorpusOperationSession]:
        with self.hydration_operation_lock():
            session = CorpusOperationSession(
                self,
                kind=kind,
                identity=identity,
                target_source=target_source,
                contract=contract,
                namespace_selectors=namespace_selectors,
                origin=origin,
                producer_provenance=producer_provenance,
            )
            session.begin()
            try:
                yield session
            finally:
                self._operation_owner = None

    @contextmanager
    def export_snapshot_view(self) -> Iterator[NativeSnapshotView]:
        with self.hydration_operation_lock(), self.namespace_lock_only():
            with self._read_only_db() as conn:
                state = self._require_ready_locked(conn)
                if not state.get("membership_complete"):
                    raise RuntimeError(
                        "Export requires explicitly reconciled corpus membership"
                    )
                with h5py.File(self.h5_path, "r") as h5:
                    yield NativeSnapshotView(self, conn, h5, state)


class NativeSnapshotView:
    """Bounded read view held under the corpus and namespace locks."""

    def __init__(self, cache, conn, h5, state):
        self.cache, self.conn, self.h5 = cache, conn, h5
        self.contract = EmbeddingContract(**state["contract"])
        self.completed_source = CorpusSource(**state["source"])
        self.namespace_serialization = state["namespace_serialization"]
        self.namespace_selectors = state["namespace_selectors"]
        self.source_records_examined = state["source_records_examined"]
        self.producer_provenance = state.get("producer_provenance", {})
        self.origin = state.get("origin")
        self.calibration_ranges = (
            np.asarray(h5[constants.CALIBRATION_RANGES_DATASET_NAME])
            if self.contract.stored_dtype == "int8"
            else None
        )

    def require_complete_certified_corpus(self):
        return None

    def iter_active_rows(self, *, batch_size=8192, order_by_native_row=True):
        cursor = self.conn.execute(
            "SELECT p.paper_id,p.text_hash,p.row_idx,p.title,p.abstract,p.year,"
            "p.authors_json,p.categories_json,p.venue,p.arxiv_id,p.doi,"
            "p.chronology_key,m.source_ordinal FROM corpus_membership m "
            "JOIN papers p ON p.paper_id=m.paper_id ORDER BY p.row_idx"
        )
        dataset = self.h5.get(constants.EMBEDDINGS_DATASET_NAME)
        while rows := cursor.fetchmany(batch_size):
            records = []
            for row in rows:
                records.append(
                    dict(
                        zip(
                            (
                                "paper_id",
                                "text_hash",
                                "row_idx",
                                "title",
                                "abstract",
                                "year",
                                "authors",
                                "categories",
                                "venue",
                                "arxiv_id",
                                "doi",
                                "chronology_key",
                                "source_ordinal",
                            ),
                            row,
                            strict=True,
                        )
                    )
                )
                records[-1]["authors"] = _parse_json_list(row[6])
                records[-1]["categories"] = _parse_json_list(row[7])
            indices = np.asarray([row[2] for row in rows], dtype=np.int64)
            yield records, np.asarray(dataset[indices])


class CorpusOperationSession:
    """An operation owns writes while its pending identity remains persisted."""

    def __init__(
        self,
        cache,
        *,
        kind,
        identity,
        target_source,
        contract,
        namespace_selectors,
        origin=None,
        producer_provenance=None,
    ):
        if kind not in {"import", "refresh"}:
            raise ValueError("Unsupported corpus operation")
        self.cache = cache
        self.kind, self.identity = kind, identity
        self.target_source, self.contract = target_source, contract
        self.namespace_selectors, self.origin = namespace_selectors, origin
        self.producer_provenance = producer_provenance
        self.already_complete = False
        self._committed_rows = None

    def begin(self):
        require_namespace_contract(
            self.contract, self.namespace_selectors, self.cache.model_name
        )
        with self.cache.namespace_lock_only(), self.cache._connect_db() as conn:
            conn.execute(CORPUS_MEMBERSHIP_TABLE_CREATE_SQL)
            conn.execute(CORPUS_MEMBERSHIP_ORDINAL_INDEX_SQL)
            state = _read_state(conn)
            if state.get("contract") and state["contract"] != asdict(self.contract):
                raise ValueError("Occupied cache has a different embedding contract")
            existing_source = state.get("source") or state.get("pending", {}).get(
                "target_source"
            )
            if existing_source and any(
                existing_source.get(key) != asdict(self.target_source)[key]
                for key in ("repo_id", "config_name", "split")
            ):
                raise ValueError("Occupied cache represents a different source")
            pending = state.get("pending")
            if pending and (
                pending["kind"] != self.kind or pending["identity"] != self.identity
            ):
                raise ValueError(
                    "Another incomplete corpus operation owns this namespace: "
                    + _json(pending)
                )
            if pending and pending["target_source"] != asdict(self.target_source):
                raise ValueError("Resume must use the recorded source target")
            if self.kind == "refresh" or not pending:
                conn.execute("DELETE FROM corpus_membership")
            state.update(
                contract=asdict(self.contract),
                namespace_serialization=self.cache.model_name,
                namespace_selectors=self.namespace_selectors,
                update_policy="manual",
                membership_complete=False,
                pending={
                    "kind": self.kind,
                    "identity": self.identity,
                    "target_source": asdict(self.target_source),
                },
            )
            if self.origin is not None and not state.get("origin"):
                state["origin"] = self.origin
            if self.producer_provenance is not None and not state.get(
                "producer_provenance"
            ):
                state["producer_provenance"] = self.producer_provenance
            if not state.get("producer_provenance"):
                metadata = self.cache._load_cache_metadata(conn)
                state["producer_provenance"] = {
                    key: metadata[key]
                    for key in (
                        constants.SOURCE_TORCH_DTYPE_KEY,
                        constants.EMBEDDING_VECTOR_DTYPE_KEY,
                    )
                    if key in metadata
                }
            _write_state(conn, state)
            self.cache._set_cache_metadata(
                conn, {constants.HYDRATION_COMPLETE_KEY: "0"}
            )
        self.cache._operation_owner = (threading.get_ident(), _json(self.identity))

    @property
    def committed_rows(self) -> int:
        if self._committed_rows is not None:
            return self._committed_rows
        with self.cache.namespace_lock_only(), self.cache._connect_db() as conn:
            with h5py.File(self.cache.h5_path, "a") as h5:
                self.cache._recover_pending_replacements_locked(
                    conn=conn,
                    h5_file=h5,
                    validate_runtime_contract=False,
                )
                dataset = self.cache._get_embeddings_dataset(h5)
                if dataset is not None:
                    self.cache._recover_trailing_rows(
                        conn=conn,
                        h5_file=h5,
                        embeddings_dataset=dataset,
                    )
                conn.execute(
                    "DELETE FROM corpus_membership WHERE paper_id NOT IN (SELECT paper_id FROM papers)"
                )
                count, distinct, minimum, maximum = conn.execute(
                    "SELECT COUNT(*),COUNT(DISTINCT row_idx),MIN(row_idx),MAX(row_idx) FROM papers"
                ).fetchone()
                if count != distinct or (
                    count and (minimum != 0 or maximum != count - 1)
                ):
                    raise RuntimeError(
                        "Import mappings are not a dense committed prefix"
                    )
                self._committed_rows = count
                return count

    @property
    def membership_count(self) -> int:
        with self.cache.namespace_lock_only(), self.cache._read_only_db() as conn:
            return int(
                conn.execute("SELECT COUNT(*) FROM corpus_membership").fetchone()[0]
            )

    def validate_records(self, records: Sequence[dict]) -> list[dict]:
        """Compare duplicate source IDs against their first committed metadata."""
        unique = {}
        for record in records:
            paper_id = record["paper_id"]
            prior = unique.get(paper_id)
            if prior is not None:
                if self.cache._normalized_metadata_fields(
                    prior
                ) != self.cache._normalized_metadata_fields(record):
                    raise ValueError(f"Conflicting duplicate paper ID {paper_id!r}")
                continue
            unique[paper_id] = record
        with self.cache.namespace_lock_only(), self.cache._connect_db() as conn:
            self.cache._require_write_owned_locked(conn)
            ids = list(unique)
            seen = []
            for start in range(0, len(ids), constants.SQLITE_QUERY_BATCH_SIZE):
                batch = ids[start : start + constants.SQLITE_QUERY_BATCH_SIZE]
                marks = ",".join("?" for _ in batch)
                seen.extend(
                    row[0]
                    for row in conn.execute(
                        f"SELECT paper_id FROM corpus_membership WHERE paper_id IN ({marks})",
                        batch,
                    )
                )
            existing = self.cache._load_existing_rows(conn, seen)
            for paper_id, row in existing.items():
                if self.cache._metadata_fields_changed(row, unique[paper_id]):
                    raise ValueError(f"Conflicting duplicate paper ID {paper_id!r}")
        return list(unique.values())

    def _insert_membership(self, conn, records):
        conn.executemany(
            "INSERT OR IGNORE INTO corpus_membership (paper_id,source_ordinal) VALUES (?,?)",
            [(record["paper_id"], record["source_ordinal"]) for record in records],
        )

    def record_membership(self, records: Sequence[dict]) -> None:
        records = self.validate_records(records)
        with self.cache.namespace_lock_only(), self.cache._connect_db() as conn:
            self.cache._require_write_owned_locked(conn)
            existing = self.cache._load_existing_rows(
                conn, [record["paper_id"] for record in records]
            )
            if len(existing) != len(records):
                raise ValueError("Membership requires a committed vector row")
            self._insert_membership(conn, records)

    def append_stored_embeddings(self, records, vectors, *, first_row: int) -> int:
        if vectors.dtype != np.dtype(self.contract.stored_dtype):
            raise ValueError("Stored dtype differs from snapshot contract")
        if vectors.shape != (len(records), self.contract.dimension):
            raise ValueError("Stored matrix shape differs from snapshot contract")
        count = self.committed_rows
        self._committed_rows = None
        with self.cache.namespace_lock_only(), self.cache._connect_db() as conn:
            self.cache._require_write_owned_locked(conn)
            with h5py.File(self.cache.h5_path, "a") as h5:
                self.cache._set_h5_attrs(h5)
                dataset = self.cache._ensure_embeddings_dataset(
                    h5, self.contract.dimension
                )
                if int(dataset.shape[0]) != count:
                    count = self.cache._recover_trailing_rows(
                        conn=conn, h5_file=h5, embeddings_dataset=dataset
                    )
                if first_row != count:
                    raise ValueError("Import batch does not start at committed prefix")
                ids = [record["paper_id"] for record in records]
                if len(ids) != len(set(ids)):
                    raise ValueError("Import contains duplicate paper IDs")
                if self.cache._load_existing_rows(conn, ids):
                    raise ValueError("Import contains existing paper IDs")
                dataset.resize((count + len(records), self.contract.dimension))
                dataset[count:] = vectors
                self.cache._flush_h5_file(h5)
                tuples = []
                for offset, record in enumerate(records):
                    row = list(
                        self.cache._metadata_tuple(
                            record["paper_id"],
                            record,
                            record["text_hash"],
                            self.contract.dimension,
                            count + offset,
                        )
                    )
                    row[-1] = record.get("chronology_key")
                    tuples.append(tuple(row))
                conn.executemany(
                    PAPER_ROW_UPSERT_SQL.replace("INSERT OR REPLACE", "INSERT"), tuples
                )
                self._insert_membership(conn, records)
        self._committed_rows = count + len(records)
        return self._committed_rows

    def finish(self, *, expected_rows: int, source_records_examined: int) -> None:
        with self.cache.namespace_lock_only(), self.cache._connect_db() as conn:
            self.cache._require_write_owned_locked(conn)
            count = conn.execute("SELECT COUNT(*) FROM corpus_membership").fetchone()[0]
            dangling = conn.execute(
                "SELECT 1 FROM corpus_membership m LEFT JOIN papers p "
                "ON p.paper_id=m.paper_id WHERE p.row_idx IS NULL LIMIT 1"
            ).fetchone()
            if count != expected_rows or dangling:
                raise ValueError(
                    "Completed membership does not match expected row coverage"
                )
            with h5py.File(self.cache.h5_path, "a") as h5:
                self.cache._set_h5_attrs(h5)
                dataset = self.cache._get_embeddings_dataset(h5)
                physical_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[
                    0
                ]
                if dataset is None:
                    dataset = self.cache._ensure_embeddings_dataset(
                        h5, self.contract.dimension
                    )
                if int(dataset.shape[0]) != physical_count:
                    raise ValueError(
                        "Completed matrix does not match native row mappings"
                    )
                if self.contract.stored_dtype == "int8":
                    self.cache._require_calibration_ranges(h5, self.contract.dimension)
                    self.cache._ensure_binary_dataset(h5, self.contract.dimension)
                self.cache._flush_h5_file(h5)
            state = _read_state(conn)
            state.update(
                source=asdict(self.target_source),
                membership_complete=True,
                membership_version=1,
                source_records_examined=source_records_examined,
            )
            if self.kind == "import":
                state["completed_import_identity"] = self.identity
            else:
                state.pop("completed_import_identity", None)
            state.pop("pending", None)
            _write_state(conn, state)
            self.cache._set_cache_metadata(
                conn,
                {
                    constants.HYDRATION_COMPLETE_KEY: "1",
                    constants.HYDRATION_DATASET_SOURCE_KEY: self.target_source.repo_id,
                    constants.HYDRATION_SPLIT_KEY: self.target_source.split,
                    constants.HYDRATION_CORPUS_SIZE_KEY: constants._corpus_size_token(
                        self.target_source.limit
                    ),
                    constants.CORPUS_METADATA_VERSION_KEY: constants.CORPUS_METADATA_VERSION,
                    constants.CORPUS_IDENTITY_VERSION_KEY: constants.CORPUS_IDENTITY_VERSION,
                    constants.MODEL_FINGERPRINT_KEY: self.contract.model_artifact_identity,
                    constants.HYDRATION_RECONCILED_CACHE_ROWS_KEY: count,
                    constants.HYDRATION_RECONCILED_UPSTREAM_ROWS_KEY: source_records_examined,
                },
            )


def _cache_shell(cache_dir, namespace_serialization, contract):
    from .store import EmbeddingCache

    return EmbeddingCache(
        cache_dir=cache_dir,
        model_name=namespace_serialization,
        storage_precision=contract.stored_dtype,
        calibration_sample_size=contract.calibration_sample_size or 2000,
        text_formatter_fingerprint=contract.formatter_identity,
        _initialize=False,
    )


def select_existing_cache(
    namespace_id: str, cache_dir: Path | None = None, *, require_ready=True
):
    """Read the installed selection before constructing a builder or model."""
    if len(namespace_id) != 12 or any(
        c not in "0123456789abcdef" for c in namespace_id
    ):
        raise ValueError("Expected a native 12-character namespace ID")
    directory = (
        Path(cache_dir) if cache_dir else get_cache_dir(create=False) / "embeddings"
    )
    db_path = directory / f"metadata_{namespace_id}.db"
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        state = _read_state(conn)
    finally:
        conn.close()
    if not state or not state.get("contract"):
        raise RuntimeError(
            "Selected cache has no certified embedding contract; refresh/adopt it first"
        )
    contract = EmbeddingContract(**state["contract"])
    source = state.get("source") or state.get("pending", {}).get("target_source")
    selection = ExistingCacheSelection(
        namespace_id,
        state["namespace_serialization"],
        contract,
        source["repo_id"],
        source["config_name"],
        source["split"],
    )
    open_existing_cache(selection, directory, require_ready=require_ready)
    return selection


read_existing_cache_selection = select_existing_cache


def read_existing_cache_metadata(namespace_id: str, cache_dir: Path | None = None):
    """Read legacy metadata under native locks without constructing/repairing payloads."""
    if len(namespace_id) != 12 or any(
        c not in "0123456789abcdef" for c in namespace_id
    ):
        raise ValueError("Expected a native 12-character namespace ID")
    from .store import EmbeddingCache

    configured_root = get_cache_dir(create=False)
    directory = Path(cache_dir) if cache_dir else configured_root / "embeddings"
    managed_root = EmbeddingCache._resolve_managed_cache_root(
        directory, configured_root
    )
    operation_lock = (
        cache_operation_lock(managed_root) if managed_root else nullcontext()
    )
    timeout = constants._resolve_cache_lock_timeout_seconds()
    with (
        operation_lock,
        FileLock(str(directory / f"hydration_{namespace_id}.lock"), timeout=timeout),
        FileLock(str(directory / f"cache_{namespace_id}.lock"), timeout=timeout),
    ):
        db_path = directory / f"metadata_{namespace_id}.db"
        conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            return dict(conn.execute("SELECT key,value FROM cache_metadata"))
        finally:
            conn.close()


def read_legacy_model_fingerprint(namespace_id: str, cache_dir: Path | None = None):
    """Return the artifact identity witnessed by an existing native namespace."""
    return read_existing_cache_metadata(namespace_id, cache_dir).get(
        constants.MODEL_FINGERPRINT_KEY
    )


def open_existing_cache(
    selection: ExistingCacheSelection,
    cache_dir: Path | None = None,
    *,
    require_ready=True,
    allow_legacy=False,
):
    cache = _cache_shell(
        cache_dir, selection.namespace_serialization, selection.contract
    )
    if cache.db_path.stem != "metadata_" + selection.namespace_id:
        raise ValueError(
            "Stored namespace serialization does not match selected address"
        )
    with cache.namespace_lock_only():
        cache._inspect_existing_locked(
            contract=selection.contract, allow_legacy=allow_legacy
        )
        if require_ready:
            with cache._read_only_db() as conn:
                state = cache._require_ready_locked(conn)
                if not state.get("membership_complete"):
                    raise RuntimeError("Selected cache membership is not certified")
    cache._strict_existing = True
    return cache


def preflight_import_destination(
    cache_dir,
    namespace_serialization,
    contract,
    source,
    identity,
    namespace_selectors=None,
    calibration_ranges=None,
    origin=None,
):
    """Classify an import destination without creating or repairing payloads."""
    require_namespace_contract(contract, namespace_selectors, namespace_serialization)
    cache = _cache_shell(cache_dir, namespace_serialization, contract)
    with cache.hydration_operation_lock(), cache.namespace_lock_only():
        return _classify_import_locked(
            cache, contract, source, identity, calibration_ranges, origin
        )


def _classify_import_locked(cache, contract, source, identity, ranges, origin):
    db, h5 = cache.db_path.exists(), cache.h5_path.exists()
    if not db and not h5:
        return "create"
    if not db:
        raise ValueError(
            "Occupied import destination has HDF5 without SQLite; files preserved"
        )
    with cache._read_only_db() as conn:
        try:
            state = _read_state(conn)
            count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        except sqlite3.DatabaseError as exc:
            raise ValueError(
                "Occupied import database is unreadable; files preserved"
            ) from exc
    pending = state.get("pending", {})
    if state.get("contract") != asdict(contract):
        raise ValueError(
            "Occupied import destination has a different or unknown contract"
        )
    if pending:
        if pending.get("kind") != "import" or pending.get("identity") != identity:
            publication = state.get("origin") or {}
            resume = ""
            if all(
                publication.get(key)
                for key in ("bucket_id", "config_name", "snapshot_id")
            ):
                resume = " Resume with: " + shlex.join(
                    [
                        "citemesh",
                        "cache",
                        "import",
                        "--bucket",
                        publication["bucket_id"],
                        "--config",
                        publication["config_name"],
                        "--snapshot",
                        publication["snapshot_id"],
                    ]
                )
            raise ValueError(
                "Conflicting incomplete import; resume its pinned identity: "
                + _json(pending)
                + resume
            )
        if pending["target_source"] != asdict(source) or state.get("origin") != origin:
            raise ValueError("Import provenance differs from pending import")
        if not h5 and count:
            raise ValueError(
                "Incomplete import has committed rows but no HDF5; files preserved"
            )
        action = "resume_same_import"
    elif (
        state.get("completed_import_identity") == identity
        and state.get("source") == asdict(source)
        and state.get("membership_complete")
        and state.get("origin") == origin
    ):
        if not h5:
            raise ValueError("Completed import has no HDF5; files preserved")
        action = "already_complete"
    else:
        raise ValueError(
            "Import destination is occupied; replacement/merging is not supported"
        )
    if h5:
        cache._inspect_existing_locked(contract=contract)
        if contract.stored_dtype == "int8":
            with h5py.File(cache.h5_path, "r") as handle:
                stored = handle.get(constants.CALIBRATION_RANGES_DATASET_NAME)
                if stored is not None and (
                    ranges is not None and not np.array_equal(stored[:], ranges)
                ):
                    raise ValueError(
                        "Import calibration ranges differ from occupied destination"
                    )
    return action


@contextmanager
def import_destination_session(
    cache_dir,
    namespace_serialization,
    contract,
    source,
    identity,
    namespace_selectors,
    calibration_ranges=None,
    origin=None,
    producer_provenance=None,
):
    """Recognize the same import before any initialization or native recovery."""
    require_namespace_contract(contract, namespace_selectors, namespace_serialization)
    cache = _cache_shell(cache_dir, namespace_serialization, contract)
    if producer_provenance:
        cache.source_torch_dtype = str(
            producer_provenance.get("source_torch_dtype", "float32")
        )
    with cache.hydration_operation_lock():
        with cache.namespace_lock_only():
            action = _classify_import_locked(
                cache, contract, source, identity, calibration_ranges, origin
            )
            if action == "create":
                cache._init_db()
                with cache._connect_db() as conn:
                    _write_state(
                        conn,
                        {
                            "contract": asdict(contract),
                            "namespace_serialization": namespace_serialization,
                            "namespace_selectors": namespace_selectors,
                            "origin": origin,
                            "update_policy": "manual",
                            "membership_complete": False,
                            "pending": {
                                "kind": "import",
                                "identity": identity,
                                "target_source": asdict(source),
                            },
                        },
                    )
            if action != "already_complete":
                with h5py.File(cache.h5_path, "a") as h5:
                    cache._set_h5_attrs(h5)
                    if contract.stored_dtype == "int8":
                        if calibration_ranges is None:
                            raise ValueError(
                                "Int8 import requires exact calibration ranges"
                            )
                        if constants.CALIBRATION_RANGES_DATASET_NAME not in h5:
                            h5.create_dataset(
                                constants.CALIBRATION_RANGES_DATASET_NAME,
                                data=calibration_ranges,
                            )
                    cache._flush_h5_file(h5)
        if action == "already_complete":
            session = CorpusOperationSession(
                cache,
                kind="import",
                identity=identity,
                target_source=source,
                contract=contract,
                namespace_selectors=namespace_selectors,
                origin=origin,
                producer_provenance=producer_provenance,
            )
            session.already_complete = True
            yield session
        else:
            with cache.operation_session(
                kind="import",
                identity=identity,
                target_source=source,
                contract=contract,
                namespace_selectors=namespace_selectors,
                origin=origin,
                producer_provenance=producer_provenance,
            ) as session:
                yield session
