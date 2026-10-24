"""Native corpus certification, strict selection and owned import/refresh sessions."""

from __future__ import annotations

import json
import shlex
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
)

import h5py
import numpy as np
import numpy.typing as npt
from filelock import FileLock, Timeout

from citemesh.data.cache import cache_operation_lock, get_cache_dir, path_exists
from citemesh.data.embedding_contract import (
    CorpusSource,
    CorpusSourceData,
    EmbeddingContract,
    EmbeddingContractData,
    ExistingCacheSelection,
    NamespaceSelectors,
    newest_corpus_order_sql,
    require_namespace_contract,
)

from . import constants
from .models import (
    ActiveCorpusRecord,
    CorpusRecord,
    CorpusSourceRecord,
    PortableImportIdentity,
    StoredEmbedding,
)
from .models import (
    CorpusOrigin as CorpusOrigin,
)
from .models import (
    PublicationOrigin as PublicationOrigin,
)
from .models import (
    SnapshotOrigin as SnapshotOrigin,
)
from .sql import (
    CORPUS_MEMBERSHIP_ORDINAL_INDEX_SQL,
    CORPUS_MEMBERSHIP_TABLE_CREATE_SQL,
    PAPER_ROW_UPSERT_SQL,
    _parse_json_list,
)

if TYPE_CHECKING:
    from ._host import _EmbeddingCacheHost
    from .store import EmbeddingCache
else:
    _EmbeddingCacheHost = object

CORPUS_STATE_KEY = "corpus_state"


class RefreshOperationIdentity(TypedDict):
    """Source selection that identifies a corpus refresh."""

    source: CorpusSourceData


OperationIdentity: TypeAlias = RefreshOperationIdentity | PortableImportIdentity
_LifecycleCacheT = TypeVar("_LifecycleCacheT", bound="_LifecycleMixin")
_CacheT = TypeVar("_CacheT", bound=_EmbeddingCacheHost)
_CorpusRecordT = TypeVar("_CorpusRecordT", CorpusRecord, CorpusSourceRecord)


class PendingOperation(TypedDict):
    """Persisted identity and source for an unfinished cache operation."""

    kind: Literal["import", "refresh"]
    identity: OperationIdentity
    target_source: CorpusSourceData


class CorpusState(TypedDict, total=False):
    """Fields persisted for a corpus namespace and its current operation."""

    contract: EmbeddingContractData
    namespace_serialization: str
    namespace_selectors: NamespaceSelectors
    source: CorpusSourceData
    update_policy: Literal["manual"]
    membership_complete: bool
    membership_version: Literal[1]
    source_records_examined: int
    pending: PendingOperation
    origin: CorpusOrigin | None
    producer_provenance: Mapping[str, object]
    completed_import_identity: PortableImportIdentity


def _json(value: OperationIdentity | PendingOperation | CorpusState) -> str:
    """Serialize cache state or an operation identity with stable compact JSON.

    :param OperationIdentity | PendingOperation | CorpusState value: JSON data to encode.
    :return str: Compact JSON with keys in sorted order.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _read_state(conn: sqlite3.Connection) -> CorpusState:
    """Read the persisted corpus state, or return an empty mapping.

    :param sqlite3.Connection conn: Open cache database connection.
    :return CorpusState: Stored state, or an empty state when no value exists.
    """
    row = conn.execute(
        "SELECT value FROM cache_metadata WHERE key = ?", (CORPUS_STATE_KEY,)
    ).fetchone()
    return json.loads(row[0]) if row else {}


def _write_state(conn: sqlite3.Connection, state: CorpusState) -> None:
    """Replace the persisted corpus state in SQLite metadata.

    :param sqlite3.Connection conn: Open cache database connection.
    :param CorpusState state: Current contract, source, and operation state.
    """
    conn.execute(
        "INSERT OR REPLACE INTO cache_metadata(key,value) VALUES (?,?)",
        (CORPUS_STATE_KEY, _json(state)),
    )


@contextmanager
def _existing_database(
    db_path: Path, *, timeout: float = 5.0
) -> Iterator[sqlite3.Connection]:
    """Open existing SQLite state, permitting journal recovery.

    :param Path db_path: Existing native metadata database.
    :param float timeout: Maximum SQLite lock wait in seconds.
    Yields:
        The connection, closed on context exit.
    """
    if not db_path.is_file():
        namespace_id = db_path.stem.removeprefix("metadata_")
        raise ValueError(f"Namespace {namespace_id} is not installed; see cache scan")
    conn = sqlite3.connect(
        db_path.resolve().as_uri() + "?mode=rw", uri=True, timeout=timeout
    )
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def _existing_namespace_connection(
    namespace_id: str, cache_dir: Path | None = None, *, blocking: bool = True
) -> Iterator[sqlite3.Connection]:
    """Hold native cache locks while reading existing SQLite state.

    :param str namespace_id: Twelve-character native namespace ID.
    :param Path | None cache_dir: Optional embedding cache directory.
    :param bool blocking: Wait for native locks and SQLite transactions when occupied.
    Yields:
        The existing database connection, with hot-journal recovery enabled.
    """
    if len(namespace_id) != 12 or any(
        c not in "0123456789abcdef" for c in namespace_id
    ):
        raise ValueError("Expected a native 12-character namespace ID")
    from .store import EmbeddingCache

    configured_root = get_cache_dir(create=False)
    directory = Path(cache_dir) if cache_dir else configured_root / "embeddings"
    db_path = directory / f"metadata_{namespace_id}.db"
    if not db_path.is_file():
        raise ValueError(f"Namespace {namespace_id} is not installed; see cache scan")
    managed_root = EmbeddingCache._resolve_managed_cache_root(
        directory, configured_root
    )
    timeout = constants._resolve_cache_lock_timeout_seconds()
    operation_lock = (
        cache_operation_lock(managed_root, timeout=timeout, blocking=blocking)
        if managed_root
        else nullcontext()
    )
    with (
        operation_lock,
        FileLock(
            str(directory / f"cache_{namespace_id}.lock"),
            timeout=timeout,
            blocking=blocking,
        ),
        _existing_database(db_path, timeout=5.0 if blocking else 0.0) as conn,
    ):
        yield conn


@dataclass(frozen=True)
class InstalledCorpusSummary:
    """Schema, persisted corpus state and row counts used by namespace inspection."""

    schema_version: str | None
    state: CorpusState
    physical_rows: int
    active_rows: int


class CorpusMetadataBusyError(RuntimeError):
    """Report SQLite contention while inspecting installed corpus metadata."""


class CorpusRecoveryPendingError(RuntimeError):
    """Report a hot journal that could not be rolled back."""


def _read_installed_corpus_summary(conn: sqlite3.Connection) -> InstalledCorpusSummary:
    """Read schema and corpus counts from one SQLite snapshot.

    :param sqlite3.Connection conn: Fresh connection used for the read transaction.
    :return InstalledCorpusSummary: Schema, corpus state and matching row counts.
    """
    conn.execute("BEGIN")
    row = conn.execute(
        "SELECT value FROM cache_metadata WHERE key = ?",
        (constants.SCHEMA_VERSION_KEY,),
    ).fetchone()
    schema_version = (str(row[0]).strip() or None) if row else None
    state = _read_state(conn)
    physical = active = 0
    if state != {}:
        physical = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        active = conn.execute("SELECT COUNT(*) FROM corpus_membership").fetchone()[0]
    return InstalledCorpusSummary(schema_version, state, physical, active)


def read_installed_corpus_summary(db_path: Path) -> InstalledCorpusSummary:
    """Inspect a read-only snapshot, recovering a hot journal under native locks.

    :param Path db_path: Native metadata database to inspect.
    :return InstalledCorpusSummary: Schema, corpus state and counts; ordinary caches
        have empty state and zero corpus counts.
    :raises CorpusMetadataBusyError: If SQLite is locked by another connection.
    :raises CorpusRecoveryPendingError: If hot-journal recovery cannot complete.
    :raises sqlite3.Error: If metadata is otherwise unreadable.
    :raises OSError: If read-only metadata access fails.
    :raises filelock.Timeout: If recovery conflicts with an active cache operation.
    """
    try:
        with closing(
            sqlite3.connect(
                db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0
            )
        ) as conn:
            return _read_installed_corpus_summary(conn)
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc):
            raise CorpusMetadataBusyError(str(exc)) from exc
        if "readonly" not in str(exc):
            raise
    namespace_id = db_path.stem.removeprefix("metadata_")
    try:
        with _existing_namespace_connection(
            namespace_id, db_path.parent, blocking=False
        ) as conn:
            return _read_installed_corpus_summary(conn)
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc):
            raise CorpusMetadataBusyError(str(exc)) from exc
        raise CorpusRecoveryPendingError(str(exc)) from exc
    except Timeout:
        raise
    except OSError as exc:
        raise CorpusRecoveryPendingError(str(exc)) from exc


class _LifecycleMixin(_EmbeddingCacheHost):
    """Operations that preserve a corpus's fixed contract and certified membership."""

    @contextmanager
    def _inspect_db(self) -> Iterator[sqlite3.Connection]:
        """Open existing SQLite state, allowing hot-journal rollback.

        The namespace lock must be held.

        Yields:
            The connection, closed on context exit.
        """
        with _existing_database(self.db_path) as conn:
            yield conn

    def _has_corpus_state(self) -> bool:
        """Return whether this database contains nonempty corpus state."""
        if not self.db_path.exists():
            return False
        with self._inspect_db() as conn:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='cache_metadata'"
            ).fetchone():
                return False
            return bool(_read_state(conn))

    def _inspect_existing_locked(
        self, *, contract: EmbeddingContract | None = None, allow_legacy: bool = False
    ) -> CorpusState:
        """Validate existing SQLite state and HDF5 layout without initializing them.

        :param EmbeddingContract | None contract: Optional contract required of the cache.
        :param bool allow_legacy: Accept matching legacy fingerprint metadata when
            no corpus state has been recorded.
        :return CorpusState: The persisted state, or an empty mapping for a legacy cache.
        :raises RuntimeError: If the existing files do not form a readable cache.
        :raises ValueError: If the stored contract or matrix differs from ``contract``.
        """
        if not self.db_path.is_file():
            raise RuntimeError(
                "Existing embedding cache requires both SQLite and HDF5 files; "
                "the existing files were preserved"
            )
        with (
            self._inspect_db() as conn,
            h5py.File(self.h5_path, "r")
            if self.h5_path.is_file()
            else nullcontext() as h5,
        ):
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
            count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            if h5 is None:
                pending = state.get("pending")
                if (
                    pending
                    and pending["kind"] == "refresh"
                    and not state.get("source")
                    and not state.get("membership_complete")
                    and count == 0
                    and conn.execute(
                        "SELECT 1 FROM replacement_journal LIMIT 1"
                    ).fetchone()
                    is None
                ):
                    return state
                raise RuntimeError(
                    "Existing embedding cache requires both SQLite and HDF5 files; "
                    "the existing files were preserved"
                )
            dataset = self._get_embeddings_dataset(h5)
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

    def corpus_state(self) -> CorpusState:
        """Read the source and operation state without model construction.

        :return CorpusState: State stored in the cache metadata table.
        """
        with self.namespace_lock_only(), self._inspect_db() as conn:
            return _read_state(conn)

    def _require_ready_locked(self, conn: sqlite3.Connection) -> CorpusState:
        """Return corpus state after rejecting pending or uncertified membership.

        :param sqlite3.Connection conn: Open cache database connection.
        :return CorpusState: Ready state, or an empty mapping for an unmanaged cache.
        :raises RuntimeError: If an operation is pending or membership is uncertified.
        """
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

    def require_complete_certified_corpus(
        self, *, corpus_size: int | None = None
    ) -> CorpusState:
        """Require ready state with explicitly certified source membership.

        :param int | None corpus_size: Optional newest-N view within installed coverage.
        :return CorpusState: Complete persisted corpus state.
        :raises RuntimeError: If membership has not been explicitly reconciled.
        :raises ValueError: If the view is invalid or exceeds installed coverage.
        """
        with self.namespace_lock_only(), self._inspect_db() as conn:
            state = self._require_corpus_view_locked(conn, corpus_size)
            if not state.get("membership_complete"):
                raise RuntimeError(
                    "Export requires explicitly reconciled corpus membership"
                )
            return state

    def _require_write_owned_locked(self, conn: sqlite3.Connection) -> None:
        """Require writes to respect any pending or manually managed operation.

        :param sqlite3.Connection conn: Open cache database connection.
        :raises RuntimeError: If this thread does not own the pending operation.
        """
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
        """Return the twelve-character native namespace ID from the DB filename."""
        return self.db_path.stem.removeprefix("metadata_")

    def _require_corpus_view_locked(
        self, conn: sqlite3.Connection, corpus_size: int | None
    ) -> CorpusState:
        """Require a ready corpus whose certified coverage contains the view.

        :param sqlite3.Connection conn: Open cache database connection.
        :param int | None corpus_size: Optional newest-N limit within installed coverage.
        :return CorpusState: Ready persisted state, or an unmanaged cache's empty state.
        :raises RuntimeError: If corpus membership is pending or uncertified.
        :raises ValueError: If the view is nonpositive or exceeds installed coverage.
        """
        state = self._require_ready_locked(conn)
        if corpus_size is not None:
            if not state.get("membership_complete"):
                raise RuntimeError(
                    "Exact corpus views require certified source membership"
                )
            if corpus_size < 1:
                raise ValueError("corpus_size must be positive")
            source = state["source"]
            if source["selection"] == "newest" and corpus_size > cast(
                int, source["limit"]
            ):
                raise ValueError(
                    f"Requested corpus view {corpus_size} exceeds installed coverage "
                    f"{source['limit']}"
                )
        return state

    def _corpus_view_query_locked(
        self, conn: sqlite3.Connection, corpus_size: int | None
    ) -> tuple[str, tuple[int, ...]] | None:
        """Build SQL for the active corpus view.

        Whole-coverage requests use unordered membership; smaller views rank
        members by newest submission.

        :param sqlite3.Connection conn: Open cache database connection.
        :param int | None corpus_size: Optional newest-N limit within installed coverage.
        :return tuple[str, tuple[int, ...]] | None: Row query and bindings, or ``None`` for unmanaged caches.
        """
        state = self._require_corpus_view_locked(conn, corpus_size)
        if not state.get("membership_complete"):
            return None
        query = (
            "SELECT p.row_idx FROM corpus_membership m "
            "JOIN papers p ON p.paper_id=m.paper_id"
        )
        if corpus_size is None:
            return query, ()
        source = state["source"]
        if source["selection"] == "newest" and corpus_size == source["limit"]:
            return query, ()
        reconciled = conn.execute(
            "SELECT value FROM cache_metadata WHERE key = ?",
            (constants.HYDRATION_RECONCILED_CACHE_ROWS_KEY,),
        ).fetchone()
        if (
            reconciled
            and reconciled[0].isdecimal()
            and corpus_size >= int(reconciled[0])
        ):
            return query, ()
        return (
            query
            + " ORDER BY "
            + newest_corpus_order_sql("p.chronology_key", "m.source_ordinal")
            + " LIMIT ?",
            (corpus_size,),
        )

    def _eligible_rows_locked(
        self,
        conn: sqlite3.Connection,
        row_count: int,
        corpus_size: int | None = None,
    ) -> npt.NDArray[np.bool_] | None:
        """Build the physical-row mask for the active corpus view.

        :param sqlite3.Connection conn: Open cache database connection.
        :param int row_count: Number of physical embedding rows.
        :param int | None corpus_size: Optional newest-N active-member limit.
        :return npt.NDArray[np.bool_] | None: Eligible rows, or no unmanaged filter.
        """
        view = self._corpus_view_query_locked(conn, corpus_size)
        if view is None:
            return None
        query, parameters = view
        mask = np.zeros(row_count, dtype=bool)
        cursor = conn.execute(query, parameters)
        for rows in iter(lambda: cursor.fetchmany(8192), []):
            indices = np.asarray([row[0] for row in rows], dtype=np.int64)
            if np.any(indices < 0) or np.any(indices >= row_count):
                raise RuntimeError("Corpus membership refers to missing vector rows")
            mask[indices] = True
        return mask

    def get_cached_embeddings(
        self, paper_ids: Sequence[str], *, corpus_size: int | None = None
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Read active cached vectors without invoking an encoder.

        :param Sequence[str] paper_ids: Paper IDs whose stored vectors to read.
        :param int | None corpus_size: Optional newest-N view within active membership.
        :return dict[str, npt.NDArray[np.float32]]: Normalized float32 vectors keyed by
            paper ID; missing or out-of-view IDs are omitted.
        :raises RuntimeError: If corpus membership is unavailable or recovery fails.
        :raises OSError: If a populated namespace's vector storage is unavailable.
        """
        with self._cache_lock(), self._connect_db() as conn:
            view = self._corpus_view_query_locked(conn, corpus_size)
            self._recover_pending_replacements_with_connection_locked(conn)
            if (
                not path_exists(self.h5_path)
                and view is None
                and conn.execute("SELECT 1 FROM papers LIMIT 1").fetchone() is None
            ):
                return {}
            with h5py.File(self.h5_path, "r") as h5:
                dataset = self._get_embeddings_dataset(h5)
                if dataset is None:
                    return {}
                cached_rows = [
                    (row[0], cast(int, row[2]))
                    for row in self._query_paper_rows(
                        conn,
                        list(dict.fromkeys(paper_ids)),
                        lookup_column="paper_id",
                        corpus_view=view,
                    )
                ]
                return self._load_cached_embeddings(h5, dataset, cached_rows)

    @contextmanager
    def operation_session(
        self: _LifecycleCacheT,
        *,
        kind: Literal["import", "refresh"],
        identity: OperationIdentity,
        target_source: CorpusSource,
        contract: EmbeddingContract,
        namespace_selectors: NamespaceSelectors,
        origin: CorpusOrigin | None = None,
        producer_provenance: Mapping[str, object] | None = None,
    ) -> Iterator[CorpusOperationSession[_LifecycleCacheT]]:
        """Own cache writes while recording an import or refresh operation.

        :param str kind: Operation kind, either ``"import"`` or ``"refresh"``.
        :param OperationIdentity identity: Stable identity used to resume the operation.
        :param CorpusSource target_source: Source selection being installed.
        :param EmbeddingContract contract: Fixed vector storage contract.
        :param NamespaceSelectors namespace_selectors: Inputs that reproduce the native namespace.
        :param CorpusOrigin | None origin: Optional import origin recorded with the corpus.
        :param Mapping[str, object] | None producer_provenance: Producer metadata carried into cache state.
        Yields:
            The operation session that validates and records corpus batches.
        """
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
                self._operation_owner: tuple[int, str] | None = None

    @contextmanager
    def export_snapshot_view(
        self: _LifecycleCacheT,
    ) -> Iterator[NativeSnapshotView]:
        """Hold the locks needed to read a complete corpus snapshot.

        Yields:
            A bounded view over active metadata rows and their vectors.
        :raises RuntimeError: If the corpus membership is not certified.
        """
        with self.hydration_operation_lock(), self.namespace_lock_only():
            with self._inspect_db() as conn:
                state = self._require_ready_locked(conn)
                if not state.get("membership_complete"):
                    raise RuntimeError(
                        "Export requires explicitly reconciled corpus membership"
                    )
                with h5py.File(self.h5_path, "r") as h5:
                    yield NativeSnapshotView(self, conn, h5, state)


class NativeSnapshotView:
    """Bounded read view held under the corpus and namespace locks."""

    def __init__(
        self,
        cache: _EmbeddingCacheHost,
        conn: sqlite3.Connection,
        h5: h5py.File,
        state: CorpusState,
    ) -> None:
        """Create a read view over cache handles held by the caller.

        :param _EmbeddingCacheHost cache: Cache supplying the native row layout and locks.
        :param sqlite3.Connection conn: Open metadata database connection.
        :param h5py.File h5: Open read-only embedding matrix file.
        :param CorpusState state: Complete corpus state returned by the cache.
        """
        self.cache, self.conn, self.h5 = cache, conn, h5
        self.contract = EmbeddingContract(**state["contract"])
        self.completed_source = CorpusSource(**state["source"])
        self.namespace_serialization = state["namespace_serialization"]
        self.namespace_selectors = state["namespace_selectors"]
        self.source_records_examined = state["source_records_examined"]
        self.producer_provenance = state.get("producer_provenance", {})
        self.origin = state.get("origin")
        self.calibration_ranges: npt.NDArray[np.float32] | None = (
            np.asarray(h5[constants.CALIBRATION_RANGES_DATASET_NAME])
            if self.contract.stored_dtype == "int8"
            else None
        )

    def iter_active_rows(
        self, *, batch_size: int = 8192
    ) -> Iterator[tuple[list[ActiveCorpusRecord], StoredEmbedding]]:
        """Yield active metadata and vectors in ascending physical row order.

        :param int batch_size: Maximum number of rows in each yielded batch.
        Yields:
            Metadata records and their matching stored embedding matrix batch.
        """
        cursor = self.conn.execute(
            "SELECT p.paper_id,p.text_hash,p.row_idx,p.title,p.abstract,p.year,"
            "p.authors_json,p.categories_json,p.venue,p.arxiv_id,p.doi,"
            "p.chronology_key,m.source_ordinal FROM corpus_membership m "
            "JOIN papers p ON p.paper_id=m.paper_id ORDER BY p.row_idx"
        )
        dataset = self.h5.get(constants.EMBEDDINGS_DATASET_NAME)
        while rows := cursor.fetchmany(batch_size):
            records: list[ActiveCorpusRecord] = []
            for row in rows:
                (
                    paper_id,
                    text_hash,
                    row_idx,
                    title,
                    abstract,
                    year,
                    authors_json,
                    categories_json,
                    venue,
                    arxiv_id,
                    doi,
                    chronology_key,
                    source_ordinal,
                ) = row
                records.append(
                    {
                        "paper_id": paper_id,
                        "text_hash": text_hash,
                        "row_idx": row_idx,
                        "title": title,
                        "abstract": abstract,
                        "year": year,
                        "authors": _parse_json_list(authors_json),
                        "categories": _parse_json_list(categories_json),
                        "venue": venue,
                        "arxiv_id": arxiv_id,
                        "doi": doi,
                        "chronology_key": chronology_key,
                        "source_ordinal": source_ordinal,
                    }
                )
            indices = np.asarray(
                [record["row_idx"] for record in records], dtype=np.int64
            )
            yield records, np.asarray(dataset[indices])


class CorpusOperationSession(Generic[_CacheT]):
    """An operation owns writes while its pending identity remains persisted."""

    def __init__(
        self,
        cache: _CacheT,
        *,
        kind: Literal["import", "refresh"],
        identity: OperationIdentity,
        target_source: CorpusSource,
        contract: EmbeddingContract,
        namespace_selectors: NamespaceSelectors,
        origin: CorpusOrigin | None = None,
        producer_provenance: Mapping[str, object] | None = None,
    ) -> None:
        """Create a session for one owned corpus import or refresh.

        :param EmbeddingCache cache: Cache whose namespace this session owns.
        :param str kind: Operation kind, either ``"import"`` or ``"refresh"``.
        :param OperationIdentity identity: Stable operation identity persisted for resume.
        :param CorpusSource target_source: Source selection to commit.
        :param EmbeddingContract contract: Fixed vector storage contract.
        :param NamespaceSelectors namespace_selectors: Inputs that reproduce the native namespace.
        :param CorpusOrigin | None origin: Optional import origin stored with corpus state.
        :param Mapping[str, object] | None producer_provenance: Producer metadata carried into cache state.
        :raises ValueError: If ``kind`` is not ``"import"`` or ``"refresh"``.
        """
        if kind not in {"import", "refresh"}:
            raise ValueError("Unsupported corpus operation")
        self.cache = cache
        self.kind, self.identity = kind, identity
        self.target_source, self.contract = target_source, contract
        self.namespace_selectors, self.origin = namespace_selectors, origin
        self.producer_provenance = producer_provenance
        self.already_complete = False
        self._committed_rows: int | None = None

    def begin(self) -> None:
        """Persist the pending identity and begin owning cache writes.

        :raises ValueError: If the namespace, source, or pending identity conflicts.
        """
        require_namespace_contract(
            self.contract, self.namespace_selectors, self.cache.model_name
        )
        with self.cache.namespace_lock_only(), self.cache._connect_db() as conn:
            conn.execute(CORPUS_MEMBERSHIP_TABLE_CREATE_SQL)
            conn.execute(CORPUS_MEMBERSHIP_ORDINAL_INDEX_SQL)
            state = _read_state(conn)
            if state.get("contract") and state["contract"] != asdict(self.contract):
                raise ValueError("Occupied cache has a different embedding contract")
            pending = state.get("pending")
            existing_source = state.get("source") or (
                pending["target_source"] if pending else None
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
            state["contract"] = self.contract.to_dict()
            state["namespace_serialization"] = self.cache.model_name
            state["namespace_selectors"] = self.namespace_selectors
            state["update_policy"] = "manual"
            state["membership_complete"] = False
            state["pending"] = {
                "kind": self.kind,
                "identity": self.identity,
                "target_source": self.target_source.to_dict(),
            }
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
        if self.kind == "refresh":
            with self.cache.namespace_lock_only():
                if not self.cache.h5_path.exists():
                    self.cache._inspect_existing_locked(contract=self.contract)
                    with h5py.File(self.cache.h5_path, "a") as h5:
                        self.cache._set_h5_attrs(h5)
                        self.cache._flush_h5_file(h5)
        self.cache._operation_owner = (threading.get_ident(), _json(self.identity))

    @property
    def committed_rows(self) -> int:
        """Return the dense prefix of vector rows committed in the cache."""
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
        """Return the number of active source records in corpus membership."""
        with self.cache.namespace_lock_only(), self.cache._inspect_db() as conn:
            return int(
                conn.execute("SELECT COUNT(*) FROM corpus_membership").fetchone()[0]
            )

    def validate_records(
        self, records: Sequence[_CorpusRecordT]
    ) -> list[_CorpusRecordT]:
        """Remove identical duplicate IDs and reject conflicting metadata.

        :param Sequence[_CorpusRecordT] records: Normalized source records in one batch.
        :return list[_CorpusRecordT]: Unique records in their first-seen order.
        :raises ValueError: If duplicate or previously stored metadata conflicts.
        """
        unique: dict[str, _CorpusRecordT] = {}
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
            seen: list[str] = []
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

    def _insert_membership(
        self,
        conn: sqlite3.Connection,
        records: Sequence[CorpusRecord | CorpusSourceRecord],
    ) -> None:
        """Insert source IDs and ordinals without replacing existing membership.

        :param sqlite3.Connection conn: Open cache database connection.
        :param Sequence[CorpusRecord | CorpusSourceRecord] records: Validated source records to add.
        """
        conn.executemany(
            "INSERT OR IGNORE INTO corpus_membership (paper_id,source_ordinal) VALUES (?,?)",
            [(record["paper_id"], record["source_ordinal"]) for record in records],
        )

    def record_membership(self, records: Sequence[_CorpusRecordT]) -> None:
        """Record active source rows whose vectors are already committed.

        :param Sequence[_CorpusRecordT] records: Normalized source records to certify.
        :raises ValueError: If a record conflicts or lacks a committed vector row.
        """
        records = self.validate_records(records)
        with self.cache.namespace_lock_only(), self.cache._connect_db() as conn:
            self.cache._require_write_owned_locked(conn)
            existing = self.cache._load_existing_rows(
                conn, [record["paper_id"] for record in records]
            )
            if len(existing) != len(records):
                raise ValueError("Membership requires a committed vector row")
            self._insert_membership(conn, records)

    def append_stored_embeddings(
        self,
        records: Sequence[CorpusRecord],
        vectors: StoredEmbedding,
        *,
        first_row: int,
    ) -> int:
        """Append an import batch at the current dense vector-row prefix.

        The method stores the vectors, paper metadata, and active membership in
        their native cache files.

        :param Sequence[CorpusRecord] records: Paper metadata paired with the vectors.
        :param StoredEmbedding vectors: Matrix in the contract's stored dtype and width.
        :param int first_row: Physical row at which this batch begins.
        :return int: Number of vector rows committed after this batch.
        :raises ValueError: If dtype, shape, row order, or IDs conflict.
        """
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
        """Certify completed membership and publish source state to the cache.

        :param int expected_rows: Number of active records expected in membership.
        :param int source_records_examined: Number of source records scanned.
        :raises ValueError: If membership or matrix row coverage is incomplete.
        """
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
            state["source"] = self.target_source.to_dict()
            state["membership_complete"] = True
            state["membership_version"] = 1
            state["source_records_examined"] = source_records_examined
            if self.kind == "import":
                # The operation kind defines which identity shape this call uses.
                state["completed_import_identity"] = cast(
                    PortableImportIdentity, self.identity
                )
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


def _cache_shell(
    cache_dir: Path | None,
    namespace_serialization: str,
    contract: EmbeddingContract,
) -> EmbeddingCache:
    """Construct an uninitialized cache object for an existing namespace.

    :param Path | None cache_dir: Directory containing the namespace files, or ``None`` for the configured cache directory.
    :param str namespace_serialization: Serialized native namespace address.
    :param EmbeddingContract contract: Vector contract expected by the cache.
    :return EmbeddingCache: Cache object configured for inspection or import.
    """
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
    namespace_id: str,
    cache_dir: Path | None = None,
    *,
    require_ready: bool = True,
) -> ExistingCacheSelection:
    """Resolve and open a native cache using its persisted source and contract.

    :param str namespace_id: Twelve-character native namespace ID.
    :param Path | None cache_dir: Optional directory containing the namespace files.
    :param bool require_ready: Require completed, certified corpus membership.
    :return ExistingCacheSelection: Selection metadata for the installed cache.
    :raises RuntimeError: If the cache has no certified embedding contract.
    :raises ValueError: If the namespace ID is invalid or is not installed.
    """
    with _existing_namespace_connection(namespace_id, cache_dir) as conn:
        state = _read_state(conn)
    directory = (
        Path(cache_dir) if cache_dir else get_cache_dir(create=False) / "embeddings"
    )
    if not state or not state.get("contract"):
        raise RuntimeError(
            "Selected cache has no certified embedding contract; refresh/adopt it first"
        )
    contract = EmbeddingContract(**state["contract"])
    pending = state.get("pending")
    source = state.get("source") or (pending["target_source"] if pending else None)
    # Keep the original source-field access behavior for incomplete stored state.
    source = cast(CorpusSourceData, source)
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


def read_existing_cache_metadata(
    namespace_id: str, cache_dir: Path | None = None
) -> dict[str, str]:
    """Read SQLite cache metadata under the native cache locks.

    :param str namespace_id: Twelve-character native namespace ID.
    :param Path | None cache_dir: Optional directory containing the namespace files.
    :return dict[str, str]: Cache metadata keyed by name.
    :raises ValueError: If the namespace ID is invalid or is not installed.
    """
    with _existing_namespace_connection(namespace_id, cache_dir) as conn:
        return dict(conn.execute("SELECT key,value FROM cache_metadata"))


def open_existing_cache(
    selection: ExistingCacheSelection,
    cache_dir: Path | None = None,
    *,
    require_ready: bool = True,
    allow_legacy: bool = False,
) -> EmbeddingCache:
    """Open an existing native cache and enforce its selected contract.

    :param ExistingCacheSelection selection: Contract and source to match.
    :param Path | None cache_dir: Optional directory containing the namespace files.
    :param bool require_ready: Require completed, certified corpus membership.
    :param bool allow_legacy: Allow matching legacy fingerprint metadata.
    :return EmbeddingCache: Cache configured for strict use of the selected namespace.
    :raises RuntimeError: If the selected corpus is incomplete or uncertified.
    :raises ValueError: If the selected address or contract differs.
    """
    cache = _cache_shell(
        cache_dir, selection.namespace_serialization, selection.contract
    )
    if cache.db_path.stem != "metadata_" + selection.namespace_id:
        raise ValueError(
            "Stored namespace serialization does not match selected address"
        )
    with cache.namespace_lock_only():
        state = cache._inspect_existing_locked(
            contract=selection.contract, allow_legacy=allow_legacy
        )
        if allow_legacy and not state:
            cache._init_db()
        if require_ready:
            with cache._inspect_db() as conn:
                state = cache._require_ready_locked(conn)
                if not state.get("membership_complete"):
                    raise RuntimeError("Selected cache membership is not certified")
    cache._strict_existing = True
    return cache


def preflight_import_destination(
    cache_dir: Path | None,
    namespace_serialization: str,
    contract: EmbeddingContract,
    source: CorpusSource,
    identity: OperationIdentity,
    namespace_selectors: NamespaceSelectors | None = None,
    calibration_ranges: npt.NDArray[np.float32] | None = None,
    origin: CorpusOrigin | None = None,
) -> Literal["create", "resume_same_import", "already_complete"]:
    """Classify whether an import destination can be created or resumed.

    :param Path | None cache_dir: Destination directory, or ``None`` for the configured cache directory.
    :param str namespace_serialization: Serialized native namespace address.
    :param EmbeddingContract contract: Embedding contract in the snapshot.
    :param CorpusSource source: Source represented by the snapshot.
    :param OperationIdentity identity: Stable identity of the requested import.
    :param NamespaceSelectors | None namespace_selectors: Inputs that reproduce the native namespace.
    :param npt.NDArray[np.float32] | None calibration_ranges: Optional exact int8 calibration ranges.
    :param CorpusOrigin | None origin: Optional snapshot or publication origin.
    :return Literal["create", "resume_same_import", "already_complete"]: Destination action.
    :raises ValueError: If the destination is occupied by a conflicting cache.
    """
    # Preserve the existing namespace validator's handling of a missing selector map.
    require_namespace_contract(
        contract, cast(NamespaceSelectors, namespace_selectors), namespace_serialization
    )
    cache = _cache_shell(cache_dir, namespace_serialization, contract)
    with cache.hydration_operation_lock(), cache.namespace_lock_only():
        return _classify_import_locked(
            cache, contract, source, identity, calibration_ranges, origin
        )


def _classify_import_locked(
    cache: EmbeddingCache,
    contract: EmbeddingContract,
    source: CorpusSource,
    identity: OperationIdentity,
    ranges: npt.NDArray[np.float32] | None,
    origin: CorpusOrigin | None,
) -> Literal["create", "resume_same_import", "already_complete"]:
    """Classify a destination while the hydration and namespace locks are held.

    :param EmbeddingCache cache: Cache whose destination files are being inspected.
    :param EmbeddingContract contract: Embedding contract in the snapshot.
    :param CorpusSource source: Source represented by the snapshot.
    :param OperationIdentity identity: Stable identity of the requested import.
    :param npt.NDArray[np.float32] | None ranges: Optional exact int8 calibration ranges.
    :param CorpusOrigin | None origin: Optional snapshot or publication origin.
    :return Literal["create", "resume_same_import", "already_complete"]: Destination action.
    :raises ValueError: If existing files conflict with the requested import.
    """
    db, h5 = cache.db_path.exists(), cache.h5_path.exists()
    if not db and not h5:
        return "create"
    if not db:
        raise ValueError(
            "Occupied import destination has HDF5 without SQLite; files preserved"
        )
    with cache._inspect_db() as conn:
        try:
            state = _read_state(conn)
            count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        except sqlite3.DatabaseError as exc:
            raise ValueError(
                "Occupied import database is unreadable; files preserved"
            ) from exc
    pending = state.get("pending")
    if state.get("contract") != asdict(contract):
        raise ValueError(
            "Occupied import destination has a different or unknown contract"
        )
    action: Literal["create", "resume_same_import", "already_complete"]
    if pending:
        if pending.get("kind") != "import" or pending.get("identity") != identity:
            publication = state.get("origin")
            resume = ""
            if publication is not None and all(
                publication.get(key)
                for key in ("bucket_id", "config_name", "snapshot_id")
            ):
                # The truthy publication fields above define the values used in this hint.
                publication_values = cast(PublicationOrigin, publication)
                resume = " Resume with: " + shlex.join(
                    [
                        "citemesh",
                        "cache",
                        "import",
                        "--bucket",
                        publication_values["bucket_id"],
                        "--config",
                        publication_values["config_name"],
                        "--snapshot",
                        publication_values["snapshot_id"],
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
    cache_dir: Path | None,
    namespace_serialization: str,
    contract: EmbeddingContract,
    source: CorpusSource,
    identity: OperationIdentity,
    namespace_selectors: NamespaceSelectors,
    calibration_ranges: npt.NDArray[np.float32] | None = None,
    origin: CorpusOrigin | None = None,
    producer_provenance: Mapping[str, object] | None = None,
) -> Iterator[CorpusOperationSession[EmbeddingCache]]:
    """Create or resume an import session for a portable snapshot.

    The destination is classified before initialization; a matching completed
    import yields a session marked ``already_complete``.

    :param Path | None cache_dir: Destination directory, or ``None`` for the configured cache directory.
    :param str namespace_serialization: Serialized native namespace address.
    :param EmbeddingContract contract: Embedding contract in the snapshot.
    :param CorpusSource source: Source represented by the snapshot.
    :param OperationIdentity identity: Stable identity used to recognize an interrupted import.
    :param NamespaceSelectors namespace_selectors: Inputs that reproduce the native namespace.
    :param npt.NDArray[np.float32] | None calibration_ranges: Exact ranges for int8 snapshots.
    :param CorpusOrigin | None origin: Optional snapshot or publication origin.
    :param Mapping[str, object] | None producer_provenance: Producer metadata carried into cache state.
    Yields:
        The owned operation session used to append and certify rows.
    :raises ValueError: If the destination conflicts with the requested import.
    """
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
                    state: CorpusState = {
                        "contract": contract.to_dict(),
                        "namespace_serialization": namespace_serialization,
                        "namespace_selectors": namespace_selectors,
                        "origin": origin,
                        "update_policy": "manual",
                        "membership_complete": False,
                        "pending": {
                            "kind": "import",
                            "identity": identity,
                            "target_source": source.to_dict(),
                        },
                    }
                    _write_state(conn, state)
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
