"""Source certification and native import durability independent of Arrow or models."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from filelock import FileLock

from citemesh.cli import cache_ops
from citemesh.data.cache import cache_operation_lock
from citemesh.data.embedding_cache import EmbeddingCache, constants, lifecycle
from citemesh.data.embedding_cache.lifecycle import (
    import_destination_session,
    open_existing_cache,
    preflight_import_destination,
    select_existing_cache,
)
from citemesh.data.embedding_contract import (
    CorpusSource,
    EmbeddingContract,
    serialize_embedding_namespace,
)
from tests._helpers import BlockingEncodeModel, ConstantEncodeModel


def _settings(tmp_path, dtype="float32", dimension=4):
    contract = EmbeddingContract(
        model_id="test/model",
        model_revision="a" * 40,
        model_artifact_identity="hf::test/model::" + "a" * 40,
        profile_key="test",
        profile_schema="test-v1",
        representation="retrieval-document-v1",
        formatter_identity="formatter",
        normalization="l2-v1",
        dimension=dimension,
        stored_dtype=dtype,
        quantization="affine-int8-v1" if dtype == "int8" else None,
        calibration_sample_size=20 if dtype == "int8" else None,
    )
    selectors = dict(
        model_identity=contract.model_id,
        requested_revision="main",
        artifact_identity=contract.model_artifact_identity,
        profile_schema=contract.profile_schema,
        representation=contract.representation,
        normalization=contract.normalization,
        dimension=dimension,
        stored_dtype=dtype,
        calibration_sample_size=contract.calibration_sample_size,
        formatter_identity=contract.formatter_identity,
        semantic_source="arxiv-corpus",
    )
    return dict(
        cache_dir=tmp_path,
        namespace_serialization=serialize_embedding_namespace(**selectors),
        contract=contract,
        source=CorpusSource("test/source", None, "train", "b" * 40, "all", None),
        identity={"snapshot_id": "snapshot-one"},
        namespace_selectors=selectors,
        calibration_ranges=np.vstack(
            (np.full(dimension, -2), np.full(dimension, 2))
        ).astype(np.float32)
        if dtype == "int8"
        else None,
    )


def _records():
    return [
        dict(
            paper_id=f"paper-{i}",
            title=f"Title {i}",
            abstract=f"Text {i}",
            text_hash=f"hash-{i}",
            authors=[f"Author {i}"],
            categories=["cs.CL"],
            year=2020 + i,
            arxiv_id="",
            doi="",
            venue="Test",
            chronology_key=chrono,
            source_ordinal=ordinal,
        )
        for i, (chrono, ordinal) in enumerate([(10, 1), (20, 2), (20, 3), (None, 0)])
    ]


def _populate(settings):
    records = _records()
    vectors = np.eye(4, dtype=np.float32)
    if settings["contract"].stored_dtype == "int8":
        vectors = (vectors * 100).astype(np.int8)
    with import_destination_session(**settings) as session:
        session.append_stored_embeddings(records, vectors, first_row=0)
        session.finish(expected_rows=len(records), source_records_examined=len(records))
        return session.cache, records, vectors


@pytest.mark.parametrize("dtype", ["int8", "float32"])
def test_native_import_preserves_rows_and_uses_original_address(tmp_path, dtype):
    settings = _settings(tmp_path, dtype)
    cache, records, vectors = _populate(settings)
    selection = select_existing_cache(cache.namespace_id, tmp_path)
    assert "revision=main::" in selection.namespace_serialization
    assert selection.contract.model_revision == "a" * 40
    with open_existing_cache(selection, tmp_path).export_snapshot_view() as view:
        batches = list(view.iter_active_rows(batch_size=1))
        assert [batch[0][0]["source_ordinal"] for batch in batches] == [1, 2, 3, 0]
        np.testing.assert_array_equal(np.concatenate([v for _, v in batches]), vectors)
        assert view.completed_source == settings["source"]
    before = cache.db_path.read_bytes(), cache.h5_path.read_bytes()
    with import_destination_session(**settings) as session:
        assert session.already_complete
    assert before == (cache.db_path.read_bytes(), cache.h5_path.read_bytes())


@pytest.mark.parametrize(
    "dtype,binary", [("float32", False), ("int8", False), ("int8", True)]
)
def test_native_search_intersects_membership_and_exact_newest_view(
    tmp_path, monkeypatch, dtype, binary
):
    settings = _settings(tmp_path, dtype)
    cache, records, _ = _populate(settings)
    source = replace(settings["source"], revision="c" * 40)
    with cache.operation_session(
        kind="refresh",
        identity={"revision": source.revision},
        target_source=source,
        contract=settings["contract"],
        namespace_selectors=settings["namespace_selectors"],
    ) as session:
        session.record_membership(records[1:])
        session.finish(expected_rows=3, source_records_examined=3)
    monkeypatch.setattr(constants, "EMBEDDING_SEARCH_CHUNK_ROWS", 1)
    kwargs = dict(binary_prefilter=binary, binary_rescore_multiplier=10)
    results = cache.search(np.array([1, 0, 0, 0], dtype=np.float32), 10, **kwargs)
    assert {r.paper_id for r in results} == {"paper-1", "paper-2", "paper-3"}
    ids = [record["paper_id"] for record in records]
    for size, expected in [(None, {"paper-1", "paper-2", "paper-3"}), (1, {"paper-2"})]:
        assert cache.get_cached_paper_ids(corpus_size=size) == expected
        assert set(cache.get_paper_metadata_batch(ids, corpus_size=size)) == expected
        assert (
            set(cache.get_paper_metadata_alias_candidates(ids, corpus_size=size))
            == expected
        )
        assert set(cache.get_cached_embeddings(ids, corpus_size=size)) == expected
    results = cache.search(
        np.ones(4, dtype=np.float32),
        10,
        corpus_size=1,
        **kwargs,
    )
    assert [r.paper_id for r in results] == ["paper-2"]
    assert cache.last_search_physical_embeddings == 4
    assert cache.last_search_total_embeddings == 1
    assert cache.last_search_rescored_embeddings == 1
    results = cache.search(np.ones(4, dtype=np.float32), 10, corpus_size=3, **kwargs)
    assert {r.paper_id for r in results} == {"paper-1", "paper-2", "paper-3"}


def test_manual_corpus_rejects_hit_updates_and_ordinary_writes(tmp_path):
    settings = _settings(tmp_path)
    cache, records, _ = _populate(settings)

    class NeverEncode:
        def encode(self, *args, **kwargs):
            raise AssertionError("encoder must not be reached")

    before = cache.db_path.read_bytes(), cache.h5_path.read_bytes()
    with pytest.raises(RuntimeError, match="owned import/refresh"):
        cache.get_embeddings(
            {"paper-0": {**records[0], "venue": "Changed"}}, NeverEncode()
        )
    with pytest.raises(RuntimeError, match="owned import/refresh"):
        cache.upsert_embeddings({"new": {}}, NeverEncode())
    with pytest.raises(RuntimeError, match="owned import/refresh"):
        cache.update_corpus_metadata(
            [dict(paper_id="paper-0", year=2000, doi="", venue="Changed")]
        )
    assert before == (cache.db_path.read_bytes(), cache.h5_path.read_bytes())
    assert list(cache.get_cached_embeddings(["paper-0"])) == ["paper-0"]


def test_interrupted_append_recovers_committed_prefix_on_resume(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    records, vectors = _records(), np.eye(4, dtype=np.float32)
    with pytest.raises(RuntimeError, match="after vector append"):
        with import_destination_session(**settings) as session:
            session.append_stored_embeddings(records[:2], vectors[:2], first_row=0)
            original = session.cache._flush_h5_file

            def fail_after_flush(handle):
                original(handle)
                raise RuntimeError("after vector append")

            monkeypatch.setattr(session.cache, "_flush_h5_file", fail_after_flush)
            session.append_stored_embeddings(records[2:], vectors[2:], first_row=2)
    with import_destination_session(**settings) as session:
        assert session.committed_rows == 2
        session.append_stored_embeddings(records[2:], vectors[2:], first_row=2)
        session.finish(expected_rows=4, source_records_examined=4)
        with h5py.File(session.cache.h5_path, "r") as h5:
            np.testing.assert_array_equal(h5["embeddings"][:], vectors)


def test_pending_import_refuses_search_export_ingest_and_other_identity(tmp_path):
    settings = _settings(tmp_path)
    with import_destination_session(**settings) as session:
        session.append_stored_embeddings(
            _records()[:1], np.eye(4, dtype=np.float32)[:1], first_row=0
        )
        cache = session.cache
    with pytest.raises(RuntimeError, match="incomplete import"):
        cache.search(np.ones(4), 1, binary_prefilter=False, binary_rescore_multiplier=1)
    with pytest.raises(RuntimeError, match="incomplete import"):
        with cache.export_snapshot_view():
            pass
    with pytest.raises(RuntimeError, match="incomplete import"):
        cache.get_cached_paper_ids()
    with pytest.raises(RuntimeError, match="incomplete import"):
        cache.get_paper_metadata_batch(["paper-0"])
    with pytest.raises(RuntimeError, match="incomplete import"):
        cache.get_paper_metadata_alias_candidates(["paper-0"])
    with pytest.raises(RuntimeError, match="incomplete import"):
        cache.get_cached_embeddings(["paper-0"])
    with pytest.raises(RuntimeError, match="owned import/refresh"):
        cache.upsert_embeddings({"x": {}}, None)
    with pytest.raises(ValueError, match="Conflicting incomplete import"):
        preflight_import_destination(
            **{**settings, "identity": {"snapshot_id": "different"}}
        )


def test_strict_missing_half_preserves_database_and_never_repairs(tmp_path):
    settings = _settings(tmp_path)
    cache, _, _ = _populate(settings)
    cache.h5_path.unlink()
    before = cache.db_path.read_bytes()
    with pytest.raises(RuntimeError, match="both SQLite and HDF5"):
        EmbeddingCache(
            cache_dir=tmp_path,
            model_name=settings["namespace_serialization"],
            storage_precision="float32",
            text_formatter_fingerprint="formatter",
        )
    assert before == cache.db_path.read_bytes()
    assert not cache.h5_path.exists()


@pytest.mark.parametrize("missing_half", ["db", "h5"])
@pytest.mark.parametrize("require_ready", [True, False])
def test_strict_selection_refuses_each_missing_half(
    tmp_path, missing_half, require_ready
):
    settings = _settings(tmp_path)
    cache, _, _ = _populate(settings)
    selection = select_existing_cache(cache.namespace_id, tmp_path)
    missing_path = cache.db_path if missing_half == "db" else cache.h5_path
    remaining_path = cache.h5_path if missing_half == "db" else cache.db_path
    missing_path.unlink()
    before = remaining_path.read_bytes()
    with pytest.raises(RuntimeError, match="both SQLite and HDF5"):
        open_existing_cache(selection, tmp_path, require_ready=require_ready)
    assert remaining_path.read_bytes() == before
    assert not missing_path.exists()


def test_unreadable_import_database_is_preserved(tmp_path):
    settings = _settings(tmp_path)
    cache, _, _ = _populate(settings)
    cache.db_path.write_bytes(b"unreadable SQLite payload")
    before = cache.db_path.read_bytes(), cache.h5_path.read_bytes()
    with pytest.raises(ValueError, match="database is unreadable"):
        preflight_import_destination(**settings)
    assert (cache.db_path.read_bytes(), cache.h5_path.read_bytes()) == before


def test_initialization_without_pending_identity_is_preserved_and_refused(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)

    def interrupt_pending_identity(*args):
        raise RuntimeError("before pending identity")

    with monkeypatch.context() as patch:
        patch.setattr(lifecycle, "_write_state", interrupt_pending_identity)
        with pytest.raises(RuntimeError, match="before pending identity"):
            with import_destination_session(**settings):
                pytest.fail("Initialization should have been interrupted")
    database = next(tmp_path.glob("metadata_*.db"))
    before = database.read_bytes()
    assert not list(tmp_path.glob("*.h5"))
    with pytest.raises(ValueError, match="different or unknown contract"):
        with import_destination_session(**settings):
            pytest.fail("Ambiguous initialization must not be resumed")
    assert database.read_bytes() == before
    assert not list(tmp_path.glob("*.h5"))


def test_recognized_empty_import_initialization_resumes_missing_hdf5(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)

    def interrupt_hdf5_creation(*args, **kwargs):
        raise RuntimeError("before initial HDF5 creation")

    with monkeypatch.context() as patch:
        patch.setattr(lifecycle.h5py, "File", interrupt_hdf5_creation)
        with pytest.raises(RuntimeError, match="before initial HDF5 creation"):
            with import_destination_session(**settings):
                pytest.fail("Initialization should have been interrupted")
    assert not list(tmp_path.glob("*.h5"))
    with import_destination_session(**settings) as session:
        assert session.committed_rows == 0
        session.append_stored_embeddings(
            _records(), np.eye(4, dtype=np.float32), first_row=0
        )
        session.finish(expected_rows=4, source_records_examined=4)


def test_export_preserves_active_storage_order_with_interleaved_history(tmp_path):
    settings = _settings(tmp_path)
    cache, records, vectors = _populate(settings)
    source = replace(settings["source"], revision="c" * 40)
    with cache.operation_session(
        kind="refresh",
        identity={"revision": source.revision},
        target_source=source,
        contract=settings["contract"],
        namespace_selectors=settings["namespace_selectors"],
    ) as session:
        session.record_membership([records[3], records[1]])
        session.finish(expected_rows=2, source_records_examined=2)
    with cache.export_snapshot_view() as view:
        rows, stored = next(view.iter_active_rows(batch_size=10))
        assert [r["paper_id"] for r in rows] == ["paper-1", "paper-3"]
        np.testing.assert_array_equal(stored, vectors[[1, 3]])


def test_completion_transaction_failure_leaves_operation_unavailable(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    with import_destination_session(**settings) as session:
        session.append_stored_embeddings(
            _records(), np.eye(4, dtype=np.float32), first_row=0
        )
        cache = session.cache

        @contextmanager
        def fail_commit():
            conn = sqlite3.connect(cache.db_path)
            try:
                yield conn
                raise RuntimeError("completion commit interrupted")
            finally:
                conn.rollback()
                conn.close()

        with monkeypatch.context() as patch:
            patch.setattr(cache, "_connect_db", fail_commit)
            with pytest.raises(RuntimeError, match="completion commit"):
                session.finish(expected_rows=4, source_records_examined=4)
    assert cache.corpus_state()["pending"]["kind"] == "import"
    with import_destination_session(**settings) as session:
        assert session.committed_rows == 4
        session.finish(expected_rows=4, source_records_examined=4)


def test_fresh_import_preflight_does_not_initialize_payload(tmp_path):
    settings = _settings(tmp_path)
    assert preflight_import_destination(**settings) == "create"
    assert not list(tmp_path.glob("*.db"))
    assert not list(tmp_path.glob("*.h5"))
    namespace_id = hashlib.sha256(
        settings["namespace_serialization"].encode()
    ).hexdigest()[:12]
    (tmp_path / f"embeddings_{namespace_id}.h5").write_bytes(b"occupied")
    with pytest.raises(ValueError, match="HDF5 without SQLite"):
        preflight_import_destination(**settings)
    assert not list(tmp_path.glob("*.db"))


def test_import_rebuilds_binary_only_once_after_all_batches(
    tmp_path, monkeypatch, caplog
):
    settings = _settings(tmp_path, "int8")
    caplog.set_level("DEBUG")
    calls = []
    original = EmbeddingCache._rebuild_binary_dataset

    def observe(self, **kwargs):
        calls.append(int(kwargs["embeddings_dataset"].shape[0]))
        return original(self, **kwargs)

    monkeypatch.setattr(EmbeddingCache, "_rebuild_binary_dataset", observe)
    with import_destination_session(**settings) as session:
        for offset, record in enumerate(_records()):
            session.append_stored_embeddings(
                [record],
                (np.eye(4)[offset : offset + 1] * 100).astype(np.int8),
                first_row=offset,
            )
            assert calls == []
        session.finish(expected_rows=4, source_records_examined=4)
        assert calls == [4]
    rebuilds = [
        record
        for record in caplog.records
        if "Rebuilding binary index" in record.message
    ]
    assert len(rebuilds) == 1
    assert rebuilds[0].levelname == "DEBUG"


def test_refresh_prevents_later_import_of_bootstrap_snapshot(tmp_path):
    settings = _settings(tmp_path)
    cache, records, _ = _populate(settings)
    with cache.operation_session(
        kind="refresh",
        identity={"refresh": 1},
        target_source=settings["source"],
        contract=settings["contract"],
        namespace_selectors=settings["namespace_selectors"],
    ) as session:
        session.record_membership(records)
        session.finish(expected_rows=4, source_records_examined=4)
    with pytest.raises(ValueError, match="occupied"):
        preflight_import_destination(**settings)


def test_legacy_membership_table_is_unknown_until_explicit_reconciliation(tmp_path):
    cache = EmbeddingCache(
        cache_dir=tmp_path, model_name="legacy", storage_precision="float32"
    )
    with pytest.raises(RuntimeError, match="reconciled corpus membership"):
        with cache.export_snapshot_view():
            pass
    with sqlite3.connect(cache.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM corpus_membership").fetchone() == (0,)
    assert cache.corpus_state() == {}


def test_publisher_provenance_survives_native_import(tmp_path):
    settings = {
        **_settings(tmp_path),
        "producer_provenance": {"source_torch_dtype": "bfloat16"},
    }
    cache, _, _ = _populate(settings)
    with h5py.File(cache.h5_path, "r") as h5:
        assert h5.attrs[constants.SOURCE_TORCH_DTYPE_KEY] == "bfloat16"
    with sqlite3.connect(cache.db_path) as conn:
        assert conn.execute(
            "SELECT value FROM cache_metadata WHERE key=?",
            (constants.SOURCE_TORCH_DTYPE_KEY,),
        ).fetchone() == ("bfloat16",)
    with cache.export_snapshot_view() as view:
        assert view.producer_provenance == settings["producer_provenance"]


def test_writer_started_before_adoption_cannot_commit_after_certification(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    settings = _settings(tmp_path)
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        model_name=settings["namespace_serialization"],
        storage_precision="float32",
        text_formatter_fingerprint="formatter",
    )

    class ImmediateModel:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)

    cache.upsert_embeddings({"paper-0": _records()[0]}, ImmediateModel())
    entered, release = threading.Event(), threading.Event()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            cache.upsert_embeddings,
            {"new-paper": {"title": "New"}},
            BlockingEncodeModel(entered, release, 4),
        )
        assert entered.wait(5)
        try:
            with cache.operation_session(
                kind="refresh",
                identity={"revision": 1},
                target_source=settings["source"],
                contract=settings["contract"],
                namespace_selectors=settings["namespace_selectors"],
            ) as session:
                session.record_membership(_records()[:1])
                session.finish(expected_rows=1, source_records_examined=1)
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="owned import/refresh"):
            future.result(timeout=5)
    assert cache.embedding_count() == 1
    assert not cache.get_paper_metadata_batch(["new-paper"])


def test_strict_selection_preserves_existing_lzf_compression(tmp_path):
    settings = _settings(tmp_path)
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        model_name=settings["namespace_serialization"],
        storage_precision="float32",
        compression="lzf",
        compression_level=0,
        text_formatter_fingerprint="formatter",
    )
    with cache.operation_session(
        kind="import",
        identity=settings["identity"],
        target_source=settings["source"],
        contract=settings["contract"],
        namespace_selectors=settings["namespace_selectors"],
    ) as session:
        session.append_stored_embeddings(
            _records(), np.eye(4, dtype=np.float32), first_row=0
        )
        session.finish(expected_rows=4, source_records_examined=4)
    selected = open_existing_cache(
        select_existing_cache(cache.namespace_id, tmp_path), tmp_path
    )
    assert selected._effective_compression == "lzf"
    with h5py.File(selected.h5_path, "r") as h5:
        assert h5["embeddings"].compression == "lzf"
        np.testing.assert_array_equal(h5["embeddings"][:], np.eye(4, dtype=np.float32))


@pytest.mark.parametrize(
    "entrypoint",
    [
        "constructor",
        "selection",
        "metadata",
        "import",
        "scan",
        "scan_namespace_locked",
        "scan_root_locked",
        "scan_readonly",
    ],
)
@pytest.mark.parametrize("kind", ["import", "refresh"])
def test_existing_cache_entrypoints_recover_a_hot_sqlite_journal(
    tmp_path, entrypoint, kind, monkeypatch
):
    """Abrupt SQLite writes must recover before import or refresh resumes."""
    if entrypoint == "scan_root_locked":
        monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path))
    settings = _settings(tmp_path)
    with import_destination_session(**settings) as session:
        session.append_stored_embeddings(
            _records(), np.eye(4, dtype=np.float32), first_row=0
        )
        cache = session.cache
    if kind == "refresh":
        with import_destination_session(**settings) as session:
            session.finish(expected_rows=4, source_records_examined=4)
        with cache.operation_session(
            kind="refresh",
            identity={"source": settings["source"].to_dict()},
            target_source=settings["source"],
            contract=settings["contract"],
            namespace_selectors=settings["namespace_selectors"],
        ):
            pass
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sqlite3, sys; "
            "conn = sqlite3.connect(sys.argv[1]); "
            "conn.execute('PRAGMA cache_size=8'); "
            "conn.execute('BEGIN IMMEDIATE'); "
            "conn.executemany('INSERT INTO papers(paper_id, abstract) VALUES (?, ?)', "
            "((f'uncommitted-{i}', 'x' * 1500) for i in range(6000))); "
            "os._exit(9)",
            str(cache.db_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 9, child.stderr
    assert cache.db_path.with_name(cache.db_path.name + "-journal").is_file()
    connection = sqlite3.connect(cache.db_path.as_uri() + "?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("SELECT COUNT(*) FROM papers").fetchone()
    finally:
        connection.close()
    if entrypoint == "constructor":
        reopened = EmbeddingCache(
            cache_dir=tmp_path,
            model_name=settings["namespace_serialization"],
            storage_precision="float32",
            text_formatter_fingerprint=settings["contract"].formatter_identity,
        )
        assert reopened.corpus_state()["pending"]["kind"] == kind
    elif entrypoint == "selection":
        assert select_existing_cache(cache.namespace_id, tmp_path, require_ready=False)
    elif entrypoint == "metadata":
        assert lifecycle.read_existing_cache_metadata(cache.namespace_id, tmp_path)
    elif entrypoint.startswith("scan"):
        if entrypoint != "scan":
            if entrypoint == "scan_namespace_locked":
                lock = FileLock(str(cache.lock_path))
            elif entrypoint == "scan_root_locked":
                lock = cache_operation_lock(tmp_path, exclusive=True)
            else:
                payloads = list(tmp_path.iterdir())
                for path in payloads:
                    path.chmod(0o444)
                tmp_path.chmod(0o555)
                lock = nullcontext()
            try:
                with lock:
                    blocked_rows = cache_ops._scan_embedding_namespaces(tmp_path)
                assert len(blocked_rows) == 1
                assert blocked_rows[0].status.startswith("SQLite recovery pending")
                assert cache.db_path.with_name(
                    cache.db_path.name + "-journal"
                ).is_file()
            finally:
                if entrypoint == "scan_readonly":
                    tmp_path.chmod(0o755)
                    for path in payloads:
                        path.chmod(0o644)
        rows = cache_ops._scan_embedding_namespaces(tmp_path)
        assert len(rows) == 1
        assert f"pending {kind}" in rows[0].status
        assert f"active {4 if kind == 'import' else 0} / physical 4" in rows[0].status
    elif kind == "import":
        assert preflight_import_destination(**settings) == "resume_same_import"
    else:
        with pytest.raises(ValueError, match="Conflicting incomplete import"):
            preflight_import_destination(**settings)
    if kind == "import":
        with import_destination_session(**settings) as session:
            assert session.committed_rows == 4
            session.finish(expected_rows=4, source_records_examined=4)
    else:
        with cache.operation_session(
            kind="refresh",
            identity={"source": settings["source"].to_dict()},
            target_source=settings["source"],
            contract=settings["contract"],
            namespace_selectors=settings["namespace_selectors"],
        ) as session:
            session.record_membership(_records())
            session.finish(expected_rows=4, source_records_examined=4)
    assert cache.get_cached_paper_ids() == {record["paper_id"] for record in _records()}


def test_clear_releases_an_imported_namespace_for_ordinary_rebuild(tmp_path):
    """Clearing an installed cache must permit writes through that same object."""
    settings = _settings(tmp_path)
    cache, _, _ = _populate(settings)
    cache = open_existing_cache(
        select_existing_cache(cache.namespace_id, tmp_path), tmp_path
    )
    cache.clear("explicit rebuild")
    cache.upsert_embeddings(
        {"replacement": {"title": "Replacement"}},
        ConstantEncodeModel((1.0, 0.0, 0.0, 0.0)),
    )
    assert cache.get_cached_paper_ids() == {"replacement"}
    assert cache.corpus_state() == {}


def test_legacy_open_migrates_chronology_before_persisting_pending_state(tmp_path):
    """Adoption must migrate supported SQLite columns before refresh begins."""
    settings = _settings(tmp_path)
    cache, _, _ = _populate(settings)
    selection = select_existing_cache(cache.namespace_id, tmp_path)
    with cache._connect_db() as connection:
        connection.execute(
            "DELETE FROM cache_metadata WHERE key = ?", (lifecycle.CORPUS_STATE_KEY,)
        )
        connection.execute("DROP INDEX idx_papers_chronology_key")
        connection.execute("ALTER TABLE papers DROP COLUMN chronology_key")
    adopted = open_existing_cache(
        selection, tmp_path, require_ready=False, allow_legacy=True
    )
    with adopted._connect_db() as connection:
        assert "chronology_key" in {
            row[1] for row in connection.execute("PRAGMA table_info(papers)")
        }
    assert adopted.corpus_state() == {}
    with adopted.operation_session(
        kind="refresh",
        identity={"source": settings["source"].to_dict()},
        target_source=settings["source"],
        contract=settings["contract"],
        namespace_selectors=settings["namespace_selectors"],
    ) as session:
        session.record_membership(_records())
        session.finish(expected_rows=4, source_records_examined=4)
    assert adopted.corpus_state()["membership_complete"]


@pytest.mark.parametrize(
    "reader", [select_existing_cache, lifecycle.read_existing_cache_metadata]
)
def test_unknown_namespace_inspection_reports_not_installed(tmp_path, reader):
    """Inspection must report a missing namespace without creating its database."""
    directory = tmp_path / "absent"
    with pytest.raises(
        ValueError, match="Namespace deadbeef0000 is not installed; see cache scan"
    ):
        reader("deadbeef0000", directory)
    assert not directory.exists()


def test_installed_corpus_summary_can_inspect_an_active_import(tmp_path, monkeypatch):
    """Scan metadata must remain readable while an importer owns hydration."""
    monkeypatch.setattr(constants, "_resolve_cache_lock_timeout_seconds", lambda: 0.05)
    with import_destination_session(**_settings(tmp_path)) as session:
        summary = lifecycle.read_installed_corpus_summary(session.cache.db_path)
        assert summary.schema_version == str(constants.EMBEDDING_CACHE_SCHEMA_VERSION)
        assert summary.state["pending"]["kind"] == "import"
        assert summary.physical_rows == summary.active_rows == 0


@pytest.mark.parametrize("kind", ["import", "refresh"])
@pytest.mark.parametrize("reader", ["selection", "metadata"])
def test_namespace_inspection_does_not_wait_for_active_corpus_operation(
    tmp_path, monkeypatch, kind, reader
):
    """Metadata inspection must expose pending state without waiting for hydration."""
    monkeypatch.setattr(constants, "_resolve_cache_lock_timeout_seconds", lambda: 0.01)
    settings = _settings(tmp_path)
    if kind == "import":
        operation = import_destination_session(**settings)
    else:
        cache, _, _ = _populate(settings)
        operation = cache.operation_session(
            kind="refresh",
            identity={"source": settings["source"].to_dict()},
            target_source=settings["source"],
            contract=settings["contract"],
            namespace_selectors=settings["namespace_selectors"],
        )
    with operation as session:
        if kind == "import":
            session.append_stored_embeddings(
                _records(), np.eye(4, dtype=np.float32), first_row=0
            )
        if reader == "selection":
            with pytest.raises(RuntimeError, match=f"incomplete {kind} operation"):
                select_existing_cache(session.cache.namespace_id, tmp_path)
        else:
            metadata = lifecycle.read_existing_cache_metadata(
                session.cache.namespace_id, tmp_path
            )
            assert (
                json.loads(metadata[lifecycle.CORPUS_STATE_KEY])["pending"]["kind"]
                == kind
            )


def test_namespace_scan_reads_committed_state_while_native_locks_are_held(
    tmp_path, monkeypatch
):
    """Inventory must remain available while another operation owns namespace locks."""
    monkeypatch.setattr(constants, "_resolve_cache_lock_timeout_seconds", lambda: 0.01)
    cache, _, _ = _populate(_settings(tmp_path))
    with FileLock(str(cache.lock_path)), FileLock(str(cache.hydration_lock_path)):
        rows = cache_ops._scan_embedding_namespaces(tmp_path)
    assert len(rows) == 1
    assert rows[0].status.endswith("ready")


def test_namespace_scan_reads_a_readonly_managed_cache_without_creating_locks(
    tmp_path, monkeypatch
):
    """A readable cache tree needs no write access or coordination files to scan."""
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path))
    directory = tmp_path / "embeddings"
    cache, _, _ = _populate(_settings(directory))
    shutil.rmtree(tmp_path / ".locks")
    cache.lock_path.unlink()
    cache.hydration_lock_path.unlink()
    payloads = list(directory.iterdir())
    for path in payloads:
        path.chmod(0o444)
    directory.chmod(0o555)
    tmp_path.chmod(0o555)
    try:
        rows = cache_ops._scan_embedding_namespaces(directory)
        assert len(rows) == 1
        assert rows[0].status.endswith("ready")
        assert not (tmp_path / ".locks").exists()
        assert set(directory.iterdir()) == set(payloads)
    finally:
        tmp_path.chmod(0o755)
        directory.chmod(0o755)
        for path in payloads:
            path.chmod(0o644)


def test_namespace_scan_reports_busy_sqlite_without_waiting(tmp_path):
    """An exclusive SQLite transaction must yield a busy row without recovery."""
    cache, _, _ = _populate(_settings(tmp_path))
    connection = sqlite3.connect(cache.db_path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        rows = cache_ops._scan_embedding_namespaces(tmp_path)
    finally:
        connection.close()
    assert len(rows) == 1
    assert rows[0].status == "Metadata busy (scan again after active operation)"


@pytest.mark.parametrize(
    "failure,status",
    [
        (
            sqlite3.OperationalError("unable to open database file"),
            "SQLite recovery pending",
        ),
        (sqlite3.OperationalError("disk I/O error"), "SQLite recovery pending"),
        (
            sqlite3.OperationalError("attempt to write a readonly database"),
            "SQLite recovery pending",
        ),
        (PermissionError("write access denied"), "SQLite recovery pending"),
        (sqlite3.OperationalError("database is locked"), "Metadata busy"),
    ],
)
@pytest.mark.parametrize("failure_stage", ["connect", "read"])
def test_namespace_scan_classifies_hot_journal_recovery_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: sqlite3.OperationalError | PermissionError,
    status: str,
    failure_stage: str,
) -> None:
    """Recovery failures retain their cause even without sqlite_errorcode."""
    assert not hasattr(failure, "sqlite_errorcode")
    cache, _, _ = _populate(_settings(tmp_path))
    read_summary = lifecycle._read_installed_corpus_summary
    recovering = False

    def read(connection: sqlite3.Connection) -> lifecycle.InstalledCorpusSummary:
        if not recovering:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        if failure_stage == "read":
            raise failure
        return read_summary(connection)

    @contextmanager
    def recover(
        namespace_id: str, cache_dir: Path | None = None, *, blocking: bool = True
    ) -> Iterator[sqlite3.Connection]:
        nonlocal recovering
        recovering = True
        assert namespace_id == cache.namespace_id
        assert cache_dir == tmp_path
        assert not blocking
        if failure_stage == "connect":
            raise failure
        with cache._connect_db() as connection:
            yield connection

    monkeypatch.setattr(lifecycle, "_read_installed_corpus_summary", read)
    monkeypatch.setattr(lifecycle, "_existing_namespace_connection", recover)
    rows = cache_ops._scan_embedding_namespaces(tmp_path)
    assert len(rows) == 1
    assert rows[0].status.startswith(status)
    recovering = False
    error_type = (
        lifecycle.CorpusMetadataBusyError
        if status == "Metadata busy"
        else lifecycle.CorpusRecoveryPendingError
    )
    with pytest.raises(error_type) as error:
        lifecycle.read_installed_corpus_summary(cache.db_path)
    assert error.value.__cause__ is failure


def test_namespace_scan_counts_share_one_committed_sqlite_snapshot(
    tmp_path, monkeypatch
):
    """Counts must describe one commit even when membership changes during a scan."""
    cache, _, _ = _populate(_settings(tmp_path))
    writer = sqlite3.connect(cache.db_path)
    writer.execute("PRAGMA journal_mode=WAL")
    read_state = lifecycle._read_state

    def replace_membership_after_state(connection):
        state = read_state(connection)
        writer.execute("DELETE FROM corpus_membership")
        writer.commit()
        return state

    monkeypatch.setattr(lifecycle, "_read_state", replace_membership_after_state)
    try:
        summary = lifecycle.read_installed_corpus_summary(cache.db_path)
        assert summary.active_rows == summary.physical_rows == 4
        assert (
            writer.execute("SELECT COUNT(*) FROM corpus_membership").fetchone()[0] == 0
        )
    finally:
        writer.close()


def test_metadata_alias_lookup_ignores_removed_primary_before_secondary_matching(
    tmp_path,
):
    """An inactive primary ID must not hide an active row sharing its DOI."""
    settings = _settings(tmp_path)
    records = _records()
    doi = "10.1234/shared"
    records[0]["paper_id"] = doi
    records[1]["doi"] = doi
    with import_destination_session(**settings) as session:
        session.append_stored_embeddings(
            records, np.eye(4, dtype=np.float32), first_row=0
        )
        session.finish(expected_rows=4, source_records_examined=4)
        cache = session.cache
    with cache.operation_session(
        kind="refresh",
        identity={"source": settings["source"].to_dict()},
        target_source=settings["source"],
        contract=settings["contract"],
        namespace_selectors=settings["namespace_selectors"],
    ) as session:
        session.record_membership(records[1:])
        session.finish(expected_rows=3, source_records_examined=3)
    assert set(cache.get_paper_metadata_alias_candidates([doi])) == {"paper-1"}
    assert cache.get_paper_metadata_alias_candidates([doi], corpus_size=1) == {}
    assert cache.embedding_count() == 4


def test_nonsearch_readers_enforce_installed_coverage(tmp_path):
    """Ready-state and reader checks share the installed newest-N coverage limit."""
    settings = _settings(tmp_path)
    settings["source"] = replace(settings["source"], selection="newest", limit=4)
    cache, records, _ = _populate(settings)
    ids = [record["paper_id"] for record in records]
    for size, message in [
        (0, "corpus_size must be positive"),
        (5, "Requested corpus view 5 exceeds installed coverage 4"),
    ]:
        with pytest.raises(ValueError, match=message):
            cache.require_complete_certified_corpus(corpus_size=size)
        with pytest.raises(ValueError, match=message):
            cache.get_cached_paper_ids(corpus_size=size)
        with pytest.raises(ValueError, match=message):
            cache.get_paper_metadata_batch(ids, corpus_size=size)
        with pytest.raises(ValueError, match=message):
            cache.get_paper_metadata_alias_candidates(ids, corpus_size=size)
        with pytest.raises(ValueError, match=message):
            cache.get_cached_embeddings(ids, corpus_size=size)
    assert cache.require_complete_certified_corpus(corpus_size=2)["membership_complete"]
    assert cache.get_cached_paper_ids(corpus_size=2) == {"paper-1", "paper-2"}


@pytest.mark.parametrize(
    "selection,limit", [("all", None), ("newest", 4), ("newest", 8)]
)
@pytest.mark.parametrize("refresh", [False, True])
def test_whole_coverage_views_avoid_sorting(
    tmp_path, monkeypatch, selection, limit, refresh
):
    """Native readers skip ranking whole coverage while retaining smaller views."""
    from citemesh.data.embedding_cache import layout

    settings = _settings(tmp_path)
    settings["source"] = replace(settings["source"], selection=selection, limit=limit)
    records = _records()
    for record in records:
        record["doi"] = f"10.1234/{record['paper_id']}"
    with import_destination_session(**settings) as session:
        session.append_stored_embeddings(
            records, np.eye(4, dtype=np.float32), first_row=0
        )
        session.finish(expected_rows=4, source_records_examined=4)
        cache = session.cache
    active = records[1:] if refresh else records
    if refresh:
        with cache.operation_session(
            kind="refresh",
            identity={"source": settings["source"].to_dict()},
            target_source=settings["source"],
            contract=settings["contract"],
            namespace_selectors=settings["namespace_selectors"],
        ) as session:
            session.record_membership(active)
            session.finish(expected_rows=len(active), source_records_examined=4)
    with cache._connect_db() as conn:
        assert conn.execute(
            "SELECT value FROM cache_metadata WHERE key = ?",
            (constants.HYDRATION_RECONCILED_CACHE_ROWS_KEY,),
        ).fetchone() == (str(len(active)),)

    statements = []
    connect = cache._connect_db

    @contextmanager
    def observe_queries():
        """Capture statements issued through native readers' shared connection."""
        with connect() as conn:
            conn.set_trace_callback(statements.append)
            yield conn

    monkeypatch.setattr(cache, "_connect_db", observe_queries)
    monkeypatch.setattr(layout, "SQLITE_QUERY_BATCH_SIZE", 1)
    monkeypatch.setattr(constants, "EMBEDDING_SEARCH_CHUNK_ROWS", 1)
    ids = [record["paper_id"] for record in records]
    aliases = [record["doi"] for record in records]
    for size in {1, len(active), limit or len(active) + 1}:
        statements.clear()
        expected = (
            {record["paper_id"] for record in active}
            if size >= len(active)
            else {"paper-2"}
        )
        assert cache.get_cached_paper_ids(corpus_size=size) == expected
        assert set(cache.get_paper_metadata_batch(ids, corpus_size=size)) == expected
        assert (
            set(cache.get_paper_metadata_alias_candidates(aliases, corpus_size=size))
            == expected
        )
        assert set(cache.get_cached_embeddings(ids, corpus_size=size)) == expected
        assert {
            result.paper_id
            for result in cache.search(
                np.ones(4, dtype=np.float32),
                10,
                corpus_size=size,
                binary_prefilter=False,
                binary_rescore_multiplier=1,
            )
        } == expected
        membership_reads = [sql for sql in statements if "corpus_membership" in sql]
        assert membership_reads
        assert any("ORDER BY" in sql for sql in membership_reads) == (
            size < len(active)
        )
        point_reads = [sql for sql in membership_reads if "papers AS owners" in sql]
        assert point_reads
        assert all(
            (" AND EXISTS " in sql) == (size >= len(active)) for sql in point_reads
        )
        assert not any("COUNT(*) FROM corpus_membership" in sql for sql in statements)


@pytest.mark.parametrize("marker", [None, "", "-1", "unknown"])
def test_missing_reconciliation_count_preserves_view_semantics(tmp_path, marker):
    """Unavailable row counts retain ranking except at the certified source limit."""
    settings = _settings(tmp_path)
    settings["source"] = replace(settings["source"], selection="newest", limit=4)
    cache, _, _ = _populate(settings)
    with cache._connect_db() as conn:
        if marker is None:
            conn.execute(
                "DELETE FROM cache_metadata WHERE key = ?",
                (constants.HYDRATION_RECONCILED_CACHE_ROWS_KEY,),
            )
        else:
            conn.execute(
                "UPDATE cache_metadata SET value = ? WHERE key = ?",
                (marker, constants.HYDRATION_RECONCILED_CACHE_ROWS_KEY),
            )
        query, parameters = cache._corpus_view_query_locked(conn, 4)
        assert parameters == ()
        assert "ORDER BY" not in query
    assert cache.get_cached_paper_ids(corpus_size=1) == {"paper-2"}
    assert cache.get_cached_paper_ids(corpus_size=4) == {
        record["paper_id"] for record in _records()
    }
