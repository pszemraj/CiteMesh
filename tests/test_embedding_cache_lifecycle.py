"""Source certification and native import durability independent of Arrow or models."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from dataclasses import replace

import h5py
import numpy as np
import pytest

from citemesh.data.embedding_cache import EmbeddingCache, constants
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
    results = cache.search(
        np.ones(4, dtype=np.float32),
        10,
        corpus_size=1,
        eligible_rows=np.ones(4, dtype=bool),
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


def test_recognized_empty_import_initialization_resumes_missing_hdf5(tmp_path):
    settings = _settings(tmp_path)
    with import_destination_session(**settings) as session:
        cache = session.cache
    cache.h5_path.unlink()
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


def test_import_rebuilds_binary_only_once_after_all_batches(tmp_path, monkeypatch):
    settings = _settings(tmp_path, "int8")
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

    class PausedModel:
        def encode(self, texts, **kwargs):
            entered.set()
            assert release.wait(5)
            return np.ones((len(texts), 4), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            cache.upsert_embeddings, {"new-paper": {"title": "New"}}, PausedModel()
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
