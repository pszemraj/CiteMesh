"""Contract tests for embedding cache layout, reset, and retrieval behavior."""

from __future__ import annotations

import builtins
import logging
import multiprocessing as mp
import os
import sqlite3
import sys
import tempfile
import threading
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from queue import Empty
from typing import Any, Iterator

import h5py
import numpy as np
import pytest

import citemesh.data.embedding_cache as embedding_cache_module
from citemesh.data.embedding_cache import (
    BINARY_INDEX_DATASET_NAME,
    BINARY_INDEX_ENCODING,
    BINARY_INDEX_ENCODING_KEY,
    CALIBRATION_SAMPLE_SIZE_KEY,
    COMPRESSION_FILTER_KEY,
    COMPRESSION_LEVEL_KEY,
    EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR,
    EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS,
    EMBEDDING_DATASET_CHUNK_ROWS,
    EMBEDDINGS_DATASET_NAME,
    H5_LAYOUT_KEY,
    H5_LAYOUT_MATRIX_VERSION,
    HYDRATION_COMPLETE_KEY,
    HYDRATION_CORPUS_SIZE_KEY,
    HYDRATION_DATASET_SOURCE_KEY,
    HYDRATION_SPLIT_KEY,
    INT8_CLIPPED_VALUE_COUNT_KEY,
    INT8_TOTAL_VALUE_COUNT_KEY,
    MODEL_FINGERPRINT_KEY,
    SCHEMA_VERSION_KEY,
    SOURCE_TORCH_DTYPE_KEY,
    TEXT_FORMATTER_FINGERPRINT_KEY,
    EmbeddingCache,
    _corpus_size_token,
    _resolve_cache_lock_timeout_seconds,
)
from citemesh.data.model_profiles import get_embedding_model_profile
from tests._helpers import LookupEncodeModel, SeededRandomEncodeModel


class _FakeInferenceTensor:
    """Small NumPy-backed tensor surface for prefetch proxy unit tests."""

    def __init__(self, values: object) -> None:
        """Store array-like values.

        :param object values: Values represented by the fake tensor.
        """
        self.values = np.asarray(values)

    def __getitem__(self, key: object) -> "_FakeInferenceTensor":
        """Return a sliced fake tensor.

        :param object key: NumPy-compatible index or slice.
        :return _FakeInferenceTensor: Sliced values.
        """
        return _FakeInferenceTensor(self.values[key])

    def float(self) -> "_FakeInferenceTensor":
        """Return FP32 values.

        :return _FakeInferenceTensor: FP32 tensor view.
        """
        return _FakeInferenceTensor(self.values.astype(np.float32))

    def cpu(self) -> "_FakeInferenceTensor":
        """Return the already-hosted tensor.

        :return _FakeInferenceTensor: This tensor.
        """
        return self

    def numpy(self) -> np.ndarray:
        """Return the backing NumPy array.

        :return np.ndarray: Array values.
        """
        return self.values

    def tolist(self) -> list:
        """Return values as Python lists.

        :return list: Nested or flat list values.
        """
        return self.values.tolist()


class _PrefetchModelStub:
    """SentenceTransformer-like text model for CPU-only proxy tests."""

    device = "cuda"
    truncate_dim = None

    def __init__(
        self,
        vectors: dict[str, list[float]],
        token_lengths: dict[str, int],
        *,
        max_seq_length: int = 100,
        prompt: str = "prompt: ",
        fail_forward_call: int | None = None,
        overlap_event: threading.Event | None = None,
    ) -> None:
        """Configure deterministic preprocessing and forward behavior.

        :param dict[str, list[float]] vectors: Embeddings keyed by raw input text.
        :param dict[str, int] token_lengths: Untruncated lengths keyed by raw text.
        :param int max_seq_length: Encoder token window.
        :param str prompt: Default prompt resolved by the model.
        :param Optional[int] fail_forward_call: Optional forward call number to fail.
        :param Optional[threading.Event] overlap_event: Event set by second preprocessing call.
        """
        self.vectors = vectors
        self.token_lengths = token_lengths
        self.max_seq_length = max_seq_length
        self.prompt = prompt
        self.fail_forward_call = fail_forward_call
        self.overlap_event = overlap_event
        self.preprocess_calls: list[tuple[list[str], int]] = []
        self.forward_threads: list[int] = []
        self.tokenizer_calls: list[list[str]] = []
        self.forward_calls = 0
        self.eval_calls = 0

    def eval(self) -> None:
        """Record model evaluation setup.

        :return None: Records the call.
        """
        self.eval_calls += 1

    def _resolve_prompt(self, prompt: None, prompt_name: None) -> str:
        """Return the configured default prompt.

        :param None prompt: Unused explicit prompt.
        :param None prompt_name: Unused prompt name.
        :return str: Configured prompt.
        """
        del prompt, prompt_name
        return self.prompt

    def preprocess(self, texts: list[str], *, prompt: str) -> dict[str, object]:
        """Build flattened-input boundaries for one batch.

        :param list[str] texts: Batch texts.
        :param str prompt: Resolved model prompt.
        :return dict[str, object]: Fake model features.
        """
        assert prompt == self.prompt
        self.preprocess_calls.append((list(texts), threading.get_ident()))
        if len(self.preprocess_calls) > 1 and self.overlap_event is not None:
            self.overlap_event.set()
        lengths = [min(self.token_lengths[text], self.max_seq_length) for text in texts]
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        return {
            "cu_seq_lens_q": _FakeInferenceTensor(boundaries),
            "texts": list(texts),
        }

    def tokenizer(self, inputs: list[str], **kwargs: object) -> dict[str, list[int]]:
        """Return exact untruncated lengths for candidate texts.

        :param list[str] inputs: Prompt-prefixed candidate texts.
        :param object kwargs: Tokenization controls.
        :return dict[str, list[int]]: Candidate token lengths.
        """
        assert kwargs == {
            "truncation": False,
            "padding": False,
            "return_length": True,
            "verbose": False,
        }
        self.tokenizer_calls.append(list(inputs))
        return {
            "length": [
                self.token_lengths[text.removeprefix(self.prompt)] for text in inputs
            ]
        }

    def __call__(self, features: dict[str, object]) -> dict[str, object]:
        """Return batch embeddings or inject a configured compile failure.

        :param dict[str, object] features: Prepared batch features.
        :return dict[str, object]: Fake sentence embeddings.
        """
        self.forward_calls += 1
        self.forward_threads.append(threading.get_ident())
        if self.forward_calls == 1 and self.overlap_event is not None:
            assert self.overlap_event.wait(timeout=2)
        if self.forward_calls == self.fail_forward_call:
            raise RuntimeError("compiled forward failed")
        texts = features["texts"]
        return {
            "sentence_embedding": _FakeInferenceTensor(
                [self.vectors[text] for text in texts]
            )
        }


def _make_prefetch_proxy(
    monkeypatch: pytest.MonkeyPatch,
    model: _PrefetchModelStub,
    restore_eager: object,
) -> object:
    """Create the precision proxy with fake optional-dependency surfaces.

    :param pytest.MonkeyPatch monkeypatch: Module patch fixture.
    :param _PrefetchModelStub model: Fake base model.
    :param object restore_eager: Compile recovery callback.
    :return object: Configured precision proxy.
    """
    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.__path__ = []
    fake_util = types.ModuleType("sentence_transformers.util")
    fake_util.batch_to_device = lambda features, _device: features
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", fake_sentence_transformers
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers.util", fake_util)

    from citemesh.strategies import embedding as embedding_module

    fake_torch = types.SimpleNamespace(inference_mode=nullcontext)
    monkeypatch.setattr(embedding_module, "_import_torch", lambda: fake_torch)
    return embedding_module._PrecisionEncodeProxy(
        model,
        nullcontext,
        restore_eager,
        prefetch_batches=True,
    )


def _set_test_int8_calibration(cache: EmbeddingCache, embedding_dim: int = 2) -> None:
    """Seed deterministic calibration ranges for direct int8 cache tests.

    :param EmbeddingCache cache: Cache instance under test.
    :param int embedding_dim: Embedding width covered by the ranges.
    :return None: Persists calibration ranges when cache uses int8 storage.
    """
    if cache.storage_precision != "int8":
        return

    ranges = np.vstack(
        (
            np.full(embedding_dim, -1.0, dtype=np.float32),
            np.full(embedding_dim, 1.0, dtype=np.float32),
        )
    )
    cache.set_calibration_ranges(ranges=ranges, embedding_dim=embedding_dim)


def _multiprocess_cache_worker(
    cache_dir: str, worker_idx: int, queue: mp.Queue
) -> None:
    """Write embeddings in subprocess and report success/failure via queue."""
    try:
        cache = EmbeddingCache(cache_dir=cache_dir, model_name="process-lock-test")
        _set_test_int8_calibration(cache)
        model = SeededRandomEncodeModel()
        papers = {
            f"p{worker_idx}_{offset}": {
                "title": f"Title {offset}",
                "abstract": f"Abstract {offset}",
            }
            for offset in range(100)
        }
        cache.get_embeddings(papers, model, show_progress=False)
        queue.put(("ok", worker_idx))
    except Exception as exc:  # pragma: no cover - subprocess path
        queue.put(("err", worker_idx, repr(exc)))


def _multiprocess_cache_init_worker(cache_dir: str, queue: mp.Queue) -> None:
    """Initialize cache namespace in subprocess and report success/failure."""
    try:
        EmbeddingCache(cache_dir=cache_dir, model_name="process-init-recovery")
        queue.put(("ok",))
    except Exception as exc:  # pragma: no cover - subprocess path
        queue.put(("err", repr(exc)))


def _replacement_interrupt_worker(
    cache_dir: str, model_name: str, storage_precision: str
) -> None:
    """Persist a replacement vector and terminate before its SQLite commit.

    :param str cache_dir: Isolated cache directory shared with the parent test.
    :param str model_name: Existing namespace model name.
    :param str storage_precision: Persisted vector format under test.
    :return None: Does not return because it terminates the child process.
    """
    cache = EmbeddingCache(
        cache_dir=cache_dir,
        model_name=model_name,
        storage_precision=storage_precision,
    )
    model = LookupEncodeModel(
        {
            "Original. Abstract": np.asarray([1.0, 0.0], dtype=np.float32),
            "Replacement. Abstract": np.asarray([-1.0, 0.0], dtype=np.float32),
        }
    )

    def exit_after_flush(h5_file: h5py.File) -> None:
        """Terminate after the replacement rows have reached HDF5 storage.

        :param h5py.File h5_file: HDF5 handle containing the replaced rows.
        :return None: Does not return because it exits the child process.
        """
        h5_file.flush()
        os._exit(91)

    cache._flush_h5_file = exit_after_flush  # type: ignore[method-assign]
    cache.get_embeddings(
        {"p1": {"title": "Replacement", "abstract": "Abstract"}},
        model,
        show_progress=False,
    )


def test_embedding_cache_lifecycle_contract() -> None:
    """Cache lifecycle should handle hit/miss, rewrites, and quantized layout."""
    model = SeededRandomEncodeModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="test-model")
        _set_test_int8_calibration(cache)
        papers_v1 = {
            "p1": {"title": "Paper One", "abstract": "Abstract one"},
            "p2": {"title": "Paper Two", "abstract": "Abstract two"},
        }
        papers_v2 = {
            "p1": {"title": "Updated", "abstract": "Abstract one"},
            "p2": {"title": "Paper Two", "abstract": "Abstract two"},
        }
        papers_v3 = {
            "p1": {"title": "Updated", "abstract": "Abstract one"},
            "p2": {"title": "Paper Two", "abstract": "Abstract two", "year": 2024},
        }
        first = cache.get_embeddings(papers_v1, model, show_progress=False)
        second = cache.get_embeddings(papers_v1, model, show_progress=False)
        cache.get_embeddings(papers_v2, model, show_progress=False)
        cache.get_embeddings(papers_v3, model, show_progress=False)

        with cache._connect_db() as conn:
            rows = conn.execute(
                "SELECT paper_id, row_idx FROM papers ORDER BY row_idx"
            ).fetchall()
            row_idx_after_update = conn.execute(
                "SELECT row_idx FROM papers WHERE paper_id = 'p1'"
            ).fetchone()[0]
            p2_metadata_after_refresh = conn.execute(
                "SELECT year FROM papers WHERE paper_id = 'p2'"
            ).fetchone()[0]

        with h5py.File(cache.h5_path, "r") as h5:
            assert set(h5.keys()) == {
                "embeddings",
                "binary_index",
                "calibration_ranges",
            }
            assert h5["embeddings"].shape == (2, 2)
            assert h5["embeddings"].dtype == np.int8
            assert h5["binary_index"].shape == (2, 1)
            assert h5["binary_index"].dtype == np.uint8
            assert h5["calibration_ranges"].shape == (2, 2)
            assert h5["calibration_ranges"].dtype == np.float32

    assert model.encode_calls == 2
    assert first["p1"].shape == second["p1"].shape
    assert rows == [("p1", 0), ("p2", 1)]
    assert row_idx_after_update == 0
    assert p2_metadata_after_refresh == 2024


@pytest.mark.parametrize("storage_precision", ["float32", "int8"])
def test_embedding_cache_replacement_journal_recovers_failed_sqlite_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage_precision: str,
) -> None:
    """A failed replacement commit must restore its old metadata/vector pair.

    :param Path tmp_path: Isolated cache directory.
    :param pytest.MonkeyPatch monkeypatch: Replaces only the final SQLite commit.
    :param str storage_precision: Persisted vector format under test.
    :return None: Checks immediate same-instance undo recovery.
    """
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        model_name=f"replacement-sqlite-{storage_precision}",
        storage_precision=storage_precision,
    )
    _set_test_int8_calibration(cache)
    original = {"p1": {"title": "Original", "abstract": "Abstract"}}
    replacement = {
        "p1": {"title": "Replacement", "abstract": "Abstract"},
        "p2": {"title": "Other", "abstract": "Abstract"},
    }
    model = LookupEncodeModel(
        {
            "Original. Abstract": np.asarray([1.0, 0.0], dtype=np.float32),
            "Replacement. Abstract": np.asarray([-1.0, 0.0], dtype=np.float32),
            "Other. Abstract": np.asarray([0.0, 1.0], dtype=np.float32),
        }
    )
    cache.get_embeddings(original, model, show_progress=False)
    with h5py.File(cache.h5_path, "r") as h5:
        original_embedding = h5[EMBEDDINGS_DATASET_NAME][:].copy()
        original_binary = (
            h5[BINARY_INDEX_DATASET_NAME][:].copy()
            if BINARY_INDEX_DATASET_NAME in h5
            else None
        )

    connection_count = 0

    @contextmanager
    def fail_final_commit() -> Iterator[sqlite3.Connection]:
        """Fail the metadata transaction after its undo journal has committed.

        :return Iterator[sqlite3.Connection]: SQLite connection used by one cache phase.
        """
        nonlocal connection_count
        conn = sqlite3.connect(cache.db_path)
        connection_count += 1
        try:
            yield conn
            if connection_count == 2:
                raise sqlite3.OperationalError("forced final commit failure")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    with monkeypatch.context() as patch:
        patch.setattr(cache, "_connect_db", fail_final_commit)
        with pytest.raises(sqlite3.OperationalError, match="forced final commit"):
            cache.get_embeddings(replacement, model, show_progress=False)

    with cache._connect_db() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM replacement_journal").fetchone()[0] == 1
        )
        assert (
            conn.execute("SELECT title FROM papers WHERE paper_id = 'p1'").fetchone()[0]
            == "Original"
        )
    with h5py.File(cache.h5_path, "r") as h5:
        assert not np.array_equal(h5[EMBEDDINGS_DATASET_NAME][:], original_embedding)

    cache.get_embeddings(original, model, show_progress=False)
    with cache._connect_db() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM replacement_journal").fetchone()[0] == 0
        )
        assert (
            conn.execute("SELECT title FROM papers WHERE paper_id = 'p1'").fetchone()[0]
            == "Original"
        )
        assert conn.execute(
            "SELECT paper_id FROM papers ORDER BY paper_id"
        ).fetchall() == [("p1",)]
    with h5py.File(cache.h5_path, "r") as h5:
        np.testing.assert_array_equal(
            h5[EMBEDDINGS_DATASET_NAME][:], original_embedding
        )
        if original_binary is not None:
            np.testing.assert_array_equal(
                h5[BINARY_INDEX_DATASET_NAME][:], original_binary
            )

    reopened = EmbeddingCache(
        cache_dir=tmp_path,
        model_name=f"replacement-sqlite-{storage_precision}",
        storage_precision=storage_precision,
    )
    assert reopened.get_cached_paper_ids() == {"p1"}
    with h5py.File(reopened.h5_path, "r") as h5:
        np.testing.assert_array_equal(
            h5[EMBEDDINGS_DATASET_NAME][:], original_embedding
        )


@pytest.mark.parametrize("storage_precision", ["float32", "int8"])
def test_embedding_cache_replacement_journal_recovers_vector_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage_precision: str,
) -> None:
    """A failed replacement vector write must leave a replayable undo record.

    :param Path tmp_path: Isolated cache directory.
    :param pytest.MonkeyPatch monkeypatch: Injects one HDF5 vector-write failure.
    :param str storage_precision: Persisted vector format under test.
    :return None: Checks journal cleanup after recovery.
    """
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        model_name=f"replacement-vector-{storage_precision}",
        storage_precision=storage_precision,
    )
    _set_test_int8_calibration(cache)
    original = {"p1": {"title": "Original", "abstract": "Abstract"}}
    replacement = {"p1": {"title": "Replacement", "abstract": "Abstract"}}
    model = LookupEncodeModel(
        {
            "Original. Abstract": np.asarray([1.0, 0.0], dtype=np.float32),
            "Replacement. Abstract": np.asarray([-1.0, 0.0], dtype=np.float32),
        }
    )
    cache.get_embeddings(original, model, show_progress=False)
    with h5py.File(cache.h5_path, "r") as h5:
        original_embedding = h5[EMBEDDINGS_DATASET_NAME][:].copy()

    original_setitem = h5py.Dataset.__setitem__
    failed = False

    def fail_replacement_vector_write(
        dataset: h5py.Dataset, key: object, value: object
    ) -> None:
        """Raise once when the replacement writes the primary vector matrix.

        :param h5py.Dataset dataset: HDF5 dataset accepting an assignment.
        :param object key: Dataset row/slice receiving the assignment.
        :param object value: Replacement value.
        :return None: Delegates after the injected failure.
        """
        nonlocal failed
        if dataset.name == f"/{EMBEDDINGS_DATASET_NAME}" and not failed:
            failed = True
            raise OSError("forced vector write failure")
        original_setitem(dataset, key, value)

    with monkeypatch.context() as patch:
        patch.setattr(h5py.Dataset, "__setitem__", fail_replacement_vector_write)
        with pytest.raises(OSError, match="forced vector write failure"):
            cache.get_embeddings(replacement, model, show_progress=False)

    assert failed
    with cache._connect_db() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM replacement_journal").fetchone()[0] == 1
        )
    assert cache.embedding_count() == 1
    with cache._connect_db() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM replacement_journal").fetchone()[0] == 0
        )
    with h5py.File(cache.h5_path, "r") as h5:
        np.testing.assert_array_equal(
            h5[EMBEDDINGS_DATASET_NAME][:], original_embedding
        )


def test_embedding_cache_replacement_journal_recovers_binary_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed binary replacement write must restore vector and binary rows.

    :param Path tmp_path: Isolated cache directory.
    :param pytest.MonkeyPatch monkeypatch: Injects one packed-binary write failure.
    :return None: Checks binary prefilter data is restored with the primary vector.
    """
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="replacement-binary")
    _set_test_int8_calibration(cache)
    original = {"p1": {"title": "Original", "abstract": "Abstract"}}
    replacement = {"p1": {"title": "Replacement", "abstract": "Abstract"}}
    model = LookupEncodeModel(
        {
            "Original. Abstract": np.asarray([1.0, 0.0], dtype=np.float32),
            "Replacement. Abstract": np.asarray([-1.0, 0.0], dtype=np.float32),
        }
    )
    cache.get_embeddings(original, model, show_progress=False)
    with h5py.File(cache.h5_path, "r") as h5:
        original_embedding = h5[EMBEDDINGS_DATASET_NAME][:].copy()
        original_binary = h5[BINARY_INDEX_DATASET_NAME][:].copy()

    original_setitem = h5py.Dataset.__setitem__
    failed = False

    def fail_replacement_binary_write(
        dataset: h5py.Dataset, key: object, value: object
    ) -> None:
        """Raise once when the replacement writes the packed binary index.

        :param h5py.Dataset dataset: HDF5 dataset accepting an assignment.
        :param object key: Dataset row/slice receiving the assignment.
        :param object value: Replacement value.
        :return None: Delegates after the injected failure.
        """
        nonlocal failed
        if dataset.name == f"/{BINARY_INDEX_DATASET_NAME}" and not failed:
            failed = True
            raise OSError("forced binary write failure")
        original_setitem(dataset, key, value)

    with monkeypatch.context() as patch:
        patch.setattr(h5py.Dataset, "__setitem__", fail_replacement_binary_write)
        with pytest.raises(OSError, match="forced binary write failure"):
            cache.get_embeddings(replacement, model, show_progress=False)

    assert failed
    with h5py.File(cache.h5_path, "a") as h5:
        del h5[BINARY_INDEX_DATASET_NAME]
    with pytest.raises(RuntimeError, match="binary index that is unavailable"):
        cache.search(
            np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=1,
            binary_prefilter=True,
            binary_rescore_multiplier=1,
        )
    with cache._connect_db() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM replacement_journal").fetchone()[0] == 1
        )
    with h5py.File(cache.h5_path, "a") as h5:
        binary = h5.create_dataset(
            BINARY_INDEX_DATASET_NAME,
            data=np.zeros_like(original_binary),
            dtype=np.uint8,
        )
        binary.attrs[BINARY_INDEX_ENCODING_KEY] = BINARY_INDEX_ENCODING
    assert (
        cache.search(
            np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=1,
            binary_prefilter=True,
            binary_rescore_multiplier=1,
        )[0].paper_id
        == "p1"
    )
    with cache._connect_db() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM replacement_journal").fetchone()[0] == 0
        )
    with h5py.File(cache.h5_path, "r") as h5:
        np.testing.assert_array_equal(
            h5[EMBEDDINGS_DATASET_NAME][:], original_embedding
        )
        np.testing.assert_array_equal(h5[BINARY_INDEX_DATASET_NAME][:], original_binary)


def test_embedding_cache_replacement_journal_retains_rows_when_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed HDF5 fsync must retain the undo journal for a later replay.

    :param Path tmp_path: Isolated cache directory.
    :param pytest.MonkeyPatch monkeypatch: Fails the explicit OS durability sync.
    :return None: Checks recovery remains possible after the sync failure.
    """
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        model_name="replacement-fsync",
        storage_precision="float32",
    )
    original = {"p1": {"title": "Original", "abstract": "Abstract"}}
    replacement = {"p1": {"title": "Replacement", "abstract": "Abstract"}}
    model = LookupEncodeModel(
        {
            "Original. Abstract": np.asarray([1.0, 0.0], dtype=np.float32),
            "Replacement. Abstract": np.asarray([-1.0, 0.0], dtype=np.float32),
        }
    )
    cache.get_embeddings(original, model, show_progress=False)

    sync_calls = 0

    def fail_fsync(file_descriptor: int) -> None:
        """Raise instead of durably syncing the HDF5 file descriptor.

        :param int file_descriptor: Descriptor selected for synchronization.
        :return None: Always raises the injected storage error.
        """
        nonlocal sync_calls
        del file_descriptor
        sync_calls += 1
        raise OSError("forced fsync failure")

    with monkeypatch.context() as patch:
        patch.setattr(embedding_cache_module.os, "fsync", fail_fsync)
        with pytest.raises(RuntimeError, match="failed to durably flush"):
            cache.get_embeddings(replacement, model, show_progress=False)

    assert sync_calls == 1
    with cache._connect_db() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM replacement_journal").fetchone()[0] == 1
        )

    cache.get_embeddings(original, model, show_progress=False)
    with cache._connect_db() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM replacement_journal").fetchone()[0] == 0
        )


def test_embedding_cache_append_syncs_vectors_before_mapping_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Appended vectors must be durable before SQLite maps rows onto them.

    :param Path tmp_path: Isolated cache directory.
    :param pytest.MonkeyPatch monkeypatch: Installs the ordering recorders.
    :return None: Checks the append path shares the replacement path's durability.
    """
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        model_name="append-durability",
        storage_precision="float32",
    )
    events: list[str] = []
    open_connection = cache._connect_db

    @contextmanager
    def recording_connect_db() -> Iterator[sqlite3.Connection]:
        """Record the row-mapping upsert issued on a cache connection.

        :return Iterator[sqlite3.Connection]: Context manager yielding a traced connection.
        """

        def record_statement(statement: str) -> None:
            """Append the upsert marker when row mappings are written.

            :param str statement: SQL statement submitted on this connection.
            :return None: Mutates the shared ordering log.
            """
            if "INSERT OR REPLACE INTO papers" in statement:
                events.append("sqlite-upsert")

        with open_connection() as conn:
            conn.set_trace_callback(record_statement)
            yield conn

    def recording_flush(h5_file: h5py.File) -> None:
        """Record the durability sync before performing it.

        :param h5py.File h5_file: Writable HDF5 handle for the namespace.
        :return None: Mutates the shared ordering log and syncs the file.
        """
        events.append("h5-sync")
        EmbeddingCache._flush_h5_file(h5_file)

    monkeypatch.setattr(cache, "_connect_db", recording_connect_db)
    monkeypatch.setattr(cache, "_flush_h5_file", recording_flush)
    cache.get_embeddings(
        {"p1": {"title": "Seed", "abstract": "Abstract"}},
        LookupEncodeModel({"Seed. Abstract": np.asarray([1.0, 0.0], dtype=np.float32)}),
        show_progress=False,
    )

    assert events == ["h5-sync", "sqlite-upsert"]
    assert cache.embedding_count() == 1


def test_embedding_cache_rebuilds_binary_index_after_disabled_replacement(
    tmp_path: Path,
) -> None:
    """Re-enabling binary prefilter must rebuild rows changed while disabled.

    :param Path tmp_path: Isolated cache directory.
    :return None: Checks binary signs track a replacement after toggle/reopen.
    """
    model_name = "binary-toggle-replacement"
    cache = EmbeddingCache(cache_dir=tmp_path, model_name=model_name)
    _set_test_int8_calibration(cache, embedding_dim=8)
    original = {
        "p1": {"title": "First", "abstract": "Abstract"},
        "p2": {"title": "Second", "abstract": "Abstract"},
    }
    replacement = {"p1": {"title": "Updated", "abstract": "Abstract"}}
    model = LookupEncodeModel(
        {
            "First. Abstract": np.ones(8, dtype=np.float32),
            "Second. Abstract": np.asarray(
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
                dtype=np.float32,
            ),
            "Updated. Abstract": -np.ones(8, dtype=np.float32),
        }
    )
    cache.get_embeddings(original, model, show_progress=False)

    disabled = EmbeddingCache(
        cache_dir=tmp_path,
        model_name=model_name,
        binary_prefilter=False,
    )
    disabled.get_embeddings(replacement, model, show_progress=False)
    with h5py.File(disabled.h5_path, "r") as h5:
        assert BINARY_INDEX_DATASET_NAME not in h5

    # The original object still has prefiltering enabled, but the shared index
    # was removed by the other writer; its search must score primary rows.
    live_results = cache.search(
        -np.ones(8, dtype=np.float32),
        top_k=1,
        binary_prefilter=True,
        binary_rescore_multiplier=1,
    )
    assert [result.paper_id for result in live_results] == ["p1"]
    assert cache.last_search_used_binary_prefilter is False

    reenabled = EmbeddingCache(cache_dir=tmp_path, model_name=model_name)
    results = reenabled.search(
        -np.ones(8, dtype=np.float32),
        top_k=1,
        binary_prefilter=True,
        binary_rescore_multiplier=1,
    )
    assert [result.paper_id for result in results] == ["p1"]
    assert reenabled.last_search_used_binary_prefilter is True

    disabled.get_embeddings(replacement, model, show_progress=False)
    direct_results = disabled.search(
        -np.ones(8, dtype=np.float32),
        top_k=1,
        binary_prefilter=True,
        binary_rescore_multiplier=1,
    )
    assert [result.paper_id for result in direct_results] == ["p1"]
    assert disabled.last_search_used_binary_prefilter is False


@pytest.mark.parametrize("storage_precision", ["float32", "int8"])
def test_embedding_cache_replacement_journal_recovers_interruption_on_reopen(
    tmp_path: Path,
    storage_precision: str,
) -> None:
    """A process exit after HDF5 flush must replay rows when reopening.

    :param Path tmp_path: Isolated cache directory.
    :param str storage_precision: Persisted vector format under test.
    :return None: Checks constructor-time replay before data is readable.
    """
    model_name = f"replacement-interrupt-{storage_precision}"
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        model_name=model_name,
        storage_precision=storage_precision,
    )
    _set_test_int8_calibration(cache)
    original = {"p1": {"title": "Original", "abstract": "Abstract"}}
    model = LookupEncodeModel(
        {
            "Original. Abstract": np.asarray([1.0, 0.0], dtype=np.float32),
            "Replacement. Abstract": np.asarray([-1.0, 0.0], dtype=np.float32),
        }
    )
    cache.get_embeddings(original, model, show_progress=False)
    with h5py.File(cache.h5_path, "r") as h5:
        original_embedding = h5[EMBEDDINGS_DATASET_NAME][:].copy()
        original_binary = (
            h5[BINARY_INDEX_DATASET_NAME][:].copy()
            if BINARY_INDEX_DATASET_NAME in h5
            else None
        )

    process = mp.Process(
        target=_replacement_interrupt_worker,
        args=(str(tmp_path), model_name, storage_precision),
    )
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join()
    assert process.exitcode == 91

    with cache._connect_db() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM replacement_journal").fetchone()[0] == 1
        )

    reopened = EmbeddingCache(
        cache_dir=tmp_path,
        model_name=model_name,
        storage_precision=storage_precision,
    )
    with reopened._connect_db() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM replacement_journal").fetchone()[0] == 0
        )
        assert (
            conn.execute("SELECT title FROM papers WHERE paper_id = 'p1'").fetchone()[0]
            == "Original"
        )
    with h5py.File(reopened.h5_path, "r") as h5:
        np.testing.assert_array_equal(
            h5[EMBEDDINGS_DATASET_NAME][:], original_embedding
        )
        if original_binary is not None:
            np.testing.assert_array_equal(
                h5[BINARY_INDEX_DATASET_NAME][:], original_binary
            )


def test_embedding_cache_reencodes_legacy_schema_before_reusing_vectors(
    tmp_path: Path,
) -> None:
    """A schema-v2 namespace must not be trusted after unsafe replacements existed.

    :param Path tmp_path: Isolated cache directory.
    :return None: Checks the schema boundary clears old vectors before re-encoding.
    """
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        model_name="replacement-schema-invalidation",
        storage_precision="float32",
    )
    original = {"p1": {"title": "Original", "abstract": "Abstract"}}
    original_model = LookupEncodeModel(
        {"Original. Abstract": np.asarray([1.0, 0.0], dtype=np.float32)}
    )
    cache.get_embeddings(original, original_model, show_progress=False)
    with h5py.File(cache.h5_path, "a") as h5:
        h5[EMBEDDINGS_DATASET_NAME][0] = np.asarray([-1.0, 0.0], dtype=np.float32)
        h5.attrs.modify(SCHEMA_VERSION_KEY, 2)

    reopened = EmbeddingCache(
        cache_dir=tmp_path,
        model_name="replacement-schema-invalidation",
        storage_precision="float32",
    )
    assert reopened.get_cached_paper_ids() == set()
    assert reopened.embedding_count() == 0

    reencoded = reopened.get_embeddings(
        original,
        LookupEncodeModel(
            {"Original. Abstract": np.asarray([0.0, 1.0], dtype=np.float32)}
        ),
        show_progress=False,
    )
    np.testing.assert_array_equal(
        reencoded["p1"], np.asarray([0.0, 1.0], dtype=np.float32)
    )


def test_embedding_cache_rejects_float16_persistent_storage() -> None:
    """Persistent cache vectors must use int8 or float32 storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="storage_precision must be one of"):
            EmbeddingCache(
                cache_dir=tmpdir,
                model_name="float16-storage-rejected",
                storage_precision="float16",
            )


def test_embedding_cache_removes_legacy_unused_text_hash_index(tmp_path: Path) -> None:
    """Reopening a namespace should discard its obsolete text-hash index."""
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="legacy-text-hash-index")
    with cache._connect_db() as conn:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_text_hash ON papers(text_hash)"
        )

    reopened = EmbeddingCache(cache_dir=tmp_path, model_name="legacy-text-hash-index")
    with reopened._connect_db() as conn:
        indexes = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }

    assert "idx_papers_text_hash" not in indexes
    assert "idx_papers_row_idx" in indexes


def test_embedding_cache_lock_timeout_env_override_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock-timeout env override should parse valid values and reject invalid ones."""
    monkeypatch.delenv(EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR, raising=False)
    assert _resolve_cache_lock_timeout_seconds() == EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS

    monkeypatch.setenv(EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR, "120.5")
    assert _resolve_cache_lock_timeout_seconds() == pytest.approx(120.5)

    monkeypatch.setenv(EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR, "0")
    assert _resolve_cache_lock_timeout_seconds() == EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS

    monkeypatch.setenv(EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR, "not-a-number")
    assert _resolve_cache_lock_timeout_seconds() == EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS


def test_embedding_cache_int8_requires_explicit_calibration_ranges() -> None:
    """Fresh int8 namespaces should fail closed until calibration is persisted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="int8-needs-calibration")

        with pytest.raises(
            RuntimeError, match="Missing persisted int8 calibration ranges"
        ):
            cache.get_embeddings(
                {"p1": {"title": "Alpha", "abstract": "First"}},
                LookupEncodeModel(
                    {"Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32)}
                ),
                show_progress=False,
            )


def test_embedding_cache_rechecks_misses_after_encode_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encode races must return the persisted winner on this and later cache hits.

    :param pytest.MonkeyPatch monkeypatch: Runtime patch helper.
    :return None: Assertions validate the race winner and warm cache behavior.
    """
    monkeypatch.setenv(EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR, "0.05")

    class _RaceEncodeModel:
        """Encode model that inserts the same row through a nested cache write."""

        def __init__(self, cache: EmbeddingCache) -> None:
            """Initialize the race model with the shared embedding cache.

            :param EmbeddingCache cache: Cache used by the concurrent winner.
            """
            self.cache = cache
            self.encode_calls = 0

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            """Persist a winner vector while producing a distinct loser vector.

            :param list[str] texts: Input texts requested for embedding.
            :param object kwargs: Encoder options not used by this fixture.
            :return np.ndarray: Fresh vector that loses the concurrent cache race.
            """
            del texts, kwargs
            self.encode_calls += 1
            self.cache.get_embeddings(
                {"p1": {"title": "Alpha", "abstract": "First"}},
                LookupEncodeModel(
                    {"Alpha. First": np.asarray([0.0, 1.0], dtype=np.float32)}
                ),
                show_progress=False,
            )
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="two-phase-lock-race",
            storage_precision="float32",
        )
        model = _RaceEncodeModel(cache)
        embeddings = cache.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            model,
            show_progress=False,
        )

        with cache._connect_db() as conn:
            paper_rows = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            row_idx = conn.execute(
                "SELECT row_idx FROM papers WHERE paper_id = 'p1'"
            ).fetchone()[0]
        with h5py.File(cache.h5_path, "r") as h5:
            embedding_rows = int(h5["embeddings"].shape[0])
        warm_embeddings = cache.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            model,
            show_progress=False,
        )

    assert model.encode_calls == 1
    assert paper_rows == 1
    assert embedding_rows == 1
    assert row_idx == 0
    np.testing.assert_allclose(embeddings["p1"], np.asarray([0.0, 1.0], np.float32))
    np.testing.assert_allclose(
        warm_embeddings["p1"], np.asarray([0.0, 1.0], np.float32)
    )


def test_embedding_cache_uses_length_bucketed_encode_batches() -> None:
    """Cache encode work should batch similarly sized texts together."""

    class _CaptureEncodeModel:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            del kwargs
            self.calls.append(list(texts))
            return np.asarray(
                [[float(len(text)), float(idx)] for idx, text in enumerate(texts)],
                dtype=np.float32,
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="length-bucketed-cache",
            storage_precision="float32",
        )
        model = _CaptureEncodeModel()
        papers = {
            "p1": {"title": "Long", "abstract": "x " * 120},
            "p2": {"title": "Tiny", "abstract": "short"},
            "p3": {"title": "Medium", "abstract": "x " * 80},
            "p4": {"title": "Small", "abstract": "tiny words"},
        }

        embeddings = cache.get_embeddings(
            papers, model, batch_size=2, show_progress=False
        )

    assert [len(batch) for batch in model.calls] == [2, 2]
    call_lengths = [[len(text) for text in batch] for batch in model.calls]
    assert call_lengths == sorted(call_lengths, key=lambda item: (max(item), item))
    assert list(embeddings) == ["p1", "p2", "p3", "p4"]


def test_embedding_cache_dispatches_explicit_prefetch_encoder() -> None:
    """Cache encoding should hand the full miss set to an enabled prefetch proxy.

    :return None: Verifies explicit fast-path dispatch and complete miss-set input.
    """

    class _PrefetchEncodeModel:
        """Capture the optimized cache encoding dispatch."""

        prefetch_batches = True

        def __init__(self) -> None:
            """Initialize captured prefetch calls.

            :return None: Creates an empty call log.
            """
            self.calls: list[tuple[list[str], int]] = []

        def encode_prefetched(self, texts: list[str], *, batch_size: int) -> np.ndarray:
            """Record one prefetch dispatch and return deterministic vectors.

            :param list[str] texts: Full cache-miss text set.
            :param int batch_size: Requested model batch size.
            :return np.ndarray: Stable FP32 embeddings.
            """
            self.calls.append((list(texts), batch_size))
            return np.asarray(
                [[float(len(text)), 1.0] for text in texts], dtype=np.float32
            )

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            """Fail if the generic encode path is selected.

            :param list[str] texts: Unexpected generic encode payload.
            :param object kwargs: Unexpected generic encode options.
            :return np.ndarray: Never returns.
            """
            del texts, kwargs
            raise AssertionError("generic encode path should not run")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="prefetched-cache",
            storage_precision="float32",
        )
        model = _PrefetchEncodeModel()
        papers = {
            "p1": {"title": "Long", "abstract": "x " * 40},
            "p2": {"title": "Tiny", "abstract": "short"},
            "p3": {"title": "Medium", "abstract": "x " * 20},
        }

        embeddings = cache.get_embeddings(
            papers, model, batch_size=2, show_progress=False
        )

    assert model.calls == [
        (
            [
                "Long. " + ("x " * 40).strip(),
                "Tiny. short",
                "Medium. " + ("x " * 20).strip(),
            ],
            2,
        )
    ]
    assert list(embeddings) == ["p1", "p2", "p3"]


def test_prefetch_proxy_overlaps_preprocessing_and_restores_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The next CPU batch should prepare during forward without changing row order.

    :param pytest.MonkeyPatch monkeypatch: Module patch fixture.
    :return None: Verifies overlap, ordering, and normalized FP32 results.
    """
    overlap_event = threading.Event()
    texts = ["xxxxxxxx", "a", "mmmm"]
    vectors = {
        "xxxxxxxx": [3.0, 4.0],
        "a": [1.0, 0.0],
        "mmmm": [0.0, 2.0],
    }
    model = _PrefetchModelStub(
        vectors,
        {text: len(text) for text in texts},
        overlap_event=overlap_event,
    )
    proxy = _make_prefetch_proxy(monkeypatch, model, lambda _error: False)

    embeddings = proxy.encode_prefetched(texts, batch_size=2)

    np.testing.assert_allclose(
        embeddings,
        np.asarray([[0.6, 0.8], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    assert overlap_event.is_set()
    assert len(model.preprocess_calls) == 2
    assert all(
        worker_thread != model.forward_threads[0]
        for _, worker_thread in model.preprocess_calls
    )
    assert model.eval_calls == 1


def test_prefetch_proxy_retokenizes_only_saturated_truncation_candidates(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exact-window inputs should be checked but only oversized inputs should warn.

    :param pytest.MonkeyPatch monkeypatch: Module patch fixture.
    :param pytest.LogCaptureFixture caplog: Captured warning records.
    :return None: Verifies candidate filtering and exact warning counts.
    """
    texts = ["short", "exact window", "oversized input payload"]
    model = _PrefetchModelStub(
        {text: [1.0, 0.0] for text in texts},
        {
            "short": 2,
            "exact window": 4,
            "oversized input payload": 7,
        },
        max_seq_length=4,
    )
    proxy = _make_prefetch_proxy(monkeypatch, model, lambda _error: False)

    with caplog.at_level(logging.WARNING):
        proxy.encode_prefetched(texts, batch_size=2)

    checked_inputs = [text for call in model.tokenizer_calls for text in call]
    assert checked_inputs == [
        "prompt: exact window",
        "prompt: oversized input payload",
    ]
    assert "truncate 1 of 3 inputs" in caplog.text


def test_prefetch_proxy_compile_retry_clears_partial_warning_count(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An eager retry should replace partial outputs and truncation accounting.

    :param pytest.MonkeyPatch monkeypatch: Module patch fixture.
    :param pytest.LogCaptureFixture caplog: Captured warning records.
    :return None: Verifies complete retry output and reset truncation counts.
    """
    texts = ["oversized input", "short"]
    model = _PrefetchModelStub(
        {text: [1.0, 0.0] for text in texts},
        {"oversized input": 8, "short": 2},
        max_seq_length=4,
        fail_forward_call=2,
    )
    restored_errors: list[str] = []

    def restore_eager(error: Exception) -> bool:
        """Record the compile failure and allow one eager retry.

        :param Exception error: Failed compiled call.
        :return bool: Always permits retry.
        """
        restored_errors.append(str(error))
        return True

    proxy = _make_prefetch_proxy(monkeypatch, model, restore_eager)

    with caplog.at_level(logging.WARNING):
        embeddings = proxy.encode_prefetched(texts, batch_size=1)

    np.testing.assert_array_equal(
        embeddings, np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    )
    assert restored_errors == ["compiled forward failed"]
    assert model.forward_calls == 4
    assert "truncate 1 of 2 inputs" in caplog.text
    assert "truncate 2 of 2 inputs" not in caplog.text


def test_int8_quantization_uses_uniform_buckets_and_clips_tails() -> None:
    """Signed conversion must not merge the two buckets surrounding zero.

    :return None: Checks SBERT bucket boundaries, offsets, and saturation behavior.
    """
    ranges = np.asarray([[0.0, 10.0], [255.0, 265.0]], dtype=np.float32)
    values = np.asarray([-1.0, 0.0, 0.9, 127.9, 128.9, 255.0, 256.0], dtype=np.float32)
    embeddings = np.column_stack((values, values + 10.0))

    quantized = embedding_cache_module._quantize_int8_embeddings(embeddings, ranges)

    expected = np.asarray([-128, -128, -128, -1, 0, 127, 127], dtype=np.int8)
    np.testing.assert_array_equal(quantized, np.column_stack((expected, expected)))


def test_int8_dequantization_centres_buckets_without_exceeding_ranges(
    tmp_path: Path,
) -> None:
    """Bucket-centre reconstruction removes floor bias within the calibration range.

    :param Path tmp_path: Isolated cache directory.
    :return None: Checks round-trip error, average bias, and saturated endpoints.
    """
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="int8-roundtrip")
    ranges = np.asarray([[0.0, -10.0], [255.0, 500.0]], dtype=np.float32)
    cache.set_calibration_ranges(ranges, embedding_dim=2)
    levels = (np.arange(25500, dtype=np.float32) + 0.5) / 100.0
    steps = (ranges[1] - ranges[0]) / 255.0
    embeddings = ranges[0] + levels[:, None] * steps
    quantized = embedding_cache_module._quantize_int8_embeddings(embeddings, ranges)
    with h5py.File(cache.h5_path, "r") as h5:
        reconstructed = cache._dequantize_int8(h5, quantized)
        endpoints = cache._dequantize_int8(
            h5, np.asarray([[-128, -128], [127, 127]], dtype=np.int8)
        )

    errors = (reconstructed - embeddings) / steps
    floors = ranges[0] + (quantized.astype(np.float32) + 128.0) * steps
    floor_errors = (floors - embeddings) / steps
    assert np.max(np.abs(errors)) <= 0.5
    assert abs(float(np.mean(errors))) < 1e-5
    assert np.mean(errors**2) < np.mean(floor_errors**2) / 3.9
    assert np.all(endpoints >= ranges[0])
    assert np.all(endpoints <= ranges[1])
    np.testing.assert_array_equal(endpoints[-1], ranges[1])


@pytest.mark.parametrize("storage_precision", ["float32", "int8"])
def test_embedding_cache_returns_stored_vector_fidelity_on_first_encode(
    tmp_path: Path, storage_precision: str
) -> None:
    """Encoding and later hits must return the same vector for a paper.

    :param Path tmp_path: Isolated cache directory.
    :param str storage_precision: Persisted vector format under test.
    :return None: Checks int8 misses return the persisted round-trip, not raw output.
    """
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        model_name=f"stored-fidelity-{storage_precision}",
        storage_precision=storage_precision,
    )
    _set_test_int8_calibration(cache)
    papers = {"p1": {"title": "Seed", "abstract": "Abstract"}}
    encoded = np.asarray([0.6, 0.8], dtype=np.float32)
    miss = cache.get_embeddings(
        papers,
        LookupEncodeModel({"Seed. Abstract": encoded}),
        show_progress=False,
    )
    # An empty lookup would raise if this call re-encoded instead of hitting.
    hit = cache.get_embeddings(papers, LookupEncodeModel({}), show_progress=False)

    np.testing.assert_array_equal(miss["p1"], hit["p1"])
    if storage_precision == "int8":
        assert not np.array_equal(miss["p1"], encoded)
    else:
        np.testing.assert_array_equal(miss["p1"], encoded)


def test_embedding_cache_upsert_records_int8_saturation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Int8 cache writes should persist saturation telemetry when values clip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="int8-saturation-stats",
            storage_precision="int8",
        )
        cache.set_calibration_ranges(
            ranges=np.vstack(
                (
                    np.zeros(2, dtype=np.float32),
                    np.ones(2, dtype=np.float32),
                )
            ),
            embedding_dim=2,
        )

        out_of_range_model = LookupEncodeModel(
            {"Alpha. First": np.asarray([2.0, -1.0], dtype=np.float32)}
        )
        with caplog.at_level("WARNING"):
            cache.upsert_embeddings(
                {"p1": {"title": "Alpha", "abstract": "First"}},
                out_of_range_model,
                show_progress=False,
            )

        assert any(
            "Int8 calibration saturation detected" in record.message
            for record in caplog.records
        )

        with h5py.File(cache.h5_path, "r") as h5:
            assert int(h5.attrs[INT8_CLIPPED_VALUE_COUNT_KEY]) == 2
            assert int(h5.attrs[INT8_TOTAL_VALUE_COUNT_KEY]) == 2


def test_embedding_cache_int8_saturation_warning_emits_once_per_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Repeated clipped writes should warn once per cache instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="int8-warn-once")
        cache.set_calibration_ranges(
            ranges=np.vstack(
                (
                    np.zeros(2, dtype=np.float32),
                    np.ones(2, dtype=np.float32),
                )
            ),
            embedding_dim=2,
        )
        model = LookupEncodeModel(
            {
                "Alpha. First": np.asarray([2.0, -1.0], dtype=np.float32),
                "Beta. Second": np.asarray([3.0, -2.0], dtype=np.float32),
            }
        )

        with caplog.at_level("WARNING"):
            cache.upsert_embeddings(
                {"p1": {"title": "Alpha", "abstract": "First"}},
                model,
                show_progress=False,
            )
            cache.upsert_embeddings(
                {"p2": {"title": "Beta", "abstract": "Second"}},
                model,
                show_progress=False,
            )

        warning_messages = [
            record.message
            for record in caplog.records
            if "Int8 calibration saturation detected" in record.message
        ]
        assert len(warning_messages) == 1
        assert "Further warnings are suppressed for this run" in warning_messages[0]
        assert "not retrieval recall" in warning_messages[0]
        assert "--force-rebuild-cache" in warning_messages[0]
        assert cache.h5_path.name in warning_messages[0]

        with h5py.File(cache.h5_path, "r") as h5:
            assert int(h5.attrs[INT8_CLIPPED_VALUE_COUNT_KEY]) == 4
            assert int(h5.attrs[INT8_TOTAL_VALUE_COUNT_KEY]) == 4


@pytest.mark.parametrize("populated", [False, True])
def test_embedding_cache_calibration_changes_require_empty_cache(
    tmp_path: Path, populated: bool
) -> None:
    """Replacing ranges must never reinterpret already stored int8 embeddings.

    :param Path tmp_path: Isolated persistent cache directory.
    :param bool populated: Whether the cache already contains an encoded row.
    :return None: Range changes preserve existing vectors or apply to an empty cache.
    """
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="fixed-ranges")
    ranges = np.asarray([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    cache.set_calibration_ranges(ranges, embedding_dim=2)
    if populated:
        cache.upsert_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            LookupEncodeModel(
                {"Alpha. First": np.asarray([0.6, 0.8], dtype=np.float32)}
            ),
            show_progress=False,
        )
        with h5py.File(cache.h5_path, "r") as h5:
            stored = h5["embeddings"][:]
        cache = EmbeddingCache(cache_dir=tmp_path, model_name="fixed-ranges")
        with pytest.raises(ValueError, match="Cannot change int8 calibration ranges"):
            cache.set_calibration_ranges(ranges * 2, embedding_dim=2)
        cache.set_calibration_ranges(ranges, embedding_dim=2)
        with h5py.File(cache.h5_path, "r") as h5:
            np.testing.assert_array_equal(h5["calibration_ranges"][:], ranges)
            np.testing.assert_array_equal(h5["embeddings"][:], stored)
    else:
        cache.set_calibration_ranges(ranges * 2, embedding_dim=2)
        with h5py.File(cache.h5_path, "r") as h5:
            np.testing.assert_array_equal(h5["calibration_ranges"][:], ranges * 2)


def test_embedding_cache_repeated_attribute_updates_do_not_bloat_hdf5_metadata(
    tmp_path: Path,
) -> None:
    """Repeated metadata updates should not accumulate replaced HDF5 attributes."""
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="hdf5-attribute-growth")
    cache.set_calibration_ranges(
        ranges=np.vstack(
            (
                np.full(512, -1.0, dtype=np.float32),
                np.full(512, 1.0, dtype=np.float32),
            )
        ),
        embedding_dim=512,
    )
    initial_size = cache.h5_path.stat().st_size
    for _ in range(256):
        with h5py.File(cache.h5_path, "a") as h5:
            cache._set_h5_attrs(h5)
            cache._record_int8_saturation(
                h5,
                clipped_value_count=0,
                total_value_count=512 * 256,
            )

    metadata_growth = cache.h5_path.stat().st_size - initial_size
    with h5py.File(cache.h5_path, "r") as h5:
        assert int(h5.attrs[INT8_CLIPPED_VALUE_COUNT_KEY]) == 0
        assert int(h5.attrs[INT8_TOTAL_VALUE_COUNT_KEY]) == 512 * 256 * 256

    assert metadata_growth < 64 * 1024


def test_embedding_cache_default_int8_path_stays_local_to_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default int8 writes/search should not import sentence-transformers quantization."""

    original_import = builtins.__import__

    def _guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "sentence_transformers.quantization":
            raise AssertionError("EmbeddingCache quantization should stay local.")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="local-quantizer-default")
        _set_test_int8_calibration(cache)
        lookup = LookupEncodeModel(
            {
                "Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32),
                "Beta. Second": np.asarray([0.0, 1.0], dtype=np.float32),
            }
        )
        cache.get_embeddings(
            {
                "p1": {"title": "Alpha", "abstract": "First"},
                "p2": {"title": "Beta", "abstract": "Second"},
            },
            lookup,
            show_progress=False,
        )

        results = cache.search(
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=1,
            binary_prefilter=True,
            binary_rescore_multiplier=4,
        )

    assert [result.paper_id for result in results] == ["p1"]


def test_embedding_cache_search_and_calibration_reuse_contract() -> None:
    """Search should return metadata and reuse calibration ranges in one namespace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="search-metadata")
        _set_test_int8_calibration(cache)
        papers = {
            "p1": {
                "title": "Alpha",
                "abstract": "First",
                "year": 2020,
                "authors": ["Alice", "Bob"],
                "categories": ["cs.AI"],
                "venue": "NeurIPS",
                "arxiv_id": "2411.03884",
                "doi": "10.1145/3133956.3134029",
            },
            "p2": {
                "title": "Beta",
                "abstract": "Second",
                "year": 2021,
                "authors": ["Carol"],
                "categories": ["cs.LG"],
                "venue": "ICML",
                "arxiv_id": "2501.00001",
                "doi": "",
            },
        }
        first_model = LookupEncodeModel(
            {
                "Alpha. First": np.array([1.0, 0.0], dtype=np.float32),
                "Beta. Second": np.array([0.0, 1.0], dtype=np.float32),
            }
        )

        cache.get_embeddings(papers, first_model, show_progress=False)
        with h5py.File(cache.h5_path, "r") as h5:
            ranges_before = np.asarray(h5["calibration_ranges"], dtype=np.float32)

        results = cache.search(
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=2,
            binary_prefilter=True,
            binary_rescore_multiplier=4,
        )

        second_model = LookupEncodeModel(
            {"Gamma. Third": np.array([0.3, 0.7], dtype=np.float32)}
        )
        cache.get_embeddings(
            {"p3": {"title": "Gamma", "abstract": "Third"}},
            second_model,
            show_progress=False,
        )
        with h5py.File(cache.h5_path, "r") as h5:
            ranges_after = np.asarray(h5["calibration_ranges"], dtype=np.float32)

    np.testing.assert_allclose(ranges_before, ranges_after)
    assert [result.paper_id for result in results[:2]] == ["p1", "p2"]
    assert np.linalg.norm(results[0].embedding) == pytest.approx(1.0)
    assert results[0].metadata["authors"] == ["Alice", "Bob"]
    assert results[0].metadata["categories"] == ["cs.AI"]
    assert results[0].metadata["year"] == 2020
    assert results[0].metadata["venue"] == "NeurIPS"
    assert results[0].metadata["arxiv_id"] == "2411.03884"
    assert results[0].metadata["doi"] == "10.1145/3133956.3134029"
    assert results[0].embedding.dtype == np.float32
    assert cache.embedding_vector_dtype == "float32"
    assert cache.storage_precision == "int8"


@pytest.mark.parametrize("storage_precision", ["float32", "int8"])
def test_embedding_cache_search_normalizes_scaled_queries(
    storage_precision: str,
) -> None:
    """Scaled copies of one query should produce identical rankings and scores.

    :param str storage_precision: Persisted vector format under test.
    :return None: Assertions verify query-scale-independent search results.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(
            cache_dir=tmpdir,
            model_name=f"scaled-query-{storage_precision}",
            storage_precision=storage_precision,
        )
        _set_test_int8_calibration(cache)
        papers = {
            "p1": {"title": "Alpha", "abstract": "First"},
            "p2": {"title": "Beta", "abstract": "Second"},
        }
        cache.get_embeddings(
            papers,
            LookupEncodeModel(
                {
                    "Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32),
                    "Beta. Second": np.asarray([0.0, 1.0], dtype=np.float32),
                }
            ),
            show_progress=False,
        )

        query = np.asarray([1.0, 0.25], dtype=np.float32)
        baseline = cache.search(
            query_embedding=query,
            top_k=2,
            binary_prefilter=False,
            binary_rescore_multiplier=1,
        )
        scaled = cache.search(
            query_embedding=query * 2.0,
            top_k=2,
            binary_prefilter=False,
            binary_rescore_multiplier=1,
        )

    assert [result.paper_id for result in baseline] == [
        result.paper_id for result in scaled
    ]
    np.testing.assert_allclose(
        [result.score for result in baseline],
        [result.score for result in scaled],
    )


@pytest.mark.parametrize("storage_precision", ["float32", "int8"])
def test_embedding_cache_search_keeps_tied_top_k_order_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
    storage_precision: str,
) -> None:
    """Float and int8 searches should rank ties by row over multiple chunks."""
    monkeypatch.setattr(embedding_cache_module, "EMBEDDING_SEARCH_CHUNK_ROWS", 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(
            cache_dir=tmpdir,
            model_name=f"tied-search-{storage_precision}",
            storage_precision=storage_precision,
        )
        _set_test_int8_calibration(cache)
        papers = {
            f"p{idx}": {"title": f"Title {idx}", "abstract": "Same"} for idx in range(5)
        }
        model = LookupEncodeModel(
            {
                f"Title {idx}. Same": np.asarray([1.0, 0.0], dtype=np.float32)
                for idx in range(5)
            }
        )
        cache.get_embeddings(papers, model, show_progress=False)
        bounded_results = cache.search(
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=3,
            binary_prefilter=False,
            binary_rescore_multiplier=1,
        )
        all_results = cache.search(
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=10,
            binary_prefilter=False,
            binary_rescore_multiplier=1,
        )

    assert [result.paper_id for result in bounded_results] == ["p0", "p1", "p2"]
    assert [result.paper_id for result in all_results] == [
        f"p{idx}" for idx in range(5)
    ]
    assert [result.score for result in all_results] == pytest.approx(
        [all_results[0].score] * len(all_results)
    )


def test_embedding_cache_search_rejects_non_finite_query_embedding() -> None:
    """Search should reject NaN query vectors before opening the cache payload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="non-finite-query",
            storage_precision="float32",
        )
        with pytest.raises(
            ValueError, match="query_embedding must contain only finite values"
        ):
            cache.search(
                query_embedding=np.asarray([np.nan, 0.0], dtype=np.float32),
                top_k=1,
                binary_prefilter=False,
                binary_rescore_multiplier=1,
            )


def test_embedding_cache_search_fails_closed_on_non_finite_stored_vector() -> None:
    """Search should identify non-finite matrix data as cache corruption."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="non-finite-stored-vector",
            storage_precision="float32",
        )
        cache.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            LookupEncodeModel(
                {"Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32)}
            ),
            show_progress=False,
        )
        with h5py.File(cache.h5_path, "a") as h5:
            h5[EMBEDDINGS_DATASET_NAME][0, 0] = np.nan

        with pytest.raises(RuntimeError, match="non-finite scores"):
            cache.search(
                query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                top_k=1,
                binary_prefilter=False,
                binary_rescore_multiplier=1,
            )


@pytest.mark.parametrize("storage_precision", ["float32", "int8"])
def test_embedding_cache_rejects_non_finite_model_output_before_write(
    storage_precision: str,
) -> None:
    """Non-finite encoder output must not create vector or metadata rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(
            cache_dir=tmpdir,
            model_name=f"non-finite-model-output-{storage_precision}",
            storage_precision=storage_precision,
        )
        _set_test_int8_calibration(cache)
        model = LookupEncodeModel(
            {"Alpha. First": np.asarray([np.nan, np.inf], dtype=np.float32)}
        )

        with pytest.raises(ValueError, match="non-finite values"):
            cache.upsert_embeddings(
                {"p1": {"title": "Alpha", "abstract": "First"}},
                model,
                show_progress=False,
            )

        assert cache.embedding_count() == 0
        with cache._connect_db() as conn:
            assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_embedding_cache_rejects_non_finite_calibration_ranges(
    bad_value: float,
) -> None:
    """Calibration data must be finite before it is persisted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="non-finite-calibration")
        ranges = np.asarray([[-1.0, -1.0], [1.0, bad_value]], dtype=np.float32)

        with pytest.raises(ValueError, match="only finite values"):
            cache.set_calibration_ranges(ranges=ranges, embedding_dim=2)

        assert cache.has_calibration_ranges() is False


def test_embedding_cache_matrix_dataset_helper_validates_dimensions_and_dtypes() -> (
    None
):
    """Shared matrix helper should create and validate both cache matrix types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="matrix-dataset-helper")
        with h5py.File(cache.h5_path, "a") as h5:
            embeddings = cache._ensure_embeddings_dataset(h5, embedding_dim=9)
            binary = cache._ensure_binary_dataset(h5, embedding_dim=9)
            assert binary is not None
            assert embeddings.shape == (0, 9)
            assert embeddings.maxshape == (None, 9)
            assert embeddings.dtype == np.int8
            assert binary.shape == (0, 2)
            assert binary.maxshape == (None, 2)
            assert binary.dtype == np.uint8

        with h5py.File(cache.h5_path, "w") as h5:
            h5.create_dataset(
                EMBEDDINGS_DATASET_NAME,
                shape=(0, 9),
                maxshape=(None, 9),
                dtype=np.float32,
            )
            with pytest.raises(ValueError, match="dtype mismatch"):
                cache._ensure_embeddings_dataset(h5, embedding_dim=9)

        with h5py.File(cache.h5_path, "w") as h5:
            h5.create_dataset(
                BINARY_INDEX_DATASET_NAME,
                shape=(0, 2),
                maxshape=(None, 2),
                dtype=np.int8,
            )
            with pytest.raises(ValueError, match="dtype mismatch"):
                cache._ensure_binary_dataset(h5, embedding_dim=9)


def test_embedding_cache_metadata_loaders_preserve_serialized_and_parsed_shapes() -> (
    None
):
    """Shared SQLite decoder should retain each caller's JSON-list contract."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="metadata-loader-shapes",
            storage_precision="float32",
        )
        cache.get_embeddings(
            {
                "p1": {
                    "title": "Alpha",
                    "abstract": "First",
                    "authors": ["Alice", "Bob"],
                    "categories": ["cs.AI"],
                    "venue": "NeurIPS",
                    "arxiv_id": "2411.03884",
                    "doi": "10.1/example",
                }
            },
            LookupEncodeModel(
                {"Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32)}
            ),
            show_progress=False,
        )
        with cache._connect_db() as conn:
            serialized = cache._load_existing_rows(conn, ["p1"])["p1"]
            parsed = cache._load_metadata_by_rows(conn, [0])[0]

    assert serialized["authors_json"] == '["Alice", "Bob"]'
    assert serialized["categories_json"] == '["cs.AI"]'
    assert "authors" not in serialized
    assert parsed["authors"] == ["Alice", "Bob"]
    assert parsed["categories"] == ["cs.AI"]
    assert "authors_json" not in parsed
    assert parsed["paper_id"] == "p1"


def test_embedding_cache_metadata_refresh_survives_mixed_batch_encode_failure() -> None:
    """Metadata-only cache-hit refresh should persist even if miss encoding fails."""

    class _FailingEncodeModel:
        """Encode model stub that always fails for negative-path simulation."""

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            del texts, kwargs
            raise RuntimeError("encode failure")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="metadata-refresh-durable")
        _set_test_int8_calibration(cache)
        cache.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            LookupEncodeModel(
                {"Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32)}
            ),
            show_progress=False,
        )

        with pytest.raises(RuntimeError, match="encode failure"):
            cache.get_embeddings(
                {
                    "p1": {
                        "title": "Alpha",
                        "abstract": "First",
                        "year": 2025,
                        "venue": "ICLR",
                        "arxiv_id": "2411.03884",
                        "doi": "10.1145/3133956.3134029",
                    },
                    "p2": {"title": "Beta", "abstract": "Second"},
                },
                _FailingEncodeModel(),
                show_progress=False,
            )

        with cache._connect_db() as conn:
            refreshed_row = conn.execute(
                """
                SELECT year, venue, arxiv_id, doi
                FROM papers
                WHERE paper_id = 'p1'
                """
            ).fetchone()

    assert refreshed_row == (
        2025,
        "ICLR",
        "2411.03884",
        "10.1145/3133956.3134029",
    )


def test_embedding_cache_search_returns_empty_when_h5_is_missing() -> None:
    """Search should safely return no candidates when cache file is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="missing-h5")
        results = cache.search(
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=3,
            binary_prefilter=True,
            binary_rescore_multiplier=2,
        )

    assert results == []


def test_corpus_size_token_encodes_newest_slice_policy() -> None:
    """Capped tokens carry the slice policy so legacy head-slice caches rehydrate."""
    assert _corpus_size_token(None) == "all"
    assert _corpus_size_token(1000) == "newest:1000"


def test_legacy_head_slice_hydration_metadata_fails_is_hydrated() -> None:
    """Caches hydrated under the head-slice policy must not pass is_hydrated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="newest-slice-migration")
        _set_test_int8_calibration(cache)
        cache.get_embeddings(
            {
                "p1": {"title": "Alpha", "abstract": "First"},
                "p2": {"title": "Beta", "abstract": "Second"},
            },
            LookupEncodeModel(
                {
                    "Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32),
                    "Beta. Second": np.asarray([0.0, 1.0], dtype=np.float32),
                }
            ),
            show_progress=False,
        )
        cache.mark_hydrated(
            dataset_source="fake/source",
            dataset_split="train",
            corpus_size=2,
            complete=True,
        )
        assert cache.is_hydrated("train", 2, dataset_source="fake/source")

        # Simulate a cache hydrated before the newest-slice policy landed.
        with cache._connect_db() as conn:
            cache._set_cache_metadata(conn, {HYDRATION_CORPUS_SIZE_KEY: "2"})
        assert not cache.is_hydrated("train", 2, dataset_source="fake/source")


def test_embedding_cache_embedding_count_tracks_persisted_rows() -> None:
    """embedding_count should report 0 for empty namespaces and rows after upserts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="embedding-count")
        assert cache.embedding_count() == 0

        _set_test_int8_calibration(cache)
        cache.get_embeddings(
            {
                "p1": {"title": "Alpha", "abstract": "First"},
                "p2": {"title": "Beta", "abstract": "Second"},
            },
            LookupEncodeModel(
                {
                    "Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32),
                    "Beta. Second": np.asarray([0.0, 1.0], dtype=np.float32),
                }
            ),
            show_progress=False,
        )
        assert cache.embedding_count() == 2


def test_embedding_cache_default_text_builder_matches_profile_formatter() -> None:
    """Default cache text builder should match the default profile formatter."""

    class _CaptureEncodeModel:
        """Capture encoded texts while returning deterministic float32 embeddings."""

        def __init__(self) -> None:
            self.texts: list[str] = []

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            del kwargs
            self.texts.extend(texts)
            return np.repeat(
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                len(texts),
                axis=0,
            )

    papers = {
        "p1": {"title": "Alpha", "abstract": "First"},
        "p2": {"title": "Beta", "abstract": "  "},
        "p3": {"title": "   ", "abstract": "Gamma"},
    }
    profile = get_embedding_model_profile("org/generic-embedding-model")
    expected_texts = [profile.format_document(metadata) for metadata in papers.values()]

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="text-builder-parity")
        _set_test_int8_calibration(cache)
        model = _CaptureEncodeModel()
        embeddings = cache.get_embeddings(papers, model, show_progress=False)

    assert list(embeddings) == list(papers)
    assert model.texts == expected_texts
    assert expected_texts == ["Alpha. First", "Beta", "Gamma"]


def test_embedding_cache_search_raises_on_missing_metadata_rows() -> None:
    """Search should fail closed when scored rows have no metadata payload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="missing-search-metadata")
        _set_test_int8_calibration(cache)
        cache.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            LookupEncodeModel({"Alpha. First": np.array([1.0, 0.0], dtype=np.float32)}),
            show_progress=False,
        )

        with cache._connect_db() as conn:
            conn.execute("DELETE FROM papers")
            conn.commit()

        with pytest.raises(
            RuntimeError,
            match=(
                "Embedding cache integrity error: "
                "(missing metadata rows|embedding row mapping mismatch)"
            ),
        ):
            cache.search(
                query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                top_k=1,
                binary_prefilter=True,
                binary_rescore_multiplier=2,
            )


def test_embedding_cache_search_rejects_non_vector_queries() -> None:
    """Search should reject non-1D query embeddings instead of flattening them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="invalid-query-shape")
        _set_test_int8_calibration(cache)
        cache.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            LookupEncodeModel({"Alpha. First": np.array([1.0, 0.0], dtype=np.float32)}),
            show_progress=False,
        )

        with pytest.raises(ValueError, match="query_embedding must be 1-dimensional"):
            cache.search(
                query_embedding=np.asarray([[1.0, 0.0]], dtype=np.float32),
                top_k=1,
                binary_prefilter=True,
                binary_rescore_multiplier=2,
            )


@pytest.mark.parametrize(
    ("model_name", "cache_kwargs", "metadata_key", "metadata_value", "match"),
    [
        pytest.param(
            "search-metadata-mismatch",
            {},
            "storage_precision",
            "float16",
            "metadata key 'storage_precision' mismatch",
            id="storage_precision",
        ),
        pytest.param(
            "search-source-dtype-mismatch",
            {"source_torch_dtype": "float32"},
            SOURCE_TORCH_DTYPE_KEY,
            "bfloat16",
            "metadata key 'source_torch_dtype' mismatch",
            id="source_torch_dtype",
        ),
        pytest.param(
            "search-calibration-sample-mismatch",
            {"storage_precision": "int8", "calibration_sample_size": 8},
            CALIBRATION_SAMPLE_SIZE_KEY,
            "32",
            "metadata key 'calibration_sample_size' mismatch",
            id="calibration_sample_size",
        ),
        pytest.param(
            "search-text-formatter-mismatch",
            {"text_formatter_fingerprint": "fmt-a"},
            TEXT_FORMATTER_FINGERPRINT_KEY,
            "fmt-b",
            "metadata key 'text_formatter_fingerprint' mismatch",
            id="text_formatter_fingerprint",
        ),
        pytest.param(
            "search-compression-filter-mismatch",
            {"compression": "gzip"},
            COMPRESSION_FILTER_KEY,
            "lzf",
            "metadata key 'compression_filter' mismatch",
            id="compression_filter",
        ),
        pytest.param(
            "search-compression-level-mismatch",
            {"compression": "gzip", "compression_level": 1},
            COMPRESSION_LEVEL_KEY,
            "9",
            "metadata key 'compression_level' mismatch",
            id="compression_level",
        ),
    ],
)
def test_embedding_cache_search_fails_closed_on_metadata_provenance_mismatch(
    model_name: str,
    cache_kwargs: dict[str, object],
    metadata_key: str,
    metadata_value: str,
    match: str,
) -> None:
    """Search should fail closed when persisted provenance metadata drifts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name=model_name, **cache_kwargs)
        _set_test_int8_calibration(cache)
        cache.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            LookupEncodeModel({"Alpha. First": np.array([1.0, 0.0], dtype=np.float32)}),
            show_progress=False,
        )

        with cache._connect_db() as conn:
            conn.execute(
                "UPDATE cache_metadata SET value = ? WHERE key = ?",
                (metadata_value, metadata_key),
            )
            conn.commit()

        with pytest.raises(RuntimeError, match=match):
            cache.search(
                query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                top_k=1,
                binary_prefilter=True,
                binary_rescore_multiplier=2,
            )


def test_embedding_cache_search_handles_unsorted_prefilter_candidates() -> None:
    """Search should normalize unsorted prefilter rows before HDF5 fancy indexing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="unsorted-prefilter-cands")
        _set_test_int8_calibration(cache)
        papers = {
            f"p{idx}": {"title": f"Title {idx}", "abstract": "Abstract"}
            for idx in range(6)
        }
        lookup = LookupEncodeModel(
            {
                **{
                    f"Title {idx}. Abstract": np.asarray([0.0, 1.0], dtype=np.float32)
                    for idx in range(5)
                },
                "Title 5. Abstract": np.asarray([1.0, 0.0], dtype=np.float32),
            }
        )
        cache.get_embeddings(papers, lookup, show_progress=False)

        cache._binary_prefilter_rows = (  # type: ignore[method-assign]
            lambda **_: np.asarray([5, 1], dtype=np.int64)
        )
        results = cache.search(
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=1,
            binary_prefilter=True,
            binary_rescore_multiplier=2,
        )

    assert [result.paper_id for result in results] == ["p5"]


def test_embedding_cache_binary_prefilter_rows_are_monotonic_subset() -> None:
    """Binary prefilter should return monotonic candidate rows for HDF5 locality/safety."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="prefilter-monotonic")
        _set_test_int8_calibration(cache)
        papers = {
            f"p{idx}": {"title": f"Title {idx}", "abstract": "Abstract"}
            for idx in range(6)
        }
        lookup = LookupEncodeModel(
            {
                **{
                    f"Title {idx}. Abstract": np.asarray([0.0, 1.0], dtype=np.float32)
                    for idx in range(5)
                },
                "Title 5. Abstract": np.asarray([1.0, 0.0], dtype=np.float32),
            }
        )
        cache.get_embeddings(papers, lookup, show_progress=False)

        with h5py.File(cache.h5_path, "r") as h5:
            rows = cache._binary_prefilter_rows(
                binary_dataset=h5["binary_index"],
                query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                candidate_count=2,
            )

    assert rows.shape == (2,)
    assert np.all(rows[:-1] < rows[1:])


def test_embedding_cache_binary_prefilter_resolves_local_cutoff_ties_by_row() -> None:
    """Chunk-local Hamming ties should retain the lowest absolute row indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="prefilter-local-ties")
        with h5py.File(cache.h5_path, "a") as h5:
            binary = h5.create_dataset(
                BINARY_INDEX_DATASET_NAME,
                data=np.zeros((65536, 1), dtype=np.uint8),
            )
            rows = cache._binary_prefilter_rows(
                binary_dataset=binary,
                query_embedding=np.zeros(8, dtype=np.float32),
                candidate_count=64,
            )

    np.testing.assert_array_equal(rows, np.arange(64, dtype=np.int64))


def test_embedding_cache_compression_codec_contracts() -> None:
    """Supported codecs should hydrate; unsupported codecs should fail fast."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="lzf-codec",
            compression="lzf",
            compression_level=1,
        )
        _set_test_int8_calibration(cache)
        assert cache.compression_level == 0
        cache.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            LookupEncodeModel(
                {"Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32)}
            ),
            show_progress=False,
        )

        with cache._connect_db() as conn:
            metadata = dict(conn.execute("SELECT key, value FROM cache_metadata"))
            assert metadata[COMPRESSION_FILTER_KEY] == "lzf"
            assert metadata[COMPRESSION_LEVEL_KEY] == "0"

        with h5py.File(cache.h5_path, "r") as h5:
            assert h5["embeddings"].compression == "lzf"
            assert h5["binary_index"].compression == "lzf"
            assert h5["embeddings"].compression_opts is None
            assert str(h5.attrs[COMPRESSION_FILTER_KEY]) == "lzf"
            assert int(h5.attrs[COMPRESSION_LEVEL_KEY]) == 0

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="compression='szip' is unsupported"):
            EmbeddingCache(
                cache_dir=tmpdir,
                model_name="szip-codec",
                compression="szip",
                compression_level=1,
            )


def test_embedding_cache_adopts_existing_compression_until_explicit_clear() -> None:
    """Requested compression should apply only to new or explicitly rebuilt payloads."""
    source = "librarian-bots/arxiv-metadata-snapshot"
    with tempfile.TemporaryDirectory() as tmpdir:
        original = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="compression-adoption",
            compression="lzf",
            compression_level=0,
        )
        _set_test_int8_calibration(original)
        original.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            LookupEncodeModel(
                {"Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32)}
            ),
            show_progress=False,
        )
        original.mark_hydrated(
            dataset_source=source,
            dataset_split="train",
            corpus_size=2,
            complete=True,
        )

        reopened = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="compression-adoption",
            compression="gzip",
            compression_level=1,
        )
        assert reopened.embedding_count() == 1
        assert reopened.is_hydrated(
            dataset_split="train",
            corpus_size=2,
            dataset_source=source,
        )
        results = reopened.search(
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=1,
            binary_prefilter=True,
            binary_rescore_multiplier=2,
        )
        assert [result.paper_id for result in results] == ["p1"]

        reopened.get_embeddings(
            {"p2": {"title": "Beta", "abstract": "Second"}},
            LookupEncodeModel(
                {"Beta. Second": np.asarray([0.0, 1.0], dtype=np.float32)}
            ),
            show_progress=False,
        )
        with reopened._connect_db() as conn:
            metadata = dict(conn.execute("SELECT key, value FROM cache_metadata"))
            assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
        with h5py.File(reopened.h5_path, "r") as h5:
            assert h5[EMBEDDINGS_DATASET_NAME].compression == "lzf"
            assert h5[BINARY_INDEX_DATASET_NAME].compression == "lzf"
            assert str(h5.attrs[COMPRESSION_FILTER_KEY]) == "lzf"
            assert int(h5.attrs[COMPRESSION_LEVEL_KEY]) == 0
        assert metadata[COMPRESSION_FILTER_KEY] == "lzf"
        assert metadata[COMPRESSION_LEVEL_KEY] == "0"

        reopened.clear(reason="test requested compression after explicit rebuild")
        _set_test_int8_calibration(reopened)
        reopened.get_embeddings(
            {"p3": {"title": "Gamma", "abstract": "Third"}},
            LookupEncodeModel(
                {"Gamma. Third": np.asarray([1.0, 0.0], dtype=np.float32)}
            ),
            show_progress=False,
        )
        with reopened._connect_db() as conn:
            rebuilt_metadata = dict(
                conn.execute("SELECT key, value FROM cache_metadata")
            )
            assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
        with h5py.File(reopened.h5_path, "r") as h5:
            assert h5[EMBEDDINGS_DATASET_NAME].compression == "gzip"
            assert h5[EMBEDDINGS_DATASET_NAME].compression_opts == 1
            assert h5[BINARY_INDEX_DATASET_NAME].compression == "gzip"
            assert h5[BINARY_INDEX_DATASET_NAME].compression_opts == 1
        assert rebuilt_metadata[COMPRESSION_FILTER_KEY] == "gzip"
        assert rebuilt_metadata[COMPRESSION_LEVEL_KEY] == "1"


def test_embedding_cache_adopts_existing_gzip_level_without_masking_corruption() -> (
    None
):
    """Physical-level adoption must not weaken unrelated HDF5 provenance checks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="compression-level-adoption",
            compression="gzip",
            compression_level=9,
        )
        _set_test_int8_calibration(original)
        original.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            LookupEncodeModel(
                {"Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32)}
            ),
            show_progress=False,
        )

        reopened = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="compression-level-adoption",
            compression="gzip",
            compression_level=1,
        )
        assert reopened.embedding_count() == 1
        with h5py.File(reopened.h5_path, "r") as h5:
            assert h5[EMBEDDINGS_DATASET_NAME].compression_opts == 9

        with h5py.File(reopened.h5_path, "a") as h5:
            h5.attrs[SOURCE_TORCH_DTYPE_KEY] = "corrupt-dtype"
        with pytest.raises(RuntimeError, match="source_torch_dtype"):
            reopened.search(
                query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                top_k=1,
                binary_prefilter=True,
                binary_rescore_multiplier=2,
            )


def test_float_cache_chunk_layout_ignores_calibration_sample_size() -> None:
    """Float cache chunk layout should not vary with int8 calibration settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        small = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="float-chunks-small",
            storage_precision="float32",
            calibration_sample_size=8,
        )
        large = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="float-chunks-large",
            storage_precision="float32",
            calibration_sample_size=4096,
        )
        papers = {"p1": {"title": "Alpha", "abstract": "First"}}
        lookup = LookupEncodeModel(
            {"Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32)}
        )
        small.get_embeddings(papers, lookup, show_progress=False)
        large.get_embeddings(papers, lookup, show_progress=False)

        with h5py.File(small.h5_path, "r") as small_h5:
            small_chunks = small_h5["embeddings"].chunks
            assert small_chunks[0] == EMBEDDING_DATASET_CHUNK_ROWS
            assert "binary_index" not in small_h5
        with h5py.File(large.h5_path, "r") as large_h5:
            large_chunks = large_h5["embeddings"].chunks
            assert large_chunks[0] == EMBEDDING_DATASET_CHUNK_ROWS
            assert "binary_index" not in large_h5

    assert small_chunks == large_chunks


def test_embedding_cache_restart_persistence_contracts() -> None:
    """Hydration/fingerprint metadata and payload state should survive restarts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hydration_cache = EmbeddingCache(
            cache_dir=tmpdir, model_name="hydration-persistence"
        )
        _set_test_int8_calibration(hydration_cache)
        hydration_cache.get_embeddings(
            {"p1": {"title": "Seed", "abstract": "Abstract"}},
            SeededRandomEncodeModel(),
            show_progress=False,
        )
        hydration_cache.mark_hydrated(
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
            dataset_split="train",
            corpus_size=1024,
            complete=True,
        )
        assert hydration_cache.is_hydrated(
            dataset_split="train",
            corpus_size=1024,
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
        )
        reloaded_hydration = EmbeddingCache(
            cache_dir=tmpdir, model_name="hydration-persistence"
        )
        assert reloaded_hydration.is_hydrated(
            dataset_split="train",
            corpus_size=1024,
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
        )

        fingerprint_cache = EmbeddingCache(
            cache_dir=tmpdir, model_name="fingerprint-persistence"
        )
        assert fingerprint_cache.get_model_fingerprint() is None
        fingerprint_cache.set_model_fingerprint("hf::org/model::abc123")
        assert fingerprint_cache.get_model_fingerprint() == "hf::org/model::abc123"
        reloaded_fingerprint = EmbeddingCache(
            cache_dir=tmpdir, model_name="fingerprint-persistence"
        )
        assert reloaded_fingerprint.get_model_fingerprint() == "hf::org/model::abc123"
        with reloaded_fingerprint._connect_db() as conn:
            metadata = {
                key: value
                for key, value in conn.execute("SELECT key, value FROM cache_metadata")
            }
        assert metadata[MODEL_FINGERPRINT_KEY] == "hf::org/model::abc123"

        cache = EmbeddingCache(cache_dir=tmpdir, model_name="payload-presence")
        assert not cache.has_cached_payload()
        _set_test_int8_calibration(cache)
        cache.get_embeddings(
            {"p1": {"title": "Seed", "abstract": "Abstract"}},
            SeededRandomEncodeModel(),
            show_progress=False,
        )
        assert cache.has_cached_payload()


def test_embedding_cache_get_cached_paper_ids_contract() -> None:
    """Cached paper ID listing should reflect persisted SQLite rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="paper-id-listing")
        assert cache.get_cached_paper_ids() == set()
        _set_test_int8_calibration(cache)
        cache.get_embeddings(
            {
                "p1": {"title": "Seed 1", "abstract": "A"},
                "p2": {"title": "Seed 2", "abstract": "B"},
            },
            SeededRandomEncodeModel(),
            show_progress=False,
        )
        assert cache.get_cached_paper_ids() == {"p1", "p2"}


def test_embedding_cache_hydration_rowcount_reconciliation_marker_contract() -> None:
    """Row-count reconciliation marker metadata should persist and reset cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="rowcount-marker")
        assert cache.get_hydration_rowcount_reconciliation() is None

        cache.set_hydration_rowcount_reconciliation(upstream_rows=110, cached_rows=100)
        assert cache.get_hydration_rowcount_reconciliation() == (110, 100)

        cache.clear_hydration_rowcount_reconciliation()
        assert cache.get_hydration_rowcount_reconciliation() is None

        cache.set_hydration_rowcount_reconciliation(upstream_rows=111, cached_rows=101)
        cache.mark_hydrated(
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
            dataset_split="train",
            corpus_size=None,
            complete=True,
        )
        assert cache.get_hydration_rowcount_reconciliation() is None


def test_embedding_cache_hydration_validation_contracts() -> None:
    """Hydration validity should fail closed when payload or metadata integrity drifts."""

    source = "librarian-bots/arxiv-metadata-snapshot"

    def _remove_h5_payload(cache: EmbeddingCache) -> None:
        cache.h5_path.unlink(missing_ok=True)

    def _delete_metadata_rows(cache: EmbeddingCache) -> None:
        with cache._connect_db() as conn:
            conn.execute("DELETE FROM papers")
            conn.commit()

    def _clear_dataset_source(cache: EmbeddingCache) -> None:
        with cache._connect_db() as conn:
            conn.execute(
                "UPDATE cache_metadata SET value = '' WHERE key = ?",
                (HYDRATION_DATASET_SOURCE_KEY,),
            )
            conn.commit()

    cases = [
        {
            "label": "missing h5 payload",
            "model_name": "hydration-payload",
            "corpus_size": 512,
            "invalidate": _remove_h5_payload,
            "include_source_arg_after_invalidation": True,
        },
        {
            "label": "orphaned embedding metadata rows",
            "model_name": "hydration-metadata-rows",
            "corpus_size": 256,
            "invalidate": _delete_metadata_rows,
            "include_source_arg_after_invalidation": True,
        },
        {
            "label": "missing dataset source metadata",
            "model_name": "hydration-source-required",
            "corpus_size": 128,
            "invalidate": _clear_dataset_source,
            "include_source_arg_after_invalidation": False,
        },
    ]

    for case in cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=tmpdir, model_name=case["model_name"])
            _set_test_int8_calibration(cache)
            cache.get_embeddings(
                {"p1": {"title": "Seed", "abstract": "Abstract"}},
                SeededRandomEncodeModel(),
                show_progress=False,
            )
            cache.mark_hydrated(
                dataset_source=source,
                dataset_split="train",
                corpus_size=case["corpus_size"],
                complete=True,
            )
            assert cache.is_hydrated(
                dataset_split="train",
                corpus_size=case["corpus_size"],
                dataset_source=source,
            ), case["label"]

            case["invalidate"](cache)

            hydrated_kwargs: dict[str, Any] = {
                "dataset_split": "train",
                "corpus_size": case["corpus_size"],
            }
            if case["include_source_arg_after_invalidation"]:
                hydrated_kwargs["dataset_source"] = source
            if case["label"] == "orphaned embedding metadata rows":
                with pytest.raises(RuntimeError, match="cache files were preserved"):
                    cache.is_hydrated(**hydrated_kwargs)
            else:
                assert not cache.is_hydrated(**hydrated_kwargs), case["label"]


def test_embedding_cache_mark_hydrated_rejects_empty_source_when_complete() -> None:
    """Complete hydration markers should reject empty dataset source tokens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="hydration-empty-source")
        with pytest.raises(ValueError, match="dataset_source must be non-empty"):
            cache.mark_hydrated(
                dataset_source="  ",
                dataset_split="train",
                corpus_size=16,
                complete=True,
            )


def test_embedding_cache_recovery_clears_hydration_metadata() -> None:
    """Invalid HDF5 layout should reset all hydration markers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="layout-recovery-metadata")
        cache.mark_hydrated(
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
            dataset_split="train",
            corpus_size=2048,
            complete=True,
        )
        with h5py.File(cache.h5_path, "w") as h5:
            h5.create_dataset(
                "legacy_payload", data=np.array([1, 2, 3], dtype=np.float32)
            )

        cache = EmbeddingCache(cache_dir=tmpdir, model_name="layout-recovery-metadata")

        assert not cache.is_hydrated(
            dataset_split="train",
            corpus_size=2048,
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
        )
        with cache._connect_db() as conn:
            cursor = conn.cursor()
            metadata = {
                key: value
                for key, value in cursor.execute(
                    "SELECT key, value FROM cache_metadata"
                )
            }

        assert metadata[HYDRATION_COMPLETE_KEY] == "0"
        assert metadata[HYDRATION_DATASET_SOURCE_KEY] == ""
        assert metadata[HYDRATION_SPLIT_KEY] == ""
        assert metadata[HYDRATION_CORPUS_SIZE_KEY] == ""


def test_embedding_cache_recovery_when_h5_missing_clears_stale_sqlite_rows() -> None:
    """Missing HDF5 payload should clear stale SQLite rows and hydration metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="missing-h5-stale-db")
        _set_test_int8_calibration(cache)
        cache.get_embeddings(
            {"p1": {"title": "Seed", "abstract": "Abstract"}},
            SeededRandomEncodeModel(),
            show_progress=False,
        )
        cache.mark_hydrated(
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
            dataset_split="train",
            corpus_size=64,
            complete=True,
        )
        cache.h5_path.unlink(missing_ok=True)

        reloaded = EmbeddingCache(cache_dir=tmpdir, model_name="missing-h5-stale-db")
        with reloaded._connect_db() as conn:
            paper_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            metadata = {
                key: value
                for key, value in conn.execute("SELECT key, value FROM cache_metadata")
            }

    assert paper_count == 0
    assert metadata[HYDRATION_COMPLETE_KEY] == "0"
    assert metadata[HYDRATION_DATASET_SOURCE_KEY] == ""
    assert metadata[HYDRATION_SPLIT_KEY] == ""
    assert metadata[HYDRATION_CORPUS_SIZE_KEY] == ""


def test_embedding_cache_reopen_keeps_namespace_after_failed_first_encode(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The datasetless file a failed first encode leaves is an empty cache.

    :param Path tmp_path: Isolated cache directory.
    :param pytest.LogCaptureFixture caplog: Captures any layout-rebuild warning.
    :return None: Checks the namespace survives and stays writable after reopen.
    """
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        model_name="failed-first-encode",
        storage_precision="float32",
    )
    cache.set_model_fingerprint("fingerprint-1")
    cache.mark_corpus_metadata_current()
    papers = {"p1": {"title": "Seed", "abstract": "Abstract"}}
    with pytest.raises(KeyError):
        cache.get_embeddings(papers, LookupEncodeModel({}), show_progress=False)

    assert cache.h5_path.exists()
    with h5py.File(cache.h5_path, "r") as h5:
        assert len(h5) == 0
        assert len(h5.attrs) == 0

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        reopened = EmbeddingCache(
            cache_dir=tmp_path,
            model_name="failed-first-encode",
            storage_precision="float32",
        )

    assert "incompatible with current schema" not in caplog.text
    assert reopened.get_model_fingerprint() == "fingerprint-1"
    assert reopened.has_current_corpus_metadata()
    reopened.get_embeddings(
        papers,
        LookupEncodeModel({"Seed. Abstract": np.asarray([1.0, 0.0], dtype=np.float32)}),
        show_progress=False,
    )
    assert reopened.embedding_count() == 1


def test_embedding_cache_reopen_rebuilds_datasetless_file_with_schema_attrs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Losing the matrix from a stamped namespace is still a proven mismatch.

    :param Path tmp_path: Isolated cache directory.
    :param pytest.LogCaptureFixture caplog: Captures the layout-rebuild warning.
    :return None: Checks schema attrs without vectors keep taking the repair path.
    """
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        model_name="datasetless-with-attrs",
        storage_precision="float32",
    )
    cache.mark_corpus_metadata_current()
    cache.get_embeddings(
        {"p1": {"title": "Seed", "abstract": "Abstract"}},
        LookupEncodeModel({"Seed. Abstract": np.asarray([1.0, 0.0], dtype=np.float32)}),
        show_progress=False,
    )
    with h5py.File(cache.h5_path, "a") as h5:
        del h5[EMBEDDINGS_DATASET_NAME]
        assert len(h5.attrs) > 0

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        reopened = EmbeddingCache(
            cache_dir=tmp_path,
            model_name="datasetless-with-attrs",
            storage_precision="float32",
        )

    assert "incompatible with current schema" in caplog.text
    assert not reopened.h5_path.exists()
    assert not reopened.has_current_corpus_metadata()
    with reopened._connect_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0


@pytest.mark.parametrize("binary_rows", [1, 3])
def test_embedding_cache_search_falls_back_when_binary_index_rows_mismatch(
    binary_rows: int,
) -> None:
    """Search should fall back to direct int8 scoring when binary rows mismatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(
            cache_dir=tmpdir, model_name=f"binary-row-mismatch-{binary_rows}"
        )
        _set_test_int8_calibration(cache)
        lookup = LookupEncodeModel(
            {
                "Alpha. First": np.array([1.0, 0.0], dtype=np.float32),
                "Beta. Second": np.array([0.0, 1.0], dtype=np.float32),
            }
        )
        cache.get_embeddings(
            {
                "p1": {"title": "Alpha", "abstract": "First"},
                "p2": {"title": "Beta", "abstract": "Second"},
            },
            lookup,
            show_progress=False,
        )
        with h5py.File(cache.h5_path, "a") as h5:
            binary = h5["binary_index"]
            binary.resize((binary_rows, binary.shape[1]))
            if binary_rows > 2:
                binary[2:binary_rows] = 0

        results = cache.search(
            query_embedding=np.asarray([0.0, 1.0], dtype=np.float32),
            top_k=2,
            binary_prefilter=True,
            binary_rescore_multiplier=4,
        )

    assert len(results) == 2
    assert {result.paper_id for result in results} == {"p1", "p2"}


def test_embedding_cache_reload_rebuilds_stale_binary_index_before_next_write() -> None:
    """Reload should rebuild a stale binary index before appending more rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="binary-dtype-mismatch")
        _set_test_int8_calibration(cache)
        cache.get_embeddings(
            {
                "p0": {"title": "Bad", "abstract": "Opposite"},
                "p1": {"title": "Good", "abstract": "Match"},
            },
            LookupEncodeModel(
                {
                    "Bad. Opposite": np.asarray([-1.0, 0.0], dtype=np.float32),
                    "Good. Match": np.asarray([1.0, 0.0], dtype=np.float32),
                }
            ),
            show_progress=False,
        )
        with h5py.File(cache.h5_path, "a") as h5:
            binary_shape = h5[BINARY_INDEX_DATASET_NAME].shape
            del h5[BINARY_INDEX_DATASET_NAME]
            h5.create_dataset(
                BINARY_INDEX_DATASET_NAME,
                data=np.zeros(binary_shape, dtype=np.float32),
            )

        reloaded = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="binary-dtype-mismatch",
        )
        with h5py.File(reloaded.h5_path, "r") as h5:
            binary = h5[BINARY_INDEX_DATASET_NAME]
            assert binary.shape == (2, 1)
            assert binary.dtype == np.uint8

        reloaded.get_embeddings(
            {"p2": {"title": "Other", "abstract": "Orthogonal"}},
            LookupEncodeModel(
                {"Other. Orthogonal": np.asarray([0.0, 1.0], dtype=np.float32)}
            ),
            show_progress=False,
        )
        results = reloaded.search(
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=1,
            binary_prefilter=True,
            binary_rescore_multiplier=1,
        )

    assert [result.paper_id for result in results] == ["p1"]
    assert reloaded.last_search_used_binary_prefilter is True


@pytest.mark.parametrize("index_state", ["missing", "legacy"])
def test_embedding_cache_binary_rebuild_preserves_signs_and_prefilter_results(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, index_state: str
) -> None:
    """Binary writes and rebuilds must use the same persisted-vector signs.

    :param Path tmp_path: Isolated cache directory.
    :param pytest.LogCaptureFixture caplog: Captures one-time index rebuild messages.
    :param str index_state: Missing index or old raw-float sign encoding.
    :return None: Checks index bytes and selected neighbors around the zero bucket.
    """
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="binary-rebuild-signs")
    _set_test_int8_calibration(cache)
    cache.get_embeddings(
        {
            "p1": {"title": "Near zero", "abstract": ""},
            "p2": {"title": "Positive", "abstract": ""},
        },
        LookupEncodeModel(
            {
                "Near zero": np.asarray([0.001, 1.0], dtype=np.float32),
                "Positive": np.asarray([0.01, 1.0], dtype=np.float32),
            }
        ),
        show_progress=False,
    )
    search_args = {
        "query_embedding": np.asarray([0.01, 1.0], dtype=np.float32),
        "top_k": 1,
        "binary_prefilter": True,
        "binary_rescore_multiplier": 1,
    }
    original_results = cache.search(**search_args)
    cache.mark_hydrated(
        dataset_source="test-corpus",
        dataset_split="train",
        corpus_size=None,
        complete=True,
    )
    with h5py.File(cache.h5_path, "a") as h5:
        original_bits = h5[BINARY_INDEX_DATASET_NAME][:]
        original_vectors = h5[EMBEDDINGS_DATASET_NAME][:]
        if index_state == "legacy":
            binary = h5[BINARY_INDEX_DATASET_NAME]
            binary[...] = np.asarray([[192], [192]], dtype=np.uint8)
            del binary.attrs[BINARY_INDEX_ENCODING_KEY]
            assert not np.array_equal(binary[:], original_bits)
        else:
            del h5[BINARY_INDEX_DATASET_NAME]

    reloaded = EmbeddingCache(cache_dir=tmp_path, model_name="binary-rebuild-signs")
    with h5py.File(reloaded.h5_path, "r") as h5:
        np.testing.assert_array_equal(h5[BINARY_INDEX_DATASET_NAME][:], original_bits)
        np.testing.assert_array_equal(h5[EMBEDDINGS_DATASET_NAME][:], original_vectors)
        assert (
            h5[BINARY_INDEX_DATASET_NAME].attrs[BINARY_INDEX_ENCODING_KEY]
            == BINARY_INDEX_ENCODING
        )
    assert reloaded.get_cached_paper_ids() == {"p1", "p2"}
    assert reloaded.is_hydrated(
        dataset_source="test-corpus", dataset_split="train", corpus_size=None
    )
    reloaded_results = reloaded.search(**search_args)
    assert [(result.paper_id, result.score) for result in reloaded_results] == [
        (result.paper_id, result.score) for result in original_results
    ]
    assert [result.paper_id for result in original_results] == ["p2"]
    caplog.clear()
    EmbeddingCache(cache_dir=tmp_path, model_name="binary-rebuild-signs")
    assert "Rebuilding binary index" not in caplog.text
    with h5py.File(reloaded.h5_path, "a") as h5:
        del h5[BINARY_INDEX_DATASET_NAME]
    rebuilt = EmbeddingCache(cache_dir=tmp_path, model_name="binary-rebuild-signs")
    assert [result.paper_id for result in rebuilt.search(**search_args)] == ["p2"]


def test_embedding_cache_enabling_binary_prefilter_preserves_populated_namespace(
    tmp_path: Path,
) -> None:
    """Enabling the auxiliary binary index should preserve and index existing rows."""
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        model_name="enable-binary-prefilter",
        binary_prefilter=False,
    )
    _set_test_int8_calibration(cache)
    cache.get_embeddings(
        {
            "p0": {"title": "Bad", "abstract": "Opposite"},
            "p1": {"title": "Good", "abstract": "Match"},
        },
        LookupEncodeModel(
            {
                "Bad. Opposite": np.asarray([-1.0, 0.0], dtype=np.float32),
                "Good. Match": np.asarray([1.0, 0.0], dtype=np.float32),
            }
        ),
        show_progress=False,
    )
    with h5py.File(cache.h5_path, "r") as h5:
        assert BINARY_INDEX_DATASET_NAME not in h5

    reloaded = EmbeddingCache(
        cache_dir=tmp_path,
        model_name="enable-binary-prefilter",
        binary_prefilter=True,
    )
    results = reloaded.search(
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=1,
        binary_prefilter=True,
        binary_rescore_multiplier=1,
    )

    assert reloaded.embedding_count() == 2
    assert [result.paper_id for result in results] == ["p1"]
    assert reloaded.last_search_used_binary_prefilter is True


def test_embedding_cache_serializes_multiprocess_writes(tmp_path: Path) -> None:
    """Concurrent processes should serialize writes without HDF5 lock failures."""
    queue: mp.Queue = mp.Queue()
    processes = [
        mp.Process(target=_multiprocess_cache_worker, args=(str(tmp_path), idx, queue))
        for idx in range(4)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join()

    results = []
    for _ in processes:
        try:
            results.append(queue.get(timeout=5))
        except Empty:
            results.append(("err", "missing", "worker did not report result"))

    errors = [result for result in results if result[0] == "err"]
    assert not errors, f"Concurrent cache writes failed: {errors}"


def test_embedding_cache_serializes_multiprocess_initialization_recovery(
    tmp_path: Path,
) -> None:
    """Concurrent init/recovery should not race when repairing stale namespace state."""
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="process-init-recovery")
    with cache._connect_db() as conn:
        conn.execute(
            """
            INSERT INTO papers (paper_id, title, abstract, year, text_hash, embedding_dim, row_idx)
            VALUES ('seed', 'seed', '', NULL, 'hash', 2, 0)
            """
        )
        conn.commit()
    cache.h5_path.unlink(missing_ok=True)

    queue: mp.Queue = mp.Queue()
    processes = [
        mp.Process(target=_multiprocess_cache_init_worker, args=(str(tmp_path), queue))
        for _ in range(4)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join()

    results = []
    for _ in processes:
        try:
            results.append(queue.get(timeout=5))
        except Empty:
            results.append(("err", "worker did not report result"))

    errors = [result for result in results if result[0] == "err"]
    assert not errors, f"Concurrent cache init recovery failed: {errors}"

    reloaded = EmbeddingCache(cache_dir=tmp_path, model_name="process-init-recovery")
    with reloaded._connect_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0


def test_embedding_cache_recovery_contracts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Legacy schema recovery and clear() should restore a healthy namespace."""
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="legacy-recovery")
    _set_test_int8_calibration(cache)
    model = SeededRandomEncodeModel()

    cache.h5_path.unlink(missing_ok=True)
    with h5py.File(cache.h5_path, "w") as h5:
        h5.create_dataset("legacy_payload", data=np.array([1, 2, 3], dtype=np.float32))

    with cache._connect_db() as conn:
        conn.execute(
            """
            INSERT INTO papers (paper_id, title, abstract, year, text_hash, embedding_dim, row_idx)
            VALUES ('seed', 'seed', '', NULL, 'hash', 3, 2)
            """
        )
        conn.commit()

    reloaded = EmbeddingCache(cache_dir=tmp_path, model_name="legacy-recovery")
    assert "REBUILDING EMBEDDING CACHE" in caplog.text
    assert "Reason: incompatible embedding cache layout" in caplog.text
    with reloaded._connect_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0

    _set_test_int8_calibration(reloaded)
    reloaded.get_embeddings(
        {"seed": {"title": "Seed", "abstract": "x", "year": None}}, model
    )
    with h5py.File(reloaded.h5_path, "r") as h5:
        assert "embeddings" in h5
        assert h5["embeddings"].shape[0] == 1

    reloaded.clear()
    assert reloaded.db_path.exists()
    assert not reloaded.h5_path.exists()


@pytest.mark.parametrize(
    "error_type", [OSError, RuntimeError, ValueError, sqlite3.DatabaseError]
)
@pytest.mark.parametrize(
    ("operation", "backend"),
    [
        ("open", "hdf5"),
        ("hydration", "hdf5"),
        ("hydration", "sqlite"),
        ("stats", "hdf5"),
        ("stats", "sqlite"),
        ("presence", "sqlite"),
        ("open", "stat-hdf5"),
        ("hydration", "stat-hdf5"),
        ("stats", "stat-hdf5"),
        ("presence", "stat-sqlite"),
    ],
)
def test_embedding_cache_open_errors_preserve_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    operation: str,
    backend: str,
) -> None:
    """Storage inspection failures must preserve vectors and SQLite rows.

    :param Path tmp_path: Isolated cache directory.
    :param pytest.MonkeyPatch monkeypatch: Injects the opening failure.
    :param type[Exception] error_type: Failure previously mistaken for corruption.
    :param str operation: Cache opening or state inspection operation.
    :param str backend: Storage backend whose read fails.
    :return None: Checks preserved bytes and successful reuse after the failure.
    """
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="open-failure")
    _set_test_int8_calibration(cache)
    papers = {"p1": {"title": "Seed", "abstract": "Abstract"}}
    model = SeededRandomEncodeModel()
    cache.get_embeddings(papers, model, show_progress=False)
    cache.mark_hydrated(
        dataset_source="fixture/source",
        dataset_split="train",
        corpus_size=1,
        complete=True,
    )
    original_payload = cache.h5_path.read_bytes()
    injected_error = error_type("transient open failure")
    original_stat = Path.stat
    original_exists = Path.exists
    stat_target = cache.db_path if backend == "stat-sqlite" else cache.h5_path

    def fail_open(*args: Any, **kwargs: Any) -> None:
        """Raise the injected failure instead of opening storage.

        :param Any args: Storage positional arguments.
        :param Any kwargs: Storage keyword arguments.
        :return None: Always raises the injected exception.
        """
        raise injected_error

    def fail_stat(path: Path, **kwargs: Any) -> Any:
        """Fail inspection of one payload path.

        :param Path path: Filesystem path under inspection.
        :param Any kwargs: Path.stat keyword arguments.
        :return Any: Real file statistics for unaffected paths.
        """
        if path == stat_target:
            raise injected_error
        return original_stat(path, **kwargs)

    def suppress_exists_error(path: Path, **kwargs: Any) -> bool:
        """Emulate Python 3.14's exists error suppression on every test runtime.

        :param Path path: Filesystem path under inspection.
        :param Any kwargs: Path.exists keyword arguments.
        :return bool: False for the inaccessible target, actual presence otherwise.
        """
        return False if path == stat_target else original_exists(path, **kwargs)

    with monkeypatch.context() as patch:
        if backend.startswith("stat-"):
            patch.setattr(Path, "stat", fail_stat)
            patch.setattr(Path, "exists", suppress_exists_error)
        elif backend == "hdf5":
            patch.setattr(embedding_cache_module.h5py, "File", fail_open)
        else:
            patch.setattr(cache, "_connect_db", fail_open)
        if operation == "open":
            with pytest.raises(error_type, match="transient open failure"):
                EmbeddingCache(cache_dir=tmp_path, model_name="open-failure")
        else:
            with pytest.raises(
                RuntimeError, match="cache files were preserved"
            ) as caught:
                if operation == "hydration":
                    cache.is_hydrated("train", 1, dataset_source="fixture/source")
                elif operation == "stats":
                    cache.payload_stats()
                else:
                    cache.has_cached_payload()
            assert caught.value.__cause__ is injected_error
            assert str(cache.db_path) in str(caught.value)
            assert str(cache.h5_path) in str(caught.value)

    assert cache.h5_path.read_bytes() == original_payload
    assert cache.get_cached_paper_ids() == {"p1"}
    reloaded = EmbeddingCache(cache_dir=tmp_path, model_name="open-failure")
    reloaded.get_embeddings(papers, model, show_progress=False)
    assert model.encode_calls == 1


def test_embedding_cache_readonly_hdf5_handle_preserves_namespace(
    tmp_path: Path,
) -> None:
    """A real read-only HDF5 handle must not cause namespace deletion.

    :param Path tmp_path: Isolated cache directory.
    :return None: Reopens and reuses the namespace after releasing the reader.
    """
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="readonly-hdf5")
    _set_test_int8_calibration(cache)
    cache.get_embeddings(
        {"p1": {"title": "Seed", "abstract": "Abstract"}},
        SeededRandomEncodeModel(),
        show_progress=False,
    )
    with h5py.File(cache.h5_path, "r"):
        with pytest.raises(OSError):
            EmbeddingCache(cache_dir=tmp_path, model_name="readonly-hdf5")
        assert cache.h5_path.exists()
        assert cache.get_cached_paper_ids() == {"p1"}

    reloaded = EmbeddingCache(cache_dir=tmp_path, model_name="readonly-hdf5")
    assert reloaded.embedding_count() == 1


def test_embedding_cache_reload_preserves_calibration_before_first_write(
    tmp_path: Path,
) -> None:
    """A calibrated namespace without embeddings is a valid resumable state.

    :param Path tmp_path: Isolated cache directory.
    :return None: Checks that persisted ranges survive and support the first write.
    """
    cache = EmbeddingCache(
        cache_dir=tmp_path, model_name="calibration-only", compression_level=9
    )
    ranges = np.asarray([[-0.75, -0.5], [0.75, 0.5]], dtype=np.float32)
    cache.set_calibration_ranges(ranges, embedding_dim=2)

    reloaded = EmbeddingCache(
        cache_dir=tmp_path,
        model_name="calibration-only",
        compression_level=1,
        binary_prefilter=False,
    )
    with h5py.File(reloaded.h5_path, "r") as h5:
        np.testing.assert_array_equal(h5["calibration_ranges"][:], ranges)
    reloaded.get_embeddings(
        {"p1": {"title": "Seed", "abstract": "Abstract"}},
        SeededRandomEncodeModel(),
        show_progress=False,
    )
    assert reloaded.embedding_count() == 1
    results = reloaded.search(
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=1,
        binary_prefilter=False,
        binary_rescore_multiplier=1,
    )
    assert [result.paper_id for result in results] == ["p1"]


@pytest.mark.parametrize(
    "changed_setting, changed_value",
    [
        ("text_formatter_fingerprint", "new"),
        ("calibration_sample_size", 2000),
        ("source_torch_dtype", "bfloat16"),
    ],
)
def test_embedding_cache_calibration_only_rejects_changed_runtime_contract(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    changed_setting: str,
    changed_value: object,
) -> None:
    """Calibration-only recovery must validate the runtime that produced ranges.

    :param Path tmp_path: Isolated cache directory.
    :param pytest.LogCaptureFixture caplog: Captures the proven mismatch reason.
    :param str changed_setting: Runtime identity field changed on reopen.
    :param object changed_value: Incompatible value for that field.
    :return None: Checks old ranges cannot silently acquire new provenance.
    """
    settings = {
        "text_formatter_fingerprint": "old",
        "calibration_sample_size": 10,
        "source_torch_dtype": "float32",
    }
    cache = EmbeddingCache(
        cache_dir=tmp_path, model_name="calibration-contract", **settings
    )
    _set_test_int8_calibration(cache)

    settings[changed_setting] = changed_value
    reloaded = EmbeddingCache(
        cache_dir=tmp_path, model_name="calibration-contract", **settings
    )

    assert not reloaded.has_calibration_ranges()
    assert not reloaded.h5_path.exists()
    assert changed_setting in caplog.text
    with pytest.raises(RuntimeError, match="Missing persisted int8 calibration ranges"):
        reloaded.get_embeddings(
            {"p1": {"title": "Seed", "abstract": "Abstract"}},
            SeededRandomEncodeModel(),
            show_progress=False,
        )


def test_embedding_cache_detects_diverged_sqlite_contract_value(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Stored contract values must be compared on reopen, not silently restamped.

    :param Path tmp_path: Isolated cache directory.
    :param pytest.LogCaptureFixture caplog: Captures the proven mismatch reason.
    :return None: Checks a diverged SQLite contract value reaches the repair path.
    """
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        model_name="sqlite-contract-divergence",
        storage_precision="float32",
    )
    cache.get_embeddings(
        {"p1": {"title": "Seed", "abstract": "Abstract"}},
        LookupEncodeModel({"Seed. Abstract": np.asarray([1.0, 0.0], dtype=np.float32)}),
        show_progress=False,
    )
    with cache._connect_db() as conn:
        conn.execute(
            "UPDATE cache_metadata SET value = ? WHERE key = ?",
            ("matrix-v1-legacy", H5_LAYOUT_KEY),
        )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        reopened = EmbeddingCache(
            cache_dir=tmp_path,
            model_name="sqlite-contract-divergence",
            storage_precision="float32",
        )

    assert "incompatible with current schema" in caplog.text
    assert H5_LAYOUT_KEY in caplog.text
    assert reopened.embedding_count() == 0
    with reopened._connect_db() as conn:
        stored_layout = conn.execute(
            "SELECT value FROM cache_metadata WHERE key = ?", (H5_LAYOUT_KEY,)
        ).fetchone()[0]
    assert stored_layout == H5_LAYOUT_MATRIX_VERSION


@pytest.mark.parametrize("persisted_rows", [0, 1])
def test_embedding_cache_recovery_preserves_sqlite_ahead_prefix(
    tmp_path: Path, persisted_rows: int
) -> None:
    """SQLite-ahead recovery discards only mappings whose vectors were lost.

    :param Path tmp_path: Isolated cache directory.
    :param int persisted_rows: Number of vectors surviving the interrupted append.
    :return None: Checks surviving vectors, incomplete hydration, and resumed writes.
    """
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="sqlite-ahead")
    _set_test_int8_calibration(cache)
    papers = {
        "p1": {"title": "Seed", "abstract": "Abstract"},
        "p2": {"title": "Other", "abstract": "Abstract"},
    }
    model = SeededRandomEncodeModel()
    cache.get_embeddings(papers, model, show_progress=False)
    cache.mark_hydrated(
        dataset_source="test-corpus",
        dataset_split="train",
        corpus_size=None,
        complete=True,
    )
    with h5py.File(cache.h5_path, "a") as h5:
        h5[EMBEDDINGS_DATASET_NAME].resize((persisted_rows, 2))
        surviving = h5[EMBEDDINGS_DATASET_NAME][:]

    reloaded = EmbeddingCache(cache_dir=tmp_path, model_name="sqlite-ahead")
    assert reloaded.embedding_count() == persisted_rows
    assert reloaded.get_cached_paper_ids() == ({"p1"} if persisted_rows else set())
    with reloaded._connect_db() as conn:
        metadata = reloaded._load_cache_metadata(conn)
    assert metadata[HYDRATION_COMPLETE_KEY] == "0"
    assert metadata[HYDRATION_DATASET_SOURCE_KEY] == "test-corpus"
    with h5py.File(reloaded.h5_path, "r") as h5:
        np.testing.assert_array_equal(h5[EMBEDDINGS_DATASET_NAME][:], surviving)
        assert h5[BINARY_INDEX_DATASET_NAME].shape[0] == persisted_rows

    reloaded.get_embeddings(papers, model, show_progress=False)
    assert reloaded.get_cached_paper_ids() == {"p1", "p2"}
    assert reloaded.embedding_count() == 2


@pytest.mark.parametrize("duplicate_interior_row", [False, True])
def test_embedding_cache_reload_preserves_unrecoverable_row_mapping(
    tmp_path: Path,
    duplicate_interior_row: bool,
) -> None:
    """An ambiguous mapping must fail without deleting recoverable vector data.

    :param Path tmp_path: Isolated cache directory.
    :param bool duplicate_interior_row: Corrupt coverage while keeping valid bounds.
    :return None: Checks that a gap does not trigger a namespace wipe.
    """
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="mapping-gap")
    _set_test_int8_calibration(cache)
    papers = {
        paper_id: {"title": "Seed", "abstract": "Abstract"}
        for paper_id in (["p1", "p2", "p3"] if duplicate_interior_row else ["p1"])
    }
    cache.get_embeddings(
        papers,
        SeededRandomEncodeModel(),
        show_progress=False,
    )
    with cache._connect_db() as conn:
        if duplicate_interior_row:
            conn.execute("UPDATE papers SET row_idx = 0 WHERE paper_id = 'p2'")
        else:
            conn.execute("UPDATE papers SET row_idx = 1")

    with pytest.raises(
        RuntimeError, match="row.*mapping mismatch|row_idx coverage mismatch"
    ):
        cache.get_embeddings(
            {"p1": papers["p1"]}, SeededRandomEncodeModel(), show_progress=False
        )

    with pytest.raises(RuntimeError, match="row_idx coverage mismatch"):
        EmbeddingCache(cache_dir=tmp_path, model_name="mapping-gap")
    assert cache.h5_path.exists()
    assert cache.get_cached_paper_ids() == set(papers)


@pytest.mark.parametrize("operation", ["hit", "append", "search"])
def test_embedding_cache_operation_sql_cost_does_not_scan_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """A small cache operation must not scan unrelated metadata as the corpus grows.

    :param Path tmp_path: Isolated persistent cache.
    :param pytest.MonkeyPatch monkeypatch: Instruments SQLite execution cost.
    :param str operation: Public cache operation whose SQL cost is measured.
    :return None: A 64-fold corpus increase leaves metadata work bounded.
    """
    cache = EmbeddingCache(cache_dir=tmp_path, storage_precision="float32")
    model = SeededRandomEncodeModel()
    connect = cache._connect_db
    steps = 0

    def count_steps() -> int:
        """Count SQLite work in fixed instruction blocks.

        :return int: Zero allows SQLite execution to continue.
        """
        nonlocal steps
        steps += 1
        return 0

    @contextmanager
    def measured_connection() -> Iterator[sqlite3.Connection]:
        """Instrument connections used by the public cache operations.

        :return Iterator[sqlite3.Connection]: Connection counting every 100 VM steps.
        """
        with connect() as conn:
            conn.set_progress_handler(count_steps, 100)
            yield conn

    monkeypatch.setattr(cache, "_connect_db", measured_connection)
    costs = []
    previous = 0
    for size in (128, 8192):
        cache.upsert_embeddings(
            {f"p{i}": {"title": f"Title {i}"} for i in range(previous, size)},
            model,
            show_progress=False,
        )
        steps = 0
        if operation == "hit":
            assert "p0" in cache.get_embeddings(
                {"p0": {"title": "Title 0"}}, model, show_progress=False
            )
        elif operation == "append":
            cache.upsert_embeddings(
                {f"new-{size}": {"title": "New paper"}}, model, show_progress=False
            )
        else:
            assert cache.search(
                np.array([1.0, 0.0], dtype=np.float32),
                top_k=1,
                binary_prefilter=False,
                binary_rescore_multiplier=8,
            )
        costs.append(steps)
        previous = size
    assert costs[1] <= costs[0] + 20, costs


def test_embedding_cache_recovery_truncates_orphan_h5_rows(tmp_path: Path) -> None:
    """Reload should truncate uncommitted HDF5 rows and preserve committed mappings."""
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="orphan-h5-row-recovery")
    _set_test_int8_calibration(cache)
    cache.get_embeddings(
        {
            "seed": {"title": "Seed", "abstract": "x"},
            "other": {"title": "Other", "abstract": "y"},
        },
        LookupEncodeModel(
            {
                "Seed. x": np.asarray([1.0, 0.0], dtype=np.float32),
                "Other. y": np.asarray([0.0, 1.0], dtype=np.float32),
            }
        ),
        show_progress=False,
    )
    with h5py.File(cache.h5_path, "a") as h5:
        embeddings = h5["embeddings"]
        binary = h5["binary_index"]
        current_rows = int(embeddings.shape[0])
        embeddings.resize((current_rows + 1, int(embeddings.shape[1])))
        embeddings[current_rows] = np.asarray([0, 0], dtype=np.int8)
        binary.resize((current_rows + 1, int(binary.shape[1])))
        binary[current_rows] = np.asarray([0], dtype=np.uint8)

    reloaded = EmbeddingCache(cache_dir=tmp_path, model_name="orphan-h5-row-recovery")
    with reloaded._connect_db() as conn:
        paper_rows = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    with h5py.File(reloaded.h5_path, "r") as h5:
        embedding_rows = int(h5[EMBEDDINGS_DATASET_NAME].shape[0])
        binary_rows = int(h5[BINARY_INDEX_DATASET_NAME].shape[0])
    results = reloaded.search(
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=1,
        binary_prefilter=True,
        binary_rescore_multiplier=2,
    )

    assert paper_rows == 2
    assert embedding_rows == 2
    assert binary_rows == 2
    assert reloaded.has_cached_payload()
    assert [result.paper_id for result in results] == ["seed"]


def test_embedding_cache_clear_releases_file_handles(tmp_path: Path) -> None:
    """clear() must fully release SQLite/HDF5 handles before unlinking.

    On Windows, unclosed sqlite3 connections prevent file deletion with
    ``[WinError 32]``.  This test verifies the clear path does not leak
    handles on any platform.
    """
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="clear-handle-test")
    _set_test_int8_calibration(cache)
    cache.get_embeddings(
        {"p1": {"title": "Test", "abstract": "Abstract"}},
        SeededRandomEncodeModel(),
        show_progress=False,
    )
    assert cache.h5_path.exists()
    assert cache.db_path.exists()

    # Must not raise on any platform (WinError 32 on Windows if handles leak)
    cache.clear(reason="handle release test")

    assert not cache.h5_path.exists()
    # DB is recreated by clear() via _init_db, so it should exist but be empty
    assert cache.db_path.exists()
    with cache._connect_db() as conn:
        paper_rows = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    assert paper_rows == 0


def test_embedding_cache_clear_logs_cached_hydration_scope(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Material cache clears should warn concisely and debug full scope.

    :param Path tmp_path: Pytest temporary directory.
    :param pytest.LogCaptureFixture caplog: Captured logging fixture.
    :return None: Assertions verify warning and debug detail separation.
    """
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="clear-log-scope")
    _set_test_int8_calibration(cache)
    cache.get_embeddings(
        {"p1": {"title": "Test", "abstract": "Abstract"}},
        SeededRandomEncodeModel(),
        show_progress=False,
    )
    cache.mark_hydrated(
        dataset_source="librarian-bots/arxiv-metadata-snapshot",
        dataset_split="train",
        corpus_size=50000,
        complete=True,
    )

    caplog.clear()
    with caplog.at_level("DEBUG"):
        cache.clear(reason="scope test")

    warnings = [
        record.message for record in caplog.records if record.levelno == logging.WARNING
    ]
    debug_messages = [
        record.message for record in caplog.records if record.levelno == logging.DEBUG
    ]
    assert any("1 cached paper(s)" in message for message in warnings)
    assert not any("clear-log-scope" in message for message in warnings)
    assert any(
        "cached_split=train, cached_corpus=newest:50000, "
        "cached_source=librarian-bots/arxiv-metadata-snapshot" in message
        for message in debug_messages
    )


def test_embedding_cache_clear_keeps_empty_namespace_reset_at_debug(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Resetting metadata-only cache files should not emit a warning.

    :param Path tmp_path: Pytest temporary directory.
    :param pytest.LogCaptureFixture caplog: Captured logging fixture.
    :return None: Assertions verify empty namespace resets remain debug-only
        and that an omitted reason is reported as unspecified.
    """
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="empty-clear-log")

    caplog.clear()
    with caplog.at_level("DEBUG"):
        cache.clear(reason="empty scope test")

    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
    assert any(
        "Embedding cache clear details: namespace=empty-clear-log" in record.message
        for record in caplog.records
        if record.levelno == logging.DEBUG
    )

    caplog.clear()
    with caplog.at_level("DEBUG"):
        cache.clear()

    assert any(
        "reason=unspecified" in record.message
        for record in caplog.records
        if record.levelno == logging.DEBUG
    )
