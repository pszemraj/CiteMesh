"""Contract tests for embedding cache layout, reset, and retrieval behavior."""

from __future__ import annotations

import builtins
import multiprocessing as mp
import tempfile
from pathlib import Path
from queue import Empty
from typing import Any

import h5py
import numpy as np
import pytest

import citemesh.data.embedding_cache as embedding_cache_module
from citemesh.data.embedding_cache import (
    BINARY_INDEX_DATASET_NAME,
    CALIBRATION_SAMPLE_SIZE_KEY,
    COMPRESSION_FILTER_KEY,
    COMPRESSION_LEVEL_KEY,
    EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR,
    EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS,
    EMBEDDING_DATASET_CHUNK_ROWS,
    EMBEDDINGS_DATASET_NAME,
    HYDRATION_COMPLETE_KEY,
    HYDRATION_CORPUS_SIZE_KEY,
    HYDRATION_DATASET_SOURCE_KEY,
    HYDRATION_SPLIT_KEY,
    INT8_CLIPPED_VALUE_COUNT_KEY,
    INT8_TOTAL_VALUE_COUNT_KEY,
    MODEL_FINGERPRINT_KEY,
    SOURCE_TORCH_DTYPE_KEY,
    TEXT_FORMATTER_FINGERPRINT_KEY,
    EmbeddingCache,
    _corpus_size_token,
    _resolve_cache_lock_timeout_seconds,
)
from citemesh.data.model_profiles import get_embedding_model_profile
from tests._helpers import LookupEncodeModel, SeededRandomEncodeModel


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
    """Encode phase should run outside the lock and reuse rows inserted mid-flight."""
    monkeypatch.setenv(EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR, "0.05")

    class _RaceEncodeModel:
        """Encode model that inserts the same row through a nested cache write."""

        def __init__(self, cache: EmbeddingCache) -> None:
            self.cache = cache
            self.encode_calls = 0

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            del texts, kwargs
            self.encode_calls += 1
            self.cache.get_embeddings(
                {"p1": {"title": "Alpha", "abstract": "First"}},
                LookupEncodeModel(
                    {"Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32)}
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

    assert model.encode_calls == 1
    assert paper_rows == 1
    assert embedding_rows == 1
    assert row_idx == 0
    np.testing.assert_allclose(embeddings["p1"], np.asarray([1.0, 0.0], np.float32))


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

        with h5py.File(cache.h5_path, "r") as h5:
            assert int(h5.attrs[INT8_CLIPPED_VALUE_COUNT_KEY]) == 4
            assert int(h5.attrs[INT8_TOTAL_VALUE_COUNT_KEY]) == 4


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
    profile = get_embedding_model_profile("sentence-transformers/all-MiniLM-L6-v2")
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


def test_embedding_cache_reload_drops_binary_index_with_wrong_dtype() -> None:
    """Reload should discard a shape-compatible binary index that is not uint8."""
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
            assert BINARY_INDEX_DATASET_NAME not in h5
        results = reloaded.search(
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=1,
            binary_prefilter=True,
            binary_rescore_multiplier=1,
        )

    assert [result.paper_id for result in results] == ["p1"]
    assert reloaded.last_search_used_binary_prefilter is False


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


def test_embedding_cache_recovery_contracts(tmp_path: Path) -> None:
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


def test_embedding_cache_recovery_clears_orphan_h5_rows(tmp_path: Path) -> None:
    """Reload should clear namespace when HDF5 has rows missing SQLite metadata mappings."""
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="orphan-h5-row-recovery")
    _set_test_int8_calibration(cache)
    cache.get_embeddings(
        {"seed": {"title": "Seed", "abstract": "x"}},
        LookupEncodeModel({"Seed. x": np.asarray([1.0, 0.0], dtype=np.float32)}),
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

    assert paper_rows == 0
    assert not reloaded.has_cached_payload()


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
    """clear() logs should describe the cached payload being replaced."""
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

    with caplog.at_level("WARNING"):
        cache.clear(reason="scope test")

    assert any(
        "cached_split=train, cached_corpus=newest:50000, "
        "cached_source=librarian-bots/arxiv-metadata-snapshot" in record.message
        for record in caplog.records
    )
