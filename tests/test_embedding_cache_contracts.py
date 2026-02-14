"""Contract tests for embedding cache layout, reset, and retrieval behavior."""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
import tempfile
from pathlib import Path
from queue import Empty

import h5py
import numpy as np
import pytest

from citemesh.data.embedding_cache import (
    HYDRATION_COMPLETE_KEY,
    HYDRATION_CORPUS_SIZE_KEY,
    HYDRATION_DATASET_SOURCE_KEY,
    HYDRATION_SPLIT_KEY,
    MODEL_FINGERPRINT_KEY,
    EmbeddingCache,
)
from tests._helpers import LookupEncodeModel, SeededRandomEncodeModel


def _multiprocess_cache_worker(
    cache_dir: str, worker_idx: int, queue: mp.Queue
) -> None:
    """Write embeddings in subprocess and report success/failure via queue."""
    try:
        cache = EmbeddingCache(cache_dir=cache_dir, model_name="process-lock-test")
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

        with sqlite3.connect(cache.db_path) as conn:
            rows = conn.execute(
                "SELECT paper_id, row_idx FROM papers ORDER BY row_idx"
            ).fetchall()
            row_idx_after_update = conn.execute(
                "SELECT row_idx FROM papers WHERE paper_id = 'p1'"
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

    assert model.encode_calls == 3
    assert first["p1"].shape == second["p1"].shape
    assert rows == [("p1", 0), ("p2", 1)]
    assert row_idx_after_update == 0


def test_embedding_cache_search_and_calibration_reuse_contract() -> None:
    """Search should return metadata and reuse calibration ranges in one namespace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="search-metadata")
        papers = {
            "p1": {
                "title": "Alpha",
                "abstract": "First",
                "year": 2020,
                "authors": ["Alice", "Bob"],
                "categories": ["cs.AI"],
            },
            "p2": {
                "title": "Beta",
                "abstract": "Second",
                "year": 2021,
                "authors": ["Carol"],
                "categories": ["cs.LG"],
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
    assert results[0].metadata["authors"] == ["Alice", "Bob"]
    assert results[0].metadata["categories"] == ["cs.AI"]
    assert results[0].metadata["year"] == 2020
    assert results[0].embedding.dtype == np.float32
    assert results[0].embedding_dtype == "float32"
    assert results[0].storage_precision == "int8"


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


def test_embedding_cache_search_raises_on_missing_metadata_rows() -> None:
    """Search should fail closed when scored rows have no metadata payload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="missing-search-metadata")
        cache.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            LookupEncodeModel({"Alpha. First": np.array([1.0, 0.0], dtype=np.float32)}),
            show_progress=False,
        )

        with sqlite3.connect(cache.db_path) as conn:
            conn.execute("DELETE FROM papers")
            conn.commit()

        with pytest.raises(
            RuntimeError,
            match="Embedding cache integrity error: missing metadata rows",
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


def test_embedding_cache_search_fails_closed_on_metadata_dtype_mismatch() -> None:
    """Search should fail closed when metadata precision diverges from payload dtype."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="search-metadata-mismatch")
        cache.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            LookupEncodeModel({"Alpha. First": np.array([1.0, 0.0], dtype=np.float32)}),
            show_progress=False,
        )

        with sqlite3.connect(cache.db_path) as conn:
            conn.execute(
                "UPDATE cache_metadata SET value = 'float16' WHERE key = 'storage_precision'"
            )
            conn.commit()

        with pytest.raises(
            RuntimeError,
            match="metadata key 'storage_precision' mismatch",
        ):
            cache.search(
                query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                top_k=1,
                binary_prefilter=True,
                binary_rescore_multiplier=2,
            )


def test_embedding_cache_preserves_hydration_metadata_across_restarts() -> None:
    """Hydration completion should survive cache re-open in same namespace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="hydration-persistence")
        cache.get_embeddings(
            {"p1": {"title": "Seed", "abstract": "Abstract"}},
            SeededRandomEncodeModel(),
            show_progress=False,
        )
        cache.mark_hydrated(
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
            dataset_split="train",
            corpus_size=1024,
            complete=True,
        )
        assert cache.is_hydrated(
            dataset_split="train",
            corpus_size=1024,
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
        )

        reloaded = EmbeddingCache(cache_dir=tmpdir, model_name="hydration-persistence")
        assert reloaded.is_hydrated(
            dataset_split="train",
            corpus_size=1024,
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
        )


def test_embedding_cache_model_fingerprint_persists_across_restarts() -> None:
    """Model fingerprint metadata should persist and be queryable across restarts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="fingerprint-persistence")
        assert cache.get_model_fingerprint() is None
        cache.set_model_fingerprint("hf::org/model::abc123")
        assert cache.get_model_fingerprint() == "hf::org/model::abc123"

        reloaded = EmbeddingCache(
            cache_dir=tmpdir, model_name="fingerprint-persistence"
        )
        assert reloaded.get_model_fingerprint() == "hf::org/model::abc123"
        with sqlite3.connect(reloaded.db_path) as conn:
            metadata = {
                key: value
                for key, value in conn.execute("SELECT key, value FROM cache_metadata")
            }

    assert metadata[MODEL_FINGERPRINT_KEY] == "hf::org/model::abc123"


def test_embedding_cache_has_cached_payload_contract() -> None:
    """Payload indicator should reflect whether namespace contains embedding rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="payload-presence")
        assert not cache.has_cached_payload()
        cache.get_embeddings(
            {"p1": {"title": "Seed", "abstract": "Abstract"}},
            SeededRandomEncodeModel(),
            show_progress=False,
        )
        assert cache.has_cached_payload()


def test_embedding_cache_hydration_requires_h5_payload() -> None:
    """Hydration should be false when completion metadata exists but HDF5 is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="hydration-payload")
        cache.get_embeddings(
            {"p1": {"title": "Seed", "abstract": "Abstract"}},
            SeededRandomEncodeModel(),
            show_progress=False,
        )
        cache.mark_hydrated(
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
            dataset_split="train",
            corpus_size=512,
            complete=True,
        )

        assert cache.is_hydrated(
            dataset_split="train",
            corpus_size=512,
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
        )

        cache.h5_path.unlink(missing_ok=True)
        assert not cache.is_hydrated(
            dataset_split="train",
            corpus_size=512,
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
        )


def test_embedding_cache_hydration_requires_metadata_row_integrity() -> None:
    """Hydration should be false when embedding rows have no matching metadata rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="hydration-metadata-rows")
        cache.get_embeddings(
            {"p1": {"title": "Seed", "abstract": "Abstract"}},
            SeededRandomEncodeModel(),
            show_progress=False,
        )
        cache.mark_hydrated(
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
            dataset_split="train",
            corpus_size=256,
            complete=True,
        )
        assert cache.is_hydrated(
            dataset_split="train",
            corpus_size=256,
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
        )

        with sqlite3.connect(cache.db_path) as conn:
            conn.execute("DELETE FROM papers")
            conn.commit()

        assert not cache.is_hydrated(
            dataset_split="train",
            corpus_size=256,
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
        )


def test_embedding_cache_hydration_requires_dataset_source_metadata() -> None:
    """Hydration should be false when completion exists but dataset source is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="hydration-source-required")
        cache.get_embeddings(
            {"p1": {"title": "Seed", "abstract": "Abstract"}},
            SeededRandomEncodeModel(),
            show_progress=False,
        )
        cache.mark_hydrated(
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
            dataset_split="train",
            corpus_size=128,
            complete=True,
        )
        assert cache.is_hydrated(
            dataset_split="train",
            corpus_size=128,
            dataset_source="librarian-bots/arxiv-metadata-snapshot",
        )

        with sqlite3.connect(cache.db_path) as conn:
            conn.execute(
                "UPDATE cache_metadata SET value = '' WHERE key = ?",
                (HYDRATION_DATASET_SOURCE_KEY,),
            )
            conn.commit()

        assert not cache.is_hydrated(dataset_split="train", corpus_size=128)


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
        with sqlite3.connect(cache.db_path) as conn:
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
        with sqlite3.connect(reloaded.db_path) as conn:
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
    with sqlite3.connect(cache.db_path) as conn:
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
    with sqlite3.connect(reloaded.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0


def test_embedding_cache_recovery_contracts(tmp_path: Path) -> None:
    """Legacy schema recovery and clear() should restore a healthy namespace."""
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="legacy-recovery")
    model = SeededRandomEncodeModel()

    cache.h5_path.unlink(missing_ok=True)
    with h5py.File(cache.h5_path, "w") as h5:
        h5.create_dataset("legacy_payload", data=np.array([1, 2, 3], dtype=np.float32))

    with sqlite3.connect(cache.db_path) as conn:
        conn.execute(
            """
            INSERT INTO papers (paper_id, title, abstract, year, text_hash, embedding_dim, row_idx)
            VALUES ('seed', 'seed', '', NULL, 'hash', 3, 2)
            """
        )
        conn.commit()

    reloaded = EmbeddingCache(cache_dir=tmp_path, model_name="legacy-recovery")
    with sqlite3.connect(reloaded.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0

    reloaded.get_embeddings(
        {"seed": {"title": "Seed", "abstract": "x", "year": None}}, model
    )
    with h5py.File(reloaded.h5_path, "r") as h5:
        assert "embeddings" in h5
        assert h5["embeddings"].shape[0] == 1

    reloaded.clear()
    assert reloaded.db_path.exists()
    assert not reloaded.h5_path.exists()
