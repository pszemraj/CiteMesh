"""Contract tests for embedding cache layout, reset, and retrieval behavior."""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
import tempfile
from pathlib import Path
from queue import Empty
from typing import Any

import h5py
import numpy as np
import pytest

from citemesh.data.embedding_cache import (
    CALIBRATION_SAMPLE_SIZE_KEY,
    EMBEDDING_CACHE_LOCK_TIMEOUT_ENV_VAR,
    EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS,
    HYDRATION_COMPLETE_KEY,
    HYDRATION_CORPUS_SIZE_KEY,
    HYDRATION_DATASET_SOURCE_KEY,
    HYDRATION_SPLIT_KEY,
    MODEL_FINGERPRINT_KEY,
    SOURCE_TORCH_DTYPE_KEY,
    TEXT_FORMATTER_FINGERPRINT_KEY,
    EmbeddingCache,
    _resolve_cache_lock_timeout_seconds,
)
from citemesh.data.model_profiles import get_embedding_model_profile
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


def test_embedding_cache_metadata_refresh_survives_mixed_batch_encode_failure() -> None:
    """Metadata-only cache-hit refresh should persist even if miss encoding fails."""

    class _FailingEncodeModel:
        """Encode model stub that always fails for negative-path simulation."""

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            del texts, kwargs
            raise RuntimeError("encode failure")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="metadata-refresh-durable")
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
                    "p1": {"title": "Alpha", "abstract": "First", "year": 2025},
                    "p2": {"title": "Beta", "abstract": "Second"},
                },
                _FailingEncodeModel(),
                show_progress=False,
            )

        with sqlite3.connect(cache.db_path) as conn:
            refreshed_year = conn.execute(
                "SELECT year FROM papers WHERE paper_id = 'p1'"
            ).fetchone()[0]

    assert refreshed_year == 2025


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
        model = _CaptureEncodeModel()
        embeddings = cache.get_embeddings(papers, model, show_progress=False)

    assert list(embeddings) == list(papers)
    assert model.texts == expected_texts
    assert expected_texts == ["Alpha. First", "Beta", "Gamma"]


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
        cache.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            LookupEncodeModel({"Alpha. First": np.array([1.0, 0.0], dtype=np.float32)}),
            show_progress=False,
        )

        with sqlite3.connect(cache.db_path) as conn:
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


def test_embedding_cache_compression_codec_contracts() -> None:
    """Supported codecs should hydrate; unsupported codecs should fail fast."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(
            cache_dir=tmpdir,
            model_name="lzf-codec",
            compression="lzf",
            compression_level=1,
        )
        cache.get_embeddings(
            {"p1": {"title": "Alpha", "abstract": "First"}},
            LookupEncodeModel(
                {"Alpha. First": np.asarray([1.0, 0.0], dtype=np.float32)}
            ),
            show_progress=False,
        )

        with h5py.File(cache.h5_path, "r") as h5:
            assert h5["embeddings"].compression == "lzf"
            assert h5["binary_index"].compression == "lzf"

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="compression='szip' is unsupported"):
            EmbeddingCache(
                cache_dir=tmpdir,
                model_name="szip-codec",
                compression="szip",
                compression_level=1,
            )


def test_embedding_cache_restart_persistence_contracts() -> None:
    """Hydration/fingerprint metadata and payload state should survive restarts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hydration_cache = EmbeddingCache(
            cache_dir=tmpdir, model_name="hydration-persistence"
        )
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
        with sqlite3.connect(reloaded_fingerprint.db_path) as conn:
            metadata = {
                key: value
                for key, value in conn.execute("SELECT key, value FROM cache_metadata")
            }
        assert metadata[MODEL_FINGERPRINT_KEY] == "hf::org/model::abc123"

        cache = EmbeddingCache(cache_dir=tmpdir, model_name="payload-presence")
        assert not cache.has_cached_payload()
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
        with sqlite3.connect(cache.db_path) as conn:
            conn.execute("DELETE FROM papers")
            conn.commit()

    def _clear_dataset_source(cache: EmbeddingCache) -> None:
        with sqlite3.connect(cache.db_path) as conn:
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


def test_embedding_cache_recovery_clears_orphan_h5_rows(tmp_path: Path) -> None:
    """Reload should clear namespace when HDF5 has rows missing SQLite metadata mappings."""
    cache = EmbeddingCache(cache_dir=tmp_path, model_name="orphan-h5-row-recovery")
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
    with sqlite3.connect(reloaded.db_path) as conn:
        paper_rows = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    assert paper_rows == 0
    assert not reloaded.has_cached_payload()
