"""Opt-in regressions over a real arXiv snapshot and embedding runtime."""

from __future__ import annotations

from itertools import islice
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest

from citemesh.data import DEFAULT_EMBEDDING_MODEL_NAME
from citemesh.data.embedding_cache import EMBEDDINGS_DATASET_NAME
from citemesh.strategies.embedding import DEFAULT_DATASET_SOURCE, EmbeddingGraphBuilder
from citemesh.strategies.embedding import deps as deps_module
from citemesh.strategies.embedding.records import _dataset_record_paper_id

pytestmark = pytest.mark.slow


def _read_real_arxiv_snapshot_subset(datasets_module: Any, *, record_count: int) -> Any:
    """Read a small usable subset from the configured live arXiv snapshot.

    :param Any datasets_module: Imported ``datasets`` module.
    :param int record_count: Number of usable source records to retain.
    :return Any: Materialized Arrow-backed ``datasets.Dataset`` subset.
    """
    stream = datasets_module.load_dataset(
        DEFAULT_DATASET_SOURCE,
        split="train",
        streaming=True,
    )
    records: list[dict[str, Any]] = []
    for record in islice(stream, record_count * 4):
        copied_record = dict(record)
        if all(
            str(copied_record.get(field, "")).strip()
            for field in ("id", "title", "abstract")
        ):
            records.append(copied_record)
        if len(records) == record_count:
            break

    if len(records) < record_count:  # pragma: no cover - source-dependent
        pytest.skip(
            f"The live arXiv snapshot returned only {len(records)} usable rows."
        )
    return datasets_module.Dataset.from_list(records)


def _dataset_slice(dataset: Any, split: str) -> Any:
    """Return the real ``Dataset`` view represented by a hydration split token.

    :param Any dataset: Arrow-backed source snapshot for the current test phase.
    :param str split: Hydration split, optionally with a numeric row slice.
    :return Any: Whole source or the requested Arrow-backed row slice.
    """
    if split == "train":
        return dataset
    if not (split.startswith("train[") and split.endswith("]")):
        raise AssertionError(f"Unexpected local snapshot split: {split}")
    start_text, stop_text = split.removeprefix("train[").removesuffix("]").split(":")
    start = int(start_text) if start_text else 0
    stop = int(stop_text) if stop_text else len(dataset)
    return dataset.select(range(start, min(stop, len(dataset))))


def _read_cached_vectors(cache: Any, paper_ids: set[str]) -> dict[str, np.ndarray]:
    """Read selected persisted vectors without invoking the encoder.

    :param Any cache: Hydrated embedding cache whose SQLite and HDF5 rows agree.
    :param set[str] paper_ids: Canonical cache identifiers to retrieve.
    :return dict[str, np.ndarray]: Detached FP32 copies keyed by paper identifier.
    """
    with (
        cache._cache_lock(),
        cache._connect_db() as conn,
        h5py.File(cache.h5_path, "r") as h5_file,
    ):
        row_indices = {
            str(paper_id): int(row_index)
            for paper_id, row_index in conn.execute(
                "SELECT paper_id, row_idx FROM papers"
            ).fetchall()
            if str(paper_id) in paper_ids
        }
        embeddings = h5_file[EMBEDDINGS_DATASET_NAME]
        vectors = {
            paper_id: np.asarray(embeddings[row_index], dtype=np.float32).copy()
            for paper_id, row_index in row_indices.items()
        }
    assert vectors.keys() == paper_ids
    return vectors


def test_real_arxiv_corpus_resume_retains_all_cache_after_interrupt_and_shrink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume an incomplete covering cache at its recorded full-source scope.

    The source starts as a real snapshot subset and hydration uses the designated
    EmbeddingGemma checkpoint. The later Arrow-backed snapshots model an upstream
    append, interrupt, and shrink. A smaller request must retain the old vectors,
    restore the full-cache marker after the interrupt, then scan the shrunken
    source at ``all`` and retain new rows beyond that smaller request's cap.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to install local source views.
    :return None: Checks persisted vectors and hydration metadata across each phase.
    """
    torch = pytest.importorskip("torch")
    datasets_module = pytest.importorskip("datasets")
    pytest.importorskip("sentence_transformers")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend unavailable in this runtime")

    source_snapshot = _read_real_arxiv_snapshot_subset(datasets_module, record_count=19)
    initial_snapshot = source_snapshot.select(range(12))
    replacement_rows = source_snapshot.select(range(12, 19))
    assert isinstance(initial_snapshot, datasets_module.Dataset)
    assert {"id", "title", "abstract"} <= set(initial_snapshot.column_names)
    current_snapshot = {"dataset": initial_snapshot}
    requested_splits: list[str] = []

    def load_local_snapshot(
        dataset_source: str,
        *,
        split: str,
        streaming: bool,
        num_proc: int | None,
    ) -> Any:
        """Serve a real ``Dataset`` shape whose membership changes by phase.

        :param str dataset_source: Requested Hugging Face source identifier.
        :param str split: Requested split or row slice.
        :param bool streaming: Whether hydration requested streaming mode.
        :param Optional[int] num_proc: Worker count requested by non-streaming load.
        :return Any: Current Arrow-backed snapshot or its requested slice.
        """
        assert dataset_source == DEFAULT_DATASET_SOURCE
        assert streaming is False
        assert num_proc is not None and num_proc >= 1
        requested_splits.append(split)
        return _dataset_slice(current_snapshot["dataset"], split)

    monkeypatch.setattr(
        deps_module,
        "_import_datasets_module",
        lambda: SimpleNamespace(load_dataset=load_local_snapshot),
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        device="mps",
        semantic_source="arxiv-corpus",
        dataset_source=DEFAULT_DATASET_SOURCE,
        corpus_size=None,
        storage_precision="float32",
        encode_batch_size=4,
        enable_torch_compile=False,
    )
    monkeypatch.setattr(
        builder,
        "_resolve_dataset_split_row_count",
        lambda _source: len(current_snapshot["dataset"]),
    )
    builder._load_model()
    assert builder._active_model_name == DEFAULT_EMBEDDING_MODEL_NAME
    weight_dtypes = {
        parameter.dtype
        for parameter in builder.model.parameters()
        if parameter.is_floating_point()
    }
    assert torch.float16 not in weight_dtypes
    assert weight_dtypes <= {torch.float32, torch.bfloat16}

    builder._ensure_cache_hydrated(use_streaming=False)
    cache = builder.embedding_cache
    initial_stats = cache.payload_stats()
    assert initial_stats.hydration_complete is True
    assert initial_stats.hydration_corpus_size == "all"
    assert cache.is_hydrated("train", None, dataset_source=DEFAULT_DATASET_SOURCE)
    historical_ids = cache.get_cached_paper_ids()
    assert len(historical_ids) == len(initial_snapshot)
    historical_vectors = _read_cached_vectors(cache, historical_ids)
    assert all(vector.dtype == np.float32 for vector in historical_vectors.values())

    clear_calls: list[str] = []
    original_clear = cache.clear

    def record_clear(*args: Any, **kwargs: Any) -> None:
        """Record an unexpected destructive namespace clear.

        :param Any args: Positional arguments forwarded to cache clearing.
        :param Any kwargs: Keyword arguments forwarded to cache clearing.
        :return None: Calls the original cache clear after recording it.
        """
        clear_calls.append("clear")
        original_clear(*args, **kwargs)

    monkeypatch.setattr(cache, "clear", record_clear)
    current_snapshot["dataset"] = datasets_module.Dataset.from_list(
        [*list(initial_snapshot), *list(replacement_rows.select(range(1)))]
    )
    builder.corpus_size = 4
    original_hydrate = builder._hydrate_dataset_records

    def interrupt_refresh(**_kwargs: Any) -> int:
        """Interrupt the reused full-corpus refresh before it writes a row.

        :param Any _kwargs: Hydration parameters supplied by the refresh path.
        :return int: Never returns normally.
        :raises KeyboardInterrupt: Simulates a user interrupt during refresh.
        """
        raise KeyboardInterrupt("stop reused-cache refresh")

    monkeypatch.setattr(builder, "_hydrate_dataset_records", interrupt_refresh)
    with pytest.raises(KeyboardInterrupt, match="stop reused-cache refresh"):
        builder._ensure_cache_hydrated(use_streaming=False)
    interrupted_stats = cache.payload_stats()
    assert interrupted_stats.hydration_complete is True
    assert interrupted_stats.hydration_corpus_size == "all"
    assert cache.get_cached_paper_ids() == historical_ids

    monkeypatch.setattr(builder, "_hydrate_dataset_records", original_hydrate)
    current_snapshot["dataset"] = datasets_module.Dataset.from_list(
        [
            *list(initial_snapshot.select(range(4))),
            *list(replacement_rows),
        ]
    )
    new_ids = {
        _dataset_record_paper_id(dict(record), index)
        for index, record in enumerate(replacement_rows)
    }
    assert len(current_snapshot["dataset"]) < len(initial_snapshot)
    assert new_ids.isdisjoint(historical_ids)
    assert len(new_ids) > builder.corpus_size

    requested_splits.clear()
    builder._ensure_cache_hydrated(use_streaming=False)

    resumed_stats = cache.payload_stats()
    assert resumed_stats.hydration_complete is True
    assert resumed_stats.hydration_corpus_size == "all"
    assert cache.is_hydrated("train", None, dataset_source=DEFAULT_DATASET_SOURCE)
    assert "train" in requested_splits
    assert cache.get_cached_paper_ids() == historical_ids | new_ids
    assert cache.embedding_count() == len(historical_ids) + len(new_ids)
    assert cache.embedding_count() > builder.corpus_size
    assert cache.get_hydration_rowcount_reconciliation() == (
        len(current_snapshot["dataset"]),
        cache.embedding_count(),
    )
    requested_splits.clear()
    builder._ensure_cache_hydrated(use_streaming=False)
    builder.corpus_size = None
    builder._ensure_cache_hydrated(use_streaming=False)
    assert requested_splits == []
    restored_vectors = _read_cached_vectors(cache, historical_ids)
    for paper_id, original_vector in historical_vectors.items():
        np.testing.assert_array_equal(restored_vectors[paper_id], original_vector)
    new_vectors = _read_cached_vectors(cache, new_ids)
    assert all(vector.dtype == np.float32 for vector in new_vectors.values())
    assert all(np.isfinite(vector).all() for vector in new_vectors.values())
    with h5py.File(cache.h5_path, "r") as h5_file:
        assert h5_file[EMBEDDINGS_DATASET_NAME].dtype == np.dtype(np.float32)
    assert clear_calls == []
