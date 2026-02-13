"""Regression coverage for streaming dataset fallback behavior."""

from __future__ import annotations

import heapq
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from citemesh.strategies.embedding import (
    EmbeddingGraphBuilder,
    _query_seed_id,
    _stream_heap_key,
)


def test_streaming_embedding_falls_back_to_secondary_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing primary HF stream should transparently fallback to secondary source."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    load_calls: list[tuple[str, str, bool]] = []

    def fake_load_dataset(
        dataset_name: str, split: str, streaming: bool = False
    ) -> list[dict[str, Any]]:
        """Simulate dataset loader with primary-source failure then fallback success.

        :param str dataset_name: Requested dataset identifier.
        :param str split: Dataset split string.
        :param bool streaming: Whether streaming mode is requested.
        :return list[dict[str, Any]]: Mocked dataset records.
        :raises RuntimeError: When primary dataset is requested.
        """
        load_calls.append((dataset_name, split, streaming))
        if dataset_name == "librarian-bots/arxiv-metadata-snapshot":
            raise RuntimeError("Primary source unavailable")
        return [
            {
                "id": "fallback-paper",
                "title": "Fallback Title",
                "abstract": "Fallback abstract.",
                "year": 2020,
            }
        ]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    def fake_process_stream_batch(
        self,
        batch: list[dict[str, Any]],
        seed_embedding: np.ndarray,
        heap: list[tuple],
        max_candidates: int,
        seen_paper_ids: set[str],
    ) -> None:
        """Push deterministic candidate records into the streaming heap.

        :param EmbeddingGraphBuilder self: Builder instance.
        :param list[dict[str, Any]] batch: Batch metadata payload.
        :param np.ndarray seed_embedding: Seed embedding vector.
        :param list[tuple] heap: Candidate min-heap.
        :param int max_candidates: Heap capacity.
        :param set[str] seen_paper_ids: Set of already-seen paper IDs.
        :return None: Heap is mutated in-place.
        """
        del seed_embedding
        del seen_paper_ids
        for metadata in batch:
            paper_id = metadata["paper_id"]
            candidate = (
                _stream_heap_key(0.95, paper_id, stable_index=0),
                paper_id,
                metadata,
                np.array([0.1], dtype=np.float32),
            )
            if len(heap) < max_candidates:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)

    monkeypatch.setattr(
        EmbeddingGraphBuilder, "_process_stream_batch", fake_process_stream_batch
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        use_streaming=True,
        random_seed=0,
        client=MagicMock(),
    )
    candidates = builder._select_candidates_streaming(np.array([1.0], dtype=np.float32))

    assert [name for name, _, _ in load_calls] == [
        "librarian-bots/arxiv-metadata-snapshot",
        "CShorten/ML-ArXiv-Papers",
    ]
    assert len(candidates) == 1
    assert candidates[0][0] == "fallback-paper"


def test_streaming_with_sliced_split_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming mode should reject sliced split syntax with clear guidance."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    with pytest.raises(ValueError, match="does not support sliced dataset splits"):
        EmbeddingGraphBuilder(
            max_papers=1,
            dataset_split="train[:5%]",
            use_streaming=True,
            client=MagicMock(),
        )


def test_streaming_candidate_ties_use_paper_id_tiebreak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming candidates with equal similarity should sort by paper ID."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    def fake_load_dataset(
        dataset_name: str, split: str, streaming: bool = False
    ) -> list[dict[str, Any]]:
        """Return deterministic tiny dataset used for tie-break verification.

        :param str dataset_name: Requested dataset identifier.
        :param str split: Dataset split string.
        :param bool streaming: Whether streaming mode is requested.
        :return list[dict[str, Any]]: Mocked dataset records.
        """
        del dataset_name
        del split
        del streaming
        return [
            {"id": "b", "title": "B", "abstract": "Abstract B"},
            {"id": "a", "title": "A", "abstract": "Abstract A"},
        ]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    def fake_process_stream_batch(
        self,
        batch: list[dict[str, Any]],
        seed_embedding: np.ndarray,
        heap: list[tuple],
        max_candidates: int,
        seen_paper_ids: set[str],
    ) -> None:
        """Push deterministic tie candidates into heap for ordering assertions.

        :param EmbeddingGraphBuilder self: Builder instance.
        :param list[dict[str, Any]] batch: Batch metadata payload.
        :param np.ndarray seed_embedding: Seed embedding vector.
        :param list[tuple] heap: Candidate min-heap.
        :param int max_candidates: Heap capacity.
        :param set[str] seen_paper_ids: Set of already-seen paper IDs.
        :return None: Heap is mutated in-place.
        """
        del self
        del seed_embedding
        del seen_paper_ids
        for metadata in batch:
            paper_id = metadata["paper_id"]
            candidate = (
                _stream_heap_key(0.95, paper_id, stable_index=0),
                paper_id,
                metadata,
                np.array([0.1], dtype=np.float32),
            )
            if len(heap) < max_candidates:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)

    monkeypatch.setattr(
        EmbeddingGraphBuilder, "_process_stream_batch", fake_process_stream_batch
    )

    builder = EmbeddingGraphBuilder(
        max_papers=2,
        use_streaming=True,
        random_seed=0,
        client=MagicMock(),
    )
    candidates = builder._select_candidates_streaming(np.array([1.0], dtype=np.float32))

    assert [paper_id for paper_id, _, _ in candidates] == ["a", "b"]


def test_collect_papers_uses_hashed_query_seed_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Query-mode seeds should use deterministic hashed identifiers."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    builder = EmbeddingGraphBuilder(
        max_papers=1, use_streaming=False, client=MagicMock()
    )
    builder.client.get_paper = MagicMock(return_value=None)
    monkeypatch.setattr(builder, "_load_model", lambda: None)
    monkeypatch.setattr(
        builder,
        "_encode_texts",
        lambda texts, show_progress_bar=False: np.asarray(
            [[1.0, 0.0] for _ in texts], dtype=np.float32
        ),
    )
    monkeypatch.setattr(builder, "_load_corpus", lambda: None)
    monkeypatch.setattr(builder, "_select_candidates_from_loaded", lambda _: [])
    monkeypatch.setattr(builder, "_update_citation_counts", lambda _: None)

    query = "attention mechanism test query"
    papers = builder.collect_papers(query)
    expected_seed_id = _query_seed_id(query)

    assert list(papers.keys()) == [expected_seed_id]
    assert papers[expected_seed_id].is_seed is True


def test_stream_heap_key_handles_prefix_ids() -> None:
    """Prefix paper IDs should preserve lexicographic ordering under tie-break keys."""
    assert _stream_heap_key(0.95, "a", 0) > _stream_heap_key(0.95, "aa", 0)


def test_stream_batch_skips_duplicate_paper_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming batch processing should avoid duplicate IDs in the heap."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    builder = EmbeddingGraphBuilder(
        max_papers=2,
        use_streaming=True,
        random_seed=0,
        client=MagicMock(),
    )
    monkeypatch.setattr(builder, "_get_model_for_encoding", lambda: MagicMock())
    builder.embedding_cache.get_embeddings = MagicMock(
        return_value={"dup": np.asarray([1.0, 0.0], dtype=np.float32)}
    )

    batch = [
        {"paper_id": "dup", "title": "First", "abstract": "A"},
        {"paper_id": "dup", "title": "Second", "abstract": "B"},
    ]
    heap: list[tuple] = []
    seen: set[str] = set()
    builder._process_stream_batch(
        batch=batch,
        seed_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        heap=heap,
        max_candidates=4,
        seen_paper_ids=seen,
    )

    assert len(heap) == 1
    assert seen == {"dup"}


def test_stream_batch_keeps_strongest_candidates_under_heap_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded streaming heaps should retain the highest-similarity candidates."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    builder = EmbeddingGraphBuilder(
        max_papers=2,
        use_streaming=True,
        random_seed=0,
        client=MagicMock(),
    )
    monkeypatch.setattr(builder, "_get_model_for_encoding", lambda: MagicMock())
    builder.embedding_cache.get_embeddings = MagicMock(
        return_value={
            "top": np.asarray([1.0, 0.0], dtype=np.float32),
            "mid": np.asarray([0.5, 0.0], dtype=np.float32),
            "low": np.asarray([0.2, 0.0], dtype=np.float32),
        }
    )

    batch = [
        {"paper_id": "top", "title": "Top", "abstract": "A"},
        {"paper_id": "mid", "title": "Mid", "abstract": "B"},
        {"paper_id": "low", "title": "Low", "abstract": "C"},
    ]
    heap: list[tuple] = []
    seen: set[str] = set()
    builder._process_stream_batch(
        batch=batch,
        seed_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        heap=heap,
        max_candidates=2,
        seen_paper_ids=seen,
    )

    assert {paper_id for _, paper_id, _, _ in heap} == {"top", "mid"}
