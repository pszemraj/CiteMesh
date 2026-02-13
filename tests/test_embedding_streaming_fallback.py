"""Regression coverage for streaming dataset fallback behavior."""

from __future__ import annotations

import heapq
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from citemesh.strategies.embedding import EmbeddingGraphBuilder


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
    ) -> None:
        del seed_embedding
        for metadata in batch:
            candidate = (
                0.95,
                metadata["paper_id"],
                metadata,
                np.array([0.1], dtype=np.float32),
            )
            if len(heap) < max_candidates:
                heapq.heappush(heap, candidate)
            elif candidate[0] > heap[0][0]:
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
