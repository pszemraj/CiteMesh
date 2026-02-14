"""Regression coverage for hydration fallback and warm-cache behavior."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from citemesh.data.embedding_cache import CacheSearchResult
from citemesh.strategies.embedding import (
    EmbeddingGraphBuilder,
    _query_seed_id,
    _stream_heap_key,
)


class _FakeEncodeModel:
    """Minimal encode model used by hydration tests."""

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        """Return deterministic embeddings for input texts.

        :param list[str] texts: Input texts.
        :param Any kwargs: Ignored kwargs.
        :return np.ndarray: Float32 embedding matrix.
        """
        del kwargs
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_streaming_embedding_hydration_loader_falls_back_to_secondary_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing primary stream should fallback to a secondary dataset source."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    load_calls: list[tuple[str, str, bool]] = []

    def fake_load_dataset(
        dataset_name: str, split: str, streaming: bool = False
    ) -> list[dict[str, Any]]:
        """Simulate primary source failure and secondary source success.

        :param str dataset_name: Requested dataset identifier.
        :param str split: Dataset split.
        :param bool streaming: Streaming flag.
        :return list[dict[str, Any]]: Fake dataset rows.
        """
        load_calls.append((dataset_name, split, streaming))
        if dataset_name == "librarian-bots/arxiv-metadata-snapshot":
            raise RuntimeError("primary unavailable")
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

    builder = EmbeddingGraphBuilder(
        max_papers=1,
        use_streaming=True,
        random_seed=0,
        client=MagicMock(),
    )

    selected_name, dataset = builder._load_dataset_for_hydration(use_streaming=True)

    assert selected_name == "CShorten/ML-ArXiv-Papers"
    assert [name for name, _, _ in load_calls] == [
        "librarian-bots/arxiv-metadata-snapshot",
        "CShorten/ML-ArXiv-Papers",
    ]
    assert len(list(dataset)) == 1


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


def test_warm_cache_candidate_selection_skips_dataset_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When cache is hydrated, candidate retrieval should bypass dataset loading."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    fake_datasets = types.ModuleType("datasets")

    def fail_load_dataset(*args: Any, **kwargs: Any) -> None:
        """Fail if hydration path unexpectedly touches HF datasets.

        :param Any args: Positional args.
        :param Any kwargs: Keyword args.
        :raises AssertionError: Always.
        """
        raise AssertionError("load_dataset should not be called on warm cache")

    fake_datasets.load_dataset = fail_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    builder = EmbeddingGraphBuilder(
        max_papers=2,
        use_streaming=False,
        random_seed=0,
        client=MagicMock(),
    )
    builder.embedding_cache.is_hydrated = MagicMock(return_value=True)
    builder.embedding_cache.search = MagicMock(
        return_value=[
            CacheSearchResult(
                paper_id="a",
                score=0.9,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                metadata={
                    "title": "A",
                    "abstract": "A",
                    "year": 2020,
                    "authors": ["Alice"],
                    "categories": ["cs.AI"],
                },
            ),
            CacheSearchResult(
                paper_id="b",
                score=0.8,
                embedding=np.asarray([0.5, 0.5], dtype=np.float32),
                metadata={
                    "title": "B",
                    "abstract": "B",
                    "year": 2021,
                    "authors": ["Bob"],
                    "categories": ["cs.LG"],
                },
            ),
        ]
    )
    monkeypatch.setattr(builder, "_get_model_for_encoding", lambda: _FakeEncodeModel())

    candidates = builder._select_candidates_from_loaded(
        np.asarray([1.0, 0.0], dtype=np.float32)
    )

    assert [paper_id for paper_id, _, _ in candidates] == ["a", "b"]
