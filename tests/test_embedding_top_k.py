"""Tests for embedding top-k configuration validation."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from citemesh.strategies.embedding import EmbeddingGraphBuilder


def test_embedding_top_k_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding builder should reject non-positive ``top_k`` values."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        EmbeddingGraphBuilder(top_k=0, client=MagicMock())


def test_loaded_candidate_ties_are_sorted_by_paper_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loaded candidate ranking should use paper ID as deterministic tie-break."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    builder = EmbeddingGraphBuilder(max_papers=2, top_k=2, client=MagicMock())
    builder.arxiv_corpus = {
        "b": {"title": "Paper B", "abstract": "Abstract B"},
        "a": {"title": "Paper A", "abstract": "Abstract A"},
    }

    monkeypatch.setattr(builder, "_get_model_for_encoding", lambda: MagicMock())
    builder.embedding_cache.get_embeddings = MagicMock(
        return_value={
            "a": np.asarray([1.0, 0.0], dtype=np.float32),
            "b": np.asarray([1.0, 0.0], dtype=np.float32),
        }
    )

    candidates = builder._select_candidates_from_loaded(
        np.asarray([1.0, 0.0], dtype=np.float32)
    )

    assert [paper_id for paper_id, _, _ in candidates] == ["a", "b"]
