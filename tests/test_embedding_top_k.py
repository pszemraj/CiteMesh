"""Tests for embedding top-k configuration validation."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from citemesh.core import Paper
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


def test_collect_papers_respects_max_papers_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding collection should cap total papers (including seed) at max_papers."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    builder = EmbeddingGraphBuilder(max_papers=2, top_k=2, client=MagicMock())
    builder.client.get_paper = MagicMock(
        return_value=Paper(
            paper_id="seed",
            title="Seed",
            year=2020,
            abstract="Seed abstract",
        )
    )

    monkeypatch.setattr(builder, "_load_model", lambda: None)
    monkeypatch.setattr(
        builder,
        "_encode_texts",
        lambda texts, batch_size=None, show_progress_bar=False: np.asarray(
            [[1.0, 0.0] for _ in texts], dtype=np.float32
        ),
    )
    monkeypatch.setattr(builder, "_load_corpus", lambda: None)
    monkeypatch.setattr(
        builder,
        "_select_candidates_from_loaded",
        lambda _: [
            (
                "candidate-1",
                {"title": "Candidate 1", "abstract": "A", "authors": []},
                np.asarray([1.0, 0.0], dtype=np.float32),
            ),
            (
                "candidate-2",
                {"title": "Candidate 2", "abstract": "B", "authors": []},
                np.asarray([1.0, 0.0], dtype=np.float32),
            ),
        ],
    )
    monkeypatch.setattr(builder, "_update_citation_counts", lambda _: None)

    papers = builder.collect_papers("seed")
    assert len(papers) == 2
    assert set(papers.keys()) == {"seed", "candidate-1"}
