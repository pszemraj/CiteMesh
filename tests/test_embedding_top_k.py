"""Tests for embedding top-k configuration validation."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from citemesh.core import Paper
from citemesh.data.embedding_cache import CacheSearchResult
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
    """Cache search ties should use paper ID as deterministic tie-break."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    builder = EmbeddingGraphBuilder(max_papers=2, top_k=2, client=MagicMock())
    builder.embedding_cache.is_hydrated = MagicMock(return_value=True)
    builder.embedding_cache.search = MagicMock(
        return_value=[
            CacheSearchResult(
                paper_id="b",
                score=0.95,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                metadata={"title": "B", "abstract": "B", "authors": []},
            ),
            CacheSearchResult(
                paper_id="a",
                score=0.95,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                metadata={"title": "A", "abstract": "A", "authors": []},
            ),
        ]
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
