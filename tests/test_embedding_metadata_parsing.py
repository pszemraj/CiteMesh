"""Regression tests for embedding dataset metadata normalization."""

from unittest.mock import MagicMock

import pytest

from citemesh.strategies.embedding import EmbeddingGraphBuilder


def test_extract_paper_metadata_parses_snapshot_author_and_category_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshot records should normalize comma/space delimited author/category fields."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    metadata = builder._extract_paper_metadata(
        {
            "id": "2301.07041",
            "title": "A snapshot paper",
            "abstract": "An abstract.",
            "authors": "Alice Smith, Bob Jones, Carol White",
            "categories": "cs.LG cs.AI",
            "update_date": "2023-06-15",
        },
        fallback_index=0,
    )

    assert metadata["paper_id"] == "arxiv:2301.07041"
    assert metadata["authors"] == ["Alice Smith", "Bob Jones", "Carol White"]
    assert metadata["categories"] == ["cs.LG", "cs.AI"]
    assert metadata["year"] == 2023


def test_extract_paper_metadata_normalizes_arxiv_prefix_and_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ArXiv-like IDs should be canonicalized to stable ``arxiv:<id>`` form."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    builder = EmbeddingGraphBuilder(max_papers=1, client=MagicMock())
    metadata = builder._extract_paper_metadata(
        {
            "id": "arXiv:1706.03762v5",
            "title": "Versioned",
            "abstract": "An abstract.",
        },
        fallback_index=0,
    )

    assert metadata["paper_id"] == "arxiv:1706.03762"
