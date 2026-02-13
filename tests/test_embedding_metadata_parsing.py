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

    assert metadata["paper_id"] == "2301.07041"
    assert metadata["authors"] == ["Alice Smith", "Bob Jones", "Carol White"]
    assert metadata["categories"] == ["cs.LG", "cs.AI"]
    assert metadata["year"] == 2023
