"""Tests for embedding top-k configuration validation."""

from unittest.mock import MagicMock

import pytest

from citemesh.strategies.embedding import EmbeddingGraphBuilder


def test_embedding_top_k_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding builder should reject non-positive ``top_k`` values."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        EmbeddingGraphBuilder(top_k=0, client=MagicMock())
