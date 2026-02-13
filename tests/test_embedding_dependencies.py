"""Tests for embedding optional dependency guardrails."""

import sys

from unittest.mock import MagicMock

import pytest

from citemesh.strategies.embedding import EmbeddingGraphBuilder


def test_embedding_builder_requires_optional_deps(monkeypatch):
    """Embedding builder should fail with clear guidance when deps are missing."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setitem(sys.modules, "datasets", None)

    with pytest.raises(
        ImportError,
        match=r"Embedding strategy requires: torch, sentence-transformers, datasets\. "
        r"Install with: pip install citemesh\[embeddings\]",
    ):
        EmbeddingGraphBuilder(
            max_papers=5,
            model_name="test-model",
            random_seed=42,
            client=MagicMock(),
        )
