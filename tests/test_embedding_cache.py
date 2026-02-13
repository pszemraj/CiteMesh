"""Tests for embedding cache hit/miss behavior."""

import tempfile

import numpy as np

from citemesh.data.embedding_cache import EmbeddingCache


class _MockModel:
    """Deterministic embedding model mock."""

    def __init__(self):
        self.encode_calls = 0

    def encode(self, texts, **kwargs):
        self.encode_calls += 1
        np.random.seed(len(texts))
        return np.random.rand(len(texts), 2)


def test_embedding_cache_returns_cached_vectors():
    """Repeated lookup should avoid re-encoding unchanged records."""
    model = _MockModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="test-model")
        papers = {
            "p1": {"title": "Paper One", "abstract": "Abstract one"},
            "p2": {"title": "Paper Two", "abstract": "Abstract two"},
        }

        first = cache.get_embeddings(papers, model)
        second = cache.get_embeddings(papers, model)

    assert model.encode_calls == 1
    assert first["p1"].shape == second["p1"].shape


def test_embedding_cache_recomputes_on_text_change():
    """Cache key changes when text changes."""
    model = _MockModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir, model_name="test-model")
        papers_v1 = {"p1": {"title": "Original", "abstract": "Abstract"}}
        papers_v2 = {"p1": {"title": "Updated", "abstract": "Abstract"}}

        cache.get_embeddings(papers_v1, model)
        cache.get_embeddings(papers_v2, model)

    assert model.encode_calls == 2
