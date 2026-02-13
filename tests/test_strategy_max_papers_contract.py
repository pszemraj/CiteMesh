"""Contract tests for ``max_papers`` behavior across strategies."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from citemesh.core import Paper
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import HybridGraphBuilder
from citemesh.strategies.recommendation import RecommendationGraphBuilder


def _seed_paper(paper_id: str = "seed") -> Paper:
    """Build a deterministic seed paper fixture.

    :param str paper_id: Seed paper identifier.
    :return Paper: Seed paper fixture.
    """
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=2024,
        abstract="seed abstract",
        is_seed=True,
    )


def _build_named_paper(paper_id: str) -> Paper:
    """Build a paper for deterministic graph expansion.

    :param str paper_id: Paper identifier.
    :return Paper: Non-seed paper fixture.
    """
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=2024,
        abstract=f"Abstract {paper_id}",
        is_seed=False,
    )


def test_recommendation_max_papers_is_total_node_cap_including_seed() -> None:
    """Recommendation strategy should include the seed within the max-papers cap."""
    client = MagicMock()
    client.get_paper.return_value = _seed_paper()
    client.get_recommended_papers.return_value = [
        _build_named_paper(f"r{i}") for i in range(1, 6)
    ]

    builder = RecommendationGraphBuilder(max_papers=3, client=client)
    papers = builder.collect_papers("seed")

    assert len(papers) == 3
    assert _seed_paper().paper_id in papers


def test_citation_max_papers_is_total_node_cap_including_seed() -> None:
    """Citation strategy should include the seed within the max-papers cap."""
    client = MagicMock()
    client.get_paper.return_value = _seed_paper()
    client.get_paper_references.return_value = [
        _build_named_paper(f"r{i}") for i in range(1, 6)
    ]
    client.get_paper_citations.return_value = [
        _build_named_paper(f"c{i}") for i in range(1, 6)
    ]

    builder = CitationGraphBuilder(
        max_papers=4,
        max_citations=10,
        max_references=10,
        client=client,
    )
    papers = builder.collect_papers("seed")

    assert len(papers) == 4
    assert _seed_paper().paper_id in papers


def test_embedding_max_papers_is_total_node_cap_including_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding strategy should include the seed within the max-papers cap."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    builder = EmbeddingGraphBuilder(
        max_papers=3,
        model_name="dummy",
        top_k=2,
        corpus_size=10,
        client=MagicMock(),
    )
    builder.client.get_paper.return_value = _seed_paper()
    builder._load_model = lambda: None
    builder._get_model_for_encoding = lambda: None
    builder._load_corpus = lambda: None
    builder._update_citation_counts = lambda _: None
    builder.arxiv_corpus = {
        "c1": {"title": "Paper c1", "abstract": "A"},
        "c2": {"title": "Paper c2", "abstract": "B"},
        "c3": {"title": "Paper c3", "abstract": "C"},
    }
    builder.embedding_cache.get_embeddings = MagicMock(
        return_value={
            "c1": np.array([1.0, 0.0], dtype=np.float32),
            "c2": np.array([0.0, 1.0], dtype=np.float32),
            "c3": np.array([0.5, 0.5], dtype=np.float32),
        }
    )
    builder._encode_texts = lambda texts, **kwargs: np.array(
        [[1.0, 0.0] for _ in texts], dtype=np.float32
    )

    papers = builder.collect_papers("seed")

    assert len(papers) == 3
    assert _seed_paper().paper_id in papers


def test_hybrid_max_papers_is_total_node_cap_including_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid strategy should include the seed within the max-papers cap."""
    monkeypatch.setattr(
        "citemesh.strategies.embedding._check_embedding_deps", lambda: None
    )

    class FakeCitationBuilder:
        """Citation branch stub with explicit max-papers behavior."""

        def __init__(self, max_papers: int, *_args: object, **_kwargs: object) -> None:
            """Store citation branch cap for deterministic fixture behavior.

            :param int max_papers: Max papers passed by hybrid builder.
            :param object _args: Ignored positional arguments.
            :param object _kwargs: Ignored keyword arguments.
            :return None: Stores configured limit.
            """
            self.max_papers = max_papers

        def collect_papers(self, seed_id: str, **_: object) -> dict[str, Paper]:
            """Return deterministic citation branch papers.

            :param str seed_id: Seed paper identifier.
            :param object _: Ignored keyword arguments.
            :return dict[str, Paper]: Deterministic citation-branch papers.
            """
            _ = seed_id
            papers = {
                "seed": _seed_paper(),
                "c1": _build_named_paper("c1"),
                "c2": _build_named_paper("c2"),
            }
            return dict(list(papers.items())[: self.max_papers])

    class FakeEmbeddingBuilder:
        """Embedding branch stub with deterministic max-semantic output."""

        def __init__(self, max_papers: int, *_args: object, **_kwargs: object) -> None:
            """Store embedding branch cap for deterministic fixture behavior.

            :param int max_papers: Max papers passed by hybrid builder.
            :param object _args: Ignored positional arguments.
            :param object _kwargs: Ignored keyword arguments.
            :return None: Stores configured limit.
            """
            self.max_papers = max_papers

        def collect_papers(self, seed_id: str, **_: object) -> dict[str, Paper]:
            """Return deterministic semantic papers used by hybrid merge path.

            :param str seed_id: Seed paper identifier.
            :param object _: Ignored keyword arguments.
            :return dict[str, Paper]: Deterministic semantic-branch papers.
            """
            _ = seed_id
            return {
                "seed": _seed_paper(),
                "s1": _build_named_paper("s1"),
                "s2": _build_named_paper("s2"),
            }

    monkeypatch.setattr(
        "citemesh.strategies.hybrid.CitationGraphBuilder", FakeCitationBuilder
    )
    monkeypatch.setattr(
        "citemesh.strategies.hybrid.EmbeddingGraphBuilder", FakeEmbeddingBuilder
    )
    monkeypatch.setattr(
        "citemesh.strategies.hybrid._check_embedding_deps", lambda: None
    )

    builder = HybridGraphBuilder(max_papers=4, max_semantic=1, client=MagicMock())
    papers = builder.collect_papers("seed")

    assert len(papers) == 4
    assert _seed_paper().paper_id in papers
