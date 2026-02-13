"""Tests for centralized CLI strategy dispatch."""

import argparse

import pytest

from citemesh import cli as cli_module
from tests.conftest import build_fake_strategy_builder_factory, build_seed_graph


def _dispatch_namespace() -> argparse.Namespace:
    return argparse.Namespace(
        paper_id="seed",
        max_papers=11,
        max_citations=9,
        max_references=7,
        similarity_threshold=0.21,
        no_references=True,
        model="m",
        dataset_split="train",
        corpus_size=1234,
        all_corpus=True,
        top_k=4,
        truncate_dim=64,
        streaming=True,
        max_semantic=5,
        seed=7,
        force_rebuild_cache=False,
    )


@pytest.mark.parametrize(
    (
        "strategy",
        "builder_name",
        "expected_kwargs",
    ),
    [
        (
            "citation",
            "CitationGraphBuilder",
            {
                "max_papers": 11,
                "max_citations": 9,
                "max_references": 7,
                "similarity_threshold": 0.21,
                "fetch_references": False,
                "random_seed": 7,
            },
        ),
        (
            "recommendation",
            "RecommendationGraphBuilder",
            {
                "max_papers": 11,
                "fetch_references": False,
                "similarity_threshold": 0.21,
                "random_seed": 7,
            },
        ),
        (
            "embedding",
            "EmbeddingGraphBuilder",
            {
                "max_papers": 11,
                "model_name": "m",
                "dataset_split": "train",
                "corpus_size": None,
                "truncate_dim": 64,
                "top_k": 4,
                "force_rebuild_cache": False,
                "use_streaming": True,
                "random_seed": 7,
            },
        ),
        (
            "hybrid",
            "HybridGraphBuilder",
            {
                "max_papers": 11,
                "max_citations": 9,
                "max_references": 7,
                "fetch_references": False,
                "max_semantic": 5,
                "model_name": "m",
                "dataset_split": "train",
                "corpus_size": None,
                "truncate_dim": 64,
                "use_streaming": True,
                "random_seed": 7,
                "force_rebuild_cache": False,
            },
        ),
    ],
)
def test_strategy_dispatches_to_matching_builder_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    builder_name: str,
    expected_kwargs: dict[str, object],
) -> None:
    """Strategy dispatch should pass CLI arguments into the selected builder."""
    captured: dict[str, object] = {}
    namespace = _dispatch_namespace()

    monkeypatch.setattr(
        cli_module,
        builder_name,
        build_fake_strategy_builder_factory(captured, graph=build_seed_graph("seed")),
    )

    graph, seed_id = cli_module._build_strategy_graph(namespace, strategy)
    assert seed_id == "seed"
    assert graph.number_of_nodes() == 1
    assert captured == expected_kwargs


@pytest.mark.parametrize("strategy,builder_name", [("embedding", "EmbeddingGraphBuilder"), ("hybrid", "HybridGraphBuilder")])
def test_force_rebuild_cache_passes_through_embedding_strategies(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    builder_name: str,
) -> None:
    """Forceful cache rebuild flag should pass through embedding builders."""
    captured: dict[str, object] = {}
    namespace = _dispatch_namespace()
    namespace.force_rebuild_cache = True

    monkeypatch.setattr(
        cli_module,
        builder_name,
        build_fake_strategy_builder_factory(captured, graph=build_seed_graph("seed")),
    )

    _ = cli_module._build_strategy_graph(namespace, strategy)
    assert captured["force_rebuild_cache"] is True


def test_build_strategy_graph_rejects_invalid_strategy() -> None:
    """Unsupported strategies should be rejected with a clear ValueError."""
    namespace = _dispatch_namespace()
    with pytest.raises(ValueError, match="Unsupported strategy: unknown"):
        cli_module._build_strategy_graph(namespace, "unknown")
