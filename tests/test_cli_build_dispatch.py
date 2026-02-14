"""Tests for centralized CLI strategy dispatch."""

import argparse

import pytest

from citemesh import cli as cli_module
from tests.conftest import build_fake_strategy_builder_factory, build_seed_graph


def _dispatch_namespace() -> argparse.Namespace:
    """Build argparse namespace fixture for strategy dispatch tests.

    :return argparse.Namespace: Namespace mirroring parsed CLI arguments.
    """
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
        storage_precision="int8",
        binary_prefilter=True,
        binary_rescore_multiplier=9,
        calibration_sample_size=123,
        cache_compression="gzip",
        cache_compression_level=1,
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
                "storage_precision": "int8",
                "binary_prefilter": True,
                "binary_rescore_multiplier": 9,
                "calibration_sample_size": 123,
                "cache_compression": "gzip",
                "cache_compression_level": 1,
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
                "storage_precision": "int8",
                "binary_prefilter": True,
                "binary_rescore_multiplier": 9,
                "calibration_sample_size": 123,
                "cache_compression": "gzip",
                "cache_compression_level": 1,
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
    """Strategy dispatch should pass CLI arguments into the selected builder.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to patch builder class.
    :param str strategy: Strategy key passed to dispatch.
    :param str builder_name: Builder attribute name patched on CLI module.
    :param dict[str, object] expected_kwargs: Expected constructor kwargs.
    :return None: Asserts dispatch behavior.
    """
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


@pytest.mark.parametrize(
    "strategy,builder_name",
    [("embedding", "EmbeddingGraphBuilder"), ("hybrid", "HybridGraphBuilder")],
)
def test_force_rebuild_cache_passes_through_embedding_strategies(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    builder_name: str,
) -> None:
    """Forceful cache rebuild flag should pass through embedding builders.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to patch builder class.
    :param str strategy: Embedding-like strategy name.
    :param str builder_name: Builder attribute name patched on CLI module.
    :return None: Asserts force-rebuild argument threading.
    """
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
