"""Contract tests for CLI argument threading and path normalization."""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import networkx as nx
import pytest

from citemesh import cli as cli_module
from citemesh.cli import canonicalize_paper_id_for_metadata, resolve_output_paths
from citemesh.visualization import generate_output_path
from tests._helpers import (
    build_fake_strategy_builder_factory,
    build_seed_graph,
    get_paper_id_normalization_cases,
)


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


def _run_cli_command(args: list[str]) -> tuple[int, str, str]:
    """Run CLI and capture return code/stdout/stderr.

    :param list[str] args: CLI arguments.
    :return tuple[int, str, str]: Return code, stdout, stderr.
    """
    previous_argv = sys.argv[:]
    sys.argv = ["citemesh"] + list(args)

    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                cli_module.main()
                code = 0
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.argv = previous_argv

    return code, stdout.getvalue(), stderr.getvalue()


def test_strategy_dispatches_to_matching_builder_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch should pass parsed CLI settings into strategy builders."""
    cases = [
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
    ]
    for strategy, builder_name, expected_kwargs in cases:
        captured: dict[str, object] = {}
        namespace = _dispatch_namespace()
        monkeypatch.setattr(
            cli_module,
            builder_name,
            build_fake_strategy_builder_factory(
                captured, graph=build_seed_graph("seed")
            ),
        )
        graph, seed_id = cli_module._build_strategy_graph(namespace, strategy)
        assert seed_id == "seed"
        assert graph.number_of_nodes() == 1
        assert captured == expected_kwargs


def test_build_strategy_graph_rejects_invalid_strategy() -> None:
    """Unsupported strategies should raise clear errors."""
    namespace = _dispatch_namespace()
    with pytest.raises(ValueError, match="Unsupported strategy: unknown"):
        cli_module._build_strategy_graph(namespace, "unknown")


def test_cli_help_contracts() -> None:
    """CLI help output should expose stable semantic contracts."""
    cases = [
        (["--help"], ["CiteMesh", "build", "cache", "search"]),
        (
            ["build", "--help"],
            [
                "default: recommendation",
                "--seed",
                "--all-corpus",
                "--storage-precision",
                "--binary-prefilter",
                "--binary-rescore-multiplier",
                "--calibration-sample-size",
                "--spring-iterations",
                "citation/recommendation",
            ],
        ),
        (["search", "--help"], ["search", "--limit"]),
        (["cache", "--help"], ["clear", "scan"]),
    ]
    for args, expected_tokens in cases:
        returncode, stdout, _stderr = _run_cli_command(args)
        assert returncode == 0
        lowered = stdout.lower()
        for token in expected_tokens:
            assert token.lower() in lowered


def test_cli_rejects_invalid_numeric_inputs() -> None:
    """Argparse validators should reject out-of-range numeric values."""
    cases = [
        (["build", "arxiv:1706.03762", "--max-papers", "0"], "must be at least 1"),
        (
            ["build", "arxiv:1706.03762", "--similarity-threshold", "1.2"],
            "must be between 0.0 and 1.0",
        ),
        (
            ["build", "arxiv:1706.03762", "--similarity-threshold", "nan"],
            "must be a finite float",
        ),
        (["search", "attention", "--limit", "0"], "must be at least 1"),
    ]
    for args, expected_error in cases:
        returncode, _stdout, stderr = _run_cli_command(args)
        assert returncode != 0
        assert expected_error in stderr


def test_resolve_output_paths_contract() -> None:
    """Output path resolver should preserve/replace suffixes correctly."""
    cases = [
        (
            Path("out/arxiv-2508.14040-example"),
            ["png", "html", "json"],
            True,
            {
                "png": Path("out/arxiv-2508.14040-example.png"),
                "html": Path("out/arxiv-2508.14040-example.html"),
                "json": Path("out/arxiv-2508.14040-example.json"),
            },
        ),
        (
            Path("reports/example.graphml"),
            ["png"],
            True,
            {"png": Path("reports/example.png")},
        ),
        (
            Path("reports/example.plotly.html"),
            ["plotly"],
            True,
            {"plotly": Path("reports/example.plotly.html")},
        ),
    ]
    for base_output_path, formats, explicit_output, expected in cases:
        paths = resolve_output_paths(
            base_output_path=base_output_path,
            selected_formats=formats,
            explicit_output=explicit_output,
        )
        assert paths == expected


def test_canonicalize_paper_id_for_metadata_normalizes_urls() -> None:
    """Metadata IDs should canonicalize URL-like arXiv/DOI forms."""
    for raw_id, expected in get_paper_id_normalization_cases():
        if raw_id.startswith("http://") or raw_id.startswith("https://"):
            assert canonicalize_paper_id_for_metadata(raw_id) == expected


def test_generate_output_path_includes_seed_suffix_for_collision_safety() -> None:
    """Same title with different seeds should produce unique output directories."""
    graph = nx.Graph()
    graph.add_node("seed-a", title="A Survey of Transformers")
    graph.add_node("seed-b", title="A Survey of Transformers")

    path_a = generate_output_path(graph, seed_id="seed-a", output_dir=Path("out"))
    path_b = generate_output_path(graph, seed_id="seed-b", output_dir=Path("out"))

    assert path_a.parent != path_b.parent
    assert path_a.parent.name.startswith("a-survey-of-transformers-")
    assert path_b.parent.name.startswith("a-survey-of-transformers-")
