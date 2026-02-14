"""Execution-path tests for CLI commands."""

from __future__ import annotations

import argparse
import io
import runpy
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import networkx as nx
import pytest

from citemesh import cli as cli_module
from citemesh.cli import canonicalize_paper_id_for_metadata, resolve_output_paths
from citemesh.core import Author, Paper
from citemesh.visualization import generate_output_path
from tests._helpers import (
    build_fake_exporter_factory,
    build_fake_strategy_builder_factory,
    build_seed_graph,
    get_paper_id_normalization_cases,
)


def run_cli_command(args: list[str]) -> SimpleNamespace:
    """Run CLI in-process and capture stdout/stderr.

    :param list[str] args: CLI arguments.
    :return SimpleNamespace: Return code and captured streams.
    """
    previous_argv = sys.argv[:]
    sys.argv = ["citemesh"] + list(args)

    stdout = io.StringIO()
    stderr = io.StringIO()

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                cli_module.main()
                returncode = 0
            except SystemExit as exc:
                code = exc.code
                if isinstance(code, int):
                    returncode = code
                elif code is None:
                    returncode = 0
                else:
                    returncode = 1
    finally:
        sys.argv = previous_argv

    return SimpleNamespace(
        returncode=returncode,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def _dispatch_namespace() -> argparse.Namespace:
    """Build argparse namespace fixture for strategy dispatch tests."""
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


def test_cache_commands_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache clear/scan should honor configured cache root and print usage summary."""
    cache_root = tmp_path / "citemesh-cache-root"
    (cache_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (cache_root / "misc").mkdir(parents=True, exist_ok=True)
    (cache_root / "references").mkdir(parents=True, exist_ok=True)
    (cache_root / "embeddings" / "vectors.bin").write_bytes(b"a" * 2048)
    (cache_root / "references" / "payload.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(cache_root))

    scan_result = run_cli_command(["cache", "scan"])
    assert scan_result.returncode == 0, (
        f"STDOUT: {scan_result.stdout}\nSTDERR: {scan_result.stderr}"
    )
    for token in [
        "CiteMesh Cache Scan",
        "embeddings",
        "references",
        "TOTAL",
        "Cache root:",
    ]:
        assert token in scan_result.stdout

    clear_result = run_cli_command(["cache", "clear", "--yes"])
    assert clear_result.returncode == 0, (
        f"STDOUT: {clear_result.stdout}\nSTDERR: {clear_result.stderr}"
    )
    assert not cache_root.exists()


@pytest.mark.slow
@pytest.mark.integration
def test_citation_strategy_runs() -> None:
    """Citation strategy should complete successfully with a small graph."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "test_output.png"
        result = run_cli_command(
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "citation",
                "-p",
                "5",
                "-c",
                "3",
                "-r",
                "3",
                "--seed",
                "42",
                "-o",
                str(output),
            ],
        )
        assert result.returncode == 0, (
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        assert output.exists()
        assert output.stat().st_size > 1000


def test_search_command_prints_results_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search should print table + full IDs for shell workflows."""
    long_paper_id = "0123456789abcdef0123456789abcdef01234567"
    mock_client = MagicMock()
    mock_client.search_papers.return_value = [
        Paper(
            paper_id=long_paper_id,
            title="Attention Is All You Need",
            year=2017,
            authors=[Author(name="Ashish Vaswani")],
            citation_count=12345,
            abstract="Transformer model paper",
        )
    ]
    monkeypatch.setattr(cli_module, "get_client", lambda: mock_client)

    result = run_cli_command(["search", "attention", "--limit", "1"])
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "Search results for 'attention'" in result.stdout
    assert "Full paper IDs:" in result.stdout
    assert long_paper_id in result.stdout


def test_invalid_paper_id_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid build errors should produce a clean non-zero exit."""
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)
    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        MagicMock(side_effect=ValueError("Seed paper not found")),
    )

    result = run_cli_command(
        ["build", "this-is-not-a-real-paper-id-12345", "--strategy", "citation"]
    )
    assert result.returncode != 0
    assert error_mock.call_count == 1
    assert "Seed paper not found" in str(error_mock.call_args)
    assert "Traceback" not in result.stderr


def test_cli_argument_validation_contracts() -> None:
    """Missing args and numeric validators should fail with clear messages."""
    result = run_cli_command(["build", "--strategy", "citation"])
    assert result.returncode != 0
    assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    numeric_cases = [
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
    for args, expected_error in numeric_cases:
        result = run_cli_command(args)
        assert result.returncode != 0
        assert expected_error in result.stderr


def test_layout_and_json_export_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build path should share seeded layout and skip it for JSON-only export."""
    graph = nx.Graph()
    graph.add_node(
        "seed", title="Seed", year=2020, authors=[], citation_count=0, is_seed=True
    )

    shared_layout = {"seed": (0.0, 0.0)}
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module, "_build_strategy_graph", lambda args, strategy: (graph, "seed")
    )

    def _fake_compute_layout(
        graph_arg: nx.Graph, iterations: int, layout_seed: int | None
    ) -> dict[str, tuple[float, float]]:
        del graph_arg
        del iterations
        captured["layout_seed"] = layout_seed
        return shared_layout

    monkeypatch.setattr(cli_module, "compute_layout", _fake_compute_layout)
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        build_fake_exporter_factory(captured, methods=("to_json",)),
    )

    def _fake_visualize(*args: Any, **kwargs: Any) -> None:
        del args
        captured["visualize_layout"] = kwargs.get("layout")

    monkeypatch.setattr(cli_module, "visualize_graph", _fake_visualize)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "seeded.png"
        result = run_cli_command(
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "recommendation",
                "--seed",
                "123",
                "--export",
                "png",
                "-o",
                str(output),
            ],
        )

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert captured["layout_seed"] == 123
    assert captured["layout"] is shared_layout
    assert captured["visualize_layout"] is shared_layout

    def _fail_compute_layout(
        *args: Any, **kwargs: Any
    ) -> dict[str, tuple[float, float]]:
        del args
        del kwargs
        raise AssertionError("compute_layout should not run for JSON-only export")

    monkeypatch.setattr(cli_module, "compute_layout", _fail_compute_layout)
    captured.clear()
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        build_fake_exporter_factory(captured, methods=("to_json",)),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "graph.json"
        result = run_cli_command(
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "recommendation",
                "--export",
                "json",
                "-o",
                str(output),
            ],
        )
        assert output.exists()

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert captured["layout"] is None


def test_metadata_timestamp_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metadata timestamp should be opt-in only."""
    for include_timestamp, expected_key in [(False, False), (True, True)]:
        graph = nx.Graph()
        graph.add_node(
            "seed", title="Seed", year=2020, authors=[], citation_count=0, is_seed=True
        )

        captured: dict[str, object] = {}
        monkeypatch.setattr(
            cli_module, "_build_strategy_graph", lambda args, strategy: (graph, "seed")
        )
        monkeypatch.setattr(
            cli_module,
            "GraphExporter",
            build_fake_exporter_factory(captured, methods=("to_json",)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "graph.json"
            args = [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "recommendation",
                "--export",
                "json",
                "-o",
                str(output),
            ]
            if include_timestamp:
                args.insert(4, "--include-timestamp")
            result = run_cli_command(args)
            assert output.exists()

        assert result.returncode == 0, (
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        metadata = captured["metadata"]
        assert isinstance(metadata, dict)
        assert ("timestamp" in metadata) is expected_key


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


def test_build_strategy_graph_invalid_and_lazy_exports_contract() -> None:
    """Unsupported strategies should error and top-level lazy exports should resolve."""
    namespace = _dispatch_namespace()
    with pytest.raises(ValueError, match="Unsupported strategy: unknown"):
        cli_module._build_strategy_graph(namespace, "unknown")

    import citemesh

    assert citemesh.CitationGraphBuilder.__name__ == "CitationGraphBuilder"
    assert citemesh.RecommendationGraphBuilder.__name__ == "RecommendationGraphBuilder"
    assert citemesh.EmbeddingGraphBuilder.__name__ == "EmbeddingGraphBuilder"
    assert citemesh.HybridGraphBuilder.__name__ == "HybridGraphBuilder"


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
        result = run_cli_command(args)
        assert result.returncode == 0
        lowered = result.stdout.lower()
        for token in expected_tokens:
            assert token.lower() in lowered


def test_output_path_and_slug_contracts() -> None:
    """Output path resolver and auto-output slug generation should stay stable."""
    path_cases = [
        (
            Path("out/arxiv-2508.14040-example"),
            ["png", "html", "json"],
            True,
            {
                "png": Path("out/arxiv-2508.14040-example/hybrid.png"),
                "html": Path("out/arxiv-2508.14040-example/hybrid.html"),
                "json": Path("out/arxiv-2508.14040-example/hybrid.json"),
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
    for base_output_path, formats, explicit_output, expected in path_cases:
        paths = resolve_output_paths(
            base_output_path=base_output_path,
            selected_formats=formats,
            explicit_output=explicit_output,
            strategy="hybrid",
        )
        assert paths == expected

    for raw_id, expected in get_paper_id_normalization_cases():
        if raw_id.startswith("http://") or raw_id.startswith("https://"):
            assert canonicalize_paper_id_for_metadata(raw_id) == expected

    graph = nx.Graph()
    graph.add_node("seed-a", title="A Survey of Transformers")
    graph.add_node("seed-b", title="A Survey of Transformers")
    path_a = generate_output_path(graph, seed_id="seed-a", output_dir=Path("out"))
    path_b = generate_output_path(graph, seed_id="seed-b", output_dir=Path("out"))
    assert path_a.parent == path_b.parent
    assert path_a.parent.name == "a-survey-of-transformers"

    graph = nx.Graph()
    graph.add_node(
        "seed",
        title="This title should definitely exceed forty characters for the slug",
    )
    output_path = generate_output_path(graph, seed_id="seed", output_dir=Path("out"))
    assert len(output_path.parent.name) <= 40


def test_main_module_invokes_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running ``citemesh.__main__`` should invoke ``citemesh.cli.main``."""
    called = {"main": False}

    def fake_main() -> None:
        called["main"] = True

    monkeypatch.setattr("citemesh.cli.main", fake_main)
    runpy.run_module("citemesh.__main__", run_name="__main__")
    assert called["main"] is True
