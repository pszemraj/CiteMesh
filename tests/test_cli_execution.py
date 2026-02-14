"""Execution-path tests for CLI commands."""

from __future__ import annotations

import io
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
from citemesh.core import Author, Paper
from tests._helpers import build_fake_exporter_factory


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


def test_cache_clear_removes_configured_cache_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache clear should delete configured cache root when --yes is provided."""
    cache_root = tmp_path / "citemesh-cache-root"
    (cache_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (cache_root / "embeddings" / "payload.txt").write_text("cache bytes")
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(cache_root))

    result = run_cli_command(["cache", "clear", "--yes"])
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert not cache_root.exists()


def test_cache_scan_reports_usage_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache scan should print per-section and total usage stats."""
    cache_root = tmp_path / "citemesh-cache-root"
    (cache_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (cache_root / "misc").mkdir(parents=True, exist_ok=True)
    (cache_root / "references").mkdir(parents=True, exist_ok=True)
    (cache_root / "embeddings" / "vectors.bin").write_bytes(b"a" * 2048)
    (cache_root / "references" / "payload.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(cache_root))

    result = run_cli_command(["cache", "scan"])
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    for token in [
        "CiteMesh Cache Scan",
        "embeddings",
        "references",
        "TOTAL",
        "Cache root:",
    ]:
        assert token in result.stdout


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


def test_missing_required_argument_fails() -> None:
    """Missing required build args should return an error."""
    result = run_cli_command(["build", "--strategy", "citation"])
    assert result.returncode != 0
    assert "required" in result.stderr.lower() or "error" in result.stderr.lower()


def test_seed_is_threaded_to_shared_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build path should compute one seeded layout and share with exporters."""
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


def test_json_export_skips_layout_computation(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON-only export should not trigger layout computation."""
    graph = nx.Graph()
    graph.add_node(
        "seed", title="Seed", year=2020, authors=[], citation_count=0, is_seed=True
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module, "_build_strategy_graph", lambda args, strategy: (graph, "seed")
    )

    def _fail_compute_layout(
        *args: Any, **kwargs: Any
    ) -> dict[str, tuple[float, float]]:
        del args
        del kwargs
        raise AssertionError("compute_layout should not run for JSON-only export")

    monkeypatch.setattr(cli_module, "compute_layout", _fail_compute_layout)
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
