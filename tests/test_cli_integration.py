"""Integration tests for CLI functionality."""

import io
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import networkx as nx
import pytest

from citemesh import cli as cli_module
from citemesh.core import Author, Paper
from citemesh.visualization import generate_output_path
from tests.conftest import (
    build_fake_exporter_factory,
    build_fake_strategy_builder_factory,
    build_seed_graph,
)


def run_cli_command(args: list[str]) -> SimpleNamespace:
    """Run the CLI in-process and capture stdout/stderr.

    :param list[str] args: CLI arguments to parse.
    :return SimpleNamespace: Namespace with ``returncode``, ``stdout`` and ``stderr``.
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


class TestCLIBasics:
    """Test basic CLI functionality and argument parsing."""

    def test_cli_help_works(self) -> None:
        """Test CLI help command runs without error."""
        result = run_cli_command(["--help"])
        assert result.returncode == 0
        assert "CiteMesh" in result.stdout
        assert "build" in result.stdout

    def test_build_help_works(self) -> None:
        """Test build subcommand help."""
        result = run_cli_command(["build", "--help"])
        assert result.returncode == 0
        assert "strategy" in result.stdout
        assert "recommendation" in result.stdout
        assert "citation" in result.stdout
        assert "embedding" in result.stdout
        assert "hybrid" in result.stdout

    def test_search_help_works(self) -> None:
        """Test search subcommand help."""
        result = run_cli_command(["search", "--help"])
        assert result.returncode == 0
        assert "search" in result.stdout.lower()
        assert "--limit" in result.stdout

    def test_seed_argument_exists(self) -> None:
        """Test --seed argument is exposed (caught bug: was implemented but not exposed)."""
        result = run_cli_command(["build", "--help"])
        assert result.returncode == 0
        assert "--seed" in result.stdout
        assert "reproducibility" in result.stdout.lower()

    def test_invalid_strategy_rejected(self) -> None:
        """Test invalid strategy name is rejected by argparse."""
        result = run_cli_command(
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "invalid",
            ]
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()


class TestCLIExecution:
    """Test actual CLI execution with real (but small) workloads."""

    @pytest.mark.slow
    def test_citation_strategy_runs(self) -> None:
        """Test citation strategy completes successfully with small graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_output.png"
            result = run_cli_command(
                [
                    "build",
                    "arxiv:1706.03762",
                    "--strategy",
                    "citation",
                    "-p",
                    "5",  # Only 5 papers for speed
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
            assert output.exists(), "Output PNG file not created"
            assert output.stat().st_size > 1000, "Output file suspiciously small"

    def test_citation_no_references_sets_builder_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI should pass ``--no-references`` through to citation builder config."""
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            cli_module,
            "CitationGraphBuilder",
            build_fake_strategy_builder_factory(
                captured, graph=build_seed_graph("seed")
            ),
        )
        monkeypatch.setattr(
            cli_module,
            "GraphExporter",
            build_fake_exporter_factory({}, methods=("to_json",)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_cli_no_refs.json"
            result = run_cli_command(
                [
                    "build",
                    "arxiv:1810.04805",
                    "--strategy",
                    "citation",
                    "-p",
                    "5",
                    "--no-references",
                    "--export",
                    "json",
                    "-o",
                    str(output),
                ],
            )
            assert result.returncode == 0, (
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
            assert output.exists()

        assert captured["fetch_references"] is False

    def test_embedding_strategy_passes_top_k(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI should pass embedding-specific knobs through to embedding builder."""
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            cli_module,
            "EmbeddingGraphBuilder",
            build_fake_strategy_builder_factory(
                captured, graph=build_seed_graph("seed")
            ),
        )
        monkeypatch.setattr(
            cli_module,
            "GraphExporter",
            build_fake_exporter_factory({}, methods=("to_json",)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_cli_embedding.json"
            result = run_cli_command(
                [
                    "build",
                    "arxiv:1706.03762",
                    "--strategy",
                    "embedding",
                    "--top-k",
                    "1",
                    "--truncate-dim",
                    "128",
                    "--export",
                    "json",
                    "-o",
                    str(output),
                ],
            )
            assert result.returncode == 0, (
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
            assert output.exists()

        assert captured["top_k"] == 1
        assert captured["truncate_dim"] == 128

    def test_hybrid_no_references_sets_builder_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI should pass ``--no-references`` through to hybrid builder config."""
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            cli_module,
            "HybridGraphBuilder",
            build_fake_strategy_builder_factory(
                captured, graph=build_seed_graph("seed")
            ),
        )
        monkeypatch.setattr(
            cli_module,
            "GraphExporter",
            build_fake_exporter_factory({}, methods=("to_json",)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_cli_hybrid_no_refs.json"
            result = run_cli_command(
                [
                    "build",
                    "arxiv:1810.04805",
                    "--strategy",
                    "hybrid",
                    "-p",
                    "5",
                    "--no-references",
                    "--export",
                    "json",
                    "-o",
                    str(output),
                ],
            )
            assert result.returncode == 0, (
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
            assert output.exists()

        assert captured["fetch_references"] is False

    def test_hybrid_strategy_passes_embedding_knobs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hybrid builder should receive embedding/runtime-related CLI settings."""
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            cli_module,
            "HybridGraphBuilder",
            build_fake_strategy_builder_factory(
                captured, graph=build_seed_graph("seed")
            ),
        )
        monkeypatch.setattr(
            cli_module,
            "GraphExporter",
            build_fake_exporter_factory({}, methods=("to_json",)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_cli_hybrid_knobs.json"
            result = run_cli_command(
                [
                    "build",
                    "arxiv:1706.03762",
                    "--strategy",
                    "hybrid",
                    "--max-semantic",
                    "3",
                    "--truncate-dim",
                    "128",
                    "--streaming",
                    "--all-corpus",
                    "--export",
                    "json",
                    "-o",
                    str(output),
                ],
            )
            assert result.returncode == 0, (
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
            assert output.exists()

        assert captured["max_semantic"] == 3
        assert captured["truncate_dim"] == 128
        assert captured["use_streaming"] is True
        assert captured["corpus_size"] is None

    def test_embedding_strategy_defaults_to_bounded_corpus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Embedding CLI defaults should pass a bounded corpus size."""
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            cli_module,
            "EmbeddingGraphBuilder",
            build_fake_strategy_builder_factory(
                captured, graph=build_seed_graph("seed")
            ),
        )
        monkeypatch.setattr(
            cli_module,
            "GraphExporter",
            build_fake_exporter_factory({}, methods=("to_json",)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_cli_embedding_defaults.json"
            result = run_cli_command(
                [
                    "build",
                    "arxiv:1706.03762",
                    "--strategy",
                    "embedding",
                    "--export",
                    "json",
                    "-o",
                    str(output),
                ],
            )
            assert result.returncode == 0, (
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
            assert output.exists()

        assert captured["corpus_size"] == 50000

    def test_embedding_all_corpus_removes_default_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Embedding ``--all-corpus`` should pass an uncapped corpus size."""
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            cli_module,
            "EmbeddingGraphBuilder",
            build_fake_strategy_builder_factory(
                captured, graph=build_seed_graph("seed")
            ),
        )
        monkeypatch.setattr(
            cli_module,
            "GraphExporter",
            build_fake_exporter_factory({}, methods=("to_json",)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_cli_embedding_all_corpus.json"
            result = run_cli_command(
                [
                    "build",
                    "arxiv:1706.03762",
                    "--strategy",
                    "embedding",
                    "--all-corpus",
                    "--export",
                    "json",
                    "-o",
                    str(output),
                ],
            )
            assert result.returncode == 0, (
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
            assert output.exists()

        assert captured["corpus_size"] is None

    def test_search_command_prints_results_to_stdout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Search results table should be emitted on stdout for shell piping."""
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

        assert result.returncode == 0, (
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        assert "Search results for 'attention'" in result.stdout
        assert "Full paper IDs:" in result.stdout
        assert "Use the paper ID with:" in result.stdout
        assert long_paper_id in result.stdout


class TestCLIErrorHandling:
    """Test error handling and exit codes."""

    def test_invalid_paper_id_fails_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid build errors should produce clean non-zero exits without traceback spam."""
        monkeypatch.setattr(
            cli_module,
            "build_citation_graph",
            MagicMock(side_effect=ValueError("Seed paper not found")),
        )

        result = run_cli_command(
            [
                "build",
                "this-is-not-a-real-paper-id-12345",
                "--strategy",
                "citation",
            ],
        )
        assert result.returncode != 0, "Should fail with non-zero exit code"
        assert "not found" in result.stderr.lower()
        assert "Traceback" not in result.stderr

    def test_missing_required_argument_fails(self) -> None:
        """Test missing paper_id argument fails."""
        result = run_cli_command(["build", "--strategy", "citation"])
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()


def test_default_strategy_is_recommendation() -> None:
    """Default strategy should be recommendation."""
    result = run_cli_command(["build", "--help"])
    assert result.returncode == 0
    assert "default: recommendation" in result.stdout


def test_generated_output_path_uses_paper_directory_and_strategy_basename() -> None:
    """Generated path should include stable seed suffix and strategy basename."""
    graph = nx.Graph()
    graph.add_node("seed", title="Attention Is All You Need")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_output_path(
            graph,
            seed_id="seed",
            output_dir=Path(tmpdir),
            strategy="recommendation",
        )
        assert path.parent.name.startswith("attention-is-all-you-need-")
        assert path.name == "recommendation.png"


class TestCLIDefaults:
    """Test default values match between CLI and strategy classes."""

    def test_dataset_split_default_matches(self) -> None:
        """Test dataset-split default in CLI help matches strategy (caught bug: train[:2%] vs train)."""
        result = run_cli_command(["build", "--help"])
        assert result.returncode == 0
        # Help should show full dataset default, not train[:2%]
        help_text = result.stdout.lower()
        assert "dataset-split" in help_text
        # Should mention full corpus or ~117k papers
        assert "train" in help_text
        assert "50000" in help_text
        assert "--all-corpus" in help_text
        # Should NOT have the old buggy default
        assert "[:2%]" not in help_text

    def test_seed_help_text_matches_deterministic_default(self) -> None:
        """Seed help text should describe deterministic default behavior."""
        result = run_cli_command(["build", "--help"])
        assert result.returncode == 0
        assert "deterministic built-in seed" in result.stdout

    def test_threshold_help_text_mentions_scope(self) -> None:
        """Threshold help should clarify strategy scope."""
        result = run_cli_command(["build", "--help"])
        assert result.returncode == 0
        assert "citation/recommendation" in result.stdout


class TestCLIReproducibility:
    """Test seed functionality for reproducible layout wiring."""

    def test_seed_is_threaded_to_shared_layout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Build path should compute one seeded layout and share it with exporters."""
        graph = nx.Graph()
        graph.add_node(
            "seed",
            title="Seed",
            year=2020,
            authors=[],
            citation_count=0,
            is_seed=True,
        )

        shared_layout = {"seed": (0.0, 0.0)}
        captured: dict[str, object] = {}

        monkeypatch.setattr(
            cli_module,
            "build_recommendation_graph",
            lambda args: (graph, "seed"),
        )

        def _fake_compute_layout(graph_arg, iterations, layout_seed):
            del graph_arg
            del iterations
            captured["layout_seed"] = layout_seed
            return shared_layout

        monkeypatch.setattr(
            cli_module,
            "compute_layout",
            _fake_compute_layout,
        )

        monkeypatch.setattr(
            cli_module,
            "GraphExporter",
            build_fake_exporter_factory(captured, methods=("to_json",)),
        )

        def _fake_visualize(*args, **kwargs) -> None:
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

        assert result.returncode == 0, (
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        assert captured["layout_seed"] == 123
        assert captured["layout"] is shared_layout
        assert captured["visualize_layout"] is shared_layout

    def test_json_export_skips_layout_computation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """JSON-only exports should not compute layout."""
        graph = nx.Graph()
        graph.add_node(
            "seed",
            title="Seed",
            year=2020,
            authors=[],
            citation_count=0,
            is_seed=True,
        )

        captured: dict[str, object] = {}

        monkeypatch.setattr(
            cli_module,
            "build_recommendation_graph",
            lambda args: (graph, "seed"),
        )

        def _fail_compute_layout(*args, **kwargs):
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

        assert result.returncode == 0, (
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        assert captured["layout"] is None

    def test_metadata_omits_timestamp_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI metadata should be deterministic by default (no timestamp key)."""
        graph = nx.Graph()
        graph.add_node(
            "seed",
            title="Seed",
            year=2020,
            authors=[],
            citation_count=0,
            is_seed=True,
        )

        captured: dict[str, object] = {}

        monkeypatch.setattr(
            cli_module,
            "build_recommendation_graph",
            lambda args: (graph, "seed"),
        )

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

        assert result.returncode == 0, (
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        metadata = captured["metadata"]
        assert isinstance(metadata, dict)
        assert "timestamp" not in metadata

    def test_metadata_includes_timestamp_when_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI should include timestamp only when ``--include-timestamp`` is set."""
        graph = nx.Graph()
        graph.add_node(
            "seed",
            title="Seed",
            year=2020,
            authors=[],
            citation_count=0,
            is_seed=True,
        )

        captured: dict[str, object] = {}

        monkeypatch.setattr(
            cli_module,
            "build_recommendation_graph",
            lambda args: (graph, "seed"),
        )

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
                    "--include-timestamp",
                    "--export",
                    "json",
                    "-o",
                    str(output),
                ],
            )
            assert output.exists()

        assert result.returncode == 0, (
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        metadata = captured["metadata"]
        assert isinstance(metadata, dict)
        assert "timestamp" in metadata
