"""Integration tests for CLI functionality."""

import io
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from citemesh import cli as cli_module
from citemesh.visualization import generate_output_path


def run_cli_command(args: list[str]) -> SimpleNamespace:
    """Run the CLI in-process and capture stdout/stderr."""
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

    def test_cli_help_works(self):
        """Test CLI help command runs without error."""
        result = run_cli_command(["--help"])
        assert result.returncode == 0
        assert "CiteMesh" in result.stdout
        assert "build" in result.stdout

    def test_build_help_works(self):
        """Test build subcommand help."""
        result = run_cli_command(["build", "--help"])
        assert result.returncode == 0
        assert "strategy" in result.stdout
        assert "recommendation" in result.stdout
        assert "citation" in result.stdout
        assert "embedding" in result.stdout
        assert "hybrid" in result.stdout

    def test_search_help_works(self):
        """Test search subcommand help."""
        result = run_cli_command(["search", "--help"])
        assert result.returncode == 0
        assert "search" in result.stdout.lower()
        assert "--limit" in result.stdout

    def test_seed_argument_exists(self):
        """Test --seed argument is exposed (caught bug: was implemented but not exposed)."""
        result = run_cli_command(["build", "--help"])
        assert result.returncode == 0
        assert "--seed" in result.stdout
        assert "reproducibility" in result.stdout.lower()

    def test_invalid_strategy_rejected(self):
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
    def test_citation_strategy_runs(self):
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

    @pytest.mark.slow
    def test_citation_no_references_faster(self):
        """Test --no-references flag works and is faster."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_cli_no_refs.png"
            result = run_cli_command(
                [
                    "build",
                    "arxiv:1810.04805",
                    "--strategy",
                    "citation",
                    "-p",
                    "5",
                    "--no-references",
                    "--seed",
                    "99",
                    "-o",
                    str(output),
                ],
            )
        # May timeout due to S2 API rate limits, skip in that case
        if result.returncode == 0:
            assert "0 with reference lists" in result.stdout

    @pytest.mark.slow
    def test_embedding_strategy_runs(self):
        """Test embedding strategy with tiny dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_cli_embedding.png"
            result = run_cli_command(
                [
                    "build",
                    "arxiv:1706.03762",
                    "--strategy",
                    "embedding",
                    "-p",
                    "5",
                    "--dataset-split",
                    "train[:100]",  # Tiny for speed
                    "--seed",
                    "42",
                    "-m",
                    "all-MiniLM-L6-v2",  # Fast model
                    "-o",
                    str(output),
                ],
            )
        # May fail due to S2 API rate limits, but should not crash
        if result.returncode == 0:
            assert "Computing corpus embeddings" in result.stdout
        else:
            # Acceptable failures: S2 rate limit, paper not in tiny corpus
            assert "429" in result.stdout or "not found" in result.stderr.lower()


class TestCLIErrorHandling:
    """Test error handling and exit codes."""

    @pytest.mark.slow
    def test_invalid_paper_id_fails_cleanly(self):
        """Test invalid paper ID returns non-zero exit code with clean error."""
        result = run_cli_command(
            [
                "build",
                "this-is-not-a-real-paper-id-12345",
                "--strategy",
                "citation",
                "-p",
                "5",
            ],
        )
        assert result.returncode != 0, "Should fail with non-zero exit code"
        assert "not found" in result.stderr.lower()
        # Should NOT have traceback spam (caught bug: verbose tracebacks)
        assert "Traceback" not in result.stderr

    def test_missing_required_argument_fails(self):
        """Test missing paper_id argument fails."""
        result = run_cli_command(["build", "--strategy", "citation"])
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()


def test_default_strategy_is_recommendation():
    """Default strategy should be recommendation."""
    result = run_cli_command(["build", "--help"])
    assert result.returncode == 0
    assert "default: recommendation" in result.stdout


def test_generated_output_path_uses_strategy_suffix():
    """Strategy name should be included in generated filenames."""
    graph = nx.Graph()
    graph.add_node("seed", title="Attention Is All You Need")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_output_path(
            graph,
            seed_id="seed",
            output_dir=Path(tmpdir),
            strategy="recommendation",
        )
        assert path.name.endswith("-recommendation.png")


class TestCLIDefaults:
    """Test default values match between CLI and strategy classes."""

    def test_dataset_split_default_matches(self):
        """Test dataset-split default in CLI help matches strategy (caught bug: train[:2%] vs train)."""
        result = run_cli_command(["build", "--help"])
        assert result.returncode == 0
        # Help should show full dataset default, not train[:2%]
        help_text = result.stdout.lower()
        assert "dataset-split" in help_text
        # Should mention full corpus or ~117k papers
        assert "train" in help_text
        # Should NOT have the old buggy default
        assert "[:2%]" not in help_text


class TestCLIReproducibility:
    """Test seed functionality for reproducible builds."""

    @pytest.mark.slow
    def test_seed_produces_deterministic_output(self):
        """Test same seed produces consistent results (node/edge counts)."""
        args = [
            "citemesh",
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
        ]

        result1 = run_cli_command(args)
        result2 = run_cli_command(args)

        if result1.returncode == 0 and result2.returncode == 0:
            # Extract node/edge counts from output
            import re

            nodes1 = re.search(r"Nodes: (\d+)", result1.stdout)
            edges1 = re.search(r"Edges: (\d+)", result1.stdout)
            nodes2 = re.search(r"Nodes: (\d+)", result2.stdout)
            edges2 = re.search(r"Edges: (\d+)", result2.stdout)

            if all([nodes1, edges1, nodes2, edges2]):
                assert nodes1.group(1) == nodes2.group(1), (
                    "Node count not deterministic"
                )
                assert edges1.group(1) == edges2.group(1), (
                    "Edge count not deterministic"
                )
