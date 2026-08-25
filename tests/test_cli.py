"""Execution-path tests for CLI commands."""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import runpy
import shlex
import tempfile
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import networkx as nx
import pytest

from citemesh import cli as cli_module
from citemesh.cli import (
    _is_standalone_dashboard_output,
    build_dashboard_collection_bundle,
    canonicalize_paper_id_for_metadata,
    resolve_dashboard_collection_outputs,
    resolve_graph_config_path,
    resolve_output_paths,
)
from citemesh.core import Author, Paper
from citemesh.data import DEFAULT_EMBEDDING_MODEL_NAME
from citemesh.strategies.embedding import ENCODE_BATCH_SIZE
from citemesh.strategies.hybrid import (
    DEFAULT_MAX_SEMANTIC,
    HYBRID_DEFAULT_MAX_CITATIONS,
    HYBRID_DEFAULT_MAX_PAPERS,
    HYBRID_DEFAULT_MAX_REFERENCES,
)
from citemesh.visualization import generate_output_path
from tests._helpers import (
    build_seed_graph,
    get_paper_id_normalization_cases,
)


def run_cli_command(args: list[str]) -> SimpleNamespace:
    """Run CLI in-process and capture stdout/stderr.

    :param list[str] args: CLI arguments.
    :return SimpleNamespace: Return code and captured streams.
    """
    return _run_captured_cli(lambda: cli_module.main(args))


def run_cli_command_via_sys_argv(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> SimpleNamespace:
    """Run CLI through ``sys.argv`` to exercise ``main(argv=None)``."""
    monkeypatch.setattr("sys.argv", ["citemesh", *args])
    return _run_captured_cli(cli_module.main)


def _run_captured_cli(entrypoint: Any) -> SimpleNamespace:
    """Run a CLI entrypoint and capture stdout/stderr."""
    stdout = io.StringIO()
    stderr = io.StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            returncode = entrypoint()
        except SystemExit as exc:
            code = exc.code
            if isinstance(code, int):
                returncode = code
            elif code is None:
                returncode = 0
            else:
                returncode = 1

    return SimpleNamespace(
        returncode=returncode,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def _make_builder_stub(
    captured_kwargs: dict[str, object],
    *,
    graph: nx.Graph | None = None,
    seed_id: str = "seed",
) -> Any:
    """Create a lightweight strategy-builder stub for CLI dispatch tests."""
    base_graph = graph if graph is not None else build_seed_graph(seed_id)

    def _factory(**kwargs: object) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(build_graph=lambda _paper_id: (base_graph, seed_id))

    return _factory


def _make_exporter_stub(
    captured_data: dict[str, object], *, methods: tuple[str, ...] | None = None
) -> Any:
    """Create a lightweight exporter stub for CLI artifact tests."""
    requested_methods = set(methods or tuple(cli_module._EXPORTER_METHOD.values()))
    payloads = {
        "to_json": "{}",
        "to_interactive_html": "<html/>",
        "to_plotly_html": "<html/>",
        "to_dashboard_html": "<html/>",
        "to_graphml": "<graphml/>",
        "to_csv": "id,title\n",
        "to_bibtex": "@article{test,}\n",
    }

    def _factory(*_args: object, **kwargs: object) -> SimpleNamespace:
        captured_data["kwargs"] = kwargs
        captured_data["metadata"] = kwargs.get("metadata")
        captured_data["layout"] = kwargs.get("layout")

        def _write_payload(
            path: Path,
            method_name: str,
            *_method_args: object,
            **_method_kwargs: object,
        ) -> None:
            del _method_args, _method_kwargs
            if method_name in requested_methods:
                path.write_text(payloads[method_name], encoding="utf-8")

        return SimpleNamespace(
            **{
                method_name: (
                    lambda path, *args, _method=method_name, **kwargs: _write_payload(
                        path,
                        _method,
                        *args,
                        **kwargs,
                    )
                )
                for method_name in payloads
            }
        )

    return _factory


def _dispatch_namespace(**overrides: object) -> argparse.Namespace:
    """Build argparse namespace fixture for strategy dispatch tests."""
    values = {
        "paper_id": "seed",
        "max_papers": 11,
        "max_citations": 25,
        "max_references": 25,
        "similarity_threshold": 0.2,
        "no_references": False,
        "refresh_reference_cache": False,
        "model": DEFAULT_EMBEDDING_MODEL_NAME,
        "model_revision": None,
        "dataset_split": "train",
        "corpus_size": 50000,
        "all_corpus": False,
        "top_k": 4,
        "truncate_dim": None,
        "streaming": False,
        "max_semantic": None,
        "seed": 7,
        "force_rebuild_cache": False,
        "overwrite_cache": False,
        "cache_overwrite_reason": None,
        "storage_precision": "int8",
        "binary_prefilter": True,
        "binary_rescore_multiplier": 8,
        "calibration_sample_size": 2000,
        "cache_compression": "gzip",
        "cache_compression_level": 1,
        "encode_batch_size": ENCODE_BATCH_SIZE,
        "torch_compile": False,
        "device": "auto",
        "semantic_source": "candidates",
        "candidate_pool_size": 400,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _reset_cli_logging_state() -> tuple[list[logging.Handler], int, bool]:
    """Reset root/CLI logging state for direct logging configuration tests."""
    root_logger = logging.getLogger()
    saved_handlers = list(root_logger.handlers)
    saved_level = int(root_logger.level)
    saved_configured = bool(cli_module._LOGGING_CONFIGURED)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    cli_module._LOGGING_CONFIGURED = False
    return saved_handlers, saved_level, saved_configured


def _restore_cli_logging_state(
    saved_handlers: list[logging.Handler], saved_level: int, saved_configured: bool
) -> None:
    """Restore root/CLI logging state after direct logging configuration tests."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    for handler in saved_handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(saved_level)
    cli_module._LOGGING_CONFIGURED = saved_configured


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

    scan_debug_result = run_cli_command(["cache", "scan", "--log-level", "debug"])
    assert scan_debug_result.returncode == 0, (
        f"STDOUT: {scan_debug_result.stdout}\nSTDERR: {scan_debug_result.stderr}"
    )
    assert "CiteMesh Cache Scan" in scan_debug_result.stdout

    clear_result = run_cli_command(
        ["cache", "clear", "--yes", "--reason", "manual local reset"]
    )
    assert clear_result.returncode == 0, (
        f"STDOUT: {clear_result.stdout}\nSTDERR: {clear_result.stderr}"
    )
    assert not cache_root.exists()


def test_force_rebuild_cache_confirmation_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force-rebuild should require explicit confirmation or overwrite acknowledgement."""
    graph = build_seed_graph("seed")

    build_graph_mock = MagicMock(return_value=(graph, "seed"))
    monkeypatch.setattr(cli_module, "_build_strategy_graph", build_graph_mock)
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        _make_exporter_stub({}, methods=("to_json",)),
    )

    monkeypatch.setattr(cli_module, "stdin_isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    cancelled = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "embedding",
            "--force-rebuild-cache",
            "--export",
            "json",
            "-o",
            str(Path("out") / "cancelled.json"),
        ]
    )
    assert cancelled.returncode != 0
    assert build_graph_mock.call_count == 0

    monkeypatch.setattr(cli_module, "stdin_isatty", lambda: False)
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)
    non_interactive = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "embedding",
            "--force-rebuild-cache",
            "--export",
            "json",
            "-o",
            str(Path("out") / "non-interactive.json"),
        ]
    )
    assert non_interactive.returncode != 0
    assert any("--overwrite-cache" in str(call) for call in error_mock.call_args_list)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "graph.json"
        acknowledged = run_cli_command(
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--force-rebuild-cache",
                "--overwrite-cache",
                "--export",
                "json",
                "-o",
                str(output),
            ]
        )
        assert output.exists()

    assert acknowledged.returncode == 0, (
        f"STDOUT: {acknowledged.stdout}\nSTDERR: {acknowledged.stderr}"
    )


def test_cli_logging_flags_are_position_agnostic() -> None:
    """Logging options should parse identically before/after subcommands."""
    parser, _, _, _ = cli_module._create_parser()
    cases = [
        ["--log-level", "debug", "build", "arxiv:1706.03762"],
        ["build", "arxiv:1706.03762", "--log-level", "debug"],
        ["--log-level", "debug", "cache", "scan"],
        ["cache", "--log-level", "debug", "scan"],
        ["cache", "scan", "--log-level", "debug"],
        ["--log-width", "0", "build", "arxiv:1706.03762"],
        ["build", "arxiv:1706.03762", "--log-width", "0"],
        ["--log-file", "run.log", "build", "arxiv:1706.03762"],
        ["build", "arxiv:1706.03762", "--log-file", "run.log"],
    ]

    for argv in cases:
        parsed = parser.parse_args(argv)
        if "--log-level" in argv:
            assert parsed.log_level == "debug"
        if "--log-width" in argv:
            assert parsed.log_width == 0
        if "--log-file" in argv:
            assert parsed.log_file == "run.log"


def test_resolve_console_width_uses_auto_width_for_tty_streams() -> None:
    """TTY streams should default Rich consoles to auto width."""
    assert cli_module._resolve_console_width(0, interactive=True) is None


def test_resolve_console_width_uses_fixed_width_for_redirected_streams() -> None:
    """Redirected streams should keep a stable fallback width by default."""
    assert (
        cli_module._resolve_console_width(0, interactive=False)
        == cli_module.REDIRECTED_LOG_WIDTH
    )
    assert cli_module._resolve_console_width(96, interactive=True) == 96


def test_configure_logging_writes_plaintext_log_file(tmp_path: Path) -> None:
    """File logging should keep debug details off the console by default."""
    saved_handlers, saved_level, saved_configured = _reset_cli_logging_state()
    log_path = tmp_path / "logs" / "cli-debug.log"
    stderr = io.StringIO()
    noisy_logger_names = ("filelock", "matplotlib", "urllib3", "semanticscholar")
    saved_logger_levels = {
        name: logging.getLogger(name).level for name in noisy_logger_names
    }

    try:
        with redirect_stderr(stderr):
            cli_module._configure_logging(
                log_level="debug",
                log_width=0,
                log_file=str(log_path),
            )
            cli_module.logger.debug("debug file sink test")
            cli_module.logger.info("info file sink test")
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                handler.flush()
            assert logging.getLogger("filelock").level == logging.WARNING
            assert logging.getLogger("matplotlib").level == logging.WARNING
            assert logging.getLogger("urllib3").level == logging.WARNING
            assert logging.getLogger("semanticscholar").level == logging.WARNING
    finally:
        for name, level in saved_logger_levels.items():
            logging.getLogger(name).setLevel(level)
        _restore_cli_logging_state(saved_handlers, saved_level, saved_configured)

    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "DEBUG" in content
    assert "debug file sink test" in content
    assert "INFO" in content
    assert "info file sink test" in content
    assert "\x1b[" not in content
    assert "debug file sink test" not in stderr.getvalue()
    assert "info file sink test" in stderr.getvalue()


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
        (["search", ""], "must be a non-empty string"),
        (["build", "", "--strategy", "citation"], "must be a non-empty string"),
    ]
    for args, expected_error in numeric_cases:
        result = run_cli_command(args)
        assert result.returncode != 0
        assert expected_error in result.stderr


def test_cli_rejects_strategy_incompatible_options() -> None:
    """Build should reject unsupported options for each strategy and argv shape."""
    cases = [
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "citation",
                "--model",
                "all-MiniLM-L6-v2",
            ],
            "--model",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "citation",
                "-mall-MiniLM-L6-v2",
            ],
            "--model",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "recommendation",
                "--max-citations",
                "10",
            ],
            "--max-citations",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "recommendation",
                "-k4",
            ],
            "--top-k",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "recommendation",
                "--max-s",
                "4",
            ],
            "--max-semantic",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "hybrid",
                "--top-k",
                "4",
            ],
            "--top-k",
        ),
        (
            [
                "--log-level",
                "debug",
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "citation",
                "--top-k",
                "4",
            ],
            "--top-k",
        ),
    ]
    for args, token in cases:
        result = run_cli_command(args)
        assert result.returncode != 0
        assert "Unsupported option(s)" in result.stderr
        assert token in result.stderr


def test_cli_validates_embedding_option_dependencies_at_parse_time() -> None:
    """Build should fail fast for invalid embedding/hybrid option combinations."""
    cases = [
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--streaming",
                "--dataset-split",
                "train[:5%]",
            ],
            "Streaming mode does not support sliced --dataset-split",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "hybrid",
                "--max-papers",
                "5",
                "--max-semantic",
                "5",
            ],
            "--max-semantic must be between 0 and --max-papers - 1",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--storage-precision",
                "float32",
                "--binary-prefilter",
            ],
            "--binary-prefilter requires --storage-precision int8",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--storage-precision",
                "float32",
                "--binary-rescore-multiplier",
                "5",
            ],
            "--binary-rescore-multiplier requires --storage-precision int8",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--all-corpus",
                "--corpus-size",
                "200",
            ],
            "--all-corpus cannot be combined with explicit --corpus-size",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--overwrite-cache",
            ],
            "--overwrite-cache requires --force-rebuild-cache",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--cache-overwrite-reason",
                "sync stale branch cache",
            ],
            "--cache-overwrite-reason requires --force-rebuild-cache",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--storage-precision",
                "float32",
                "--calibration-sample-size",
                "512",
            ],
            "--calibration-sample-size requires --storage-precision int8",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "hybrid",
                "--max-semantic",
                "0",
                "--model",
                "all-MiniLM-L6-v2",
            ],
            "Hybrid semantic branch is disabled with --max-semantic 0",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "hybrid",
                "--max-semantic",
                "0",
                "--model",
                DEFAULT_EMBEDDING_MODEL_NAME,
            ],
            "Hybrid semantic branch is disabled with --max-semantic 0",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--cache-compression",
                "lzf",
                "--cache-compression-level",
                "0",
            ],
            "--cache-compression-level is unsupported with --cache-compression lzf",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--cache-compression",
                "brotli",
            ],
            "invalid choice",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--cache-compression",
                "szip",
            ],
            "invalid choice",
        ),
    ]
    for args, token in cases:
        result = run_cli_command(args)
        assert result.returncode != 0
        assert token in result.stderr


@pytest.mark.parametrize(
    ("args", "expected_tokens"),
    [
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "citation",
                "--storage-precision",
                "int8",
            ],
            ["Unsupported option(s)", "--storage-precision"],
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--all-corpus",
                "--corpus-size",
                "50000",
            ],
            ["--all-corpus cannot be combined with explicit --corpus-size"],
        ),
    ],
)
def test_main_without_argv_preserves_explicit_default_valued_flags(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    expected_tokens: list[str],
) -> None:
    """``main()`` should validate explicit default-valued flags from ``sys.argv``."""
    result = run_cli_command_via_sys_argv(
        monkeypatch,
        args,
    )

    assert result.returncode != 0
    for token in expected_tokens:
        assert token in result.stderr


def test_hybrid_allows_embedding_options_when_max_semantic_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid should accept embedding options when default semantic budget is non-zero."""
    graph = build_seed_graph("seed")

    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "graph.json"
        result = run_cli_command(
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "hybrid",
                "--model",
                "all-MiniLM-L6-v2",
                "--dataset-split",
                "train",
                "--export",
                "json",
                "-o",
                str(output),
            ]
        )
        assert output.exists()

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert (
        "Hybrid semantic branch is disabled with --max-semantic 0" not in result.stderr
    )


def test_embedding_lzf_compression_level_normalization_contract() -> None:
    """Embedding validation should normalize implicit lzf compression level to 0."""
    _, build_parser, _, _ = cli_module._create_parser()
    args = build_parser.parse_args(
        ["seed", "--strategy", "embedding", "--cache-compression", "lzf"]
    )
    cli_module._validate_build_cli_contract(
        args, build_parser, provided={"cache_compression"}
    )
    assert args.cache_compression == "lzf"
    assert args.cache_compression_level == 0


def test_hybrid_implicit_budget_defaults_contract() -> None:
    """Hybrid should apply tuned defaults only when budget knobs are omitted."""
    _, build_parser, _, _ = cli_module._create_parser()
    hybrid_defaults = build_parser.parse_args(["seed", "--strategy", "hybrid"])
    cli_module._validate_build_cli_contract(
        hybrid_defaults, build_parser, provided=set()
    )
    assert hybrid_defaults.max_papers == HYBRID_DEFAULT_MAX_PAPERS
    assert hybrid_defaults.max_citations == HYBRID_DEFAULT_MAX_CITATIONS
    assert hybrid_defaults.max_references == HYBRID_DEFAULT_MAX_REFERENCES
    assert cli_module._resolved_hybrid_max_semantic(hybrid_defaults) == min(
        DEFAULT_MAX_SEMANTIC, HYBRID_DEFAULT_MAX_PAPERS - 1
    )

    explicit_hybrid = build_parser.parse_args(
        [
            "seed",
            "--strategy",
            "hybrid",
            "--max-papers",
            "30",
            "--max-citations",
            "6",
            "--max-references",
            "7",
            "--max-semantic",
            "5",
        ]
    )
    cli_module._validate_build_cli_contract(
        explicit_hybrid,
        build_parser,
        provided={"max_papers", "max_citations", "max_references", "max_semantic"},
    )
    assert explicit_hybrid.max_papers == 30
    assert explicit_hybrid.max_citations == 6
    assert explicit_hybrid.max_references == 7
    assert cli_module._resolved_hybrid_max_semantic(explicit_hybrid) == 5


def test_layout_and_json_export_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build path should share seeded layout and skip it for JSON-only export."""
    graph = build_seed_graph("seed")

    shared_layout = {"seed": (0.0, 0.0)}
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
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
        _make_exporter_stub(captured, methods=("to_json",)),
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
        _make_exporter_stub(captured, methods=("to_json",)),
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


def test_dashboard_export_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dashboard exports should use shared shell + per-run collection artifacts."""
    graph = build_seed_graph("seed")
    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        _make_exporter_stub(
            captured,
            methods=("to_dashboard_html", "to_json"),
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "graph"
        result = run_cli_command(
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "recommendation",
                "--export",
                "dashboard",
                "-o",
                str(output),
            ],
        )
        run_dir = generate_output_path(
            graph,
            seed_id="seed",
            output_dir=output,
            strategy="recommendation",
        ).parent
        assert (output / "dashboard.html").exists()
        assert (output / "dashboard.manifest.json").exists()
        assert (run_dir / "recommendation.json").exists()
        assert (run_dir / "recommendation.config.json").exists()
        collection_bundle = captured["metadata"]["dashboard_collection"]
        assert collection_bundle["current_result_id"] == "recommendation:seed"
        assert collection_bundle["results"][0]["json_path"].endswith(
            "recommendation.json"
        )
        assert "recommendation:seed" in collection_bundle["payloads"]
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"

    captured.clear()
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        _make_exporter_stub(
            captured,
            methods=(
                "to_dashboard_html",
                "to_json",
                "to_csv",
                "to_bibtex",
                "to_graphml",
                "to_interactive_html",
                "to_plotly_html",
            ),
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "exports"
        result = run_cli_command(
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "recommendation",
                "--export",
                "all",
                "-o",
                str(output_dir),
            ],
        )
        run_dir = generate_output_path(
            graph,
            seed_id="seed",
            output_dir=output_dir,
            strategy="recommendation",
        ).parent
        assert (output_dir / "dashboard.html").exists()
        assert (output_dir / "dashboard.manifest.json").exists()
        assert not (run_dir / "recommendation.dashboard.html").exists()
        assert (run_dir / "recommendation.csv").exists()
        assert (run_dir / "recommendation.bib").exists()
        config_files = sorted(output_dir.rglob("*.config.json"))
        assert len(config_files) == 1
        config_payload = json.loads(config_files[0].read_text())
        assert "dashboard" in config_payload["outputs"]
        assert "csv" in config_payload["outputs"]
        assert "bibtex" in config_payload["outputs"]
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_dashboard_collection_manifest_tracks_multiple_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared dashboard collections should retain multiple run payload entries."""
    first_graph = build_seed_graph("seed-a")
    first_graph.nodes["seed-a"]["title"] = "First Seed"
    second_graph = build_seed_graph("seed-b")
    second_graph.nodes["seed-b"]["title"] = "Second Seed"
    captured: dict[str, object] = {}
    build_results = iter([(first_graph, "seed-a"), (second_graph, "seed-b")])
    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: next(build_results),
    )
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        _make_exporter_stub(
            captured,
            methods=("to_dashboard_html", "to_json"),
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "collection"
        first_result = run_cli_command(
            [
                "build",
                "arxiv:1111.1111",
                "--strategy",
                "recommendation",
                "--export",
                "dashboard",
                "-o",
                str(output_dir),
            ],
        )
        second_result = run_cli_command(
            [
                "build",
                "arxiv:2222.2222",
                "--strategy",
                "recommendation",
                "--export",
                "dashboard",
                "-o",
                str(output_dir),
            ],
        )

        manifest_payload = json.loads(
            (output_dir / "dashboard.manifest.json").read_text()
        )
        assert len(manifest_payload["results"]) == 2
        json_paths = {entry["json_path"] for entry in manifest_payload["results"]}
        assert len(json_paths) == 2
        first_run_dir = generate_output_path(
            first_graph,
            seed_id="seed-a",
            output_dir=output_dir,
            strategy="recommendation",
        ).parent
        second_run_dir = generate_output_path(
            second_graph,
            seed_id="seed-b",
            output_dir=output_dir,
            strategy="recommendation",
        ).parent
        assert (output_dir / "dashboard.html").exists()
        assert (first_run_dir / "recommendation.json").exists()
        assert (second_run_dir / "recommendation.json").exists()
        collection_bundle = captured["metadata"]["dashboard_collection"]
        assert collection_bundle["current_result_id"] == "recommendation:seed-b"
        assert len(collection_bundle["results"]) == 2
        assert set(collection_bundle["payloads"]) == {
            "recommendation:seed-a",
            "recommendation:seed-b",
        }

    assert first_result.returncode == 0
    assert second_result.returncode == 0


def test_dashboard_collection_manifest_refreshes_same_seed_strategy_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running the same seed/strategy should refresh the existing selector slot."""
    seed_graph = build_seed_graph("seed")
    updated_graph = build_seed_graph("seed")
    updated_graph.add_node(
        "extra",
        title="Extra Paper",
        year=2025,
        authors=["Author B"],
        citation_count=1,
    )
    updated_graph.add_edge("seed", "extra", weight=0.4)

    captured: dict[str, object] = {}
    build_results = iter([(seed_graph, "seed"), (updated_graph, "seed")])
    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: next(build_results),
    )
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        _make_exporter_stub(
            captured,
            methods=("to_dashboard_html", "to_json"),
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "collection"
        first_result = run_cli_command(
            [
                "build",
                "arxiv:1111.1111",
                "--strategy",
                "recommendation",
                "--export",
                "dashboard",
                "-o",
                str(output_dir),
            ],
        )
        second_result = run_cli_command(
            [
                "build",
                "arxiv:1111.1111",
                "--strategy",
                "recommendation",
                "--export",
                "dashboard",
                "-o",
                str(output_dir),
            ],
        )

        manifest_payload = json.loads(
            (output_dir / "dashboard.manifest.json").read_text()
        )
        assert len(manifest_payload["results"]) == 1
        entry = manifest_payload["results"][0]
        assert entry["result_id"] == "recommendation:seed"
        assert entry["summary"] == {"nodes": 2, "edges": 1}
        collection_bundle = captured["metadata"]["dashboard_collection"]
        assert collection_bundle["current_result_id"] == "recommendation:seed"
        assert len(collection_bundle["results"]) == 1

    assert first_result.returncode == 0
    assert second_result.returncode == 0


def test_dashboard_collection_manifest_serializes_concurrent_updates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Manifest updates should hold a shared lock across read/merge/write."""
    manifest_path = tmp_path / "dashboard.manifest.json"
    collection_root = tmp_path
    first_graph = build_seed_graph("seed-a")
    second_graph = build_seed_graph("seed-b")
    first_graph.nodes["seed-a"]["title"] = "First Seed"
    second_graph.nodes["seed-b"]["title"] = "Second Seed"

    first_write_started = threading.Event()
    allow_first_write = threading.Event()
    second_write_started = threading.Event()
    write_counter = 0
    write_counter_lock = threading.Lock()
    original_atomic_write = cli_module.atomic_write_json

    def delayed_atomic_write(
        path: Path, payload: dict[str, object], **kwargs: object
    ) -> None:
        nonlocal write_counter
        with write_counter_lock:
            write_counter += 1
            call_number = write_counter
        if call_number == 1:
            first_write_started.set()
            assert allow_first_write.wait(timeout=5), "first write never released"
        else:
            second_write_started.set()
        original_atomic_write(path, payload, **kwargs)

    monkeypatch.setattr(cli_module, "atomic_write_json", delayed_atomic_write)

    errors: list[BaseException] = []

    def worker(graph: nx.Graph, seed_id: str) -> None:
        try:
            cli_module.update_dashboard_manifest(
                manifest_path,
                collection_root=collection_root,
                graph=graph,
                seed_id=seed_id,
                strategy="recommendation",
                json_path=collection_root / seed_id / "recommendation.json",
                config_path=collection_root / seed_id / "recommendation.config.json",
                metadata={
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                },
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    first_thread = threading.Thread(target=worker, args=(first_graph, "seed-a"))
    second_thread = threading.Thread(target=worker, args=(second_graph, "seed-b"))

    first_thread.start()
    assert first_write_started.wait(timeout=5), "first write never started"
    second_thread.start()
    assert not second_write_started.wait(timeout=0.25), (
        "second update reached write path before first released manifest lock"
    )

    allow_first_write.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not errors

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {entry["result_id"] for entry in manifest_payload["results"]} == {
        "recommendation:seed-a",
        "recommendation:seed-b",
    }


@pytest.mark.parametrize(
    ("extra_exports", "exporter_methods", "expected_extra_outputs"),
    [
        ([], ("to_dashboard_html",), []),
        (["json"], ("to_dashboard_html", "to_json"), ["report.json"]),
    ],
)
def test_dashboard_standalone_export_preserves_explicit_single_file(
    monkeypatch: pytest.MonkeyPatch,
    extra_exports: list[str],
    exporter_methods: tuple[str, ...],
    expected_extra_outputs: list[str],
) -> None:
    """Explicit dashboard filenames should bypass collection mode."""
    graph = build_seed_graph("seed")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        _make_exporter_stub(
            captured,
            methods=exporter_methods,
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "report.dashboard.html"
        collection_root = output_file.parent / "report"
        command = [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "recommendation",
            "--export",
            "dashboard",
        ]
        for export_format in extra_exports:
            command.extend(["--export", export_format])
        command.extend(["-o", str(output_file)])
        result = run_cli_command(command)
        assert output_file.exists()
        assert (output_file.parent / "report.config.json").exists()
        for expected_output in expected_extra_outputs:
            assert (output_file.parent / expected_output).exists()
        assert not (collection_root / "dashboard.html").exists()
        assert not (collection_root / "dashboard.manifest.json").exists()
        assert not (output_file.parent / "recommendation.json").exists()
        assert "dashboard_collection" not in captured["metadata"]

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_dashboard_collection_helpers_cover_resolver_and_bundle_loading() -> None:
    """Collection helpers should imply JSON and skip invalid embedded payloads."""
    graph = nx.Graph()
    graph.add_node("seed", title="Seed Title")
    assert _is_standalone_dashboard_output(
        Path("reports/example.dashboard.html"),
        ["dashboard"],
        True,
    )
    assert _is_standalone_dashboard_output(
        Path("reports/example.dashboard.html"),
        ["dashboard", "json"],
        True,
    )
    assert not _is_standalone_dashboard_output(
        Path("reports/session"),
        ["dashboard"],
        True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        output_paths, manifest_path = resolve_dashboard_collection_outputs(
            base_output_path=root / "session",
            selected_formats=["dashboard"],
            explicit_output=True,
            strategy="recommendation",
            graph=graph,
            seed_id="seed",
        )
        assert output_paths["dashboard"] == root / "session" / "dashboard.html"
        assert output_paths["json"].parent.parent == root / "session"
        assert output_paths["json"].parent.name.startswith("seed-title-")
        assert output_paths["json"].name == "recommendation.json"
        assert manifest_path == root / "session" / "dashboard.manifest.json"

        valid_path = root / "seed-a" / "recommendation.json"
        valid_path.parent.mkdir(parents=True, exist_ok=True)
        valid_path.write_text(
            json.dumps(
                {
                    "seed_id": "seed-a",
                    "meta": {"strategy": "recommendation"},
                    "summary": {"nodes": 1, "edges": 0},
                    "nodes": [],
                    "edges": [],
                    "dashboard": {"meta": {}},
                }
            ),
            encoding="utf-8",
        )
        broken_path = root / "seed-b" / "recommendation.json"
        broken_path.parent.mkdir(parents=True, exist_ok=True)
        broken_path.write_text("{not-json", encoding="utf-8")
        manifest_path = root / "dashboard.manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "results": [
                        {
                            "result_id": "recommendation:seed-a",
                            "seed_id": "seed-a",
                            "title": "Seed A",
                            "strategy": "recommendation",
                            "summary": {"nodes": 1, "edges": 0},
                            "json_path": "seed-a/recommendation.json",
                        },
                        {
                            "result_id": "recommendation:seed-b",
                            "seed_id": "seed-b",
                            "title": "Seed B",
                            "strategy": "recommendation",
                            "summary": {"nodes": 1, "edges": 0},
                            "json_path": "seed-b/recommendation.json",
                        },
                        {
                            "result_id": "recommendation:seed-c",
                            "seed_id": "seed-c",
                            "title": "Seed C",
                            "strategy": "recommendation",
                            "summary": {"nodes": 1, "edges": 0},
                            "json_path": "seed-c/recommendation.json",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        bundle = build_dashboard_collection_bundle(
            manifest_path,
            current_result_id="recommendation:seed-c",
        )

    assert bundle["current_result_id"] == "recommendation:seed-c"
    assert len(bundle["results"]) == 3
    assert set(bundle["payloads"]) == {"recommendation:seed-a"}


def test_build_dashboard_collection_bundle_rejects_escaping_payload_paths() -> None:
    """Collection bundles should ignore manifest payload paths outside the root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        root = tmp_root / "collection"
        root.mkdir(parents=True, exist_ok=True)

        valid_path = root / "seed-a" / "recommendation.json"
        valid_path.parent.mkdir(parents=True, exist_ok=True)
        valid_path.write_text(
            json.dumps({"result_id": "recommendation:seed-a", "safe": True}),
            encoding="utf-8",
        )

        outside_path = tmp_root / "outside.json"
        outside_path.write_text(
            json.dumps({"result_id": "outside", "leaked": True}),
            encoding="utf-8",
        )

        manifest_path = root / "dashboard.manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "results": [
                        {
                            "result_id": "recommendation:seed-a",
                            "json_path": "seed-a/recommendation.json",
                        },
                        {
                            "result_id": "recommendation:seed-b",
                            "json_path": "../outside.json",
                        },
                        {
                            "result_id": "recommendation:seed-c",
                            "json_path": str(outside_path),
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        bundle = build_dashboard_collection_bundle(
            manifest_path,
            current_result_id="recommendation:seed-a",
        )

    assert set(bundle["payloads"]) == {"recommendation:seed-a"}


def test_dashboard_collection_mode_logs_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collection-mode dashboard exports should log their extra saved-artifact flow."""
    graph = build_seed_graph("seed")
    info_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "info", info_mock)
    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        _make_exporter_stub(
            {},
            methods=("to_dashboard_html", "to_json"),
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "collection"
        result = run_cli_command(
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "recommendation",
                "--export",
                "dashboard",
                "-o",
                str(output_dir),
            ],
        )

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    info_messages = [
        str(call.args[0]) for call in info_mock.call_args_list if call.args
    ]
    assert any("Dashboard collection mode:" in msg for msg in info_messages)


def test_build_uses_compact_plot_metadata_and_summary_export_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should pass compact PNG metadata and emit one export summary line."""
    graph = build_seed_graph("seed")

    captured: dict[str, object] = {}
    logged: list[str] = []
    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        _make_exporter_stub(
            captured,
            methods=(
                "to_json",
                "to_csv",
                "to_bibtex",
                "to_graphml",
                "to_interactive_html",
                "to_plotly_html",
                "to_dashboard_html",
            ),
        ),
    )

    def _fake_visualize(*args: Any, **kwargs: Any) -> None:
        captured["plot_metadata"] = kwargs.get("metadata")
        Path(args[2]).write_bytes(b"png")

    def _capture_info(message: str, *args: Any, **kwargs: Any) -> None:
        del kwargs
        rendered = message % args if args else message
        logged.append(str(rendered))

    monkeypatch.setattr(cli_module, "visualize_graph", _fake_visualize)
    monkeypatch.setattr(cli_module.logger, "info", _capture_info)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "graph.png"
        result = run_cli_command(
            [
                "build",
                "https://arxiv.org/abs/2508.14040",
                "--strategy",
                "hybrid",
                "--export",
                "all",
                "-o",
                str(output),
            ],
        )
        config_files = sorted(Path(tmpdir).rglob("*.config.json"))
        assert len(config_files) == 1
        config_payload = json.loads(config_files[0].read_text())

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "plot_metadata" in captured
    assert captured["plot_metadata"] == {
        "paper_id": "arxiv:2508.14040",
        "strategy": "hybrid",
        "nodes": 1,
        "edges": 0,
        "theme": "light",
    }
    assert any("export artifacts saved" in message for message in logged)
    assert all("PNG saved to" not in message for message in logged)
    assert all("Graph JSON saved to" not in message for message in logged)
    assert all("Creating visualization..." not in message for message in logged)
    assert config_payload["build"]["strategy"] == "hybrid"
    assert config_payload["build"]["paper_id_canonical"] == "arxiv:2508.14040"
    assert config_payload["metadata"]["strategy"] == "hybrid"
    assert config_payload["metadata"]["score_contract"]["score_type"] == (
        "hybrid_similarity_composite"
    )


def test_export_metadata_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build export metadata should satisfy score, timestamp, and embedding contracts."""

    def _capture_metadata(
        *,
        strategy: str,
        extra_args: list[str] | None = None,
        graph: nx.Graph | None = None,
    ) -> dict[str, Any]:
        local_graph = graph if graph is not None else nx.Graph()
        if "seed" not in local_graph:
            local_graph.add_node(
                "seed",
                title="Seed",
                year=2020,
                authors=[],
                citation_count=0,
                is_seed=True,
            )

        graph = local_graph.copy()
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            cli_module,
            "_build_strategy_graph",
            lambda args, _strategy_name, **_kwargs: (graph, "seed"),
        )
        monkeypatch.setattr(
            cli_module,
            "GraphExporter",
            _make_exporter_stub(captured, methods=("to_json",)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "graph.json"
            command = [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                strategy,
                "--export",
                "json",
                "-o",
                str(output),
            ]
            if extra_args:
                command = command[:4] + extra_args + command[4:]
            result = run_cli_command(command)
            assert output.exists()
        assert result.returncode == 0, (
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        metadata = captured["metadata"]
        assert isinstance(metadata, dict)
        return metadata

    hybrid_metadata = _capture_metadata(strategy="hybrid")
    score_contract = hybrid_metadata["score_contract"]
    assert isinstance(score_contract, dict)
    assert score_contract["strategy"] == "hybrid"
    assert score_contract["comparable_across_strategies"] is False
    assert "adjudication_policy" in score_contract

    for include_timestamp, expected_key in [(False, False), (True, True)]:
        extra_args = ["--include-timestamp"] if include_timestamp else []
        metadata = _capture_metadata(
            strategy="recommendation",
            extra_args=extra_args,
        )
        assert ("timestamp" in metadata) is expected_key

    metadata = _capture_metadata(
        strategy="embedding",
        extra_args=["--storage-precision", "float32"],
    )
    assert metadata["embedding"] == {
        "effective_vector_dtype": "float32",
        "effective_device": None,
        "effective_compute_dtype": None,
        "semantic_source": "candidates",
        "candidate_pool_size": 400,
        "storage_precision": "float32",
        "binary_prefilter_enabled": False,
        "binary_prefilter_used_for_query": False,
        "binary_rescore_multiplier": 1,
        "cache_overwrite_reason": None,
    }

    runtime_graph = nx.Graph()
    runtime_graph.graph["embedding_runtime"] = {"binary_prefilter_used": False}
    runtime_graph.add_node(
        "seed",
        title="Seed",
        year=2020,
        authors=[],
        citation_count=0,
        is_seed=True,
    )
    metadata = _capture_metadata(
        strategy="embedding",
        extra_args=[
            "--semantic-source",
            "arxiv-corpus",
            "--storage-precision",
            "int8",
            "--binary-prefilter",
        ],
        graph=runtime_graph,
    )
    assert metadata["embedding"]["binary_prefilter_enabled"] is True
    assert metadata["embedding"]["binary_prefilter_used_for_query"] is False

    metadata = _capture_metadata(
        strategy="hybrid",
        extra_args=["--max-semantic", "0"],
    )
    assert "embedding" not in metadata


def test_graph_config_payload_omits_citation_budgets_for_recommendation() -> None:
    """Recommendation sidecars should only record settings that affect the run."""
    _, build_parser, _, _ = cli_module._create_parser()
    cli_args = build_parser.parse_args(
        [
            "seed",
            "--strategy",
            "recommendation",
            "--no-references",
            "--refresh-reference-cache",
        ]
    )

    payload = cli_module._build_graph_config_payload(
        cli_args=cli_args,
        seed_id="seed",
        metadata={"strategy": "recommendation"},
        selected_formats=["json"],
        output_paths={"json": Path("out/recommendation.json")},
    )

    citation_config = payload["build"]["citation"]
    assert citation_config == {
        "fetch_references": False,
        "refresh_reference_cache": True,
    }
    assert "max_citations" not in citation_config
    assert "max_references" not in citation_config


def test_embedding_build_logs_side_effect_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding build should emit concise runtime config logs."""
    graph = build_seed_graph("seed")

    info_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "info", info_mock)
    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        _make_exporter_stub({}, methods=("to_json",)),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "graph.json"
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
        assert output.exists()

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    messages = [str(call.args[0]) for call in info_mock.call_args_list if call.args]
    assert any("Embedding config:" in msg for msg in messages)


def test_hybrid_disabled_semantic_branch_skips_embedding_side_effect_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid runs without semantic enrichment should not advertise embedding work."""
    graph = build_seed_graph("seed")

    info_mock = MagicMock()
    warning_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "info", info_mock)
    monkeypatch.setattr(cli_module.logger, "warning", warning_mock)
    monkeypatch.setattr(
        cli_module,
        "_embedding_cache_directory_stats",
        lambda: (Path("/tmp/cache"), 0, 0),
    )
    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        _make_exporter_stub({}, methods=("to_json",)),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "graph.json"
        result = run_cli_command(
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "hybrid",
                "--max-semantic",
                "0",
                "--export",
                "json",
                "-o",
                str(output),
            ],
        )
        assert output.exists()

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    info_messages = [
        str(call.args[0]) for call in info_mock.call_args_list if call.args
    ]
    warning_messages = [
        str(call.args[0]) for call in warning_mock.call_args_list if call.args
    ]
    assert not any("Embedding config:" in msg for msg in info_messages)
    assert not any("No embedding cache found" in msg for msg in warning_messages)


def test_strategy_dispatches_to_matching_builder_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch should pass parsed CLI settings into strategy builders."""
    cases = [
        (
            "citation",
            "CitationGraphBuilder",
            {
                "max_citations": 9,
                "max_references": 7,
                "similarity_threshold": 0.21,
                "no_references": True,
            },
            {
                "max_papers": 11,
                "max_citations": 9,
                "max_references": 7,
                "similarity_threshold": 0.21,
                "fetch_references": False,
                "refresh_reference_cache": False,
            },
        ),
        (
            "recommendation",
            "RecommendationGraphBuilder",
            {
                "similarity_threshold": 0.21,
                "no_references": True,
            },
            {
                "max_papers": 11,
                "fetch_references": False,
                "refresh_reference_cache": False,
                "similarity_threshold": 0.21,
            },
        ),
        (
            "embedding",
            "EmbeddingGraphBuilder",
            {
                "model": "m",
                "corpus_size": 1234,
                "all_corpus": False,
                "top_k": 4,
                "truncate_dim": 64,
                "streaming": True,
                "binary_rescore_multiplier": 9,
                "calibration_sample_size": 123,
                "encode_batch_size": 48,
            },
            {
                "max_papers": 11,
                "model_name": "m",
                "model_revision": None,
                "dataset_split": "train",
                "corpus_size": 1234,
                "truncate_dim": 64,
                "top_k": 4,
                "force_rebuild_cache": False,
                "force_rebuild_reason": None,
                "use_streaming": True,
                "storage_precision": "int8",
                "binary_prefilter": True,
                "binary_rescore_multiplier": 9,
                "calibration_sample_size": 123,
                "cache_compression": "gzip",
                "cache_compression_level": 1,
                "encode_batch_size": 48,
                "enable_torch_compile": False,
                "device": "auto",
                "semantic_source": "arxiv-corpus",
                "candidate_pool_size": 400,
            },
        ),
        (
            "hybrid",
            "HybridGraphBuilder",
            {
                "max_citations": 9,
                "max_references": 7,
                "no_references": True,
                "max_semantic": 5,
                "model": "m",
                "corpus_size": 1234,
                "all_corpus": False,
                "truncate_dim": 64,
                "streaming": True,
                "binary_rescore_multiplier": 9,
                "calibration_sample_size": 123,
                "encode_batch_size": 48,
            },
            {
                "max_papers": 11,
                "max_citations": 9,
                "max_references": 7,
                "fetch_references": False,
                "refresh_reference_cache": False,
                "max_semantic": 5,
                "model_name": "m",
                "model_revision": None,
                "dataset_split": "train",
                "corpus_size": 1234,
                "truncate_dim": 64,
                "use_streaming": True,
                "force_rebuild_cache": False,
                "force_rebuild_reason": None,
                "storage_precision": "int8",
                "binary_prefilter": True,
                "binary_rescore_multiplier": 9,
                "calibration_sample_size": 123,
                "cache_compression": "gzip",
                "cache_compression_level": 1,
                "encode_batch_size": 48,
                "enable_torch_compile": False,
                "device": "auto",
                "semantic_source": "arxiv-corpus",
                "candidate_pool_size": 400,
            },
        ),
    ]
    for strategy, builder_name, namespace_overrides, expected_kwargs in cases:
        captured: dict[str, object] = {}
        namespace = _dispatch_namespace(**namespace_overrides)
        monkeypatch.setattr(
            cli_module,
            builder_name,
            _make_builder_stub(captured, graph=build_seed_graph("seed")),
        )
        graph, seed_id = cli_module._build_strategy_graph(namespace, strategy)
        assert seed_id == "seed"
        assert graph.number_of_nodes() == 1
        assert captured == expected_kwargs


def test_programmatic_hybrid_implicit_defaults_flow_into_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Programmatic hybrid dispatch should carry normalized implicit defaults."""
    _, build_parser, _, _ = cli_module._create_parser()
    namespace = build_parser.parse_args(["seed", "--strategy", "hybrid"])
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "HybridGraphBuilder",
        _make_builder_stub(captured, graph=build_seed_graph("seed")),
    )

    graph, seed_id = cli_module._build_strategy_graph(namespace, "hybrid")
    assert seed_id == "seed"
    assert graph.number_of_nodes() == 1
    assert captured["max_papers"] == HYBRID_DEFAULT_MAX_PAPERS
    assert captured["max_citations"] == HYBRID_DEFAULT_MAX_CITATIONS
    assert captured["max_references"] == HYBRID_DEFAULT_MAX_REFERENCES
    assert namespace.max_papers == HYBRID_DEFAULT_MAX_PAPERS
    assert namespace.max_citations == HYBRID_DEFAULT_MAX_CITATIONS
    assert namespace.max_references == HYBRID_DEFAULT_MAX_REFERENCES


def test_programmatic_embedding_dispatch_normalizes_lzf_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Programmatic embedding dispatch should pass normalized lzf level to builder."""
    _, build_parser, _, _ = cli_module._create_parser()
    namespace = build_parser.parse_args(
        ["seed", "--strategy", "embedding", "--cache-compression", "lzf"]
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "EmbeddingGraphBuilder",
        _make_builder_stub(captured, graph=build_seed_graph("seed")),
    )

    graph, seed_id = cli_module._build_strategy_graph(namespace, "embedding")
    assert seed_id == "seed"
    assert graph.number_of_nodes() == 1
    assert captured["cache_compression"] == "lzf"
    assert captured["cache_compression_level"] == 0
    assert namespace.cache_compression_level == 0


def test_programmatic_embedding_dispatch_propagates_normalized_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Programmatic embedding dispatch should pass normalized scalar values to builders."""
    namespace = _dispatch_namespace(
        top_k="4",
        encode_batch_size="32",
        cache_compression="lzf",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "EmbeddingGraphBuilder",
        _make_builder_stub(captured, graph=build_seed_graph("seed")),
    )

    graph, seed_id = cli_module._build_strategy_graph(namespace, "embedding")
    assert seed_id == "seed"
    assert graph.number_of_nodes() == 1
    assert captured["top_k"] == 4
    assert captured["encode_batch_size"] == 32
    assert captured["cache_compression_level"] == 0
    assert namespace.top_k == 4
    assert namespace.encode_batch_size == 32
    assert namespace.cache_compression_level == 0


def test_programmatic_strategy_dispatch_contracts() -> None:
    """Programmatic dispatch should enforce strategy validation."""
    namespace = _dispatch_namespace()
    with pytest.raises(ValueError, match="Unsupported strategy: unknown"):
        cli_module._build_strategy_graph(namespace, "unknown")

    invalid_namespace = _dispatch_namespace(
        similarity_threshold=0.21,
        model="all-MiniLM-L6-v2",
    )
    with pytest.raises(ValueError, match="Unsupported option\\(s\\).*--model"):
        cli_module._build_strategy_graph(invalid_namespace, "recommendation")

    invalid_embedding_namespace = _dispatch_namespace(
        storage_precision="float32",
        calibration_sample_size=512,
    )
    with pytest.raises(
        ValueError,
        match="--calibration-sample-size requires --storage-precision int8",
    ):
        cli_module._build_strategy_graph(invalid_embedding_namespace, "embedding")


def test_programmatic_strategy_dispatch_validates_scalar_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Programmatic dispatch should enforce parser-equivalent scalar validation."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "CitationGraphBuilder",
        _make_builder_stub(captured, graph=build_seed_graph("seed")),
    )

    with pytest.raises(ValueError, match="must be at least 1"):
        cli_module._build_strategy_graph(_dispatch_namespace(max_papers=0), "citation")
    assert captured == {}

    with pytest.raises(ValueError, match="must be a float"):
        cli_module._build_strategy_graph(
            _dispatch_namespace(similarity_threshold="banana"),
            "citation",
        )
    assert captured == {}

    with pytest.raises(ValueError, match="must be a non-empty string"):
        cli_module._build_strategy_graph(
            _dispatch_namespace(paper_id="   "),
            "citation",
        )
    assert captured == {}


def test_cli_help_contracts() -> None:
    """CLI help output should expose stable semantic contracts."""
    cases = [
        (
            ["--help"],
            [
                "CiteMesh",
                "build",
                "cache",
                "search",
                "S2_API_KEY",
                "CITEMESH_CACHE_DIR",
            ],
        ),
        (
            ["build", "--help"],
            [
                "default: recommendation",
                "--seed",
                "dashboard",
                "--all-corpus",
                "--storage-precision",
                "--binary-prefilter",
                "--binary-rescore-multiplier",
                "--calibration-sample-size",
                "--encode-batch-size",
                "--no-torch-compile",
                "--spring-iterations",
                "citation/recommendation",
                "repeat for multiple",
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


def test_build_help_discloses_dashboard_collection_mode_contracts() -> None:
    """Build help should disclose dashboard collection-vs-standalone behavior."""
    result = run_cli_command(["build", "--help"])
    assert result.returncode == 0
    lowered = result.stdout.lower()
    for token in ["dashboard.html", "dashboard.manifest.json", ".dashboard.html"]:
        assert token in lowered


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
        (
            Path("reports/example.dashboard.html"),
            ["dashboard"],
            True,
            {"dashboard": Path("reports/example.dashboard.html")},
        ),
        (
            Path("reports/example.dashboard.html"),
            ["dashboard", "json"],
            True,
            {
                "dashboard": Path("reports/example.dashboard.html"),
                "json": Path("reports/example.json"),
            },
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
    config_cases = [
        (
            {"png": Path("out/seed/hybrid.png")},
            Path("out/seed/hybrid.config.json"),
        ),
        (
            {"json": Path("reports/example.json")},
            Path("reports/example.config.json"),
        ),
        (
            {"plotly": Path("reports/example.plotly.html")},
            Path("reports/example.config.json"),
        ),
    ]
    for output_paths, expected_path in config_cases:
        assert (
            resolve_graph_config_path(output_paths, strategy="hybrid") == expected_path
        )

    for raw_id, expected in get_paper_id_normalization_cases():
        if raw_id.startswith("http://") or raw_id.startswith("https://"):
            assert canonicalize_paper_id_for_metadata(raw_id) == expected

    graph = nx.Graph()
    graph.add_node("seed-a", title="A Survey of Transformers")
    graph.add_node("seed-b", title="A Survey of Transformers")
    path_a = generate_output_path(graph, seed_id="seed-a", output_dir=Path("out"))
    path_b = generate_output_path(graph, seed_id="seed-b", output_dir=Path("out"))
    assert path_a != path_b
    assert path_a.parent.name != path_b.parent.name
    assert re.search(r"-[0-9a-f]{8}$", path_a.parent.name)
    assert re.search(r"-[0-9a-f]{8}$", path_b.parent.name)

    graph = nx.Graph()
    graph.add_node(
        "seed",
        title="This title should definitely exceed forty characters for the slug",
    )
    output_path = generate_output_path(graph, seed_id="seed", output_dir=Path("out"))
    assert re.search(r"-[0-9a-f]{8}$", output_path.parent.name)
    assert len(output_path.parent.name) <= 40


def test_main_module_invokes_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running ``citemesh.__main__`` should invoke ``citemesh.cli.main``."""
    called: dict[str, object] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        called["argv"] = argv
        return 0

    monkeypatch.setattr("citemesh.cli.main", fake_main)
    monkeypatch.setattr(
        "sys.argv", ["python", "build", "seed", "--strategy", "citation"]
    )
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("citemesh.__main__", run_name="__main__")
    assert exc_info.value.code == 0
    assert called["argv"] is None


def _extract_citemesh_doc_commands(markdown_text: str) -> list[list[str]]:
    """Extract parseable ``citemesh`` command argv vectors from Markdown bash blocks."""
    commands: list[list[str]] = []
    blocks = re.findall(r"```bash\s+(.*?)```", markdown_text, flags=re.DOTALL)
    for block in blocks:
        pending = ""
        for raw_line in block.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if pending:
                continuation = (
                    stripped[:-1].strip() if stripped.endswith("\\") else stripped
                )
                pending = f"{pending} {continuation}".strip()
                if stripped.endswith("\\"):
                    continue
                tokens = shlex.split(pending)
                pending = ""
                if tokens and tokens[0] == "citemesh":
                    commands.append(tokens[1:])
                continue
            if not stripped.startswith("citemesh "):
                continue
            if stripped.endswith("\\"):
                pending = stripped[:-1].strip()
                continue
            tokens = shlex.split(stripped)
            if tokens and tokens[0] == "citemesh":
                commands.append(tokens[1:])
    return commands


def test_documented_cli_examples_are_parseable() -> None:
    """README and CLI guide command examples should remain parseable in CI."""
    parser, _, _, _ = cli_module._create_parser()
    docs = [Path("README.md"), Path("docs/guides/cli.md")]

    commands: list[list[str]] = []
    for doc_path in docs:
        markdown_text = doc_path.read_text(encoding="utf-8")
        commands.extend(_extract_citemesh_doc_commands(markdown_text))

    assert commands, "No citemesh commands found in docs; example parser test is stale."
    for argv in commands:
        if any(token.startswith("[") or token.endswith("]") for token in argv):
            continue
        parser.parse_args(argv)


def test_multi_export_flag_selects_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeating --export should produce exactly the requested formats."""
    graph = build_seed_graph("seed")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        _make_exporter_stub(captured, methods=("to_json", "to_csv", "to_bibtex")),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "multi"
        result = run_cli_command(
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "recommendation",
                "-e",
                "json",
                "-e",
                "csv",
                "-o",
                str(output),
            ]
        )
        json_path = output / "recommendation.json"
        csv_path = output / "recommendation.csv"
        assert json_path.exists()
        assert csv_path.exists()
        # bibtex NOT requested — should not exist
        bib_path = output / "recommendation.bib"
        assert not bib_path.exists()

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_multi_export_deduplicates_repeated_formats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeating the same --export value should not produce duplicate work."""
    graph = build_seed_graph("seed")
    call_counts: dict[str, int] = {}

    class _CountingExporter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def to_json(self, path: Any) -> None:
            call_counts["json"] = call_counts.get("json", 0) + 1
            Path(path).write_text("{}")

    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(cli_module, "GraphExporter", _CountingExporter)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "dedup.json"
        result = run_cli_command(
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "recommendation",
                "-e",
                "json",
                "-e",
                "json",
                "-o",
                str(output),
            ]
        )

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert call_counts.get("json", 0) == 1


def test_export_dispatch_table_covers_all_declared_formats() -> None:
    """Every EXPORT_FORMATS entry must have a dispatch mapping or be 'png'."""
    covered = set(cli_module._EXPORTER_METHOD) | {"png"}
    assert covered == set(cli_module.EXPORT_FORMATS), (
        f"Dispatch gap: covered={sorted(covered)}, "
        f"declared={sorted(cli_module.EXPORT_FORMATS)}"
    )


def test_programmatic_dispatch_respects_explicit_provided_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_build_strategy_graph with explicit provided set should bypass inference."""
    _, build_parser, _, _ = cli_module._create_parser()
    namespace = build_parser.parse_args(["seed", "--strategy", "hybrid"])
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "HybridGraphBuilder",
        _make_builder_stub(captured, graph=build_seed_graph("seed")),
    )

    # Explicitly mark max_papers as provided → hybrid override should NOT apply
    graph, seed_id = cli_module._build_strategy_graph(
        namespace, "hybrid", provided={"max_papers"}
    )
    assert seed_id == "seed"
    # max_papers should remain the parser default (40), not the hybrid override (45)
    assert captured["max_papers"] == 40


def test_builder_defaults_match_cli_defaults() -> None:
    """Strategy builder constructor defaults should match CLI parser defaults."""
    from citemesh.strategies.citation import CitationGraphBuilder
    from citemesh.strategies.recommendation import RecommendationGraphBuilder

    _, build_parser, _, _ = cli_module._create_parser()
    defaults = build_parser.parse_args(["seed", "--strategy", "citation"])

    assert CitationGraphBuilder.__init__.__defaults__ is not None
    import inspect

    cit_sig = inspect.signature(CitationGraphBuilder.__init__)
    assert cit_sig.parameters["max_citations"].default == defaults.max_citations
    assert cit_sig.parameters["max_references"].default == defaults.max_references
    assert (
        cit_sig.parameters["similarity_threshold"].default
        == defaults.similarity_threshold
    )

    rec_sig = inspect.signature(RecommendationGraphBuilder.__init__)
    assert (
        rec_sig.parameters["similarity_threshold"].default
        == defaults.similarity_threshold
    )
    # CLI dispatches fetch_references=not(no_references); default no_references=False → True
    assert rec_sig.parameters["fetch_references"].default is (
        not defaults.no_references
    )


def test_build_rejects_device_flag_for_non_embedding_strategies() -> None:
    """--device is embedding/hybrid-scoped and rejected elsewhere."""
    result = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "citation",
            "--device",
            "cpu",
        ]
    )
    assert result.returncode != 0
    assert "Unsupported option(s)" in result.stderr
    assert "--device" in result.stderr


def test_build_rejects_device_flag_when_hybrid_semantic_disabled() -> None:
    """--device is embedding-only for hybrid and invalid with --max-semantic 0."""
    result = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "hybrid",
            "--max-semantic",
            "0",
            "--device",
            "cpu",
        ]
    )
    assert result.returncode != 0
    assert "Hybrid semantic branch is disabled" in result.stderr
    assert "--device" in result.stderr


def test_build_rejects_unavailable_explicit_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicitly requesting an unavailable device fails as a clean parser error."""

    def _raise_unavailable(_requested: str) -> str:
        raise ValueError(
            "device='cuda' was requested but CUDA is not available in this runtime."
        )

    monkeypatch.setattr(cli_module, "resolve_embedding_device", _raise_unavailable)
    result = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "embedding",
            "--device",
            "cuda",
        ]
    )
    assert result.returncode == 2
    assert "device='cuda' was requested but CUDA is not available" in result.stderr


def test_graph_config_payload_records_device() -> None:
    """Embedding sidecar config should persist the requested device token."""
    _, build_parser, _, _ = cli_module._create_parser()
    cli_args = build_parser.parse_args(
        ["seed", "--strategy", "embedding", "--device", "cpu"]
    )

    payload = cli_module._build_graph_config_payload(
        cli_args=cli_args,
        seed_id="seed",
        metadata={"strategy": "embedding"},
        selected_formats=["json"],
        output_paths={"json": Path("out/embedding.json")},
    )

    assert payload["build"]["embedding"]["device"] == "cpu"


def test_export_metadata_records_effective_device() -> None:
    """Export metadata should surface effective device/dtype from runtime."""
    namespace = _dispatch_namespace()
    metadata = cli_module._embedding_export_metadata(
        namespace,
        runtime_metadata={
            "binary_prefilter_used": True,
            "device": "mps",
            "compute_dtype": "bfloat16",
        },
    )
    assert metadata["effective_device"] == "mps"
    assert metadata["effective_compute_dtype"] == "bfloat16"


def test_build_corpus_flags_imply_arxiv_corpus_source() -> None:
    """Corpus-only flags without --semantic-source should imply arxiv-corpus."""
    _, build_parser, _, _ = cli_module._create_parser()
    args = build_parser.parse_args(
        ["seed", "--strategy", "embedding", "--corpus-size", "1234"]
    )
    provided = cli_module._pop_tracked_option_dests(args)
    cli_module._validate_build_cli_contract(args, build_parser, provided)
    assert args.semantic_source == "arxiv-corpus"

    args = build_parser.parse_args(["seed", "--strategy", "embedding"])
    provided = cli_module._pop_tracked_option_dests(args)
    cli_module._validate_build_cli_contract(args, build_parser, provided)
    assert args.semantic_source == "candidates"
    assert args.storage_precision == "float32"


def test_build_rejects_corpus_flags_with_explicit_candidates_source() -> None:
    """Explicit candidates mode should reject corpus-only flags."""
    result = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "embedding",
            "--semantic-source",
            "candidates",
            "--corpus-size",
            "5000",
        ]
    )
    assert result.returncode != 0
    assert "Corpus-only option(s) require --semantic-source arxiv-corpus" in (
        result.stderr
    )
    assert "--corpus-size" in result.stderr


def test_build_rejects_int8_storage_in_candidate_mode() -> None:
    """Explicit int8 storage should require the corpus source."""
    result = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "embedding",
            "--storage-precision",
            "int8",
        ]
    )
    assert result.returncode != 0
    assert "--storage-precision int8 requires --semantic-source arxiv-corpus" in (
        result.stderr
    )


def test_build_rejects_candidate_pool_size_in_corpus_mode() -> None:
    """--candidate-pool-size should be candidates-mode only."""
    result = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "embedding",
            "--semantic-source",
            "arxiv-corpus",
            "--candidate-pool-size",
            "100",
        ]
    )
    assert result.returncode != 0
    assert "--candidate-pool-size requires --semantic-source candidates" in (
        result.stderr
    )
