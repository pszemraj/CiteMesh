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
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import networkx as nx
import pytest

from citemesh import cli as cli_module
from citemesh.cli import (
    DASHBOARD_COLLECTION_KIND,
    DASHBOARD_COLLECTION_SCHEMA_VERSION,
    DASHBOARD_PACKAGE_FILENAME,
    DashboardPackageError,
    _is_standalone_dashboard_output,
    canonicalize_paper_id_for_metadata,
    load_dashboard_package,
    render_dashboard_collection_snapshot,
    resolve_dashboard_collection_outputs,
    resolve_graph_config_path,
    resolve_output_paths,
    update_dashboard_package,
)
from citemesh.core import Author, Paper
from citemesh.core.user_config import UserConfig
from citemesh.data import DEFAULT_EMBEDDING_MODEL_NAME
from citemesh.strategies.candidates import CandidateAcquisitionError
from citemesh.strategies.embedding import ENCODE_BATCH_SIZE
from citemesh.strategies.hybrid import (
    DEFAULT_MAX_SEMANTIC,
    HYBRID_DEFAULT_MAX_CITATIONS,
    HYBRID_DEFAULT_MAX_PAPERS,
    HYBRID_DEFAULT_MAX_REFERENCES,
)
from citemesh.visualization import GraphExporter as ProductionGraphExporter
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

    def _factory(*factory_args: object, **kwargs: object) -> SimpleNamespace:
        captured_data["kwargs"] = kwargs
        captured_data["metadata"] = kwargs.get("metadata")
        captured_data["layout"] = kwargs.get("layout")
        graph = factory_args[0]
        seed_id = str(factory_args[1])
        assert isinstance(graph, nx.Graph)

        def _write_payload(
            path: Path,
            method_name: str,
            *_method_args: object,
            **_method_kwargs: object,
        ) -> None:
            del _method_args, _method_kwargs
            if method_name == "to_json":
                path.write_text(json.dumps(_graph_payload()), encoding="utf-8")
            elif method_name in requested_methods:
                path.write_text(payloads[method_name], encoding="utf-8")

        def _graph_payload() -> dict[str, object]:
            """Build the package-compatible payload for a JSON export.

            :return dict[str, object]: Canonical graph payload for the stub exporter.
            """
            metadata = kwargs.get("metadata")
            strategy = (
                str(metadata.get("strategy") or "")
                if isinstance(metadata, dict)
                else ""
            )
            return _dashboard_graph_payload(graph, seed_id, strategy)

        namespace = {
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
            },
            "graph_payload": _graph_payload,
        }
        return SimpleNamespace(**namespace)

    return _factory


def _dashboard_graph_payload(
    graph: nx.Graph, seed_id: str, strategy: str
) -> dict[str, object]:
    """Build a canonical graph payload for package-focused CLI tests.

    :param nx.Graph graph: Source graph.
    :param str seed_id: Seed node identifier.
    :param str strategy: Strategy descriptor.
    :return dict[str, object]: Canonical graph payload accepted by package helpers.
    """
    layout = {
        node_id: (float(index), float(index % 2))
        for index, node_id in enumerate(graph.nodes)
    }
    return ProductionGraphExporter(
        graph,
        seed_id,
        metadata={"strategy": strategy},
        layout=layout,
    ).graph_payload()


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
        "model_profile": "auto",
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


def _populate_cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a populated cache root and point ``CITEMESH_CACHE_DIR`` at it.

    :param Path tmp_path: Per-test temporary directory.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to set the cache root env var.
    :return Path: Cache root containing sample embedding/reference payloads.
    """
    cache_root = tmp_path / "citemesh-cache-root"
    (cache_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (cache_root / "misc").mkdir(parents=True, exist_ok=True)
    (cache_root / "references").mkdir(parents=True, exist_ok=True)
    (cache_root / "embeddings" / "vectors.bin").write_bytes(b"a" * 2048)
    (cache_root / "references" / "payload.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(cache_root))
    return cache_root


def test_cache_commands_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache clear/scan should honor configured cache root and print usage summary."""
    cache_root = _populate_cache_root(tmp_path, monkeypatch)

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
    assert cache_root.is_dir()
    assert {child.name for child in cache_root.iterdir()} <= {"config.toml.lock"}


def test_cache_clear_declined_at_prompt_keeps_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answering the cache clear prompt with ``n`` should abort and keep every file."""
    cache_root = _populate_cache_root(tmp_path, monkeypatch)
    monkeypatch.setattr(cli_module, "stdin_isatty", lambda: True)
    prompt_mock = MagicMock(return_value="n")
    monkeypatch.setattr(cli_module.Console, "input", prompt_mock)

    declined = run_cli_command(["cache", "clear"])

    assert declined.returncode == 1
    assert prompt_mock.call_count == 1
    assert (cache_root / "embeddings" / "vectors.bin").read_bytes() == b"a" * 2048
    assert (cache_root / "references" / "payload.json").exists()
    assert (cache_root / "misc").is_dir()


def test_cache_clear_prompt_eof_keeps_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``EOFError`` while reading the confirmation should be treated as a decline."""
    cache_root = _populate_cache_root(tmp_path, monkeypatch)
    monkeypatch.setattr(cli_module, "stdin_isatty", lambda: True)
    monkeypatch.setattr(cli_module.Console, "input", MagicMock(side_effect=EOFError))
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)

    aborted = run_cli_command(["cache", "clear"])

    assert aborted.returncode == 1
    assert any(
        "No confirmation input received" in str(call)
        for call in error_mock.call_args_list
    )
    assert (cache_root / "embeddings" / "vectors.bin").read_bytes() == b"a" * 2048
    assert (cache_root / "references" / "payload.json").exists()


def test_cache_clear_without_tty_refuses_before_prompting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-interactive stdin without ``--yes`` should refuse and point at ``--yes``."""
    cache_root = _populate_cache_root(tmp_path, monkeypatch)
    monkeypatch.setattr(cli_module, "stdin_isatty", lambda: False)
    prompt_mock = MagicMock(return_value="y")
    monkeypatch.setattr(cli_module.Console, "input", prompt_mock)
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)

    refused = run_cli_command(["cache", "clear"])

    assert refused.returncode == 1
    assert prompt_mock.call_count == 0
    assert any(
        "Refusing to clear cache in non-interactive mode without --yes" in str(call)
        and "citemesh cache clear --yes" in str(call)
        for call in error_mock.call_args_list
    )
    assert (cache_root / "embeddings" / "vectors.bin").read_bytes() == b"a" * 2048
    assert (cache_root / "references" / "payload.json").exists()


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
    monkeypatch.setattr(
        cli_module.Console, "input", lambda self, prompt="", **kwargs: "n"
    )
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
        ["--log-level", "debug", "build", "seed", "--strategy", "hybrid"],
        ["--log-level", "debug", "cache", "--log-width", "0", "clear", "--yes"],
    ]

    for argv in cases:
        parsed = parser.parse_args(argv)
        if "--log-level" in argv:
            assert parsed.log_level == "debug"
        if "--log-width" in argv:
            assert parsed.log_width == 0
        if "--log-file" in argv:
            assert parsed.log_file == "run.log"
        expected_provided = {
            token.removeprefix("--").replace("-", "_")
            for token in argv
            if token.startswith("--")
        }
        assert cli_module._pop_tracked_option_dests(parsed) == expected_provided

    assert (
        cli_module._pop_tracked_option_dests(parser.parse_args(["cache", "scan"]))
        == set()
    )


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


@pytest.mark.parametrize("preconfigured", [False, True])
def test_configure_logging_honors_debug_console_with_plaintext_log_file(
    tmp_path: Path,
    preconfigured: bool,
) -> None:
    """An explicit debug level should apply to both console and file handlers.

    :param Path tmp_path: Pytest temporary directory.
    :param bool preconfigured: Whether a library already installed a root handler.
    :return None: Assertions verify both logging sinks honor the requested level.
    """
    saved_handlers, saved_level, saved_configured = _reset_cli_logging_state()
    log_path = tmp_path / "logs" / "cli-debug.log"
    stderr = io.StringIO()
    noisy_logger_names = ("filelock", "matplotlib", "urllib3", "semanticscholar")
    saved_logger_levels = {
        name: logging.getLogger(name).level for name in noisy_logger_names
    }

    try:
        if preconfigured:
            logging.getLogger().addHandler(logging.NullHandler())
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
    assert "debug file sink test" in stderr.getvalue()
    assert "info file sink test" in stderr.getvalue()


def test_rich_logging_preserves_unknown_config_table_name(tmp_path: Path) -> None:
    """Config warnings should keep bracket-like table identifiers visible.

    :param Path tmp_path: Pytest temporary directory.
    :return None: Assertions verify Rich markup does not consume the table name.
    """
    saved_handlers, saved_level, saved_configured = _reset_cli_logging_state()
    config_path = tmp_path / "config.toml"
    config_path.write_text("[plugin_settings]\nenabled = true\n", encoding="utf-8")
    stderr = io.StringIO()

    try:
        with redirect_stderr(stderr):
            cli_module._configure_logging(log_level="info", log_width=0)
            cli_module.load_user_config(config_path)
            for handler in logging.getLogger().handlers:
                handler.flush()
    finally:
        _restore_cli_logging_state(saved_handlers, saved_level, saved_configured)

    warning = stderr.getvalue()
    assert "Ignoring unknown config table" in warning
    assert "[plugin_settings]" in warning


def test_search_command_prints_results_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search should print table + full IDs for shell workflows."""
    long_paper_id = "0123456789abcdef0123456789abcdef01234567"
    mock_client = MagicMock()
    mock_client.search_papers.return_value = [
        Paper(
            paper_id=long_paper_id,
            title="[Attention] Is All You Need [/bold]",
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
    assert result.stdout.count(long_paper_id) == 1
    assert "[Attention] Is All You Need [/bold]" in result.stdout


def _fake_local_search_builder(
    *, cached_count: int, results: list[Any] | None = None
) -> MagicMock:
    """Build a fake EmbeddingGraphBuilder for local-search CLI tests."""
    fake_builder = MagicMock()
    fake_builder.search_local.return_value = list(results or [])
    fake_builder.has_persistent_embedding_artifacts.return_value = cached_count > 0
    fake_builder.embedding_cache = SimpleNamespace(
        embedding_count=lambda: cached_count,
        last_search_total_embeddings=cached_count,
        h5_path=Path("namespace.h5"),
    )
    return fake_builder


_FAKE_LOCAL_RESULT = SimpleNamespace(
    paper_id="feedfacefeedfacefeedfacefeedfacefeedface",
    score=0.876,
    metadata={
        "title": "Cached Paper",
        "year": 2024,
        "authors": ["Ada Lovelace", "Alan Turing", "Grace Hopper"],
    },
)


def test_search_mode_s2_rejects_local_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding namespace flags are meaningless for explicit S2 keyword search."""
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)
    result = run_cli_command(
        [
            "search",
            "attention",
            "--mode",
            "s2",
            "--model-profile",
            "embeddinggemma",
        ]
    )
    assert result.returncode == 2
    assert "--model, --model-profile, and --device only apply" in str(
        error_mock.call_args
    )


def test_search_rejects_empty_model_override() -> None:
    """An empty model token should fail parsing instead of silently no-oping.

    :return None: Assertions verify the non-empty model contract.
    """
    result = run_cli_command(["search", "attention", "--model", ""])

    assert result.returncode == 2
    assert "--model" in result.stderr
    assert "must be a non-empty string" in result.stderr


def test_search_mode_local_prints_cached_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--mode local renders cached results and mirrors flagless build defaults."""
    fake_builder = _fake_local_search_builder(
        cached_count=42, results=[_FAKE_LOCAL_RESULT]
    )
    builder_factory = MagicMock(return_value=fake_builder)
    monkeypatch.setattr(cli_module, "EmbeddingGraphBuilder", builder_factory)

    result = run_cli_command(["search", "cached topic", "--mode", "local", "-n", "1"])
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    # Rich folds table cells at console width; compare on whitespace-normalized text.
    plain_stdout = " ".join(result.stdout.split())
    assert "Local semantic search for 'cached topic'" in plain_stdout
    assert _FAKE_LOCAL_RESULT.paper_id in plain_stdout
    assert "0.876" in plain_stdout
    assert "Ada Lovelace" in plain_stdout
    assert "Searched 42 locally cached embeddings" in plain_stdout
    fake_builder.search_local.assert_called_once_with("cached topic", top_k=1)
    fake_builder.prepare_embedding_cache.assert_called_once_with()

    # Namespace parity with a flagless build: candidates mode with the int8
    # default normalized to float32 storage.
    builder_kwargs = builder_factory.call_args.kwargs
    assert builder_kwargs["model_name"] == DEFAULT_EMBEDDING_MODEL_NAME
    assert builder_kwargs["semantic_source"] == "candidates"
    assert builder_kwargs["storage_precision"] == "float32"


@pytest.mark.parametrize("search_args", [[], ["--mode", "local"]])
def test_search_forces_embedding_strategy_over_configured_build_strategy(
    monkeypatch: pytest.MonkeyPatch,
    search_args: list[str],
) -> None:
    """Build strategy defaults must not corrupt local search normalization.

    :param pytest.MonkeyPatch monkeypatch: Pytest patch helper.
    :param list[str] search_args: Auto or explicit-local search mode arguments.
    :return None: Assertions verify local search keeps embedding candidate defaults.
    """
    fake_builder = _fake_local_search_builder(
        cached_count=42, results=[_FAKE_LOCAL_RESULT]
    )
    builder_factory = MagicMock(return_value=fake_builder)
    monkeypatch.setattr(cli_module, "EmbeddingGraphBuilder", builder_factory)
    config = UserConfig(
        path=Path("cfg-home") / "config.toml",
        defaults={"strategy": "recommendation"},
    )
    monkeypatch.setattr(cli_module, "load_user_config", lambda: config)
    client_factory = MagicMock()
    monkeypatch.setattr(cli_module, "get_client", client_factory)

    result = run_cli_command(["search", "cached topic", *search_args])

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    builder_kwargs = builder_factory.call_args.kwargs
    assert builder_kwargs["semantic_source"] == "candidates"
    assert builder_kwargs["storage_precision"] == "float32"
    assert builder_kwargs["binary_prefilter"] is False
    client_factory.assert_not_called()


def test_search_auto_falls_back_on_config_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto search should catch local config failures and use S2 without build usage.

    :param pytest.MonkeyPatch monkeypatch: Pytest patch helper.
    :return None: Assertions verify the config failure remains catchable.
    """
    config = UserConfig(
        path=Path("cfg-home") / "config.toml", defaults={"device": "cuda"}
    )
    monkeypatch.setattr(cli_module, "load_user_config", lambda: config)
    monkeypatch.setattr(
        cli_module,
        "resolve_embedding_device",
        MagicMock(side_effect=ValueError("CUDA is unavailable")),
    )
    mock_client = MagicMock()
    mock_client.search_papers.return_value = [
        Paper(
            paper_id="0123456789abcdef0123456789abcdef01234567",
            title="Fallback Result",
            year=2025,
            authors=[Author(name="Ada Lovelace")],
            citation_count=1,
            abstract="Fallback result",
        )
    ]
    monkeypatch.setattr(cli_module, "get_client", lambda: mock_client)
    info = MagicMock()
    monkeypatch.setattr(cli_module.logger, "info", info)

    result = run_cli_command(["search", "attention"])

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "Fallback Result" in result.stdout
    assert "usage: citemesh build" not in result.stderr
    notices = str(info.call_args_list)
    assert "defaults.device" in notices
    assert str(config.path) in notices
    assert "searching the Semantic Scholar API instead" in notices


def test_search_local_reports_config_contract_error_without_build_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit local search should report config failures as local errors.

    :param pytest.MonkeyPatch monkeypatch: Pytest patch helper.
    :return None: Assertions verify no build parser ``SystemExit`` escapes.
    """
    config = UserConfig(
        path=Path("cfg-home") / "config.toml", defaults={"device": "cuda"}
    )
    monkeypatch.setattr(cli_module, "load_user_config", lambda: config)
    monkeypatch.setattr(
        cli_module,
        "resolve_embedding_device",
        MagicMock(side_effect=ValueError("CUDA is unavailable")),
    )
    error = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error)

    result = run_cli_command(["search", "attention", "--mode", "local"])

    assert result.returncode == 1
    assert "usage: citemesh build" not in result.stderr
    message = str(error.call_args)
    assert "Local search unavailable" in message
    assert "defaults.device" in message
    assert str(config.path) in message


def test_search_device_flag_overrides_config_before_local_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit local device should replace config before contract validation.

    :param pytest.MonkeyPatch monkeypatch: Pytest patch helper.
    :return None: Assertions verify CLI-over-config precedence during validation.
    """
    config = UserConfig(
        path=Path("cfg-home") / "config.toml", defaults={"device": "cuda"}
    )
    monkeypatch.setattr(cli_module, "load_user_config", lambda: config)
    resolver = MagicMock(return_value="cpu")
    monkeypatch.setattr(cli_module, "resolve_embedding_device", resolver)
    fake_builder = _fake_local_search_builder(
        cached_count=42, results=[_FAKE_LOCAL_RESULT]
    )
    builder_factory = MagicMock(return_value=fake_builder)
    monkeypatch.setattr(cli_module, "EmbeddingGraphBuilder", builder_factory)

    result = run_cli_command(
        ["search", "cached topic", "--mode", "local", "--device", "cpu"]
    )

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert builder_factory.call_args.kwargs["device"] == "cpu"
    assert resolver.call_count == 2
    assert all(call.args == ("cpu",) for call in resolver.call_args_list)


@pytest.mark.parametrize(
    ("namespace_args", "expected_kwarg", "expected_value"),
    [
        (["--model", "custom/model"], "model_name", "custom/model"),
        (["--model-profile", "embeddinggemma"], "model_profile", "embeddinggemma"),
    ],
)
def test_search_namespace_flag_implies_local_mode(
    monkeypatch: pytest.MonkeyPatch,
    namespace_args: list[str],
    expected_kwarg: str,
    expected_value: str,
) -> None:
    """A namespace override without --mode should select local search."""
    fake_builder = _fake_local_search_builder(
        cached_count=7, results=[_FAKE_LOCAL_RESULT]
    )
    builder_factory = MagicMock(return_value=fake_builder)
    monkeypatch.setattr(cli_module, "EmbeddingGraphBuilder", builder_factory)
    client_factory = MagicMock()
    monkeypatch.setattr(cli_module, "get_client", client_factory)

    result = run_cli_command(["search", "cached topic", *namespace_args])
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "Local semantic search for 'cached topic'" in " ".join(result.stdout.split())
    assert builder_factory.call_args.kwargs[expected_kwarg] == expected_value
    client_factory.assert_not_called()


def test_search_auto_uses_local_when_cache_populated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (auto) mode prefers local search and says so when vectors exist."""
    fake_builder = _fake_local_search_builder(
        cached_count=42, results=[_FAKE_LOCAL_RESULT]
    )
    monkeypatch.setattr(
        cli_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )
    client_factory = MagicMock()
    monkeypatch.setattr(cli_module, "get_client", client_factory)
    info_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "info", info_mock)

    result = run_cli_command(["search", "cached topic", "-n", "1"])
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "Local semantic search for 'cached topic'" in " ".join(result.stdout.split())
    notices = str(info_mock.call_args_list)
    assert "locally cached embeddings" in notices
    assert "--mode s2" in notices
    client_factory.assert_not_called()


def test_search_auto_resolves_artifact_namespace_when_cache_files_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto search must resolve the artifact before opening any cache namespace."""
    fake_builder = _fake_local_search_builder(
        cached_count=0, results=[_FAKE_LOCAL_RESULT]
    )
    fake_builder.embedding_cache.embedding_count = MagicMock(return_value=42)
    fake_builder.has_persistent_embedding_artifacts.return_value = True
    monkeypatch.setattr(
        cli_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )
    client_factory = MagicMock()
    monkeypatch.setattr(cli_module, "get_client", client_factory)

    result = run_cli_command(["search", "cached topic", "-n", "1"])

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    fake_builder.prepare_embedding_cache.assert_called_once_with()
    fake_builder.embedding_cache.embedding_count.assert_called_once_with()
    assert fake_builder.method_calls[:2] == [
        ("has_persistent_embedding_artifacts", (), {}),
        ("prepare_embedding_cache", (), {}),
    ]
    fake_builder.search_local.assert_called_once_with("cached topic", top_k=1)
    client_factory.assert_not_called()


def test_search_auto_falls_back_to_s2_when_cache_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (auto) mode falls back to S2 keyword search with a notice."""
    fake_builder = _fake_local_search_builder(cached_count=0)
    monkeypatch.setattr(
        cli_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )
    mock_client = MagicMock()
    mock_client.search_papers.return_value = [
        Paper(
            paper_id="0123456789abcdef0123456789abcdef01234567",
            title="Attention Is All You Need",
            year=2017,
            authors=[Author(name="Ashish Vaswani")],
            citation_count=12345,
            abstract="Transformer model paper",
        )
    ]
    monkeypatch.setattr(cli_module, "get_client", lambda: mock_client)
    info_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "info", info_mock)

    result = run_cli_command(["search", "attention", "--limit", "1"])
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "Search results for 'attention'" in result.stdout
    assert "searching the Semantic Scholar API instead" in str(info_mock.call_args_list)
    fake_builder.search_local.assert_not_called()
    fake_builder.prepare_embedding_cache.assert_not_called()


def test_search_mode_local_empty_cache_fails_with_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit local mode treats an empty cache as an error with guidance."""
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)
    fake_builder = _fake_local_search_builder(cached_count=0)
    monkeypatch.setattr(
        cli_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )

    result = run_cli_command(["search", "anything", "--mode", "local"])
    assert result.returncode == 1
    message = str(error_mock.call_args)
    assert "Local search was requested via" in message
    assert "--mode local" in message
    assert "has no vectors" in message
    fake_builder.prepare_embedding_cache.assert_not_called()
    fake_builder.search_local.assert_not_called()


def test_search_explicit_auto_with_namespace_flag_keeps_s2_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit --mode auto must keep its S2 fallback despite --model.

    :param pytest.MonkeyPatch monkeypatch: Builder and client stubs.
    :return None: Assertions verify the fallback path runs instead of erroring.
    """
    fake_builder = _fake_local_search_builder(cached_count=0)
    monkeypatch.setattr(
        cli_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )
    mock_client = MagicMock()
    mock_client.search_papers.return_value = [
        Paper(
            paper_id="0123456789abcdef0123456789abcdef01234567",
            title="Fallback Result",
            year=2020,
            abstract="fallback",
        )
    ]
    monkeypatch.setattr(cli_module, "get_client", lambda: mock_client)
    info_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "info", info_mock)

    result = run_cli_command(
        ["search", "anything", "--mode", "auto", "--model", "custom/model"]
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "searching the Semantic Scholar API instead" in str(info_mock.call_args_list)
    mock_client.search_papers.assert_called_once()
    fake_builder.search_local.assert_not_called()


def test_search_namespace_flag_empty_cache_error_names_the_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The implied-local error must not claim the user passed --mode local.

    :param pytest.MonkeyPatch monkeypatch: Builder stub and error capture.
    :return None: Assertions pin the namespace-flag attribution text.
    """
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)
    fake_builder = _fake_local_search_builder(cached_count=0)
    monkeypatch.setattr(
        cli_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )

    result = run_cli_command(["search", "anything", "--model", "custom/model"])
    assert result.returncode == 1
    message = str(error_mock.call_args)
    assert "namespace flags imply" in message
    assert "--mode local" not in message
    fake_builder.search_local.assert_not_called()


def test_search_mode_from_config_local_empty_cache_cites_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config-driven local mode errors on an empty cache and cites config.toml."""
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)
    fake_builder = _fake_local_search_builder(cached_count=0)
    monkeypatch.setattr(
        cli_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )
    config = UserConfig(
        path=Path("cfg-home") / "config.toml", defaults={"search_mode": "local"}
    )
    monkeypatch.setattr(cli_module, "load_user_config", lambda: config)

    result = run_cli_command(["search", "anything"])
    assert result.returncode == 1
    message = str(error_mock.call_args)
    assert "defaults.search_mode" in message
    assert "config.toml" in message


def test_invalid_paper_id_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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

    output = tmp_path / "unavailable.json"
    error_mock.reset_mock()
    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        MagicMock(
            side_effect=CandidateAcquisitionError(
                "All requested Semantic Scholar sources were unavailable"
            )
        ),
    )
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
        ]
    )
    assert result.returncode != 0
    assert not output.exists()
    assert "All requested Semantic Scholar sources" in str(error_mock.call_args)


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
                "org/generic-embedding-model",
            ],
            "--model",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "citation",
                "-morg/generic-embedding-model",
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
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "citation",
                "--device",
                "cpu",
            ],
            "--device",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "citation",
                "--no-streaming",
            ],
            "--no-streaming",
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
                "org/generic-embedding-model",
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
                "--device",
                "cpu",
            ],
            "remove embedding-only option(s): --device",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--semantic-source",
                "candidates",
                "--corpus-size",
                "5000",
            ],
            "Corpus-only option(s) require --semantic-source arxiv-corpus: --corpus-size",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--storage-precision",
                "int8",
            ],
            "--storage-precision int8 requires --semantic-source arxiv-corpus",
        ),
        (
            [
                "build",
                "arxiv:1706.03762",
                "--strategy",
                "embedding",
                "--semantic-source",
                "arxiv-corpus",
                "--candidate-pool-size",
                "100",
            ],
            "--candidate-pool-size requires --semantic-source candidates",
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


def test_cli_rejects_float16_persistent_storage_precision() -> None:
    """The build parser should reject float16 persistent storage."""
    result = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "embedding",
            "--storage-precision",
            "float16",
        ]
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert "float16" in result.stderr


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
                "org/generic-embedding-model",
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
    """Build path should share one seeded layout across PNG and JSON exports."""
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

    # JSON embeds dashboard geometry, so a JSON-only build computes the same
    # shared layout (with the run's parameters) instead of skipping it.
    json_layout = {"seed": (0.5, 0.5)}
    json_layout_params: dict[str, object] = {}

    def _fake_json_compute_layout(
        graph_arg: nx.Graph, iterations: int, layout_seed: int | None
    ) -> dict[str, tuple[float, float]]:
        """Record JSON layout options and return the shared fake layout.

        :param nx.Graph graph_arg: Graph passed to the layout function.
        :param int iterations: Requested spring-layout iterations.
        :param int | None layout_seed: Requested layout seed.
        :return dict[str, tuple[float, float]]: Shared layout for the JSON export.
        """
        del graph_arg
        json_layout_params["iterations"] = iterations
        json_layout_params["layout_seed"] = layout_seed
        return json_layout

    monkeypatch.setattr(cli_module, "compute_layout", _fake_json_compute_layout)
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
    assert captured["layout"] is json_layout
    assert json_layout_params == {"iterations": 100, "layout_seed": None}


def test_dashboard_export_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dashboard collections should retain the shared viewer and per-seed graph."""
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
            methods=("to_dashboard_html",),
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
            graph, "seed", output_dir=output, strategy="recommendation"
        ).parent
        assert {path.name for path in output.iterdir()} == {
            "dashboard.html",
            DASHBOARD_PACKAGE_FILENAME,
            run_dir.name,
        }
        package = load_dashboard_package(output / DASHBOARD_PACKAGE_FILENAME)
        assert package["kind"] == DASHBOARD_COLLECTION_KIND
        assert package["schema_version"] == DASHBOARD_COLLECTION_SCHEMA_VERSION
        assert package["current_result_id"] == "recommendation:seed"
        assert [entry["result_id"] for entry in package["results"]] == [
            "recommendation:seed"
        ]
        entry = package["results"][0]
        assert entry["payload"]["seed_id"] == "seed"
        assert entry["build"]["strategy"] == "recommendation"
        assert (
            json.loads((run_dir / "recommendation.json").read_text())
            == entry["payload"]
        )
        config = json.loads((run_dir / "recommendation.config.json").read_text())
        assert config["build"]["exports_requested"] == ["dashboard"]
        assert config["outputs"]["json"] == str(run_dir / "recommendation.json")
        assert captured["metadata"]["dashboard_collection"] == package
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
        assert (output_dir / DASHBOARD_PACKAGE_FILENAME).exists()
        assert not (run_dir / "recommendation.dashboard.html").exists()
        assert (run_dir / "recommendation.csv").exists()
        assert (run_dir / "recommendation.bib").exists()
        config_files = sorted(output_dir.rglob("*.config.json"))
        assert len(config_files) == 1
        config_payload = json.loads(config_files[0].read_text())
        assert "dashboard" in config_payload["outputs"]
        assert config_payload["outputs"]["dashboard_package"].endswith(
            DASHBOARD_PACKAGE_FILENAME
        )
        assert "csv" in config_payload["outputs"]
        assert "bibtex" in config_payload["outputs"]
        assert (output_dir / DASHBOARD_PACKAGE_FILENAME).exists()
        assert not (output_dir / "dashboard.manifest.json").exists()
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_dashboard_default_collection_retains_seed_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default output root should retain a graph JSON and sidecar per seed."""
    graph = build_seed_graph("seed")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        _make_exporter_stub({}, methods=("to_dashboard_html",)),
    )

    result = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "recommendation",
            "--export",
            "dashboard",
        ]
    )

    output_root = tmp_path / "out"
    assert result.returncode == 0, result.stderr
    run_dir = generate_output_path(
        graph, "seed", output_dir=output_root, strategy="recommendation"
    ).parent
    assert {path.name for path in output_root.iterdir()} == {
        "dashboard.html",
        DASHBOARD_PACKAGE_FILENAME,
        run_dir.name,
    }
    assert {path.name for path in run_dir.iterdir()} == {
        "recommendation.json",
        "recommendation.config.json",
    }


def test_dashboard_package_tracks_multiple_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One portable dashboard package should retain multiple result payloads."""
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
            methods=("to_dashboard_html",),
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
        first_json = generate_output_path(
            first_graph, "seed-a", output_dir=output_dir, strategy="recommendation"
        ).with_suffix(".json")
        first_bytes = first_json.read_bytes()
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

        package = load_dashboard_package(output_dir / DASHBOARD_PACKAGE_FILENAME)
        assert first_json.read_bytes() == first_bytes
        assert (output_dir / "dashboard.html").exists()
        assert package["current_result_id"] == "recommendation:seed-b"
        assert [entry["result_id"] for entry in package["results"]] == [
            "recommendation:seed-b",
            "recommendation:seed-a",
        ]
        for seed_graph, seed_id in ((first_graph, "seed-a"), (second_graph, "seed-b")):
            run_dir = generate_output_path(
                seed_graph, seed_id, output_dir=output_dir, strategy="recommendation"
            ).parent
            saved = json.loads((run_dir / "recommendation.json").read_text())
            entry = next(
                item for item in package["results"] if item["seed_id"] == seed_id
            )
            assert saved == entry["payload"]
        collection_bundle = captured["metadata"]["dashboard_collection"]
        assert collection_bundle["current_result_id"] == "recommendation:seed-b"
        assert len(collection_bundle["results"]) == 2
        assert {entry["result_id"] for entry in collection_bundle["results"]} == {
            "recommendation:seed-a",
            "recommendation:seed-b",
        }

    assert first_result.returncode == 0
    assert second_result.returncode == 0


def test_dashboard_package_refreshes_same_seed_strategy_slot(
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
            methods=("to_dashboard_html",),
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

        package = load_dashboard_package(output_dir / DASHBOARD_PACKAGE_FILENAME)
        assert len(package["results"]) == 1
        entry = package["results"][0]
        assert entry["result_id"] == "recommendation:seed"
        assert entry["summary"] == {"nodes": 2, "edges": 1}
        assert entry["payload"]["summary"] == {"nodes": 2, "edges": 1}
        run_dir = generate_output_path(
            updated_graph, "seed", output_dir=output_dir, strategy="recommendation"
        ).parent
        assert (
            json.loads((run_dir / "recommendation.json").read_text())
            == entry["payload"]
        )
        collection_bundle = captured["metadata"]["dashboard_collection"]
        assert collection_bundle["current_result_id"] == "recommendation:seed"
        assert len(collection_bundle["results"]) == 1

    assert first_result.returncode == 0
    assert second_result.returncode == 0


def test_dashboard_package_deduplicates_existing_slots_deterministically(
    tmp_path: Path,
) -> None:
    """The first existing duplicate should win when a later result is upserted."""
    package_path = tmp_path / DASHBOARD_PACKAGE_FILENAME
    first_graph = build_seed_graph("seed-a")
    first_graph.nodes["seed-a"]["title"] = "First Copy"
    initial = update_dashboard_package(
        package_path,
        graph=first_graph,
        seed_id="seed-a",
        strategy="recommendation",
        payload=_dashboard_graph_payload(first_graph, "seed-a", "recommendation"),
        build={"strategy": "recommendation"},
    )
    duplicate = dict(initial["results"][0])
    duplicate["title"] = "Ignored Duplicate"
    package_path.write_text(
        json.dumps({**initial, "results": [initial["results"][0], duplicate]}),
        encoding="utf-8",
    )
    second_graph = build_seed_graph("seed-b")

    updated = update_dashboard_package(
        package_path,
        graph=second_graph,
        seed_id="seed-b",
        strategy="recommendation",
        payload=_dashboard_graph_payload(second_graph, "seed-b", "recommendation"),
        build={"strategy": "recommendation"},
    )

    assert [entry["result_id"] for entry in updated["results"]] == [
        "recommendation:seed-b",
        "recommendation:seed-a",
    ]
    assert updated["results"][1]["title"] == "First Copy"


def test_dashboard_package_serializes_concurrent_updates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Package updates should hold one lock across read, merge, and atomic write."""
    package_path = tmp_path / DASHBOARD_PACKAGE_FILENAME
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
            update_dashboard_package(
                package_path,
                graph=graph,
                seed_id=seed_id,
                strategy="recommendation",
                payload=_dashboard_graph_payload(graph, seed_id, "recommendation"),
                build={"strategy": "recommendation"},
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    first_thread = threading.Thread(target=worker, args=(first_graph, "seed-a"))
    second_thread = threading.Thread(target=worker, args=(second_graph, "seed-b"))

    first_thread.start()
    assert first_write_started.wait(timeout=5), "first write never started"
    second_thread.start()
    assert not second_write_started.wait(timeout=0.25), (
        "second update reached write path before first released package lock"
    )

    allow_first_write.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not errors

    package = load_dashboard_package(package_path)
    assert {entry["result_id"] for entry in package["results"]} == {
        "recommendation:seed-a",
        "recommendation:seed-b",
    }


@pytest.mark.parametrize(
    (
        "extra_exports",
        "exporter_methods",
        "expected_extra_outputs",
        "expects_config",
    ),
    [
        ([], ("to_dashboard_html",), [], False),
        (["json"], ("to_dashboard_html", "to_json"), ["report.json"], True),
    ],
)
def test_dashboard_standalone_export_preserves_explicit_single_file(
    monkeypatch: pytest.MonkeyPatch,
    extra_exports: list[str],
    exporter_methods: tuple[str, ...],
    expected_extra_outputs: list[str],
    expects_config: bool,
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
        assert (output_file.parent / "report.config.json").exists() is expects_config
        for expected_output in expected_extra_outputs:
            assert (output_file.parent / expected_output).exists()
        assert not (collection_root / "dashboard.html").exists()
        assert not (collection_root / "dashboard.manifest.json").exists()
        assert not (collection_root / DASHBOARD_PACKAGE_FILENAME).exists()
        assert not (output_file.parent / "recommendation.json").exists()
        assert "dashboard_collection" not in captured["metadata"]

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_dashboard_collection_resolver_always_includes_graph_json(
    tmp_path: Path,
) -> None:
    """A collection should plan a per-seed JSON even without an explicit JSON flag."""
    graph = nx.Graph()
    graph.add_node("seed", title="Seed Title")
    assert _is_standalone_dashboard_output(
        Path("reports/example.dashboard.html"), ["dashboard"], True
    )
    assert _is_standalone_dashboard_output(
        Path("reports/example.dashboard.html"), ["dashboard", "json"], True
    )
    assert not _is_standalone_dashboard_output(
        Path("reports/session"), ["dashboard"], True
    )

    root = tmp_path / "reports" / "session"
    output_paths, package_path = resolve_dashboard_collection_outputs(
        base_output_path=root,
        selected_formats=["dashboard"],
        explicit_output=True,
        strategy="recommendation",
        graph=graph,
        seed_id="seed",
    )
    assert set(output_paths) == {"dashboard", "json"}
    assert output_paths["dashboard"] == root / "dashboard.html"
    implicit_json = output_paths["json"]
    assert package_path == root / DASHBOARD_PACKAGE_FILENAME

    output_paths, package_path = resolve_dashboard_collection_outputs(
        base_output_path=root,
        selected_formats=["dashboard", "json"],
        explicit_output=True,
        strategy="recommendation",
        graph=graph,
        seed_id="seed",
    )
    assert output_paths["json"].parent.parent == root
    assert output_paths["json"].parent.name.startswith("seed-title-")
    assert output_paths["json"].name == "recommendation.json"
    assert output_paths["json"] == implicit_json
    assert package_path == root / DASHBOARD_PACKAGE_FILENAME


@pytest.mark.parametrize(
    "existing_bytes",
    [
        b"{not-json",
        b"\xff",
        json.dumps(
            {
                "kind": DASHBOARD_COLLECTION_KIND,
                "schema_version": 999,
                "current_result_id": None,
                "results": [],
            }
        ).encode("utf-8"),
    ],
    ids=["malformed-json", "invalid-utf8", "unsupported-schema"],
)
@pytest.mark.parametrize("stat_error", [False, True])
def test_dashboard_package_existing_errors_fail_closed(
    tmp_path: Path,
    existing_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
    stat_error: bool,
) -> None:
    """Invalid or uninspectable packages must remain byte-for-byte untouched.

    :param Path tmp_path: Isolated output directory.
    :param bytes existing_bytes: Persisted invalid package.
    :param pytest.MonkeyPatch monkeypatch: Filesystem fault injection fixture.
    :param bool stat_error: Emulate suppressed filesystem errors on Python 3.14.
    :return None: Checks that an update cannot replace unreadable prior results.
    """
    package_path = tmp_path / DASHBOARD_PACKAGE_FILENAME
    package_path.write_bytes(existing_bytes)
    graph = build_seed_graph("seed")

    original_stat, original_exists = Path.stat, Path.exists

    def fail_stat(path: Path, **kwargs: Any) -> Any:
        """Fail one package inspection while leaving other paths usable.

        :param Path path: Inspected path.
        :param Any kwargs: Remaining stat options.
        :return Any: Statistics for unaffected paths.
        """
        if path == package_path:
            raise OSError("package stat unavailable")
        return original_stat(path, **kwargs)

    with monkeypatch.context() as fault:
        if stat_error:
            fault.setattr(Path, "stat", fail_stat)
            fault.setattr(
                Path,
                "exists",
                lambda path: False if path == package_path else original_exists(path),
            )
        with pytest.raises((DashboardPackageError, OSError)):
            update_dashboard_package(
                package_path,
                graph=graph,
                seed_id="seed",
                strategy="recommendation",
                payload=_dashboard_graph_payload(graph, "seed", "recommendation"),
                build={"strategy": "recommendation"},
            )

    assert package_path.read_bytes() == existing_bytes


def test_dashboard_build_preflights_invalid_package_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An incompatible collection must block all new dashboard artifacts."""
    output_root = tmp_path / "collection"
    output_root.mkdir()
    package_path = output_root / DASHBOARD_PACKAGE_FILENAME
    package_path.write_text(
        json.dumps(
            {
                "kind": DASHBOARD_COLLECTION_KIND,
                "schema_version": 999,
                "current_result_id": None,
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    original_package = package_path.read_bytes()
    build_graph = MagicMock(
        side_effect=AssertionError("graph construction must not run before preflight")
    )
    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        build_graph,
    )
    monkeypatch.setattr(
        cli_module,
        "GraphExporter",
        _make_exporter_stub({}, methods=("to_dashboard_html", "to_json")),
    )

    result = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "recommendation",
            "--export",
            "dashboard",
            "--export",
            "json",
            "--output",
            str(output_root),
        ]
    )

    assert result.returncode == 1
    build_graph.assert_not_called()
    assert package_path.read_bytes() == original_package
    assert not (output_root / "dashboard.html").exists()
    assert list(output_root.iterdir()) == [package_path]


def test_dashboard_package_rejects_graph_summary_drift(tmp_path: Path) -> None:
    """Package summaries must describe the arrays they accompany."""
    package_path = tmp_path / DASHBOARD_PACKAGE_FILENAME
    graph = build_seed_graph("seed")
    package = update_dashboard_package(
        package_path,
        graph=graph,
        seed_id="seed",
        strategy="recommendation",
        payload=_dashboard_graph_payload(graph, "seed", "recommendation"),
        build={"strategy": "recommendation"},
    )
    package["results"][0]["payload"]["summary"]["nodes"] = 999
    package_path.write_text(json.dumps(package), encoding="utf-8")

    with pytest.raises(DashboardPackageError, match="node and edge arrays"):
        load_dashboard_package(package_path)


@pytest.mark.parametrize(
    ("mutate_payload", "error_match"),
    [
        (
            lambda payload: payload.pop("dashboard"),
            "missing dashboard geometry",
        ),
        (
            lambda payload: payload["dashboard"]["meta"][
                "plotly_positions"
            ].__setitem__(0, [float("nan"), 0.0]),
            "invalid layout position",
        ),
        (
            lambda payload: payload["dashboard"]["meta"][
                "plotly_node_sizes"
            ].__setitem__(0, 0.0),
            "invalid node sizes",
        ),
    ],
    ids=["missing", "non-finite-position", "non-positive-size"],
)
def test_dashboard_package_rejects_unrenderable_geometry(
    tmp_path: Path,
    mutate_payload: Callable[[dict[str, Any]], object],
    error_match: str,
) -> None:
    """Collection payloads must contain complete, finite render geometry."""
    graph = build_seed_graph("seed")
    payload = _dashboard_graph_payload(graph, "seed", "recommendation")
    mutate_payload(payload)

    with pytest.raises(DashboardPackageError, match=error_match):
        update_dashboard_package(
            tmp_path / DASHBOARD_PACKAGE_FILENAME,
            graph=graph,
            seed_id="seed",
            strategy="recommendation",
            payload=payload,
            build={"strategy": "recommendation"},
        )


def test_dashboard_package_rejects_padded_identity_tokens(tmp_path: Path) -> None:
    """Versioned graph identities must not rely on reader-specific trimming."""
    graph = build_seed_graph("seed")
    payload = _dashboard_graph_payload(graph, "seed", "recommendation")
    payload["seed_id"] = " seed "

    with pytest.raises(DashboardPackageError, match="canonical token"):
        update_dashboard_package(
            tmp_path / DASHBOARD_PACKAGE_FILENAME,
            graph=graph,
            seed_id="seed",
            strategy="recommendation",
            payload=payload,
            build={"strategy": "recommendation"},
        )


def test_dashboard_package_rejects_mismatched_dashboard_metadata(
    tmp_path: Path,
) -> None:
    """Duplicated dashboard identity metadata must agree with canonical fields."""
    graph = build_seed_graph("seed")
    payload = _dashboard_graph_payload(graph, "seed", "recommendation")
    payload["dashboard"]["meta"]["strategy"] = "embedding"

    with pytest.raises(DashboardPackageError, match="inconsistent dashboard metadata"):
        update_dashboard_package(
            tmp_path / DASHBOARD_PACKAGE_FILENAME,
            graph=graph,
            seed_id="seed",
            strategy="recommendation",
            payload=payload,
            build={"strategy": "recommendation"},
        )


def test_dashboard_snapshot_rereads_latest_package_under_shared_lock(
    tmp_path: Path,
) -> None:
    """HTML refresh should embed the latest package, not an earlier upsert result."""
    package_path = tmp_path / DASHBOARD_PACKAGE_FILENAME
    first_graph = build_seed_graph("seed-a")
    first_package = update_dashboard_package(
        package_path,
        graph=first_graph,
        seed_id="seed-a",
        strategy="recommendation",
        payload=_dashboard_graph_payload(first_graph, "seed-a", "recommendation"),
        build={"strategy": "recommendation"},
    )
    second_graph = build_seed_graph("seed-b")
    update_dashboard_package(
        package_path,
        graph=second_graph,
        seed_id="seed-b",
        strategy="recommendation",
        payload=_dashboard_graph_payload(second_graph, "seed-b", "recommendation"),
        build={"strategy": "recommendation"},
    )
    metadata: dict[str, object] = {"dashboard_collection": first_package}
    dashboard_path = tmp_path / "dashboard.html"

    class SnapshotExporter:
        """Serialize the package visible through the shared metadata mapping."""

        def to_dashboard_html(self, path: Path, *, theme: str) -> None:
            """Write the captured package as a lightweight HTML surrogate.

            :param Path path: Snapshot destination.
            :param str theme: Theme token supplied by the CLI helper.
            :return None: Writes the captured metadata payload.
            """
            assert theme == "dark"
            path.write_text(json.dumps(metadata["dashboard_collection"]))

    latest = render_dashboard_collection_snapshot(
        package_path,
        dashboard_path=dashboard_path,
        exporter=SnapshotExporter(),
        metadata=metadata,
        theme="dark",
    )

    assert len(latest["results"]) == 2
    assert json.loads(dashboard_path.read_text()) == latest


def test_dashboard_render_failure_preserves_package_and_reports_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A viewer failure must not imply that the completed graph build was lost."""
    graph = build_seed_graph("seed")
    monkeypatch.setattr(
        cli_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    base_factory = _make_exporter_stub({}, methods=("to_dashboard_html",))

    def failing_exporter(*args: object, **kwargs: object) -> SimpleNamespace:
        """Return an exporter whose final viewer write fails.

        :param object args: GraphExporter positional arguments.
        :param object kwargs: GraphExporter keyword arguments.
        :return SimpleNamespace: Exporter stub with a failing dashboard writer.
        """
        exporter = base_factory(*args, **kwargs)

        def fail_dashboard(*_args: object, **_kwargs: object) -> None:
            """Simulate a renderer failure after package persistence.

            :param object _args: Ignored dashboard writer positional arguments.
            :param object _kwargs: Ignored dashboard writer keyword arguments.
            :return None: Always raises.
            """
            raise RuntimeError("plot renderer unavailable")

        exporter.to_dashboard_html = fail_dashboard
        return exporter

    monkeypatch.setattr(cli_module, "GraphExporter", failing_exporter)
    error_log = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_log)
    output_root = tmp_path / "collection"
    output_root.mkdir()
    dashboard_path = output_root / "dashboard.html"
    original_dashboard = b"<html>previous working viewer</html>"
    dashboard_path.write_bytes(original_dashboard)

    result = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "recommendation",
            "--export",
            "dashboard",
            "--output",
            str(output_root),
        ]
    )

    package_path = output_root / DASHBOARD_PACKAGE_FILENAME
    assert result.returncode == 1
    assert len(load_dashboard_package(package_path)["results"]) == 1
    assert dashboard_path.read_bytes() == original_dashboard
    messages = [
        str(call.args[0]) % tuple(call.args[1:]) for call in error_log.call_args_list
    ]
    assert any("Dashboard data was saved safely" in message for message in messages)
    assert any("Add Results" in message for message in messages)


def test_dashboard_package_migrates_safe_legacy_results_non_destructively(
    tmp_path: Path,
) -> None:
    """A valid legacy manifest should migrate confined payload/config artifacts."""
    root = tmp_path / "collection"
    legacy_dir = root / "legacy-seed"
    legacy_dir.mkdir(parents=True)
    legacy_graph = build_seed_graph("legacy")
    legacy_payload_path = legacy_dir / "recommendation.json"
    legacy_config_path = legacy_dir / "recommendation.config.json"
    legacy_payload = _dashboard_graph_payload(legacy_graph, "legacy", "recommendation")
    legacy_payload.pop("kind")
    legacy_payload.pop("schema_version")
    legacy_payload_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    legacy_config_path.write_text(
        json.dumps({"schema_version": 1, "build": {"max_papers": 17}}),
        encoding="utf-8",
    )
    outside_payload = tmp_path / "outside.json"
    outside_config = tmp_path / "outside.config.json"
    outside_payload.write_text(json.dumps(legacy_payload), encoding="utf-8")
    outside_config.write_text(json.dumps({"build": {}}), encoding="utf-8")
    manifest_path = root / "dashboard.manifest.json"
    manifest_payload = {
        "schema_version": 1,
        "results": [
            {
                "result_id": "recommendation:legacy",
                "seed_id": "legacy",
                "title": "Legacy Seed",
                "strategy": "recommendation",
                "summary": legacy_payload["summary"],
                "json_path": "legacy-seed/recommendation.json",
                "config_path": "legacy-seed/recommendation.config.json",
                "updated_at": "2026-01-01T00:00:00",
            },
            {
                "result_id": "recommendation:outside",
                "seed_id": "outside",
                "title": "Unsafe Result",
                "strategy": "recommendation",
                "summary": legacy_payload["summary"],
                "json_path": "../outside.json",
                "config_path": "../outside.config.json",
                "updated_at": "2026-01-01T00:00:00",
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    original_manifest = manifest_path.read_bytes()
    original_graph = legacy_payload_path.read_bytes()
    current_graph = build_seed_graph("current")

    package = update_dashboard_package(
        root / DASHBOARD_PACKAGE_FILENAME,
        graph=current_graph,
        seed_id="current",
        strategy="hybrid",
        payload=_dashboard_graph_payload(current_graph, "current", "hybrid"),
        build={"strategy": "hybrid"},
    )

    assert [entry["result_id"] for entry in package["results"]] == [
        "hybrid:current",
        "recommendation:legacy",
    ]
    migrated = package["results"][1]
    assert migrated["payload"]["kind"] == "citemesh-graph"
    assert migrated["payload"]["schema_version"] == 1
    assert migrated["build"] == {"max_papers": 17}
    assert manifest_path.read_bytes() == original_manifest
    assert legacy_payload_path.read_bytes() == original_graph


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
            methods=("to_dashboard_html",),
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
        "theme": "dark",
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

    status_graph = nx.Graph()
    status_graph.graph["candidate_source_status"] = {
        "references": "unavailable",
        "citations": "empty",
    }
    metadata = _capture_metadata(
        strategy="citation",
        graph=status_graph,
    )
    assert metadata["candidate_source_status"] == {
        "citations": "empty",
        "references": "unavailable",
    }

    metadata = _capture_metadata(
        strategy="embedding",
        extra_args=["--storage-precision", "float32"],
    )
    assert metadata["embedding"] == {
        "effective_vector_dtype": "float32",
        "effective_device": None,
        "effective_compute_dtype": None,
        "model_profile": "auto",
        "retrieval_representation": "retrieval-query/retrieval-document",
        "graph_representation": "graph-similarity",
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
    """Embedding build should keep detailed runtime configuration at debug.

    :param pytest.MonkeyPatch monkeypatch: Pytest patch helper.
    :return None: Assertions verify routine configuration does not fill info logs.
    """
    graph = build_seed_graph("seed")

    debug_mock = MagicMock()
    info_mock = MagicMock()
    warning_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "debug", debug_mock)
    monkeypatch.setattr(cli_module.logger, "info", info_mock)
    monkeypatch.setattr(cli_module.logger, "warning", warning_mock)
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
    debug_messages = [
        str(call.args[0]) for call in debug_mock.call_args_list if call.args
    ]
    info_messages = [
        str(call.args[0]) for call in info_mock.call_args_list if call.args
    ]
    warning_messages = [
        str(call.args[0]) for call in warning_mock.call_args_list if call.args
    ]
    assert any("Embedding config:" in msg for msg in debug_messages)
    assert not any("Embedding config:" in msg for msg in info_messages)
    assert any("No embedding cache found" in msg for msg in info_messages)
    assert not any("No embedding cache found" in msg for msg in warning_messages)


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
                "model_profile": "embeddinggemma",
                "corpus_size": 1234,
                "all_corpus": False,
                "top_k": 4,
                "truncate_dim": 64,
                "min_semantic_similarity": 0.69,
                "streaming": True,
                "binary_rescore_multiplier": 9,
                "calibration_sample_size": 123,
                "encode_batch_size": 48,
            },
            {
                "max_papers": 11,
                "model_name": "m",
                "model_profile": "embeddinggemma",
                "model_revision": None,
                "dataset_split": "train",
                "corpus_size": 1234,
                "truncate_dim": 64,
                "top_k": 4,
                "min_semantic_similarity": 0.69,
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
                "model_profile": "embeddinggemma",
                "corpus_size": 1234,
                "all_corpus": False,
                "truncate_dim": 64,
                "min_semantic_similarity": 0.69,
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
                "model_profile": "embeddinggemma",
                "model_revision": None,
                "dataset_split": "train",
                "corpus_size": 1234,
                "truncate_dim": 64,
                "use_streaming": True,
                "min_semantic_similarity": 0.69,
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
        client_factory = MagicMock()
        monkeypatch.setattr(cli_module, "SemanticScholarClient", client_factory)
        namespace.refresh_paper_cache = True
        namespace._s2_api_key = "configured-key"
        cli_module._build_strategy_graph(namespace, strategy, validate_contract=False)
        client_factory.assert_called_once_with(
            api_key="configured-key", refresh_paper_cache=True
        )
        assert captured == {**expected_kwargs, "client": client_factory.return_value}


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
        model="org/generic-embedding-model",
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


@pytest.mark.parametrize("width", [60, 80, 120])
def test_cli_help_contracts(width: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every command should expose its help without color leaks or clipped lines.

    :param int width: Simulated terminal width in columns.
    :param pytest.MonkeyPatch monkeypatch: Environment override fixture.
    :return None: Assertions verify help content, wrapping, and stream routing.
    """
    monkeypatch.setenv("COLUMNS", str(width))
    monkeypatch.delenv("FORCE_COLOR", raising=False)
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
                "--refresh-paper-cache",
                "--storage-precision",
                "--binary-prefilter",
                "--binary-rescore-multiplier",
                "--calibration-sample-size",
                "--encode-batch-size",
                "--no-torch-compile",
                "--spring-iterations",
                "citation/recommendation",
                "repeat for multiple",
                "dashboard.html",
                DASHBOARD_PACKAGE_FILENAME,
                ".dashboard.html",
            ],
        ),
        (["search", "--help"], ["search", "--limit"]),
        (["cache", "--help"], ["clear", "scan", "Examples"]),
        (["cache", "scan", "--help"], ["cache scan", "--log-level"]),
        (["cache", "clear", "--help"], ["cache clear", "--yes", "config.toml"]),
        (["config", "--help"], ["config", "set", "unset", "Examples"]),
        (["config", "list", "--help"], ["config list", "--log-level"]),
        (["config", "get", "--help"], ["config get KEY", "Dotted config key"]),
        (["config", "set", "--help"], ["config set KEY VALUE", "comma-separated"]),
        (["config", "unset", "--help"], ["config unset KEY", "Dotted config key"]),
        (["config", "path", "--help"], ["config path", "--log-level"]),
    ]
    for args, expected_tokens in cases:
        result = run_cli_command(args)
        assert result.returncode == 0
        assert result.stderr == ""
        assert "\x1b[" not in result.stdout
        assert max(map(len, result.stdout.splitlines())) <= width
        lowered = " ".join(result.stdout.lower().split())
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


def test_single_format_output_writes_into_an_existing_directory(
    tmp_path: Path,
) -> None:
    """An existing --output directory must receive the artifact, not name it.

    :param Path tmp_path: Temporary directory serving as the output target.
    :return None: Assertions pin file-or-directory semantics for one format.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    paths = resolve_output_paths(
        base_output_path=results_dir,
        selected_formats=["json"],
        explicit_output=True,
        strategy="citation",
    )

    assert paths == {"json": results_dir / "citation.json"}
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
    """README and CLI guide command examples should remain parseable."""
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
        try:
            parser.parse_args(argv)
        except SystemExit as exc:
            assert exc.code == 0, f"Invalid documented command: {argv}"


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


def test_build_rejects_unavailable_explicit_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicitly requesting an unavailable device fails as a clean parser error."""

    def _raise_unavailable(_requested: str) -> str:
        """Simulate rejecting a requested unavailable device.

        :param str _requested: Requested device token, ignored by this fixed stub.
        :return str: No value; this stub always raises ``ValueError``.
        """
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


def test_export_metadata_omits_candidate_pool_size_in_corpus_mode() -> None:
    """Corpus builds must not record a candidate pool size they never consult.

    :return None: Assertions align export metadata with the config sidecar.
    """
    corpus_metadata = cli_module._embedding_export_metadata(
        _dispatch_namespace(semantic_source="arxiv-corpus")
    )
    candidates_metadata = cli_module._embedding_export_metadata(
        _dispatch_namespace(semantic_source="candidates", candidate_pool_size=400)
    )

    assert "candidate_pool_size" not in corpus_metadata
    assert candidates_metadata["candidate_pool_size"] == 400


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
