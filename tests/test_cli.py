"""Execution-path tests for CLI commands."""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import runpy
import shlex
import sqlite3
import tempfile
import threading
import webbrowser
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

from citemesh import cli as cli_module
from citemesh.cli import (
    DASHBOARD_COLLECTION_KIND,
    DASHBOARD_COLLECTION_SCHEMA_VERSION,
    DASHBOARD_PACKAGE_FILENAME,
    DashboardPackageError,
    canonicalize_paper_id_for_metadata,
    load_dashboard_package,
    render_dashboard_collection_snapshot,
    resolve_dashboard_collection_outputs,
    resolve_graph_config_path,
    resolve_output_paths,
    update_dashboard_package,
)
from citemesh.cli import build_contract as build_contract_module
from citemesh.cli import build_options as build_options_module
from citemesh.cli import cache_ops as cache_ops_module
from citemesh.cli import console as console_module
from citemesh.cli import graph_config as graph_config_module
from citemesh.cli import outputs as outputs_module
from citemesh.cli import parser as parser_module
from citemesh.cli.commands import build as build_module
from citemesh.cli.commands import search as search_module
from citemesh.cli.outputs import _is_standalone_dashboard_output
from citemesh.core import API_CONFIG, Author, Paper
from citemesh.data import DEFAULT_EMBEDDING_MODEL_NAME
from citemesh.data.cache import CACHE_COORDINATION_DIRNAME
from citemesh.data.embedding_cache import EmbeddingCache
from citemesh.data.user_config import UserConfig
from citemesh.services import SemanticScholarClient
from citemesh.services import semantic_scholar as s2
from citemesh.strategies.candidates import CandidateAcquisitionError
from citemesh.strategies.embedding import (
    DEFAULT_DATASET_SOURCE,
    ENCODE_BATCH_SIZE,
    EmbeddingCacheFingerprintMismatchError,
    EmbeddingGraphBuilder,
)
from citemesh.strategies.hybrid import (
    DEFAULT_MAX_SEMANTIC,
    HYBRID_DEFAULT_MAX_CITATIONS,
    HYBRID_DEFAULT_MAX_PAPERS,
    HYBRID_DEFAULT_MAX_REFERENCES,
)
from citemesh.visualization import GraphExporter as ProductionGraphExporter
from citemesh.visualization import generate_output_path
from citemesh.visualization.dashboard import package as dashboard_package_module
from tests._helpers import (
    PausedFirstWrite,
    build_seed_graph,
    get_paper_id_normalization_cases,
    run_captured_cli,
)


def run_cli_command(args: list[str]) -> SimpleNamespace:
    """Run CLI in-process and capture stdout/stderr.

    :param list[str] args: CLI arguments.
    :return SimpleNamespace: Return code and captured streams.
    """
    return run_captured_cli(lambda: cli_module.main(args))


def run_cli_command_via_sys_argv(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> SimpleNamespace:
    """Run CLI through ``sys.argv`` to exercise ``main(argv=None)``."""
    monkeypatch.setattr("sys.argv", ["citemesh", *args])
    return run_captured_cli(cli_module.main)


def flatten_console_text(text: str) -> str:
    """Collapse Rich's soft line wrapping so phrase assertions survive re-wrapping.

    Log records render through ``RichHandler`` at a fixed 140-column width, so a
    long interpolated path pushes later words onto the next line and splits
    asserted phrases. Collapsing every whitespace run makes those assertions
    independent of how much of the line the path consumed.

    :param str text: Captured stdout or stderr from a CLI run.
    :return str: Text with each whitespace run replaced by a single space.
    """
    return " ".join(text.split())


def unwrapped_console_token(text: str) -> str:
    """Drop every whitespace character so a hard-folded long token rejoins.

    Rich folds a token longer than the remaining line without inserting a
    separator, so a temporary-directory path can be split mid-segment. Removing
    whitespace entirely restores such tokens; use this only to look for values
    that contain no whitespace of their own, such as filesystem paths.

    :param str text: Captured stdout or stderr from a CLI run.
    :return str: Text with every whitespace character removed.
    """
    return "".join(text.split())


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
    requested_methods = set(methods or tuple(outputs_module._EXPORTER_METHOD.values()))
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
        "s2_retry_budget": None,
        "model": DEFAULT_EMBEDDING_MODEL_NAME,
        "model_profile": "auto",
        "model_revision": None,
        "dataset_source": DEFAULT_DATASET_SOURCE,
        "dataset_split": "train",
        "corpus_size": None,
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
    saved_configured = bool(console_module._LOGGING_CONFIGURED)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    console_module._LOGGING_CONFIGURED = False
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
    console_module._LOGGING_CONFIGURED = saved_configured


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


@pytest.mark.parametrize(
    ("arguments", "html_path", "browser_name"),
    [
        ([], "out/dashboard.html", None),
        (["collection dir"], "collection dir/dashboard.html", None),
        (
            ["report #1.html", "--browser", "google-chrome"],
            "report #1.html",
            "google-chrome",
        ),
    ],
)
def test_view_opens_saved_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    html_path: str,
    browser_name: str | None,
) -> None:
    """View resolves saved results and sends a file URI to the chosen browser.

    :param Path tmp_path: Directory containing saved results.
    :param pytest.MonkeyPatch monkeypatch: Isolated browser and working directory.
    :param list[str] arguments: Optional view arguments.
    :param str html_path: Saved HTML path relative to the working directory.
    :param str | None browser_name: Explicit browser override, if any.
    :return None: Checks browser dispatch without loading build configuration.
    """
    monkeypatch.chdir(tmp_path)
    saved_html = tmp_path / html_path
    saved_html.parent.mkdir(parents=True, exist_ok=True)
    saved_html.write_text("<html>Saved result</html>", encoding="utf-8")
    open_default = MagicMock(return_value=True)
    controller = MagicMock()
    controller.open_new_tab.return_value = True
    get_browser = MagicMock(return_value=controller)
    monkeypatch.setattr(webbrowser, "open_new_tab", open_default)
    monkeypatch.setattr(webbrowser, "get", get_browser)
    load_config = MagicMock(side_effect=AssertionError("view loaded build config"))
    monkeypatch.setattr(cli_module, "load_user_config", load_config)

    result = run_cli_command(["view", *arguments])

    assert result.returncode == 0, result.stderr
    if browser_name:
        get_browser.assert_called_once_with(browser_name)
        controller.open_new_tab.assert_called_once_with(saved_html.as_uri())
        open_default.assert_not_called()
    else:
        open_default.assert_called_once_with(saved_html.as_uri())
        get_browser.assert_not_called()
    load_config.assert_not_called()


@pytest.mark.parametrize("target", ["missing.html", "empty-collection", "graph.json"])
def test_view_rejects_missing_or_non_html_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """View should report unusable inputs without launching a browser.

    :param Path tmp_path: Directory containing invalid result selections.
    :param pytest.MonkeyPatch monkeypatch: Isolated browser and working directory.
    :param str target: Missing file, incomplete collection, or graph data file.
    :return None: Checks a useful error and nonzero exit status.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "empty-collection").mkdir()
    (tmp_path / "graph.json").write_text("{}", encoding="utf-8")
    open_default = MagicMock()
    monkeypatch.setattr(webbrowser, "open_new_tab", open_default)

    result = run_cli_command(["view", target])

    assert result.returncode == 1
    flat_stderr = flatten_console_text(result.stderr)
    assert target in unwrapped_console_token(result.stderr)
    if target == "graph.json":
        assert "HTML" in flat_stderr
        assert "Add Results" in flat_stderr
    else:
        assert "Cannot read saved results" in flat_stderr
    open_default.assert_not_called()


@pytest.mark.parametrize("browser_name", [None, "unavailable-browser"])
def test_view_reports_browser_launch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, browser_name: str | None
) -> None:
    """View should report browser failures with the file available to open manually.

    :param Path tmp_path: Directory containing saved HTML.
    :param pytest.MonkeyPatch monkeypatch: Isolated browser launch behavior.
    :param str | None browser_name: Unavailable named browser or default launch failure.
    :return None: Checks the failure exit status and actionable browser error.
    """
    saved_html = tmp_path / "dashboard.html"
    saved_html.write_text("<html/>", encoding="utf-8")
    monkeypatch.setattr(webbrowser, "open_new_tab", lambda _url: False)
    monkeypatch.setattr(
        webbrowser,
        "get",
        MagicMock(side_effect=webbrowser.Error("browser unavailable")),
    )
    arguments = ["view", str(saved_html)]
    if browser_name:
        arguments.extend(["--browser", browser_name])

    result = run_cli_command(arguments)

    assert result.returncode == 1
    assert "browser" in flatten_console_text(result.stderr).lower()
    assert str(saved_html) in unwrapped_console_token(result.stderr)


def test_cache_commands_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache clear/scan should count links without traversing their targets.

    :param Path tmp_path: Temporary cache and external directory locations.
    :param pytest.MonkeyPatch monkeypatch: Overrides the cache-root environment.
    :return None: Checks scan totals, cache removal, and intact symlink targets.
    """
    cache_root = _populate_cache_root(tmp_path, monkeypatch)
    external = tmp_path / "external"
    external.mkdir()
    external_file = external / "payload.bin"
    external_file.write_bytes(b"outside cache" * 1000)
    links = [
        (cache_root / "linked-directory", external),
        (cache_root / "linked-file", external_file),
        (cache_root / "broken-link", external / "missing"),
        (cache_root / "embeddings" / "nested-link", external),
    ]
    for link, target in links:
        link.symlink_to(target, target_is_directory=target == external)
        assert cache_ops_module._scan_path_stats(link) == (1, link.lstat().st_size)
    assert cache_ops_module._scan_path_stats(cache_root) == (
        2 + len(links),
        2050 + sum(link.lstat().st_size for link, _target in links),
    )

    legacy_namespace = "0123456789ab"
    legacy_metadata = cache_root / "embeddings" / f"metadata_{legacy_namespace}.db"
    with sqlite3.connect(legacy_metadata) as connection:
        connection.execute(
            "CREATE TABLE cache_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO cache_metadata (key, value) VALUES (?, ?)",
            ("schema_version", "3"),
        )
    (cache_root / "embeddings" / f"embeddings_{legacy_namespace}.h5").write_bytes(
        b"legacy embeddings"
    )

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
        "Embedding Namespace Details",
        "Legacy dtype-keyed namespace",
    ]:
        assert token in flatten_console_text(scan_result.stdout)

    scan_debug_result = run_cli_command(["cache", "scan", "--log-level", "debug"])
    assert scan_debug_result.returncode == 0, (
        f"STDOUT: {scan_debug_result.stdout}\nSTDERR: {scan_debug_result.stderr}"
    )
    assert "CiteMesh Cache Scan" in flatten_console_text(scan_debug_result.stdout)

    clear_result = run_cli_command(
        ["cache", "clear", "--yes", "--reason", "manual local reset"]
    )
    assert clear_result.returncode == 0, (
        f"STDOUT: {clear_result.stdout}\nSTDERR: {clear_result.stderr}"
    )
    assert cache_root.is_dir()
    assert {child.name for child in cache_root.iterdir()} <= {
        "config.toml.lock",
        CACHE_COORDINATION_DIRNAME,
    }
    assert external_file.read_bytes() == b"outside cache" * 1000


def test_cache_clear_refuses_active_embedding_encode() -> None:
    """Cache clear must not delete a namespace between encode and commit.

    :return None: Validates that an active shared cache operation rejects clear.
    """
    encode_started = threading.Event()
    release_encode = threading.Event()

    class _BlockingEncodeModel:
        """Hold an embedding operation between its cache lookup and commit."""

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            """Wait until the test permits the cache commit.

            :param list[str] texts: Text payload to embed.
            :param object kwargs: Unused encoder keyword arguments.
            :return np.ndarray: One deterministic FP32 vector per requested text.
            """
            del kwargs
            encode_started.set()
            assert release_encode.wait(timeout=5), "encode was never released"
            return np.ones((len(texts), 2), dtype=np.float32)

    cache = EmbeddingCache(model_name="clear-race", storage_precision="float32")
    model = _BlockingEncodeModel()
    with ThreadPoolExecutor(max_workers=1) as executor:
        write = executor.submit(
            cache.get_embeddings,
            {"paper": {"title": "Title", "abstract": "Abstract"}},
            model,
            show_progress=False,
        )
        assert encode_started.wait(timeout=5), "embedding encode never started"
        assert (
            cache_ops_module._clear_cache_directory(assume_yes=True, clear_reason=None)
            == 1
        )
        assert cache.h5_path.is_file()
        release_encode.set()
        write.result(timeout=5)

    assert (
        cache_ops_module._clear_cache_directory(assume_yes=True, clear_reason=None) == 0
    )
    assert cache.cache_dir.parent.joinpath(CACHE_COORDINATION_DIRNAME).is_dir()
    assert not cache.cache_dir.exists()

    repopulated = cache.get_embeddings(
        {"paper-after-clear": {"title": "Title", "abstract": "Abstract"}},
        model,
        show_progress=False,
    )
    assert set(repopulated) == {"paper-after-clear"}
    assert cache.h5_path.is_file()


def test_cache_clear_refuses_active_s2_paper_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache clear must not race an active Semantic Scholar cache operation.

    :param pytest.MonkeyPatch monkeypatch: Pauses paper metadata persistence.
    :return None: Checks the shared root lock keeps the cache intact until completion.
    """
    write_started = threading.Event()
    release_write = threading.Event()
    original_persist = s2.disk_cache._persist_paper

    def blocked_persist(paper: Paper, requested_id: str) -> None:
        """Pause one metadata write while its operation lock remains held.

        :param Paper paper: Resolved paper metadata.
        :param str requested_id: Identifier used for lookup.
        :return None: Persists after the test releases the writer.
        """
        write_started.set()
        assert release_write.wait(timeout=5), "paper write was never released"
        original_persist(paper, requested_id)

    monkeypatch.setattr(s2.disk_cache, "_persist_paper", blocked_persist)
    with SemanticScholarClient(api_key="") as client:
        client._request_json = MagicMock(
            return_value={"paperId": "seed", "title": "Seed"}
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            write = executor.submit(client.get_paper, "seed")
            assert write_started.wait(timeout=5), "paper write never started"
            assert (
                cache_ops_module._clear_cache_directory(
                    assume_yes=True, clear_reason=None
                )
                == 1
            )
            release_write.set()
            assert write.result(timeout=5).paper_id == "seed"

    paper_path = s2.disk_cache._paper_cache_path("seed")
    assert paper_path.is_file()
    assert (
        cache_ops_module._clear_cache_directory(assume_yes=True, clear_reason=None) == 0
    )
    assert not paper_path.exists()


def test_cache_clear_refuses_active_reference_cache_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache clear must not race a public cache-only normalization write.

    :param pytest.MonkeyPatch monkeypatch: Pauses normalized reference persistence.
    :return None: Checks clear reports busy until the reference lookup completes.
    """
    cache_path = s2.disk_cache._reference_cache_path("seed")
    cache_path.write_text(
        json.dumps(
            {
                "paper_id": "seed",
                "references": ["ref", "ref"],
                "version": s2.disk_cache.REFERENCE_CACHE_VERSION,
            }
        ),
        encoding="utf-8",
    )
    write_started = threading.Event()
    release_write = threading.Event()
    with SemanticScholarClient(api_key="") as client:
        original_persist = client._persist_reference_cache_entry

        def blocked_persist(
            path: Path, paper_id: str, reference_ids: list[str]
        ) -> None:
            """Pause normalized persistence while the shared root lock is held.

            :param Path path: Reference cache file being normalized.
            :param str paper_id: Normalized paper identifier.
            :param list[str] reference_ids: Deduplicated reference identifiers.
            :return None: Persists after the test releases the writer.
            """
            write_started.set()
            assert release_write.wait(timeout=5), "reference write was never released"
            original_persist(path, paper_id, reference_ids)

        monkeypatch.setattr(client, "_persist_reference_cache_entry", blocked_persist)
        with ThreadPoolExecutor(max_workers=1) as executor:
            lookup = executor.submit(client.get_cached_reference_ids, "seed")
            assert write_started.wait(timeout=5), "reference write never started"
            assert (
                cache_ops_module._clear_cache_directory(
                    assume_yes=True, clear_reason=None
                )
                == 1
            )
            release_write.set()
            assert lookup.result(timeout=5) == ["ref"]

    assert cache_path.is_file()
    assert json.loads(cache_path.read_text())["references"] == ["ref"]
    assert (
        cache_ops_module._clear_cache_directory(assume_yes=True, clear_reason=None) == 0
    )
    assert not cache_path.exists()


def test_cache_clear_reports_config_inspection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed config stat must abort clearing even when exists hides errors.

    :param Path tmp_path: Temporary cache location.
    :param pytest.MonkeyPatch monkeypatch: Emulates suppressed exists errors.
    :return None: Checks that the error is reported before deleting cache data.
    """
    cache_root = _populate_cache_root(tmp_path, monkeypatch)
    config_path = cache_root / "config.toml"
    config_path.write_text('[defaults]\ntheme = "dark"\n', encoding="utf-8")
    original_stat = Path.stat
    original_exists = Path.exists

    def denied_stat(path: Path, *args: Any, **kwargs: Any) -> Any:
        """Deny config inspection while allowing all other file operations.

        :param Path path: Path being inspected.
        :param Any args: Forwarded positional stat arguments.
        :param Any kwargs: Forwarded keyword stat arguments.
        :return Any: Original stat result for other paths.
        """
        if path == config_path and kwargs.get("follow_symlinks", True):
            raise PermissionError("config stat denied")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied_stat)
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: False if path == config_path else original_exists(path),
    )
    monkeypatch.setattr(
        cache_ops_module, "_confirmed_cache_clear", lambda *_args, **_kwargs: True
    )

    assert (
        cache_ops_module._clear_cache_directory(assume_yes=True, clear_reason=None) == 1
    )
    assert (cache_root / "embeddings" / "vectors.bin").is_file()


def test_cache_clear_declined_at_prompt_keeps_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answering the cache clear prompt with ``n`` should abort and keep every file."""
    cache_root = _populate_cache_root(tmp_path, monkeypatch)
    monkeypatch.setattr(cache_ops_module, "stdin_isatty", lambda: True)
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
    monkeypatch.setattr(cache_ops_module, "stdin_isatty", lambda: True)
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
    monkeypatch.setattr(cache_ops_module, "stdin_isatty", lambda: False)
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
    monkeypatch.setattr(build_module, "_build_strategy_graph", build_graph_mock)
    monkeypatch.setattr(
        build_module,
        "GraphExporter",
        _make_exporter_stub({}, methods=("to_json",)),
    )

    monkeypatch.setattr(cache_ops_module, "stdin_isatty", lambda: True)
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

    monkeypatch.setattr(cache_ops_module, "stdin_isatty", lambda: False)
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
    parser, _, _, _ = parser_module._create_parser()
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
        assert parser_module._pop_tracked_option_dests(parsed) == expected_provided

    assert (
        parser_module._pop_tracked_option_dests(parser.parse_args(["cache", "scan"]))
        == set()
    )


@pytest.mark.parametrize(
    ("argv", "expected_level"),
    [
        (["--verbose", "build", "arxiv:1706.03762"], "debug"),
        (
            [
                "--log-level",
                "warning",
                "build",
                "arxiv:1706.03762",
                "--verbose",
            ],
            "debug",
        ),
        (
            [
                "--verbose",
                "build",
                "arxiv:1706.03762",
                "--log-level",
                "error",
            ],
            "error",
        ),
        (["cache", "scan", "--verbose"], "debug"),
    ],
)
def test_cli_verbose_alias_is_position_agnostic_and_last_option_wins(
    argv: list[str], expected_level: str
) -> None:
    """`--verbose` should share `--log-level` precedence across command nesting."""
    parser, _, _, _ = parser_module._create_parser()

    parsed = parser.parse_args(argv)

    assert parsed.log_level == expected_level
    assert parser_module._pop_tracked_option_dests(parsed) == {"log_level"}


def test_resolve_console_width_uses_auto_width_for_tty_streams() -> None:
    """TTY streams should default Rich consoles to auto width."""
    assert console_module._resolve_console_width(0, interactive=True) is None


def test_resolve_console_width_uses_fixed_width_for_redirected_streams() -> None:
    """Redirected streams should keep a stable fallback width by default."""
    assert (
        console_module._resolve_console_width(0, interactive=False)
        == cli_module.REDIRECTED_LOG_WIDTH
    )
    assert console_module._resolve_console_width(96, interactive=True) == 96


@pytest.mark.parametrize(
    ("log_level", "expected_progress"),
    [("debug", True), ("info", True), ("warning", False), ("error", False)],
)
def test_configure_logging_sets_progress_policy_from_severity(
    monkeypatch: pytest.MonkeyPatch,
    log_level: str,
    expected_progress: bool,
) -> None:
    """Progress bars should appear only for normal and verbose CLI output."""
    saved_handlers, saved_level, saved_configured = _reset_cli_logging_state()
    progress_policy: list[bool] = []
    monkeypatch.setattr(
        console_module,
        "set_progress_enabled",
        lambda enabled: progress_policy.append(enabled),
    )

    try:
        console_module._configure_logging(log_level=log_level)
    finally:
        _restore_cli_logging_state(saved_handlers, saved_level, saved_configured)

    assert progress_policy == [expected_progress]


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
    noisy_logger_names = ("filelock", "matplotlib", "urllib3")
    saved_logger_levels = {
        name: logging.getLogger(name).level for name in noisy_logger_names
    }

    try:
        if preconfigured:
            logging.getLogger().addHandler(logging.NullHandler())
        with redirect_stderr(stderr):
            console_module._configure_logging(
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
            console_module._configure_logging(log_level="info", log_width=0)
            cli_module.load_user_config(config_path)
            for handler in logging.getLogger().handlers:
                handler.flush()
    finally:
        _restore_cli_logging_state(saved_handlers, saved_level, saved_configured)

    warning = stderr.getvalue()
    assert "Ignoring unknown config table" in warning
    assert "[plugin_settings]" in warning


@pytest.mark.parametrize(
    "author_names",
    [[], ["Ashish Vaswani"], ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"]],
)
def test_search_command_prints_results_to_stdout(
    monkeypatch: pytest.MonkeyPatch, author_names: list[str]
) -> None:
    """Search should print ranked metadata and full IDs for shell workflows.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the search client.
    :param list[str] author_names: Empty, single, or abbreviated author list.
    :return None: Verifies safe text, citation formatting, and full identifiers.
    """
    long_paper_id = "0123456789abcdef0123456789abcdef01234567"
    mock_client = MagicMock()
    mock_client.search_papers.return_value = [
        Paper(
            paper_id=long_paper_id,
            title="[Attention] Is All You Need [/bold]",
            year=2017 if author_names else None,
            authors=[Author(name=name) for name in author_names],
            citation_count=12345,
            abstract="Transformer model paper",
        )
    ]
    monkeypatch.setattr(search_module, "get_client", lambda: mock_client)

    result = run_cli_command(["search", "attention", "--limit", "1"])
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "Search results for 'attention'" in result.stdout
    assert "Full paper IDs:" in result.stdout
    assert long_paper_id in result.stdout
    assert result.stdout.count(long_paper_id) == 1
    assert "[Attention] Is All You Need [/bold]" in result.stdout

    plain_stdout = flatten_console_text(result.stdout)
    assert "Citations" in plain_stdout
    assert "12,345" in plain_stdout
    assert "None" not in plain_stdout
    for name in author_names[:2]:
        assert name in plain_stdout
    assert ("et al." in plain_stdout) == (len(author_names) > 2)
    for name in author_names[2:]:
        assert name not in plain_stdout
    assert plain_stdout.index("12,345") < plain_stdout.index("Full paper IDs:")


def test_s2_search_forwards_result_count_and_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2 search should route its result count and recovery budget.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the S2 client factory.
    :return None: Checks search-specific client settings and request arguments.
    """
    monkeypatch.delenv("S2_API_KEY", raising=False)
    mock_client = MagicMock()
    mock_client.search_papers.return_value = [
        Paper(paper_id="result", title="Result", year=None, abstract="Abstract")
    ]
    client_factory = MagicMock(return_value=mock_client)
    monkeypatch.setattr(build_options_module, "SemanticScholarClient", client_factory)

    result = run_cli_command(
        [
            "search",
            "attention",
            "--mode",
            "s2",
            "-n",
            "101",
            "--s2-retry-budget",
            "37.5",
        ]
    )

    assert result.returncode == 0
    client_factory.assert_called_once_with(
        api_key=None,
        refresh_paper_cache=False,
        retry_budget_seconds=37.5,
    )
    mock_client.search_papers.assert_called_once_with(
        "attention",
        limit=101,
        raise_on_unavailable=True,
    )


def _fake_local_search_builder(
    *, cached_count: int, results: list[Any] | None = None
) -> MagicMock:
    """Build a fake EmbeddingGraphBuilder for local-search CLI tests."""
    fake_builder = MagicMock()
    fake_builder.device = "cpu"
    fake_builder.compute_dtype = "float32"
    fake_builder._active_model_name = None
    fake_builder.search_local.return_value = list(results or [])
    fake_builder.has_persistent_embedding_artifacts.return_value = cached_count > 0
    fake_builder.embedding_cache = SimpleNamespace(
        embedding_count=lambda: cached_count,
        last_search_total_embeddings=cached_count,
        h5_path=Path("namespace.h5"),
        hydration_operation_lock=nullcontext,
        payload_stats=lambda: SimpleNamespace(
            hydration_split="train", hydration_corpus_size="all"
        ),
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


@pytest.mark.parametrize(
    "local_flags",
    [
        ["--model", "custom/model"],
        ["--model-profile", "embeddinggemma"],
        ["--model-revision", "frozen-release"],
        ["--device", "cpu"],
        ["--semantic-source", "arxiv-corpus"],
        ["--dataset-source", "research/arxiv-snapshot"],
        ["--truncate-dim", "256"],
        ["--storage-precision", "float32"],
        [
            "--semantic-source",
            "arxiv-corpus",
            "--storage-precision",
            "int8",
            "--calibration-sample-size",
            "100",
        ],
    ],
)
def test_search_mode_s2_rejects_local_flags(
    monkeypatch: pytest.MonkeyPatch, local_flags: list[str]
) -> None:
    """Embedding namespace flags are meaningless for explicit S2 keyword search.

    :param pytest.MonkeyPatch monkeypatch: Fixture capturing the CLI error.
    :param list[str] local_flags: One local-only flag pair rejected by S2 mode.
    :return None: Assertions verify the rejection names every local-only flag.
    """
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)
    result = run_cli_command(
        [
            "search",
            "attention",
            "--mode",
            "s2",
            *local_flags,
        ]
    )
    assert result.returncode == 2
    message = str(error_mock.call_args)
    assert "only apply to local semantic search" in message
    for flag in local_flags[::2]:
        assert flag in message


@pytest.mark.parametrize("mode_args", [[], ["--mode", "local"], ["--mode", "auto"]])
def test_search_rejects_a_dataset_source_against_the_candidates_namespace(
    monkeypatch: pytest.MonkeyPatch, mode_args: list[str]
) -> None:
    """Contradictory namespace flags must fail rather than quietly pick one.

    ``--dataset-source`` names a corpus the build hydrated, so it only means
    anything in ``arxiv-corpus`` mode. Paired with ``--semantic-source
    candidates`` it used to be accepted and then discarded: the search read the
    candidates namespace while the dataset the user named never reached it, in
    every mode. The build contract already refuses this pairing, so it is the
    contract that has to see which options this command line supplied.

    :param pytest.MonkeyPatch monkeypatch: Pytest patch helper.
    :param list[str] mode_args: Implicit, explicit-local, or explicit-auto mode.
    :return None: Asserts the usage exit code, both flag names, and no search.
    """
    builder_factory = MagicMock()
    monkeypatch.setattr(search_module, "EmbeddingGraphBuilder", builder_factory)
    client_factory = MagicMock()
    monkeypatch.setattr(search_module, "get_client", client_factory)
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)

    result = run_cli_command(
        [
            "search",
            "attention",
            *mode_args,
            "--semantic-source",
            "candidates",
            "--dataset-source",
            "research/arxiv-snapshot",
        ]
    )

    assert result.returncode == 2
    message = str(error_mock.call_args)
    assert "--dataset-source" in message
    assert "--semantic-source" in message
    builder_factory.assert_not_called()
    client_factory.assert_not_called()


def test_search_rejects_empty_model_override() -> None:
    """An empty model token should fail parsing instead of silently no-oping.

    :return None: Assertions verify the non-empty model contract.
    """
    result = run_cli_command(["search", "attention", "--model", ""])

    assert result.returncode == 2
    assert "--model" in result.stderr
    assert "must be a non-empty string" in result.stderr


@pytest.mark.parametrize(
    ("option", "prefix"),
    [
        ("--model-revision", ["--strategy", "embedding"]),
        ("--output", []),
    ],
)
def test_build_rejects_empty_option_values(option: str, prefix: list[str]) -> None:
    """Build options with required text values should reject an empty token.

    :param str option: Build option receiving the empty value.
    :param list[str] prefix: Arguments needed to put the option in scope.
    :return None: Assertions verify an argparse usage failure before execution.
    """
    result = run_cli_command(["build", "seed", *prefix, option, ""])

    assert result.returncode == 2
    assert option in result.stderr
    assert "must be a non-empty string" in result.stderr


def test_search_mode_local_prints_cached_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--mode local renders cached results and mirrors flagless build defaults."""
    fake_builder = _fake_local_search_builder(
        cached_count=42, results=[_FAKE_LOCAL_RESULT]
    )
    builder_factory = MagicMock(return_value=fake_builder)
    monkeypatch.setattr(search_module, "EmbeddingGraphBuilder", builder_factory)

    result = run_cli_command(["search", "cached topic", "--mode", "local", "-n", "1"])
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    # Rich folds table cells at console width; compare on whitespace-normalized text.
    plain_stdout = flatten_console_text(result.stdout)
    assert "Local semantic search for 'cached topic'" in plain_stdout
    assert _FAKE_LOCAL_RESULT.paper_id in plain_stdout
    assert "0.876" in plain_stdout
    assert "Ada Lovelace" in plain_stdout
    assert "Searched 42 locally cached embeddings" in plain_stdout
    assert "Score" in plain_stdout
    assert plain_stdout.index("0.876") < plain_stdout.index("Searched 42")
    assert plain_stdout.index("Searched 42") < plain_stdout.index("Full paper IDs:")
    fake_builder.search_local.assert_called_once_with("cached topic", top_k=1)
    fake_builder.prepare_embedding_cache.assert_called_once_with()

    # Namespace parity with a flagless build: candidates mode with the int8
    # default normalized to float32 storage.
    builder_kwargs = builder_factory.call_args.kwargs
    assert builder_kwargs["model_name"] == DEFAULT_EMBEDDING_MODEL_NAME
    assert builder_kwargs["semantic_source"] == "candidates"
    assert builder_kwargs["dataset_source"] == DEFAULT_DATASET_SOURCE
    assert builder_kwargs["storage_precision"] == "float32"


def test_search_mode_local_has_no_s2_result_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local search should accept counts beyond S2's relevance-search window.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to inject the local builder.
    :return None: Checks a large positive count reaches local search unchanged.
    """
    fake_builder = _fake_local_search_builder(
        cached_count=1500,
        results=[_FAKE_LOCAL_RESULT],
    )
    monkeypatch.setattr(
        search_module,
        "EmbeddingGraphBuilder",
        MagicMock(return_value=fake_builder),
    )

    result = run_cli_command(
        ["search", "cached topic", "--mode", "local", "-n", "1001"]
    )

    assert result.returncode == 0
    fake_builder.search_local.assert_called_once_with("cached topic", top_k=1001)


@pytest.mark.parametrize("binary_prefilter", [False, True])
def test_search_local_uses_configured_corpus_dataset_source(
    monkeypatch: pytest.MonkeyPatch,
    binary_prefilter: bool,
) -> None:
    """Local corpus search should target the configured int8 cache namespace.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing local runtime dependencies.
    :param bool binary_prefilter: Persisted binary-prefilter setting.
    :return None: Assertions validate the build-defaults namespace passed to search.
    """
    fake_builder = _fake_local_search_builder(
        cached_count=42, results=[_FAKE_LOCAL_RESULT]
    )
    builder_factory = MagicMock(return_value=fake_builder)
    monkeypatch.setattr(search_module, "EmbeddingGraphBuilder", builder_factory)
    dataset_source = "research/arxiv-snapshot"
    for key, value in {
        "semantic_source": "arxiv-corpus",
        "dataset_source": dataset_source,
        "calibration_sample_size": "200",
        "binary_prefilter": str(binary_prefilter).lower(),
    }.items():
        cli_module.set_config_value(f"defaults.{key}", value)

    result = run_cli_command(["search", "cached topic", "--mode", "local"])

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    builder_kwargs = builder_factory.call_args.kwargs
    assert builder_kwargs["semantic_source"] == "arxiv-corpus"
    assert builder_kwargs["dataset_source"] == dataset_source
    assert builder_kwargs["storage_precision"] == "int8"
    assert builder_kwargs["calibration_sample_size"] == 200
    assert builder_kwargs["binary_prefilter"] is binary_prefilter


@pytest.mark.parametrize(
    ("corpus_token", "expected_corpus_size", "expected_all_corpus"),
    [("newest:7", 7, False), ("all", None, True)],
)
def test_search_local_build_footer_preserves_effective_namespace_and_scope(
    monkeypatch: pytest.MonkeyPatch,
    corpus_token: str,
    expected_corpus_size: int | None,
    expected_all_corpus: bool,
) -> None:
    """The copyable result command must reopen the searched namespace and scope.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing local search runtime.
    :param str corpus_token: Corpus scope recorded on the searched cache.
    :param Optional[int] expected_corpus_size: Expected generated finite cap.
    :param bool expected_all_corpus: Whether the command should request all rows.
    :return None: Parses the footer and compares selectors plus recorded scope.
    """
    fake_builder = _fake_local_search_builder(
        cached_count=42, results=[_FAKE_LOCAL_RESULT]
    )
    fake_builder.embedding_cache.payload_stats = lambda: SimpleNamespace(
        hydration_split="train[:5%]", hydration_corpus_size=corpus_token
    )
    builder_factory = MagicMock(return_value=fake_builder)
    monkeypatch.setattr(search_module, "EmbeddingGraphBuilder", builder_factory)
    search_args = [
        "search",
        "cached topic",
        "--mode",
        "local",
        "--model",
        "custom/model",
        "--model-profile",
        "default",
        "--model-revision",
        "release candidate",
        "--semantic-source",
        "arxiv-corpus",
        "--dataset-source",
        "research/arxiv-snapshot",
        "--truncate-dim",
        "256",
        "--storage-precision",
        "int8",
        "--calibration-sample-size",
        "100",
    ]

    result = run_cli_command(search_args)

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    footer = next(
        line for line in result.stdout.splitlines() if line.startswith("Use a paper ID")
    )
    command = footer.removeprefix("Use a paper ID with: ")
    command_args = shlex.split(command)
    assert command_args[:3] == ["citemesh", "build", "<ID>"]
    _parser, build_parser, _cache_parser, _config_parser = (
        parser_module._create_parser()
    )
    generated = build_parser.parse_args(command_args[2:])
    searched = builder_factory.call_args.kwargs

    assert generated.strategy == "embedding"
    assert generated.model == searched["model_name"]
    assert generated.model_profile == searched["model_profile"]
    assert generated.model_revision == searched["model_revision"]
    assert generated.semantic_source == searched["semantic_source"]
    assert generated.dataset_source == searched["dataset_source"]
    assert generated.truncate_dim == searched["truncate_dim"]
    assert generated.storage_precision == searched["storage_precision"]
    assert generated.calibration_sample_size == searched["calibration_sample_size"]
    assert generated.dataset_split == "train[:5%]"
    assert generated.all_corpus is expected_all_corpus
    if expected_corpus_size is not None:
        assert generated.corpus_size == expected_corpus_size


@pytest.mark.parametrize("storage_precision", ["float32", "int8"])
def test_search_local_captures_result_and_replay_scope_under_one_lock(
    monkeypatch: pytest.MonkeyPatch,
    storage_precision: str,
) -> None:
    """A competing rebuild must not change the footer scope after search.

    :param pytest.MonkeyPatch monkeypatch: CLI and cache isolation fixture.
    :param str storage_precision: Persistent float32 or int8 representation.
    :return None: Parses the rendered command and verifies the searched split.
    """
    fake_builder = _fake_local_search_builder(
        cached_count=1,
        results=[_FAKE_LOCAL_RESULT],
    )
    cache = fake_builder.embedding_cache
    depth = 0
    search_completed = False
    persisted_split = "train[:1]"
    events: list[str] = []

    @contextmanager
    def operation_lock() -> Any:
        """Replace the corpus immediately after the consuming lock is released."""
        nonlocal depth, persisted_split
        depth += 1
        try:
            yield
        finally:
            depth -= 1
            if depth == 0 and search_completed and "replacement" not in events:
                persisted_split = "validation[:1]"
                events.append("replacement")

    def search_local(_query: str, *, top_k: int) -> list[Any]:
        """Return training results from the builder's normal nested lock."""
        nonlocal search_completed
        assert top_k == 1
        with operation_lock():
            events.append("search")
            search_completed = True
            return [_FAKE_LOCAL_RESULT]

    def payload_stats() -> SimpleNamespace:
        """Expose whichever corpus owns the namespace at capture time."""
        events.append("scope")
        return SimpleNamespace(
            hydration_split=persisted_split,
            hydration_corpus_size="newest:1",
        )

    cache.hydration_operation_lock = MagicMock(side_effect=operation_lock)
    cache.payload_stats = payload_stats
    fake_builder.search_local.side_effect = search_local
    builder_factory = MagicMock(return_value=fake_builder)
    monkeypatch.setattr(search_module, "EmbeddingGraphBuilder", builder_factory)
    args = [
        "search",
        "cached topic",
        "--mode",
        "local",
        "--limit",
        "1",
        "--semantic-source",
        "arxiv-corpus",
        "--dataset-source",
        "fixture/corpus",
        "--storage-precision",
        storage_precision,
    ]
    if storage_precision == "int8":
        args.extend(["--calibration-sample-size", "2"])

    result = run_cli_command(args)

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert _FAKE_LOCAL_RESULT.paper_id in flatten_console_text(result.stdout)
    footer = next(
        line for line in result.stdout.splitlines() if line.startswith("Use a paper ID")
    )
    command_args = shlex.split(footer.removeprefix("Use a paper ID with: "))
    _parser, build_parser, _cache_parser, _config_parser = (
        parser_module._create_parser()
    )
    generated = build_parser.parse_args(command_args[2:])
    assert generated.dataset_split == "train[:1]"
    assert generated.corpus_size == 1
    assert events.index("scope") < events.index("replacement")
    assert depth == 0


def test_search_local_candidate_mode_does_not_acquire_corpus_operation_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate-cache search keeps its existing lock-free render path.

    :param pytest.MonkeyPatch monkeypatch: CLI builder isolation fixture.
    :return None: Verifies candidate mode never opens the corpus operation lock.
    """
    fake_builder = _fake_local_search_builder(
        cached_count=1,
        results=[_FAKE_LOCAL_RESULT],
    )
    operation_lock = MagicMock(
        side_effect=AssertionError("candidate mode must not acquire the corpus lock")
    )
    fake_builder.embedding_cache.hydration_operation_lock = operation_lock
    monkeypatch.setattr(
        search_module,
        "EmbeddingGraphBuilder",
        MagicMock(return_value=fake_builder),
    )

    result = run_cli_command(["search", "cached topic", "--mode", "local"])

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    operation_lock.assert_not_called()


def test_search_local_keeps_results_when_corpus_scope_was_not_hydrated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python-populated corpus cache rows should remain searchable without a footer.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing local search runtime.
    :return None: Checks usable results and focused missing-scope guidance.
    """
    fake_builder = _fake_local_search_builder(
        cached_count=42, results=[_FAKE_LOCAL_RESULT]
    )
    fake_builder.embedding_cache.payload_stats = lambda: SimpleNamespace(
        hydration_split=None, hydration_corpus_size=None
    )
    monkeypatch.setattr(
        search_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )
    warning = MagicMock()
    monkeypatch.setattr(cli_module.logger, "warning", warning)

    result = run_cli_command(
        [
            "search",
            "cached topic",
            "--mode",
            "local",
            "--semantic-source",
            "arxiv-corpus",
        ]
    )

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    plain_stdout = flatten_console_text(result.stdout)
    assert _FAKE_LOCAL_RESULT.paper_id in plain_stdout
    assert "Use a paper ID with:" not in plain_stdout
    assert "no recorded dataset split" in str(warning.call_args)
    assert "Search results remain usable" in str(warning.call_args)


def test_search_local_build_footer_uses_active_fallback_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local-search footer must reopen the active fallback cache namespace.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing local search runtime.
    :return None: Parses the footer and checks active model plus stable selectors.
    """
    active_model = "google/embeddinggemma-300m"
    fake_builder = _fake_local_search_builder(
        cached_count=42, results=[_FAKE_LOCAL_RESULT]
    )
    fake_builder._active_model_name = active_model
    builder_factory = MagicMock(return_value=fake_builder)
    monkeypatch.setattr(search_module, "EmbeddingGraphBuilder", builder_factory)

    result = run_cli_command(
        [
            "search",
            "cached topic",
            "--mode",
            "local",
            "--model-profile",
            "embeddinggemma",
            "--truncate-dim",
            "256",
            "--storage-precision",
            "float32",
        ]
    )

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    plain_stdout = flatten_console_text(result.stdout)
    assert f"model={active_model}" in plain_stdout
    assert f"model={DEFAULT_EMBEDDING_MODEL_NAME}" not in plain_stdout
    footer = next(
        line for line in result.stdout.splitlines() if line.startswith("Use a paper ID")
    )
    command_args = shlex.split(footer.removeprefix("Use a paper ID with: "))
    assert command_args[:3] == ["citemesh", "build", "<ID>"]
    _parser, build_parser, _cache_parser, _config_parser = (
        parser_module._create_parser()
    )
    generated = build_parser.parse_args(command_args[2:])
    searched = builder_factory.call_args.kwargs

    assert generated.model == active_model
    assert generated.model_profile == searched["model_profile"]
    assert generated.model_revision == searched["model_revision"]
    assert generated.truncate_dim == searched["truncate_dim"]
    assert generated.storage_precision == searched["storage_precision"]
    assert generated.calibration_sample_size == searched["calibration_sample_size"]


def test_search_semantic_source_flag_selects_corpus_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--semantic-source arxiv-corpus must reach the corpus cache namespace.

    Storage precision is part of the namespace and the build contract coerces
    ``int8`` to ``float32`` outside corpus mode, so the override has to land
    before validation or the search targets a namespace no corpus build wrote.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the local builder.
    :return None: Assertions pin the corpus namespace the builder receives.
    """
    fake_builder = _fake_local_search_builder(
        cached_count=5999, results=[_FAKE_LOCAL_RESULT]
    )
    builder_factory = MagicMock(return_value=fake_builder)
    monkeypatch.setattr(search_module, "EmbeddingGraphBuilder", builder_factory)

    result = run_cli_command(
        [
            "search",
            "cached topic",
            "--mode",
            "local",
            "--semantic-source",
            "arxiv-corpus",
        ]
    )

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    builder_kwargs = builder_factory.call_args.kwargs
    assert builder_kwargs["semantic_source"] == "arxiv-corpus"
    assert builder_kwargs["storage_precision"] == "int8"
    assert builder_kwargs["dataset_source"] == DEFAULT_DATASET_SOURCE


def test_search_identity_flags_select_one_off_build_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local search should accept every user-selectable cache identity field.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the local builder.
    :return None: Assertions verify one-off build selectors reach the builder.
    """
    fake_builder = _fake_local_search_builder(
        cached_count=5999, results=[_FAKE_LOCAL_RESULT]
    )
    builder_factory = MagicMock(return_value=fake_builder)
    monkeypatch.setattr(search_module, "EmbeddingGraphBuilder", builder_factory)

    result = run_cli_command(
        [
            "search",
            "cached topic",
            "--model-revision",
            "frozen-release",
            "--semantic-source",
            "arxiv-corpus",
            "--truncate-dim",
            "256",
            "--storage-precision",
            "int8",
            "--calibration-sample-size",
            "4000",
        ]
    )

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    builder_kwargs = builder_factory.call_args.kwargs
    assert builder_kwargs["model_revision"] == "frozen-release"
    assert builder_kwargs["semantic_source"] == "arxiv-corpus"
    assert builder_kwargs["truncate_dim"] == 256
    assert builder_kwargs["storage_precision"] == "int8"
    assert builder_kwargs["calibration_sample_size"] == 4000


def test_search_dataset_source_flag_implies_corpus_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--dataset-source alone implies arxiv-corpus, mirroring the build contract.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the local builder.
    :return None: Assertions verify the implied corpus namespace.
    """
    fake_builder = _fake_local_search_builder(
        cached_count=5999, results=[_FAKE_LOCAL_RESULT]
    )
    builder_factory = MagicMock(return_value=fake_builder)
    monkeypatch.setattr(search_module, "EmbeddingGraphBuilder", builder_factory)
    dataset_source = "research/arxiv-snapshot"

    result = run_cli_command(
        [
            "search",
            "cached topic",
            "--mode",
            "local",
            "--dataset-source",
            dataset_source,
        ]
    )

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    builder_kwargs = builder_factory.call_args.kwargs
    assert builder_kwargs["semantic_source"] == "arxiv-corpus"
    assert builder_kwargs["dataset_source"] == dataset_source
    assert builder_kwargs["storage_precision"] == "int8"


def test_configured_local_corpus_hybrid_build_uses_full_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Configured corpus builds should hydrate the full selected split by default.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing runtime dependencies.
    :param Path tmp_path: Isolated dashboard collection location.
    :return None: Assertions verify the builder and generated sidecar contracts.
    """
    captured_builder: dict[str, object] = {}
    graph = build_seed_graph("arxiv:2609.03430")
    monkeypatch.setattr(
        cli_module,
        "load_user_config",
        lambda: UserConfig(
            path=Path("cfg-home") / "config.toml",
            defaults={
                "search_mode": "local",
                "semantic_source": "arxiv-corpus",
            },
        ),
    )
    monkeypatch.setattr(
        build_options_module,
        "HybridGraphBuilder",
        _make_builder_stub(
            captured_builder,
            graph=graph,
            seed_id="arxiv:2609.03430",
        ),
    )
    monkeypatch.setattr(
        build_module,
        "GraphExporter",
        _make_exporter_stub({}, methods=("to_dashboard_html",)),
    )

    output_dir = tmp_path / "dashboard"
    result = run_cli_command(
        [
            "build",
            "arxiv:2609.03430",
            "--strategy",
            "hybrid",
            "--export",
            "dashboard",
            "-o",
            str(output_dir),
        ]
    )

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert captured_builder["semantic_source"] == "arxiv-corpus"
    assert captured_builder["corpus_size"] is None
    config_files = sorted(output_dir.rglob("*.config.json"))
    assert len(config_files) == 1
    payload = json.loads(config_files[0].read_text(encoding="utf-8"))
    embedding = payload["build"]["embedding"]
    assert embedding["semantic_source"] == "arxiv-corpus"
    assert embedding.get("corpus_size") is None
    assert embedding["all_corpus"] is True


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
    monkeypatch.setattr(search_module, "EmbeddingGraphBuilder", builder_factory)
    config = UserConfig(
        path=Path("cfg-home") / "config.toml",
        defaults={"strategy": "recommendation", "calibration_sample_size": 100},
    )
    monkeypatch.setattr(cli_module, "load_user_config", lambda: config)
    client_factory = MagicMock()
    monkeypatch.setattr(search_module, "get_client", client_factory)

    result = run_cli_command(["search", "cached topic", *search_args])

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    builder_kwargs = builder_factory.call_args.kwargs
    assert builder_kwargs["semantic_source"] == "candidates"
    assert builder_kwargs["dataset_source"] == DEFAULT_DATASET_SOURCE
    assert builder_kwargs["storage_precision"] == "float32"
    assert builder_kwargs["binary_prefilter"] is False
    assert builder_kwargs["calibration_sample_size"] == (
        cli_module.EMBEDDING_STORAGE_CONFIG.calibration_sample_size
    )
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
    device_resolver = MagicMock(side_effect=ValueError("CUDA is unavailable"))
    monkeypatch.setattr(search_module, "resolve_embedding_device", device_resolver)
    monkeypatch.setattr(
        build_contract_module, "resolve_embedding_device", device_resolver
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
    monkeypatch.setattr(search_module, "get_client", lambda: mock_client)
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
    device_resolver = MagicMock(side_effect=ValueError("CUDA is unavailable"))
    monkeypatch.setattr(search_module, "resolve_embedding_device", device_resolver)
    monkeypatch.setattr(
        build_contract_module, "resolve_embedding_device", device_resolver
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
    monkeypatch.setattr(search_module, "resolve_embedding_device", resolver)
    monkeypatch.setattr(build_contract_module, "resolve_embedding_device", resolver)
    fake_builder = _fake_local_search_builder(
        cached_count=42, results=[_FAKE_LOCAL_RESULT]
    )
    builder_factory = MagicMock(return_value=fake_builder)
    monkeypatch.setattr(search_module, "EmbeddingGraphBuilder", builder_factory)

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
        (["--semantic-source", "arxiv-corpus"], "semantic_source", "arxiv-corpus"),
        (
            ["--dataset-source", "research/arxiv-snapshot"],
            "dataset_source",
            "research/arxiv-snapshot",
        ),
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
    monkeypatch.setattr(search_module, "EmbeddingGraphBuilder", builder_factory)
    client_factory = MagicMock()
    monkeypatch.setattr(search_module, "get_client", client_factory)

    result = run_cli_command(["search", "cached topic", *namespace_args])
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "Local semantic search for 'cached topic'" in flatten_console_text(
        result.stdout
    )
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
        search_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )
    client_factory = MagicMock()
    monkeypatch.setattr(search_module, "get_client", client_factory)
    info_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "info", info_mock)

    result = run_cli_command(["search", "cached topic", "-n", "1"])
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "Local semantic search for 'cached topic'" in flatten_console_text(
        result.stdout
    )
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
        search_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )
    client_factory = MagicMock()
    monkeypatch.setattr(search_module, "get_client", client_factory)

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
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Default (auto) mode falls back to S2 keyword search with a notice."""
    fake_builder = _fake_local_search_builder(cached_count=0)
    monkeypatch.setattr(
        search_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
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
    monkeypatch.setattr(search_module, "get_client", lambda: mock_client)
    with caplog.at_level(logging.INFO, logger=cli_module.logger.name):
        result = run_cli_command(["search", "attention", "--limit", "1"])
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "Search results for 'attention'" in result.stdout
    notices = [record.getMessage() for record in caplog.records]
    assert any(
        "searching the Semantic Scholar API instead" in notice for notice in notices
    )
    empty_notice = next(
        notice for notice in notices if "Local embedding cache is empty" in notice
    )
    assert "semantic-source=candidates" in empty_notice
    assert f"dataset-source={DEFAULT_DATASET_SOURCE}" in empty_notice
    assert "pass --semantic-source arxiv-corpus" in empty_notice
    # The namespace has no device or compute-dtype token; guidance must not imply one.
    assert "device=" not in empty_notice
    assert "compute dtype" not in empty_notice
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
        search_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )

    result = run_cli_command(["search", "anything", "--mode", "local"])
    assert result.returncode == 1
    message = str(error_mock.call_args)
    assert "Local search was requested via" in message
    assert "--mode local" in message
    assert "has no vectors" in message
    assert "pass --semantic-source arxiv-corpus" in message
    # Device and compute dtype are absent from the namespace contract.
    assert "device=" not in message
    assert "compute_dtype" not in message
    assert "semantic-source=%s" in error_mock.call_args.args[0]
    assert error_mock.call_args.args[3] == "candidates"
    assert error_mock.call_args.args[4] == DEFAULT_DATASET_SOURCE
    fake_builder.prepare_embedding_cache.assert_not_called()
    fake_builder.search_local.assert_not_called()


def test_search_auto_reports_selectors_when_cache_prepare_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Auto fallback should identify a constructed namespace after a load failure.

    :param pytest.MonkeyPatch monkeypatch: Replaces local and S2 search dependencies.
    :param pytest.LogCaptureFixture caplog: Captures fallback diagnostics.
    :return None: Assertions verify the effective namespace selectors are logged.
    """
    fake_builder = _fake_local_search_builder(cached_count=1)
    fake_builder.device = "cuda"
    fake_builder.compute_dtype = "bfloat16"
    fake_builder.prepare_embedding_cache.side_effect = RuntimeError("model unavailable")
    monkeypatch.setattr(
        search_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )
    s2_search = MagicMock(return_value=0)
    monkeypatch.setattr(search_module, "_run_s2_search", s2_search)

    parser, build_parser, _cache_parser, _config_parser = parser_module._create_parser()
    args = parser.parse_args(["search", "attention"])
    with caplog.at_level(logging.INFO, logger=cli_module.logger.name):
        result = search_module._run_search_command(
            args, build_parser, UserConfig(path=Path("config.toml"), defaults={})
        )

    assert result == 0
    assert any(
        "device=cuda compute_dtype=bfloat16" in record.getMessage()
        for record in caplog.records
    )
    s2_search.assert_called_once_with(args)


def test_search_auto_refuses_corpus_fingerprint_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto search must surface a protected corpus fingerprint mismatch.

    :param pytest.MonkeyPatch monkeypatch: Replaces the local cache and S2 path.
    :return None: Asserts the mismatch exits non-zero without a keyword fallback.
    """
    fake_builder = _fake_local_search_builder(cached_count=1)
    fake_builder.prepare_embedding_cache.side_effect = EmbeddingCacheFingerprintMismatchError(
        "protected corpus cache; run `citemesh build <paper-id> --strategy embedding "
        "--semantic-source arxiv-corpus --force-rebuild-cache --overwrite-cache` "
        "with the same model and corpus options"
    )
    monkeypatch.setattr(
        search_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )
    s2_search = MagicMock(return_value=0)
    monkeypatch.setattr(search_module, "_run_s2_search", s2_search)
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)

    parser, build_parser, _cache_parser, _config_parser = parser_module._create_parser()
    args = parser.parse_args(["search", "attention"])
    result = search_module._run_search_command(
        args, build_parser, UserConfig(path=Path("config.toml"), defaults={})
    )

    assert result == 1
    message = str(error_mock.call_args)
    assert "protected corpus cache" in message
    assert "citemesh build <paper-id>" in message
    assert "--strategy embedding" in message
    assert "--force-rebuild-cache --overwrite-cache" in message
    s2_search.assert_not_called()


def test_search_auto_empty_corpus_cache_names_the_selected_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto fallback must name a corpus selector without recommending it again.

    :param pytest.MonkeyPatch monkeypatch: Replaces the local cache and S2 path.
    :return None: Asserts the fallback notices both effective source selectors.
    """
    dataset_source = "research/arxiv-snapshot"
    fake_builder = _fake_local_search_builder(cached_count=0)
    monkeypatch.setattr(
        search_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )
    s2_search = MagicMock(return_value=0)
    monkeypatch.setattr(search_module, "_run_s2_search", s2_search)
    info_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "info", info_mock)

    parser, build_parser, _cache_parser, _config_parser = parser_module._create_parser()
    args = parser.parse_args(["search", "attention"])
    config = UserConfig(
        path=Path("config.toml"),
        defaults={
            "semantic_source": "arxiv-corpus",
            "dataset_source": dataset_source,
        },
    )
    result = search_module._run_search_command(args, build_parser, config)

    assert result == 0
    assert "semantic-source=%s" in info_mock.call_args.args[0]
    assert info_mock.call_args.args[2] == "arxiv-corpus"
    assert info_mock.call_args.args[3] == dataset_source
    assert "already the arXiv-corpus namespace" in info_mock.call_args.args[4]
    assert "pass --semantic-source arxiv-corpus" not in info_mock.call_args.args[4]
    s2_search.assert_called_once_with(args)


def test_search_local_empty_corpus_cache_names_the_selected_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit local search must not recommend its active corpus source.

    :param pytest.MonkeyPatch monkeypatch: Replaces the local cache and logger.
    :return None: Asserts the empty-cache error names both source selectors.
    """
    dataset_source = "research/arxiv-snapshot"
    fake_builder = _fake_local_search_builder(cached_count=0)
    monkeypatch.setattr(
        search_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)

    result = run_cli_command(
        [
            "search",
            "attention",
            "--mode",
            "local",
            "--semantic-source",
            "arxiv-corpus",
            "--dataset-source",
            dataset_source,
        ]
    )

    assert result.returncode == 1
    assert "semantic-source=%s" in error_mock.call_args.args[0]
    assert error_mock.call_args.args[3] == "arxiv-corpus"
    assert error_mock.call_args.args[4] == dataset_source
    assert "already the arXiv-corpus namespace" in error_mock.call_args.args[5]
    assert "pass --semantic-source arxiv-corpus" not in error_mock.call_args.args[5]


def test_search_explicit_auto_with_namespace_flag_keeps_s2_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit --mode auto must keep its S2 fallback despite --model.

    :param pytest.MonkeyPatch monkeypatch: Builder and client stubs.
    :return None: Assertions verify the fallback path runs instead of erroring.
    """
    fake_builder = _fake_local_search_builder(cached_count=0)
    monkeypatch.setattr(
        search_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
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
    monkeypatch.setattr(search_module, "get_client", lambda: mock_client)
    info_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "info", info_mock)

    result = run_cli_command(
        ["search", "anything", "--mode", "auto", "--model", "custom/model"]
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "searching the Semantic Scholar API instead" in str(info_mock.call_args_list)
    mock_client.search_papers.assert_called_once()
    fake_builder.search_local.assert_not_called()


def test_search_explicit_auto_with_namespace_flag_falls_back_for_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrelated namespace flag must not claim a config device failure.

    :param pytest.MonkeyPatch monkeypatch: Device resolver and S2 fallback stubs.
    :return None: Assertions verify config attribution preserves auto fallback.
    """
    parser, build_parser, _cache_parser, _config_parser = parser_module._create_parser()
    args = parser.parse_args(
        ["search", "anything", "--mode", "auto", "--model", "custom/model"]
    )
    config = UserConfig(
        path=Path("cfg-home") / "config.toml", defaults={"device": "cuda"}
    )
    monkeypatch.setattr(
        build_contract_module,
        "resolve_embedding_device",
        MagicMock(side_effect=ValueError("Configured cuda unavailable")),
    )
    s2_search = MagicMock(return_value=0)
    monkeypatch.setattr(search_module, "_run_s2_search", s2_search)

    result = search_module._run_search_command(args, build_parser, config)

    assert result == 0
    s2_search.assert_called_once_with(args)


def test_search_namespace_flag_empty_cache_error_names_the_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The implied-local error must not claim the user passed --mode local.

    :param pytest.MonkeyPatch monkeypatch: Builder stub and error capture.
    :return None: Assertions pin the implied-local attribution text.
    """
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)
    fake_builder = _fake_local_search_builder(cached_count=0)
    monkeypatch.setattr(
        search_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
    )

    result = run_cli_command(["search", "anything", "--model", "custom/model"])
    assert result.returncode == 1
    message = str(error_mock.call_args)
    assert "these flags imply local search" in message
    assert "--mode local" not in message
    assert "--model" in message
    assert "--model-profile" not in message
    assert "--model-revision" not in message
    fake_builder.search_local.assert_not_called()


def test_search_mode_from_config_local_empty_cache_cites_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config-driven local mode errors on an empty cache and cites config.toml."""
    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)
    fake_builder = _fake_local_search_builder(cached_count=0)
    monkeypatch.setattr(
        search_module, "EmbeddingGraphBuilder", MagicMock(return_value=fake_builder)
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
        build_module,
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
        build_module,
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


def test_required_discovery_failure_preserves_existing_exports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed current-discovery check must not replace completed exports.

    :param pytest.MonkeyPatch monkeypatch: Replaces graph acquisition and logging.
    :param Path tmp_path: Existing artifact directory.
    :return None: Verifies failure wording and byte-for-byte output preservation.
    """
    output_dir = tmp_path / "existing-results"
    output_dir.mkdir()
    original_outputs = {
        output_dir / "recommendation.png": b"previous png",
        output_dir / "recommendation.json": b'{"previous": "json"}',
        output_dir / "recommendation.html": b"<html>previous</html>",
    }
    for path, content in original_outputs.items():
        path.write_bytes(content)

    error_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", error_mock)
    monkeypatch.setattr(
        build_module,
        "_build_strategy_graph",
        MagicMock(
            side_effect=CandidateAcquisitionError(
                "Could not complete current recommendation discovery."
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
            "png",
            "--export",
            "json",
            "--export",
            "html",
            "--output",
            str(output_dir),
        ]
    )

    assert result.returncode == 1
    assert {path: path.read_bytes() for path in original_outputs} == original_outputs
    assert error_mock.call_count == 1
    assert (
        error_mock.call_args.args[0]
        == "Build incomplete: current Semantic Scholar discovery could not be acquired. %s"
    )
    assert "current recommendation discovery" in str(error_mock.call_args.args[1])


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
        (["build", "seed", "--similarity-threshold", "banana"], "must be a float"),
        (["build", "   "], "must be a non-empty string"),
        (["build", "seed", "--strategy", "unknown"], "invalid choice"),
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
            ["build", "seed", "--strategy", "citation", "--batch-size", "128"],
            "--batch-size",
        ),
        (
            ["build", "seed", "--strategy", "citation", "-bs", "128"],
            "--batch-size",
        ),
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
        build_module,
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
    _, build_parser, _, _ = parser_module._create_parser()
    args = build_parser.parse_args(
        ["seed", "--strategy", "embedding", "--cache-compression", "lzf"]
    )
    build_contract_module._validate_build_cli_contract(
        args, build_parser, provided={"cache_compression"}
    )
    assert args.cache_compression == "lzf"
    assert args.cache_compression_level == 0


def test_hybrid_implicit_budget_defaults_contract() -> None:
    """Hybrid should apply tuned defaults only when budget knobs are omitted."""
    _, build_parser, _, _ = parser_module._create_parser()
    hybrid_defaults = build_parser.parse_args(["seed", "--strategy", "hybrid"])
    build_contract_module._validate_build_cli_contract(
        hybrid_defaults, build_parser, provided=set()
    )
    assert hybrid_defaults.max_papers == HYBRID_DEFAULT_MAX_PAPERS
    assert hybrid_defaults.max_citations == HYBRID_DEFAULT_MAX_CITATIONS
    assert hybrid_defaults.max_references == HYBRID_DEFAULT_MAX_REFERENCES
    assert build_options_module._resolved_hybrid_max_semantic(hybrid_defaults) == min(
        DEFAULT_MAX_SEMANTIC, HYBRID_DEFAULT_MAX_PAPERS - 1
    )

    explicit_hybrid = build_parser.parse_args(
        [
            "seed",
            "--strategy",
            "hybrid",
            "--max-papers",
            "40",
            "--max-citations",
            "6",
            "--max-references",
            "7",
            "--max-semantic",
            "5",
        ]
    )
    build_contract_module._validate_build_cli_contract(
        explicit_hybrid,
        build_parser,
        provided={"max_papers", "max_citations", "max_references", "max_semantic"},
    )
    assert explicit_hybrid.max_papers == 40
    assert explicit_hybrid.max_citations == 6
    assert explicit_hybrid.max_references == 7
    assert build_options_module._resolved_hybrid_max_semantic(explicit_hybrid) == 5


def test_layout_and_json_export_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build path should share one seeded layout across PNG and JSON exports."""
    graph = build_seed_graph("seed")

    shared_layout = {"seed": (0.0, 0.0)}
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        build_module,
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

    monkeypatch.setattr(build_module, "compute_layout", _fake_compute_layout)
    monkeypatch.setattr(
        build_module,
        "GraphExporter",
        _make_exporter_stub(captured, methods=("to_json",)),
    )

    def _fake_visualize(*args: Any, **kwargs: Any) -> None:
        del args
        captured["visualize_layout"] = kwargs.get("layout")

    monkeypatch.setattr(build_module, "visualize_graph", _fake_visualize)

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

    monkeypatch.setattr(build_module, "compute_layout", _fake_json_compute_layout)
    captured.clear()
    monkeypatch.setattr(
        build_module,
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
        build_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        build_module,
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
            f".{DASHBOARD_PACKAGE_FILENAME}.lock",
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
        build_module,
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
        build_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        build_module,
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
        f".{DASHBOARD_PACKAGE_FILENAME}.lock",
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
        build_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: next(build_results),
    )
    monkeypatch.setattr(
        build_module,
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
    """Refresh one stable result slot and remove its obsolete export formats."""
    seed_graph = build_seed_graph("seed")
    seed_graph.nodes["seed"]["title"] = "Original Seed Title"
    updated_graph = build_seed_graph("seed")
    updated_graph.nodes["seed"]["title"] = "Corrected Seed Title"
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
        build_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: next(build_results),
    )
    monkeypatch.setattr(
        build_module,
        "GraphExporter",
        _make_exporter_stub(
            captured,
            methods=(
                "to_dashboard_html",
                "to_interactive_html",
                "to_plotly_html",
                "to_csv",
                "to_bibtex",
                "to_graphml",
            ),
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
                "--export",
                "html",
                "--export",
                "plotly",
                "--export",
                "csv",
                "--export",
                "bibtex",
                "--export",
                "graphml",
                "-o",
                str(output_dir),
            ],
        )
        first_run_dir = generate_output_path(
            seed_graph, "seed", output_dir=output_dir, strategy="recommendation"
        ).parent
        optional_suffixes = (".html", ".plotly.html", ".csv", ".bib", ".graphml")
        for suffix in optional_suffixes:
            assert (first_run_dir / f"recommendation{suffix}").exists()
        sentinel_path = first_run_dir / "notes.txt"
        sentinel_path.write_text("keep user file", encoding="utf-8")
        citation_payload = _dashboard_graph_payload(seed_graph, "seed", "citation")
        citation_paths = {
            first_run_dir / "citation.json": json.dumps(citation_payload),
            first_run_dir / "citation.config.json": '{"strategy": "citation"}',
            first_run_dir / "citation.csv": "keep citation export",
        }
        for path, content in citation_paths.items():
            path.write_text(content, encoding="utf-8")
        update_dashboard_package(
            output_dir / DASHBOARD_PACKAGE_FILENAME,
            graph=seed_graph,
            seed_id="seed",
            strategy="citation",
            payload=citation_payload,
            build={"strategy": "citation"},
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
        assert len(package["results"]) == 2
        entry = package["results"][0]
        assert entry["result_id"] == "recommendation:seed"
        assert package["results"][1]["result_id"] == "citation:seed"
        assert entry["summary"] == {"nodes": 2, "edges": 1}
        assert entry["payload"]["summary"] == {"nodes": 2, "edges": 1}
        run_dir = generate_output_path(
            updated_graph, "seed", output_dir=output_dir, strategy="recommendation"
        ).parent
        assert run_dir == first_run_dir
        assert (
            json.loads((run_dir / "recommendation.json").read_text())
            == entry["payload"]
        )
        for suffix in optional_suffixes:
            assert not (run_dir / f"recommendation{suffix}").exists()
        assert sentinel_path.read_text(encoding="utf-8") == "keep user file"
        assert {
            path: path.read_text(encoding="utf-8") for path in citation_paths
        } == citation_paths
        collection_bundle = captured["metadata"]["dashboard_collection"]
        assert collection_bundle["current_result_id"] == "recommendation:seed"
        assert len(collection_bundle["results"]) == 2

    assert first_result.returncode == 0
    assert second_result.returncode == 0


def test_dashboard_build_serializes_all_result_artifacts_with_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Overlapping builds of one result must keep every artifact consistent.

    :param pytest.MonkeyPatch monkeypatch: Replaces building and delays one upsert.
    :param Path tmp_path: Temporary collection root.
    :return None: Assertions verify every artifact matches the winning entry.
    """
    first_graph = build_seed_graph("seed")
    second_graph = first_graph.copy()
    second_graph.add_node("extra", title="Second Run Paper")
    second_graph.add_edge("seed", "extra", weight=0.4)
    monkeypatch.setattr(
        build_module,
        "_build_strategy_graph",
        lambda args, strategy, **kwargs: (
            first_graph if args.max_papers == 1 else second_graph,
            "seed",
        ),
    )
    exporter_factory = _make_exporter_stub({}, methods=("to_dashboard_html", "to_csv"))

    def graph_sensitive_exporter(
        *factory_args: object, **kwargs: object
    ) -> SimpleNamespace:
        """Mark CSV output with the source graph's node count.

        :param object factory_args: Exporter constructor positional arguments.
        :param object kwargs: Exporter constructor keyword arguments.
        :return SimpleNamespace: Stub exporter with graph-sensitive CSV output.
        """
        graph = factory_args[0]
        assert isinstance(graph, nx.Graph)
        exporter = exporter_factory(*factory_args, **kwargs)

        def write_csv(path: Path) -> None:
            """Write the source graph's node count to the CSV fixture.

            :param Path path: CSV output path.
            :return None: Writes the fixture marker.
            """
            path.write_text(str(graph.number_of_nodes()), encoding="utf-8")

        exporter.to_csv = write_csv
        return exporter

    monkeypatch.setattr(build_module, "GraphExporter", graph_sensitive_exporter)

    first_update_started = threading.Event()
    allow_first_update = threading.Event()
    original_update = build_module.update_dashboard_package

    def delayed_update(package_path: Path, **kwargs: Any) -> dict[str, Any]:
        """Let the second build finish before the first acquires the package lock.

        :param Path package_path: Shared dashboard package path.
        :param Any kwargs: Result and staged artifacts passed by the CLI.
        :return dict[str, Any]: Updated collection package.
        """
        if kwargs["graph"] is first_graph:
            first_update_started.set()
            assert allow_first_update.wait(timeout=10), "first update never released"
        return original_update(package_path, **kwargs)

    monkeypatch.setattr(build_module, "update_dashboard_package", delayed_update)
    returncodes: list[int] = []
    errors: list[BaseException] = []

    def worker(max_papers: int) -> None:
        """Build the same result with distinguishable graph and build settings.

        :param int max_papers: Selects the first or second fixture graph.
        :return None: Records the CLI result or an unexpected worker error.
        """
        try:
            returncodes.append(
                cli_module.main(
                    [
                        "build",
                        "arxiv:1706.03762",
                        "--strategy",
                        "recommendation",
                        "--export",
                        "dashboard",
                        "--export",
                        "csv",
                        "--max-papers",
                        str(max_papers),
                        "--output",
                        str(tmp_path),
                    ]
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    first_thread = threading.Thread(target=worker, args=(1,))
    first_thread.start()
    try:
        assert first_update_started.wait(timeout=10), "first update never started"
        worker(2)
        assert not errors
        assert returncodes == [0]
    finally:
        allow_first_update.set()
        first_thread.join(timeout=10)

    assert not first_thread.is_alive()
    assert not errors
    assert returncodes == [0, 0]
    package = load_dashboard_package(tmp_path / DASHBOARD_PACKAGE_FILENAME)
    assert len(package["results"]) == 1
    entry = package["results"][0]
    assert entry["summary"] == {"nodes": 1, "edges": 0}
    assert entry["build"]["max_papers"] == 1
    run_dir = generate_output_path(
        first_graph, "seed", output_dir=tmp_path, strategy="recommendation"
    ).parent
    assert (
        json.loads((run_dir / "recommendation.json").read_text(encoding="utf-8"))
        == entry["payload"]
    )
    config = json.loads(
        (run_dir / "recommendation.config.json").read_text(encoding="utf-8")
    )
    assert config["build"] == entry["build"]
    assert (run_dir / "recommendation.csv").read_text(encoding="utf-8") == "1"


def test_dashboard_export_failure_preserves_previous_result_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed staged export must leave the completed result bundle unchanged.

    :param pytest.MonkeyPatch monkeypatch: Installs graph and exporter fixtures.
    :param Path tmp_path: Isolated dashboard collection root.
    :return None: Checks prior artifacts and package bytes after a failed refresh.
    """
    first_graph = build_seed_graph("seed")
    second_graph = first_graph.copy()
    second_graph.add_node("extra", title="Uncommitted Paper")
    second_graph.add_edge("seed", "extra", weight=0.4)
    build_results = iter([(first_graph, "seed"), (second_graph, "seed")])
    monkeypatch.setattr(
        build_module,
        "_build_strategy_graph",
        lambda args, strategy, **kwargs: next(build_results),
    )
    base_factory = _make_exporter_stub(
        {}, methods=("to_dashboard_html", "to_csv", "to_bibtex")
    )

    def failing_exporter(*args: object, **kwargs: object) -> SimpleNamespace:
        """Fail the second graph's BibTeX export after writing partial data.

        :param object args: Exporter constructor positional arguments.
        :param object kwargs: Exporter constructor keyword arguments.
        :return SimpleNamespace: Exporter fixture with graph-sensitive failure.
        """
        graph = args[0]
        exporter = base_factory(*args, **kwargs)

        def write_bibtex(path: Path) -> None:
            """Write staged data and fail only for the replacement graph.

            :param Path path: Staged BibTeX path.
            :return None: Writes a fixture or raises for the second graph.
            """
            path.write_text("partial replacement", encoding="utf-8")
            if graph is second_graph:
                raise RuntimeError("bibtex export failed")

        exporter.to_bibtex = write_bibtex
        return exporter

    monkeypatch.setattr(build_module, "GraphExporter", failing_exporter)
    first_result = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "recommendation",
            "--export",
            "dashboard",
            "--export",
            "csv",
            "--output",
            str(tmp_path),
        ]
    )
    run_dir = generate_output_path(
        first_graph, "seed", output_dir=tmp_path, strategy="recommendation"
    ).parent
    completed_paths = (
        tmp_path / DASHBOARD_PACKAGE_FILENAME,
        tmp_path / "dashboard.html",
        run_dir / "recommendation.json",
        run_dir / "recommendation.config.json",
        run_dir / "recommendation.csv",
    )
    completed_bytes = {path: path.read_bytes() for path in completed_paths}

    second_result = run_cli_command(
        [
            "build",
            "arxiv:1706.03762",
            "--strategy",
            "recommendation",
            "--export",
            "dashboard",
            "--export",
            "csv",
            "--export",
            "bibtex",
            "--output",
            str(tmp_path),
        ]
    )

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 1
    assert {path: path.read_bytes() for path in completed_paths} == completed_bytes
    assert not (run_dir / "recommendation.bib").exists()
    assert not any(".stage-" in path.name for path in tmp_path.rglob("*"))


@pytest.mark.parametrize("failure_stage", ["promotion", "package"])
def test_dashboard_commit_failure_restores_promoted_and_removed_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_stage: str
) -> None:
    """A commit failure must roll back result promotion and pruning.

    :param pytest.MonkeyPatch monkeypatch: Installs the selected filesystem failure.
    :param Path tmp_path: Isolated transaction directory.
    :param str failure_stage: Transaction stage that raises after one promotion.
    :return None: Checks exact restoration of old, absent, and obsolete targets.
    """
    package_path = tmp_path / DASHBOARD_PACKAGE_FILENAME
    existing_path = tmp_path / "recommendation.json"
    new_path = tmp_path / "recommendation.config.json"
    obsolete_path = tmp_path / "recommendation.csv"
    staged_existing = tmp_path / "staged.json"
    staged_new = tmp_path / "staged.config.json"
    package_path.write_bytes(b"old package")
    existing_path.write_bytes(b"old graph")
    existing_path.chmod(0o000)
    obsolete_path.write_bytes(b"old csv")
    staged_existing.write_bytes(b"new graph")
    staged_new.write_bytes(b"new config")

    def fail_package_write(path: Path, payload: object, **kwargs: object) -> None:
        """Fail the final package write after result-file promotion.

        :param Path path: Package destination.
        :param object payload: New package payload.
        :param object kwargs: JSON serialization options.
        :return None: Always raises before replacing the package.
        """
        del path, payload, kwargs
        raise OSError("package write failed")

    original_replace = Path.replace
    promoted_count = 0

    def fail_second_promotion(source: Path, target: Path) -> Path:
        """Fail after the first staged file has reached its final path.

        :param Path source: Source path for the replacement.
        :param Path target: Destination path for the replacement.
        :return Path: Result from replacements outside the injected failure.
        """
        nonlocal promoted_count
        if source in {staged_existing, staged_new}:
            promoted_count += 1
            if promoted_count == 2:
                raise OSError("promotion write failed")
        return original_replace(source, target)

    if failure_stage == "promotion":
        monkeypatch.setattr(Path, "replace", fail_second_promotion)
    else:
        monkeypatch.setattr(
            dashboard_package_module, "atomic_write_json", fail_package_write
        )

    with pytest.raises(OSError, match=f"{failure_stage} write failed"):
        dashboard_package_module._commit_staged_dashboard_artifacts(
            package_path,
            {"results": []},
            staged_result_files={
                existing_path: staged_existing,
                new_path: staged_new,
            },
            obsolete_result_paths={obsolete_path},
        )

    assert package_path.read_bytes() == b"old package"
    assert existing_path.stat().st_mode & 0o7777 == 0o000
    existing_path.chmod(0o600)
    assert existing_path.read_bytes() == b"old graph"
    assert not new_path.exists()
    assert obsolete_path.read_bytes() == b"old csv"
    assert not any(
        path.name.startswith(".citemesh-dashboard-backup-")
        for path in tmp_path.iterdir()
    )


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


@pytest.mark.parametrize("separate_cache_roots", [False, True])
def test_dashboard_package_serializes_concurrent_updates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, separate_cache_roots: bool
) -> None:
    """Serialize package updates independently of each writer's cache root.

    :param pytest.MonkeyPatch monkeypatch: Controls cache roots and write timing.
    :param Path tmp_path: Isolated collection and cache directories.
    :param bool separate_cache_roots: Whether writers use different cache roots.
    :return None: Checks exclusion and retention of both graph results.
    """
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path / "cache-a"))
    package_path = tmp_path / DASHBOARD_PACKAGE_FILENAME
    first_graph = build_seed_graph("seed-a")
    second_graph = build_seed_graph("seed-b")
    first_graph.nodes["seed-a"]["title"] = "First Seed"
    second_graph.nodes["seed-b"]["title"] = "Second Seed"

    delayed_atomic_write = PausedFirstWrite(dashboard_package_module.atomic_write_json)

    monkeypatch.setattr(
        dashboard_package_module, "atomic_write_json", delayed_atomic_write
    )

    errors: list[BaseException] = []

    def worker(graph: nx.Graph, seed_id: str) -> None:
        """Update one graph while recording thread failures for the test.

        :param nx.Graph graph: Graph to add to the collection.
        :param str seed_id: Seed identifying the graph result.
        :return None: Stores the result or records its exception.
        """
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
    assert delayed_atomic_write.first_started.wait(timeout=5), (
        "first write never started"
    )
    # The first writer already holds its lock, so each writer observes one root.
    if separate_cache_roots:
        monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path / "cache-b"))
    second_thread.start()
    second_wrote_early = delayed_atomic_write.second_started.wait(timeout=0.25)

    delayed_atomic_write.release_first.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not errors
    assert not second_wrote_early, (
        "second update reached write path before first released package lock"
    )

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
        build_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        build_module,
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


@pytest.mark.parametrize(
    "directory_name", ["session", "session.json", "session.dashboard.html"]
)
def test_dashboard_collection_resolver_always_includes_graph_json(
    tmp_path: Path,
    directory_name: str,
) -> None:
    """A collection directory should receive the viewer and per-seed JSON.

    :param Path tmp_path: Temporary parent for the collection directory.
    :param str directory_name: Directory name, including export-like suffixes.
    :return None: Assertions verify directory routing and implicit graph JSON.
    """
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

    root = tmp_path / "reports" / directory_name
    root.mkdir(parents=True)
    assert not _is_standalone_dashboard_output(root, ["dashboard"], True)
    assert not _is_standalone_dashboard_output(root, ["dashboard", "json"], True)
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
        build_module,
        "_build_strategy_graph",
        build_graph,
    )
    monkeypatch.setattr(
        build_module,
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


def test_dashboard_package_rejects_non_finite_payload_values(tmp_path: Path) -> None:
    """Persisted non-finite graph values must fail package preflight.

    :param Path tmp_path: Isolated dashboard package directory.
    :return None: Checks strict numeric validation during package loading.
    """
    package_path = tmp_path / DASHBOARD_PACKAGE_FILENAME
    graph = build_seed_graph("seed")
    graph.add_node(
        "other",
        title="Other",
        year=2021,
        authors=[],
        citation_count=0,
    )
    graph.add_edge("seed", "other", weight=0.5)
    package = update_dashboard_package(
        package_path,
        graph=graph,
        seed_id="seed",
        strategy="recommendation",
        payload=_dashboard_graph_payload(graph, "seed", "recommendation"),
        build={"strategy": "recommendation"},
    )
    package["results"][0]["payload"]["edges"][0]["weight"] = float("nan")
    package_path.write_text(json.dumps(package), encoding="utf-8")

    with pytest.raises(DashboardPackageError, match="non-finite numeric values"):
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


def test_dashboard_package_rejects_descriptor_payload_seed_mismatch(
    tmp_path: Path,
) -> None:
    """The package boundary should report mismatched seed identities cleanly.

    :param Path tmp_path: Isolated dashboard package directory.
    :return None: Checks the descriptor-to-payload identity validation.
    """
    graph = build_seed_graph("seed")
    payload = _dashboard_graph_payload(graph, "seed", "recommendation")

    with pytest.raises(DashboardPackageError, match="descriptor does not match"):
        update_dashboard_package(
            tmp_path / DASHBOARD_PACKAGE_FILENAME,
            graph=graph,
            seed_id="other-seed",
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
        build_module,
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

    monkeypatch.setattr(build_module, "GraphExporter", failing_exporter)
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
    """Collection-mode dashboard exports should debug-log planning details."""
    graph = build_seed_graph("seed")
    debug_mock = MagicMock()
    monkeypatch.setattr(cli_module.logger, "debug", debug_mock)
    monkeypatch.setattr(
        build_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        build_module,
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
    debug_messages = [
        str(call.args[0]) for call in debug_mock.call_args_list if call.args
    ]
    assert any("Dashboard collection mode:" in msg for msg in debug_messages)


def test_build_run_summary_reports_requested_outputs_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The closing INFO record should contain graph counts and requested paths."""
    info_mock = MagicMock()
    debug_mock = MagicMock()
    monkeypatch.setattr(build_module.logger, "info", info_mock)
    monkeypatch.setattr(build_module.logger, "debug", debug_mock)
    graph = build_seed_graph("seed")
    output_paths = {
        "dashboard": tmp_path / "dashboard.html",
        "json": tmp_path / "graph.json",
    }

    build_module._log_run_summary(
        graph,
        output_paths=output_paths,
        graph_config_path=tmp_path / "graph.config.json",
        dashboard_package_path=tmp_path / "dashboard.citemesh.json",
    )

    info_mock.assert_called_once_with(
        "Build complete: nodes=%d, edges=%d; outputs: %s",
        1,
        0,
        f"dashboard={output_paths['dashboard']}, json={output_paths['json']}",
    )
    debug_mock.assert_called_once_with(
        "Build auxiliary artifacts: %s",
        "config="
        f"{tmp_path / 'graph.config.json'}, dashboard_package="
        f"{tmp_path / 'dashboard.citemesh.json'}",
    )


def test_build_uses_compact_plot_metadata_and_summary_export_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should pass compact PNG metadata and emit one export summary line."""
    graph = build_seed_graph("seed")

    captured: dict[str, object] = {}
    logged: list[str] = []
    monkeypatch.setattr(
        build_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        build_module,
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

    monkeypatch.setattr(build_module, "visualize_graph", _fake_visualize)
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
    assert any(
        "Build complete: nodes=1, edges=0; outputs:" in message for message in logged
    )
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
            build_module,
            "_build_strategy_graph",
            lambda args, _strategy_name, **_kwargs: (graph, "seed"),
        )
        monkeypatch.setattr(
            build_module,
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
        "effective_model": DEFAULT_EMBEDDING_MODEL_NAME,
        "effective_model_revision": None,
        "model_fingerprint": None,
        "effective_truncate_dim": None,
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
    _, build_parser, _, _ = parser_module._create_parser()
    cli_args = build_parser.parse_args(
        [
            "seed",
            "--strategy",
            "recommendation",
            "--no-references",
            "--refresh-reference-cache",
        ]
    )

    payload = graph_config_module._build_graph_config_payload(
        cli_args=cli_args,
        seed_id="seed",
        metadata={"strategy": "recommendation"},
        selected_formats=["json"],
        output_paths={"json": Path("out/recommendation.json")},
        s2_retry_budget=API_CONFIG.anonymous_retry_budget_seconds,
    )

    citation_config = payload["build"]["citation"]
    assert citation_config == {
        "fetch_references": False,
        "refresh_reference_cache": True,
        "similarity_threshold": cli_args.similarity_threshold,
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
        build_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        build_module,
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
        build_contract_module,
        "_embedding_cache_directory_stats",
        lambda: (Path("/tmp/cache"), 0, 0),
    )
    monkeypatch.setattr(
        build_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        build_module,
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
    tmp_path: Path,
) -> None:
    """Dispatch and sidecars should retain the same graph-shaping settings.

    :param pytest.MonkeyPatch monkeypatch: Replaces builders with recording stubs.
    :param Path tmp_path: Isolated export directory.
    :return None: Checks builder arguments and strategy-specific sidecar values.
    """
    monkeypatch.delenv("S2_API_KEY", raising=False)
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
                "dataset_source": "research/arxiv-snapshot",
                "corpus_size": 1234,
                "all_corpus": False,
                "top_k": 4,
                "truncate_dim": 64,
                "min_semantic_similarity": 0.69,
                "streaming": True,
                "binary_rescore_multiplier": 9,
                "calibration_sample_size": 123,
                "encode_batch_size": 48,
                "cache_compression": "lzf",
            },
            {
                "max_papers": 11,
                "model_name": "m",
                "model_profile": "embeddinggemma",
                "model_revision": None,
                "dataset_source": "research/arxiv-snapshot",
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
                "cache_compression": "lzf",
                "cache_compression_level": 0,
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
                "dataset_source": "research/arxiv-snapshot",
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
                "dataset_source": "research/arxiv-snapshot",
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
        output = tmp_path / f"{strategy}.json"
        argv = [
            "build",
            "seed",
            "--strategy",
            strategy,
            "--max-papers",
            "11",
            "--export",
            "json",
            "--output",
            str(output),
        ]
        for dest, value in namespace_overrides.items():
            flag = (
                "--batch-size"
                if dest == "encode_batch_size"
                else "--" + dest.replace("_", "-")
            )
            if isinstance(value, bool):
                if value:
                    argv.append(flag)
            else:
                argv.extend([flag, str(value)])
        monkeypatch.setattr(
            build_options_module,
            builder_name,
            _make_builder_stub(captured, graph=build_seed_graph("seed")),
        )
        result = run_cli_command(argv)
        assert result.returncode == 0, result.stderr
        assert captured == expected_kwargs
        build_config = json.loads(output.with_suffix(".config.json").read_text())[
            "build"
        ]
        if "similarity_threshold" in captured:
            assert (
                build_config["citation"]["similarity_threshold"]
                == captured["similarity_threshold"]
            )
        else:
            assert "similarity_threshold" not in build_config.get("citation", {})
        if "top_k" in captured:
            assert build_config["embedding"]["top_k"] == captured["top_k"]
        else:
            assert "top_k" not in build_config.get("embedding", {})
        client_factory = MagicMock()
        client_factory.return_value.retry_budget_seconds = 0.0
        monkeypatch.setattr(
            build_options_module, "SemanticScholarClient", client_factory
        )
        monkeypatch.setenv("S2_API_KEY", "configured-key")
        result = run_cli_command([*argv, "--refresh-paper-cache"])
        assert result.returncode == 0, result.stderr
        client_factory.assert_called_once_with(
            api_key="configured-key", refresh_paper_cache=True
        )
        assert captured == {**expected_kwargs, "client": client_factory.return_value}
        monkeypatch.delenv("S2_API_KEY")


@pytest.mark.parametrize(
    ("token", "expected"),
    [("0", 0.0), ("37.5", 37.5)],
)
def test_build_s2_retry_budget_routes_explicit_override(
    monkeypatch: pytest.MonkeyPatch, token: str, expected: float
) -> None:
    """An explicit S2 recovery budget should construct a configured client.

    :param pytest.MonkeyPatch monkeypatch: Replaces the S2 client constructor.
    :param str token: CLI retry-budget token.
    :param float expected: Parsed retry-budget value.
    :return None: Assertions verify client routing and default behavior.
    """
    monkeypatch.delenv("S2_API_KEY", raising=False)
    _, build_parser, _, _ = parser_module._create_parser()
    default_args = build_parser.parse_args(["seed"])
    factory = MagicMock()
    monkeypatch.setattr(build_options_module, "SemanticScholarClient", factory)

    assert build_options_module._configured_client_kwargs(default_args) == {}
    factory.assert_not_called()

    args = build_parser.parse_args(["seed", "--s2-retry-budget", token])
    configured = build_options_module._configured_client_kwargs(args)

    factory.assert_called_once_with(
        api_key=None,
        refresh_paper_cache=False,
        retry_budget_seconds=expected,
    )
    assert configured == {"client": factory.return_value}
    assert args._s2_client is factory.return_value


@pytest.mark.parametrize("token", ["-1", "nan", "inf", "-inf"])
def test_build_rejects_invalid_s2_retry_budget(token: str) -> None:
    """The S2 recovery budget must be a finite non-negative float.

    :param str token: Invalid CLI retry-budget token.
    :return None: Assertions verify clean usage failure.
    """
    result = run_cli_command(["build", "seed", "--s2-retry-budget", token])

    assert result.returncode == 2
    assert "--s2-retry-budget" in result.stderr


@pytest.mark.parametrize(
    ("token", "api_key", "selected_client_budget"),
    [
        (None, None, API_CONFIG.anonymous_retry_budget_seconds),
        (None, "configured-key", 0.0),
        ("0", None, 0.0),
        ("37.5", "configured-key", 12.5),
    ],
)
def test_graph_config_payload_records_effective_s2_retry_budget(
    token: str | None, api_key: str | None, selected_client_budget: float
) -> None:
    """Graph sidecars should retain the selected client's S2 recovery budget.

    :param str | None token: Optional explicit CLI budget token.
    :param str | None api_key: Resolved key presence used for the default policy.
    :param float selected_client_budget: Effective budget from the selected client.
    :return None: Checks the sidecar does not recalculate client policy.
    """
    _, build_parser, _, _ = parser_module._create_parser()
    argv = ["seed"]
    if token is not None:
        argv.extend(["--s2-retry-budget", token])
    cli_args = build_parser.parse_args(argv)
    cli_args._s2_api_key = api_key

    payload = graph_config_module._build_graph_config_payload(
        cli_args=cli_args,
        seed_id="seed",
        metadata={"strategy": "recommendation"},
        selected_formats=["json"],
        output_paths={"json": Path("out/recommendation.json")},
        s2_retry_budget=selected_client_budget,
    )

    assert payload["build"]["s2_retry_budget"] == selected_client_budget


def test_in_process_cli_sidecar_matches_reconfigured_shared_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A prior keyed singleton must not change a later anonymous build policy.

    :param pytest.MonkeyPatch monkeypatch: Replaces the builder and credential state.
    :param Path tmp_path: Isolated output directory for the config sidecar.
    :return None: Checks the builder client and sidecar report the same budget.
    """
    keyed_client: SemanticScholarClient | None = None
    captured: dict[str, object] = {}
    try:
        s2.reset_client()
        monkeypatch.setenv("S2_API_KEY", "configured-key")
        keyed_client = s2.get_client()
        assert keyed_client.retry_budget_seconds == 0.0
        monkeypatch.delenv("S2_API_KEY")

        def builder_factory(**kwargs: object) -> SimpleNamespace:
            """Capture the client selected by normal recommendation dispatch.

            :param object kwargs: Recommendation builder keyword arguments.
            :return SimpleNamespace: Builder stub returning a seed-only graph.
            """
            client = kwargs.get("client") or s2.get_client()
            captured["client"] = client
            return SimpleNamespace(
                build_graph=lambda _paper_id: (build_seed_graph("seed"), "seed")
            )

        monkeypatch.setattr(
            build_options_module, "RecommendationGraphBuilder", builder_factory
        )
        monkeypatch.setattr(
            build_module,
            "GraphExporter",
            _make_exporter_stub({}, methods=("to_json",)),
        )
        output = tmp_path / "graph.json"

        result = run_cli_command(
            ["build", "seed", "--export", "json", "--output", str(output)]
        )

        assert result.returncode == 0, result.stderr
        selected_client = captured["client"]
        assert isinstance(selected_client, SemanticScholarClient)
        payload = json.loads(output.with_suffix(".config.json").read_text())
        assert (
            selected_client.retry_budget_seconds == payload["build"]["s2_retry_budget"]
        )
        assert payload["build"]["s2_retry_budget"] == (
            API_CONFIG.anonymous_retry_budget_seconds
        )
    finally:
        s2.reset_client()
        if keyed_client is not None:
            keyed_client.close()


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
        (["--version"], [f"citemesh {cli_module.__version__}"]),
        (
            ["--help"],
            [
                "CiteMesh",
                "build",
                "cache",
                "search",
                "view",
                "S2_API_KEY",
                "CITEMESH_CACHE_DIR",
                "--version",
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
                "--s2-retry-budget",
                "--storage-precision",
                "--binary-prefilter",
                "--binary-rescore-multiplier",
                "--calibration-sample-size",
                "--batch-size",
                "-bs",
                "--no-torch-compile",
                "--spring-iterations",
                "citation/recommendation",
                "repeat for multiple",
                "dashboard.html",
                DASHBOARD_PACKAGE_FILENAME,
                ".dashboard.html",
            ],
        ),
        (
            ["view", "--help"],
            ["view [PATH]", "out/dashboard.html", "--browser NAME", "system browser"],
        ),
        (["search", "--help"], ["search", "--limit"]),
        (["cache", "--help"], ["clear", "scan", "Examples"]),
        (["cache", "scan", "--help"], ["cache scan", "--log-level"]),
        (
            ["cache", "clear", "--help"],
            ["cache clear", "--yes", "config.toml", "cached papers"],
        ),
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
        lowered = flatten_console_text(result.stdout).lower()
        for token in expected_tokens:
            assert token.lower() in lowered


def test_cli_help_survives_unusable_terminal_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Help must remain readable when the terminal reports one column.

    :param pytest.MonkeyPatch monkeypatch: Terminal-size override fixture.
    :return None: Checks nonempty root help at the minimum formatter width.
    """
    with monkeypatch.context() as terminal_patch:
        terminal_patch.setattr(
            parser_module.shutil,
            "get_terminal_size",
            lambda: SimpleNamespace(columns=1),
        )
        help_text = parser_module._create_parser()[0].format_help()

    assert "usage: citemesh COMMAND [options]" in help_text
    assert "CiteMesh" in help_text
    assert "Options:" in help_text


def test_output_path_and_slug_contracts(tmp_path: Path) -> None:
    """Output path resolver and auto-output slug generation should stay stable.

    :param Path tmp_path: Isolated directory for output-name collision checks.
    :return None: Verifies artifact paths, sidecar paths, and stable seed slugs.
    """
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
    output_dir = tmp_path / "out"
    path_a = generate_output_path(graph, seed_id="seed-a", output_dir=output_dir)
    path_b = generate_output_path(graph, seed_id="seed-b", output_dir=output_dir)
    graph.nodes["seed-a"]["title"] = "A Corrected Transformer Survey Title"
    corrected_path_a = generate_output_path(
        graph, seed_id="seed-a", output_dir=output_dir
    )
    assert path_a != path_b
    assert path_a.parent.name != path_b.parent.name
    assert corrected_path_a == path_a
    assert path_a.parent.name.startswith("a-survey-of-transformers-")
    assert path_b.parent.name.startswith("a-survey-of-transformers-")
    assert re.search(r"-[0-9a-f]{8}$", path_a.parent.name)
    assert re.search(r"-[0-9a-f]{8}$", path_b.parent.name)

    graph = nx.Graph()
    graph.add_node(
        "seed",
        title="This title should definitely exceed forty characters for the slug",
    )
    output_path = generate_output_path(graph, seed_id="seed", output_dir=output_dir)
    assert output_path.parent.name.startswith("this-title-should-")
    assert re.search(r"-[0-9a-f]{8}$", output_path.parent.name)
    assert len(output_path.parent.name) <= 40


@pytest.mark.parametrize(
    "directory_name", ["results", "results.json", "results.dashboard.html"]
)
@pytest.mark.parametrize("formats", [["json"], ["png"], ["json", "png"]])
def test_output_writes_into_an_existing_directory(
    tmp_path: Path,
    directory_name: str,
    formats: list[str],
) -> None:
    """An existing --output directory must receive the artifact, not name it.

    :param Path tmp_path: Temporary directory serving as the output target.
    :param str directory_name: Directory name, including export-like suffixes.
    :param list[str] formats: Single or multiple requested export formats.
    :return None: Assertions pin file-or-directory semantics.
    """
    results_dir = tmp_path / directory_name
    results_dir.mkdir()

    paths = resolve_output_paths(
        base_output_path=results_dir,
        selected_formats=formats,
        explicit_output=True,
        strategy="citation",
    )

    assert paths == {fmt: results_dir / f"citation.{fmt}" for fmt in formats}


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


def test_multi_export_flag_selects_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeating --export should produce exactly the requested formats."""
    graph = build_seed_graph("seed")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        build_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(
        build_module,
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
        build_module,
        "_build_strategy_graph",
        lambda args, strategy, **_kwargs: (graph, "seed"),
    )
    monkeypatch.setattr(build_module, "GraphExporter", _CountingExporter)

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
    covered = set(outputs_module._EXPORTER_METHOD) | {"png"}
    assert covered == set(cli_module.EXPORT_FORMATS), (
        f"Dispatch gap: covered={sorted(covered)}, "
        f"declared={sorted(cli_module.EXPORT_FORMATS)}"
    )


def test_builder_defaults_match_cli_defaults() -> None:
    """Strategy builder constructor defaults should match CLI parser defaults."""
    from citemesh.strategies.citation import CitationGraphBuilder
    from citemesh.strategies.recommendation import RecommendationGraphBuilder

    _, build_parser, _, _ = parser_module._create_parser()
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

    monkeypatch.setattr(
        build_contract_module, "resolve_embedding_device", _raise_unavailable
    )
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
    _, build_parser, _, _ = parser_module._create_parser()
    cli_args = build_parser.parse_args(
        ["seed", "--strategy", "embedding", "--device", "cpu"]
    )

    payload = graph_config_module._build_graph_config_payload(
        cli_args=cli_args,
        seed_id="seed",
        metadata={"strategy": "embedding"},
        selected_formats=["json"],
        output_paths={"json": Path("out/embedding.json")},
        s2_retry_budget=API_CONFIG.anonymous_retry_budget_seconds,
    )

    assert payload["build"]["embedding"]["device"] == "cpu"


def test_export_metadata_records_effective_embedding_runtime() -> None:
    """Export metadata should surface the active model, dimension, and device."""
    _, build_parser, _, _ = parser_module._create_parser()
    namespace = build_parser.parse_args(["seed", "--strategy", "embedding"])
    active_model = "google/embeddinggemma-300m"
    resolved_revision = "0123456789abcdef0123456789abcdef01234567"
    fingerprint = f"hf::{active_model}::{resolved_revision}"
    metadata = build_options_module._embedding_export_metadata(
        namespace,
        runtime_metadata={
            "binary_prefilter_used": True,
            "active_model": active_model,
            "model_fingerprint": fingerprint,
            "resolved_model_revision": resolved_revision,
            "truncate_dim": 512,
            "device": "mps",
            "compute_dtype": "bfloat16",
        },
    )
    assert metadata["effective_model"] == active_model
    assert metadata["effective_model_revision"] == resolved_revision
    assert metadata["model_fingerprint"] == fingerprint
    assert metadata["effective_truncate_dim"] == 512
    assert metadata["effective_device"] == "mps"
    assert metadata["effective_compute_dtype"] == "bfloat16"

    sidecar = graph_config_module._build_graph_config_payload(
        cli_args=namespace,
        seed_id="seed",
        metadata={"strategy": "embedding", "embedding": metadata},
        selected_formats=["json"],
        output_paths={"json": Path("out/embedding.json")},
        s2_retry_budget=API_CONFIG.anonymous_retry_budget_seconds,
    )
    replay = sidecar["build"]["embedding"]
    assert replay["model"] == active_model
    assert "model_revision" not in replay
    assert replay["truncate_dim"] == 512

    original_builder = EmbeddingGraphBuilder(
        model_name=namespace.model,
        model_revision=namespace.model_revision,
        client=MagicMock(),
    )
    original_builder._active_model_name = active_model
    original_builder._resolved_model_fingerprint = fingerprint
    replay_builder = EmbeddingGraphBuilder(
        model_name=replay["model"],
        model_revision=replay.get("model_revision"),
        truncate_dim=replay["truncate_dim"],
        client=MagicMock(),
    )
    replay_builder._resolved_model_fingerprint = fingerprint
    assert original_builder._embedding_cache_namespace(
        artifact_identity=fingerprint
    ) == replay_builder._embedding_cache_namespace(artifact_identity=fingerprint)

    selector_namespace = build_parser.parse_args(
        ["seed", "--strategy", "embedding", "--model-revision", "main"]
    )
    selector_fingerprint = f"hf::{selector_namespace.model}::{resolved_revision}"
    selector_metadata = build_options_module._embedding_export_metadata(
        selector_namespace,
        runtime_metadata={
            "active_model": selector_namespace.model,
            "model_fingerprint": selector_fingerprint,
            "resolved_model_revision": resolved_revision,
        },
    )
    selector_sidecar = graph_config_module._build_graph_config_payload(
        cli_args=selector_namespace,
        seed_id="seed",
        metadata={"strategy": "embedding", "embedding": selector_metadata},
        selected_formats=["json"],
        output_paths={"json": Path("out/embedding.json")},
        s2_retry_budget=API_CONFIG.anonymous_retry_budget_seconds,
    )
    assert selector_metadata["effective_model_revision"] == resolved_revision
    assert selector_sidecar["build"]["embedding"]["model_revision"] == "main"


def test_export_metadata_omits_candidate_pool_size_in_corpus_mode() -> None:
    """Corpus builds must not record a candidate pool size they never consult.

    :return None: Assertions align export metadata with the config sidecar.
    """
    corpus_metadata = build_options_module._embedding_export_metadata(
        _dispatch_namespace(semantic_source="arxiv-corpus")
    )
    candidates_metadata = build_options_module._embedding_export_metadata(
        _dispatch_namespace(semantic_source="candidates", candidate_pool_size=400)
    )

    assert "candidate_pool_size" not in corpus_metadata
    assert candidates_metadata["candidate_pool_size"] == 400


def test_build_corpus_flags_imply_arxiv_corpus_source() -> None:
    """Corpus-only flags without --semantic-source should imply arxiv-corpus."""
    _, build_parser, _, _ = parser_module._create_parser()
    args = build_parser.parse_args(
        ["seed", "--strategy", "embedding", "--corpus-size", "1234"]
    )
    provided = parser_module._pop_tracked_option_dests(args)
    build_contract_module._validate_build_cli_contract(args, build_parser, provided)
    assert args.semantic_source == "arxiv-corpus"

    args = build_parser.parse_args(["seed", "--strategy", "embedding"])
    provided = parser_module._pop_tracked_option_dests(args)
    build_contract_module._validate_build_cli_contract(args, build_parser, provided)
    assert args.semantic_source == "candidates"
    assert args.storage_precision == "float32"
    assert args.corpus_size is None
    assert args.candidate_pool_size == 400


def test_dataset_source_implies_corpus_mode_and_reaches_sidecar() -> None:
    """A selected metadata source should activate and describe corpus hydration.

    :return None: Assertions validate corpus routing and sidecar provenance.
    """
    _, build_parser, _, _ = parser_module._create_parser()
    dataset_source = "research/arxiv-snapshot"
    args = build_parser.parse_args(
        ["seed", "--strategy", "embedding", "--dataset-source", dataset_source]
    )
    provided = parser_module._pop_tracked_option_dests(args)
    build_contract_module._validate_build_cli_contract(args, build_parser, provided)

    assert args.semantic_source == "arxiv-corpus"
    payload = graph_config_module._build_graph_config_payload(
        cli_args=args,
        seed_id="seed",
        metadata={"strategy": "embedding"},
        selected_formats=["json"],
        output_paths={"json": Path("out/embedding.json")},
        s2_retry_budget=API_CONFIG.anonymous_retry_budget_seconds,
    )
    assert payload["build"]["embedding"]["dataset_source"] == dataset_source

    candidate_args = build_parser.parse_args(["seed", "--strategy", "embedding"])
    candidate_provided = parser_module._pop_tracked_option_dests(candidate_args)
    build_contract_module._validate_build_cli_contract(
        candidate_args, build_parser, candidate_provided
    )
    candidate_payload = graph_config_module._build_graph_config_payload(
        cli_args=candidate_args,
        seed_id="seed",
        metadata={"strategy": "embedding"},
        selected_formats=["json"],
        output_paths={"json": Path("out/embedding.json")},
        s2_retry_budget=API_CONFIG.anonymous_retry_budget_seconds,
    )
    assert "dataset_source" not in candidate_payload["build"]["embedding"]
