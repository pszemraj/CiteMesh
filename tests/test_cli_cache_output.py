"""Rendered cache reports retain identifiers and useful status at terminal widths."""

from __future__ import annotations

import io
import shlex
from pathlib import Path

import pytest
from rich.console import Console

from citemesh.cli import cache_ops, console
from citemesh.cli.commands import cache


@pytest.mark.parametrize("width", [60, 80, 120])
def test_cache_namespace_report_preserves_source_and_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    """Long provenance strings remain readable rather than ellipsized in a cell."""
    output = io.StringIO()
    monkeypatch.setattr(
        console, "output_console", Console(file=output, width=width, color_system=None)
    )
    monkeypatch.setattr(cache_ops, "get_cache_dir", lambda **kwargs: tmp_path)
    revision = "abcdef0123456789abcdef0123456789abcdef0123"
    origin = "user/arxiv-embeddinggemma[archive]"
    monkeypatch.setattr(
        cache_ops,
        "_scan_embedding_namespaces",
        lambda _path: [
            cache_ops._EmbeddingNamespaceScan(
                namespace_id="0123456789ab",
                file_count=4,
                size_bytes=12 * 1024**3,
                status=(
                    "768d int8; active 3,156,800 / physical 3,156,800; "
                    f"source {revision}; coverage full; origin {origin}; ready"
                ),
            )
        ],
    )

    assert cache_ops._scan_cache_directory() == 0

    rendered = output.getvalue()
    assert "0123456789ab" in rendered
    assert "4 files" in rendered
    assert "12.0 GiB" in rendered
    assert "3,156,800" in rendered
    assert revision in rendered
    assert origin in rendered
    assert "…" not in rendered
    assert all(len(line) <= width for line in rendered.splitlines())


def test_installed_corpus_hint_keeps_command_copyable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command occupies its own unmodified line even in a narrow terminal."""
    output = io.StringIO()
    monkeypatch.setattr(
        console, "output_console", Console(file=output, width=60, color_system=None)
    )

    cache._print_installed_namespace("0123456789ab")

    lines = output.getvalue().splitlines()
    command = next(line for line in lines if line.startswith("citemesh search "))
    assert shlex.split(command) == [
        "citemesh",
        "search",
        "long-context language models",
        "--mode",
        "local",
        "--cache-namespace",
        "0123456789ab",
    ]
    assert lines[lines.index(command) - 1].strip() == "Search it with:"
