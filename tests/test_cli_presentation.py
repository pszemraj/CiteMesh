"""Console help stays readable and exposes the inputs needed for each command."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from citemesh.cli.parser import (
    _create_parser,
    corpus_build_example,
    corpus_search_example,
)
from citemesh.data.user_config import known_config_keys
from citemesh.strategies.embedding.config import PUBLIC_ARXIV_INDEX_IMPORT


@pytest.mark.parametrize("width", [60, 80, 120])
def test_cache_adoption_help_labels_every_selector(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], width: int
) -> None:
    monkeypatch.setenv("COLUMNS", str(width))
    parser, *_ = _create_parser()
    with pytest.raises(SystemExit, match="0"):
        parser.parse_args(["cache", "refresh", "--help"])
    output = capsys.readouterr().out
    assert max(map(len, output.splitlines())) <= width
    flattened = " ".join(output.split())
    assert "--namespace NAMESPACE [options]" in flattened
    for description in (
        "Original model",
        "Original int8 calibration sample count",
        "Original cap on corpus papers",
        "Original corpus used the full selected split",
    ):
        assert description in flattened


@pytest.mark.parametrize("width", [60, 80])
@pytest.mark.parametrize(
    ("command", "examples"),
    [
        (
            ["build"],
            [
                'citemesh build "arxiv:2404.08801" --strategy hybrid --export dashboard',
                'citemesh build "arxiv:1706.03762" --strategy recommendation --export dashboard',
                corpus_build_example(),
            ],
        ),
        (
            ["cache", "import"],
            [
                "citemesh cache import --path out/corpus",
                " ".join(PUBLIC_ARXIV_INDEX_IMPORT),
            ],
        ),
        (
            ["search"],
            [
                'citemesh search "long-context language models" -n 10',
                'citemesh search "long-context language models" --mode local',
                corpus_search_example(),
                'citemesh search "Megalodon" --mode s2',
            ],
        ),
    ],
    ids=["build", "cache-import", "search"],
)
def test_help_examples_stay_pasteable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: list[str],
    examples: list[str],
    width: int,
) -> None:
    """A long example breaks only at option boundaries, each cut ending in a backslash."""
    monkeypatch.setenv("COLUMNS", str(width))
    parser, *_ = _create_parser()
    with pytest.raises(SystemExit, match="0"):
        parser.parse_args([*command, "--help"])
    lines = capsys.readouterr().out.splitlines()
    assert max(map(len, lines)) <= width
    for line, following in zip(lines, lines[1:], strict=False):
        if line.endswith(" \\"):
            assert following.startswith("      -")
    joined = "\n".join(lines).replace(" \\\n      ", " ")
    pasteable = {line.strip() for line in joined.splitlines()}
    assert set(examples) <= pasteable


@pytest.mark.parametrize("command", ["get", "set", "unset"])
def test_config_key_help_is_complete(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    parser, *_ = _create_parser()
    with pytest.raises(SystemExit, match="0"):
        parser.parse_args(["config", command, "--help"])
    output = capsys.readouterr().out
    assert "Config keys:" in output
    assert all(key in output for key in known_config_keys())


@pytest.mark.parametrize("width", [60, 80, 120])
def test_publisher_help_uses_compact_cli_style(width: int) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "publish_arxiv_index.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": str(width), "NO_COLOR": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "BUCKET OUTPUT [options]" in result.stdout
    assert "(default: 512)" in result.stdout
    assert "default: None" not in result.stdout
    assert "--dataset-revision REV" in result.stdout
    assert max(map(len, result.stdout.splitlines())) <= width


def test_usage_error_wraps_to_the_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("COLUMNS", "60")
    parser, *_ = _create_parser()
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["build", "focal", "--strategy", "unknown"])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid choice" in " ".join(captured.err.split())
    assert max(map(len, captured.err.splitlines())) <= 60
