"""Tests for runtime TTY helper semantics."""

from __future__ import annotations

import io

import pytest

from citemesh import _runtime as runtime_module


class _StreamWithoutIsatty:
    """Sentinel stream object that omits an ``isatty`` method."""


def test_stdin_isatty_respects_replaced_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stdin TTY detection should respect replaced ``sys.stdin`` semantics."""
    monkeypatch.setattr(runtime_module, "_fd_isatty", lambda _fd: True)
    monkeypatch.setattr(runtime_module.sys, "stdin", io.StringIO("scripted input"))

    assert runtime_module.stdin_isatty() is False


def test_stderr_isatty_falls_back_to_fd_when_stream_lacks_isatty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stderr TTY detection should fall back to fd checks for shim streams."""
    monkeypatch.setattr(runtime_module, "_fd_isatty", lambda fd: fd == 2)
    monkeypatch.setattr(runtime_module.sys, "stderr", _StreamWithoutIsatty())

    assert runtime_module.stderr_isatty() is True
