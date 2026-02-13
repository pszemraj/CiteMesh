"""Tests for ``python -m citemesh`` entrypoint wrapper."""

from __future__ import annotations

import runpy

import pytest


def test_main_module_invokes_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running ``citemesh.__main__`` as a script should call ``citemesh.cli.main``."""
    called = {"main": False}

    def fake_main() -> None:
        """Track invocation from module wrapper."""
        called["main"] = True

    monkeypatch.setattr("citemesh.cli.main", fake_main)
    runpy.run_module("citemesh.__main__", run_name="__main__")
    assert called["main"] is True
