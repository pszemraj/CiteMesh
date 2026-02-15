"""Pytest configuration shared across test modules."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_citemesh_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate CITEMESH cache paths per-test to avoid mutating user cache state."""
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path / "citemesh-cache"))
