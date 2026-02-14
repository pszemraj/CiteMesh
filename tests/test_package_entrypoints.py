"""Tests for package-level exports and module entrypoints."""

from __future__ import annotations

import runpy

import pytest


def test_main_module_invokes_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running ``citemesh.__main__`` should invoke ``citemesh.cli.main``."""
    called = {"main": False}

    def fake_main() -> None:
        called["main"] = True

    monkeypatch.setattr("citemesh.cli.main", fake_main)
    runpy.run_module("citemesh.__main__", run_name="__main__")
    assert called["main"] is True


def test_lazy_strategy_exports_resolve() -> None:
    """Top-level strategy exports should resolve via lazy loading."""
    import citemesh

    assert citemesh.CitationGraphBuilder.__name__ == "CitationGraphBuilder"
    assert citemesh.RecommendationGraphBuilder.__name__ == "RecommendationGraphBuilder"
    assert citemesh.EmbeddingGraphBuilder.__name__ == "EmbeddingGraphBuilder"
    assert citemesh.HybridGraphBuilder.__name__ == "HybridGraphBuilder"
