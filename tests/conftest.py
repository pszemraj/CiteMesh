"""Pytest configuration shared across test modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import huggingface_hub
import pytest
import requests


@pytest.fixture(autouse=True)
def _isolate_citemesh_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate CITEMESH cache paths per-test to avoid mutating user cache state."""
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path / "citemesh-cache"))


@pytest.fixture(autouse=True)
def _forbid_external_model_resolution(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail non-slow tests that cross a Hugging Face download boundary."""
    if request.node.get_closest_marker("slow") is not None:
        return

    # Import before patching the provider so teardown restores the original alias.
    import transformers.utils.hub as transformers_hub

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        """Reject external model resolution from a nominal unit test."""
        raise AssertionError(
            "A non-slow test attempted external model resolution. Inject a local "
            "artifact or mock the model/fingerprint boundary."
        )

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", blocked)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", blocked)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", blocked)

    monkeypatch.setattr(transformers_hub, "hf_hub_download", blocked)


@pytest.fixture(autouse=True)
def _forbid_external_http(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail non-slow tests that reach an unmocked HTTP transport."""
    if request.node.get_closest_marker("slow") is not None:
        return

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        """Reject network access below provider-specific client seams."""
        raise AssertionError(
            "A non-slow test attempted an external HTTP request. Mock the client "
            "or transport boundary."
        )

    async def blocked_async(*_args: Any, **_kwargs: Any) -> Any:
        """Reject asynchronous network access below SDK client seams."""
        raise AssertionError(
            "A non-slow test attempted an external HTTP request. Mock the client "
            "or transport boundary."
        )

    monkeypatch.setattr(requests.sessions.Session, "send", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked_async)


@pytest.fixture(autouse=True)
def _forbid_arxiv_requests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail non-slow tests that cross the optional arXiv transport boundary."""
    if request.node.get_closest_marker("slow") is not None:
        return

    from citemesh.services import arxiv as arxiv_module

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        """Reject an arXiv HTML or Atom request from a nominal unit test."""
        raise AssertionError(
            "A non-slow test attempted an arXiv request. Mock the arXiv client or "
            "its transport boundary."
        )

    monkeypatch.setattr(arxiv_module.requests, "get", blocked)


@pytest.fixture
def arxiv_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep existing provider-focused tests independent of live arXiv HTML.

    :param pytest.MonkeyPatch monkeypatch: Test-scoped transport replacement.
    :return None: Makes the optional HTML source unavailable.
    """
    from citemesh.services.arxiv import ArxivClient

    monkeypatch.setattr(ArxivClient, "get_bibliography", lambda self, identifier: None)
