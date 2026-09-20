"""Pytest configuration shared across test modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import huggingface_hub
import pytest


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


@pytest.fixture
def arxiv_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep existing provider-focused tests independent of live arXiv HTML.

    :param pytest.MonkeyPatch monkeypatch: Test-scoped transport replacement.
    :return None: Makes the optional HTML source unavailable.
    """
    from citemesh.services.arxiv import ArxivClient

    monkeypatch.setattr(ArxivClient, "get_bibliography", lambda self, identifier: None)
