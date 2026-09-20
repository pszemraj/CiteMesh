"""Regression tests for model-download isolation across test lifetimes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("preimport_transformers", [False, True])
def test_model_download_guard_blocks_and_restores_imported_aliases(
    preimport_transformers: bool,
) -> None:
    """The guard must block both aliases and restore them regardless of import order.

    :param bool preimport_transformers: Whether Transformers predates fixture setup.
    :return None: Checks download blocking and restoration in a fresh interpreter.
    """
    script = """
import importlib
import sys
from types import SimpleNamespace

import huggingface_hub
import pytest

from tests.conftest import _forbid_external_model_resolution

assert "transformers.utils.hub" not in sys.modules
original = huggingface_hub.hf_hub_download
if sys.argv[1] == "True":
    importlib.import_module("transformers.utils.hub")
request = SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda name: None))
with pytest.MonkeyPatch.context() as patches:
    _forbid_external_model_resolution.__wrapped__(request, patches)
    transformers_hub = importlib.import_module("transformers.utils.hub")
    for download in (
        huggingface_hub.hf_hub_download,
        transformers_hub.hf_hub_download,
    ):
        # No arguments also prevent an unpatched function from accessing the network.
        with pytest.raises(AssertionError, match="external model resolution"):
            download()
assert huggingface_hub.hf_hub_download is original
assert transformers_hub.hf_hub_download is original
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(preimport_transformers)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_arxiv_request_guard_blocks_non_slow_tests() -> None:
    """The shared test guard rejects accidental direct arXiv transport use.

    :return None: Verifies unit tests cannot silently reach arXiv.org.
    """
    from citemesh.services import arxiv as arxiv_module

    with pytest.raises(AssertionError, match="attempted an arXiv request"):
        arxiv_module.requests.get("https://arxiv.org/html/2608.27147")
