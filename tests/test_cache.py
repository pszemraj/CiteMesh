"""Direct contracts for cross-platform cache path and atomic-write helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import citemesh.data.cache as cache_module


def test_default_cache_root_honors_override_before_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit cache override should win on every operating system.

    :param Path tmp_path: Temporary cache root.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate environment state.
    :return None: Validates override precedence.
    """
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_module.platform, "system", lambda: "Windows")

    assert cache_module._default_cache_root() == tmp_path
    monkeypatch.setenv("CITEMESH_CACHE_DIR", "~/paper-cache")
    assert cache_module._default_cache_root() == Path.home() / "paper-cache"


def test_default_cache_root_uses_shared_xdg_layout_on_macos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS should use the documented cross-platform XDG-style cache root.

    :param Path tmp_path: Temporary XDG cache base.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate environment state.
    :return None: Validates the macOS cache path contract.
    """
    monkeypatch.delenv("CITEMESH_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(cache_module.platform, "system", lambda: "Darwin")

    assert cache_module._default_cache_root() == tmp_path / "citemesh"


def test_default_cache_root_uses_local_app_data_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows should prefer LOCALAPPDATA for the CiteMesh cache root.

    :param Path tmp_path: Temporary LOCALAPPDATA base.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate environment state.
    :return None: Validates the Windows cache path contract.
    """
    monkeypatch.delenv("CITEMESH_CACHE_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(cache_module.platform, "system", lambda: "Windows")

    assert cache_module._default_cache_root() == tmp_path / "CiteMesh"


def test_legacy_macos_cache_root_requires_darwin_and_existing_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy discovery should expose only an existing macOS cache path.

    :param Path tmp_path: Temporary fake home directory.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate platform state.
    :return None: Validates legacy cache discovery boundaries.
    """
    monkeypatch.setattr(cache_module.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("CITEMESH_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(cache_module.platform, "system", lambda: "Linux")
    assert cache_module.legacy_macos_cache_root() is None

    monkeypatch.setattr(cache_module.platform, "system", lambda: "Darwin")
    assert cache_module.legacy_macos_cache_root() is None

    legacy = tmp_path / "Library" / "Caches" / "citemesh"
    legacy.mkdir(parents=True)
    assert cache_module.legacy_macos_cache_root() == legacy
    for override in ("CITEMESH_CACHE_DIR", "XDG_CACHE_HOME"):
        monkeypatch.setenv(override, str(tmp_path / "custom"))
        assert cache_module.legacy_macos_cache_root() is None
        monkeypatch.delenv(override)


def test_atomic_write_text_uses_binary_temp_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Atomic text writes should leave newline translation to the text wrapper.

    :param Path tmp_path: Temporary directory for the atomic-write target.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to observe descriptor mode.
    :return None: Validates descriptor and persisted newline bytes.
    """
    original_mkstemp = cache_module.tempfile.mkstemp
    observed_text_modes: list[bool] = []

    def _recording_mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
        """Record the descriptor mode while delegating temporary-file creation.

        :param Any args: Positional arguments forwarded to ``tempfile.mkstemp``.
        :param Any kwargs: Keyword arguments forwarded to ``tempfile.mkstemp``.
        :return tuple[int, str]: Open descriptor and temporary path.
        """
        observed_text_modes.append(bool(kwargs.get("text")))
        return original_mkstemp(*args, **kwargs)

    monkeypatch.setattr(cache_module.tempfile, "mkstemp", _recording_mkstemp)
    destination = tmp_path / "payload.txt"

    cache_module.atomic_write_text(destination, "first\nsecond\n")

    assert observed_text_modes == [False]
    assert destination.read_bytes() == b"first\nsecond\n"


def test_atomic_write_text_preserves_target_permissions(tmp_path: Path) -> None:
    """Rewriting an existing file must not narrow its permission bits.

    :param Path tmp_path: Temporary directory for the atomic-write target.
    :return None: Validates preserved, defaulted, and explicit modes.
    """
    shared = tmp_path / "graph.html"
    shared.write_text("original")
    shared.chmod(0o644)

    cache_module.atomic_write_text(shared, "rewritten")
    assert shared.stat().st_mode & 0o7777 == 0o644

    current_umask = os.umask(0)
    os.umask(current_umask)
    fresh = tmp_path / "fresh.json"
    cache_module.atomic_write_text(fresh, "{}")
    assert fresh.stat().st_mode & 0o7777 == 0o666 & ~current_umask

    secret = tmp_path / "config.toml"
    secret.write_text("old")
    secret.chmod(0o644)
    cache_module.atomic_write_text(secret, "new", mode=0o600)
    assert secret.stat().st_mode & 0o7777 == 0o600
