"""Direct contracts for cross-platform cache path and atomic-write helpers."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import Any

import pytest
from filelock import Timeout

import citemesh.data.cache as cache_module


def test_cache_operation_lock_blocks_nonblocking_clear(tmp_path: Path) -> None:
    """An exclusive clear lock must reject while a shared operation is live.

    :param Path tmp_path: Temporary cache root.
    :return None: Validates shared/exclusive root coordination.
    """
    with cache_module.cache_operation_lock(tmp_path):
        with pytest.raises(Timeout):
            with cache_module.cache_operation_lock(
                tmp_path, exclusive=True, blocking=False
            ):
                pass

    with cache_module.cache_operation_lock(tmp_path, exclusive=True, blocking=False):
        pass


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

    def _denied(_path: Path) -> bool:
        """Simulate an inspection failure that is not absence.

        :param Path _path: Ignored inspected path.
        :return bool: Never returns; always raises.
        """
        raise PermissionError("denied")

    monkeypatch.setattr(cache_module, "path_exists", _denied)
    assert cache_module.legacy_macos_cache_root() == legacy


def test_path_exists_propagates_inspection_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only absence may read as False; other OS errors must propagate.

    :param Path tmp_path: Temporary directory with a real file.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to fake a stat failure.
    :return None: Validates the helper's error contract.
    """
    assert cache_module.path_exists(tmp_path) is True
    assert cache_module.path_exists(tmp_path / "absent") is False

    def _denied_stat(_self: Path) -> None:
        """Simulate a stat call rejected by the OS.

        :param Path _self: Ignored inspected path.
        :return None: Never returns; always raises.
        """
        raise PermissionError("denied")

    monkeypatch.setattr(cache_module.Path, "stat", _denied_stat)
    with pytest.raises(PermissionError):
        cache_module.path_exists(tmp_path)


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


def test_atomic_writes_without_fchmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomic text and JSON writes must work without the Unix-only descriptor API.

    :param Path tmp_path: Temporary directory for atomic-write targets.
    :param pytest.MonkeyPatch monkeypatch: Fixture simulating older Windows Python.
    :return None: Validates both shared-writer entry points persist their payloads.
    """
    monkeypatch.delattr(cache_module.os, "fchmod", raising=False)
    text_path = tmp_path / "config.toml"
    json_path = tmp_path / "paper.json"

    cache_module.atomic_write_text(text_path, 'theme = "dark"\n', mode=0o600)
    cache_module.atomic_write_json(json_path, {"paper_id": "seed"})

    assert text_path.read_text(encoding="utf-8") == 'theme = "dark"\n'
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"paper_id": "seed"}


@pytest.mark.parametrize("write_json", [False, True], ids=["text", "json"])
@pytest.mark.parametrize(
    "failure_stage", ["stat", "chmod", "fdopen", "fsync", "replace"]
)
def test_atomic_write_failure_closes_descriptor_and_preserves_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_json: bool,
    failure_stage: str,
) -> None:
    """Failed atomic writes must release their descriptor without replacing data.

    :param Path tmp_path: Temporary directory for the write target.
    :param pytest.MonkeyPatch monkeypatch: Fixture injecting filesystem failures.
    :param bool write_json: Whether to use the JSON entry point.
    :param str failure_stage: Operation that raises before replacement completes.
    :return None: Checks descriptor ownership, temporary cleanup, and old contents.
    """
    target = tmp_path / "payload.json"
    target.write_text("original", encoding="utf-8")
    original_mkstemp = cache_module.tempfile.mkstemp
    original_stat = Path.stat
    descriptors: list[int] = []

    def recording_mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
        """Record ownership of the real temporary descriptor.

        :param Any args: Arguments forwarded to the original factory.
        :param Any kwargs: Keyword arguments forwarded to the original factory.
        :return tuple[int, str]: Open descriptor and temporary path.
        """
        fd, name = original_mkstemp(*args, **kwargs)
        descriptors.append(fd)
        return fd, name

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        """Reject the selected filesystem operation.

        :param Any _args: Ignored positional arguments.
        :param Any _kwargs: Ignored keyword arguments.
        :return Any: Never returns.
        """
        raise PermissionError(f"injected {failure_stage} failure")

    def stat(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        """Fail only the destination lookup, allowing temporary-file setup.

        :param Path path: Inspected path.
        :param Any args: Additional stat arguments.
        :param Any kwargs: Additional stat keyword arguments.
        :return os.stat_result: Original metadata for unrelated paths.
        """
        if path == target:
            fail()
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(cache_module.tempfile, "mkstemp", recording_mkstemp)
    if failure_stage == "stat":
        monkeypatch.setattr(Path, "stat", stat)
    elif failure_stage == "chmod":
        monkeypatch.setattr(Path, "chmod", fail)
    else:
        monkeypatch.setattr(cache_module.os, failure_stage, fail)

    try:
        with pytest.raises(PermissionError, match=f"injected {failure_stage} failure"):
            if write_json:
                cache_module.atomic_write_json(target, {"updated": True})
            else:
                cache_module.atomic_write_text(target, "updated")
        assert len(descriptors) == 1
        with pytest.raises(OSError) as error:
            os.fstat(descriptors[0])
        assert error.value.errno == errno.EBADF
        assert target.read_text(encoding="utf-8") == "original"
        assert list(tmp_path.iterdir()) == [target]
    finally:
        for fd in descriptors:
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.mark.parametrize(
    "target_mode",
    [0o644, 0o444, 0o222, 0o000],
    ids=["read-write", "read-only", "write-only", "no-access"],
)
def test_atomic_write_text_preserves_target_permissions(
    tmp_path: Path, target_mode: int
) -> None:
    """Rewriting an existing file must not narrow its permission bits.

    :param Path tmp_path: Temporary directory for the atomic-write target.
    :param int target_mode: Existing permissions, including missing read/write access.
    :return None: Validates preserved, defaulted, and explicit modes.
    """
    shared = tmp_path / "graph.html"
    shared.write_text("original", encoding="utf-8")
    shared.chmod(target_mode)

    try:
        cache_module.atomic_write_text(shared, "rewritten")
        assert shared.stat().st_mode & 0o7777 == target_mode
    finally:
        shared.chmod(0o600)
    assert shared.read_text(encoding="utf-8") == "rewritten"

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
