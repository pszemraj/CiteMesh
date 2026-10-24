"""Direct contracts for cross-platform cache path and atomic-write helpers."""

from __future__ import annotations

import errno
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from filelock import Timeout

import citemesh.data.cache as cache_module
from citemesh.cli import cache_ops as cache_ops_module
from citemesh.data.embedding_cache.constants import (
    EMBEDDING_CACHE_SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
)
from tests._helpers import fail_stat_for


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


def test_embedding_namespace_paper_probe_distinguishes_empty_and_unreadable(
    tmp_path: Path,
) -> None:
    """The cheap namespace probe should preserve unknown metadata failures.

    :param Path tmp_path: Temporary directory holding metadata fixtures.
    :return None: Verifies populated, empty, and corrupt tri-state results.
    """
    metadata_path = tmp_path / "metadata.db"
    with sqlite3.connect(metadata_path) as connection:
        connection.execute("CREATE TABLE papers (paper_id TEXT PRIMARY KEY)")

    assert cache_module.embedding_namespace_has_papers(metadata_path) is False
    with sqlite3.connect(metadata_path) as connection:
        connection.execute("INSERT INTO papers (paper_id) VALUES ('paper')")
    assert cache_module.embedding_namespace_has_papers(metadata_path) is True

    corrupt_path = tmp_path / "corrupt.db"
    corrupt_path.write_bytes(b"not SQLite")
    assert cache_module.embedding_namespace_has_papers(corrupt_path) is None


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


@pytest.mark.parametrize(
    ("platform_name", "environment_variable", "expected_leaf"),
    [("Darwin", "XDG_CACHE_HOME", "citemesh"), ("Windows", "LOCALAPPDATA", "CiteMesh")],
)
def test_default_cache_root_uses_platform_environment_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    environment_variable: str,
    expected_leaf: str,
) -> None:
    """Platform defaults should use their configured cache environment path.

    :param Path tmp_path: Temporary cache base.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to isolate environment state.
    :param str platform_name: Operating system branch under test.
    :param str environment_variable: Environment key supplying the base path.
    :param str expected_leaf: Platform-specific final cache directory name.
    :return None: Validates the platform cache path contract.
    """
    monkeypatch.delenv("CITEMESH_CACHE_DIR", raising=False)
    monkeypatch.setenv(environment_variable, str(tmp_path))
    monkeypatch.setattr(cache_module.platform, "system", lambda: platform_name)

    assert cache_module._default_cache_root() == tmp_path / expected_leaf


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
    :return None: Validates parent creation, descriptor mode, and persisted bytes.
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
    destination = tmp_path / "missing" / "nested" / "payload.txt"
    assert not destination.parent.exists()

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


def test_atomic_output_path_replaces_existing_writable_file(tmp_path: Path) -> None:
    """A library-written staged output replaces its writable destination.

    :param Path tmp_path: Temporary directory for the atomic output target.
    :return None: Validates staged-file synchronization and replacement.
    """
    destination = tmp_path / "graph.png"
    destination.write_bytes(b"old")

    with cache_module.atomic_output_path(destination) as staged:
        staged.write_bytes(b"new")

    assert destination.read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [destination]


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

    monkeypatch.setattr(cache_module.tempfile, "mkstemp", recording_mkstemp)
    if failure_stage == "stat":
        monkeypatch.setattr(
            Path,
            "stat",
            fail_stat_for(target, PermissionError(f"injected {failure_stage} failure")),
        )
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
@pytest.mark.skipif(os.name != "posix", reason="exact mode bits are POSIX semantics")
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


def _write_embedding_namespace_schema(db_path: Path, schema_version: int) -> None:
    """Create the minimal read-only metadata shape used by the cache scanner.

    :param Path db_path: SQLite metadata database to create.
    :param int schema_version: Embedding cache schema token to persist.
    :return None: Writes the metadata database.
    """
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE cache_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO cache_metadata (key, value) VALUES (?, ?)",
            (SCHEMA_VERSION_KEY, str(schema_version)),
        )


def test_embedding_namespace_scan_groups_artifacts_and_flags_dtype_upgrade_cache(
    tmp_path: Path,
) -> None:
    """The scanner must expose reclaimable dtype-keyed namespaces individually.

    :param Path tmp_path: Temporary embedding-cache directory.
    :return None: Checks grouping, byte accounting, and legacy upgrade status.
    """
    current_namespace = "0123456789ab"
    legacy_namespace = "fedcba987654"
    orphaned_namespace = "a1b2c3d4e5f6"
    cleaned_namespace = "123456789abc"

    current_db = tmp_path / f"metadata_{current_namespace}.db"
    legacy_db = tmp_path / f"metadata_{legacy_namespace}.db"
    _write_embedding_namespace_schema(current_db, EMBEDDING_CACHE_SCHEMA_VERSION)
    _write_embedding_namespace_schema(legacy_db, 3)

    current_artifacts = [
        current_db,
        tmp_path / f"embeddings_{current_namespace}.h5",
        tmp_path / f"cache_{current_namespace}.lock",
    ]
    legacy_artifacts = [
        legacy_db,
        tmp_path / f"embeddings_{legacy_namespace}.h5",
        tmp_path / f"metadata_{legacy_namespace}.db-journal",
        tmp_path / f"metadata_{legacy_namespace}.db-wal",
        tmp_path / f"hydration_{legacy_namespace}.lock",
    ]
    orphaned_artifact = tmp_path / f"embeddings_{orphaned_namespace}.h5"
    cleaned_locks = [
        tmp_path / f"cache_{cleaned_namespace}.lock",
        tmp_path / f"hydration_{cleaned_namespace}.lock",
    ]
    for index, artifact_path in enumerate(
        current_artifacts[1:]
        + legacy_artifacts[1:]
        + [orphaned_artifact]
        + cleaned_locks,
        start=1,
    ):
        artifact_path.write_bytes(bytes(index))

    rows = {
        row.namespace_id: row
        for row in cache_ops_module._scan_embedding_namespaces(tmp_path)
    }

    assert set(rows) == {current_namespace, legacy_namespace, orphaned_namespace}
    assert rows[current_namespace].file_count == len(current_artifacts)
    assert rows[current_namespace].size_bytes == sum(
        path.stat().st_size for path in current_artifacts
    )
    assert rows[current_namespace].status == "Current schema"
    assert rows[legacy_namespace].file_count == len(legacy_artifacts)
    assert rows[legacy_namespace].size_bytes == sum(
        path.stat().st_size for path in legacy_artifacts
    )
    assert rows[legacy_namespace].status.startswith("Legacy dtype-keyed namespace")
    assert rows[orphaned_namespace].status == (
        "Orphaned artifact (metadata database missing)"
    )
    assert cleaned_namespace not in rows


@pytest.mark.parametrize(
    "state",
    [[], {"source": [1]}, {"source": {"selection": 1, "limit": 5}}],
)
def test_embedding_namespace_scan_tolerates_malformed_corpus_state(
    tmp_path: Path, state: object
) -> None:
    """Unreadable corpus state must not prevent inspecting other cache files."""
    db_path = tmp_path / "metadata_0123456789ab.db"
    _write_embedding_namespace_schema(db_path, EMBEDDING_CACHE_SCHEMA_VERSION)
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE papers (paper_id TEXT)")
        connection.execute("CREATE TABLE corpus_membership (paper_id TEXT)")
        connection.execute(
            "INSERT INTO cache_metadata (key, value) VALUES (?, ?)",
            ("corpus_state", json.dumps(state)),
        )

    rows = cache_ops_module._scan_embedding_namespaces(tmp_path)

    assert len(rows) == 1
    assert rows[0].status == "Corpus metadata unavailable (inspect before removing)"
