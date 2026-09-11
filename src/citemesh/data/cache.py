"""
Helpers for determining cache directories in a cross-platform way.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO

from filelock import ReadWriteLock

CACHE_COORDINATION_DIRNAME = ".locks"
CACHE_OPERATION_LOCK_FILENAME = "cache-operations.db"


def path_exists(path: Path) -> bool:
    """Distinguish an absent persisted file from a failed filesystem inspection.

    ``Path.exists()`` suppresses all OS errors on Python 3.14 and later.

    :param Path path: Persisted file path to inspect.
    :return bool: Whether the path exists.
    :raises OSError: If inspection fails for a reason other than absence.
    """
    try:
        path.stat()
    except FileNotFoundError:
        return False
    return True


def _default_cache_root() -> Path:
    """Return user-level cache root honoring platform conventions.

    :return Path: Base cache directory for CiteMesh artifacts.
    """
    override = os.getenv("CITEMESH_CACHE_DIR")
    if override:
        return Path(override).expanduser()

    system = platform.system()

    if system == "Windows":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        if base:
            return Path(base) / "CiteMesh"
        return Path.home() / "AppData" / "Local" / "CiteMesh"

    # macOS and Linux share the HuggingFace-style ~/.cache/citemesh layout so
    # cache paths (and config.toml) are predictable across machines.
    xdg_cache = os.getenv("XDG_CACHE_HOME")
    cache_root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache_root / "citemesh"


def legacy_macos_cache_root() -> Path | None:
    """Return the pre-unification macOS cache root when it still exists.

    :return Path | None: Legacy ``~/Library/Caches/citemesh`` path or ``None``.
    """
    if platform.system() != "Darwin" or any(
        os.getenv(name) for name in ("CITEMESH_CACHE_DIR", "XDG_CACHE_HOME")
    ):
        return None
    legacy = Path.home() / "Library" / "Caches" / "citemesh"
    try:
        return legacy if path_exists(legacy) else None
    except OSError:
        # Inspection failed for a reason other than absence: the directory is
        # present but unreadable, which is exactly when the hint matters.
        return legacy


def get_cache_dir(*parts: str, create: bool = True) -> Path:
    """
    Get (and optionally create) a cache directory scoped to CiteMesh.

    :param str parts: Additional subdirectories to append.
    :param bool create: Whether to create the directory if it does not exist.
    :return Path: Path to the requested cache directory.
    """
    path = _default_cache_root()
    if parts:
        path = path.joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def cache_operation_lock(
    cache_root: Path,
    *,
    exclusive: bool = False,
    timeout: float = -1,
    blocking: bool = True,
) -> Iterator[None]:
    """Coordinate a cache operation with cache-root clearing.

    :param Path cache_root: CiteMesh cache root whose artifacts are coordinated.
    :param bool exclusive: Whether to acquire the exclusive clear lock.
    :param float timeout: Maximum acquisition wait in seconds; ``-1`` waits indefinitely.
    :param bool blocking: Whether acquisition may wait for conflicting operations.
    :return Iterator[None]: Context manager holding a shared operation or exclusive clear lock.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_dir = cache_root / CACHE_COORDINATION_DIRNAME
    lock_dir.mkdir(exist_ok=True)
    lock = ReadWriteLock(
        lock_dir / CACHE_OPERATION_LOCK_FILENAME,
        timeout=timeout,
        blocking=blocking,
        is_singleton=False,
    )
    lock_context = lock.write_lock() if exclusive else lock.read_lock()
    try:
        with lock_context:
            yield
    finally:
        lock.close()


def _atomic_write_text_payload(
    path: Path,
    writer: Callable[[TextIO], object],
    *,
    newline: str | None,
    mode: int | None = None,
) -> None:
    """Atomically replace a UTF-8 text file using a writer callback.

    :param Path path: Target text file path.
    :param Callable[[TextIO], object] writer: Callback that writes the payload.
    :param str | None newline: Text-mode newline translation policy.
    :param int | None mode: Explicit permission bits, or ``None`` to keep the
        target's existing mode (umask default for new files).
    :return None: Writes and durably replaces the target file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=False,
    )
    tmp_path = Path(tmp_name)
    try:
        try:
            # Keep descriptor ownership even when constructing the wrapper fails.
            with os.fdopen(
                fd, "w", encoding="utf-8", newline=newline, closefd=False
            ) as tmp_file:
                # mkstemp creates the temp file 0600 and os.replace carries that mode
                # onto the target; without an explicit chmod every rewrite would
                # silently restrict shared artifacts to owner-only.
                if mode is None:
                    try:
                        mode = path.stat().st_mode & 0o7777
                    except FileNotFoundError:
                        current_umask = os.umask(0)
                        os.umask(current_umask)
                        mode = 0o666 & ~current_umask
                tmp_path.chmod(mode)
                writer(tmp_file)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
        finally:
            os.close(fd)

        # The payload is already synced; its preserved mode may forbid reopening.
        os.replace(tmp_name, path)
        directory_fd: int | None = None
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


@contextmanager
def atomic_output_path(path: Path) -> Iterator[Path]:
    """Publish a library-written file with an atomic replacement.

    The yielded temporary path is closed, resides beside the destination, and
    retains the destination suffix so libraries can infer their output format.

    :param Path path: Final output path.
    :return Iterator[Path]: Closed temporary path for the library writer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.stat().st_mode & 0o7777
    except FileNotFoundError:
        current_umask = os.umask(0)
        os.umask(current_umask)
        mode = 0o666 & ~current_umask

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        yield tmp_path
        with tmp_path.open("rb") as tmp_file:
            tmp_path.chmod(mode)
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
        directory_fd: int | None = None
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_text(
    path: Path,
    content: str,
    *,
    newline: str | None = "",
    mode: int | None = None,
) -> None:
    """Persist complete UTF-8 text with a crash-safe atomic rename.

    :param Path path: Target text file path.
    :param str content: Complete text payload.
    :param str | None newline: Text-mode newline translation policy.
    :param int | None mode: Explicit permission bits, or ``None`` to keep the
        target's existing mode (umask default for new files).
    :return None: Writes the target file in place.
    """
    _atomic_write_text_payload(
        path,
        lambda handle: handle.write(content),
        newline=newline,
        mode=mode,
    )


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int | None = None,
    sort_keys: bool = True,
) -> None:
    """Persist JSON content with a crash-safe atomic rename.

    :param Path path: Target JSON file path.
    :param Any payload: JSON-serializable payload to write.
    :param int | None indent: Optional JSON indentation level.
    :param bool sort_keys: Whether to sort object keys during serialization.
    :return None: Writes the target file in place.
    """
    writer = partial(json.dump, payload, indent=indent, sort_keys=sort_keys)
    _atomic_write_text_payload(path, writer, newline=None)


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Read a JSON object from disk, treating any unusable payload as absent.

    :param Path path: JSON file path to read.
    :return dict[str, Any] | None: Parsed mapping, or ``None`` when the file
        cannot be read or decoded (``OSError``, ``UnicodeDecodeError``,
        ``json.JSONDecodeError``) or its parsed value is not a JSON object.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def format_bytes(num_bytes: int) -> str:
    """Format byte counts into readable binary units.

    :param int num_bytes: Raw byte count.
    :return str: Human-readable size string.
    """
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(max(int(num_bytes), 0))
    unit = units[0]
    for candidate in units:
        unit = candidate
        if value < 1024.0 or candidate == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"
