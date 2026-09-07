"""
Helpers for determining cache directories in a cross-platform way.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Callable, TextIO


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
    return legacy if legacy.exists() else None


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
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as tmp_file:
            writer(tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.replace(tmp_name, path)
        with path.open("r+b") as final_file:
            os.fsync(final_file.fileno())
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
