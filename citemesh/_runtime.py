"""Small runtime helpers shared across CLI and library modules."""

from __future__ import annotations

import os
import sys


def _fd_isatty(fd: int) -> bool:
    """Return whether a file descriptor is attached to a TTY.

    :param int fd: File descriptor number.
    :return bool: ``True`` when the descriptor is interactive.
    """
    try:
        return os.isatty(fd)
    except OSError:
        return False


def _stream_isatty(stream: object, fd: int) -> bool:
    """Return whether a stream is interactive, with a file-descriptor fallback.

    :param object stream: Stream object that may expose ``isatty``.
    :param int fd: Fallback file descriptor number.
    :return bool: ``True`` when the stream is interactive.
    """
    isatty = getattr(stream, "isatty", None)
    if callable(isatty):
        try:
            return bool(isatty())
        except (OSError, ValueError):
            return False
    return _fd_isatty(fd)


def stdin_isatty() -> bool:
    """Return whether stdin is attached to a TTY.

    :return bool: ``True`` when stdin is interactive.
    """
    return _stream_isatty(sys.stdin, 0)


def stderr_isatty() -> bool:
    """Return whether stderr is attached to a TTY.

    :return bool: ``True`` when stderr is interactive.
    """
    return _stream_isatty(sys.stderr, 2)
