"""Small runtime helpers shared across CLI and library modules."""

from __future__ import annotations

import os


def _fd_isatty(fd: int) -> bool:
    """Return whether a file descriptor is attached to a TTY.

    :param int fd: File descriptor number.
    :return bool: ``True`` when the descriptor is interactive.
    """
    try:
        return os.isatty(fd)
    except OSError:
        return False


def stdin_isatty() -> bool:
    """Return whether stdin is attached to a TTY.

    :return bool: ``True`` when stdin is interactive.
    """
    return _fd_isatty(0)


def stderr_isatty() -> bool:
    """Return whether stderr is attached to a TTY.

    :return bool: ``True`` when stderr is interactive.
    """
    return _fd_isatty(2)
