"""
Helpers for determining cache directories in a cross-platform way.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def _default_cache_root() -> Path:
    """Return user-level cache root honoring platform conventions."""
    override = os.getenv("CITEMESH_CACHE_DIR")
    if override:
        return Path(override)

    if sys.platform.startswith("win"):
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        if base:
            return Path(base) / "CiteMesh"
        return Path.home() / "AppData" / "Local" / "CiteMesh"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "citemesh"

    xdg_cache = os.getenv("XDG_CACHE_HOME")
    cache_root = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return cache_root / "citemesh"


def get_cache_dir(*parts: str, create: bool = True) -> Path:
    """
    Get (and optionally create) a cache directory scoped to CiteMesh.

    Args:
        *parts: Additional subdirectories to append.
        create: Whether to create the directory if it does not exist.

    Returns:
        Path to the requested cache directory.
    """
    path = _default_cache_root()
    if parts:
        path = path.joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
