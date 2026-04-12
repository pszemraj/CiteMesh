"""
CiteMesh: Citation mesh visualization toolkit.

A unified package for creating academic paper similarity graphs using
multiple strategies: citation networks, semantic embeddings, or hybrid approaches.
"""

from typing import Any

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - fallback for editable/source environments
    __version__ = "0.0.dev0"

__author__ = "CiteMesh Contributors"

from citemesh.core import Author, Paper

from . import strategies as _strategies

__all__ = [
    "__version__",
    "Paper",
    "Author",
    *_strategies.__all__,
]


def __getattr__(name: str) -> Any:
    """Lazily resolve strategy exports to keep optional boundaries lightweight."""
    if name in _strategies.__all__:
        value = getattr(_strategies, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'citemesh' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Expose lazy-exported names to IDEs and runtime introspection."""
    return sorted(set(globals()) | set(__all__))
