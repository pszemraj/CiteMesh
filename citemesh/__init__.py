"""CiteMesh package exports."""

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - fallback for editable/source environments
    __version__ = "0.0.dev0"

__author__ = "CiteMesh Contributors"

from .core import Author, Paper

__all__ = [
    "__version__",
    "Paper",
    "Author",
]
