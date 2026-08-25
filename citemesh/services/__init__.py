"""External service client exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .semantic_scholar import (
        SemanticScholarClient,
        SemanticScholarUnavailableError,
        get_client,
        reset_client,
    )

__all__ = [
    "SemanticScholarClient",
    "SemanticScholarUnavailableError",
    "get_client",
    "reset_client",
]


def __getattr__(name: str) -> Any:
    """Resolve service exports lazily to avoid eager optional-client imports.

    :param str name: Requested module attribute.
    :return Any: Export resolved from ``semantic_scholar``.
    :raises AttributeError: If the attribute is not a supported export.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(".semantic_scholar", __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    """Return sorted module attribute names for interactive inspection.

    :return list[str]: Sorted module attribute names plus lazy exports.
    """
    return sorted(set(globals()) | set(__all__))
