"""External service client exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._lazy import install_lazy_exports

if TYPE_CHECKING:
    from .semantic_scholar import (
        SemanticScholarClient,
        SemanticScholarRequestError,
        SemanticScholarUnavailableError,
        get_client,
        reset_client,
    )

__all__ = [
    "SemanticScholarClient",
    "SemanticScholarRequestError",
    "SemanticScholarUnavailableError",
    "get_client",
    "reset_client",
]

# Lazy so importing the package does not pull the optional Semantic Scholar client.
__getattr__, __dir__ = install_lazy_exports(
    globals(),
    {name: (".semantic_scholar", name) for name in __all__},
)
