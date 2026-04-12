"""External service clients with lazy exports for optional boundaries."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .semantic_scholar import SemanticScholarClient

__all__ = ["get_client", "SemanticScholarClient", "reset_client"]


def get_client(*args: Any, **kwargs: Any) -> Any:
    """Lazily resolve and call the Semantic Scholar client factory."""
    module = import_module("citemesh.services.semantic_scholar")
    return module.get_client(*args, **kwargs)


def reset_client() -> None:
    """Lazily resolve and clear the process-wide Semantic Scholar client."""
    module = import_module("citemesh.services.semantic_scholar")
    module.reset_client()


def __getattr__(name: str) -> Any:
    """Lazily expose Semantic Scholar service helpers on first use."""
    if name != "SemanticScholarClient":
        raise AttributeError(f"module 'citemesh.services' has no attribute {name!r}")

    module = import_module("citemesh.services.semantic_scholar")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy service exports to runtime introspection."""
    return sorted(set(globals()) | set(__all__))
