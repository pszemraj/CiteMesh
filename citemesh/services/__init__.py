"""External service clients."""

from .semantic_scholar import SemanticScholarClient, get_client, reset_client

__all__ = ["get_client", "SemanticScholarClient", "reset_client"]
