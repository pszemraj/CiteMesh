"""External service client exports."""

from .semantic_scholar import SemanticScholarClient, get_client, reset_client

__all__ = ["SemanticScholarClient", "get_client", "reset_client"]
