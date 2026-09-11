"""Semantic Scholar API client with error handling and caching.

The package splits the client into focused modules: :mod:`errors` (failure
taxonomy), :mod:`retry` (backoff policy), :mod:`disk_cache` (persisted
reference/paper caches), :mod:`payloads` (payload parsing and conversion), and
:mod:`client` (transport, endpoints, and the shared singleton). Only the names
re-exported here are part of the supported public surface.
"""

from __future__ import annotations

from citemesh.core.paper_ids import normalize_paper_id

from .client import SemanticScholarClient, get_client, reset_client
from .errors import SemanticScholarRequestError, SemanticScholarUnavailableError

__all__ = [
    "SemanticScholarClient",
    "SemanticScholarRequestError",
    "SemanticScholarUnavailableError",
    "get_client",
    "normalize_paper_id",
    "reset_client",
]
