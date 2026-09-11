"""Value types and the layout-error signal shared across the cache package.

Owns the frozen dataclasses handed across the cache boundary
(:class:`CacheSearchResult`, :class:`CacheNamespacePayloadStats`,
:class:`PendingEmbeddingRecord`) and the internal
:class:`_EmbeddingCacheLayoutError` raised when a persisted layout is proven
incompatible and the namespace must be rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


class _EmbeddingCacheLayoutError(ValueError):
    """A proven persisted layout mismatch that requires a namespace rebuild."""


@dataclass(frozen=True)
class CacheSearchResult:
    """Search result returned by ``EmbeddingCache.search``."""

    paper_id: str
    score: float
    embedding: np.ndarray
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class CacheNamespacePayloadStats:
    """Namespace payload summary used for clear-impact reporting."""

    file_count: int
    size_bytes: int
    sqlite_rows: int
    embedding_rows: int
    hydration_complete: bool
    hydration_split: Optional[str]
    hydration_corpus_size: Optional[str]
    hydration_dataset_source: Optional[str]


@dataclass(frozen=True)
class PendingEmbeddingRecord:
    """Cache-miss record staged across lookup/encode/commit phases."""

    paper_id: str
    metadata: Dict[str, object]
    text_hash: str
    text: str
    row_idx: Optional[int]
