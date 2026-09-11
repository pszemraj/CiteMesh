"""Value types and the layout-error signal shared across the cache package.

Owns the frozen dataclasses handed across the cache boundary
(:class:`CacheSearchResult`, :class:`CacheNamespacePayloadStats`,
:class:`PendingEmbeddingRecord`) and the internal
:class:`_EmbeddingCacheLayoutError` raised when a persisted layout is proven
incompatible and the namespace must be rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass(frozen=True)
class _CacheLookupPlan:
    """Outcome of the pre-encode cache scan for one embedding request."""

    cached_embeddings: Dict[str, np.ndarray]
    papers_to_embed: List[PendingEmbeddingRecord]


@dataclass(frozen=True)
class _Int8Saturation:
    """Clipping counts observed while quantizing one encode batch to int8."""

    clipped_values: int
    total_values: int


@dataclass
class _VectorWritePlan:
    """Row-level write plan derived from one encoded batch.

    ``replacement_rows`` carries ``(row_idx, previous_embedding, previous_binary,
    replacement_embedding, replacement_binary)`` so the journal can record the
    prior vectors before the new ones overwrite them.
    """

    existing_row_count: int
    new_embeddings: Dict[str, np.ndarray]
    rows_to_upsert: List[Tuple[Any, ...]]
    append_embeddings: List[np.ndarray]
    append_binary_embeddings: List[np.ndarray]
    append_records: List[Tuple[str, Dict[str, object], str]]
    replacement_rows: List[
        Tuple[int, np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]
    ]
