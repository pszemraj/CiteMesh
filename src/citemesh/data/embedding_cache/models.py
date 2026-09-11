"""Value types and the layout-error signal shared across the cache package.

Owns the frozen dataclasses handed across the cache boundary
(:class:`CacheSearchResult`, :class:`CacheNamespacePayloadStats`,
:class:`PendingEmbeddingRecord`) and the internal
:class:`_EmbeddingCacheLayoutError` raised when a persisted layout is proven
incompatible and the namespace must be rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class _EmbeddingCacheLayoutError(ValueError):
    """A proven persisted layout mismatch that requires a namespace rebuild."""


@dataclass(frozen=True)
class CacheSearchResult:
    """Search result returned by ``EmbeddingCache.search``."""

    paper_id: str
    score: float
    embedding: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CacheNamespacePayloadStats:
    """Namespace payload summary used for clear-impact reporting."""

    file_count: int
    size_bytes: int
    sqlite_rows: int
    embedding_rows: int
    hydration_complete: bool
    hydration_split: str | None
    hydration_corpus_size: str | None
    hydration_dataset_source: str | None


@dataclass(frozen=True)
class PendingEmbeddingRecord:
    """Cache-miss record staged across lookup/encode/commit phases."""

    paper_id: str
    metadata: dict[str, object]
    text_hash: str
    text: str
    row_idx: int | None


@dataclass(frozen=True)
class _CacheLookupPlan:
    """Outcome of the pre-encode cache scan for one embedding request."""

    cached_embeddings: dict[str, np.ndarray]
    papers_to_embed: list[PendingEmbeddingRecord]


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
    new_embeddings: dict[str, np.ndarray]
    rows_to_upsert: list[tuple[Any, ...]]
    append_embeddings: list[np.ndarray]
    append_binary_embeddings: list[np.ndarray]
    append_records: list[tuple[str, dict[str, object], str]]
    replacement_rows: list[
        tuple[int, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]
    ]
