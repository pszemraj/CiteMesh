"""Value types and the layout-error signal shared across the cache package.

Owns the frozen dataclasses handed across the cache boundary
(:class:`CacheSearchResult`, :class:`CacheNamespacePayloadStats`,
:class:`PendingEmbeddingRecord`) and the internal
:class:`_EmbeddingCacheLayoutError` raised when a persisted layout is proven
incompatible and the namespace must be rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from ..embedding_contract import CorpusSourceData, EmbeddingContractData

StoredEmbedding = NDArray[np.int8] | NDArray[np.float32]
VectorReplacement = tuple[
    int,
    StoredEmbedding,
    NDArray[np.uint8] | None,
    StoredEmbedding,
    NDArray[np.uint8] | None,
]


class PaperMetadata(TypedDict, total=False):
    """Optional bibliographic fields accepted by the embedding cache."""

    title: str | None
    abstract: str | None
    year: int | str | None
    authors: list[str] | None
    categories: list[str] | None
    venue: str | None
    arxiv_id: str | None
    doi: str | None


class EmbeddingInputMetadata(PaperMetadata, total=False):
    """Known metadata fields passed with an embedding-cache request."""

    paper_id: str
    source_ordinal: int
    chronology_key: int | None


class CachedPaperMetadata(TypedDict):
    """Normalized metadata returned from a persisted paper row."""

    title: str
    abstract: str
    year: int | None
    authors: list[str]
    categories: list[str]
    venue: str
    arxiv_id: str
    doi: str


class CachedPaperRecord(CachedPaperMetadata):
    """Normalized cache metadata with its primary paper ID."""

    paper_id: str


class CorpusSourceRecord(PaperMetadata):
    """Normalized source row selected for corpus hydration."""

    paper_id: str
    source_ordinal: int
    chronology_key: int | None


class CorpusMetadataUpdate(PaperMetadata):
    """Source metadata fields used to refresh a cached paper row."""

    paper_id: str


CorpusMetadataUpdateValues: TypeAlias = CorpusMetadataUpdate | CachedPaperRecord


class CorpusRecord(TypedDict):
    """Persisted corpus row carried by a portable snapshot."""

    paper_id: str
    arxiv_id: str | None
    doi: str | None
    title: str | None
    abstract: str | None
    year: int | None
    authors: list[str] | None
    categories: list[str] | None
    venue: str | None
    text_hash: str
    chronology_key: int | None
    source_ordinal: int


CacheInputMetadata: TypeAlias = (
    EmbeddingInputMetadata | CorpusSourceRecord | CachedPaperRecord
)
CacheMetadataValues: TypeAlias = CacheInputMetadata | CorpusRecord
PaperRequest = tuple[str, CacheInputMetadata]


class ActiveCorpusRecord(CorpusRecord):
    """Persisted corpus row with its physical embedding-row index."""

    row_idx: int


class PortableShardData(TypedDict):
    """Serialized row range and payload path for one portable shard."""

    path: str
    first_row: int
    row_count: int


class PortableImportIdentity(TypedDict):
    """Manifest fields that identify an import operation for resumption."""

    snapshot_id: str
    contract: EmbeddingContractData
    source: CorpusSourceData
    shards: list[PortableShardData]


class RuntimeVersions(TypedDict):
    """Runtime package versions recorded when exporting a snapshot."""

    python: str
    numpy: str
    h5py: str
    pyarrow: str


class PublicationOrigin(TypedDict):
    """Bucket publication identity attached to an imported cache."""

    bucket_id: str
    config_name: str
    snapshot_id: str


class SnapshotOrigin(TypedDict):
    """Local snapshot identity recorded without a bucket publication."""

    snapshot_id: str


CorpusOrigin = SnapshotOrigin | PublicationOrigin


class PortableProducerProvenance(TypedDict, total=False):
    """Known producer and runtime details carried by a portable manifest."""

    source_torch_dtype: str
    embedding_vector_dtype: str
    binary_prefilter_used: bool | None
    active_model: str
    model_fingerprint: str | None
    resolved_model_revision: str | None
    truncate_dim: int | None
    device: str
    requested_device: str
    compute_dtype: str
    autocast: bool
    retrieval_representation: str
    graph_representation: str
    model_profile: str
    dataset_revision: str | None
    cache_namespace: str
    index_origin: CorpusOrigin | None
    export_runtime_versions: RuntimeVersions


PaperRowValues = tuple[
    str,
    str,
    str,
    int | None,
    str,
    int,
    int,
    str,
    str,
    str,
    str,
    str,
    int | None,
]

NormalizedMetadataFields = tuple[str, str, int | None, str, str, str, str, str]


class _EmbeddingCacheLayoutError(ValueError):
    """A proven persisted layout mismatch that requires a namespace rebuild."""


@dataclass(frozen=True)
class CacheSearchResult:
    """Search result returned by ``EmbeddingCache.search``."""

    paper_id: str
    score: float
    embedding: NDArray[np.float32]
    metadata: CachedPaperMetadata


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
    metadata: CacheInputMetadata
    text_hash: str
    text: str
    row_idx: int | None


@dataclass(frozen=True)
class _CacheLookupPlan:
    """Outcome of the pre-encode cache scan for one embedding request."""

    cached_embeddings: dict[str, NDArray[np.float32]]
    papers_to_embed: list[PendingEmbeddingRecord]


@dataclass(frozen=True)
class _Int8Saturation:
    """Clipping counts observed while quantizing one encode batch to int8."""

    clipped_values: int
    total_values: int


@dataclass
class _VectorWritePlan:
    """Describe metadata and vector writes prepared for one encoded batch."""

    existing_row_count: int
    new_embeddings: dict[str, NDArray[np.float32]]
    rows_to_upsert: list[PaperRowValues]
    append_embeddings: list[StoredEmbedding]
    append_binary_embeddings: list[NDArray[np.uint8]]
    append_records: list[tuple[str, CacheInputMetadata, str]]
    replacement_rows: list[VectorReplacement]
