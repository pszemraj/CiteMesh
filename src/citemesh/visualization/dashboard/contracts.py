"""Versioned schema identities shared by dashboard producers and consumers."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Final, Literal, TypedDict

if TYPE_CHECKING:
    from citemesh.data.embedding_cache.models import CorpusOrigin


class _ScoreContractRequired(TypedDict):
    """Scoring fields present in every strategy contract."""

    strategy: str
    comparable_across_strategies: Literal[False]
    range_hint: str
    score_type: Literal[
        "citation_similarity_composite",
        "recommendation_similarity_composite",
        "embedding_similarity_composite",
        "hybrid_similarity_composite",
        "unknown",
    ]


class ScoreContract(_ScoreContractRequired, total=False):
    """Scoring semantics recorded for one exported graph."""

    adjudication_policy: str


class _DashboardCollectionResultRequired(TypedDict):
    """Identity and graph data required in a dashboard collection result."""

    result_id: str
    payload: Mapping[str, object]


class DashboardCollectionResult(_DashboardCollectionResultRequired, total=False):
    """Optional descriptor fields retained for one collection result."""

    seed_id: object
    title: object
    strategy: object
    summary: object
    updated_at: object
    build: dict[object, object]


class _DashboardCollectionBundleRequired(TypedDict):
    """Fields present in every dashboard collection bundle."""

    current_result_id: str | None
    results: list[DashboardCollectionResult]


class DashboardCollectionBundle(_DashboardCollectionBundleRequired, total=False):
    """Collection bundle with optional version markers for legacy metadata."""

    kind: Literal["citemesh-dashboard-collection"]
    schema_version: Literal[1]


class _EmbeddingExportMetadataRequired(TypedDict):
    """Embedding runtime fields always present in candidate-build metadata."""

    effective_vector_dtype: Literal["float32"]
    effective_model: str
    model_profile: str
    retrieval_representation: str
    graph_representation: str
    semantic_source: str
    storage_precision: str
    binary_prefilter_enabled: bool
    binary_rescore_multiplier: int


class EmbeddingExportMetadata(_EmbeddingExportMetadataRequired, total=False):
    """Embedding metadata with nullable and all-corpus fields omitted as needed."""

    effective_model_revision: str | None
    model_fingerprint: str | None
    effective_truncate_dim: int | None
    effective_device: str | None
    effective_compute_dtype: str | None
    binary_prefilter_used_for_query: bool | None
    cache_overwrite_reason: str | None
    candidate_pool_size: int
    dataset_revision: str | None
    cache_namespace: str
    index_origin: CorpusOrigin | None


class ExportMetadata(TypedDict, total=False):
    """Optional metadata accepted by graph exporters."""

    paper_id: str
    seed_id: str
    strategy: str
    nodes: int
    edges: int
    theme: str
    score_contract: ScoreContract
    candidate_source_status: dict[str, str]
    embedding: EmbeddingExportMetadata
    timestamp: str
    dashboard_collection: DashboardCollectionBundle


class PlotOverlayMetadata(TypedDict, total=False):
    """Metadata fields used in static plot annotations."""

    paper_id: str
    strategy: str
    nodes: int
    edges: int
    theme: str
    timestamp: str


class EffectiveNodeMetadata(TypedDict):
    """Known node fields before the exporter normalizes raw graph attributes."""

    id: Hashable
    title: object
    year: object
    authors: object
    citation_count: object
    abstract: object
    venue: object
    arxiv_id: object
    doi: object
    categories: object
    is_seed: bool
    is_local_corpus: bool


class EnrichedNodePayload(TypedDict):
    """Bibliographic and relationship fields exported for one graph node."""

    id: str
    title: object
    year: int
    authors: list[str]
    citation_count: int
    abstract: str
    venue: str
    arxiv_id: str
    doi: str
    categories: list[str]
    is_seed: bool
    is_local_corpus: bool
    provenance: str
    provenance_base: str
    seed_relation: str
    seed_relevance: float
    links: dict[str, str | None]
    bibtex: str


class GraphSummary(TypedDict):
    """Node and edge counts shared by graph payloads and dashboards."""

    nodes: int
    edges: int


class YearRange(TypedDict):
    """Minimum and maximum publication years present in a graph."""

    min: int
    max: int


class _DashboardMetadataRequired(TypedDict):
    """Summary fields present in every dashboard metadata object."""

    seed_id: str
    strategy: str
    theme: str
    summary: GraphSummary
    year_range: YearRange | None


class DashboardMetadata(_DashboardMetadataRequired, total=False):
    """Dashboard summary and optional geometry aligned with node order."""

    candidate_source_status: dict[str, str]
    plotly_node_order: list[str]
    plotly_positions: list[list[float]]
    plotly_node_sizes: list[float]


class GraphPayloadEdge(TypedDict):
    """Enriched edge record in the portable graph payload."""

    source: str
    target: str
    source_title: str
    target_title: str
    source_label: str
    target_label: str
    weight: float


class DashboardEdge(TypedDict):
    """Edge record used by the interactive dashboard viewer."""

    source: str
    target: str
    weight: float


class _PortableGraphMetadataRequired(TypedDict):
    """Fields always present in portable graph metadata."""

    strategy: str
    year_range: YearRange | None


class PortableGraphMetadata(_PortableGraphMetadataRequired, total=False):
    """Strategy and chronology fields attached to a graph payload."""

    candidate_source_status: dict[str, str]


class GraphPayloadDashboard(TypedDict):
    """Dashboard metadata embedded in a portable graph payload."""

    meta: DashboardMetadata


class GraphPayload(TypedDict):
    """Versioned graph export consumed by dashboard collection loaders."""

    kind: Literal["citemesh-graph"]
    schema_version: Literal[1]
    seed_id: str
    meta: PortableGraphMetadata
    summary: GraphSummary
    nodes: list[EnrichedNodePayload]
    dashboard: GraphPayloadDashboard
    edges: list[GraphPayloadEdge]


class DashboardPayload(TypedDict):
    """Graph data consumed by the dashboard JavaScript viewer."""

    meta: DashboardMetadata
    nodes: list[EnrichedNodePayload]
    edges: list[DashboardEdge]


class ValidatedPayloadNode(TypedDict):
    """Node field required by dashboard package validation."""

    id: str


class ValidatedPayloadEdge(TypedDict):
    """Edge endpoints required by dashboard package validation."""

    source: str
    target: str


class _ValidatedPortableGraphMetadataRequired(TypedDict):
    """Portable graph metadata field checked by the package loader."""

    strategy: str


class ValidatedPortableGraphMetadata(
    _ValidatedPortableGraphMetadataRequired, total=False
):
    """Portable metadata subset guaranteed by graph package validation."""


class ValidatedDashboardMetadata(TypedDict):
    """Dashboard fields checked before a package can be loaded."""

    seed_id: str
    strategy: str
    summary: GraphSummary
    plotly_node_order: list[str]
    plotly_positions: list[list[int | float]]
    plotly_node_sizes: list[int | float]


class ValidatedGraphPayloadDashboard(TypedDict):
    """Dashboard section guaranteed by graph package validation."""

    meta: ValidatedDashboardMetadata


class ValidatedGraphPayload(TypedDict):
    """Graph payload fields guaranteed by collection package validation."""

    kind: Literal["citemesh-graph"]
    schema_version: Literal[1]
    seed_id: str
    meta: ValidatedPortableGraphMetadata
    summary: GraphSummary
    nodes: list[ValidatedPayloadNode]
    dashboard: ValidatedGraphPayloadDashboard
    edges: list[ValidatedPayloadEdge]


class DashboardResultEntry(TypedDict):
    """Normalized result descriptor stored in a dashboard collection."""

    result_id: str
    seed_id: str
    title: str
    strategy: str
    summary: GraphSummary
    updated_at: str
    payload: ValidatedGraphPayload
    build: dict[object, object]


class DashboardCollectionPackage(TypedDict):
    """Versioned collection package returned by its loader and updater."""

    kind: Literal["citemesh-dashboard-collection"]
    schema_version: Literal[1]
    current_result_id: str | None
    results: list[DashboardResultEntry]


class CitationConfig(TypedDict, total=False):
    """Citation and reference options recorded for one graph build."""

    max_citations: int
    max_references: int
    fetch_references: bool
    refresh_reference_cache: bool
    similarity_threshold: float


class HybridConfig(TypedDict):
    """Semantic candidate budget recorded for a hybrid graph build."""

    max_semantic: int


class EmbeddingConfig(TypedDict, total=False):
    """Effective embedding options recorded in a build sidecar."""

    cache_namespace: str | None
    model: str
    model_profile: str
    model_revision: str | None
    semantic_source: str
    candidate_pool_size: int
    dataset_source: str
    dataset_split: str
    dataset_revision: str | None
    index_origin: CorpusOrigin | None
    corpus_size: int | None
    all_corpus: bool
    truncate_dim: int | None
    min_semantic_similarity: float
    streaming: bool
    storage_precision: str
    binary_prefilter: bool
    binary_rescore_multiplier: int
    calibration_sample_size: int
    cache_compression: str
    cache_compression_level: int
    encode_batch_size: int
    torch_compile: bool
    device: str
    force_rebuild_cache: bool
    overwrite_cache: bool
    cache_overwrite_reason: str | None
    top_k: int


class _LayoutConfigRequired(TypedDict):
    """Layout settings always recorded in build sidecars."""

    spring_iterations: int
    dpi: int
    theme: str


class LayoutConfig(_LayoutConfigRequired, total=False):
    """Layout settings with an optional deterministic layout seed."""

    seed: int


class _BuildConfigRequired(TypedDict):
    """Build options present for every strategy in a sidecar."""

    paper_id_input: str
    paper_id_canonical: str
    seed_id: str
    strategy: str
    max_papers: int
    refresh_paper_cache: bool
    s2_retry_budget: float
    layout: LayoutConfig
    timestamp_included: bool
    exports_requested: list[str]


class BuildConfig(_BuildConfigRequired, total=False):
    """Build options with settings for the active strategy branches."""

    citation: CitationConfig
    hybrid: HybridConfig
    embedding: EmbeddingConfig


class GraphConfigData(TypedDict):
    """Configuration sidecar emitted beside a graph export."""

    schema_version: Literal[1]
    build: BuildConfig
    outputs: dict[str, str]
    metadata: ExportMetadata


GRAPH_PAYLOAD_KIND: Final[Literal["citemesh-graph"]] = "citemesh-graph"
GRAPH_PAYLOAD_SCHEMA_VERSION: Final[Literal[1]] = 1
DASHBOARD_COLLECTION_KIND: Final[Literal["citemesh-dashboard-collection"]] = (
    "citemesh-dashboard-collection"
)
DASHBOARD_COLLECTION_SCHEMA_VERSION: Final[Literal[1]] = 1

__all__ = [
    "ScoreContract",
    "EmbeddingExportMetadata",
    "ExportMetadata",
    "PlotOverlayMetadata",
    "EffectiveNodeMetadata",
    "EnrichedNodePayload",
    "GraphSummary",
    "YearRange",
    "DashboardMetadata",
    "GraphPayloadEdge",
    "DashboardEdge",
    "PortableGraphMetadata",
    "GraphPayloadDashboard",
    "GraphPayload",
    "DashboardPayload",
    "ValidatedPayloadNode",
    "ValidatedPayloadEdge",
    "ValidatedPortableGraphMetadata",
    "ValidatedDashboardMetadata",
    "ValidatedGraphPayloadDashboard",
    "ValidatedGraphPayload",
    "DashboardCollectionResult",
    "DashboardCollectionBundle",
    "DashboardResultEntry",
    "DashboardCollectionPackage",
    "CitationConfig",
    "HybridConfig",
    "EmbeddingConfig",
    "LayoutConfig",
    "BuildConfig",
    "GraphConfigData",
    "GRAPH_PAYLOAD_KIND",
    "GRAPH_PAYLOAD_SCHEMA_VERSION",
    "DASHBOARD_COLLECTION_KIND",
    "DASHBOARD_COLLECTION_SCHEMA_VERSION",
]
