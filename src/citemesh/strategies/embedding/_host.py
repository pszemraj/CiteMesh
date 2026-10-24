"""Structural types required by the embedding builder mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import torch
    from numpy.typing import NDArray
    from sentence_transformers import SentenceTransformer

    from citemesh.core.text_batching import _TextEncoder
    from citemesh.data.embedding_cache import EmbeddingCache
    from citemesh.data.embedding_cache.models import (
        CachedPaperMetadata,
        CacheInputMetadata,
        CorpusOrigin,
        EmbeddingInputMetadata,
    )
    from citemesh.data.embedding_contract import ExistingCacheSelection
    from citemesh.data.model_profiles import (
        DocumentFormatter,
        EmbeddingModelProfile,
        SimilarityFormatter,
    )

    from .builder import EmbeddingRuntimeMetadata


class _EmbeddingBuilderHost(Protocol):
    """State and cross-mixin methods provided by ``EmbeddingGraphBuilder``."""

    model: SentenceTransformer | None
    model_profile: EmbeddingModelProfile
    truncate_dim: int | None
    device: str
    requested_device: str
    model_name: str
    model_revision: str | None
    requested_model_profile: str
    semantic_source: str
    dataset_source: str
    dataset_split: str
    corpus_size: int | None
    max_papers: int
    storage_precision: Literal["int8", "float32"]
    min_semantic_similarity: float
    enable_torch_compile: bool
    encode_batch_size: int
    binary_prefilter: bool
    binary_rescore_multiplier: int
    calibration_sample_size: int

    _requested_truncate_dim: int | None
    _active_model_name: str | None
    _resolved_model_fingerprint: str | None
    _source_dtype_hint: str
    _attention_implementation_hint: str | None
    _document_formatter_fingerprint: str | None
    _similarity_formatter_fingerprint: str | None
    _autocast_dtype: torch.dtype | None
    _autocast_device_type: str | None
    _autocast_enabled: bool
    _encode_model: _TextEncoder | None
    _inner_model_compiled: bool
    _eager_inner_transformer: torch.nn.Module | None
    _compile_status_reason: str | None
    _profile_logged: bool
    _dim_logged: bool
    _runtime_summary_logged: bool
    _tf32_runtime_configured: bool
    _tf32_mode: str
    _last_search_used_binary_prefilter: bool | None
    _cache_selection: ExistingCacheSelection | None
    _selected_cache_handle: EmbeddingCache | None
    _completed_source_revision: str | None
    _index_origin: CorpusOrigin | None
    _embedding_cache: EmbeddingCache | None
    _graph_embedding_cache: EmbeddingCache | None

    @property
    def embedding_cache(self) -> EmbeddingCache: ...

    @property
    def graph_embedding_cache(self) -> EmbeddingCache: ...

    @property
    def compute_dtype(self) -> str: ...

    def _resolve_formatter_fingerprint(
        self,
        *,
        formatter: DocumentFormatter | SimilarityFormatter,
        probe_renderer: Callable[[EmbeddingInputMetadata], str],
    ) -> str: ...

    def _format_retrieval_document_metadata(
        self, metadata: CacheInputMetadata | CachedPaperMetadata
    ) -> str: ...

    def _format_graph_similarity_metadata(
        self, metadata: CacheInputMetadata | CachedPaperMetadata
    ) -> str: ...

    def _resolve_truncate_dim(self, requested_dim: int | None) -> int | None: ...

    def _bind_embedding_cache_to_active_model(self) -> None: ...

    def _ensure_cache_model_fingerprint(
        self, *, representation: str = "retrieval-document-v1"
    ) -> None: ...

    def _cache_model_identity(self) -> str: ...

    def _require_selected_runtime_match(self) -> None: ...

    def _clear_embedding_cache(self, reason: str) -> None: ...

    def _load_model(self) -> None: ...

    def _encode_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> NDArray[np.float32]: ...

    def _get_model_for_encoding(self) -> _TextEncoder: ...

    def _cached_payload_row_count(self) -> int: ...

    def _embedding_runtime_metadata(self) -> EmbeddingRuntimeMetadata: ...

    def _cache_hydrated_for_active_spec(self) -> bool: ...
