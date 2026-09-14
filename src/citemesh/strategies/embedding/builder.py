"""Embedding-based graph building strategy.

This strategy uses semantic similarity from sentence transformers
to find conceptually similar papers without relying on citations.

Owns :class:`EmbeddingGraphBuilder` itself: construction, the cache bindings and
namespaces, and the public ``GraphBuilderStrategy`` surface. The model runtime,
fingerprinting and corpus hydration behavior arrive from the mixins in sibling
modules.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
)

import networkx as nx
import numpy as np

from citemesh._runtime import stderr_isatty
from citemesh.core import EMBEDDING_CONFIG, EMBEDDING_STORAGE_CONFIG, Author, Paper
from citemesh.core.paper_ids import (
    canonicalize_or_none,
    external_ids_from_canonical_paper_id,
    normalize_paper_id,
    recognize_arxiv_identifier,
)
from citemesh.core.paper_ids import (
    is_local_corpus_paper_id as _is_local_corpus_paper_id,
)
from citemesh.core.text_batching import l2_normalize_embeddings
from citemesh.data import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    EmbeddingCache,
    get_cache_dir,
    resolve_embedding_model_profile,
    validate_compression_filter,
)
from citemesh.data.cache import (
    embedding_namespace_has_papers,
    path_exists,
    read_embedding_namespace_schema,
)
from citemesh.data.embedding_cache import (
    EMBEDDING_CACHE_SCHEMA_VERSION,
    CacheSearchResult,
)
from citemesh.progress import progress_iterator
from citemesh.services import get_client
from citemesh.strategies.base import (
    GraphBuilderStrategy,
    build_capped_undirected_graph,
    deterministic_sort_key,
    validate_embedding_vectors,
)
from citemesh.strategies.candidates import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    SEMANTIC_SOURCE_CHOICES,
    IdentityRegistry,
    fetch_candidate_pool,
    paper_embedding_metadata,
    register_aliases,
    resolve_aliases,
    scope_candidate_collection,
)

from . import deps
from .config import (
    CITATION_COUNT_ENRICHMENT_LIMIT,
    DEFAULT_DATASET_SOURCE,
    ENCODE_BATCH_SIZE,
)
from .fingerprint import _FingerprintMixin
from .hydration import _CorpusHydrationMixin
from .model_runtime import _ModelRuntimeMixin
from .records import _query_seed_id
from .runtime import resolve_embedding_device
from .text import (
    _GRAPH_SIMILARITY_REPRESENTATION,
    _RETRIEVAL_DOCUMENT_REPRESENTATION,
    EmbeddingTask,
    format_embedding_metadata,
    format_paper_for_embedding,
)

if TYPE_CHECKING:
    from citemesh.services.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)


def _cache_scan_progress(values: Sequence[Any], description: str) -> Iterable[Any]:
    """Render a progress bar over an embedding-cache scan.

    Injected into :class:`EmbeddingCache` so the storage layer stays free of UI
    imports: the TTY gate and the bar's presentation options live here, with the
    rest of this module's progress handling.

    :param Sequence[Any] values: Items the cache is about to iterate.
    :param str description: Label shown ahead of the bar.
    :return Iterable[Any]: Items from ``values``, unchanged.
    """
    return progress_iterator(
        values,
        description=description,
        unit="papers",
        enabled=stderr_isatty(),
    )


class EmbeddingGraphBuilder(
    _ModelRuntimeMixin,
    _FingerprintMixin,
    _CorpusHydrationMixin,
    GraphBuilderStrategy,
):
    """
    Build similarity graphs using semantic embeddings.

    This strategy:
    - Hydrates a quantized corpus cache from ArXiv metadata
    - Executes cache-native semantic retrieval
    - Combines semantic with temporal/category/author factors
    """

    strategy_name = "embedding"

    def __init__(
        self,
        max_papers: int = 40,
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        model_profile: str = "auto",
        model_revision: str | None = None,
        dataset_split: str = "train",
        corpus_size: int | None = None,
        truncate_dim: int | None = None,
        top_k: int = 4,
        use_streaming: bool = False,
        force_rebuild_cache: bool = False,
        force_rebuild_reason: str | None = None,
        storage_precision: str = EMBEDDING_STORAGE_CONFIG.storage_precision,
        binary_prefilter: bool | None = None,
        binary_rescore_multiplier: int | None = None,
        calibration_sample_size: int = EMBEDDING_STORAGE_CONFIG.calibration_sample_size,
        cache_compression: str = EMBEDDING_STORAGE_CONFIG.compression,
        cache_compression_level: int = EMBEDDING_STORAGE_CONFIG.compression_level,
        encode_batch_size: int = ENCODE_BATCH_SIZE,
        enable_torch_compile: bool = False,
        device: str | None = None,
        semantic_source: str = "candidates",
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
        client: SemanticScholarClient | None = None,
        min_semantic_similarity: float = EMBEDDING_CONFIG.min_semantic_similarity,
        dataset_source: str = DEFAULT_DATASET_SOURCE,
    ):
        """
        Initialize embedding graph builder.

        :param int max_papers: Maximum papers in final graph
        :param str model_name: Sentence transformer model name
        :param str model_profile: Model task/runtime profile override
            (``auto``, ``embeddinggemma``, or ``default``).
        :param Optional[str] model_revision: Optional model revision token for hub-backed models.
        :param str dataset_split: HuggingFace dataset split
        :param Optional[int] corpus_size: Optional cap on papers to embed and cache
            after selecting newest submissions (default ``None`` = full selected split).
        :param Optional[int] truncate_dim: Optional embedding truncation dimension. If ``None``,
            uses profile defaults (e.g. EmbeddingGemma defaults to 512d MRL).
        :param int top_k: Number of most similar neighbors per node
        :param bool use_streaming: Whether to stream the HuggingFace dataset instead of loading it
        :param bool force_rebuild_cache: Whether to force an explicit cache rebuild.
        :param Optional[str] force_rebuild_reason: Optional operator rationale logged
            when ``force_rebuild_cache`` clears the embedding namespace.
        :param str storage_precision: Persistent cache precision (``int8`` or ``float32``).
        :param Optional[bool] binary_prefilter: Whether cache search uses binary
            Hamming prefiltering. When ``None``, defaults to enabled only for
            ``int8`` storage precision.
        :param Optional[int] binary_rescore_multiplier: Candidate oversampling factor
            for binary prefilter search. When ``None``, defaults to configured value
            for ``int8`` and ``1`` for non-int8 precision.
        :param int calibration_sample_size: Calibration sample size used for int8 quantization ranges.
        :param str cache_compression: HDF5 compression filter for embedding datasets.
        :param int cache_compression_level: HDF5 compression level.
        :param int encode_batch_size: Batch size used when encoding text payloads.
        :param bool enable_torch_compile: Whether to enable best-effort inner-model
            ``torch.compile`` optimization for supported profiles.
        :param Optional[str] device: Requested compute device token
            (``auto``/``cuda``/``mps``/``cpu``). ``None`` means ``auto``
            (cuda, then mps, then cpu). Explicit unavailable devices raise.
        :param str semantic_source: Candidate sourcing mode: ``candidates``
            (default; embed S2 seed neighbors only) or ``arxiv-corpus``
            (hydrate a local arXiv corpus).
        :param int candidate_pool_size: Maximum S2 candidate pool size fetched
            in ``candidates`` mode.
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        :param float min_semantic_similarity: Minimum semantic cosine for graph edges.
        :param str dataset_source: HuggingFace dataset repository with arXiv metadata fields.
        """
        normalized_semantic_source = str(semantic_source).strip().lower()
        if normalized_semantic_source not in SEMANTIC_SOURCE_CHOICES:
            formatted = ", ".join(SEMANTIC_SOURCE_CHOICES)
            raise ValueError(f"semantic_source must be one of: {formatted}")
        deps._check_embedding_deps(
            require_corpus=normalized_semantic_source == "arxiv-corpus"
        )
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be at least 1")
        if not 0.0 <= min_semantic_similarity <= 1.0:
            raise ValueError("min_semantic_similarity must be between 0.0 and 1.0")
        self.min_semantic_similarity = float(min_semantic_similarity)
        int8_rewritten_for_candidates = (
            normalized_semantic_source != "arxiv-corpus" and storage_precision == "int8"
        )
        if int8_rewritten_for_candidates:
            # int8 calibration ranges are only computed during corpus hydration;
            # candidate pools are small enough that float32 storage is free.
            logger.debug("Candidate mode does not support int8 storage; using float32.")
            storage_precision = "float32"
            if binary_prefilter is None:
                binary_prefilter = False
            if binary_rescore_multiplier is None:
                binary_rescore_multiplier = 1
        normalized_model_name = self._required_text(model_name, "model_name")
        normalized_dataset_split = self._required_text(dataset_split, "dataset_split")
        normalized_dataset_source = self._required_text(
            dataset_source, "dataset_source"
        )
        corpus_size = self._normalized_corpus_size(corpus_size)
        self._validate_scalar_options(
            storage_precision=storage_precision,
            top_k=top_k,
            binary_rescore_multiplier=binary_rescore_multiplier,
            calibration_sample_size=calibration_sample_size,
            encode_batch_size=encode_batch_size,
            cache_compression=cache_compression,
        )
        super().__init__(max_papers)

        self.semantic_source = normalized_semantic_source
        self.candidate_pool_size = int(candidate_pool_size)
        self.model_name = normalized_model_name
        self.requested_model_profile = str(model_profile).strip().casefold()
        self._requested_truncate_dim = truncate_dim
        normalized_revision = (
            str(model_revision).strip() if model_revision is not None else ""
        )
        self.model_revision = normalized_revision or None
        self.dataset_source = normalized_dataset_source
        self.dataset_split = normalized_dataset_split
        self.corpus_size = corpus_size
        self.storage_precision = str(storage_precision)
        self._bind_binary_search_options(
            storage_precision=storage_precision,
            binary_prefilter=binary_prefilter,
            binary_rescore_multiplier=binary_rescore_multiplier,
            calibration_sample_size=calibration_sample_size,
            int8_rewritten_for_candidates=int8_rewritten_for_candidates,
        )
        self.cache_compression = cache_compression
        self.cache_compression_level = int(cache_compression_level)
        self.encode_batch_size = int(encode_batch_size)
        self.enable_torch_compile = bool(enable_torch_compile)
        self._bind_requested_model_contract(device=device, truncate_dim=truncate_dim)

        self.top_k = top_k
        self.model = None
        self.retrieval_embeddings: dict[str, np.ndarray] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        self.candidate_source_status: dict[str, str] = {}
        self._client = client
        self._active_model_name: str | None = None
        self._embedding_cache: EmbeddingCache | None = None
        self._graph_embedding_cache: EmbeddingCache | None = None
        self._pending_force_rebuild_reason = self._deferred_force_rebuild_reason(
            force_rebuild_cache, force_rebuild_reason
        )
        self.use_streaming = use_streaming
        self._validate_streaming_contract()
        self._reset_model_runtime_state()

    @staticmethod
    def _required_text(value: object, field: str) -> str:
        """Normalize a required string option, rejecting blank input.

        :param object value: Raw option value.
        :param str field: Option name quoted back in the error message.
        :return str: Stripped, non-empty option value.
        :raises ValueError: If the option is empty after stripping.
        """
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{field} must be a non-empty string")
        return normalized

    @staticmethod
    def _normalized_corpus_size(corpus_size: int | None) -> int | None:
        """Coerce the optional corpus-size cap to a positive integer.

        ``bool`` is rejected before ``int`` conversion because ``True`` would
        otherwise silently become a one-paper corpus.

        :param Optional[int] corpus_size: Requested cap, or ``None`` for no cap.
        :return Optional[int]: Parsed cap, or ``None`` when uncapped.
        :raises ValueError: If a cap is provided but is not an integer >= 1.
        """
        if corpus_size is None:
            return None
        if isinstance(corpus_size, bool):
            raise ValueError("corpus_size must be at least 1 when provided")
        try:
            parsed_corpus_size = int(corpus_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("corpus_size must be at least 1 when provided") from exc
        if parsed_corpus_size < 1:
            raise ValueError("corpus_size must be at least 1 when provided")
        return parsed_corpus_size

    @staticmethod
    def _validate_scalar_options(
        *,
        storage_precision: str,
        top_k: int,
        binary_rescore_multiplier: int | None,
        calibration_sample_size: int,
        encode_batch_size: int,
        cache_compression: str,
    ) -> None:
        """Reject out-of-range scalar options before any state is bound.

        :param str storage_precision: Requested persistent cache precision.
        :param int top_k: Requested neighbor count per node.
        :param Optional[int] binary_rescore_multiplier: Requested oversampling factor.
        :param int calibration_sample_size: Requested int8 calibration sample size.
        :param int encode_batch_size: Requested encode batch size.
        :param str cache_compression: Requested HDF5 compression filter.
        :return None: Raises on the first violated constraint.
        :raises ValueError: If any option is outside its supported range.
        """
        if str(storage_precision) not in {"int8", "float32"}:
            raise ValueError("storage_precision must be one of {'float32', 'int8'}")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if binary_rescore_multiplier is not None and binary_rescore_multiplier < 1:
            raise ValueError("binary_rescore_multiplier must be at least 1")
        if calibration_sample_size < 1:
            raise ValueError("calibration_sample_size must be at least 1")
        if encode_batch_size < 1:
            raise ValueError("encode_batch_size must be at least 1")
        validate_compression_filter(cache_compression)

    def _bind_binary_search_options(
        self,
        *,
        storage_precision: str,
        binary_prefilter: bool | None,
        binary_rescore_multiplier: int | None,
        calibration_sample_size: int,
        int8_rewritten_for_candidates: bool,
    ) -> None:
        """Resolve the int8-only search options and reject them for other precisions.

        :param str storage_precision: Effective persistent cache precision.
        :param Optional[bool] binary_prefilter: Requested prefilter toggle, or ``None``.
        :param Optional[int] binary_rescore_multiplier: Requested oversampling factor.
        :param int calibration_sample_size: Requested int8 calibration sample size.
        :param bool int8_rewritten_for_candidates: Whether candidate mode downgraded
            an int8 request to float32.
        :return None: Binds the prefilter, multiplier and calibration attributes.
        :raises ValueError: If an int8-only option was requested for other storage.
        """
        resolved_prefilter = (
            EMBEDDING_STORAGE_CONFIG.binary_prefilter
            if binary_prefilter is None and storage_precision == "int8"
            else False
            if binary_prefilter is None
            else bool(binary_prefilter)
        )
        requested_multiplier = (
            EMBEDDING_STORAGE_CONFIG.binary_rescore_multiplier
            if binary_rescore_multiplier is None and storage_precision == "int8"
            else 1
            if binary_rescore_multiplier is None
            else int(binary_rescore_multiplier)
        )

        def _int8_option_error(option: str) -> ValueError:
            """Name the actual constraint an int8-only option violated.

            After the candidate-mode rewrite the caller may well have passed
            int8; telling them the option "requires int8" would be
            self-contradictory. The real constraint is the semantic source.

            :param str option: CLI spelling of the rejected option.
            :return ValueError: Error naming the violated constraint.
            """
            if int8_rewritten_for_candidates:
                return ValueError(
                    f"{option} with int8 storage requires "
                    "semantic_source='arxiv-corpus' (int8 calibration ranges "
                    "are computed during corpus hydration)."
                )
            return ValueError(f"{option} requires storage_precision='int8'")

        if storage_precision != "int8" and resolved_prefilter:
            raise _int8_option_error("--binary-prefilter")
        if storage_precision != "int8" and requested_multiplier != 1:
            raise _int8_option_error("--binary-rescore-multiplier")
        if storage_precision != "int8" and int(calibration_sample_size) != int(
            EMBEDDING_STORAGE_CONFIG.calibration_sample_size
        ):
            raise _int8_option_error("--calibration-sample-size")
        self.binary_prefilter = bool(resolved_prefilter)
        self.binary_rescore_multiplier = int(requested_multiplier)
        self.calibration_sample_size = int(calibration_sample_size)

    def _bind_requested_model_contract(
        self, *, device: str | None, truncate_dim: int | None
    ) -> None:
        """Resolve the model profile, device and dimension contract for this build.

        This is the construction-time counterpart of :meth:`_bind_model_contract`,
        which rebinds the same contract to whichever checkpoint actually loads.

        :param Optional[str] device: Requested compute device token, or ``None``.
        :param Optional[int] truncate_dim: Requested embedding truncation dimension.
        :return None: Binds the profile, device, formatter fingerprints and hints.
        :raises ValueError: If the requested device is unknown or unavailable.
        """
        self.model_profile = resolve_embedding_model_profile(
            self.model_name,
            self.requested_model_profile,
        )
        # Device must resolve before the attention/dtype hints below: those hints
        # feed the cache namespace computed for EmbeddingCache further down.
        self.requested_device = str(device or "auto").strip().lower()
        self.device = resolve_embedding_device(self.requested_device)
        self._document_formatter_fingerprint = self._resolve_formatter_fingerprint(
            formatter=self.model_profile.document_formatter,
            probe_renderer=self._format_retrieval_document_metadata,
        )
        self._similarity_formatter_fingerprint = self._resolve_formatter_fingerprint(
            formatter=self.model_profile.similarity_formatter,
            probe_renderer=self._format_graph_similarity_metadata,
        )
        self.truncate_dim = self._resolve_truncate_dim(truncate_dim)
        self._source_dtype_hint = self._resolve_source_dtype_hint()
        self._attention_implementation_hint = (
            self._resolve_attention_implementation_hint()
        )

    @staticmethod
    def _deferred_force_rebuild_reason(
        force_rebuild_cache: bool, force_rebuild_reason: str | None
    ) -> str | None:
        """Compose the rebuild rationale logged once the namespace is known.

        The clear itself is deferred: the namespace depends on the model that
        actually loads, so the reason is recorded here and consumed later.

        :param bool force_rebuild_cache: Whether an explicit rebuild was requested.
        :param Optional[str] force_rebuild_reason: Optional operator rationale.
        :return Optional[str]: Reason string, or ``None`` when no rebuild is pending.
        """
        if not force_rebuild_cache:
            return None
        logger.info(
            "Embedding cache rebuild requested; deferring clear until the "
            "runtime-active model namespace is resolved."
        )
        normalized_reason = (
            " ".join(str(force_rebuild_reason).split())
            if force_rebuild_reason is not None
            else ""
        )
        clear_reason = "explicit --force-rebuild-cache request"
        if normalized_reason:
            clear_reason = f"{clear_reason}; user_reason={normalized_reason}"
        return clear_reason

    def _validate_streaming_contract(self) -> None:
        """Reject streaming corpus hydration over a sliced dataset split.

        :return None: Raises when streaming cannot honor the requested slice.
        :raises ValueError: If a sliced split is streamed in corpus mode.
        """
        if (
            self.semantic_source == "arxiv-corpus"
            and self.use_streaming
            and ":" in self.dataset_split
        ):
            raise ValueError(
                "Streaming mode does not support sliced dataset splits such as "
                f"'{self.dataset_split}'. Use --dataset-split train with "
                "--corpus-size to cap runtime, or disable --streaming."
            )

    def _reset_model_runtime_state(self) -> None:
        """Clear the per-load model runtime flags to their unloaded defaults.

        :return None: Resets logging latches, autocast/TF32 state and compile state.
        """
        self._profile_logged = False
        self._dim_logged = False
        self._autocast_dtype: Any | None = None
        self._autocast_device_type: str | None = None
        self._autocast_enabled = False
        self._encode_model: Any | None = None
        self._inner_model_compiled = False
        self._eager_inner_transformer: Any | None = None
        self._compile_status_reason: str | None = None
        self._runtime_summary_logged = False
        self._tf32_runtime_configured = False
        self._tf32_mode = "off"
        self._resolved_model_fingerprint: str | None = None
        self._last_search_used_binary_prefilter: bool | None = None

    @property
    def client(self) -> SemanticScholarClient:
        """Return the injected S2 client or create one when an API call needs it.

        :return SemanticScholarClient: Shared client for Semantic Scholar operations.
        """
        if self._client is None:
            self._client = get_client()
        return self._client

    @client.setter
    def client(self, client: SemanticScholarClient) -> None:
        """Replace the Semantic Scholar client for tests and specialized callers.

        :param SemanticScholarClient client: Client instance to use for S2 operations.
        :return None: Replaces the current client reference.
        """
        self._client = client

    @property
    def compute_dtype(self) -> str:
        """Return the effective compute dtype for the active cache namespace.

        :return str: ``"bfloat16"`` when verified autocast is selected, otherwise
            ``"float32"``.
        """
        return self._source_dtype_hint

    @property
    def embedding_cache(self) -> EmbeddingCache:
        """Return the current cache, creating the requested namespace lazily.

        Actual embedding operations call :meth:`_load_model` first; model
        fallback can therefore rebind this property to the runtime-active
        checkpoint before any vector is read or written.

        :return EmbeddingCache: Current persistent cache namespace.
        """
        if self._embedding_cache is None:
            self._embedding_cache = self._create_embedding_cache(
                self._embedding_cache_namespace()
            )
        return self._embedding_cache

    @embedding_cache.setter
    def embedding_cache(self, cache: EmbeddingCache) -> None:
        """Replace the active cache object for tests and specialized callers.

        :param EmbeddingCache cache: Cache-compatible object to install.
        :return None: Replaces the current cache reference.
        """
        self._embedding_cache = cache

    @property
    def graph_embedding_cache(self) -> EmbeddingCache:
        """Return the float32 cache for symmetric graph-similarity vectors.

        :return EmbeddingCache: Graph-only persistent cache namespace.
        """
        if self._graph_embedding_cache is None:
            self._graph_embedding_cache = self._create_graph_embedding_cache()
        return self._graph_embedding_cache

    @graph_embedding_cache.setter
    def graph_embedding_cache(self, cache: EmbeddingCache) -> None:
        """Replace the graph cache object for tests and specialized callers.

        :param EmbeddingCache cache: Cache-compatible object to install.
        :return None: Replaces the current graph-cache reference.
        """
        self._graph_embedding_cache = cache

    def _create_embedding_cache(
        self,
        namespace: str,
        *,
        storage_precision: str | None = None,
        binary_prefilter: bool | None = None,
        formatter_identity: str | None = None,
    ) -> EmbeddingCache:
        """Construct one cache for the supplied representation namespace.

        :param str namespace: Complete cache partition key.
        :param Optional[str] storage_precision: Representation-specific storage mode.
        :param Optional[bool] binary_prefilter: Representation-specific prefilter mode.
        :param Optional[str] formatter_identity: Representation formatter fingerprint.
        :return EmbeddingCache: Initialized persistent cache.
        """
        resolved_storage = storage_precision or self.storage_precision
        resolved_prefilter = (
            self.binary_prefilter
            if binary_prefilter is None
            else bool(binary_prefilter)
        )
        return EmbeddingCache(
            model_name=namespace,
            progress=_cache_scan_progress,
            storage_precision=resolved_storage,
            binary_prefilter=resolved_prefilter,
            calibration_sample_size=self.calibration_sample_size,
            compression=self.cache_compression,
            compression_level=self.cache_compression_level,
            source_torch_dtype=self._source_dtype_hint,
            text_formatter_fingerprint=(
                formatter_identity or self._document_formatter_fingerprint
            ),
        )

    def _graph_embedding_cache_namespace(self) -> str:
        """Return the complete graph-similarity cache namespace.

        :return str: Namespace for the active graph representation and artifact.
        """
        return self._embedding_cache_namespace(
            representation=_GRAPH_SIMILARITY_REPRESENTATION,
            artifact_identity=self._resolved_model_fingerprint,
            storage_precision="float32",
            formatter_identity=self._similarity_formatter_fingerprint,
        )

    def _create_graph_embedding_cache(
        self, namespace: str | None = None
    ) -> EmbeddingCache:
        """Construct the float32 cache for graph-similarity vectors.

        :param Optional[str] namespace: Precomputed namespace override.
        :return EmbeddingCache: Graph-only persistent cache.
        """
        return self._create_embedding_cache(
            namespace or self._graph_embedding_cache_namespace(),
            storage_precision="float32",
            binary_prefilter=False,
            formatter_identity=self._similarity_formatter_fingerprint,
        )

    def _bind_embedding_cache_to_active_model(self) -> None:
        """Bind persistent state to the checkpoint that actually loaded.

        :return None: Replaces a provisional requested-model cache when fallback
            selected another checkpoint and applies a deferred explicit clear once.
        """
        if self._resolved_model_fingerprint is None:
            # A pre-load cache object is only a cheap discovery handle. Never
            # retain it after model fallback has selected the active checkpoint.
            self._embedding_cache = None
            self._graph_embedding_cache = None
            return

        namespace = self._embedding_cache_namespace(
            artifact_identity=self._resolved_model_fingerprint
        )
        if (
            self._embedding_cache is None
            or str(getattr(self._embedding_cache, "model_name", "")) != namespace
        ):
            self._embedding_cache = self._create_embedding_cache(namespace)
        if self._graph_embedding_cache is not None:
            graph_namespace = self._graph_embedding_cache_namespace()
            if (
                str(getattr(self._graph_embedding_cache, "model_name", ""))
                != graph_namespace
            ):
                self._graph_embedding_cache = self._create_graph_embedding_cache(
                    graph_namespace
                )
        if self._pending_force_rebuild_reason is not None:
            operation_lock = (
                self._embedding_cache.hydration_operation_lock()
                if self.semantic_source == "arxiv-corpus"
                else nullcontext()
            )
            with operation_lock:
                logger.warning(
                    "REBUILDING EMBEDDING CACHE — please hang tight. "
                    "Embeddings will be regenerated automatically; this may take a while. "
                    "No action is needed. Reason: explicit rebuild request."
                )
                self._embedding_cache.clear(reason=self._pending_force_rebuild_reason)
                self.graph_embedding_cache.clear(
                    reason=self._pending_force_rebuild_reason
                )
                self._pending_force_rebuild_reason = None

    def _clear_embedding_cache(self, reason: str) -> None:
        """Clear embedding namespace payload with explicit reason logging.

        :param str reason: Human-readable reason for cache reset.
        :return None: Mutates cache files/metadata in-place.
        """
        normalized_reason = str(reason).strip() or "unspecified"
        self.embedding_cache.clear(reason=normalized_reason)

    def _embedding_runtime_metadata(self) -> dict[str, object]:
        """Return runtime metadata describing effective embedding retrieval behavior.

        :return Dict[str, object]: Runtime metadata payload for downstream export.
        """
        return {
            "binary_prefilter_used": self._last_search_used_binary_prefilter,
            "device": self.device,
            "requested_device": self.requested_device,
            "compute_dtype": self._source_dtype_hint,
            "autocast": self._autocast_enabled,
            "retrieval_representation": (
                f"{EmbeddingTask.RETRIEVAL_QUERY.value}/"
                f"{EmbeddingTask.RETRIEVAL_DOCUMENT.value}"
            ),
            "graph_representation": EmbeddingTask.GRAPH_SIMILARITY.value,
            "model_profile": self.model_profile.schema_token,
        }

    def _cache_model_identity(self) -> str:
        """Return model identity used for cache fingerprinting.

        When model loading falls back to another checkpoint, cache validation
        should follow the active checkpoint identity rather than the requested
        model token.

        :return str: Active model identifier for cache fingerprint checks.
        """
        active_model_name = str(self._active_model_name or "").strip()
        if active_model_name:
            return active_model_name
        return str(self.model_name).strip()

    def _resolve_truncate_dim(self, requested_dim: int | None) -> int | None:
        """Resolve effective embedding dimension from request + model profile defaults.

        :param Optional[int] requested_dim: Requested truncate dimension from caller.
        :return Optional[int]: Effective truncate dimension or ``None`` for full embeddings.
        :raises ValueError: If dimension is invalid for the selected model profile.
        """
        if requested_dim is not None and requested_dim < 1:
            raise ValueError("truncate_dim must be at least 1 when provided")

        available_dims = self.model_profile.available_truncate_dims
        if requested_dim is None:
            return self.model_profile.recommended_truncate_dim

        if available_dims and requested_dim not in available_dims:
            formatted = ", ".join(str(dim) for dim in available_dims)
            raise ValueError(
                f"truncate_dim={requested_dim} is not supported for {self.model_name}. "
                f"Expected one of: {formatted}"
            )
        return requested_dim

    def _embedding_cache_namespace(
        self,
        *,
        representation: str = _RETRIEVAL_DOCUMENT_REPRESENTATION,
        artifact_identity: str | None = None,
        storage_precision: str | None = None,
        formatter_identity: str | None = None,
    ) -> str:
        """Build a cache key for the active model and representation contract.

        :param str representation: Semantic role and schema version of stored vectors.
        :param Optional[str] artifact_identity: Immutable resolved checkpoint identity.
        :param Optional[str] storage_precision: Representation-specific storage mode.
        :param Optional[str] formatter_identity: Representation formatter fingerprint.
        :return str: Namespace key used for embedding cache partitioning.
        """
        resolved_storage = storage_precision or self.storage_precision
        resolved_formatter = formatter_identity or self._document_formatter_fingerprint
        parts = [
            f"model={self._cache_model_identity()}",
            f"revision={self._requested_hf_revision_token()}",
            f"artifact={artifact_identity or 'unresolved'}",
            f"profile={self.model_profile.schema_token}",
            f"representation={representation}",
            "normalization=l2-v1",
        ]
        if self.truncate_dim is not None:
            parts.append(f"truncate_dim={self.truncate_dim}")
        parts.append(f"storage_precision={resolved_storage}")
        # Keep the historical default token so existing default caches remain
        # reachable; the auxiliary index setting does not identify vector rows.
        parts.append(f"binary_prefilter={int(resolved_storage == 'int8')}")
        if resolved_storage == "int8":
            parts.append(f"calibration_sample_size={self.calibration_sample_size}")
        # Compute dtype is provenance, not identity. It is auto-resolved from the
        # active device, and the contract already ignores the other execution
        # choices that shift numerics (attention backend, TF32, torch.compile);
        # binding it forked one corpus into a per-hardware cache for no gain.
        parts.append(f"formatter={resolved_formatter}")
        # Candidate mode gets its own namespace so incremental candidate rows
        # never mix with (and never distort row counts of) corpus hydrations.
        # The corpus namespace stays token-free for legacy cache compatibility.
        if (
            representation == _RETRIEVAL_DOCUMENT_REPRESENTATION
            and self.semantic_source != "arxiv-corpus"
        ):
            parts.append(f"mode={self.semantic_source}")
        return "::".join(parts)

    @scope_candidate_collection
    def collect_papers(
        self,
        seed_id: str,
        *,
        seed_paper: Paper | None = None,
        **kwargs: Any,
    ) -> dict[str, Paper]:
        """
        Collect papers via semantic similarity search.

        :param str seed_id: Seed paper identifier (ArXiv ID or text query)
        :param Optional[Paper] seed_paper: Optional pre-fetched seed paper metadata to
            reuse instead of fetching the seed from Semantic Scholar again.
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Dict[str, Paper]: Dictionary of paper_id -> Paper objects
        """
        papers: dict[str, Paper] = {}
        self.retrieval_embeddings = {}
        self.embeddings = {}
        self.candidate_source_status = {}

        # Load model lazily.
        self._load_model()

        resolved_seed_paper = self._resolve_seed_paper(seed_id, seed_paper)
        papers[resolved_seed_paper.paper_id] = resolved_seed_paper

        seed_identities = IdentityRegistry()
        register_aliases(
            seed_identities,
            resolved_seed_paper.paper_id,
            resolved_seed_paper,
        )
        seed_embedding = self._encode_seed_embedding(resolved_seed_paper)

        if self.semantic_source != "arxiv-corpus":
            self._collect_candidate_pool_papers(
                papers, seed_identities, seed_embedding, resolved_seed_paper
            )
            return papers

        self._collect_corpus_cache_papers(papers, seed_identities, seed_embedding)
        return papers

    def _resolve_seed_paper(self, seed_id: str, seed_paper: Paper | None) -> Paper:
        """Resolve a seed from caller metadata, the corpus cache, S2, or query text.

        Generated corpus identifiers have no meaning to Semantic Scholar, so a
        matching cached metadata row is reopened before external resolution. An
        unresolved identifier is treated as a free-text query, which gets a
        synthetic seed node so the graph still has a root.

        :param str seed_id: Seed paper identifier or free-text query.
        :param Optional[Paper] seed_paper: Pre-fetched seed metadata to reuse.
        :return Paper: Seed paper flagged as the graph seed.
        """
        # Reuse caller-provided seed metadata when available to avoid redundant
        # Semantic Scholar fetches in hybrid mode.
        resolved_seed_paper = seed_paper
        local_only_id = _is_local_corpus_paper_id(seed_id)
        if (
            resolved_seed_paper is None
            and local_only_id
            and self.semantic_source != "arxiv-corpus"
        ):
            raise ValueError(
                f"Local corpus seed ID '{seed_id}' requires "
                "semantic_source='arxiv-corpus'. Re-run the build with "
                "--semantic-source arxiv-corpus and the same embedding settings "
                "used for local search."
            )
        if resolved_seed_paper is None and self.semantic_source == "arxiv-corpus":
            cache = self.embedding_cache
            with cache.hydration_operation_lock():
                cached_source = cache.get_hydrated_dataset_source()
                source_matches = (
                    not cached_source or cached_source == self.dataset_source
                )
                if local_only_id and not source_matches:
                    self._validate_cached_corpus_source(cache)
                metadata = (
                    cache.get_paper_metadata_batch([seed_id]).get(seed_id)
                    if source_matches
                    else None
                )
            if metadata is not None:
                resolved_seed_paper = self._paper_from_cached_metadata(
                    seed_id, metadata
                )
            elif local_only_id:
                raise ValueError(
                    f"Local corpus seed ID '{seed_id}' was not found in the selected "
                    "embedding cache namespace. Re-run local search with the same "
                    "model, profile, revision, dimensions, and cache settings, then "
                    "build that result again."
                )
        if resolved_seed_paper is None:
            resolved_seed_paper = self.client.get_paper(
                seed_id, raise_on_unavailable=True
            )

        if resolved_seed_paper:
            # Found via S2 API
            resolved_seed_paper.is_seed = True
            return resolved_seed_paper

        # Treat as text query
        logger.info(f"Using '{seed_id}' as text query")
        # Create dummy seed paper
        return Paper(
            paper_id=_query_seed_id(seed_id),
            title=seed_id,
            year=None,
            is_seed=True,
        )

    def _validate_cached_corpus_source(self, cache: EmbeddingCache) -> None:
        """Require cached corpus provenance to match the requested dataset.

        :param EmbeddingCache cache: Selected embedding cache namespace.
        :return None: Returns after matching or absent provenance.
        :raises RuntimeError: If the cache records another dataset source.
        """
        cached_source = cache.get_hydrated_dataset_source()
        if cached_source and cached_source != self.dataset_source:
            raise RuntimeError(
                f"Local corpus cache contains {cached_source!r}, but the "
                f"configured dataset source is {self.dataset_source!r}. "
                "Run an embedding or hybrid build with "
                f"--dataset-source {self.dataset_source!r} first."
            )

    def _encode_seed_embedding(self, seed_paper: Paper) -> np.ndarray:
        """Encode the seed in the model's query prompt space and record it.

        :param Paper seed_paper: Resolved seed paper.
        :return np.ndarray: Normalized seed embedding, also stored for graph export.
        """
        logger.debug("Computing seed embedding...")
        formatted_seed_text = format_paper_for_embedding(
            profile=self.model_profile,
            paper=seed_paper,
            task=EmbeddingTask.RETRIEVAL_QUERY,
        )
        seed_embedding = self._encode_texts(
            [formatted_seed_text], show_progress_bar=False
        )[0]
        self.retrieval_embeddings[seed_paper.paper_id] = seed_embedding
        return seed_embedding

    def _collect_candidate_pool_papers(
        self,
        papers: dict[str, Paper],
        seed_identities: IdentityRegistry,
        seed_embedding: np.ndarray,
        seed_paper: Paper,
    ) -> None:
        """Rank a Semantic Scholar candidate pool and admit the closest papers.

        :param Dict[str, Paper] papers: Accumulating selection, seeded with the seed.
        :param IdentityRegistry seed_identities: Alias registry for the seed paper.
        :param np.ndarray seed_embedding: Normalized seed embedding.
        :param Paper seed_paper: Resolved seed paper used to fetch the pool.
        :return None: Extends ``papers`` and ``retrieval_embeddings`` in place.
        """
        logger.debug("Using candidate-pool semantic search...")
        pool_candidates = self._select_candidates_from_pool(seed_embedding, seed_paper)
        for paper_id, paper, embedding in pool_candidates:
            if len(papers) >= self.max_papers:
                break
            if paper_id in papers or resolve_aliases(seed_identities, paper):
                continue
            papers[paper_id] = paper
            self.retrieval_embeddings[paper_id] = embedding

    def _collect_corpus_cache_papers(
        self,
        papers: dict[str, Paper],
        seed_identities: IdentityRegistry,
        seed_embedding: np.ndarray,
    ) -> None:
        """Search the hydrated corpus cache and admit the closest papers.

        :param Dict[str, Paper] papers: Accumulating selection, seeded with the seed.
        :param IdentityRegistry seed_identities: Alias registry for the seed paper.
        :param np.ndarray seed_embedding: Normalized seed embedding.
        :return None: Extends ``papers`` in place and backfills citation counts.
        """
        use_streaming = self.use_streaming
        if use_streaming:
            logger.debug(
                "Using streaming hydration path for cache-native semantic search..."
            )
        else:
            logger.debug("Using cache-native semantic search...")
        candidates = self._select_candidates(
            seed_embedding,
            use_streaming=use_streaming,
        )

        # Convert candidates to Paper objects while respecting max_papers total.
        for paper_id, metadata, embedding in candidates:
            if len(papers) >= self.max_papers:
                break
            if paper_id in papers:
                continue

            paper = self._paper_from_cached_metadata(paper_id, metadata)
            if resolve_aliases(seed_identities, paper):
                continue

            papers[paper_id] = paper
            self.retrieval_embeddings[paper_id] = embedding

        self._update_citation_counts(papers)

    @staticmethod
    def _paper_from_cached_metadata(paper_id: str, metadata: dict) -> Paper:
        """Build a Paper from one cached corpus metadata row.

        :param str paper_id: Canonical cached paper identifier.
        :param Dict metadata: Normalized metadata row persisted alongside vectors.
        :return Paper: Paper carrying the cached bibliographic fields.
        """
        return Paper(
            paper_id=paper_id,
            title=metadata.get("title", "Unknown"),
            year=metadata.get("year"),
            authors=[Author(name=name) for name in metadata.get("authors", [])],
            abstract=metadata.get("abstract", ""),
            venue=metadata.get("venue", ""),
            arxiv_id=metadata.get("arxiv_id", ""),
            doi=metadata.get("doi", ""),
            categories=metadata.get("categories", []),
            citation_count=0,  # ArXiv data lacks citation counts
            is_seed=False,
            is_local_corpus=True,
        )

    def _candidate_pool_budgets(self) -> tuple[int, int, int]:
        """Split the candidate pool size into per-source fetch budgets.

        :return Tuple[int, int, int]: ``(max_references, max_citations,
            max_recommendations)`` budgets.
        """
        pool_size = self.candidate_pool_size
        max_recommendations = min(100, max(1, pool_size // 4))
        remaining = max(0, pool_size - max_recommendations)
        max_references = min(100, (remaining + 2) // 3)
        max_citations = max(0, remaining - max_references)
        return max_references, max_citations, max_recommendations

    def _select_candidates_from_pool(
        self, seed_embedding: np.ndarray, seed_paper: Paper
    ) -> list[tuple[str, Paper, np.ndarray]]:
        """Rank S2 candidate-pool papers by cosine similarity to the seed.

        :param np.ndarray seed_embedding: Normalized seed embedding vector.
        :param Paper seed_paper: Resolved seed paper (S2-backed or query seed).
        :return List[Tuple[str, Paper, np.ndarray]]: Ranked candidate tuples.
        """
        max_references, max_citations, max_recommendations = (
            self._candidate_pool_budgets()
        )
        pool = fetch_candidate_pool(
            self.client,
            seed_paper,
            max_references=max_references,
            max_citations=max_citations,
            max_recommendations=max_recommendations,
        )
        self.candidate_source_status = dict(pool.source_status)
        if not pool.papers:
            logger.warning(
                "Candidate pool for %s is empty; graph will only contain the seed.",
                seed_paper.paper_id,
            )
            return []

        embeddings = self.embed_papers(pool.papers)
        query = np.asarray(seed_embedding, dtype=np.float32)
        scored: list[tuple[float, str, Paper, np.ndarray, int]] = []
        for idx, (paper_id, paper) in enumerate(pool.papers.items()):
            embedding = embeddings.get(paper_id)
            if embedding is None:
                continue
            vector = np.asarray(embedding, dtype=np.float32)
            score = float(np.clip(np.dot(query, vector), -1.0, 1.0))
            scored.append((score, paper_id, paper, vector, idx))
        scored.sort(
            key=lambda item: deterministic_sort_key(
                item[0], item[1], stable_index=item[4]
            )
        )
        logger.info(
            "Candidate semantic search ranked %d of %d pooled papers.",
            len(scored),
            len(pool.papers),
        )
        return [(paper_id, paper, vector) for _, paper_id, paper, vector, _ in scored]

    def embed_papers(self, papers: dict[str, Paper]) -> dict[str, np.ndarray]:
        """Embed retrieval documents through cache, encoding only missing ones.

        :param Dict[str, Paper] papers: Mapping of paper ID to paper payload.
        :return Dict[str, np.ndarray]: Mapping of paper IDs to float32 embeddings.
        """
        if not papers:
            return {}
        self._load_model()
        self._ensure_cache_model_fingerprint()
        metadata_map = {
            paper_id: {
                **paper_embedding_metadata(paper),
                "paper_id": paper_id,
            }
            for paper_id, paper in papers.items()
        }
        embeddings = self.embedding_cache.get_embeddings(
            metadata_map,
            self._get_model_for_encoding(),
            batch_size=self.encode_batch_size,
            show_progress=False,
            text_builder=self._format_retrieval_document_metadata,
        )
        self.retrieval_embeddings.update(embeddings)
        return embeddings

    def _format_retrieval_document_metadata(self, metadata: dict[str, object]) -> str:
        """Format cached metadata for retrieval-document encoding.

        :param Dict[str, object] metadata: Cache metadata including paper identity.
        :return str: Profile-formatted retrieval-document input.
        """
        return format_embedding_metadata(
            profile=self.model_profile,
            metadata=metadata,
            paper_id=metadata.get("paper_id", ""),
            task=EmbeddingTask.RETRIEVAL_DOCUMENT,
        )

    def _format_graph_similarity_metadata(self, metadata: dict[str, object]) -> str:
        """Format cached paper metadata for symmetric graph similarity.

        :param Dict[str, object] metadata: Cache metadata including paper identity.
        :return str: Profile-formatted symmetric similarity input.
        """
        return format_embedding_metadata(
            profile=self.model_profile,
            metadata=metadata,
            paper_id=metadata.get("paper_id", ""),
            task=EmbeddingTask.GRAPH_SIMILARITY,
        )

    def materialize_graph_embeddings(
        self, papers: dict[str, Paper]
    ) -> dict[str, np.ndarray]:
        """Materialize selected papers in a symmetric graph-similarity space.

        :param Dict[str, Paper] papers: Final graph papers by canonical ID.
        :return Dict[str, np.ndarray]: Validated graph-similarity vectors.
        """
        if not papers:
            self.embeddings = {}
            return {}
        self._load_model()
        self._ensure_cache_model_fingerprint(
            representation=_GRAPH_SIMILARITY_REPRESENTATION
        )
        metadata_map = {}
        for paper_id, paper in papers.items():
            metadata = paper_embedding_metadata(paper)
            metadata["paper_id"] = paper_id
            metadata_map[paper_id] = metadata
        embeddings = self.graph_embedding_cache.get_embeddings(
            metadata_map,
            self._get_model_for_encoding(),
            batch_size=self.encode_batch_size,
            show_progress=False,
            text_builder=self._format_graph_similarity_metadata,
        )
        self.embeddings = validate_embedding_vectors(
            papers,
            embeddings,
            context="Graph-similarity embedding",
        )
        return dict(self.embeddings)

    def prepare_graph_scoring(self, papers: dict[str, Paper]) -> None:
        """Populate symmetric vectors immediately before pairwise edge scoring.

        :param Dict[str, Paper] papers: Final selected graph papers.
        :return None: Materializes the graph-similarity cache and vector map.
        """
        self.materialize_graph_embeddings(papers)

    def search_local(self, query: str, top_k: int) -> list[CacheSearchResult]:
        """Semantically search this builder's persistent embedding cache.

        Encodes the free-text query in the model's query prompt space and
        ranks it against every embedding already persisted in the cache
        namespace (candidate-mode vectors accumulated across builds, or a
        hydrated corpus). Purely local except for the one-time model download.

        :param str query: Free-text search query.
        :param int top_k: Number of results to return.
        :return List[CacheSearchResult]: Ranked results with cached metadata.
        :raises ValueError: If the query is empty or ``top_k`` is below 1.
        :raises RuntimeError: If the corpus cache records another dataset source.
        """
        normalized_query = str(query).strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if int(top_k) < 1:
            raise ValueError("top_k must be at least 1")
        cache = self.prepare_embedding_cache()
        query_text = self.model_profile.format_query(normalized_query, {})
        query_embedding = self._encode_texts([query_text])[0]
        operation_lock = (
            cache.hydration_operation_lock()
            if self.semantic_source == "arxiv-corpus"
            else nullcontext()
        )
        with operation_lock:
            if self.semantic_source == "arxiv-corpus":
                # Split and cap describe the rows a build contributed. Searching
                # every cached row keeps sliced builds reachable with default
                # query settings; only dataset provenance must agree here.
                self._validate_cached_corpus_source(cache)
            return cache.search(
                query_embedding=np.asarray(query_embedding, dtype=np.float32),
                top_k=int(top_k),
                binary_prefilter=self.binary_prefilter,
                binary_rescore_multiplier=self.binary_rescore_multiplier,
            )

    def prepare_embedding_cache(self) -> EmbeddingCache:
        """Resolve the runtime-active model and its exact persistent namespace.

        :return EmbeddingCache: Artifact-bound cache ready for safe access.
        """
        self._load_model()
        self._ensure_cache_model_fingerprint()
        return self.embedding_cache

    def has_persistent_embedding_artifacts(self) -> bool:
        """Return whether a potentially searchable vector cache exists.

        This cheap probe lets auto local-search avoid loading a model when the
        user has never built an embedding cache. Exact namespace selection still
        occurs before any vector is searched.

        :return bool: Whether at least one non-empty HDF5/metadata pair may
            contain searchable papers.
        """
        cache_directory = (
            self._embedding_cache.h5_path.parent
            if self._embedding_cache is not None
            else get_cache_dir("embeddings", create=False)
        )
        for h5_path in cache_directory.glob("embeddings_*.h5"):
            if not path_exists(h5_path) or h5_path.stat().st_size == 0:
                continue
            namespace_id = h5_path.name.removeprefix("embeddings_").removesuffix(".h5")
            db_path = cache_directory / f"metadata_{namespace_id}.db"
            if not path_exists(db_path):
                continue
            has_papers = embedding_namespace_has_papers(db_path)
            if has_papers is False:
                continue
            schema_version = read_embedding_namespace_schema(db_path)
            if schema_version == "3" and EMBEDDING_CACHE_SCHEMA_VERSION == 4:
                continue
            # Unreadable metadata or a non-empty malformed HDF5 file is not an
            # empty-cache signal. Let full cache preparation surface that error.
            return True
        return False

    def _select_candidates(
        self, seed_embedding: np.ndarray, use_streaming: bool
    ) -> list[tuple[str, dict, np.ndarray]]:
        """Select candidates after hydrating cache with a specific loading mode.

        :param np.ndarray seed_embedding: Normalized seed embedding vector.
        :param bool use_streaming: Whether hydration should stream the dataset.
        :return List[Tuple[str, Dict, np.ndarray]]: Candidate tuples sorted by similarity.
        """
        self._ensure_cache_model_fingerprint()
        with self.embedding_cache.hydration_operation_lock():
            self._ensure_cache_hydrated(use_streaming=use_streaming)
            return self._search_cache_candidates(seed_embedding)

    @staticmethod
    def _citation_enrichment_identifier(paper_id: str, paper: Paper) -> str | None:
        """Resolve an S2-compatible identifier for citation-count enrichment.

        Corpus rows may use arbitrary source-local primary keys. Their explicit
        arXiv or DOI metadata remains a valid lookup route, while a source key
        with no recognized external identity must stay local.

        :param str paper_id: Graph/cache primary identifier.
        :param Paper paper: Paper metadata carrying corpus provenance and aliases.
        :return Optional[str]: Identifier safe to send to S2, or ``None``.
        """
        normalized_id = str(paper_id).strip()
        if normalized_id.startswith("query:"):
            return None
        if not paper.is_local_corpus:
            return None if _is_local_corpus_paper_id(normalized_id) else normalized_id

        arxiv_id = recognize_arxiv_identifier(paper.arxiv_id, allow_bare=True)
        if arxiv_id:
            return arxiv_id
        raw_doi = str(paper.doi or "").strip()
        canonical_doi = canonicalize_or_none(f"doi:{raw_doi}") if raw_doi else None
        _, doi = external_ids_from_canonical_paper_id(canonical_doi or "")
        if doi:
            return doi
        canonical_primary = canonicalize_or_none(normalized_id) or normalized_id
        primary_arxiv = recognize_arxiv_identifier(canonical_primary, allow_bare=True)
        if primary_arxiv:
            return primary_arxiv
        if re.fullmatch(r"(?:s2:)?[0-9a-f]{40}", normalized_id, flags=re.IGNORECASE):
            return normalized_id
        _, primary_doi = external_ids_from_canonical_paper_id(canonical_primary)
        if primary_doi:
            return primary_doi.lower()
        return None

    def _update_citation_counts(self, papers: dict[str, Paper]) -> None:
        """
        Enrich top semantic candidates with citation counts from Semantic Scholar.

        :param Dict[str, Paper] papers: Dictionary of collected papers (including seed)
        """
        targets: list[tuple[str, Paper, str]] = []
        for paper_id, paper in papers.items():
            if paper.is_seed:
                continue
            lookup_id = self._citation_enrichment_identifier(paper_id, paper)
            if lookup_id is None:
                continue
            targets.append((paper_id, paper, lookup_id))
            if len(targets) >= CITATION_COUNT_ENRICHMENT_LIMIT:
                break

        if not targets:
            return

        logger.info(
            "Fetching citation counts from Semantic Scholar for up to %d papers...",
            len(targets),
        )

        from citemesh.services import SemanticScholarRequestError

        try:
            batch_results = self.client.get_papers(
                [lookup_id for _paper_id, _paper, lookup_id in targets]
            )
        except SemanticScholarRequestError as exc:
            logger.warning(
                "Citation-count enrichment was rejected; continuing with "
                "existing citation counts: %s",
                exc,
            )
            return
        for _paper_id, paper, lookup_id in targets:
            batch_paper = batch_results.get(normalize_paper_id(lookup_id))
            if batch_paper is not None:
                paper.citation_count = batch_paper.citation_count

    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        """
        Compute multi-factor similarity.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :return float: Combined similarity score (0.0 to 1.0)
        """
        # Semantic similarity from embeddings
        if paper1.paper_id in self.embeddings and paper2.paper_id in self.embeddings:
            emb1 = l2_normalize_embeddings(self.embeddings[paper1.paper_id])
            emb2 = l2_normalize_embeddings(self.embeddings[paper2.paper_id])
            semantic_sim = float(np.clip(np.dot(emb1, emb2), -1.0, 1.0))
        else:
            semantic_sim = 0.0

        if semantic_sim < self.min_semantic_similarity:
            return 0.0

        # Temporal factor
        temporal_factor = self.temporal_similarity(paper1, paper2)

        # Category overlap
        category_overlap = paper1.category_overlap(paper2)

        # Author collaboration
        author_factor = (
            EMBEDDING_CONFIG.shared_author_bonus
            if paper1.shares_authors_with(paper2)
            else 1.0
        )

        # Combined similarity
        similarity = (
            EMBEDDING_CONFIG.semantic_weight * semantic_sim
            + EMBEDDING_CONFIG.temporal_weight * temporal_factor
            + EMBEDDING_CONFIG.category_weight * category_overlap
        ) * author_factor

        return min(similarity, 1.0)  # Cap at 1.0

    def should_create_edge(
        self, paper1: Paper, paper2: Paper, similarity: float
    ) -> bool:
        """
        Create edges using top-k strategy.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :param float similarity: Computed similarity
        :return bool: True if edge should be created
        """
        # For embedding strategy, we'll compute top-k after all similarities
        # For now, return True for all non-zero similarities
        # The build_graph method will filter to top-k
        return similarity > 0.1

    def build_graph(self, seed_id: str, **kwargs: Any) -> tuple[nx.Graph, str]:
        """
        Build graph with top-k edge selection.

        :param str seed_id: Seed paper identifier
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Tuple[nx.Graph, str]: Tuple of (NetworkX graph, seed paper ID)
        """
        # Use base class to collect papers and create nodes
        graph, actual_seed_id = super().build_graph(seed_id, **kwargs)
        graph.graph["embedding_runtime"] = self._embedding_runtime_metadata()
        graph.graph["candidate_source_status"] = dict(
            sorted(self.candidate_source_status.items())
        )
        # Enforce a strict per-node top-k cap by greedily keeping strongest edges.
        filtered_graph = build_capped_undirected_graph(
            graph, self.top_k, seed_id=actual_seed_id
        )

        logger.info(
            "Graph complete: %s nodes, %s edges",
            filtered_graph.number_of_nodes(),
            filtered_graph.number_of_edges(),
        )

        return filtered_graph, actual_seed_id
