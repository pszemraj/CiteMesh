"""
Embedding-based graph building strategy.

This strategy uses semantic similarity from sentence transformers
to find conceptually similar papers without relying on citations.
"""

import logging
import re
import sys
from contextlib import nullcontext
from hashlib import sha1
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from citemesh.core import EMBEDDING_CONFIG, EMBEDDING_STORAGE_CONFIG, Author, Paper
from citemesh.data import EmbeddingCache, get_embedding_model_profile
from citemesh.services import SemanticScholarClient, get_client
from citemesh.strategies.base import (
    GraphBuilderStrategy,
    deterministic_sort_key,
    select_capped_undirected_edges,
)

logger = logging.getLogger(__name__)


def _check_embedding_deps() -> None:
    """Verify embedding dependencies are installed."""
    missing: list[str] = []

    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        missing.append("sentence-transformers")

    try:
        import datasets  # noqa: F401
    except ImportError:
        missing.append("datasets")

    if missing:
        raise ImportError(
            f"Embedding strategy requires: {', '.join(missing)}. "
            f"Install with: pip install citemesh[embeddings]"
        )


STREAMING_BATCH_SIZE = 32
CANDIDATE_MULTIPLIER = 4
ARXIV_DATASET_CANDIDATES = (
    "librarian-bots/arxiv-metadata-snapshot",
    "CShorten/ML-ArXiv-Papers",
    "gfissore/arxiv-abstracts-2021",
)
ARXIV_IDENTIFIER_PATTERN = re.compile(
    r"^(?:arxiv:)?((?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z\-]+)?/\d{7})(?:v\d+)?)$",
    re.IGNORECASE,
)


def _canonicalize_embedding_paper_id(raw_id: Any) -> str:
    """Canonicalize arXiv-like identifiers for downstream lookups.

    :param Any raw_id: Raw dataset record identifier.
    :return str: Canonicalized identifier.
    """
    text = str(raw_id).strip() if raw_id is not None else ""
    if not text:
        return ""

    if text.startswith("arxiv_"):
        return text

    match = ARXIV_IDENTIFIER_PATTERN.match(text)
    if not match:
        return text

    normalized = re.sub(r"v\d+$", "", match.group(1), flags=re.IGNORECASE)
    return f"arxiv:{normalized}"


def _query_seed_id(query_text: str) -> str:
    """Build deterministic query-mode seed node identifier.

    :param str query_text: Raw user query text.
    :return str: Stable hashed query seed identifier.
    """
    digest = sha1(query_text.encode("utf-8")).hexdigest()[:8]
    return f"query:{digest}"


class _AutocastEncodeProxy:
    """Wrap model encode calls in a precision context manager."""

    def __init__(self, model: Any, context_factory: Callable[[], Any]):
        """Create a model proxy for encode-time autocast.

        :param Any model: Wrapped model object exposing ``encode``.
        :param Callable[[], Any] context_factory: Callable returning a context manager.
        """
        self._model = model
        self._context_factory = context_factory

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        """Run ``encode`` within the configured context manager.

        :param Any args: Positional arguments forwarded to ``encode``.
        :param Any kwargs: Keyword arguments forwarded to ``encode``.
        :return Any: Model ``encode`` return value.
        """
        with self._context_factory():
            return self._model.encode(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped model.

        :param str name: Attribute name.
        :return Any: Delegated attribute value.
        """
        return getattr(self._model, name)


def _parse_year(paper: Dict[str, Any]) -> Optional[int]:
    """Extract publication year from dataset metadata.

    :param Dict[str, Any] paper: Raw dataset record.
    :return Optional[int]: Parsed year or ``None`` if missing/invalid.
    """
    if paper.get("year"):
        try:
            return int(paper["year"])
        except (TypeError, ValueError):
            pass

    update_date = paper.get("update_date")
    if update_date:
        try:
            return int(str(update_date)[:4])
        except (TypeError, ValueError):
            pass

    return None


def _parse_authors(authors_data: Any) -> List[str]:
    """Normalize author metadata to a list of names.

    :param Any authors_data: Raw ``authors`` field from dataset.
    :return List[str]: Author names.
    """
    if isinstance(authors_data, str):
        return [name.strip() for name in authors_data.split(",") if name.strip()]

    if isinstance(authors_data, list):
        author_names: List[str] = []
        for author in authors_data:
            if isinstance(author, str):
                normalized = author.strip()
                if normalized:
                    author_names.append(normalized)
                continue

            if isinstance(author, dict):
                name = author.get("name")
                if isinstance(name, str):
                    normalized = name.strip()
                    if normalized:
                        author_names.append(normalized)
        return author_names

    return []


def _parse_categories(categories_data: Any) -> List[str]:
    """Normalize category metadata to a list of arXiv category codes.

    :param Any categories_data: Raw ``categories`` field from dataset.
    :return List[str]: Category code list.
    """
    if isinstance(categories_data, str):
        normalized = categories_data.replace(",", " ")
        return [category.strip() for category in normalized.split() if category.strip()]

    if isinstance(categories_data, list):
        categories: List[str] = []
        for raw_value in categories_data:
            if isinstance(raw_value, str):
                normalized = raw_value.replace(",", " ")
                categories.extend(
                    [
                        category.strip()
                        for category in normalized.split()
                        if category.strip()
                    ]
                )
        return categories

    return []


def _extract_dataset_paper_metadata(paper: Dict[str, Any], fallback_index: int) -> Dict:
    """Normalize a raw dataset record to embedding metadata fields.

    :param Dict[str, Any] paper: Raw dataset record.
    :param int fallback_index: Index used for synthetic IDs when missing.
    :return Dict: Normalized metadata used by embedding selection.
    """
    raw_paper_id = (
        paper.get("id")
        or paper.get("paper_id")
        or paper.get("paperId")
        or f"arxiv_{fallback_index}"
    )
    paper_id = _canonicalize_embedding_paper_id(raw_paper_id)
    title = paper.get("title", "Unknown")
    if not isinstance(title, str) or not title.strip():
        title = "Unknown"
    abstract = paper.get("abstract", paper.get("summary", ""))
    if not isinstance(abstract, str):
        abstract = ""

    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "year": _parse_year(paper),
        "authors": _parse_authors(paper.get("authors", [])),
        "categories": _parse_categories(paper.get("categories", [])),
    }


class EmbeddingGraphBuilder(GraphBuilderStrategy):
    """
    Build similarity graphs using semantic embeddings.

    This strategy:
    - Hydrates a quantized corpus cache from ArXiv metadata
    - Executes cache-native semantic retrieval
    - Combines semantic with temporal/category/author factors
    """

    def __init__(
        self,
        max_papers: int = 40,
        model_name: str = "google/embeddinggemma-300m",
        dataset_split: str = "train",  # Full snapshot split; use corpus_size to bound runtime.
        corpus_size: Optional[int] = 50000,
        truncate_dim: Optional[int] = None,
        top_k: int = 2,
        random_seed: Optional[int] = None,
        use_streaming: bool = False,
        force_rebuild_cache: bool = False,
        storage_precision: str = EMBEDDING_STORAGE_CONFIG.storage_precision,
        binary_prefilter: bool = EMBEDDING_STORAGE_CONFIG.binary_prefilter,
        binary_rescore_multiplier: int = EMBEDDING_STORAGE_CONFIG.binary_rescore_multiplier,
        calibration_sample_size: int = EMBEDDING_STORAGE_CONFIG.calibration_sample_size,
        cache_compression: str = EMBEDDING_STORAGE_CONFIG.compression,
        cache_compression_level: int = EMBEDDING_STORAGE_CONFIG.compression_level,
        client: Optional[SemanticScholarClient] = None,
    ):
        """
        Initialize embedding graph builder.

        :param int max_papers: Maximum papers in final graph
        :param str model_name: Sentence transformer model name
        :param str dataset_split: HuggingFace dataset split
        :param Optional[int] corpus_size: Maximum papers to load from corpus (``None`` = all in split)
        :param Optional[int] truncate_dim: Optional embedding truncation dimension. If ``None``,
            uses profile defaults (e.g. EmbeddingGemma defaults to 256d MRL).
        :param int top_k: Number of most similar neighbors per node
        :param Optional[int] random_seed: Random seed for reproducibility
        :param bool use_streaming: Whether to stream the HuggingFace dataset instead of loading it
        :param bool force_rebuild_cache: Whether to force an explicit cache rebuild.
        :param str storage_precision: Persistent cache precision (``int8``, ``float16``, ``float32``).
        :param bool binary_prefilter: Whether cache search uses binary Hamming prefiltering.
        :param int binary_rescore_multiplier: Candidate oversampling factor for binary prefilter search.
        :param int calibration_sample_size: Calibration sample size used for int8 quantization ranges.
        :param str cache_compression: HDF5 compression filter for embedding datasets.
        :param int cache_compression_level: HDF5 compression level.
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        """
        _check_embedding_deps()
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if binary_rescore_multiplier < 1:
            raise ValueError("binary_rescore_multiplier must be at least 1")
        if calibration_sample_size < 1:
            raise ValueError("calibration_sample_size must be at least 1")
        super().__init__(max_papers, random_seed)
        self.model_name = model_name
        self.dataset_split = dataset_split
        self.corpus_size = corpus_size
        self.storage_precision = storage_precision
        self.binary_prefilter = bool(binary_prefilter)
        self.binary_rescore_multiplier = int(binary_rescore_multiplier)
        self.calibration_sample_size = int(calibration_sample_size)
        self.cache_compression = cache_compression
        self.cache_compression_level = int(cache_compression_level)
        self.model_profile = get_embedding_model_profile(model_name)
        self.truncate_dim = self._resolve_truncate_dim(truncate_dim)
        self._source_dtype_hint = self._resolve_source_dtype_hint()
        self.top_k = top_k
        self.model = None
        self.embeddings: Dict[str, np.ndarray] = {}
        self.client = client or get_client()
        self.embedding_cache = EmbeddingCache(
            model_name=self._embedding_cache_namespace(),
            storage_precision=self.storage_precision,
            binary_prefilter=self.binary_prefilter,
            calibration_sample_size=self.calibration_sample_size,
            compression=self.cache_compression,
            compression_level=self.cache_compression_level,
            source_torch_dtype=self._source_dtype_hint,
        )
        if force_rebuild_cache:
            logger.info("Forcing embedding cache rebuild as requested.")
            self.embedding_cache.clear()
        self.use_streaming = use_streaming
        if self.use_streaming and ":" in self.dataset_split:
            raise ValueError(
                "Streaming mode does not support sliced dataset splits such as "
                f"'{self.dataset_split}'. Use --dataset-split train with "
                "--corpus-size to cap runtime, or disable --streaming."
            )
        self._profile_logged = False
        self._dim_logged = False
        self._autocast_dtype: Optional[Any] = None
        self._autocast_device_type: Optional[str] = None
        self._autocast_enabled = False
        self._encode_model: Optional[Any] = None
        self._inner_model_compiled = False
        self._runtime_summary_logged = False
        self._tf32_runtime_configured = False
        self._tf32_mode = "off"

    def _resolve_truncate_dim(self, requested_dim: Optional[int]) -> Optional[int]:
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

    def _embedding_cache_namespace(self) -> str:
        """Build cache namespace key for the active model + embedding dimension.

        :return str: Namespace key used for embedding cache partitioning.
        """
        parts = [self.model_name]
        if self.truncate_dim is not None:
            parts.append(f"truncate_dim={self.truncate_dim}")
        parts.append(f"storage_precision={self.storage_precision}")
        parts.append(f"binary_prefilter={int(self.binary_prefilter)}")
        parts.append(f"source_dtype={self._source_dtype_hint}")
        return "::".join(parts)

    def _resolve_source_dtype_hint(self) -> str:
        """Resolve source dtype token used in cache namespace metadata.

        :return str: Source dtype token.
        """
        preferred_dtype = (self.model_profile.preferred_torch_dtype or "").lower()
        if preferred_dtype != "bfloat16":
            return "float32"

        try:
            import torch
        except ImportError:
            return "float32"

        if not torch.cuda.is_available():
            return "float32"
        if not bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
            return "float32"
        return "bfloat16"

    def _log_dimension_policy(self) -> None:
        """Emit one-time debug log for active embedding dimensionality."""
        if self._dim_logged:
            return

        available_dims = self.model_profile.available_truncate_dims
        if available_dims:
            selected_dim = (
                self.truncate_dim
                if self.truncate_dim is not None
                else available_dims[0]
            )
            available_text = ", ".join(f"{dim}d" for dim in available_dims)
            recommended_dim = self.model_profile.recommended_truncate_dim
            if recommended_dim is not None:
                logger.debug(
                    "%s embedding dimension: using %sd (recommended: %sd; available: %s).",
                    self.model_name,
                    selected_dim,
                    recommended_dim,
                    available_text,
                )
            else:
                logger.debug(
                    "%s embedding dimension: using %sd (available: %s).",
                    self.model_name,
                    selected_dim,
                    available_text,
                )
            self._dim_logged = True
            return

        if self.truncate_dim is not None:
            logger.debug(
                "%s embedding dimension: using truncate_dim=%sd.",
                self.model_name,
                self.truncate_dim,
            )
            self._dim_logged = True

    def _effective_embedding_dim(self) -> Optional[int]:
        """Return effective embedding dimension used by this builder.

        :return Optional[int]: Active embedding dimension, or ``None`` for full model output.
        """
        if self.truncate_dim is not None:
            return self.truncate_dim
        available_dims = self.model_profile.available_truncate_dims
        if available_dims:
            return available_dims[0]
        return None

    def _reset_precision_runtime(self) -> None:
        """Clear runtime precision/autocast state."""
        self._autocast_dtype = None
        self._autocast_device_type = None
        self._autocast_enabled = False
        self._encode_model = None

    def _resolve_model_kwargs(self) -> Dict[str, Any]:
        """Compute SentenceTransformer kwargs for model precision policy.

        :return Dict[str, Any]: ``SentenceTransformer`` constructor kwargs.
        """
        self._reset_precision_runtime()
        preferred_dtype = (self.model_profile.preferred_torch_dtype or "").lower()
        if preferred_dtype != "bfloat16":
            self._source_dtype_hint = "float32"
            return {}

        try:
            import torch
        except ImportError:
            logger.warning(
                "%s prefers bfloat16, but torch is unavailable; using float32.",
                self.model_name,
            )
            self._source_dtype_hint = "float32"
            return {}

        if not torch.cuda.is_available():
            logger.info(
                "%s prefers bfloat16, but CUDA is unavailable; using float32.",
                self.model_name,
            )
            self._source_dtype_hint = "float32"
            return {}

        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if not bf16_supported:
            logger.info(
                "%s prefers bfloat16, but CUDA bfloat16 is unsupported; using float32.",
                self.model_name,
            )
            self._source_dtype_hint = "float32"
            return {}

        self._autocast_dtype = torch.bfloat16
        self._autocast_device_type = "cuda"
        self._autocast_enabled = bool(self.model_profile.use_cuda_autocast)
        self._source_dtype_hint = "bfloat16"

        if self._autocast_enabled:
            logger.info(
                "%s will run with torch_dtype=bfloat16 and CUDA autocast.",
                self.model_name,
            )
        else:
            logger.info("%s will run with torch_dtype=bfloat16.", self.model_name)

        return {"torch_dtype": torch.bfloat16}

    def _autocast_context(self) -> Any:
        """Return autocast context for model encoding.

        :return Any: Active autocast context manager or no-op context.
        """
        if (
            not self._autocast_enabled
            or self._autocast_dtype is None
            or self._autocast_device_type is None
        ):
            return nullcontext()

        try:
            import torch
        except ImportError:
            return nullcontext()

        return torch.autocast(
            device_type=self._autocast_device_type,
            dtype=self._autocast_dtype,
        )

    def _get_model_for_encoding(self) -> Any:
        """Return model object used for embedding encode calls.

        :return Any: Base model or autocast-enabled proxy.
        """
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded.")

        if not self._autocast_enabled:
            return self.model

        if self._encode_model is None:
            self._encode_model = _AutocastEncodeProxy(
                self.model, self._autocast_context
            )

        return self._encode_model

    def _encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode text inputs and return normalized float32 embeddings.

        :param List[str] texts: Text payload(s) to encode.
        :param Optional[int] batch_size: Optional batch size override.
        :param bool show_progress_bar: Whether to display encoding progress.
        :return np.ndarray: Embeddings with shape ``(len(texts), dim)``.
        """
        encode_model = self._get_model_for_encoding()

        encode_kwargs: Dict[str, Any] = {
            "convert_to_tensor": False,
            "normalize_embeddings": True,
            "show_progress_bar": show_progress_bar,
        }
        if batch_size is not None:
            encode_kwargs["batch_size"] = batch_size

        embeddings = encode_model.encode(texts, **encode_kwargs)
        return np.asarray(embeddings, dtype=np.float32)

    def _load_model(self) -> None:
        """Lazy load sentence transformer model.

        :return None: Model is initialized in-place on first access.
        """
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            model_kwargs = self._resolve_model_kwargs()
            st_kwargs: Dict[str, Any] = {}
            if model_kwargs:
                st_kwargs["model_kwargs"] = model_kwargs
            if self.truncate_dim is not None:
                st_kwargs["truncate_dim"] = self.truncate_dim

            if st_kwargs:
                self.model = SentenceTransformer(self.model_name, **st_kwargs)
            else:
                self.model = SentenceTransformer(self.model_name)

            self._configure_tf32_runtime()
            self._maybe_compile_inner_transformer()

            if (
                not self.model_profile.float16_supported
                and not model_kwargs
                and not self.model_profile.preferred_torch_dtype
            ):
                logger.info(
                    "%s does not support float16 activations; using float32.",
                    self.model_name,
                )
            self._log_dimension_policy()
            if self.model_profile.notes and not self._profile_logged:
                logger.debug(self.model_profile.notes)
                self._profile_logged = True
            self._log_runtime_summary()

    def _configure_tf32_runtime(self) -> None:
        """Best-effort TF32 enablement for Ampere+ CUDA devices.

        :return None: Updates runtime backend configuration in-place when available.
        """
        if self._tf32_runtime_configured:
            return
        self._tf32_runtime_configured = True
        self._tf32_mode = "off"

        try:
            import torch
        except ImportError:
            logger.debug(
                "Skipping TF32 config for %s: torch unavailable.", self.model_name
            )
            return

        cuda_module = getattr(torch, "cuda", None)
        cuda_available = getattr(cuda_module, "is_available", None)
        if not callable(cuda_available) or not bool(cuda_available()):
            logger.debug(
                "Skipping TF32 config for %s: CUDA unavailable.", self.model_name
            )
            return

        get_capability = getattr(cuda_module, "get_device_capability", None)
        capability = get_capability(0) if callable(get_capability) else None
        if not isinstance(capability, tuple) or len(capability) < 2:
            self._tf32_mode = "unknown"
            logger.debug(
                "Skipping TF32 config for %s: CUDA capability unavailable.",
                self.model_name,
            )
            return
        major, minor = int(capability[0]), int(capability[1])
        if major < 8:
            logger.debug(
                "Skipping TF32 config for %s: GPU capability %s.%s is pre-Ampere.",
                self.model_name,
                major,
                minor,
            )
            return

        backends = getattr(torch, "backends", None)
        cuda_backends = getattr(backends, "cuda", None)
        matmul_backend = getattr(cuda_backends, "matmul", None)
        cudnn_backend = getattr(backends, "cudnn", None)
        cudnn_conv = getattr(cudnn_backend, "conv", None)

        try:
            if hasattr(matmul_backend, "fp32_precision") and hasattr(
                cudnn_conv, "fp32_precision"
            ):
                matmul_backend.fp32_precision = "tf32"
                cudnn_conv.fp32_precision = "tf32"
                self._tf32_mode = "tf32"
                logger.info("Enabled TF32 kernels for CUDA matmul/conv (Ampere+ GPU).")
                return
        except Exception as exc:
            logger.debug(
                "Failed setting TF32 precision APIs for %s: %s",
                self.model_name,
                exc,
            )

        try:
            if hasattr(matmul_backend, "allow_tf32") and hasattr(
                cudnn_backend, "allow_tf32"
            ):
                matmul_backend.allow_tf32 = True
                cudnn_backend.allow_tf32 = True
                self._tf32_mode = "legacy-tf32"
                logger.info(
                    "Enabled TF32 kernels via legacy backend flags (Ampere+ GPU)."
                )
                return
        except Exception as exc:
            logger.debug(
                "Failed setting legacy TF32 flags for %s: %s",
                self.model_name,
                exc,
            )

        self._tf32_mode = "unsupported"
        logger.debug("TF32 configuration API unavailable for %s.", self.model_name)

    def _log_runtime_summary(self) -> None:
        """Emit concise one-time runtime summary at info level."""
        if self._runtime_summary_logged:
            return

        selected_dim = self._effective_embedding_dim()
        dim_label = "full" if selected_dim is None else f"{selected_dim}d"
        logger.info(
            "%s runtime: dim=%s, compile=%s, tf32=%s.",
            self.model_name,
            dim_label,
            "on" if self._inner_model_compiled else "off",
            self._tf32_mode,
        )
        self._runtime_summary_logged = True

    def _maybe_compile_inner_transformer(self) -> None:
        """Best-effort compile of the wrapped HF model for selected profiles.

        SentenceTransformer itself is not compiled due wrapper incompatibilities.
        For supported profiles (currently EmbeddingGemma), we compile only
        ``model[0].auto_model`` and keep the outer SentenceTransformer intact.

        :return None: Mutates ``self.model`` in place when compilation succeeds.
        """
        if self.model is None or self._inner_model_compiled:
            return

        if not self.model_profile.compile_inner_transformer:
            return

        try:
            import torch
        except ImportError:
            logger.info(
                "%s profile supports inner-model torch.compile, but torch is unavailable.",
                self.model_name,
            )
            return

        compile_fn = getattr(torch, "compile", None)
        if not callable(compile_fn):
            logger.info(
                "%s profile supports inner-model torch.compile, but torch.compile is unavailable.",
                self.model_name,
            )
            return

        try:
            transformer_block = self.model[0]
        except Exception as exc:  # pragma: no cover - defensive for upstream API drift
            logger.warning(
                "Skipping torch.compile for %s: could not access model[0] (%s).",
                self.model_name,
                exc,
            )
            return

        auto_model = getattr(transformer_block, "auto_model", None)
        if auto_model is None:
            logger.warning(
                "Skipping torch.compile for %s: model[0].auto_model is unavailable.",
                self.model_name,
            )
            return

        if auto_model.__class__.__name__ == "OptimizedModule":
            self._inner_model_compiled = True
            return

        try:
            transformer_block.auto_model = compile_fn(auto_model)
        except Exception as exc:
            logger.warning(
                "torch.compile failed for %s inner transformer; continuing without compile: %s",
                self.model_name,
                exc,
            )
            return

        self._inner_model_compiled = True
        logger.debug(
            "Enabled torch.compile for %s inner transformer (model[0].auto_model).",
            self.model_name,
        )

    def collect_papers(self, seed_id: str, **kwargs: Any) -> Dict[str, Paper]:
        """
        Collect papers via semantic similarity search.

        :param str seed_id: Seed paper identifier (ArXiv ID or text query)
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Dict[str, Paper]: Dictionary of paper_id -> Paper objects
        """
        papers: Dict[str, Paper] = {}
        self.embeddings = {}

        # Load model lazily.
        self._load_model()

        # Try to get seed from Semantic Scholar first
        seed_paper = self.client.get_paper(seed_id)

        seed_metadata: Dict[str, str] = {}

        if seed_paper:
            # Found via S2 API
            seed_paper.is_seed = True
            papers[seed_paper.paper_id] = seed_paper
            seed_title = (seed_paper.title or "").strip()
            seed_abstract = (seed_paper.abstract or "").strip()
            pieces = [part for part in (seed_title, seed_abstract) if part]
            seed_text = ". ".join(pieces) if pieces else seed_id
            seed_metadata = {
                "title": seed_title,
                "abstract": seed_abstract,
            }
        else:
            # Treat as text query
            logger.info(f"Using '{seed_id}' as text query")
            seed_text = seed_id
            # Create dummy seed paper
            query_seed = _query_seed_id(seed_id)
            seed_paper = Paper(
                paper_id=query_seed,
                title=seed_id,
                year=None,
                is_seed=True,
            )
            papers[query_seed] = seed_paper
            seed_metadata = {"title": seed_id, "abstract": ""}

        # Compute normalized seed embedding
        logger.info("Computing seed embedding...")
        formatted_seed_text = self.model_profile.format_query(seed_text, seed_metadata)
        seed_embedding = self._encode_texts(
            [formatted_seed_text], show_progress_bar=False
        )[0]
        self.embeddings[seed_paper.paper_id] = seed_embedding

        use_streaming = self.use_streaming

        if use_streaming:
            logger.info(
                "Using streaming hydration path for cache-native semantic search..."
            )
            candidates = self._select_candidates_streaming(seed_embedding)
        else:
            logger.info("Using cache-native semantic search...")
            candidates = self._select_candidates_from_loaded(seed_embedding)

        # Convert candidates to Paper objects while respecting max_papers total.
        for paper_id, metadata, embedding in candidates:
            if len(papers) >= self.max_papers:
                break
            if paper_id in papers:
                continue

            authors = [Author(name=name) for name in metadata.get("authors", [])[:3]]

            paper = Paper(
                paper_id=paper_id,
                title=metadata.get("title", "Unknown"),
                year=metadata.get("year"),
                authors=authors,
                abstract=metadata.get("abstract", ""),
                categories=metadata.get("categories", []),
                citation_count=0,  # ArXiv data lacks citation counts
                is_seed=False,
            )

            papers[paper_id] = paper
            self.embeddings[paper_id] = embedding

        self._update_citation_counts(papers)
        return papers

    def _select_candidates_from_loaded(
        self, seed_embedding: np.ndarray
    ) -> List[Tuple[str, Dict, np.ndarray]]:
        """
        Select top candidates from cache using non-streaming hydration policy.

        :param np.ndarray seed_embedding: Normalized seed embedding vector
        :return List[Tuple[str, Dict, np.ndarray]]: List of (paper_id, metadata, embedding) tuples sorted by similarity
        """
        self._ensure_cache_hydrated(use_streaming=False)
        return self._search_cache_candidates(seed_embedding)

    def _select_candidates_streaming(
        self, seed_embedding: np.ndarray
    ) -> List[Tuple[str, Dict, np.ndarray]]:
        """
        Select top candidates from cache using streaming hydration policy.

        :param np.ndarray seed_embedding: Normalized seed embedding vector
        :return List[Tuple[str, Dict, np.ndarray]]: List of (paper_id, metadata, embedding) tuples sorted by similarity
        """
        self._ensure_cache_hydrated(use_streaming=True)
        return self._search_cache_candidates(seed_embedding)

    def _search_cache_candidates(
        self, seed_embedding: np.ndarray
    ) -> List[Tuple[str, Dict, np.ndarray]]:
        """Run cache-native retrieval and map results to candidate tuples.

        :param np.ndarray seed_embedding: Normalized seed embedding vector.
        :return List[Tuple[str, Dict, np.ndarray]]: Candidate tuples ordered by score.
        """
        top_k = max(self.max_papers * CANDIDATE_MULTIPLIER, self.max_papers)
        search_results = self.embedding_cache.search(
            query_embedding=np.asarray(seed_embedding, dtype=np.float32),
            top_k=top_k,
            binary_prefilter=self.binary_prefilter,
            binary_rescore_multiplier=self.binary_rescore_multiplier,
        )

        scored_candidates = [
            (
                float(result.score),
                str(result.paper_id),
                dict(result.metadata),
                np.asarray(result.embedding, dtype=np.float32),
                idx,
            )
            for idx, result in enumerate(search_results)
        ]
        scored_candidates.sort(
            key=lambda item: deterministic_sort_key(
                item[0], item[1], stable_index=item[4]
            )
        )
        limited = min(top_k, len(scored_candidates))

        return [
            (paper_id, metadata, embedding)
            for _, paper_id, metadata, embedding, _ in scored_candidates[:limited]
        ]

    def _ensure_cache_hydrated(self, use_streaming: bool) -> None:
        """Ensure cache contains hydrated corpus embeddings for current split/cap.

        :param bool use_streaming: Whether to use streaming dataset hydration.
        :return None: Mutates cache state in-place when hydration is required.
        """
        # Warm-cache fast path: avoid dataset/network work when split/cap already match.
        if self.embedding_cache.is_hydrated(self.dataset_split, self.corpus_size):
            return

        dataset_source: Optional[str]
        dataset: Iterable[Dict[str, Any]]

        try:
            dataset_source, dataset = self._load_dataset_for_hydration(
                use_streaming=use_streaming
            )
        except Exception:
            if self.embedding_cache.is_hydrated(self.dataset_split, self.corpus_size):
                logger.info(
                    "Using existing hydrated cache; dataset hydration source could not be "
                    "resolved in current environment."
                )
                return
            raise

        if self.embedding_cache.is_hydrated(
            self.dataset_split,
            self.corpus_size,
            dataset_source=dataset_source,
        ):
            return

        logger.info(
            "Hydrating embedding cache for split=%s corpus_size=%s (streaming=%s).",
            self.dataset_split,
            "all" if self.corpus_size is None else self.corpus_size,
            use_streaming,
        )
        self.embedding_cache.clear()
        self.embedding_cache.mark_hydrated(
            dataset_source=dataset_source,
            dataset_split=self.dataset_split,
            corpus_size=self.corpus_size,
            complete=False,
        )

        hydrated_records = 0
        calibration_records: List[Dict] = []
        calibration_ready = (
            self.storage_precision != "int8"
            or self.embedding_cache.has_calibration_ranges()
        )
        progress_total = self.corpus_size if self.corpus_size else None
        if not use_streaming and self.corpus_size is None:
            try:
                progress_total = len(dataset)
            except TypeError:  # pragma: no cover - defensive for dataset APIs
                progress_total = None

        with tqdm(
            total=progress_total,
            desc=f"Hydrating {dataset_source}",
            unit="papers",
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        ) as progress:
            batch: List[Dict] = []
            for idx, raw_record in enumerate(dataset):
                if self.corpus_size is not None and idx >= self.corpus_size:
                    break

                metadata = self._extract_paper_metadata(raw_record, idx)
                if (
                    not calibration_ready
                    and len(calibration_records) < self.calibration_sample_size
                ):
                    calibration_records.append(metadata)
                    if len(calibration_records) == self.calibration_sample_size:
                        self._initialize_calibration_ranges(calibration_records)
                        calibration_ready = True
                        batch.extend(calibration_records)
                        calibration_records = []
                    progress.update(1)
                    continue

                batch.append(metadata)
                if len(batch) >= STREAMING_BATCH_SIZE:
                    hydrated_records += self._cache_metadata_batch(batch)
                    batch = []
                progress.update(1)

            if calibration_records and not calibration_ready:
                self._initialize_calibration_ranges(calibration_records)
                calibration_ready = True
                batch.extend(calibration_records)
                calibration_records = []

            if batch:
                hydrated_records += self._cache_metadata_batch(batch)

            if progress_total is None:
                progress.set_postfix_str(f"processed {progress.n}")

        if hydrated_records == 0:
            logger.warning(
                "Hydration produced zero records for split=%s corpus_size=%s; "
                "cache remains incomplete.",
                self.dataset_split,
                "all" if self.corpus_size is None else self.corpus_size,
            )
            return

        self.embedding_cache.mark_hydrated(
            dataset_source=dataset_source,
            dataset_split=self.dataset_split,
            corpus_size=self.corpus_size,
            complete=True,
        )

    def _load_dataset_for_hydration(
        self, use_streaming: bool
    ) -> Tuple[str, Iterable[Dict[str, Any]]]:
        """Load first available ArXiv dataset for hydration.

        :param bool use_streaming: Whether to load streaming dataset iterator.
        :return Tuple[str, Iterable[Dict[str, Any]]]: Dataset source name and iterable.
        """
        from datasets import load_dataset

        last_error: Optional[Exception] = None
        for dataset_name in ARXIV_DATASET_CANDIDATES:
            try:
                dataset = load_dataset(
                    dataset_name,
                    split=self.dataset_split,
                    streaming=use_streaming,
                )
            except Exception as exc:  # pragma: no cover - source/network dependent
                last_error = exc
                logger.warning(
                    "Could not load dataset %s for hydration: %s. Trying fallback.",
                    dataset_name,
                    exc,
                )
                continue
            logger.info(
                "Hydration dataset selected: %s (split=%s, streaming=%s).",
                dataset_name,
                self.dataset_split,
                use_streaming,
            )
            return dataset_name, dataset

        if last_error is not None:
            raise last_error
        raise RuntimeError("Could not load any ArXiv dataset for hydration.")

    def _initialize_calibration_ranges(self, records: List[Dict]) -> None:
        """Compute and persist int8 calibration ranges from metadata records.

        :param List[Dict] records: Records used for calibration embedding sample.
        :return None: Persists calibration ranges in cache.
        """
        if self.storage_precision != "int8":
            return
        if self.embedding_cache.has_calibration_ranges():
            return
        if not records:
            return

        sample_texts = [
            self.model_profile.format_document(
                {
                    "title": metadata.get("title", ""),
                    "abstract": metadata.get("abstract", ""),
                }
            )
            for metadata in records
        ]
        sample_embeddings = self._encode_texts(
            sample_texts,
            batch_size=STREAMING_BATCH_SIZE,
            show_progress_bar=False,
        )
        ranges = np.vstack(
            (
                np.min(sample_embeddings, axis=0),
                np.max(sample_embeddings, axis=0),
            )
        )
        self.embedding_cache.set_calibration_ranges(
            ranges=ranges,
            embedding_dim=int(sample_embeddings.shape[1]),
        )

    def _cache_metadata_batch(self, batch: List[Dict]) -> int:
        """Encode/cache a batch of metadata records.

        :param List[Dict] batch: Metadata records including ``paper_id``.
        :return int: Number of paper IDs routed into cache encoding.
        """
        if not batch:
            return 0

        metadata_map: Dict[str, Dict] = {}
        for metadata in batch:
            paper_id = str(metadata.get("paper_id", "")).strip()
            if not paper_id:
                continue
            payload = dict(metadata)
            payload.pop("paper_id", None)
            metadata_map[paper_id] = payload

        if not metadata_map:
            return 0

        self.embedding_cache.get_embeddings(
            metadata_map,
            self._get_model_for_encoding(),
            batch_size=min(STREAMING_BATCH_SIZE, len(metadata_map)),
            show_progress=False,
            text_builder=self.model_profile.format_document,
        )
        return len(metadata_map)

    def _extract_paper_metadata(self, paper: Dict, fallback_index: int) -> Dict:
        """
        Normalize dataset record into metadata dictionary.

        :param Dict paper: Raw dataset record
        :param int fallback_index: Index used to generate ID if missing
        :return Dict: Dictionary with normalized fields
        """
        return _extract_dataset_paper_metadata(paper, fallback_index)

    def _update_citation_counts(self, papers: Dict[str, Paper]) -> None:
        """
        Optionally enrich top papers with citation counts from Semantic Scholar.

        :param Dict[str, Paper] papers: Dictionary of collected papers (including seed)
        """
        logger.info("Fetching citation counts from Semantic Scholar (optional)...")

        targets = [
            (pid, paper)
            for pid, paper in list(papers.items())[:10]
            if not paper.is_seed
            and not (isinstance(pid, str) and pid.startswith("query:"))
            and not (isinstance(pid, str) and pid.startswith("arxiv_"))
        ]

        if not targets:
            return

        progress_enabled = sys.stderr.isatty()
        iterator = (
            tqdm(
                targets,
                desc="Citation metadata",
                unit="papers",
                leave=False,
                dynamic_ncols=True,
            )
            if progress_enabled
            else targets
        )

        for paper_id, paper in iterator:
            try:
                s2_paper = self.client.get_paper(paper_id)
                if s2_paper:
                    paper.citation_count = s2_paper.citation_count
            except Exception as exc:
                logger.warning(f"Could not fetch citation count for {paper_id}: {exc}")

        if progress_enabled:
            iterator.close()

    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        """
        Compute multi-factor similarity.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :return float: Combined similarity score (0.0 to 1.0)
        """
        # Semantic similarity from embeddings
        if paper1.paper_id in self.embeddings and paper2.paper_id in self.embeddings:
            emb1 = self.embeddings[paper1.paper_id]
            emb2 = self.embeddings[paper2.paper_id]
            semantic_sim = float(np.clip(np.dot(emb1, emb2), -1.0, 1.0))
        else:
            semantic_sim = 0.0

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

    def build_graph(self, seed_id: str, **kwargs: Any) -> Tuple[nx.Graph, str]:
        """
        Build graph with top-k edge selection.

        :param str seed_id: Seed paper identifier
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Tuple[nx.Graph, str]: Tuple of (NetworkX graph, seed paper ID)
        """
        # Use base class to collect papers and create nodes
        graph, actual_seed_id = super().build_graph(seed_id, **kwargs)
        # Enforce a strict per-node top-k cap by greedily keeping strongest edges.
        filtered_graph = nx.Graph()
        filtered_graph.add_nodes_from(graph.nodes(data=True))

        for u, v, weight in select_capped_undirected_edges(
            graph.edges(data=True), self.top_k
        ):
            filtered_graph.add_edge(u, v, weight=weight)

        logger.info(
            f"Filtered graph: {filtered_graph.number_of_nodes()} nodes, "
            f"{filtered_graph.number_of_edges()} edges (top-{self.top_k})"
        )

        return filtered_graph, actual_seed_id
