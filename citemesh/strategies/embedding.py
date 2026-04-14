"""
Embedding-based graph building strategy.

This strategy uses semantic similarity from sentence transformers
to find conceptually similar papers without relying on citations.
"""

from __future__ import annotations

import importlib.util
import logging
import random
import re
import warnings
from contextlib import nullcontext
from hashlib import sha1, sha256
from itertools import islice
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from citemesh._runtime import stderr_isatty
from citemesh.core import EMBEDDING_CONFIG, EMBEDDING_STORAGE_CONFIG, Author, Paper
from citemesh.data import (
    DEFAULT_EMBEDDING_MODEL_FALLBACKS,
    DEFAULT_EMBEDDING_MODEL_NAME,
    EmbeddingCache,
    get_embedding_model_profile,
    validate_compression_filter,
)
from citemesh.data.model_profiles import compose_title_abstract_text
from citemesh.paper_ids import external_ids_from_canonical_paper_id, normalize_paper_id
from citemesh.services import get_client
from citemesh.strategies.base import (
    GraphBuilderStrategy,
    build_capped_undirected_graph,
    deterministic_sort_key,
)
from citemesh.text_batching import (
    encode_texts_in_length_buckets,
    l2_normalize_embeddings,
)

if TYPE_CHECKING:
    from citemesh.services.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)
_EMBEDDING_MIN_TORCH_VERSION = (2, 9)


def _parse_torch_major_minor(version: str) -> tuple[int, int]:
    """Parse major/minor tuple from a torch version string.

    :param str version: Raw torch version string.
    :return tuple[int, int]: Parsed ``(major, minor)`` tuple, ``(0, 0)`` on parse miss.
    """
    version_match = re.match(r"^(\d+)\.(\d+)", str(version).strip())
    if version_match:
        return int(version_match.group(1)), int(version_match.group(2))
    return (0, 0)


def _check_embedding_deps() -> None:
    """Verify embedding dependencies are installed."""
    missing: list[str] = []
    torch_module: Any | None = None

    try:
        torch_module = _import_torch()
    except ImportError:
        missing.append("torch")

    try:
        _import_sentence_transformer_class()
    except ImportError:
        missing.append("sentence-transformers")

    try:
        _import_datasets_module()
    except ImportError:
        missing.append("datasets")

    if missing:
        raise ImportError(
            f"Embedding strategy requires: {', '.join(missing)}. "
            f"Install with: pip install citemesh[embeddings]"
        )

    raw_torch_version = str(getattr(torch_module, "__version__", "")).strip()
    torch_version = _parse_torch_major_minor(raw_torch_version)
    if torch_version < _EMBEDDING_MIN_TORCH_VERSION:
        raise ImportError(
            "Embedding strategy requires torch>=2.9.0 (runtime precision policy). "
            f"Detected torch=={raw_torch_version or 'unknown'}."
        )


def _module_available(module_name: str) -> bool:
    """Return whether a Python module can be imported in the current runtime.

    :param str module_name: Absolute module name to probe.
    :return bool: ``True`` when the module exists and is importable.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _import_torch() -> Any:
    """Import and return the ``torch`` module.

    :return Any: Imported ``torch`` module object.
    """
    import torch

    return torch


def _import_sentence_transformer_class() -> Any:
    """Import and return ``SentenceTransformer``.

    :return Any: Imported ``SentenceTransformer`` class.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def _import_datasets_module() -> Any:
    """Import and return the ``datasets`` module.

    :return Any: Imported ``datasets`` module object.
    """
    import datasets

    return datasets


def _import_huggingface_hub_module() -> Any:
    """Import and return the ``huggingface_hub`` module.

    :return Any: Imported ``huggingface_hub`` module object.
    """
    import huggingface_hub

    return huggingface_hub


ENCODE_BATCH_SIZE = 32
HYDRATION_FLUSH_SIZE = 256
CANDIDATE_MULTIPLIER = 4
CITATION_COUNT_ENRICHMENT_LIMIT = 20
CALIBRATION_RESERVOIR_SEED = 0
CALIBRATION_LOWER_PERCENTILE = 0.1
CALIBRATION_UPPER_PERCENTILE = 99.9
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


def _parse_venue(paper: Dict[str, Any]) -> str:
    """Normalize venue/journal metadata from dataset records.

    :param Dict[str, Any] paper: Raw dataset record.
    :return str: Best-effort venue string (empty when unavailable).
    """
    for key in ("venue", "journal_ref", "journal"):
        value = paper.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            name = value.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return ""


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
    arxiv_id, doi = external_ids_from_canonical_paper_id(paper_id)
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
        "venue": _parse_venue(paper),
        "arxiv_id": arxiv_id,
        "doi": doi,
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

    strategy_name = "embedding"

    def __init__(
        self,
        max_papers: int = 40,
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        model_revision: Optional[str] = None,
        dataset_split: str = "train",  # Full snapshot split; use corpus_size to bound runtime.
        corpus_size: Optional[int] = 50000,
        truncate_dim: Optional[int] = None,
        top_k: int = 4,
        use_streaming: bool = False,
        force_rebuild_cache: bool = False,
        force_rebuild_reason: Optional[str] = None,
        storage_precision: str = EMBEDDING_STORAGE_CONFIG.storage_precision,
        binary_prefilter: Optional[bool] = None,
        binary_rescore_multiplier: Optional[int] = None,
        calibration_sample_size: int = EMBEDDING_STORAGE_CONFIG.calibration_sample_size,
        cache_compression: str = EMBEDDING_STORAGE_CONFIG.compression,
        cache_compression_level: int = EMBEDDING_STORAGE_CONFIG.compression_level,
        encode_batch_size: int = ENCODE_BATCH_SIZE,
        enable_torch_compile: bool = False,
        client: Optional[SemanticScholarClient] = None,
    ):
        """
        Initialize embedding graph builder.

        :param int max_papers: Maximum papers in final graph
        :param str model_name: Sentence transformer model name
        :param Optional[str] model_revision: Optional model revision token for hub-backed models.
        :param str dataset_split: HuggingFace dataset split
        :param Optional[int] corpus_size: Maximum papers to load from corpus (``None`` = all in split)
        :param Optional[int] truncate_dim: Optional embedding truncation dimension. If ``None``,
            uses profile defaults (e.g. EmbeddingGemma defaults to 256d MRL).
        :param int top_k: Number of most similar neighbors per node
        :param bool use_streaming: Whether to stream the HuggingFace dataset instead of loading it
        :param bool force_rebuild_cache: Whether to force an explicit cache rebuild.
        :param Optional[str] force_rebuild_reason: Optional operator rationale logged
            when ``force_rebuild_cache`` clears the embedding namespace.
        :param str storage_precision: Persistent cache precision (``int8``, ``float16``, ``float32``).
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
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        """
        _check_embedding_deps()
        normalized_model_name = str(model_name).strip()
        if not normalized_model_name:
            raise ValueError("model_name must be a non-empty string")
        normalized_dataset_split = str(dataset_split).strip()
        if not normalized_dataset_split:
            raise ValueError("dataset_split must be a non-empty string")
        if corpus_size is not None:
            if isinstance(corpus_size, bool):
                raise ValueError("corpus_size must be at least 1 when provided")
            try:
                corpus_size = int(corpus_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "corpus_size must be at least 1 when provided"
                ) from exc
            if corpus_size < 1:
                raise ValueError("corpus_size must be at least 1 when provided")
        if str(storage_precision) not in {"int8", "float16", "float32"}:
            raise ValueError(
                "storage_precision must be one of {'float32', 'float16', 'int8'}"
            )
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if binary_rescore_multiplier is not None and binary_rescore_multiplier < 1:
            raise ValueError("binary_rescore_multiplier must be at least 1")
        if calibration_sample_size < 1:
            raise ValueError("calibration_sample_size must be at least 1")
        if encode_batch_size < 1:
            raise ValueError("encode_batch_size must be at least 1")
        validate_compression_filter(cache_compression)
        super().__init__(max_papers)
        self.model_name = normalized_model_name
        normalized_revision = (
            str(model_revision).strip() if model_revision is not None else ""
        )
        self.model_revision = normalized_revision or None
        self.dataset_split = normalized_dataset_split
        self.corpus_size = corpus_size
        self.storage_precision = str(storage_precision)
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
        if storage_precision != "int8" and resolved_prefilter:
            raise ValueError("--binary-prefilter requires storage_precision='int8'")
        if storage_precision != "int8" and requested_multiplier != 1:
            raise ValueError(
                "--binary-rescore-multiplier requires storage_precision='int8'"
            )
        if storage_precision != "int8" and int(calibration_sample_size) != int(
            EMBEDDING_STORAGE_CONFIG.calibration_sample_size
        ):
            raise ValueError(
                "--calibration-sample-size requires storage_precision='int8'"
            )
        self.binary_prefilter = bool(resolved_prefilter)
        self.binary_rescore_multiplier = int(requested_multiplier)
        self.calibration_sample_size = int(calibration_sample_size)
        self.cache_compression = cache_compression
        self.cache_compression_level = int(cache_compression_level)
        self.encode_batch_size = int(encode_batch_size)
        self.enable_torch_compile = bool(enable_torch_compile)
        self.model_profile = get_embedding_model_profile(self.model_name)
        self._document_formatter_fingerprint = (
            self._resolve_document_formatter_fingerprint()
        )
        self.truncate_dim = self._resolve_truncate_dim(truncate_dim)
        self._attention_implementation_hint = (
            self._resolve_attention_implementation_hint()
        )
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
            text_formatter_fingerprint=self._document_formatter_fingerprint,
        )
        normalized_force_rebuild_reason = (
            " ".join(str(force_rebuild_reason).split())
            if force_rebuild_reason is not None
            else ""
        )
        if force_rebuild_cache:
            logger.info("Forcing embedding cache rebuild as requested.")
            clear_reason = "explicit --force-rebuild-cache request"
            if normalized_force_rebuild_reason:
                clear_reason = (
                    f"{clear_reason}; user_reason={normalized_force_rebuild_reason}"
                )
            self._clear_embedding_cache(clear_reason)
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
        self._compile_status_reason: Optional[str] = None
        self._runtime_summary_logged = False
        self._tf32_runtime_configured = False
        self._tf32_mode = "off"
        self._active_model_name: Optional[str] = None
        self._resolved_model_fingerprint: Optional[str] = None
        self._resolved_offline_fingerprint: Optional[str] = None
        self._last_search_used_binary_prefilter: Optional[bool] = None

    def _clear_embedding_cache(self, reason: str) -> None:
        """Clear embedding namespace payload with explicit reason logging.

        :param str reason: Human-readable reason for cache reset.
        :return None: Mutates cache files/metadata in-place.
        """
        normalized_reason = str(reason).strip() or "unspecified"
        self.embedding_cache.clear(reason=normalized_reason)

    def _embedding_runtime_metadata(self) -> Dict[str, object]:
        """Return runtime metadata describing effective embedding retrieval behavior.

        :return Dict[str, object]: Runtime metadata payload for downstream export.
        """
        return {
            "binary_prefilter_used": self._last_search_used_binary_prefilter,
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
        parts.append(f"binary_prefilter={int(self._cache_binary_prefilter_enabled())}")
        if self.storage_precision == "int8":
            parts.append(f"calibration_sample_size={self.calibration_sample_size}")
        parts.append(f"source_dtype={self._source_dtype_hint}")
        parts.append(f"doc_formatter={self._document_formatter_fingerprint}")
        return "::".join(parts)

    def _resolve_document_formatter_fingerprint(self) -> str:
        """Resolve deterministic formatter fingerprint used by embedding cache.

        :return str: SHA-256 digest of profile formatter probes.
        """
        probes = (
            {"title": "Alpha", "abstract": "Beta"},
            {"title": "Alpha", "abstract": ""},
            {"title": "", "abstract": "Beta"},
            {"title": "  Alpha  ", "abstract": "  Beta  "},
        )
        outputs = [
            self.model_profile.format_document(dict(payload)) for payload in probes
        ]
        payload = "||".join(
            (
                str(self.model_profile.name),
                str(getattr(self.model_profile.document_formatter, "__module__", "")),
                str(
                    getattr(
                        self.model_profile.document_formatter,
                        "__qualname__",
                        getattr(
                            self.model_profile.document_formatter,
                            "__name__",
                            "formatter",
                        ),
                    )
                ),
                *outputs,
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _cache_binary_prefilter_enabled(self) -> bool:
        """Return whether binary-prefilter behavior is active for this cache namespace.

        Only int8 caches can use binary-prefilter indexing, so non-int8 precisions
        always map to ``False`` regardless of the configured flag.

        :return bool: Effective binary-prefilter state for cache partitioning.
        """
        return self.binary_prefilter

    def _resolve_attention_implementation_hint(self) -> Optional[str]:
        """Resolve preferred CUDA attention implementation for torch CUDA runs.

        :return Optional[str]: Attention implementation token or ``None``.
        """
        try:
            torch = _import_torch()
        except ImportError:
            return None

        cuda_module = getattr(torch, "cuda", None)
        cuda_available = getattr(cuda_module, "is_available", None)
        if not callable(cuda_available) or not bool(cuda_available()):
            return None

        preferred_attention = str(
            self.model_profile.cuda_attention_implementation or ""
        ).strip()
        if preferred_attention == "flash_attention_2" and _module_available(
            "flash_attn"
        ):
            return preferred_attention
        return "sdpa"

    def _resolve_source_dtype_hint(self) -> str:
        """Resolve source dtype token used for cache provenance metadata.

        :return str: Source dtype token.
        """
        try:
            torch = _import_torch()
        except ImportError:
            return "float32"

        cuda_module = getattr(torch, "cuda", None)
        cuda_available = getattr(cuda_module, "is_available", None)
        if not callable(cuda_available) or not bool(cuda_available()):
            return "float32"

        preferred_dtype = (self.model_profile.preferred_torch_dtype or "").lower()
        if preferred_dtype == "bfloat16" and bool(
            getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        ):
            return "bfloat16"
        if bool(self.model_profile.float16_supported):
            return "float16"
        return "float32"

    def _resolve_model_fingerprint(self) -> str:
        """Resolve deterministic model fingerprint for cache validity checks.

        :return str: Model fingerprint token.
        :raises RuntimeError: If no strong/weak local fingerprint can be resolved.
        """
        if self._resolved_model_fingerprint is not None:
            return self._resolved_model_fingerprint

        model_identity = self._cache_model_identity()
        resolved_path = Path(model_identity).expanduser()
        if resolved_path.exists():
            fingerprint = f"local-path::{resolved_path.resolve()}"
            self._resolved_model_fingerprint = fingerprint
            return fingerprint

        model_id = model_identity
        if "/" not in model_id:
            revision_token = self.model_revision or "default"
            fingerprint = f"model-alias::{model_id}::revision={revision_token}"
            self._resolved_model_fingerprint = fingerprint
            return fingerprint

        requested_revision = self._requested_hf_revision_token()
        resolved_sha = ""
        resolution_error: Optional[Exception] = None

        try:
            huggingface_hub = _import_huggingface_hub_module()
            model_info = huggingface_hub.HfApi().model_info(
                repo_id=model_id,
                revision=requested_revision,
            )
            resolved_sha = str(getattr(model_info, "sha", "") or "").strip()
        except Exception as exc:
            resolution_error = exc

        if not resolved_sha:
            local_snapshot_sha = self._resolve_local_hf_snapshot_sha(
                model_id=model_id,
                requested_revision=requested_revision,
            )
            if local_snapshot_sha is not None:
                resolved_sha = local_snapshot_sha

        if not resolved_sha:
            local_artifact_fingerprint = self._resolve_local_hf_artifact_fingerprint(
                model_id=model_id,
                requested_revision=requested_revision,
            )
            if local_artifact_fingerprint is not None:
                logger.warning(
                    "Could not resolve Hugging Face commit SHA for %s (revision=%s). "
                    "Using local artifact fingerprint from config.json + model.safetensors; "
                    "assuming local artifacts are unchanged.",
                    model_id,
                    requested_revision,
                )
                self._resolved_model_fingerprint = local_artifact_fingerprint
                return local_artifact_fingerprint

        if not resolved_sha:
            raise RuntimeError(
                "Could not resolve Hugging Face commit SHA for "
                f"{model_id!r} (revision={requested_revision!r}); "
                "refusing to use embedding cache without model fingerprint."
            ) from resolution_error

        fingerprint = f"hf::{model_id}::{resolved_sha}"
        self._resolved_model_fingerprint = fingerprint
        return fingerprint

    def _requested_hf_revision_token(self) -> str:
        """Return normalized Hugging Face revision token for current builder config.

        :return str: Normalized requested revision token.
        """
        return (self.model_revision or "main").strip() or "main"

    def _resolve_local_hf_snapshot_sha(
        self, model_id: str, requested_revision: str
    ) -> Optional[str]:
        """Best-effort local SHA resolution from existing HF snapshot cache.

        :param str model_id: Hugging Face repository ID.
        :param str requested_revision: Requested model revision token.
        :return Optional[str]: Locally resolved snapshot SHA, if available.
        """
        try:
            snapshot_download = _import_huggingface_hub_module().snapshot_download
        except Exception:
            return None

        try:
            snapshot_path = Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=requested_revision,
                    local_files_only=True,
                )
            )
        except Exception:
            return None

        parts = snapshot_path.resolve().parts
        for idx, part in enumerate(parts):
            if part != "snapshots":
                continue
            if idx + 1 >= len(parts):
                continue
            candidate = str(parts[idx + 1]).strip()
            if re.fullmatch(r"[0-9a-f]{40}", candidate, flags=re.IGNORECASE):
                logger.warning(
                    "Resolved Hugging Face snapshot SHA for %s (revision=%s) from local cache.",
                    model_id,
                    requested_revision,
                )
                return candidate.lower()
        return None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Return hex SHA-256 digest for a file path.

        :param Path path: File path to hash.
        :return str: Lowercase SHA-256 hex digest.
        """
        digest = sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _resolve_local_hf_artifact_fingerprint(
        self, model_id: str, requested_revision: str
    ) -> Optional[str]:
        """Best-effort local artifact fingerprint using config + weights files.

        :param str model_id: Hugging Face repository ID.
        :param str requested_revision: Requested model revision token.
        :return Optional[str]: Deterministic local artifact fingerprint if available.
        """
        try:
            snapshot_download = _import_huggingface_hub_module().snapshot_download
        except Exception:
            return None

        try:
            snapshot_path = Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=requested_revision,
                    local_files_only=True,
                )
            ).resolve()
        except Exception:
            return None

        config_path = snapshot_path / "config.json"
        weights_path = snapshot_path / "model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            return None

        try:
            config_hash = self._sha256_file(config_path)
            weights_hash = self._sha256_file(weights_path)
        except OSError:
            return None

        return (
            f"hf::{model_id}::revision={requested_revision}"
            f"::config={config_hash}::weights={weights_hash}"
        )

    def _offline_model_fingerprint_fallback(self) -> str:
        """Build a deterministic fallback fingerprint token for offline verification gaps.

        :return str: Deterministic fingerprint proxy derived from model identity hints.
        """
        if self._resolved_offline_fingerprint is not None:
            return self._resolved_offline_fingerprint

        model_id = self._cache_model_identity()
        if "/" not in model_id:
            revision_token = self.model_revision or "default"
            fingerprint = f"model-alias::{model_id}::revision={revision_token}"
            self._resolved_offline_fingerprint = fingerprint
            return fingerprint

        requested_revision = (self.model_revision or "main").strip() or "main"
        local_artifact_fingerprint = self._resolve_local_hf_artifact_fingerprint(
            model_id=model_id,
            requested_revision=requested_revision,
        )
        if local_artifact_fingerprint is not None:
            self._resolved_offline_fingerprint = local_artifact_fingerprint
            return local_artifact_fingerprint

        fingerprint = (
            f"hf::{model_id}::revision={requested_revision}::offline-unverified"
        )
        self._resolved_offline_fingerprint = fingerprint
        return fingerprint

    def _cached_fingerprint_compatible_with_requested_identity(
        self, cached_fingerprint: Optional[str]
    ) -> bool:
        """Return whether cached fingerprint is compatible with active model identity.

        This is used only when strong SHA verification cannot be resolved.

        :param Optional[str] cached_fingerprint: Existing cached fingerprint token.
        :return bool: ``True`` when offline reuse is safe for the requested identity.
        """
        if cached_fingerprint is None:
            return False

        fingerprint = str(cached_fingerprint).strip()
        if not fingerprint:
            return False

        model_id = self._cache_model_identity()
        if "/" not in model_id:
            return fingerprint == self._offline_model_fingerprint_fallback()

        expected_prefix = f"hf::{model_id}::"
        if not fingerprint.startswith(expected_prefix):
            return False

        requested_revision = self._requested_hf_revision_token()
        if fingerprint.endswith("::offline"):
            return fingerprint == self._offline_model_fingerprint_fallback()
        if fingerprint.endswith("::offline-unverified"):
            return fingerprint == self._offline_model_fingerprint_fallback()
        if fingerprint == self._offline_model_fingerprint_fallback():
            return True

        suffix = fingerprint[len(expected_prefix) :]
        if not re.fullmatch(r"[0-9a-f]{40}", suffix, flags=re.IGNORECASE):
            return False

        if re.fullmatch(r"[0-9a-f]{40}", requested_revision, flags=re.IGNORECASE):
            return suffix.lower() == requested_revision.lower()
        return False

    def _ensure_cache_model_fingerprint(self) -> None:
        """Verify cache payload is bound to the active model fingerprint."""
        has_cached_payload = self.embedding_cache.has_cached_payload()
        cached_fingerprint = self.embedding_cache.get_model_fingerprint()

        if has_cached_payload:
            try:
                model_fingerprint = self._resolve_model_fingerprint()
            except Exception as exc:
                fallback_fingerprint = self._offline_model_fingerprint_fallback()
                if self._cached_fingerprint_compatible_with_requested_identity(
                    cached_fingerprint
                ):
                    self._resolved_model_fingerprint = str(cached_fingerprint)
                    logger.warning(
                        "Could not resolve Hugging Face model fingerprint for %s while "
                        "reuse checks are active. Reusing compatible cached fingerprint %s.",
                        self.model_name,
                        cached_fingerprint,
                    )
                    logger.debug(
                        "Skipping model-fingerprint enforcement due resolution failure: %s",
                        exc,
                    )
                    return
                if cached_fingerprint:
                    logger.warning(
                        "Could not resolve Hugging Face model fingerprint for %s and "
                        "cached fingerprint %s is incompatible with requested identity %s. "
                        "Clearing namespace cache to avoid stale embedding reuse.",
                        self.model_name,
                        cached_fingerprint,
                        fallback_fingerprint,
                    )
                    self._clear_embedding_cache(
                        "cached fingerprint incompatible with requested identity "
                        f"(cached={cached_fingerprint}, requested={fallback_fingerprint})"
                    )
                    self.embedding_cache.set_model_fingerprint(fallback_fingerprint)
                    self._resolved_model_fingerprint = fallback_fingerprint
                    logger.debug(
                        "Skipping model-fingerprint enforcement due resolution failure: %s",
                        exc,
                    )
                    return
                self._resolved_model_fingerprint = fallback_fingerprint
                logger.warning(
                    "Could not resolve Hugging Face model fingerprint for %s with no "
                    "stored fingerprint. Reusing cached payload with fallback identity %s "
                    "and recording this assumption for future offline checks.",
                    self.model_name,
                    fallback_fingerprint,
                )
                logger.debug(
                    "Skipping model-fingerprint enforcement due resolution failure: %s",
                    exc,
                )
                self.embedding_cache.set_model_fingerprint(fallback_fingerprint)
                return

            if (
                cached_fingerprint is not None
                and cached_fingerprint != model_fingerprint
            ):
                logger.warning(
                    "Embedding cache model fingerprint mismatch (cached=%s, active=%s). "
                    "Clearing namespace cache.",
                    cached_fingerprint or "missing",
                    model_fingerprint,
                )
                self._clear_embedding_cache(
                    "model fingerprint mismatch "
                    f"(cached={cached_fingerprint or 'missing'}, active={model_fingerprint})"
                )
                self._resolved_model_fingerprint = model_fingerprint
                self.embedding_cache.set_model_fingerprint(model_fingerprint)
                return
            if cached_fingerprint is None:
                self.embedding_cache.set_model_fingerprint(model_fingerprint)

            self._resolved_model_fingerprint = model_fingerprint
            return

        try:
            model_fingerprint = self._resolve_model_fingerprint()
        except Exception as exc:
            fallback_fingerprint = self._offline_model_fingerprint_fallback()
            self._resolved_model_fingerprint = fallback_fingerprint
            logger.warning(
                "Could not resolve Hugging Face model fingerprint for %s while "
                "initializing cache metadata. Using fallback identity %s for "
                "offline initialization.",
                self.model_name,
                fallback_fingerprint,
            )
            logger.debug(
                "Skipping model-fingerprint enforcement due resolution failure: %s",
                exc,
            )
            self.embedding_cache.set_model_fingerprint(fallback_fingerprint)
            return

        if cached_fingerprint is not None and cached_fingerprint != model_fingerprint:
            logger.warning(
                "Embedding cache model fingerprint mismatch (cached=%s, active=%s). "
                "Clearing namespace cache.",
                cached_fingerprint or "missing",
                model_fingerprint,
            )
            # Metadata-only mismatches do not indicate payload corruption here.
            # Keep existing hydration metadata only when payload is absent and proceed
            # with refreshed model identity.
        self._resolved_model_fingerprint = model_fingerprint
        self.embedding_cache.set_model_fingerprint(model_fingerprint)

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
        model_kwargs: Dict[str, Any] = {}
        if self._attention_implementation_hint is not None:
            model_kwargs["attn_implementation"] = self._attention_implementation_hint

        try:
            torch = _import_torch()
        except ImportError:
            self._source_dtype_hint = "float32"
            return model_kwargs

        cuda_module = getattr(torch, "cuda", None)
        cuda_available = getattr(cuda_module, "is_available", None)
        if not callable(cuda_available) or not bool(cuda_available()):
            self._source_dtype_hint = "float32"
            return model_kwargs

        if self._source_dtype_hint == "bfloat16":
            self._autocast_dtype = torch.bfloat16
            model_kwargs["dtype"] = torch.bfloat16
        elif self._source_dtype_hint == "float16":
            float16_dtype = getattr(torch, "float16", None)
            if (
                float16_dtype is None
            ):  # pragma: no cover - defensive for torch API drift
                logger.warning(
                    "%s selected float16 runtime, but torch.float16 is unavailable; using float32.",
                    self.model_name,
                )
                self._source_dtype_hint = "float32"
                return model_kwargs
            self._autocast_dtype = float16_dtype
            model_kwargs["dtype"] = float16_dtype
        else:
            return model_kwargs

        self._autocast_device_type = "cuda"
        self._autocast_enabled = bool(self.model_profile.use_cuda_autocast)

        if self._autocast_enabled:
            logger.debug(
                "%s will run with dtype=%s and CUDA autocast.",
                self.model_name,
                self._source_dtype_hint,
            )
        else:
            logger.debug(
                "%s will run with dtype=%s.",
                self.model_name,
                self._source_dtype_hint,
            )

        return model_kwargs

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
            torch = _import_torch()
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
        effective_batch_size = len(texts) if batch_size is None else int(batch_size)
        if effective_batch_size < 1:
            raise ValueError("batch_size must be at least 1 when provided")

        return encode_texts_in_length_buckets(
            texts,
            batch_size=min(effective_batch_size, len(texts)),
            show_progress_bar=show_progress_bar,
            encode_batch=lambda batch_texts, batch_progress: np.asarray(
                encode_model.encode(
                    batch_texts,
                    batch_size=min(effective_batch_size, len(batch_texts)),
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=batch_progress,
                ),
                dtype=np.float32,
            ),
        )

    def _model_load_candidates(self) -> Tuple[str, ...]:
        """Return ordered candidate model IDs used for lazy model loading.

        Fallbacks are intentionally scoped to default-revision checkpoints so
        explicit revision pins remain deterministic.

        :return Tuple[str, ...]: Ordered model IDs to try.
        """
        requested = str(self.model_name).strip()
        candidates: List[str] = [requested]
        if self.model_revision is not None:
            return tuple(candidates)
        for fallback_model in DEFAULT_EMBEDDING_MODEL_FALLBACKS.get(requested, ()):
            if fallback_model not in candidates:
                candidates.append(fallback_model)
        return tuple(candidates)

    @staticmethod
    def _model_load_error_summary(
        errors: List[Tuple[str, Exception]],
    ) -> str:
        """Build compact model-load failure summary.

        :param List[Tuple[str, Exception]] errors: Ordered candidate failures.
        :return str: Readable failure summary for exception messages.
        """
        return "; ".join(
            f"{model_id}: {type(exc).__name__}: {exc}" for model_id, exc in errors
        )

    def _cache_hydrated_for_active_spec(self) -> bool:
        """Return whether cache is hydrated for active split/corpus selection.

        :return bool: ``True`` when active cache namespace has a matching hydrated payload.
        """
        cached_dataset_source = self.embedding_cache.get_hydrated_dataset_source()
        return self.embedding_cache.is_hydrated(
            self.dataset_split,
            self.corpus_size,
            dataset_source=cached_dataset_source,
        )

    def _should_defer_compile_for_cache_hydration(self) -> bool:
        """Return whether compile should be deferred until cache is hydrated.

        :return bool: ``True`` when runtime should skip compile for current cold-cache run.
        """
        if not self.enable_torch_compile:
            return False
        if not self.model_profile.compile_inner_transformer:
            return False
        try:
            return not self._cache_hydrated_for_active_spec()
        except Exception:
            # Conservative fallback: avoid compile when cache state cannot be
            # validated before hydration.
            return True

    def _load_model(self) -> None:
        """Lazy load sentence transformer model.

        :return None: Model is initialized in-place on first access.
        """
        if self.model is None:
            sentence_transformer_cls = _import_sentence_transformer_class()

            logger.info(f"Loading embedding model: {self.model_name}")
            model_kwargs = self._resolve_model_kwargs()
            st_kwargs: Dict[str, Any] = {}
            if model_kwargs:
                st_kwargs["model_kwargs"] = model_kwargs
            if self.truncate_dim is not None:
                st_kwargs["truncate_dim"] = self.truncate_dim
            if self.model_revision is not None:
                st_kwargs["revision"] = self.model_revision

            load_candidates = self._model_load_candidates()
            model_errors: List[Tuple[str, Exception]] = []
            for idx, candidate_model in enumerate(load_candidates):
                try:
                    if st_kwargs:
                        self.model = sentence_transformer_cls(
                            candidate_model, **st_kwargs
                        )
                    else:
                        self.model = sentence_transformer_cls(candidate_model)
                except Exception as exc:
                    model_errors.append((candidate_model, exc))
                    has_more_candidates = idx + 1 < len(load_candidates)
                    if has_more_candidates:
                        logger.warning(
                            "Failed to load embedding model %s (%s: %s). "
                            "Trying fallback checkpoint...",
                            candidate_model,
                            type(exc).__name__,
                            exc,
                        )
                        continue
                    summary = self._model_load_error_summary(model_errors)
                    raise RuntimeError(
                        "Could not load embedding model from candidate chain "
                        f"{load_candidates}: {summary}"
                    ) from exc

                if candidate_model != self.model_name:
                    logger.info(
                        "Using fallback embedding checkpoint: requested=%s active=%s.",
                        self.model_name,
                        candidate_model,
                    )
                if self._active_model_name != candidate_model:
                    self._resolved_model_fingerprint = None
                    self._resolved_offline_fingerprint = None
                self._active_model_name = candidate_model
                break

            self._configure_tf32_runtime()
            if self._should_defer_compile_for_cache_hydration():
                self._compile_status_reason = (
                    "deferred while hydrating cache; compile resumes on warm-cache runs"
                )
                logger.debug(
                    "Deferring torch.compile for %s until cache hydration completes.",
                    self.model_name,
                )
            else:
                self._maybe_compile_inner_transformer()

            if (
                not self.model_profile.float16_supported
                and not model_kwargs
                and not self.model_profile.preferred_torch_dtype
            ):
                logger.debug(
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
            torch = _import_torch()
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

        torch_version = _parse_torch_major_minor(getattr(torch, "__version__", ""))
        compile_fn = getattr(torch, "compile", None)
        should_use_compile_bridge = (
            self.enable_torch_compile
            and self.model_profile.compile_inner_transformer
            and callable(compile_fn)
            and torch_version in {(2, 9), (2, 10)}
        )
        if should_use_compile_bridge:
            set_matmul_precision = getattr(torch, "set_float32_matmul_precision", None)
            if not callable(set_matmul_precision):
                self._tf32_mode = "unsupported"
                logger.debug(
                    "Skipping TF32 config for %s: compile-safe matmul precision API unavailable.",
                    self.model_name,
                )
                return
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Please use the new API settings to control TF32 behavior.*",
                        category=UserWarning,
                    )
                    set_matmul_precision("high")
                self._tf32_mode = "tf32-matmul-high"
                logger.debug(
                    "Configured compile-safe TF32 matmul precision for %s on torch %s.",
                    self.model_name,
                    getattr(torch, "__version__", "unknown"),
                )
                return
            except Exception as exc:
                self._tf32_mode = "unsupported"
                logger.debug(
                    "Failed setting compile-safe TF32 matmul precision for %s: %s",
                    self.model_name,
                    exc,
                )
                return

        backends = getattr(torch, "backends", None)
        if backends is None or not hasattr(backends, "fp32_precision"):
            self._tf32_mode = "unsupported"
            logger.debug(
                "Skipping TF32 config for %s: torch.backends.fp32_precision unavailable.",
                self.model_name,
            )
            return

        try:
            backends.fp32_precision = "tf32"
            self._tf32_mode = "tf32"
            logger.debug("Enabled TF32 kernels for CUDA matmul/conv (Ampere+ GPU).")
            return
        except Exception as exc:
            self._tf32_mode = "unsupported"
            logger.debug(
                "Failed setting TF32 precision API for %s: %s",
                self.model_name,
                exc,
            )

    def _log_runtime_summary(self) -> None:
        """Emit concise one-time runtime summary at info level."""
        if self._runtime_summary_logged:
            return

        selected_dim = self._effective_embedding_dim()
        dim_label = "full" if selected_dim is None else f"{selected_dim}d"
        compute_dtype_label = self._source_dtype_hint
        if self._autocast_enabled:
            compute_dtype_label = f"{compute_dtype_label}+autocast"
        attention_label = self._attention_implementation_hint or "auto"
        logger.info(
            "%s runtime: dim=%s, compute=%s, attn=%s, output=float32, cache=%s, compile=%s, tf32=%s.",
            self.model_name,
            dim_label,
            compute_dtype_label,
            attention_label,
            self.storage_precision,
            "on" if self._inner_model_compiled else "off",
            self._tf32_mode,
        )
        if not self._inner_model_compiled and self._compile_status_reason:
            logger.debug(
                "%s compile status: %s", self.model_name, self._compile_status_reason
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

        if not self.enable_torch_compile:
            self._compile_status_reason = "disabled by configuration"
            logger.debug(
                "Skipping torch.compile for %s: disabled by configuration.",
                self.model_name,
            )
            return

        if not self.model_profile.compile_inner_transformer:
            self._compile_status_reason = "profile does not support inner-model compile"
            return

        try:
            torch = _import_torch()
        except ImportError:
            self._compile_status_reason = "torch unavailable"
            logger.debug(
                "%s profile supports inner-model torch.compile, but torch is unavailable.",
                self.model_name,
            )
            return

        compile_fn = getattr(torch, "compile", None)
        if not callable(compile_fn):
            self._compile_status_reason = "torch.compile unavailable"
            logger.debug(
                "%s profile supports inner-model torch.compile, but torch.compile is unavailable.",
                self.model_name,
            )
            return

        try:
            transformer_block = self.model[0]
        except Exception as exc:  # pragma: no cover - defensive for upstream API drift
            self._compile_status_reason = f"model[0] unavailable ({type(exc).__name__})"
            logger.warning(
                "Skipping torch.compile for %s: could not access model[0] (%s).",
                self.model_name,
                exc,
            )
            return

        auto_model = getattr(transformer_block, "auto_model", None)
        if auto_model is None:
            self._compile_status_reason = "inner auto_model unavailable"
            logger.warning(
                "Skipping torch.compile for %s: model[0].auto_model is unavailable.",
                self.model_name,
            )
            return

        if auto_model.__class__.__name__ == "OptimizedModule":
            self._inner_model_compiled = True
            self._compile_status_reason = None
            return

        try:
            transformer_block.auto_model = compile_fn(auto_model)
        except Exception as exc:
            self._compile_status_reason = f"compile failed ({type(exc).__name__})"
            logger.warning(
                "torch.compile failed for %s inner transformer; continuing without compile: %s",
                self.model_name,
                exc,
            )
            return

        self._inner_model_compiled = True
        self._compile_status_reason = None
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
            seed_text = (
                compose_title_abstract_text(
                    {"title": seed_title, "abstract": seed_abstract}
                )
                or seed_id
            )
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
        logger.debug("Computing seed embedding...")
        formatted_seed_text = self._format_seed_for_embedding(
            seed_text=seed_text,
            seed_metadata=seed_metadata,
            seed_is_free_text_query=seed_paper.paper_id.startswith("query:"),
        )
        seed_embedding = self._encode_texts(
            [formatted_seed_text], show_progress_bar=False
        )[0]
        self.embeddings[seed_paper.paper_id] = seed_embedding

        use_streaming = self.use_streaming

        if use_streaming:
            logger.debug(
                "Using streaming hydration path for cache-native semantic search..."
            )
            candidates = self._select_candidates_streaming(seed_embedding)
        else:
            logger.debug("Using cache-native semantic search...")
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
                venue=metadata.get("venue", ""),
                arxiv_id=metadata.get("arxiv_id", ""),
                doi=metadata.get("doi", ""),
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
        return self._select_candidates(seed_embedding, use_streaming=False)

    def _select_candidates_streaming(
        self, seed_embedding: np.ndarray
    ) -> List[Tuple[str, Dict, np.ndarray]]:
        """
        Select top candidates from cache using streaming hydration policy.

        :param np.ndarray seed_embedding: Normalized seed embedding vector
        :return List[Tuple[str, Dict, np.ndarray]]: List of (paper_id, metadata, embedding) tuples sorted by similarity
        """
        return self._select_candidates(seed_embedding, use_streaming=True)

    def _format_seed_for_embedding(
        self,
        seed_text: str,
        seed_metadata: Dict[str, str],
        *,
        seed_is_free_text_query: bool,
    ) -> str:
        """Format a seed input into the correct embedding prompt space.

        Paper-to-paper retrieval stays in document space. Only free-text user
        queries should cross from query space into the hydrated document corpus.

        :param str seed_text: Raw seed text or fallback identifier.
        :param Dict[str, str] seed_metadata: Seed metadata payload.
        :param bool seed_is_free_text_query: Whether the seed came from user query text.
        :return str: Prompt-formatted seed text for embedding encode.
        """
        if seed_is_free_text_query:
            return self.model_profile.format_query(seed_text, seed_metadata)

        document_text = self.model_profile.format_document(
            {
                "title": seed_metadata.get("title", ""),
                "abstract": seed_metadata.get("abstract", ""),
            }
        )
        return document_text or seed_text

    def _select_candidates(
        self, seed_embedding: np.ndarray, use_streaming: bool
    ) -> List[Tuple[str, Dict, np.ndarray]]:
        """Select candidates after hydrating cache with a specific loading mode.

        :param np.ndarray seed_embedding: Normalized seed embedding vector.
        :param bool use_streaming: Whether hydration should stream the dataset.
        :return List[Tuple[str, Dict, np.ndarray]]: Candidate tuples sorted by similarity.
        """
        self._ensure_cache_hydrated(use_streaming=use_streaming)
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
        self._last_search_used_binary_prefilter = (
            self.embedding_cache.last_search_used_binary_prefilter
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
        compared_embeddings = getattr(
            self.embedding_cache, "last_search_total_embeddings", None
        )
        rescored_embeddings = getattr(
            self.embedding_cache, "last_search_rescored_embeddings", None
        )
        if compared_embeddings is not None:
            prefilter_used = (
                self._last_search_used_binary_prefilter
                if self._last_search_used_binary_prefilter is not None
                else False
            )
            compared_label = f"{int(compared_embeddings):,}"
            rescored_label = (
                f"{int(rescored_embeddings):,}"
                if rescored_embeddings is not None
                else "unknown"
            )
            logger.info(
                "Semantic cache search compared against %s embeddings "
                "(rescored=%s, prefilter=%s).",
                compared_label,
                rescored_label,
                "on" if prefilter_used else "off",
            )

        return [
            (paper_id, metadata, embedding)
            for _, paper_id, metadata, embedding, _ in scored_candidates[:limited]
        ]

    def _ensure_cache_hydrated(self, use_streaming: bool) -> None:
        """Ensure cache contains hydrated corpus embeddings for current split/cap.

        :param bool use_streaming: Whether to use streaming dataset hydration.
        :return None: Mutates cache state in-place when hydration is required.
        """
        self._ensure_cache_model_fingerprint()
        cached_dataset_source = self.embedding_cache.get_hydrated_dataset_source()
        if self.embedding_cache.is_hydrated(
            self.dataset_split,
            self.corpus_size,
            dataset_source=cached_dataset_source,
        ):
            try:
                self._refresh_hydrated_full_corpus_cache(
                    use_streaming=use_streaming,
                    cached_dataset_source=cached_dataset_source,
                )
            except Exception as exc:
                logger.warning(
                    "Incremental full-corpus refresh check failed for source=%s; "
                    "continuing with existing hydrated cache: %s",
                    cached_dataset_source or "unknown",
                    exc,
                )
            cached_dataset_source = self.embedding_cache.get_hydrated_dataset_source()
            if self.embedding_cache.is_hydrated(
                self.dataset_split,
                self.corpus_size,
                dataset_source=cached_dataset_source,
            ):
                logger.debug(
                    "Embedding cache already hydrated for split=%s corpus_size=%s source=%s; "
                    "skipping dataset load.",
                    self.dataset_split,
                    "all" if self.corpus_size is None else self.corpus_size,
                    cached_dataset_source or "unknown",
                )
                return
            logger.warning(
                "Hydrated cache revalidation invalidated source=%s for split=%s corpus_size=%s; "
                "performing full source revalidation.",
                cached_dataset_source or "unknown",
                self.dataset_split,
                "all" if self.corpus_size is None else self.corpus_size,
            )

        dataset_source: Optional[str]
        dataset: Iterable[Dict[str, Any]]

        try:
            dataset_source, dataset = self._load_dataset_for_hydration(
                use_streaming=use_streaming,
                preferred_dataset_source=cached_dataset_source,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to resolve hydration dataset source; refusing to reuse "
                "existing hydrated cache without source revalidation."
            ) from exc

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
        retained_fingerprint = (
            str(self._resolved_model_fingerprint).strip()
            if self._resolved_model_fingerprint is not None
            else ""
        )
        if not retained_fingerprint:
            fallback_fingerprint = self.embedding_cache.get_model_fingerprint()
            retained_fingerprint = (
                str(fallback_fingerprint).strip()
                if fallback_fingerprint is not None
                else ""
            )
        self._clear_embedding_cache(
            "hydration metadata mismatch requires rebuild "
            f"(requested_split={self.dataset_split}, "
            f"requested_corpus={'all' if self.corpus_size is None else self.corpus_size}, "
            f"resolved_source={dataset_source}, cached_source={cached_dataset_source or 'unknown'})"
        )
        if retained_fingerprint:
            self.embedding_cache.set_model_fingerprint(retained_fingerprint)
        self.embedding_cache.mark_hydrated(
            dataset_source=dataset_source,
            dataset_split=self.dataset_split,
            corpus_size=self.corpus_size,
            complete=False,
        )

        progress_total = self._resolve_hydration_progress_total(
            dataset,
            use_streaming=use_streaming,
        )
        self._ensure_int8_calibration_ranges(
            use_streaming=use_streaming,
            dataset_source=dataset_source,
        )

        hydrated_records = self._hydrate_dataset_records(
            dataset=dataset,
            progress_total=progress_total,
            progress_label=f"Hydrating {dataset_source}",
        )

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

    def _resolve_hydration_progress_total(
        self, dataset: Iterable[Dict[str, Any]], *, use_streaming: bool
    ) -> Optional[int]:
        """Resolve best-effort progress totals for hydration-related passes.

        :param Iterable[Dict[str, Any]] dataset: Dataset iterable used by the pass.
        :param bool use_streaming: Whether the iterable came from streaming mode.
        :return Optional[int]: Progress-bar total when it can be inferred.
        """
        progress_total = self.corpus_size if self.corpus_size else None
        if not use_streaming and self.corpus_size is None:
            try:
                progress_total = len(dataset)
            except TypeError:  # pragma: no cover - defensive for dataset APIs
                progress_total = None
        return progress_total

    def _needs_explicit_int8_calibration(self) -> bool:
        """Return whether int8 hydration must initialize persisted ranges first.

        :return bool: ``True`` when int8 cache writes would otherwise fail closed.
        """
        return (
            self.storage_precision == "int8"
            and not self.embedding_cache.has_calibration_ranges()
        )

    def _sample_calibration_records(
        self,
        dataset: Iterable[Dict[str, Any]],
        *,
        progress_total: Optional[int],
        progress_label: str,
    ) -> List[Dict]:
        """Reservoir-sample representative metadata records for int8 calibration.

        The sample is deterministic so cache bootstrap remains reproducible under
        tests and across repeated local runs given the same corpus slice/order.

        :param Iterable[Dict[str, Any]] dataset: Dataset records to sample.
        :param Optional[int] progress_total: Optional progress-bar total.
        :param str progress_label: Progress-bar description label.
        :return List[Dict]: Reservoir-sampled metadata records.
        """
        rng = random.Random(CALIBRATION_RESERVOIR_SEED)
        sampled_records: List[Dict] = []

        with tqdm(
            total=progress_total,
            desc=progress_label,
            unit="papers",
            dynamic_ncols=True,
            disable=not stderr_isatty(),
        ) as progress:
            for idx, raw_record in enumerate(dataset):
                if self.corpus_size is not None and idx >= self.corpus_size:
                    break

                metadata = _extract_dataset_paper_metadata(raw_record, idx)
                if len(sampled_records) < self.calibration_sample_size:
                    sampled_records.append(metadata)
                else:
                    replace_idx = rng.randint(0, idx)
                    if replace_idx < self.calibration_sample_size:
                        sampled_records[replace_idx] = metadata
                progress.update(1)

            if progress_total is None:
                progress.set_postfix_str(f"processed {progress.n}")

        return sampled_records

    def _ensure_int8_calibration_ranges(
        self,
        *,
        use_streaming: bool,
        dataset_source: str,
    ) -> None:
        """Initialize representative int8 calibration ranges before hydration writes.

        :param bool use_streaming: Whether the hydration source streams records.
        :param str dataset_source: Resolved dataset source token for hydration.
        :return None: Persists calibration ranges in cache when required.
        :raises RuntimeError: If calibration source resolution or sampling fails.
        """
        if not self._needs_explicit_int8_calibration():
            return

        logger.info(
            "Initializing representative int8 calibration ranges from %s (sample_size=%d).",
            dataset_source,
            self.calibration_sample_size,
        )
        calibration_source, calibration_dataset = self._load_dataset_for_hydration(
            use_streaming=use_streaming,
            preferred_dataset_source=dataset_source,
            allow_source_fallback=False,
        )
        if calibration_source != dataset_source:
            raise RuntimeError(
                "Calibration prepass resolved unexpected dataset source "
                f"{calibration_source!r} (expected {dataset_source!r})."
            )

        calibration_records = self._sample_calibration_records(
            calibration_dataset,
            progress_total=self._resolve_hydration_progress_total(
                calibration_dataset,
                use_streaming=use_streaming,
            ),
            progress_label=f"Calibrating {dataset_source}",
        )
        if not calibration_records:
            logger.warning(
                "Representative int8 calibration prepass produced zero records for %s; "
                "hydration will remain incomplete until a non-empty source is available.",
                dataset_source,
            )
            return
        self._initialize_calibration_ranges(calibration_records)

    def _hydrate_dataset_records(
        self,
        dataset: Iterable[Dict[str, Any]],
        *,
        progress_total: Optional[int],
        progress_label: str,
        existing_paper_ids: Optional[Set[str]] = None,
        max_new_records: Optional[int] = None,
    ) -> int:
        """Hydrate cache records from dataset iterator without clearing namespace.

        :param Iterable[Dict[str, Any]] dataset: Dataset records to process.
        :param Optional[int] progress_total: Optional progress-bar total.
        :param str progress_label: Progress-bar description label.
        :param Optional[Set[str]] existing_paper_ids: Optional set used to skip
            already-cached paper IDs while hydrating.
        :param Optional[int] max_new_records: Optional cap on newly selected records.
        :return int: Number of records routed into cache batching.
        """
        if max_new_records is not None and int(max_new_records) < 1:
            raise ValueError("max_new_records must be at least 1 when provided")

        hydrated_records = 0
        selected_records = 0

        with tqdm(
            total=progress_total,
            desc=progress_label,
            unit="papers",
            dynamic_ncols=True,
            disable=not stderr_isatty(),
        ) as progress:
            batch: List[Dict] = []
            for idx, raw_record in enumerate(dataset):
                if self.corpus_size is not None and idx >= self.corpus_size:
                    break

                metadata = _extract_dataset_paper_metadata(raw_record, idx)
                if existing_paper_ids is not None:
                    paper_id = str(metadata.get("paper_id", "")).strip()
                    if not paper_id or paper_id in existing_paper_ids:
                        progress.update(1)
                        continue
                    existing_paper_ids.add(paper_id)

                selected_records += 1
                batch.append(metadata)
                if len(batch) >= HYDRATION_FLUSH_SIZE:
                    hydrated_records += self._cache_metadata_batch(batch)
                    batch = []
                progress.update(1)
                if max_new_records is not None and selected_records >= int(
                    max_new_records
                ):
                    break

            if batch:
                hydrated_records += self._cache_metadata_batch(batch)

            if progress_total is None:
                progress.set_postfix_str(f"processed {progress.n}")

        return hydrated_records

    def _cached_payload_row_count(self) -> int:
        """Return best-effort hydrated payload row count for this namespace.

        :return int: Maximum of SQLite and HDF5 embedding row counts.
        """
        stats = self.embedding_cache.payload_stats()
        return max(int(stats.sqlite_rows), int(stats.embedding_rows))

    def _resolve_dataset_split_row_count(self, dataset_source: str) -> Optional[int]:
        """Resolve dataset split row count from HuggingFace metadata when available.

        :param str dataset_source: Dataset source identifier.
        :return Optional[int]: Split row count, or ``None`` when unavailable.
        """
        if ":" in str(self.dataset_split):
            return None

        load_dataset_builder = _import_datasets_module().load_dataset_builder

        try:
            builder = load_dataset_builder(dataset_source)
            splits = getattr(getattr(builder, "info", None), "splits", None)
            if splits is None:
                return None
            if hasattr(splits, "get"):
                split_info = splits.get(self.dataset_split)
            elif self.dataset_split in splits:
                split_info = splits[self.dataset_split]
            else:
                split_info = None
            if split_info is None:
                return None
            num_examples = getattr(split_info, "num_examples", None)
            if num_examples is None:
                return None
            parsed = int(num_examples)
            return parsed if parsed >= 0 else None
        except Exception as exc:  # pragma: no cover - source/network dependent
            logger.warning(
                "Could not resolve split row count for %s/%s: %s",
                dataset_source,
                self.dataset_split,
                exc,
            )
            return None

    def _refresh_hydrated_full_corpus_cache(
        self, *, use_streaming: bool, cached_dataset_source: Optional[str]
    ) -> None:
        """Incrementally refresh hydrated full-corpus cache when source row count grows.

        This avoids clearing/re-encoding existing payload when a source only appends
        new records.

        :param bool use_streaming: Whether hydration mode is streaming.
        :param Optional[str] cached_dataset_source: Hydrated dataset source token.
        :return None: Mutates cache in-place when incremental refresh is required.
        """
        if self.corpus_size is not None:
            return
        if ":" in str(self.dataset_split):
            return
        source = str(cached_dataset_source or "").strip()
        if not source:
            return

        cached_rows = self._cached_payload_row_count()
        if cached_rows < 1:
            return

        upstream_rows = self._resolve_dataset_split_row_count(source)
        if upstream_rows is None:
            return
        if upstream_rows <= cached_rows:
            if upstream_rows < cached_rows:
                logger.warning(
                    "Cached embedding payload rows (%d) exceed upstream split rows (%d) "
                    "for %s/%s; marking hydration incomplete for full source revalidation.",
                    cached_rows,
                    upstream_rows,
                    source,
                    self.dataset_split,
                )
                self.embedding_cache.mark_hydrated(
                    dataset_source=source,
                    dataset_split=self.dataset_split,
                    corpus_size=self.corpus_size,
                    complete=False,
                )
            self.embedding_cache.clear_hydration_rowcount_reconciliation()
            return

        self._ensure_int8_calibration_ranges(
            use_streaming=use_streaming,
            dataset_source=source,
        )

        previous_reconciliation = (
            self.embedding_cache.get_hydration_rowcount_reconciliation()
        )
        if previous_reconciliation == (upstream_rows, cached_rows):
            logger.info(
                "Skipping incremental refresh for %s/%s: prior reconciliation already "
                "verified this row-count delta (cache_rows=%d, upstream_rows=%d).",
                source,
                self.dataset_split,
                cached_rows,
                upstream_rows,
            )
            return

        delta_rows = upstream_rows - cached_rows
        logger.info(
            "Detected %d new dataset rows for %s/%s (cached=%d, upstream=%d). "
            "Running incremental cache refresh.",
            delta_rows,
            source,
            self.dataset_split,
            cached_rows,
            upstream_rows,
        )
        refreshed_source, dataset = self._load_dataset_for_hydration(
            use_streaming=use_streaming,
            preferred_dataset_source=source,
            row_limit=delta_rows,
            row_offset=cached_rows,
            allow_source_fallback=False,
        )
        if refreshed_source != source:
            raise RuntimeError(
                "Incremental refresh resolved unexpected dataset source "
                f"{refreshed_source!r} (expected {source!r})."
            )

        tail_refreshed_records = self._hydrate_dataset_records(
            dataset=dataset,
            progress_total=delta_rows,
            progress_label=f"Refreshing {source}",
        )
        updated_rows = self._cached_payload_row_count()
        head_reconciled_records = 0
        full_reconciled_records = 0

        if updated_rows < upstream_rows:
            remaining_rows = upstream_rows - updated_rows
            logger.warning(
                "Tail delta refresh left %d unresolved rows for %s/%s "
                "(cache_rows=%d, upstream=%d). Running head-slice missing-ID reconciliation.",
                remaining_rows,
                source,
                self.dataset_split,
                updated_rows,
                upstream_rows,
            )
            cached_paper_ids = self.embedding_cache.get_cached_paper_ids()
            reconciled_source, head_dataset = self._load_dataset_for_hydration(
                use_streaming=use_streaming,
                preferred_dataset_source=source,
                row_limit=delta_rows,
                row_offset=0,
                allow_source_fallback=False,
            )
            if reconciled_source != source:
                raise RuntimeError(
                    "Head-slice reconciliation resolved unexpected dataset source "
                    f"{reconciled_source!r} (expected {source!r})."
                )
            head_reconciled_records = self._hydrate_dataset_records(
                dataset=head_dataset,
                progress_total=delta_rows,
                progress_label=f"Reconciling head {source}",
                existing_paper_ids=cached_paper_ids,
                max_new_records=remaining_rows,
            )
            updated_rows = self._cached_payload_row_count()
            if updated_rows < upstream_rows:
                remaining_rows = upstream_rows - updated_rows
                logger.warning(
                    "Head-slice reconciliation left %d unresolved rows for %s/%s "
                    "(cache_rows=%d, upstream=%d). Running full-split missing-ID reconciliation.",
                    remaining_rows,
                    source,
                    self.dataset_split,
                    updated_rows,
                    upstream_rows,
                )
                reconciled_source, full_dataset = self._load_dataset_for_hydration(
                    use_streaming=use_streaming,
                    preferred_dataset_source=source,
                    allow_source_fallback=False,
                )
                if reconciled_source != source:
                    raise RuntimeError(
                        "Full-split reconciliation resolved unexpected dataset source "
                        f"{reconciled_source!r} (expected {source!r})."
                    )
                full_reconciled_records = self._hydrate_dataset_records(
                    dataset=full_dataset,
                    progress_total=upstream_rows,
                    progress_label=f"Reconciling full {source}",
                    existing_paper_ids=cached_paper_ids,
                    max_new_records=None,
                )
                updated_rows = self._cached_payload_row_count()

        if updated_rows > 0:
            self.embedding_cache.mark_hydrated(
                dataset_source=source,
                dataset_split=self.dataset_split,
                corpus_size=self.corpus_size,
                complete=True,
            )
        logger.info(
            "Incremental refresh processed tail=%d head=%d full=%d rows "
            "for %s/%s (cache_rows=%d, upstream_rows=%d).",
            tail_refreshed_records,
            head_reconciled_records,
            full_reconciled_records,
            source,
            self.dataset_split,
            updated_rows,
            upstream_rows,
        )
        if updated_rows < upstream_rows:
            self.embedding_cache.set_hydration_rowcount_reconciliation(
                upstream_rows=upstream_rows,
                cached_rows=updated_rows,
            )
            logger.info(
                "Full-split reconciliation completed for %s/%s with cache_rows=%d "
                "and upstream_rows=%d. Remaining row-count delta likely reflects "
                "duplicate upstream paper IDs; this state is memoized to skip "
                "repeat full-split scans until row counts change.",
                source,
                self.dataset_split,
                updated_rows,
                upstream_rows,
            )
        else:
            self.embedding_cache.clear_hydration_rowcount_reconciliation()

    def _load_dataset_for_hydration(
        self,
        use_streaming: bool,
        preferred_dataset_source: Optional[str] = None,
        row_limit: Optional[int] = None,
        row_offset: Optional[int] = None,
        allow_source_fallback: bool = True,
    ) -> Tuple[str, Iterable[Dict[str, Any]]]:
        """Load first available ArXiv dataset for hydration.

        :param bool use_streaming: Whether to load streaming dataset iterator.
        :param Optional[str] preferred_dataset_source: Preferred source if already cached.
        :param Optional[int] row_limit: Optional row cap override for dataset loading.
        :param Optional[int] row_offset: Optional row offset for delta refresh loading.
        :param bool allow_source_fallback: Whether alternate sources may be tried.
        :return Tuple[str, Iterable[Dict[str, Any]]]: Dataset source name and iterable.
        """
        load_dataset = _import_datasets_module().load_dataset

        parsed_row_limit: Optional[int] = None
        if row_limit is not None:
            parsed_row_limit = int(row_limit)
            if parsed_row_limit < 1:
                raise ValueError("row_limit must be at least 1 when provided")

        parsed_row_offset = 0 if row_offset is None else int(row_offset)
        if parsed_row_offset < 0:
            raise ValueError("row_offset must be at least 0 when provided")

        last_error: Optional[Exception] = None
        if preferred_dataset_source is not None and not allow_source_fallback:
            dataset_names = (preferred_dataset_source,)
        elif (
            preferred_dataset_source is not None
            and preferred_dataset_source in ARXIV_DATASET_CANDIDATES
        ):
            dataset_names = (
                preferred_dataset_source,
                *[
                    dataset_name
                    for dataset_name in ARXIV_DATASET_CANDIDATES
                    if dataset_name != preferred_dataset_source
                ],
            )
        else:
            dataset_names = ARXIV_DATASET_CANDIDATES

        for dataset_name in dataset_names:
            split_for_load = self.dataset_split
            if not use_streaming and parsed_row_offset > 0 and ":" in split_for_load:
                raise ValueError(
                    "row_offset requires a non-sliced dataset_split in non-streaming mode"
                )
            if (
                not use_streaming
                and ":" not in split_for_load
                and parsed_row_limit is not None
                and parsed_row_offset > 0
            ):
                stop_idx = parsed_row_offset + parsed_row_limit
                split_for_load = f"{split_for_load}[{parsed_row_offset}:{stop_idx}]"
            elif (
                not use_streaming
                and ":" not in split_for_load
                and parsed_row_limit is not None
            ):
                split_for_load = f"{split_for_load}[:{parsed_row_limit}]"
            elif (
                not use_streaming
                and ":" not in split_for_load
                and parsed_row_offset > 0
            ):
                split_for_load = f"{split_for_load}[{parsed_row_offset}:]"
            elif (
                not use_streaming
                and self.corpus_size is not None
                and ":" not in split_for_load
            ):
                split_for_load = f"{split_for_load}[:{int(self.corpus_size)}]"
            try:
                dataset = load_dataset(
                    dataset_name,
                    split=split_for_load,
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
            if use_streaming and (
                parsed_row_limit is not None or parsed_row_offset > 0
            ):
                stop_idx = (
                    None
                    if parsed_row_limit is None
                    else parsed_row_offset + parsed_row_limit
                )
                dataset = islice(dataset, parsed_row_offset, stop_idx)
            logger.debug(
                "Hydration dataset selected: %s (split=%s, streaming=%s, row_limit=%s, row_offset=%s).",
                dataset_name,
                split_for_load,
                use_streaming,
                "none" if parsed_row_limit is None else parsed_row_limit,
                parsed_row_offset,
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
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
        )
        ranges = np.percentile(
            sample_embeddings,
            [CALIBRATION_LOWER_PERCENTILE, CALIBRATION_UPPER_PERCENTILE],
            axis=0,
        ).astype(np.float32)
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

        self.embedding_cache.upsert_embeddings(
            metadata_map,
            self._get_model_for_encoding(),
            batch_size=min(self.encode_batch_size, len(metadata_map)),
            show_progress=False,
            text_builder=self.model_profile.format_document,
        )
        return len(metadata_map)

    def _update_citation_counts(self, papers: Dict[str, Paper]) -> None:
        """
        Enrich top semantic candidates with citation counts from Semantic Scholar.

        :param Dict[str, Paper] papers: Dictionary of collected papers (including seed)
        """
        targets = [
            (pid, paper)
            for pid, paper in list(papers.items())[:CITATION_COUNT_ENRICHMENT_LIMIT]
            if not paper.is_seed
            and not (isinstance(pid, str) and pid.startswith("query:"))
            and not (isinstance(pid, str) and pid.startswith("arxiv_"))
        ]

        if not targets:
            return

        logger.info(
            "Fetching citation counts from Semantic Scholar for up to %d papers...",
            len(targets),
        )

        progress_enabled = stderr_isatty() and len(targets) > 1
        progress_bar = (
            tqdm(
                targets,
                desc="Citation counts",
                unit="papers",
                dynamic_ncols=True,
                leave=False,
            )
            if progress_enabled
            else None
        )
        iterator = progress_bar if progress_bar is not None else targets

        batch_results: Dict[str, Paper] = {}
        batch_fetch = getattr(self.client, "get_papers", None)
        if callable(batch_fetch):
            try:
                batch_response = batch_fetch([paper_id for paper_id, _paper in targets])
                if isinstance(batch_response, dict):
                    batch_results = {
                        normalize_paper_id(str(paper_id)): paper
                        for paper_id, paper in batch_response.items()
                        if isinstance(paper, Paper)
                    }
                else:
                    logger.debug(
                        "Ignoring unexpected batch citation-count response type %s.",
                        type(batch_response).__name__,
                    )
            except Exception as exc:
                logger.warning("Could not batch fetch citation counts: %s", exc)

        for paper_id, paper in iterator:
            batch_paper = batch_results.get(normalize_paper_id(paper_id))
            if batch_paper is not None:
                paper.citation_count = batch_paper.citation_count
                continue
            try:
                s2_paper = self.client.get_paper(paper_id)
                if s2_paper:
                    paper.citation_count = s2_paper.citation_count
            except Exception as exc:
                logger.warning(f"Could not fetch citation count for {paper_id}: {exc}")

        if progress_bar is not None:
            progress_bar.close()

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
        graph.graph["embedding_runtime"] = self._embedding_runtime_metadata()
        # Enforce a strict per-node top-k cap by greedily keeping strongest edges.
        filtered_graph = build_capped_undirected_graph(graph, self.top_k)

        logger.info(
            f"Filtered graph: {filtered_graph.number_of_nodes()} nodes, "
            f"{filtered_graph.number_of_edges()} edges (top-{self.top_k})"
        )

        return filtered_graph, actual_seed_id
