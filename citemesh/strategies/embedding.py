"""
Embedding-based graph building strategy.

This strategy uses semantic similarity from sentence transformers
to find conceptually similar papers without relying on citations.
"""

from __future__ import annotations

import heapq
import importlib.util
import json
import logging
import os
import random
import re
import warnings
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from hashlib import sha1, sha256
from importlib import metadata as importlib_metadata
from itertools import chain, islice
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
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
    get_cache_dir,
    resolve_embedding_model_profile,
    validate_compression_filter,
)
from citemesh.data.embedding_cache import (
    EMBEDDING_DATASET_CHUNK_ROWS,
    CacheSearchResult,
)
from citemesh.data.model_profiles import (
    EmbeddingModelProfile,
    compose_title_abstract_text,
)
from citemesh.paper_ids import (
    external_ids_from_canonical_paper_id,
    normalize_paper_id,
    recognize_arxiv_identifier,
)
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
)
from citemesh.text_batching import (
    encode_texts_in_length_buckets,
    l2_normalize_embeddings,
)

if TYPE_CHECKING:
    from citemesh.services.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)
_FA2_LOAD_DTYPE_WARNING_PREFIX = (
    "Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes"
)
_EMBEDDING_MIN_TORCH_VERSION = (2, 9)
# bf16-on-MPS is only enabled on torch releases verified on Apple Silicon; this is
# a policy floor, not a hard technical cliff — lower it once older wheels are vetted.
_MPS_MIN_TORCH_VERSION = (2, 13)
EMBEDDING_DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")
_COMPILE_ELIGIBLE_DEVICES = frozenset({"cuda", "mps", "cpu"})
# Legacy release-branch workaround window where Inductor conflicted with the
# fp32_precision TF32 API; later torch releases use the modern API directly.
_TF32_COMPILE_BRIDGE_TORCH_VERSIONS = frozenset({(2, 9), (2, 10)})
_INFERENCE_ARTIFACT_SUFFIXES = frozenset(
    {
        ".bin",
        ".json",
        ".merges",
        ".model",
        ".onnx",
        ".pt",
        ".pth",
        ".py",
        ".safetensors",
        ".tflite",
        ".tiktoken",
        ".txt",
        ".vocab",
    }
)
_IGNORED_ARTIFACT_DIRECTORIES = frozenset({".git", "logs", "runs", "wandb"})
_IGNORED_ARTIFACT_FILES = frozenset(
    {
        ".ds_store",
        ".gitattributes",
        ".gitignore",
        "optimizer.pt",
        "rng_state.pth",
        "scaler.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
)
_RETRIEVAL_DOCUMENT_REPRESENTATION = "retrieval-document-v1"
_GRAPH_SIMILARITY_REPRESENTATION = "graph-similarity-v1"
_PLACEHOLDER_EMBEDDING_TITLES = frozenset(
    {"", "n/a", "na", "none", "unknown", "untitled"}
)
_FORMATTER_FINGERPRINT_PROBES = (
    {"title": "Alpha", "abstract": "Beta"},
    {"title": "Alpha", "abstract": ""},
    {"title": "", "abstract": "Beta"},
    {"title": "  Alpha  ", "abstract": "  Beta  "},
)


@contextmanager
def _suppress_expected_fa2_load_dtype_warning(*, enabled: bool) -> Iterator[None]:
    """Suppress Transformers' redundant FA2 warning for verified bf16 autocast.

    Transformers validates the checkpoint weight dtype before the first forward
    pass, so automatic FP32 weights trigger a warning even though its FA2 adapter
    uses the active CUDA autocast dtype for attention inputs.

    :param bool enabled: Whether verified CUDA bf16 autocast and FA2 are active.
    :return Iterator[None]: Scoped logging filter context.
    """
    if not enabled:
        yield
        return

    def keep_relevant_warning(record: logging.LogRecord) -> bool:
        """Reject only the expected FP32 checkpoint warning.

        :param logging.LogRecord record: Candidate Transformers log record.
        :return bool: Whether the log record should be emitted.
        """
        message = record.getMessage()
        return not (
            message.startswith(_FA2_LOAD_DTYPE_WARNING_PREFIX)
            and " is torch.float32." in message
        )

    transformers_logger = logging.getLogger("transformers.modeling_utils")
    transformers_logger.addFilter(keep_relevant_warning)
    try:
        yield
    finally:
        transformers_logger.removeFilter(keep_relevant_warning)


class EmbeddingTask(str, Enum):
    """Prompt-conditioned embedding roles used by CiteMesh."""

    RETRIEVAL_QUERY = "retrieval-query"
    RETRIEVAL_DOCUMENT = "retrieval-document"
    GRAPH_SIMILARITY = "graph-similarity"


class EmbeddingBackendCompatibilityError(RuntimeError):
    """The active embedding model requires a newer inference backend."""


class EmbeddingPrecisionCompatibilityError(RuntimeError):
    """The checkpoint's automatic weight dtype violates runtime precision policy."""


def _embedding_text_metadata(title: object, abstract: object) -> Dict[str, str]:
    """Normalize title/abstract fields without treating placeholders as content.

    :param object title: Raw paper title.
    :param object abstract: Raw paper abstract.
    :return Dict[str, str]: Clean text metadata for prompt formatting.
    """
    normalized_title = str(title or "").strip()
    if normalized_title.casefold() in _PLACEHOLDER_EMBEDDING_TITLES:
        normalized_title = ""
    return {
        "title": normalized_title,
        "abstract": str(abstract or "").strip(),
    }


def format_paper_for_embedding(
    *, profile: Any, paper: Paper, task: EmbeddingTask
) -> str:
    """Format one paper for a specific retrieval or graph task.

    :param Any profile: Active embedding model profile.
    :param Paper paper: Paper whose title/abstract should be formatted.
    :param EmbeddingTask task: Required prompt-conditioned vector role.
    :return str: Model input text, falling back to the paper ID when needed.
    :raises ValueError: If ``task`` is unsupported.
    """
    return format_embedding_metadata(
        profile=profile,
        metadata={"title": paper.title, "abstract": paper.abstract},
        paper_id=paper.paper_id,
        task=task,
    )


def format_embedding_metadata(
    *,
    profile: Any,
    metadata: Mapping[str, object],
    task: EmbeddingTask,
    paper_id: object = "",
) -> str:
    """Format paper metadata for one prompt-conditioned embedding role.

    :param Any profile: Active embedding model profile.
    :param Mapping[str, object] metadata: Paper metadata containing text fields.
    :param EmbeddingTask task: Required prompt-conditioned vector role.
    :param object paper_id: Identity fallback when title and abstract are empty.
    :return str: Profile-formatted input with a stable identity fallback.
    :raises ValueError: If ``task`` is unsupported.
    """
    text_metadata = _embedding_text_metadata(
        metadata.get("title"), metadata.get("abstract")
    )
    fallback_id = str(paper_id or metadata.get("paper_id") or "unknown-paper")
    content = compose_title_abstract_text(text_metadata) or fallback_id
    if task is EmbeddingTask.RETRIEVAL_QUERY:
        return str(profile.format_query(content, text_metadata))
    if task is EmbeddingTask.RETRIEVAL_DOCUMENT:
        document_metadata = dict(text_metadata)
        if not compose_title_abstract_text(document_metadata):
            document_metadata["title"] = fallback_id
        return str(profile.format_document(document_metadata)) or content
    if task is EmbeddingTask.GRAPH_SIMILARITY:
        return str(profile.format_similarity(content, text_metadata)) or content
    raise ValueError(f"Unsupported embedding task: {task}")


@dataclass(frozen=True)
class _HydrationSourceSliceResult:
    """Outcome of consuming one exact-source hydration slice.

    ``hydrated_records`` counts unique records routed into cache batching, while
    ``source_rows_consumed`` and ``source_exhausted`` describe source traversal.
    Keeping those concepts separate prevents duplicate/invalid source rows from
    being mistaken for an interrupted resume.

    :ivar int hydrated_records: Records routed into cache batching.
    :ivar int source_rows_consumed: Raw source rows yielded to hydration.
    :ivar bool source_exhausted: Whether the selected source slice reached clean EOF.
    """

    hydrated_records: int
    source_rows_consumed: int
    source_exhausted: bool


def _parse_torch_major_minor(version: str) -> tuple[int, int]:
    """Parse major/minor tuple from a torch version string.

    :param str version: Raw torch version string.
    :return tuple[int, int]: Parsed ``(major, minor)`` tuple, ``(0, 0)`` on parse miss.
    """
    version_match = re.match(r"^(\d+)\.(\d+)", str(version).strip())
    if version_match:
        return int(version_match.group(1)), int(version_match.group(2))
    return (0, 0)


def _parse_major_minor(version: str) -> Optional[tuple[int, int]]:
    """Parse a leading semantic-version major/minor pair.

    :param str version: Raw package version string.
    :return Optional[tuple[int, int]]: Parsed pair, or ``None`` when unavailable.
    """
    version_match = re.match(r"^(\d+)\.(\d+)", str(version).strip())
    if version_match is None:
        return None
    return int(version_match.group(1)), int(version_match.group(2))


def _installed_transformers_major_minor(transformers: Any) -> Optional[tuple[int, int]]:
    """Resolve the installed Transformers version from module or distribution data.

    Editable and development builds sometimes expose a nonstandard module
    ``__version__`` even though their installed distribution metadata is valid.

    :param Any transformers: Imported Transformers module.
    :return Optional[tuple[int, int]]: Verified major/minor pair when discoverable.
    """
    module_version = _parse_major_minor(getattr(transformers, "__version__", ""))
    if module_version is not None:
        return module_version
    try:
        distribution_version = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError:
        return None
    return _parse_major_minor(distribution_version)


def _require_transformers_compatibility(profile: EmbeddingModelProfile) -> None:
    """Require the Transformers floor declared by an embedding model profile.

    :param EmbeddingModelProfile profile: Runtime-active model contract.
    :return None: The installed backend satisfies the model contract.
    :raises EmbeddingBackendCompatibilityError: If Transformers is too old.
    """
    minimum = profile.minimum_transformers_version
    if minimum is None:
        return
    required = ".".join(str(part) for part in minimum)

    transformers = importlib.import_module("transformers")
    raw_version = str(getattr(transformers, "__version__", "")).strip()
    detected = _installed_transformers_major_minor(transformers)
    if detected is None:
        raise EmbeddingBackendCompatibilityError(
            f"Could not determine the installed Transformers version for {profile.name}; "
            "the bidirectional-attention compatibility floor cannot be verified. "
            f"Reinstall a supported transformers>={required} release."
        )
    if detected < minimum:
        detected_label = (
            raw_version
            if _parse_major_minor(raw_version) is not None
            else ".".join(str(part) for part in detected)
        )
        raise EmbeddingBackendCompatibilityError(
            f"{profile.name} requires transformers>={required} because older "
            "Gemma 3 implementations ignore bidirectional attention. "
            f"Detected transformers=={detected_label}."
        )


def _accelerator_available(torch: Any, backend: str) -> bool:
    """Return whether a torch accelerator backend is usable.

    :param Any torch: Imported ``torch`` module object.
    :param str backend: Accelerator token (``cuda`` or ``mps``).
    :return bool: ``True`` when the backend reports as available.
    """
    if backend == "mps":
        _built, available = _mps_capabilities(torch)
        return available

    owner = torch
    backend_module = getattr(owner, backend, None)
    is_available = getattr(backend_module, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def _mps_capabilities(torch: Any) -> tuple[bool, bool]:
    """Return whether the torch MPS backend is built and currently available.

    :param Any torch: Imported torch module object.
    :return tuple[bool, bool]: ``(is_built, is_available)`` capability flags.
    """
    backends = getattr(torch, "backends", None)
    backend_mps = getattr(backends, "mps", None) if backends is not None else None
    torch_mps = getattr(torch, "mps", None)

    is_built = getattr(backend_mps, "is_built", None)
    try:
        built = bool(is_built()) if callable(is_built) else False
    except Exception:
        built = False

    available = False
    for owner in (torch_mps, backend_mps):
        is_available = getattr(owner, "is_available", None)
        if not callable(is_available):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                available = bool(is_available())
        except Exception:
            available = False
        if available:
            break

    if available:
        built = True
    return built, available


def _cuda_native_bf16_supported(torch: Any) -> bool:
    """Return whether CUDA provides native rather than emulated bfloat16.

    :param Any torch: Imported torch module object.
    :return bool: ``True`` only for a live CUDA backend with native bf16 support.
    """
    cuda_module = getattr(torch, "cuda", None)
    is_supported = getattr(cuda_module, "is_bf16_supported", None)
    if not callable(is_supported):
        return False
    try:
        return bool(is_supported(including_emulation=False))
    except TypeError:
        get_capability = getattr(cuda_module, "get_device_capability", None)
        try:
            capability = get_capability(0) if callable(get_capability) else None
        except Exception:
            return False
        return bool(
            isinstance(capability, tuple)
            and capability
            and int(capability[0]) >= 8
            and is_supported()
        )
    except Exception:
        return False


def _cpu_native_bf16_supported(torch: Any) -> bool:
    """Check native CPU BF16 instructions using available torch capability APIs.

    :param Any torch: Imported torch module object.
    :return bool: Whether x86 or ARM BF16 support can be established.
    """
    cpu_module = getattr(torch, "cpu", None)
    capabilities = getattr(cpu_module, "get_capabilities", None)
    try:
        if callable(capabilities):
            supported = capabilities()
            return any(
                supported.get(name, False)
                for name in (
                    "avx512_bf16",
                    "amx_bf16",
                    "bf16",
                    "sve_bf16",
                )
            )
        # Older supported torch versions expose only this x86 BF16 probe.
        probe = getattr(cpu_module, "_is_avx512_bf16_supported", None)
        return bool(probe()) if callable(probe) else False
    except Exception:
        return False


def resolve_embedding_device(requested: Optional[str]) -> str:
    """Resolve a requested device token to a concrete torch device string.

    ``auto`` (or ``None``) prefers ``cuda``, then ``mps``, then ``cpu``. An
    explicit accelerator request fails loudly when that backend is unavailable
    rather than silently downgrading to a slower device.

    :param Optional[str] requested: Requested device token
        (``auto``/``cuda``/``mps``/``cpu`` or ``None``).
    :return str: Resolved device token: ``cuda``, ``mps``, or ``cpu``.
    :raises ValueError: If the token is unknown or names an unavailable backend.
    """
    normalized = str(requested or "auto").strip().lower()
    if normalized not in EMBEDDING_DEVICE_CHOICES:
        formatted = ", ".join(EMBEDDING_DEVICE_CHOICES)
        raise ValueError(f"device must be one of: {formatted} (got {requested!r})")

    try:
        torch = _import_torch()
    except ImportError:
        if normalized in {"auto", "cpu"}:
            return "cpu"
        raise ValueError(
            f"device='{normalized}' requires torch; install citemesh[embeddings]."
        ) from None

    if normalized == "auto":
        if _accelerator_available(torch, "cuda"):
            return "cuda"
        if _accelerator_available(torch, "mps"):
            return "mps"
        return "cpu"
    if normalized == "cuda" and not _accelerator_available(torch, "cuda"):
        raise ValueError(
            "device='cuda' was requested but CUDA is not available in this runtime."
        )
    if normalized == "mps":
        mps_built, mps_available = _mps_capabilities(torch)
        if not mps_built:
            raise ValueError(
                "device='mps' was requested but this torch build has no MPS support."
            )
        if not mps_available:
            raise ValueError(
                "device='mps' was requested but the built MPS backend is not "
                "available on this machine."
            )
    return normalized


def _check_embedding_deps(require_corpus: bool = True) -> None:
    """Verify embedding dependencies are installed.

    :param bool require_corpus: Whether corpus hydration deps (``datasets``)
        are required. Candidate mode only needs the encoder stack.
    """
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

    if require_corpus:
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
HYDRATION_FLUSH_SIZE = EMBEDDING_DATASET_CHUNK_ROWS
CANDIDATE_MULTIPLIER = 4
CITATION_COUNT_ENRICHMENT_LIMIT = 20
CALIBRATION_RESERVOIR_SEED = 0
ARXIV_DATASET_CANDIDATES = (
    "librarian-bots/arxiv-metadata-snapshot",
    "CShorten/ML-ArXiv-Papers",
    "gfissore/arxiv-abstracts-2021",
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

    return recognize_arxiv_identifier(text, allow_bare=True) or text


def _query_seed_id(query_text: str) -> str:
    """Build deterministic query-mode seed node identifier.

    :param str query_text: Raw user query text.
    :return str: Stable hashed query seed identifier.
    """
    digest = sha1(query_text.encode("utf-8")).hexdigest()[:8]
    return f"query:{digest}"


def _model_floating_dtype_names(model: Any) -> Optional[Set[str]]:
    """Return floating-point parameter and buffer dtypes from a loaded model.

    :param Any model: Model object that may expose ``parameters()``.
    :return Optional[Set[str]]: Normalized dtype names, or ``None`` when live
        tensor inspection is unavailable.
    """
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    buffers = getattr(model, "buffers", None)

    observed: Set[str] = set()
    aliases = {
        "float": "float32",
        "float16": "float16",
        "half": "float16",
        "bfloat16": "bfloat16",
        "float32": "float32",
        "double": "float64",
        "float64": "float64",
    }
    try:
        for tensor in chain(parameters(), buffers() if callable(buffers) else ()):
            dtype = getattr(tensor, "dtype", None)
            dtype_name = str(dtype or "").casefold()
            normalized = dtype_name.removeprefix("torch.")
            is_floating = getattr(dtype, "is_floating_point", None)
            if is_floating is False:
                continue
            mapped = aliases.get(normalized)
            if mapped is not None:
                observed.add(mapped)
            elif is_floating is True or normalized.startswith(("float", "bfloat")):
                observed.add(normalized)
    except Exception:
        logger.debug("Could not inspect loaded model tensor dtypes", exc_info=True)
        return None
    return observed


class _PrecisionEncodeProxy:
    """Apply encode-time precision and lazy-compile recovery for every caller."""

    def __init__(
        self,
        model: Any,
        context_factory: Callable[[], Any],
        restore_eager: Callable[[Exception], bool],
    ):
        """Create a model proxy for encode-time precision controls.

        :param Any model: Wrapped model object exposing ``encode``.
        :param Callable[[], Any] context_factory: Callable returning a context manager.
        :param Callable[[Exception], bool] restore_eager: Restore eager execution
            after a compiled-call failure; return whether the call can be retried.
        """
        self._model = model
        self._context_factory = context_factory
        self._restore_eager = restore_eager

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        """Run ``encode`` within the configured context manager.

        :param Any args: Positional arguments forwarded to ``encode``.
        :param Any kwargs: Keyword arguments forwarded to ``encode``.
        :return Any: Model ``encode`` return value.
        """
        normalize_embeddings = kwargs.get("normalize_embeddings", False)
        if normalize_embeddings:
            kwargs["normalize_embeddings"] = False
        try:
            with self._context_factory():
                embeddings = self._model.encode(*args, **kwargs)
        except Exception as exc:
            compiled_failure = exc
            if not self._restore_eager(compiled_failure):
                raise
            try:
                with self._context_factory():
                    embeddings = self._model.encode(*args, **kwargs)
            except Exception as eager_error:
                raise eager_error from compiled_failure
        # Normalize once in FP32; CPU autocast would round ST's division to BF16.
        if normalize_embeddings:
            return l2_normalize_embeddings(embeddings)
        return embeddings

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

    raw_id = paper.get("id") or paper.get("paper_id") or paper.get("paperId")
    chronology = _arxiv_id_chronology_key(raw_id)
    return chronology[0] if chronology is not None else None


_NEW_STYLE_ARXIV_ID_RE = re.compile(r"^(\d{2})(\d{2})\.(\d{4,5})(?:v\d+)?$")
_OLD_STYLE_ARXIV_ID_RE = re.compile(
    r"^[a-z][a-z-]*(?:\.[a-z-]+)?/(\d{2})(\d{2})(\d{3})(?:v\d+)?$"
)


def _arxiv_id_chronology_key(raw_id: Any) -> Optional[Tuple[int, int, int]]:
    """Return a sortable submission-chronology key for an arXiv identifier.

    Both identifier styles encode the submission year/month: new-style
    ``YYMM.NNNNN`` and old-style ``archive/YYMMNNN``. Snapshot row order and
    ``update_date`` do not track submission time (revisions bump old papers),
    so this key is the only reliable "newest papers" ordering.

    :param Any raw_id: Raw identifier value from a dataset record.
    :return Optional[Tuple[int, int, int]]: ``(year, month, sequence)`` or
        ``None`` when the identifier is not a parseable arXiv ID.
    """
    text = str(raw_id or "").strip().lower()
    if text.startswith("arxiv:"):
        text = text[len("arxiv:") :]
    match = _NEW_STYLE_ARXIV_ID_RE.match(text) or _OLD_STYLE_ARXIV_ID_RE.match(text)
    if match is None:
        return None
    year_token, month, sequence = (int(group) for group in match.groups())
    # arXiv started in 1991; two-digit years wrap at the century boundary.
    year = 1900 + year_token if year_token >= 91 else 2000 + year_token
    return (year, month, sequence)


def _newest_records_by_arxiv_id(
    records: Iterable[Dict[str, Any]], limit: int
) -> List[Dict[str, Any]]:
    """Select the ``limit`` most recently submitted records in one bounded pass.

    Prefer records with parseable arXiv IDs, filling any shortfall from the
    first records without parseable IDs so hydration reaches the requested cap.

    :param Iterable[Dict[str, Any]] records: Dataset records to scan.
    :param int limit: Number of newest records to keep.
    :return List[Dict[str, Any]]: Selected records in chronological order.
    """
    heap: List[Tuple[Tuple[int, int, int], int, Dict[str, Any]]] = []
    head_fallback: List[Dict[str, Any]] = []
    for order, record in enumerate(records):
        key = _arxiv_id_chronology_key((record or {}).get("id"))
        if key is None:
            if len(head_fallback) < limit:
                head_fallback.append(record)
            continue
        entry = (key, order, record)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    selected = [record for _, _, record in sorted(heap, key=lambda entry: entry[:2])]
    return selected + head_fallback[: limit - len(selected)]


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
    source_doi = str(paper.get("doi") or "").strip()
    if source_doi:
        _, normalized_doi = external_ids_from_canonical_paper_id(
            normalize_paper_id(source_doi)
        )
        doi = normalized_doi or doi
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
        model_profile: str = "auto",
        model_revision: Optional[str] = None,
        dataset_split: str = "train",
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
        device: Optional[str] = None,
        semantic_source: str = "candidates",
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
        client: Optional[SemanticScholarClient] = None,
    ):
        """
        Initialize embedding graph builder.

        :param int max_papers: Maximum papers in final graph
        :param str model_name: Sentence transformer model name
        :param str model_profile: Model task/runtime profile override
            (``auto``, ``embeddinggemma``, or ``default``).
        :param Optional[str] model_revision: Optional model revision token for hub-backed models.
        :param str dataset_split: HuggingFace dataset split
        :param Optional[int] corpus_size: Maximum papers to embed and cache after
            scanning the selected split for newest submissions (``None`` = all).
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
        """
        normalized_semantic_source = str(semantic_source).strip().lower()
        if normalized_semantic_source not in SEMANTIC_SOURCE_CHOICES:
            formatted = ", ".join(SEMANTIC_SOURCE_CHOICES)
            raise ValueError(f"semantic_source must be one of: {formatted}")
        _check_embedding_deps(
            require_corpus=normalized_semantic_source == "arxiv-corpus"
        )
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be at least 1")
        if normalized_semantic_source != "arxiv-corpus" and storage_precision == "int8":
            # int8 calibration ranges are only computed during corpus hydration;
            # candidate pools are small enough that float32 storage is free.
            logger.debug("Candidate mode does not support int8 storage; using float32.")
            storage_precision = "float32"
            if binary_prefilter is None:
                binary_prefilter = False
            if binary_rescore_multiplier is None:
                binary_rescore_multiplier = 1
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
        self.top_k = top_k
        self.model = None
        self.retrieval_embeddings: Dict[str, np.ndarray] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.candidate_source_status: Dict[str, str] = {}
        self.client = client or get_client()
        self._active_model_name: Optional[str] = None
        self._embedding_cache: Optional[EmbeddingCache] = None
        self._graph_embedding_cache: Optional[EmbeddingCache] = None
        self._pending_force_rebuild_reason: Optional[str] = None
        normalized_force_rebuild_reason = (
            " ".join(str(force_rebuild_reason).split())
            if force_rebuild_reason is not None
            else ""
        )
        if force_rebuild_cache:
            logger.info(
                "Embedding cache rebuild requested; deferring clear until the "
                "runtime-active model namespace is resolved."
            )
            clear_reason = "explicit --force-rebuild-cache request"
            if normalized_force_rebuild_reason:
                clear_reason = (
                    f"{clear_reason}; user_reason={normalized_force_rebuild_reason}"
                )
            self._pending_force_rebuild_reason = clear_reason
        self.use_streaming = use_streaming
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
        self._profile_logged = False
        self._dim_logged = False
        self._autocast_dtype: Optional[Any] = None
        self._autocast_device_type: Optional[str] = None
        self._autocast_enabled = False
        self._encode_model: Optional[Any] = None
        self._inner_model_compiled = False
        self._eager_inner_transformer: Optional[Any] = None
        self._compile_status_reason: Optional[str] = None
        self._runtime_summary_logged = False
        self._tf32_runtime_configured = False
        self._tf32_mode = "off"
        self._resolved_model_fingerprint: Optional[str] = None
        self._last_search_used_binary_prefilter: Optional[bool] = None

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
        storage_precision: Optional[str] = None,
        binary_prefilter: Optional[bool] = None,
        formatter_identity: Optional[str] = None,
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
            binary_prefilter=False,
            formatter_identity=self._similarity_formatter_fingerprint,
        )

    def _create_graph_embedding_cache(
        self, namespace: Optional[str] = None
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
            self._embedding_cache.clear(reason=self._pending_force_rebuild_reason)
            self.graph_embedding_cache.clear(reason=self._pending_force_rebuild_reason)
            self._pending_force_rebuild_reason = None

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

    def _embedding_cache_namespace(
        self,
        *,
        representation: str = _RETRIEVAL_DOCUMENT_REPRESENTATION,
        artifact_identity: Optional[str] = None,
        storage_precision: Optional[str] = None,
        binary_prefilter: Optional[bool] = None,
        formatter_identity: Optional[str] = None,
    ) -> str:
        """Build a cache key for the active model and representation contract.

        :param str representation: Semantic role and schema version of stored vectors.
        :param Optional[str] artifact_identity: Immutable resolved checkpoint identity.
        :param Optional[str] storage_precision: Representation-specific storage mode.
        :param Optional[bool] binary_prefilter: Representation-specific prefilter mode.
        :param Optional[str] formatter_identity: Representation formatter fingerprint.
        :return str: Namespace key used for embedding cache partitioning.
        """
        resolved_storage = storage_precision or self.storage_precision
        resolved_prefilter = (
            self.binary_prefilter
            if binary_prefilter is None
            else bool(binary_prefilter)
        )
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
        parts.append(f"binary_prefilter={int(resolved_prefilter)}")
        if resolved_storage == "int8":
            parts.append(f"calibration_sample_size={self.calibration_sample_size}")
        parts.append(f"source_dtype={self._source_dtype_hint}")
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

    def _resolve_formatter_fingerprint(
        self,
        *,
        formatter: Callable[..., str],
        probe_renderer: Callable[[Dict[str, str]], str],
    ) -> str:
        """Resolve a deterministic cache fingerprint for one formatter role.

        :param Callable[..., str] formatter: Underlying formatter identity.
        :param Callable[[Dict[str, str]], str] probe_renderer: Probe rendering callback.
        :return str: SHA-256 digest of profile formatter probes.
        """
        outputs = [
            probe_renderer(dict(metadata)) for metadata in _FORMATTER_FINGERPRINT_PROBES
        ]
        payload = "||".join(
            (
                str(self.model_profile.name),
                str(getattr(formatter, "__module__", "")),
                str(
                    getattr(
                        formatter,
                        "__qualname__",
                        getattr(formatter, "__name__", "formatter"),
                    )
                ),
                *outputs,
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _bind_model_contract(self, model_name_or_path: str) -> None:
        """Bind profile-derived formatting and runtime policy to one load candidate.

        :param str model_name_or_path: Hub identifier or local checkpoint path.
        :return None: Recomputes profile-dependent state when the contract changes.
        """
        resolved_profile = resolve_embedding_model_profile(
            model_name_or_path,
            self.requested_model_profile,
        )
        if resolved_profile == self.model_profile:
            return

        self.model_profile = resolved_profile
        self._document_formatter_fingerprint = self._resolve_formatter_fingerprint(
            formatter=self.model_profile.document_formatter,
            probe_renderer=self._format_retrieval_document_metadata,
        )
        self._similarity_formatter_fingerprint = self._resolve_formatter_fingerprint(
            formatter=self.model_profile.similarity_formatter,
            probe_renderer=self._format_graph_similarity_metadata,
        )
        self.truncate_dim = self._resolve_truncate_dim(self._requested_truncate_dim)
        self._source_dtype_hint = self._resolve_source_dtype_hint()
        self._attention_implementation_hint = (
            self._resolve_attention_implementation_hint()
        )
        self._embedding_cache = None
        self._graph_embedding_cache = None
        self._resolved_model_fingerprint = None
        self._autocast_dtype = None
        self._autocast_device_type = None
        self._autocast_enabled = False
        self._encode_model = None
        self._inner_model_compiled = False
        self._eager_inner_transformer = None
        self._compile_status_reason = None
        self._profile_logged = False
        self._dim_logged = False
        self._runtime_summary_logged = False

    def _resolve_attention_implementation_hint(self) -> Optional[str]:
        """Resolve preferred attention implementation for the resolved device.

        :return Optional[str]: Attention implementation token or ``None``.
        """
        if self.device == "cpu":
            return None

        preferred_attention = str(
            self.model_profile.preferred_attention_implementation or ""
        ).strip()
        if not preferred_attention:
            return None
        if preferred_attention != "flash_attention_2":
            return preferred_attention
        if (
            self.device == "cuda"
            and self._source_dtype_hint == "bfloat16"
            and _module_available("flash_attn")
        ):
            return preferred_attention
        # flash_attn ships CUDA-only kernels; the profile's portable fallback is SDPA.
        return "sdpa"

    def _resolve_source_dtype_hint(self) -> str:
        """Resolve effective compute dtype used for cache provenance metadata.

        :return str: Effective compute dtype token.
        """
        try:
            torch = _import_torch()
        except ImportError:
            return "float32"

        preferred_dtype = (self.model_profile.preferred_compute_dtype or "").lower()
        if preferred_dtype == "bfloat16" and self._bf16_autocast_allowed(torch):
            return "bfloat16"
        return "float32"

    def _bf16_autocast_allowed(self, torch: Any) -> bool:
        """Return whether the active device can execute bf16 through autocast.

        Model weights use the checkpoint/library's automatic dtype resolution.
        Reduced compute precision is enabled only through a verified autocast
        context; any missing or rejected runtime capability falls back to float32.

        :param Any torch: Imported torch module.
        :return bool: ``True`` when bf16 autocast is usable for the active device.
        """
        if self.device not in self.model_profile.autocast_devices:
            return False

        bf16_dtype = getattr(torch, "bfloat16", None)
        autocast = getattr(torch, "autocast", None)
        if bf16_dtype is None or not callable(autocast):
            logger.warning(
                "%s prefers bfloat16 autocast on %s, but this torch runtime does "
                "not expose the required APIs; falling back to float32.",
                self.model_name,
                self.device,
            )
            return False

        if self.device == "cpu":
            if not _cpu_native_bf16_supported(torch):
                return False
        elif self.device == "cuda":
            if not _cuda_native_bf16_supported(torch):
                return False
        elif self.device == "mps" and not self._mps_bf16_allowed(torch):
            return False

        try:
            with autocast(device_type=self.device, dtype=bf16_dtype):
                pass
        except Exception as exc:
            logger.warning(
                "%s prefers bfloat16 autocast on %s, but the runtime rejected "
                "that context (%s: %s); falling back to float32.",
                self.model_name,
                self.device,
                type(exc).__name__,
                exc,
            )
            return False
        return True

    def _mps_bf16_allowed(self, torch: Any) -> bool:
        """Return whether bf16 autocast is allowed on MPS for this torch build.

        :param Any torch: Imported ``torch`` module object.
        :return bool: ``True`` when torch meets the MPS bf16 policy floor.
        """
        torch_version = _parse_torch_major_minor(getattr(torch, "__version__", ""))
        if torch_version >= _MPS_MIN_TORCH_VERSION:
            return True
        logger.warning(
            "%s prefers bfloat16 on MPS, but torch %s predates the verified "
            "MPS floor %s; falling back to float32.",
            self.model_name,
            getattr(torch, "__version__", "unknown"),
            ".".join(str(part) for part in _MPS_MIN_TORCH_VERSION),
        )
        return False

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
            artifact_digest = self._resolve_inference_artifact_digest(resolved_path)
            fingerprint = (
                f"local::{resolved_path.resolve()}::artifact={artifact_digest}"
            )
            self._resolved_model_fingerprint = fingerprint
            return fingerprint

        model_id = model_identity
        if "/" not in model_id:
            raise RuntimeError(
                "Could not derive immutable artifact identity for embedding model alias "
                f"{model_id!r}; use a local checkpoint path or a full Hugging Face "
                "repository ID."
            )

        requested_revision = self._requested_hf_revision_token()
        if re.fullmatch(r"[0-9a-f]{40}", requested_revision, flags=re.IGNORECASE):
            fingerprint = f"hf::{model_id}::{requested_revision.lower()}"
            self._resolved_model_fingerprint = fingerprint
            return fingerprint
        resolved_sha = ""
        resolution_error: Optional[Exception] = None

        local_snapshot_sha = self._resolve_local_hf_snapshot_sha(
            model_id=model_id,
            requested_revision=requested_revision,
        )
        if local_snapshot_sha is not None:
            resolved_sha = local_snapshot_sha

        if not resolved_sha:
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
            local_artifact_fingerprint = self._resolve_local_hf_artifact_fingerprint(
                model_id=model_id,
                requested_revision=requested_revision,
            )
            if local_artifact_fingerprint is not None:
                logger.debug(
                    "Could not resolve Hugging Face commit SHA for %s (revision=%s). "
                    "Using a content fingerprint of the complete local inference "
                    "artifact manifest.",
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

    @staticmethod
    def _resolve_local_hf_snapshot_path(
        model_id: str, requested_revision: str
    ) -> Optional[Path]:
        """Resolve an existing Hugging Face snapshot without network access.

        :param str model_id: Hugging Face repository ID.
        :param str requested_revision: Requested model revision token.
        :return Optional[Path]: Resolved local snapshot directory, if available.
        """
        try:
            snapshot_download = _import_huggingface_hub_module().snapshot_download
            return Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=requested_revision,
                    local_files_only=True,
                )
            ).resolve()
        except Exception:
            return None

    def _resolve_local_hf_snapshot_sha(
        self, model_id: str, requested_revision: str
    ) -> Optional[str]:
        """Best-effort local SHA resolution from existing HF snapshot cache.

        :param str model_id: Hugging Face repository ID.
        :param str requested_revision: Requested model revision token.
        :return Optional[str]: Locally resolved snapshot SHA, if available.
        """
        snapshot_path = self._resolve_local_hf_snapshot_path(
            model_id, requested_revision
        )
        if snapshot_path is None:
            return None

        parts = snapshot_path.parts
        for idx, part in enumerate(parts):
            if part != "snapshots":
                continue
            if idx + 1 >= len(parts):
                continue
            candidate = str(parts[idx + 1]).strip()
            if re.fullmatch(r"[0-9a-f]{40}", candidate, flags=re.IGNORECASE):
                logger.debug(
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

    @staticmethod
    def _safe_artifact_reference(root: Path, source: Path, raw_path: object) -> Path:
        """Resolve one manifest reference without allowing lexical path escape.

        Hugging Face snapshot files may be symlinks into the blob store, so this
        validates the logical path before following the file rather than rejecting
        legitimate cache symlink targets.

        :param Path root: Artifact root directory.
        :param Path source: Manifest file containing the reference.
        :param object raw_path: Referenced relative path.
        :return Path: Validated logical path below ``root``.
        :raises RuntimeError: If the reference is absolute or escapes the root.
        """
        reference = str(raw_path or "").strip()
        if not reference:
            raise RuntimeError(f"Empty artifact reference in {source}.")
        candidate = source.parent / reference
        root_absolute = os.path.abspath(root)
        candidate_absolute = os.path.abspath(candidate)
        if os.path.commonpath([root_absolute, candidate_absolute]) != root_absolute:
            raise RuntimeError(
                f"Artifact reference {reference!r} in {source} escapes {root}."
            )
        return Path(candidate_absolute)

    @classmethod
    def _inference_artifact_paths(cls, artifact_root: Path) -> List[Path]:
        """Enumerate and validate inference-relevant checkpoint artifacts.

        :param Path artifact_root: Local model file or directory.
        :return List[Path]: Stable sorted artifact paths.
        :raises RuntimeError: If the layout is empty, malformed, or incomplete.
        """
        root = artifact_root.expanduser().resolve()
        if root.is_file():
            return [root]
        if not root.is_dir():
            raise RuntimeError(f"Local embedding model path is not readable: {root}")

        artifacts: Set[Path] = set()
        for directory, child_directories, file_names in os.walk(root):
            child_directories[:] = [
                name
                for name in child_directories
                if name.lower() not in _IGNORED_ARTIFACT_DIRECTORIES
                and not name.lower().startswith("checkpoint-")
            ]
            directory_path = Path(directory)
            for file_name in file_names:
                lowered = file_name.lower()
                if (
                    lowered in _IGNORED_ARTIFACT_FILES
                    or lowered.startswith("readme")
                    or lowered.startswith("license")
                ):
                    continue
                path = directory_path / file_name
                if path.suffix.lower() in _INFERENCE_ARTIFACT_SUFFIXES:
                    artifacts.add(path)

        modules_path = root / "modules.json"
        if modules_path.is_file():
            try:
                modules_payload = json.loads(modules_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Malformed SentenceTransformers modules.json: {exc}"
                ) from exc
            if not isinstance(modules_payload, list):
                raise RuntimeError("SentenceTransformers modules.json must be a list.")
            for module in modules_payload:
                if not isinstance(module, dict):
                    raise RuntimeError(
                        "SentenceTransformers modules.json entries must be objects."
                    )
                module_path = str(module.get("path", "") or "").strip()
                if not module_path:
                    continue
                resolved_module = cls._safe_artifact_reference(
                    root, modules_path, module_path
                )
                if not resolved_module.exists():
                    raise RuntimeError(
                        f"SentenceTransformers module path is missing: {module_path}"
                    )

        index_paths = [
            path for path in artifacts if path.name.lower().endswith(".index.json")
        ]
        for index_path in index_paths:
            try:
                index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Malformed weight index {index_path}: {exc}"
                ) from exc
            weight_map = (
                index_payload.get("weight_map")
                if isinstance(index_payload, dict)
                else None
            )
            if not isinstance(weight_map, dict) or not weight_map:
                raise RuntimeError(f"Weight index has no weight_map: {index_path}")
            for shard_name in sorted({str(value) for value in weight_map.values()}):
                shard_path = cls._safe_artifact_reference(root, index_path, shard_name)
                if not shard_path.is_file():
                    raise RuntimeError(
                        f"Weight index references a missing shard: {shard_name}"
                    )
                artifacts.add(shard_path)

        if not artifacts:
            raise RuntimeError(
                f"No inference-relevant artifacts found under local model path {root}."
            )
        return sorted(artifacts, key=lambda path: path.relative_to(root).as_posix())

    @classmethod
    def _resolve_inference_artifact_digest(cls, artifact_root: Path) -> str:
        """Hash a canonical manifest of inference-relevant model contents.

        :param Path artifact_root: Local model file or directory.
        :return str: SHA-256 manifest digest.
        """
        root = artifact_root.expanduser().resolve()
        paths = cls._inference_artifact_paths(root)
        digest = sha256()
        for path in paths:
            relative = (
                path.name if root.is_file() else path.relative_to(root).as_posix()
            )
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(cls._sha256_file(path).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _resolve_local_hf_artifact_fingerprint(
        self, model_id: str, requested_revision: str
    ) -> Optional[str]:
        """Best-effort local artifact fingerprint using config + weights files.

        :param str model_id: Hugging Face repository ID.
        :param str requested_revision: Requested model revision token.
        :return Optional[str]: Deterministic local artifact fingerprint if available.
        """
        snapshot_path = self._resolve_local_hf_snapshot_path(
            model_id, requested_revision
        )
        if snapshot_path is None:
            return None

        try:
            artifact_digest = self._resolve_inference_artifact_digest(snapshot_path)
        except (OSError, RuntimeError):
            return None

        return (
            f"hf::{model_id}::revision={requested_revision}::artifact={artifact_digest}"
        )

    def _ensure_cache_model_fingerprint(
        self,
        *,
        representation: str = _RETRIEVAL_DOCUMENT_REPRESENTATION,
    ) -> None:
        """Verify one task cache is bound to the active model fingerprint.

        :param str representation: Retrieval-document or graph-similarity role.
        :return None: Validates and records model identity on the selected cache.
        """
        try:
            model_fingerprint = self._resolve_model_fingerprint()
        except Exception as exc:
            raise RuntimeError(
                "Could not verify the runtime-active embedding model identity; "
                "refusing persistent cache access instead of adopting an "
                f"unverified payload for {self._cache_model_identity()!r}."
            ) from exc

        self._resolved_model_fingerprint = model_fingerprint
        self._bind_embedding_cache_to_active_model()
        if representation == _RETRIEVAL_DOCUMENT_REPRESENTATION:
            cache = self.embedding_cache
        elif representation == _GRAPH_SIMILARITY_REPRESENTATION:
            cache = self.graph_embedding_cache
        else:
            raise ValueError(
                f"Unsupported embedding cache representation: {representation}"
            )
        has_cached_payload = cache.has_cached_payload()
        cached_fingerprint = cache.get_model_fingerprint()
        if has_cached_payload and cached_fingerprint != model_fingerprint:
            logger.warning(
                "Embedding cache model fingerprint mismatch (cached=%s, active=%s). "
                "Clearing namespace cache.",
                cached_fingerprint or "missing",
                model_fingerprint,
            )
            cache.clear(
                reason=(
                    "model fingerprint mismatch "
                    f"(cached={cached_fingerprint or 'missing'}, "
                    f"active={model_fingerprint})"
                )
            )
        if not has_cached_payload or cached_fingerprint != model_fingerprint:
            cache.set_model_fingerprint(model_fingerprint)

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

    def _resolve_model_kwargs(self) -> Dict[str, Any]:
        """Compute SentenceTransformer kwargs and configure autocast policy.

        :return Dict[str, Any]: ``SentenceTransformer`` constructor kwargs.
        """
        self._autocast_dtype = None
        self._autocast_device_type = None
        self._autocast_enabled = False
        self._encode_model = None
        model_kwargs: Dict[str, Any] = {"dtype": "auto"}
        if self._attention_implementation_hint is not None:
            model_kwargs["attn_implementation"] = self._attention_implementation_hint

        try:
            torch = _import_torch()
        except ImportError:
            self._source_dtype_hint = "float32"
            return model_kwargs

        if self._source_dtype_hint != "bfloat16":
            return model_kwargs

        self._autocast_dtype = torch.bfloat16
        self._autocast_device_type = self.device
        self._autocast_enabled = True
        logger.debug(
            "%s will load weights with automatic dtype selection and run with "
            "bfloat16 autocast on %s.",
            self.model_name,
            self.device,
        )

        return model_kwargs

    def _validate_loaded_model_precision(
        self,
        model: Any,
        model_name_or_path: str,
    ) -> None:
        """Reject automatic checkpoint dtypes outside the verified compute policy.

        CiteMesh deliberately keeps Transformers automatic checkpoint loading so
        compatible weights are not needlessly coerced. Live parameters and buffers
        are therefore the authoritative dtype check: float16 is forbidden
        everywhere, and bfloat16 tensors are accepted only when this runtime has
        already selected the verified bfloat16 autocast path.

        :param Any model: Newly loaded SentenceTransformer-compatible model.
        :param str model_name_or_path: Candidate checkpoint that produced ``model``.
        :return None: The model's automatic dtype is compatible.
        :raises EmbeddingPrecisionCompatibilityError: If live tensors resolve to an
            unsupported precision.
        """
        tensor_dtypes = _model_floating_dtype_names(model)
        if not tensor_dtypes:
            raise EmbeddingPrecisionCompatibilityError(
                f"Could not inspect live parameter and buffer dtypes for embedding "
                f"checkpoint {model_name_or_path!r}; automatic precision cannot be verified."
            )
        if "float16" in tensor_dtypes:
            raise EmbeddingPrecisionCompatibilityError(
                f"Embedding checkpoint {model_name_or_path!r} resolved float16 tensors "
                "under automatic dtype loading. CiteMesh forbids float16; choose a "
                "current checkpoint whose saved weights are float32 or bfloat16."
            )
        unsupported_dtypes = tensor_dtypes - {"float32", "bfloat16"}
        if unsupported_dtypes:
            raise EmbeddingPrecisionCompatibilityError(
                f"Embedding checkpoint {model_name_or_path!r} resolved unsupported "
                "automatic tensor dtype(s): "
                f"{', '.join(sorted(unsupported_dtypes))}. CiteMesh supports only "
                "float32 weights or bfloat16 weights on a verified bfloat16 runtime."
            )
        if "bfloat16" in tensor_dtypes and self._source_dtype_hint != "bfloat16":
            raise EmbeddingPrecisionCompatibilityError(
                f"Embedding checkpoint {model_name_or_path!r} resolved bfloat16 tensors "
                f"on device={self.device}, but this runtime has not verified bfloat16 "
                "compute for the active model profile. Choose a float32 checkpoint "
                "instead of mixing bfloat16 execution into a float32 cache namespace."
            )
        logger.debug(
            "%s automatic parameter/buffer dtype(s): %s.",
            model_name_or_path,
            ", ".join(sorted(tensor_dtypes)),
        )

    def _validate_loaded_model_contract(
        self,
        model: Any,
        model_name_or_path: str,
    ) -> None:
        """Require live transformer settings declared by the active model profile.

        :param Any model: Newly loaded SentenceTransformer-compatible model.
        :param str model_name_or_path: Candidate checkpoint that produced ``model``.
        :return None: The runtime-active transformer satisfies its profile.
        :raises EmbeddingBackendCompatibilityError: If a required live setting
            is absent, inaccessible, or disabled.
        """
        if not self.model_profile.requires_bidirectional_attention:
            return

        try:
            active_config = model[0].auto_model.config
            bidirectional_attention = active_config.use_bidirectional_attention
        except Exception as exc:
            raise EmbeddingBackendCompatibilityError(
                f"Embedding checkpoint {model_name_or_path!r} requires bidirectional "
                "attention, but the runtime-active setting at "
                "model[0].auto_model.config.use_bidirectional_attention could not "
                "be verified."
            ) from exc

        if bidirectional_attention is not True:
            raise EmbeddingBackendCompatibilityError(
                f"Embedding checkpoint {model_name_or_path!r} requires bidirectional "
                "attention, but the runtime-active transformer reports "
                f"use_bidirectional_attention={bidirectional_attention!r}."
            )

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

    @contextmanager
    def _tf32_context(self) -> Iterator[None]:
        """Apply CUDA-only TF32 controls for one encode call and restore them.

        :return Iterator[None]: Context that scopes process-global torch settings.
        """
        if self._tf32_mode not in {"tf32", "tf32-matmul-high"}:
            yield
            return

        try:
            torch = _import_torch()
        except ImportError:
            yield
            return

        if self._tf32_mode == "tf32-matmul-high":
            get_precision = getattr(torch, "get_float32_matmul_precision", None)
            set_precision = getattr(torch, "set_float32_matmul_precision", None)
            if not callable(get_precision) or not callable(set_precision):
                yield
                return
            original = get_precision()
            try:
                set_precision("high")
            except Exception:
                logger.warning("Could not enable scoped CUDA TF32 matmul precision.")
                yield
                return
            try:
                yield
            finally:
                try:
                    set_precision(original)
                except Exception:
                    logger.warning(
                        "Could not restore the prior torch float32 matmul precision."
                    )
            return

        backends = getattr(torch, "backends", None)
        cuda_backend = getattr(backends, "cuda", None)
        cudnn_backend = getattr(backends, "cudnn", None)
        targets = (
            getattr(cuda_backend, "matmul", None),
            getattr(cudnn_backend, "conv", None),
        )
        if any(
            target is None or not hasattr(target, "fp32_precision")
            for target in targets
        ):
            yield
            return

        originals = [(target, target.fp32_precision) for target in targets]
        changed: list[tuple[Any, Any]] = []
        try:
            for target, original in originals:
                target.fp32_precision = "tf32"
                changed.append((target, original))
        except Exception:
            for target, original in reversed(changed):
                try:
                    target.fp32_precision = original
                except Exception:
                    pass
            logger.warning("Could not enable scoped CUDA TF32 backend precision.")
            yield
            return

        try:
            yield
        finally:
            for target, original in reversed(originals):
                try:
                    target.fp32_precision = original
                except Exception:
                    logger.warning(
                        "Could not restore a prior CUDA TF32 backend setting."
                    )

    @contextmanager
    def _precision_context(self) -> Iterator[None]:
        """Combine scoped TF32 and autocast controls around model encoding.

        :return Iterator[None]: Active encode-time precision context.
        """
        with ExitStack() as stack:
            stack.enter_context(self._tf32_context())
            stack.enter_context(self._autocast_context())
            if self.device == "cpu" and self._inner_model_compiled:
                # Dynamo must see freezing while capturing weights; backend-only
                # compile options arrive too late. Keep eager fallback weights.
                stack.enter_context(
                    _import_torch()._inductor.config.patch(
                        freezing=True, freezing_discard_parameters=False
                    )
                )
            if (
                self._inner_model_compiled
                and self._attention_implementation_hint == "flash_attention_2"
            ):
                from torch._dynamo import config as dynamo_config
                from transformers.modeling_flash_attention_utils import (
                    logger as flash_logger,
                )

                # Transformers logs an autocast conversion inside attention.
                # Tracing that log breaks the decoder loop into eager fragments.
                logging_option = (
                    "ignore_logging_functions"
                    if hasattr(dynamo_config, "ignore_logging_functions")
                    else "reorderable_logging_functions"
                )
                stack.enter_context(
                    dynamo_config.patch(
                        **{
                            logging_option: getattr(dynamo_config, logging_option)
                            | {flash_logger.warning_once}
                        }
                    )
                )
            yield

    def _get_model_for_encoding(self) -> Any:
        """Return model object used for embedding encode calls.

        :return Any: Base model or encode-time precision proxy.
        """
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded.")

        if (
            not self._inner_model_compiled
            and not self._autocast_enabled
            and self._tf32_mode not in {"tf32", "tf32-matmul-high"}
        ):
            return self.model

        if self._encode_model is None:
            self._encode_model = _PrecisionEncodeProxy(
                self.model,
                self._precision_context,
                self._restore_eager_model_after_compile_failure,
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
        if self.device in {"cuda", "cpu"}:
            return False
        if self.semantic_source != "arxiv-corpus":
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
            load_candidates = self._model_load_candidates()
            model_errors: List[Tuple[str, Exception]] = []
            for idx, candidate_model in enumerate(load_candidates):
                try:
                    self._bind_model_contract(candidate_model)
                    _require_transformers_compatibility(self.model_profile)
                    model_kwargs = self._resolve_model_kwargs()
                    st_kwargs: Dict[str, Any] = {"device": self.device}
                    if model_kwargs:
                        st_kwargs["model_kwargs"] = model_kwargs
                    if self.truncate_dim is not None:
                        st_kwargs["truncate_dim"] = self.truncate_dim
                    if self.model_revision is not None:
                        st_kwargs["revision"] = self.model_revision
                    try:
                        with _suppress_expected_fa2_load_dtype_warning(
                            enabled=(
                                self._attention_implementation_hint
                                == "flash_attention_2"
                                and self._autocast_enabled
                                and self._autocast_device_type == "cuda"
                            )
                        ):
                            loaded_model = sentence_transformer_cls(
                                candidate_model, **st_kwargs
                            )
                    except Exception as exc:
                        if self._attention_implementation_hint != "flash_attention_2":
                            raise
                        logger.warning(
                            "FlashAttention 2 could not load for %s (%s); retrying with SDPA.",
                            candidate_model,
                            exc,
                        )
                        self._attention_implementation_hint = "sdpa"
                        model_kwargs["attn_implementation"] = "sdpa"
                        loaded_model = sentence_transformer_cls(
                            candidate_model, **st_kwargs
                        )
                    self._validate_loaded_model_precision(loaded_model, candidate_model)
                    self._validate_loaded_model_contract(loaded_model, candidate_model)
                    self.model = loaded_model
                except (
                    EmbeddingBackendCompatibilityError,
                    EmbeddingPrecisionCompatibilityError,
                ):
                    raise
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
                    summary = "; ".join(
                        f"{model_id}: {type(error).__name__}: {error}"
                        for model_id, error in model_errors
                    )
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
                self._active_model_name = candidate_model
                break

            self._bind_embedding_cache_to_active_model()
            self._configure_tf32_runtime()
            if (
                self.enable_torch_compile
                and self.semantic_source == "arxiv-corpus"
                and self.device == "mps"
            ):
                # Warm-cache compile policy needs the exact artifact namespace.
                self._ensure_cache_model_fingerprint()
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

        if self.device != "cuda":
            logger.debug(
                "Skipping TF32 config for %s: device=%s is not CUDA.",
                self.model_name,
                self.device,
            )
            return

        try:
            torch = _import_torch()
        except ImportError:
            logger.debug(
                "Skipping TF32 config for %s: torch unavailable.", self.model_name
            )
            return

        cuda_module = getattr(torch, "cuda", None)
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
            and torch_version in _TF32_COMPILE_BRIDGE_TORCH_VERSIONS
        )
        if should_use_compile_bridge:
            get_matmul_precision = getattr(torch, "get_float32_matmul_precision", None)
            set_matmul_precision = getattr(torch, "set_float32_matmul_precision", None)
            if not callable(get_matmul_precision) or not callable(set_matmul_precision):
                self._tf32_mode = "unsupported"
                logger.debug(
                    "Skipping TF32 config for %s: restorable compile-safe matmul "
                    "precision APIs unavailable.",
                    self.model_name,
                )
                return
            self._tf32_mode = "tf32-matmul-high"
            logger.debug(
                "Will scope compile-safe TF32 matmul precision to encode calls for "
                "%s on torch %s.",
                self.model_name,
                getattr(torch, "__version__", "unknown"),
            )
            return

        backends = getattr(torch, "backends", None)
        cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
        cudnn_backend = (
            getattr(backends, "cudnn", None) if backends is not None else None
        )
        matmul_backend = getattr(cuda_backend, "matmul", None)
        conv_backend = getattr(cudnn_backend, "conv", None)
        if any(
            owner is None or not hasattr(owner, "fp32_precision")
            for owner in (matmul_backend, conv_backend)
        ):
            self._tf32_mode = "unsupported"
            logger.debug(
                "Skipping TF32 config for %s: CUDA-scoped fp32_precision APIs "
                "unavailable.",
                self.model_name,
            )
            return

        self._tf32_mode = "tf32"
        logger.debug("Will scope TF32 to CUDA matmul/conv encode calls (Ampere+ GPU).")

    def _log_runtime_summary(self) -> None:
        """Emit concise one-time runtime summary at info level."""
        if self._runtime_summary_logged:
            return

        selected_dim = self.truncate_dim
        if selected_dim is None and self.model_profile.available_truncate_dims:
            selected_dim = self.model_profile.available_truncate_dims[0]
        dim_label = "full" if selected_dim is None else f"{selected_dim}d"
        compute_dtype_label = self._source_dtype_hint
        if self._autocast_enabled:
            compute_dtype_label = f"{compute_dtype_label}+autocast"
        attention_label = self._attention_implementation_hint or "auto"
        logger.info(
            "%s runtime: device=%s, dim=%s, compute=%s, attn=%s, output=float32, cache=%s, compile=%s, tf32=%s.",
            self.model_name,
            self.device,
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
        the active inner transformer and keep the outer SentenceTransformer intact.

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

        if self.device not in _COMPILE_ELIGIBLE_DEVICES:
            self._compile_status_reason = f"compile disabled for device={self.device}"
            logger.debug(
                "Skipping torch.compile for %s: device=%s is not compile-eligible.",
                self.model_name,
                self.device,
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

        # SentenceTransformers 6 forwards through .model; assigning its legacy
        # .auto_model alias can register an unused module instead.
        model_attribute = (
            "model" if hasattr(transformer_block, "model") else "auto_model"
        )
        auto_model = getattr(transformer_block, model_attribute, None)
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
            compile_kwargs: Dict[str, Any] = (
                {"dynamic": True} if self.device in {"cuda", "cpu"} else {}
            )
            if self.device == "cpu":
                compile_kwargs["options"] = {"max_autotune": True}
            compiled_model = compile_fn(auto_model, **compile_kwargs)
            setattr(transformer_block, model_attribute, compiled_model)
        except Exception as exc:
            try:
                setattr(transformer_block, model_attribute, auto_model)
            except Exception:
                pass
            self._compile_status_reason = f"compile failed ({type(exc).__name__})"
            logger.warning(
                "torch.compile failed for %s inner transformer; continuing without compile: %s",
                self.model_name,
                exc,
            )
            return

        self._eager_inner_transformer = auto_model
        self._inner_model_compiled = True
        self._compile_status_reason = None
        if self.device == "mps":
            logger.info(
                "torch.compile on MPS (Inductor/Metal) is experimental; the first "
                "encode will retry from the eager inner model if compilation fails."
            )
        logger.debug(
            "Enabled torch.compile for %s inner transformer (model[0].%s).",
            self.model_name,
            model_attribute,
        )

    def _restore_eager_model_after_compile_failure(self, error: Exception) -> bool:
        """Restore the original inner model after a lazy compiled-call failure.

        ``torch.compile`` normally defers backend compilation until the first
        invocation, so wrapping the module successfully is not proof that it can
        execute. Retrying the failed encode batch is safe: cache writes happen
        only after all batches finish encoding successfully.

        :param Exception error: Failure raised while the compiled model executed.
        :return bool: Whether an eager model was restored for one retry.
        """
        if (
            not self._inner_model_compiled
            or self._eager_inner_transformer is None
            or self.model is None
        ):
            return False
        try:
            transformer_block = self.model[0]
            model_attribute = (
                "model" if hasattr(transformer_block, "model") else "auto_model"
            )
            setattr(transformer_block, model_attribute, self._eager_inner_transformer)
        except Exception:
            logger.warning(
                "Compiled embedding execution failed and the eager inner model "
                "could not be restored.",
                exc_info=True,
            )
            return False

        self._inner_model_compiled = False
        self._eager_inner_transformer = None
        self._encode_model = None
        self._compile_status_reason = (
            f"compiled execution failed ({type(error).__name__}); restored eager model"
        )
        logger.warning(
            "Compiled embedding execution failed for %s (%s: %s); retrying the "
            "affected encode batch with the eager inner model.",
            self.model_name,
            type(error).__name__,
            error,
        )
        return True

    def collect_papers(
        self,
        seed_id: str,
        *,
        seed_paper: Optional[Paper] = None,
        **kwargs: Any,
    ) -> Dict[str, Paper]:
        """
        Collect papers via semantic similarity search.

        :param str seed_id: Seed paper identifier (ArXiv ID or text query)
        :param Optional[Paper] seed_paper: Optional pre-fetched seed paper metadata to
            reuse instead of fetching the seed from Semantic Scholar again.
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Dict[str, Paper]: Dictionary of paper_id -> Paper objects
        """
        papers: Dict[str, Paper] = {}
        self.retrieval_embeddings = {}
        self.embeddings = {}
        self.candidate_source_status = {}

        # Load model lazily.
        self._load_model()

        # Reuse caller-provided seed metadata when available to avoid redundant
        # Semantic Scholar fetches in hybrid mode.
        resolved_seed_paper = seed_paper
        if resolved_seed_paper is None:
            resolved_seed_paper = self.client.get_paper(
                seed_id, raise_on_unavailable=True
            )

        if resolved_seed_paper:
            # Found via S2 API
            resolved_seed_paper.is_seed = True
            papers[resolved_seed_paper.paper_id] = resolved_seed_paper
        else:
            # Treat as text query
            logger.info(f"Using '{seed_id}' as text query")
            # Create dummy seed paper
            query_seed = _query_seed_id(seed_id)
            resolved_seed_paper = Paper(
                paper_id=query_seed,
                title=seed_id,
                year=None,
                is_seed=True,
            )
            papers[query_seed] = resolved_seed_paper

        seed_identities = IdentityRegistry()
        register_aliases(
            seed_identities,
            resolved_seed_paper.paper_id,
            resolved_seed_paper,
        )

        # Compute normalized seed embedding
        logger.debug("Computing seed embedding...")
        formatted_seed_text = format_paper_for_embedding(
            profile=self.model_profile,
            paper=resolved_seed_paper,
            task=EmbeddingTask.RETRIEVAL_QUERY,
        )
        seed_embedding = self._encode_texts(
            [formatted_seed_text], show_progress_bar=False
        )[0]
        self.retrieval_embeddings[resolved_seed_paper.paper_id] = seed_embedding

        if self.semantic_source != "arxiv-corpus":
            logger.debug("Using candidate-pool semantic search...")
            pool_candidates = self._select_candidates_from_pool(
                seed_embedding, resolved_seed_paper
            )
            for paper_id, paper, embedding in pool_candidates:
                if len(papers) >= self.max_papers:
                    break
                if paper_id in papers or resolve_aliases(seed_identities, paper):
                    continue
                papers[paper_id] = paper
                self.retrieval_embeddings[paper_id] = embedding
            return papers

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

            if resolve_aliases(seed_identities, paper):
                continue

            papers[paper_id] = paper
            self.retrieval_embeddings[paper_id] = embedding

        self._update_citation_counts(papers)
        return papers

    def _candidate_pool_budgets(self) -> Tuple[int, int, int]:
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
    ) -> List[Tuple[str, Paper, np.ndarray]]:
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
        scored: List[Tuple[float, str, Paper, np.ndarray, int]] = []
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

    def embed_papers(self, papers: Dict[str, Paper]) -> Dict[str, np.ndarray]:
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

    def _format_retrieval_document_metadata(self, metadata: Dict[str, object]) -> str:
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

    def _format_graph_similarity_metadata(self, metadata: Dict[str, object]) -> str:
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
        self, papers: Dict[str, Paper]
    ) -> Dict[str, np.ndarray]:
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

    def prepare_graph_scoring(self, papers: Dict[str, Paper]) -> None:
        """Populate symmetric vectors immediately before pairwise edge scoring.

        :param Dict[str, Paper] papers: Final selected graph papers.
        :return None: Materializes the graph-similarity cache and vector map.
        """
        self.materialize_graph_embeddings(papers)

    def search_local(self, query: str, top_k: int) -> List[CacheSearchResult]:
        """Semantically search this builder's persistent embedding cache.

        Encodes the free-text query in the model's query prompt space and
        ranks it against every embedding already persisted in the cache
        namespace (candidate-mode vectors accumulated across builds, or a
        hydrated corpus). Purely local except for the one-time model download.

        :param str query: Free-text search query.
        :param int top_k: Number of results to return.
        :return List[CacheSearchResult]: Ranked results with cached metadata.
        :raises ValueError: If the query is empty or ``top_k`` is below 1.
        """
        normalized_query = str(query).strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if int(top_k) < 1:
            raise ValueError("top_k must be at least 1")
        self.prepare_embedding_cache()
        query_text = self.model_profile.format_query(normalized_query, {})
        query_embedding = self._encode_texts([query_text])[0]
        return self.embedding_cache.search(
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
        """Return whether any physical vector cache exists under this cache root.

        This cheap probe lets auto local-search avoid loading a model when the
        user has never built an embedding cache. Exact namespace selection still
        occurs before any vector is searched.

        :return bool: Whether at least one HDF5 embedding payload exists.
        """
        cache_directory = (
            self._embedding_cache.h5_path.parent
            if self._embedding_cache is not None
            else get_cache_dir("embeddings", create=False)
        )
        return any(cache_directory.glob("embeddings_*.h5"))

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
            self._refresh_cached_corpus_metadata(cached_dataset_source, use_streaming)
            self._refresh_hydrated_full_corpus_cache(
                use_streaming=use_streaming,
                cached_dataset_source=cached_dataset_source,
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

        if self._resume_incomplete_full_corpus_cache(
            use_streaming=use_streaming,
            cached_dataset_source=cached_dataset_source,
        ):
            self._refresh_cached_corpus_metadata(cached_dataset_source, use_streaming)
            return

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
            self._refresh_cached_corpus_metadata(dataset_source, use_streaming)
            return

        logger.info(
            "Hydrating embedding cache for split=%s corpus_size=%s (streaming=%s).",
            self.dataset_split,
            "all" if self.corpus_size is None else self.corpus_size,
            use_streaming,
        )
        if self.corpus_size is not None and ":" not in str(self.dataset_split):
            logger.info(
                "Capped corpus hydration selects the %d most recently "
                "submitted papers (by arXiv ID chronology). Use --all-corpus "
                "for full coverage or an explicit --dataset-split slice for "
                "a custom positional window.",
                int(self.corpus_size),
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

        self.embedding_cache.mark_corpus_metadata_current()
        if self.corpus_size is None and ":" not in str(self.dataset_split):
            upstream_rows = self._resolve_dataset_split_row_count(dataset_source)
            updated_rows = self._cached_payload_row_count()
            rows_reconciled = self._finalize_full_corpus_hydration_rows(
                source=dataset_source,
                updated_rows=updated_rows,
                upstream_rows=upstream_rows,
                mark_complete=True,
            )
            if not rows_reconciled:
                logger.info(
                    "Initial full-corpus hydration for %s/%s completed with "
                    "cache_rows=%d and upstream_rows=%d; recording the expected "
                    "duplicate-ID row-count deficit.",
                    dataset_source,
                    self.dataset_split,
                    updated_rows,
                    upstream_rows,
                )
            return

        self.embedding_cache.mark_hydrated(
            dataset_source=dataset_source,
            dataset_split=self.dataset_split,
            corpus_size=self.corpus_size,
            complete=True,
        )

    def _refresh_cached_corpus_metadata(
        self, source: Optional[str], use_streaming: bool
    ) -> None:
        """Backfill corpus years and DOIs without changing persisted vectors.

        :param Optional[str] source: Dataset recorded on the matching cache.
        :param bool use_streaming: Whether source rows should be streamed.
        :return None: Refreshes existing SQLite rows once after an adapter change.
        """
        cache = self.embedding_cache
        if (
            not source
            or cache.has_current_corpus_metadata()
            or not cache.has_cached_payload()
        ):
            return

        logger.info("Refreshing cached publication years and DOIs from %s.", source)
        # A capped cache retains its original paper selection. Inspect the full
        # selected split so older cached papers can still receive metadata fixes.
        dataset = _import_datasets_module().load_dataset(
            source, split=self.dataset_split, streaming=use_streaming
        )
        batch: List[Dict] = []
        for index, record in enumerate(dataset):
            batch.append(_extract_dataset_paper_metadata(record, index))
            if len(batch) >= HYDRATION_FLUSH_SIZE:
                cache.update_corpus_metadata(batch)
                batch = []
        if batch:
            cache.update_corpus_metadata(batch)
        # Older resume code could memoize a deficit without reconciling IDs.
        cache.clear_hydration_rowcount_reconciliation()
        cache.mark_corpus_metadata_current()

    def _resume_incomplete_full_corpus_cache(
        self,
        *,
        use_streaming: bool,
        cached_dataset_source: Optional[str],
    ) -> bool:
        """Resume an incomplete full-corpus hydration when cached rows are reusable.

        :param bool use_streaming: Whether hydration mode is streaming.
        :param Optional[str] cached_dataset_source: Dataset source recorded on the
            incomplete cache attempt.
        :return bool: ``True`` when the incomplete cache was resumed or safely
            retained without requiring a full namespace clear.
        """
        if self.corpus_size is not None:
            return False
        if ":" in str(self.dataset_split):
            return False

        source = str(cached_dataset_source or "").strip()
        if not source:
            return False

        stats = self.embedding_cache.payload_stats()
        if stats.hydration_complete:
            return False
        if stats.hydration_split != self.dataset_split:
            return False
        if stats.hydration_corpus_size != "all":
            return False
        if stats.hydration_dataset_source != source:
            return False
        if stats.sqlite_rows < 1 or stats.embedding_rows < 1:
            return False
        if stats.sqlite_rows != stats.embedding_rows:
            logger.warning(
                "Incomplete full-corpus cache rows diverged for %s/%s "
                "(sqlite_rows=%d, embedding_rows=%d); performing full rebuild.",
                source,
                self.dataset_split,
                stats.sqlite_rows,
                stats.embedding_rows,
            )
            return False
        if (
            self.storage_precision == "int8"
            and not self.embedding_cache.has_calibration_ranges()
        ):
            logger.warning(
                "Incomplete full-corpus cache for %s/%s is missing int8 calibration "
                "ranges; performing full rebuild.",
                source,
                self.dataset_split,
            )
            return False

        cached_rows = int(stats.sqlite_rows)
        upstream_rows = self._resolve_dataset_split_row_count(source)
        if upstream_rows is not None:
            if upstream_rows < cached_rows:
                logger.warning(
                    "Incomplete full-corpus cache rows (%d) exceed upstream split rows "
                    "(%d) for %s/%s; performing full rebuild.",
                    cached_rows,
                    upstream_rows,
                    source,
                    self.dataset_split,
                )
                return False
            if upstream_rows == cached_rows:
                logger.info(
                    "Incomplete full-corpus cache for %s/%s already matches upstream "
                    "row count (%d); marking hydration complete.",
                    source,
                    self.dataset_split,
                    cached_rows,
                )
                self._finalize_full_corpus_hydration_rows(
                    source=source,
                    updated_rows=cached_rows,
                    upstream_rows=upstream_rows,
                    mark_complete=True,
                )
                return True

        row_limit = None if upstream_rows is None else upstream_rows - cached_rows
        logger.info(
            "Resuming incomplete full-corpus cache for %s/%s from cached_rows=%d%s.",
            source,
            self.dataset_split,
            cached_rows,
            "" if upstream_rows is None else f" toward upstream_rows={upstream_rows}",
        )
        self._ensure_int8_calibration_ranges(
            use_streaming=use_streaming,
            dataset_source=source,
        )
        resume_result = self._hydrate_exact_hydration_source_slice(
            use_streaming=use_streaming,
            source=source,
            row_limit=row_limit,
            row_offset=cached_rows,
            progress_total=row_limit,
            progress_label=f"Resuming {source}",
            operation="Incomplete hydration resume",
        )
        resumed_records = resume_result.hydrated_records
        updated_rows = self._cached_payload_row_count()
        if updated_rows < cached_rows:
            raise RuntimeError(
                "Incomplete hydration resume reduced cached row count unexpectedly "
                f"({updated_rows} < {cached_rows})."
            )

        if upstream_rows is not None:
            expected_source_rows = int(row_limit or 0)
            if (
                not resume_result.source_exhausted
                or resume_result.source_rows_consumed != expected_source_rows
            ):
                logger.warning(
                    "Incomplete full-corpus resume for %s/%s consumed %d of %d "
                    "expected source rows (slice_exhausted=%s); performing full "
                    "rebuild.",
                    source,
                    self.dataset_split,
                    resume_result.source_rows_consumed,
                    expected_source_rows,
                    resume_result.source_exhausted,
                )
                return False
        elif not resume_result.source_exhausted:
            logger.warning(
                "Incomplete full-corpus resume for %s/%s did not exhaust its "
                "unknown-cardinality source slice; performing full revalidation.",
                source,
                self.dataset_split,
            )
            return False

        # Cached unique IDs are not a source offset after reordered growth.
        # Exhausting the tail alone cannot establish a duplicate-ID deficit.
        if upstream_rows is not None and updated_rows < upstream_rows:
            reconciled = self._hydrate_exact_hydration_source_slice(
                use_streaming=use_streaming,
                source=source,
                progress_total=upstream_rows,
                progress_label=f"Reconciling {source}",
                operation="Resume missing-ID reconciliation",
                existing_paper_ids=self.embedding_cache.get_cached_paper_ids(),
            )
            resumed_records += reconciled.hydrated_records
            updated_rows = self._cached_payload_row_count()

        rows_reconciled = self._finalize_full_corpus_hydration_rows(
            source=source,
            updated_rows=updated_rows,
            upstream_rows=upstream_rows,
            mark_complete=True,
        )
        if not rows_reconciled:
            logger.info(
                "Incomplete full-corpus resume for %s/%s exhausted its expected "
                "source and reconciled missing IDs with cache_rows=%d and upstream_rows=%d; recording "
                "the duplicate/invalid-ID row-count deficit.",
                source,
                self.dataset_split,
                updated_rows,
                upstream_rows,
            )
        logger.info(
            "Resumed incomplete full-corpus cache for %s/%s "
            "(source_rows=%d, added=%d, cache_rows=%d).",
            source,
            self.dataset_split,
            resume_result.source_rows_consumed,
            resumed_records,
            updated_rows,
        )
        return True

    def _load_exact_hydration_source_slice(
        self,
        *,
        use_streaming: bool,
        source: str,
        operation: str,
        row_limit: Optional[int] = None,
        row_offset: Optional[int] = None,
    ) -> Iterable[Dict[str, Any]]:
        """Load a hydration slice while requiring the recorded source exactly.

        :param bool use_streaming: Whether to load a streaming dataset iterator.
        :param str source: Previously recorded dataset source to load.
        :param str operation: Caller-facing operation name for mismatch errors.
        :param Optional[int] row_limit: Optional number of rows to load.
        :param Optional[int] row_offset: Optional source row offset.
        :return Iterable[Dict[str, Any]]: The exact-source dataset slice.
        :raises RuntimeError: If the loader resolves a different source.
        """
        resolved_source, dataset = self._load_dataset_for_hydration(
            use_streaming=use_streaming,
            preferred_dataset_source=source,
            row_limit=row_limit,
            row_offset=row_offset,
            allow_source_fallback=False,
        )
        if resolved_source != source:
            raise RuntimeError(
                f"{operation} resolved unexpected dataset source {resolved_source!r} "
                f"(expected {source!r})."
            )
        return dataset

    def _hydrate_exact_hydration_source_slice(
        self,
        *,
        use_streaming: bool,
        source: str,
        progress_total: Optional[int],
        progress_label: str,
        operation: str,
        row_limit: Optional[int] = None,
        row_offset: Optional[int] = None,
        existing_paper_ids: Optional[Set[str]] = None,
        max_new_records: Optional[int] = None,
    ) -> _HydrationSourceSliceResult:
        """Load an exact-source slice and report cache and source progress.

        Exceptions from source iteration or cache writes propagate before a
        result is returned, so callers cannot mistake a failed pass for clean EOF.

        :param bool use_streaming: Whether to load a streaming dataset iterator.
        :param str source: Previously recorded dataset source to load.
        :param Optional[int] progress_total: Expected row count for progress display.
        :param str progress_label: Progress-bar description label.
        :param str operation: Caller-facing operation name for mismatch errors.
        :param Optional[int] row_limit: Optional number of rows to load.
        :param Optional[int] row_offset: Optional source row offset.
        :param Optional[Set[str]] existing_paper_ids: IDs to skip during reconciliation.
        :param Optional[int] max_new_records: Optional cap on newly hydrated records.
        :return _HydrationSourceSliceResult: Cache writes and source-consumption state.
        """
        dataset = self._load_exact_hydration_source_slice(
            use_streaming=use_streaming,
            source=source,
            operation=operation,
            row_limit=row_limit,
            row_offset=row_offset,
        )
        source_rows_consumed = 0
        source_exhausted = False

        def tracked_dataset() -> Iterable[Dict[str, Any]]:
            """Yield source rows while recording clean iterator exhaustion.

            :return Iterable[Dict[str, Any]]: Tracked source records.
            """
            nonlocal source_rows_consumed, source_exhausted
            iterator = iter(dataset)
            while True:
                try:
                    raw_record = next(iterator)
                except StopIteration:
                    source_exhausted = True
                    return
                source_rows_consumed += 1
                yield raw_record

        hydrated_records = self._hydrate_dataset_records(
            dataset=tracked_dataset(),
            progress_total=progress_total,
            progress_label=progress_label,
            existing_paper_ids=existing_paper_ids,
            max_new_records=max_new_records,
            fallback_index_offset=int(row_offset or 0),
        )
        return _HydrationSourceSliceResult(
            hydrated_records=hydrated_records,
            source_rows_consumed=source_rows_consumed,
            source_exhausted=source_exhausted,
        )

    def _finalize_full_corpus_hydration_rows(
        self,
        *,
        source: str,
        updated_rows: int,
        upstream_rows: Optional[int],
        mark_complete: bool,
    ) -> bool:
        """Finalize hydration completion and row-count reconciliation metadata.

        :param str source: Exact dataset source used for hydration.
        :param int updated_rows: Current cached payload row count.
        :param Optional[int] upstream_rows: Upstream split row count, if known.
        :param bool mark_complete: Whether this path has completed hydration.
        :return bool: ``True`` when row counts match or cannot be compared.
        """
        if mark_complete:
            self.embedding_cache.mark_hydrated(
                dataset_source=source,
                dataset_split=self.dataset_split,
                corpus_size=self.corpus_size,
                complete=True,
            )
        if upstream_rows is not None and updated_rows < upstream_rows:
            self.embedding_cache.set_hydration_rowcount_reconciliation(
                upstream_rows=upstream_rows,
                cached_rows=updated_rows,
            )
            return False
        self.embedding_cache.clear_hydration_rowcount_reconciliation()
        return True

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
        calibration_dataset = self._load_exact_hydration_source_slice(
            use_streaming=use_streaming,
            source=dataset_source,
            operation="Calibration prepass",
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
        fallback_index_offset: int = 0,
    ) -> int:
        """Hydrate cache records from dataset iterator without clearing namespace.

        :param Iterable[Dict[str, Any]] dataset: Dataset records to process.
        :param Optional[int] progress_total: Optional progress-bar total.
        :param str progress_label: Progress-bar description label.
        :param Optional[Set[str]] existing_paper_ids: Optional set used to skip
            already-cached paper IDs while hydrating.
        :param Optional[int] max_new_records: Optional cap on newly selected records.
        :param int fallback_index_offset: Source offset for synthetic paper IDs.
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
            for local_idx, raw_record in enumerate(dataset):
                if self.corpus_size is not None and local_idx >= self.corpus_size:
                    break

                metadata = _extract_dataset_paper_metadata(
                    raw_record,
                    fallback_index_offset + local_idx,
                )
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

    def _select_newest_corpus_rows(
        self, dataset: Iterable[Dict[str, Any]], dataset_source: str
    ) -> Iterable[Dict[str, Any]]:
        """Select the ``corpus_size`` most recently submitted rows by arXiv ID.

        Snapshot datasets are not ordered by submission time (the arXiv
        snapshot ships newest-``update_date`` first, with pre-2007 IDs at the
        tail), so positional slicing cannot express "newest papers". Rows are
        instead ranked by the submission chronology encoded in their arXiv
        IDs. Any shortfall is filled from rows without parseable IDs in source
        order, with a warning.

        :param Iterable[Dict[str, Any]] dataset: Loaded dataset or record stream.
        :param str dataset_source: Dataset source identifier (for logging).
        :return Iterable[Dict[str, Any]]: Selected rows (or the original
            iterable when no arXiv IDs are parseable).
        """
        limit = int(self.corpus_size)
        column_names = getattr(dataset, "column_names", None)
        select_by_index = column_names is not None and hasattr(dataset, "select")
        if select_by_index:
            if "id" not in column_names:
                logger.warning(
                    "Dataset %s has no 'id' column; capped hydration takes "
                    "the first %d rows instead of the newest.",
                    dataset_source,
                    limit,
                )
                return dataset
            selected = _newest_records_by_arxiv_id(
                (
                    {"id": raw_id, "source_index": idx}
                    for idx, raw_id in enumerate(dataset["id"])
                ),
                limit,
            )
        else:
            selected = _newest_records_by_arxiv_id(dataset, limit)

        chronology_keys = [
            key
            for record in selected
            if (key := _arxiv_id_chronology_key(record.get("id"))) is not None
        ]
        if not chronology_keys:
            logger.warning(
                "No parseable arXiv IDs in %s; capped hydration takes the "
                "first %d rows instead of the newest.",
                dataset_source,
                limit,
            )
            return dataset if select_by_index else selected
        else:
            if len(chronology_keys) < len(selected):
                logger.warning(
                    "Only %d rows in %s have parseable arXiv IDs; filling "
                    "the corpus cap with %d rows in source order.",
                    len(chronology_keys),
                    dataset_source,
                    len(selected) - len(chronology_keys),
                )
            logger.info(
                "Selected the %d most recently submitted rows from %s by "
                "arXiv ID chronology (submission window %04d-%02d..%04d-%02d).",
                len(chronology_keys),
                dataset_source,
                chronology_keys[0][0],
                chronology_keys[0][1],
                chronology_keys[-1][0],
                chronology_keys[-1][1],
            )
        if select_by_index:
            return dataset.select(
                sorted(int(record["source_index"]) for record in selected)
            )
        return selected

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

        self.embedding_cache.mark_hydrated(
            dataset_source=source,
            dataset_split=self.dataset_split,
            corpus_size=self.corpus_size,
            complete=False,
        )
        self._ensure_int8_calibration_ranges(
            use_streaming=use_streaming,
            dataset_source=source,
        )

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
        tail_refreshed_records = self._hydrate_exact_hydration_source_slice(
            use_streaming=use_streaming,
            source=source,
            row_limit=delta_rows,
            row_offset=cached_rows,
            progress_total=delta_rows,
            progress_label=f"Refreshing {source}",
            operation="Incremental refresh",
        ).hydrated_records
        updated_rows = self._cached_payload_row_count()
        reconciled_records = [0, 0]

        if updated_rows < upstream_rows:
            cached_paper_ids = self.embedding_cache.get_cached_paper_ids()
            reconciliation_passes = (
                (
                    "Head-slice reconciliation",
                    delta_rows,
                    0,
                    delta_rows,
                    "head",
                ),
                (
                    "Full-split reconciliation",
                    None,
                    None,
                    upstream_rows,
                    "full",
                ),
            )
            previous_operation = "Tail delta refresh"
            for pass_index, (
                operation,
                row_limit,
                row_offset,
                progress_total,
                progress_scope,
            ) in enumerate(reconciliation_passes):
                if updated_rows >= upstream_rows:
                    break
                remaining_rows = upstream_rows - updated_rows
                logger.warning(
                    "%s left %d unresolved rows for %s/%s "
                    "(cache_rows=%d, upstream=%d). Running %s.",
                    previous_operation,
                    remaining_rows,
                    source,
                    self.dataset_split,
                    updated_rows,
                    upstream_rows,
                    operation.lower().replace(
                        "reconciliation", "missing-ID reconciliation"
                    ),
                )
                reconciled_records[pass_index] = (
                    self._hydrate_exact_hydration_source_slice(
                        use_streaming=use_streaming,
                        source=source,
                        row_limit=row_limit,
                        row_offset=row_offset,
                        progress_total=progress_total,
                        progress_label=f"Reconciling {progress_scope} {source}",
                        operation=operation,
                        existing_paper_ids=cached_paper_ids,
                        max_new_records=(remaining_rows if pass_index == 0 else None),
                    ).hydrated_records
                )
                updated_rows = self._cached_payload_row_count()
                previous_operation = operation

        logger.info(
            "Incremental refresh processed tail=%d head=%d full=%d rows "
            "for %s/%s (cache_rows=%d, upstream_rows=%d).",
            tail_refreshed_records,
            *reconciled_records,
            source,
            self.dataset_split,
            updated_rows,
            upstream_rows,
        )
        rows_reconciled = self._finalize_full_corpus_hydration_rows(
            source=source,
            updated_rows=updated_rows,
            upstream_rows=upstream_rows,
            mark_complete=updated_rows > 0,
        )
        if not rows_reconciled:
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
                and (parsed_row_limit is not None or parsed_row_offset > 0)
            ):
                start_idx = str(parsed_row_offset) if parsed_row_offset else ""
                stop_idx = (
                    ""
                    if parsed_row_limit is None
                    else str(parsed_row_offset + parsed_row_limit)
                )
                split_for_load = f"{split_for_load}[{start_idx}:{stop_idx}]"
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
            elif (
                self.corpus_size is not None
                and parsed_row_limit is None
                and parsed_row_offset == 0
                and ":" not in str(self.dataset_split)
            ):
                # Newest-first capped hydration: snapshot row order does not
                # track submission time, so select by arXiv ID chronology.
                dataset = self._select_newest_corpus_rows(dataset, dataset_name)
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
            self._format_retrieval_document_metadata(metadata) for metadata in records
        ]
        sample_embeddings = self._encode_texts(
            sample_texts,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
        )
        ranges = np.stack(
            (sample_embeddings.min(axis=0), sample_embeddings.max(axis=0))
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
            metadata_map[paper_id] = payload

        if not metadata_map:
            return 0

        self.embedding_cache.upsert_embeddings(
            metadata_map,
            self._get_model_for_encoding(),
            batch_size=min(self.encode_batch_size, len(metadata_map)),
            show_progress=False,
            text_builder=self._format_retrieval_document_metadata,
        )
        return len(metadata_map)

    def _update_citation_counts(self, papers: Dict[str, Paper]) -> None:
        """
        Enrich top semantic candidates with citation counts from Semantic Scholar.

        :param Dict[str, Paper] papers: Dictionary of collected papers (including seed)
        """
        targets = [
            (pid, paper)
            for pid, paper in papers.items()
            if not paper.is_seed
            and not (isinstance(pid, str) and pid.startswith("query:"))
            and not (isinstance(pid, str) and pid.startswith("arxiv_"))
        ][:CITATION_COUNT_ENRICHMENT_LIMIT]

        if not targets:
            return

        logger.info(
            "Fetching citation counts from Semantic Scholar for up to %d papers...",
            len(targets),
        )

        from citemesh.services import SemanticScholarRequestError

        try:
            batch_results = self.client.get_papers(
                [paper_id for paper_id, _paper in targets]
            )
        except SemanticScholarRequestError as exc:
            logger.warning(
                "Citation-count enrichment was rejected; continuing with "
                "existing citation counts: %s",
                exc,
            )
            return
        for paper_id, paper in targets:
            batch_paper = batch_results.get(normalize_paper_id(paper_id))
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
        graph.graph["candidate_source_status"] = dict(
            sorted(self.candidate_source_status.items())
        )
        # Enforce a strict per-node top-k cap by greedily keeping strongest edges.
        filtered_graph = build_capped_undirected_graph(
            graph, self.top_k, seed_id=actual_seed_id
        )

        logger.info(
            f"Filtered graph: {filtered_graph.number_of_nodes()} nodes, "
            f"{filtered_graph.number_of_edges()} edges (top-{self.top_k})"
        )

        return filtered_graph, actual_seed_id
