"""Embedding-based graph building strategy.

The strategy is split across focused modules and re-exported here so
``citemesh.strategies.embedding`` stays the single public import path:

- :mod:`~citemesh.strategies.embedding.runtime`: torch/Transformers version
  floors, accelerator and bf16 probes, device resolution, compatibility errors.
- :mod:`~citemesh.strategies.embedding.deps`: optional-dependency checks and the
  ``_import_*`` shims every third-party import funnels through.
- :mod:`~citemesh.strategies.embedding.text`: embedding task roles and prompt
  formatting of paper title/abstract text.
- :mod:`~citemesh.strategies.embedding.records`: arXiv dataset record
  normalization into embedding metadata.
- :mod:`~citemesh.strategies.embedding.precision`: the precision-scoped encode
  proxy wrapping a loaded SentenceTransformer.
- :mod:`~citemesh.strategies.embedding.config`: builder tunables.
- :mod:`~citemesh.strategies.embedding.fingerprint`,
  :mod:`~citemesh.strategies.embedding.model_runtime` and
  :mod:`~citemesh.strategies.embedding.hydration`: the mixins assembled into
  :class:`~citemesh.strategies.embedding.builder.EmbeddingGraphBuilder`.
"""

from __future__ import annotations

from .builder import EmbeddingGraphBuilder
from .config import (
    CALIBRATION_RESERVOIR_SEED,
    CANDIDATE_MULTIPLIER,
    CITATION_COUNT_ENRICHMENT_LIMIT,
    DEFAULT_DATASET_SOURCE,
    ENCODE_BATCH_SIZE,
)
from .deps import _check_embedding_deps
from .fingerprint import EmbeddingCacheFingerprintMismatchError
from .hydration import HYDRATION_FLUSH_SIZE
from .precision import _PrecisionEncodeProxy
from .records import _extract_dataset_paper_metadata, _query_seed_id
from .runtime import (
    EMBEDDING_DEVICE_CHOICES,
    EmbeddingBackendCompatibilityError,
    EmbeddingPrecisionCompatibilityError,
    resolve_embedding_device,
)
from .text import EmbeddingTask, format_embedding_metadata, format_paper_for_embedding

__all__ = [
    "CALIBRATION_RESERVOIR_SEED",
    "CANDIDATE_MULTIPLIER",
    "CITATION_COUNT_ENRICHMENT_LIMIT",
    "DEFAULT_DATASET_SOURCE",
    "EMBEDDING_DEVICE_CHOICES",
    "ENCODE_BATCH_SIZE",
    "HYDRATION_FLUSH_SIZE",
    "EmbeddingBackendCompatibilityError",
    "EmbeddingCacheFingerprintMismatchError",
    "EmbeddingGraphBuilder",
    "EmbeddingPrecisionCompatibilityError",
    "EmbeddingTask",
    "_PrecisionEncodeProxy",
    "_check_embedding_deps",
    "_extract_dataset_paper_metadata",
    "_query_seed_id",
    "format_embedding_metadata",
    "format_paper_for_embedding",
    "resolve_embedding_device",
]
