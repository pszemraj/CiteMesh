"""Tunable constants for the embedding graph builder.

Batch sizes, candidate-pool multipliers and the default corpus dataset live here
so the builder and the CLI can share one definition. ``HYDRATION_FLUSH_SIZE``
deliberately lives in :mod:`citemesh.strategies.embedding.hydration` instead,
next to the loop that reads it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from citemesh.data.model_profiles import EMBEDDINGGEMMA_MODEL_NAME

if TYPE_CHECKING:
    from ._host import _EmbeddingBuilderHost

ENCODE_BATCH_SIZE = 32
CANDIDATE_MULTIPLIER = 4
CALIBRATION_RESERVOIR_SEED = 0
_ARXIV_METADATA_DATASET = "librarian-bots/arxiv-metadata-snapshot"
DEFAULT_DATASET_SOURCE = _ARXIV_METADATA_DATASET

PUBLIC_ARXIV_INDEX_IMPORT = (
    "citemesh cache import",
    "--bucket pszemraj/citemesh-arxiv-indexes",
    "--config embeddinggemma-512-int8",
)


def matches_public_arxiv_index(builder: _EmbeddingBuilderHost) -> bool:
    """Check the published index settings, which stay fixed as local defaults change."""
    return (
        builder.dataset_source == _ARXIV_METADATA_DATASET
        and builder.dataset_split == "train"
        and builder.model_name == EMBEDDINGGEMMA_MODEL_NAME
        and builder.model_revision is None
        and builder.requested_model_profile in {"auto", "embeddinggemma"}
        and builder.truncate_dim == 512
        and builder.storage_precision == "int8"
        and builder.calibration_sample_size == 2000
    )
