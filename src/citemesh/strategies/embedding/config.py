"""Tunable constants for the embedding graph builder.

Batch sizes, candidate-pool multipliers and the default corpus dataset live here
so the builder and the CLI can share one definition. ``HYDRATION_FLUSH_SIZE``
deliberately lives in :mod:`citemesh.strategies.embedding.hydration` instead,
next to the loop that reads it.
"""

from __future__ import annotations

ENCODE_BATCH_SIZE = 32
CANDIDATE_MULTIPLIER = 4
CITATION_COUNT_ENRICHMENT_LIMIT = 20
CALIBRATION_RESERVOIR_SEED = 0
DEFAULT_DATASET_SOURCE = "librarian-bots/arxiv-metadata-snapshot"
