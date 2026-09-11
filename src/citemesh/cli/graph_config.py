"""Reproducible build-configuration payloads written beside graph exports.

Owns the ``*.config.json`` payload assembly for citation and hybrid/embedding
runs, plus the paper-id canonicalization used in export metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from citemesh.core.paper_ids import canonicalize_or_none

from .build_options import (
    _CORPUS_ONLY_OPTION_DESTS,
    _hybrid_semantic_branch_enabled,
    _normalized_cache_reason,
    _resolved_hybrid_max_semantic,
)


def _drop_none_values(value: Any) -> Any:
    """Recursively drop ``None`` entries from dictionaries/lists.

    :param Any value: Arbitrary JSON-serializable object.
    :return Any: Copy with ``None``-valued mapping entries removed.
    """
    if isinstance(value, dict):
        return {
            key: _drop_none_values(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value]
    return value


def _build_citation_config_payload(
    cli_args: argparse.Namespace, *, strategy: str
) -> dict[str, Any] | None:
    """Build strategy-scoped citation/reference settings for config sidecars.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :param str strategy: Active build strategy.
    :return Optional[Dict[str, Any]]: Citation config payload when applicable.
    """
    fetch_references = not bool(cli_args.no_references)
    refresh_reference_cache = bool(cli_args.refresh_reference_cache)

    if strategy == "recommendation":
        return {
            "fetch_references": fetch_references,
            "refresh_reference_cache": refresh_reference_cache,
            "similarity_threshold": cli_args.similarity_threshold,
        }
    if strategy in {"citation", "hybrid"}:
        config = {
            "max_citations": int(cli_args.max_citations),
            "max_references": int(cli_args.max_references),
            "fetch_references": fetch_references,
            "refresh_reference_cache": refresh_reference_cache,
        }
        if strategy == "citation":
            config["similarity_threshold"] = cli_args.similarity_threshold
        return config
    return None


def _build_graph_config_payload(
    cli_args: argparse.Namespace,
    seed_id: str,
    metadata: dict[str, Any],
    selected_formats: list[str],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    """Build sidecar graph-config payload for reproducibility and auditability.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :param str seed_id: Resolved seed paper identifier.
    :param Dict[str, Any] metadata: Run metadata payload used by exporters.
    :param List[str] selected_formats: Formats requested for export.
    :param Dict[str, Path] output_paths: Resolved export artifact paths.
    :return Dict[str, Any]: JSON-safe run configuration payload.
    """
    strategy = str(cli_args.strategy)
    semantic_enabled = strategy == "embedding" or (
        strategy == "hybrid" and _hybrid_semantic_branch_enabled(cli_args)
    )
    embedding_config: dict[str, Any] | None = None
    if semantic_enabled:
        embedding_config = {
            "model": cli_args.model,
            "model_profile": cli_args.model_profile,
            "model_revision": cli_args.model_revision,
            "semantic_source": str(cli_args.semantic_source),
            "candidate_pool_size": int(cli_args.candidate_pool_size),
            "dataset_source": cli_args.dataset_source,
            "dataset_split": cli_args.dataset_split,
            "corpus_size": (
                None
                if cli_args.all_corpus or cli_args.corpus_size is None
                else int(cli_args.corpus_size)
            ),
            "all_corpus": bool(cli_args.all_corpus or cli_args.corpus_size is None),
            "truncate_dim": cli_args.truncate_dim,
            "min_semantic_similarity": cli_args.min_semantic_similarity,
            "streaming": bool(cli_args.streaming),
            "storage_precision": cli_args.storage_precision,
            "binary_prefilter": bool(cli_args.binary_prefilter),
            "binary_rescore_multiplier": int(cli_args.binary_rescore_multiplier),
            "calibration_sample_size": int(cli_args.calibration_sample_size),
            "cache_compression": cli_args.cache_compression,
            "cache_compression_level": int(cli_args.cache_compression_level),
            "encode_batch_size": int(cli_args.encode_batch_size),
            "torch_compile": bool(cli_args.torch_compile),
            "device": str(cli_args.device),
            "force_rebuild_cache": bool(cli_args.force_rebuild_cache),
            "overwrite_cache": bool(cli_args.overwrite_cache),
            "cache_overwrite_reason": _normalized_cache_reason(
                getattr(cli_args, "cache_overwrite_reason", None)
            ),
        }
        if strategy == "embedding":
            embedding_config["top_k"] = int(cli_args.top_k)
        if cli_args.semantic_source != "arxiv-corpus":
            for dest in _CORPUS_ONLY_OPTION_DESTS:
                embedding_config.pop(dest, None)
        else:
            embedding_config.pop("candidate_pool_size")
        if cli_args.storage_precision != "int8":
            for dest in (
                "binary_prefilter",
                "binary_rescore_multiplier",
                "calibration_sample_size",
            ):
                embedding_config.pop(dest)

    payload = {
        "schema_version": 1,
        "build": {
            "paper_id_input": cli_args.paper_id,
            "paper_id_canonical": canonicalize_paper_id_for_metadata(cli_args.paper_id),
            "seed_id": seed_id,
            "strategy": strategy,
            "max_papers": int(cli_args.max_papers),
            "refresh_paper_cache": bool(
                getattr(cli_args, "refresh_paper_cache", False)
            ),
            "citation": _build_citation_config_payload(cli_args, strategy=strategy),
            "hybrid": (
                {"max_semantic": _resolved_hybrid_max_semantic(cli_args)}
                if strategy == "hybrid"
                else None
            ),
            "embedding": embedding_config,
            "layout": {
                "spring_iterations": int(cli_args.spring_iterations),
                "seed": cli_args.seed,
                "dpi": int(cli_args.dpi),
                "theme": cli_args.theme,
            },
            "timestamp_included": bool(cli_args.include_timestamp),
            "exports_requested": list(selected_formats),
        },
        "outputs": {fmt: str(path) for fmt, path in sorted(output_paths.items())},
        "metadata": metadata,
    }
    return _drop_none_values(payload)


def canonicalize_paper_id_for_metadata(paper_id: str) -> str:
    """
    Best-effort canonical paper ID for output metadata display.

    :param str paper_id: Raw CLI paper identifier.
    :return str: Canonicalized identifier when possible; otherwise original input.
    """
    canonical = canonicalize_or_none(paper_id)
    return canonical if canonical is not None else paper_id
