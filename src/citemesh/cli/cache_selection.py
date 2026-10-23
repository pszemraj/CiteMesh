"""Apply an installed corpus contract before constructing embedding builders."""

from __future__ import annotations

import argparse
from typing import Protocol


class _ErrorSink(Protocol):
    """Provide argparse-style error reporting for selection validation."""

    def error(self, message: str) -> None: ...


def _apply_cache_selection(
    args: argparse.Namespace,
    provided: set[str],
    error_sink: _ErrorSink,
) -> set[str]:
    """Apply an installed corpus contract to parsed build arguments.

    :param argparse.Namespace args: Parsed build options to update.
    :param set[str] provided: Option destinations explicitly supplied by the user.
    :param _ErrorSink error_sink: Parser-compatible error reporter.
    :return set[str]: Option destinations bound from the installed contract.
    """
    if not getattr(args, "cache_namespace", None):
        return set()
    if args.strategy not in {"embedding", "hybrid"}:
        error_sink.error("--cache-namespace requires --strategy embedding or hybrid.")
    from citemesh.data.embedding_cache.lifecycle import (
        open_existing_cache,
        select_existing_cache,
    )

    selection = select_existing_cache(args.cache_namespace)
    contract = selection.contract
    selectors = {
        "model": contract.model_id,
        "model_revision": contract.model_revision,
        "model_profile": contract.profile_key,
        "truncate_dim": contract.dimension,
        "storage_precision": contract.stored_dtype,
        "calibration_sample_size": contract.calibration_sample_size,
        "semantic_source": "arxiv-corpus",
        "dataset_source": selection.source_repo_id,
        "dataset_split": selection.source_split,
    }
    for dest, value in selectors.items():
        if dest in provided and getattr(args, dest) != value:
            error_sink.error(
                f"--{dest.replace('_', '-')} conflicts with --cache-namespace "
                f"{selection.namespace_id} (stored value: {value})."
            )
        setattr(args, dest, value)
    if args.force_rebuild_cache:
        error_sink.error("--force-rebuild-cache cannot be used with --cache-namespace.")
    cache = open_existing_cache(selection)
    with cache.hydration_operation_lock():
        source = cache.corpus_state().get("source")
    limit = source.get("limit") if source is not None else None
    if limit is not None and not {"all_corpus", "corpus_size"}.intersection(provided):
        args.all_corpus = False
        args.corpus_size = min(args.corpus_size, limit) if args.corpus_size else limit
    args._cache_selection = selection
    return set(selectors)
