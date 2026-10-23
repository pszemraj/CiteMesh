#!/usr/bin/env python3
"""Refresh, export, and publish one complete CiteMesh arXiv index privately.

Requires the installed CiteMesh embeddings extra. Create a private Hugging Face
dataset repository with an initial README before running this publisher.
Existing configurations retain their pinned model contract and calibration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the private publication destination and local corpus settings."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("repo", help="existing private Hugging Face dataset repository")
    parser.add_argument(
        "output", type=Path, help="new directory for the complete portable export"
    )
    parser.add_argument(
        "--config",
        default="embeddinggemma-512-int8",
        help="published configuration name",
    )
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument(
        "--namespace", help="existing native cache namespace to refresh"
    )
    existing.add_argument(
        "--adopt-namespace",
        help="legacy namespace to certify using the supplied original model settings",
    )
    parser.add_argument(
        "--dataset-revision",
        help="source revision; latest for new work, recorded target for interrupted work",
    )
    parser.add_argument(
        "--model-revision", help="model revision for a new native configuration"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=512,
        help="embedding dimension for a new native configuration",
    )
    parser.add_argument(
        "--storage-precision",
        choices=("int8", "float32"),
        default="int8",
        help="stored vector dtype for a new native configuration",
    )
    parser.add_argument("--device", help="encoder device")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase diagnostic verbosity",
    )
    return parser.parse_args(argv)


def publish_arxiv_index(
    repo_id: str,
    output: Path,
    *,
    config_name: str,
    namespace: str | None = None,
    adopt_namespace: str | None = None,
    dataset_revision: str | None = None,
    model_revision: str | None = None,
    dimension: int = 512,
    storage_precision: str = "int8",
    device: str | None = None,
) -> str:
    """Use the native refresh and portable export paths before publication."""
    from citemesh.data.embedding_cache.portable import export_snapshot
    from citemesh.services.embedding_indexes import publish_snapshot
    from citemesh.strategies.embedding import EmbeddingGraphBuilder

    options = {"semantic_source": "arxiv-corpus", "device": device}
    if namespace is not None:
        from citemesh.data.embedding_cache.lifecycle import (
            open_existing_cache,
            select_existing_cache,
        )

        options["cache_selection"] = select_existing_cache(
            namespace, require_ready=False
        )
    else:
        options.update(
            model_revision=model_revision,
            truncate_dim=dimension,
            storage_precision=storage_precision,
        )
    builder = EmbeddingGraphBuilder(**options)
    if namespace is not None:
        builder.embedding_cache = open_existing_cache(
            options["cache_selection"], require_ready=False
        )
    if adopt_namespace is not None:
        from citemesh.strategies.embedding.refresh import adopt_legacy_cache

        adopt_legacy_cache(builder, adopt_namespace, revision=dataset_revision)
    else:
        builder.refresh_corpus(revision=dataset_revision)
    export_snapshot(builder.embedding_cache, output, config_name=config_name)
    return publish_snapshot(repo_id, output)


def main(argv: list[str] | None = None) -> int:
    """Run one manual publication and print its immutable repository revision."""
    args = parse_args(argv)
    logging.basicConfig(
        level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG),
        stream=sys.stderr,
        format="%(levelname)s: %(message)s",
    )
    try:
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        revision = publish_arxiv_index(
            args.repo,
            args.output,
            config_name=args.config,
            namespace=args.namespace,
            adopt_namespace=args.adopt_namespace,
            dataset_revision=args.dataset_revision,
            model_revision=args.model_revision,
            dimension=args.dimension,
            storage_precision=args.storage_precision,
            device=args.device,
        )
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, ImportError, RuntimeError, HfHubHTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
