#!/usr/bin/env python3
"""Refresh, export, and publish one complete CiteMesh arXiv index privately.

Requires the installed CiteMesh embeddings extra and an existing private
Hugging Face bucket.
Existing configurations retain their pinned model contract and calibration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the private publication destination and local corpus settings.

    :param list[str] argv: Optional argument list; defaults to process arguments.
    :return argparse.Namespace: Parsed publication options.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "bucket", help="existing private Hugging Face bucket (owner/name)"
    )
    parser.add_argument(
        "output", type=Path, help="new directory for the complete portable export"
    )
    parser.add_argument(
        "--config",
        help="published configuration name; inferred from the stored contract when omitted",
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
    bucket_id: str,
    output: Path,
    *,
    config_name: str | None = None,
    namespace: str | None = None,
    adopt_namespace: str | None = None,
    dataset_revision: str | None = None,
    model_revision: str | None = None,
    dimension: int = 512,
    storage_precision: str = "int8",
    device: str | None = None,
) -> str:
    """Refresh or adopt a corpus, export it, and publish the snapshot.

    :param str bucket_id: Existing private Hugging Face bucket identifier.
    :param Path output: Directory for the portable snapshot.
    :param str | None config_name: Published name; inferred from the stored contract when omitted.
    :param Optional[str] namespace: Existing native cache namespace to refresh.
    :param Optional[str] adopt_namespace: Legacy namespace to certify first.
    :param Optional[str] dataset_revision: Source revision for refresh or adoption.
    :param Optional[str] model_revision: Model revision for a new cache.
    :param int dimension: Embedding dimension for a new cache.
    :param str storage_precision: Stored vector dtype for a new cache.
    :param Optional[str] device: Encoder device override.
    :return str: Published snapshot identifier.
    """
    from citemesh.data.embedding_cache.portable import export_snapshot
    from citemesh.services.embedding_indexes import publish_snapshot
    from citemesh.strategies.embedding import EmbeddingGraphBuilder

    if output.exists():
        raise FileExistsError(f"Export output already exists: {output}")
    if namespace is not None:
        from citemesh.data.embedding_cache.lifecycle import (
            open_existing_cache,
            select_existing_cache,
        )
        from citemesh.strategies.embedding.refresh import refresh_cache_selection

        selection = select_existing_cache(namespace, require_ready=False)
        refresh_cache_selection(selection, revision=dataset_revision, device=device)
        cache = open_existing_cache(selection)
    else:
        builder = EmbeddingGraphBuilder(
            semantic_source="arxiv-corpus",
            device=device,
            model_revision=model_revision,
            truncate_dim=dimension,
            storage_precision=storage_precision,
        )
        if adopt_namespace is not None:
            from citemesh.strategies.embedding.refresh import adopt_legacy_cache

            adopt_legacy_cache(builder, adopt_namespace, revision=dataset_revision)
        else:
            builder.refresh_corpus(revision=dataset_revision)
        cache = builder.embedding_cache
    export_snapshot(cache, output, config_name=config_name)
    return publish_snapshot(bucket_id, output)


def main(argv: list[str] | None = None) -> int:
    """Run one manual publication and print its snapshot ID.

    :param list[str] argv: Optional argument list; defaults to process arguments.
    :return int: Process-style exit code.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG),
        stream=sys.stderr,
        format="%(levelname)s: %(message)s",
    )
    try:
        snapshot_id = publish_arxiv_index(
            args.bucket,
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
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(snapshot_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
