"""Inspect, transfer, and refresh native corpus caches."""

from __future__ import annotations

import argparse
import shlex

from rich.text import Text

from .. import console
from ..cache_ops import _clear_cache_directory, _scan_cache_directory
from ..console import logger

_ADOPTION_SELECTORS = (
    "model",
    "model_revision",
    "model_profile",
    "truncate_dim",
    "storage_precision",
    "calibration_sample_size",
    "dataset_source",
    "dataset_split",
    "corpus_size",
    "all_corpus",
)


class _CacheUsageError(ValueError):
    """Report portable-source selection errors through argparse."""


def _hub_http_error_types() -> tuple[type[Exception], ...]:
    """Resolve the Hub HTTP exception across supported SDK versions on failure."""
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        try:
            from huggingface_hub.utils import HfHubHTTPError
        except ImportError:
            return ()
    return (HfHubHTTPError,)


def _adopt_legacy_namespace(
    args: argparse.Namespace, error_sink: argparse.ArgumentParser
) -> None:
    """Validate selectors and certify a legacy cache namespace.

    :param argparse.Namespace args: Parsed refresh options for the namespace.
    :param argparse.ArgumentParser error_sink: Parser used to report invalid options.
    """
    from citemesh.strategies.embedding import EmbeddingGraphBuilder
    from citemesh.strategies.embedding.refresh import adopt_legacy_cache

    from ..build_contract import _validate_build_cli_contract
    from ..build_options import _shared_embedding_builder_kwargs
    from ..parser import _create_parser, _pop_tracked_option_dests

    _, build_parser, _, _ = _create_parser()
    defaults = build_parser.parse_args(["legacy-adoption", "--strategy", "embedding"])
    _pop_tracked_option_dests(defaults)
    provided = {"semantic_source"}
    defaults.semantic_source = "arxiv-corpus"
    for dest in _ADOPTION_SELECTORS:
        value = getattr(args, dest, None)
        if value is not None:
            setattr(defaults, dest, value)
            provided.add(dest)
    _validate_build_cli_contract(defaults, error_sink, provided)
    builder = EmbeddingGraphBuilder(**_shared_embedding_builder_kwargs(defaults))
    adopt_legacy_cache(builder, args.namespace, revision=args.dataset_revision)


def _print_installed_namespace(namespace: str) -> None:
    """Show a copyable route from corpus installation to local retrieval."""
    command = shlex.join(
        [
            "citemesh",
            "search",
            "long-context language models",
            "--mode",
            "local",
            "--cache-namespace",
            namespace,
        ]
    )
    console.output_console.print(Text(f"Installed corpus namespace: {namespace}"))
    console.output_console.print(Text(f"Search it with: {command}"))
    console.output_console.print(
        Text(
            "Corpus embeddings are installed; query and graph encoding still use "
            "the selected model."
        )
    )


def _run_portable_command(args: argparse.Namespace) -> int:
    """Export or import a cache snapshot, loading transfer dependencies on demand.

    :param argparse.Namespace args: Parsed export or import options.
    :return int: Process-style exit code.
    """
    from citemesh.data.embedding_cache.lifecycle import (
        open_existing_cache,
        select_existing_cache,
    )
    from citemesh.data.embedding_cache.portable import export_snapshot, import_snapshot

    if args.cache_command == "export":
        selection = select_existing_cache(args.namespace)
        contract = selection.contract
        cache = open_existing_cache(selection)
        config_name = (
            f"{contract.profile_key}-{contract.dimension}-{contract.stored_dtype}"
        )
        manifest = export_snapshot(cache, args.output, config_name=config_name)
        logger.info(
            "Exported %s corpus rows to %s.", f"{manifest.row_count:,}", args.output
        )
        return 0
    publication = None
    snapshot_path = args.path
    if args.bucket:
        from citemesh.services.embedding_indexes import (
            IndexSelectionError,
            download_publication,
            preflight_publication_destination,
            resolve_publication,
        )

        try:
            resolved = resolve_publication(
                args.bucket, config_name=args.config, snapshot_id=args.snapshot
            )
        except IndexSelectionError as exc:
            raise _CacheUsageError(str(exc)) from exc
        preflight_publication_destination(resolved)
        snapshot_path = download_publication(resolved)
        publication = resolved.origin
    cache = import_snapshot(snapshot_path, publication=publication)
    namespace = cache.db_path.stem.removeprefix("metadata_")
    _print_installed_namespace(namespace)
    return 0


def run_cache_command(
    args: argparse.Namespace, cache_parser: argparse.ArgumentParser
) -> int:
    """Run the ``cache`` subcommand.

    :param argparse.Namespace args: Parsed CLI namespace for ``cache``.
    :param argparse.ArgumentParser cache_parser: Parser used for usage output.
    :return int: Process-style exit code.
    """
    if (
        args.cache_command == "import"
        and args.path is not None
        and (args.config is not None or args.snapshot is not None)
    ):
        cache_parser.error(
            "--config and --snapshot require --bucket; local files use their manifest."
        )
    if args.cache_command in {"export", "import", "refresh"}:
        if (
            args.cache_command == "refresh"
            and not args.adopt
            and any(
                getattr(args, dest, None) is not None for dest in _ADOPTION_SELECTORS
            )
        ):
            cache_parser.error("Legacy model/source selectors require --adopt.")
        try:
            if args.cache_command == "refresh":
                if args.adopt:
                    _adopt_legacy_namespace(args, cache_parser)
                    logger.info("Adopted corpus namespace %s.", args.namespace)
                    return 0
                from citemesh.data.embedding_cache.lifecycle import (
                    select_existing_cache,
                )
                from citemesh.strategies.embedding.refresh import (
                    refresh_cache_selection,
                )

                selection = select_existing_cache(args.namespace, require_ready=False)
                refresh_cache_selection(selection, revision=args.dataset_revision)
                logger.info("Refreshed corpus namespace %s.", args.namespace)
                return 0
            return _run_portable_command(args)
        except _CacheUsageError as exc:
            cache_parser.error(str(exc))
        except ImportError as exc:
            logger.error(
                "Corpus transfer/refresh dependencies are unavailable; install "
                "CiteMesh with the embeddings extra: pip install 'citemesh[embeddings]'. %s",
                exc,
            )
            return 1
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Corpus %s failed: %s", args.cache_command, exc)
            return 1
        except _hub_http_error_types() as exc:
            logger.error("Corpus %s failed: %s", args.cache_command, exc)
            return 1
    if args.cache_command == "scan":
        exit_code = _scan_cache_directory()
        if exit_code != 0:
            return exit_code
    elif args.cache_command == "clear":
        exit_code = _clear_cache_directory(
            assume_yes=bool(args.yes),
            clear_reason=getattr(args, "reason", None),
        )
        if exit_code != 0:
            return exit_code
    else:
        cache_parser.print_help()
        return 1

    return 0
