"""Inspect, transfer, and refresh native corpus caches."""

from __future__ import annotations

import argparse
import logging
import sqlite3

from rich.text import Text

from citemesh.data.user_config import UserConfig

from .. import console
from ..cache_ops import _clear_cache_directory, _scan_cache_directory
from ..console import logger
from ..parser import _missing_command, corpus_build_example, corpus_search_example

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


def _default_corpus_args() -> argparse.Namespace:
    """Return the built-in build defaults for an arXiv-corpus embedding build.

    :return argparse.Namespace: Parsed defaults without config.toml overrides.
    """
    from ..parser import _create_parser, _pop_tracked_option_dests

    _, build_parser, _, _ = _create_parser()
    defaults = build_parser.parse_args(["corpus-defaults", "--strategy", "embedding"])
    _pop_tracked_option_dests(defaults)
    defaults.semantic_source = "arxiv-corpus"
    return defaults


def _settings_select(
    namespace: str,
    build_parser: argparse.ArgumentParser,
    user_config: UserConfig,
) -> bool:
    """Check whether the user's settings select an installed corpus on their own.

    Resolves the settings of ``citemesh search QUERY --semantic-source
    arxiv-corpus`` (built-in defaults, then config.toml), which a hybrid build
    with ``--semantic-source both`` shares for its corpus, and asks a builder
    with them whether it binds the namespace. That loads no model and uses no
    network.

    :param str namespace: Installed native namespace ID.
    :param argparse.ArgumentParser build_parser: Build subparser supplying the
        built-in defaults.
    :param UserConfig user_config: Loaded user configuration snapshot.
    :return bool: ``True`` when those settings reach the namespace; ``False``
        also when config.toml makes them invalid for a corpus build or the
        embeddings extra is not installed.
    """
    from citemesh.strategies.embedding import EmbeddingGraphBuilder

    from ..build_contract import _resolve_namespace_settings
    from ..build_options import _shared_embedding_builder_kwargs

    try:
        settings = _resolve_namespace_settings(
            build_parser, user_config, {"semantic_source": "arxiv-corpus"}
        )
        builder = EmbeddingGraphBuilder(**_shared_embedding_builder_kwargs(settings))
    except ImportError:
        return False
    except ValueError as exc:
        logger.debug("Current settings cannot build from a corpus: %s", exc)
        return False
    return builder.selects_installed_corpus(namespace)


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

    if args.all_corpus:
        from citemesh.data.embedding_cache.constants import (
            HYDRATION_CORPUS_SIZE_KEY,
            _corpus_size_from_token,
        )
        from citemesh.data.embedding_cache.lifecycle import (
            read_existing_cache_metadata,
        )

        metadata = read_existing_cache_metadata(args.namespace)
        coverage = metadata.get(HYDRATION_CORPUS_SIZE_KEY)
        if coverage is not None and _corpus_size_from_token(coverage) is not None:
            raise ValueError(
                "--all-corpus conflicts with the recorded legacy corpus limit"
            )

    defaults = _default_corpus_args()
    provided = {"semantic_source"}
    for dest in _ADOPTION_SELECTORS:
        value = getattr(args, dest, None)
        if value is not None:
            setattr(defaults, dest, value)
            provided.add(dest)
    _validate_build_cli_contract(defaults, error_sink, provided)
    builder = EmbeddingGraphBuilder(**_shared_embedding_builder_kwargs(defaults))
    adopt_legacy_cache(builder, args.namespace, revision=args.dataset_revision)


def _print_installed_namespace(namespace: str, *, automatic: bool) -> None:
    """Show a copyable route from corpus installation to local retrieval.

    :param str namespace: Installed native namespace ID.
    :param bool automatic: Whether the user's settings select the corpus without
        ``--cache-namespace``.
    """
    console.output_console.print(
        Text.assemble(("Installed corpus namespace: ", "bold"), (namespace, "cyan"))
    )
    if automatic:
        selector = None
        note = "Your settings select this index automatically."
    else:
        selector = namespace
        note = (
            "Your settings do not select this index; add "
            f"--cache-namespace {namespace} to the commands below."
        )
    console.output_console.print(Text(note))
    for label, command in (
        ("Build a hybrid graph", corpus_build_example(selector)),
        ("Search the corpus", corpus_search_example(selector)),
        (
            "Use the corpus for future hybrid builds",
            "citemesh config set defaults.semantic_source both",
        ),
    ):
        console.output_console.print(label, style="bold")
        console.output_console.print(Text(command), soft_wrap=True)
    console.output_console.print(
        Text(
            "Corpus embeddings are installed; query and graph encoding still use "
            "the selected model.",
            style="dim",
        )
    )


def _run_portable_command(
    args: argparse.Namespace,
    build_parser: argparse.ArgumentParser,
    user_config: UserConfig,
) -> int:
    """Export or import a cache snapshot, loading transfer dependencies on demand.

    :param argparse.Namespace args: Parsed export or import options.
    :param argparse.ArgumentParser build_parser: Build subparser whose defaults,
        with ``user_config``, decide how the import report selects the corpus.
    :param UserConfig user_config: Loaded user configuration snapshot.
    :return int: Process-style exit code.
    """
    from citemesh.data.embedding_cache.lifecycle import (
        open_existing_cache,
        select_existing_cache,
    )
    from citemesh.data.embedding_cache.portable import export_snapshot, import_snapshot

    if args.cache_command == "export":
        selection = select_existing_cache(args.namespace)
        cache = open_existing_cache(selection)
        manifest = export_snapshot(cache, args.output)
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
    _print_installed_namespace(
        cache.namespace_id,
        automatic=_settings_select(cache.namespace_id, build_parser, user_config),
    )
    return 0


def run_cache_command(
    args: argparse.Namespace,
    cache_parser: argparse.ArgumentParser,
    build_parser: argparse.ArgumentParser,
    user_config: UserConfig,
) -> int:
    """Run the ``cache`` subcommand.

    Only the import report reads ``user_config``; refresh and adoption use
    built-in defaults and the selectors passed to them.

    :param argparse.Namespace args: Parsed CLI namespace for ``cache``.
    :param argparse.ArgumentParser cache_parser: Parser used for usage output.
    :param argparse.ArgumentParser build_parser: Build subparser supplying the
        settings the import report checks.
    :param UserConfig user_config: Loaded user configuration snapshot.
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
            return _run_portable_command(args, build_parser, user_config)
        except _CacheUsageError as exc:
            cache_parser.error(str(exc))
        except ImportError as exc:
            logger.error(
                "Corpus transfer/refresh dependencies are unavailable; install "
                "CiteMesh with the embeddings extra: pip install 'citemesh[embeddings]'. %s",
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return 1
        except (OSError, ValueError, RuntimeError, sqlite3.DatabaseError) as exc:
            logger.error(
                "Corpus %s failed: %s",
                args.cache_command,
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
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
        return _missing_command(cache_parser)

    return 0
