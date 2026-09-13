"""``citemesh search``: local embedding-cache search and Semantic Scholar lookup."""

from __future__ import annotations

import argparse
import logging

from rich.text import Text

from citemesh.data.user_config import UserConfig
from citemesh.services import SemanticScholarUnavailableError, get_client
from citemesh.strategies.embedding import (
    EmbeddingGraphBuilder,
    resolve_embedding_device,
)

from .. import console
from ..build_contract import _validate_build_cli_contract, _ValueErrorParserErrorSink
from ..build_options import (
    _apply_user_config_defaults,
    _configured_client_kwargs,
    _resolve_user_config_api_key,
    _shared_embedding_builder_kwargs,
)
from ..console import logger
from ..parser import _output_table, _pop_tracked_option_dests


def _resolve_search_mode(
    args: argparse.Namespace, user_config: UserConfig
) -> tuple[str, str]:
    """Resolve the effective search mode and where it came from.

    Precedence: explicit ``--mode`` flag > config.toml
    ``defaults.search_mode`` > built-in ``auto``.

    :param argparse.Namespace args: Parsed search command arguments.
    :param UserConfig user_config: Loaded user configuration snapshot.
    :return Tuple[str, str]: ``(mode, origin)`` with origin one of ``flag``,
        ``config``, ``default``.
    """
    if args.mode:
        return str(args.mode), "flag"
    configured = user_config.defaults.get("search_mode")
    if configured:
        return str(configured), "config"
    return "auto", "default"


def _prepare_local_search_builder(
    args: argparse.Namespace,
    build_parser: argparse.ArgumentParser,
    user_config: UserConfig,
) -> tuple[EmbeddingGraphBuilder, argparse.Namespace]:
    """Construct the embedding builder local search runs against.

    Mirrors a flagless build's defaults pipeline (config.toml defaults plus
    candidate-mode storage normalization) so the search targets the same cache
    namespace a default build writes to; ``--model``, ``--model-profile``, and
    ``--device`` override.

    :param argparse.Namespace args: Parsed search command arguments.
    :param argparse.ArgumentParser build_parser: Build subparser used to
        derive build-equivalent defaults.
    :param UserConfig user_config: Loaded user configuration snapshot.
    :return Tuple[EmbeddingGraphBuilder, argparse.Namespace]: Builder and the
        effective build-equivalent defaults namespace.
    """
    defaults = build_parser.parse_args(["local-search-placeholder-seed"])
    defaults._s2_api_key = _resolve_user_config_api_key(user_config)
    _pop_tracked_option_dests(defaults)
    config_default_dests = _apply_user_config_defaults(
        defaults, {"strategy"}, user_config
    )
    defaults.strategy = "embedding"
    if args.model:
        defaults.model = args.model
        config_default_dests.discard("model")
    if args.model_profile:
        defaults.model_profile = args.model_profile
        config_default_dests.discard("model_profile")
    if args.device:
        defaults.device = args.device
        config_default_dests.discard("device")
    _validate_build_cli_contract(
        defaults,
        _ValueErrorParserErrorSink(),
        frozenset(),
        config_defaults=config_default_dests,
        config_path=user_config.path,
    )
    builder = EmbeddingGraphBuilder(**_shared_embedding_builder_kwargs(defaults))
    return builder, defaults


def _render_local_search(
    args: argparse.Namespace,
    builder: EmbeddingGraphBuilder,
    defaults: argparse.Namespace,
) -> int:
    """Encode the query, search the local cache, and render ranked results.

    :param argparse.Namespace args: Parsed search command arguments.
    :param EmbeddingGraphBuilder builder: Builder targeting the search namespace.
    :param argparse.Namespace defaults: Effective build-equivalent defaults.
    :return int: Process-style exit code.
    """
    try:
        results = builder.search_local(args.query, top_k=args.limit)
    except Exception as exc:
        logger.error(
            "Local search failed: %s",
            exc,
            exc_info=logging.getLogger().level == logging.DEBUG,
        )
        return 1

    cache = builder.embedding_cache
    if not results:
        logger.error(
            "Local search returned no results for model=%s (cache: %s).",
            defaults.model,
            cache.h5_path,
        )
        return 1

    table = _output_table(f"Local semantic search for '{args.query}'")
    table.leading = 1
    table.add_column("#", style="dim", width=3)
    table.add_column("Paper / authors", ratio=1)
    table.add_column("Year", justify="right", no_wrap=True)
    table.add_column("Score", justify="right", width=6)

    for i, result in enumerate(results, 1):
        metadata = result.metadata or {}
        authors = [str(name) for name in (metadata.get("authors") or [])]
        authors_str = ", ".join(authors[:2])
        if len(authors) > 2:
            authors_str += " et al."
        year_value = metadata.get("year")
        table.add_row(
            str(i),
            Text.assemble(
                (str(metadata.get("title") or ""), "bold"),
                ("\n" + authors_str, "dim") if authors_str else "",
            ),
            str(year_value) if year_value is not None else "",
            f"{float(result.score):.3f}",
        )

    console.output_console.print(table)
    total = getattr(cache, "last_search_total_embeddings", None)
    if total is not None:
        console.output_console.print(
            Text(
                f"Searched {int(total):,} locally cached embeddings "
                f"(model={defaults.model}, source={defaults.semantic_source}).",
                style="dim",
            )
        )
    console.output_console.print("\n[dim]Full paper IDs:[/dim]")
    for i, result in enumerate(results, 1):
        console.output_console.print(
            Text.assemble((f"{i}. ", "dim"), (str(result.paper_id), "cyan")),
            soft_wrap=True,
        )
    console.output_console.print(
        '\nUse a paper ID with: citemesh build "<ID>"', style="dim"
    )
    return 0


def _run_s2_search(args: argparse.Namespace) -> int:
    """Run keyword search on the Semantic Scholar API and render results.

    :param argparse.Namespace args: Parsed search command arguments.
    :return int: Process-style exit code.
    """
    try:
        client = _configured_client_kwargs(args).get("client") or get_client()
        logger.info(f"Searching for: {args.query}")
        results = client.search_papers(
            args.query, limit=args.limit, raise_on_unavailable=True
        )

        if not results:
            logger.error("No results found.")
            return 1

        table = _output_table(f"Search results for '{args.query}'")
        table.leading = 1
        table.add_column("#", style="dim", width=3)
        table.add_column("Paper / authors", ratio=1)
        table.add_column("Year", justify="right", no_wrap=True)
        table.add_column("Citations", justify="right", no_wrap=True)

        for i, paper in enumerate(results, 1):
            authors_str = ", ".join(a.name for a in paper.authors[:2])
            if len(paper.authors) > 2:
                authors_str += " et al."

            table.add_row(
                str(i),
                Text.assemble(
                    (paper.title, "bold"),
                    ("\n" + authors_str, "dim") if authors_str else "",
                ),
                str(paper.year) if paper.year is not None else "",
                f"{paper.citation_count:,}",
            )

        console.output_console.print(table)
        console.output_console.print("\n[dim]Full paper IDs:[/dim]")
        for i, paper in enumerate(results, 1):
            console.output_console.print(
                Text.assemble((f"{i}. ", "dim"), (paper.paper_id, "cyan")),
                soft_wrap=True,
            )
        console.output_console.print(
            '\nUse a paper ID with: citemesh build "<ID>"', style="dim"
        )

    except SemanticScholarUnavailableError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1
    return 0


def _run_search_command(
    args: argparse.Namespace,
    build_parser: argparse.ArgumentParser,
    user_config: UserConfig,
) -> int:
    """Dispatch `citemesh search` across local and S2 modes.

    Mode ``auto`` prefers the local embedding cache when it has vectors and
    falls back to the Semantic Scholar API otherwise, logging which one ran.
    Explicitly requested local mode treats an empty cache as an error.

    :param argparse.Namespace args: Parsed search command arguments.
    :param argparse.ArgumentParser build_parser: Build subparser used to
        derive build-equivalent defaults for local search.
    :param UserConfig user_config: Loaded user configuration snapshot.
    :return int: Process-style exit code.
    """
    mode, origin = _resolve_search_mode(args, user_config)
    if args.model or args.model_profile or args.device:
        if args.mode == "s2":
            logger.error(
                "--model, --model-profile, and --device only apply to local "
                "semantic search; drop them or use --mode local."
            )
            return 2
        if origin != "flag" and mode != "local":
            # Local-only flags are explicit local intent; they outrank a
            # config-level s2/auto default but never an explicit --mode, so
            # --mode auto keeps its S2 fallback.
            mode, origin = "local", "namespace-flag"
    if args.device:
        try:
            resolve_embedding_device(args.device)
        except ValueError as exc:
            logger.error(str(exc))
            return 2

    if mode == "s2":
        if origin == "config":
            logger.info(
                "Searching the Semantic Scholar API (defaults.search_mode = "
                "'s2' in %s).",
                user_config.path,
            )
        return _run_s2_search(args)

    builder: EmbeddingGraphBuilder | None = None
    try:
        builder, defaults = _prepare_local_search_builder(
            args, build_parser, user_config
        )
        # Resolve the artifact before opening a namespace: opening the provisional
        # cache just to count rows leaves unused metadata and lock files behind.
        if builder.has_persistent_embedding_artifacts():
            builder.prepare_embedding_cache()
            cached_count = builder.embedding_cache.embedding_count()
        else:
            cached_count = 0
    except Exception as exc:
        runtime_selectors = ""
        if builder is not None:
            runtime_selectors = (
                f" (device={builder.device} compute_dtype={builder.compute_dtype})"
            )
        if mode == "auto":
            logger.info(
                "Local semantic search unavailable%s (%s); searching the "
                "Semantic Scholar API instead.",
                runtime_selectors,
                exc,
            )
            return _run_s2_search(args)
        logger.error(
            "Local search unavailable%s: %s",
            runtime_selectors,
            exc,
            exc_info=logging.getLogger().level == logging.DEBUG,
        )
        return 1

    if mode == "auto":
        if cached_count > 0:
            logger.info(
                "Searching %s locally cached embeddings (model=%s). "
                "Use --mode s2 for Semantic Scholar keyword search.",
                f"{cached_count:,}",
                defaults.model,
            )
            return _render_local_search(args, builder, defaults)
        logger.info(
            "Local embedding cache is empty for model=%s semantic-source=%s; "
            "searching the Semantic Scholar API instead. The namespace is keyed "
            "by model, profile, truncate dim, storage precision, and formatter "
            "(never the device), so run `citemesh build` with this configuration "
            "to populate it. To search an existing arXiv-corpus build instead, "
            "run `citemesh config set defaults.semantic_source arxiv-corpus` "
            "(plus defaults.dataset_source when it is not the default). Use "
            "--mode s2 for keyword search.",
            defaults.model,
            defaults.semantic_source,
        )
        return _run_s2_search(args)

    # Explicit local mode: an empty cache is an error, not a fallback.
    if cached_count == 0:
        if origin == "flag":
            requested_via = "--mode local"
        elif origin == "namespace-flag":
            requested_via = (
                "--model/--model-profile/--device (these flags imply local search)"
            )
        else:
            requested_via = f"defaults.search_mode in {user_config.path}"
        logger.error(
            "Local search was requested via %s, but the local embedding cache "
            "has no vectors for model=%s semantic-source=%s. The namespace is "
            "keyed by model, profile, truncate dim, storage precision, and "
            "formatter (never the device), so run `citemesh build` with this "
            "configuration to populate it. To search an existing arXiv-corpus "
            "build instead, run `citemesh config set defaults.semantic_source "
            "arxiv-corpus` (plus defaults.dataset_source when it is not the "
            "default). Use --mode s2 for keyword search.",
            requested_via,
            defaults.model,
            defaults.semantic_source,
        )
        return 1
    return _render_local_search(args, builder, defaults)
