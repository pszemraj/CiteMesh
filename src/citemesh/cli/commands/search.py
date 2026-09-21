"""``citemesh search``: local embedding-cache search and Semantic Scholar lookup."""

from __future__ import annotations

import argparse
import logging
import shlex
from collections.abc import Iterable, Sequence
from contextlib import nullcontext

from rich.text import Text

from citemesh.data.embedding_cache import EmbeddingCache, _corpus_size_from_token
from citemesh.data.user_config import UserConfig
from citemesh.services import SemanticScholarUnavailableError, get_client
from citemesh.strategies.embedding import (
    EmbeddingCacheFingerprintMismatchError,
    EmbeddingGraphBuilder,
    resolve_embedding_device,
)

from .. import console
from ..build_contract import (
    _BuildContractValueError,
    _validate_build_cli_contract,
    _ValueErrorParserErrorSink,
)
from ..build_options import (
    _apply_user_config_defaults,
    _configured_client_kwargs,
    _resolve_user_config_api_key,
    _shared_embedding_builder_kwargs,
)
from ..console import logger
from ..parser import _output_table, _pop_tracked_option_dests

#: Search flags that override a build-equivalent namespace selector. They share
#: their destinations with the build parser and all default to ``None``, so a
#: set value is exactly the set of options this invocation supplied.
_NAMESPACE_OVERRIDE_DESTS: frozenset[str] = frozenset(
    {
        "model",
        "model_profile",
        "model_revision",
        "device",
        "semantic_source",
        "dataset_source",
        "truncate_dim",
        "storage_precision",
        "calibration_sample_size",
    }
)


def _namespace_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Return cache-selection values supplied directly to ``search``.

    Every search selector defaults to ``None``, including integer selectors, so
    this single presence test stays aligned with the parser contracts.

    :param argparse.Namespace args: Parsed search command arguments.
    :return dict[str, object]: Supplied selector destinations and their values.
    """
    return {
        dest: value
        for dest in _NAMESPACE_OVERRIDE_DESTS
        if (value := getattr(args, dest, None)) is not None
    }


def _namespace_override_labels(overrides: dict[str, object]) -> list[str]:
    """Return stable CLI labels for supplied search namespace selectors.

    :param dict[str, object] overrides: Supplied selector destinations and values.
    :return list[str]: Sorted long-option labels.
    """
    return sorted(f"--{dest.replace('_', '-')}" for dest in overrides)


class _LocalSearchOptionError(ValueError):
    """A namespace-flag combination on this command line the contract refuses.

    Distinguished from every other local-search failure because it is a usage
    error, not an unavailable cache: ``auto`` mode must report the bad flags
    rather than treat them as a reason to quietly search Semantic Scholar. A
    contract failure this invocation did not cause — a device in config.toml
    that does not resolve, say — stays an availability failure and keeps the
    fallback it has always had.
    """


def _empty_cache_alternative_guidance(defaults: argparse.Namespace) -> str:
    """Describe the next search option after an empty local namespace.

    :param argparse.Namespace defaults: Effective build-equivalent namespace.
    :return str: Guidance that does not repeat the active source selector.
    """
    if defaults.semantic_source == "arxiv-corpus":
        return (
            "This is already the arXiv-corpus namespace; use the same "
            "--dataset-source when building it. Use --mode s2 for keyword search."
        )
    return (
        "To search an existing arXiv-corpus build instead, pass "
        "--semantic-source arxiv-corpus (plus --dataset-source when it is not "
        "the default). Use --mode s2 for keyword search."
    )


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
    namespace a default build writes to; ``--model``, ``--model-profile``,
    ``--model-revision``, ``--device``, ``--semantic-source``,
    ``--dataset-source``, ``--truncate-dim``, ``--storage-precision``, and
    ``--calibration-sample-size`` override.

    :param argparse.Namespace args: Parsed search command arguments.
    :param argparse.ArgumentParser build_parser: Build subparser used to
        derive build-equivalent defaults.
    :param UserConfig user_config: Loaded user configuration snapshot.
    :return Tuple[EmbeddingGraphBuilder, argparse.Namespace]: Builder and the
        effective build-equivalent defaults namespace.
    :raises _LocalSearchOptionError: If the supplied namespace flags do not form
        a build configuration the contract accepts.
    """
    defaults = build_parser.parse_args(["local-search-placeholder-seed"])
    defaults._s2_api_key = _resolve_user_config_api_key(user_config)
    _pop_tracked_option_dests(defaults)
    config_default_dests = _apply_user_config_defaults(
        defaults, {"strategy"}, user_config
    )
    defaults.strategy = "embedding"
    overrides = _namespace_overrides(args)
    for dest, value in overrides.items():
        setattr(defaults, dest, value)
        config_default_dests.discard(dest)
    # The overrides must land before contract validation: it coerces int8
    # storage to float32 outside arxiv-corpus mode, and storage precision is
    # part of the cache namespace, so a late override would compute a
    # float32 namespace that no corpus build ever wrote.
    #
    # The destinations this invocation actually supplied go with them, so the
    # build contract applies its own rules to them instead of search restating
    # a subset: --dataset-source alone implies arxiv-corpus, and pairing it with
    # --semantic-source candidates is the contradiction the contract rejects.
    # Passed an empty set, the contract can only see the resolved values, and a
    # contradiction reads as a plain candidates search that silently discards
    # the dataset the user named.
    provided_dests = set(overrides)
    try:
        _validate_build_cli_contract(
            defaults,
            _ValueErrorParserErrorSink(),
            provided_dests,
            config_defaults=config_default_dests,
            config_path=user_config.path,
        )
    except _BuildContractValueError as exc:
        if not provided_dests.intersection(exc.related_dests):
            # This command line did not produce the failing value, so it stays
            # what it has always been: a config-sourced reason local search is
            # unavailable, which auto mode may answer from Semantic Scholar.
            raise
        raise _LocalSearchOptionError(str(exc)) from exc
    builder = EmbeddingGraphBuilder(**_shared_embedding_builder_kwargs(defaults))
    return builder, defaults


def _apply_active_model_identity(
    defaults: argparse.Namespace, builder: EmbeddingGraphBuilder
) -> None:
    """Record the checkpoint that local search actually opened.

    Model loading may replace the requested default checkpoint with its configured
    fallback. The active model name must then drive result labels and the generated
    build command because cache fingerprinting already binds the namespace to that
    fallback. The remaining selectors stay as validated: an explicit revision
    disables fallback, while the profile token, truncation, storage precision, and
    calibration size reproduce the active builder contract without changing its
    compute or storage policy.

    :param argparse.Namespace defaults: Effective build-equivalent namespace.
    :param EmbeddingGraphBuilder builder: Prepared builder with an active model.
    :return None: Updates the namespace model selector in place when resolved.
    """
    active_model_name = str(getattr(builder, "_active_model_name", "") or "").strip()
    if active_model_name:
        defaults.model = active_model_name


def _local_result_build_command(
    defaults: argparse.Namespace, cache: EmbeddingCache
) -> str | None:
    """Build a shell-safe command targeting the local search namespace.

    :param argparse.Namespace defaults: Effective build-equivalent defaults.
    :param EmbeddingCache cache: Cache whose recorded corpus scope was searched.
    :return str | None: Copyable embedding build command with namespace selectors,
        or ``None`` when the cache lacks the corpus scope needed to reproduce it.
    """
    command = [
        "citemesh",
        "build",
        "<ID>",
        "--strategy",
        "embedding",
        "--model",
        str(defaults.model),
        "--model-profile",
        str(defaults.model_profile),
        "--semantic-source",
        str(defaults.semantic_source),
        "--storage-precision",
        str(defaults.storage_precision),
    ]
    if defaults.model_revision is not None:
        command.extend(["--model-revision", str(defaults.model_revision)])
    if defaults.truncate_dim is not None:
        command.extend(["--truncate-dim", str(defaults.truncate_dim)])
    if defaults.semantic_source == "arxiv-corpus":
        with cache.hydration_operation_lock():
            stats = cache.payload_stats()
        if stats.hydration_split is None:
            return None
        command.extend(["--dataset-source", str(defaults.dataset_source)])
        command.extend(["--dataset-split", stats.hydration_split])
        recorded_corpus_size = _corpus_size_from_token(stats.hydration_corpus_size)
        if recorded_corpus_size is None:
            command.append("--all-corpus")
        else:
            command.extend(["--corpus-size", str(recorded_corpus_size)])
    if defaults.storage_precision == "int8":
        command.extend(
            ["--calibration-sample-size", str(defaults.calibration_sample_size)]
        )
    return shlex.join(command)


def _print_search_results(
    title: str,
    rows: Iterable[tuple[str, str, Sequence[str], object, str]],
    *,
    metric: str,
    metric_width: int | None = None,
    summary: str | None = None,
) -> None:
    """Render ranked papers with source-specific values and copyable identifiers.

    :param str title: Table heading.
    :param Iterable rows: Paper ID, title, author names, year, and formatted metric.
    :param str metric: Label for the source-specific final column.
    :param int | None metric_width: Fixed metric width, or an unwrapped auto width.
    :param str | None summary: Optional context between the table and identifiers.
    :return None: Prints the table, optional summary, and full paper IDs.
    """
    table = _output_table(title)
    table.leading = 1
    table.add_column("#", style="dim", width=3)
    table.add_column("Paper / authors", ratio=1)
    table.add_column("Year", justify="right", no_wrap=True)
    table.add_column(
        metric, justify="right", width=metric_width, no_wrap=metric_width is None
    )
    paper_ids = []
    for i, (paper_id, paper_title, authors, year, value) in enumerate(rows, 1):
        authors_str = ", ".join(authors[:2])
        if len(authors) > 2:
            authors_str += " et al."
        table.add_row(
            str(i),
            Text.assemble(
                (paper_title, "bold"),
                ("\n" + authors_str, "dim") if authors_str else "",
            ),
            str(year) if year is not None else "",
            value,
        )
        paper_ids.append(paper_id)
    console.output_console.print(table)
    if summary is not None:
        console.output_console.print(Text(summary, style="dim"))
    console.output_console.print("\n[dim]Full paper IDs:[/dim]")
    for i, paper_id in enumerate(paper_ids, 1):
        console.output_console.print(
            Text.assemble((f"{i}. ", "dim"), (paper_id, "cyan")),
            soft_wrap=True,
        )


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
    cache = builder.embedding_cache
    operation_lock = (
        cache.hydration_operation_lock()
        if defaults.semantic_source == "arxiv-corpus"
        else nullcontext()
    )
    try:
        with operation_lock:
            results = builder.search_local(args.query, top_k=args.limit)
            total = getattr(cache, "last_search_total_embeddings", None)
            build_command = (
                _local_result_build_command(defaults, cache) if results else None
            )
    except Exception as exc:
        logger.error(
            "Local search failed: %s",
            exc,
            exc_info=logging.getLogger().level == logging.DEBUG,
        )
        return 1

    if not results:
        logger.error(
            "Local search returned no results for model=%s (cache: %s).",
            defaults.model,
            cache.h5_path,
        )
        return 1

    rows = []
    for result in results:
        metadata = result.metadata or {}
        rows.append(
            (
                str(result.paper_id),
                str(metadata.get("title") or ""),
                [str(name) for name in (metadata.get("authors") or [])],
                metadata.get("year"),
                f"{float(result.score):.3f}",
            )
        )
    _print_search_results(
        f"Local semantic search for '{args.query}'",
        rows,
        metric="Score",
        metric_width=6,
        summary=(
            f"Searched {int(total):,} locally cached embeddings "
            f"(model={defaults.model}, source={defaults.semantic_source})."
            if total is not None
            else None
        ),
    )
    if build_command is None:
        logger.warning(
            "Build command omitted because the local corpus cache has no recorded "
            "dataset split. Search results remain usable; build a returned paper "
            "ID with the same embedding settings and an explicit --dataset-split."
        )
    else:
        console.output_console.print(
            Text(f"\nUse a paper ID with: {build_command}", style="dim"),
            soft_wrap=True,
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

        _print_search_results(
            f"Search results for '{args.query}'",
            (
                (
                    paper.paper_id,
                    paper.title,
                    [author.name for author in paper.authors],
                    paper.year,
                    f"{paper.citation_count:,}",
                )
                for paper in results
            ),
            metric="Citations",
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
    overrides = _namespace_overrides(args)
    override_labels = _namespace_override_labels(overrides)
    if overrides:
        if args.mode == "s2":
            logger.error(
                "%s only apply to local semantic search; drop them or use "
                "--mode local.",
                ", ".join(override_labels),
            )
            return 2
        # An already-local default retains its mode provenance; selectors still
        # choose which namespace that local search reads.
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
            logger.debug(
                "Search mode resolved to Semantic Scholar from config: %s.",
                user_config.path,
            )
        return _run_s2_search(args)

    builder: EmbeddingGraphBuilder | None = None
    try:
        try:
            builder, defaults = _prepare_local_search_builder(
                args, build_parser, user_config
            )
        except ImportError as exc:
            if mode == "auto":
                logger.info(
                    "Local semantic search dependencies are unavailable; "
                    "searching the Semantic Scholar API instead."
                )
                logger.debug("Local search dependency details: %s", exc, exc_info=True)
                return _run_s2_search(args)
            raise
        # Resolve the artifact before opening a namespace: opening the provisional
        # cache just to count rows leaves unused metadata and lock files behind.
        if builder.has_persistent_embedding_artifacts():
            builder.prepare_embedding_cache()
            _apply_active_model_identity(defaults, builder)
            cached_count = builder.embedding_cache.embedding_count()
        else:
            cached_count = 0
    except _LocalSearchOptionError as exc:
        # A usage error, so it is reported in every mode: falling back to S2
        # would answer a question the flags say was not asked.
        logger.error("%s", exc)
        return 2
    except EmbeddingCacheFingerprintMismatchError as exc:
        # This refusal preserves a costly hydrated corpus. Returning keyword
        # results would hide the operator action needed to resolve it.
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        runtime_selectors = ""
        if builder is not None:
            runtime_selectors = (
                f" (device={builder.device} compute_dtype={builder.compute_dtype})"
            )
        if mode == "auto":
            logger.warning(
                "Local semantic search unavailable; searching the Semantic Scholar "
                "API instead."
            )
            logger.debug(
                "Local search fallback details%s: %s",
                runtime_selectors,
                exc,
                exc_info=True,
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
            logger.info("Searching locally cached embeddings...")
            logger.debug(
                "Local search cache: embeddings=%s, model=%s.",
                f"{cached_count:,}",
                defaults.model,
            )
            # Query failures stay visible once local results were selected.
            return _render_local_search(args, builder, defaults)
        logger.info(
            "No local embeddings available; searching the Semantic Scholar API instead."
        )
        logger.debug(
            "Empty local embedding namespace: model=%s, semantic_source=%s, "
            "dataset_source=%s. Namespace keys include model, revision, profile, "
            "semantic source, truncate dim, storage precision, int8 calibration "
            "size, and formatter; corpus dataset source is validated against "
            "hydration metadata. %s",
            defaults.model,
            defaults.semantic_source,
            defaults.dataset_source,
            _empty_cache_alternative_guidance(defaults),
        )
        return _run_s2_search(args)

    # Explicit local mode: an empty cache is an error, not a fallback.
    if cached_count == 0:
        if origin == "flag":
            requested_via = "--mode local"
        elif origin == "namespace-flag":
            requested_via = (
                f"{', '.join(override_labels)} (these flags imply local search)"
            )
        else:
            requested_via = f"defaults.search_mode in {user_config.path}"
        logger.error(
            "Local search was requested via %s, but the local embedding cache "
            "has no vectors for model=%s semantic-source=%s dataset-source=%s. "
            "The namespace is keyed by model, revision, profile, semantic source, "
            "truncate dim, storage precision, int8 calibration size, and formatter "
            "(never the device); corpus dataset source is validated against its "
            "hydration metadata. Run `citemesh build` with this configuration to "
            "populate it. %s",
            requested_via,
            defaults.model,
            defaults.semantic_source,
            defaults.dataset_source,
            _empty_cache_alternative_guidance(defaults),
        )
        return 1
    # Local vectors are available and the search source is now selected.
    # Query/scoring failures stay visible instead of changing result semantics.
    return _render_local_search(args, builder, defaults)
