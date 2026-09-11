"""Validation of the ``build`` CLI contract and strategy graph construction.

Owns the parser error sinks, cross-option validation for ``build`` (including
config-sourced defaults), side-effect logging, programmatic-invocation value
checks, and the dispatch that turns a validated namespace into a graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NoReturn, Protocol

import networkx as nx

from citemesh.core import EMBEDDING_STORAGE_CONFIG
from citemesh.data import get_cache_dir, validate_compression_filter
from citemesh.strategies.embedding import resolve_embedding_device

from .build_options import (
    _BUILD_STRATEGY_OPTION_SUPPORT,
    _CANDIDATE_ONLY_OPTION_DESTS,
    _CORPUS_ONLY_OPTION_BUILTIN_DEFAULTS,
    _CORPUS_ONLY_OPTION_DESTS,
    _HYBRID_EMBEDDING_OPTION_DESTS,
    _PROGRAMMATIC_BUILD_VALUE_DESTS,
    _STRATEGY_DISPATCH,
    _apply_hybrid_default_overrides,
    _build_option_label,
    _embedding_branch_enabled,
    _normalized_cache_reason,
    _resolved_hybrid_max_semantic,
)
from .cache_ops import _embedding_cache_directory_stats
from .console import logger
from .parser import _create_parser


class _ParserErrorSink(Protocol):
    """Protocol for argparse-compatible validation error sinks."""

    def error(self, message: str) -> NoReturn:
        """Stop validation with a user-facing error.

        :param str message: Validation failure message.
        :raises SystemExit: Argparse implementations terminate CLI parsing.
        """
        ...


class _ValueErrorParserErrorSink:
    """Translate parser-style validation failures into catchable exceptions."""

    @staticmethod
    def error(message: str) -> NoReturn:
        """Raise a parser-style validation message as ``ValueError``.

        :param str message: Validation failure message.
        :raises ValueError: Always raised with ``message``.
        """
        raise ValueError(message)


def _config_error_context(
    *,
    related_dests: set[str],
    config_defaults: set[str],
    config_path: Path | None,
) -> str:
    """Describe config values that contributed to a contract failure.

    :param Set[str] related_dests: Option destinations relevant to the failure.
    :param Set[str] config_defaults: Destinations filled from config.toml.
    :param Path | None config_path: Active config.toml path when available.
    :return str: Diagnostic suffix, or an empty string for CLI-only failures.
    """
    configured = sorted(related_dests & set(config_defaults))
    if not configured:
        return ""
    key_text = ", ".join(f"defaults.{dest}" for dest in configured)
    source = str(config_path) if config_path is not None else "config.toml"
    return (
        f" Config source: {key_text} in {source}; update or unset the configured value."
    )


def _build_contract_error(
    error_sink: _ParserErrorSink,
    message: str,
    *,
    related_dests: set[str] = frozenset(),
    config_defaults: set[str] = frozenset(),
    config_path: Path | None = None,
) -> NoReturn:
    """Route a build-contract failure through the active error sink.

    :param _ParserErrorSink error_sink: Argparse or exception-raising error sink.
    :param str message: Base validation failure message.
    :param Set[str] related_dests: Option destinations relevant to the failure.
    :param Set[str] config_defaults: Destinations filled from config.toml.
    :param Path | None config_path: Active config.toml path when available.
    :raises SystemExit: When ``error_sink`` is an argparse parser.
    :raises ValueError: When ``error_sink`` raises catchable validation errors.
    """
    context = _config_error_context(
        related_dests=related_dests,
        config_defaults=config_defaults,
        config_path=config_path,
    )
    error_sink.error(f"{message}{context}")


def _validate_build_cli_contract(
    args: argparse.Namespace,
    build_parser: _ParserErrorSink,
    provided: set[str],
    config_defaults: set[str] = frozenset(),
    config_path: Path | None = None,
) -> None:
    """Validate strategy-scoped and dependent build options before execution.

    :param argparse.Namespace args: Parsed build arguments.
    :param _ParserErrorSink build_parser: Parser-like validation error sink.
    :param Set[str] provided: Explicit option destinations found in argv.
    :param Set[str] config_defaults: Destinations filled from config.toml; they
        outrank built-in defaults (hybrid implicit budgets) but never count as
        explicit flags for strategy gating or corpus-mode implication.
    :param Path | None config_path: Active config.toml path for diagnostics.
    :return None: Mutates normalized args for effective no-op elimination.
    """
    strategy = str(args.strategy)
    _apply_hybrid_default_overrides(args, provided | set(config_defaults))
    contract_provided = set(provided)
    unsupported: list[str] = []
    for dest in sorted(contract_provided):
        allowed = _BUILD_STRATEGY_OPTION_SUPPORT.get(dest)
        if allowed is None:
            continue
        if strategy not in allowed:
            unsupported.append(_build_option_label(args, dest))
    if unsupported:
        unsupported_text = ", ".join(unsupported)
        _build_contract_error(
            build_parser,
            f"Unsupported option(s) for --strategy {strategy}: {unsupported_text}. "
            "Use --help to view strategy-scoped option applicability.",
            related_dests={"strategy"},
            config_defaults=config_defaults,
            config_path=config_path,
        )
    if not bool(getattr(args, "streaming", False)):
        contract_provided.discard("streaming")

    if strategy in {"embedding", "hybrid"}:
        provided_corpus_flags = sorted(
            _build_option_label(args, dest)
            for dest in contract_provided
            if dest in _CORPUS_ONLY_OPTION_DESTS
        )
        provided_candidate_flags = sorted(
            _build_option_label(args, dest)
            for dest in contract_provided
            if dest in _CANDIDATE_ONLY_OPTION_DESTS
        )
        if "semantic_source" not in contract_provided:
            if provided_corpus_flags and provided_candidate_flags:
                build_parser.error(
                    "Corpus-only and candidate-only options cannot be combined: "
                    f"{', '.join(provided_corpus_flags + provided_candidate_flags)}."
                )
            if provided_corpus_flags:
                args.semantic_source = "arxiv-corpus"
                logger.debug(
                    "Corpus option(s) %s imply --semantic-source arxiv-corpus.",
                    ", ".join(provided_corpus_flags),
                )
            elif provided_candidate_flags:
                args.semantic_source = "candidates"
                logger.debug(
                    "Candidate option(s) %s imply --semantic-source candidates.",
                    ", ".join(provided_candidate_flags),
                )
        if args.semantic_source != "arxiv-corpus":
            ignored_config_corpus_dests = sorted(
                set(config_defaults) & _CORPUS_ONLY_OPTION_DESTS
            )
            if ignored_config_corpus_dests:
                ignored_text = ", ".join(
                    f"defaults.{dest}" for dest in ignored_config_corpus_dests
                )
                source = str(config_path) if config_path is not None else "config.toml"
                logger.info(
                    "Ignoring corpus-only config default(s) %s from %s because "
                    "the effective semantic source is candidates; set "
                    "defaults.semantic_source='arxiv-corpus' to apply them.",
                    ignored_text,
                    source,
                )
                # "Ignored" must mean ignored: reset the values so the builder
                # never receives corpus settings a candidates run announced it
                # would not apply.
                for dest in ignored_config_corpus_dests:
                    setattr(args, dest, _CORPUS_ONLY_OPTION_BUILTIN_DEFAULTS[dest])
            if provided_corpus_flags:
                option_text = ", ".join(provided_corpus_flags)
                _build_contract_error(
                    build_parser,
                    f"Corpus-only option(s) require --semantic-source arxiv-corpus: "
                    f"{option_text}.",
                    related_dests={"semantic_source"},
                    config_defaults=config_defaults,
                    config_path=config_path,
                )
            if (
                "storage_precision" in contract_provided
                and args.storage_precision == "int8"
            ):
                _build_contract_error(
                    build_parser,
                    "--storage-precision int8 requires --semantic-source "
                    "arxiv-corpus (int8 calibration ranges are computed during "
                    "corpus hydration).",
                    related_dests={"semantic_source"},
                    config_defaults=config_defaults,
                    config_path=config_path,
                )
            if args.storage_precision == "int8":
                # Normalize the implicit int8 default to candidate-mode storage.
                args.storage_precision = "float32"
                logger.debug(
                    "Candidate mode stores embeddings as float32 "
                    "(int8 calibration requires corpus hydration)."
                )
        elif provided_candidate_flags:
            _build_contract_error(
                build_parser,
                "--candidate-pool-size requires --semantic-source candidates.",
                related_dests={"semantic_source"},
                config_defaults=config_defaults,
                config_path=config_path,
            )
        if (
            args.semantic_source == "arxiv-corpus"
            and args.streaming
            and ":" in str(args.dataset_split)
        ):
            _build_contract_error(
                build_parser,
                "Streaming mode does not support sliced --dataset-split values "
                "(for example train[:5%]). Use unsliced split (e.g. train) or "
                "disable --streaming.",
                related_dests={"dataset_split", "streaming"},
                config_defaults=config_defaults,
                config_path=config_path,
            )
        if bool(args.overwrite_cache) and not bool(args.force_rebuild_cache):
            build_parser.error("--overwrite-cache requires --force-rebuild-cache.")
        if _normalized_cache_reason(
            getattr(args, "cache_overwrite_reason", None)
        ) and not bool(args.force_rebuild_cache):
            build_parser.error(
                "--cache-overwrite-reason requires --force-rebuild-cache."
            )
        if args.all_corpus and "corpus_size" in contract_provided:
            build_parser.error(
                "--all-corpus cannot be combined with explicit --corpus-size."
            )
        if _embedding_branch_enabled(args) and str(args.device) != "auto":
            try:
                resolve_embedding_device(args.device)
            except ValueError as exc:
                _build_contract_error(
                    build_parser,
                    str(exc),
                    related_dests={"device"},
                    config_defaults=config_defaults,
                    config_path=config_path,
                )
        try:
            args.cache_compression = validate_compression_filter(
                str(args.cache_compression)
            )
        except ValueError as exc:
            build_parser.error(str(exc))
        try:
            resolved_compression_level = int(args.cache_compression_level)
        except (TypeError, ValueError):
            build_parser.error("--cache-compression-level must be an integer.")
        if resolved_compression_level < 0:
            build_parser.error("--cache-compression-level must be at least 0.")
        if args.cache_compression == "lzf":
            if "cache_compression_level" in contract_provided:
                build_parser.error(
                    "--cache-compression-level is unsupported with "
                    "--cache-compression lzf."
                )
            resolved_compression_level = 0
        args.cache_compression_level = int(resolved_compression_level)
        if str(args.storage_precision) != "int8":
            if "binary_prefilter" in contract_provided and bool(args.binary_prefilter):
                _build_contract_error(
                    build_parser,
                    "--binary-prefilter requires --storage-precision int8.",
                    related_dests={"storage_precision"},
                    config_defaults=config_defaults,
                    config_path=config_path,
                )
            if "binary_rescore_multiplier" in contract_provided:
                _build_contract_error(
                    build_parser,
                    "--binary-rescore-multiplier requires --storage-precision int8.",
                    related_dests={"storage_precision"},
                    config_defaults=config_defaults,
                    config_path=config_path,
                )
            if "calibration_sample_size" in contract_provided:
                _build_contract_error(
                    build_parser,
                    "--calibration-sample-size requires --storage-precision int8.",
                    related_dests={"storage_precision"},
                    config_defaults=config_defaults,
                    config_path=config_path,
                )
            # Normalize implicit non-int8 defaults to effective values to avoid
            # strategy-level runtime warnings about ignored options.
            args.binary_prefilter = False
            args.binary_rescore_multiplier = 1
            args.calibration_sample_size = (
                EMBEDDING_STORAGE_CONFIG.calibration_sample_size
            )

    if strategy == "hybrid":
        if args.max_semantic is not None and int(args.max_semantic) >= int(
            args.max_papers
        ):
            _build_contract_error(
                build_parser,
                "--max-semantic must be between 0 and --max-papers - 1 for hybrid.",
                related_dests={"max_papers", "max_semantic"},
                config_defaults=config_defaults,
                config_path=config_path,
            )
        resolved_max_semantic = _resolved_hybrid_max_semantic(args)

        if resolved_max_semantic == 0:
            ignored_embedding_options = sorted(
                _build_option_label(args, dest)
                for dest in contract_provided
                if dest in _HYBRID_EMBEDDING_OPTION_DESTS
            )
            if ignored_embedding_options:
                option_text = ", ".join(ignored_embedding_options)
                if args.max_semantic is None:
                    message = (
                        "Hybrid semantic branch is disabled (effective --max-semantic "
                        "is 0 from --max-papers defaulting); remove embedding-only "
                        f"option(s): {option_text}."
                    )
                elif "max_semantic" in config_defaults:
                    message = (
                        "Hybrid semantic branch is disabled by defaults.max_semantic=0; "
                        f"remove embedding-only option(s): {option_text}."
                    )
                else:
                    message = (
                        "Hybrid semantic branch is disabled with --max-semantic 0; "
                        f"remove embedding-only option(s): {option_text}."
                    )
                _build_contract_error(
                    build_parser,
                    message,
                    related_dests={"max_semantic", "max_papers"},
                    config_defaults=config_defaults,
                    config_path=config_path,
                )


def _log_build_side_effect_contract(args: argparse.Namespace) -> None:
    """Log build side-effect contract summary for transparency before execution.

    :param argparse.Namespace args: Parsed build arguments.
    :return None: Emits contract details at their appropriate logging levels.
    """
    if not _embedding_branch_enabled(args):
        return

    corpus_mode = str(args.semantic_source) == "arxiv-corpus"
    _, cache_files, _ = _embedding_cache_directory_stats()
    if cache_files == 0:
        if corpus_mode:
            logger.info(
                "No embedding cache found; model and corpus downloads may be "
                "required (network access needed, may take several minutes on "
                "first use)."
            )
        else:
            logger.info(
                "No embedding cache found; the embedding model will be "
                "downloaded on first use (network access needed)."
            )

    cache_root = get_cache_dir("embeddings")
    revision_label = args.model_revision or "default"
    logger.debug("Embedding cache namespace root: %s.", cache_root)
    if corpus_mode:
        corpus_label = (
            "all"
            if args.all_corpus or args.corpus_size is None
            else str(args.corpus_size)
        )
        logger.debug(
            "Embedding config: model=%s@%s device=%s source=arxiv-corpus dataset=%s split=%s corpus=%s streaming=%s storage=%s encode_batch=%s.",
            args.model,
            revision_label,
            args.device,
            args.dataset_source,
            args.dataset_split,
            corpus_label,
            bool(args.streaming),
            args.storage_precision,
            int(args.encode_batch_size),
        )
    else:
        logger.debug(
            "Embedding config: model=%s@%s device=%s source=%s pool=%s storage=%s encode_batch=%s.",
            args.model,
            revision_label,
            args.device,
            args.semantic_source,
            int(args.candidate_pool_size),
            args.storage_precision,
            int(args.encode_batch_size),
        )
    if args.force_rebuild_cache:
        overwrite_reason = _normalized_cache_reason(
            getattr(args, "cache_overwrite_reason", None)
        )
        if bool(args.overwrite_cache):
            logger.warning(
                "--force-rebuild-cache enabled with --overwrite-cache; existing embedding namespace payload will be cleared without prompt."
            )
        else:
            logger.warning(
                "--force-rebuild-cache enabled; existing embedding namespace payload will be cleared after confirmation."
            )
        if overwrite_reason:
            logger.warning(
                "Cache overwrite rationale: %s",
                overwrite_reason,
            )


def _normalize_programmatic_build_value(
    action: argparse.Action,
    value: object,
    parser_error_sink: _ParserErrorSink,
) -> object:
    """Normalize a programmatic build value using the parser action contract.

    :param argparse.Action action: Parser action defining the value contract.
    :param object value: Programmatic value to validate.
    :param _ParserErrorSink parser_error_sink: Parser-like error sink.
    :return object: Normalized value compatible with CLI parsing rules.
    """
    if value is None:
        return None

    primary_label = action.option_strings[0] if action.option_strings else action.dest
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if not isinstance(value, bool):
            parser_error_sink.error(f"{primary_label} must be a boolean.")
        return value

    normalized = value
    if action.type is not None:
        try:
            normalized = action.type(value)
        except argparse.ArgumentTypeError as exc:
            parser_error_sink.error(str(exc))
        except (TypeError, ValueError) as exc:
            parser_error_sink.error(str(exc))

    if action.choices is not None and normalized not in action.choices:
        choices_text = ", ".join(str(choice) for choice in action.choices)
        parser_error_sink.error(f"{primary_label} must be one of: {choices_text}.")
    return normalized


def _validate_programmatic_build_values(
    args: argparse.Namespace,
    build_parser: argparse.ArgumentParser,
    parser_error_sink: _ParserErrorSink,
) -> None:
    """Validate programmatic build namespaces against CLI scalar contracts.

    :param argparse.Namespace args: Candidate build namespace.
    :param argparse.ArgumentParser build_parser: Build parser used for action metadata.
    :param _ParserErrorSink parser_error_sink: Parser-like error sink.
    :return None: Mutates ``args`` with normalized CLI-equivalent values.
    """
    for action in build_parser._actions:
        dest = str(getattr(action, "dest", "") or "")
        if dest not in _PROGRAMMATIC_BUILD_VALUE_DESTS or not hasattr(args, dest):
            continue
        normalized = _normalize_programmatic_build_value(
            action,
            getattr(args, dest),
            parser_error_sink,
        )
        setattr(args, dest, normalized)


def _synchronize_namespace_values(
    target: argparse.Namespace, source: argparse.Namespace
) -> None:
    """Copy validated namespace state back to the caller namespace.

    :param argparse.Namespace target: Namespace mutated in-place.
    :param argparse.Namespace source: Namespace carrying validated CLI-equivalent values.
    :return None: Copies every field from ``source`` onto ``target``.
    """
    for field, value in vars(source).items():
        setattr(target, field, value)


def _build_strategy_graph(
    args: argparse.Namespace,
    strategy: str,
    *,
    validate_contract: bool = True,
    provided: set[str] | None = None,
) -> tuple[nx.Graph, str]:
    """Build a graph for a strategy selected from CLI arguments.

    :param argparse.Namespace args: Parsed arguments.
    :param str strategy: Strategy name.
    :param bool validate_contract: Whether to run strategy-option contract checks.
    :param Optional[Set[str]] provided: Explicit set of option destinations that were
        provided by the caller.  When ``None``, provided fields are inferred by
        comparing namespace values against parser defaults (note: re-specifying a
        default value is invisible to the heuristic).
    :return tuple[nx.Graph, str]: Graph and normalized seed paper ID.
    :raises ValueError: If strategy is unsupported.
    """
    if strategy not in _STRATEGY_DISPATCH:
        raise ValueError(f"Unsupported strategy: {strategy}")

    args_for_validation = argparse.Namespace(**vars(args))
    setattr(args_for_validation, "strategy", strategy)

    if validate_contract:
        parser_snapshot, build_parser_snapshot, _, _ = _create_parser()
        del parser_snapshot
        defaults_namespace = build_parser_snapshot.parse_args(["seed"])
        merged_values = vars(defaults_namespace)
        merged_values.update(vars(args_for_validation))
        args_for_validation = argparse.Namespace(**merged_values)
        setattr(args_for_validation, "strategy", strategy)
        inferred_provided = (
            provided
            if provided is not None
            else _infer_provided_build_option_dests(
                args=args_for_validation,
                build_parser=build_parser_snapshot,
            )
        )

        _validate_programmatic_build_values(
            args_for_validation,
            build_parser_snapshot,
            _ValueErrorParserErrorSink(),
        )
        _validate_build_cli_contract(
            args_for_validation,
            _ValueErrorParserErrorSink(),
            inferred_provided,
        )
        _synchronize_namespace_values(args, args_for_validation)

    builder = _STRATEGY_DISPATCH[strategy].factory(args)
    return builder.build_graph(args.paper_id)


def _infer_provided_build_option_dests(
    args: argparse.Namespace, build_parser: argparse.ArgumentParser
) -> set[str]:
    """Infer likely explicit build options from a parsed namespace.

    This heuristic compares namespace values against parser defaults. It cannot detect
    explicit re-specification of the same default, but it prevents most silent
    programmatic bypasses for strategy-scoped option contracts.

    For precise control, programmatic callers should pass the ``provided`` parameter
    to :func:`_build_strategy_graph` directly, bypassing this heuristic entirely.

    :param argparse.Namespace args: Candidate parsed namespace.
    :param argparse.ArgumentParser build_parser: Build subcommand parser.
    :return Set[str]: Option destinations inferred as explicitly set.
    """
    provided: set[str] = set()
    for action in build_parser._actions:
        if not action.option_strings:
            continue
        dest = action.dest
        if not hasattr(args, dest):
            continue
        current_value = getattr(args, dest)
        if isinstance(action, argparse._AppendAction):
            if current_value is not None:
                provided.add(dest)
        else:
            default_value = build_parser.get_default(dest)
            if current_value != default_value:
                provided.add(dest)
    return provided
