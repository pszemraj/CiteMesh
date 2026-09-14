"""Validation of the ``build`` CLI contract and strategy graph construction.

Owns the parser error sinks, cross-option validation for ``build`` (including
config-sourced defaults), side-effect logging, and dispatch of validated arguments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    _STRATEGY_DISPATCH,
    _apply_hybrid_default_overrides,
    _build_option_label,
    _embedding_branch_enabled,
    _normalized_cache_reason,
    _resolved_hybrid_max_semantic,
)
from .cache_ops import _embedding_cache_directory_stats
from .console import logger


class _ParserErrorSink(Protocol):
    """Protocol for argparse-compatible validation error sinks."""

    def error(self, message: str) -> NoReturn:
        """Stop validation with a user-facing error.

        :param str message: Validation failure message.
        :raises SystemExit: Argparse implementations terminate CLI parsing.
        """
        ...


class _BuildContractValueError(ValueError):
    """Catchable build-contract failure with its contributing destinations."""

    def __init__(
        self, message: str, *, related_dests: set[str] | frozenset[str] = frozenset()
    ) -> None:
        """Record the option destinations associated with one validation error.

        :param str message: User-facing validation failure message.
        :param Set[str] related_dests: Option destinations relevant to the failure.
        :return None: Initializes the exception and immutable destination set.
        """
        super().__init__(message)
        self.related_dests = frozenset(related_dests)


class _ValueErrorParserErrorSink:
    """Translate parser-style validation failures into catchable exceptions."""

    @staticmethod
    def error(message: str) -> NoReturn:
        """Raise a parser-style validation message as ``ValueError``.

        :param str message: Validation failure message.
        :raises ValueError: Always raised with ``message``.
        """
        raise _BuildContractValueError(message)

    @staticmethod
    def error_for_dests(message: str, related_dests: set[str]) -> NoReturn:
        """Raise a validation error retaining its contributing destinations.

        :param str message: Validation failure message.
        :param Set[str] related_dests: Option destinations relevant to the failure.
        :return NoReturn: Always raises a catchable validation error.
        :raises _BuildContractValueError: Always raised with ``message``.
        """
        raise _BuildContractValueError(message, related_dests=related_dests)


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
    related_dests: set[str] | frozenset[str] = frozenset(),
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
    resolved_message = f"{message}{context}"
    if isinstance(error_sink, _ValueErrorParserErrorSink):
        error_sink.error_for_dests(resolved_message, related_dests)
    error_sink.error(resolved_message)


@dataclass(frozen=True)
class _BuildContractContext:
    """Inputs shared by every check in one ``build`` contract validation pass.

    :param argparse.Namespace args: Parsed build arguments, normalized in place.
    :param _ParserErrorSink error_sink: Parser-like validation error sink.
    :param set[str] contract_provided: Destinations that count as explicitly
        supplied for gating, after effective no-ops have been discarded.
    :param set[str] config_defaults: Destinations filled from config.toml.
    :param Path | None config_path: Active config.toml path for diagnostics.
    """

    args: argparse.Namespace
    error_sink: _ParserErrorSink
    contract_provided: set[str]
    config_defaults: set[str]
    config_path: Path | None

    @property
    def strategy(self) -> str:
        """Return the selected build strategy token.

        :return str: Strategy name from the parsed namespace.
        """
        return str(self.args.strategy)

    def fail(self, message: str, *, related_dests: set[str]) -> NoReturn:
        """Report a contract failure annotated with any config-default origin.

        :param str message: Base validation failure message.
        :param Set[str] related_dests: Option destinations relevant to the failure.
        :raises SystemExit: When the sink is an argparse parser.
        :raises ValueError: When the sink raises catchable validation errors.
        """
        _build_contract_error(
            self.error_sink,
            message,
            related_dests=related_dests,
            config_defaults=self.config_defaults,
            config_path=self.config_path,
        )

    def reject(self, message: str) -> NoReturn:
        """Report a usage failure that no config default can explain.

        :param str message: Validation failure message.
        :raises SystemExit: When the sink is an argparse parser.
        :raises ValueError: When the sink raises catchable validation errors.
        """
        _build_contract_error(
            self.error_sink,
            message,
            related_dests=set(self.contract_provided),
        )

    def provided_labels(self, dests: set[str]) -> list[str]:
        """List sorted CLI labels for the explicitly provided destinations.

        :param Set[str] dests: Destinations to intersect with explicit argv usage.
        :return list[str]: Sorted option labels; empty when none were supplied.
        """
        return sorted(
            _build_option_label(self.args, dest)
            for dest in self.contract_provided
            if dest in dests
        )


def _reject_unsupported_strategy_options(context: _BuildContractContext) -> None:
    """Reject options the selected strategy does not implement.

    :param _BuildContractContext context: Shared validation inputs.
    :return None: Returns only when every provided option is in scope.
    """
    strategy = context.strategy
    unsupported: list[str] = []
    for dest in sorted(context.contract_provided):
        allowed = _BUILD_STRATEGY_OPTION_SUPPORT.get(dest)
        if allowed is None:
            continue
        if strategy not in allowed:
            unsupported.append(_build_option_label(context.args, dest))
    if unsupported:
        unsupported_text = ", ".join(unsupported)
        context.fail(
            f"Unsupported option(s) for --strategy {strategy}: {unsupported_text}. "
            "Use --help to view strategy-scoped option applicability.",
            related_dests={"strategy"},
        )


def _resolve_implied_semantic_source(
    context: _BuildContractContext,
) -> tuple[list[str], list[str]]:
    """Infer ``--semantic-source`` from the corpus- or candidate-only flags used.

    :param _BuildContractContext context: Shared validation inputs.
    :return tuple[list[str], list[str]]: Provided corpus-only and candidate-only
        option labels, in that order.
    """
    args = context.args
    provided_corpus_flags = context.provided_labels(_CORPUS_ONLY_OPTION_DESTS)
    provided_candidate_flags = context.provided_labels(_CANDIDATE_ONLY_OPTION_DESTS)
    if "semantic_source" not in context.contract_provided:
        if provided_corpus_flags and provided_candidate_flags:
            context.reject(
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
    return provided_corpus_flags, provided_candidate_flags


def _drop_ignored_corpus_config_defaults(context: _BuildContractContext) -> None:
    """Reset corpus-only config defaults a candidates run announced it ignores.

    "Ignored" must mean ignored: the values are reset so the builder never
    receives corpus settings the run said it would not apply.

    :param _BuildContractContext context: Shared validation inputs.
    :return None: Mutates the namespace in place.
    """
    ignored_config_corpus_dests = sorted(
        set(context.config_defaults) & _CORPUS_ONLY_OPTION_DESTS
    )
    if not ignored_config_corpus_dests:
        return
    ignored_text = ", ".join(f"defaults.{dest}" for dest in ignored_config_corpus_dests)
    source = (
        str(context.config_path) if context.config_path is not None else "config.toml"
    )
    logger.info(
        "Ignoring corpus-only config default(s) %s from %s because "
        "the effective semantic source is candidates; set "
        "defaults.semantic_source='arxiv-corpus' to apply them.",
        ignored_text,
        source,
    )
    for dest in ignored_config_corpus_dests:
        setattr(context.args, dest, _CORPUS_ONLY_OPTION_BUILTIN_DEFAULTS[dest])


def _validate_semantic_source_options(
    context: _BuildContractContext,
    provided_corpus_flags: list[str],
    provided_candidate_flags: list[str],
) -> None:
    """Enforce which options each semantic source accepts.

    :param _BuildContractContext context: Shared validation inputs.
    :param list[str] provided_corpus_flags: Corpus-only labels found in argv.
    :param list[str] provided_candidate_flags: Candidate-only labels found in argv.
    :return None: Normalizes candidate-mode storage precision in place.
    """
    args = context.args
    if args.semantic_source != "arxiv-corpus":
        _drop_ignored_corpus_config_defaults(context)
        if provided_corpus_flags:
            option_text = ", ".join(provided_corpus_flags)
            context.fail(
                f"Corpus-only option(s) require --semantic-source arxiv-corpus: "
                f"{option_text}.",
                related_dests={"semantic_source"},
            )
        if (
            "storage_precision" in context.contract_provided
            and args.storage_precision == "int8"
        ):
            context.fail(
                "--storage-precision int8 requires --semantic-source "
                "arxiv-corpus (int8 calibration ranges are computed during "
                "corpus hydration).",
                related_dests={"semantic_source", "storage_precision"},
            )
        if args.storage_precision == "int8":
            # Normalize the implicit int8 default to candidate-mode storage.
            args.storage_precision = "float32"
            logger.debug(
                "Candidate mode stores embeddings as float32 "
                "(int8 calibration requires corpus hydration)."
            )
    elif provided_candidate_flags:
        context.fail(
            "--candidate-pool-size requires --semantic-source candidates.",
            related_dests={"semantic_source"},
        )


def _validate_streaming_dataset_split(context: _BuildContractContext) -> None:
    """Reject sliced dataset splits that streaming corpus reads cannot honor.

    :param _BuildContractContext context: Shared validation inputs.
    :return None: Returns when the split and streaming flags are compatible.
    """
    args = context.args
    if (
        args.semantic_source == "arxiv-corpus"
        and args.streaming
        and ":" in str(args.dataset_split)
    ):
        context.fail(
            "Streaming mode does not support sliced --dataset-split values "
            "(for example train[:5%]). Use unsliced split (e.g. train) or "
            "disable --streaming.",
            related_dests={"dataset_split", "streaming"},
        )


def _validate_cache_flag_combinations(context: _BuildContractContext) -> None:
    """Enforce the destructive cache flags that require an explicit rebuild.

    :param _BuildContractContext context: Shared validation inputs.
    :return None: Returns when the cache flag combination is coherent.
    """
    args = context.args
    if bool(args.overwrite_cache) and not bool(args.force_rebuild_cache):
        context.reject("--overwrite-cache requires --force-rebuild-cache.")
    if _normalized_cache_reason(
        getattr(args, "cache_overwrite_reason", None)
    ) and not bool(args.force_rebuild_cache):
        context.reject("--cache-overwrite-reason requires --force-rebuild-cache.")
    if args.all_corpus and "corpus_size" in context.contract_provided:
        context.reject("--all-corpus cannot be combined with explicit --corpus-size.")


def _validate_embedding_device(context: _BuildContractContext) -> None:
    """Fail fast when an explicitly requested encode device is unavailable.

    :param _BuildContractContext context: Shared validation inputs.
    :return None: Returns when the device resolves in this runtime.
    """
    args = context.args
    if _embedding_branch_enabled(args) and str(args.device) != "auto":
        try:
            resolve_embedding_device(args.device)
        except ValueError as exc:
            context.fail(str(exc), related_dests={"device"})


def _normalize_cache_compression(context: _BuildContractContext) -> None:
    """Validate the cache compression filter and normalize its level.

    :param _BuildContractContext context: Shared validation inputs.
    :return None: Writes the normalized filter and level back to the namespace.
    """
    args = context.args
    try:
        args.cache_compression = validate_compression_filter(
            str(args.cache_compression)
        )
    except ValueError as exc:
        context.reject(str(exc))
    try:
        resolved_compression_level = int(args.cache_compression_level)
    except (TypeError, ValueError):
        context.reject("--cache-compression-level must be an integer.")
    if resolved_compression_level < 0:
        context.reject("--cache-compression-level must be at least 0.")
    if args.cache_compression == "lzf":
        if "cache_compression_level" in context.contract_provided:
            context.reject(
                "--cache-compression-level is unsupported with --cache-compression lzf."
            )
        resolved_compression_level = 0
    args.cache_compression_level = int(resolved_compression_level)


def _validate_storage_precision_options(context: _BuildContractContext) -> None:
    """Gate the int8-only storage options and normalize their inert defaults.

    :param _BuildContractContext context: Shared validation inputs.
    :return None: Writes effective non-int8 values back to the namespace.
    """
    args = context.args
    if str(args.storage_precision) == "int8":
        return
    if "binary_prefilter" in context.contract_provided and bool(args.binary_prefilter):
        context.fail(
            "--binary-prefilter requires --storage-precision int8.",
            related_dests={"binary_prefilter", "storage_precision"},
        )
    if "binary_rescore_multiplier" in context.contract_provided:
        context.fail(
            "--binary-rescore-multiplier requires --storage-precision int8.",
            related_dests={"binary_rescore_multiplier", "storage_precision"},
        )
    if "calibration_sample_size" in context.contract_provided:
        context.fail(
            "--calibration-sample-size requires --storage-precision int8.",
            related_dests={"calibration_sample_size", "storage_precision"},
        )
    # Normalize implicit non-int8 defaults to effective values to avoid
    # strategy-level runtime warnings about ignored options.
    args.binary_prefilter = False
    args.binary_rescore_multiplier = 1
    args.calibration_sample_size = EMBEDDING_STORAGE_CONFIG.calibration_sample_size


def _validate_hybrid_semantic_budget(context: _BuildContractContext) -> None:
    """Check the hybrid semantic budget and reject options it would silently ignore.

    :param _BuildContractContext context: Shared validation inputs.
    :return None: Returns when the hybrid budget and options agree.
    """
    args = context.args
    if args.max_semantic is not None and int(args.max_semantic) >= int(args.max_papers):
        context.fail(
            "--max-semantic must be between 0 and --max-papers - 1 for hybrid.",
            related_dests={"max_papers", "max_semantic"},
        )
    if _resolved_hybrid_max_semantic(args) != 0:
        return

    ignored_embedding_options = context.provided_labels(_HYBRID_EMBEDDING_OPTION_DESTS)
    if not ignored_embedding_options:
        return
    option_text = ", ".join(ignored_embedding_options)
    if args.max_semantic is None:
        message = (
            "Hybrid semantic branch is disabled (effective --max-semantic "
            "is 0 from --max-papers defaulting); remove embedding-only "
            f"option(s): {option_text}."
        )
    elif "max_semantic" in context.config_defaults:
        message = (
            "Hybrid semantic branch is disabled by defaults.max_semantic=0; "
            f"remove embedding-only option(s): {option_text}."
        )
    else:
        message = (
            "Hybrid semantic branch is disabled with --max-semantic 0; "
            f"remove embedding-only option(s): {option_text}."
        )
    context.fail(message, related_dests={"max_semantic", "max_papers"})


def _validate_build_cli_contract(
    args: argparse.Namespace,
    build_parser: _ParserErrorSink,
    provided: set[str],
    config_defaults: set[str] = frozenset(),
    config_path: Path | None = None,
) -> None:
    """Validate strategy-scoped and dependent build options before execution.

    Checks run in a fixed order so the first error reported for any given argv
    is stable: strategy scope, then semantic-source implication and its option
    gates, then streaming, cache, device, compression, and storage precision,
    and finally the hybrid budget.

    :param argparse.Namespace args: Parsed build arguments.
    :param _ParserErrorSink build_parser: Parser-like validation error sink.
    :param Set[str] provided: Explicit option destinations found in argv.
    :param Set[str] config_defaults: Destinations filled from config.toml; they
        outrank built-in defaults (hybrid implicit budgets) but never count as
        explicit flags for strategy gating or corpus-mode implication.
    :param Path | None config_path: Active config.toml path for diagnostics.
    :return None: Mutates normalized args for effective no-op elimination.
    """
    _apply_hybrid_default_overrides(args, provided | set(config_defaults))
    contract_provided = set(provided)
    context = _BuildContractContext(
        args=args,
        error_sink=build_parser,
        contract_provided=contract_provided,
        config_defaults=set(config_defaults),
        config_path=config_path,
    )

    _reject_unsupported_strategy_options(context)
    if not bool(getattr(args, "streaming", False)):
        contract_provided.discard("streaming")

    if context.strategy in {"embedding", "hybrid"}:
        corpus_flags, candidate_flags = _resolve_implied_semantic_source(context)
        _validate_semantic_source_options(context, corpus_flags, candidate_flags)
        _validate_streaming_dataset_split(context)
        _validate_cache_flag_combinations(context)
        _validate_embedding_device(context)
        _normalize_cache_compression(context)
        _validate_storage_precision_options(context)

    if context.strategy == "hybrid":
        _validate_hybrid_semantic_budget(context)


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


def _build_strategy_graph(
    args: argparse.Namespace, strategy: str
) -> tuple[nx.Graph, str]:
    """Dispatch a namespace already normalized by the build command.

    :param argparse.Namespace args: Validated build arguments.
    :param str strategy: Selected strategy name.
    :return tuple[nx.Graph, str]: Graph and normalized seed paper ID.
    :raises ValueError: If the strategy is unsupported.
    """
    if strategy not in _STRATEGY_DISPATCH:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return _STRATEGY_DISPATCH[strategy](args).build_graph(args.paper_id)
