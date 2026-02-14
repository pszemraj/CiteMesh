#!/usr/bin/env python3
"""
CiteMesh: Unified CLI for CiteMesh visualizations.

This is the main entry point for the refactored CiteMesh package,
providing a single interface to all graph building strategies.
"""

import argparse
import copy
import logging
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Protocol, Set, Tuple

import networkx as nx
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from citemesh.core import EMBEDDING_STORAGE_CONFIG
from citemesh.data import get_cache_dir, validate_compression_filter
from citemesh.services import get_client
from citemesh.services.semantic_scholar import normalize_paper_id
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import HybridGraphBuilder
from citemesh.strategies.recommendation import RecommendationGraphBuilder
from citemesh.visualization import (
    GraphExporter,
    compute_layout,
    generate_output_path,
    visualize_graph,
)

log_console = Console(stderr=True)
output_console = Console()
_LOGGING_CONFIGURED = False
logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure CLI logging once at runtime."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=log_console,
                show_time=False,
                show_path=False,
                rich_tracebacks=False,
                markup=True,
            )
        ],
    )
    # Keep third-party HTTP logs concise without import-time side effects.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _LOGGING_CONFIGURED = True


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI argument.

    :param str value: Raw argparse value.
    :return int: Parsed integer.
    :raises argparse.ArgumentTypeError: If value is not >= 1.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    """Parse a non-negative integer CLI argument.

    :param str value: Raw argparse value.
    :return int: Parsed integer.
    :raises argparse.ArgumentTypeError: If value is negative.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def _threshold_float(value: str) -> float:
    """Parse similarity-threshold CLI argument constrained to [0, 1].

    :param str value: Raw argparse value.
    :return float: Parsed threshold value.
    :raises argparse.ArgumentTypeError: If value is outside [0, 1].
    """
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a float") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite float")
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return parsed


def _non_empty_str(value: str) -> str:
    """Parse a non-empty string argument after trimming whitespace.

    :param str value: Raw argparse value.
    :return str: Trimmed non-empty string.
    :raises argparse.ArgumentTypeError: If value is empty/whitespace.
    """
    normalized = str(value).strip()
    if not normalized:
        raise argparse.ArgumentTypeError("must be a non-empty string")
    return normalized


EXPORT_FORMATS = ("png", "html", "plotly", "json", "graphml")
EXPORT_EXTENSIONS: Dict[str, str] = {
    "png": ".png",
    "html": ".html",
    "plotly": ".plotly.html",
    "json": ".json",
    "graphml": ".graphml",
}
KNOWN_EXPORT_SUFFIXES: List[str] = sorted(
    EXPORT_EXTENSIONS.values(), key=len, reverse=True
)


class _StrategyBuilderProtocol(Protocol):
    """Protocol describing strategy builder objects used by CLI dispatch."""

    def build_graph(self, paper_id: str) -> tuple[nx.Graph, str]:
        """Build a graph for a paper identifier.

        :param str paper_id: Raw or normalized seed paper identifier.
        :return tuple[nx.Graph, str]: Built graph and normalized seed paper ID.
        """
        ...


StrategyFactory = Callable[[argparse.Namespace], _StrategyBuilderProtocol]

# Strategy-scoped build options and CLI-token aliases for strict post-parse validation.
_BUILD_STRATEGY_OPTION_SUPPORT: Dict[str, Set[str]] = {
    "max_citations": {"citation", "hybrid"},
    "max_references": {"citation", "hybrid"},
    "similarity_threshold": {"citation", "recommendation"},
    "no_references": {"citation", "recommendation", "hybrid"},
    "refresh_reference_cache": {"citation", "recommendation", "hybrid"},
    "model": {"embedding", "hybrid"},
    "model_revision": {"embedding", "hybrid"},
    "dataset_split": {"embedding", "hybrid"},
    "corpus_size": {"embedding", "hybrid"},
    "all_corpus": {"embedding", "hybrid"},
    "top_k": {"embedding"},
    "truncate_dim": {"embedding", "hybrid"},
    "streaming": {"embedding", "hybrid"},
    "force_rebuild_cache": {"embedding", "hybrid"},
    "storage_precision": {"embedding", "hybrid"},
    "binary_prefilter": {"embedding", "hybrid"},
    "binary_rescore_multiplier": {"embedding", "hybrid"},
    "calibration_sample_size": {"embedding", "hybrid"},
    "cache_compression": {"embedding", "hybrid"},
    "cache_compression_level": {"embedding", "hybrid"},
    "torch_compile": {"embedding", "hybrid"},
    "max_semantic": {"hybrid"},
}
_BUILD_OPTION_FLAGS: Dict[str, List[str]] = {
    "max_citations": ["--max-citations", "-c"],
    "max_references": ["--max-references", "-r"],
    "similarity_threshold": ["--similarity-threshold", "-t"],
    "no_references": ["--no-references"],
    "refresh_reference_cache": ["--refresh-reference-cache"],
    "model": ["--model", "-m"],
    "model_revision": ["--model-revision"],
    "dataset_split": ["--dataset-split"],
    "corpus_size": ["--corpus-size"],
    "all_corpus": ["--all-corpus"],
    "top_k": ["--top-k", "-k"],
    "truncate_dim": ["--truncate-dim"],
    "streaming": ["--streaming"],
    "force_rebuild_cache": ["--force-rebuild-cache"],
    "storage_precision": ["--storage-precision"],
    "binary_prefilter": ["--binary-prefilter", "--no-binary-prefilter"],
    "binary_rescore_multiplier": ["--binary-rescore-multiplier"],
    "calibration_sample_size": ["--calibration-sample-size"],
    "cache_compression": ["--cache-compression"],
    "cache_compression_level": ["--cache-compression-level"],
    "torch_compile": ["--torch-compile", "--no-torch-compile"],
    "max_semantic": ["--max-semantic"],
}
_BUILD_OPTION_PRIMARY_FLAG: Dict[str, str] = {
    dest: flags[0] for dest, flags in _BUILD_OPTION_FLAGS.items()
}
_BUILD_OPTION_DEST_BY_FLAG: Dict[str, str] = {
    flag: dest for dest, flags in _BUILD_OPTION_FLAGS.items() for flag in flags
}
_CACHE_COMPRESSION_CHOICES = ("gzip", "lzf", "szip")
_HYBRID_EMBEDDING_OPTION_DESTS: Set[str] = {
    "model",
    "model_revision",
    "dataset_split",
    "corpus_size",
    "all_corpus",
    "truncate_dim",
    "streaming",
    "force_rebuild_cache",
    "storage_precision",
    "binary_prefilter",
    "binary_rescore_multiplier",
    "calibration_sample_size",
    "cache_compression",
    "cache_compression_level",
    "torch_compile",
}


@dataclass(frozen=True)
class _StrategyDispatchSpec:
    """Strategy dispatch metadata for CLI construction."""

    factory: StrategyFactory


def _shared_embedding_builder_kwargs(cli_args: argparse.Namespace) -> Dict[str, object]:
    """Build shared embedding kwargs for embedding-aware strategy builders.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :return Dict[str, object]: Shared kwargs consumed by embedding/hybrid builders.
    """
    return {
        "model_name": cli_args.model,
        "model_revision": cli_args.model_revision,
        "dataset_split": cli_args.dataset_split,
        "corpus_size": None if cli_args.all_corpus else cli_args.corpus_size,
        "truncate_dim": cli_args.truncate_dim,
        "use_streaming": cli_args.streaming,
        "force_rebuild_cache": cli_args.force_rebuild_cache,
        "storage_precision": cli_args.storage_precision,
        "binary_prefilter": cli_args.binary_prefilter,
        "binary_rescore_multiplier": cli_args.binary_rescore_multiplier,
        "calibration_sample_size": cli_args.calibration_sample_size,
        "cache_compression": cli_args.cache_compression,
        "cache_compression_level": cli_args.cache_compression_level,
        "enable_torch_compile": cli_args.torch_compile,
    }


def _embedding_export_metadata(cli_args: argparse.Namespace) -> Dict[str, object]:
    """Build embedding provenance payload persisted in export metadata.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :return Dict[str, object]: Effective embedding cache/vector provenance fields.
    """
    int8_mode = str(cli_args.storage_precision) == "int8"
    return {
        "effective_vector_dtype": "float32",
        "storage_precision": str(cli_args.storage_precision),
        "binary_prefilter_enabled": bool(cli_args.binary_prefilter and int8_mode),
        "binary_rescore_multiplier": (
            int(cli_args.binary_rescore_multiplier) if int8_mode else 1
        ),
    }


def _strategy_score_contract(strategy: str) -> Dict[str, object]:
    """Return strategy-specific score semantics metadata for export payloads.

    :param str strategy: Active build strategy.
    :return Dict[str, object]: Score semantics metadata.
    """
    base_contract = {
        "strategy": strategy,
        "comparable_across_strategies": False,
        "range_hint": "[0,1] strategy-specific composite score",
    }
    if strategy == "citation":
        return {
            **base_contract,
            "score_type": "citation_similarity_composite",
        }
    if strategy == "recommendation":
        return {
            **base_contract,
            "score_type": "recommendation_similarity_composite",
        }
    if strategy == "embedding":
        return {
            **base_contract,
            "score_type": "embedding_similarity_composite",
        }
    if strategy == "hybrid":
        return {
            **base_contract,
            "score_type": "hybrid_similarity_composite",
            "adjudication_policy": (
                "citation-first union; semantic additions include only new papers "
                "up to max_semantic."
            ),
        }
    return {
        **base_contract,
        "score_type": "unknown",
    }


def _collect_provided_build_option_dests(
    build_parser: argparse.ArgumentParser, argv: List[str]
) -> Set[str]:
    """Return build-option destinations explicitly present in CLI argv.

    :param argparse.ArgumentParser build_parser: Build-subcommand parser.
    :param List[str] argv: Raw argv list without executable name.
    :return Set[str]: Explicitly provided build option destinations.
    """
    if not argv or argv[0] != "build":
        return set()

    provided: Set[str] = set()
    if not build_parser:
        return provided

    probe_parser = copy.deepcopy(build_parser)
    probe_default = object()
    for action in probe_parser._actions:
        if action.option_strings:
            action.default = probe_default
    probe_parser.set_defaults(**{key: probe_default for key in probe_parser._defaults})

    try:
        parsed, _ = probe_parser.parse_known_args(argv[1:])
    except SystemExit:
        return provided

    for action in probe_parser._actions:
        if (
            not action.option_strings
            or not hasattr(parsed, action.dest)
            or getattr(parsed, action.dest) is probe_default
        ):
            continue
        provided.add(action.dest)

    return provided


def _validate_build_cli_contract(
    args: argparse.Namespace, build_parser: argparse.ArgumentParser, provided: Set[str]
) -> None:
    """Validate strategy-scoped and dependent build options before execution.

    :param argparse.Namespace args: Parsed build arguments.
    :param argparse.ArgumentParser build_parser: Build subcommand parser.
    :param Set[str] provided: Explicit option destinations found in argv.
    :return None: Mutates normalized args for effective no-op elimination.
    """
    strategy = str(args.strategy)
    unsupported: List[str] = []
    for dest in sorted(provided):
        allowed = _BUILD_STRATEGY_OPTION_SUPPORT.get(dest)
        if allowed is None:
            continue
        if strategy not in allowed:
            unsupported.append(_BUILD_OPTION_PRIMARY_FLAG[dest])
    if unsupported:
        unsupported_text = ", ".join(unsupported)
        build_parser.error(
            f"Unsupported option(s) for --strategy {strategy}: {unsupported_text}. "
            "Use --help to view strategy-scoped option applicability."
        )

    if strategy in {"embedding", "hybrid"}:
        if args.streaming and ":" in str(args.dataset_split):
            build_parser.error(
                "Streaming mode does not support sliced --dataset-split values "
                "(for example train[:5%]). Use unsliced split (e.g. train) or "
                "disable --streaming."
            )
        if args.all_corpus and "corpus_size" in provided:
            build_parser.error(
                "--all-corpus cannot be combined with explicit --corpus-size."
            )
        try:
            validate_compression_filter(str(args.cache_compression))
        except ValueError as exc:
            build_parser.error(str(exc))
        if str(args.storage_precision) != "int8":
            if "binary_prefilter" in provided and bool(args.binary_prefilter):
                build_parser.error(
                    "--binary-prefilter requires --storage-precision int8."
                )
            if "binary_rescore_multiplier" in provided:
                build_parser.error(
                    "--binary-rescore-multiplier requires --storage-precision int8."
                )
            # Normalize implicit non-int8 defaults to effective values to avoid
            # strategy-level runtime warnings about ignored options.
            args.binary_prefilter = False
            args.binary_rescore_multiplier = 1

    if strategy == "hybrid":
        if args.max_semantic is not None and int(args.max_semantic) >= int(
            args.max_papers
        ):
            build_parser.error(
                "--max-semantic must be between 0 and --max-papers - 1 for hybrid."
            )
        if int(args.max_semantic or 0) == 0:
            ignored_embedding_options = sorted(
                _BUILD_OPTION_PRIMARY_FLAG[dest]
                for dest in provided
                if dest in _HYBRID_EMBEDDING_OPTION_DESTS
            )
            if ignored_embedding_options:
                option_text = ", ".join(ignored_embedding_options)
                build_parser.error(
                    "Hybrid semantic branch is disabled with --max-semantic 0; "
                    f"remove embedding-only option(s): {option_text}."
                )


def _log_build_side_effect_contract(args: argparse.Namespace) -> None:
    """Log build side-effect contract summary for transparency before execution.

    :param argparse.Namespace args: Parsed build arguments.
    :return None: Emits info-level contract summary logs.
    """
    if args.strategy not in {"embedding", "hybrid"}:
        return

    corpus_label = "all" if args.all_corpus else str(args.corpus_size)
    cache_root = get_cache_dir("embeddings")
    logger.info(
        "Embedding workflow contract: may download model/dataset artifacts and mutate "
        "cache namespace at %s.",
        cache_root,
    )
    logger.info(
        "Embedding run config: model=%s revision=%s split=%s corpus=%s streaming=%s "
        "precision=%s.",
        args.model,
        args.model_revision or "default",
        args.dataset_split,
        corpus_label,
        bool(args.streaming),
        args.storage_precision,
    )
    if args.force_rebuild_cache:
        logger.warning(
            "--force-rebuild-cache enabled; existing embedding namespace payload will be cleared."
        )


_STRATEGY_DISPATCH: Dict[str, _StrategyDispatchSpec] = {
    "citation": _StrategyDispatchSpec(
        factory=lambda cli_args: CitationGraphBuilder(
            max_papers=cli_args.max_papers,
            max_citations=cli_args.max_citations,
            max_references=cli_args.max_references,
            similarity_threshold=cli_args.similarity_threshold,
            fetch_references=not cli_args.no_references,
            refresh_reference_cache=cli_args.refresh_reference_cache,
        ),
    ),
    "recommendation": _StrategyDispatchSpec(
        factory=lambda cli_args: RecommendationGraphBuilder(
            max_papers=cli_args.max_papers,
            fetch_references=not cli_args.no_references,
            refresh_reference_cache=cli_args.refresh_reference_cache,
            similarity_threshold=cli_args.similarity_threshold,
        ),
    ),
    "embedding": _StrategyDispatchSpec(
        factory=lambda cli_args: EmbeddingGraphBuilder(
            max_papers=cli_args.max_papers,
            top_k=cli_args.top_k,
            **_shared_embedding_builder_kwargs(cli_args),
        ),
    ),
    "hybrid": _StrategyDispatchSpec(
        factory=lambda cli_args: HybridGraphBuilder(
            max_papers=cli_args.max_papers,
            max_citations=cli_args.max_citations,
            max_references=cli_args.max_references,
            fetch_references=not cli_args.no_references,
            refresh_reference_cache=cli_args.refresh_reference_cache,
            max_semantic=cli_args.max_semantic,
            **_shared_embedding_builder_kwargs(cli_args),
        ),
    ),
}


def _build_strategy_graph(
    args: argparse.Namespace, strategy: str, *, validate_contract: bool = True
) -> tuple[nx.Graph, str]:
    """Build a graph for a strategy selected from CLI arguments.

    :param argparse.Namespace args: Parsed arguments.
    :param str strategy: Strategy name.
    :param bool validate_contract: Whether to run strategy-option contract checks.
    :return tuple[nx.Graph, str]: Graph and normalized seed paper ID.
    :raises ValueError: If strategy is unsupported.
    """
    if strategy not in _STRATEGY_DISPATCH:
        raise ValueError(f"Unsupported strategy: {strategy}")

    args_for_validation = argparse.Namespace(**vars(args))
    setattr(args_for_validation, "strategy", strategy)

    if validate_contract:
        parser_snapshot, build_parser_snapshot, _ = _create_parser()
        del parser_snapshot
        defaults_namespace = build_parser_snapshot.parse_args(["seed"])
        merged_values = vars(defaults_namespace)
        merged_values.update(vars(args_for_validation))
        args_for_validation = argparse.Namespace(**merged_values)
        setattr(args_for_validation, "strategy", strategy)
        inferred_provided = _infer_provided_build_option_dests(
            args=args_for_validation,
            build_parser=build_parser_snapshot,
        )

        class _ProgrammaticBuildParser:
            """Tiny parser shim translating parser.error into ValueError."""

            @staticmethod
            def error(message: str) -> None:
                """Raise parser-style validation messages as ``ValueError``.

                :param str message: Validation error message.
                :raises ValueError: Always raised with ``message``.
                """
                raise ValueError(message)

        _validate_build_cli_contract(
            args_for_validation,
            _ProgrammaticBuildParser(),  # type: ignore[arg-type]
            inferred_provided,
        )
        # Preserve validation-time no-op normalization for downstream builder parity.
        for field in ("binary_prefilter", "binary_rescore_multiplier"):
            if hasattr(args_for_validation, field):
                setattr(args, field, getattr(args_for_validation, field))

    builder = _STRATEGY_DISPATCH[strategy].factory(args)
    return builder.build_graph(args.paper_id)


def _infer_provided_build_option_dests(
    args: argparse.Namespace, build_parser: argparse.ArgumentParser
) -> Set[str]:
    """Infer likely explicit build options from a parsed namespace.

    This heuristic compares namespace values against parser defaults. It cannot detect
    explicit re-specification of the same default, but it prevents most silent
    programmatic bypasses for strategy-scoped option contracts.

    :param argparse.Namespace args: Candidate parsed namespace.
    :param argparse.ArgumentParser build_parser: Build subcommand parser.
    :return Set[str]: Option destinations inferred as explicitly set.
    """
    provided: Set[str] = set()
    for dest in _BUILD_STRATEGY_OPTION_SUPPORT:
        if not hasattr(args, dest):
            continue
        current_value = getattr(args, dest)
        default_value = build_parser.get_default(dest)
        if current_value != default_value:
            provided.add(dest)
    return provided


def _create_parser() -> Tuple[
    argparse.ArgumentParser, argparse.ArgumentParser, argparse.ArgumentParser
]:
    """Create and return the root parser and key subcommand parsers.

    :return Tuple[argparse.ArgumentParser, argparse.ArgumentParser, argparse.ArgumentParser]:
        Root parser, build subcommand parser, cache subcommand parser.
    """
    parser = argparse.ArgumentParser(
        description="CiteMesh: Create citation graph visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Citation-based graph (fast, uses S2 API)
  citemesh build "arxiv:1706.03762" --strategy citation

  # Recommendation graph (semantic-aware by default)
  citemesh build "arxiv:1706.03762"

  # Embedding-based graph (semantic similarity)
  citemesh build "arxiv:1706.03762" --strategy embedding

  # Hybrid approach (combines both)
  citemesh build "arxiv:1706.03762" --strategy hybrid

  # Custom output path
  citemesh build "10.1038/nature14539" -o my_graph.png

  # Quick test with fewer papers
  citemesh build "arxiv:1810.04805" -p 20 --strategy citation
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Build command
    build_parser = subparsers.add_parser(
        "build", help="Build and visualize paper graph"
    )

    # Required arguments
    build_parser.add_argument(
        "paper_id",
        type=_non_empty_str,
        help="Paper identifier (DOI, arXiv ID, or S2 ID)",
    )

    # Strategy selection
    build_parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        choices=["recommendation", "citation", "embedding", "hybrid"],
        default="recommendation",
        help="Graph building strategy (default: recommendation)",
    )

    # Common arguments
    build_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Output file path for single export, or output directory base for "
            "multi-export runs (auto-named if not specified)"
        ),
    )

    build_parser.add_argument(
        "--export",
        "-e",
        choices=["png", "html", "plotly", "json", "graphml", "all"],
        default="png",
        help="Export format (default: png)",
    )

    build_parser.add_argument(
        "--theme",
        choices=["light", "dark", "solarized", "auto"],
        default="light",
        help="Visualization theme to use",
    )

    build_parser.add_argument(
        "--max-papers",
        "-p",
        type=_positive_int,
        default=40,
        help=("Maximum papers in final graph (seed included; default: 40)"),
    )

    build_parser.add_argument(
        "--spring-iterations",
        "-i",
        type=_positive_int,
        default=100,
        help="Spring fallback layout iterations (default: 100)",
    )

    build_parser.add_argument(
        "--dpi",
        "-d",
        type=_positive_int,
        default=150,
        help="Output image resolution (default: 150)",
    )

    build_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed for deterministic layout generation in layout-based exports "
            "(default: deterministic built-in seed)"
        ),
    )
    build_parser.add_argument(
        "--include-timestamp",
        action="store_true",
        help="Include generation timestamp in output metadata annotations",
    )

    # Citation strategy arguments
    citation_group = build_parser.add_argument_group("citation strategy options")
    citation_group.add_argument(
        "--max-citations",
        "-c",
        type=_non_negative_int,
        default=20,
        help="Maximum citing papers to fetch (default: 20)",
    )

    citation_group.add_argument(
        "--max-references",
        "-r",
        type=_non_negative_int,
        default=20,
        help="Maximum referenced papers to fetch (default: 20)",
    )

    citation_group.add_argument(
        "--similarity-threshold",
        "-t",
        type=_threshold_float,
        default=0.2,
        help="Minimum edge similarity for citation/recommendation strategies (default: 0.2)",
    )

    citation_group.add_argument(
        "--no-references",
        action="store_true",
        help="Disable fetching reference lists (faster but no real bibliographic coupling)",
    )
    citation_group.add_argument(
        "--refresh-reference-cache",
        action="store_true",
        help=(
            "Bypass persisted reference-cache reads and force fresh API fetches "
            "for recommendation/citation lookups."
        ),
    )

    # Embedding strategy arguments
    embedding_group = build_parser.add_argument_group("embedding strategy options")
    embedding_group.add_argument(
        "--model",
        "-m",
        type=_non_empty_str,
        default="google/embeddinggemma-300m",
        help="Sentence transformer model name",
    )
    embedding_group.add_argument(
        "--model-revision",
        type=str,
        default=None,
        help=(
            "Optional model revision token (branch/tag/commit) for hub-backed "
            "embedding models."
        ),
    )

    embedding_group.add_argument(
        "--dataset-split",
        type=_non_empty_str,
        default="train",
        help="ArXiv dataset split (default: train = full snapshot split; combine with --corpus-size to cap runtime)",
    )

    embedding_group.add_argument(
        "--corpus-size",
        type=_positive_int,
        default=50000,
        help=(
            "Maximum papers to load from corpus "
            "(default: 50000; use --all-corpus to remove cap)"
        ),
    )

    embedding_group.add_argument(
        "--all-corpus",
        action="store_true",
        help="Disable corpus cap and process the full selected split",
    )

    embedding_group.add_argument(
        "--top-k",
        "-k",
        type=_positive_int,
        default=2,
        help="Top-k neighbors per node (default: 2)",
    )

    embedding_group.add_argument(
        "--truncate-dim",
        type=_positive_int,
        default=None,
        help=(
            "Optional embedding output dimension truncation "
            "(for EmbeddingGemma: 768, 512, 256, 128; default uses profile recommendation)"
        ),
    )

    embedding_group.add_argument(
        "--streaming",
        action="store_true",
        help="Stream HuggingFace dataset instead of loading it into memory (requires non-sliced --dataset-split)",
    )

    embedding_group.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Forcefully clear and rebuild embedding cache for this model before running.",
    )

    embedding_group.add_argument(
        "--storage-precision",
        choices=["int8", "float16", "float32"],
        default=EMBEDDING_STORAGE_CONFIG.storage_precision,
        help=("Persistent embedding cache precision (default: %(default)s)"),
    )

    binary_prefilter_group = embedding_group.add_mutually_exclusive_group()
    binary_prefilter_group.add_argument(
        "--binary-prefilter",
        dest="binary_prefilter",
        action="store_true",
        help="Enable binary Hamming prefilter + rescoring (recommended for large corpora).",
    )
    binary_prefilter_group.add_argument(
        "--no-binary-prefilter",
        dest="binary_prefilter",
        action="store_false",
        help="Disable binary prefilter and use direct cache scoring.",
    )
    build_parser.set_defaults(
        binary_prefilter=EMBEDDING_STORAGE_CONFIG.binary_prefilter
    )

    embedding_group.add_argument(
        "--binary-rescore-multiplier",
        type=_positive_int,
        default=EMBEDDING_STORAGE_CONFIG.binary_rescore_multiplier,
        help=(
            "Oversampling factor for binary prefilter rescoring (default: %(default)s)"
        ),
    )

    embedding_group.add_argument(
        "--calibration-sample-size",
        type=_positive_int,
        default=EMBEDDING_STORAGE_CONFIG.calibration_sample_size,
        help=(
            "Calibration sample size for int8 quantization ranges "
            "(default: %(default)s)"
        ),
    )

    embedding_group.add_argument(
        "--cache-compression",
        type=str,
        choices=list(_CACHE_COMPRESSION_CHOICES),
        default=EMBEDDING_STORAGE_CONFIG.compression,
        help=(
            "HDF5 compression filter for embedding cache datasets "
            "(default: %(default)s)"
        ),
    )

    embedding_group.add_argument(
        "--cache-compression-level",
        type=_non_negative_int,
        default=EMBEDDING_STORAGE_CONFIG.compression_level,
        help=(
            "HDF5 compression level for embedding cache datasets (default: %(default)s)"
        ),
    )

    torch_compile_group = embedding_group.add_mutually_exclusive_group()
    torch_compile_group.add_argument(
        "--torch-compile",
        dest="torch_compile",
        action="store_true",
        help=(
            "Enable best-effort torch.compile for supported embedding profiles "
            "(default: enabled)."
        ),
    )
    torch_compile_group.add_argument(
        "--no-torch-compile",
        dest="torch_compile",
        action="store_false",
        help="Disable torch.compile and keep eager runtime for embedding models.",
    )
    build_parser.set_defaults(torch_compile=True)

    # Hybrid strategy arguments
    hybrid_group = build_parser.add_argument_group("hybrid strategy options")
    hybrid_group.add_argument(
        "--max-semantic",
        type=_non_negative_int,
        default=None,
        help=(
            "Maximum non-seed semantic papers to add. Reserves citation capacity via "
            "max-papers - max-semantic and must be <= max-papers - 1 "
            "(default: min(10, max-papers - 1))"
        ),
    )

    # Search subcommand
    search_parser = subparsers.add_parser(
        "search", help="Search papers by title or keyword"
    )
    search_parser.add_argument("query", type=_non_empty_str, help="Search query")
    search_parser.add_argument(
        "--limit",
        "-n",
        type=_positive_int,
        default=10,
        help="Maximum results (default: 10)",
    )
    cache_parser = subparsers.add_parser("cache", help="Manage local CiteMesh caches")
    cache_subparsers = cache_parser.add_subparsers(
        dest="cache_command", help="Cache operations"
    )
    cache_clear_parser = cache_subparsers.add_parser(
        "clear", help="Delete the entire CiteMesh cache directory"
    )
    cache_clear_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt and clear cache immediately",
    )
    cache_subparsers.add_parser(
        "scan", help="Scan cache usage (sections, file counts, and total size)"
    )
    return parser, build_parser, cache_parser


def resolve_output_paths(
    base_output_path: Path,
    selected_formats: List[str],
    explicit_output: bool,
    strategy: str,
) -> Dict[str, Path]:
    """
    Resolve final output paths for selected export formats.

    :param Path base_output_path: Path provided by the user or auto-generated filename.
    :param List[str] selected_formats: Export formats selected for this run.
    :param bool explicit_output: True when the user provided ``--output``.
    :param str strategy: Active strategy name used for multi-format directory outputs.
    :return Dict[str, Path]: Mapping of export format -> resolved output path.
    """
    base_str = str(base_output_path)
    matched_suffix = next(
        (ext for ext in KNOWN_EXPORT_SUFFIXES if base_str.lower().endswith(ext)),
        None,
    )

    output_paths: Dict[str, Path] = {}
    if explicit_output and len(selected_formats) > 1:
        output_dir = (
            Path(base_str[: -len(matched_suffix)])
            if matched_suffix
            else base_output_path
        )
        basename = strategy or "graph"
        for fmt in selected_formats:
            output_paths[fmt] = output_dir / f"{basename}{EXPORT_EXTENSIONS[fmt]}"
        return output_paths

    if explicit_output and len(selected_formats) == 1:
        fmt = selected_formats[0]
        desired_ext = EXPORT_EXTENSIONS[fmt]

        if matched_suffix == desired_ext:
            output_paths[fmt] = base_output_path
            return output_paths

        if matched_suffix:
            output_paths[fmt] = Path(base_str[: -len(matched_suffix)] + desired_ext)
            return output_paths

        output_paths[fmt] = Path(base_str + desired_ext)
        return output_paths

    if matched_suffix:
        output_base = base_str[: -len(matched_suffix)]
    else:
        output_base = base_str

    for fmt in selected_formats:
        output_paths[fmt] = Path(output_base + EXPORT_EXTENSIONS[fmt])

    return output_paths


def canonicalize_paper_id_for_metadata(paper_id: str) -> str:
    """
    Best-effort canonical paper ID for output metadata display.

    :param str paper_id: Raw CLI paper identifier.
    :return str: Canonicalized identifier when possible; otherwise original input.
    """
    try:
        return normalize_paper_id(paper_id)
    except ValueError:
        return paper_id


def _confirmed_cache_clear(cache_root: Path, assume_yes: bool) -> bool:
    """Return whether cache directory deletion is confirmed.

    :param Path cache_root: Cache root directory targeted for deletion.
    :param bool assume_yes: Skip interactive prompt when ``True``.
    :return bool: ``True`` if cache deletion should proceed.
    """
    if assume_yes:
        return True

    if not sys.stdin.isatty():
        logger.error(
            "Refusing to clear cache in non-interactive mode without --yes. "
            "Re-run with: citemesh cache clear --yes"
        )
        return False

    prompt = f"Delete CiteMesh cache directory '{cache_root}'? [y/N]: "
    try:
        response = input(prompt).strip().lower()
    except EOFError:
        logger.error("No confirmation input received; cache clear aborted.")
        return False
    return response in {"y", "yes"}


def _clear_cache_directory(*, assume_yes: bool) -> int:
    """Clear the entire CiteMesh cache root.

    :param bool assume_yes: Whether to bypass interactive confirmation.
    :return int: Process exit code (``0`` success, ``1`` failure/cancelled).
    """
    raw_cache_root = get_cache_dir(create=False)
    cache_root = raw_cache_root.expanduser().resolve()

    if len(cache_root.parts) <= 1:
        logger.error("Refusing to clear unsafe cache path: %s", cache_root)
        return 1
    if cache_root == Path.home().expanduser().resolve():
        logger.error("Refusing to clear home directory path: %s", cache_root)
        return 1

    if not cache_root.exists():
        logger.info("Cache directory does not exist: %s", cache_root)
        return 0

    if not _confirmed_cache_clear(cache_root, assume_yes):
        logger.info("Cache clear aborted.")
        return 1

    try:
        shutil.rmtree(cache_root)
    except OSError as exc:
        logger.error("Failed to clear cache directory %s: %s", cache_root, exc)
        return 1

    logger.info("✓ Cleared cache directory: %s", cache_root)
    return 0


def _format_bytes(num_bytes: int) -> str:
    """Format byte counts into readable binary units.

    :param int num_bytes: Raw byte count.
    :return str: Human-readable size string.
    """
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(max(num_bytes, 0))
    unit = units[0]
    for candidate in units:
        unit = candidate
        if value < 1024.0 or candidate == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def _scan_path_stats(path: Path) -> tuple[int, int]:
    """Return file-count and total size stats for a path.

    :param Path path: Directory or file path to scan.
    :return tuple[int, int]: ``(file_count, size_bytes)`` totals.
    """
    if path.is_file():
        try:
            return 1, path.stat().st_size
        except OSError:
            return 1, 0

    file_count = 0
    size_bytes = 0
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        file_count += 1
        try:
            size_bytes += candidate.stat().st_size
        except OSError:
            continue
    return file_count, size_bytes


def _scan_cache_directory() -> int:
    """Scan the CiteMesh cache root and print a usage summary.

    :return int: Process exit code (``0`` success, ``1`` failure).
    """
    raw_cache_root = get_cache_dir(create=False)
    cache_root = raw_cache_root.expanduser().resolve()

    if not cache_root.exists():
        logger.info("Cache directory does not exist: %s", cache_root)
        return 0
    if not cache_root.is_dir():
        logger.error("Cache path exists but is not a directory: %s", cache_root)
        return 1

    section_rows: list[tuple[str, int, int]] = []
    for child in sorted(cache_root.iterdir(), key=lambda item: item.name):
        files, size_bytes = _scan_path_stats(child)
        section_rows.append((child.name, files, size_bytes))

    total_files = sum(row[1] for row in section_rows)
    total_bytes = sum(row[2] for row in section_rows)

    output_console.print(f"[bold]Cache root:[/bold] {cache_root}")
    table = Table(title="CiteMesh Cache Scan")
    table.add_column("Section")
    table.add_column("Files", justify="right")
    table.add_column("Size", justify="right")

    if section_rows:
        for name, files, size_bytes in section_rows:
            table.add_row(name, str(files), _format_bytes(size_bytes))
    else:
        table.add_row("(empty)", "0", "0 B")

    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_files}[/bold]",
        f"[bold]{_format_bytes(total_bytes)}[/bold]",
    )
    output_console.print(table)
    return 0


def main() -> None:
    """Main CLI entry point."""
    _configure_logging()
    parser, build_parser, cache_parser = _create_parser()

    args = parser.parse_args()
    provided_build_options = _collect_provided_build_option_dests(
        build_parser, sys.argv[1:]
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "build":
        _validate_build_cli_contract(args, build_parser, provided_build_options)
        try:
            _log_build_side_effect_contract(args)
            # Build graph based on strategy
            logger.info(f"Building graph using {args.strategy} strategy...")
            graph, seed_id = _build_strategy_graph(
                args, args.strategy, validate_contract=False
            )

            # Determine output paths
            if args.output:
                base_output_path = Path(args.output)
            else:
                base_output_path = generate_output_path(
                    graph, seed_id, strategy=args.strategy
                )

            selected_formats = (
                list(EXPORT_FORMATS) if args.export == "all" else [args.export]
            )
            output_paths = resolve_output_paths(
                base_output_path=base_output_path,
                selected_formats=selected_formats,
                explicit_output=bool(args.output),
                strategy=args.strategy,
            )
            for parent in {path.parent for path in output_paths.values()}:
                if parent and not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)

            # Visualize / export
            logger.info("Creating visualization...")
            metadata = {
                "paper_id": canonicalize_paper_id_for_metadata(args.paper_id),
                "seed_id": seed_id,
                "strategy": args.strategy,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "theme": args.theme,
                "score_contract": _strategy_score_contract(args.strategy),
            }
            if args.strategy in {"embedding", "hybrid"}:
                metadata["embedding"] = _embedding_export_metadata(args)
            if args.include_timestamp:
                metadata["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            layout_required = any(fmt in output_paths for fmt in ("png", "plotly"))
            shared_layout = (
                compute_layout(
                    graph,
                    iterations=args.spring_iterations,
                    layout_seed=args.seed,
                )
                if layout_required
                else None
            )

            exporter = GraphExporter(
                graph,
                seed_id,
                metadata=metadata,
                theme_name=args.theme,
                layout=shared_layout,
            )

            if "png" in output_paths:
                visualize_graph(
                    graph,
                    seed_id,
                    output_paths["png"],
                    iterations=args.spring_iterations,
                    dpi=args.dpi,
                    metadata=metadata,
                    theme_name=args.theme,
                    layout=shared_layout,
                )
                logger.info(f"✓ PNG saved to {output_paths['png']}")

            if "html" in output_paths:
                exporter.to_interactive_html(output_paths["html"], theme=args.theme)
                logger.info(f"✓ Interactive HTML saved to {output_paths['html']}")

            if "plotly" in output_paths:
                exporter.to_plotly_html(output_paths["plotly"], theme=args.theme)
                logger.info(f"✓ Plotly HTML saved to {output_paths['plotly']}")

            if "json" in output_paths:
                exporter.to_json(output_paths["json"])
                logger.info(f"✓ Graph JSON saved to {output_paths['json']}")

            if "graphml" in output_paths:
                exporter.to_graphml(output_paths["graphml"])
                logger.info(f"✓ GraphML saved to {output_paths['graphml']}")

            logger.info(
                f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}"
            )

        except Exception as e:
            logger.error(
                "Failed to build graph: %s",
                e,
                exc_info=logging.getLogger().level == logging.DEBUG,
            )
            sys.exit(1)
    elif args.command == "search":
        try:
            client = get_client()
            logger.info(f"Searching for: {args.query}")
            results = client.search_papers(args.query, limit=args.limit)

            if not results:
                logger.error("No results found.")
                sys.exit(1)

            table = Table(title=f"Search results for '{args.query}'")
            table.add_column("#", style="dim", width=3)
            # Keep full IDs copyable for direct use in `citemesh build`.
            table.add_column("ID", style="cyan", overflow="fold")
            table.add_column("Title", overflow="fold")
            table.add_column("Year", justify="right", width=6)
            table.add_column("Citations", justify="right", width=10)
            table.add_column("Authors", max_width=30)

            for i, paper in enumerate(results, 1):
                authors_str = ", ".join(a.name for a in paper.authors[:2])
                if len(paper.authors) > 2:
                    authors_str += " et al."

                table.add_row(
                    str(i),
                    paper.paper_id,
                    paper.title,
                    str(paper.year) if paper.year is not None else "",
                    f"{paper.citation_count:,}",
                    authors_str,
                )

            output_console.print(table)
            output_console.print("\n[dim]Full paper IDs:[/dim]")
            for i, paper in enumerate(results, 1):
                output_console.print(f"[dim]{i}.[/dim] {paper.paper_id}")
            output_console.print(
                "\n[dim]Use the paper ID with:[/dim] "
                'citemesh build "<ID>" --strategy recommendation'
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            sys.exit(1)
    elif args.command == "cache":
        if args.cache_command == "scan":
            exit_code = _scan_cache_directory()
            if exit_code != 0:
                sys.exit(exit_code)
        elif args.cache_command == "clear":
            exit_code = _clear_cache_directory(assume_yes=bool(args.yes))
            if exit_code != 0:
                sys.exit(exit_code)
        else:
            cache_parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
