#!/usr/bin/env python3
"""
CiteMesh: Unified CLI for CiteMesh visualizations.

This is the main entry point for the refactored CiteMesh package,
providing a single interface to all graph building strategies.
"""

import argparse
import json
import logging
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set, Tuple

import networkx as nx
from filelock import FileLock, Timeout
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from citemesh._runtime import stderr_isatty, stdin_isatty, stdout_isatty
from citemesh.core import EMBEDDING_STORAGE_CONFIG
from citemesh.core.user_config import (
    SEARCH_MODE_CHOICES,
    USER_CONFIG_FILENAME,
    ConfigFileError,
    ConfigKeyError,
    ConfigValueError,
    UserConfig,
    format_config_value,
    load_user_config,
    parse_config_key,
    set_config_value,
    unset_config_value,
    user_config_path,
)
from citemesh.data import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    format_bytes,
    get_cache_dir,
    validate_compression_filter,
)
from citemesh.data.cache import atomic_write_json, legacy_macos_cache_root
from citemesh.paper_ids import normalize_paper_id
from citemesh.services import SemanticScholarUnavailableError, get_client
from citemesh.strategies.candidates import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    SEMANTIC_SOURCE_CHOICES,
)
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import (
    EMBEDDING_DEVICE_CHOICES,
    ENCODE_BATCH_SIZE,
    EmbeddingGraphBuilder,
    resolve_embedding_device,
)
from citemesh.strategies.hybrid import (
    DEFAULT_MAX_SEMANTIC,
    HYBRID_DEFAULT_MAX_CITATIONS,
    HYBRID_DEFAULT_MAX_PAPERS,
    HYBRID_DEFAULT_MAX_REFERENCES,
    HybridGraphBuilder,
)
from citemesh.strategies.recommendation import RecommendationGraphBuilder
from citemesh.visualization import (
    GraphExporter,
    compute_layout,
    generate_output_path,
    visualize_graph,
)

DEFAULT_LOG_WIDTH = 0
REDIRECTED_LOG_WIDTH = 140
LARGE_CACHE_CLEAR_WARNING_BYTES = 1024 * 1024 * 1024
LOG_LEVEL_CHOICES = ("debug", "info", "warning", "error")
DASHBOARD_MANIFEST_LOCK_TIMEOUT_SECONDS = 60.0


def _resolve_console_width(log_width: int, *, interactive: bool) -> Optional[int]:
    """Resolve the configured Rich console width for a target stream.

    :param int log_width: Requested Rich console width in columns.
    :param bool interactive: Whether the target stream is attached to a TTY.
    :return Optional[int]: Explicit column width or ``None`` for auto sizing.
    """
    resolved_width = int(log_width)
    if resolved_width > 0:
        return resolved_width
    if interactive:
        return None
    return REDIRECTED_LOG_WIDTH


log_console = Console(
    stderr=True,
    width=_resolve_console_width(DEFAULT_LOG_WIDTH, interactive=stderr_isatty()),
)
output_console = Console(
    width=_resolve_console_width(DEFAULT_LOG_WIDTH, interactive=stdout_isatty())
)
_LOGGING_CONFIGURED = False
logger = logging.getLogger(__name__)
_TRACKED_OPTION_DESTS_ATTR = "_citemesh_provided_option_dests"
_TRACKED_ACTION_CACHE: Dict[type[argparse.Action], type[argparse.Action]] = {}


def _tracking_action_class(
    action_cls: type[argparse.Action],
) -> type[argparse.Action]:
    """Wrap an argparse action so explicit CLI usage records its destination.

    :param type[argparse.Action] action_cls: Action class to wrap.
    :return type[argparse.Action]: Wrapper class recording explicit option use.
    """
    cached = _TRACKED_ACTION_CACHE.get(action_cls)
    if cached is not None:
        return cached

    class _TrackedAction(action_cls):
        """Action wrapper that records explicit option usage on the namespace."""

        _citemesh_tracks_presence = True

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: object,
            option_string: str | None = None,
        ) -> None:
            if self.option_strings:
                provided = getattr(namespace, _TRACKED_OPTION_DESTS_ATTR, None)
                if not isinstance(provided, set):
                    provided = set()
                    setattr(namespace, _TRACKED_OPTION_DESTS_ATTR, provided)
                provided.add(self.dest)
            super().__call__(parser, namespace, values, option_string)

    _TrackedAction.__name__ = f"CiteMeshTracked{action_cls.__name__}"
    _TRACKED_ACTION_CACHE[action_cls] = _TrackedAction
    return _TrackedAction


def _instrument_parser_actions(parser: argparse.ArgumentParser) -> None:
    """Wrap parser actions so explicit CLI option usage is recorded at parse time.

    The instrumentation walks the parser tree after construction, including all
    subparsers, which keeps presence tracking correct for arguments added via
    argument groups and mutually exclusive groups.

    :param argparse.ArgumentParser parser: Root or subparser to instrument.
    :return None: Mutates parser action classes in place.
    """
    for action in parser._actions:
        if action.option_strings and not getattr(
            action.__class__, "_citemesh_tracks_presence", False
        ):
            tracked_cls = _tracking_action_class(action.__class__)
            try:
                action.__class__ = tracked_cls
            except TypeError:
                pass
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                _instrument_parser_actions(subparser)


def _pop_tracked_option_dests(args: argparse.Namespace) -> Set[str]:
    """Return and remove parser-tracked explicit option destinations.

    :param argparse.Namespace args: Parsed CLI namespace.
    :return Set[str]: Destinations explicitly supplied by the caller.
    """
    raw_provided = getattr(args, _TRACKED_OPTION_DESTS_ATTR, None)
    if hasattr(args, _TRACKED_OPTION_DESTS_ATTR):
        delattr(args, _TRACKED_OPTION_DESTS_ATTR)
    if not isinstance(raw_provided, set):
        return set()
    return {str(dest).strip() for dest in raw_provided if str(dest).strip()}


def _configure_logging(
    *,
    log_level: str = "info",
    log_width: int = DEFAULT_LOG_WIDTH,
    log_file: str | None = None,
) -> None:
    """Configure CLI logging once at runtime.

    :param str log_level: Log level token.
    :param int log_width: Rich console width; non-positive values use stream defaults.
    :param str | None log_file: Optional plain-text log file path.
    :return None: Mutates global logging handlers and consoles once.
    """
    global _LOGGING_CONFIGURED
    global log_console
    global output_console
    if _LOGGING_CONFIGURED:
        return

    level_name = str(log_level).strip().lower()
    if level_name not in LOG_LEVEL_CHOICES:
        level_name = "info"
    resolved_level = getattr(logging, level_name.upper(), logging.INFO)
    console_level = (
        max(resolved_level, logging.INFO) if log_file is not None else resolved_level
    )
    log_console = Console(
        stderr=True,
        width=_resolve_console_width(log_width, interactive=stderr_isatty()),
    )
    output_console = Console(
        width=_resolve_console_width(log_width, interactive=stdout_isatty())
    )
    console_handler = RichHandler(
        console=log_console,
        show_time=False,
        show_path=False,
        rich_tracebacks=False,
        markup=True,
    )
    console_handler.setLevel(console_level)
    handlers: list[logging.Handler] = [console_handler]
    if log_file is not None:
        resolved_log_file = Path(log_file).expanduser()
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            resolved_log_file,
            mode="w",
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=resolved_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    # Keep third-party HTTP logs concise without import-time side effects.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("semanticscholar").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    _LOGGING_CONFIGURED = True


def _bounded_int(value: str, *, minimum: int) -> int:
    """Parse an integer CLI argument constrained by a minimum value.

    :param str value: Raw argparse value.
    :param int minimum: Inclusive lower bound for parsed values.
    :return int: Parsed integer.
    :raises argparse.ArgumentTypeError: If parsing fails or value is below minimum.
    """
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("must be an integer")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < minimum:
        raise argparse.ArgumentTypeError(f"must be at least {minimum}")
    return parsed


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI argument.

    :param str value: Raw argparse value.
    :return int: Parsed integer constrained to be >= 1.
    """
    return _bounded_int(value, minimum=1)


def _non_negative_int(value: str) -> int:
    """Parse a non-negative integer CLI argument.

    :param str value: Raw argparse value.
    :return int: Parsed integer constrained to be >= 0.
    """
    return _bounded_int(value, minimum=0)


def _threshold_float(value: str) -> float:
    """Parse similarity-threshold CLI argument constrained to [0, 1].

    :param str value: Raw argparse value.
    :return float: Parsed threshold value.
    :raises argparse.ArgumentTypeError: If value is outside [0, 1].
    """
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("must be a float")
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


def _add_logging_arguments(
    target: argparse.ArgumentParser, *, suppress_defaults: bool = False
) -> None:
    """Add shared logging arguments to a parser.

    :param argparse.ArgumentParser target: Parser receiving logging options.
    :param bool suppress_defaults: Whether logging defaults should be suppressed.
    :return None: Mutates parser in-place.
    """
    default_log_level: object = "info"
    default_log_width: object = DEFAULT_LOG_WIDTH
    default_log_file: object = None
    if suppress_defaults:
        default_log_level = argparse.SUPPRESS
        default_log_width = argparse.SUPPRESS
        default_log_file = argparse.SUPPRESS

    target.add_argument(
        "--log-level",
        choices=list(LOG_LEVEL_CHOICES),
        default=default_log_level,
        help="Console log level (default: info)",
    )
    target.add_argument(
        "--log-width",
        type=_non_negative_int,
        default=default_log_width,
        help="Rich console wrap width in columns (0 = auto width; default: 0)",
    )
    target.add_argument(
        "--log-file",
        type=_non_empty_str,
        default=default_log_file,
        help="Optional plain-text log file path (overwrites existing file).",
    )


EXPORT_FORMATS = (
    "png",
    "html",
    "plotly",
    "dashboard",
    "json",
    "csv",
    "bibtex",
    "graphml",
)
EXPORT_EXTENSIONS: Dict[str, str] = {
    "png": ".png",
    "html": ".html",
    "plotly": ".plotly.html",
    "dashboard": ".dashboard.html",
    "json": ".json",
    "csv": ".csv",
    "bibtex": ".bib",
    "graphml": ".graphml",
}
KNOWN_EXPORT_SUFFIXES: List[str] = sorted(
    EXPORT_EXTENSIONS.values(), key=len, reverse=True
)
# Table-driven export dispatch: format → GraphExporter method name.
# ``png`` is handled separately (uses ``visualize_graph``, not the exporter).
_EXPORTER_METHOD: Dict[str, str] = {
    "html": "to_interactive_html",
    "plotly": "to_plotly_html",
    "dashboard": "to_dashboard_html",
    "json": "to_json",
    "csv": "to_csv",
    "bibtex": "to_bibtex",
    "graphml": "to_graphml",
}
_THEME_AWARE_FORMATS: frozenset = frozenset({"html", "plotly", "dashboard"})
DASHBOARD_COLLECTION_FILENAME = "dashboard.html"
DASHBOARD_MANIFEST_FILENAME = "dashboard.manifest.json"
# Verify dispatch coverage at import time — a new EXPORT_FORMATS entry without
# a dispatch mapping will fail fast here rather than silently skip at runtime.
assert set(_EXPORTER_METHOD) | {"png"} == set(EXPORT_FORMATS), (
    f"Export dispatch gap: covered={sorted(set(_EXPORTER_METHOD) | {'png'})}, "
    f"declared={sorted(EXPORT_FORMATS)}"
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
    "overwrite_cache": {"embedding", "hybrid"},
    "cache_overwrite_reason": {"embedding", "hybrid"},
    "storage_precision": {"embedding", "hybrid"},
    "binary_prefilter": {"embedding", "hybrid"},
    "binary_rescore_multiplier": {"embedding", "hybrid"},
    "calibration_sample_size": {"embedding", "hybrid"},
    "cache_compression": {"embedding", "hybrid"},
    "cache_compression_level": {"embedding", "hybrid"},
    "encode_batch_size": {"embedding", "hybrid"},
    "torch_compile": {"embedding", "hybrid"},
    "device": {"embedding", "hybrid"},
    "semantic_source": {"embedding", "hybrid"},
    "candidate_pool_size": {"embedding", "hybrid"},
    "max_semantic": {"hybrid"},
}
# Flags that only affect arxiv-corpus hydration; providing them implies (or
# requires) --semantic-source arxiv-corpus.
_CORPUS_ONLY_OPTION_DESTS: Set[str] = {
    "dataset_split",
    "corpus_size",
    "all_corpus",
    "streaming",
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
    "overwrite_cache": ["--overwrite-cache"],
    "cache_overwrite_reason": ["--cache-overwrite-reason"],
    "storage_precision": ["--storage-precision"],
    "binary_prefilter": ["--binary-prefilter", "--no-binary-prefilter"],
    "binary_rescore_multiplier": ["--binary-rescore-multiplier"],
    "calibration_sample_size": ["--calibration-sample-size"],
    "cache_compression": ["--cache-compression"],
    "cache_compression_level": ["--cache-compression-level"],
    "encode_batch_size": ["--encode-batch-size"],
    "torch_compile": ["--torch-compile", "--no-torch-compile"],
    "device": ["--device"],
    "semantic_source": ["--semantic-source"],
    "candidate_pool_size": ["--candidate-pool-size"],
    "max_semantic": ["--max-semantic"],
}
_BUILD_OPTION_PRIMARY_FLAG: Dict[str, str] = {
    dest: flags[0] for dest, flags in _BUILD_OPTION_FLAGS.items()
}
_CACHE_COMPRESSION_CHOICES = ("gzip", "lzf")
_HYBRID_BEST_PRACTICE_DEFAULTS: Dict[str, int] = {
    "max_papers": HYBRID_DEFAULT_MAX_PAPERS,
    "max_citations": HYBRID_DEFAULT_MAX_CITATIONS,
    "max_references": HYBRID_DEFAULT_MAX_REFERENCES,
}
_PROGRAMMATIC_BUILD_VALUE_DESTS: Set[str] = set(_BUILD_STRATEGY_OPTION_SUPPORT) | {
    "paper_id",
    "max_papers",
}
_HYBRID_EMBEDDING_OPTION_DESTS: Set[str] = {
    "model",
    "model_revision",
    "dataset_split",
    "corpus_size",
    "all_corpus",
    "truncate_dim",
    "streaming",
    "force_rebuild_cache",
    "overwrite_cache",
    "cache_overwrite_reason",
    "storage_precision",
    "binary_prefilter",
    "binary_rescore_multiplier",
    "calibration_sample_size",
    "cache_compression",
    "cache_compression_level",
    "encode_batch_size",
    "torch_compile",
    "device",
    "semantic_source",
    "candidate_pool_size",
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
        "force_rebuild_reason": getattr(cli_args, "cache_overwrite_reason", None),
        "storage_precision": cli_args.storage_precision,
        "binary_prefilter": cli_args.binary_prefilter,
        "binary_rescore_multiplier": cli_args.binary_rescore_multiplier,
        "calibration_sample_size": cli_args.calibration_sample_size,
        "cache_compression": cli_args.cache_compression,
        "cache_compression_level": cli_args.cache_compression_level,
        "encode_batch_size": cli_args.encode_batch_size,
        "enable_torch_compile": cli_args.torch_compile,
        "device": cli_args.device,
        "semantic_source": cli_args.semantic_source,
        "candidate_pool_size": cli_args.candidate_pool_size,
    }


def _normalized_cache_reason(raw_reason: Optional[str]) -> Optional[str]:
    """Normalize optional cache-clear rationale into a compact single-line token.

    :param Optional[str] raw_reason: Raw user-provided rationale text.
    :return Optional[str]: Normalized reason, or ``None`` when absent.
    """
    if raw_reason is None:
        return None
    normalized = " ".join(str(raw_reason).split())
    return normalized or None


def _embedding_export_metadata(
    cli_args: argparse.Namespace, runtime_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, object]:
    """Build embedding provenance payload persisted in export metadata.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :param Optional[Dict[str, Any]] runtime_metadata: Optional runtime retrieval metadata.
    :return Dict[str, object]: Embedding cache/vector provenance + runtime fields.
    """
    int8_mode = str(cli_args.storage_precision) == "int8"
    binary_prefilter_enabled = bool(cli_args.binary_prefilter and int8_mode)
    binary_prefilter_used_for_query: Optional[bool]
    binary_prefilter_used_for_query = None
    if int8_mode and isinstance(runtime_metadata, dict):
        raw_used = runtime_metadata.get("binary_prefilter_used")
        if isinstance(raw_used, bool):
            binary_prefilter_used_for_query = raw_used
    elif not int8_mode:
        binary_prefilter_used_for_query = False

    effective_device: Optional[str] = None
    effective_compute_dtype: Optional[str] = None
    if isinstance(runtime_metadata, dict):
        raw_device = runtime_metadata.get("device")
        raw_compute_dtype = runtime_metadata.get("compute_dtype")
        if isinstance(raw_device, str) and raw_device:
            effective_device = raw_device
        if isinstance(raw_compute_dtype, str) and raw_compute_dtype:
            effective_compute_dtype = raw_compute_dtype

    return {
        "effective_vector_dtype": "float32",
        "effective_device": effective_device,
        "effective_compute_dtype": effective_compute_dtype,
        "semantic_source": str(cli_args.semantic_source),
        "candidate_pool_size": int(cli_args.candidate_pool_size),
        "storage_precision": str(cli_args.storage_precision),
        "binary_prefilter_enabled": binary_prefilter_enabled,
        "binary_prefilter_used_for_query": binary_prefilter_used_for_query,
        "binary_rescore_multiplier": (
            int(cli_args.binary_rescore_multiplier) if int8_mode else 1
        ),
        "cache_overwrite_reason": _normalized_cache_reason(
            getattr(cli_args, "cache_overwrite_reason", None)
        ),
    }


def _plot_overlay_metadata(export_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact metadata suitable for static image overlays.

    :param Dict[str, Any] export_metadata: Full export metadata payload.
    :return Dict[str, Any]: Reduced metadata subset for on-plot annotation.
    """
    overlay_keys = ("paper_id", "strategy", "nodes", "edges", "theme", "timestamp")
    return {
        key: export_metadata[key]
        for key in overlay_keys
        if key in export_metadata and export_metadata[key] is not None
    }


def _resolved_hybrid_max_semantic(cli_args: argparse.Namespace) -> int:
    """Resolve effective hybrid semantic cap from CLI arguments.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :return int: Effective ``max_semantic`` value.
    """
    if cli_args.max_semantic is None:
        return max(0, min(int(DEFAULT_MAX_SEMANTIC), int(cli_args.max_papers) - 1))
    return int(cli_args.max_semantic)


def _apply_hybrid_default_overrides(
    args: argparse.Namespace, provided: Set[str]
) -> None:
    """Apply tuned hybrid defaults when budget knobs are omitted.

    :param argparse.Namespace args: Parsed build arguments.
    :param Set[str] provided: Explicit option destinations found in argv.
    :return None: Mutates ``args`` in place for omitted hybrid budget fields.
    """
    if str(args.strategy) != "hybrid":
        return

    for dest, value in _HYBRID_BEST_PRACTICE_DEFAULTS.items():
        if dest in provided:
            continue
        setattr(args, dest, int(value))


def _apply_user_config_defaults(
    args: argparse.Namespace, provided: Set[str], user_config: UserConfig
) -> Set[str]:
    """Overlay config.toml defaults onto build args the user did not set.

    Precedence: explicit CLI flag > config.toml > built-in default. Values are
    already whitelist-validated at config load time.

    :param argparse.Namespace args: Parsed build arguments.
    :param Set[str] provided: Explicit option destinations found in argv.
    :param UserConfig user_config: Loaded user configuration snapshot.
    :return Set[str]: Destinations that were filled from config.toml.
    """
    applied: Set[str] = set()
    for dest in sorted(user_config.defaults):
        if dest in provided or not hasattr(args, dest):
            continue
        value = user_config.defaults[dest]
        setattr(args, dest, list(value) if isinstance(value, list) else value)
        applied.add(dest)
    if applied:
        summary = ", ".join(
            f"{dest}={format_config_value(user_config.defaults[dest])}"
            for dest in sorted(applied)
        )
        logger.info("Applying config defaults from %s: %s", user_config.path, summary)
    return applied


def _apply_user_config_api_key(user_config: UserConfig) -> None:
    """Export the configured S2 API key unless the environment already set one.

    Environment presence wins even for an empty value, so ``S2_API_KEY=""``
    still explicitly disables the configured key.

    :param UserConfig user_config: Loaded user configuration snapshot.
    :return None: May set ``S2_API_KEY`` in the process environment.
    """
    if not user_config.s2_api_key or "S2_API_KEY" in os.environ:
        return
    os.environ["S2_API_KEY"] = user_config.s2_api_key
    logger.debug("Using api.s2_api_key from %s.", user_config.path)


def _hybrid_semantic_branch_enabled(cli_args: argparse.Namespace) -> bool:
    """Return whether hybrid semantic branch is effectively enabled.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :return bool: ``True`` when hybrid semantic branch can run.
    """
    return _resolved_hybrid_max_semantic(cli_args) > 0


def _embedding_branch_enabled(cli_args: argparse.Namespace) -> bool:
    """Return whether the active build will execute an embedding-backed workflow.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :return bool: ``True`` when embedding runtime, cache, and corpus work may run.
    """
    strategy = str(getattr(cli_args, "strategy", "")).strip().lower()
    if strategy == "embedding":
        return True
    if strategy == "hybrid":
        return _hybrid_semantic_branch_enabled(cli_args)
    return False


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
                "seed-relevance-ranked union across citation+semantic candidates "
                "with overlap priority; purely semantic additions are capped by "
                "max_semantic."
            ),
        }
    return {
        **base_contract,
        "score_type": "unknown",
    }


def _validate_build_cli_contract(
    args: argparse.Namespace,
    build_parser: argparse.ArgumentParser,
    provided: Set[str],
    config_defaults: Set[str] = frozenset(),
) -> None:
    """Validate strategy-scoped and dependent build options before execution.

    :param argparse.Namespace args: Parsed build arguments.
    :param argparse.ArgumentParser build_parser: Build subcommand parser.
    :param Set[str] provided: Explicit option destinations found in argv.
    :param Set[str] config_defaults: Destinations filled from config.toml; they
        outrank built-in defaults (hybrid implicit budgets) but never count as
        explicit flags for strategy gating or corpus-mode implication.
    :return None: Mutates normalized args for effective no-op elimination.
    """
    strategy = str(args.strategy)
    _apply_hybrid_default_overrides(args, provided | set(config_defaults))
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
        provided_corpus_flags = sorted(
            _BUILD_OPTION_PRIMARY_FLAG[dest]
            for dest in provided
            if dest in _CORPUS_ONLY_OPTION_DESTS
        )
        if provided_corpus_flags and "semantic_source" not in provided:
            args.semantic_source = "arxiv-corpus"
            logger.info(
                "Corpus option(s) %s imply --semantic-source arxiv-corpus.",
                ", ".join(provided_corpus_flags),
            )
        if args.semantic_source != "arxiv-corpus":
            if provided_corpus_flags:
                option_text = ", ".join(provided_corpus_flags)
                build_parser.error(
                    f"Corpus-only option(s) require --semantic-source arxiv-corpus: "
                    f"{option_text}."
                )
            if "storage_precision" in provided and args.storage_precision == "int8":
                build_parser.error(
                    "--storage-precision int8 requires --semantic-source "
                    "arxiv-corpus (int8 calibration ranges are computed during "
                    "corpus hydration)."
                )
            if args.storage_precision == "int8":
                # Normalize the implicit int8 default to candidate-mode storage.
                args.storage_precision = "float32"
                logger.info(
                    "Candidate mode stores embeddings as float32 "
                    "(int8 calibration requires corpus hydration)."
                )
        elif "candidate_pool_size" in provided:
            build_parser.error(
                "--candidate-pool-size requires --semantic-source candidates."
            )
        if args.streaming and ":" in str(args.dataset_split):
            build_parser.error(
                "Streaming mode does not support sliced --dataset-split values "
                "(for example train[:5%]). Use unsliced split (e.g. train) or "
                "disable --streaming."
            )
        if bool(args.overwrite_cache) and not bool(args.force_rebuild_cache):
            build_parser.error("--overwrite-cache requires --force-rebuild-cache.")
        if _normalized_cache_reason(
            getattr(args, "cache_overwrite_reason", None)
        ) and not bool(args.force_rebuild_cache):
            build_parser.error(
                "--cache-overwrite-reason requires --force-rebuild-cache."
            )
        if args.all_corpus and "corpus_size" in provided:
            build_parser.error(
                "--all-corpus cannot be combined with explicit --corpus-size."
            )
        if str(args.device) != "auto":
            try:
                resolve_embedding_device(args.device)
            except ValueError as exc:
                build_parser.error(str(exc))
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
            if "cache_compression_level" in provided:
                build_parser.error(
                    "--cache-compression-level is unsupported with "
                    "--cache-compression lzf."
                )
            resolved_compression_level = 0
        args.cache_compression_level = int(resolved_compression_level)
        if str(args.storage_precision) != "int8":
            if "binary_prefilter" in provided and bool(args.binary_prefilter):
                build_parser.error(
                    "--binary-prefilter requires --storage-precision int8."
                )
            if "binary_rescore_multiplier" in provided:
                build_parser.error(
                    "--binary-rescore-multiplier requires --storage-precision int8."
                )
            if "calibration_sample_size" in provided:
                build_parser.error(
                    "--calibration-sample-size requires --storage-precision int8."
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
        resolved_max_semantic = _resolved_hybrid_max_semantic(args)

        if resolved_max_semantic == 0:
            ignored_embedding_options = sorted(
                _BUILD_OPTION_PRIMARY_FLAG[dest]
                for dest in provided
                if dest in _HYBRID_EMBEDDING_OPTION_DESTS
            )
            if ignored_embedding_options:
                option_text = ", ".join(ignored_embedding_options)
                if args.max_semantic is None:
                    build_parser.error(
                        "Hybrid semantic branch is disabled (effective --max-semantic "
                        "is 0 from --max-papers defaulting); remove embedding-only "
                        f"option(s): {option_text}."
                    )
                else:
                    build_parser.error(
                        "Hybrid semantic branch is disabled with --max-semantic 0; "
                        f"remove embedding-only option(s): {option_text}."
                    )


def _log_build_side_effect_contract(args: argparse.Namespace) -> None:
    """Log build side-effect contract summary for transparency before execution.

    :param argparse.Namespace args: Parsed build arguments.
    :return None: Emits info-level contract summary logs.
    """
    if not _embedding_branch_enabled(args):
        return

    corpus_mode = str(args.semantic_source) == "arxiv-corpus"
    _, cache_files, _ = _embedding_cache_directory_stats()
    if cache_files == 0:
        if corpus_mode:
            logger.warning(
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
        corpus_label = "all" if args.all_corpus else str(args.corpus_size)
        logger.info(
            "Embedding config: model=%s@%s device=%s source=arxiv-corpus split=%s corpus=%s streaming=%s storage=%s encode_batch=%s.",
            args.model,
            revision_label,
            args.device,
            args.dataset_split,
            corpus_label,
            bool(args.streaming),
            args.storage_precision,
            int(args.encode_batch_size),
        )
    else:
        logger.info(
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
    parser_error_sink: argparse.ArgumentParser,
) -> object:
    """Normalize a programmatic build value using the parser action contract.

    :param argparse.Action action: Parser action defining the value contract.
    :param object value: Programmatic value to validate.
    :param argparse.ArgumentParser parser_error_sink: Parser-like error sink.
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
    parser_error_sink: argparse.ArgumentParser,
) -> None:
    """Validate programmatic build namespaces against CLI scalar contracts.

    :param argparse.Namespace args: Candidate build namespace.
    :param argparse.ArgumentParser build_parser: Build parser used for action metadata.
    :param argparse.ArgumentParser parser_error_sink: Parser-like error sink.
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
    args: argparse.Namespace,
    strategy: str,
    *,
    validate_contract: bool = True,
    provided: Optional[Set[str]] = None,
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

        class _ProgrammaticBuildParser:
            """Tiny parser shim translating parser.error into ValueError."""

            @staticmethod
            def error(message: str) -> None:
                """Raise parser-style validation messages as ``ValueError``.

                :param str message: Validation error message.
                :raises ValueError: Always raised with ``message``.
                """
                raise ValueError(message)

        _validate_programmatic_build_values(
            args_for_validation,
            build_parser_snapshot,
            _ProgrammaticBuildParser(),  # type: ignore[arg-type]
        )
        _validate_build_cli_contract(
            args_for_validation,
            _ProgrammaticBuildParser(),  # type: ignore[arg-type]
            inferred_provided,
        )
        _synchronize_namespace_values(args, args_for_validation)

    builder = _STRATEGY_DISPATCH[strategy].factory(args)
    return builder.build_graph(args.paper_id)


def _infer_provided_build_option_dests(
    args: argparse.Namespace, build_parser: argparse.ArgumentParser
) -> Set[str]:
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
    provided: Set[str] = set()
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


def _create_parser() -> Tuple[
    argparse.ArgumentParser,
    argparse.ArgumentParser,
    argparse.ArgumentParser,
    argparse.ArgumentParser,
]:
    """Create and return the root parser and key subcommand parsers.

    :return Tuple[argparse.ArgumentParser, argparse.ArgumentParser, argparse.ArgumentParser, argparse.ArgumentParser]:
        Root parser, build subcommand parser, cache subcommand parser,
        config subcommand parser.
    """
    root_logging_parent = argparse.ArgumentParser(add_help=False)
    _add_logging_arguments(root_logging_parent)
    command_logging_parent = argparse.ArgumentParser(add_help=False)
    _add_logging_arguments(command_logging_parent, suppress_defaults=True)

    parser = argparse.ArgumentParser(
        description="CiteMesh: Create citation graph visualizations",
        parents=[root_logging_parent],
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

Environment variables:
  S2_API_KEY                                      Semantic Scholar API key (higher rate limits)
  CITEMESH_CACHE_DIR                              Override cache directory location
  CITEMESH_EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS   Embedding cache lock timeout (default: 900s)

User configuration:
  Persistent defaults live in <cache_root>/config.toml (see `citemesh config --help`).
  Precedence: explicit CLI flag > environment variable > config.toml > built-in default.

  # Always default to corpus-backed semantic sourcing
  citemesh config set defaults.semantic_source arxiv-corpus
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Commands",
    )

    # Build command
    build_parser = subparsers.add_parser(
        "build",
        help="Build and visualize paper graph",
        parents=[command_logging_parent],
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
            "Output file path for single export, or output/collection root for "
            "multi-export runs. With dashboard export, an explicit "
            "*.dashboard.html path keeps standalone mode; otherwise CiteMesh "
            "writes dashboard.html + dashboard.manifest.json under the output root."
        ),
    )

    build_parser.add_argument(
        "--export",
        "-e",
        choices=[*EXPORT_FORMATS, "all"],
        action="append",
        default=None,
        help=(
            "Export format; repeat for multiple (default: png). Dashboard export "
            "normally uses collection mode (shared dashboard.html + manifest + "
            "per-run JSON/config artifacts). Use -o <name>.dashboard.html for a "
            "standalone one-file dashboard."
        ),
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
        help=(
            "Maximum papers in final graph (seed included; default: 40; "
            f"hybrid implicit default: {HYBRID_DEFAULT_MAX_PAPERS})"
        ),
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
        default=25,
        help=(
            "Maximum citing papers to fetch (default: 25; "
            f"hybrid implicit default: {HYBRID_DEFAULT_MAX_CITATIONS})"
        ),
    )

    citation_group.add_argument(
        "--max-references",
        "-r",
        type=_non_negative_int,
        default=25,
        help=(
            "Maximum referenced papers to fetch (default: 25; "
            f"hybrid implicit default: {HYBRID_DEFAULT_MAX_REFERENCES})"
        ),
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
        default=DEFAULT_EMBEDDING_MODEL_NAME,
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
        default=4,
        help="Top-k neighbors per node (default: 4)",
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
        "--overwrite-cache",
        action="store_true",
        help=(
            "Acknowledge destructive cache overwrite for --force-rebuild-cache and "
            "skip interactive confirmation."
        ),
    )
    embedding_group.add_argument(
        "--cache-overwrite-reason",
        type=str,
        default=None,
        help=(
            "Optional rationale string logged when --force-rebuild-cache clears "
            "embedding cache state."
        ),
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
            "HDF5 compression level for embedding cache datasets (default: %(default)s; "
            "only applies to --cache-compression gzip)"
        ),
    )

    embedding_group.add_argument(
        "--encode-batch-size",
        type=_positive_int,
        default=ENCODE_BATCH_SIZE,
        help=(
            "Batch size for embedding model encode passes during hydration/search "
            "(default: %(default)s)"
        ),
    )

    torch_compile_group = embedding_group.add_mutually_exclusive_group()
    torch_compile_group.add_argument(
        "--torch-compile",
        dest="torch_compile",
        action="store_true",
        help=(
            "Enable best-effort torch.compile for supported embedding profiles "
            "(default: disabled)."
        ),
    )
    torch_compile_group.add_argument(
        "--no-torch-compile",
        dest="torch_compile",
        action="store_false",
        help="Disable torch.compile and keep eager runtime for embedding models.",
    )
    build_parser.set_defaults(torch_compile=False)

    embedding_group.add_argument(
        "--semantic-source",
        dest="semantic_source",
        choices=list(SEMANTIC_SOURCE_CHOICES),
        default="candidates",
        help=(
            "Semantic candidate sourcing: 'candidates' embeds only S2 seed "
            "neighbors (references/citations/recommendations; fast, no local "
            "corpus); 'arxiv-corpus' hydrates a local arXiv corpus. Corpus-only "
            "flags imply arxiv-corpus for backwards compatibility "
            "(default: %(default)s)."
        ),
    )
    embedding_group.add_argument(
        "--candidate-pool-size",
        dest="candidate_pool_size",
        type=_positive_int,
        default=DEFAULT_CANDIDATE_POOL_SIZE,
        help=(
            "Maximum S2 candidate pool size fetched in candidates mode "
            "(default: %(default)s)."
        ),
    )
    embedding_group.add_argument(
        "--device",
        dest="device",
        choices=list(EMBEDDING_DEVICE_CHOICES),
        default="auto",
        help=(
            "Compute device for embedding model runs. 'auto' prefers CUDA, then "
            "MPS (Apple Silicon), then CPU; explicit unavailable devices fail "
            "fast (default: %(default)s)."
        ),
    )

    # Hybrid strategy arguments
    hybrid_group = build_parser.add_argument_group("hybrid strategy options")
    hybrid_group.add_argument(
        "--max-semantic",
        type=_non_negative_int,
        default=None,
        help=(
            "Maximum non-seed semantic papers to add (must be <= max-papers - 1). "
            "When omitted, hybrid uses implicit citation-depth reservation before "
            "semantic expansion (default cap: "
            f"min({DEFAULT_MAX_SEMANTIC}, max-papers - 1))."
        ),
    )

    # Search subcommand
    search_parser = subparsers.add_parser(
        "search",
        help="Search papers (local semantic index or the Semantic Scholar API)",
        description=(
            "Find papers to pass to `citemesh build`. Mode `local` runs "
            "semantic search over the embeddings already persisted in your "
            "local cache (candidate vectors accumulated across builds, or a "
            "hydrated corpus) with no Semantic Scholar traffic. Mode `s2` "
            "runs keyword search on the Semantic Scholar API (shares an "
            "anonymous rate-limit pool unless S2_API_KEY is set). The default "
            "mode `auto` searches locally when your cache has embeddings and "
            "falls back to `s2` otherwise, logging which one ran. Persist a "
            "preference with `citemesh config set defaults.search_mode "
            "<mode>`."
        ),
        parents=[command_logging_parent],
    )
    search_parser.add_argument("query", type=_non_empty_str, help="Search query")
    search_parser.add_argument(
        "--limit",
        "-n",
        type=_positive_int,
        default=10,
        help="Maximum results (default: 10)",
    )
    search_parser.add_argument(
        "--mode",
        choices=list(SEARCH_MODE_CHOICES),
        default=None,
        help=(
            "Search mode (default: config.toml defaults.search_mode, else "
            "auto: local when cached embeddings exist, s2 otherwise)"
        ),
    )
    search_parser.add_argument(
        "--model",
        "-m",
        default=None,
        help=(
            "Embedding model for local search (implies --mode local); must "
            "match the model used at build time (default: config.toml "
            "default or built-in default)"
        ),
    )
    search_parser.add_argument(
        "--device",
        choices=list(EMBEDDING_DEVICE_CHOICES),
        default=None,
        help="Compute device for local query encoding (implies --mode local)",
    )
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage local CiteMesh caches",
        parents=[command_logging_parent],
    )
    cache_subparsers = cache_parser.add_subparsers(
        dest="cache_command",
        help="Cache operations",
    )
    cache_clear_parser = cache_subparsers.add_parser(
        "clear",
        help="Delete cached data under the CiteMesh cache root (config.toml is preserved)",
        parents=[command_logging_parent],
    )
    cache_clear_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt and clear cache immediately",
    )
    cache_clear_parser.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Optional rationale string logged when cache clear is executed.",
    )
    cache_subparsers.add_parser(
        "scan",
        help="Scan cache usage (sections, file counts, and total size)",
        parents=[command_logging_parent],
    )

    # Config subcommand
    config_parser = subparsers.add_parser(
        "config",
        help="Manage persistent user configuration (config.toml)",
        parents=[command_logging_parent],
        description=(
            "Manage persistent CiteMesh defaults stored in config.toml under "
            "the cache root. Precedence: explicit CLI flag > environment "
            "variable > config.toml > built-in default."
        ),
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
        help="Config operations",
    )
    config_subparsers.add_parser(
        "list",
        help="Show configured values and the config file path",
        parents=[command_logging_parent],
    )
    config_get_parser = config_subparsers.add_parser(
        "get",
        help="Print one configured value",
        parents=[command_logging_parent],
    )
    config_get_parser.add_argument(
        "key",
        type=_non_empty_str,
        help="Dotted config key (for example defaults.semantic_source)",
    )
    config_set_parser = config_subparsers.add_parser(
        "set",
        help="Set and persist one config value",
        parents=[command_logging_parent],
    )
    config_set_parser.add_argument(
        "key",
        type=_non_empty_str,
        help="Dotted config key (for example defaults.semantic_source)",
    )
    config_set_parser.add_argument(
        "value",
        type=_non_empty_str,
        help="Value to persist (booleans: true/false; lists: comma-separated)",
    )
    config_unset_parser = config_subparsers.add_parser(
        "unset",
        help="Remove one configured value",
        parents=[command_logging_parent],
    )
    config_unset_parser.add_argument(
        "key",
        type=_non_empty_str,
        help="Dotted config key (for example defaults.semantic_source)",
    )
    config_subparsers.add_parser(
        "path",
        help="Print the config file path",
        parents=[command_logging_parent],
    )
    _instrument_parser_actions(parser)
    return parser, build_parser, cache_parser, config_parser


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
    stripped_base = _strip_known_export_suffix(base_str)
    has_known_suffix = stripped_base != base_str

    output_paths: Dict[str, Path] = {}
    if explicit_output and len(selected_formats) > 1:
        if "dashboard" in selected_formats and base_str.lower().endswith(
            EXPORT_EXTENSIONS["dashboard"]
        ):
            for fmt in selected_formats:
                if fmt == "dashboard":
                    output_paths[fmt] = base_output_path
                else:
                    output_paths[fmt] = Path(stripped_base + EXPORT_EXTENSIONS[fmt])
            return output_paths

        output_dir = Path(stripped_base) if has_known_suffix else base_output_path
        basename = strategy or "graph"
        for fmt in selected_formats:
            output_paths[fmt] = output_dir / f"{basename}{EXPORT_EXTENSIONS[fmt]}"
        return output_paths

    if explicit_output and len(selected_formats) == 1:
        fmt = selected_formats[0]
        desired_ext = EXPORT_EXTENSIONS[fmt]

        if base_str.lower().endswith(desired_ext):
            output_paths[fmt] = base_output_path
            return output_paths

        if has_known_suffix:
            output_paths[fmt] = Path(stripped_base + desired_ext)
            return output_paths

        output_paths[fmt] = Path(base_str + desired_ext)
        return output_paths

    for fmt in selected_formats:
        output_paths[fmt] = Path(stripped_base + EXPORT_EXTENSIONS[fmt])

    return output_paths


def _is_standalone_dashboard_output(
    base_output_path: Path,
    selected_formats: List[str],
    explicit_output: bool,
) -> bool:
    """Return whether dashboard export should remain a standalone HTML artifact.

    :param Path base_output_path: User-provided or generated base output path.
    :param List[str] selected_formats: Requested export formats.
    :param bool explicit_output: Whether ``--output`` was provided.
    :return bool: ``True`` when an explicit standalone dashboard path was
        requested, even if additional sibling exports were also selected.
    """
    return (
        explicit_output
        and "dashboard" in selected_formats
        and str(base_output_path).lower().endswith(EXPORT_EXTENSIONS["dashboard"])
    )


def _resolve_dashboard_collection_root(
    base_output_path: Path,
    *,
    explicit_output: bool,
) -> Path:
    """Resolve root directory for shared dashboard collection artifacts.

    :param Path base_output_path: User-provided or generated base output path.
    :param bool explicit_output: Whether ``--output`` was provided.
    :return Path: Collection root directory containing shared dashboard shell.
    """
    if explicit_output:
        base_str = str(base_output_path)
        stripped_base = _strip_known_export_suffix(base_str)
        if stripped_base != base_str:
            return Path(stripped_base)
        return base_output_path

    parent = base_output_path.parent
    grandparent = parent.parent
    if str(grandparent) and grandparent != Path("."):
        return grandparent
    return parent


def resolve_dashboard_collection_outputs(
    *,
    base_output_path: Path,
    selected_formats: List[str],
    explicit_output: bool,
    strategy: str,
    graph: nx.Graph,
    seed_id: str,
) -> tuple[Dict[str, Path], Path]:
    """Resolve shared-dashboard output paths plus per-run result artifact paths.

    The dashboard shell lives at the collection root, while each build run writes
    its JSON/config and any other requested artifacts into a unique seed-specific
    result directory beneath that root.

    :param Path base_output_path: User-provided or generated base output path.
    :param List[str] selected_formats: Requested export formats.
    :param bool explicit_output: Whether ``--output`` was provided.
    :param str strategy: Active strategy name.
    :param nx.Graph graph: Built graph used for run-specific output naming.
    :param str seed_id: Seed node identifier.
    :return tuple[Dict[str, Path], Path]: Resolved output paths and manifest path.
    """
    collection_root = _resolve_dashboard_collection_root(
        base_output_path,
        explicit_output=explicit_output,
    )
    run_base_output_path = generate_output_path(
        graph,
        seed_id,
        output_dir=collection_root,
        strategy=strategy,
    )
    run_formats = [fmt for fmt in selected_formats if fmt != "dashboard"]
    if "json" not in run_formats:
        run_formats.append("json")
    output_paths = resolve_output_paths(
        base_output_path=run_base_output_path,
        selected_formats=run_formats,
        explicit_output=False,
        strategy=strategy,
    )
    output_paths["dashboard"] = collection_root / DASHBOARD_COLLECTION_FILENAME
    return output_paths, collection_root / DASHBOARD_MANIFEST_FILENAME


def _strip_known_export_suffix(filename: str) -> str:
    """Strip a known export suffix from a filename-like token.

    :param str filename: Candidate filename token.
    :return str: Filename with trailing known export suffix removed.
    """
    lowered = filename.lower()
    for suffix in KNOWN_EXPORT_SUFFIXES:
        if lowered.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def resolve_graph_config_path(output_paths: Dict[str, Path], strategy: str) -> Path:
    """Resolve sidecar graph-config output path for a build run.

    :param Dict[str, Path] output_paths: Resolved export artifact paths.
    :param str strategy: Active strategy name.
    :return Path: Graph-config JSON output path.
    """
    if not output_paths:
        return Path(f"{strategy or 'graph'}.config.json")

    anchor_path = next(iter(output_paths.values()))
    stem = _strip_known_export_suffix(anchor_path.name)
    if not stem:
        stem = strategy or "graph"
    return anchor_path.parent / f"{stem}.config.json"


def _relative_output_path(path: Path, root: Path) -> str:
    """Resolve a stable manifest path relative to a collection root when possible.

    :param Path path: Absolute or relative artifact path.
    :param Path root: Collection root directory.
    :return str: Relative path when possible, else normalized string form.
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def update_dashboard_manifest(
    manifest_path: Path,
    *,
    collection_root: Path,
    graph: nx.Graph,
    seed_id: str,
    strategy: str,
    json_path: Path,
    config_path: Path,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Create or update shared dashboard manifest for collection-style outputs.

    Dashboard collections intentionally keep one manifest slot per
    ``(strategy, seed_id)`` pair. Re-running the same seed/strategy refreshes
    that slot because the per-run JSON/config artifact paths are also stable for
    a given seed title and identifier.

    :param Path manifest_path: Manifest JSON path to write.
    :param Path collection_root: Shared dashboard collection root directory.
    :param nx.Graph graph: Built graph used for seed metadata.
    :param str seed_id: Seed node identifier.
    :param str strategy: Active strategy name.
    :param Path json_path: Per-run JSON payload path.
    :param Path config_path: Per-run config sidecar path.
    :param Dict[str, Any] metadata: Export metadata payload for summary fields.
    :return Dict[str, Any]: Manifest payload written to disk.
    """
    seed_title = str(graph.nodes[seed_id].get("title", seed_id))
    result_id = f"{strategy}:{seed_id}"
    entry = {
        "result_id": result_id,
        "seed_id": seed_id,
        "title": seed_title,
        "strategy": strategy,
        "summary": {
            "nodes": int(metadata.get("nodes", 0) or 0),
            "edges": int(metadata.get("edges", 0) or 0),
        },
        "json_path": _relative_output_path(json_path, collection_root),
        "config_path": _relative_output_path(config_path, collection_root),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_name(f".{manifest_path.name}.lock")
    lock = FileLock(str(lock_path), timeout=DASHBOARD_MANIFEST_LOCK_TIMEOUT_SECONDS)
    try:
        with lock:
            results: list[Dict[str, Any]] = []
            if manifest_path.exists():
                try:
                    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
                if isinstance(existing, dict) and isinstance(
                    existing.get("results"), list
                ):
                    results = [
                        item for item in existing["results"] if isinstance(item, dict)
                    ]

            filtered = [item for item in results if item.get("result_id") != result_id]
            manifest_payload = {
                "schema_version": 1,
                "results": [entry, *filtered],
            }
            atomic_write_json(manifest_path, manifest_payload, indent=2)
            return manifest_payload
    except Timeout as exc:
        raise RuntimeError(
            "Timed out waiting for dashboard manifest lock "
            f"at {lock_path} after {DASHBOARD_MANIFEST_LOCK_TIMEOUT_SECONDS:.1f}s."
        ) from exc


def _resolve_collection_payload_path(
    collection_root: Path, json_rel_path: str
) -> Optional[Path]:
    """Resolve a manifest JSON payload path confined to the collection root.

    :param Path collection_root: Shared dashboard collection directory.
    :param str json_rel_path: Manifest-provided JSON payload path.
    :return Optional[Path]: Resolved payload path, or ``None`` when invalid.
    """
    candidate = Path(json_rel_path)
    if candidate.is_absolute():
        return None

    resolved_root = collection_root.resolve()
    resolved_candidate = (resolved_root / candidate).resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved_candidate


def build_dashboard_collection_bundle(
    manifest_path: Path,
    *,
    current_result_id: str,
) -> Dict[str, Any]:
    """Load manifest entries plus embedded JSON payloads for shared dashboard UX.

    :param Path manifest_path: Manifest JSON path for the collection.
    :param str current_result_id: Result identifier for the graph just built.
    :return Dict[str, Any]: Embedded dashboard collection bundle.
    """
    collection_root = manifest_path.parent
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest_payload = {}

    raw_results = (
        manifest_payload.get("results") if isinstance(manifest_payload, dict) else []
    )
    results = [entry for entry in raw_results if isinstance(entry, dict)]
    payloads: Dict[str, Any] = {}
    for entry in results:
        result_id = str(entry.get("result_id") or "").strip()
        json_rel_path = str(entry.get("json_path") or "").strip()
        if not result_id or not json_rel_path:
            continue
        payload_path = _resolve_collection_payload_path(collection_root, json_rel_path)
        if payload_path is None:
            continue
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        payloads[result_id] = payload

    return {
        "current_result_id": current_result_id,
        "results": results,
        "payloads": payloads,
    }


def _drop_none_values(value: Any) -> Any:
    """Recursively drop ``None`` entries from dictionaries/lists.

    :param Any value: Arbitrary JSON-serializable object.
    :return Any: Copy with ``None``-valued mapping entries removed.
    """
    if isinstance(value, dict):
        return {
            key: _drop_none_values(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value]
    return value


def _build_citation_config_payload(
    cli_args: argparse.Namespace, *, strategy: str
) -> Optional[Dict[str, Any]]:
    """Build strategy-scoped citation/reference settings for config sidecars.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :param str strategy: Active build strategy.
    :return Optional[Dict[str, Any]]: Citation config payload when applicable.
    """
    fetch_references = not bool(cli_args.no_references)
    refresh_reference_cache = bool(cli_args.refresh_reference_cache)

    if strategy == "recommendation":
        return {
            "fetch_references": fetch_references,
            "refresh_reference_cache": refresh_reference_cache,
        }
    if strategy in {"citation", "hybrid"}:
        return {
            "max_citations": int(cli_args.max_citations),
            "max_references": int(cli_args.max_references),
            "fetch_references": fetch_references,
            "refresh_reference_cache": refresh_reference_cache,
        }
    return None


def _build_graph_config_payload(
    cli_args: argparse.Namespace,
    seed_id: str,
    metadata: Dict[str, Any],
    selected_formats: List[str],
    output_paths: Dict[str, Path],
) -> Dict[str, Any]:
    """Build sidecar graph-config payload for reproducibility and auditability.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :param str seed_id: Resolved seed paper identifier.
    :param Dict[str, Any] metadata: Run metadata payload used by exporters.
    :param List[str] selected_formats: Formats requested for export.
    :param Dict[str, Path] output_paths: Resolved export artifact paths.
    :return Dict[str, Any]: JSON-safe run configuration payload.
    """
    strategy = str(cli_args.strategy)
    semantic_enabled = strategy == "embedding" or (
        strategy == "hybrid" and _hybrid_semantic_branch_enabled(cli_args)
    )
    embedding_config: Optional[Dict[str, Any]] = None
    if semantic_enabled:
        embedding_config = {
            "model": cli_args.model,
            "model_revision": cli_args.model_revision,
            "semantic_source": str(cli_args.semantic_source),
            "candidate_pool_size": int(cli_args.candidate_pool_size),
            "dataset_split": cli_args.dataset_split,
            "corpus_size": None if cli_args.all_corpus else int(cli_args.corpus_size),
            "all_corpus": bool(cli_args.all_corpus),
            "truncate_dim": cli_args.truncate_dim,
            "streaming": bool(cli_args.streaming),
            "storage_precision": cli_args.storage_precision,
            "binary_prefilter": bool(cli_args.binary_prefilter),
            "binary_rescore_multiplier": int(cli_args.binary_rescore_multiplier),
            "calibration_sample_size": int(cli_args.calibration_sample_size),
            "cache_compression": cli_args.cache_compression,
            "cache_compression_level": int(cli_args.cache_compression_level),
            "encode_batch_size": int(cli_args.encode_batch_size),
            "torch_compile": bool(cli_args.torch_compile),
            "device": str(cli_args.device),
            "force_rebuild_cache": bool(cli_args.force_rebuild_cache),
            "overwrite_cache": bool(cli_args.overwrite_cache),
            "cache_overwrite_reason": _normalized_cache_reason(
                getattr(cli_args, "cache_overwrite_reason", None)
            ),
        }

    payload = {
        "schema_version": 1,
        "build": {
            "paper_id_input": cli_args.paper_id,
            "paper_id_canonical": canonicalize_paper_id_for_metadata(cli_args.paper_id),
            "seed_id": seed_id,
            "strategy": strategy,
            "max_papers": int(cli_args.max_papers),
            "citation": _build_citation_config_payload(cli_args, strategy=strategy),
            "hybrid": (
                {"max_semantic": _resolved_hybrid_max_semantic(cli_args)}
                if strategy == "hybrid"
                else None
            ),
            "embedding": embedding_config,
            "layout": {
                "spring_iterations": int(cli_args.spring_iterations),
                "seed": cli_args.seed,
                "dpi": int(cli_args.dpi),
                "theme": cli_args.theme,
            },
            "timestamp_included": bool(cli_args.include_timestamp),
            "exports_requested": list(selected_formats),
        },
        "outputs": {fmt: str(path) for fmt, path in sorted(output_paths.items())},
        "metadata": metadata,
    }
    return _drop_none_values(payload)


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


def _embedding_cache_directory_stats() -> tuple[Path, int, int]:
    """Return embedding cache directory path + file/size totals.

    :return tuple[Path, int, int]: ``(path, files, size_bytes)``.
    """
    embedding_cache_dir = (
        get_cache_dir("embeddings", create=False).expanduser().resolve()
    )
    if not embedding_cache_dir.exists():
        return embedding_cache_dir, 0, 0
    files, size_bytes = _scan_path_stats(embedding_cache_dir)
    return embedding_cache_dir, files, size_bytes


def _confirm_force_rebuild_cache(args: argparse.Namespace) -> bool:
    """Confirm destructive embedding namespace rebuild requested by CLI flags.

    :param argparse.Namespace args: Parsed build CLI arguments.
    :return bool: ``True`` when build may proceed.
    """
    if (
        str(args.command) != "build"
        or str(args.strategy) not in {"embedding", "hybrid"}
        or not bool(args.force_rebuild_cache)
    ):
        return True

    embedding_cache_dir, total_files, total_bytes = _embedding_cache_directory_stats()
    total_size_label = format_bytes(total_bytes)
    large_cache = total_bytes >= LARGE_CACHE_CLEAR_WARNING_BYTES
    large_threshold_label = format_bytes(LARGE_CACHE_CLEAR_WARNING_BYTES)
    overwrite_reason = _normalized_cache_reason(
        getattr(args, "cache_overwrite_reason", None)
    )

    if bool(args.overwrite_cache):
        logger.warning(
            "--overwrite-cache acknowledged destructive rebuild "
            "(embedding cache dir=%s files=%d size=%s).",
            embedding_cache_dir,
            total_files,
            total_size_label,
        )
        if large_cache:
            logger.warning(
                "Large embedding cache footprint detected (%s >= %s).",
                total_size_label,
                large_threshold_label,
            )
        if overwrite_reason:
            logger.warning("Cache overwrite rationale: %s", overwrite_reason)
        return True

    if not stdin_isatty():
        logger.error(
            "Refusing --force-rebuild-cache in non-interactive mode without "
            "--overwrite-cache. Re-run with --overwrite-cache to proceed."
        )
        return False

    logger.warning(
        "--force-rebuild-cache will clear the active embedding cache namespace before this run."
    )
    logger.warning(
        "Embedding cache directory snapshot: root=%s files=%d size=%s.",
        embedding_cache_dir,
        total_files,
        total_size_label,
    )
    if large_cache:
        logger.warning(
            "Large cache warning: %s >= %s. Clearing may require long rehydration.",
            total_size_label,
            large_threshold_label,
        )
    if overwrite_reason:
        logger.warning("Cache overwrite rationale: %s", overwrite_reason)
    logger.warning(
        "Use --overwrite-cache to bypass this prompt in scripted/non-interactive workflows."
    )
    try:
        response = (
            input("Proceed with embedding cache overwrite? [y/N]: ").strip().lower()
        )
    except EOFError:
        logger.error("No confirmation input received; build aborted.")
        return False
    return response in {"y", "yes"}


def _confirmed_cache_clear(
    cache_root: Path, assume_yes: bool, clear_reason: Optional[str]
) -> bool:
    """Return whether cache directory deletion is confirmed.

    :param Path cache_root: Cache root directory targeted for deletion.
    :param bool assume_yes: Skip interactive prompt when ``True``.
    :param Optional[str] clear_reason: Optional operator rationale for cache clear.
    :return bool: ``True`` if cache deletion should proceed.
    """
    total_files, total_bytes = _scan_path_stats(cache_root)
    total_size_label = format_bytes(total_bytes)
    large_cache = total_bytes >= LARGE_CACHE_CLEAR_WARNING_BYTES
    large_threshold_label = format_bytes(LARGE_CACHE_CLEAR_WARNING_BYTES)
    normalized_reason = _normalized_cache_reason(clear_reason)

    if assume_yes:
        logger.warning(
            "--yes acknowledged destructive cache clear (root=%s files=%d size=%s).",
            cache_root,
            total_files,
            total_size_label,
        )
        if large_cache:
            logger.warning(
                "Large cache warning: %s >= %s.",
                total_size_label,
                large_threshold_label,
            )
        if normalized_reason:
            logger.warning("Cache clear rationale: %s", normalized_reason)
        return True

    if not stdin_isatty():
        logger.error(
            "Refusing to clear cache in non-interactive mode without --yes. "
            "Re-run with: citemesh cache clear --yes"
        )
        return False

    logger.warning(
        "Cache directory snapshot: root=%s files=%d size=%s.",
        cache_root,
        total_files,
        total_size_label,
    )
    if large_cache:
        logger.warning(
            "Large cache warning: %s >= %s. Deletion is immediate and irreversible.",
            total_size_label,
            large_threshold_label,
        )
    if normalized_reason:
        logger.warning("Cache clear rationale: %s", normalized_reason)
    prompt = f"Delete CiteMesh cache directory '{cache_root}'? [y/N]: "
    try:
        response = input(prompt).strip().lower()
    except EOFError:
        logger.error("No confirmation input received; cache clear aborted.")
        return False
    return response in {"y", "yes"}


def _clear_cache_directory(*, assume_yes: bool, clear_reason: Optional[str]) -> int:
    """Clear the entire CiteMesh cache root.

    :param bool assume_yes: Whether to bypass interactive confirmation.
    :param Optional[str] clear_reason: Optional operator rationale for cache clear.
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

    if not _confirmed_cache_clear(cache_root, assume_yes, clear_reason=clear_reason):
        logger.info("Cache clear aborted.")
        return 1

    # Clear cache payloads but never the persistent user config file.
    config_path = cache_root / USER_CONFIG_FILENAME
    preserved_config = config_path.is_file()
    try:
        for child in sorted(cache_root.iterdir()):
            if child == config_path:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        if not preserved_config:
            cache_root.rmdir()
    except OSError as exc:
        logger.error("Failed to clear cache directory %s: %s", cache_root, exc)
        return 1

    suffix_parts = []
    if preserved_config:
        suffix_parts.append(f"preserved {USER_CONFIG_FILENAME}")
    normalized_reason = _normalized_cache_reason(clear_reason)
    if normalized_reason:
        suffix_parts.append(f"reason={normalized_reason}")
    if suffix_parts:
        logger.info(
            "✓ Cleared cache directory: %s (%s)", cache_root, ", ".join(suffix_parts)
        )
    else:
        logger.info("✓ Cleared cache directory: %s", cache_root)
    return 0


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
        _log_legacy_macos_cache_hint(cache_root)
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
            table.add_row(name, str(files), format_bytes(size_bytes))
    else:
        table.add_row("(empty)", "0", "0 B")

    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_files}[/bold]",
        f"[bold]{format_bytes(total_bytes)}[/bold]",
    )
    output_console.print(table)
    _log_legacy_macos_cache_hint(cache_root)
    return 0


def _log_legacy_macos_cache_hint(cache_root: Path) -> None:
    """Point at the pre-unification macOS cache directory when it lingers.

    :param Path cache_root: Active resolved cache root.
    :return None: Emits an info-level migration hint when applicable.
    """
    legacy_root = legacy_macos_cache_root()
    if legacy_root is None:
        return
    try:
        if legacy_root.resolve() == cache_root:
            return
    except OSError:
        return
    logger.info(
        "Legacy macOS cache directory detected at %s. CiteMesh now uses %s; "
        "move or delete the old directory to reclaim space.",
        legacy_root,
        cache_root,
    )


def _masked_secret(value: str) -> str:
    """Mask a secret for display, keeping a short recognizable suffix.

    :param str value: Secret value to mask.
    :return str: Masked representation.
    """
    if len(value) <= 8:
        return "****"
    return f"****{value[-4:]}"


def _run_config_command(
    args: argparse.Namespace, config_parser: argparse.ArgumentParser
) -> int:
    """Execute the ``citemesh config`` subcommand.

    :param argparse.Namespace args: Parsed config arguments.
    :param argparse.ArgumentParser config_parser: Config subcommand parser.
    :return int: Process-style exit code.
    """
    command = getattr(args, "config_command", None)
    if not command:
        config_parser.print_help()
        return 1

    if command == "path":
        # Plain print keeps `citemesh config path`/`get` output unwrapped and
        # markup-free for shell substitution.
        print(user_config_path())
        return 0

    if command == "list":
        user_config = load_user_config()
        output_console.print(f"[bold]Config file:[/bold] {user_config.path}")
        if not user_config.path.is_file():
            output_console.print(
                "[dim]File does not exist yet; using built-in defaults. "
                "Create it with `citemesh config set <key> <value>`.[/dim]"
            )
        rows: List[Tuple[str, str]] = [
            (f"defaults.{key}", format_config_value(value))
            for key, value in sorted(user_config.defaults.items())
        ]
        if user_config.s2_api_key:
            rows.append(("api.s2_api_key", _masked_secret(user_config.s2_api_key)))
        table = Table(title="CiteMesh User Config")
        table.add_column("Key")
        table.add_column("Value")
        if rows:
            for key, value in rows:
                table.add_row(key, value)
        else:
            table.add_row("(no values set)", "")
        output_console.print(table)
        return 0

    if command == "get":
        try:
            table_name, key, _spec = parse_config_key(args.key)
        except ConfigKeyError as exc:
            config_parser.error(str(exc))
        user_config = load_user_config()
        if table_name == "api":
            value: Any = user_config.s2_api_key
        else:
            value = user_config.defaults.get(key)
        if value is None:
            logger.error("Config key '%s' is not set.", args.key)
            return 1
        print(format_config_value(value))
        return 0

    if command == "set":
        try:
            value = set_config_value(args.key, args.value)
        except (ConfigKeyError, ConfigValueError) as exc:
            config_parser.error(str(exc))
        except ConfigFileError as exc:
            logger.error("%s", exc)
            return 1
        display_value = (
            _masked_secret(str(value))
            if str(args.key).strip() == "api.s2_api_key"
            else format_config_value(value)
        )
        logger.info("✓ Set %s = %s in %s", args.key, display_value, user_config_path())
        return 0

    if command == "unset":
        try:
            removed = unset_config_value(args.key)
        except ConfigKeyError as exc:
            config_parser.error(str(exc))
        except ConfigFileError as exc:
            logger.error("%s", exc)
            return 1
        if removed:
            logger.info("✓ Unset %s in %s", args.key, user_config_path())
        else:
            logger.info("Config key '%s' was not set.", args.key)
        return 0

    config_parser.print_help()
    return 1


def _resolve_search_mode(
    args: argparse.Namespace, user_config: UserConfig
) -> Tuple[str, str]:
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
) -> Tuple[EmbeddingGraphBuilder, argparse.Namespace]:
    """Construct the embedding builder local search runs against.

    Mirrors a flagless build's defaults pipeline (config.toml defaults plus
    candidate-mode storage normalization) so the search targets the same cache
    namespace a default build writes to; ``--model``/``--device`` override.

    :param argparse.Namespace args: Parsed search command arguments.
    :param argparse.ArgumentParser build_parser: Build subparser used to
        derive build-equivalent defaults.
    :param UserConfig user_config: Loaded user configuration snapshot.
    :return Tuple[EmbeddingGraphBuilder, argparse.Namespace]: Builder and the
        effective build-equivalent defaults namespace.
    """
    defaults = build_parser.parse_args(["local-search-placeholder-seed"])
    _pop_tracked_option_dests(defaults)
    defaults.strategy = "embedding"
    config_default_dests = _apply_user_config_defaults(
        defaults, frozenset(), user_config
    )
    _validate_build_cli_contract(
        defaults, build_parser, frozenset(), config_defaults=config_default_dests
    )
    if args.model:
        defaults.model = args.model
    if args.device:
        defaults.device = args.device
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

    table = Table(title=f"Local semantic search for '{args.query}'")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", width=6)
    # Keep full IDs copyable for direct use in `citemesh build`.
    table.add_column("ID", style="cyan", overflow="fold")
    table.add_column("Title", overflow="fold")
    table.add_column("Year", justify="right", width=6)
    table.add_column("Authors", max_width=30)

    for i, result in enumerate(results, 1):
        metadata = result.metadata or {}
        authors = [str(name) for name in (metadata.get("authors") or [])]
        authors_str = ", ".join(authors[:2])
        if len(authors) > 2:
            authors_str += " et al."
        year_value = metadata.get("year")
        table.add_row(
            str(i),
            f"{float(result.score):.3f}",
            str(result.paper_id),
            str(metadata.get("title") or ""),
            str(year_value) if year_value is not None else "",
            authors_str,
        )

    output_console.print(table)
    total = getattr(cache, "last_search_total_embeddings", None)
    if total is not None:
        output_console.print(
            f"[dim]Searched {int(total):,} locally cached embeddings "
            f"(model={defaults.model}, source={defaults.semantic_source}).[/dim]"
        )
    output_console.print("\n[dim]Full paper IDs:[/dim]")
    for i, result in enumerate(results, 1):
        output_console.print(f"[dim]{i}.[/dim] {result.paper_id}")
    return 0


def _run_s2_search(args: argparse.Namespace) -> int:
    """Run keyword search on the Semantic Scholar API and render results.

    :param argparse.Namespace args: Parsed search command arguments.
    :return int: Process-style exit code.
    """
    try:
        client = get_client()
        logger.info(f"Searching for: {args.query}")
        results = client.search_papers(
            args.query, limit=args.limit, raise_on_unavailable=True
        )

        if not results:
            logger.error("No results found.")
            return 1

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
    if args.model or args.device:
        if args.mode == "s2":
            logger.error(
                "--model and --device only apply to local semantic search; "
                "drop them or use --mode local."
            )
            return 2
        if mode != "local":
            # Namespace-selecting flags are explicit local intent; they
            # outrank a config-level s2/auto default.
            mode, origin = "local", "flag"
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

    try:
        builder, defaults = _prepare_local_search_builder(
            args, build_parser, user_config
        )
        cached_count = builder.embedding_cache.embedding_count()
    except Exception as exc:
        if mode == "auto":
            logger.info(
                "Local semantic search unavailable (%s); searching the "
                "Semantic Scholar API instead.",
                exc,
            )
            return _run_s2_search(args)
        logger.error(
            "Local search unavailable: %s",
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
            "Local embedding cache is empty; searching the Semantic Scholar "
            "API instead. Local semantic search activates once your builds "
            "have embedded papers."
        )
        return _run_s2_search(args)

    # Explicit local mode: an empty cache is an error, not a fallback.
    if cached_count == 0:
        requested_via = (
            "--mode local"
            if origin == "flag"
            else f"defaults.search_mode in {user_config.path}"
        )
        logger.error(
            "Local search was requested via %s, but the local embedding cache "
            "has no vectors for model=%s semantic-source=%s (cache: %s). "
            "Local search covers papers your builds have already embedded - "
            "run `citemesh build` with the embedding or hybrid strategy to "
            "populate it, or use --mode s2 for keyword search.",
            requested_via,
            defaults.model,
            defaults.semantic_source,
            builder.embedding_cache.h5_path,
        )
        return 1
    return _render_local_search(args, builder, defaults)


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point.

    :param Sequence[str] | None argv: Optional CLI argument list without executable.
    :return int: Process-style exit code.
    """
    parser, build_parser, cache_parser, config_parser = _create_parser()
    argv_list = list(argv) if argv is not None else None
    args = parser.parse_args(argv_list)
    _configure_logging(
        log_level=args.log_level,
        log_width=args.log_width,
        log_file=getattr(args, "log_file", None),
    )
    provided_build_options = _pop_tracked_option_dests(args)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "config":
        return _run_config_command(args, config_parser)

    user_config = load_user_config()
    _apply_user_config_api_key(user_config)

    if args.command == "build":
        config_default_dests = _apply_user_config_defaults(
            args, provided_build_options, user_config
        )
        _validate_build_cli_contract(
            args,
            build_parser,
            provided_build_options,
            config_defaults=config_default_dests,
        )
        try:
            if not _confirm_force_rebuild_cache(args):
                logger.info("Build aborted.")
                return 1
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

            raw_exports = args.export or ["png"]
            if "all" in raw_exports:
                selected_formats = list(EXPORT_FORMATS)
            else:
                selected_formats = list(dict.fromkeys(raw_exports))
            dashboard_manifest_path: Optional[Path] = None
            if "dashboard" in selected_formats and not _is_standalone_dashboard_output(
                base_output_path=base_output_path,
                selected_formats=selected_formats,
                explicit_output=bool(args.output),
            ):
                output_paths, dashboard_manifest_path = (
                    resolve_dashboard_collection_outputs(
                        base_output_path=base_output_path,
                        selected_formats=selected_formats,
                        explicit_output=bool(args.output),
                        strategy=args.strategy,
                        graph=graph,
                        seed_id=seed_id,
                    )
                )
            else:
                output_paths = resolve_output_paths(
                    base_output_path=base_output_path,
                    selected_formats=selected_formats,
                    explicit_output=bool(args.output),
                    strategy=args.strategy,
                )
            if dashboard_manifest_path is not None:
                run_artifact_root = (
                    output_paths["json"].parent
                    if "json" in output_paths
                    else next(iter(output_paths.values())).parent
                )
                logger.info(
                    "Dashboard collection mode: shell=%s manifest=%s run_artifacts=%s.",
                    output_paths["dashboard"],
                    dashboard_manifest_path,
                    run_artifact_root,
                )
            for parent in {path.parent for path in output_paths.values()}:
                if parent and not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)

            # Visualize / export
            metadata = {
                "paper_id": canonicalize_paper_id_for_metadata(args.paper_id),
                "seed_id": seed_id,
                "strategy": args.strategy,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "theme": args.theme,
                "score_contract": _strategy_score_contract(args.strategy),
            }
            include_embedding_metadata = _embedding_branch_enabled(args)
            if include_embedding_metadata:
                runtime_embedding_metadata: Optional[Dict[str, Any]] = None
                raw_runtime_metadata = graph.graph.get("embedding_runtime")
                if isinstance(raw_runtime_metadata, dict):
                    runtime_embedding_metadata = raw_runtime_metadata
                metadata["embedding"] = _embedding_export_metadata(
                    args, runtime_embedding_metadata
                )
            if args.include_timestamp:
                metadata["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            plot_metadata = _plot_overlay_metadata(metadata)
            layout_required = any(
                fmt in output_paths for fmt in ("png", "plotly", "dashboard")
            )
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
                    metadata=plot_metadata,
                    theme_name=args.theme,
                    layout=shared_layout,
                )

            for fmt, method_name in _EXPORTER_METHOD.items():
                if fmt not in output_paths:
                    continue
                if fmt == "dashboard" and dashboard_manifest_path is not None:
                    # Shared dashboards are rendered after the manifest update so the
                    # shell can embed the current collection bundle for offline reuse.
                    continue
                method = getattr(exporter, method_name)
                if fmt in _THEME_AWARE_FORMATS:
                    method(output_paths[fmt], theme=args.theme)
                else:
                    method(output_paths[fmt])

            graph_config_path = resolve_graph_config_path(
                output_paths=output_paths,
                strategy=args.strategy,
            )
            graph_config_payload = _build_graph_config_payload(
                cli_args=args,
                seed_id=seed_id,
                metadata=metadata,
                selected_formats=selected_formats,
                output_paths=output_paths,
            )
            graph_config_path.write_text(
                json.dumps(graph_config_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if (
                dashboard_manifest_path is not None
                and "json" in output_paths
                and output_paths["json"].exists()
            ):
                manifest_payload = update_dashboard_manifest(
                    dashboard_manifest_path,
                    collection_root=dashboard_manifest_path.parent,
                    graph=graph,
                    seed_id=seed_id,
                    strategy=args.strategy,
                    json_path=output_paths["json"],
                    config_path=graph_config_path,
                    metadata=metadata,
                )
                metadata["dashboard_collection"] = build_dashboard_collection_bundle(
                    dashboard_manifest_path,
                    current_result_id=str(
                        manifest_payload["results"][0].get(
                            "result_id", f"{args.strategy}:{seed_id}"
                        )
                    ),
                )
                exporter.to_dashboard_html(output_paths["dashboard"], theme=args.theme)

            artifact_paths = dict(output_paths)
            artifact_paths["config"] = graph_config_path
            if dashboard_manifest_path is not None:
                artifact_paths["dashboard_manifest"] = dashboard_manifest_path
            saved_artifact_count = len(artifact_paths)
            output_dirs = sorted({str(path.parent) for path in artifact_paths.values()})
            if saved_artifact_count:
                if len(output_dirs) == 1:
                    logger.info(
                        "%d export artifacts saved to:\t%s",
                        saved_artifact_count,
                        output_dirs[0],
                    )
                else:
                    logger.info(
                        "%d export artifacts saved across %d directories: %s",
                        saved_artifact_count,
                        len(output_dirs),
                        ", ".join(output_dirs),
                    )

            logger.info(
                "Graph summary: nodes=%d, edges=%d",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )

        except Exception as e:
            logger.error(
                "Failed to build graph: %s",
                e,
                exc_info=logging.getLogger().level == logging.DEBUG,
            )
            return 1
    elif args.command == "search":
        return _run_search_command(args, build_parser, user_config)
    elif args.command == "cache":
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


if __name__ == "__main__":
    raise SystemExit(main())
