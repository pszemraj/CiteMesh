"""Argparse construction, styling, and shared argument validators.

Owns the Rich-styled parser and help formatter, explicit-option presence
tracking, the reusable argparse ``type`` validators, and :func:`_create_parser`,
which builds the full command tree.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import NoReturn

from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text
from rich_argparse import RichHelpFormatter

from citemesh import __version__
from citemesh.core import EMBEDDING_CONFIG, EMBEDDING_STORAGE_CONFIG
from citemesh.core.user_config import SEARCH_MODE_CHOICES
from citemesh.core.validation import (
    ValueValidationError,
    parse_bounded_int,
    parse_unit_interval_float,
)
from citemesh.data import DEFAULT_EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_PROFILE_CHOICES
from citemesh.strategies.candidates import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    SEMANTIC_SOURCE_CHOICES,
)
from citemesh.strategies.embedding import (
    DEFAULT_DATASET_SOURCE,
    EMBEDDING_DEVICE_CHOICES,
    ENCODE_BATCH_SIZE,
)
from citemesh.strategies.hybrid import (
    DEFAULT_MAX_SEMANTIC,
    HYBRID_DEFAULT_MAX_CITATIONS,
    HYBRID_DEFAULT_MAX_PAPERS,
    HYBRID_DEFAULT_MAX_REFERENCES,
)
from citemesh.visualization.dashboard.package import DASHBOARD_COLLECTION_FILENAME

from . import console
from .build_options import _CACHE_COMPRESSION_CHOICES
from .console import DEFAULT_LOG_WIDTH, LOG_LEVEL_CHOICES
from .outputs import EXPORT_FORMATS

_TRACKED_OPTION_DESTS_ATTR = "_citemesh_provided_option_dests"
_TRACKED_ACTION_CACHE: dict[type[argparse.Action], type[argparse.Action]] = {}


class _HelpFormatter(RichHelpFormatter):
    """Style argparse help without interpreting dataset slices as markup."""

    styles = {
        **RichHelpFormatter.styles,
        "argparse.groups": "bold",
        "argparse.args": "cyan",
        "argparse.metavar": "dim cyan",
        "argparse.prog": "bold cyan",
    }
    group_name_formatter = str
    help_markup = False
    text_markup = False

    def __init__(self, prog: str) -> None:
        """Keep option columns readable on both narrow and wide terminals.

        :param str prog: Command name supplied by argparse.
        :return None: Initialize the help renderer.
        """
        super().__init__(
            prog,
            max_help_position=32,
            width=max(40, min(110, shutil.get_terminal_size().columns - 2)),
        )


class _ArgumentParser(argparse.ArgumentParser):
    """Argparse parser that styles usage errors like the rest of the CLI.

    ``RichHelpFormatter`` only styles help and usage rendering; argparse still
    reports ``error()`` through a plain ``sys.stderr`` write. Subparsers default
    to their parent's class, so building the root parser from this class styles
    every command's usage errors without a per-parser opt-in.
    """

    def error(self, message: str) -> NoReturn:
        """Report a usage error through the Rich console, then exit.

        :param str message: Failure description supplied by argparse.
        :return NoReturn: Always exits with argparse's usage status code.
        """
        self.print_usage(sys.stderr)
        console.log_console.print(
            Text.assemble(
                (f"{self.prog}: ", "bold"),
                ("error: ", "bold red"),
                (message, "red"),
            ),
            soft_wrap=True,
        )
        self.exit(2)


def _help_examples(*examples: tuple[str, str]) -> Table:
    """Render labeled command examples that wrap with the help's terminal width.

    :param tuple[str, str] examples: Description and command pairs.
    :return Table: Borderless, single-column example block.
    """
    table = Table(
        box=None,
        show_header=False,
        padding=(0, 2),
        leading=1,
        title="Examples:",
        title_style="bold",
        title_justify="left",
    )
    for label, command in examples:
        table.add_row(Text.assemble((label + "\n", "dim"), (command + "\n", "cyan")))
    return table


def _output_table(title: str) -> Table:
    """Create the shared compact table style for human-readable CLI results.

    :param str title: Literal table title.
    :return Table: Table with light rules and a left-aligned title.
    """
    return Table(
        title=Text(title),
        title_style="bold",
        title_justify="left",
        box=box.SIMPLE_HEAD,
        border_style="dim",
        header_style="bold",
        padding=(0, 1),
        collapse_padding=True,
    )


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
            """Record options and retain parent options across subparser namespaces.

            :param argparse.ArgumentParser parser: Parser invoking this action.
            :param argparse.Namespace namespace: Destination namespace.
            :param object values: Parsed action values.
            :param str | None option_string: Explicit option spelling, if any.
            :return None: Apply the action and merge presence tracking.
            """
            parent_provided = set(getattr(namespace, _TRACKED_OPTION_DESTS_ATTR, set()))
            if self.option_strings:
                provided = getattr(namespace, _TRACKED_OPTION_DESTS_ATTR, None)
                if not isinstance(provided, set):
                    provided = set()
                    setattr(namespace, _TRACKED_OPTION_DESTS_ATTR, provided)
                provided.add(self.dest)
            super().__call__(parser, namespace, values, option_string)
            if isinstance(self, argparse._SubParsersAction):
                # argparse copies a fresh child namespace over its parent.
                child_provided = getattr(namespace, _TRACKED_OPTION_DESTS_ATTR, set())
                setattr(
                    namespace,
                    _TRACKED_OPTION_DESTS_ATTR,
                    parent_provided | child_provided,
                )

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
        if (
            action.option_strings or isinstance(action, argparse._SubParsersAction)
        ) and not getattr(action.__class__, "_citemesh_tracks_presence", False):
            tracked_cls = _tracking_action_class(action.__class__)
            try:
                action.__class__ = tracked_cls
            except TypeError:
                pass
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                _instrument_parser_actions(subparser)


def _pop_tracked_option_dests(args: argparse.Namespace) -> set[str]:
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


def _bounded_int(value: str, *, minimum: int) -> int:
    """Parse an integer CLI argument constrained by a minimum value.

    :param str value: Raw argparse value.
    :param int minimum: Inclusive lower bound for parsed values.
    :return int: Parsed integer.
    :raises argparse.ArgumentTypeError: If parsing fails or value is below minimum.
    """
    try:
        return parse_bounded_int(value, minimum=minimum)
    except ValueValidationError as exc:
        if exc.reason == "below_minimum":
            raise argparse.ArgumentTypeError(f"must be at least {minimum}") from exc
        raise argparse.ArgumentTypeError("must be an integer") from exc


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
    try:
        return parse_unit_interval_float(value)
    except ValueValidationError as exc:
        if exc.reason == "not_finite":
            raise argparse.ArgumentTypeError("must be a finite float") from exc
        if exc.reason == "out_of_range":
            raise argparse.ArgumentTypeError("must be between 0.0 and 1.0") from exc
        raise argparse.ArgumentTypeError("must be a float") from exc


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

    group = target.add_argument_group("Logging")
    group.add_argument(
        "--log-level",
        metavar="LEVEL",
        choices=list(LOG_LEVEL_CHOICES),
        default=default_log_level,
        help="Console verbosity: debug, info, warning, error (default: info)",
    )
    group.add_argument(
        "--log-width",
        metavar="COLS",
        type=_non_negative_int,
        default=default_log_width,
        help="Output/log width in columns; 0 uses terminal width (default: 0)",
    )
    group.add_argument(
        "--log-file",
        metavar="PATH",
        type=_non_empty_str,
        default=default_log_file,
        help="Write plain-text logs to PATH (overwrites an existing file)",
    )


def _create_root_parser() -> tuple[_ArgumentParser, argparse._SubParsersAction]:
    """Build the root ``citemesh`` parser and its command subparser action.

    :return tuple[_ArgumentParser, argparse._SubParsersAction]: Root parser and
        the subparser action every command attaches itself to.
    """
    parser = _ArgumentParser(
        prog="citemesh",
        description="CiteMesh — discover related research and build paper graphs.",
        epilog=Group(
            _help_examples(
                (
                    "Build a graph from a known paper",
                    'citemesh build "arxiv:1706.03762" --strategy hybrid',
                ),
                ("Find a seed paper", 'citemesh search "attention mechanisms"'),
                ("Open saved results", "citemesh view"),
                ("Inspect local storage", "citemesh cache scan"),
            ),
            Text(
                "\nUse citemesh COMMAND --help for command-specific options.",
                style="dim",
            ),
            Text("\nEnvironment", style="bold"),
            Text(
                "  S2_API_KEY          Semantic Scholar API key (higher rate limits)\n"
                "  CITEMESH_CACHE_DIR  Override the cache directory"
            ),
            Text("\nPersonal defaults: citemesh config --help", style="dim"),
        ),
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        metavar="COMMAND",
    )

    return parser, subparsers


def _add_build_graph_arguments(
    graph_group: argparse._ArgumentGroup, export_group: argparse._ArgumentGroup
) -> None:
    """Add strategy selection and the export/output options shared by every strategy.

    :param argparse._ArgumentGroup graph_group: "Graph" help section.
    :param argparse._ArgumentGroup export_group: "Output" help section.
    :return None: Mutates the groups in place.
    """
    # Strategy selection
    graph_group.add_argument(
        "--strategy",
        "-s",
        metavar="NAME",
        type=str,
        choices=["recommendation", "citation", "embedding", "hybrid"],
        default="recommendation",
        help="recommendation, citation, embedding, hybrid (default: recommendation)",
    )

    # Common arguments
    export_group.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "File or directory (default: out/ with automatic names). Dashboard collections share "
            "dashboard.html + dashboard.citemesh.json and save each seed's JSON "
            "under <title-slug>-<hash>/; use *.dashboard.html for a standalone file."
        ),
    )

    export_group.add_argument(
        "--export",
        "-e",
        metavar="FORMAT",
        choices=[*EXPORT_FORMATS, "all"],
        action="append",
        default=None,
        help=(
            "png, html, plotly, dashboard, json, csv, bibtex, graphml, all; "
            "repeat for multiple (default: png)."
        ),
    )

    export_group.add_argument(
        "--theme",
        metavar="THEME",
        choices=["light", "dark", "solarized", "auto"],
        default="dark",
        help="light, dark, solarized, auto (default: %(default)s)",
    )

    graph_group.add_argument(
        "--max-papers",
        "-p",
        type=_positive_int,
        metavar="N",
        default=40,
        help=(
            "Maximum papers including the seed (default: 40; "
            f"hybrid: {HYBRID_DEFAULT_MAX_PAPERS})"
        ),
    )

    export_group.add_argument(
        "--spring-iterations",
        "-i",
        type=_positive_int,
        metavar="N",
        default=100,
        help="Spring fallback layout iterations (default: 100)",
    )

    export_group.add_argument(
        "--dpi",
        "-d",
        type=_positive_int,
        metavar="N",
        default=150,
        help="Output image resolution (default: 150)",
    )

    export_group.add_argument(
        "--seed",
        type=int,
        metavar="N",
        default=None,
        help=(
            "Seed for deterministic layout generation in layout-based exports "
            "(default: deterministic built-in seed)"
        ),
    )
    export_group.add_argument(
        "--include-timestamp",
        action="store_true",
        help="Include generation timestamp in output metadata annotations",
    )


def _add_build_citation_arguments(
    citation_group: argparse._ArgumentGroup,
) -> None:
    """Add the citation/reference traversal options.

    :param argparse._ArgumentGroup citation_group: "Citations and references" section.
    :return None: Mutates the group in place.
    """
    # Citation strategy arguments
    citation_group.add_argument(
        "--max-citations",
        "-c",
        type=_non_negative_int,
        metavar="N",
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
        metavar="N",
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
        metavar="SCORE",
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


def _add_build_corpus_arguments(
    build_parser: argparse.ArgumentParser,
    embedding_group: argparse._ArgumentGroup,
    corpus_group: argparse._ArgumentGroup,
    semantic_group: argparse._ArgumentGroup,
) -> None:
    """Add embedding model selection and arXiv corpus sourcing options.

    :param argparse.ArgumentParser build_parser: Build parser carrying the
        streaming tri-state default.
    :param argparse._ArgumentGroup embedding_group: "Embedding runtime" section.
    :param argparse._ArgumentGroup corpus_group: "arXiv corpus" section.
    :param argparse._ArgumentGroup semantic_group: "Semantic discovery" section.
    :return None: Mutates the parser and groups in place.
    """
    # Embedding strategy arguments
    embedding_group.add_argument(
        "--model",
        "-m",
        type=_non_empty_str,
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help="Model name or local checkpoint path (default: %(default)s)",
    )
    embedding_group.add_argument(
        "--model-profile",
        metavar="PROFILE",
        choices=list(EMBEDDING_MODEL_PROFILE_CHOICES),
        default="auto",
        help=(
            "auto, default, embeddinggemma (default: auto). "
            "Use an explicit profile for stripped local exports."
        ),
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

    corpus_group.add_argument(
        "--dataset-source",
        type=_non_empty_str,
        default=DEFAULT_DATASET_SOURCE,
        help=(
            "HuggingFace arXiv metadata dataset source "
            f"(default: {DEFAULT_DATASET_SOURCE})"
        ),
    )

    corpus_group.add_argument(
        "--dataset-split",
        type=_non_empty_str,
        default="train",
        help=(
            "ArXiv dataset split (default: train; non-streaming slices such as "
            "train[:1000] bound rows exposed to CiteMesh)"
        ),
    )

    corpus_group.add_argument(
        "--corpus-size",
        type=_positive_int,
        metavar="N",
        default=None,
        help=(
            "Opt in to caching only the N newest submissions after scanning the split "
            "(default: full selected split)"
        ),
    )

    corpus_group.add_argument(
        "--all-corpus",
        action="store_true",
        help="Use the full selected split (the default), overriding a configured corpus cap",
    )

    semantic_group.add_argument(
        "--top-k",
        "-k",
        type=_positive_int,
        metavar="N",
        default=4,
        help="Neighbors per node for the embedding strategy only (default: 4)",
    )

    embedding_group.add_argument(
        "--truncate-dim",
        type=_positive_int,
        metavar="N",
        default=None,
        help=(
            "Optional embedding output dimension truncation "
            "(for EmbeddingGemma: 768, 512, 256, 128; default: 512; "
            "other models use their profile recommendation)"
        ),
    )

    semantic_group.add_argument(
        "--min-semantic-similarity",
        type=_threshold_float,
        default=EMBEDDING_CONFIG.min_semantic_similarity,
        help=(
            "Minimum semantic cosine for embedding/hybrid graph edges "
            "(default: %(default)s; calibrated for EmbeddingGemma at 512 dimensions)"
        ),
    )

    streaming_group = corpus_group.add_mutually_exclusive_group()
    streaming_group.add_argument(
        "--streaming",
        dest="streaming",
        action="store_true",
        help="Stream HuggingFace dataset instead of loading it into memory (requires non-sliced --dataset-split)",
    )
    streaming_group.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Load cached HuggingFace dataset shards instead of streaming them.",
    )
    build_parser.set_defaults(streaming=False)


def _add_build_cache_arguments(
    build_parser: argparse.ArgumentParser,
    storage_group: argparse._ArgumentGroup,
) -> None:
    """Add embedding cache storage, precision, and prefilter options.

    :param argparse.ArgumentParser build_parser: Build parser carrying the
        binary-prefilter tri-state default.
    :param argparse._ArgumentGroup storage_group: "Embedding cache" section.
    :return None: Mutates the parser and group in place.
    """
    storage_group.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Forcefully clear and rebuild embedding cache for this model before running.",
    )
    storage_group.add_argument(
        "--overwrite-cache",
        action="store_true",
        help=(
            "Acknowledge destructive cache overwrite for --force-rebuild-cache and "
            "skip interactive confirmation."
        ),
    )
    storage_group.add_argument(
        "--cache-overwrite-reason",
        type=str,
        default=None,
        help=(
            "Optional rationale string logged when --force-rebuild-cache clears "
            "embedding cache state."
        ),
    )

    storage_group.add_argument(
        "--storage-precision",
        metavar="PRECISION",
        choices=["int8", "float32"],
        default=EMBEDDING_STORAGE_CONFIG.storage_precision,
        help="Persistent vectors (default: int8 for corpus, float32 for candidates)",
    )

    binary_prefilter_group = storage_group.add_mutually_exclusive_group()
    binary_prefilter_group.add_argument(
        "--binary-prefilter",
        dest="binary_prefilter",
        action="store_true",
        help="Binary Hamming prefilter + rescoring (default: on for int8 corpus)",
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

    storage_group.add_argument(
        "--binary-rescore-multiplier",
        type=_positive_int,
        metavar="N",
        default=EMBEDDING_STORAGE_CONFIG.binary_rescore_multiplier,
        help=(
            "Oversampling factor for binary prefilter rescoring (default: %(default)s)"
        ),
    )

    storage_group.add_argument(
        "--calibration-sample-size",
        type=_positive_int,
        metavar="N",
        default=EMBEDDING_STORAGE_CONFIG.calibration_sample_size,
        help=(
            "Calibration sample size for int8 quantization ranges "
            "(default: %(default)s)"
        ),
    )

    storage_group.add_argument(
        "--cache-compression",
        metavar="FILTER",
        type=str,
        choices=list(_CACHE_COMPRESSION_CHOICES),
        default=EMBEDDING_STORAGE_CONFIG.compression,
        help=("HDF5 compression: gzip, lzf (default: %(default)s)"),
    )

    storage_group.add_argument(
        "--cache-compression-level",
        type=_non_negative_int,
        metavar="N",
        default=EMBEDDING_STORAGE_CONFIG.compression_level,
        help=(
            "HDF5 compression level for embedding cache datasets (default: %(default)s; "
            "only applies to --cache-compression gzip)"
        ),
    )


def _add_build_runtime_arguments(
    build_parser: argparse.ArgumentParser,
    embedding_group: argparse._ArgumentGroup,
    semantic_group: argparse._ArgumentGroup,
) -> None:
    """Add encode-time runtime options: batching, compile, pool size, and device.

    :param argparse.ArgumentParser build_parser: Build parser carrying the
        torch-compile tri-state default.
    :param argparse._ArgumentGroup embedding_group: "Embedding runtime" section.
    :param argparse._ArgumentGroup semantic_group: "Semantic discovery" section.
    :return None: Mutates the parser and groups in place.
    """
    embedding_group.add_argument(
        "--batch-size",
        "-bs",
        dest="encode_batch_size",
        type=_positive_int,
        metavar="N",
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

    semantic_group.add_argument(
        "--semantic-source",
        metavar="SOURCE",
        dest="semantic_source",
        choices=list(SEMANTIC_SOURCE_CHOICES),
        default="candidates",
        help=(
            "candidates embeds S2 seed neighbors; arxiv-corpus builds a local "
            "corpus (default: %(default)s). Corpus flags imply arxiv-corpus."
        ),
    )
    semantic_group.add_argument(
        "--candidate-pool-size",
        dest="candidate_pool_size",
        type=_positive_int,
        metavar="N",
        default=DEFAULT_CANDIDATE_POOL_SIZE,
        help=(
            "Maximum S2 candidate pool size fetched in candidates mode "
            "(default: %(default)s)."
        ),
    )
    embedding_group.add_argument(
        "--device",
        metavar="DEVICE",
        dest="device",
        choices=list(EMBEDDING_DEVICE_CHOICES),
        default="auto",
        help=(
            "auto, cuda, mps, cpu (default: auto). Auto prefers CUDA, "
            "then MPS (Apple Silicon), then CPU."
        ),
    )


def _add_build_hybrid_arguments(
    hybrid_group: argparse._ArgumentGroup,
) -> None:
    """Add the hybrid expansion budget option.

    :param argparse._ArgumentGroup hybrid_group: "Hybrid expansion" section.
    :return None: Mutates the group in place.
    """
    # Hybrid strategy arguments
    hybrid_group.add_argument(
        "--max-semantic",
        type=_non_negative_int,
        metavar="N",
        default=None,
        help=(
            "Semantic additions, at most max-papers - 1; 0 disables embeddings. "
            f"Default: reserve citation depth, then add up to {DEFAULT_MAX_SEMANTIC}."
        ),
    )


def _add_build_arguments(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Add the ``build`` subcommand and every option it accepts.

    Argument groups are created up front because help sections render in
    creation order, while each group's options render in the order they are
    added; the per-section helpers below therefore run in argv-documentation
    order rather than group order.

    :param argparse._SubParsersAction subparsers: Root command subparser action.
    :return argparse.ArgumentParser: The ``build`` subcommand parser.
    """
    # Build command
    build_parser = subparsers.add_parser(
        "build",
        help="Build a graph from a paper or topic",
        description="Discover related papers using recommendations, citations, embeddings, or a hybrid of citations and embeddings.",
        epilog=_help_examples(
            (
                "Create or extend the dashboard collection in out/",
                'citemesh build "arxiv:1706.03762" -s hybrid -e dashboard',
            ),
            (
                "Search 10,000 recent papers by topic",
                'citemesh build "long-context models" -s embedding --corpus-size 10000',
            ),
        ),
    )

    # Required arguments
    build_parser.add_argument(
        "paper_id",
        type=_non_empty_str,
        metavar="PAPER",
        help="DOI, arXiv ID/URL, S2 ID, or free text with --strategy embedding",
    )

    graph_group = build_parser.add_argument_group("Graph")
    graph_group.add_argument(
        "--refresh-paper-cache",
        action="store_true",
        help="Fetch fresh S2 paper metadata and citation counts, retaining embedding caches.",
    )
    export_group = build_parser.add_argument_group("Output")
    citation_group = build_parser.add_argument_group("Citations and references")
    semantic_group = build_parser.add_argument_group(
        "Semantic discovery", "For embedding and hybrid strategies."
    )
    embedding_group = build_parser.add_argument_group("Embedding runtime")
    corpus_group = build_parser.add_argument_group(
        "arXiv corpus",
        "These options select arxiv-corpus sourcing when --semantic-source is omitted.",
    )
    storage_group = build_parser.add_argument_group("Embedding cache")
    hybrid_group = build_parser.add_argument_group("Hybrid expansion")

    _add_build_graph_arguments(graph_group, export_group)
    _add_build_citation_arguments(citation_group)
    _add_build_corpus_arguments(
        build_parser, embedding_group, corpus_group, semantic_group
    )
    _add_build_cache_arguments(build_parser, storage_group)
    _add_build_runtime_arguments(build_parser, embedding_group, semantic_group)
    _add_build_hybrid_arguments(hybrid_group)
    return build_parser


def _add_view_arguments(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Add the ``view`` subcommand.

    :param argparse._SubParsersAction subparsers: Root command subparser action.
    :return argparse.ArgumentParser: The ``view`` subcommand parser.
    """
    view_parser = subparsers.add_parser(
        "view",
        help="Open saved results in a browser",
        description="Open an existing HTML export or a collection directory's dashboard.html in a browser.",
        epilog=_help_examples(
            ("Open the default saved dashboard", "citemesh view"),
            ("Open another collection", "citemesh view out/my-collection"),
            ("Choose Chrome on Linux", "citemesh view --browser google-chrome"),
        ),
    )
    view_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("out") / DASHBOARD_COLLECTION_FILENAME,
        help="HTML file or collection directory (default: out/dashboard.html)",
    )
    view_parser.add_argument(
        "--browser",
        type=_non_empty_str,
        default=None,
        help="Browser name, such as google-chrome (default: system browser)",
    )

    return view_parser


def _add_search_arguments(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Add the ``search`` subcommand.

    :param argparse._SubParsersAction subparsers: Root command subparser action.
    :return argparse.ArgumentParser: The ``search`` subcommand parser.
    """
    search_parser = subparsers.add_parser(
        "search",
        help="Search papers (local semantic index or the Semantic Scholar API)",
        description=(
            "Find papers to pass to citemesh build. Auto mode searches cached "
            "embeddings when available, otherwise Semantic Scholar. Local mode "
            "uses the same model and cache settings as your configured builds."
        ),
        epilog=_help_examples(
            (
                "Search available sources automatically",
                'citemesh search "long-context language models" -n 10',
            ),
            (
                "Search cached embeddings",
                'citemesh search "long-context language models" --mode local',
            ),
            (
                "Search Semantic Scholar by keyword",
                'citemesh search "Megalodon" --mode s2',
            ),
        ),
    )
    search_parser.add_argument(
        "query",
        type=_non_empty_str,
        metavar="QUERY",
        help="Search text (quote phrases with spaces)",
    )
    search_parser.add_argument(
        "--limit",
        "-n",
        type=_positive_int,
        metavar="N",
        default=10,
        help="Maximum results (default: 10)",
    )
    search_parser.add_argument(
        "--mode",
        metavar="MODE",
        choices=list(SEARCH_MODE_CHOICES),
        default=None,
        help=(
            "auto, local, s2 (default: saved search_mode, else auto). "
            "Auto uses local embeddings when available, otherwise S2."
        ),
    )
    search_parser.add_argument(
        "--model",
        "-m",
        type=_non_empty_str,
        default=None,
        help=(
            "Embedding model for local search (implies --mode local); must "
            "match the model used at build time (default: config.toml "
            "default or built-in default)"
        ),
    )
    search_parser.add_argument(
        "--model-profile",
        metavar="PROFILE",
        choices=list(EMBEDDING_MODEL_PROFILE_CHOICES),
        default=None,
        help=(
            "auto, default, embeddinggemma; match the build profile "
            "(implies --mode local)"
        ),
    )
    search_parser.add_argument(
        "--device",
        metavar="DEVICE",
        choices=list(EMBEDDING_DEVICE_CHOICES),
        default=None,
        help="auto, cuda, mps, cpu for query encoding (implies --mode local)",
    )

    return search_parser


def _add_cache_arguments(
    subparsers: argparse._SubParsersAction,
) -> tuple[argparse.ArgumentParser, list[tuple[argparse.ArgumentParser, str]]]:
    """Add the ``cache`` subcommand and its ``clear``/``scan`` operations.

    :param argparse._SubParsersAction subparsers: Root command subparser action.
    :return tuple[argparse.ArgumentParser, list[tuple[argparse.ArgumentParser, str]]]:
        The ``cache`` parser and the usage synopses it owns, in help order.
    """
    cache_parser = subparsers.add_parser(
        "cache",
        help="Inspect or clear local caches",
        description="Inspect local cache usage or delete cached data. Clearing preserves your config.toml settings.",
        epilog=_help_examples(
            ("Show storage by cache section", "citemesh cache scan"),
            ("Clear cached data after confirmation", "citemesh cache clear"),
        ),
    )
    cache_subparsers = cache_parser.add_subparsers(
        dest="cache_command",
        title="Cache operations",
        metavar="COMMAND",
    )
    cache_clear_parser = cache_subparsers.add_parser(
        "clear",
        help="Delete cached data; preserve config.toml",
        description="Delete cached papers, references, and embeddings. config.toml is preserved. Prompts for confirmation; scripts require --yes.",
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
    cache_scan_parser = cache_subparsers.add_parser(
        "scan",
        help="Show sections, file counts, and storage usage",
        description="Show the cache root, usage by section, and total disk space.",
    )

    return cache_parser, [
        (cache_parser, "COMMAND [options]"),
        (cache_clear_parser, "[options]"),
        (cache_scan_parser, "[options]"),
    ]


def _add_config_arguments(
    subparsers: argparse._SubParsersAction,
) -> tuple[argparse.ArgumentParser, list[tuple[argparse.ArgumentParser, str]]]:
    """Add the ``config`` subcommand and its list/get/set/unset/path operations.

    :param argparse._SubParsersAction subparsers: Root command subparser action.
    :return tuple[argparse.ArgumentParser, list[tuple[argparse.ArgumentParser, str]]]:
        The ``config`` parser and the usage synopses it owns, in help order.
    """
    config_parser = subparsers.add_parser(
        "config",
        help="View or change personal defaults",
        description=(
            "Manage persistent CiteMesh defaults stored in config.toml under "
            "the cache root. Precedence: explicit CLI flag > environment "
            "variable > config.toml > built-in default."
        ),
        epilog=_help_examples(
            ("Show saved settings", "citemesh config list"),
            (
                "Use the local corpus by default",
                "citemesh config set defaults.semantic_source arxiv-corpus",
            ),
            (
                "Reset one setting to its default",
                "citemesh config unset defaults.semantic_source",
            ),
        ),
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
        title="Config operations",
        metavar="COMMAND",
    )
    config_list_parser = config_subparsers.add_parser(
        "list",
        help="Show configured values and the config file path",
    )
    config_get_parser = config_subparsers.add_parser(
        "get",
        help="Print one configured value",
    )
    config_get_parser.add_argument(
        "key",
        type=_non_empty_str,
        help="Dotted config key (for example defaults.semantic_source)",
    )
    config_set_parser = config_subparsers.add_parser(
        "set",
        help="Set and persist one config value",
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
    )
    config_unset_parser.add_argument(
        "key",
        type=_non_empty_str,
        help="Dotted config key (for example defaults.semantic_source)",
    )
    config_path_parser = config_subparsers.add_parser(
        "path",
        help="Print the config file path",
    )

    return config_parser, [
        (config_parser, "COMMAND [options]"),
        (config_list_parser, "[options]"),
        (config_get_parser, "KEY [options]"),
        (config_set_parser, "KEY VALUE [options]"),
        (config_unset_parser, "KEY [options]"),
        (config_path_parser, "[options]"),
    ]


def _normalize_action_metavars(command_parser: argparse.ArgumentParser) -> None:
    """Give every action an explicit metavar so help columns stay predictable.

    Positionals fall back to their upper-cased destination; free-text options
    get a short placeholder keyed by destination so ``--output PATH`` reads
    better than ``--output OUTPUT``.

    :param argparse.ArgumentParser command_parser: Parser whose actions are labeled.
    :return None: Mutates each action's ``metavar`` in place.
    """
    for action in command_parser._actions:
        if not action.option_strings and not isinstance(
            action, argparse._SubParsersAction
        ):
            action.metavar = (
                action.dest.upper() if action.metavar is None else action.metavar
            )
        elif (
            action.option_strings
            and action.type in (str, _non_empty_str)
            and action.choices is None
        ):
            action.metavar = {
                "output": "PATH",
                "log_file": "PATH",
                "model": "MODEL",
                "model_revision": "REV",
                "dataset_source": "DATASET",
                "dataset_split": "SPLIT",
                "browser": "NAME",
            }.get(action.dest, "TEXT")


def _finalize_command_parser(
    command_parser: argparse.ArgumentParser, synopsis: str, *, is_root: bool
) -> None:
    """Apply the shared help chrome to one command parser.

    :param argparse.ArgumentParser command_parser: Parser to finalize.
    :param str synopsis: Usage suffix rendered after the program name.
    :param bool is_root: Whether this is the root parser, which keeps logging
        defaults instead of suppressing them.
    :return None: Mutates the parser's help presentation in place.
    """
    command_parser.formatter_class = _HelpFormatter
    command_parser.usage = f"%(prog)s {synopsis}"
    command_parser._positionals.title = "Arguments"
    command_parser._optionals.title = "Options"
    _add_logging_arguments(command_parser, suppress_defaults=not is_root)
    command_parser._action_groups.sort(
        key=lambda group: {"Options": 1, "Logging": 2}.get(group.title, 0)
    )
    _normalize_action_metavars(command_parser)


def _create_parser() -> tuple[
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
    parser, subparsers = _create_root_parser()
    build_parser = _add_build_arguments(subparsers)
    view_parser = _add_view_arguments(subparsers)
    search_parser = _add_search_arguments(subparsers)
    cache_parser, cache_synopses = _add_cache_arguments(subparsers)
    config_parser, config_synopses = _add_config_arguments(subparsers)

    for command_parser, synopsis in (
        (parser, "COMMAND [options]"),
        (build_parser, "PAPER [options]"),
        (view_parser, "[PATH] [options]"),
        (search_parser, "QUERY [options]"),
        *cache_synopses,
        *config_synopses,
    ):
        _finalize_command_parser(
            command_parser, synopsis, is_root=command_parser is parser
        )

    _instrument_parser_actions(parser)
    return parser, build_parser, cache_parser, config_parser
