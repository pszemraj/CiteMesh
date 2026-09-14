#!/usr/bin/env python3
"""CiteMesh: unified CLI for CiteMesh visualizations.

This package owns the ``citemesh`` entry point: it builds the parser, configures
logging once, and dispatches to the per-subcommand modules in
:mod:`citemesh.cli.commands`. The pieces live in focused submodules --
:mod:`~citemesh.cli.console` (consoles and logging), :mod:`~citemesh.cli.parser`
(argparse construction and validators), :mod:`~citemesh.cli.outputs` (export
formats and output paths), :mod:`~citemesh.cli.build_options` and
:mod:`~citemesh.cli.build_contract` (build option tables and validation),
:mod:`~citemesh.cli.graph_config` (reproducible build payloads), and
:mod:`~citemesh.cli.cache_ops` (cache scan/clear).

Names re-exported here form the CLI's stable import surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from types import ModuleType

from rich.console import Console as Console

from citemesh import __version__ as __version__
from citemesh._lazy import install_lazy_exports
from citemesh.core import EMBEDDING_STORAGE_CONFIG as EMBEDDING_STORAGE_CONFIG
from citemesh.data.cache import atomic_write_json as atomic_write_json
from citemesh.data.user_config import load_user_config
from citemesh.data.user_config import set_config_value as set_config_value
from citemesh.visualization.dashboard.contracts import (
    DASHBOARD_COLLECTION_KIND,
    DASHBOARD_COLLECTION_SCHEMA_VERSION,
)
from citemesh.visualization.dashboard.package import (
    DASHBOARD_PACKAGE_FILENAME,
    DashboardPackageError,
    load_dashboard_package,
    render_dashboard_collection_snapshot,
    update_dashboard_package,
)

from .build_options import _resolve_user_config_api_key
from .console import REDIRECTED_LOG_WIDTH as REDIRECTED_LOG_WIDTH
from .console import _configure_logging
from .console import logger as logger
from .graph_config import canonicalize_paper_id_for_metadata
from .outputs import (
    EXPORT_FORMATS,
    resolve_dashboard_collection_outputs,
    resolve_graph_config_path,
    resolve_output_paths,
)
from .parser import _create_parser, _pop_tracked_option_dests

__all__ = [
    "DASHBOARD_COLLECTION_KIND",
    "DASHBOARD_COLLECTION_SCHEMA_VERSION",
    "DASHBOARD_PACKAGE_FILENAME",
    "DashboardPackageError",
    "EXPORT_FORMATS",
    "canonicalize_paper_id_for_metadata",
    "load_dashboard_package",
    "main",
    "render_dashboard_collection_snapshot",
    "resolve_dashboard_collection_outputs",
    "resolve_graph_config_path",
    "resolve_output_paths",
    "update_dashboard_package",
]

# Deferred so ``import citemesh.cli`` -- and therefore ``citemesh config``,
# ``citemesh view`` and ``citemesh cache`` -- never pulls the embedding stack
# through this module. Attribute access keeps the historical import surface.
__getattr__, __dir__ = install_lazy_exports(
    globals(),
    {
        "EmbeddingGraphBuilder": (
            "citemesh.strategies.embedding",
            "EmbeddingGraphBuilder",
        ),
    },
)


def _command_module(name: str) -> ModuleType:
    """Import a subcommand module on demand.

    Command modules pull heavy optional dependencies (matplotlib via the
    exporters, h5py via the embedding cache), so ``main`` resolves only the one
    the parsed command actually needs.

    :param str name: Subcommand module name under :mod:`citemesh.cli.commands`.
    :return ModuleType: Imported command module.
    """
    return import_module(f".commands.{name}", __name__)


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
        return _command_module("config")._run_config_command(args, config_parser)

    if args.command == "view":
        return _command_module("view")._run_view_command(args.path, args.browser)

    user_config = load_user_config()
    args._s2_api_key = _resolve_user_config_api_key(user_config)

    if args.command == "build":
        return _command_module("build").run_build_command(
            args,
            build_parser,
            provided_build_options=provided_build_options,
            user_config=user_config,
        )
    if args.command == "search":
        return _command_module("search")._run_search_command(
            args, build_parser, user_config
        )
    if args.command == "cache":
        return _command_module("cache").run_cache_command(args, cache_parser)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
