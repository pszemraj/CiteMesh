"""``citemesh cache``: scan and clear the on-disk embedding cache."""

from __future__ import annotations

import argparse

from ..cache_ops import _clear_cache_directory, _scan_cache_directory


def run_cache_command(
    args: argparse.Namespace, cache_parser: argparse.ArgumentParser
) -> int:
    """Run the ``cache`` subcommand.

    :param argparse.Namespace args: Parsed CLI namespace for ``cache``.
    :param argparse.ArgumentParser cache_parser: Parser used for usage output.
    :return int: Process-style exit code.
    """
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
