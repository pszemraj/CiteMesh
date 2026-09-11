"""``citemesh config``: read, set, and unset persisted user defaults."""

from __future__ import annotations

import argparse
from typing import Any

from rich.text import Text

from citemesh.data.user_config import (
    ConfigFileError,
    ConfigKeyError,
    ConfigValueError,
    format_config_value,
    load_user_config,
    parse_config_key,
    set_config_value,
    unset_config_value,
    user_config_path,
)

from .. import console
from ..console import logger
from ..parser import _output_table


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
        console.output_console.print(
            Text.assemble(("Config file: ", "bold"), str(user_config.path))
        )
        if not user_config.path.is_file():
            console.output_console.print(
                "[dim]File does not exist yet; using built-in defaults. "
                "Create it with `citemesh config set <key> <value>`.[/dim]"
            )
        rows: list[tuple[str, str]] = [
            (f"defaults.{key}", format_config_value(value))
            for key, value in sorted(user_config.defaults.items())
        ]
        if user_config.s2_api_key:
            rows.append(("api.s2_api_key", _masked_secret(user_config.s2_api_key)))
        table = _output_table("CiteMesh User Config")
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        if rows:
            for key, value in rows:
                table.add_row(Text(key), Text(value))
        else:
            table.add_row("(no values set)", "")
        console.output_console.print(table)
        return 0

    if command == "get":
        try:
            table_name, key, _spec = parse_config_key(args.key)
        except ConfigKeyError as exc:
            config_parser.error(str(exc))
        user_config = load_user_config()
        if table_name == "api":
            # Direct get is intentionally raw for shell substitution; list is masked.
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
