"""Keep the user-config reference and example in step with the loader and parser.

``docs/reference/config/config_reference.toml`` is hand-maintained in loadable
form: an uncommented key holds its built-in default, and a commented-out key
(``# key = value``) has no single literal default. ``example.toml`` must load
through the real loader as written.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

from citemesh.cli import parser as parser_module
from citemesh.cli.build_options import _HYBRID_BEST_PRACTICE_DEFAULTS
from citemesh.cli.commands.build import _resolve_build_plan
from citemesh.cli.commands.search import _resolve_search_mode
from citemesh.data.user_config import (
    ConfigValue,
    UserConfig,
    known_config_keys,
    load_user_config,
)

_CONFIG_DOCS = Path(__file__).resolve().parents[1] / "docs" / "reference" / "config"
REFERENCE_PATH = _CONFIG_DOCS / "config_reference.toml"
EXAMPLE_PATH = _CONFIG_DOCS / "example.toml"

_TABLE_LINE = re.compile(r"^\[(?P<table>[a-z_]+)\]$")
_COMMENTED_KEY_LINE = re.compile(r"^#\s*(?P<key>[a-z][a-z0-9_]*) = (?P<value>.+)$")


def _uncommented_keys(path: Path) -> set[str]:
    """Return the dotted keys a TOML file sets.

    :param Path path: TOML file to parse.
    :return set[str]: ``table.key`` paths for every key the file sets.
    """
    document: dict[str, object] = tomllib.loads(path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for table, entries in document.items():
        assert isinstance(entries, dict), f"{path}: [{table}] is not a table"
        keys.update(f"{table}.{key}" for key in entries)
    return keys


def _commented_out_keys(path: Path) -> set[str]:
    """Return the dotted keys a loadable reference documents as commented out.

    :param Path path: Reference file to scan.
    :return set[str]: ``table.key`` paths of ``# key = value`` lines.
    """
    table: str | None = None
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if (table_match := _TABLE_LINE.match(line)) is not None:
            table = table_match["table"]
        elif (key_match := _COMMENTED_KEY_LINE.match(line)) is not None:
            assert table is not None, f"commented-out key before any table: {line}"
            # The example value must stay valid TOML so uncommenting it parses.
            tomllib.loads(f"{key_match['key']} = {key_match['value']}")
            keys.add(f"{table}.{key_match['key']}")
    return keys


def _built_in_build_defaults() -> argparse.Namespace:
    """Parse a bare ``build`` invocation to read the parser's built-in defaults.

    :return argparse.Namespace: Build namespace with no options supplied.
    """
    _, build_parser, _, _ = parser_module._create_parser()
    args = build_parser.parse_args(["paper-id"])
    parser_module._pop_tracked_option_dests(args)
    return args


def _effective_export_formats(
    built_in: argparse.Namespace, value: ConfigValue | None
) -> list[str]:
    """Resolve the formats a build writes for one ``export`` value.

    :param argparse.Namespace built_in: Built-in build defaults.
    :param ConfigValue | None value: Configured export value, or ``None`` for unset.
    :return list[str]: Formats the build plan selects.
    """
    args = argparse.Namespace(**vars(built_in))
    args.export = value
    return _resolve_build_plan(args).selected_formats


def test_reference_documents_exactly_the_accepted_keys() -> None:
    """Every accepted key is documented once, and nothing else is.

    :return None: Assertions compare documented and accepted key paths.
    """
    uncommented = _uncommented_keys(REFERENCE_PATH)
    commented = _commented_out_keys(REFERENCE_PATH)
    accepted = set(known_config_keys())

    assert sorted(uncommented & commented) == []
    documented = uncommented | commented
    assert sorted(documented - accepted) == [], "documented but not accepted"
    assert sorted(accepted - documented) == [], "accepted but not documented"


def test_reference_values_are_the_built_in_defaults() -> None:
    """Uncommented values load unchanged and equal the program's defaults.

    :return None: Assertions compare loaded values with parser defaults.
    """
    loaded = load_user_config(REFERENCE_PATH)
    uncommented = _uncommented_keys(REFERENCE_PATH)
    # An invalid value would be dropped with a warning, so equal key sets mean
    # every uncommented value passed validation.
    assert {f"defaults.{key}" for key in loaded.defaults} == {
        key for key in uncommented if key.startswith("defaults.")
    }
    assert loaded.s2_api_key is None

    built_in = _built_in_build_defaults()
    for key, value in loaded.defaults.items():
        if key == "search_mode":
            mode_args = argparse.Namespace(mode=None)
            unset = UserConfig(path=REFERENCE_PATH)
            assert (value, "default") == _resolve_search_mode(mode_args, unset)
        elif key == "export":
            assert _effective_export_formats(built_in, value) == (
                _effective_export_formats(built_in, None)
            )
        else:
            assert value == getattr(built_in, key), key

    # A key is commented out only when no single literal default exists.
    for dotted_key in _commented_out_keys(REFERENCE_PATH):
        table, _, key = dotted_key.partition(".")
        if table == "defaults":
            assert key in _HYBRID_BEST_PRACTICE_DEFAULTS or (
                getattr(built_in, key) is None
            ), dotted_key


def test_example_loads_through_the_real_loader() -> None:
    """The example config loads as written, with every key accepted.

    :return None: Assertions compare the example's keys with the loaded config.
    """
    loaded = load_user_config(EXAMPLE_PATH)
    loaded_keys = {f"defaults.{key}" for key in loaded.defaults}
    if loaded.s2_api_key is not None:
        loaded_keys.add("api.s2_api_key")

    assert loaded_keys == _uncommented_keys(EXAMPLE_PATH)
