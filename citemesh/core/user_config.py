"""
Persistent user configuration (``config.toml``) for CiteMesh.

The config file lives at ``<cache_root>/config.toml`` (HuggingFace-style,
``~/.cache/citemesh/config.toml`` by default) and stores durable personal
defaults for build flags plus optional API credentials. Effective precedence
is: explicit CLI flag > environment variable > config.toml > built-in default.

The module is deliberately dependency-light: choice tuples are declared as
literals here and kept in sync with the CLI parser by contract tests, so
loading configuration never imports strategy or visualization modules.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

from citemesh.data.cache import get_cache_dir

logger = logging.getLogger(__name__)

USER_CONFIG_FILENAME = "config.toml"
DEFAULTS_TABLE = "defaults"
API_TABLE = "api"

# Kept in sync with the CLI parser by tests/test_user_config.py contract tests.
STRATEGY_CHOICES: Tuple[str, ...] = (
    "recommendation",
    "citation",
    "embedding",
    "hybrid",
)
THEME_CHOICES: Tuple[str, ...] = ("light", "dark", "solarized", "auto")
DEVICE_CHOICES: Tuple[str, ...] = ("auto", "cuda", "mps", "cpu")
SEMANTIC_SOURCE_CHOICES: Tuple[str, ...] = ("candidates", "arxiv-corpus")
STORAGE_PRECISION_CHOICES: Tuple[str, ...] = ("int8", "float32")
SEARCH_MODE_CHOICES: Tuple[str, ...] = ("auto", "local", "s2")
EXPORT_CHOICES: Tuple[str, ...] = (
    "png",
    "html",
    "plotly",
    "dashboard",
    "json",
    "csv",
    "bibtex",
    "graphml",
    "all",
)

_TRUTHY_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSEY_STRINGS = frozenset({"false", "0", "no", "off"})


class ConfigKeyError(ValueError):
    """Raised for unknown or malformed config key names."""


class ConfigValueError(ValueError):
    """Raised for config values that fail whitelist validation."""


class ConfigFileError(RuntimeError):
    """Raised when the config file cannot be read or rewritten safely."""


def _cast_str(value: Any) -> str:
    """Validate a non-empty string value.

    :param Any value: Raw TOML or CLI-provided value.
    :return str: Stripped string value.
    """
    if not isinstance(value, str) or not value.strip():
        raise ConfigValueError("expected a non-empty string")
    return value.strip()


def _cast_bool(value: Any) -> bool:
    """Validate a boolean value, accepting common CLI string spellings.

    :param Any value: Raw TOML or CLI-provided value.
    :return bool: Parsed boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUTHY_STRINGS:
            return True
        if lowered in _FALSEY_STRINGS:
            return False
    raise ConfigValueError("expected a boolean (true/false)")


def _int_caster(minimum: int) -> Callable[[Any], int]:
    """Build a bounded integer caster.

    :param int minimum: Minimum accepted value (inclusive).
    :return Callable[[Any], int]: Caster enforcing type and lower bound.
    """

    def _cast(value: Any) -> int:
        """Cast a raw configuration value to a lower-bounded integer.

        :param Any value: Raw TOML or CLI-provided value.
        :return int: Parsed integer greater than or equal to ``minimum``.
        """
        if isinstance(value, bool):
            raise ConfigValueError("expected an integer, got a boolean")
        if isinstance(value, str):
            try:
                value = int(value.strip())
            except ValueError as exc:
                raise ConfigValueError("expected an integer") from exc
        if not isinstance(value, int):
            raise ConfigValueError("expected an integer")
        if value < minimum:
            raise ConfigValueError(f"expected an integer >= {minimum}")
        return value

    return _cast


def _choice_caster(choices: Tuple[str, ...]) -> Callable[[Any], str]:
    """Build a caster validating membership in a fixed choice tuple.

    :param Tuple[str, ...] choices: Accepted string values.
    :return Callable[[Any], str]: Caster enforcing choice membership.
    """

    def _cast(value: Any) -> str:
        """Cast a raw configuration value to an allowed string choice.

        :param Any value: Raw TOML or CLI-provided value.
        :return str: Stripped string present in ``choices``.
        """
        text = _cast_str(value)
        if text not in choices:
            raise ConfigValueError(f"expected one of: {', '.join(choices)}")
        return text

    return _cast


def _cast_export(value: Any) -> list[str]:
    """Validate an export format list (TOML array or comma-separated string).

    :param Any value: Raw TOML or CLI-provided value.
    :return list[str]: Deduplicated export format list.
    """
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ConfigValueError("expected a list of export format strings")
            items.append(item.strip())
    else:
        raise ConfigValueError(
            "expected an export format string (comma-separated) or list"
        )
    if not items:
        raise ConfigValueError("expected at least one export format")
    invalid = sorted({item for item in items if item not in EXPORT_CHOICES})
    if invalid:
        raise ConfigValueError(
            f"unknown export format(s): {', '.join(invalid)} "
            f"(valid: {', '.join(EXPORT_CHOICES)})"
        )
    return list(dict.fromkeys(items))


@dataclass(frozen=True)
class ConfigKeySpec:
    """Validation spec for one whitelisted config key."""

    caster: Callable[[Any], Any]
    description: str


CONFIG_DEFAULT_KEY_SPECS: Dict[str, ConfigKeySpec] = {
    "strategy": ConfigKeySpec(
        _choice_caster(STRATEGY_CHOICES), "Default --strategy value"
    ),
    "export": ConfigKeySpec(
        _cast_export, "Default --export format list (comma-separated when set)"
    ),
    "theme": ConfigKeySpec(_choice_caster(THEME_CHOICES), "Default --theme value"),
    "model": ConfigKeySpec(_cast_str, "Default --model checkpoint name"),
    "model_revision": ConfigKeySpec(_cast_str, "Default --model-revision token"),
    "device": ConfigKeySpec(_choice_caster(DEVICE_CHOICES), "Default --device value"),
    "semantic_source": ConfigKeySpec(
        _choice_caster(SEMANTIC_SOURCE_CHOICES), "Default --semantic-source value"
    ),
    "candidate_pool_size": ConfigKeySpec(
        _int_caster(1), "Default --candidate-pool-size value"
    ),
    "encode_batch_size": ConfigKeySpec(
        _int_caster(1), "Default --encode-batch-size value"
    ),
    "storage_precision": ConfigKeySpec(
        _choice_caster(STORAGE_PRECISION_CHOICES),
        "Default --storage-precision value",
    ),
    "max_papers": ConfigKeySpec(_int_caster(1), "Default --max-papers value"),
    "max_semantic": ConfigKeySpec(_int_caster(0), "Default --max-semantic value"),
    "max_citations": ConfigKeySpec(_int_caster(0), "Default --max-citations value"),
    "max_references": ConfigKeySpec(_int_caster(0), "Default --max-references value"),
    "top_k": ConfigKeySpec(_int_caster(1), "Default --top-k value"),
    "truncate_dim": ConfigKeySpec(_int_caster(1), "Default --truncate-dim value"),
    "corpus_size": ConfigKeySpec(_int_caster(1), "Default --corpus-size value"),
    "dataset_split": ConfigKeySpec(_cast_str, "Default --dataset-split value"),
    "streaming": ConfigKeySpec(_cast_bool, "Default --streaming/--no-streaming toggle"),
    "torch_compile": ConfigKeySpec(_cast_bool, "Default --torch-compile toggle"),
    # Search-command default (not a build flag): mode for `citemesh search`.
    "search_mode": ConfigKeySpec(
        _choice_caster(SEARCH_MODE_CHOICES),
        "Default `citemesh search` mode (auto = local when cached embeddings "
        "exist, else s2)",
    ),
}

CONFIG_API_KEY_SPECS: Dict[str, ConfigKeySpec] = {
    "s2_api_key": ConfigKeySpec(
        _cast_str, "Semantic Scholar API key (S2_API_KEY env var wins when set)"
    ),
}

_TABLE_SPECS: Dict[str, Dict[str, ConfigKeySpec]] = {
    DEFAULTS_TABLE: CONFIG_DEFAULT_KEY_SPECS,
    API_TABLE: CONFIG_API_KEY_SPECS,
}


@dataclass(frozen=True)
class UserConfig:
    """Validated snapshot of persisted user configuration."""

    path: Path
    defaults: Dict[str, Any] = field(default_factory=dict)
    s2_api_key: Optional[str] = None


def user_config_path() -> Path:
    """Return the config file path under the active cache root.

    :return Path: ``<cache_root>/config.toml`` (not created).
    """
    return get_cache_dir(create=False) / USER_CONFIG_FILENAME


def known_config_keys() -> list[str]:
    """Return all valid dotted config keys.

    :return list[str]: Sorted dotted keys (for example ``defaults.device``).
    """
    keys = [f"{DEFAULTS_TABLE}.{key}" for key in CONFIG_DEFAULT_KEY_SPECS]
    keys.extend(f"{API_TABLE}.{key}" for key in CONFIG_API_KEY_SPECS)
    return sorted(keys)


def parse_config_key(dotted_key: str) -> Tuple[str, str, ConfigKeySpec]:
    """Resolve a dotted config key into its table, key, and validation spec.

    :param str dotted_key: Dotted key such as ``defaults.semantic_source``.
    :return Tuple[str, str, ConfigKeySpec]: ``(table, key, spec)`` triple.
    """
    table, separator, key = str(dotted_key).strip().partition(".")
    specs = _TABLE_SPECS.get(table)
    if not separator or specs is None or key not in specs:
        raise ConfigKeyError(
            f"Unknown config key '{dotted_key}'. "
            f"Valid keys: {', '.join(known_config_keys())}"
        )
    return table, key, specs[key]


def _validated_table(
    document: Dict[str, Any],
    table: str,
    specs: Dict[str, ConfigKeySpec],
    config_path: Path,
) -> Dict[str, Any]:
    """Extract and validate one config table, warning on invalid entries.

    :param Dict[str, Any] document: Parsed TOML document.
    :param str table: Table name to validate.
    :param Dict[str, ConfigKeySpec] specs: Whitelisted key specs for the table.
    :param Path config_path: Source file path used in warning messages.
    :return Dict[str, Any]: Validated key/value pairs (invalid entries dropped).
    """
    raw_table = document.get(table)
    if raw_table is None:
        return {}
    if not isinstance(raw_table, dict):
        logger.warning(
            "Ignoring [%s] in %s: expected a table, got %s.",
            table,
            config_path,
            type(raw_table).__name__,
        )
        return {}
    validated: Dict[str, Any] = {}
    for key, raw_value in raw_table.items():
        spec = specs.get(key)
        if spec is None:
            logger.warning(
                "Ignoring unknown config key '%s.%s' in %s.", table, key, config_path
            )
            continue
        try:
            validated[key] = spec.caster(raw_value)
        except ConfigValueError as exc:
            logger.warning(
                "Ignoring invalid config value for '%s.%s' in %s: %s.",
                table,
                key,
                config_path,
                exc,
            )
    return validated


def load_user_config(path: Optional[Path] = None) -> UserConfig:
    """Load and validate the persisted user configuration.

    Missing files yield an empty config; unreadable files and invalid entries
    are ignored with warnings so a bad config never blocks CLI usage.

    :param Optional[Path] path: Optional explicit config file path override.
    :return UserConfig: Validated configuration snapshot.
    """
    config_path = path if path is not None else user_config_path()
    if not config_path.is_file():
        return UserConfig(path=config_path)
    try:
        with config_path.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Ignoring unreadable config file %s: %s", config_path, exc)
        return UserConfig(path=config_path)

    for table in document:
        if table not in _TABLE_SPECS:
            logger.warning(
                "Ignoring unknown config table [%s] in %s.", table, config_path
            )
    defaults = _validated_table(
        document, DEFAULTS_TABLE, CONFIG_DEFAULT_KEY_SPECS, config_path
    )
    api = _validated_table(document, API_TABLE, CONFIG_API_KEY_SPECS, config_path)
    return UserConfig(
        path=config_path,
        defaults=defaults,
        s2_api_key=api.get("s2_api_key"),
    )


def _read_raw_document(config_path: Path) -> Dict[str, Any]:
    """Read the raw TOML document for read-modify-write operations.

    Unlike :func:`load_user_config`, corrupt files fail loudly here so a
    ``config set`` never silently discards unreadable user content.

    :param Path config_path: Config file path.
    :return Dict[str, Any]: Parsed document (empty when the file is missing).
    """
    if not config_path.is_file():
        return {}
    try:
        with config_path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigFileError(
            f"Cannot rewrite config file {config_path}: {exc}. "
            "Fix or remove the file, then retry."
        ) from exc


def _write_document(config_path: Path, document: Dict[str, Any]) -> None:
    """Atomically persist a TOML document (temp file + rename).

    :param Path config_path: Target config file path.
    :param Dict[str, Any] document: TOML-serializable document.
    :return None: Writes the config file in place.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = tomli_w.dumps(document)
    except (TypeError, ValueError) as exc:
        raise ConfigFileError(
            f"Cannot serialize config document for {config_path}: {exc}"
        ) from exc
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{config_path.name}.",
        suffix=".tmp",
        dir=config_path.parent,
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(payload)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_name, config_path)
    except OSError as exc:
        raise ConfigFileError(
            f"Failed to write config file {config_path}: {exc}"
        ) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def set_config_value(
    dotted_key: str, raw_value: Any, *, path: Optional[Path] = None
) -> Any:
    """Validate and persist one config value.

    Unknown keys already present in the file are preserved on rewrite;
    comments are not (TOML round-trip is value-level).

    :param str dotted_key: Dotted key such as ``defaults.semantic_source``.
    :param Any raw_value: Raw value (CLI string or TOML-native type).
    :param Optional[Path] path: Optional explicit config file path override.
    :return Any: The validated, persisted value.
    """
    table, key, spec = parse_config_key(dotted_key)
    try:
        value = spec.caster(raw_value)
    except ConfigValueError as exc:
        raise ConfigValueError(f"Invalid value for '{dotted_key}': {exc}") from exc
    config_path = path if path is not None else user_config_path()
    document = _read_raw_document(config_path)
    section = document.setdefault(table, {})
    if not isinstance(section, dict):
        raise ConfigFileError(
            f"Cannot rewrite config file {config_path}: [{table}] is not a table."
        )
    section[key] = value
    _write_document(config_path, document)
    return value


def unset_config_value(dotted_key: str, *, path: Optional[Path] = None) -> bool:
    """Remove one config value if present.

    :param str dotted_key: Dotted key such as ``defaults.semantic_source``.
    :param Optional[Path] path: Optional explicit config file path override.
    :return bool: ``True`` when a value was removed, ``False`` if unset.
    """
    table, key, _spec = parse_config_key(dotted_key)
    config_path = path if path is not None else user_config_path()
    document = _read_raw_document(config_path)
    section = document.get(table)
    if not isinstance(section, dict) or key not in section:
        return False
    del section[key]
    if not section:
        del document[table]
    _write_document(config_path, document)
    return True


def format_config_value(value: Any) -> str:
    """Render a config value in the same form ``config set`` accepts.

    :param Any value: Validated config value.
    :return str: CLI-friendly string form.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)
