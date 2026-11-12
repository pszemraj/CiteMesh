"""Tests for the persistent user config system (config.toml + `citemesh config`)."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from citemesh import cli as cli_module
from citemesh.cli import build_contract as build_contract_module
from citemesh.cli import build_options as build_options_module
from citemesh.cli import cache_ops as cache_ops_module
from citemesh.cli import graph_config as graph_config_module
from citemesh.cli import parser as parser_module
from citemesh.core import API_CONFIG
from citemesh.core.choices import (
    DEVICE_CHOICES,
    EXPORT_CHOICES,
    HYBRID_SEMANTIC_SOURCE_CHOICES,
    MODEL_PROFILE_CHOICES,
    SEARCH_MODE_CHOICES,
    STORAGE_PRECISION_CHOICES,
    STRATEGY_CHOICES,
    THEME_CHOICES,
)
from citemesh.core.config import EmbeddingStorageConfig
from citemesh.core.validation import ValueValidationError, parse_unit_interval_float
from citemesh.data import cache as cache_module
from citemesh.data import user_config as user_config_module
from citemesh.data.user_config import (
    CONFIG_DEFAULT_KEY_SPECS,
    ConfigFileError,
    ConfigKeyError,
    ConfigValue,
    ConfigValueError,
    UserConfig,
    format_config_value,
    load_user_config,
    set_config_value,
    unset_config_value,
    user_config_path,
)
from citemesh.strategies import embedding as embedding_module
from citemesh.strategies.hybrid import HYBRID_DEFAULT_MAX_REFERENCES
from tests._helpers import PausedFirstWrite, run_captured_cli


def _deny_mkdir(monkeypatch: pytest.MonkeyPatch, denied_path: Path) -> None:
    """Reject creation of one directory while allowing other mkdir calls.

    :param pytest.MonkeyPatch monkeypatch: Fixture installing the scoped patch.
    :param Path denied_path: Directory whose creation should fail.
    :return None: Patches ``Path.mkdir`` for this test.
    """
    original_mkdir = Path.mkdir

    def deny_mkdir(
        path: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        """Reject the selected path and forward other mkdir options.

        :param Path path: Directory being created.
        :param int mode: Permissions passed to ``mkdir``.
        :param bool parents: Whether parent directories are created.
        :param bool exist_ok: Whether an existing directory is accepted.
        :return None: Creates unrelated directories normally.
        """
        if path == denied_path:
            raise PermissionError("config directory denied")
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", deny_mkdir)


def _run_cli(args: list[str]) -> SimpleNamespace:
    """Run the CLI in-process and capture stdout/stderr.

    :param list[str] args: CLI arguments excluding the program name.
    :return SimpleNamespace: Return code and captured output streams.
    """
    return run_captured_cli(lambda: cli_module.main(args))


def _parsed_build_args(argv: list[str]) -> tuple:
    """Parse build args and return ``(args, provided, build_parser)``.

    :param list[str] argv: Build arguments beginning with the paper identifier.
    :return tuple: Parsed namespace, explicit option names, and parser.
    """
    _, build_parser, _, _ = parser_module._create_parser()
    args = build_parser.parse_args(argv)
    provided = parser_module._pop_tracked_option_dests(args)
    return args, provided, build_parser


# ---------------------------------------------------------------------------
# Module-level load/set/unset behavior
# ---------------------------------------------------------------------------


def test_load_missing_file_returns_empty_config(tmp_path: Path) -> None:
    """Missing config files should load as empty configuration.

    :param Path tmp_path: Pytest temporary directory.
    :return None: Assertions validate the empty configuration.
    """
    config = load_user_config(tmp_path / "config.toml")
    assert config.defaults == {}
    assert config.s2_api_key is None


def test_set_get_unset_round_trip(tmp_path: Path) -> None:
    """Supported values should survive set, load, and unset operations.

    :param Path tmp_path: Pytest temporary directory.
    :return None: Assertions validate the persisted round trip.
    """
    config_path = tmp_path / "config.toml"
    assert (
        set_config_value("defaults.semantic_source", "arxiv-corpus", path=config_path)
        == "arxiv-corpus"
    )
    assert (
        set_config_value(
            "defaults.dataset_source", "research/arxiv-snapshot", path=config_path
        )
        == "research/arxiv-snapshot"
    )
    assert set_config_value("defaults.max_papers", "25", path=config_path) == 25
    assert set_config_value("defaults.streaming", "true", path=config_path) is True
    assert set_config_value("defaults.export", "json,dashboard", path=config_path) == [
        "json",
        "dashboard",
    ]

    config = load_user_config(config_path)
    assert config.defaults == {
        "semantic_source": "arxiv-corpus",
        "dataset_source": "research/arxiv-snapshot",
        "max_papers": 25,
        "streaming": True,
        "export": ["json", "dashboard"],
    }

    assert unset_config_value("defaults.streaming", path=config_path) is True
    assert unset_config_value("defaults.streaming", path=config_path) is False
    reloaded = load_user_config(config_path)
    assert "streaming" not in reloaded.defaults
    assert reloaded.defaults["semantic_source"] == "arxiv-corpus"
    assert reloaded.defaults["dataset_source"] == "research/arxiv-snapshot"


def test_unset_removes_an_unknown_key_present_in_the_file(tmp_path: Path) -> None:
    """A key left by a removed option can be unset; an absent unknown key cannot.

    :param Path tmp_path: Pytest temporary directory.
    :return None: Assertions validate removal and the unknown-key error.
    """
    config_path = tmp_path / "config.toml"
    with pytest.raises(ConfigKeyError, match="Unknown config key"):
        unset_config_value("defaults.max_semantic", path=config_path)
    assert not config_path.exists()

    config_path.write_text(
        '[defaults]\nmax_semantic = 30\ntheme = "dark"\n', encoding="utf-8"
    )
    assert unset_config_value("defaults.max_semantic", path=config_path) is True
    assert config_path.read_text(encoding="utf-8") == '[defaults]\ntheme = "dark"\n'
    with pytest.raises(ConfigKeyError, match="Unknown config key"):
        unset_config_value("defaults.max_semantic", path=config_path)


def test_config_file_is_written_owner_only(tmp_path: Path) -> None:
    """config.toml must stay owner-only on POSIX because it can hold an API key.

    :param Path tmp_path: Pytest temporary directory.
    :return None: Assertions validate persisted values and POSIX permission bits.
    """
    config_path = tmp_path / "config.toml"
    set_config_value("defaults.max_papers", "25", path=config_path)
    if os.name == "posix":
        assert config_path.stat().st_mode & 0o7777 == 0o600
    assert load_user_config(config_path).defaults["max_papers"] == 25

    if os.name == "posix":
        config_path.chmod(0o644)
    set_config_value("defaults.max_papers", "30", path=config_path)
    if os.name == "posix":
        assert config_path.stat().st_mode & 0o7777 == 0o600
    assert load_user_config(config_path).defaults["max_papers"] == 30


def test_config_set_serializes_concurrent_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """config set must hold one lock across read, merge, and atomic write.

    :param pytest.MonkeyPatch monkeypatch: Fixture patching the module's writer.
    :param Path tmp_path: Pytest temporary directory.
    :return None: Assertions validate that neither concurrent update is lost.
    """
    config_path = tmp_path / "config.toml"

    delayed_atomic_write = PausedFirstWrite(user_config_module.atomic_write_text)

    monkeypatch.setattr(user_config_module, "atomic_write_text", delayed_atomic_write)

    errors: list[BaseException] = []

    def worker(dotted_key: str, raw_value: str) -> None:
        """Persist one config value from a worker thread.

        :param str dotted_key: Dotted config key to set.
        :param str raw_value: Raw CLI-style value string.
        :return None: Records any raised exception for the main thread.
        """
        try:
            set_config_value(dotted_key, raw_value, path=config_path)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    first_thread = threading.Thread(target=worker, args=("defaults.max_papers", "25"))
    second_thread = threading.Thread(target=worker, args=("defaults.streaming", "true"))

    first_thread.start()
    assert delayed_atomic_write.first_started.wait(timeout=5), (
        "first write never started"
    )
    second_thread.start()
    second_wrote_early = delayed_atomic_write.second_started.wait(timeout=0.25)

    delayed_atomic_write.release_first.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not errors
    assert not second_wrote_early, (
        "second config set reached write path before first released config lock"
    )

    config = load_user_config(config_path)
    assert config.defaults["max_papers"] == 25
    assert config.defaults["streaming"] is True


def test_set_rejects_unknown_key(tmp_path: Path) -> None:
    """Config mutation should reject unknown and incomplete keys.

    :param Path tmp_path: Pytest temporary directory.
    :return None: Assertions validate key errors.
    """
    with pytest.raises(ConfigKeyError, match="Unknown config key"):
        set_config_value("defaults.nope", "1", path=tmp_path / "config.toml")
    with pytest.raises(ConfigKeyError, match="Unknown config key"):
        set_config_value("semantic_source", "candidates", path=tmp_path / "config.toml")


def test_set_rejects_invalid_value_without_writing(tmp_path: Path) -> None:
    """Invalid values should fail without creating a config file.

    :param Path tmp_path: Pytest temporary directory.
    :return None: Assertions validate value errors and filesystem state.
    """
    config_path = tmp_path / "config.toml"
    with pytest.raises(ConfigValueError, match="defaults.device"):
        set_config_value("defaults.device", "warp", path=config_path)
    with pytest.raises(ConfigValueError, match="expected an integer"):
        set_config_value("defaults.max_papers", "many", path=config_path)
    with pytest.raises(ConfigValueError, match="expected an integer >= 1"):
        set_config_value("defaults.max_papers", "0", path=config_path)
    with pytest.raises(ConfigValueError, match="expected a boolean"):
        set_config_value("defaults.streaming", "maybe", path=config_path)
    with pytest.raises(ConfigValueError, match="unknown export format"):
        set_config_value("defaults.export", "json,docx", path=config_path)
    with pytest.raises(ConfigValueError, match="defaults.storage_precision"):
        set_config_value("defaults.storage_precision", "float16", path=config_path)
    assert not config_path.exists()


def test_embedding_storage_config_rejects_float16() -> None:
    """Core persistent-storage config should allow only int8 and float32.

    :return None: Assertions validate the float16 rejection.
    """
    with pytest.raises(ValueError, match="storage_precision must be one of"):
        EmbeddingStorageConfig(storage_precision="float16").validate()


def test_load_skips_unknown_and_invalid_entries(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Loading should skip invalid entries while preserving valid defaults.

    :param Path tmp_path: Pytest temporary directory.
    :param pytest.LogCaptureFixture caplog: Captured log fixture.
    :return None: Assertions validate filtered defaults and warnings.
    """
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[defaults]",
                'theme = "dark"',
                "max_papers = true",  # bool where int expected
                'mystery_key = "x"',
                "[unknown_table]",
                "value = 1",
            ]
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="citemesh.data.user_config"):
        config = load_user_config(config_path)
    assert config.defaults == {"theme": "dark"}
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "mystery_key" in messages
    assert "max_papers" in messages
    assert "unknown_table" in messages


# Files tomllib cannot read. The last raises a plain ValueError (Python's
# integer-string digit limit) instead of TOMLDecodeError.
_CORRUPT_CONFIG_PAYLOADS = [
    pytest.param(b"not [valid toml", id="syntax"),
    pytest.param(b"\xff", id="not-utf8"),
    pytest.param(b"[defaults]\nmax_papers = " + b"1" * 5000, id="5000-digit-int"),
]


@pytest.mark.parametrize("payload", _CORRUPT_CONFIG_PAYLOADS)
def test_load_corrupt_file_warns_and_returns_empty(
    payload: bytes, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Corrupt config bytes should warn and load as empty configuration.

    :param bytes payload: Invalid TOML or UTF-8 bytes.
    :param Path tmp_path: Pytest temporary directory.
    :param pytest.LogCaptureFixture caplog: Captured log fixture.
    :return None: Assertions validate recovery and warning behavior.
    """
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(payload)
    with caplog.at_level(logging.WARNING, logger="citemesh.data.user_config"):
        config = load_user_config(config_path)
    assert config.defaults == {}
    assert any(
        "unreadable config file" in record.getMessage() for record in caplog.records
    )


@pytest.mark.parametrize("payload", _CORRUPT_CONFIG_PAYLOADS)
def test_set_on_corrupt_file_fails_loudly(tmp_path: Path, payload: bytes) -> None:
    """Mutation should reject corrupt config files without overwriting them.

    :param Path tmp_path: Pytest temporary directory.
    :param bytes payload: Invalid TOML or UTF-8 bytes.
    :return None: Assertions validate failure and byte preservation.
    """
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(payload)
    with pytest.raises(ConfigFileError, match="Cannot rewrite config file"):
        set_config_value("defaults.theme", "dark", path=config_path)
    # Corrupt content must remain untouched for manual repair.
    assert config_path.read_bytes() == payload


_HUGE_TOML_INT = "1" + "0" * 400

# Valid TOML values of every kind a hand-edited config can hold.
_ODD_TOML_LITERALS = [
    pytest.param(_HUGE_TOML_INT, id="huge-int"),
    pytest.param(f"-{_HUGE_TOML_INT}", id="huge-negative-int"),
    pytest.param("3.5", id="float"),
    pytest.param("inf", id="inf"),
    pytest.param("-inf", id="negative-inf"),
    pytest.param("nan", id="nan"),
    pytest.param("true", id="bool"),
    pytest.param("[]", id="empty-array"),
    pytest.param("[1, 2]", id="array"),
    pytest.param("[[1], ['a']]", id="nested-array"),
    pytest.param("[{ a = 1 }]", id="array-of-tables"),
    pytest.param("{ a = 1 }", id="inline-table"),
    pytest.param("1979-05-27T07:32:00Z", id="datetime"),
    pytest.param("1979-05-27", id="date"),
]


@pytest.mark.parametrize("literal", _ODD_TOML_LITERALS)
def test_load_accepts_or_warns_on_odd_values_for_every_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, literal: str
) -> None:
    """No TOML value under ``[defaults]`` may make loading raise.

    Each key gets the value directly and inside a nested table, and either
    accepts it or drops it with a warning naming the key.

    :param Path tmp_path: Pytest temporary directory.
    :param pytest.LogCaptureFixture caplog: Captured log fixture.
    :param str literal: TOML literal written for every key.
    :return None: Assertions validate that loading returns and warns.
    """
    config_path = tmp_path / "config.toml"
    for key in CONFIG_DEFAULT_KEY_SPECS:
        for document in (
            f"[defaults]\n{key} = {literal}\n",
            f"[defaults.{key}]\nvalue = {literal}\n",
        ):
            config_path.write_text(document, encoding="utf-8")
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="citemesh.data.user_config"):
                config = load_user_config(config_path)
            if key not in config.defaults:
                assert f"Ignoring invalid config value for 'defaults.{key}'" in (
                    caplog.text
                ), document


_HUGE_INT_PARAMS = [
    pytest.param(10**400, id="huge-int"),
    pytest.param(-(10**400), id="huge-negative-int"),
]


@pytest.mark.parametrize("literal", _ODD_TOML_LITERALS[:2])
def test_huge_toml_threshold_is_ignored_with_a_warning(
    caplog: pytest.LogCaptureFixture, literal: str
) -> None:
    """An integer beyond float range is an invalid threshold, not a crash.

    :param pytest.LogCaptureFixture caplog: Captured log fixture.
    :param str literal: TOML integer too large to convert to a float.
    :return None: Assertions validate the warning, other keys and ``config list``.
    """
    config_path = user_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f'[defaults]\ntheme = "light"\nmin_semantic_similarity = {literal}\n',
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="citemesh.data.user_config"):
        config = load_user_config()

    assert config.defaults == {"theme": "light"}
    assert (
        "Ignoring invalid config value for 'defaults.min_semantic_similarity'"
        in caplog.text
    )
    assert "expected a finite float between 0.0 and 1.0" in caplog.text
    assert _run_cli(["config", "list"]).returncode == 0


@pytest.mark.parametrize("value", _HUGE_INT_PARAMS)
def test_unit_interval_parser_rejects_ints_beyond_float_range(value: int) -> None:
    """The shared CLI/config float parser reports overflow as out of range.

    :param int value: Integer too large to convert to a float.
    :return None: Assertions validate the validation reason.
    """
    with pytest.raises(ValueValidationError) as error:
        parse_unit_interval_float(value)
    assert error.value.reason == "out_of_range"


def test_config_write_wraps_parent_directory_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config write preparation failures should use the public error type.

    :param Path tmp_path: Temporary directory supplying the target path.
    :param pytest.MonkeyPatch monkeypatch: Filesystem failure injector.
    :return None: Assertions verify the stable configuration error boundary.
    """
    config_path = tmp_path / "blocked" / "config.toml"
    _deny_mkdir(monkeypatch, config_path.parent)

    with pytest.raises(ConfigFileError, match="Failed to prepare config directory"):
        user_config_module._write_document(config_path, {"defaults": {"theme": "dark"}})


@pytest.mark.parametrize("operation", ["set", "unset"])
@pytest.mark.parametrize("failure_site", ["parent", "lock"])
def test_config_cli_wraps_mutation_filesystem_errors(
    monkeypatch: pytest.MonkeyPatch, operation: str, failure_site: str
) -> None:
    """Config mutations should report filesystem failures without a traceback.

    :param pytest.MonkeyPatch monkeypatch: Lock failure injector.
    :param str operation: Config mutation subcommand under test.
    :param str failure_site: Parent-directory or lock-acquisition failure to inject.
    :return None: Assertions verify the CLI returns its ordinary failure status.
    """
    lock: MagicMock | None = None
    if failure_site == "parent":
        config_parent = user_config_path().parent
        _deny_mkdir(monkeypatch, config_parent)
    else:
        lock = MagicMock()
        lock.acquire.side_effect = PermissionError("config lock denied")
        monkeypatch.setattr(
            user_config_module, "FileLock", MagicMock(return_value=lock)
        )

    command = ["config", operation, "defaults.theme"]
    if operation == "set":
        command.append("dark")
    result = _run_cli(command)

    assert result.returncode == 1
    if lock is not None:
        lock.release.assert_not_called()


def test_config_lock_wraps_release_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lock release filesystem failures should retain the config error contract.

    :param Path tmp_path: Temporary directory supplying the lock path.
    :param pytest.MonkeyPatch monkeypatch: Lock failure injector.
    :return None: Assertions verify release errors use ``ConfigFileError``.
    """
    lock = MagicMock()
    lock.release.side_effect = PermissionError("config unlock denied")
    monkeypatch.setattr(user_config_module, "FileLock", MagicMock(return_value=lock))

    with pytest.raises(ConfigFileError, match="Failed to release config file lock"):
        with user_config_module.config_lock(tmp_path / "config.toml"):
            pass


@pytest.mark.parametrize("is_file_false", [False, True])
def test_set_preserves_unrecognized_raw_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_file_false: bool
) -> None:
    """Mutation should preserve unrecognized tables and keys.

    :param Path tmp_path: Pytest temporary directory.
    :param pytest.MonkeyPatch monkeypatch: Filesystem probe override.
    :param bool is_file_false: Emulate an is_file probe hiding an inspection error.
    :return None: Assertions validate forward-compatible preservation.
    """
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[defaults]\nfuture_key = "kept"\n[custom]\nnote = 1\n', encoding="utf-8"
    )
    if is_file_false:
        original_is_file = Path.is_file
        monkeypatch.setattr(
            Path,
            "is_file",
            lambda path: False if path == config_path else original_is_file(path),
        )
    set_config_value("defaults.theme", "dark", path=config_path)
    raw = config_path.read_text(encoding="utf-8")
    assert "future_key" in raw
    assert "custom" in raw
    assert 'theme = "dark"' in raw


def test_export_accepts_toml_list_and_int_rejects_toml_bool(tmp_path: Path) -> None:
    """Typed TOML values should retain list and integer semantics.

    :param Path tmp_path: Pytest temporary directory.
    :return None: Assertions validate coercion and boolean rejection.
    """
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[defaults]\nexport = ["json", "png", "json"]\ntop_k = 6\n', encoding="utf-8"
    )
    config = load_user_config(config_path)
    assert config.defaults["export"] == ["json", "png"]
    assert config.defaults["top_k"] == 6

    assert set_config_value("defaults.torch_compile", True, path=config_path) is True
    with pytest.raises(ConfigValueError, match="boolean"):
        set_config_value("defaults.top_k", True, path=config_path)


def test_format_config_value_round_trips_cli_forms() -> None:
    """Config values should format into accepted CLI representations.

    :return None: Assertions validate each supported value form.
    """
    assert format_config_value(True) == "true"
    assert format_config_value(False) == "false"
    assert format_config_value(["json", "png"]) == "json,png"
    assert format_config_value(42) == "42"


# ---------------------------------------------------------------------------
# Contract: config whitelist stays in sync with the build parser
# ---------------------------------------------------------------------------


# Config defaults consumed by non-build commands; `_apply_user_config_defaults`
# skips them for build args because the build namespace lacks the attribute.
_NON_BUILD_CONFIG_KEYS = frozenset({"search_mode"})


def test_config_default_keys_exist_as_build_dests() -> None:
    """Build defaults should map to parser destinations unless exempted.

    :return None: Assertions validate the config-to-parser key contract.
    """
    args, _, _ = _parsed_build_args(["paper-id"])
    for dest in CONFIG_DEFAULT_KEY_SPECS:
        if dest in _NON_BUILD_CONFIG_KEYS:
            assert not hasattr(args, dest), (
                f"non-build config key '{dest}' unexpectedly collides with a "
                "build parser dest"
            )
            continue
        assert hasattr(args, dest), f"config key '{dest}' is not a build parser dest"


def test_search_mode_round_trip_and_validation(tmp_path: Path) -> None:
    """Search mode should round-trip and reject unsupported values.

    :param Path tmp_path: Pytest temporary directory.
    :return None: Assertions validate persistence and validation.
    """
    config_path = tmp_path / "config.toml"
    assert set_config_value("defaults.search_mode", "local", path=config_path) == (
        "local"
    )
    assert load_user_config(config_path).defaults["search_mode"] == "local"
    with pytest.raises(ConfigValueError, match="defaults.search_mode"):
        set_config_value("defaults.search_mode", "hybrid", path=config_path)


# ---------------------------------------------------------------------------
# Precedence: CLI flag > config.toml > built-in default
# ---------------------------------------------------------------------------


def test_config_default_applied_when_flag_omitted() -> None:
    """Config defaults should apply when their CLI flags are omitted.

    :return None: Assertions validate applied values and provenance.
    """
    args, provided, _ = _parsed_build_args(["paper-id"])
    config = UserConfig(
        path=Path("unused"), defaults={"theme": "dark", "max_papers": 22}
    )
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    assert applied == {"theme", "max_papers"}
    assert args.theme == "dark"
    assert args.max_papers == 22


def test_build_parser_uses_dark_theme_by_default() -> None:
    """The built-in visualization theme should be dark-first.

    :return None: Assertions validate the parser default.
    """
    args, _, _ = _parsed_build_args(["paper-id"])

    assert args.theme == "dark"


def test_explicit_cli_flag_wins_over_config_default() -> None:
    """Explicit CLI flags should override persisted defaults.

    :return None: Assertions validate precedence and provenance.
    """
    args, provided, _ = _parsed_build_args(["paper-id", "--theme", "solarized"])
    config = UserConfig(
        path=Path("unused"), defaults={"theme": "dark", "max_papers": 22}
    )
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    assert applied == {"max_papers"}
    assert args.theme == "solarized"


@pytest.mark.parametrize("explicit", [False, True])
def test_semantic_threshold_config_reaches_builder_kwargs(
    tmp_path: Path, explicit: bool
) -> None:
    """Persisted thresholds route to both strategies with CLI precedence.

    :param Path tmp_path: Isolated config file location.
    :param bool explicit: Whether the CLI overrides the persisted threshold.
    :return None: Checks persisted float parsing and shared builder dispatch.
    """
    path = tmp_path / "config.toml"
    set_config_value("defaults.min_semantic_similarity", "0.68", path=path)
    config = load_user_config(path=path)
    flags = ["--min-semantic-similarity", "0.75"] if explicit else []
    args, provided, _ = _parsed_build_args(
        ["paper-id", "--strategy", "embedding", *flags]
    )
    build_options_module._apply_user_config_defaults(args, provided, config)
    kwargs = build_options_module._shared_embedding_builder_kwargs(args)
    assert kwargs["min_semantic_similarity"] == (0.75 if explicit else 0.68)


def test_configured_dataset_source_is_ignored_in_candidate_mode(
    tmp_path: Path,
) -> None:
    """A corpus-only persisted metadata source must stay inert in candidate mode.

    :param Path tmp_path: Isolated config file location.
    :return None: Assertions validate reset to the built-in source.
    """
    path = tmp_path / "config.toml"
    dataset_source = "research/arxiv-snapshot"
    set_config_value("defaults.dataset_source", dataset_source, path=path)
    config = load_user_config(path=path)
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "embedding"]
    )
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    build_contract_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied, config_path=config.path
    )

    assert args.semantic_source == "candidates"
    assert (
        args.dataset_source
        == build_options_module._CORPUS_ONLY_OPTION_BUILTIN_DEFAULTS["dataset_source"]
    )
    assert (
        build_options_module._shared_embedding_builder_kwargs(args)["dataset_source"]
        == (build_options_module._CORPUS_ONLY_OPTION_BUILTIN_DEFAULTS["dataset_source"])
    )


@pytest.mark.parametrize(
    "value", ["nan", "inf", "-0.1", "1.1", True, *_HUGE_INT_PARAMS]
)
def test_semantic_threshold_config_rejects_invalid_values(value: object) -> None:
    """Keep persisted thresholds within the same range as CLI arguments.

    :param object value: Invalid threshold representation.
    :return None: Checks config's existing validation contract.
    """
    with pytest.raises(ConfigValueError):
        CONFIG_DEFAULT_KEY_SPECS["min_semantic_similarity"](value)


def test_explicit_no_streaming_wins_over_enabled_config_default() -> None:
    """The negative streaming flag should override a persisted enabled default.

    :return None: Assertions validate negative-flag precedence.
    """
    args, provided, build_parser = _parsed_build_args(
        [
            "paper-id",
            "--strategy",
            "embedding",
            "--no-streaming",
            "--dataset-split",
            "train[:5%]",
        ]
    )
    config = UserConfig(path=Path("unused"), defaults={"streaming": True})

    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    build_contract_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )

    assert "streaming" in provided
    assert "streaming" not in applied
    assert args.streaming is False
    assert args.semantic_source == "arxiv-corpus"


def test_no_streaming_does_not_imply_or_conflict_with_candidate_mode() -> None:
    """The negative streaming toggle should be value-aware during mode selection.

    :return None: Assertions verify ``--no-streaming`` remains a candidate-mode no-op.
    """
    args, provided, build_parser = _parsed_build_args(
        [
            "paper-id",
            "--strategy",
            "embedding",
            "--no-streaming",
            "--candidate-pool-size",
            "50",
        ]
    )

    build_contract_module._validate_build_cli_contract(args, build_parser, provided)

    assert "streaming" in provided
    assert args.streaming is False
    assert args.semantic_source == "candidates"
    assert args.storage_precision == "float32"


def test_config_default_outranks_hybrid_implicit_defaults() -> None:
    """Persisted defaults should outrank hybrid mode's implicit defaults.

    :return None: Assertions validate hybrid default precedence.
    """
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "hybrid"]
    )
    config = UserConfig(path=Path("unused"), defaults={"max_citations": 7})
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    build_contract_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )
    assert args.max_citations == 7
    assert args.max_references == HYBRID_DEFAULT_MAX_REFERENCES


def test_cli_corpus_flag_overrides_config_semantic_source() -> None:
    """Explicit corpus flags should override persisted candidate mode.

    :return None: Assertions validate semantic-source selection.
    """
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "embedding", "--corpus-size", "1000"]
    )
    config = UserConfig(path=Path("unused"), defaults={"semantic_source": "candidates"})
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    build_contract_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )
    assert args.semantic_source == "arxiv-corpus"


def test_cli_dataset_source_overrides_config_default() -> None:
    """An explicit metadata source should outrank a persisted source.

    :return None: Assertions validate CLI-over-config precedence in corpus mode.
    """
    cli_source = "cli/arxiv-snapshot"
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "embedding", "--dataset-source", cli_source]
    )
    config = UserConfig(
        path=Path("unused"),
        defaults={
            "semantic_source": "arxiv-corpus",
            "dataset_source": "config/arxiv-snapshot",
        },
    )
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    build_contract_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )

    assert "dataset_source" in provided
    assert "dataset_source" not in applied
    assert args.semantic_source == "arxiv-corpus"
    assert args.dataset_source == cli_source


def test_cli_candidate_flag_overrides_corpus_config_defaults() -> None:
    """Explicit candidate options should make unused corpus defaults inert.

    :return None: Assertions validate candidate-mode precedence.
    """
    args, provided, build_parser = _parsed_build_args(
        [
            "paper-id",
            "--strategy",
            "embedding",
            "--candidate-pool-size",
            "50",
        ]
    )
    config = UserConfig(
        path=Path("unused"),
        defaults={
            "semantic_source": "arxiv-corpus",
            "dataset_source": "research/arxiv-snapshot",
            "streaming": True,
            "dataset_split": "train[:5%]",
        },
    )
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    build_contract_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )

    assert args.semantic_source == "candidates"
    assert args.candidate_pool_size == 50
    # Ignored corpus defaults are reset outright, not merely left unused.
    assert args.streaming is False
    assert (
        args.dataset_source
        == build_options_module._CORPUS_ONLY_OPTION_BUILTIN_DEFAULTS["dataset_source"]
    )
    assert args.dataset_split == "train"


def test_config_semantic_source_corpus_applies_without_flags() -> None:
    """Persisted corpus mode should apply without explicit corpus flags.

    :return None: Assertions validate corpus mode and storage defaults.
    """
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "embedding"]
    )
    dataset_source = "research/arxiv-snapshot"
    config = UserConfig(
        path=Path("unused"),
        defaults={
            "semantic_source": "arxiv-corpus",
            "dataset_source": dataset_source,
        },
    )
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    build_contract_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )
    assert args.semantic_source == "arxiv-corpus"
    assert args.dataset_source == dataset_source
    assert build_options_module._shared_embedding_builder_kwargs(args)[
        "dataset_source"
    ] == (dataset_source)
    # Corpus mode keeps the int8 storage default (no candidate-mode downgrade).
    assert args.storage_precision == "int8"


@pytest.mark.parametrize(
    ("flags", "expected_corpus_size", "expected_all_corpus"),
    [
        ([], 1234, False),
        (["--corpus-size", "5678"], 5678, False),
        (["--all-corpus"], None, True),
    ],
)
def test_configured_corpus_cap_and_full_split_override(
    flags: list[str],
    expected_corpus_size: int | None,
    expected_all_corpus: bool,
) -> None:
    """Corpus caps should apply unless an explicit full-split flag overrides them.

    :param list[str] flags: Corpus-size flag variation.
    :param int | None expected_corpus_size: Expected builder hydration cap.
    :param bool expected_all_corpus: Expected sidecar full-split marker.
    :return None: Assertions validate config and CLI precedence.
    """
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "embedding", *flags]
    )
    config = UserConfig(
        path=Path("cfg-home") / "config.toml",
        defaults={"semantic_source": "arxiv-corpus", "corpus_size": 1234},
    )

    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    build_contract_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied, config_path=config.path
    )

    assert args.semantic_source == "arxiv-corpus"
    assert build_options_module._shared_embedding_builder_kwargs(args)[
        "corpus_size"
    ] == (expected_corpus_size)
    payload = graph_config_module._build_graph_config_payload(
        args,
        "seed",
        {},
        ["json"],
        {"json": Path("graph.json")},
        API_CONFIG.anonymous_retry_budget_seconds,
    )
    embedding = payload["build"]["embedding"]
    assert embedding.get("corpus_size") == expected_corpus_size
    assert embedding["all_corpus"] is expected_all_corpus


@pytest.mark.parametrize("strategy", ["embedding", "hybrid"])
@pytest.mark.parametrize(
    ("semantic_source", "storage_precision", "expected_precision"),
    [
        ("candidates", "int8", "float32"),
        ("arxiv-corpus", "float32", "float32"),
        ("arxiv-corpus", "int8", "int8"),
    ],
)
def test_config_calibration_defaults_follow_effective_storage(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    semantic_source: str,
    storage_precision: str,
    expected_precision: str,
) -> None:
    """Configured calibration sizes should only reach builders using int8 storage.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing runtime probes.
    :param str strategy: Embedding-aware build strategy.
    :param str semantic_source: Configured candidate or corpus source.
    :param str storage_precision: Configured storage precision.
    :param str expected_precision: Effective builder storage precision.
    :return None: Validates constructor compatibility and int8 calibration retention.
    """
    monkeypatch.setattr(embedding_module, "_check_embedding_deps", MagicMock())
    monkeypatch.setattr(
        embedding_module, "resolve_embedding_device", MagicMock(return_value="cpu")
    )
    monkeypatch.setattr(
        cli_module.EmbeddingGraphBuilder,
        "_resolve_source_dtype_hint",
        MagicMock(return_value="float32"),
    )
    monkeypatch.setattr(
        cli_module.EmbeddingGraphBuilder,
        "_resolve_attention_implementation_hint",
        MagicMock(return_value="sdpa"),
    )
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", strategy]
    )
    config = UserConfig(
        path=Path("unused"),
        defaults={
            "semantic_source": semantic_source,
            "storage_precision": storage_precision,
            "calibration_sample_size": 100,
        },
    )
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    build_contract_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )
    builder = cli_module.EmbeddingGraphBuilder(
        **build_options_module._shared_embedding_builder_kwargs(args)
    )
    assert builder.semantic_source == semantic_source
    assert builder.storage_precision == expected_precision
    assert builder.calibration_sample_size == (
        100
        if expected_precision == "int8"
        else cli_module.EMBEDDING_STORAGE_CONFIG.calibration_sample_size
    )


def test_config_embedding_defaults_do_not_gate_citation_strategy() -> None:
    """Embedding-only defaults should not gate citation builds.

    :return None: Completion validates strategy-specific option handling.
    """
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "citation"]
    )
    config = UserConfig(
        path=Path("unused"),
        defaults={"device": "cuda", "semantic_source": "arxiv-corpus"},
    )
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    # Must not raise SystemExit: config defaults never count as explicit flags.
    build_contract_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )


def test_config_device_validation_names_the_config_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime device failures should identify a config-sourced value.

    :param pytest.MonkeyPatch monkeypatch: Pytest patch helper.
    :return None: Assertions verify actionable config provenance.
    """
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "embedding"]
    )
    config = UserConfig(
        path=Path("cfg-home") / "config.toml", defaults={"device": "cuda"}
    )
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    monkeypatch.setattr(
        build_contract_module,
        "resolve_embedding_device",
        MagicMock(side_effect=ValueError("CUDA is unavailable")),
    )

    with pytest.raises(ValueError) as error:
        build_contract_module._validate_build_cli_contract(
            args,
            build_contract_module._ValueErrorParserErrorSink(),
            provided,
            config_defaults=applied,
            config_path=config.path,
        )

    message = str(error.value)
    assert "CUDA is unavailable" in message
    assert "defaults.device" in message
    assert str(config.path) in message
    assert "update or unset" in message


def test_config_strategy_gating_names_the_config_source() -> None:
    """Strategy-gated option failures must identify a config-sourced strategy.

    :return None: Assertions verify the user is pointed at defaults.strategy
        instead of a --strategy flag they never passed.
    """
    args, provided, build_parser = _parsed_build_args(["paper-id", "--top-k", "9"])
    config = UserConfig(
        path=Path("cfg-home") / "config.toml", defaults={"strategy": "citation"}
    )
    applied = build_options_module._apply_user_config_defaults(args, provided, config)

    with pytest.raises(ValueError) as error:
        build_contract_module._validate_build_cli_contract(
            args,
            build_contract_module._ValueErrorParserErrorSink(),
            provided,
            config_defaults=applied,
            config_path=config.path,
        )

    message = str(error.value)
    assert "Unsupported option(s) for --strategy citation" in message
    assert "defaults.strategy" in message
    assert str(config.path) in message


def test_ignored_corpus_config_defaults_are_reset_before_the_builder() -> None:
    """Corpus-only config defaults announced as ignored must not reach builders.

    :return None: Assertions verify the values reset to built-in defaults and
        that the reset table matches the real parser defaults.
    """
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "embedding"]
    )
    config = UserConfig(
        path=Path("cfg-home") / "config.toml",
        defaults={
            "dataset_source": "research/arxiv-snapshot",
            "streaming": True,
            "dataset_split": "train[:99]",
            "corpus_size": 123,
        },
    )
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    assert args.streaming is True

    build_contract_module._validate_build_cli_contract(
        args,
        build_contract_module._ValueErrorParserErrorSink(),
        provided,
        config_defaults=applied,
        config_path=config.path,
    )

    assert args.semantic_source == "candidates"
    assert args.streaming is False
    assert (
        args.dataset_source
        == build_options_module._CORPUS_ONLY_OPTION_BUILTIN_DEFAULTS["dataset_source"]
    )
    assert args.dataset_split == "train"
    assert args.corpus_size is None
    for (
        dest,
        expected,
    ) in build_options_module._CORPUS_ONLY_OPTION_BUILTIN_DEFAULTS.items():
        assert build_parser.get_default(dest) == expected


def test_legacy_max_semantic_config_key_is_ignored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A config written before hybrid ranking replaced the semantic cap still loads.

    :param Path tmp_path: Isolated config directory.
    :param pytest.LogCaptureFixture caplog: Captured load warnings.
    :return None: The removed key is dropped with a warning; other keys apply.
    """
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[defaults]\nstrategy = 'hybrid'\nmax_semantic = 5\nmax_papers = 25\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        config = load_user_config(config_path)

    assert config.defaults == {"strategy": "hybrid", "max_papers": 25}
    assert "Ignoring unknown config key 'defaults.max_semantic'" in caplog.text
    args, provided, build_parser = _parsed_build_args(["paper-id"])
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    build_contract_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )
    assert (args.strategy, args.max_papers) == ("hybrid", 25)


def test_candidate_mode_announces_ignored_corpus_config_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate mode should explain why corpus-only config keys are inert.

    :param pytest.MonkeyPatch monkeypatch: Pytest patch helper.
    :return None: Assertions verify the ignored-key guidance.
    """
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "embedding"]
    )
    config = UserConfig(
        path=Path("cfg-home") / "config.toml",
        defaults={
            "dataset_source": "research/arxiv-snapshot",
            "corpus_size": 1234,
            "streaming": True,
        },
    )
    debug = MagicMock()
    info = MagicMock()
    monkeypatch.setattr(cli_module.logger, "debug", debug)
    monkeypatch.setattr(cli_module.logger, "info", info)

    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    build_contract_module._validate_build_cli_contract(
        args,
        build_parser,
        provided,
        config_defaults=applied,
        config_path=config.path,
    )

    assert "Loaded config defaults" in str(debug.call_args_list)
    info.assert_not_called()
    messages = [call.args[0] % call.args[1:] for call in debug.call_args_list]
    assert (
        "Ignoring corpus-only config default(s) defaults.corpus_size, "
        f"defaults.dataset_source, defaults.streaming from {config.path} "
        "because the effective semantic source is candidates."
    ) in messages
    payload = graph_config_module._build_graph_config_payload(
        args,
        "seed",
        {},
        ["json"],
        {"json": Path("graph.json")},
        API_CONFIG.anonymous_retry_budget_seconds,
    )
    embedding = payload["build"]["embedding"]
    assert embedding["semantic_source"] == "candidates"
    assert embedding["candidate_pool_size"] == args.candidate_pool_size
    assert not set(embedding) & build_options_module._CORPUS_ONLY_OPTION_DESTS
    assert "calibration_sample_size" not in embedding


# ---------------------------------------------------------------------------
# Invariant: no single accepted config default breaks a plain build
# ---------------------------------------------------------------------------

_CFG_PATH = Path("cfg-home") / "config.toml"

# Every choice of a choice-typed key; the lower bound and a typical value of an
# integer; both booleans; the bounds of the similarity threshold; one string,
# plus a sliced split because streaming rejects those.
_REPRESENTATIVE_CONFIG_VALUES: dict[str, tuple[object, ...]] = {
    "strategy": STRATEGY_CHOICES,
    "export": tuple([choice] for choice in EXPORT_CHOICES),
    "theme": THEME_CHOICES,
    "model": ("org/generic-embedding-model",),
    "model_profile": MODEL_PROFILE_CHOICES,
    "model_revision": ("main",),
    "device": DEVICE_CHOICES,
    "semantic_source": HYBRID_SEMANTIC_SOURCE_CHOICES,
    "candidate_pool_size": (1, 400),
    "encode_batch_size": (1, 64),
    "storage_precision": STORAGE_PRECISION_CHOICES,
    "binary_prefilter": (True, False),
    "calibration_sample_size": (1, 2000),
    "max_papers": (1, 40),
    "max_citations": (0, 20),
    "max_references": (0, 20),
    "top_k": (1, 5),
    "truncate_dim": (1, 256),
    "min_semantic_similarity": (0.0, 1.0),
    "corpus_size": (1, 5000),
    "dataset_source": ("research/arxiv-snapshot",),
    "dataset_split": ("train", "train[:5%]"),
    "streaming": (True, False),
    "torch_compile": (True, False),
    "search_mode": SEARCH_MODE_CHOICES,
}


def test_single_default_sweep_covers_every_config_key() -> None:
    """The sweep below must name a value for every accepted ``[defaults]`` key.

    :return None: Assertion fails when a config key is added or removed.
    """
    assert set(_REPRESENTATIVE_CONFIG_VALUES) == set(CONFIG_DEFAULT_KEY_SPECS)


@pytest.mark.parametrize(
    ("key", "raw_value", "strategy"),
    [
        pytest.param(
            key,
            raw_value,
            strategy,
            id=f"{key}={raw_value}-{strategy}",
        )
        for key, values in _REPRESENTATIVE_CONFIG_VALUES.items()
        for raw_value in values
        for strategy in STRATEGY_CHOICES
    ],
)
def test_single_accepted_config_default_never_breaks_a_plain_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw_value: object,
    strategy: str,
) -> None:
    """A default the loader accepts must not fail ``build --strategy S <seed>``.

    Device availability is a property of the host, not of the config: the probe
    is patched to resolve every device, so ``cuda`` and ``mps`` are checked like
    ``cpu`` on any machine. A missing device still fails builds, with the config
    source named, as ``test_config_device_validation_names_the_config_source``
    pins.

    :param Path tmp_path: Isolated config file location.
    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the device probe.
    :param str key: Config ``[defaults]`` key under test.
    :param object raw_value: Value persisted through ``config set``.
    :param str strategy: Strategy passed on the command line.
    :return None: Completion shows the contract accepts the overlaid default.
    """
    monkeypatch.setattr(
        build_contract_module, "resolve_embedding_device", lambda device: device
    )
    config_path = tmp_path / "config.toml"
    value = set_config_value(f"defaults.{key}", raw_value, path=config_path)
    config = load_user_config(config_path)
    assert config.defaults == {key: value}
    args, provided, _ = _parsed_build_args(["paper-id", "--strategy", strategy])
    applied = build_options_module._apply_user_config_defaults(args, provided, config)

    build_contract_module._validate_build_cli_contract(
        args,
        build_contract_module._ValueErrorParserErrorSink(),
        provided,
        config_defaults=applied,
        config_path=config.path,
    )


@pytest.mark.parametrize("strategy_from_config", [False, True])
@pytest.mark.parametrize("strategy", STRATEGY_CHOICES)
def test_configured_combined_source_narrows_outside_hybrid(
    monkeypatch: pytest.MonkeyPatch, strategy: str, strategy_from_config: bool
) -> None:
    """A configured ``both`` means the corpus for every strategy except hybrid.

    :param pytest.MonkeyPatch monkeypatch: Fixture capturing debug logging.
    :param str strategy: Effective build strategy.
    :param bool strategy_from_config: Whether the strategy is configured too.
    :return None: Assertions check the value, its provenance and the debug line.
    """
    debug = MagicMock()
    monkeypatch.setattr(build_options_module.logger, "debug", debug)
    defaults: dict[str, ConfigValue] = {"semantic_source": "both"}
    if strategy_from_config:
        defaults["strategy"] = strategy
    argv = (
        ["paper-id"] if strategy_from_config else ["paper-id", "--strategy", strategy]
    )
    args, provided, _ = _parsed_build_args(argv)

    applied = build_options_module._apply_user_config_defaults(
        args, provided, UserConfig(path=_CFG_PATH, defaults=defaults)
    )

    narrowed = strategy != "hybrid"
    assert args.semantic_source == ("arxiv-corpus" if narrowed else "both")
    assert "semantic_source" in applied
    assert "semantic_source" not in provided
    messages = [call.args[0] % call.args[1:] for call in debug.call_args_list]
    narrowing = [line for line in messages if "defaults.semantic_source='both'" in line]
    assert len(narrowing) == int(narrowed)
    if narrowed:
        assert str(_CFG_PATH) in narrowing[0]
        assert narrowing[0].endswith(f"the active strategy is {strategy}.")


def _configured_embedding_outcome(
    semantic_source: str, flags: list[str]
) -> tuple[str, str]:
    """Validate an embedding build over one configured semantic source.

    :param str semantic_source: Configured ``defaults.semantic_source``.
    :param list[str] flags: Extra build flags after ``--strategy embedding``.
    :return tuple[str, str]: Effective source and storage precision, or
        ``("error", message)`` when the contract rejects the build.
    """
    args, provided, _ = _parsed_build_args(
        ["paper-id", "--strategy", "embedding", *flags]
    )
    config = UserConfig(path=_CFG_PATH, defaults={"semantic_source": semantic_source})
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    try:
        build_contract_module._validate_build_cli_contract(
            args,
            build_contract_module._ValueErrorParserErrorSink(),
            provided,
            config_defaults=applied,
            config_path=config.path,
        )
    except ValueError as exc:
        return "error", str(exc)
    return str(args.semantic_source), str(args.storage_precision)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], ("arxiv-corpus", "int8")),
        (["--candidate-pool-size", "50"], ("candidates", "float32")),
        (["--corpus-size", "100"], ("arxiv-corpus", "int8")),
    ],
)
def test_configured_combined_source_resolves_like_configured_corpus(
    flags: list[str], expected: tuple[str, str]
) -> None:
    """Outside hybrid, a configured ``both`` behaves exactly like ``arxiv-corpus``.

    It stays a config default, so candidate-only flags still imply candidates.

    :param list[str] flags: Extra build flags after ``--strategy embedding``.
    :param tuple[str, str] expected: Effective source and precision.
    :return None: Assertions compare both configured values with the expectation.
    """
    assert _configured_embedding_outcome("both", flags) == expected
    assert _configured_embedding_outcome("arxiv-corpus", flags) == expected


_INT8_CANDIDATES_ERROR = (
    "--storage-precision int8 requires --semantic-source arxiv-corpus "
    "(int8 calibration ranges are computed during corpus hydration)."
)


@pytest.mark.parametrize(
    ("semantic_source", "flags", "config_context"),
    [
        (
            "candidates",
            ["--storage-precision", "int8"],
            f" Config source: defaults.semantic_source in {_CFG_PATH}; "
            "update or unset the configured value.",
        ),
        (
            "arxiv-corpus",
            ["--candidate-pool-size", "50", "--storage-precision", "int8"],
            "",
        ),
        ("both", ["--candidate-pool-size", "50", "--storage-precision", "int8"], ""),
    ],
)
def test_semantic_source_error_context_names_config_only_when_it_decided(
    semantic_source: str, flags: list[str], config_context: str
) -> None:
    """A source implied by explicit flags is no longer blamed on config.toml.

    :param str semantic_source: Configured ``defaults.semantic_source``.
    :param list[str] flags: Extra build flags after ``--strategy embedding``.
    :param str config_context: Expected config provenance suffix.
    :return None: Assertions pin the error and its provenance suffix.
    """
    assert _configured_embedding_outcome(semantic_source, flags) == (
        "error",
        _INT8_CANDIDATES_ERROR + config_context,
    )


@pytest.mark.parametrize(
    ("argv", "defaults", "config_context"),
    [
        (
            ["paper-id", "--strategy", "embedding", "--semantic-source", "both"],
            {"semantic_source": "both"},
            "",
        ),
        (
            ["paper-id", "--semantic-source", "both"],
            {"strategy": "embedding"},
            f" Config source: defaults.strategy in {_CFG_PATH}; "
            "update or unset the configured value.",
        ),
    ],
)
def test_explicit_combined_source_still_requires_hybrid(
    argv: list[str], defaults: dict[str, ConfigValue], config_context: str
) -> None:
    """Only a configured ``both`` is narrowed; the flag stays a usage error.

    :param list[str] argv: Build arguments including ``--semantic-source both``.
    :param dict[str, ConfigValue] defaults: Configured defaults for the run.
    :param str config_context: Expected config provenance suffix.
    :return None: Assertions pin the unchanged error message.
    """
    args, provided, _ = _parsed_build_args(argv)
    config = UserConfig(path=_CFG_PATH, defaults=defaults)
    applied = build_options_module._apply_user_config_defaults(args, provided, config)

    with pytest.raises(ValueError) as error:
        build_contract_module._validate_build_cli_contract(
            args,
            build_contract_module._ValueErrorParserErrorSink(),
            provided,
            config_defaults=applied,
            config_path=config.path,
        )

    assert str(error.value) == (
        "--semantic-source both requires --strategy hybrid." + config_context
    )


@pytest.mark.parametrize("flags", [[], ["--corpus-size", "100"]])
def test_configured_combined_source_is_kept_for_hybrid(flags: list[str]) -> None:
    """Hybrid keeps a configured ``both``, and corpus flags refine it.

    :param list[str] flags: Extra build flags after ``--strategy hybrid``.
    :return None: Assertions check the combined source survives validation.
    """
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "hybrid", *flags]
    )
    config = UserConfig(path=_CFG_PATH, defaults={"semantic_source": "both"})
    applied = build_options_module._apply_user_config_defaults(args, provided, config)
    build_contract_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied, config_path=config.path
    )

    assert args.semantic_source == "both"
    assert args.storage_precision == "int8"


# ---------------------------------------------------------------------------
# API key precedence
# ---------------------------------------------------------------------------


def test_config_api_key_resolved_without_environment_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config API key should reach the client without subprocess inheritance.

    :param pytest.MonkeyPatch monkeypatch: Pytest patch helper.
    :return None: Assertions validate config-key application.
    """
    monkeypatch.delenv("S2_API_KEY", raising=False)
    config = UserConfig(path=Path("unused"), s2_api_key="config-key")
    key = build_options_module._resolve_user_config_api_key(config)
    assert key == "config-key"
    assert "S2_API_KEY" not in os.environ
    factory = MagicMock()
    monkeypatch.setattr(build_options_module, "SemanticScholarClient", factory)
    args = argparse.Namespace(_s2_api_key=key, refresh_paper_cache=True)
    kwargs = build_options_module._configured_client_kwargs(args)
    factory.assert_called_once_with(api_key="config-key", refresh_paper_cache=True)
    assert kwargs["client"] is factory.return_value
    assert args._s2_client is factory.return_value
    assert "S2_API_KEY" not in os.environ


def test_env_api_key_wins_even_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly empty environment API key should outrank config.

    :param pytest.MonkeyPatch monkeypatch: Pytest patch helper.
    :return None: Assertions validate environment precedence.
    """
    monkeypatch.setenv("S2_API_KEY", "")
    config = UserConfig(path=Path("unused"), s2_api_key="config-key")
    assert build_options_module._resolve_user_config_api_key(config) == ""
    assert os.environ["S2_API_KEY"] == ""


# ---------------------------------------------------------------------------
# `citemesh config` subcommand
# ---------------------------------------------------------------------------


def test_config_cli_round_trip() -> None:
    """The config CLI should support path, set, get, list, and unset.

    :return None: Assertions validate the complete CLI round trip.
    """
    path_result = _run_cli(["config", "path"])
    assert path_result.returncode == 0
    reported_path = Path(path_result.stdout.strip())
    assert reported_path == user_config_path()

    set_result = _run_cli(["config", "set", "defaults.semantic_source", "arxiv-corpus"])
    assert set_result.returncode == 0
    assert reported_path.is_file()

    get_result = _run_cli(["config", "get", "defaults.semantic_source"])
    assert get_result.returncode == 0
    assert get_result.stdout.strip() == "arxiv-corpus"

    list_result = _run_cli(["config", "list"])
    assert list_result.returncode == 0
    assert "defaults.semantic_source" in list_result.stdout
    assert "arxiv-corpus" in list_result.stdout

    unset_result = _run_cli(["config", "unset", "defaults.semantic_source"])
    assert unset_result.returncode == 0
    get_after_unset = _run_cli(["config", "get", "defaults.semantic_source"])
    assert get_after_unset.returncode == 1


def test_config_cli_masks_api_key_in_list() -> None:
    """Config listing should mask API keys while direct get remains scriptable.

    :return None: Assertions validate secret display behavior.
    """
    secret = "secret-api-key-123456"
    assert _run_cli(["config", "set", "api.s2_api_key", secret]).returncode == 0
    list_result = _run_cli(["config", "list"])
    assert list_result.returncode == 0
    assert secret not in list_result.stdout
    assert "****" in list_result.stdout
    # `get` still returns the full value for scripting.
    get_result = _run_cli(["config", "get", "api.s2_api_key"])
    assert get_result.stdout.strip() == secret


def test_config_cli_rejects_unknown_key_and_bad_value() -> None:
    """The config CLI should reject unknown keys and invalid values.

    :return None: Assertions validate CLI error responses.
    """
    unknown = _run_cli(["config", "set", "defaults.nope", "1"])
    assert unknown.returncode == 2
    assert "Unknown config key" in unknown.stderr

    bad_value = _run_cli(["config", "set", "defaults.device", "warp"])
    assert bad_value.returncode == 2
    assert "expected one of" in bad_value.stderr


def test_config_cli_without_subcommand_prints_help() -> None:
    """Invoking config without an operation should print help and fail.

    :return None: Assertions validate the incomplete command response.
    """
    result = _run_cli(["config"])
    assert result.returncode == 1
    assert "Config operations" in result.stdout


@pytest.mark.parametrize("payload", _CORRUPT_CONFIG_PAYLOADS)
def test_config_cli_list_survives_corrupt_file(payload: bytes) -> None:
    """Config listing should remain usable with a corrupt config file.

    :param bytes payload: Invalid TOML or UTF-8 bytes.
    :return None: Assertions validate the recovery response.
    """
    config_path = user_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(payload)
    result = _run_cli(["config", "list"])
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Cache clear must never delete config.toml
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_kind", ["file", "directory", "broken_symlink"])
def test_cache_clear_preserves_config_file(config_kind: str) -> None:
    """Cache clearing should preserve persistent user configuration.

    :param str config_kind: Kind of the entry occupying the reserved config path.
    :return None: Assertions validate selective cache removal.
    """
    assert _run_cli(["config", "set", "defaults.theme", "dark"]).returncode == 0
    config_path = user_config_path()
    if config_kind != "file":
        config_path.unlink()
        if config_kind == "directory":
            config_path.mkdir()
        else:
            config_path.symlink_to("missing-config.toml")
    cache_root = config_path.parent
    (cache_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (cache_root / "embeddings" / "payload.bin").write_bytes(b"data")
    (cache_root / "references").mkdir(exist_ok=True)

    result = _run_cli(["cache", "clear", "--yes"])
    assert result.returncode == 0
    assert config_path.exists() or config_path.is_symlink()
    assert not (cache_root / "embeddings").exists()
    assert not (cache_root / "references").exists()
    if config_kind == "file":
        assert load_user_config().defaults == {"theme": "dark"}


def test_cache_clear_waits_for_pending_config_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache clearing must wait until an atomic configuration write completes.

    :param pytest.MonkeyPatch monkeypatch: Fixture pausing a pending config rename.
    :return None: Validates serialization, config preservation, and cache deletion.
    """
    config_path = user_config_path()
    cache_payload = config_path.parent / "papers" / "paper.json"
    cache_payload.parent.mkdir(parents=True, exist_ok=True)
    cache_payload.write_text("{}", encoding="utf-8")
    write_pending = threading.Event()
    allow_write = threading.Event()
    clear_started = threading.Event()
    clear_finished = threading.Event()
    original_replace = cache_module.os.replace

    def delayed_replace(source: str | Path, destination: str | Path) -> None:
        """Pause the config writer after creating its temporary file.

        :param str | Path source: Temporary file ready for replacement.
        :param str | Path destination: Target file being written.
        :return None: Replaces the file once the test releases the writer.
        """
        if Path(destination) == config_path:
            write_pending.set()
            assert allow_write.wait(timeout=5), "config write was never released"
        original_replace(source, destination)

    def clear_cache() -> int:
        """Clear cached data while recording when the operation completes.

        :return int: Cache-clear exit code.
        """
        clear_started.set()
        try:
            return cache_ops_module._clear_cache_directory(
                assume_yes=True, clear_reason=None
            )
        finally:
            clear_finished.set()

    monkeypatch.setattr(cache_module.os, "replace", delayed_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(set_config_value, "defaults.max_papers", "25")
        try:
            assert write_pending.wait(timeout=5), "config write never started"
            clearing = executor.submit(clear_cache)
            assert clear_started.wait(timeout=5), "cache clear never started"
            assert not clear_finished.wait(timeout=0.25), (
                "cache clear completed while a config write was pending"
            )
        finally:
            allow_write.set()
        assert writer.result(timeout=5) == 25
        assert clearing.result(timeout=5) == 0

    assert load_user_config().defaults == {"max_papers": 25}
    assert not cache_payload.exists()


def test_cache_clear_without_config_keeps_coordination_root() -> None:
    """Cache clearing must leave the root available for configuration locking.

    :return None: Assertions validate removal of every cached payload.
    """
    cache_root = user_config_path().parent
    (cache_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (cache_root / "embeddings" / "payload.bin").write_bytes(b"data")

    result = _run_cli(["cache", "clear", "--yes"])
    assert result.returncode == 0
    assert cache_root.is_dir()
    assert {child.name for child in cache_root.iterdir()} <= {
        "config.toml.lock",
        cache_module.CACHE_COORDINATION_DIRNAME,
    }
