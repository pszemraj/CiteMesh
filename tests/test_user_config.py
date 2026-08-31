"""Tests for the persistent user config system (config.toml + `citemesh config`)."""

from __future__ import annotations

import argparse
import io
import logging
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

from citemesh import cli as cli_module
from citemesh.core.config import EmbeddingStorageConfig
from citemesh.core.user_config import (
    CONFIG_DEFAULT_KEY_SPECS,
    DEVICE_CHOICES,
    EXPORT_CHOICES,
    MODEL_PROFILE_CHOICES,
    SEARCH_MODE_CHOICES,
    SEMANTIC_SOURCE_CHOICES,
    STORAGE_PRECISION_CHOICES,
    STRATEGY_CHOICES,
    THEME_CHOICES,
    ConfigFileError,
    ConfigKeyError,
    ConfigValueError,
    UserConfig,
    format_config_value,
    load_user_config,
    set_config_value,
    unset_config_value,
    user_config_path,
)
from citemesh.strategies.hybrid import HYBRID_DEFAULT_MAX_REFERENCES


def _run_cli(args: list[str]) -> SimpleNamespace:
    """Run the CLI in-process and capture stdout/stderr."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            returncode = cli_module.main(args)
        except SystemExit as exc:
            code = exc.code
            if isinstance(code, int):
                returncode = code
            elif code is None:
                returncode = 0
            else:
                returncode = 1
    return SimpleNamespace(
        returncode=returncode, stdout=stdout.getvalue(), stderr=stderr.getvalue()
    )


def _parsed_build_args(argv: list[str]) -> tuple:
    """Parse build args and return ``(args, provided, build_parser)``."""
    _, build_parser, _, _ = cli_module._create_parser()
    args = build_parser.parse_args(argv)
    provided = cli_module._pop_tracked_option_dests(args)
    return args, provided, build_parser


# ---------------------------------------------------------------------------
# Module-level load/set/unset behavior
# ---------------------------------------------------------------------------


def test_load_missing_file_returns_empty_config(tmp_path: Path) -> None:
    config = load_user_config(tmp_path / "config.toml")
    assert config.defaults == {}
    assert config.s2_api_key is None


def test_set_get_unset_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    assert (
        set_config_value("defaults.semantic_source", "arxiv-corpus", path=config_path)
        == "arxiv-corpus"
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
        "max_papers": 25,
        "streaming": True,
        "export": ["json", "dashboard"],
    }

    assert unset_config_value("defaults.streaming", path=config_path) is True
    assert unset_config_value("defaults.streaming", path=config_path) is False
    reloaded = load_user_config(config_path)
    assert "streaming" not in reloaded.defaults
    assert reloaded.defaults["semantic_source"] == "arxiv-corpus"


def test_set_rejects_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(ConfigKeyError, match="Unknown config key"):
        set_config_value("defaults.nope", "1", path=tmp_path / "config.toml")
    with pytest.raises(ConfigKeyError, match="Unknown config key"):
        set_config_value("semantic_source", "candidates", path=tmp_path / "config.toml")


def test_set_rejects_invalid_value_without_writing(tmp_path: Path) -> None:
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
    """Core persistent-storage config should allow only int8 and float32."""
    with pytest.raises(ValueError, match="storage_precision must be one of"):
        EmbeddingStorageConfig(storage_precision="float16").validate()


def test_load_skips_unknown_and_invalid_entries(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
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
    with caplog.at_level(logging.WARNING, logger="citemesh.core.user_config"):
        config = load_user_config(config_path)
    assert config.defaults == {"theme": "dark"}
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "mystery_key" in messages
    assert "max_papers" in messages
    assert "unknown_table" in messages


@pytest.mark.parametrize("payload", [b"not [valid toml", b"\xff"])
def test_load_corrupt_file_warns_and_returns_empty(
    payload: bytes, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(payload)
    with caplog.at_level(logging.WARNING, logger="citemesh.core.user_config"):
        config = load_user_config(config_path)
    assert config.defaults == {}
    assert any(
        "unreadable config file" in record.getMessage() for record in caplog.records
    )


@pytest.mark.parametrize("payload", [b"not [valid toml", b"\xff"])
def test_set_on_corrupt_file_fails_loudly(tmp_path: Path, payload: bytes) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(payload)
    with pytest.raises(ConfigFileError, match="Cannot rewrite config file"):
        set_config_value("defaults.theme", "dark", path=config_path)
    # Corrupt content must remain untouched for manual repair.
    assert config_path.read_bytes() == payload


def test_set_preserves_unrecognized_raw_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[defaults]\nfuture_key = "kept"\n[custom]\nnote = 1\n', encoding="utf-8"
    )
    set_config_value("defaults.theme", "dark", path=config_path)
    raw = config_path.read_text(encoding="utf-8")
    assert "future_key" in raw
    assert "custom" in raw
    assert 'theme = "dark"' in raw


def test_export_accepts_toml_list_and_int_rejects_toml_bool(tmp_path: Path) -> None:
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
    assert format_config_value(True) == "true"
    assert format_config_value(False) == "false"
    assert format_config_value(["json", "png"]) == "json,png"
    assert format_config_value(42) == "42"


# ---------------------------------------------------------------------------
# Contract: config whitelist stays in sync with the build parser
# ---------------------------------------------------------------------------


def test_config_choice_specs_match_build_parser_choices() -> None:
    _, build_parser, _, _ = cli_module._create_parser()
    parser_choices = {
        action.dest: action.choices
        for action in build_parser._actions
        if action.choices is not None
    }
    assert set(parser_choices["strategy"]) == set(STRATEGY_CHOICES)
    assert set(parser_choices["theme"]) == set(THEME_CHOICES)
    assert set(parser_choices["device"]) == set(DEVICE_CHOICES)
    assert set(parser_choices["model_profile"]) == set(MODEL_PROFILE_CHOICES)
    assert set(parser_choices["semantic_source"]) == set(SEMANTIC_SOURCE_CHOICES)
    assert set(parser_choices["storage_precision"]) == set(STORAGE_PRECISION_CHOICES)
    assert STORAGE_PRECISION_CHOICES == ("int8", "float32")
    assert set(parser_choices["export"]) == set(EXPORT_CHOICES)


# Config defaults consumed by non-build commands; `_apply_user_config_defaults`
# skips them for build args because the build namespace lacks the attribute.
_NON_BUILD_CONFIG_KEYS = frozenset({"search_mode"})


def test_config_default_keys_exist_as_build_dests() -> None:
    args, _, _ = _parsed_build_args(["paper-id"])
    for dest in CONFIG_DEFAULT_KEY_SPECS:
        if dest in _NON_BUILD_CONFIG_KEYS:
            assert not hasattr(args, dest), (
                f"non-build config key '{dest}' unexpectedly collides with a "
                "build parser dest"
            )
            continue
        assert hasattr(args, dest), f"config key '{dest}' is not a build parser dest"


def test_search_mode_choices_match_search_parser() -> None:
    parser, _, _, _ = cli_module._create_parser()
    subparsers_action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    search_parser = subparsers_action.choices["search"]
    mode_action = next(
        action for action in search_parser._actions if action.dest == "mode"
    )
    assert set(mode_action.choices) == set(SEARCH_MODE_CHOICES)


def test_search_mode_round_trip_and_validation(tmp_path: Path) -> None:
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
    args, provided, _ = _parsed_build_args(["paper-id"])
    config = UserConfig(
        path=Path("unused"), defaults={"theme": "dark", "max_papers": 22}
    )
    applied = cli_module._apply_user_config_defaults(args, provided, config)
    assert applied == {"theme", "max_papers"}
    assert args.theme == "dark"
    assert args.max_papers == 22


def test_build_parser_uses_dark_theme_by_default() -> None:
    """The built-in visualization theme should be dark-first."""
    args, _, _ = _parsed_build_args(["paper-id"])

    assert args.theme == "dark"


def test_explicit_cli_flag_wins_over_config_default() -> None:
    args, provided, _ = _parsed_build_args(["paper-id", "--theme", "solarized"])
    config = UserConfig(
        path=Path("unused"), defaults={"theme": "dark", "max_papers": 22}
    )
    applied = cli_module._apply_user_config_defaults(args, provided, config)
    assert applied == {"max_papers"}
    assert args.theme == "solarized"


def test_explicit_no_streaming_wins_over_enabled_config_default() -> None:
    """The negative streaming flag should override a persisted enabled default."""
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

    applied = cli_module._apply_user_config_defaults(args, provided, config)
    cli_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )

    assert "streaming" in provided
    assert "streaming" not in applied
    assert args.streaming is False
    assert args.semantic_source == "arxiv-corpus"


def test_config_default_outranks_hybrid_implicit_defaults() -> None:
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "hybrid"]
    )
    config = UserConfig(path=Path("unused"), defaults={"max_citations": 7})
    applied = cli_module._apply_user_config_defaults(args, provided, config)
    cli_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )
    assert args.max_citations == 7
    assert args.max_references == HYBRID_DEFAULT_MAX_REFERENCES


def test_cli_corpus_flag_overrides_config_semantic_source() -> None:
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "embedding", "--corpus-size", "1000"]
    )
    config = UserConfig(path=Path("unused"), defaults={"semantic_source": "candidates"})
    applied = cli_module._apply_user_config_defaults(args, provided, config)
    cli_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )
    assert args.semantic_source == "arxiv-corpus"


def test_cli_candidate_flag_overrides_corpus_config_defaults() -> None:
    """Explicit candidate options should make unused corpus defaults inert."""
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
            "streaming": True,
            "dataset_split": "train[:5%]",
        },
    )
    applied = cli_module._apply_user_config_defaults(args, provided, config)
    cli_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )

    assert args.semantic_source == "candidates"
    assert args.candidate_pool_size == 50
    assert args.streaming is True
    assert args.dataset_split == "train[:5%]"


def test_config_semantic_source_corpus_applies_without_flags() -> None:
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "embedding"]
    )
    config = UserConfig(
        path=Path("unused"), defaults={"semantic_source": "arxiv-corpus"}
    )
    applied = cli_module._apply_user_config_defaults(args, provided, config)
    cli_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )
    assert args.semantic_source == "arxiv-corpus"
    # Corpus mode keeps the int8 storage default (no candidate-mode downgrade).
    assert args.storage_precision == "int8"


def test_config_int8_normalized_in_candidate_mode() -> None:
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "embedding"]
    )
    config = UserConfig(path=Path("unused"), defaults={"storage_precision": "int8"})
    applied = cli_module._apply_user_config_defaults(args, provided, config)
    cli_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )
    assert args.semantic_source == "candidates"
    assert args.storage_precision == "float32"


def test_config_embedding_defaults_do_not_gate_citation_strategy() -> None:
    args, provided, build_parser = _parsed_build_args(
        ["paper-id", "--strategy", "citation"]
    )
    config = UserConfig(
        path=Path("unused"),
        defaults={"device": "cuda", "semantic_source": "arxiv-corpus"},
    )
    applied = cli_module._apply_user_config_defaults(args, provided, config)
    # Must not raise SystemExit: config defaults never count as explicit flags.
    cli_module._validate_build_cli_contract(
        args, build_parser, provided, config_defaults=applied
    )


# ---------------------------------------------------------------------------
# API key precedence
# ---------------------------------------------------------------------------


def test_config_api_key_applied_when_env_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("S2_API_KEY", raising=False)
    config = UserConfig(path=Path("unused"), s2_api_key="config-key")
    cli_module._apply_user_config_api_key(config)
    assert os.environ["S2_API_KEY"] == "config-key"


def test_env_api_key_wins_even_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S2_API_KEY", "")
    config = UserConfig(path=Path("unused"), s2_api_key="config-key")
    cli_module._apply_user_config_api_key(config)
    assert os.environ["S2_API_KEY"] == ""


# ---------------------------------------------------------------------------
# `citemesh config` subcommand
# ---------------------------------------------------------------------------


def test_config_cli_round_trip() -> None:
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
    unknown = _run_cli(["config", "set", "defaults.nope", "1"])
    assert unknown.returncode == 2
    assert "Unknown config key" in unknown.stderr

    bad_value = _run_cli(["config", "set", "defaults.device", "warp"])
    assert bad_value.returncode == 2
    assert "expected one of" in bad_value.stderr


def test_config_cli_without_subcommand_prints_help() -> None:
    result = _run_cli(["config"])
    assert result.returncode == 1
    assert "Config operations" in result.stdout


@pytest.mark.parametrize("payload", [b"not [valid toml", b"\xff"])
def test_config_cli_list_survives_corrupt_file(payload: bytes) -> None:
    config_path = user_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(payload)
    result = _run_cli(["config", "list"])
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Cache clear must never delete config.toml
# ---------------------------------------------------------------------------


def test_cache_clear_preserves_config_file() -> None:
    assert _run_cli(["config", "set", "defaults.theme", "dark"]).returncode == 0
    config_path = user_config_path()
    cache_root = config_path.parent
    (cache_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (cache_root / "embeddings" / "payload.bin").write_bytes(b"data")
    (cache_root / "references").mkdir(exist_ok=True)

    result = _run_cli(["cache", "clear", "--yes"])
    assert result.returncode == 0
    assert config_path.is_file()
    assert not (cache_root / "embeddings").exists()
    assert not (cache_root / "references").exists()
    assert load_user_config().defaults == {"theme": "dark"}


def test_cache_clear_without_config_removes_root() -> None:
    cache_root = user_config_path().parent
    (cache_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (cache_root / "embeddings" / "payload.bin").write_bytes(b"data")

    result = _run_cli(["cache", "clear", "--yes"])
    assert result.returncode == 0
    assert not cache_root.exists()
