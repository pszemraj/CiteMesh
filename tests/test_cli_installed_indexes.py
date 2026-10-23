"""Installed-index command selection and model-free option handling."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from citemesh.cli.build_contract import _validate_build_cli_contract
from citemesh.cli.build_options import _shared_embedding_builder_kwargs
from citemesh.cli.cache_selection import _apply_cache_selection
from citemesh.cli.commands import search
from citemesh.cli.commands.cache import run_cache_command
from citemesh.cli.parser import _create_parser, _pop_tracked_option_dests
from citemesh.data.embedding_contract import EmbeddingContract, ExistingCacheSelection


@pytest.fixture
def installed_selection(monkeypatch: pytest.MonkeyPatch) -> ExistingCacheSelection:
    """Use a stored contract without opening native files or a model."""
    from citemesh.data.embedding_cache import lifecycle

    selection = ExistingCacheSelection(
        namespace_id="0123456789ab",
        namespace_serialization="legacy-address",
        contract=EmbeddingContract(
            model_id="publisher/model",
            model_revision="immutable-revision",
            model_artifact_identity="artifact",
            profile_key="embeddinggemma",
            profile_schema="embeddinggemma-v1",
            representation="retrieval-document-v1",
            formatter_identity="formatter",
            normalization="l2-v1",
            dimension=256,
            stored_dtype="int8",
            quantization="scalar-int8",
            calibration_sample_size=2000,
        ),
        source_repo_id="publisher/source",
        source_config_name=None,
        source_split="train",
    )
    monkeypatch.setattr(lifecycle, "select_existing_cache", lambda _: selection)
    monkeypatch.setattr(
        lifecycle,
        "open_existing_cache",
        lambda _: SimpleNamespace(
            hydration_operation_lock=nullcontext,
            corpus_state=lambda: {"source": {"selection": "all", "limit": None}},
        ),
    )
    return selection


@pytest.mark.parametrize("strategy", ["embedding", "hybrid"])
def test_selected_contract_precedes_build_validation(
    installed_selection: ExistingCacheSelection, strategy: str
) -> None:
    parser, build_parser, _, _ = _create_parser()
    args = parser.parse_args(
        [
            "build",
            "arxiv:2404.08801",
            "--strategy",
            strategy,
            "--cache-namespace",
            installed_selection.namespace_id,
        ]
    )
    provided = _pop_tracked_option_dests(args)
    args.model = "unrelated/config-default"
    args.semantic_source = "candidates"
    _apply_cache_selection(args, provided, build_parser)
    _validate_build_cli_contract(args, build_parser, provided)
    kwargs = _shared_embedding_builder_kwargs(args)
    assert kwargs["semantic_source"] == "arxiv-corpus"
    assert kwargs["model_name"] == "publisher/model"
    assert kwargs["model_revision"] == "immutable-revision"
    assert kwargs["truncate_dim"] == 256
    assert kwargs["cache_selection"] is installed_selection


@pytest.mark.parametrize(
    "flags",
    [
        ["--model", "different/model"],
        ["--semantic-source", "candidates"],
        ["--truncate-dim", "512"],
        ["--dataset-split", "test"],
        ["--storage-precision", "float32"],
    ],
)
def test_explicit_installed_contract_conflicts_are_usage_errors(
    installed_selection: ExistingCacheSelection, flags: list[str]
) -> None:
    parser, build_parser, _, _ = _create_parser()
    args = parser.parse_args(
        [
            "build",
            "seed",
            "--strategy",
            "embedding",
            "--cache-namespace",
            installed_selection.namespace_id,
            *flags,
        ]
    )
    with pytest.raises(SystemExit) as exc:
        _apply_cache_selection(args, _pop_tracked_option_dests(args), build_parser)
    assert exc.value.code == 2


@pytest.mark.parametrize("mode", [[], ["--mode", "auto"], ["--mode", "local"]])
def test_selected_search_never_falls_back(
    monkeypatch: pytest.MonkeyPatch, mode: list[str]
) -> None:
    parser, build_parser, _, _ = _create_parser()
    args = parser.parse_args(
        [
            "search",
            "query",
            "--cache-namespace",
            "0123456789ab",
            *mode,
        ]
    )
    user_config = SimpleNamespace(
        defaults={"search_mode": "s2"}, path=Path("config.toml")
    )

    def unavailable(*args: object) -> None:
        raise ValueError("incomplete selected corpus")

    def unexpected_s2(*args: object) -> None:
        pytest.fail("selected cache search fell back to Semantic Scholar")

    monkeypatch.setattr(search, "_prepare_local_search_builder", unavailable)
    monkeypatch.setattr(search, "_run_s2_search", unexpected_s2)
    assert search._run_search_command(args, build_parser, user_config) == 1


def test_search_corpus_view_options_flow_into_overrides() -> None:
    parser, _, _, _ = _create_parser()
    args = parser.parse_args(
        [
            "search",
            "query",
            "--dataset-split",
            "train",
            "--corpus-size",
            "5",
        ]
    )
    assert search._namespace_overrides(args) == {
        "dataset_split": "train",
        "corpus_size": 5,
    }


def test_cache_import_requires_one_source() -> None:
    parser, _, _, _ = _create_parser()
    for flags in ([], ["--path", "snapshot", "--repo", "owner/repo"]):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["cache", "import", *flags])
        assert exc.value.code == 2


def test_local_import_reports_namespace_and_copyable_use(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from citemesh.data.embedding_cache import portable

    parser, _, cache_parser, _ = _create_parser()
    args = parser.parse_args(["cache", "import", "--path", str(tmp_path)])
    observed = []

    def install(path: Path, *, publication: dict | None) -> SimpleNamespace:
        observed.append((path, publication))
        return SimpleNamespace(db_path=tmp_path / "metadata_0123456789ab.db")

    monkeypatch.setattr(portable, "import_snapshot", install)
    assert run_cache_command(args, cache_parser) == 0
    assert observed == [(tmp_path, None)]
    output = capsys.readouterr().out
    assert "--cache-namespace 0123456789ab" in " ".join(output.split())
    assert "query and graph encoding" in " ".join(output.split())


def test_hub_import_preflights_before_payload_transfer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from citemesh.data.embedding_cache import portable
    from citemesh.services import embedding_indexes

    parser, _, cache_parser, _ = _create_parser()
    args = parser.parse_args(
        [
            "cache",
            "import",
            "--repo",
            "owner/index",
            "--config",
            "variant",
        ]
    )
    calls = []
    publication = SimpleNamespace(
        repo_id="owner/index", config_name="variant", revision="immutable-pin"
    )

    def resolve(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append("resolve")
        return publication

    def preflight(value: object) -> None:
        assert value is publication
        calls.append("preflight")
        raise ValueError("resume pinned pending import")

    def transfer(*args: object, **kwargs: object) -> None:
        pytest.fail("payload downloaded after destination refusal")

    monkeypatch.setattr(embedding_indexes, "resolve_publication", resolve)
    monkeypatch.setattr(
        embedding_indexes, "preflight_publication_destination", preflight
    )
    monkeypatch.setattr(embedding_indexes, "download_publication", transfer)
    monkeypatch.setattr(portable, "import_snapshot", transfer)
    assert run_cache_command(args, cache_parser) == 1
    assert calls == ["resolve", "preflight"]


def test_refresh_requires_explicit_legacy_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from citemesh.data.embedding_cache import lifecycle
    from citemesh.strategies.embedding import refresh

    parser, _, cache_parser, _ = _create_parser()
    args = parser.parse_args(["cache", "refresh", "--namespace", "0123456789ab"])

    def uncertified(*args: object, **kwargs: object) -> None:
        raise ValueError("Legacy membership is uncertified; explicit adoption required")

    def unexpected_adoption(*args: object, **kwargs: object) -> None:
        pytest.fail("ordinary refresh adopted an uncertified legacy cache")

    monkeypatch.setattr(lifecycle, "select_existing_cache", uncertified)
    monkeypatch.setattr(refresh, "adopt_legacy_cache", unexpected_adoption)
    assert run_cache_command(args, cache_parser) == 1


def test_legacy_adoption_passes_historical_selectors_for_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from citemesh.strategies import embedding
    from citemesh.strategies.embedding import refresh

    parser, _, cache_parser, _ = _create_parser()
    args = parser.parse_args(
        [
            "cache",
            "refresh",
            "--namespace",
            "0123456789ab",
            "--adopt",
            "--model-revision",
            "main",
            "--dataset-revision",
            "source-pin",
            "--truncate-dim",
            "256",
            "--dataset-source",
            "owner/source",
        ]
    )
    captured = {}

    def builder(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    def adopt(value: SimpleNamespace, namespace_id: str, *, revision: str) -> None:
        captured.update(builder=value, namespace=namespace_id, revision=revision)

    monkeypatch.setattr(embedding, "EmbeddingGraphBuilder", builder)
    monkeypatch.setattr(refresh, "adopt_legacy_cache", adopt)
    assert run_cache_command(args, cache_parser) == 0
    assert captured["builder"].model_revision == "main"
    assert captured["builder"].truncate_dim == 256
    assert captured["builder"].dataset_source == "owner/source"
    assert captured["builder"].semantic_source == "arxiv-corpus"
    assert captured["namespace"] == "0123456789ab"
    assert captured["revision"] == "source-pin"


def test_legacy_selectors_require_adopt() -> None:
    parser, _, cache_parser, _ = _create_parser()
    args = parser.parse_args(
        [
            "cache",
            "refresh",
            "--namespace",
            "0123456789ab",
            "--model-revision",
            "main",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        run_cache_command(args, cache_parser)
    assert exc.value.code == 2


def test_unknown_published_config_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from citemesh.services import embedding_indexes

    parser, _, cache_parser, _ = _create_parser()
    args = parser.parse_args(
        [
            "cache",
            "import",
            "--repo",
            "owner/index",
            "--config",
            "unknown",
        ]
    )

    def unknown(*args: object, **kwargs: object) -> None:
        raise embedding_indexes.IndexSelectionError(
            "Unknown index configuration 'unknown'"
        )

    monkeypatch.setattr(embedding_indexes, "resolve_publication", unknown)
    with pytest.raises(SystemExit) as exc:
        run_cache_command(args, cache_parser)
    assert exc.value.code == 2


def test_hub_http_failures_return_an_operation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from citemesh.cli.commands.cache import _hub_http_error_types
    from citemesh.services import embedding_indexes

    parser, _, cache_parser, _ = _create_parser()
    args = parser.parse_args(["cache", "import", "--repo", "owner/index"])
    error_type = _hub_http_error_types()[0]

    def unavailable(*args: object, **kwargs: object) -> None:
        response = SimpleNamespace(headers={}, request=None)
        raise error_type("503 Service Unavailable", response=response)

    monkeypatch.setattr(embedding_indexes, "resolve_publication", unavailable)
    assert run_cache_command(args, cache_parser) == 1


def test_minimal_selected_finite_corpus_uses_certified_limit(
    installed_selection: ExistingCacheSelection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from citemesh.data.embedding_cache import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "open_existing_cache",
        lambda _: SimpleNamespace(
            hydration_operation_lock=nullcontext,
            corpus_state=lambda: {"source": {"selection": "newest", "limit": 5}},
        ),
    )
    parser, build_parser, _, _ = _create_parser()
    args = parser.parse_args(
        [
            "build",
            "seed",
            "--strategy",
            "embedding",
            "--cache-namespace",
            installed_selection.namespace_id,
        ]
    )
    _apply_cache_selection(args, _pop_tracked_option_dests(args), build_parser)
    assert args.corpus_size == 5
    assert args.all_corpus is False
