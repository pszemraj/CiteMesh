"""Installed-index command selection and model-free option handling."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import networkx as nx
import pytest

from citemesh.cli.build_contract import _validate_build_cli_contract
from citemesh.cli.build_options import _shared_embedding_builder_kwargs
from citemesh.cli.cache_selection import _apply_cache_selection
from citemesh.cli.commands import search
from citemesh.cli.commands.build import run_build_command
from citemesh.cli.commands.cache import run_cache_command
from citemesh.cli.parser import _create_parser, _pop_tracked_option_dests
from citemesh.data.embedding_contract import EmbeddingContract, ExistingCacheSelection
from citemesh.data.user_config import UserConfig
from tests._helpers import build_seed_graph


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


@pytest.mark.parametrize("committed_rows", [2, 0])
@pytest.mark.parametrize("operation", ["auto", "local", "embedding", "hybrid", "embed"])
def test_interrupted_import_survives_implicit_consumers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    operation: str,
    committed_rows: int,
) -> None:
    """Refuse implicit consumers without erasing an import's resumable prefix.

    Zero committed rows leave a managed pending namespace that holds no papers;
    it is still an installed corpus, not an empty cache to rebuild or bypass.
    """
    import pyarrow as pa

    from citemesh import cli
    from citemesh.core import Paper
    from citemesh.data import get_cache_dir
    from citemesh.data.embedding_cache import portable
    from citemesh.data.embedding_cache.lifecycle import (
        open_existing_cache,
        select_existing_cache,
    )
    from citemesh.data.embedding_cache.portable import export_snapshot, import_snapshot
    from citemesh.data.embedding_contract import InstalledCorpusError
    from citemesh.strategies.embedding import builder as builder_module
    from citemesh.strategies.embedding.builder import EmbeddingGraphBuilder
    from citemesh.strategies.embedding.refresh import _builder_contract
    from tests._helpers import disable_embedding_dep_checks
    from tests.test_embedding_portable import ExportFixture

    disable_embedding_dep_checks(monkeypatch)
    monkeypatch.setattr(builder_module, "resolve_embedding_device", lambda _: "cpu")
    monkeypatch.setattr(
        EmbeddingGraphBuilder, "_resolve_source_dtype_hint", lambda _: "float32"
    )
    monkeypatch.setattr(EmbeddingGraphBuilder, "_load_model", lambda _: None)
    fingerprint = "hf::org/example::" + "a" * 40
    monkeypatch.setattr(
        EmbeddingGraphBuilder, "_resolve_model_fingerprint", lambda _: fingerprint
    )
    builder = EmbeddingGraphBuilder(
        model_name="org/example",
        model_profile="default",
        model_revision="main",
        truncate_dim=4,
        storage_precision="float32",
        semantic_source="arxiv-corpus",
        dataset_source="test/source",
        device="cpu",
    )
    producer = ExportFixture(dtype="float32", dimension=4, count=4)
    producer.contract = _builder_contract(builder)
    snapshot = tmp_path / "snapshot"
    manifest = export_snapshot(producer, snapshot, shard_rows=2)
    broken_shard = snapshot / manifest.shards[0 if committed_rows == 0 else -1].path
    correct_bytes = broken_shard.read_bytes()
    broken_shard.write_bytes(b"interrupted shard")
    destination = get_cache_dir("embeddings")
    with pytest.raises(pa.ArrowInvalid):
        import_snapshot(snapshot, destination, batch_rows=2)
    namespace_id = next(destination.glob("metadata_*.db")).stem.removeprefix(
        "metadata_"
    )
    selection = select_existing_cache(namespace_id, destination, require_ready=False)
    cache = open_existing_cache(selection, destination, require_ready=False)
    state = cache.corpus_state()
    assert state["pending"]["kind"] == "import"
    assert not state["membership_complete"]
    assert cache.get_model_fingerprint() is None
    assert cache.embedding_count() == committed_rows
    before = cache.db_path.read_bytes(), cache.h5_path.read_bytes()

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("incomplete import started provider work or cache rebuild")

    monkeypatch.setattr(search, "_run_s2_search", unexpected)
    monkeypatch.setattr(EmbeddingGraphBuilder, "_ensure_cache_hydrated", unexpected)
    monkeypatch.setattr(cli, "_configure_logging", lambda **kwargs: None)
    caplog.set_level(logging.INFO)
    flags = [
        "--model",
        "org/example",
        "--model-profile",
        "default",
        "--model-revision",
        "main",
        "--truncate-dim",
        "4",
        "--storage-precision",
        "float32",
        "--semantic-source",
        "arxiv-corpus",
        "--dataset-source",
        "test/source",
        "--device",
        "cpu",
    ]
    refusal = (
        f"Corpus cache {namespace_id} has an incomplete import operation; resume it "
        "with `citemesh cache import --path <directory of snapshot "
        f"{manifest.snapshot_id}>` before using the corpus"
    )
    if operation == "embed":
        with pytest.raises(InstalledCorpusError, match="incomplete import") as refused:
            builder.embed_papers(
                {"paper": Paper(paper_id="paper", title="Paper", year=2024)}
            )
        assert refusal in str(refused.value)
    else:
        command = (
            ["search", "attention", "--mode", operation]
            if operation in {"auto", "local"}
            else [
                "build",
                "arxiv:2401.00000",
                "--strategy",
                operation,
                "--output",
                str(tmp_path / "output"),
            ]
        )
        assert cli.main([*command, *flags]) == 1
        assert refusal in caplog.text
        assert not any(record.exc_info for record in caplog.records)
    assert "REBUILDING EMBEDDING CACHE" not in caplog.text
    assert cache.corpus_state() == state
    assert cache.embedding_count() == committed_rows
    assert cache.get_model_fingerprint() is None
    assert (cache.db_path.read_bytes(), cache.h5_path.read_bytes()) == before

    broken_shard.write_bytes(correct_bytes)
    resumed = portable.import_snapshot(snapshot, destination, batch_rows=2)
    assert resumed.embedding_count() == 4
    assert resumed.corpus_state()["membership_complete"]


@pytest.mark.parametrize("mode", ["auto", "local"])
def test_search_names_installed_corpus_its_settings_do_not_select(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    mode: str,
) -> None:
    """Default settings address the candidates namespace, never an installed corpus.

    The cache holds only an installed corpus, so the probe must not load a model or
    leave a provisional namespace behind, and the guidance must name the corpus.
    """
    from citemesh import cli
    from citemesh.data import get_cache_dir
    from citemesh.data.embedding_cache.portable import export_snapshot, import_snapshot
    from citemesh.strategies.embedding import builder as builder_module
    from citemesh.strategies.embedding.builder import EmbeddingGraphBuilder
    from tests._helpers import disable_embedding_dep_checks
    from tests.test_embedding_portable import ExportFixture

    disable_embedding_dep_checks(monkeypatch)
    monkeypatch.setattr(builder_module, "resolve_embedding_device", lambda _: "cpu")
    monkeypatch.setattr(
        EmbeddingGraphBuilder, "_resolve_source_dtype_hint", lambda _: "float32"
    )
    export_snapshot(
        ExportFixture(dtype="int8", dimension=4, count=7),
        tmp_path / "snapshot",
        shard_rows=4,
    )
    destination = get_cache_dir("embeddings")
    installed = import_snapshot(tmp_path / "snapshot", destination)
    namespace_id = installed.db_path.stem.removeprefix("metadata_")
    before = sorted(path.name for path in destination.iterdir())

    def unexpected_model_load(*args: object, **kwargs: object) -> None:
        pytest.fail("an installed corpus the settings do not select loaded a model")

    s2_search = MagicMock(return_value=0)
    monkeypatch.setattr(EmbeddingGraphBuilder, "_load_model", unexpected_model_load)
    monkeypatch.setattr(search, "_run_s2_search", s2_search)
    monkeypatch.setattr(cli, "_configure_logging", lambda **kwargs: None)
    caplog.set_level(logging.INFO)

    exit_code = cli.main(["search", "attention mechanism", "--mode", mode])

    assert sorted(path.name for path in destination.iterdir()) == before
    expected_command = (
        f"citemesh search 'attention mechanism' --cache-namespace {namespace_id}"
    )
    selected = f"Installed corpus {namespace_id} is selected with --cache-namespace"
    if mode == "auto":
        assert exit_code == 0
        s2_search.assert_called_once()
        warning = next(
            record for record in caplog.records if record.levelno == logging.WARNING
        )
        assert warning.getMessage() == (
            "No local embeddings match the current model settings. "
            f"{selected} (for example: {expected_command}). "
            "Searching the Semantic Scholar API instead."
        )
    else:
        assert exit_code == 1
        s2_search.assert_not_called()
        error = next(
            record for record in caplog.records if record.levelno == logging.ERROR
        )
        assert "Run `citemesh build` with this configuration" in error.getMessage()
        assert f"{selected} (for example: {expected_command})." in error.getMessage()


def test_installed_corpus_guidance_pluralizes_and_skips_the_selected_namespace(
    tmp_path: Path,
) -> None:
    """Several unselected corpora are all listed; a selected one is not."""
    from citemesh.data.embedding_cache import lifecycle

    for namespace_id in ("111111111111", "222222222222", "333333333333"):
        with sqlite3.connect(tmp_path / f"metadata_{namespace_id}.db") as conn:
            conn.execute("CREATE TABLE cache_metadata (key TEXT, value TEXT)")
            conn.execute(
                "INSERT INTO cache_metadata VALUES (?, '{\"pending\": {}}')",
                (lifecycle.CORPUS_STATE_KEY,),
            )
    args = argparse.Namespace(query="attention")

    def guidance(selected: str | None) -> str | None:
        builder = SimpleNamespace(
            _cache_selection=(
                SimpleNamespace(namespace_id=selected) if selected else None
            ),
            embedding_cache_directory=lambda: tmp_path,
        )
        return search._installed_corpus_guidance(builder, args)  # type: ignore[arg-type]

    assert guidance(None) == (
        "Installed corpora 111111111111, 222222222222, 333333333333 are selected "
        "with --cache-namespace (for example: citemesh search attention "
        "--cache-namespace 111111111111)."
    )
    assert guidance("111111111111") == (
        "Installed corpora 222222222222, 333333333333 are selected with "
        "--cache-namespace (for example: citemesh search attention "
        "--cache-namespace 222222222222)."
    )
    (tmp_path / "metadata_222222222222.db").unlink()
    assert guidance("111111111111") == (
        "Installed corpus 333333333333 is selected with --cache-namespace "
        "(for example: citemesh search attention --cache-namespace 333333333333)."
    )
    (tmp_path / "metadata_333333333333.db").unlink()
    assert guidance("111111111111") is None


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


@pytest.mark.parametrize("mode", ["local", "auto"])
@pytest.mark.parametrize(
    "flags", [["--corpus-size", "2"], ["--all-corpus"], ["--dataset-split", "train"]]
)
def test_legacy_search_rejects_unsupported_views(
    monkeypatch: pytest.MonkeyPatch, mode: str, flags: list[str]
) -> None:
    """Uncertified caches cannot silently ignore explicit corpus-view options."""
    parser, build_parser, _, _ = _create_parser()
    args = parser.parse_args(["search", "query", "--mode", mode, *flags])
    builder = SimpleNamespace(
        _cache_selection=None,
        _active_model_name=None,
        has_persistent_embedding_artifacts=lambda: True,
        prepare_embedding_cache=lambda: None,
        embedding_cache=SimpleNamespace(embedding_count=lambda: 5),
    )
    defaults = build_parser.parse_args(["seed"])
    monkeypatch.setattr(
        search, "_prepare_local_search_builder", lambda *args: (builder, defaults)
    )

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("unsupported local view fell back to Semantic Scholar")

    monkeypatch.setattr(search, "_run_s2_search", unexpected)
    assert (
        search._run_search_command(args, build_parser, UserConfig(Path("config.toml")))
        == 2
    )


@pytest.mark.parametrize("flags", [["--corpus-size", "2"], ["--all-corpus"]])
def test_implicit_installed_search_replays_its_namespace_and_view(
    installed_selection: ExistingCacheSelection,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
) -> None:
    """A discovered certified cache produces a command for the searched view."""
    parser, build_parser, _, _ = _create_parser()
    defaults = build_parser.parse_args(
        ["seed", "--strategy", "embedding", "--semantic-source", "arxiv-corpus", *flags]
    )
    builder = SimpleNamespace(
        _cache_selection=None,
        _active_model_name=installed_selection.contract.model_id,
        has_persistent_embedding_artifacts=lambda: True,
        embedding_cache=SimpleNamespace(
            embedding_count=lambda: 5,
            hydration_operation_lock=nullcontext,
            last_search_total_embeddings=2 if "--corpus-size" in flags else 5,
        ),
        search_local=lambda query, top_k: [
            SimpleNamespace(paper_id="paper", score=0.9, metadata={"title": "Result"})
        ],
    )

    def prepare() -> None:
        builder._cache_selection = installed_selection

    builder.prepare_embedding_cache = prepare
    monkeypatch.setattr(
        search, "_prepare_local_search_builder", lambda *args: (builder, defaults)
    )
    args = parser.parse_args(["search", "query", "--mode", "local", *flags])
    assert (
        search._run_search_command(args, build_parser, UserConfig(Path("config.toml")))
        == 0
    )
    output = " ".join(capsys.readouterr().out.split())
    assert f"--cache-namespace {installed_selection.namespace_id}" in output
    assert " ".join(flags) in output


def test_cache_import_requires_one_source() -> None:
    parser, _, _, _ = _create_parser()
    for flags in ([], ["--path", "snapshot", "--bucket", "owner/repo"]):
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
            "--bucket",
            "owner/index",
            "--config",
            "variant",
        ]
    )
    calls = []
    publication = SimpleNamespace(
        origin={
            "bucket_id": "owner/index",
            "config_name": "variant",
            "snapshot_id": "snapshot-pin",
        }
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


def test_bucket_import_passes_snapshot_selection_and_records_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from citemesh.data.embedding_cache import portable
    from citemesh.services import embedding_indexes

    parser, _, cache_parser, _ = _create_parser()
    args = parser.parse_args(
        [
            "cache",
            "import",
            "--bucket",
            "owner/index",
            "--config",
            "variant",
            "--snapshot",
            "snapshot-one",
        ]
    )
    origin = {
        "bucket_id": "owner/index",
        "config_name": "variant",
        "snapshot_id": "snapshot-one",
    }
    publication = SimpleNamespace(origin=origin)
    calls = []

    def resolve(bucket_id: str, **kwargs: object) -> SimpleNamespace:
        assert bucket_id == "owner/index"
        assert kwargs == {"config_name": "variant", "snapshot_id": "snapshot-one"}
        calls.append("resolve")
        return publication

    def preflight(value: object) -> None:
        assert value is publication
        calls.append("preflight")

    def download(value: object) -> Path:
        assert value is publication
        calls.append("download")
        return tmp_path

    def install(path: Path, *, publication: dict | None) -> SimpleNamespace:
        assert path == tmp_path
        assert publication == origin
        calls.append("install")
        return SimpleNamespace(db_path=tmp_path / "metadata_0123456789ab.db")

    monkeypatch.setattr(embedding_indexes, "resolve_publication", resolve)
    monkeypatch.setattr(
        embedding_indexes, "preflight_publication_destination", preflight
    )
    monkeypatch.setattr(embedding_indexes, "download_publication", download)
    monkeypatch.setattr(portable, "import_snapshot", install)
    assert run_cache_command(args, cache_parser) == 0
    assert calls == ["resolve", "preflight", "download", "install"]


@pytest.mark.parametrize("flags", [["--config", "variant"], ["--snapshot", "one"]])
def test_local_import_rejects_bucket_selection_flags(flags: list[str]) -> None:
    parser, _, cache_parser, _ = _create_parser()
    args = parser.parse_args(["cache", "import", "--path", "snapshot", *flags])
    with pytest.raises(SystemExit) as exc:
        run_cache_command(args, cache_parser)
    assert exc.value.code == 2


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
            "--bucket",
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


@pytest.mark.parametrize("failure", ["missing", "transport", "batch"])
def test_hub_http_failures_return_an_operation_error(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Real SDK failures cross the transfer boundary as clean CLI errors."""
    import huggingface_hub
    from httpx2 import ConnectError, Request, Response
    from huggingface_hub.errors import BucketBatchError, EntryNotFoundError

    parser, _, cache_parser, _ = _create_parser()
    args = parser.parse_args(["cache", "import", "--bucket", "owner/index"])
    response = Response(503, request=Request("GET", "https://huggingface.co"))
    error = {
        "missing": EntryNotFoundError("missing catalog"),
        "transport": ConnectError("offline"),
        "batch": BucketBatchError("failed bucket operation", response=response),
    }[failure]

    def unavailable(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(download_bucket_files=unavailable),
    )
    assert run_cache_command(args, cache_parser) == 1


@pytest.mark.parametrize("command", ["export", "refresh", "build"])
@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        ValueError("Namespace 0123456789ab is not installed; see cache scan"),
        RuntimeError("Selected cache membership is not certified"),
        RuntimeError("Corpus import is pending"),
        sqlite3.DatabaseError("file is not a database"),
    ],
)
def test_namespace_failures_return_operation_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    command: str,
    error: Exception,
    verbose: bool,
) -> None:
    """Namespace selection failures return exit1 before graph or transfer work."""
    from citemesh.data.embedding_cache import lifecycle

    caplog.set_level(logging.DEBUG if verbose else logging.INFO)

    def unavailable(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(lifecycle, "select_existing_cache", unavailable)
    parser, build_parser, cache_parser, _ = _create_parser()
    if command == "build":
        args = parser.parse_args(
            [
                "build",
                "seed",
                "--strategy",
                "embedding",
                "--cache-namespace",
                "0123456789ab",
            ]
        )
        result = run_build_command(
            args,
            build_parser,
            provided_build_options=_pop_tracked_option_dests(args),
            user_config=UserConfig(tmp_path / "config.toml"),
        )
    else:
        flags = ["--output", str(tmp_path / "out")] if command == "export" else []
        args = parser.parse_args(
            ["cache", command, "--namespace", "0123456789ab", *flags]
        )
        result = run_cache_command(args, cache_parser)
    assert result == 1
    assert not (tmp_path / "out").exists()
    failures = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(failures) == 1
    assert bool(failures[0].exc_info) is verbose


@pytest.mark.parametrize("failure_stage", ["resolve", "load", "iterate"])
def test_refresh_reports_offline_source_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure_stage: str,
) -> None:
    """Report source resolution and scan transport failures without a traceback."""
    from collections.abc import Iterator

    from huggingface_hub import HfApi
    from huggingface_hub.utils import httpx

    from citemesh.data.embedding_cache import lifecycle
    from citemesh.data.embedding_contract import CorpusSource
    from citemesh.strategies.embedding import corpus_source, deps, refresh
    from citemesh.strategies.embedding.records import DatasetRecord

    source = CorpusSource("test/source", None, "train", "main", "all", None)

    def offline(*args: object, **kwargs: object) -> None:
        raise httpx.RemoteProtocolError("offline")

    def interrupted() -> Iterator[DatasetRecord]:
        yield {"id": "2401.00001", "title": "Paper", "abstract": "Body"}
        raise httpx.ReadError("offline")

    def refresh_source(*args: object, **kwargs: object) -> None:
        if failure_stage == "resolve":
            corpus_source.resolve_source_snapshot(source)
        else:
            list(corpus_source.iter_source_records(source))

    monkeypatch.setattr(HfApi, "dataset_info", offline)
    monkeypatch.setattr(
        deps,
        "_import_datasets_module",
        lambda: SimpleNamespace(
            load_dataset=offline
            if failure_stage == "load"
            else lambda *a, **kw: interrupted()
        ),
    )
    monkeypatch.setattr(
        lifecycle, "select_existing_cache", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(refresh, "refresh_cache_selection", refresh_source)
    parser, _, cache_parser, _ = _create_parser()
    args = parser.parse_args(["cache", "refresh", "--namespace", "0123456789ab"])
    assert run_cache_command(args, cache_parser) == 1
    action = "resolve" if failure_stage == "resolve" else "read"
    assert f"Cannot {action} dataset test/source@main: offline" in caplog.text
    assert not any(record.exc_info for record in caplog.records)


def test_legacy_adoption_rejects_explicit_all_against_recorded_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adoption cannot silently replace an explicit all-corpus selector with a cap."""
    from citemesh.data.embedding_cache import lifecycle
    from citemesh.strategies.embedding import refresh

    monkeypatch.setattr(
        lifecycle,
        "read_existing_cache_metadata",
        lambda _: {"hydration_corpus_size": "newest:5"},
    )

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("conflicting adoption started")

    monkeypatch.setattr(refresh, "adopt_legacy_cache", unexpected)
    parser, _, cache_parser, _ = _create_parser()
    args = parser.parse_args(
        ["cache", "refresh", "--namespace", "0123456789ab", "--adopt", "--all-corpus"]
    )
    assert run_cache_command(args, cache_parser) == 1


@pytest.mark.parametrize("strategy", ["embedding", "hybrid"])
@pytest.mark.parametrize("installed_limit", [None, 5])
def test_selected_build_writes_contract_and_runtime_provenance(
    installed_selection: ExistingCacheSelection,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    strategy: str,
    installed_limit: int | None,
) -> None:
    """The build command carries selected identity through export and sidecar writing."""
    from citemesh.cli.commands import build
    from citemesh.data.embedding_cache import lifecycle

    installed_scope = "all" if installed_limit is None else "newest"
    monkeypatch.setattr(
        lifecycle,
        "open_existing_cache",
        lambda _: SimpleNamespace(
            hydration_operation_lock=nullcontext,
            corpus_state=lambda: {
                "source": {"selection": installed_scope, "limit": installed_limit}
            },
        ),
    )

    graph = build_seed_graph()
    origin = {
        "bucket_id": "owner/index",
        "config_name": "variant",
        "snapshot_id": "snapshot-one",
    }
    graph.graph["embedding_runtime"] = {
        "cache_namespace": installed_selection.namespace_id,
        "dataset_revision": "source-pin",
        "index_origin": origin,
        "installed_corpus_selection": installed_scope,
        "installed_corpus_limit": installed_limit,
    }

    def build_graph(
        args: argparse.Namespace, selected_strategy: str
    ) -> tuple[nx.Graph, str]:
        assert selected_strategy == strategy
        assert args._cache_selection is installed_selection
        assert args.model == installed_selection.contract.model_id
        assert args.truncate_dim == installed_selection.contract.dimension
        return graph, "seed"

    monkeypatch.setattr(build, "_build_strategy_graph", build_graph)
    output = tmp_path / "graph.json"
    parser, build_parser, _, _ = _create_parser()
    args = parser.parse_args(
        [
            "build",
            "seed",
            "--strategy",
            strategy,
            "--cache-namespace",
            installed_selection.namespace_id,
            "--all-corpus",
            "--export",
            "json",
            "--output",
            str(output),
        ]
    )
    assert (
        run_build_command(
            args,
            build_parser,
            provided_build_options=_pop_tracked_option_dests(args),
            user_config=UserConfig(tmp_path / "config.toml"),
        )
        == 0
    )
    payload = json.loads(output.with_suffix(".config.json").read_text())
    embedding = payload["build"]["embedding"]
    assert embedding["model"] == installed_selection.contract.model_id
    assert embedding["truncate_dim"] == installed_selection.contract.dimension
    assert embedding["cache_namespace"] == installed_selection.namespace_id
    assert embedding["dataset_revision"] == "source-pin"
    assert embedding["index_origin"] == origin
    assert embedding["all_corpus"] is True
    assert embedding["installed_corpus_selection"] == installed_scope
    assert embedding.get("installed_corpus_limit") == installed_limit


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
