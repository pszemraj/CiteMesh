"""Rendered cache reports retain identifiers and useful status at terminal widths."""

from __future__ import annotations

import io
import logging
import shlex
from pathlib import Path

import pytest
from rich.console import Console

import citemesh.strategies.embedding.builder as builder_module
from citemesh.cli import cache_ops, console
from citemesh.cli.commands import cache
from citemesh.cli.parser import _create_parser
from citemesh.data import DEFAULT_EMBEDDING_MODEL_NAME
from citemesh.data.user_config import load_user_config, user_config_path
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.embedding.config import DEFAULT_DATASET_SOURCE


@pytest.mark.parametrize("width", [60, 80, 120])
def test_cache_namespace_report_preserves_source_and_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    """Long provenance strings remain readable rather than ellipsized in a cell."""
    output = io.StringIO()
    monkeypatch.setattr(
        console, "output_console", Console(file=output, width=width, color_system=None)
    )
    monkeypatch.setattr(cache_ops, "get_cache_dir", lambda **kwargs: tmp_path)
    revision = "abcdef0123456789abcdef0123456789abcdef0123"
    origin = "user/arxiv-embeddinggemma[archive]"
    monkeypatch.setattr(
        cache_ops,
        "_scan_embedding_namespaces",
        lambda _path: [
            cache_ops._EmbeddingNamespaceScan(
                namespace_id="0123456789ab",
                file_count=4,
                size_bytes=12 * 1024**3,
                status=(
                    "768d int8; active 3,156,800 / physical 3,156,800; "
                    f"source {revision}; coverage full; origin {origin}; ready"
                ),
            )
        ],
    )

    assert cache_ops._scan_cache_directory() == 0

    rendered = output.getvalue()
    assert "0123456789ab" in rendered
    assert "4 files" in rendered
    assert "12.0 GiB" in rendered
    assert "3,156,800" in rendered
    assert revision in rendered
    assert origin in rendered
    assert "…" not in rendered
    assert all(len(line) <= width for line in rendered.splitlines())


@pytest.mark.parametrize("automatic", [True, False], ids=["selected", "explicit"])
def test_installed_corpus_hint_keeps_command_copyable(
    monkeypatch: pytest.MonkeyPatch, automatic: bool
) -> None:
    """The command occupies its own unmodified line even in a narrow terminal."""
    output = io.StringIO()
    monkeypatch.setattr(
        console, "output_console", Console(file=output, width=60, color_system=None)
    )

    cache._print_installed_namespace("0123456789ab", automatic=automatic)

    selector = [] if automatic else ["--cache-namespace", "0123456789ab"]
    lines = output.getvalue().splitlines()
    assert [shlex.split(line) for line in lines if line.startswith("citemesh ")] == [
        [
            "citemesh",
            "build",
            "arxiv:2404.08801",
            "--strategy",
            "hybrid",
            "--semantic-source",
            "both",
            *selector,
            "--export",
            "dashboard",
        ],
        [
            "citemesh",
            "search",
            "long-context language models",
            "--semantic-source",
            "arxiv-corpus",
            *selector,
        ],
        ["citemesh", "config", "set", "defaults.semantic_source", "both"],
    ]
    text = " ".join(output.getvalue().split())
    assert ("Your settings select this index automatically." in text) is automatic
    assert (
        "Your settings do not select this index; add --cache-namespace "
        "0123456789ab to the commands below." in text
    ) is not automatic
    assert ("--cache-namespace" in text) is not automatic


class _InstalledView:
    """Corpus view check of an installed corpus that covers ``limit`` papers."""

    def __init__(self, limit: int | None) -> None:
        self.limit = limit

    def require_complete_certified_corpus(self, *, corpus_size: int | None) -> None:
        if self.limit is not None and corpus_size is not None:
            if corpus_size > self.limit:
                raise ValueError("Requested corpus view exceeds installed coverage")


def _installed_corpus(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    truncate_dim: int | None = None,
    source_repo_id: str = DEFAULT_DATASET_SOURCE,
    limit: int | None = None,
) -> str:
    """Serve one installed corpus that these builder settings encoded.

    :return str: Native namespace ID of the corpus, recorded at artifact ``artifact``.
    """
    from citemesh.data.embedding_contract import (
        EmbeddingContract,
        ExistingCacheSelection,
        embedding_namespace_id,
    )

    producer = EmbeddingGraphBuilder(
        model_name=model_name, truncate_dim=truncate_dim, semantic_source="arxiv-corpus"
    )
    namespace_id = embedding_namespace_id(
        producer._embedding_cache_namespace(artifact_identity="artifact")
    )
    selection = ExistingCacheSelection(
        namespace_id=namespace_id,
        namespace_serialization="address",
        contract=EmbeddingContract(
            model_id=model_name,
            model_revision="revision",
            model_artifact_identity="artifact",
            profile_key="embeddinggemma",
            profile_schema="embeddinggemma-v2",
            representation="retrieval-document-v1",
            formatter_identity="formatter",
            normalization="l2-v1",
            dimension=producer.truncate_dim,
            stored_dtype="int8",
            quantization="affine-int8-v1",
            calibration_sample_size=2000,
        ),
        source_repo_id=source_repo_id,
        source_config_name=None,
        source_split="train",
    )

    def select(requested: str) -> ExistingCacheSelection:
        assert requested == namespace_id
        return selection

    monkeypatch.setattr(builder_module, "select_existing_cache", select)
    monkeypatch.setattr(
        builder_module, "open_existing_cache", lambda _: _InstalledView(limit)
    )
    return namespace_id


def _settings_select(namespace_id: str, config_toml: str) -> bool:
    """Decide the import report from a config.toml in the isolated cache root."""
    config_path = user_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_toml, encoding="utf-8")
    _, build_parser, _, _ = _create_parser()
    return cache._settings_select(namespace_id, build_parser, load_user_config())


@pytest.mark.parametrize(
    ("corpus", "config_toml", "selected"),
    [
        pytest.param({}, "", True, id="defaults"),
        pytest.param(
            {}, '[defaults]\nsemantic_source = "both"\n', True, id="saved-source"
        ),
        pytest.param({}, "[defaults]\ntruncate_dim = 256\n", False, id="config-dim"),
        pytest.param(
            {},
            '[defaults]\nmodel = "google/embeddinggemma-300m"\n',
            False,
            id="config-model",
        ),
        pytest.param({"truncate_dim": 256}, "", False, id="corpus-dim"),
        pytest.param(
            {"truncate_dim": 256},
            "[defaults]\ntruncate_dim = 256\n",
            True,
            id="corpus-dim-configured",
        ),
        pytest.param({"source_repo_id": "publisher/source"}, "", False, id="source"),
        pytest.param(
            {"source_repo_id": "publisher/source"},
            '[defaults]\ndataset_source = "publisher/source"\n',
            True,
            id="source-configured",
        ),
        pytest.param(
            {"model_name": "google/embeddinggemma-300m"},
            "",
            False,
            id="fallback-model",
        ),
        pytest.param(
            {"limit": 5}, "[defaults]\ncorpus_size = 5\n", True, id="view-within"
        ),
        pytest.param(
            {"limit": 5}, "[defaults]\ncorpus_size = 10\n", False, id="view-beyond"
        ),
    ],
)
def test_saved_settings_decide_whether_import_selects_the_corpus(
    monkeypatch: pytest.MonkeyPatch,
    corpus: dict[str, str | int],
    config_toml: str,
    selected: bool,
) -> None:
    """Config defaults that address a namespace flip the import report either way.

    The requested checkpoint addresses the corpus; the fallback checkpoint that
    loads only when it fails does not, and a configured view must fit coverage.
    """
    namespace_id = _installed_corpus(monkeypatch, **corpus)

    assert _settings_select(namespace_id, config_toml) is selected


@pytest.mark.parametrize(
    "config_toml",
    [
        pytest.param("[defaults]\ntruncate_dim = 100\n", id="builder-option"),
        pytest.param(
            '[defaults]\nstreaming = true\ndataset_split = "train[:5%]"\n',
            id="build-contract",
        ),
    ],
)
def test_invalid_saved_settings_report_the_explicit_route(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    config_toml: str,
) -> None:
    """A config.toml no corpus build accepts yields the namespace route, not an error."""
    namespace_id = _installed_corpus(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="citemesh")

    assert not _settings_select(namespace_id, config_toml)
    assert "Current settings cannot build from a corpus" in caplog.text


def test_saved_settings_need_the_embeddings_extra_to_select_a_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the extra the address cannot be derived, so the explicit route is shown."""
    from citemesh.strategies.embedding import deps

    def missing(*, require_corpus: bool = True) -> None:
        raise ImportError("Embedding strategy requires: torch")

    monkeypatch.setattr(deps, "_check_embedding_deps", missing)

    assert not _settings_select("0123456789ab", "")
