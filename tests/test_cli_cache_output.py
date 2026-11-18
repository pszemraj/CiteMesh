"""Rendered cache reports retain identifiers and useful status at terminal widths."""

from __future__ import annotations

import io
import shlex
from pathlib import Path

import pytest
from rich.console import Console

from citemesh.cli import cache_ops, console
from citemesh.cli.commands import cache


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
    assert ("Default settings select this index automatically." in text) is automatic
    assert (
        "Default settings do not select this index; add --cache-namespace "
        "0123456789ab to the commands below." in text
    ) is not automatic
    assert ("--cache-namespace" in text) is not automatic


def _installed_corpus(
    monkeypatch: pytest.MonkeyPatch, namespace_id: str, source_repo_id: str
) -> None:
    """Serve one installed corpus whose model artifact is ``artifact``."""
    from citemesh.data.embedding_cache import lifecycle
    from citemesh.data.embedding_contract import (
        EmbeddingContract,
        ExistingCacheSelection,
    )

    selection = ExistingCacheSelection(
        namespace_id=namespace_id,
        namespace_serialization="address",
        contract=EmbeddingContract(
            model_id="unsloth/embeddinggemma-300m",
            model_revision="revision",
            model_artifact_identity="artifact",
            profile_key="embeddinggemma",
            profile_schema="embeddinggemma-v2",
            representation="retrieval-document-v1",
            formatter_identity="formatter",
            normalization="l2-v1",
            dimension=512,
            stored_dtype="int8",
            quantization="affine-int8-v1",
            calibration_sample_size=2000,
        ),
        source_repo_id=source_repo_id,
        source_config_name=None,
        source_split="train",
    )
    monkeypatch.setattr(lifecycle, "select_existing_cache", lambda _: selection)


def test_default_settings_select_a_corpus_by_address_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an installed corpus the default builder would address is selected."""
    from citemesh.data.embedding_contract import embedding_namespace_id
    from citemesh.strategies.embedding import EmbeddingGraphBuilder
    from citemesh.strategies.embedding.config import DEFAULT_DATASET_SOURCE

    default = EmbeddingGraphBuilder(semantic_source="arxiv-corpus")
    address = default._embedding_cache_namespace(artifact_identity="artifact")
    namespace = embedding_namespace_id(address)

    _installed_corpus(monkeypatch, namespace, DEFAULT_DATASET_SOURCE)
    assert cache._default_settings_select(namespace)

    _installed_corpus(monkeypatch, namespace, "publisher/source")
    assert not cache._default_settings_select(namespace)

    _installed_corpus(monkeypatch, "0123456789ab", DEFAULT_DATASET_SOURCE)
    assert not cache._default_settings_select("0123456789ab")


def test_default_settings_need_the_embeddings_extra_to_select_a_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the extra the address cannot be derived, so the explicit route is shown."""
    from citemesh.strategies.embedding import deps

    def missing(*, require_corpus: bool = True) -> None:
        raise ImportError("Embedding strategy requires: torch")

    monkeypatch.setattr(deps, "_check_embedding_deps", missing)

    assert not cache._default_settings_select("0123456789ab")
