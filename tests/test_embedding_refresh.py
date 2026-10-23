"""Pinned source reconciliation and installed-corpus consumer contracts."""

from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from citemesh.data.embedding_cache import select_existing_cache
from citemesh.data.embedding_contract import CorpusSource
from citemesh.strategies.embedding import builder as builder_module
from citemesh.strategies.embedding import corpus_source, deps
from citemesh.strategies.embedding.builder import EmbeddingGraphBuilder
from citemesh.strategies.embedding.refresh import adopt_legacy_cache


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(deps, "_check_embedding_deps", lambda **kwargs: None)
    monkeypatch.setattr(builder_module, "resolve_embedding_device", lambda _: "cpu")
    monkeypatch.setattr(
        EmbeddingGraphBuilder, "_resolve_source_dtype_hint", lambda _: "float32"
    )
    encoded = []
    loads = []

    class Encoder:
        def encode(self, texts, **kwargs):
            encoded.extend(texts)
            return np.asarray(
                [[1.0, (sum(text.encode()) % 29) / 29] for text in texts],
                dtype=np.float32,
            )

    def load(builder):
        if builder.model is None:
            loads.append(builder.model_revision)
            builder.model = Encoder()
            builder._active_model_name = builder.model_name
            builder._bind_embedding_cache_to_active_model()

    monkeypatch.setattr(EmbeddingGraphBuilder, "_load_model", load)
    return encoded, loads


def make_builder(**kwargs):
    return EmbeddingGraphBuilder(
        model_name="org/example",
        model_revision="a" * 40,
        truncate_dim=2,
        storage_precision="float32",
        semantic_source="arxiv-corpus",
        **kwargs,
    )


def install_source(monkeypatch, revisions, active):
    reads = []
    resolutions = []

    def load_dataset(repo, **kwargs):
        reads.append(kwargs["revision"])
        return iter(revisions[kwargs["revision"]])

    def resolve(source, revision="main"):
        resolutions.append(revision)
        return replace(source, revision=active[0] if revision == "main" else revision)

    monkeypatch.setattr(
        deps,
        "_import_datasets_module",
        lambda: SimpleNamespace(load_dataset=load_dataset),
    )
    monkeypatch.setattr(corpus_source, "resolve_source_snapshot", resolve)
    return reads, resolutions


def row(identifier, abstract="body", **metadata):
    return {"id": identifier, "title": identifier, "abstract": abstract, **metadata}


def test_revision_refresh_compares_full_records_and_membership(monkeypatch, runtime):
    encoded, loads = runtime
    sources = {
        "one": [
            row("2401.00001"),
            row("2401.00002"),
            row("2401.00003"),
            row("2401.00004"),
        ],
        "two": [
            row("2401.00004"),
            row("2401.00002", venue="updated venue"),
            row("2401.00001", "revised"),
            row("2301.00001"),
        ],
    }
    active = ["one"]
    reads, _ = install_source(monkeypatch, sources, active)
    builder = make_builder()
    builder.refresh_corpus()
    cache = builder.embedding_cache
    namespace = builder._cache_selection.namespace_id
    with h5py.File(cache.h5_path, "r") as handle:
        before = handle["embeddings"][:]
    encoded.clear()
    loads.clear()
    builder.model = None
    active[0] = "two"
    builder.refresh_corpus()
    assert len(encoded) == 2
    assert any("revised" in text for text in encoded)
    assert len(loads) == 1
    assert builder._cache_selection.namespace_id == namespace
    with h5py.File(cache.h5_path, "r") as handle:
        after = handle["embeddings"][:]
    np.testing.assert_array_equal(after[[1, 3]], before[[1, 3]])
    assert len(after) == 5
    assert (
        cache.get_paper_metadata_batch(["arxiv:2401.00002"])["arxiv:2401.00002"][
            "venue"
        ]
        == "updated venue"
    )
    results = cache.search(
        np.asarray([1, 0], dtype=np.float32),
        top_k=10,
        binary_prefilter=False,
        binary_rescore_multiplier=1,
    )
    assert {result.paper_id for result in results} == {
        "arxiv:2401.00001",
        "arxiv:2401.00002",
        "arxiv:2401.00004",
        "arxiv:2301.00001",
    }
    assert reads == ["one", "two"]
    encoded.clear()
    reads.clear()
    builder.refresh_corpus()
    assert not reads and not encoded


def test_changed_source_identical_text_never_loads_encoder(monkeypatch, runtime):
    encoded, loads = runtime
    revisions = {"one": [row("2401.00001")], "two": [row("2401.00001", venue="new")]}
    active = ["one"]
    install_source(monkeypatch, revisions, active)
    builder = make_builder()
    builder.refresh_corpus()
    builder.model = None
    encoded.clear()
    loads.clear()
    active[0] = "two"
    builder.refresh_corpus()
    assert not loads and not encoded


def test_interrupted_refresh_resumes_recorded_target(monkeypatch, runtime):
    revisions = {
        "one": [row("2401.00001")],
        "two": [row("2401.00001", "new"), row("2401.00002")],
        "three": [row("2401.00003")],
    }
    active = ["one"]
    reads, resolutions = install_source(monkeypatch, revisions, active)
    builder = make_builder()
    builder.refresh_corpus()
    original = builder.embedding_cache.upsert_embeddings
    active[0] = "two"

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("interrupted after vector commit")

    monkeypatch.setattr(builder.embedding_cache, "upsert_embeddings", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        builder.refresh_corpus()
    with pytest.raises(RuntimeError, match="incomplete"):
        builder.embedding_cache.search(
            np.asarray([1, 0], dtype=np.float32),
            2,
            binary_prefilter=False,
            binary_rescore_multiplier=1,
        )
    active[0] = "three"
    monkeypatch.setattr(builder.embedding_cache, "upsert_embeddings", original)
    reads.clear()
    resolutions.clear()
    builder.refresh_corpus()
    assert reads == ["two"]
    assert not resolutions
    assert builder.embedding_cache.corpus_state()["source"]["revision"] == "two"


def test_installed_consumers_preserve_binding_and_smaller_view(monkeypatch, runtime):
    revisions = {"one": [row("2301.00001"), row("2401.00001"), row("2501.00001")]}
    reads, _ = install_source(monkeypatch, revisions, ["one"])
    original = make_builder()
    original.refresh_corpus()
    selection = select_existing_cache(original._cache_selection.namespace_id)
    builder = EmbeddingGraphBuilder(cache_selection=selection, corpus_size=1)
    reads.clear()
    builder._prepare_corpus_for_build()
    builder._embedding_cache = None
    results = builder.search_local("query", 10)
    assert [result.paper_id for result in results] == ["arxiv:2501.00001"]
    assert not reads
    assert builder.embedding_cache.db_path == original.embedding_cache.db_path
    assert builder.graph_embedding_cache.db_path != builder.embedding_cache.db_path
    assert builder._embedding_runtime_metadata()["dataset_revision"] == "one"


def test_newest_source_selection_deduplicates_before_cap(monkeypatch):
    revisions = {
        "one": [
            row("2501.00001"),
            row("2501.00001"),
            row("2401.00001"),
            row("2301.00001"),
        ]
    }
    install_source(monkeypatch, revisions, ["one"])
    source = CorpusSource("org/source", None, "train", "one", "newest", 2)
    stats = {}
    records = list(corpus_source.iter_source_records(source, stats))
    assert [record["paper_id"] for record in records] == [
        "arxiv:2501.00001",
        "arxiv:2401.00001",
    ]
    assert [record["source_ordinal"] for record in records] == [0, 2]
    assert stats["source_records_examined"] == 4


def test_explicit_legacy_adoption_retains_requested_main_address(monkeypatch, runtime):
    install_source(monkeypatch, {"one": [row("2401.00001")]}, ["one"])
    builder = make_builder()
    builder.model_revision = None
    monkeypatch.setattr(
        builder, "_resolve_model_fingerprint", lambda: "hf::org/example::" + "a" * 40
    )
    namespace = builder._embedding_cache_namespace(
        artifact_identity=builder._resolve_model_fingerprint()
    )
    cache = builder._create_embedding_cache(namespace)
    cache.set_model_fingerprint(builder._resolve_model_fingerprint())
    builder._load_model()
    cache.upsert_embeddings(
        {
            "arxiv:2401.00001": {
                "paper_id": "arxiv:2401.00001",
                "title": "2401.00001",
                "abstract": "body",
            }
        },
        builder.model,
        text_builder=builder._format_retrieval_document_metadata,
    )
    namespace_id = cache.db_path.stem.removeprefix("metadata_")
    monkeypatch.setattr(
        builder,
        "_resolve_model_fingerprint",
        lambda: pytest.fail("legacy adoption resolved moving main"),
    )
    adopt_legacy_cache(builder, namespace_id)
    assert builder.embedding_cache.db_path == cache.db_path
    assert "revision=main::" in builder._cache_selection.namespace_serialization
    assert builder.model_revision == "a" * 40


def test_managed_reranking_reads_hits_and_encodes_external_papers_in_memory(
    monkeypatch, runtime
):
    from citemesh.core import Paper

    install_source(monkeypatch, {"one": [row("2401.00001")]}, ["one"])
    builder = make_builder()
    builder.refresh_corpus()
    cache = builder.embedding_cache
    before = cache.corpus_state()
    runtime[0].clear()
    monkeypatch.setattr(
        cache,
        "get_embeddings",
        lambda *args, **kwargs: pytest.fail("managed retrieval wrote corpus"),
    )
    embeddings = builder.embed_papers(
        {
            "arxiv:2401.00001": Paper(
                paper_id="arxiv:2401.00001",
                title="2401.00001",
                abstract="body",
                year=2024,
            ),
            "outside": Paper(
                paper_id="outside",
                title="External paper",
                abstract="new text",
                year=2020,
            ),
        }
    )
    assert set(embeddings) == {"arxiv:2401.00001", "outside"}
    assert len(runtime[0]) == 1
    assert cache.corpus_state() == before
    assert cache.embedding_count() == 1


def test_conflicting_duplicates_leave_refresh_incomplete(monkeypatch, runtime):
    from citemesh.strategies.embedding import refresh

    install_source(
        monkeypatch,
        {"one": [row("2401.00001"), row("2401.00001", "conflicting")]},
        ["one"],
    )
    monkeypatch.setattr(refresh, "HYDRATION_FLUSH_SIZE", 1)
    builder = make_builder()
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        builder.refresh_corpus()
    assert (
        builder.embedding_cache.corpus_state()["pending"]["target_source"]["revision"]
        == "one"
    )


def test_first_int8_calibration_resumes_using_persisted_ranges(monkeypatch, runtime):
    install_source(
        monkeypatch, {"one": [row("2401.00001"), row("2401.00002")]}, ["one"]
    )
    builder = EmbeddingGraphBuilder(
        model_name="org/example",
        model_revision="a" * 40,
        truncate_dim=2,
        storage_precision="int8",
        semantic_source="arxiv-corpus",
        calibration_sample_size=2,
    )
    original = builder._initialize_calibration_ranges

    def interrupt(records):
        original(records)
        raise RuntimeError("interrupted after calibration")

    monkeypatch.setattr(builder, "_initialize_calibration_ranges", interrupt)
    with pytest.raises(RuntimeError, match="after calibration"):
        builder.refresh_corpus()
    with h5py.File(builder.embedding_cache.h5_path, "r") as handle:
        before = handle["calibration_ranges"][:]
    monkeypatch.setattr(
        builder,
        "_initialize_calibration_ranges",
        lambda records: pytest.fail("recalibration"),
    )
    builder.refresh_corpus()
    with h5py.File(builder.embedding_cache.h5_path, "r") as handle:
        np.testing.assert_array_equal(handle["calibration_ranges"][:], before)
    assert builder.embedding_cache.corpus_state()["membership_complete"]


def test_sliced_source_uses_nonstreaming_with_same_pinned_revision(monkeypatch):
    calls = []

    def load_dataset(repo, **kwargs):
        calls.append((repo, kwargs))
        return [row("2401.00001")]

    monkeypatch.setattr(
        deps,
        "_import_datasets_module",
        lambda: SimpleNamespace(load_dataset=load_dataset),
    )
    source = CorpusSource(
        "org/source", "metadata", "train[:32]", "pinned-revision", "all", None
    )
    records = list(corpus_source.iter_source_records(source))
    assert records[0]["source_ordinal"] == 0
    assert calls == [
        (
            "org/source",
            {
                "name": "metadata",
                "split": "train[:32]",
                "revision": "pinned-revision",
                "streaming": False,
            },
        )
    ]


def test_pending_refresh_retains_owned_handle_across_transient_model_reset(
    monkeypatch, runtime
):
    install_source(monkeypatch, {"one": [row("2401.00001")]}, ["one"])
    normal_load = EmbeddingGraphBuilder._load_model

    def reset_then_load(builder):
        owned = builder._selected_cache_handle
        builder._embedding_cache = None
        assert builder.embedding_cache is owned
        return normal_load(builder)

    monkeypatch.setattr(EmbeddingGraphBuilder, "_load_model", reset_then_load)
    builder = make_builder()
    builder.refresh_corpus()
    assert builder.embedding_cache.corpus_state()["membership_complete"]


def test_embeddinggemma_contract_reopens_with_explicit_profile_key(
    monkeypatch, runtime
):
    install_source(monkeypatch, {"one": []}, ["one"])
    builder = EmbeddingGraphBuilder(
        model_name="unsloth/embeddinggemma-300m",
        model_revision="a" * 40,
        truncate_dim=512,
        semantic_source="arxiv-corpus",
        storage_precision="float32",
    )
    builder.refresh_corpus()
    selection = select_existing_cache(builder.embedding_cache.namespace_id)
    assert selection.contract.profile_key == "embeddinggemma"
    reopened = EmbeddingGraphBuilder(cache_selection=selection)
    assert reopened.model_profile == builder.model_profile
    reopened._resolved_model_fingerprint = selection.contract.model_artifact_identity
    reopened._require_selected_runtime_match()
    assert (
        builder.embedding_cache.corpus_state()["producer_provenance"][
            "source_torch_dtype"
        ]
        == "float32"
    )


def test_explicit_rebuild_resets_managed_missing_half_before_initialization(
    monkeypatch, runtime
):
    install_source(monkeypatch, {"one": [row("2401.00001")]}, ["one"])
    builder = make_builder()
    builder.refresh_corpus()
    cache = builder.embedding_cache
    cache.h5_path.unlink()
    reset = make_builder(force_rebuild_cache=True)
    reset._resolved_model_fingerprint = (
        builder._cache_selection.contract.model_artifact_identity
    )
    reset._bind_embedding_cache_to_active_model()
    assert reset.embedding_cache.db_path == cache.db_path
    assert reset.embedding_cache.corpus_state() == {}
    assert reset.embedding_cache.embedding_count() == 0
