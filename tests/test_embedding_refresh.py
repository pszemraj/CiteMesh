"""Pinned source reconciliation and installed-corpus consumer contracts."""

import json
import runpy
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import h5py
import numpy as np
import pytest
from huggingface_hub.utils import httpx

from citemesh.data import get_cache_dir
from citemesh.data.embedding_cache import select_existing_cache
from citemesh.data.embedding_contract import CorpusSource, InstalledCorpusError
from citemesh.strategies.embedding import builder as builder_module
from citemesh.strategies.embedding import corpus_source, deps
from citemesh.strategies.embedding.builder import EmbeddingGraphBuilder
from citemesh.strategies.embedding.records import DatasetRecord
from citemesh.strategies.embedding.refresh import (
    adopt_legacy_cache,
    refresh_cache_selection,
)


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


def make_builder(*, storage_precision="float32", **kwargs):
    return EmbeddingGraphBuilder(
        model_name="org/example",
        model_revision="a" * 40,
        truncate_dim=2,
        storage_precision=storage_precision,
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


@pytest.mark.parametrize("reuse_builder", [False, True])
def test_initial_source_lookup_failure_can_be_retried(
    monkeypatch, runtime, reuse_builder
):
    reads, resolutions = install_source(
        monkeypatch, {"one": [row("2401.00001")]}, ["one"]
    )
    resolve = corpus_source.resolve_source_snapshot

    def unavailable(*args, **kwargs):
        raise OSError("source lookup unavailable")

    monkeypatch.setattr(corpus_source, "resolve_source_snapshot", unavailable)
    builder = make_builder()
    with pytest.raises(OSError, match="source lookup unavailable"):
        builder.refresh_corpus()
    directory = get_cache_dir("embeddings", create=False)
    assert not list(directory.glob("metadata_*.db"))
    assert not list(directory.glob("embeddings_*.h5"))
    assert runtime == ([], [])

    monkeypatch.setattr(corpus_source, "resolve_source_snapshot", resolve)
    retry = builder if reuse_builder else make_builder()
    assert retry.refresh_corpus()["active_rows"] == 1
    assert reads == ["one"]
    assert resolutions == ["main"]
    assert retry.embedding_cache.corpus_state()["membership_complete"]


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


@pytest.mark.parametrize("prefetch", [False, True])
def test_refresh_preserves_encoder_batching_capabilities(
    monkeypatch, runtime, caplog, prefetch
):
    install_source(
        monkeypatch,
        {"one": [row("2401.00001"), row("2401.00002"), row("2401.00003")]},
        ["one"],
    )
    builder = make_builder(encode_batch_size=1)
    builder._load_model()
    model = builder.model
    calls = []

    def encode_prefetched(texts, *, batch_size):
        calls.append((list(texts), batch_size))
        return model.encode(texts)

    model.prefetch_batches = prefetch
    model.encode_prefetched = encode_prefetched
    model.max_seq_length = 2
    model.default_prompt_name = "document"
    model.prompts = {"document": "prefix "}
    model.tokenizer = lambda texts, **kwargs: {"length": [3 for _ in texts]}
    result = builder.refresh_corpus()
    assert result["encoded_rows"] == 3
    assert len(runtime[0]) == 3
    if prefetch:
        assert len(calls) == 1
        assert calls[0][1] == 1
    else:
        assert not calls
        assert "truncate 3 of 3 inputs" in caplog.text


@pytest.mark.parametrize(
    "source_error", [None, httpx.ReadError, httpx.RemoteProtocolError]
)
def test_interrupted_refresh_resumes_recorded_target(
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[list[str], list[str]],
    source_error: type[httpx.HTTPError] | None,
) -> None:
    """Keep interrupted refreshes pending and resume their recorded source target."""
    revisions: dict[str, Iterable[DatasetRecord]] = {
        "one": [row("2401.00001")],
        "two": [row("2401.00001", "new"), row("2401.00002")],
        "three": [row("2401.00003")],
    }
    active = ["one"]
    reads, resolutions = install_source(monkeypatch, revisions, active)
    builder = make_builder()
    builder.refresh_corpus()
    cache = builder.embedding_cache
    completed_state = cache.corpus_state()
    assert completed_state["membership_complete"]
    encoded_before = len(runtime[0])
    with h5py.File(cache.h5_path, "r") as handle:
        vectors_before = handle["embeddings"][:]
    original = cache.upsert_embeddings
    active[0] = "two"

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("interrupted after vector commit")

    if source_error is None:
        monkeypatch.setattr(cache, "upsert_embeddings", interrupt)
    else:
        failure = source_error("interrupted source stream")

        def broken_source() -> Iterator[DatasetRecord]:
            """Read one changed source row before the transport fails."""
            yield row("2401.00001", "new")
            raise failure

        revisions["two"] = broken_source()
    expected_error = RuntimeError if source_error is None else OSError
    with pytest.raises(expected_error, match="interrupted"):
        builder.refresh_corpus()
    state = cache.corpus_state()
    assert state["source"] == completed_state["source"]
    assert state["contract"] == completed_state["contract"]
    assert (
        state["namespace_serialization"] == completed_state["namespace_serialization"]
    )
    assert state["pending"]["target_source"]["revision"] == "two"
    assert not state["membership_complete"]
    if source_error is not None:
        assert len(runtime[0]) == encoded_before
        assert cache.embedding_count() == len(vectors_before)
        with h5py.File(cache.h5_path, "r") as handle:
            np.testing.assert_array_equal(handle["embeddings"][:], vectors_before)
    with pytest.raises(RuntimeError, match="incomplete"):
        builder.embedding_cache.search(
            np.asarray([1, 0], dtype=np.float32),
            2,
            binary_prefilter=False,
            binary_rescore_multiplier=1,
        )
    active[0] = "three"
    revisions["two"] = [row("2401.00001", "new"), row("2401.00002")]
    monkeypatch.setattr(cache, "upsert_embeddings", original)
    reads.clear()
    resolutions.clear()
    builder.refresh_corpus()
    assert reads == ["two"]
    assert not resolutions
    assert cache.corpus_state()["source"]["revision"] == "two"
    assert cache.corpus_state()["membership_complete"]
    assert "pending" not in cache.corpus_state()
    assert cache.get_cached_paper_ids() == {"arxiv:2401.00001", "arxiv:2401.00002"}


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("installed_limit", [None, 2])
@pytest.mark.parametrize("view", [None, 1])
def test_installed_consumers_preserve_binding_and_smaller_view(
    monkeypatch, runtime, explicit, installed_limit, view
):
    revisions = {"one": [row("2301.00001"), row("2401.00001"), row("2501.00001")]}
    reads, _ = install_source(monkeypatch, revisions, ["one"])
    monkeypatch.setattr(
        EmbeddingGraphBuilder,
        "_resolve_model_fingerprint",
        lambda self: "hf::org/example::" + "a" * 40,
    )
    original = make_builder(corpus_size=installed_limit)
    original.model_revision = None
    original.refresh_corpus()
    selection = select_existing_cache(original._cache_selection.namespace_id)
    if explicit:
        builder = EmbeddingGraphBuilder(cache_selection=selection, corpus_size=view)
    else:
        builder = make_builder(corpus_size=view)
        builder.model_revision = None
        assert builder._cache_selection is None
    reads.clear()
    builder._prepare_corpus_for_build()
    builder._embedding_cache = None
    results = builder.search_local("query", 10)
    expected = {"arxiv:2501.00001"}
    if view is None:
        expected.add("arxiv:2401.00001")
        if installed_limit is None:
            expected.add("arxiv:2301.00001")
    assert {result.paper_id for result in results} == expected
    references = builder.cached_reference_metadata(
        ["arxiv:2301.00001", "arxiv:2401.00001", "arxiv:2501.00001"]
    )
    assert set(references) == expected
    assert not reads
    assert builder.model_revision == "a" * 40
    assert builder._cache_selection == selection
    assert builder.embedding_cache.db_path == original.embedding_cache.db_path
    assert builder.graph_embedding_cache.db_path != builder.embedding_cache.db_path
    metadata = builder._embedding_runtime_metadata()
    assert metadata["dataset_revision"] == "one"
    assert metadata["cache_namespace"] == selection.namespace_id
    assert "index_origin" in metadata
    assert "revision=main::" in selection.namespace_serialization


@pytest.mark.parametrize(
    "mismatch", ["source", "split", "coverage", "pending", "uncertified"]
)
def test_implicit_installed_cache_rejects_incompatible_corpus_view(
    monkeypatch, runtime, mismatch
):
    reads, _ = install_source(monkeypatch, {"one": [row("2401.00001")]}, ["one"])
    original = make_builder(corpus_size=1)
    original.refresh_corpus()
    if mismatch in {"pending", "uncertified"}:
        state = original.embedding_cache.corpus_state()
        state["membership_complete"] = False
        if mismatch == "pending":
            state["pending"] = {"kind": "refresh"}
        original.embedding_cache._write_cache_metadata(
            {"corpus_state": json.dumps(state)}
        )
    options = {
        "source": {"dataset_source": "other/source"},
        "split": {"dataset_split": "other"},
        "coverage": {"corpus_size": 2},
        "pending": {},
        "uncertified": {},
    }[mismatch]
    builder = make_builder(**options)
    reads.clear()
    with pytest.raises(
        InstalledCorpusError, match="source|coverage|incomplete|not certified"
    ):
        builder.prepare_embedding_cache()
    assert not reads


def test_refresh_refuses_missing_matrix_of_open_selected_cache(monkeypatch, runtime):
    active = ["one"]
    install_source(
        monkeypatch,
        {"one": [row("2401.00001")], "two": [row("2401.00002")]},
        active,
    )
    original = make_builder()
    original.refresh_corpus()
    selection = select_existing_cache(original.embedding_cache.namespace_id)
    builder = EmbeddingGraphBuilder(cache_selection=selection)
    cache = builder.embedding_cache
    cache.h5_path.unlink()
    runtime[0].clear()
    active[0] = "two"

    with pytest.raises(RuntimeError, match="both SQLite and HDF5"):
        builder.refresh_corpus()
    assert not cache.h5_path.exists()
    assert not runtime[0]


@pytest.mark.parametrize("selection", ["all", "newest"])
def test_source_selection_identical_duplicates_keep_first_ordinal(
    monkeypatch, selection
):
    revisions = {
        "one": [
            row("2501.00001"),
            row("2501.00001"),
            row("2401.00001"),
            row("2301.00001"),
        ]
    }
    install_source(monkeypatch, revisions, ["one"])
    source = CorpusSource(
        "org/source",
        None,
        "train",
        "one",
        selection,
        2 if selection == "newest" else None,
    )
    stats = {}
    records = list(corpus_source.iter_source_records(source, stats))
    expected = ["arxiv:2501.00001", "arxiv:2401.00001"]
    ordinals = [0, 2]
    if selection == "all":
        expected.append("arxiv:2301.00001")
        ordinals.append(3)
    assert [record["paper_id"] for record in records] == expected
    assert [record["source_ordinal"] for record in records] == ordinals
    assert stats["source_records_examined"] == 4


@pytest.mark.parametrize("missing_chronology", [False, True])
def test_explicit_legacy_adoption_retains_requested_main_address(
    monkeypatch, runtime, missing_chronology
):
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
    if missing_chronology:
        with cache._connect_db() as connection:
            connection.execute("DROP INDEX idx_papers_chronology_key")
            connection.execute("ALTER TABLE papers DROP COLUMN chronology_key")
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
    assert builder.embedding_cache.get_max_chronology_key() is not None


@pytest.mark.parametrize("stored_size", [None, 2])
def test_legacy_adoption_rejects_conflicting_explicit_coverage(
    monkeypatch, runtime, stored_size
):
    builder = make_builder(corpus_size=1)
    cache = builder.embedding_cache
    cache.set_model_fingerprint("hf::org/example::" + "a" * 40)
    cache.mark_hydrated(
        builder.dataset_source, builder.dataset_split, stored_size, complete=True
    )
    with pytest.raises(ValueError, match="coverage differs"):
        adopt_legacy_cache(builder, cache.namespace_id)
    assert cache.corpus_state() == {}
    assert runtime == ([], [])
    assert cache.payload_stats().hydration_corpus_size == (
        "all" if stored_size is None else "newest:2"
    )


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


def test_conflicting_duplicates_refresh_one_deterministic_winner(monkeypatch, runtime):
    install_source(
        monkeypatch,
        {"one": [row("2401.00001"), row("2401.00001", "conflicting")]},
        ["one"],
    )
    builder = make_builder()
    result = builder.refresh_corpus()
    assert result["source_records_examined"] == 2
    assert result["active_rows"] == 1
    assert len(runtime[0]) == 1
    assert builder.embedding_cache.corpus_state()["membership_complete"]
    assert (
        builder.embedding_cache.get_paper_metadata_batch(["arxiv:2401.00001"])[
            "arxiv:2401.00001"
        ]["abstract"]
        == "conflicting"
    )


@pytest.mark.parametrize("selection", ["all", "newest"])
@pytest.mark.parametrize("reverse", [False, True])
def test_dated_duplicate_winner_is_shared_and_precedes_newest_cap(
    monkeypatch, selection, reverse
):
    duplicate = [
        row("2501.00001", "old", update_date="2023-05-25", categories="math-ph"),
        row(
            "2501.00001",
            "new",
            update_date="2026-10-16 00:00:00",
            categories="math-ph quant-ph",
        ),
    ]
    if reverse:
        duplicate.reverse()
    rows = [duplicate[0], row("2401.00001"), duplicate[1], row("2301.00001")]
    install_source(monkeypatch, {"one": rows}, ["one"])
    stats = {}
    records = list(
        corpus_source.iter_source_records(
            CorpusSource(
                "org/source",
                None,
                "train",
                "one",
                selection,
                2 if selection == "newest" else None,
            ),
            stats,
        )
    )
    by_id = {record["paper_id"]: record for record in records}
    winner = by_id["arxiv:2501.00001"]
    assert winner["abstract"] == "new"
    assert winner["categories"] == ["math-ph", "quant-ph"]
    assert winner["source_ordinal"] == (0 if reverse else 2)
    assert len(records) == (2 if selection == "newest" else 3)
    assert "arxiv:2401.00001" in by_id
    assert stats["source_records_examined"] == 4


@pytest.mark.parametrize(
    ("older", "newer"),
    [
        ({}, {"update_date": "2026-10-16"}),
        ({"update_date": "invalid"}, {"update_date": "2026-10-16"}),
        (
            {"update_date": "2026-10-16T00:30:00+02:00"},
            {"update_date": "Thu, 15 Oct 2026 23:00:00 GMT"},
        ),
        (
            {"versions": [{"version": "v9"}]},
            {"versions": [{"version": "v10"}, {"version": "v2"}]},
        ),
        (
            {"update_date": "2023-05-25", "versions": [{"version": "v10"}]},
            {"update_date": "2026-10-16", "versions": [{"version": "v9"}]},
        ),
        (
            {"update_date": "invalid", "versions": [{"version": "v9"}]},
            {"versions": [{"version": "v10"}]},
        ),
        (
            {"update_date": "2026-10-16", "versions": [{"version": "v9"}]},
            {"update_date": "2026-10-16", "versions": [{"version": "v10"}]},
        ),
        (
            {"versions": [{"version": "v2", "created": "2023-05-24"}]},
            {
                "versions": [
                    {"version": "v2", "created": "Thu, 25 May 2023 00:00:00 GMT"}
                ]
            },
        ),
        ({"id": "2401.00001v9"}, {"id": "2401.00001v10"}),
    ],
)
def test_source_duplicate_recency_fallbacks(monkeypatch, older, newer):
    for records in (
        [row("2401.00001", "z old", **older), row("2401.00001", "a new", **newer)],
        [row("2401.00001", "a new", **newer), row("2401.00001", "z old", **older)],
    ):
        install_source(monkeypatch, {"one": records}, ["one"])
        source = CorpusSource("org/source", None, "train", "one", "all", None)
        resolved = list(corpus_source.iter_source_records(source))
        assert len(resolved) == 1
        assert resolved[0]["paper_id"] == "arxiv:2401.00001"
        assert resolved[0]["abstract"] == "a new"


@pytest.mark.parametrize("update_date", [None, "2026-10-16", "invalid"])
def test_ambiguous_duplicates_resolve_stably_with_aggregate_warning(
    monkeypatch, caplog, update_date
):
    records = [
        row(identifier, abstract, update_date=update_date)
        for identifier in ("2401.00001", "2401.00002")
        for abstract in ("alpha", "zulu")
    ]
    winners = []
    for rows in (records, list(reversed(records))):
        install_source(monkeypatch, {"one": rows}, ["one"])
        source = CorpusSource("org/source", None, "train", "one", "all", None)
        resolved = list(corpus_source.iter_source_records(source))
        winners.append({record["paper_id"]: record["abstract"] for record in resolved})
        warnings = [
            record for record in caplog.records if record.levelname == "WARNING"
        ]
        assert len(warnings) == 1
        assert "2 conflicting source rows" in warnings[0].getMessage()
        caplog.clear()
    assert winners[0] == winners[1]


def test_interrupted_old_duplicate_membership_reconciles_winner(monkeypatch, runtime):
    initial = [row("2401.00001", "old"), row("2401.00002", "unchanged")]
    sources = {
        "one": initial,
        "two": [*initial, row("2401.00001", "new", update_date="2026-10-16")],
    }
    active = ["one"]
    install_source(monkeypatch, sources, active)
    builder = EmbeddingGraphBuilder(
        model_name="org/example",
        model_revision="a" * 40,
        truncate_dim=2,
        storage_precision="int8",
        semantic_source="arxiv-corpus",
        calibration_sample_size=2,
    )
    builder.refresh_corpus()
    cache = builder.embedding_cache
    with h5py.File(cache.h5_path, "r") as handle:
        vectors = handle["embeddings"][:]
        ranges = handle["calibration_ranges"][:]
    old_records = list(
        corpus_source.iter_source_records(
            CorpusSource("org/source", None, "train", "one", "all", None)
        )
    )
    normal_batches = corpus_source.iter_source_batches

    def interrupted_old_stream(*args):
        yield old_records
        raise RuntimeError("old streaming resolver stopped at duplicate")

    active[0] = "two"
    monkeypatch.setattr(corpus_source, "iter_source_batches", interrupted_old_stream)
    with pytest.raises(RuntimeError, match="old streaming resolver"):
        builder.refresh_corpus()
    assert cache.corpus_state()["pending"]["target_source"]["revision"] == "two"
    assert cache.corpus_state()["membership_complete"] is False
    monkeypatch.setattr(corpus_source, "iter_source_batches", normal_batches)
    runtime[0].clear()
    result = builder.refresh_corpus()
    assert result["encoded_rows"] == 1
    assert len(runtime[0]) == 1 and "new" in runtime[0][0]
    assert (
        cache.get_paper_metadata_batch(["arxiv:2401.00001"])["arxiv:2401.00001"][
            "abstract"
        ]
        == "new"
    )
    with cache._inspect_db() as connection:
        assert dict(
            connection.execute("SELECT paper_id,source_ordinal FROM corpus_membership")
        ) == {
            "arxiv:2401.00001": 2,
            "arxiv:2401.00002": 1,
        }
    with h5py.File(cache.h5_path, "r") as handle:
        np.testing.assert_array_equal(handle["embeddings"][1], vectors[1])
        np.testing.assert_array_equal(handle["calibration_ranges"][:], ranges)
    assert cache.corpus_state()["membership_complete"]


@pytest.mark.parametrize("source_error", [RuntimeError, httpx.ReadTimeout])
@pytest.mark.parametrize("storage_precision", ["float32", "int8"])
@pytest.mark.parametrize("missing_h5", [False, True])
def test_source_failure_during_duplicate_staging_can_be_resumed(
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[list[str], list[str]],
    storage_precision: Literal["float32", "int8"],
    missing_h5: bool,
    source_error: type[RuntimeError] | type[httpx.HTTPError],
) -> None:
    """Resume the pinned source after staging fails before writing any vectors."""

    def broken_source() -> Iterator[DatasetRecord]:
        """Read one source row before the upstream iterator fails."""
        yield row("2401.00001")
        raise source_error("source stream interrupted")

    sources = {"one": broken_source(), "two": [row("2501.00001")]}
    active = ["one"]
    reads, resolutions = install_source(monkeypatch, sources, active)
    builder = make_builder(storage_precision=storage_precision)
    expected_error = (
        OSError if issubclass(source_error, httpx.HTTPError) else RuntimeError
    )
    with pytest.raises(expected_error, match="source stream interrupted"):
        builder.refresh_corpus()
    assert runtime == ([], [])
    cache = builder.embedding_cache
    assert cache.embedding_count() == 0
    assert not cache.corpus_state()["membership_complete"]
    assert cache.corpus_state()["pending"]["kind"] == "refresh"
    with h5py.File(cache.h5_path, "r") as handle:
        assert "embeddings" not in handle
        assert "calibration_ranges" not in handle
    if missing_h5:
        cache.h5_path.unlink()

    sources["one"] = [row("2401.00001"), row("2401.00002")]
    active[0] = "two"
    reads.clear()
    resolutions.clear()
    with pytest.raises(RuntimeError, match="incomplete refresh"):
        select_existing_cache(cache.namespace_id)
    selection = select_existing_cache(cache.namespace_id, require_ready=False)
    assert refresh_cache_selection(selection)["active_rows"] == 2
    assert set(reads) == {"one"}
    assert not resolutions
    assert cache.corpus_state()["source"]["revision"] == "one"
    assert cache.corpus_state()["membership_complete"]


def test_manual_publisher_resumes_incomplete_selected_namespace(
    monkeypatch, runtime, tmp_path
):
    from citemesh.data.embedding_cache import portable
    from citemesh.services import embedding_indexes

    active = ["one"]
    install_source(
        monkeypatch,
        {"one": [row("2401.00001")], "two": [row("2401.00001", "new")]},
        active,
    )
    builder = make_builder()
    builder.refresh_corpus()
    cache = builder.embedding_cache
    original = cache.upsert_embeddings

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("interrupted before membership")

    active[0] = "two"
    monkeypatch.setattr(cache, "upsert_embeddings", interrupt)
    with pytest.raises(RuntimeError, match="interrupted before membership"):
        builder.refresh_corpus()
    exports = []
    monkeypatch.setattr(
        portable,
        "export_snapshot",
        lambda cache, *args, **kwargs: exports.append(
            cache.require_complete_certified_corpus()
        ),
    )
    monkeypatch.setattr(
        embedding_indexes, "publish_snapshot", lambda *args: "published"
    )
    publish = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/publish_arxiv_index.py")
    )["publish_arxiv_index"]
    assert (
        publish(
            "owner/private-index",
            tmp_path / "export",
            config_name="example",
            namespace=cache.namespace_id,
            device="cpu",
        )
        == "published"
    )
    assert len(exports) == 1
    assert exports[0]["source"]["revision"] == "two"


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


@pytest.mark.parametrize("failure_stage", ["load", "iterate"])
@pytest.mark.parametrize(
    "source_error",
    [httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError],
)
def test_source_transport_failures_are_operation_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: Literal["load", "iterate"],
    source_error: type[httpx.HTTPError],
) -> None:
    """Normalize transport errors from opening and consuming the pinned dataset."""
    failure = source_error("source transport unavailable")

    def broken_source() -> Iterator[DatasetRecord]:
        """Read one source row before the upstream iterator fails."""
        yield row("2401.00001")
        raise failure

    def load_dataset(repo: str, **kwargs: str | bool) -> Iterator[DatasetRecord]:
        """Fail at dataset construction or return the interrupted source stream."""
        if failure_stage == "load":
            raise failure
        return broken_source()

    monkeypatch.setattr(
        deps,
        "_import_datasets_module",
        lambda: SimpleNamespace(load_dataset=load_dataset),
    )
    source = CorpusSource("org/source", None, "train", "pinned-revision", "all", None)
    stats: corpus_source.SourceScanStats = {"source_records_examined": 0}

    with pytest.raises(OSError, match="source transport unavailable") as captured:
        list(corpus_source.iter_source_records(source, stats))

    assert "org/source@pinned-revision" in str(captured.value)
    assert captured.value.__cause__ is failure
    assert stats["source_records_examined"] == (1 if failure_stage == "iterate" else 0)


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "b" * 40),
        ("_active_model_name", "org/other"),
        ("truncate_dim", 1),
        ("calibration_sample_size", 3),
    ],
)
def test_selected_runtime_rejects_contract_mismatch(monkeypatch, runtime, field, value):
    install_source(monkeypatch, {"one": [row("2401.00001")]}, ["one"])
    original = make_builder(storage_precision="int8", calibration_sample_size=2)
    original.refresh_corpus()
    selection = select_existing_cache(original.embedding_cache.namespace_id)
    builder = EmbeddingGraphBuilder(cache_selection=selection)
    builder._resolved_model_fingerprint = selection.contract.model_artifact_identity
    setattr(builder, field, value)
    with pytest.raises(InstalledCorpusError, match="runtime mismatch"):
        builder._require_selected_runtime_match()


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
    reset.prepare_embedding_cache()
    assert reset._cache_selection is None
    assert reset.embedding_cache.db_path == cache.db_path
    assert reset.embedding_cache.corpus_state() == {}
    assert reset.embedding_cache.embedding_count() == 0
