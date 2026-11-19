"""Pinned source reconciliation and installed-corpus consumer contracts."""

import json
import runpy
import threading
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Literal

import datasets
import h5py
import numpy as np
import pytest
from huggingface_hub.utils import httpx

from citemesh.data import get_cache_dir
from citemesh.data.embedding_cache import select_existing_cache
from citemesh.data.embedding_cache.lifecycle import CorpusOperationSession
from citemesh.data.embedding_cache.models import CorpusSourceRecord
from citemesh.data.embedding_cache.store import EmbeddingCache
from citemesh.data.embedding_contract import CorpusSource, InstalledCorpusError
from citemesh.strategies.embedding import builder as builder_module
from citemesh.strategies.embedding import corpus_source, deps, refresh
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


def source_dataset(rows: Iterable[DatasetRecord]) -> datasets.Dataset:
    """Download rows as an Arrow dataset, as a non-streaming load prepares them.

    Consuming the rows here surfaces a transport failure from the load call.
    Missing fields become nulls so every row keeps all of its columns.
    """
    records = list(rows)
    columns = dict.fromkeys(field for record in records for field in record)
    return datasets.Dataset.from_list(
        [{field: record.get(field) for field in columns} for record in records]
    )


def install_source(monkeypatch, revisions, active, loads=None):
    """Serve revisions from memory: streamed as rows, downloaded as a Dataset.

    :param loads: Optional list collecting each load's revision and streaming flag.
    """
    reads = []
    resolutions = []

    def load_dataset(repo, **kwargs):
        reads.append(kwargs["revision"])
        if loads is not None:
            loads.append((kwargs["revision"], kwargs["streaming"]))
        rows = revisions[kwargs["revision"]]
        return iter(rows) if kwargs["streaming"] else source_dataset(rows)

    def resolve(source, revision="main"):
        resolutions.append(revision)
        return replace(source, revision=active[0] if revision == "main" else revision)

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    monkeypatch.setattr(corpus_source, "resolve_source_snapshot", resolve)
    return reads, resolutions


def read_source(
    reader: Literal["stream", "download"],
    source: CorpusSource,
    stats: corpus_source.SourceScanStats | None = None,
) -> list[CorpusSourceRecord]:
    """Read the selected records of a source with one of the two readers."""
    if reader == "stream":
        return list(corpus_source.iter_source_records(source, stats))
    with TemporaryDirectory() as workdir:
        selection = corpus_source.prepare_source_selection(source, Path(workdir), stats)
        return [record for batch in selection.batches(2) for record in batch]


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

    callbacks = []

    def encode_prefetched(texts, *, batch_size, on_progress=None):
        calls.append((list(texts), batch_size))
        callbacks.append(on_progress)
        if on_progress is not None:
            on_progress(0)
            for completed in range(1, len(texts) + 1):
                on_progress(completed)
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
        assert len(callbacks) == 1 and callable(callbacks[0])
        completed = []
        callback = completed.append
        lazy_encoder = refresh._LazyEncoder(builder)
        lazy_encoder.encode_prefetched(
            ["one", "two"], batch_size=1, on_progress=callback
        )
        assert callbacks[-1] is callback
        assert completed == [0, 1, 2]
        assert lazy_encoder.encoded_count == 2
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
    with pytest.raises(
        ValueError, match="incomplete refresh is pending at two"
    ) as advance:
        builder.refresh_corpus("three")
    assert (
        f"resume it with `citemesh cache refresh --namespace {cache.namespace_id}`"
        in str(advance.value)
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


@pytest.mark.parametrize("reader", ["stream", "download"])
@pytest.mark.parametrize("selection", ["all", "newest"])
def test_source_selection_identical_duplicates_keep_first_ordinal(
    monkeypatch, selection, reader
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
    records = read_source(reader, source, stats)
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
    loads = []
    install_source(monkeypatch, {"one": [row("2401.00001")]}, ["one"], loads)
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
    assert loads == [("one", False)]
    assert builder.embedding_cache.db_path == cache.db_path
    assert "revision=main::" in builder._cache_selection.namespace_serialization
    assert builder.model_revision == "a" * 40
    assert builder.embedding_cache.get_max_chronology_key() is not None


@pytest.mark.parametrize("explicit_revision", [None, "two"])
def test_adoption_defaults_to_the_pinned_hydration_revision(
    monkeypatch, runtime, explicit_revision
):
    loads = []
    sources = {
        "one": [row("2401.00001")],
        "two": [row("2401.00001"), row("2401.00002")],
    }
    # ``main`` has moved past both; nothing may resolve or read it.
    reads, resolutions = install_source(monkeypatch, sources, ["three"], loads)
    builder = make_builder()
    builder.model_revision = None
    fingerprint = "hf::org/example::" + "a" * 40
    monkeypatch.setattr(builder, "_resolve_model_fingerprint", lambda: fingerprint)
    namespace = builder._embedding_cache_namespace(artifact_identity=fingerprint)
    cache = builder._create_embedding_cache(namespace)
    cache.set_model_fingerprint(fingerprint)
    builder._load_model()
    cache.upsert_embeddings(
        {"arxiv:2401.00001": {"title": "2401.00001", "abstract": "body"}},
        builder.model,
        text_builder=builder._format_retrieval_document_metadata,
    )
    cache.mark_hydrated(
        builder.dataset_source, builder.dataset_split, None, complete=True
    )
    cache.set_hydration_source_revision("one")
    target = explicit_revision or "one"

    result = adopt_legacy_cache(builder, cache.namespace_id, revision=explicit_revision)

    assert result["source"]["revision"] == target
    assert resolutions == [target]
    assert reads == [target]
    assert loads == [(target, False)]
    assert result["active_rows"] == len(sources[target])
    assert builder.embedding_cache.get_hydration_source_revision() == target


def test_plain_adoption_resumes_an_interrupted_adoption_at_its_target(
    monkeypatch, runtime
):
    """A pending adoption's target wins over the older build pin."""
    sources = {
        "one": [row("2401.00001")],
        "two": [row("2401.00001"), row("2401.00002")],
    }
    install_source(monkeypatch, sources, ["three"])
    fingerprint = "hf::org/example::" + "a" * 40

    def legacy_builder():
        builder = make_builder()
        builder.model_revision = None
        monkeypatch.setattr(builder, "_resolve_model_fingerprint", lambda: fingerprint)
        return builder

    builder = legacy_builder()
    namespace = builder._embedding_cache_namespace(artifact_identity=fingerprint)
    cache = builder._create_embedding_cache(namespace)
    cache.set_model_fingerprint(fingerprint)
    builder._load_model()
    cache.upsert_embeddings(
        {"arxiv:2401.00001": {"title": "2401.00001", "abstract": "body"}},
        builder.model,
        text_builder=builder._format_retrieval_document_metadata,
    )
    cache.mark_hydrated(
        builder.dataset_source, builder.dataset_split, None, complete=True
    )
    cache.set_hydration_source_revision("one")
    upsert = EmbeddingCache.upsert_embeddings

    def interrupted(self, *args, **kwargs):
        upsert(self, *args, **kwargs)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(EmbeddingCache, "upsert_embeddings", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        adopt_legacy_cache(builder, cache.namespace_id, revision="two")
    monkeypatch.setattr(EmbeddingCache, "upsert_embeddings", upsert)

    result = adopt_legacy_cache(legacy_builder(), cache.namespace_id)

    assert result["source"]["revision"] == "two"
    assert result["active_rows"] == len(sources["two"])


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


@pytest.mark.parametrize("reader", ["stream", "download"])
@pytest.mark.parametrize("selection", ["all", "newest"])
@pytest.mark.parametrize("reverse", [False, True])
def test_dated_duplicate_winner_is_shared_and_precedes_newest_cap(
    monkeypatch, selection, reverse, reader
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
    records = read_source(
        reader,
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
@pytest.mark.parametrize("reader", ["stream", "download"])
def test_source_duplicate_recency_fallbacks(monkeypatch, older, newer, reader):
    for records in (
        [row("2401.00001", "z old", **older), row("2401.00001", "a new", **newer)],
        [row("2401.00001", "a new", **newer), row("2401.00001", "z old", **older)],
    ):
        install_source(monkeypatch, {"one": records}, ["one"])
        source = CorpusSource("org/source", None, "train", "one", "all", None)
        resolved = read_source(reader, source)
        assert len(resolved) == 1
        assert resolved[0]["paper_id"] == "arxiv:2401.00001"
        assert resolved[0]["abstract"] == "a new"


def test_streaming_staging_pass_runs_on_the_calling_thread(monkeypatch):
    """Ctrl-C must stop the staging pass at once, so it must not run on the worker."""
    staging_threads = set()

    def rows():
        for index in range(20):
            staging_threads.add(threading.current_thread())
            yield row(f"2401.{index:05d}")

    install_source(monkeypatch, {"one": rows()}, ["one"])
    source = CorpusSource("org/source", None, "train", "one", "all", None)
    prepared = list(
        refresh._prepared_batches(
            corpus_source.iter_source_batches(source, 8),
            lambda record: f"text {record['paper_id']}",
        )
    )

    assert staging_threads == {threading.current_thread()}
    # The worker reads the later batches through the same staging connection.
    assert [len(records) for records, _ in prepared] == [8, 8, 4]
    texts = {key: value for _, batch in prepared for key, value in batch.items()}
    assert len(texts) == 20
    assert texts["arxiv:2401.00019"] == "text arxiv:2401.00019"


@pytest.mark.parametrize("reader", ["stream", "download"])
@pytest.mark.parametrize("update_date", [None, "2026-10-16", "invalid"])
def test_ambiguous_duplicates_resolve_stably_with_aggregate_warning(
    monkeypatch, caplog, update_date, reader
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
        resolved = read_source(reader, source)
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


@pytest.mark.parametrize("reader", ["stream", "download"])
@pytest.mark.parametrize("failure_stage", ["load", "iterate"])
@pytest.mark.parametrize(
    "source_error",
    [httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError],
)
def test_source_transport_failures_are_operation_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: Literal["load", "iterate"],
    source_error: type[httpx.HTTPError],
    reader: Literal["stream", "download"],
) -> None:
    """Normalize transport errors from opening and consuming the pinned dataset.

    A download consumes every row inside the load, so no row counts as examined
    until it completes.
    """
    failure = source_error("source transport unavailable")

    def broken_source() -> Iterator[DatasetRecord]:
        """Read one source row before the upstream iterator fails."""
        yield row("2401.00001")
        raise failure

    def load_dataset(
        repo: str, **kwargs: str | bool | int
    ) -> Iterator[DatasetRecord] | datasets.Dataset:
        """Fail at dataset construction or return the interrupted source."""
        if failure_stage == "load":
            raise failure
        return (
            broken_source() if kwargs["streaming"] else source_dataset(broken_source())
        )

    monkeypatch.setattr(
        deps,
        "_import_datasets_module",
        lambda: SimpleNamespace(load_dataset=load_dataset),
    )
    source = CorpusSource("org/source", None, "train", "pinned-revision", "all", None)
    stats: corpus_source.SourceScanStats = {"source_records_examined": 0}

    with pytest.raises(OSError, match="source transport unavailable") as captured:
        read_source(reader, source, stats)

    assert "org/source@pinned-revision" in str(captured.value)
    assert captured.value.__cause__ is failure
    examined = 1 if failure_stage == "iterate" and reader == "stream" else 0
    assert stats["source_records_examined"] == examined


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


def test_refresh_streams_only_to_reconcile_a_complete_cache(monkeypatch, runtime):
    """Download whenever the whole source is processed; stream an incremental refresh."""
    loads = []
    revisions = {
        "one": [row("2401.00001")],
        "two": [row("2401.00001", "new"), row("2401.00002")],
        "three": [row("2401.00001", "newer"), row("2401.00003")],
    }
    active = ["one"]
    install_source(monkeypatch, revisions, active, loads)
    builder = make_builder()
    builder.refresh_corpus()
    active[0] = "two"
    builder.refresh_corpus()
    cache = builder.embedding_cache
    original = cache.upsert_embeddings

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("interrupted after vector commit")

    monkeypatch.setattr(cache, "upsert_embeddings", interrupt)
    active[0] = "three"
    with pytest.raises(RuntimeError, match="interrupted"):
        builder.refresh_corpus()
    monkeypatch.setattr(cache, "upsert_embeddings", original)
    builder.refresh_corpus()
    assert loads == [("one", False), ("two", True), ("three", True), ("three", False)]
    assert cache.corpus_state()["source"]["revision"] == "three"


@pytest.mark.parametrize("lost_tail", [False, True])
def test_resumed_refresh_skips_recorded_members_and_certifies_coverage(
    monkeypatch, runtime, lost_tail
):
    """Resume validates and encodes only papers the interrupted refresh did not record.

    With ``lost_tail`` the last committed vector row is missing on resume, so
    recovery drops its mapping and the member pointing at it is re-encoded.
    """
    encoded, _ = runtime
    paper_ids = [f"arxiv:2401.0000{index}" for index in range(1, 6)]
    install_source(
        monkeypatch,
        {"one": [row(paper_id.removeprefix("arxiv:")) for paper_id in paper_ids]},
        ["one"],
    )
    monkeypatch.setattr(refresh, "HYDRATION_FLUSH_SIZE", 2)
    original = EmbeddingCache.upsert_embeddings
    batches = []

    def interrupt_third(self, papers, *args, **kwargs):
        batches.append(list(papers))
        if len(batches) == 3:
            raise RuntimeError("interrupted before the third batch committed")
        return original(self, papers, *args, **kwargs)

    monkeypatch.setattr(EmbeddingCache, "upsert_embeddings", interrupt_third)
    builder = make_builder()
    with pytest.raises(RuntimeError, match="third batch"):
        builder.refresh_corpus()
    cache = builder.embedding_cache
    with cache._inspect_db() as connection:
        recorded = dict(
            connection.execute("SELECT paper_id,source_ordinal FROM corpus_membership")
        )
    assert recorded == {
        paper_id: ordinal for ordinal, paper_id in enumerate(paper_ids[:4])
    }
    if lost_tail:
        with h5py.File(cache.h5_path, "a") as handle:
            handle["embeddings"].resize((3, 2))
    monkeypatch.setattr(EmbeddingCache, "upsert_embeddings", original)
    validate = CorpusOperationSession.validate_records
    validated = []

    def record_validation(self, records):
        validated.extend(record["paper_id"] for record in records)
        return validate(self, records)

    monkeypatch.setattr(CorpusOperationSession, "validate_records", record_validation)
    encoded.clear()

    result = builder.refresh_corpus()

    resumed = paper_ids[3:] if lost_tail else paper_ids[4:]
    assert set(validated) == set(resumed)
    assert len(encoded) == len(resumed)
    assert result["active_rows"] == 5
    assert result["encoded_rows"] == len(resumed)
    assert result["source_records_examined"] == 5
    state = cache.corpus_state()
    assert state["membership_complete"] and "pending" not in state
    with cache._inspect_db() as connection:
        assert dict(
            connection.execute("SELECT paper_id,source_ordinal FROM corpus_membership")
        ) == {paper_id: ordinal for ordinal, paper_id in enumerate(paper_ids)}
    assert cache.embedding_count() == 5
    assert cache.get_cached_paper_ids() == set(paper_ids)


def test_resume_restores_interrupted_replacement_before_skipping_members(
    monkeypatch, runtime
):
    """A replacement interrupted before its mappings commit is rolled back and redone."""
    encoded, _ = runtime
    one = [row(f"2401.0000{index}") for index in range(1, 5)]
    two = [
        row("2401.00001"),
        row("2401.00002", "revised"),
        row("2401.00003"),
        row("2401.00004", "revised"),
    ]
    active = ["one"]
    install_source(monkeypatch, {"one": one, "two": two}, active)
    monkeypatch.setattr(refresh, "HYDRATION_FLUSH_SIZE", 2)
    builder = make_builder()
    builder.refresh_corpus()
    cache = builder.embedding_cache
    with h5py.File(cache.h5_path, "r") as handle:
        before = handle["embeddings"][:]
    commit = cache._commit_row_mappings
    commits = []

    def interrupt_second(**kwargs):
        commits.append(kwargs)
        if len(commits) == 2:
            raise RuntimeError("interrupted before mappings committed")
        return commit(**kwargs)

    monkeypatch.setattr(cache, "_commit_row_mappings", interrupt_second)
    active[0] = "two"
    with pytest.raises(RuntimeError, match="mappings committed"):
        builder.refresh_corpus()
    with cache._inspect_db() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM replacement_journal"
        ).fetchone() == (1,)
        assert {
            paper_id
            for (paper_id,) in connection.execute(
                "SELECT paper_id FROM corpus_membership"
            )
        } == {"arxiv:2401.00001", "arxiv:2401.00002"}
    with h5py.File(cache.h5_path, "r") as handle:
        replaced = handle["embeddings"][:]
    assert not np.array_equal(replaced[3], before[3])
    monkeypatch.setattr(cache, "_commit_row_mappings", commit)
    validate = CorpusOperationSession.validate_records
    validated = []

    def record_validation(self, records):
        validated.extend(record["paper_id"] for record in records)
        return validate(self, records)

    monkeypatch.setattr(CorpusOperationSession, "validate_records", record_validation)
    encoded.clear()

    result = builder.refresh_corpus()

    assert set(validated) == {"arxiv:2401.00003", "arxiv:2401.00004"}
    assert len(encoded) == 1 and "revised" in encoded[0]
    assert result["active_rows"] == 4 and result["encoded_rows"] == 1
    with cache._inspect_db() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM replacement_journal"
        ).fetchone() == (0,)
    with h5py.File(cache.h5_path, "r") as handle:
        after = handle["embeddings"][:]
    np.testing.assert_array_equal(after[[0, 1, 2]], replaced[[0, 1, 2]])
    np.testing.assert_array_equal(after[3], replaced[3])
    assert (
        cache.get_paper_metadata_batch(["arxiv:2401.00004"])["arxiv:2401.00004"][
            "abstract"
        ]
        == "revised"
    )
    assert cache.corpus_state()["membership_complete"]


@pytest.mark.parametrize("selection", ["all", "newest"])
def test_stream_and_download_select_identical_records(monkeypatch, selection):
    """Both readers keep the same winners, ordinals, order and examined count."""
    rows = [
        row("2501.00002", "dated old", update_date="2023-05-25"),
        row("2401.00001"),
        row("2501.00002", "dated new", update_date="2026-10-16 00:00:00"),
        {"title": "Undated paper", "abstract": "no identifier"},
        row("2401.00001"),
        row("2301.00003", "alpha"),
        row("2301.00003", "zulu"),
        row(
            "2401.00004v2",
            versions=[{"version": "v2", "created": "Thu, 25 May 2023 00:00:00 GMT"}],
        ),
        row("2401.00004v3"),
        row(
            "hep-th/9901001",
            authors_parsed=[["Doe", "Jane", ""]],
            categories="hep-th math-ph",
            doi="doi: 10.1000/example",
        ),
    ]
    install_source(monkeypatch, {"one": rows}, ["one"])
    source = CorpusSource(
        "org/source",
        None,
        "train",
        "one",
        selection,
        4 if selection == "newest" else None,
    )
    stream_stats: corpus_source.SourceScanStats = {"source_records_examined": 0}
    download_stats: corpus_source.SourceScanStats = {"source_records_examined": 0}
    streamed = read_source("stream", source, stream_stats)
    downloaded = read_source("download", source, download_stats)
    assert downloaded == streamed
    assert [record["source_ordinal"] for record in streamed] == (
        [2, 8, 1, 6] if selection == "newest" else [1, 2, 3, 6, 8, 9]
    )
    assert stream_stats == download_stats == {"source_records_examined": len(rows)}


def test_calibration_sample_matches_hydration_reservoir(monkeypatch, runtime, tmp_path):
    """One prepared selection yields hydration's calibration sample."""
    rows = [row(f"2401.{index:05d}") for index in range(1, 12)]
    rows.insert(3, row("2401.00002"))
    install_source(monkeypatch, {"one": rows}, ["one"])
    builder = make_builder(storage_precision="int8", calibration_sample_size=3)
    source = CorpusSource("org/source", None, "train", "one", "all", None)
    expected = builder._sample_calibration_records(
        corpus_source.iter_source_records(source),
        progress_total=None,
        progress_label="Calibrating source",
        corpus_size=None,
    )
    selection = corpus_source.prepare_source_selection(source, tmp_path)
    assert refresh._calibration_sample(selection, 3) == expected


@pytest.mark.parametrize("rows", [2, 3])
def test_source_normalizes_in_workers_only_from_row_threshold(
    monkeypatch, rows, tmp_path
):
    """Small sources such as fixtures are normalized without worker processes."""
    install_source(
        monkeypatch,
        {"one": [row(f"2401.{index:05d}") for index in range(rows)]},
        ["one"],
    )
    monkeypatch.setattr(corpus_source, "PARALLEL_SOURCE_MIN_ROWS", 3)
    monkeypatch.setattr(corpus_source, "_source_processes", lambda: 2)
    original = datasets.Dataset.map
    requested = []

    def record_map(self, *args, num_proc=None, **kwargs):
        requested.append(num_proc)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(datasets.Dataset, "map", record_map)
    selection = corpus_source.prepare_source_selection(
        CorpusSource("org/source", None, "train", "one", "all", None), tmp_path
    )
    assert len(selection) == rows
    assert requested == [2 if rows >= 3 else None]
