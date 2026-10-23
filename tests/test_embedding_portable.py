"""Portable payload contracts, bounded interchange, and malformed snapshots."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from citemesh.data.embedding_cache.constants import BINARY_INDEX_DATASET_NAME
from citemesh.data.embedding_cache.portable import (
    PortableManifest,
    export_snapshot,
    import_snapshot,
    iter_snapshot_batches,
    load_manifest,
    portable_schema,
    validate_snapshot,
)
from citemesh.data.embedding_contract import (
    CorpusSource,
    EmbeddingContract,
    serialize_embedding_namespace,
)


def contract(dtype="int8", dimension=4):
    return EmbeddingContract(
        model_id="unsloth/embeddinggemma-300m",
        model_revision="a" * 40,
        model_artifact_identity="test-artifact",
        profile_key="embeddinggemma",
        profile_schema="embeddinggemma-v1",
        representation="retrieval-document-v1",
        formatter_identity="test-formatter",
        normalization="l2-v1",
        dimension=dimension,
        stored_dtype=dtype,
        quantization="affine-int8-v1" if dtype == "int8" else None,
        calibration_sample_size=2000 if dtype == "int8" else None,
    )


def records(count=7):
    return [
        {
            "paper_id": f"arxiv:2401.{i:05d}",
            "arxiv_id": f"2401.{i:05d}",
            "doi": None,
            "title": f"Title {i}",
            "abstract": "The unchanged source text.",
            "year": 2024,
            "authors": ["A. Author"],
            "categories": ["cs.AI"],
            "venue": None,
            "text_hash": f"hash-{i}",
            "chronology_key": 100 - i if i % 2 else None,
            "source_ordinal": 100 - i,
        }
        for i in range(count)
    ]


class ExportFixture:
    def __init__(self, dtype="int8", dimension=4, count=7):
        self.contract = contract(dtype, dimension)
        self.records = records(count)
        self.vectors = np.arange(count * dimension, dtype=dtype).reshape(
            count, dimension
        )
        if dtype == "float32":
            self.vectors = self.vectors / np.float32(3.14159)
        self.batches = []

    @contextmanager
    def export_snapshot_view(self):
        selectors = {
            "model_identity": self.contract.model_id,
            "requested_revision": "main",
            "artifact_identity": self.contract.model_artifact_identity,
            "profile_schema": self.contract.profile_schema,
            "representation": self.contract.representation,
            "dimension": self.contract.dimension,
            "stored_dtype": self.contract.stored_dtype,
            "formatter_identity": self.contract.formatter_identity,
        }

        def batches(batch_size):
            for first in range(0, len(self.records), batch_size):
                self.batches.append(min(batch_size, len(self.records) - first))
                yield (
                    self.records[first : first + batch_size],
                    self.vectors[first : first + batch_size],
                )

        yield SimpleNamespace(
            contract=self.contract,
            namespace_serialization=serialize_embedding_namespace(**selectors),
            namespace_selectors=selectors,
            completed_source=CorpusSource(
                "test/source", None, "train", "b" * 40, "all", None
            ),
            source_records_examined=len(self.records) + 2,
            producer_provenance={"source_torch_dtype": "bfloat16"},
            calibration_ranges=np.array(
                [[-1.1] * self.contract.dimension, [2.7] * self.contract.dimension],
                dtype=np.float32,
            ),
            iter_active_rows=batches,
        )


@pytest.mark.parametrize(
    "dtype,dimension", [("int8", 4), ("int8", 8), ("float32", 3), ("float32", 7)]
)
def test_exact_bounded_export(dtype, dimension, tmp_path):
    cache = ExportFixture(dtype, dimension)
    if dtype == "int8":
        cache.vectors.flat[:4] = [-128, -1, 0, 127]
    else:
        cache.vectors.flat[:4] = [
            0.0,
            -0.0,
            np.nextafter(np.float32(0), np.float32(1)),
            123.456,
        ]
    snapshot = tmp_path / "snapshot"
    manifest = export_snapshot(
        cache, snapshot, config_name="test", shard_rows=3, batch_rows=2
    )
    assert [shard.row_count for shard in manifest.shards] == [3, 3, 1]
    assert cache.batches == [2, 2, 2, 1]
    assert load_manifest(snapshot) == manifest
    assert validate_snapshot(snapshot, batch_rows=1) == manifest
    batches = list(iter_snapshot_batches(snapshot, manifest, batch_rows=2))
    assert max(len(item[1]) for item in batches) <= 2
    assert [record for _, batch, _ in batches for record in batch] == cache.records
    actual = np.concatenate([vectors for _, _, vectors in batches])
    assert actual.dtype == cache.vectors.dtype
    assert actual.tobytes() == cache.vectors.tobytes()
    assert manifest.producer_provenance == {"source_torch_dtype": "bfloat16"}
    assert manifest.source_records_examined == 9
    if dtype == "int8":
        expected = np.array([[-1.1] * dimension, [2.7] * dimension], dtype=np.float32)
        assert (
            np.load(snapshot / manifest.calibration_ranges).tobytes()
            == expected.tobytes()
        )
    else:
        assert manifest.calibration_ranges is None


@pytest.mark.parametrize(
    "alteration",
    [
        "row-number",
        "duplicate",
        "null-coordinate",
        "nonfinite",
        "dtype",
        "width",
        "shard-count",
    ],
)
def test_reader_refuses_malformed_payload(alteration, tmp_path):
    snapshot = tmp_path / "snapshot"
    manifest = export_snapshot(ExportFixture("float32"), snapshot, shard_rows=7)
    path = snapshot / manifest.shards[0].path
    table = pq.read_table(path)
    if alteration == "shard-count":
        table = table.slice(0, 6)
    else:
        name = "embedding"
        rows = table[name].to_pylist()
        data_type = portable_schema(manifest.contract).field(name).type
        if alteration == "row-number":
            name = "row_number"
            rows = [9] + list(range(1, 7))
            data_type = pa.int64()
        elif alteration == "duplicate":
            name = "paper_id"
            rows = table[name].to_pylist()
            rows[-1] = rows[0]
            data_type = pa.string()
        elif alteration == "null-coordinate":
            rows[-1][0] = None
        elif alteration == "nonfinite":
            rows[-1][0] = float("inf")
        elif alteration == "dtype":
            data_type = pa.list_(pa.float64(), 4)
        elif alteration == "width":
            data_type = pa.list_(pa.float32(), 3)
            rows = [row[:3] for row in rows]
        field = pa.field(name, data_type, nullable=False)
        table = table.set_column(
            table.schema.get_field_index(name), field, pa.array(rows, type=data_type)
        )
    pq.write_table(table, path)
    with pytest.raises(ValueError):
        validate_snapshot(snapshot, batch_rows=2)


@pytest.mark.parametrize(
    "path",
    [
        "../out.parquet",
        "/out.parquet",
        "index.json",
        "data//part.parquet",
        ".",
        "data/./part.parquet",
    ],
)
def test_manifest_rejects_invalid_artifact_paths(path, tmp_path):
    manifest = export_snapshot(ExportFixture(), tmp_path / "snapshot")
    value = manifest.to_dict()
    value["shards"][0]["path"] = path
    with pytest.raises(ValueError, match="artifact path"):
        PortableManifest.from_dict(value)


@pytest.mark.parametrize(
    "malformation", ["coverage", "version", "ranges", "float-ranges"]
)
def test_manifest_rejects_inconsistent_contracts(malformation, tmp_path):
    manifest = export_snapshot(ExportFixture(), tmp_path / "snapshot")
    value = manifest.to_dict()
    if malformation == "coverage":
        value["row_count"] += 1
    elif malformation == "version":
        value["format_version"] = 999
    elif malformation == "ranges":
        value["calibration_ranges"] = None
    else:
        value["contract"] = replace(manifest.contract, stored_dtype="float32").to_dict()
    with pytest.raises(ValueError):
        PortableManifest.from_dict(value)


def test_failed_export_has_no_completion_manifest(tmp_path):
    cache = ExportFixture("float32")
    cache.vectors[-1][0] = np.nan
    snapshot = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="finite"):
        export_snapshot(cache, snapshot, batch_rows=2)
    assert not (snapshot / "index.json").exists()
    with pytest.raises(FileExistsError):
        export_snapshot(cache, snapshot)


def test_invalid_ranges_refused_before_import_session(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    manifest = export_snapshot(ExportFixture(), snapshot)
    np.save(snapshot / manifest.calibration_ranges, np.zeros((2, 4), dtype=np.float64))
    with pytest.raises(ValueError, match="ranges"):
        validate_snapshot(snapshot)


def test_manifest_is_written_last(tmp_path):
    snapshot = tmp_path / "snapshot"
    export_snapshot(ExportFixture(), snapshot)
    value = json.loads((snapshot / "index.json").read_text())
    assert all((snapshot / shard["path"]).is_file() for shard in value["shards"])


@pytest.mark.parametrize("dtype,dimension", [("int8", 4), ("float32", 7)])
def test_native_import_round_trip_and_idempotence(dtype, dimension, tmp_path):
    fixture = ExportFixture(dtype, dimension)
    snapshot = tmp_path / "snapshot"
    manifest = export_snapshot(fixture, snapshot, config_name="test", shard_rows=3)
    destination = tmp_path / "native"
    cache = import_snapshot(snapshot, destination, batch_rows=2)
    with h5py.File(cache.h5_path, "r") as handle:
        assert handle["embeddings"][:].tobytes() == fixture.vectors.tobytes()
        if dtype == "int8":
            assert (
                handle["calibration_ranges"][:].tobytes()
                == np.load(snapshot / manifest.calibration_ranges).tobytes()
            )
            assert handle[BINARY_INDEX_DATASET_NAME].shape[0] == 7
        else:
            assert "calibration_ranges" not in handle
            assert BINARY_INDEX_DATASET_NAME not in handle
    state = cache.corpus_state()
    assert state["membership_complete"] is True
    assert "pending" not in state
    assert state["source"] == manifest.source.to_dict()
    assert state["producer_provenance"] == manifest.producer_provenance
    before = (cache.db_path.read_bytes(), cache.h5_path.read_bytes())
    repeated = import_snapshot(snapshot, destination, batch_rows=1)
    assert (repeated.db_path.read_bytes(), repeated.h5_path.read_bytes()) == before
    second = tmp_path / "second"
    exported = export_snapshot(cache, second, config_name="second", batch_rows=2)
    assert exported.producer_provenance == manifest.producer_provenance
    batches = list(iter_snapshot_batches(second, exported))
    assert (
        np.concatenate([item[2] for item in batches]).tobytes()
        == fixture.vectors.tobytes()
    )
    assert [record["chronology_key"] for _, rows, _ in batches for record in rows] == [
        record["chronology_key"] for record in fixture.records
    ]


def test_native_import_resumes_committed_prefix_after_later_corrupt_shard(tmp_path):
    fixture = ExportFixture()
    snapshot = tmp_path / "snapshot"
    manifest = export_snapshot(fixture, snapshot, shard_rows=3)
    last_shard = snapshot / manifest.shards[-1].path
    correct_bytes = last_shard.read_bytes()
    table = pq.read_table(last_shard)
    table = table.set_column(0, table.schema.field(0), pa.array([900], type=pa.int64()))
    pq.write_table(table, last_shard)
    with pytest.raises(ValueError, match="contiguous"):
        import_snapshot(snapshot, tmp_path / "native", batch_rows=2)
    database = next((tmp_path / "native").glob("metadata_*.db"))
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 6
        state = json.loads(
            conn.execute(
                "SELECT value FROM cache_metadata WHERE key='corpus_state'"
            ).fetchone()[0]
        )
        assert state["pending"]["kind"] == "import"
        assert not state["membership_complete"]
    last_shard.write_bytes(correct_bytes)
    cache = import_snapshot(snapshot, tmp_path / "native", batch_rows=2)
    with h5py.File(cache.h5_path, "r") as handle:
        assert handle["embeddings"][:].tobytes() == fixture.vectors.tobytes()
    assert cache.corpus_state()["membership_complete"]


@pytest.mark.parametrize("dtype", ["int8", "float32"])
@pytest.mark.parametrize("binary_prefilter", [False, True])
def test_native_active_order_and_search_agreement(dtype, binary_prefilter, tmp_path):
    fixture = ExportFixture(dtype, 8)
    first = tmp_path / "first"
    export_snapshot(fixture, first)
    publisher = import_snapshot(first, tmp_path / "publisher")
    with sqlite3.connect(publisher.db_path) as connection:
        connection.execute(
            "DELETE FROM corpus_membership WHERE source_ordinal IN (99,97)"
        )
    second = tmp_path / "second"
    manifest = export_snapshot(publisher, second, shard_rows=2, batch_rows=3)
    assert manifest.row_count == 5
    consumer = import_snapshot(second, tmp_path / "consumer", batch_rows=1)
    with h5py.File(consumer.h5_path, "r") as handle:
        assert (
            handle["embeddings"][:].tobytes()
            == fixture.vectors[[0, 2, 4, 5, 6]].tobytes()
        )
    query = np.arange(8, dtype=np.float32) - 3
    options = {
        "top_k": 4,
        "binary_prefilter": binary_prefilter,
        "binary_rescore_multiplier": 1,
    }
    expected = publisher.search(query, **options)
    actual = consumer.search(query, **options)
    assert [item.paper_id for item in actual] == [item.paper_id for item in expected]
    np.testing.assert_allclose(
        [item.score for item in actual], [item.score for item in expected], atol=1e-6
    )


@pytest.mark.parametrize("conflict", ["source-advanced", "ranges", "publication"])
def test_occupied_import_refuses_conflict_without_mutating_payload(conflict, tmp_path):
    snapshot = tmp_path / "snapshot"
    manifest = export_snapshot(ExportFixture(), snapshot)
    destination = tmp_path / "native"
    cache = import_snapshot(snapshot, destination)
    publication = None
    if conflict == "source-advanced":
        with sqlite3.connect(cache.db_path) as connection:
            state = cache.corpus_state()
            state["source"]["revision"] = "c" * 40
            connection.execute(
                "UPDATE cache_metadata SET value=? WHERE key='corpus_state'",
                (json.dumps(state),),
            )
    elif conflict == "ranges":
        ranges = np.load(snapshot / manifest.calibration_ranges)
        ranges[1] += 1
        np.save(snapshot / manifest.calibration_ranges, ranges)
    else:
        publication = {
            "repo_id": "private/index",
            "revision": "c" * 40,
            "config_name": manifest.config_name,
        }
    before = (cache.db_path.read_bytes(), cache.h5_path.read_bytes())
    with pytest.raises(ValueError):
        import_snapshot(snapshot, destination, publication=publication)
    assert (cache.db_path.read_bytes(), cache.h5_path.read_bytes()) == before


@pytest.mark.parametrize("dtype", ["int8", "float32"])
def test_empty_certified_snapshot_round_trip(dtype, tmp_path):
    snapshot = tmp_path / "empty"
    manifest = export_snapshot(ExportFixture(dtype, count=0), snapshot)
    assert manifest.shards == ()
    assert validate_snapshot(snapshot) == manifest
    cache = import_snapshot(snapshot, tmp_path / "native")
    assert cache.corpus_state()["membership_complete"]
    assert cache.embedding_count() == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_identity", "another/model"),
        ("artifact_identity", "other-artifact"),
        ("profile_schema", "other-profile"),
        ("representation", "symmetric-v1"),
        ("normalization", "none"),
        ("dimension", 9),
        ("stored_dtype", "float32"),
        ("calibration_sample_size", 1),
        ("formatter_identity", "other-formatter"),
    ],
)
def test_manifest_rejects_contract_address_mismatch(field, value, tmp_path):
    manifest = export_snapshot(ExportFixture(), tmp_path / "snapshot")
    serialized = manifest.to_dict()
    serialized["namespace_selectors"][field] = value
    serialized["namespace_serialization"] = serialize_embedding_namespace(
        **serialized["namespace_selectors"]
    )
    with pytest.raises(ValueError, match="contract"):
        PortableManifest.from_dict(serialized)


def test_manifest_rejects_unknown_quantization(tmp_path):
    manifest = export_snapshot(ExportFixture(), tmp_path / "snapshot")
    serialized = manifest.to_dict()
    serialized["contract"]["quantization"] = "other-int8-codec"
    with pytest.raises(ValueError, match="quantization"):
        PortableManifest.from_dict(serialized)
