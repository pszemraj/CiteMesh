"""Bounded, lossless interchange of completed corpus embedding caches."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import numpy as np

from citemesh.data.embedding_contract import (
    CorpusSource,
    EmbeddingContract,
    require_namespace_contract,
)

FORMAT_VERSION = 1


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or str(path) != value
        or value in {".", "index.json"}
    ):
        raise ValueError(f"Invalid portable artifact path: {value!r}")
    return value


@dataclass(frozen=True)
class PortableShard:
    """One ordered, contiguous segment of portable rows."""

    path: str
    first_row: int
    row_count: int


@dataclass(frozen=True)
class PortableManifest:
    """The complete interpretation and coverage of one exported snapshot."""

    config_name: str
    snapshot_id: str
    created_at: str
    producer_version: str
    contract: EmbeddingContract
    namespace_serialization: str
    namespace_selectors: dict[str, Any]
    source: CorpusSource
    row_count: int
    source_records_examined: int
    shards: tuple[PortableShard, ...]
    calibration_ranges: str | None = None
    producer_provenance: dict[str, Any] = field(default_factory=dict)
    format_version: int = FORMAT_VERSION
    row_order: str = "native-relative"

    @property
    def artifact_paths(self) -> list[str]:
        """Return only the payloads explicitly owned by this configuration."""
        paths = [shard.path for shard in self.shards]
        if self.calibration_ranges is not None:
            paths.append(self.calibration_ranges)
        return paths

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PortableManifest:
        """Parse and validate the supported manifest before opening a cache."""
        if "format_version" not in value or "row_order" not in value:
            raise ValueError("Portable manifest requires format and row-order versions")
        try:
            manifest = cls(
                **{
                    **value,
                    "contract": EmbeddingContract(**value["contract"]),
                    "source": CorpusSource(**value["source"]),
                    "shards": tuple(PortableShard(**item) for item in value["shards"]),
                }
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid portable manifest: {exc}") from exc
        if manifest.format_version != FORMAT_VERSION:
            raise ValueError("Unsupported portable format version")
        if manifest.row_order != "native-relative":
            raise ValueError("Unsupported portable row-order convention")
        if not all(
            isinstance(item, str) and item.strip()
            for item in (
                manifest.config_name,
                manifest.snapshot_id,
                manifest.created_at,
                manifest.producer_version,
                manifest.namespace_serialization,
            )
        ):
            raise ValueError("Portable manifest identity fields must be nonempty")
        if not isinstance(manifest.namespace_selectors, dict) or not isinstance(
            manifest.producer_provenance, dict
        ):
            raise ValueError("Portable selectors and provenance must be mappings")
        _validate_contract(manifest.contract)
        _validate_manifest_semantics(manifest)
        if (
            type(manifest.row_count) is not int
            or manifest.row_count < 0
            or type(manifest.source_records_examined) is not int
            or manifest.source_records_examined < manifest.row_count
        ):
            raise ValueError("Invalid portable source coverage")
        next_row = 0
        paths = set()
        for shard in manifest.shards:
            _relative_path(shard.path)
            if (
                shard.path in paths
                or type(shard.first_row) is not int
                or shard.first_row != next_row
                or type(shard.row_count) is not int
                or shard.row_count <= 0
            ):
                raise ValueError("Portable shards must cover distinct contiguous rows")
            paths.add(shard.path)
            next_row += shard.row_count
        if next_row != manifest.row_count:
            raise ValueError("Portable shard coverage differs from row_count")
        if manifest.contract.stored_dtype == "int8":
            if not isinstance(manifest.calibration_ranges, str):
                raise ValueError("An int8 snapshot requires calibration ranges")
            _relative_path(manifest.calibration_ranges)
            if manifest.calibration_ranges in paths:
                raise ValueError("Calibration artifact duplicates a shard path")
        elif manifest.calibration_ranges is not None:
            raise ValueError("A float32 snapshot must not contain calibration ranges")
        return manifest


def _validate_manifest_semantics(manifest: PortableManifest) -> None:
    contract = manifest.contract
    for name in (
        "model_id",
        "model_revision",
        "model_artifact_identity",
        "profile_key",
        "profile_schema",
        "formatter_identity",
    ):
        if not str(getattr(contract, name) or "").strip():
            raise ValueError(f"Portable embedding contract requires {name}")
    if contract.model_revision in {"main", "master", "latest"}:
        raise ValueError("Portable model revision must be resolved and immutable")
    if (
        contract.representation != "retrieval-document-v1"
        or contract.normalization != "l2-v1"
    ):
        raise ValueError(
            "Unsupported portable embedding representation or normalization"
        )
    if contract.stored_dtype == "int8":
        if not contract.quantization or (
            type(contract.calibration_sample_size) is not int
            or contract.calibration_sample_size <= 0
        ):
            raise ValueError(
                "Int8 contract requires quantization and calibration settings"
            )
    elif contract.quantization is not None:
        raise ValueError("Float32 contract must not declare quantization")
    source = manifest.source
    if not all(
        isinstance(item, str) and item.strip()
        for item in (source.repo_id, source.split, source.revision)
    ):
        raise ValueError(
            "Portable source requires repository, split, and resolved revision"
        )
    if source.revision in {"main", "master", "latest"}:
        raise ValueError("Portable source revision must be resolved and immutable")
    if (
        source.selection not in {"all", "newest"}
        or (source.selection == "all" and source.limit is not None)
        or (
            source.selection == "newest"
            and (type(source.limit) is not int or source.limit < 1)
        )
    ):
        raise ValueError("Portable source selection and limit disagree")
    require_namespace_contract(
        contract, manifest.namespace_selectors, manifest.namespace_serialization
    )


def _validate_contract(contract: EmbeddingContract) -> None:
    if contract.stored_dtype not in {"int8", "float32"} or (
        type(contract.dimension) is not int or contract.dimension <= 0
    ):
        raise ValueError("Unsupported embedding shape or stored dtype")


def load_manifest(path: Path | str) -> PortableManifest:
    """Read an export directory or an explicit ``index.json`` path."""
    path = Path(path)
    if path.is_dir():
        path /= "index.json"
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Portable manifest must be a JSON object")
    return PortableManifest.from_dict(value)


def portable_schema(contract: EmbeddingContract):
    """Return the version-one schema; Arrow is optional until invoked."""
    import pyarrow as pa

    _validate_contract(contract)
    value_type = pa.int8() if contract.stored_dtype == "int8" else pa.float32()
    return pa.schema(
        [
            pa.field("row_number", pa.int64(), nullable=False),
            pa.field("paper_id", pa.string(), nullable=False),
            pa.field("arxiv_id", pa.string()),
            pa.field("doi", pa.string()),
            pa.field("title", pa.string()),
            pa.field("abstract", pa.string()),
            pa.field("year", pa.int32()),
            pa.field("authors", pa.list_(pa.string())),
            pa.field("categories", pa.list_(pa.string())),
            pa.field("venue", pa.string()),
            pa.field("text_hash", pa.string(), nullable=False),
            pa.field("chronology_key", pa.int64()),
            pa.field("source_ordinal", pa.int64(), nullable=False),
            pa.field(
                "embedding", pa.list_(value_type, contract.dimension), nullable=False
            ),
        ]
    )


def _validate_vectors(vectors: np.ndarray, contract: EmbeddingContract, rows: int):
    if vectors.dtype != np.dtype(contract.stored_dtype):
        raise ValueError("Stored vector dtype differs from the snapshot contract")
    if vectors.shape != (rows, contract.dimension):
        raise ValueError("Stored vector shape differs from the snapshot contract")
    if contract.stored_dtype == "float32" and not np.isfinite(vectors).all():
        raise ValueError("Stored float32 vectors must contain only finite values")


def _validate_ranges(ranges: np.ndarray, contract: EmbeddingContract) -> None:
    if (
        ranges.dtype != np.dtype("float32")
        or ranges.shape != (2, contract.dimension)
        or not np.isfinite(ranges).all()
        or not np.all(ranges[0] < ranges[1])
    ):
        raise ValueError("Invalid float32 calibration ranges")


def _record_batch(records, vectors, contract, first_row):
    import pyarrow as pa

    _validate_vectors(vectors, contract, len(records))
    schema = portable_schema(contract)
    columns = []
    for column in schema:
        if column.name == "embedding":
            values = pa.array(vectors.reshape(-1), type=column.type.value_type)
            array = pa.FixedSizeListArray.from_arrays(values, contract.dimension)
        elif column.name == "row_number":
            array = pa.array(
                range(first_row, first_row + len(records)), type=pa.int64()
            )
        else:
            array = pa.array(
                [record.get(column.name) for record in records], type=column.type
            )
        if not column.nullable and array.null_count:
            raise ValueError(f"Null values in required portable column {column.name}")
        columns.append(array)
    return pa.RecordBatch.from_arrays(columns, schema=schema)


def export_snapshot(
    cache,
    output_dir: Path | str,
    *,
    config_name: str = "embeddinggemma-512-int8",
    shard_rows: int = 65_536,
    batch_rows: int = 8_192,
) -> PortableManifest:
    """Stream one locked native corpus into a fresh portable directory."""
    import pyarrow.parquet as pq

    if shard_rows < 1 or batch_rows < 1:
        raise ValueError("Shard and batch row counts must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "data").mkdir()
    writer = None
    shards = []
    next_row = 0
    shard_count = 0
    shard_start = 0
    shard_path = ""
    try:
        with cache.export_snapshot_view() as view:
            for records, vectors in view.iter_active_rows(batch_size=batch_rows):
                _validate_vectors(vectors, view.contract, len(records))
                offset = 0
                while offset < len(records):
                    if writer is None:
                        shard_start = next_row
                        shard_count = 0
                        shard_path = f"data/part-{len(shards):05d}.parquet"
                        writer = pq.ParquetWriter(
                            output_dir / shard_path,
                            portable_schema(view.contract),
                            compression="zstd",
                            use_dictionary=["venue", "year"],
                        )
                    count = min(len(records) - offset, shard_rows - shard_count)
                    batch = _record_batch(
                        records[offset : offset + count],
                        vectors[offset : offset + count],
                        view.contract,
                        next_row,
                    )
                    writer.write_batch(batch)
                    offset += count
                    next_row += count
                    shard_count += count
                    if shard_count == shard_rows:
                        writer.close()
                        writer = None
                        shards.append(
                            PortableShard(shard_path, shard_start, shard_count)
                        )
            if writer is not None:
                writer.close()
                writer = None
                shards.append(PortableShard(shard_path, shard_start, shard_count))
            ranges_path = None
            if view.contract.stored_dtype == "int8":
                ranges = view.calibration_ranges
                _validate_ranges(ranges, view.contract)
                ranges_path = "calibration_ranges.npy"
                np.save(output_dir / ranges_path, ranges, allow_pickle=False)
            manifest = PortableManifest(
                config_name=config_name,
                snapshot_id=str(uuid4()),
                created_at=datetime.now(timezone.utc).isoformat(),
                producer_version=version("citemesh"),
                contract=view.contract,
                namespace_serialization=view.namespace_serialization,
                namespace_selectors=view.namespace_selectors,
                source=view.completed_source,
                row_count=next_row,
                source_records_examined=view.source_records_examined,
                producer_provenance=view.producer_provenance,
                shards=tuple(shards),
                calibration_ranges=ranges_path,
            )
            PortableManifest.from_dict(manifest.to_dict())
        with (output_dir / "index.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, indent=2)
            handle.write("\n")
        return manifest
    finally:
        if writer is not None:
            writer.close()


def iter_snapshot_batches(
    snapshot_dir: Path | str, manifest: PortableManifest, *, batch_rows: int = 8_192
):
    """Validate ordered shards and yield bounded exact stored-vector batches."""

    if batch_rows < 1:
        raise ValueError("Batch row count must be positive")
    with TemporaryDirectory(prefix="citemesh-portable-") as temporary:
        with sqlite3.connect(Path(temporary) / "ids.sqlite") as seen:
            seen.execute("CREATE TABLE ids (paper_id TEXT PRIMARY KEY)")
            yield from _iter_snapshot_batches(snapshot_dir, manifest, batch_rows, seen)


def _iter_snapshot_batches(snapshot_dir, manifest, batch_rows, seen):
    import pyarrow.parquet as pq

    schema = portable_schema(manifest.contract)
    next_row = 0
    for shard in manifest.shards:
        with pq.ParquetFile(Path(snapshot_dir) / shard.path) as parquet:
            if not parquet.schema_arrow.equals(schema, check_metadata=False):
                raise ValueError(f"Portable shard schema mismatch: {shard.path}")
            if parquet.metadata.num_rows != shard.row_count:
                raise ValueError(f"Portable shard row count mismatch: {shard.path}")
            for batch in parquet.iter_batches(batch_size=batch_rows):
                for column in schema:
                    array = batch.column(schema.get_field_index(column.name))
                    if not column.nullable and array.null_count:
                        raise ValueError(
                            f"Null values in required portable column {column.name}"
                        )
                row_numbers = batch.column(0).to_numpy(zero_copy_only=False)
                if not np.array_equal(
                    row_numbers, np.arange(next_row, next_row + len(batch))
                ):
                    raise ValueError("Portable row numbers are not contiguous")
                embedding = batch.column(schema.get_field_index("embedding"))
                values = embedding.values.slice(
                    embedding.offset * manifest.contract.dimension,
                    len(batch) * manifest.contract.dimension,
                )
                if values.null_count:
                    raise ValueError("Portable embedding coordinates must not be null")
                vectors = values.to_numpy(zero_copy_only=False).reshape(
                    len(batch), manifest.contract.dimension
                )
                _validate_vectors(vectors, manifest.contract, len(batch))
                records = batch.select(schema.names[1:-1]).to_pylist()
                if any(record["source_ordinal"] < 0 for record in records):
                    raise ValueError("Portable source ordinals must be nonnegative")
                if any(
                    not record["paper_id"] or not record["text_hash"]
                    for record in records
                ):
                    raise ValueError(
                        "Portable paper IDs and text hashes must be nonempty"
                    )
                try:
                    seen.executemany(
                        "INSERT INTO ids VALUES (?)",
                        ((record["paper_id"],) for record in records),
                    )
                    seen.commit()
                except sqlite3.IntegrityError as exc:
                    raise ValueError("Portable paper IDs must be unique") from exc
                yield next_row, records, vectors
                next_row += len(batch)
    if next_row != manifest.row_count:
        raise ValueError("Portable payload differs from declared row coverage")


def validate_snapshot(snapshot_dir: Path | str, *, batch_rows: int = 8_192):
    """Validate every declared payload using bounded reads before publication."""
    manifest = load_manifest(snapshot_dir)
    if manifest.calibration_ranges is not None:
        ranges = np.load(
            Path(snapshot_dir) / manifest.calibration_ranges, allow_pickle=False
        )
        _validate_ranges(ranges, manifest.contract)
    for _batch in iter_snapshot_batches(snapshot_dir, manifest, batch_rows=batch_rows):
        pass
    return manifest


def import_snapshot(
    snapshot_dir: Path | str,
    destination: Path | None = None,
    *,
    publication: dict[str, Any] | None = None,
    batch_rows: int = 8_192,
):
    """Resume or install a snapshot through the native append-only session."""
    from .lifecycle import import_destination_session

    snapshot_dir = Path(snapshot_dir)
    manifest = load_manifest(snapshot_dir)
    ranges = None
    if manifest.calibration_ranges is not None:
        ranges = np.load(snapshot_dir / manifest.calibration_ranges, allow_pickle=False)
        _validate_ranges(ranges, manifest.contract)
    identity = manifest_import_identity(manifest)
    origin = {"snapshot_id": manifest.snapshot_id, **(publication or {})}
    with import_destination_session(
        cache_dir=destination,
        namespace_serialization=manifest.namespace_serialization,
        contract=manifest.contract,
        source=manifest.source,
        identity=identity,
        namespace_selectors=manifest.namespace_selectors,
        calibration_ranges=ranges,
        origin=origin,
        producer_provenance=manifest.producer_provenance,
    ) as session:
        if session.already_complete:
            return session.cache
        prefix = session.committed_rows
        for first_row, records, vectors in iter_snapshot_batches(
            snapshot_dir, manifest, batch_rows=batch_rows
        ):
            skip = min(len(records), max(0, prefix - first_row))
            if skip == len(records):
                continue
            session.append_stored_embeddings(
                records[skip:], vectors[skip:], first_row=first_row + skip
            )
        session.finish(
            expected_rows=manifest.row_count,
            source_records_examined=manifest.source_records_examined,
        )
        return session.cache


def manifest_import_identity(manifest: PortableManifest) -> dict[str, Any]:
    """Use one operation identity for preflight, import, and interrupted resume."""
    return {
        "snapshot_id": manifest.snapshot_id,
        "contract": asdict(manifest.contract),
        "source": asdict(manifest.source),
        "shards": [asdict(shard) for shard in manifest.shards],
    }
