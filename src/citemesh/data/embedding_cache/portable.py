"""Bounded, lossless interchange of completed corpus embedding caches."""

from __future__ import annotations

import json
import platform
import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, TypedDict, cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from citemesh.data.embedding_cache.models import (
    ActiveCorpusRecord,
    CorpusOrigin,
    CorpusRecord,
    PortableImportIdentity,
    PortableShardData,
    PublicationOrigin,
    RuntimeVersions,
    StoredEmbedding,
)
from citemesh.data.embedding_contract import (
    CorpusSource,
    CorpusSourceData,
    EmbeddingContract,
    EmbeddingContractData,
    NamespaceSelectors,
    require_namespace_contract,
)

if TYPE_CHECKING:
    import pyarrow as pa

    from .store import EmbeddingCache

FORMAT_VERSION = 1


class PortableManifestData(TypedDict):
    """Serialized fields that define a portable corpus snapshot."""

    config_name: str
    snapshot_id: str
    created_at: str
    producer_version: str
    contract: EmbeddingContractData
    namespace_serialization: str
    namespace_selectors: NamespaceSelectors
    source: CorpusSourceData
    row_count: int
    source_records_examined: int
    shards: list[PortableShardData]
    calibration_ranges: str | None
    producer_provenance: Mapping[str, object]
    format_version: int
    row_order: Literal["native-relative"]


class _PortableManifestRequiredFields(TypedDict):
    """Required fields passed to the portable manifest dataclass."""

    config_name: str
    snapshot_id: str
    created_at: str
    producer_version: str
    contract: EmbeddingContract
    namespace_serialization: str
    namespace_selectors: NamespaceSelectors
    source: CorpusSource
    row_count: int
    source_records_examined: int
    shards: tuple[PortableShard, ...]
    format_version: int
    row_order: Literal["native-relative"]


class _PortableManifestConstructor(_PortableManifestRequiredFields, total=False):
    """Typed constructor input for an already-decoded manifest mapping."""

    calibration_ranges: str | None
    producer_provenance: Mapping[str, object]


def _relative_path(value: str) -> str:
    """Return a safe relative path used by a portable artifact.

    :param str value: Artifact path from the manifest.
    :return str: Validated portable relative path.
    """
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
    namespace_selectors: NamespaceSelectors
    source: CorpusSource
    row_count: int
    source_records_examined: int
    shards: tuple[PortableShard, ...]
    calibration_ranges: str | None = None
    producer_provenance: Mapping[str, object] = field(default_factory=dict)
    format_version: int = FORMAT_VERSION
    row_order: Literal["native-relative"] = "native-relative"

    @property
    def artifact_paths(self) -> list[str]:
        """Return only the payloads explicitly owned by this configuration."""
        paths = [shard.path for shard in self.shards]
        if self.calibration_ranges is not None:
            paths.append(self.calibration_ranges)
        return paths

    def to_dict(self) -> PortableManifestData:
        """Return the manifest fields in their portable mapping form."""
        return {
            "config_name": self.config_name,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "producer_version": self.producer_version,
            "contract": self.contract.to_dict(),
            "namespace_serialization": self.namespace_serialization,
            "namespace_selectors": deepcopy(self.namespace_selectors),
            "source": self.source.to_dict(),
            "row_count": self.row_count,
            "source_records_examined": self.source_records_examined,
            "shards": [
                {
                    "path": shard.path,
                    "first_row": shard.first_row,
                    "row_count": shard.row_count,
                }
                for shard in self.shards
            ],
            "calibration_ranges": self.calibration_ranges,
            "producer_provenance": deepcopy(self.producer_provenance),
            "format_version": self.format_version,
            "row_order": self.row_order,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PortableManifest:
        """Parse a serialized manifest and validate its snapshot contract.

        :param Mapping[str, object] value: Raw serialized manifest mapping.
        :return PortableManifest: Validated snapshot manifest.
        """
        if "format_version" not in value or "row_order" not in value:
            raise ValueError("Portable manifest requires format and row-order versions")
        if value["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported portable format version")
        if value["row_order"] != "native-relative":
            raise ValueError("Unsupported portable row-order convention")
        try:
            contract_data = value["contract"]
            source_data = value["source"]
            shard_data = value["shards"]
            if (
                not isinstance(contract_data, Mapping)
                or not isinstance(source_data, Mapping)
                or not isinstance(shard_data, Iterable)
            ):
                raise TypeError("Portable manifest fields have invalid mapping shapes")
            contract_values = cast(EmbeddingContractData, contract_data)
            source_values = cast(CorpusSourceData, source_data)
            shard_values: Iterable[object] = shard_data
            shards: list[PortableShard] = []
            for item in shard_values:
                if not isinstance(item, Mapping):
                    raise TypeError("Portable shard fields must be mappings")
                shards.append(PortableShard(**cast(PortableShardData, item)))
            manifest_data: dict[str, object] = {
                **value,
                "contract": EmbeddingContract(**contract_values),
                "source": CorpusSource(**source_values),
                "shards": tuple(shards),
            }
            # Construction and the checks below validate this decoded mapping.
            manifest = cls(**cast(_PortableManifestConstructor, manifest_data))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid portable manifest: {exc}") from exc
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
    """Check that source and namespace fields agree with the vector contract.

    :param PortableManifest manifest: Manifest whose semantic fields are checked.
    :raises ValueError: If the source or namespace is incompatible.
    """
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
    if not isinstance(contract.model_revision, str) or not re.fullmatch(
        r"[0-9a-fA-F]{40}", contract.model_revision
    ):
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
    """Check that the embedding shape and stored dtype are supported.

    :param EmbeddingContract contract: Contract to validate.
    :raises ValueError: If the shape or dtype is unsupported.
    """
    if contract.stored_dtype not in {"int8", "float32"} or (
        type(contract.dimension) is not int or contract.dimension <= 0
    ):
        raise ValueError("Unsupported embedding shape or stored dtype")


def load_manifest(path: Path | str) -> PortableManifest:
    """Read and validate a manifest from a directory or ``index.json`` file.

    :param Path | str path: Export directory or manifest path.
    :return PortableManifest: Validated snapshot manifest.
    """
    path = Path(path)
    if path.is_dir():
        path /= "index.json"
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Portable manifest must be a JSON object")
    manifest_data: Mapping[str, object] = value
    return PortableManifest.from_dict(manifest_data)


def portable_schema(contract: EmbeddingContract) -> pa.Schema:
    """Build the Arrow schema for one stored embedding contract.

    :param EmbeddingContract contract: Vector dtype and dimension for the schema.
    :return pa.Schema: Ordered fields for portable corpus rows and embeddings.
    """
    import pyarrow as pa

    _validate_contract(contract)
    value_type = pa.int8() if contract.stored_dtype == "int8" else pa.float32()
    fields: list[pa.Field[pa.DataType]] = [
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
        pa.field("embedding", pa.list_(value_type, contract.dimension), nullable=False),
    ]
    return pa.schema(fields)


def _validate_vectors(
    vectors: StoredEmbedding,
    contract: EmbeddingContract,
    rows: int,
) -> None:
    """Check a stored-vector batch against its contract and row count.

    :param StoredEmbedding vectors: Stored vector matrix.
    :param EmbeddingContract contract: Contract that defines dtype and width.
    :param int rows: Expected number of vectors.
    :raises ValueError: If the batch does not match the contract.
    """
    if vectors.dtype != np.dtype(contract.stored_dtype):
        raise ValueError("Stored vector dtype differs from the snapshot contract")
    if vectors.shape != (rows, contract.dimension):
        raise ValueError("Stored vector shape differs from the snapshot contract")
    if contract.stored_dtype == "float32" and not np.isfinite(vectors).all():
        raise ValueError("Stored float32 vectors must contain only finite values")


def _validate_ranges(ranges: NDArray[np.float32], contract: EmbeddingContract) -> None:
    """Check that float32 calibration ranges match the embedding dimension.

    :param NDArray[np.float32] ranges: Lower and upper range for each coordinate.
    :param EmbeddingContract contract: Contract that defines vector width.
    :raises ValueError: If the range matrix is invalid.
    """
    if (
        ranges.dtype != np.dtype("float32")
        or ranges.shape != (2, contract.dimension)
        or not np.isfinite(ranges).all()
        or not np.all(ranges[0] < ranges[1])
    ):
        raise ValueError("Invalid float32 calibration ranges")


def _record_batch(
    records: Sequence[ActiveCorpusRecord],
    vectors: StoredEmbedding,
    contract: EmbeddingContract,
    first_row: int,
) -> pa.RecordBatch:
    """Create one Arrow batch with portable fields and ordered row numbers.

    :param Sequence[ActiveCorpusRecord] records: Corpus rows in native row order.
    :param StoredEmbedding vectors: Stored embeddings for those rows.
    :param EmbeddingContract contract: Vector dtype and dimension.
    :param int first_row: Portable row number assigned to the first record.
    :return pa.RecordBatch: Typed Arrow rows for one Parquet write.
    """
    import pyarrow as pa

    _validate_vectors(vectors, contract, len(records))
    schema = portable_schema(contract)
    columns: list[pa.Array[pa.Scalar[pa.DataType]]] = []
    for column in schema:
        array: pa.Array[pa.Scalar[pa.DataType]]
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
    cache: EmbeddingCache,
    output_dir: Path | str,
    *,
    config_name: str | None = None,
    shard_rows: int = 65_536,
    batch_rows: int = 8_192,
) -> PortableManifest:
    """Write a bounded portable snapshot from a complete native corpus.

    :param EmbeddingCache cache: Native cache to export under its snapshot lock.
    :param Path | str output_dir: New directory that receives the manifest and shards.
    :param str | None config_name: Published name; defaults to the stored profile, dimension, and dtype.
    :param int shard_rows: Maximum data rows per Parquet shard.
    :param int batch_rows: Native rows read and converted per batch.
    :return PortableManifest: Manifest describing the completed export.
    """
    import pyarrow.parquet as pq

    if shard_rows < 1 or batch_rows < 1:
        raise ValueError("Shard and batch row counts must be positive")
    output_dir = Path(output_dir)
    writer = None
    shards: list[PortableShard] = []
    next_row = 0
    shard_count = 0
    shard_start = 0
    shard_path = ""
    try:
        with cache.export_snapshot_view() as view:
            output_dir.mkdir(parents=True, exist_ok=False)
            (output_dir / "data").mkdir()
            if config_name is None:
                config_name = f"{view.contract.profile_key}-{view.contract.dimension}-{view.contract.stored_dtype}"
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
                            # PyArrow accepts a column-name list; its stub types this as bool.
                            use_dictionary=["venue", "year"],  # type: ignore[arg-type]
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
                # The int8 view carries calibration ranges by construction.
                ranges = cast(NDArray[np.float32], view.calibration_ranges)
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
                producer_provenance={
                    **view.producer_provenance,
                    "export_runtime_versions": RuntimeVersions(
                        python=platform.python_version(),
                        numpy=version("numpy"),
                        h5py=version("h5py"),
                        pyarrow=version("pyarrow"),
                    ),
                },
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
) -> Iterator[tuple[int, list[CorpusRecord], StoredEmbedding]]:
    """Validate ordered shards and yield bounded stored-vector batches.

    :param Path | str snapshot_dir: Directory containing the declared shard files.
    :param PortableManifest manifest: Validated snapshot description.
    :param int batch_rows: Maximum records yielded from each Parquet batch.
    :return Iterator[tuple[int, list[CorpusRecord], StoredEmbedding]]: Batches of
        first row number, corpus records, and their stored vectors.
    """

    if batch_rows < 1:
        raise ValueError("Batch row count must be positive")
    with TemporaryDirectory(prefix="citemesh-portable-") as temporary:
        with closing(sqlite3.connect(Path(temporary) / "ids.sqlite")) as seen:
            seen.execute("CREATE TABLE ids (paper_id TEXT PRIMARY KEY)")
            yield from _iter_snapshot_batches(snapshot_dir, manifest, batch_rows, seen)


def _iter_snapshot_batches(
    snapshot_dir: Path | str,
    manifest: PortableManifest,
    batch_rows: int,
    seen: sqlite3.Connection,
) -> Iterator[tuple[int, list[CorpusRecord], StoredEmbedding]]:
    """Read, validate, and yield rows from each declared Parquet shard.

    :param Path | str snapshot_dir: Directory containing the declared shard files.
    :param PortableManifest manifest: Validated snapshot description.
    :param int batch_rows: Maximum records read from each Parquet batch.
    :param sqlite3.Connection seen: Temporary index for duplicate paper IDs.
    :return Iterator[tuple[int, list[CorpusRecord], StoredEmbedding]]: Batches of
        first row number, corpus records, and their stored vectors.
    """
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
                # The parquet schema was checked against this contract above.
                embedding = cast(
                    "pa.FixedSizeListArray[pa.DataType, int]",
                    batch.column(schema.get_field_index("embedding")),
                )
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
                # The selected columns use the validated portable row schema.
                records = cast(
                    list[CorpusRecord],
                    batch.select(schema.names[1:-1]).to_pylist(),
                )
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


def validate_snapshot(
    snapshot_dir: Path | str, *, batch_rows: int = 8_192
) -> PortableManifest:
    """Validate every declared payload before publication.

    :param Path | str snapshot_dir: Export directory containing ``index.json``.
    :param int batch_rows: Maximum Parquet rows read at once.
    :return PortableManifest: Validated manifest for the snapshot.
    """
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
    publication: PublicationOrigin | None = None,
    batch_rows: int = 8_192,
) -> EmbeddingCache:
    """Install a portable snapshot into a native cache namespace.

    :param Path | str snapshot_dir: Directory containing the portable manifest.
    :param Path | None destination: Native cache directory, when explicitly set.
    :param PublicationOrigin publication: Bucket and snapshot identity, if published.
    :param int batch_rows: Maximum records imported at once.
    :return EmbeddingCache: Cache populated from the snapshot.
    """
    from .lifecycle import import_destination_session

    snapshot_dir = Path(snapshot_dir)
    manifest = load_manifest(snapshot_dir)
    ranges = None
    if manifest.calibration_ranges is not None:
        ranges = np.load(snapshot_dir / manifest.calibration_ranges, allow_pickle=False)
        _validate_ranges(ranges, manifest.contract)
    identity = manifest_import_identity(manifest)
    if publication is None:
        origin: CorpusOrigin = {"snapshot_id": manifest.snapshot_id}
    else:
        origin = publication.copy()
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


def manifest_import_identity(manifest: PortableManifest) -> PortableImportIdentity:
    """Return the manifest fields that identify an import operation.

    :param PortableManifest manifest: Snapshot being imported.
    :return PortableImportIdentity: Stable identity used to recognize a resumed import.
    """
    return {
        "snapshot_id": manifest.snapshot_id,
        "contract": manifest.contract.to_dict(),
        "source": manifest.source.to_dict(),
        "shards": [
            {
                "path": shard.path,
                "first_row": shard.first_row,
                "row_count": shard.row_count,
            }
            for shard in manifest.shards
        ],
    }
