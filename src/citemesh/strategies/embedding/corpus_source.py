"""Resolve and read one immutable upstream corpus snapshot.

A pinned source is read one of two ways, with one normalization and one
duplicate rule. :func:`iter_source_records` streams rows once in source order
and resolves repeated IDs in a temporary SQLite table; refresh uses it to
reconcile a complete cache with a newer revision. :func:`prepare_source_selection`
downloads the revision as Arrow, normalizes it with a cached, parallel
``Dataset.map`` and resolves repeated IDs on the columns; refresh uses it
whenever the whole source has to be processed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import tempfile
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import closing
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import numpy.typing as npt

from citemesh.core.paper_ids import encode_arxiv_id_chronology_key
from citemesh.data.embedding_cache.models import CorpusSourceRecord, PaperMetadata
from citemesh.data.embedding_contract import CorpusSource, newest_corpus_order_sql

from . import deps
from .records import (
    DatasetRecord,
    _dataset_record_raw_paper_id,
    _extract_dataset_paper_metadata,
)

if TYPE_CHECKING:
    import pyarrow as pa

logger = logging.getLogger(__name__)

PARALLEL_SOURCE_MIN_ROWS = 50_000
"""Smallest source whose rows are normalized in worker processes."""

_SourcePriority = tuple[str, int, str]
_RECORD_COLUMNS = (
    "paper_id",
    "title",
    "abstract",
    "venue",
    "arxiv_id",
    "doi",
    "year",
    "authors",
    "categories",
    "chronology_key",
    "source_ordinal",
)


class SourceScanStats(TypedDict):
    """Counts recorded while iterating an upstream source."""

    source_records_examined: int


class _SourceRecordPayload(PaperMetadata):
    """Source fields stored in SQLite before restoring the selected ordinal."""

    paper_id: str
    chronology_key: int | None


def _source_timestamp(value: object) -> str:
    """Normalize a source timestamp to a UTC ISO value.

    :param value: ISO date, date-time, or email-style timestamp.
    :return: UTC timestamp with microsecond precision, or an empty string.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        value = value.isoformat()
    if isinstance(value, str):
        raw_timestamp = value
        try:
            value = datetime.fromisoformat(raw_timestamp.strip().replace("Z", "+00:00"))
        except ValueError:
            try:
                value = parsedate_to_datetime(raw_timestamp)
            except (ValueError, TypeError, IndexError):
                return ""
    if not isinstance(value, datetime):
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _source_priority(record: DatasetRecord) -> _SourcePriority:
    """Return update time, version number, and version creation time.

    :param record: Raw dataset row with source version metadata.
    :return: Values used to choose the newest duplicate record.
    """
    versions: list[tuple[int, str]] = []
    raw_versions = record.get("versions")
    if raw_versions and not isinstance(raw_versions, Iterable):
        raise TypeError(f"'{type(raw_versions).__name__}' object is not iterable")
    source_versions: Iterable[object] = ()
    if raw_versions and isinstance(raw_versions, Iterable):
        source_versions = raw_versions
    for version in source_versions:
        if not isinstance(version, dict):
            continue
        match = re.fullmatch(r"v([0-9]+)", str(version.get("version", "")))
        if match:
            versions.append((int(match[1]), _source_timestamp(version.get("created"))))
    suffix = re.search(
        r"v([0-9]+)$", str(_dataset_record_raw_paper_id(record) or "").strip()
    )
    if suffix:
        versions.append((int(suffix[1]), ""))
    number, created = max(versions, default=(0, ""))
    return _source_timestamp(record.get("update_date")), number, created


def _source_record(raw_record: DatasetRecord, ordinal: int) -> CorpusSourceRecord:
    """Normalize one raw source row with its ordinal and chronology key.

    :param raw_record: Raw dataset row.
    :param ordinal: Position of the row in the pinned split.
    :return: Normalized record shared by both source readers.
    """
    extracted = _extract_dataset_paper_metadata(raw_record, ordinal)
    return {
        **extracted,
        "source_ordinal": ordinal,
        "chronology_key": encode_arxiv_id_chronology_key(extracted["paper_id"]),
    }


def _selection_payload(record: CorpusSourceRecord) -> str:
    """Serialize normalized content for the stable duplicate tie-break.

    :param record: Normalized source record.
    :return: Compact sorted JSON of every field except the source ordinal.
    """
    return json.dumps(
        {key: value for key, value in record.items() if key != "source_ordinal"},
        sort_keys=True,
        separators=(",", ":"),
    )


def _compare_duplicate(
    priority: _SourcePriority,
    payload: str,
    previous_priority: _SourcePriority,
    previous_payload: str,
) -> tuple[bool, bool]:
    """Compare a repeated source row with the current winner for its paper ID.

    A newer ``(updated, version, created)`` priority wins. Equal priorities fall
    back to the normalized-content ordering, so an identical row never replaces
    the first one.

    :param priority: Priority of the repeated row.
    :param payload: Normalized content of the repeated row.
    :param previous_priority: Priority of the current winner.
    :param previous_payload: Normalized content of the current winner.
    :return: Whether the repeated row replaces the winner, and whether the two
        rows conflict under equal revision metadata.
    """
    return (
        (*priority, payload) > (*previous_priority, previous_payload),
        priority == previous_priority and payload != previous_payload,
    )


def _log_duplicates(duplicates: int, ambiguous: int) -> None:
    """Report how many repeated source rows were resolved.

    :param duplicates: Rows whose paper ID was already seen.
    :param ambiguous: Conflicting rows with equal revision metadata.
    """
    if duplicates:
        logger.info(
            "Resolved %s repeated source rows before corpus selection", duplicates
        )
    if ambiguous:
        logger.warning(
            "%s conflicting source rows have equal revision metadata; "
            "using stable normalized-content ordering to choose their records",
            ambiguous,
        )


def resolve_source_snapshot(
    source: CorpusSource, revision: str = "main"
) -> CorpusSource:
    """Resolve a source reference to its immutable commit.

    :param source: Dataset repository and split to resolve.
    :param revision: Branch, tag, or commit requested from the dataset host.
    :return: Source selection pinned to the resolved commit.
    :raises OSError: If the dataset host cannot resolve the requested revision.
    """
    from dataclasses import replace

    from huggingface_hub.utils import httpx

    try:
        info = (
            deps._import_huggingface_hub_module()
            .HfApi()
            .dataset_info(repo_id=source.repo_id, revision=revision)
        )
    except httpx.HTTPError as exc:
        raise OSError(
            f"Cannot resolve dataset {source.repo_id}@{revision}: {exc}"
        ) from exc
    return replace(source, revision=info.sha)


def _source_records(
    source: CorpusSource, stats: SourceScanStats | None = None
) -> Iterator[tuple[CorpusSourceRecord, _SourcePriority]]:
    """Yield normalized rows with their source-version priority.

    :param source: Pinned dataset selection to read.
    :param stats: Optional counter for raw rows examined.
    Yields:
        Normalized source record and its duplicate-selection priority.
    :raises OSError: If opening or iterating the dataset fails in transport.
    """
    from huggingface_hub.utils import httpx

    kwargs = {
        "split": source.split,
        "revision": source.revision,
        "streaming": "[" not in source.split,
    }
    if source.config_name is not None:
        kwargs["name"] = source.config_name
    try:
        dataset = deps._import_datasets_module().load_dataset(source.repo_id, **kwargs)
        for ordinal, raw_record in enumerate(dataset):
            if stats is not None:
                stats["source_records_examined"] = ordinal + 1
            yield _source_record(raw_record, ordinal), _source_priority(raw_record)
    except httpx.HTTPError as exc:
        raise OSError(
            f"Cannot read dataset {source.repo_id}@{source.revision}: {exc}"
        ) from exc


def iter_source_records(
    source: CorpusSource, stats: SourceScanStats | None = None
) -> Iterator[CorpusSourceRecord]:
    """Stream one selected source row per paper ID.

    Newer update and version metadata win; equal-priority conflicts use stable
    normalized content ordering, while identical duplicates keep their first row.
    Sliced splits are read without streaming.

    :param source: Pinned dataset selection to read.
    :param stats: Optional counter for raw rows examined.
    Yields:
        Normalized row with its source ordinal and chronology key.
    :raises OSError: If opening or iterating the dataset fails in transport.
    """
    with (
        tempfile.TemporaryDirectory(prefix="citemesh-source-") as temporary,
        # The refresh loop reads the first batch on its own thread and the rest
        # on a worker, one read at a time.
        closing(
            sqlite3.connect(
                Path(temporary) / "selection.sqlite3", check_same_thread=False
            )
        ) as connection,
    ):
        connection.execute(
            "CREATE TABLE records (paper_id TEXT PRIMARY KEY, ordinal INTEGER, "
            "chronology INTEGER, payload TEXT, updated TEXT, version INTEGER, created TEXT)"
        )
        duplicates = ambiguous = 0
        for record, priority in _source_records(source, stats):
            payload = _selection_payload(record)
            inserted = connection.execute(
                "INSERT OR IGNORE INTO records VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record["paper_id"],
                    record["source_ordinal"],
                    record["chronology_key"],
                    payload,
                    *priority,
                ),
            )
            if inserted.rowcount:
                continue
            duplicates += 1
            previous = connection.execute(
                "SELECT payload,updated,version,created FROM records WHERE paper_id=?",
                (record["paper_id"],),
            ).fetchone()
            replaces, conflicting = _compare_duplicate(
                priority, payload, (previous[1], previous[2], previous[3]), previous[0]
            )
            ambiguous += conflicting
            if replaces:
                connection.execute(
                    "UPDATE records SET ordinal=?,chronology=?,payload=?,updated=?,"
                    "version=?,created=? WHERE paper_id=?",
                    (
                        record["source_ordinal"],
                        record["chronology_key"],
                        payload,
                        *priority,
                        record["paper_id"],
                    ),
                )
        connection.commit()
        _log_duplicates(duplicates, ambiguous)
        if source.selection == "all":
            query = "SELECT ordinal,payload FROM records ORDER BY ordinal"
            parameters: tuple[int | None, ...] = ()
        else:
            query = (
                "SELECT ordinal,payload FROM records ORDER BY "
                + newest_corpus_order_sql("chronology", "ordinal")
                + " LIMIT ?"
            )
            parameters = (source.limit,)
        for ordinal, payload in connection.execute(query, parameters):
            source_payload: _SourceRecordPayload = json.loads(payload)
            yield {**source_payload, "source_ordinal": ordinal}


def iter_source_batches(
    source: CorpusSource, batch_size: int, stats: SourceScanStats | None = None
) -> Generator[list[CorpusSourceRecord], None, None]:
    """Stream bounded batches of normalized source rows.

    :param source: Pinned dataset selection to read.
    :param batch_size: Maximum number of records in a batch.
    :param stats: Optional counter for raw rows examined.
    Yields:
        A batch of normalized source rows.
    :raises OSError: If opening or iterating the dataset fails in transport.
    """
    batch: list[CorpusSourceRecord] = []
    for record in iter_source_records(source, stats):
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _source_processes() -> int:
    """Return the worker count for downloading and normalizing a full source.

    :return: Half the logical CPUs, at least one, as build hydration uses.
    """
    return max(1, (os.cpu_count() or 1) // 2)


def _normalize_source_batch(
    batch: Mapping[str, Sequence[object]], indices: Sequence[int]
) -> dict[str, list[object]]:
    """Normalize one ``Dataset.map`` batch of raw rows into record columns.

    :param batch: Raw source columns.
    :param indices: Source ordinals of the batch rows.
    :return: Columns of the normalized records.
    """
    records = [
        _source_record(dict(zip(batch, values, strict=True)), ordinal)
        for values, ordinal in zip(zip(*batch.values()), indices, strict=True)
    ]
    return {
        column: [record.get(column) for record in records] for column in _RECORD_COLUMNS
    }


def _record_features(datasets_module: ModuleType) -> object:
    """Declare the normalized record columns for ``Dataset.map``.

    :param datasets_module: Imported ``datasets`` module.
    :return: ``datasets.Features`` for :data:`_RECORD_COLUMNS`.
    """
    text = datasets_module.Value("string")
    integer = datasets_module.Value("int64")
    texts = datasets_module.Sequence(datasets_module.Value("string"))
    return datasets_module.Features(
        {
            "paper_id": text,
            "title": text,
            "abstract": text,
            "venue": text,
            "arxiv_id": text,
            "doi": text,
            "year": integer,
            "authors": texts,
            "categories": texts,
            "chronology_key": integer,
            "source_ordinal": integer,
        }
    )


def _records_from_table(table: pa.Table) -> list[CorpusSourceRecord]:
    """Convert normalized Arrow rows back into source records.

    :param table: Rows with the :data:`_RECORD_COLUMNS` schema.
    :return: Records in table order.
    """
    columns = table.to_pydict()
    return [
        {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "venue": venue,
            "arxiv_id": arxiv_id,
            "doi": doi,
            "year": year,
            "authors": authors,
            "categories": categories,
            "chronology_key": chronology_key,
            "source_ordinal": source_ordinal,
        }
        for (
            paper_id,
            title,
            abstract,
            venue,
            arxiv_id,
            doi,
            year,
            authors,
            categories,
            chronology_key,
            source_ordinal,
        ) in zip(*(columns[column] for column in _RECORD_COLUMNS), strict=True)
    ]


class _ArrowRows:
    """Gather rows of a chunked Arrow table by position.

    ``Table.take`` concatenates every chunk of the table first, which costs a
    copy of the whole source per batch; gathering chunk by chunk does not.
    """

    def __init__(self, table: pa.Table) -> None:
        """Index the record batches of a table.

        :param table: Table whose rows to gather.
        """
        self._schema = table.schema
        self._batches = table.to_batches()
        lengths = [batch.num_rows for batch in self._batches]
        self._starts = np.cumsum([0, *lengths[:-1]], dtype=np.int64)

    def take(self, ordinals: npt.NDArray[np.int64]) -> pa.Table:
        """Return rows by position, in the order requested.

        :param ordinals: Row positions to gather.
        :return: The rows as a table.
        """
        import pyarrow as pa

        if not len(ordinals):
            return self._schema.empty_table()
        chunks = np.searchsorted(self._starts, ordinals, side="right") - 1
        by_chunk = np.argsort(chunks, kind="stable")
        groups = np.split(by_chunk, np.flatnonzero(np.diff(chunks[by_chunk])) + 1)
        gathered = pa.Table.from_batches(
            [
                self._batches[chunks[group[0]]].take(
                    ordinals[group] - self._starts[chunks[group[0]]]
                )
                for group in groups
            ],
            schema=self._schema,
        )
        return gathered.take(np.argsort(by_chunk))


def _resolve_column_duplicates(
    rows: _ArrowRows,
    ids: pa.ChunkedArray[pa.StringScalar],
    priority: Callable[[int], _SourcePriority],
) -> npt.NDArray[np.bool_]:
    """Choose one winner per paper ID, running Python only on repeated IDs.

    Rows of each repeated ID are compared in source order with the streaming
    reader's rule, so both readers keep the same winners.

    :param rows: Normalized records, one per source row.
    :param ids: The ``paper_id`` column of those records.
    :param priority: Returns the source priority of a raw row by ordinal.
    :return: Mask over source rows marking each paper ID's winner.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    winners = np.ones(len(ids), dtype=bool)
    counts = pc.value_counts(ids)
    repeated = counts.field("values").filter(
        pc.greater(counts.field("counts"), pa.scalar(1))
    )
    if not len(repeated):
        return winners
    ordinals = np.flatnonzero(
        pc.is_in(ids, value_set=repeated).to_numpy(zero_copy_only=False)
    )
    current: dict[str, tuple[int, _SourcePriority, str]] = {}
    duplicates = ambiguous = 0
    for record in _records_from_table(rows.take(ordinals)):
        candidate = (
            record["source_ordinal"],
            priority(record["source_ordinal"]),
            _selection_payload(record),
        )
        previous = current.get(record["paper_id"])
        if previous is None:
            current[record["paper_id"]] = candidate
            continue
        duplicates += 1
        replaces, conflicting = _compare_duplicate(
            candidate[1], candidate[2], previous[1], previous[2]
        )
        ambiguous += conflicting
        if replaces:
            winners[previous[0]] = False
            current[record["paper_id"]] = candidate
        else:
            winners[candidate[0]] = False
    _log_duplicates(duplicates, ambiguous)
    return winners


def _selection_order(
    source: CorpusSource, table: pa.Table, winners: npt.NDArray[np.bool_]
) -> npt.NDArray[np.int64]:
    """Order the winning rows as the source selection requires.

    :param source: Pinned selection, either ``all`` or ``newest`` with a limit.
    :param table: Normalized records, one per source row.
    :param winners: Mask of each paper ID's winning row.
    :return: Selected source ordinals in corpus order.
    """
    ordinals = np.flatnonzero(winners).astype(np.int64)
    if source.selection == "all":
        return ordinals
    chronology = table.column("chronology_key").to_pylist()
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("CREATE TABLE records (ordinal INTEGER, chronology INTEGER)")
        connection.executemany(
            "INSERT INTO records VALUES (?, ?)",
            ((ordinal, chronology[ordinal]) for ordinal in ordinals.tolist()),
        )
        rows = connection.execute(
            "SELECT ordinal FROM records ORDER BY "
            + newest_corpus_order_sql("chronology", "ordinal")
            + " LIMIT ?",
            (source.limit,),
        )
        return np.fromiter((row[0] for row in rows), dtype=np.int64)


class SourceSelection:
    """Selected rows of one pinned source, read in batches from Arrow columns.

    :ivar int source_rows: Raw source rows examined, repeated IDs included.
    :ivar npt.NDArray[np.int64] order: Selected source ordinals in corpus order.
    """

    def __init__(self, table: pa.Table, order: npt.NDArray[np.int64]) -> None:
        """Wrap normalized source rows and the ordinals selected from them.

        :param table: Normalized records, one per source row.
        :param order: Selected source ordinals in corpus order.
        """
        self._rows = _ArrowRows(table)
        self._ids = _ArrowRows(table.select(["paper_id"]))
        self.order = order
        self.source_rows = table.num_rows
        self._selected = np.zeros(self.source_rows, dtype=bool)
        self._selected[order] = True

    def __len__(self) -> int:
        """Return the number of selected records."""
        return len(self.order)

    def records(self, ordinals: npt.NDArray[np.int64]) -> list[CorpusSourceRecord]:
        """Read normalized records by source ordinal.

        :param ordinals: Source ordinals to read.
        :return: Records in the order of ``ordinals``.
        """
        return _records_from_table(self._rows.take(ordinals))

    def batches(
        self, batch_size: int, skip: npt.NDArray[np.bool_] | None = None
    ) -> Generator[list[CorpusSourceRecord], None, None]:
        """Yield selected records in corpus order.

        :param batch_size: Maximum number of records in a batch.
        :param skip: Optional mask over source ordinals whose records to leave out.
        Yields:
            A batch of normalized source rows.
        """
        order = self.order if skip is None else self.order[~skip[self.order]]
        for start in range(0, len(order), batch_size):
            yield self.records(order[start : start + batch_size])

    def selects(
        self, paper_ids: Sequence[str], ordinals: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.bool_]:
        """Report which paper ID and ordinal pairs name a selected record.

        :param paper_ids: Paper IDs, one per ordinal.
        :param ordinals: Source ordinals recorded with the paper IDs.
        :return: Mask of the pairs whose ordinal is selected for that paper ID.
        """
        selected = (ordinals >= 0) & (ordinals < self.source_rows)
        selected[selected] = self._selected[ordinals[selected]]
        positions = np.flatnonzero(selected)
        source_ids = self._ids.take(ordinals[positions]).column("paper_id").to_pylist()
        selected[positions] = [
            paper_ids[position] == source_id
            for position, source_id in zip(positions.tolist(), source_ids, strict=True)
        ]
        return selected


def prepare_source_selection(
    source: CorpusSource, workdir: Path, stats: SourceScanStats | None = None
) -> SourceSelection:
    """Download a pinned source and select one row per paper ID on its columns.

    The revision is prepared as Arrow by ``datasets``, which keeps it per
    revision. Rows are normalized by a batched ``Dataset.map`` into ``workdir``,
    in worker processes from :data:`PARALLEL_SOURCE_MIN_ROWS` rows, so every
    pass normalizes with the running code and leaves nothing behind once the
    caller removes ``workdir``. Repeated IDs and the selection order follow
    :func:`iter_source_records` exactly.

    :param source: Pinned dataset selection to read.
    :param workdir: Directory for the normalized rows; it must outlive the selection.
    :param stats: Optional counter for raw rows examined.
    :return: The selected rows, readable in batches.
    :raises OSError: If downloading the dataset fails in transport.
    """
    from huggingface_hub.utils import httpx

    datasets_module = deps._import_datasets_module()
    processes = _source_processes()
    kwargs: dict[str, str | bool | int] = {
        "split": source.split,
        "revision": source.revision,
        "streaming": False,
        "num_proc": processes,
    }
    if source.config_name is not None:
        kwargs["name"] = source.config_name
    try:
        dataset = datasets_module.load_dataset(source.repo_id, **kwargs)
    except httpx.HTTPError as exc:
        raise OSError(
            f"Cannot read dataset {source.repo_id}@{source.revision}: {exc}"
        ) from exc
    rows = len(dataset)
    features = _record_features(datasets_module)
    # Mapping an empty dataset keeps its (absent) columns, not the features.
    normalized = (
        dataset.map(
            _normalize_source_batch,
            batched=True,
            with_indices=True,
            cache_file_name=str(workdir / "normalized.arrow"),
            remove_columns=dataset.column_names,
            features=features,
            num_proc=processes if rows >= PARALLEL_SOURCE_MIN_ROWS else None,
            desc="Normalizing source rows",
        )
        if rows
        else datasets_module.Dataset.from_dict(
            {column: [] for column in _RECORD_COLUMNS}, features=features
        )
    )
    table: pa.Table = normalized.with_format("arrow")[:]
    if stats is not None:
        stats["source_records_examined"] = rows
    winners = _resolve_column_duplicates(
        _ArrowRows(table),
        table.column("paper_id"),
        lambda ordinal: _source_priority(dataset[ordinal]),
    )
    return SourceSelection(table, _selection_order(source, table, winners))
