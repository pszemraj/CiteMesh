"""Resolve and iterate one immutable upstream corpus snapshot."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import closing
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TypedDict, cast

from citemesh.core.paper_ids import encode_arxiv_id_chronology_key
from citemesh.data.embedding_cache.models import CorpusSourceRecord, PaperMetadata
from citemesh.data.embedding_contract import CorpusSource

from . import deps
from .records import (
    DatasetRecord,
    _dataset_record_raw_paper_id,
    _extract_dataset_paper_metadata,
)

logger = logging.getLogger(__name__)


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


def _source_priority(record: DatasetRecord) -> tuple[str, int, str]:
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


def resolve_source_snapshot(
    source: CorpusSource, revision: str = "main"
) -> CorpusSource:
    """Resolve a source reference to its immutable commit.

    :param source: Dataset repository and split to resolve.
    :param revision: Branch, tag, or commit requested from the dataset host.
    :return: Source selection pinned to the resolved commit.
    """
    from dataclasses import replace

    info = (
        deps._import_huggingface_hub_module()
        .HfApi()
        .dataset_info(repo_id=source.repo_id, revision=revision)
    )
    return replace(source, revision=info.sha)


def _source_records(
    source: CorpusSource, stats: SourceScanStats | None = None
) -> Iterator[tuple[CorpusSourceRecord, tuple[str, int, str]]]:
    """Yield normalized rows with their source-version priority.

    :param source: Pinned dataset selection to read.
    :param stats: Optional counter for raw rows examined.
    Yields:
        Normalized source record and its duplicate-selection priority.
    """
    kwargs = {
        "split": source.split,
        "revision": source.revision,
        "streaming": "[" not in source.split,
    }
    if source.config_name is not None:
        kwargs["name"] = source.config_name
    dataset = deps._import_datasets_module().load_dataset(source.repo_id, **kwargs)
    for ordinal, raw_record in enumerate(dataset):
        if stats is not None:
            stats["source_records_examined"] = ordinal + 1
        extracted = _extract_dataset_paper_metadata(raw_record, ordinal)
        metadata: CorpusSourceRecord = {
            **extracted,
            "source_ordinal": ordinal,
            "chronology_key": encode_arxiv_id_chronology_key(extracted["paper_id"]),
        }
        yield metadata, _source_priority(raw_record)


def iter_source_records(
    source: CorpusSource, stats: SourceScanStats | None = None
) -> Iterator[CorpusSourceRecord]:
    """Yield one selected source row per paper ID.

    Newer update and version metadata win; equal-priority conflicts use stable
    normalized content ordering, while identical duplicates keep their first row.

    :param source: Pinned dataset selection to read.
    :param stats: Optional counter for raw rows examined.
    Yields:
        Normalized row with its source ordinal and chronology key.
    """
    with (
        tempfile.TemporaryDirectory(prefix="citemesh-source-") as temporary,
        closing(sqlite3.connect(Path(temporary) / "selection.sqlite3")) as connection,
    ):
        connection.execute(
            "CREATE TABLE records (paper_id TEXT PRIMARY KEY, ordinal INTEGER, "
            "chronology INTEGER, payload TEXT, updated TEXT, version INTEGER, created TEXT)"
        )
        duplicates = ambiguous = 0
        for record, priority in _source_records(source, stats):
            payload = json.dumps(
                {k: v for k, v in record.items() if k != "source_ordinal"},
                sort_keys=True,
                separators=(",", ":"),
            )
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
            previous_priority = tuple(previous[1:])
            if priority == previous_priority and payload != previous[0]:
                ambiguous += 1
            if (*priority, payload) > (*previous_priority, previous[0]):
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
        if source.selection == "all":
            query = "SELECT ordinal,payload FROM records ORDER BY ordinal"
            parameters: tuple[int | None, ...] = ()
        else:
            query = (
                "SELECT ordinal,payload FROM records ORDER BY chronology IS NULL, "
                "chronology DESC, CASE WHEN chronology IS NOT NULL THEN ordinal END DESC, "
                "ordinal ASC LIMIT ?"
            )
            parameters = (source.limit,)
        for ordinal, payload in connection.execute(query, parameters):
            # The JSON row is serialized from the typed source record above,
            # with only source_ordinal removed before writing to SQLite.
            source_payload = cast(_SourceRecordPayload, json.loads(payload))
            yield {**source_payload, "source_ordinal": ordinal}


def iter_source_batches(
    source: CorpusSource, batch_size: int, stats: SourceScanStats | None = None
) -> Iterator[list[CorpusSourceRecord]]:
    """Yield bounded batches of normalized source rows.

    :param source: Pinned dataset selection to read.
    :param batch_size: Maximum number of records in a batch.
    :param stats: Optional counter for raw rows examined.
    Yields:
        A batch of normalized source rows.
    """
    batch: list[CorpusSourceRecord] = []
    for record in iter_source_records(source, stats):
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
