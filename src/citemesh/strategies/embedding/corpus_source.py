"""Resolve and iterate one immutable upstream corpus snapshot."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import closing
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from citemesh.core.paper_ids import encode_arxiv_id_chronology_key
from citemesh.data.embedding_contract import CorpusSource

from . import deps
from .records import _dataset_record_raw_paper_id, _extract_dataset_paper_metadata

logger = logging.getLogger(__name__)


def _source_timestamp(value) -> str:
    """Compare ISO dates and arXiv version timestamps in UTC."""
    if isinstance(value, date) and not isinstance(value, datetime):
        value = value.isoformat()
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            try:
                value = parsedate_to_datetime(value)
            except (ValueError, TypeError, IndexError):
                return ""
    if not isinstance(value, datetime):
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _source_priority(record: dict) -> tuple[str, int, str]:
    """Rank metadata updates first, then the latest numbered arXiv version."""
    versions = []
    for version in record.get("versions") or []:
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
    """Resolve a source reference once; every subsequent read uses its commit."""
    from dataclasses import replace

    info = (
        deps._import_huggingface_hub_module()
        .HfApi()
        .dataset_info(repo_id=source.repo_id, revision=revision)
    )
    return replace(source, revision=info.sha)


def _source_records(
    source: CorpusSource, stats: dict | None = None
) -> Iterator[tuple[dict, tuple[str, int, str]]]:
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
        metadata = _extract_dataset_paper_metadata(raw_record, ordinal)
        metadata["source_ordinal"] = ordinal
        metadata["chronology_key"] = encode_arxiv_id_chronology_key(
            metadata["paper_id"]
        )
        yield metadata, _source_priority(raw_record)


def iter_source_records(
    source: CorpusSource, stats: dict | None = None
) -> Iterator[dict]:
    """Resolve repeated IDs before yielding records with their source ordinals.

    All selections share a disk-backed staging pass. Newer metadata updates win,
    followed by version number and creation time. Conflicting equally recent
    records use sorted normalized JSON as a stable tie-break, without claiming
    either is newer. Identical equally recent duplicates keep their first ordinal.
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
            query, parameters = (
                "SELECT ordinal,payload FROM records ORDER BY ordinal",
                (),
            )
        else:
            query = (
                "SELECT ordinal,payload FROM records ORDER BY chronology IS NULL, "
                "chronology DESC, CASE WHEN chronology IS NOT NULL THEN ordinal END DESC, "
                "ordinal ASC LIMIT ?"
            )
            parameters = (source.limit,)
        for ordinal, payload in connection.execute(query, parameters):
            yield {**json.loads(payload), "source_ordinal": ordinal}


def iter_source_batches(
    source: CorpusSource, batch_size: int, stats: dict | None = None
) -> Iterator[list[dict]]:
    """Bound the live normalized record batch independently of corpus size."""
    batch: list[dict] = []
    for record in iter_source_records(source, stats):
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
