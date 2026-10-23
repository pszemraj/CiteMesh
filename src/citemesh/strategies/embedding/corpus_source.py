"""Resolve and iterate one immutable upstream corpus snapshot."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path

from citemesh.core.paper_ids import encode_arxiv_id_chronology_key
from citemesh.data.embedding_contract import CorpusSource

from . import deps
from .records import _extract_dataset_paper_metadata


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


def _source_records(source: CorpusSource, stats: dict | None = None) -> Iterator[dict]:
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
        yield metadata


def iter_source_records(
    source: CorpusSource, stats: dict | None = None
) -> Iterator[dict]:
    """Yield full normalized records, retaining their original split ordinals.

    Newest-N selection stages normalized records in a temporary SQLite file so
    canonical duplicates cannot consume the cap and source size cannot exhaust
    memory. Full-corpus reconciliation uses the native pending-membership table.
    """
    if source.selection == "all":
        yield from _source_records(source, stats)
        return
    with tempfile.TemporaryDirectory(prefix="citemesh-source-") as temporary:
        with sqlite3.connect(Path(temporary) / "selection.sqlite3") as connection:
            connection.execute(
                "CREATE TABLE records (paper_id TEXT PRIMARY KEY, ordinal INTEGER, "
                "chronology INTEGER, payload TEXT)"
            )
            for record in _source_records(source, stats):
                payload = json.dumps(
                    {k: v for k, v in record.items() if k != "source_ordinal"},
                    sort_keys=True,
                )
                previous = connection.execute(
                    "SELECT payload FROM records WHERE paper_id=?",
                    (record["paper_id"],),
                ).fetchone()
                if previous is not None:
                    if previous[0] != payload:
                        raise ValueError(
                            f"Conflicting canonical source ID: {record['paper_id']}"
                        )
                    continue
                connection.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?)",
                    (
                        record["paper_id"],
                        record["source_ordinal"],
                        record["chronology_key"],
                        payload,
                    ),
                )
            connection.commit()
            for ordinal, payload in connection.execute(
                "SELECT ordinal, payload FROM records ORDER BY chronology IS NULL, "
                "chronology DESC, CASE WHEN chronology IS NOT NULL THEN ordinal END DESC, "
                "ordinal ASC LIMIT ?",
                (source.limit,),
            ):
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
