"""SQLite DDL/DML text, JSON column codecs, and row decoding helpers.

Owns the schema statements executed when a namespace is created, the templated
paper-row lookup query, the JSON list serializers used by the ``papers``
columns, the shared row decoder, and the batching helper that keeps ``IN``
clauses under SQLite's variable limit.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

PAPERS_TABLE_CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY,
        title TEXT,
        abstract TEXT,
        year INTEGER,
        text_hash TEXT,
        embedding_dim INTEGER,
        row_idx INTEGER,
        authors_json TEXT,
        categories_json TEXT,
        venue TEXT,
        arxiv_id TEXT,
        doi TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

REPLACEMENT_JOURNAL_TABLE_CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS replacement_journal (
        row_idx INTEGER PRIMARY KEY,
        embedding BLOB NOT NULL,
        embedding_width INTEGER NOT NULL,
        binary_embedding BLOB,
        binary_width INTEGER
    )
    """

PAPER_ROW_UPSERT_SQL = """
    INSERT OR REPLACE INTO papers
    (paper_id, title, abstract, year, text_hash, embedding_dim, row_idx,
     authors_json, categories_json, venue, arxiv_id, doi)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

PAPER_METADATA_REFRESH_SQL = """
    UPDATE papers
    SET title = ?, abstract = ?, year = ?, authors_json = ?, categories_json = ?,
        venue = ?, arxiv_id = ?, doi = ?
    WHERE paper_id = ?
    """

PAPER_ROW_QUERY_SQL_TEMPLATE = (
    "SELECT {columns}, "
    "(SELECT COUNT(*) FROM papers AS owners "
    "WHERE owners.row_idx = papers.row_idx) FROM papers "
    "WHERE {lookup_column} IN ({placeholders})"
)


def _metadata_table_create_sql() -> str:
    """Return SQL DDL used to create cache metadata table.

    :return str: SQL statement for ``cache_metadata`` table creation.
    """
    return """
    CREATE TABLE IF NOT EXISTS cache_metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """


def _safe_json_list(value: Any) -> str:
    """Serialize metadata list values as JSON arrays.

    :param Any value: Raw metadata value.
    :return str: JSON array string.
    """
    if isinstance(value, list):
        normalized = [str(item) for item in value if str(item).strip()]
        return json.dumps(normalized)
    return json.dumps([])


def _parse_json_list(value: str | None) -> list[str]:
    """Parse JSON list payload from SQLite metadata rows.

    :param Optional[str] value: Raw JSON string.
    :return List[str]: Parsed list payload.
    """
    if not value:
        return []

    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []

    if not isinstance(decoded, list):
        return []

    return [str(item) for item in decoded if str(item).strip()]


def _decode_paper_row(
    row: tuple[Any, ...],
    *,
    parse_json_lists: bool,
) -> dict[str, Any]:
    """Normalize one common paper row while retaining JSON shape choice.

    :param Tuple[Any, ...] row: Row returned by ``_query_paper_rows``.
    :param bool parse_json_lists: Decode authors/categories into lists when true.
    :return Dict[str, Any]: Normalized scalar metadata and selected JSON shape.
    """
    (
        paper_id,
        text_hash,
        row_idx,
        title,
        abstract,
        year,
        authors_json,
        categories_json,
        venue,
        arxiv_id,
        doi,
    ) = row
    decoded: dict[str, Any] = {
        "paper_id": str(paper_id),
        "text_hash": str(text_hash),
        "row_idx": int(row_idx) if row_idx is not None else None,
        "title": str(title or ""),
        "abstract": str(abstract or ""),
        "year": int(year) if year is not None else None,
        "venue": str(venue or ""),
        "arxiv_id": str(arxiv_id or ""),
        "doi": str(doi or ""),
    }
    if parse_json_lists:
        decoded["authors"] = _parse_json_list(authors_json)
        decoded["categories"] = _parse_json_list(categories_json)
    else:
        decoded["authors_json"] = str(authors_json or "")
        decoded["categories_json"] = str(categories_json or "")
    return decoded


def _chunked(values: Sequence[Any], chunk_size: int) -> Iterable[list[Any]]:
    """Yield fixed-size chunks from a sequence.

    :param Sequence[Any] values: Sequence to split into chunks.
    :param int chunk_size: Number of items per yielded chunk.
    :return Iterable[List[Any]]: Iterator over chunk lists.
    """
    for start in range(0, len(values), chunk_size):
        yield list(values[start : start + chunk_size])
