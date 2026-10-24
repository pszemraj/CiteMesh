"""SQLite DDL/DML text, JSON column codecs, and row decoding helpers.

Owns the schema statements executed when a namespace is created, the templated
paper-row lookup query, the JSON list serializers used by the ``papers``
columns, the shared row decoder, and the batching helper that keeps ``IN``
clauses under SQLite's variable limit.

``chronology_key`` is deliberately absent from the row-read path: it is an
aggregate-only column (the newest-submission watermark), and rows whose ID has
no derivable submission date — synthetic, DOI-canonical, or candidate-mode —
store ``NULL`` there.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Literal, TypedDict, TypeVar, overload

from .models import CachedPaperMetadata, CachedPaperRecord

SQLitePaperRow = tuple[
    str,
    str | None,
    int | None,
    str | None,
    str | None,
    int | float | str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]


class _PaperMetadataFields(TypedDict):
    """Normalized metadata scalar columns shared by persisted row shapes."""

    text_hash: str
    row_idx: int | None
    title: str
    abstract: str
    year: int | None
    venue: str
    arxiv_id: str
    doi: str


class SerializedPaperMetadata(_PaperMetadataFields):
    """Paper fields read from SQLite before authors and categories are decoded."""

    authors_json: str
    categories_json: str


class _PaperRowScalarFields(_PaperMetadataFields):
    """Normalized scalar columns with the primary paper ID."""

    paper_id: str


class SerializedPaperRow(_PaperRowScalarFields):
    """Serialized metadata row with its primary paper ID."""

    authors_json: str
    categories_json: str


class DecodedPaperRow(_PaperRowScalarFields):
    """Normalized SQLite row with authors and categories decoded."""

    authors: list[str]
    categories: list[str]


PaperMetadataRefreshValues = tuple[
    str,
    str,
    int | None,
    str,
    str,
    str,
    str,
    str,
    int | None,
    str,
]


_ValueT = TypeVar("_ValueT")

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
        chronology_key INTEGER,
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

CORPUS_MEMBERSHIP_TABLE_CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS corpus_membership (
        paper_id TEXT PRIMARY KEY,
        source_ordinal INTEGER NOT NULL
    )
    """

CORPUS_MEMBERSHIP_ORDINAL_INDEX_SQL = """
    CREATE INDEX IF NOT EXISTS corpus_membership_source_ordinal
    ON corpus_membership(source_ordinal)
    """

PAPER_ROW_UPSERT_SQL = """
    INSERT OR REPLACE INTO papers
    (paper_id, title, abstract, year, text_hash, embedding_dim, row_idx,
     authors_json, categories_json, venue, arxiv_id, doi, chronology_key)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

PAPER_METADATA_REFRESH_SQL = """
    UPDATE papers
    SET title = ?, abstract = ?, year = ?, authors_json = ?, categories_json = ?,
        venue = ?, arxiv_id = ?, doi = ?, chronology_key = ?
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


def _safe_json_list(value: object) -> str:
    """Serialize metadata list values as JSON arrays.

    :param object value: Raw metadata value; only lists are serialized.
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


@overload
def _decode_paper_row(
    row: SQLitePaperRow, *, parse_json_lists: Literal[True]
) -> DecodedPaperRow: ...


@overload
def _decode_paper_row(
    row: SQLitePaperRow, *, parse_json_lists: Literal[False]
) -> SerializedPaperRow: ...


@overload
def _decode_paper_row(
    row: SQLitePaperRow, *, parse_json_lists: bool
) -> DecodedPaperRow | SerializedPaperRow: ...


def _decode_paper_row(
    row: SQLitePaperRow,
    *,
    parse_json_lists: bool,
) -> DecodedPaperRow | SerializedPaperRow:
    """Normalize one common paper row while retaining JSON shape choice.

    :param SQLitePaperRow row: Row returned by ``_query_paper_rows``.
    :param bool parse_json_lists: Decode authors/categories into lists when true.
    :return DecodedPaperRow | SerializedPaperRow: Normalized row in the requested JSON shape.
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
    common: _PaperRowScalarFields = {
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
        return {
            **common,
            "authors": _parse_json_list(authors_json),
            "categories": _parse_json_list(categories_json),
        }
    return {
        **common,
        "authors_json": str(authors_json or ""),
        "categories_json": str(categories_json or ""),
    }


def _cached_paper_metadata(row: DecodedPaperRow) -> CachedPaperMetadata:
    """Project a decoded SQLite row to its normalized metadata fields.

    :param DecodedPaperRow row: Parsed row returned by ``_decode_paper_row``.
    :return CachedPaperMetadata: Fields exposed by cache metadata lookups.
    """
    return {
        "title": row["title"],
        "abstract": row["abstract"],
        "year": row["year"],
        "authors": row["authors"],
        "categories": row["categories"],
        "venue": row["venue"],
        "arxiv_id": row["arxiv_id"],
        "doi": row["doi"],
    }


def _cached_paper_record(row: DecodedPaperRow) -> CachedPaperRecord:
    """Project a decoded SQLite row to normalized metadata with its paper ID.

    :param DecodedPaperRow row: Parsed row returned by ``_decode_paper_row``.
    :return CachedPaperRecord: Cache record exposed to search and identity lookup.
    """
    return {"paper_id": row["paper_id"], **_cached_paper_metadata(row)}


def _chunked(values: Sequence[_ValueT], chunk_size: int) -> Iterator[list[_ValueT]]:
    """Yield fixed-size chunks from a sequence.

    :param Sequence[_ValueT] values: Sequence to split into chunks.
    :param int chunk_size: Number of items per yielded chunk.
    :return Iterator[list[_ValueT]]: Chunks in their original order.
    """
    for start in range(0, len(values), chunk_size):
        yield list(values[start : start + chunk_size])
