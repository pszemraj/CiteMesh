"""Normalization of raw arXiv dataset records into embedding metadata.

Owns identifier canonicalization, the newest-first record ranking built on the
arXiv-id chronology keys derived in :mod:`citemesh.core.paper_ids`, and the
field coercion that turns a raw dataset row into the metadata dict the embedding
cache stores. The ``_parse_*`` helpers layer arXiv-specific behavior over the
shared coercers in :mod:`citemesh.core.paper_fields`.
"""

from __future__ import annotations

import heapq
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha1, sha256
from typing import TypeVar

from citemesh.core.paper_fields import coerce_authors, coerce_categories, coerce_venue
from citemesh.core.paper_ids import (
    arxiv_id_chronology_key as _arxiv_id_chronology_key,
)
from citemesh.core.paper_ids import (
    canonicalize_or_none,
    external_ids_from_canonical_paper_id,
    recognize_arxiv_identifier,
)
from citemesh.data.embedding_cache.models import CachedPaperMetadata

from .text import _embedding_text_metadata

_DATASET_PAPER_ID_FIELDS = ("id", "paper_id", "paperId")

DatasetRecord = Mapping[str, object]
_RecordT = TypeVar("_RecordT", bound=Mapping[str, object])


class ExtractedPaperMetadata(CachedPaperMetadata):
    """Normalized source metadata keyed by its resolved paper identity."""

    paper_id: str


@dataclass(frozen=True)
class _HydrationSourceSliceResult:
    """Counts and completion flags from one exact-source hydration pass.

    ``hydrated_records`` counts unique records sent to cache batching;
    ``source_rows_consumed`` counts traversed source rows. ``source_exhausted``
    reports clean EOF, while ``row_cap_reached`` reports consuming the full cap.

    :ivar int hydrated_records: Records routed into cache batching.
    :ivar int source_rows_consumed: Raw source rows yielded to hydration.
    :ivar bool source_exhausted: Whether the selected source slice reached clean EOF.
    :ivar bool row_cap_reached: Whether the pass consumed its whole corpus cap.
    """

    hydrated_records: int
    source_rows_consumed: int
    source_exhausted: bool
    row_cap_reached: bool = False


@dataclass(frozen=True)
class _CappedCorpusRecencyProbe:
    """Newest-first source selection and key from a capped recency probe.

    ``selection`` contains the ranked rows when they can be reused by hydration;
    it is ``None`` when the source selection cannot be traversed again.

    :ivar Optional[int] newest_key: Newest packed chronology key upstream holds.
    :ivar Optional[Iterable[DatasetRecord]] selection: Re-iterable selected rows.
    """

    newest_key: int | None
    selection: Iterable[DatasetRecord] | None


def _canonicalize_embedding_paper_id(raw_id: object) -> str:
    """Canonicalize arXiv-like identifiers for downstream lookups.

    :param object raw_id: Raw dataset record identifier.
    :return str: Canonicalized identifier.
    """
    text = str(raw_id).strip() if raw_id is not None else ""
    if not text:
        return ""

    if text.startswith("arxiv_"):
        return text

    return recognize_arxiv_identifier(text, allow_bare=True) or text


def _query_seed_id(query_text: str) -> str:
    """Build deterministic query-mode seed node identifier.

    :param str query_text: Raw user query text.
    :return str: Stable hashed query seed identifier.
    """
    digest = sha1(query_text.encode("utf-8")).hexdigest()[:8]
    return f"query:{digest}"


def _parse_year(paper: DatasetRecord) -> int | None:
    """Extract publication year from dataset metadata.

    :param DatasetRecord paper: Raw dataset record.
    :return Optional[int]: Parsed year or ``None`` if missing/invalid.
    """
    year = paper.get("year")
    if isinstance(year, (int, float, str)) and year:
        try:
            return int(year)
        except (TypeError, ValueError):
            pass

    raw_id = _dataset_record_raw_paper_id(paper)
    chronology = _arxiv_id_chronology_key(raw_id)
    return chronology[0] if chronology is not None else None


def _newest_records_by_arxiv_id(
    records: Iterable[_RecordT], limit: int
) -> list[_RecordT]:
    """Select the ``limit`` most recently submitted records in one bounded pass.

    Prefer records with parseable arXiv IDs, filling any shortfall from the
    first records without parseable IDs so hydration reaches the requested cap.

    :param Iterable[DatasetRecord] records: Dataset records to scan.
    :param int limit: Number of newest records to keep.
    :return List[DatasetRecord]: Selected records in chronological order.
    """
    heap: list[tuple[tuple[int, int, int], int, _RecordT]] = []
    head_fallback: list[_RecordT] = []
    for order, record in enumerate(records):
        raw_id = _dataset_record_raw_paper_id(record or {})
        key = _arxiv_id_chronology_key(raw_id)
        if key is None:
            if len(head_fallback) < limit:
                head_fallback.append(record)
            continue
        entry = (key, order, record)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    selected = [record for _, _, record in sorted(heap, key=lambda entry: entry[:2])]
    return selected + head_fallback[: limit - len(selected)]


def _dataset_record_raw_paper_id(paper: DatasetRecord) -> object | None:
    """Return the first populated identifier field from a dataset record.

    :param DatasetRecord paper: Raw dataset record.
    :return object | None: Raw identifier value, or ``None`` when every supported field is empty.
    """
    for field in _DATASET_PAPER_ID_FIELDS:
        value = paper.get(field)
        if isinstance(value, str):
            if value.strip():
                return value
        elif value:
            return value
    return None


def _parse_authors(
    authors_data: object, authors_parsed_data: object | None = None
) -> list[str]:
    """Normalize author metadata to a list of names.

    arXiv snapshots ship a structured ``authors_parsed`` field (last, first,
    suffix triples) alongside the free-text ``authors`` string; the structured
    rows win when present, and everything else defers to
    :func:`citemesh.core.paper_fields.coerce_authors` with the comma-splitting
    that arXiv's packed author strings require.

    :param object authors_data: Raw ``authors`` field from dataset.
    :param object | None authors_parsed_data: Optional structured arXiv author rows.
    :return List[str]: Author names.
    """
    if isinstance(authors_parsed_data, list):
        structured_authors = []
        for raw_author in authors_parsed_data:
            if not isinstance(raw_author, (list, tuple)):
                continue
            parts = (
                raw_author[1] if len(raw_author) > 1 else "",
                raw_author[0] if raw_author else "",
                raw_author[2] if len(raw_author) > 2 else "",
            )
            name = " ".join(
                part.strip() for part in parts if isinstance(part, str) and part.strip()
            )
            if name:
                structured_authors.append(name)
        if structured_authors:
            return structured_authors

    return coerce_authors(authors_data, split_string=True)


def _parse_categories(categories_data: object) -> list[str]:
    """Normalize category metadata to a list of arXiv category codes.

    arXiv packs several codes into one comma/whitespace-separated string, so the
    shared coercer is asked to split on whitespace.

    :param object categories_data: Raw ``categories`` field from dataset.
    :return List[str]: Category code list.
    """
    return coerce_categories(categories_data, split_whitespace=True)


def _parse_venue(paper: DatasetRecord) -> str:
    """Normalize venue/journal metadata from dataset records.

    The arXiv metadata snapshots ship this column as ``journal-ref``; other
    sources spell it ``journal_ref`` or nest it under ``journal``.

    :param DatasetRecord paper: Raw dataset record.
    :return str: Best-effort venue string (empty when unavailable).
    """
    for key in ("venue", "journal_ref", "journal-ref", "journal"):
        venue = coerce_venue(paper.get(key))
        if venue:
            return venue
    return ""


def _parse_abstract(paper: DatasetRecord) -> str:
    """Return the first usable abstract-like field from a dataset record.

    Some dataset adapters populate ``abstract`` with ``None`` or whitespace
    while retaining usable text in ``summary``. Treating the mere presence of
    ``abstract`` as authoritative would discard that document text.

    :param DatasetRecord paper: Source or cached paper metadata.
    :return str: First nonblank ``abstract`` or ``summary`` string, else empty.
    """
    for field in ("abstract", "summary"):
        value = paper.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _parse_doi(doi_data: object) -> str:
    """Normalize the first usable DOI token in dataset metadata.

    :param object doi_data: Raw DOI field from source or cached metadata.
    :return str: Canonical bare DOI, or an empty string when unavailable.
    """
    # Suffix punctuation separates records only when another DOI follows it.
    source_doi = re.split(
        r"\s+|[,;](?=\s*(?:10\.\d{4,9}/|doi:|https?://))",
        str(doi_data or "").strip(),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    canonical_source_doi = canonicalize_or_none(source_doi) if source_doi else None
    if canonical_source_doi is None:
        return ""
    _, normalized_doi = external_ids_from_canonical_paper_id(canonical_source_doi)
    return normalized_doi if normalized_doi and normalized_doi.startswith("10.") else ""


def _anonymous_dataset_paper_id(paper: DatasetRecord) -> str | None:
    """Identify a row without a source ID by normalized bibliographic metadata.

    Source positions change on insertion, ranking and slicing. Content identity
    preserves unchanged rows across those operations. Authors, year, and DOI
    keep distinct works from aliasing when they share generic title/abstract text.

    :param DatasetRecord paper: Source or cached bibliographic metadata.
    :return Optional[str]: Stable content identifier, or ``None`` for empty text.
    """
    title = paper.get("title")
    text = _embedding_text_metadata(
        title if isinstance(title, str) else "",
        _parse_abstract(paper),
    )
    if not any(text.values()):
        return None
    identity = {
        "text": text,
        "authors": _parse_authors(
            paper.get("authors", []), paper.get("authors_parsed")
        ),
        "year": _parse_year(paper),
        "doi": _parse_doi(paper.get("doi")),
    }
    content = json.dumps(identity, sort_keys=True, ensure_ascii=False)
    return f"content:{sha256(content.encode('utf-8')).hexdigest()}"


def _dataset_record_paper_id(paper: DatasetRecord, fallback_index: int) -> str:
    """Resolve the cache identifier a raw dataset record hydrates under.

    Split out so a caller that only needs to ask whether a source row is already
    cached can answer it without normalizing the rest of the record, and cannot
    drift from the identifier hydration would actually write.

    :param DatasetRecord paper: Raw dataset record.
    :param int fallback_index: Source row index reported for unusable records.
    :return str: Canonicalized paper identifier.
    :raises ValueError: If neither an identifier nor usable document text exists.
    """
    raw_paper_id = _dataset_record_raw_paper_id(paper)
    canonical_id = _canonicalize_embedding_paper_id(raw_paper_id)
    if canonical_id:
        return canonical_id
    content_id = _anonymous_dataset_paper_id(paper)
    if content_id is None:
        raise ValueError(
            f"Dataset row {fallback_index} has no paper identifier or usable "
            "title/abstract; supply a stable identifier or document text."
        )
    return content_id


def _extract_dataset_paper_metadata(
    paper: DatasetRecord, fallback_index: int
) -> ExtractedPaperMetadata:
    """Normalize a raw dataset record to embedding metadata fields.

    :param DatasetRecord paper: Raw dataset record.
    :param int fallback_index: Source row index reported for unusable records.
    :return ExtractedPaperMetadata: Normalized metadata used by embedding selection.
    """
    paper_id = _dataset_record_paper_id(paper, fallback_index)
    arxiv_id, doi = external_ids_from_canonical_paper_id(paper_id)
    doi = _parse_doi(paper.get("doi")) or doi
    title = paper.get("title", "Unknown")
    if not isinstance(title, str) or not title.strip():
        title = "Unknown"
    abstract = _parse_abstract(paper)

    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "venue": _parse_venue(paper),
        "arxiv_id": arxiv_id,
        "doi": doi,
        "year": _parse_year(paper),
        "authors": _parse_authors(
            paper.get("authors", []), paper.get("authors_parsed")
        ),
        "categories": _parse_categories(paper.get("categories", [])),
    }
