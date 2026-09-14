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
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha1, sha256
from typing import (
    Any,
)

from citemesh.core.paper_fields import coerce_authors, coerce_categories, coerce_venue
from citemesh.core.paper_ids import (
    arxiv_id_chronology_key as _arxiv_id_chronology_key,
)
from citemesh.core.paper_ids import (
    canonicalize_or_none,
    external_ids_from_canonical_paper_id,
    recognize_arxiv_identifier,
)

from .text import _embedding_text_metadata

_DATASET_PAPER_ID_FIELDS = ("id", "paper_id", "paperId")


@dataclass(frozen=True)
class _HydrationSourceSliceResult:
    """Outcome of consuming one exact-source hydration slice.

    ``hydrated_records`` counts unique records routed into cache batching, while
    ``source_rows_consumed`` and ``source_exhausted`` describe source traversal.
    Keeping those concepts separate prevents duplicate/invalid source rows from
    being mistaken for an interrupted resume.

    A capped pass has two ways to see everything it was asked to select, and
    ``source_exhausted`` only reports one of them. Where the selection is exactly
    cap-sized, draining it is reaching the cap; where the selection is wider than
    the cap — the fallbacks that hand back the whole dataset when no arXiv ID is
    parseable — hydration stops at the cap and the iterator never reports EOF.
    ``row_cap_reached`` is the other half, so a caller asking "did this pass
    finish?" does not mistake the second shape for a short read.

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
    """Outcome of asking upstream whether a capped corpus is still the newest.

    ``selection`` is the newest-first slice the probe had to load and rank to
    answer that. Ranking costs a full pass over the source — a full drain under
    ``--streaming`` — and the pass that admits the missing rows needs exactly
    this slice, so it is handed over rather than selected a second time. It is
    ``None`` when the loaded slice cannot be traversed again, leaving that
    caller to reload as before.

    :ivar Optional[int] newest_key: Newest packed chronology key upstream holds.
    :ivar Optional[Iterable[Dict[str, Any]]] selection: Re-iterable selected rows.
    """

    newest_key: int | None
    selection: Iterable[dict[str, Any]] | None


def _canonicalize_embedding_paper_id(raw_id: Any) -> str:
    """Canonicalize arXiv-like identifiers for downstream lookups.

    :param Any raw_id: Raw dataset record identifier.
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


def _parse_year(paper: dict[str, Any]) -> int | None:
    """Extract publication year from dataset metadata.

    :param Dict[str, Any] paper: Raw dataset record.
    :return Optional[int]: Parsed year or ``None`` if missing/invalid.
    """
    if paper.get("year"):
        try:
            return int(paper["year"])
        except (TypeError, ValueError):
            pass

    chronology = _arxiv_id_chronology_key(_dataset_record_raw_paper_id(paper))
    return chronology[0] if chronology is not None else None


def _newest_records_by_arxiv_id(
    records: Iterable[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Select the ``limit`` most recently submitted records in one bounded pass.

    Prefer records with parseable arXiv IDs, filling any shortfall from the
    first records without parseable IDs so hydration reaches the requested cap.

    :param Iterable[Dict[str, Any]] records: Dataset records to scan.
    :param int limit: Number of newest records to keep.
    :return List[Dict[str, Any]]: Selected records in chronological order.
    """
    heap: list[tuple[tuple[int, int, int], int, dict[str, Any]]] = []
    head_fallback: list[dict[str, Any]] = []
    for order, record in enumerate(records):
        key = _arxiv_id_chronology_key(_dataset_record_raw_paper_id(record or {}))
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


def _dataset_record_raw_paper_id(paper: dict[str, Any]) -> Any:
    """Return the first populated identifier field from a dataset record.

    :param Dict[str, Any] paper: Raw dataset record.
    :return Any: Raw identifier value, or ``None`` when every supported field is empty.
    """
    for field in _DATASET_PAPER_ID_FIELDS:
        value = paper.get(field)
        if isinstance(value, str):
            if value.strip():
                return value
        elif value:
            return value
    return None


def _parse_authors(authors_data: Any, authors_parsed_data: Any = None) -> list[str]:
    """Normalize author metadata to a list of names.

    arXiv snapshots ship a structured ``authors_parsed`` field (last, first,
    suffix triples) alongside the free-text ``authors`` string; the structured
    rows win when present, and everything else defers to
    :func:`citemesh.core.paper_fields.coerce_authors` with the comma-splitting
    that arXiv's packed author strings require.

    :param Any authors_data: Raw ``authors`` field from dataset.
    :param Any authors_parsed_data: Optional structured arXiv author rows.
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


def _parse_categories(categories_data: Any) -> list[str]:
    """Normalize category metadata to a list of arXiv category codes.

    arXiv packs several codes into one comma/whitespace-separated string, so the
    shared coercer is asked to split on whitespace.

    :param Any categories_data: Raw ``categories`` field from dataset.
    :return List[str]: Category code list.
    """
    return coerce_categories(categories_data, split_whitespace=True)


def _parse_venue(paper: dict[str, Any]) -> str:
    """Normalize venue/journal metadata from dataset records.

    The arXiv metadata snapshots ship this column as ``journal-ref``; other
    sources spell it ``journal_ref`` or nest it under ``journal``.

    :param Dict[str, Any] paper: Raw dataset record.
    :return str: Best-effort venue string (empty when unavailable).
    """
    for key in ("venue", "journal_ref", "journal-ref", "journal"):
        venue = coerce_venue(paper.get(key))
        if venue:
            return venue
    return ""


def _anonymous_dataset_paper_id(paper: dict[str, Any]) -> str | None:
    """Identify a row without a source ID by its normalized embedding content.

    Source positions change on insertion, ranking and slicing. Content identity
    preserves unchanged rows across those operations without aliasing new papers.

    :param Dict[str, Any] paper: Source or cached title/abstract metadata.
    :return Optional[str]: Stable content identifier, or ``None`` for empty text.
    """
    title = paper.get("title")
    abstract = paper.get("abstract", paper.get("summary", ""))
    text = _embedding_text_metadata(
        title if isinstance(title, str) else "",
        abstract if isinstance(abstract, str) else "",
    )
    if not any(text.values()):
        return None
    content = json.dumps(text, sort_keys=True, ensure_ascii=False)
    return f"content:{sha256(content.encode('utf-8')).hexdigest()}"


def _dataset_record_paper_id(paper: dict[str, Any], fallback_index: int) -> str:
    """Resolve the cache identifier a raw dataset record hydrates under.

    Split out so a caller that only needs to ask whether a source row is already
    cached can answer it without normalizing the rest of the record, and cannot
    drift from the identifier hydration would actually write.

    :param Dict[str, Any] paper: Raw dataset record.
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


def _extract_dataset_paper_metadata(paper: dict[str, Any], fallback_index: int) -> dict:
    """Normalize a raw dataset record to embedding metadata fields.

    :param Dict[str, Any] paper: Raw dataset record.
    :param int fallback_index: Source row index reported for unusable records.
    :return Dict: Normalized metadata used by embedding selection.
    """
    paper_id = _dataset_record_paper_id(paper, fallback_index)
    arxiv_id, doi = external_ids_from_canonical_paper_id(paper_id)
    source_doi = re.split(r"[\s,;]+", str(paper.get("doi") or "").strip())[0]
    canonical_source_doi = canonicalize_or_none(source_doi) if source_doi else None
    if canonical_source_doi is not None:
        _, normalized_doi = external_ids_from_canonical_paper_id(canonical_source_doi)
        if normalized_doi and normalized_doi.startswith("10."):
            doi = normalized_doi
    title = paper.get("title", "Unknown")
    if not isinstance(title, str) or not title.strip():
        title = "Unknown"
    abstract = paper.get("abstract", paper.get("summary", ""))
    if not isinstance(abstract, str):
        abstract = ""

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
