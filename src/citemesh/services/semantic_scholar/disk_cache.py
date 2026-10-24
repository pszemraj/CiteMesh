"""On-disk caches for Semantic Scholar reference lists and paper metadata.

Owns the cache-path layout, the persisted payload versions, and the read/write
helpers for cached paper metadata. ``REFERENCE_CACHE_DIR`` is the runtime
override consulted by every reference-cache path lookup.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict, TypeGuard

from citemesh.core import Author, Paper
from citemesh.core.paper_ids import normalize_paper_id, paper_identifier_aliases
from citemesh.data import get_cache_dir
from citemesh.data.cache import JSONValue, atomic_write_json, read_json_object

logger = logging.getLogger(__name__)


# Optional runtime override for tests and one-off callers.
# When unset, reference cache paths are resolved from ``get_cache_dir`` per call.
REFERENCE_CACHE_DIR: Path | None = None
REFERENCE_CACHE_VERSION = 1
PAPER_CACHE_VERSION = 2


class _CachedAuthorBase(TypedDict):
    """Required decoded fields in a cached author record."""

    name: str


class _CachedAuthor(_CachedAuthorBase, total=False):
    """Optional decoded fields in a cached author record."""

    author_id: str | None


class _CachedPaperBase(TypedDict):
    """Required decoded fields in cached paper metadata."""

    paper_id: str
    title: str
    year: int | None


class _PaperOptionalFields(TypedDict, total=False):
    """Optional scalar and list fields carried by paper cache records."""

    citation_count: int
    abstract: str
    venue: str
    arxiv_id: str
    doi: str
    categories: list[str]
    references: list[str]
    is_seed: bool
    is_local_corpus: bool


class _CachedPaper(_CachedPaperBase, _PaperOptionalFields, total=False):
    """Decoded paper metadata stored in the cache."""

    authors: list[_CachedAuthor]


class _PaperArguments(_CachedPaperBase, _PaperOptionalFields):
    """Paper fields accepted by the dataclass constructor."""

    authors: list[Author]


def _is_cached_author(value: JSONValue) -> TypeGuard[_CachedAuthor]:
    """Narrow decoded author data to the persisted fields.

    :param JSONValue value: Decoded author value.
    :return bool: Whether the value has the expected author fields.
    """
    if not isinstance(value, dict):
        return False
    return isinstance(value.get("name"), str) and (
        "author_id" not in value
        or value["author_id"] is None
        or isinstance(value["author_id"], str)
    )


def _is_cached_paper(value: JSONValue) -> TypeGuard[_CachedPaper]:
    """Narrow decoded paper metadata before dataclass construction.

    :param JSONValue value: Decoded paper value.
    :return bool: Whether the value has supported paper fields.
    """
    if not isinstance(value, dict):
        return False
    if not (
        isinstance(value.get("paper_id"), str)
        and isinstance(value.get("title"), str)
        and "year" in value
        and (value["year"] is None or isinstance(value["year"], int))
    ):
        return False
    scalar_fields = (
        ("citation_count", int),
        ("abstract", str),
        ("venue", str),
        ("arxiv_id", str),
        ("doi", str),
        ("is_seed", bool),
        ("is_local_corpus", bool),
    )
    if any(
        key in value and not isinstance(value[key], field_type)
        for key, field_type in scalar_fields
    ):
        return False
    for key in ("categories", "references"):
        if key in value:
            items = value[key]
            if not isinstance(items, list) or not all(
                isinstance(item, str) for item in items
            ):
                return False
    if "authors" in value:
        authors = value["authors"]
        if not isinstance(authors, list) or not all(
            _is_cached_author(author) for author in authors
        ):
            return False
    return True


def _reference_cache_dir() -> Path:
    """Resolve reference-cache directory at call time.

    :return Path: Directory where reference cache JSON files are stored.
    """
    if REFERENCE_CACHE_DIR is None:
        return get_cache_dir("references")

    resolved = Path(REFERENCE_CACHE_DIR)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _paper_lookup_keys(paper: Paper) -> set[str]:
    """Build current normalized identifiers for a paper cache entry.

    :param Paper paper: Converted paper payload from Semantic Scholar.
    :return set[str]: Normalized identifier aliases for the paper.
    """
    return set(
        paper_identifier_aliases(
            paper_id=paper.paper_id,
            arxiv_id=paper.arxiv_id,
            doi=paper.doi,
        )
    )


def _reference_cache_path(paper_id: str) -> Path:
    """Build cache file path for a normalized paper identifier.

    :param str paper_id: Normalized paper identifier.
    :return Path: JSON cache path for the paper's reference IDs.
    """
    digest = hashlib.sha1(paper_id.encode("utf-8")).hexdigest()
    return _reference_cache_dir() / f"{digest}.json"


def _paper_cache_path(paper_id: str) -> Path:
    """Locate persisted metadata for a normalized paper identifier.

    :param str paper_id: Normalized requested ID or paper alias.
    :return Path: Paper metadata JSON path.
    """
    digest = hashlib.sha1(paper_id.encode("utf-8")).hexdigest()
    return get_cache_dir("papers") / f"{digest}.json"


def _load_cached_paper(paper_id: str) -> Paper | None:
    """Read current paper metadata through one canonical cache lookup.

    :param str paper_id: Normalized requested paper identifier.
    :return Paper | None: Fresh paper instance, or ``None`` on a cache miss.
    """
    cached = read_json_object(_paper_cache_path(paper_id))
    if cached is None or cached.get("version") != PAPER_CACHE_VERSION:
        return None
    try:
        canonical_paper_id = paper_id
        if "canonical_paper_id" in cached:
            raw_canonical_paper_id = cached["canonical_paper_id"]
            if not isinstance(raw_canonical_paper_id, str):
                return None
            canonical_paper_id = raw_canonical_paper_id
            canonical = read_json_object(_paper_cache_path(canonical_paper_id))
            if canonical is None or canonical.get("version") != PAPER_CACHE_VERSION:
                return None
            cached = canonical
        else:
            initial_data = cached.get("paper")
            if not _is_cached_paper(initial_data):
                return None
            if initial_data["paper_id"] != paper_id:
                canonical_paper_id = initial_data["paper_id"]
                canonical = read_json_object(_paper_cache_path(canonical_paper_id))
                if canonical is None or canonical.get("version") != PAPER_CACHE_VERSION:
                    return None
                cached = canonical

        data = cached.get("paper")
        if not _is_cached_paper(data):
            return None
        if data["paper_id"] != canonical_paper_id:
            return None
        paper_data: _PaperArguments = {
            **data,
            "authors": [Author(**author) for author in data.get("authors", [])],
        }
        return Paper(**paper_data)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None


def _persist_paper(paper: Paper, requested_id: str) -> None:
    """Save full metadata once and point current identifiers at it.

    :param Paper paper: Converted Semantic Scholar paper metadata.
    :param str requested_id: Normalized identifier used for the request.
    :return None: Writes metadata independently of embeddings and references.
    """
    data = asdict(paper)
    data["references"] = []
    data["is_seed"] = False
    canonical_paper_id = paper.paper_id
    try:
        atomic_write_json(
            _paper_cache_path(canonical_paper_id),
            {"version": PAPER_CACHE_VERSION, "paper": data},
        )
    except OSError as exc:
        logger.debug(
            "Failed to persist canonical paper cache for %s: %s",
            canonical_paper_id,
            exc,
        )
        return

    aliases = {
        normalize_paper_id(alias)
        for alias in _paper_lookup_keys(paper) | {requested_id}
    }
    for alias in aliases - {canonical_paper_id}:
        try:
            atomic_write_json(
                _paper_cache_path(alias),
                {
                    "version": PAPER_CACHE_VERSION,
                    "canonical_paper_id": canonical_paper_id,
                },
            )
        except OSError as exc:
            logger.debug("Failed to persist paper cache for %s: %s", alias, exc)
