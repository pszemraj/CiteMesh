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

from citemesh.core import Author, Paper
from citemesh.core.paper_ids import paper_identifier_aliases
from citemesh.data import get_cache_dir
from citemesh.data.cache import atomic_write_json, read_json_object

logger = logging.getLogger(__name__)


# Optional runtime override for tests and one-off callers.
# When unset, reference cache paths are resolved from ``get_cache_dir`` per call.
REFERENCE_CACHE_DIR: Path | None = None
REFERENCE_CACHE_VERSION = 1
PAPER_CACHE_VERSION = 2


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
    """Build normalized aliases for matching batch responses to requested IDs.

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
    """Read current paper metadata, treating stale entries as cache misses.

    :param str paper_id: Normalized requested paper identifier.
    :return Paper | None: Fresh paper instance, or ``None`` on a cache miss.
    """
    cached = read_json_object(_paper_cache_path(paper_id))
    if cached is None or cached.get("version") != PAPER_CACHE_VERSION:
        return None
    try:
        data = cached["paper"]
        data["authors"] = [Author(**author) for author in data["authors"]]
        return Paper(**data)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None


def _persist_paper(paper: Paper, requested_id: str) -> None:
    """Save successful metadata under the requested ID and known aliases.

    :param Paper paper: Converted Semantic Scholar paper metadata.
    :param str requested_id: Normalized identifier used for the request.
    :return None: Writes metadata independently of embeddings and references.
    """
    data = asdict(paper)
    data["references"] = []
    data["is_seed"] = False
    cached = {"version": PAPER_CACHE_VERSION, "paper": data}
    for alias in _paper_lookup_keys(paper) | {requested_id}:
        try:
            atomic_write_json(_paper_cache_path(alias), cached)
        except OSError as exc:
            logger.debug("Failed to persist paper cache for %s: %s", alias, exc)
