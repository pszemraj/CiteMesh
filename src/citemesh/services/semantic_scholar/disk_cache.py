"""On-disk caches for Semantic Scholar reference lists and paper metadata.

Owns the cache-path layout, the persisted payload versions, and the read/write
helpers for cached paper metadata. ``REFERENCE_CACHE_DIR`` is the runtime
override consulted by every reference-cache path lookup.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
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
DISCOVERY_CACHE_VERSION = 1


def _discovery_cache_path(key: tuple[str, str, int, str]) -> Path:
    """Locate a snapshot for one bounded discovery request.

    :param tuple[str, str, int, str] key: Endpoint, normalized seed, limit and pool.
    :return Path: JSON path independent of paper and reference enrichment caches.
    """
    digest = hashlib.sha1(json.dumps(key).encode("utf-8")).hexdigest()
    return get_cache_dir("discovery") / f"{digest}.json"


def _persist_discovery(key: tuple[str, str, int, str], paper_ids: list[str]) -> None:
    """Compare and save a successfully checked ordered discovery list.

    :param tuple[str, str, int, str] key: Endpoint, normalized seed, limit and pool.
    :param list[str] paper_ids: IDs in the order returned by Semantic Scholar.
    :return None: Saves the snapshot without making it a freshness substitute.
    """
    path = _discovery_cache_path(key)
    previous = read_json_object(path)
    unchanged = (
        previous is not None
        and previous.get("version") == DISCOVERY_CACHE_VERSION
        and previous.get("paper_ids") == paper_ids
    )
    logger.info(
        "Checked %s discovery upstream for %s%s: %d IDs (%s).",
        key[0],
        key[1],
        f", {key[3]} pool" if key[3] else "",
        len(paper_ids),
        "membership and order unchanged" if unchanged else "new or changed list",
    )
    try:
        atomic_write_json(
            path,
            {
                "version": DISCOVERY_CACHE_VERSION,
                "endpoint": key[0],
                "paper_id": key[1],
                "limit": key[2],
                "pool": key[3],
                "paper_ids": paper_ids,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except OSError as exc:
        logger.debug("Failed to persist discovery cache for %s: %s", key, exc)


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
    previous = _load_cached_paper(requested_id)
    aliases = _paper_lookup_keys(paper) | {requested_id}
    if previous is not None and previous.paper_id == paper.paper_id:
        for alias in _paper_lookup_keys(previous):
            aliased = _load_cached_paper(alias)
            if aliased is not None and aliased.paper_id == previous.paper_id:
                aliases.add(alias)

    data = asdict(paper)
    data["references"] = []
    data["is_seed"] = False
    cached = {"version": PAPER_CACHE_VERSION, "paper": data}
    for alias in aliases:
        try:
            atomic_write_json(_paper_cache_path(alias), cached)
        except OSError as exc:
            logger.debug("Failed to persist paper cache for %s: %s", alias, exc)
