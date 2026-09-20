"""Cache-aware Semantic Scholar endpoint methods over the direct HTTP transport."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from typing import Any
from urllib.parse import quote

from citemesh.core import Paper
from citemesh.core.paper_ids import normalize_paper_id

from . import disk_cache, payloads
from .errors import (
    SemanticScholarUnavailableError,
    _SemanticScholarResponseContractError,
)

logger = logging.getLogger(__name__)

RECOMMENDATION_BASE_URL = (
    "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
)
PAPER_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
SEARCH_BASE_URL = f"{PAPER_BASE_URL}/search"
GRAPH_RELATION_PAGE_SIZE = 1000
PAPER_BATCH_SIZE = 500
RECOMMENDATION_MAX_RESULTS = 500
SEARCH_MAX_RESULTS = 1000
SEARCH_PAGE_SIZE = 100


def _scoped_endpoint(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run an endpoint inside the outer candidate collection scope.

    :param Callable[..., Any] method: Endpoint method.
    :return Callable[..., Any]: Scope-sharing wrapper.
    """

    @wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Enter or reuse the collection scope.

        :param Any self: Semantic Scholar client.
        :param Any args: Endpoint positional arguments.
        :param Any kwargs: Endpoint keyword arguments.
        :return Any: Endpoint result.
        """
        with self.candidate_operation_scope():
            return method(self, *args, **kwargs)

    return wrapped


class _EndpointsMixin:
    """Public Semantic Scholar endpoint surface."""

    @_scoped_endpoint
    def get_paper(
        self,
        paper_id: str,
        fetch_references: bool = False,
        *,
        raise_on_unavailable: bool = False,
    ) -> Paper | None:
        """Fetch one paper, reusing persisted metadata first.

        :param str paper_id: DOI, arXiv ID, or Semantic Scholar ID.
        :param bool fetch_references: Populate complete reference IDs, raising if
            those references cannot be retrieved.
        :param bool raise_on_unavailable: Raise when operational retries stop.
        :return Paper | None: Paper record or ``None`` when absent/unavailable.
        """
        if not paper_id or not isinstance(paper_id, str):
            raise ValueError(f"Invalid paper ID: {paper_id}")
        normalized_id = normalize_paper_id(paper_id)

        if fetch_references:
            paper = self.get_paper(
                normalized_id, raise_on_unavailable=raise_on_unavailable
            )
            if paper is not None:
                paper.references = self.get_reference_ids(
                    normalized_id, raise_on_unavailable=True
                )
            return paper

        if not self.refresh_paper_cache:
            cached = disk_cache._load_cached_paper(normalized_id)
            if cached is not None:
                logger.debug("Reused cached paper metadata for %s.", normalized_id)
                return cached

        response = self._request_json(
            f"{PAPER_BASE_URL}/{quote(normalized_id, safe='/')}",
            {"fields": ",".join(payloads._default_paper_fields())},
            context=f"fetching {normalized_id}",
            raise_on_unavailable=raise_on_unavailable,
        )
        if response is None:
            return None
        try:
            paper = payloads._convert_payload_paper(
                response, category_keys=("fields", "fieldsOfStudy")
            )
        except Exception as exc:
            raise _SemanticScholarResponseContractError(
                f"Semantic Scholar returned a malformed paper payload for "
                f"{normalized_id}: {exc}"
            ) from exc
        if paper is None:
            raise _SemanticScholarResponseContractError(
                f"Semantic Scholar returned a paper payload without a paper ID for "
                f"{normalized_id}."
            )
        disk_cache._persist_paper(paper, normalized_id)
        return paper

    @_scoped_endpoint
    def get_cached_papers(self, paper_ids: Sequence[str]) -> dict[str, Paper]:
        """Load persisted paper metadata without making provider requests.

        :param Sequence[str] paper_ids: Paper identifiers to resolve from cache.
        :return dict[str, Paper]: Cached records keyed by normalized request ID.
        """
        normalized_ids: list[str] = []
        for raw_id in paper_ids:
            if not raw_id or not isinstance(raw_id, str):
                raise ValueError(f"Invalid paper ID: {raw_id}")
            normalized_ids.append(normalize_paper_id(raw_id))
        if self.refresh_paper_cache:
            return {}
        papers = {
            paper_id: cached
            for paper_id in dict.fromkeys(normalized_ids)
            if (cached := disk_cache._load_cached_paper(paper_id)) is not None
        }
        return papers

    @_scoped_endpoint
    def get_papers(
        self, paper_ids: Sequence[str], *, raise_on_unavailable: bool = False
    ) -> dict[str, Paper]:
        """Load cached papers and batch-fetch all missing IDs in groups of 500.

        :param Sequence[str] paper_ids: Arbitrary number of paper identifiers.
        :param bool raise_on_unavailable: Raise if a required batch cannot complete.
        :return dict[str, Paper]: Results keyed by normalized requested identifier.
        """
        normalized_ids: list[str] = []
        for raw_id in paper_ids:
            if not raw_id or not isinstance(raw_id, str):
                raise ValueError(f"Invalid paper ID: {raw_id}")
            normalized_ids.append(normalize_paper_id(raw_id))
        normalized_ids = list(dict.fromkeys(normalized_ids))

        papers = self.get_cached_papers(normalized_ids)
        missing = [paper_id for paper_id in normalized_ids if paper_id not in papers]
        reused_count = len(papers)
        malformed_records = 0

        for offset in range(0, len(missing), PAPER_BATCH_SIZE):
            batch_ids = missing[offset : offset + PAPER_BATCH_SIZE]
            response = self._request_json(
                f"{PAPER_BASE_URL}/batch",
                {"fields": ",".join(payloads._default_paper_fields())},
                payload={"ids": batch_ids},
                context=f"batch fetching {len(batch_ids)} papers",
                raise_on_unavailable=raise_on_unavailable,
                retry_not_found=True,
            )
            if response is None:
                logger.debug(
                    "Paper metadata: stopped after an unavailable batch; %d "
                    "missing records were not fetched.",
                    len(missing) - offset,
                )
                break
            if not isinstance(response, list) or len(response) != len(batch_ids):
                raise _SemanticScholarResponseContractError(
                    "Semantic Scholar batch response must contain one row per requested ID."
                )
            for requested_id, record in zip(batch_ids, response):
                if record is None:
                    continue
                paper = payloads._convert_api_paper(record)
                if paper is None:
                    malformed_records += 1
                    logger.debug("Skipping malformed batch paper for %s.", requested_id)
                    continue
                papers[requested_id] = paper
                disk_cache._persist_paper(paper, requested_id)
        logger.debug(
            "Paper metadata: reused %d cached records; fetched %d missing records.",
            reused_count,
            len(papers) - reused_count,
        )
        if malformed_records:
            logger.warning(
                "Paper metadata skipped %d malformed records.", malformed_records
            )
        return {
            paper_id: papers[paper_id]
            for paper_id in normalized_ids
            if paper_id in papers
        }

    def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 20,
        *,
        raise_on_unavailable: bool = False,
    ) -> list[Paper]:
        """Fetch complete records for papers citing a seed.

        :param str paper_id: Seed identifier.
        :param int limit: Maximum unique results.
        :param bool raise_on_unavailable: Raise if discovery or enrichment fails.
        :return list[Paper]: Ordered citing papers.
        """
        return self._get_related_papers(
            paper_id,
            limit,
            relation="citations",
            nested_key="citingPaper",
            raise_on_unavailable=raise_on_unavailable,
        )

    def get_paper_references(
        self,
        paper_id: str,
        limit: int = 20,
        *,
        raise_on_unavailable: bool = False,
    ) -> list[Paper]:
        """Fetch complete records for papers referenced by a seed.

        :param str paper_id: Seed identifier.
        :param int limit: Maximum unique results.
        :param bool raise_on_unavailable: Raise if discovery or enrichment fails.
        :return list[Paper]: Ordered referenced papers.
        """
        return self._get_related_papers(
            paper_id,
            limit,
            relation="references",
            nested_key="citedPaper",
            raise_on_unavailable=raise_on_unavailable,
        )

    def _relation_ids(
        self,
        paper_id: str,
        *,
        relation: str,
        nested_key: str,
        limit: int | None,
        raise_on_unavailable: bool,
    ) -> list[str] | None:
        """Fetch ordered unique relation IDs through explicit REST pages.

        :param str paper_id: Normalized seed identifier.
        :param str relation: ``references`` or ``citations``.
        :param str nested_key: Relation row's nested paper key.
        :param int | None limit: Maximum results, or all available IDs.
        :param bool raise_on_unavailable: Raise on a failed page.
        :return list[str] | None: IDs, or ``None`` after a tolerated failed page.
        """
        ordered_ids: list[str] = []
        seen_ids: set[str] = set()
        offset = 0
        while limit is None or len(ordered_ids) < limit:
            page_limit = GRAPH_RELATION_PAGE_SIZE
            if limit is not None:
                page_limit = min(page_limit, max(1, limit - len(ordered_ids)))
            response = self._request_json(
                f"{PAPER_BASE_URL}/{quote(paper_id, safe='/')}/{relation}",
                {
                    "fields": "paperId",
                    "limit": page_limit,
                    "offset": offset,
                },
                context=f"fetching {relation} for {paper_id} at offset {offset}",
                raise_on_unavailable=raise_on_unavailable,
                retry_not_found=offset > 0,
            )
            if response is None:
                return None
            if not isinstance(response, dict) or "data" not in response:
                raise _SemanticScholarResponseContractError(
                    f"Semantic Scholar returned malformed {relation} pagination data."
                )
            rows = response["data"]
            if rows is None:
                rows = []
            if not isinstance(rows, list):
                raise _SemanticScholarResponseContractError(
                    f"Semantic Scholar returned non-list {relation} data."
                )
            for row in rows:
                if not isinstance(row, dict) or nested_key not in row:
                    raise _SemanticScholarResponseContractError(
                        f"Semantic Scholar returned malformed {relation} relation data."
                    )
                paper = payloads._payload_get(row, nested_key)
                if not isinstance(paper, dict) or "paperId" not in paper:
                    raise _SemanticScholarResponseContractError(
                        f"Semantic Scholar returned malformed {relation} paper data."
                    )
                raw_id = payloads._payload_get(paper, "paperId")
                if raw_id is None:
                    continue
                if not isinstance(raw_id, str) or not raw_id.strip():
                    raise _SemanticScholarResponseContractError(
                        f"Semantic Scholar returned invalid {relation} paper ID."
                    )
                normalized_id = normalize_paper_id(raw_id)
                if normalized_id not in seen_ids:
                    seen_ids.add(normalized_id)
                    ordered_ids.append(normalized_id)
                    if limit is not None and len(ordered_ids) >= limit:
                        break
            next_offset = response.get("next")
            if not rows or next_offset is None:
                break
            if (
                isinstance(next_offset, bool)
                or not isinstance(next_offset, int)
                or next_offset <= offset
            ):
                raise _SemanticScholarResponseContractError(
                    f"Semantic Scholar returned invalid {relation} pagination offset."
                )
            offset = next_offset
        logger.debug(
            "Checked %s discovery upstream for %s: %d IDs.",
            relation,
            paper_id,
            len(ordered_ids),
        )
        return ordered_ids

    @_scoped_endpoint
    def _get_related_papers(
        self,
        paper_id: str,
        limit: int,
        *,
        relation: str,
        nested_key: str,
        raise_on_unavailable: bool,
    ) -> list[Paper]:
        """Freshly discover relation IDs, then materialize available records.

        :param str paper_id: Seed identifier.
        :param int limit: Maximum unique results.
        :param str relation: Relation endpoint name.
        :param str nested_key: Nested response paper key.
        :param bool raise_on_unavailable: Raise on failed discovery/enrichment.
        :return list[Paper]: Ordered materialized papers.
        """
        parsed_limit = payloads._validate_integer_limit(limit, "limit", allow_zero=True)
        if parsed_limit == 0:
            return []
        normalized_id = normalize_paper_id(paper_id)
        paper_ids = self._relation_ids(
            normalized_id,
            relation=relation,
            nested_key=nested_key,
            limit=parsed_limit,
            raise_on_unavailable=raise_on_unavailable,
        )
        if paper_ids is None:
            if raise_on_unavailable:
                raise SemanticScholarUnavailableError(
                    f"Could not fetch {relation} for {normalized_id}: HTTP 404 "
                    "(paper not found)."
                )
            return []
        records = self.get_papers(paper_ids, raise_on_unavailable=raise_on_unavailable)
        unresolved = [paper_id for paper_id in paper_ids if paper_id not in records]
        if unresolved:
            logger.warning(
                "%s discovery for %s: skipping %d IDs without resolvable metadata.",
                relation,
                normalized_id,
                len(unresolved),
            )
        return [records[paper_id] for paper_id in paper_ids if paper_id in records]

    @_scoped_endpoint
    def get_cached_reference_ids(self, paper_id: str) -> list[str] | None:
        """Read a validated reference cache entry without network access.

        :param str paper_id: Seed identifier.
        :return list[str] | None: Cached list, including authoritative empty, or miss.
        """
        normalized_id = normalize_paper_id(paper_id)
        cache_path = disk_cache._reference_cache_path(normalized_id)
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("reference cache must be an object")
            if data.get("version") != disk_cache.REFERENCE_CACHE_VERSION:
                return None
            if data.get("paper_id") != normalized_id or "references" not in data:
                raise ValueError("reference cache key mismatch")
            references = payloads._coerce_cached_reference_ids(data["references"])
            if references is None:
                raise ValueError("invalid reference list")
            if data["references"] != references:
                self._persist_reference_cache_entry(
                    cache_path, normalized_id, references
                )
            return references
        except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
            with contextlib.suppress(OSError):
                cache_path.unlink(missing_ok=True)
            return None

    @_scoped_endpoint
    def get_reference_ids(
        self,
        paper_id: str,
        *,
        force_refresh: bool = False,
        raise_on_unavailable: bool = True,
    ) -> list[str]:
        """Fetch all resolved reference IDs, with an independent persistent cache.

        :param str paper_id: Seed identifier.
        :param bool force_refresh: Bypass the reference cache.
        :param bool raise_on_unavailable: Raise on a failed page.
        :return list[str]: Ordered unique reference identifiers.
        """
        normalized_id = normalize_paper_id(paper_id)
        cache_path = disk_cache._reference_cache_path(normalized_id)
        if not force_refresh:
            cached = self.get_cached_reference_ids(normalized_id)
            if cached is not None:
                self._candidate_operation.state.reference_cache_hits += 1
                return cached
        references = self._relation_ids(
            normalized_id,
            relation="references",
            nested_key="citedPaper",
            limit=None,
            raise_on_unavailable=raise_on_unavailable,
        )
        if references is None:
            return []
        self._persist_reference_cache_entry(cache_path, normalized_id, references)
        return references

    @staticmethod
    def _resolve_paper_fields(fields: list[str] | None) -> tuple[list[str], bool]:
        """Resolve requested fields and whether records are safe to cache.

        :param list[str] | None fields: Caller projection or default full metadata.
        :return tuple[list[str], bool]: Requested fields and full-cache flag.
        """
        if fields is None:
            return payloads._default_paper_fields(), True
        requested = list(fields)
        return requested, set(payloads.DEFAULT_PAPER_FIELDS).issubset(requested)

    def _papers_from_records(
        self, records: Iterable[Any], *, cache_full_metadata: bool
    ) -> tuple[list[Paper], int]:
        """Convert records, count malformed rows, and optionally cache them.

        :param Iterable[Any] records: API paper records.
        :param bool cache_full_metadata: Whether records carry default full metadata.
        :return tuple[list[Paper], int]: Converted records and malformed-row count.
        """
        papers: list[Paper] = []
        malformed_records = 0
        for record in records:
            paper = payloads._convert_recommendation(record)
            if paper is None:
                malformed_records += 1
                logger.debug("Skipping malformed Semantic Scholar paper row.")
                continue
            papers.append(paper)
            if cache_full_metadata:
                disk_cache._persist_paper(paper, paper.paper_id)
        return papers, malformed_records

    @_scoped_endpoint
    def get_recommended_papers(
        self,
        paper_id: str,
        limit: int = 50,
        fields: list[str] | None = None,
        *,
        raise_on_unavailable: bool = False,
    ) -> list[Paper]:
        """Freshly fetch recommendation IDs and materialize available papers.

        :param str paper_id: Seed identifier.
        :param int limit: Maximum results, up to 500.
        :param list[str] | None fields: Optional direct-response projection.
        :param bool raise_on_unavailable: Raise on discovery/enrichment failure.
        :return list[Paper]: Ordered recommendations.
        """
        parsed_limit = payloads._validate_integer_limit(
            limit, "limit", maximum=RECOMMENDATION_MAX_RESULTS
        )
        normalized_id = normalize_paper_id(paper_id)
        encoded_id = quote(normalized_id, safe="")
        requested_fields, cache_full_metadata = self._resolve_paper_fields(fields)
        custom_projection = fields is not None
        # The recommendations API expands ``references`` into unsupported nested
        # fields; complete reference IDs remain available through get_reference_ids.
        requested_fields = [
            field for field in requested_fields if field != "references"
        ]
        if custom_projection and "paperId" not in requested_fields:
            requested_fields.append("paperId")

        skipped_records = 0
        for pool in ("recent", "all-cs"):
            params: dict[str, Any] = {
                "fields": ",".join(requested_fields)
                if custom_projection
                else "paperId",
                "limit": parsed_limit,
            }
            if pool == "all-cs":
                params["from"] = pool
            response = self._request_json(
                f"{RECOMMENDATION_BASE_URL}/{encoded_id}",
                params,
                context=f"fetching {pool} recommendations for {normalized_id}",
                raise_on_unavailable=raise_on_unavailable,
            )
            if response is None:
                if raise_on_unavailable:
                    raise SemanticScholarUnavailableError(
                        f"Could not fetch {pool} recommendations for {normalized_id}: "
                        "HTTP 404 (paper not found)."
                    )
                return []
            if not isinstance(response, dict) or not isinstance(
                response.get("recommendedPapers"), list
            ):
                raise _SemanticScholarResponseContractError(
                    "Semantic Scholar returned malformed recommendation data."
                )
            records = response["recommendedPapers"]
            logger.debug(
                "Checked recommendations discovery upstream for %s, %s pool: %d rows.",
                normalized_id,
                pool,
                len(records),
            )
            if custom_projection:
                papers, malformed_records = self._papers_from_records(
                    records, cache_full_metadata=cache_full_metadata
                )
                skipped_records += malformed_records
                if papers:
                    if skipped_records:
                        logger.warning(
                            "Recommendations for %s skipped %d malformed or "
                            "unresolved records.",
                            normalized_id,
                            skipped_records,
                        )
                    return papers[:parsed_limit]
                continue

            paper_ids: list[str] = []
            seen_ids: set[str] = set()
            for record in records:
                raw_id = payloads._payload_get(record, "paperId")
                if not isinstance(raw_id, str) or not raw_id.strip():
                    skipped_records += 1
                    logger.debug("Skipping malformed recommendation row.")
                    continue
                candidate = normalize_paper_id(raw_id)
                if candidate not in seen_ids:
                    seen_ids.add(candidate)
                    paper_ids.append(candidate)
            if not paper_ids:
                continue
            materialized = self.get_papers(
                paper_ids, raise_on_unavailable=raise_on_unavailable
            )
            papers = [
                materialized[paper_id]
                for paper_id in paper_ids
                if paper_id in materialized
            ]
            skipped_records += len(paper_ids) - len(papers)
            if papers:
                if skipped_records:
                    logger.warning(
                        "Recommendations for %s skipped %d malformed or "
                        "unresolved records.",
                        normalized_id,
                        skipped_records,
                    )
                return papers[:parsed_limit]
        if skipped_records:
            logger.warning(
                "Recommendations for %s skipped %d malformed or unresolved records.",
                normalized_id,
                skipped_records,
            )
        return []

    @_scoped_endpoint
    def search_papers(
        self,
        query: str,
        limit: int = 10,
        fields: list[str] | None = None,
        *,
        raise_on_unavailable: bool = False,
    ) -> list[Paper]:
        """Search papers through explicit REST pagination.

        :param str query: Non-empty search text.
        :param int limit: Maximum unique results, up to 1,000.
        :param list[str] | None fields: Optional response projection.
        :param bool raise_on_unavailable: Raise when a page cannot complete.
        :return list[Paper]: Ordered search results.
        """
        parsed_limit = payloads._validate_integer_limit(
            limit, "limit", maximum=SEARCH_MAX_RESULTS
        )
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        normalized_query = query.strip()
        requested_fields, cache_full_metadata = self._resolve_paper_fields(fields)
        if fields is not None and "paperId" not in requested_fields:
            requested_fields.append("paperId")

        records: list[Any] = []
        seen_ids: set[str] = set()
        malformed_records = 0
        offset = 0
        while len(records) < parsed_limit:
            response = self._request_json(
                SEARCH_BASE_URL,
                {
                    "query": normalized_query,
                    "fields": ",".join(requested_fields),
                    "limit": min(SEARCH_PAGE_SIZE, parsed_limit - len(records)),
                    "offset": offset,
                },
                context=f"searching for {normalized_query!r} at offset {offset}",
                raise_on_unavailable=raise_on_unavailable,
                retry_not_found=offset > 0,
            )
            if response is None:
                if raise_on_unavailable:
                    raise SemanticScholarUnavailableError(
                        f"Could not search for {normalized_query!r}: HTTP 404."
                    )
                return []
            if not isinstance(response, dict) or not isinstance(
                response.get("data"), list
            ):
                raise _SemanticScholarResponseContractError(
                    "Semantic Scholar returned malformed search data."
                )
            page = response["data"]
            if not page:
                break
            for record in page:
                raw_id = payloads._payload_get(record, "paperId")
                if not isinstance(raw_id, str) or not raw_id.strip():
                    malformed_records += 1
                    logger.debug("Skipping malformed search row.")
                    continue
                paper_id = normalize_paper_id(raw_id)
                if paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    records.append(record)
                    if len(records) >= parsed_limit:
                        break
            next_offset = response.get("next")
            if next_offset is None:
                break
            if (
                isinstance(next_offset, bool)
                or not isinstance(next_offset, int)
                or next_offset <= offset
            ):
                raise _SemanticScholarResponseContractError(
                    "Semantic Scholar returned invalid search pagination offset."
                )
            offset = next_offset
        papers, conversion_malformed_records = self._papers_from_records(
            records[:parsed_limit], cache_full_metadata=cache_full_metadata
        )
        malformed_records += conversion_malformed_records
        if malformed_records:
            logger.warning(
                "Search for %r skipped %d malformed records.",
                normalized_query,
                malformed_records,
            )
        return papers
