"""Semantic Scholar API client with error handling and caching."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import unquote, urlparse

import requests
from semanticscholar import SemanticScholar
from semanticscholar.SemanticScholarException import ObjectNotFoundException

from citemesh.core import API_CONFIG, Author, Paper
from citemesh.data import get_cache_dir

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

REFERENCE_CACHE_DIR = get_cache_dir("references")
REFERENCE_CACHE_VERSION = 1
RECOMMENDATION_BASE_URL = (
    "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
)
SEARCH_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def _extract_arxiv_identifier(raw_path: str) -> Optional[str]:
    """
    Extract an arXiv identifier from an arXiv URL path.

    :param str raw_path: URL path component (for example ``/abs/2508.14040``).
    :return Optional[str]: Canonical arXiv identifier suffix without prefix (for example ``2508.14040``), or ``None`` when extraction fails.
    """
    segments = [segment for segment in raw_path.strip("/").split("/") if segment]
    if not segments:
        return None

    if segments[0] in {"abs", "pdf"}:
        candidate = "/".join(segments[1:])
    else:
        candidate = "/".join(segments)

    candidate = candidate.strip()
    if not candidate:
        return None

    if candidate.endswith(".pdf"):
        candidate = candidate[:-4]
    candidate = candidate.strip()

    return candidate or None


def normalize_paper_id(paper_id: str) -> str:
    """
    Normalize paper identifiers (including arXiv/DOI URLs) for S2 API calls.

    :param str paper_id: Raw user-provided identifier (ID or URL).
    :return str: Canonical Semantic Scholar paper identifier string.
    :raises ValueError: If the identifier is empty after trimming.
    """
    normalized = paper_id.strip()
    if not normalized:
        raise ValueError(f"Invalid paper ID: {paper_id}")

    lowered = normalized.lower()

    if lowered.startswith("arxiv:"):
        suffix = normalized.split(":", 1)[1].strip()
        if not suffix:
            raise ValueError(f"Invalid paper ID: {paper_id}")
        return f"arxiv:{suffix}"

    if lowered.startswith("http://") or lowered.startswith("https://"):
        parsed = urlparse(normalized)
        host = parsed.netloc.lower()
        path = unquote(parsed.path)

        if host.endswith("arxiv.org"):
            arxiv_id = _extract_arxiv_identifier(path)
            if arxiv_id:
                return f"arxiv:{arxiv_id}"

        if host.endswith("doi.org"):
            doi_id = path.strip("/")
            if doi_id:
                return doi_id

    if lowered.startswith("arxiv.org/"):
        parsed = urlparse(f"https://{normalized}")
        arxiv_id = _extract_arxiv_identifier(unquote(parsed.path))
        if arxiv_id:
            return f"arxiv:{arxiv_id}"

    return normalized


def _reference_cache_path(paper_id: str) -> Path:
    """Build cache file path for a normalized paper identifier.

    :param str paper_id: Normalized paper identifier.
    :return Path: JSON cache path for the paper's reference IDs.
    """
    digest = hashlib.sha1(paper_id.encode("utf-8")).hexdigest()
    return REFERENCE_CACHE_DIR / f"{digest}.json"


class SemanticScholarClient:
    """
    Wrapper for Semantic Scholar API with retry logic and caching.

    This client provides:
    - Automatic retry on transient failures
    - Rate limiting
    - Reference list cache
    - Direct recommendation and search endpoint support
    """

    def __init__(self, timeout: float = API_CONFIG.default_timeout):
        """
        Initialize the API client.

        :param float timeout: Request timeout in seconds
        """
        api_key = os.getenv("S2_API_KEY")

        self.client = SemanticScholar(timeout=timeout, api_key=api_key)
        self.timeout = timeout
        self.last_request_time = 0.0
        self._session = requests.Session()

        if api_key:
            self._session.headers["x-api-key"] = api_key
            logger.info("Using Semantic Scholar API key from S2_API_KEY")

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / API_CONFIG.requests_per_second
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """Detect rate-limit exceptions.

        :param Exception error: Exception from request/client layer.
        :return bool: ``True`` when the error indicates HTTP 429.
        """
        return "429" in str(error)

    def _retry_wait_time(self, error: Optional[Exception], attempt: int) -> float:
        """Compute adaptive retry delay for transient failures.

        :param Optional[Exception] error: Captured exception, if any.
        :param int attempt: Zero-based retry attempt index.
        :return float: Delay in seconds before retry.
        """
        if error:
            retry_after = self._get_retry_after(error)
            if retry_after is not None:
                return retry_after
        if self._is_rate_limit_error(error) if error else False:
            return API_CONFIG.retry_delay * (2**attempt) * 2
        return API_CONFIG.retry_delay * (2**attempt)

    @staticmethod
    def _get_retry_after(error: Exception) -> Optional[float]:
        """Extract Retry-After from library or HTTP errors.

        :param Exception error: Exception instance captured from request.
        :return Optional[float]: Parsed Retry-After value in seconds, if available.
        """
        if isinstance(error, requests.RequestException) and error.response is not None:
            header = error.response.headers.get("Retry-After")
            if header:
                try:
                    return float(header)
                except ValueError:
                    pass
            return None

        for candidate in ("response",):
            response = getattr(error, candidate, None)
            if response is not None and hasattr(response, "headers"):
                header = response.headers.get("Retry-After")  # type: ignore[attr-defined]
                if header:
                    try:
                        return float(header)
                    except ValueError:
                        pass
        return None

    @staticmethod
    def _safe_retry_after(response: requests.Response) -> float:
        """Extract Retry-After from direct HTTP responses safely.

        :param requests.Response response: HTTP response to inspect.
        :return float: Retry delay in seconds (header value or default delay).
        """
        header = response.headers.get("Retry-After")
        if header:
            try:
                return float(header)
            except ValueError:
                logger.debug(
                    "Ignoring invalid Retry-After value %s; using default delay",
                    header,
                )
        return API_CONFIG.retry_delay

    def _convert_api_paper(self, api_paper: Any) -> Optional[Paper]:
        """
        Convert Semantic Scholar API response to Paper model.

        :param Any api_paper: Raw paper object from S2 API
        :return Optional[Paper]: Paper object or None if conversion fails
        """
        try:
            if not api_paper or not hasattr(api_paper, "paperId"):
                return None

            # Extract authors
            authors = []
            if hasattr(api_paper, "authors") and api_paper.authors:
                for author in api_paper.authors[:3]:
                    if hasattr(author, "name") and author.name:
                        author_id = (
                            getattr(author, "authorId", None)
                            if hasattr(author, "authorId")
                            else None
                        )
                        authors.append(Author(name=author.name, author_id=author_id))

            categories = []
            if hasattr(api_paper, "fields") and api_paper.fields:
                categories = [f for f in api_paper.fields if f]
            elif hasattr(api_paper, "fieldsOfStudy") and api_paper.fieldsOfStudy:
                categories = [f for f in api_paper.fieldsOfStudy if f]

            return Paper(
                paper_id=api_paper.paperId,
                title=api_paper.title or "Unknown",
                year=getattr(api_paper, "year", None),
                authors=authors,
                citation_count=api_paper.citationCount or 0,
                abstract=getattr(api_paper, "abstract", "") or "",
                categories=categories,
                references=[],  # Will be populated separately if needed
                is_seed=False,
            )

        except Exception as exc:
            logger.warning("Failed to convert API paper: %s", exc)
            return None

    def _convert_recommendation(self, rec: Dict[str, Any]) -> Optional[Paper]:
        """Convert recommendation/search record dict to a Paper model.

        :param Dict[str, Any] rec: Record returned by recommendation/search APIs.
        :return Optional[Paper]: Parsed Paper model or ``None`` on malformed payload.
        """
        try:
            paper_id = rec.get("paperId")
            if not paper_id:
                return None

            authors = []
            for author_data in rec.get("authors", [])[:3]:
                if isinstance(author_data, dict):
                    name = author_data.get("name")
                    if name:
                        authors.append(
                            Author(
                                name=name,
                                author_id=author_data.get("authorId"),
                            )
                        )

            categories = rec.get("fieldsOfStudy") or rec.get("fields") or []
            if isinstance(categories, str):
                categories = [categories]

            return Paper(
                paper_id=paper_id,
                title=rec.get("title") or "Unknown",
                year=rec.get("year"),
                authors=authors,
                citation_count=rec.get("citationCount", 0) or 0,
                abstract=rec.get("abstract") or "",
                categories=categories,
                references=[],
                is_seed=False,
            )
        except (TypeError, ValueError) as exc:
            logger.debug("Skipping malformed recommendation record: %s", exc)
            return None

    def _request_json(
        self, url: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Request JSON payload from direct Semantic Scholar REST endpoints.

        :param str url: Endpoint URL.
        :param Dict[str, Any] params: Query parameters.
        :return Optional[Dict[str, Any]]: Parsed JSON payload or ``None`` on failure.
        """
        for attempt in range(API_CONFIG.max_retries):
            try:
                self._rate_limit()
                response = self._session.get(url, params=params, timeout=self.timeout)

                if response.status_code == 404:
                    return None

                if response.status_code == 429:
                    retry_after = self._safe_retry_after(response)
                    logger.warning(
                        "Rate limited by Semantic Scholar. Waiting %ss before retry.",
                        retry_after,
                    )
                    if attempt < API_CONFIG.max_retries - 1:
                        time.sleep(retry_after)
                        continue
                    return None

                response.raise_for_status()
                return response.json()

            except requests.RequestException as exc:
                if attempt < API_CONFIG.max_retries - 1:
                    wait_time = self._retry_wait_time(exc, attempt)
                    logger.warning(
                        "Request failed (attempt %s) for %s: %s. Retrying in %ss",
                        attempt + 1,
                        url,
                        exc,
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        "Failed to call %s after %s attempts: %s",
                        url,
                        API_CONFIG.max_retries,
                        exc,
                    )
        return None

    def get_paper(
        self, paper_id: str, fetch_references: bool = False
    ) -> Optional[Paper]:
        """
        Fetch a paper by ID with retry logic.

        :param str paper_id: Paper identifier (DOI, arXiv ID, or S2 ID)
        :param bool fetch_references: Whether to fetch reference list (slower)
        :return Optional[Paper]: Paper object or None if not found
        """
        if not paper_id or not isinstance(paper_id, str):
            raise ValueError(f"Invalid paper ID: {paper_id}")

        paper_id = normalize_paper_id(paper_id)
        for attempt in range(API_CONFIG.max_retries):
            try:
                self._rate_limit()
                fields = [
                    "paperId",
                    "title",
                    "year",
                    "authors",
                    "citationCount",
                    "abstract",
                    "fieldsOfStudy",
                ]
                if fetch_references:
                    fields.append("references")

                api_paper = self.client.get_paper(paper_id, fields=fields)
                if not api_paper:
                    logger.warning("Paper not found: %s", paper_id)
                    return None

                paper = self._convert_api_paper(api_paper)
                if fetch_references and paper and hasattr(api_paper, "references"):
                    if api_paper.references:
                        paper.references = [
                            ref.paperId
                            for ref in api_paper.references
                            if hasattr(ref, "paperId") and ref.paperId
                        ]
                return paper

            except ObjectNotFoundException:
                logger.warning("Paper not found: %s", paper_id)
                return None
            except Exception as exc:
                if attempt < API_CONFIG.max_retries - 1:
                    wait_time = self._retry_wait_time(exc, attempt)
                    logger.warning(
                        "Attempt %s failed for %s: %s. Retrying in %ss",
                        attempt + 1,
                        paper_id,
                        exc,
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        "Failed to fetch paper %s after %s attempts: %s",
                        paper_id,
                        API_CONFIG.max_retries,
                        exc,
                    )
                    return None

        return None

    def get_paper_citations(self, paper_id: str, limit: int = 20) -> List[Paper]:
        """
        Fetch papers that cite the given paper.

        :param str paper_id: Paper identifier
        :param int limit: Maximum number of citations to fetch
        :return List[Paper]: Citation Papers (may be empty).
        """
        return self._get_related_papers(
            paper_id=paper_id,
            limit=limit,
            fetch_method=self.client.get_paper_citations,
            relation_label="citations",
        )

    def get_paper_references(self, paper_id: str, limit: int = 20) -> List[Paper]:
        """
        Fetch papers referenced by the given paper.

        :param str paper_id: Paper identifier
        :param int limit: Maximum number of references to fetch
        :return List[Paper]: List of Paper objects (may be shorter than limit)
        """
        return self._get_related_papers(
            paper_id=paper_id,
            limit=limit,
            fetch_method=self.client.get_paper_references,
            relation_label="references",
        )

    def _get_related_papers(
        self,
        paper_id: str,
        limit: int,
        fetch_method: Callable[..., Any],
        relation_label: str,
    ) -> List[Paper]:
        """Fetch and convert citation-like relation payloads with shared retry logic.

        :param str paper_id: Raw paper identifier.
        :param int limit: Maximum number of relation records to fetch.
        :param Callable[..., Any] fetch_method: Semantic Scholar relation fetch method.
        :param str relation_label: Human-readable label used in logs.
        :return List[Paper]: Converted relation papers.
        """
        papers: List[Paper] = []
        normalized_paper_id = normalize_paper_id(paper_id)

        for attempt in range(API_CONFIG.max_retries):
            try:
                self._rate_limit()
                relation_records = fetch_method(normalized_paper_id, limit=limit)
                if not relation_records:
                    return papers

                for record in relation_records:
                    if (
                        hasattr(record, "paper")
                        and record.paper
                        and hasattr(record.paper, "paperId")
                    ):
                        paper = self._convert_api_paper(record.paper)
                        if paper:
                            papers.append(paper)

                    if len(papers) >= limit:
                        break

                return papers

            except ObjectNotFoundException:
                logger.warning(
                    "Paper not found for %s: %s", relation_label, normalized_paper_id
                )
                return papers
            except Exception as exc:
                if attempt < API_CONFIG.max_retries - 1:
                    wait_time = self._retry_wait_time(exc, attempt)
                    logger.warning(
                        "Failed to fetch %s for %s (attempt %s). Retrying in %ss",
                        relation_label,
                        normalized_paper_id,
                        attempt + 1,
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.warning(
                        "Failed to fetch %s for %s after %s attempts: %s",
                        relation_label,
                        normalized_paper_id,
                        API_CONFIG.max_retries,
                        exc,
                    )

        return papers

    def get_reference_ids(self, paper_id: str) -> List[str]:
        """
        Fetch only the reference IDs for a paper (faster than full references).

        :param str paper_id: Paper identifier
        :return List[str]: List of referenced paper IDs
        """
        normalized_paper_id = normalize_paper_id(paper_id)
        cache_path = _reference_cache_path(normalized_paper_id)
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                if data.get("version") == REFERENCE_CACHE_VERSION:
                    refs = data.get("references", [])
                    logger.debug(
                        "Loaded %d cached references for %s",
                        len(refs),
                        normalized_paper_id,
                    )
                    return refs
            except json.JSONDecodeError:
                cache_path.unlink(missing_ok=True)

        try:
            for attempt in range(API_CONFIG.max_retries):
                try:
                    self._rate_limit()
                    references = self.client.get_paper_references(
                        normalized_paper_id, fields=["paperId"]
                    )
                    if not references:
                        return []

                    ref_ids: List[str] = []
                    for ref in references:
                        if (
                            hasattr(ref, "paper")
                            and ref.paper
                            and hasattr(ref.paper, "paperId")
                        ):
                            ref_ids.append(ref.paper.paperId)

                    try:
                        cache_path.write_text(
                            json.dumps(
                                {
                                    "paper_id": normalized_paper_id,
                                    "references": ref_ids,
                                    "version": REFERENCE_CACHE_VERSION,
                                }
                            )
                        )
                    except OSError as exc:
                        logger.debug(
                            "Failed to persist reference cache for %s: %s",
                            normalized_paper_id,
                            exc,
                        )

                    return ref_ids

                except TypeError:
                    logger.debug(
                        "Reference payload missing for %s (treating as empty)",
                        normalized_paper_id,
                    )
                    return []
                except ObjectNotFoundException:
                    logger.warning(
                        "Paper not found for reference IDs: %s", normalized_paper_id
                    )
                    return []
                except Exception as exc:
                    if attempt < API_CONFIG.max_retries - 1:
                        wait_time = self._retry_wait_time(exc, attempt)
                        logger.warning(
                            "Failed to fetch reference IDs for %s (attempt %s). Retrying in %ss",
                            normalized_paper_id,
                            attempt + 1,
                            wait_time,
                        )
                        time.sleep(wait_time)
                    else:
                        logger.warning(
                            "Failed to fetch reference IDs for %s after %s attempts: %s",
                            normalized_paper_id,
                            API_CONFIG.max_retries,
                            exc,
                        )
                        return []

        except Exception as exc:
            logger.warning(
                "Unexpected error fetching reference IDs for %s: %s",
                normalized_paper_id,
                exc,
            )
            return []

    def get_recommended_papers(
        self, paper_id: str, limit: int = 50, fields: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Get semantically related papers using S2 recommendations.

        :param str paper_id: S2 paper ID
        :param int limit: Maximum recommendations
        :param Optional[List[str]] fields: API fields to return
        :return List[Paper]: Ranked recommendation papers.
        """
        if fields is None:
            fields = [
                "paperId",
                "title",
                "year",
                "authors",
                "citationCount",
                "abstract",
                "fieldsOfStudy",
            ]

        normalized_paper_id = normalize_paper_id(paper_id)
        payload = self._request_json(
            f"{RECOMMENDATION_BASE_URL}/{normalized_paper_id}",
            {"fields": ",".join(fields), "limit": limit},
        )
        if not payload:
            return []

        papers = []
        for rec in payload.get("recommendedPapers", []):
            paper = self._convert_recommendation(rec)
            if paper:
                papers.append(paper)
        return papers

    def search_papers(
        self, query: str, limit: int = 10, fields: Optional[List[str]] = None
    ) -> List[Paper]:
        """Search papers by title or keyword.

        :param str query: Search query string.
        :param int limit: Maximum number of results.
        :param Optional[List[str]] fields: Optional fields list for API payload.
        :return List[Paper]: Search results.
        """
        if fields is None:
            fields = [
                "paperId",
                "title",
                "year",
                "authors",
                "citationCount",
                "abstract",
                "fieldsOfStudy",
            ]

        payload = self._request_json(
            SEARCH_BASE_URL,
            {"query": query, "fields": ",".join(fields), "limit": limit},
        )
        if not payload:
            return []

        papers = []
        for rec in payload.get("data", []):
            paper = self._convert_recommendation(rec)
            if paper:
                papers.append(paper)
        return papers


_client_instance: Optional[SemanticScholarClient] = None
_client_lock = threading.Lock()


def get_client() -> SemanticScholarClient:
    """Get or create the shared Semantic Scholar API client instance.

    :return SemanticScholarClient: Process-wide singleton client.
    """
    global _client_instance
    if _client_instance is None:
        with _client_lock:
            if _client_instance is None:
                _client_instance = SemanticScholarClient()
    return _client_instance


def reset_client() -> None:
    """Reset cached client instance (for testing)."""
    global _client_instance
    with _client_lock:
        _client_instance = None
