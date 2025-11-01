"""
Semantic Scholar API client with error handling and caching.

This module wraps the Semantic Scholar API with retry logic, caching,
and better error handling to improve reliability.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional

from semanticscholar import SemanticScholar
from semanticscholar.SemanticScholarException import ObjectNotFoundException

from citemesh.core import API_CONFIG, Author, Paper
from citemesh.data import get_cache_dir

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

REFERENCE_CACHE_DIR = get_cache_dir("references")
REFERENCE_CACHE_VERSION = 1


def _reference_cache_path(paper_id: str) -> Path:
    digest = hashlib.sha1(paper_id.encode("utf-8")).hexdigest()
    return REFERENCE_CACHE_DIR / f"{digest}.json"


class SemanticScholarClient:
    """
    Wrapper for Semantic Scholar API with retry logic and caching.

    This client provides:
    - Automatic retry on transient failures
    - In-memory LRU caching for paper metadata
    - Better error messages
    - Rate limiting to avoid throttling
    """

    def __init__(self, timeout: float = API_CONFIG.default_timeout):
        """
        Initialize the API client.

        Args:
            timeout: Request timeout in seconds
        """
        self.client = SemanticScholar(timeout=timeout)
        self.timeout = timeout
        self.last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / API_CONFIG.requests_per_second

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self.last_request_time = time.time()

    def _convert_api_paper(self, api_paper: Any) -> Optional[Paper]:
        """
        Convert Semantic Scholar API response to Paper model.

        Args:
            api_paper: Raw paper object from S2 API

        Returns:
            Paper object or None if conversion fails
        """
        try:
            if not api_paper or not hasattr(api_paper, "paperId"):
                return None

            # Extract authors
            authors = []
            if hasattr(api_paper, "authors") and api_paper.authors:
                for author in api_paper.authors[:3]:  # Limit to first 3
                    if hasattr(author, "name") and author.name:
                        author_id = (
                            getattr(author, "authorId", None)
                            if hasattr(author, "authorId")
                            else None
                        )
                        authors.append(Author(name=author.name, author_id=author_id))

            # Extract fields/categories
            categories = []
            if hasattr(api_paper, "fields") and api_paper.fields:
                categories = [f for f in api_paper.fields if f]
            elif hasattr(api_paper, "fieldsOfStudy") and api_paper.fieldsOfStudy:
                categories = [f for f in api_paper.fieldsOfStudy if f]

            return Paper(
                paper_id=api_paper.paperId,
                title=api_paper.title or "Unknown",
                year=api_paper.year or 2020,
                authors=authors,
                citation_count=api_paper.citationCount or 0,
                abstract=getattr(api_paper, "abstract", "") or "",
                categories=categories,
                references=[],  # Will be populated separately if needed
                is_seed=False,
            )

        except Exception as e:
            logger.warning(f"Failed to convert API paper: {e}")
            return None

    def get_paper(
        self, paper_id: str, fetch_references: bool = False
    ) -> Optional[Paper]:
        """
        Fetch a paper by ID with retry logic.

        Args:
            paper_id: Paper identifier (DOI, arXiv ID, or S2 ID)
            fetch_references: Whether to fetch reference list (slower)

        Returns:
            Paper object or None if not found

        Raises:
            ValueError: If paper_id is invalid
        """
        if not paper_id or not isinstance(paper_id, str):
            raise ValueError(f"Invalid paper ID: {paper_id}")

        paper_id = paper_id.strip()

        for attempt in range(API_CONFIG.max_retries):
            try:
                self._rate_limit()

                # Request specific fields to reduce payload
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
                    logger.warning(f"Paper not found: {paper_id}")
                    return None

                paper = self._convert_api_paper(api_paper)

                # Extract references if requested
                if fetch_references and paper and hasattr(api_paper, "references"):
                    if api_paper.references:
                        paper.references = [
                            ref.paperId
                            for ref in api_paper.references
                            if hasattr(ref, "paperId") and ref.paperId
                        ]

                return paper

            except ObjectNotFoundException:
                logger.warning(f"Paper not found: {paper_id}")
                return None
            except Exception as e:
                if attempt < API_CONFIG.max_retries - 1:
                    wait_time = API_CONFIG.retry_delay * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {paper_id}: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to fetch paper {paper_id} after {API_CONFIG.max_retries} attempts: {e}"
                    )
                    return None

        return None

    def search_paper(self, query: str) -> Optional[Paper]:
        """Search Semantic Scholar by title when a direct identifier is unavailable."""

        if not query:
            return None

        fields = [
            "paperId",
            "title",
            "year",
            "authors",
            "citationCount",
            "abstract",
            "fieldsOfStudy",
        ]

        try:
            results = self.client.search_paper(query, limit=1, fields=fields)
        except Exception as exc:
            logger.warning(f"Semantic Scholar search failed for '{query[:60]}': {exc}")
            return None

        if not results or not getattr(results[0], "paperId", None):
            return None

        return self._convert_api_paper(results[0])

    def get_paper_citations(self, paper_id: str, limit: int = 20) -> List[Paper]:
        """
        Fetch papers that cite the given paper.

        Args:
            paper_id: Paper identifier
            limit: Maximum number of citations to fetch

        Returns:
            List of Paper objects (may be shorter than limit)
        """
        papers = []

        try:
            self._rate_limit()
            citations = self.client.get_paper_citations(paper_id, limit=limit)

            if not citations:
                return papers

            for cit in citations:
                if (
                    hasattr(cit, "paper")
                    and cit.paper
                    and hasattr(cit.paper, "paperId")
                ):
                    paper = self._convert_api_paper(cit.paper)
                    if paper:
                        papers.append(paper)

                if len(papers) >= limit:
                    break

        except ObjectNotFoundException:
            logger.warning(f"Paper not found for citations: {paper_id}")
        except Exception as e:
            logger.warning(f"Failed to fetch citations for {paper_id}: {e}")

        return papers

    def get_paper_references(self, paper_id: str, limit: int = 20) -> List[Paper]:
        """
        Fetch papers referenced by the given paper.

        Args:
            paper_id: Paper identifier
            limit: Maximum number of references to fetch

        Returns:
            List of Paper objects (may be shorter than limit)
        """
        papers = []

        try:
            self._rate_limit()
            references = self.client.get_paper_references(paper_id, limit=limit)

            if not references:
                return papers

            for ref in references:
                if (
                    hasattr(ref, "paper")
                    and ref.paper
                    and hasattr(ref.paper, "paperId")
                ):
                    paper = self._convert_api_paper(ref.paper)
                    if paper:
                        papers.append(paper)

                if len(papers) >= limit:
                    break

        except ObjectNotFoundException:
            logger.warning(f"Paper not found for references: {paper_id}")
        except Exception as e:
            logger.warning(f"Failed to fetch references for {paper_id}: {e}")

        return papers

    def get_reference_ids(self, paper_id: str) -> List[str]:
        """
        Fetch only the reference IDs for a paper (faster than full references).

        Args:
            paper_id: Paper identifier

        Returns:
            List of referenced paper IDs
        """
        cache_path = _reference_cache_path(paper_id)
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                if data.get("version") == REFERENCE_CACHE_VERSION:
                    refs = data.get("references", [])
                    logger.debug(
                        "Loaded %d cached references for %s", len(refs), paper_id
                    )
                    return refs
            except json.JSONDecodeError:
                cache_path.unlink(missing_ok=True)

        try:
            self._rate_limit()
            references = self.client.get_paper_references(paper_id, fields=["paperId"])

            if not references:
                return []

            ref_ids = []
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
                            "paper_id": paper_id,
                            "references": ref_ids,
                            "version": REFERENCE_CACHE_VERSION,
                        }
                    )
                )
            except OSError as exc:
                logger.debug(
                    "Failed to persist reference cache for %s: %s", paper_id, exc
                )

            return ref_ids

        except TypeError:
            logger.debug(
                "Reference payload missing for %s (treating as empty)", paper_id
            )
            return []
        except ObjectNotFoundException:
            logger.warning(f"Paper not found for reference IDs: {paper_id}")
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch reference IDs for {paper_id}: {e}")
            return []


# Global client instance with caching
_client_instance: Optional[SemanticScholarClient] = None


def get_client() -> SemanticScholarClient:
    """Get or create the global API client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = SemanticScholarClient()
    return _client_instance
