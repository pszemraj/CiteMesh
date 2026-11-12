"""
Reference-list hydration for bibliographic coupling.

Edge scores compare papers' reference lists. A :class:`ReferenceHydrator`
fetches each collected paper's reference IDs at most once per build, keeps
them in memory, and after the first provider outage stops requesting and reads
only persisted reference caches, so an outage costs one failed request rather
than one per paper.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterable
from typing import TYPE_CHECKING

from citemesh.core import Paper
from citemesh.progress import progress_enabled, progress_iterator
from citemesh.services import SemanticScholarUnavailableError
from citemesh.strategies.candidates import provider_lookup_identifier

if TYPE_CHECKING:
    from citemesh.services.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)


class ReferenceHydrator:
    """Fill papers' reference IDs from Semantic Scholar for one build at a time."""

    def __init__(
        self,
        client: SemanticScholarClient,
        *,
        fetch_references: bool = True,
        refresh_reference_cache: bool = False,
    ) -> None:
        """Bind the provider client and the hydration settings.

        :param SemanticScholarClient client: Provider client.
        :param bool fetch_references: Whether reference lists are fetched at all.
        :param bool refresh_reference_cache: Whether persisted reference caches
            are bypassed.
        """
        self.client = client
        self.fetch_references = fetch_references
        self.refresh_reference_cache = bool(refresh_reference_cache)
        self.reference_cache: dict[str, list[str]] = {}
        self._source_unavailable = False

    def reset(self) -> None:
        """Forget reference lists and outage state from an earlier build.

        :return None: Clears the in-memory cache and the outage flag.
        """
        self.reference_cache.clear()
        self._source_unavailable = False

    def withhold_seed_references(self, seed: Paper) -> None:
        """Record that the provider has no reference list for the seed.

        A bibliography recovered elsewhere must not stand in for the provider's
        list in coupling scores, and the empty provider answer is not repeated.

        :param Paper seed: Build seed.
        :return None: Empties the seed's reference list for this build.
        """
        self.reference_cache[seed.paper_id] = []
        seed.references = []

    def hydrate(self, papers: dict[str, Paper], seed: Paper) -> None:
        """Fill the seed's and the collected papers' reference lists.

        :param Dict[str, Paper] papers: Collected papers, possibly with the seed.
        :param Paper seed: Build seed.
        :return None: Updates paper records and the in-memory cache.
        """
        if self.fetch_references:
            logger.debug("Hydrating reference lists for %d papers...", len(papers))

        self.ensure(seed)

        progress_bar: Generator[tuple[str, Paper], None, None] | None = None
        paper_items: Iterable[tuple[str, Paper]] = papers.items()
        if self.fetch_references and progress_enabled() and len(papers) > 25:
            progress_bar = progress_iterator(
                paper_items,
                description="Hydrating reference lists",
                unit="papers",
            )
            paper_items = progress_bar

        try:
            for paper_id, paper in paper_items:
                if paper_id != seed.paper_id:
                    self.ensure(paper)
        finally:
            if progress_bar is not None:
                progress_bar.close()

    def ensure(self, paper: Paper) -> None:
        """Fill one paper's reference IDs without clobbering an existing list.

        :param Paper paper: Paper record to hydrate in place.
        :return None: Sets ``paper.references`` when a list is available.
        """
        if not self.fetch_references:
            return

        paper_id = str(paper.paper_id).strip()
        if not paper_id:
            return

        if paper.references:
            self.reference_cache.setdefault(paper_id, list(paper.references))
            return

        cached_refs = self.reference_cache.get(paper_id)
        if cached_refs is not None:
            paper.references = list(cached_refs)
            return
        provider_id = provider_lookup_identifier(paper_id, paper)
        if provider_id is None:
            return
        if self._source_unavailable:
            if not self.refresh_reference_cache:
                cached_refs = self.client.get_cached_reference_ids(provider_id)
                if cached_refs is not None:
                    self.reference_cache[paper_id] = cached_refs
                    paper.references = list(cached_refs)
            return

        try:
            paper.references = self._fetch(paper_id, provider_lookup_id=provider_id)
        except SemanticScholarUnavailableError as exc:
            self._source_unavailable = True
            logger.warning(
                "Reference hydration unavailable; continuing with available "
                "reference data."
            )
            logger.debug("Reference hydration failed for %s: %s", paper_id, exc)

    def _fetch(
        self, paper_id: str, *, provider_lookup_id: str | None = None
    ) -> list[str]:
        """Return a paper's reference IDs, from memory unless refreshing.

        :param str paper_id: In-memory cache key for the paper.
        :param Optional[str] provider_lookup_id: Provider-safe request identifier;
            defaults to ``paper_id``.
        :return list[str]: Referenced paper IDs.
        """
        if not self.refresh_reference_cache and paper_id in self.reference_cache:
            return self.reference_cache[paper_id]
        if self.refresh_reference_cache:
            self.reference_cache.pop(paper_id, None)

        ref_ids = self.client.get_reference_ids(
            provider_lookup_id or paper_id,
            force_refresh=self.refresh_reference_cache,
        )
        self.reference_cache[paper_id] = ref_ids
        return ref_ids
