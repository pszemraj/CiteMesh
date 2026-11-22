"""
Reference-list hydration for bibliographic coupling.

Edge scores compare papers' reference lists. A :class:`ReferenceHydrator`
groups collected papers' reference lookups, keeps completed lists in memory,
and reads only persisted reference caches after the provider is unavailable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from citemesh.core import Paper
from citemesh.core.paper_ids import normalize_paper_id
from citemesh.strategies.candidates import provider_lookup_identifier

if TYPE_CHECKING:
    from citemesh.services.semantic_scholar import SemanticScholarClient


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

    def withhold_focal_references(self, focal: Paper) -> None:
        """Record that the provider has no reference list for the focal paper.

        A bibliography recovered elsewhere must not stand in for the provider's
        list in coupling scores, and the empty provider answer is not repeated.

        :param Paper focal: Focal paper for this build.
        :return None: Empties the focal paper's reference list for this build.
        """
        self.reference_cache[focal.paper_id.strip()] = []
        focal.references = []

    def hydrate(self, papers: dict[str, Paper], focal: Paper) -> None:
        """Fill the focal paper's and the collected papers' reference lists.

        :param Dict[str, Paper] papers: Collected papers, possibly with the focal paper.
        :param Paper focal: Focal paper for this build.
        :return None: Updates paper records and the in-memory cache.
        """
        if not self.fetch_references:
            return

        selected = [
            focal,
            *(paper for key, paper in papers.items() if key != focal.paper_id),
        ]
        targets = [
            (paper, lookup_id)
            for paper in selected
            if (lookup_id := self._pending_lookup(paper)) is not None
        ]
        fetched = self.fetch_many([lookup_id for _, lookup_id in targets])
        for paper, lookup_id in targets:
            if (refs := fetched.get(normalize_paper_id(lookup_id))) is not None:
                self.reference_cache[paper.paper_id.strip()] = refs
                paper.references = list(refs)

    def fetch_many(self, lookup_ids: Sequence[str]) -> dict[str, list[str]]:
        """Hydrate provider lookups, returning completed lists by normalized ID."""
        if not self.fetch_references or not lookup_ids:
            return {}
        if self._source_unavailable:
            if self.refresh_reference_cache:
                return {}
            return {
                paper_id: refs
                for paper_id in dict.fromkeys(map(normalize_paper_id, lookup_ids))
                if (refs := self.client.get_cached_reference_ids(paper_id)) is not None
            }
        result = self.client.get_reference_ids_batch(
            lookup_ids, force_refresh=self.refresh_reference_cache
        )
        self._source_unavailable = result.unavailable
        return result.reference_ids

    def _pending_lookup(self, paper: Paper) -> str | None:
        """Reuse in-memory references or identify a paper requiring a lookup."""
        paper_id = paper.paper_id.strip()
        if not paper_id:
            return None
        if paper.references:
            self.reference_cache.setdefault(paper_id, list(paper.references))
            return None
        cached_refs = self.reference_cache.get(paper_id)
        if cached_refs is not None:
            paper.references = list(cached_refs)
            return None
        return provider_lookup_identifier(paper_id, paper)
