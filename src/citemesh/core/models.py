"""
Data models for CiteMesh with validation and business logic.

This module defines typed data structures to replace raw dictionaries,
providing type safety, validation, and encapsulation of paper-related logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Author:
    """
    Represents a paper author.

    :ivar name: Full name of the author
    :ivar author_id: Optional Semantic Scholar author ID
    """

    name: str
    author_id: str | None = None

    @property
    def surname(self) -> str:
        """Extract surname from full name.

        :return str: Surname extracted from ``name`` (or ``Unknown`` if missing).
        """
        if not self.name:
            return "Unknown"
        parts = self.name.strip().split()
        return parts[-1] if parts else "Unknown"


@dataclass
class Paper:
    """
    Validated paper model with business logic.

    :ivar paper_id: Unique identifier (S2 paperId, DOI, or arXiv ID)
    :ivar title: Paper title
    :ivar year: Publication year (validated to be reasonable)
    :ivar authors: List of authors
    :ivar citation_count: Number of times cited
    :ivar abstract: Paper abstract text
    :ivar venue: Publication venue/journal/conference label when available
    :ivar arxiv_id: Canonical arXiv identifier suffix when available.
    :ivar doi: DOI suffix when available.
    :ivar categories: List of subject categories (e.g., ArXiv categories)
    :ivar references: List of paper IDs this paper references
    :ivar is_seed: Whether this is the query paper
    """

    paper_id: str
    title: str
    year: int | None
    authors: list[Author] = field(default_factory=list)
    citation_count: int = 0
    abstract: str = ""
    venue: str = ""
    arxiv_id: str = ""
    doi: str = ""
    categories: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    is_seed: bool = False

    def __post_init__(self):
        """Validate paper data after initialization."""
        current_year = datetime.now().year

        # Validate year if available
        if self.year is not None and not 1900 <= self.year <= current_year + 1:
            raise ValueError(
                f"Invalid year: {self.year}. Must be between 1900 and {current_year + 1}"
            )

        # Validate citation count
        if self.citation_count < 0:
            raise ValueError(
                f"Citation count cannot be negative: {self.citation_count}"
            )

        # Ensure title is not empty
        if not self.title or self.title.strip() == "":
            self.title = "Unknown"

    @property
    def first_author_surname(self) -> str:
        """Get first author's surname safely.

        :return str: First listed author surname or ``Unknown`` when absent.
        """
        if self.authors:
            return self.authors[0].surname
        return "Unknown"

    @property
    def author_names(self) -> set[str]:
        """Get set of all author names for comparison.

        :return Set[str]: Unique author names for the paper.
        """
        return {author.name for author in self.authors}

    @property
    def label(self) -> str:
        """Generate display label used in graph annotations.

        :return str: Compact label formatted as ``"<author>, <year>"``.
        """
        year = self.year if self.year is not None else "n.d."
        return f"{self.first_author_surname}, {year}"

    def shares_authors_with(self, other: Paper) -> bool:
        """Check if this paper shares any authors with another paper.

        :param Paper other: Paper to compare authors against.
        :return bool: ``True`` when the author sets intersect.
        """
        return bool(self.author_names & other.author_names)

    def category_overlap(self, other: Paper) -> float:
        """
        Compute category overlap coefficient.

        :return float: Jaccard similarity: |intersection| / |union| Returns 0.0 if either paper has no categories.
        """
        if not self.categories or not other.categories:
            return 0.0

        cats1 = set(self.categories)
        cats2 = set(other.categories)
        intersection = len(cats1 & cats2)
        union = len(cats1 | cats2)

        return intersection / union if union > 0 else 0.0

    def reference_overlap(self, other: Paper) -> float:
        """
        Compute bibliographic coupling strength.

        :return float: Bibliographic coupling coefficient (0.0 to 1.0) Returns 0.0 if either paper has no references.

        Note:
            Semantic Scholar reference lists are API-limited and can be truncated.
            When only partial lists are available, shared-reference overlap can be
            inflated relative to full bibliographies, so treat this score as an
            approximation.
        """
        if not self.references or not other.references:
            return 0.0

        refs1 = set(self.references)
        refs2 = set(other.references)  # Fixed: was using self.references twice
        shared = len(refs1 & refs2)
        denominator = (len(refs1) * len(refs2)) ** 0.5

        return shared / denominator if denominator > 0 else 0.0

    def __hash__(self):
        """Make Paper hashable for use in sets/dicts."""
        return hash(self.paper_id)

    def __eq__(self, other):
        """Papers are equal if they have the same ID."""
        if not isinstance(other, Paper):
            return False
        return self.paper_id == other.paper_id

    def __repr__(self):
        """Concise string representation."""
        return (
            f"Paper(id={self.paper_id[:10]}..., title={self.title[:30]}..., "
            f"year={self.year or 'n.d.'})"
        )
