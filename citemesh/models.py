"""
Data models for CiteMesh with validation and business logic.

This module defines typed data structures to replace raw dictionaries,
providing type safety, validation, and encapsulation of paper-related logic.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set


@dataclass
class Author:
    """
    Represents a paper author.

    Attributes:
        name: Full name of the author
        author_id: Optional Semantic Scholar author ID
    """

    name: str
    author_id: Optional[str] = None

    @property
    def surname(self) -> str:
        """Extract surname from full name."""
        if not self.name:
            return "Unknown"
        parts = self.name.strip().split()
        return parts[-1] if parts else "Unknown"


@dataclass
class Paper:
    """
    Validated paper model with business logic.

    This replaces raw dictionary manipulation with a typed structure
    that validates data and provides convenient access methods.

    Attributes:
        paper_id: Unique identifier (S2 paperId, DOI, or arXiv ID)
        title: Paper title
        year: Publication year (validated to be reasonable)
        authors: List of authors
        citation_count: Number of times cited
        abstract: Paper abstract text
        categories: List of subject categories (e.g., ArXiv categories)
        references: List of paper IDs this paper references
        is_seed: Whether this is the query paper
    """

    paper_id: str
    title: str
    year: int
    authors: List[Author] = field(default_factory=list)
    citation_count: int = 0
    abstract: str = ""
    categories: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    is_seed: bool = False

    def __post_init__(self):
        """Validate paper data after initialization."""
        current_year = datetime.now().year

        # Validate year
        if not 1900 <= self.year <= current_year + 1:
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
        """Get first author's surname safely."""
        if self.authors:
            return self.authors[0].surname
        return "Unknown"

    @property
    def age(self) -> int:
        """Paper age in years from current date."""
        return datetime.now().year - self.year

    @property
    def author_names(self) -> Set[str]:
        """Get set of all author names for comparison."""
        return {author.name for author in self.authors}

    @property
    def label(self) -> str:
        """Generate display label: 'FirstAuthor, Year'."""
        return f"{self.first_author_surname}, {self.year}"

    def shares_authors_with(self, other: "Paper") -> bool:
        """Check if this paper shares any authors with another paper."""
        return bool(self.author_names & other.author_names)

    def category_overlap(self, other: "Paper") -> float:
        """
        Compute category overlap coefficient.

        Returns:
            Jaccard similarity: |intersection| / |union|
            Returns 0.0 if either paper has no categories.
        """
        if not self.categories or not other.categories:
            return 0.0

        cats1 = set(self.categories)
        cats2 = set(other.categories)
        intersection = len(cats1 & cats2)
        union = len(cats1 | cats2)

        return intersection / union if union > 0 else 0.0

    def reference_overlap(self, other: "Paper") -> float:
        """
        Compute bibliographic coupling strength.

        Follows Kessler (1963) formula:
            coupling = |shared_refs| / sqrt(|refs1| * |refs2|)

        Returns:
            Bibliographic coupling coefficient (0.0 to 1.0)
            Returns 0.0 if either paper has no references.
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
        return f"Paper(id={self.paper_id[:10]}..., title={self.title[:30]}..., year={self.year})"
