"""
Tests for data models.
"""

import pytest

from citemesh.models import Author, Paper


class TestAuthor:
    """Test Author model."""

    def test_surname_extraction(self) -> None:
        """Test surname extraction from full name."""
        author = Author(name="John Smith")
        assert author.surname == "Smith"

        author = Author(name="Jean-Claude Van Damme")
        assert author.surname == "Damme"

    def test_empty_name(self) -> None:
        """Test handling of empty name."""
        author = Author(name="")
        assert author.surname == "Unknown"


class TestPaper:
    """Test Paper model."""

    def test_valid_paper_creation(self) -> None:
        """Test creating a valid paper."""
        paper = Paper(
            paper_id="test123",
            title="Test Paper",
            year=2020,
            authors=[Author(name="John Smith")],
            citation_count=10,
        )

        assert paper.paper_id == "test123"
        assert paper.title == "Test Paper"
        assert paper.year == 2020
        assert paper.citation_count == 10
        assert paper.first_author_surname == "Smith"

    def test_invalid_year_raises_error(self) -> None:
        """Test that invalid year raises ValueError."""
        with pytest.raises(ValueError, match="Invalid year"):
            Paper(
                paper_id="test123",
                title="Test",
                year=1800,  # Too old
            )

        with pytest.raises(ValueError, match="Invalid year"):
            Paper(
                paper_id="test123",
                title="Test",
                year=2100,  # Too future
            )

    def test_negative_citation_count_raises_error(self) -> None:
        """Test that negative citations raise ValueError."""
        with pytest.raises(ValueError, match="Citation count cannot be negative"):
            Paper(
                paper_id="test123",
                title="Test",
                year=2020,
                citation_count=-5,
            )

    def test_paper_age(self) -> None:
        """Test paper age calculation."""
        from datetime import datetime

        current_year = datetime.now().year
        paper = Paper(paper_id="test", title="Test", year=current_year - 5)

        assert paper.age == 5

    def test_label_generation(self) -> None:
        """Test label generation."""
        paper = Paper(
            paper_id="test",
            title="Test",
            year=2020,
            authors=[Author(name="Alice Johnson")],
        )

        assert paper.label == "Johnson, 2020"

    def test_shared_authors(self) -> None:
        """Test author sharing detection."""
        paper1 = Paper(
            paper_id="p1",
            title="Test 1",
            year=2020,
            authors=[Author(name="Alice Smith"), Author(name="Bob Jones")],
        )

        paper2 = Paper(
            paper_id="p2",
            title="Test 2",
            year=2021,
            authors=[Author(name="Alice Smith"), Author(name="Carol White")],
        )

        paper3 = Paper(
            paper_id="p3",
            title="Test 3",
            year=2021,
            authors=[Author(name="Dave Brown")],
        )

        assert paper1.shares_authors_with(paper2)
        assert not paper1.shares_authors_with(paper3)

    def test_category_overlap(self) -> None:
        """Test category overlap calculation."""
        paper1 = Paper(
            paper_id="p1",
            title="Test 1",
            year=2020,
            categories=["cs.AI", "cs.LG", "cs.CL"],
        )

        paper2 = Paper(
            paper_id="p2",
            title="Test 2",
            year=2020,
            categories=["cs.AI", "cs.CL"],
        )

        overlap = paper1.category_overlap(paper2)
        # Intersection: {cs.AI, cs.CL} = 2
        # Union: {cs.AI, cs.LG, cs.CL} = 3
        # Overlap: 2/3 ≈ 0.667
        assert abs(overlap - 0.667) < 0.01

    def test_reference_overlap(self) -> None:
        """Test bibliographic coupling calculation."""
        paper1 = Paper(
            paper_id="p1",
            title="Test 1",
            year=2020,
            references=["ref1", "ref2", "ref3", "ref4"],
        )

        paper2 = Paper(
            paper_id="p2",
            title="Test 2",
            year=2020,
            references=["ref2", "ref3", "ref5", "ref6"],
        )

        coupling = paper1.reference_overlap(paper2)
        # Shared: {ref2, ref3} = 2
        # sqrt(4 * 4) = 4
        # Coupling: 2/4 = 0.5
        assert coupling == 0.5

    def test_reference_overlap_no_refs(self) -> None:
        """Test bibliographic coupling with no references."""
        paper1 = Paper(paper_id="p1", title="Test 1", year=2020, references=[])

        paper2 = Paper(paper_id="p2", title="Test 2", year=2020, references=["ref1"])

        assert paper1.reference_overlap(paper2) == 0.0
