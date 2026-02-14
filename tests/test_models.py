"""Tests for core data models."""

from __future__ import annotations

import pytest

from citemesh.core import Author, Paper


def test_author_surname_extraction() -> None:
    """Author surname extraction should use the last token."""
    assert Author(name="John Smith").surname == "Smith"
    assert Author(name="Jean-Claude Van Damme").surname == "Damme"
    assert Author(name="").surname == "Unknown"


def test_paper_validation_and_properties() -> None:
    """Paper should validate key fields and expose helper properties."""
    paper = Paper(
        paper_id="test123",
        title="Test Paper",
        year=2020,
        authors=[Author(name="John Smith")],
        citation_count=10,
    )
    assert paper.paper_id == "test123"
    assert paper.first_author_surname == "Smith"
    assert paper.label == "Smith, 2020"


def test_invalid_year_raises_error() -> None:
    """Year validation should reject implausible values."""
    for year in [1800, 2100]:
        with pytest.raises(ValueError, match="Invalid year"):
            Paper(paper_id="test123", title="Test", year=year)


def test_negative_citation_count_raises_error() -> None:
    """Citation counts must be non-negative."""
    with pytest.raises(ValueError, match="Citation count cannot be negative"):
        Paper(paper_id="test123", title="Test", year=2020, citation_count=-5)


def test_paper_age() -> None:
    """Paper age should be current_year - publication_year."""
    from datetime import datetime

    current_year = datetime.now().year
    paper = Paper(paper_id="test", title="Test", year=current_year - 5)
    assert paper.age == 5


def test_shared_authors() -> None:
    """Author-sharing detection should use overlapping author names."""
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


def test_category_overlap() -> None:
    """Category overlap should compute Jaccard-like overlap."""
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
    assert abs(paper1.category_overlap(paper2) - 0.667) < 0.01


def test_reference_overlap() -> None:
    """Reference overlap should match bibliographic coupling formula."""
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
    assert paper1.reference_overlap(paper2) == 0.5


def test_reference_overlap_no_refs() -> None:
    """Reference overlap should be zero when either side has no references."""
    paper1 = Paper(paper_id="p1", title="Test 1", year=2020, references=[])
    paper2 = Paper(paper_id="p2", title="Test 2", year=2020, references=["ref1"])
    assert paper1.reference_overlap(paper2) == 0.0
