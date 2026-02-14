"""Tests for core data models."""

from __future__ import annotations

from datetime import datetime

import pytest

from citemesh.core import Author, Paper


def test_author_surname_extraction() -> None:
    """Author surname extraction should use the last token."""
    assert Author(name="John Smith").surname == "Smith"
    assert Author(name="Jean-Claude Van Damme").surname == "Damme"
    assert Author(name="").surname == "Unknown"


def test_paper_validation_properties_and_age() -> None:
    """Paper should expose validated fields, label helpers, and computed age."""
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

    current_year = datetime.now().year
    recent = Paper(paper_id="test", title="Test", year=current_year - 5)
    assert recent.age == 5


def test_paper_validation_errors() -> None:
    """Paper should reject invalid years and negative citation counts."""
    for year in [1800, 2100]:
        with pytest.raises(ValueError, match="Invalid year"):
            Paper(paper_id="test123", title="Test", year=year)

    with pytest.raises(ValueError, match="Citation count cannot be negative"):
        Paper(paper_id="test123", title="Test", year=2020, citation_count=-5)


def test_overlap_contracts_for_authors_categories_and_references() -> None:
    """Overlap helpers should follow expected author/category/reference semantics."""
    paper1 = Paper(
        paper_id="p1",
        title="Test 1",
        year=2020,
        authors=[Author(name="Alice Smith"), Author(name="Bob Jones")],
        categories=["cs.AI", "cs.LG", "cs.CL"],
        references=["ref1", "ref2", "ref3", "ref4"],
    )
    paper2 = Paper(
        paper_id="p2",
        title="Test 2",
        year=2021,
        authors=[Author(name="Alice Smith"), Author(name="Carol White")],
        categories=["cs.AI", "cs.CL"],
        references=["ref2", "ref3", "ref5", "ref6"],
    )
    paper3 = Paper(
        paper_id="p3",
        title="Test 3",
        year=2021,
        authors=[Author(name="Dave Brown")],
        references=[],
    )

    assert paper1.shares_authors_with(paper2)
    assert not paper1.shares_authors_with(paper3)
    assert abs(paper1.category_overlap(paper2) - 0.667) < 0.01
    assert paper1.reference_overlap(paper2) == 0.5
    assert paper3.reference_overlap(paper2) == 0.0
