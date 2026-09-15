"""Tests for arXiv dataset record normalization in the embedding strategy.

These pin the arXiv-specific behavior layered over the shared coercers in
:mod:`citemesh.core.paper_fields`: structured ``authors_parsed`` rows winning
over the free-text ``authors`` string, whitespace-packed category codes, and
the venue key precedence used by the corpus cache, including the hyphenated
``journal-ref`` column the arXiv metadata snapshot actually ships.
"""

from __future__ import annotations

from typing import Any

import pytest

from citemesh.strategies.embedding.records import (
    _anonymous_dataset_paper_id,
    _dataset_record_raw_paper_id,
    _extract_dataset_paper_metadata,
    _newest_records_by_arxiv_id,
    _parse_authors,
    _parse_categories,
    _parse_venue,
)


@pytest.mark.parametrize("identifier_field", ["id", "paper_id", "paperId"])
def test_newest_record_ranking_uses_supported_identifier_fields(
    identifier_field: str,
) -> None:
    """Newest-first ranking must use every identifier hydration accepts.

    :param str identifier_field: Dataset identifier field under test.
    :return None: Selects the chronologically newest paper through that field.
    """
    records = [
        {identifier_field: "2401.00001", "title": "Older"},
        {identifier_field: "2601.00001", "title": "Newer"},
    ]

    assert _newest_records_by_arxiv_id(records, 1) == [records[1]]


@pytest.mark.parametrize("empty_id", [None, "", "   "])
def test_dataset_record_identifier_follows_hydration_precedence(
    empty_id: str | None,
) -> None:
    """Blank preferred fields must fall through to the next supported identifier.

    :param Optional[str] empty_id: Missing or whitespace-only preferred identifier.
    :return None: Resolves ``paper_id`` ahead of ``paperId`` when ``id`` is blank.
    """
    assert (
        _dataset_record_raw_paper_id(
            {"id": empty_id, "paper_id": "2601.00001", "paperId": "2602.00002"}
        )
        == "2601.00001"
    )


@pytest.mark.parametrize(
    ("authors_data", "authors_parsed_data", "expected"),
    [
        ("Ada Example, Blaise Example", None, ["Ada Example", "Blaise Example"]),
        ("  Ada Example ,, Blaise Example  ", None, ["Ada Example", "Blaise Example"]),
        ("", None, []),
        (
            ["Ada Example", "  ", "Blaise Example"],
            None,
            ["Ada Example", "Blaise Example"],
        ),
        ([{"name": " Ada Example "}, {"id": "x"}], None, ["Ada Example"]),
        (None, None, []),
        (
            "ignored, also ignored",
            [["Example", "Ada", ""], ["Example", "Blaise", "Jr."]],
            ["Ada Example", "Blaise Example Jr."],
        ),
        ("Ada Example", [], ["Ada Example"]),
        ("Ada Example", [["", "", ""]], ["Ada Example"]),
        ("Ada Example", ["not-a-row"], ["Ada Example"]),
    ],
)
def test_parse_authors_prefers_structured_arxiv_rows(
    authors_data: Any, authors_parsed_data: Any, expected: list[str]
) -> None:
    """Structured rows win when usable; otherwise the packed string is split.

    :param Any authors_data: Raw ``authors`` dataset field.
    :param Any authors_parsed_data: Raw ``authors_parsed`` dataset field.
    :param list[str] expected: Normalized author names.
    :return None: Checks the structured-first contract and comma splitting.
    """
    assert _parse_authors(authors_data, authors_parsed_data) == expected


@pytest.mark.parametrize(
    ("categories_data", "expected"),
    [
        ("cs.CL cs.LG", ["cs.CL", "cs.LG"]),
        ("cs.CL, cs.LG", ["cs.CL", "cs.LG"]),
        ("  cs.CL  ", ["cs.CL"]),
        (["cs.CL cs.LG", "stat.ML"], ["cs.CL", "cs.LG", "stat.ML"]),
        (["cs.CL", 7, None], ["cs.CL"]),
        ("cs.CL cs.CL", ["cs.CL"]),
        ("", []),
        (None, []),
    ],
)
def test_parse_categories_splits_packed_arxiv_codes(
    categories_data: Any, expected: list[str]
) -> None:
    """arXiv packs codes into one string, so commas and whitespace both split.

    Repeats collapse to the first occurrence, matching the shared coercer.

    :param Any categories_data: Raw ``categories`` dataset field.
    :param list[str] expected: Normalized category codes.
    :return None: Checks splitting, non-string rejection, and deduplication.
    """
    assert _parse_categories(categories_data) == expected


@pytest.mark.parametrize(
    ("paper", "expected"),
    [
        ({"venue": " ICLR "}, "ICLR"),
        ({"venue": "  ", "journal_ref": "Nature 2020"}, "Nature 2020"),
        # The arXiv metadata snapshot spells the column with a hyphen.
        ({"journal-ref": " Nature 2020 "}, "Nature 2020"),
        ({"venue": "  ", "journal-ref": "Nature 2020"}, "Nature 2020"),
        ({"journal": {"name": " JMLR "}}, "JMLR"),
        ({"venue": "ICLR", "journal_ref": "Nature"}, "ICLR"),
        ({"venue": None, "journal_ref": None, "journal": None}, ""),
        ({"venue": None, "journal-ref": None, "journal": None}, ""),
        ({}, ""),
    ],
)
def test_parse_venue_follows_key_precedence(paper: dict, expected: str) -> None:
    """Venue walks ``venue``, ``journal_ref``, ``journal-ref``, then ``journal``.

    :param dict paper: Raw dataset record.
    :param str expected: Normalized venue string.
    :return None: Checks key precedence and the blank-skips-to-next rule.
    """
    assert _parse_venue(paper) == expected


def test_extract_dataset_paper_metadata_normalizes_an_arxiv_row() -> None:
    """A full arXiv row normalizes into the metadata dict the cache stores.

    The row uses the arXiv metadata snapshot's own column spellings, hyphenated
    ``journal-ref`` included.

    :return None: Checks identity, text, and the three coerced fields together.
    """
    metadata = _extract_dataset_paper_metadata(
        {
            "id": "1706.03762v5",
            "title": "  Attention Is All You Need  ",
            "abstract": " We propose the Transformer. ",
            "authors": "ignored",
            "authors_parsed": [["Vaswani", "Ashish", ""]],
            "categories": "cs.CL cs.LG, cs.CL",
            "journal-ref": " NeurIPS 2017 ",
            "update_date": "2017-12-06",
        },
        0,
    )

    assert metadata["paper_id"] == "arxiv:1706.03762"
    assert metadata["arxiv_id"] == "1706.03762"
    assert metadata["title"] == "  Attention Is All You Need  "
    assert metadata["authors"] == ["Ashish Vaswani"]
    assert metadata["categories"] == ["cs.CL", "cs.LG"]
    assert metadata["venue"] == "NeurIPS 2017"
    assert metadata["year"] == 2017


@pytest.mark.parametrize("abstract", [None, "", "   "])
def test_extract_dataset_paper_metadata_uses_summary_when_abstract_is_blank(
    abstract: str | None,
) -> None:
    """A populated summary remains usable when abstract is present but blank.

    :param Optional[str] abstract: Unusable abstract representation under test.
    :return None: Checks that normalized metadata retains the summary text.
    """
    metadata = _extract_dataset_paper_metadata(
        {
            "title": "Summary-only paper",
            "abstract": abstract,
            "summary": "Usable summary",
        },
        0,
    )

    assert metadata["abstract"] == "Usable summary"


def test_anonymous_identity_distinguishes_bibliographically_distinct_works() -> None:
    """Generic shared text must not collapse works with distinct metadata.

    :return None: Checks authors, year, and DOI all participate in identity.
    """
    common = {"title": "Editorial", "abstract": "An editorial note."}
    base_id = _anonymous_dataset_paper_id(
        {
            **common,
            "authors": ["Alice Example"],
            "year": 2024,
            "doi": "10.1234/alpha",
        }
    )

    assert base_id is not None
    assert base_id != _anonymous_dataset_paper_id(
        {
            **common,
            "authors": ["Bob Example"],
            "year": 2024,
            "doi": "10.1234/alpha",
        }
    )
    assert base_id != _anonymous_dataset_paper_id(
        {
            **common,
            "authors": ["Alice Example"],
            "year": 2025,
            "doi": "10.1234/alpha",
        }
    )
    assert base_id != _anonymous_dataset_paper_id(
        {
            **common,
            "authors": ["Alice Example"],
            "year": 2024,
            "doi": "10.1234/beta",
        }
    )


def test_anonymous_identity_is_stable_across_raw_and_cached_metadata() -> None:
    """Normalization must give one identity before and after cache persistence.

    :return None: Checks structured authors, DOI URLs, and summary normalization.
    """
    raw = {
        "title": "  Stable work  ",
        "abstract": None,
        "summary": "  Stable summary  ",
        "authors": "ignored",
        "authors_parsed": [["Example", "Ada", ""]],
        "year": "2024",
        "doi": "https://doi.org/10.1234/EXAMPLE",
    }
    cached = _extract_dataset_paper_metadata(raw, 0)

    assert _anonymous_dataset_paper_id(raw) == _anonymous_dataset_paper_id(cached)


def test_anonymous_identity_uses_distinct_summary_when_abstract_is_blank() -> None:
    """Fallback summaries must contribute to anonymous content identity.

    :return None: Checks blank abstracts cannot collapse distinct summaries.
    """
    first = _anonymous_dataset_paper_id(
        {"title": "Editorial", "abstract": "", "summary": "First summary"}
    )
    second = _anonymous_dataset_paper_id(
        {"title": "Editorial", "abstract": None, "summary": "Second summary"}
    )

    assert first is not None
    assert second is not None
    assert first != second


def test_extract_dataset_paper_metadata_falls_back_for_empty_rows() -> None:
    """Unidentifiable empty rows must not generate meaningless corpus embeddings.

    :return None: Checks the actionable source-row error.
    """
    with pytest.raises(ValueError, match="Dataset row 7 has no paper identifier"):
        _extract_dataset_paper_metadata({"title": "   "}, 7)
