"""Tests for the shared paper-field coercers used by every ingestion source."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from citemesh.core.paper_fields import (
    coerce_author_name,
    coerce_authors,
    coerce_categories,
    coerce_venue,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  ICLR  ", "ICLR"),
        ("   ", ""),
        ({"name": " NeurIPS "}, "NeurIPS"),
        ({"name": "   "}, ""),
        ({"name": None}, ""),
        ({}, ""),
        (SimpleNamespace(name=" JMLR "), "JMLR"),
        (SimpleNamespace(), ""),
        (None, ""),
        (42, ""),
    ],
)
def test_coerce_venue_accepts_every_source_shape(raw: object, expected: str) -> None:
    """Strings, ``name`` mappings, objects, and junk all resolve to a display string.

    :param object raw: Raw venue value from a payload or dataset record.
    :param str expected: Normalized venue string.
    :return None: Checks stripping and the empty-when-unavailable contract.
    """
    assert coerce_venue(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Ada Example ", "Ada Example"),
        ({"name": " Ada Example ", "authorId": "a-1"}, "Ada Example"),
        ({"name": " "}, ""),
        ({}, ""),
        (SimpleNamespace(name="Blaise Example"), "Blaise Example"),
        (None, ""),
    ],
)
def test_coerce_author_name_reads_records_and_plain_names(
    raw: object, expected: str
) -> None:
    """One author entry normalizes from a record, an object, or a bare name.

    :param object raw: Raw author entry.
    :param str expected: Normalized author name.
    :return None: Checks stripping and the empty-when-unavailable contract.
    """
    assert coerce_author_name(raw) == expected


def test_coerce_authors_normalizes_list_shapes_and_keeps_order() -> None:
    """Mixed record/plain-name lists keep source order and drop unusable entries.

    :return None: Checks blank, missing-name, and non-record entries are dropped.
    """
    assert coerce_authors(
        [
            {"name": " Ada Example ", "authorId": "a-1"},
            {"name": " "},
            {},
            "Blaise Example",
            SimpleNamespace(name="Chien Example"),
            None,
            7,
        ]
    ) == ["Ada Example", "Blaise Example", "Chien Example"]


def test_coerce_authors_preserves_duplicate_display_names() -> None:
    """Two authors may legitimately share a name, so names are never deduplicated.

    :return None: Checks both occurrences survive.
    """
    assert coerce_authors(["Ada Example", "Ada Example"]) == [
        "Ada Example",
        "Ada Example",
    ]


def test_coerce_authors_splits_a_delimited_string_only_when_asked() -> None:
    """arXiv packs authors into one comma-separated string; S2 always sends a list.

    :return None: Checks both sides of the documented ``split_string`` divergence.
    """
    packed = " Ada Example,  Blaise Example ,, "
    assert coerce_authors(packed) == ["Ada Example", "Blaise Example"]
    assert coerce_authors(packed, split_string=False) == []


@pytest.mark.parametrize("raw", [None, 7, {"authors": []}])
def test_coerce_authors_rejects_unusable_payloads(raw: object) -> None:
    """Anything that is neither a list nor a string yields no authors.

    :param object raw: Unusable authors payload.
    :return None: Checks the empty-list contract.
    """
    assert coerce_authors(raw) == []


def test_coerce_categories_keeps_multiword_labels_whole_by_default() -> None:
    """Semantic Scholar's ``fieldsOfStudy`` labels contain spaces and must survive.

    :return None: Checks no whitespace splitting without the opt-in keyword.
    """
    assert coerce_categories(["Computer Science", "Mathematics"]) == [
        "Computer Science",
        "Mathematics",
    ]
    assert coerce_categories("Computer Science") == ["Computer Science"]


def test_coerce_categories_splits_packed_arxiv_codes_when_asked() -> None:
    """arXiv packs several codes into one string, separated by spaces or commas.

    :return None: Checks both the single-string and list-of-packed-strings shapes.
    """
    assert coerce_categories("cs.CL cs.LG", split_whitespace=True) == [
        "cs.CL",
        "cs.LG",
    ]
    assert coerce_categories(["cs.CL,cs.AI", "stat.ML"], split_whitespace=True) == [
        "cs.CL",
        "cs.AI",
        "stat.ML",
    ]


def test_coerce_categories_deduplicates_and_drops_unusable_entries() -> None:
    """Repeats keep their first position; blanks and non-strings are dropped.

    :return None: Checks deduplication, stripping, and entry filtering.
    """
    assert coerce_categories([" cs.AI ", "cs.LG", "cs.AI", "  ", None, 3, ""]) == [
        "cs.AI",
        "cs.LG",
    ]


@pytest.mark.parametrize("raw", [None, 7, {"categories": []}])
def test_coerce_categories_rejects_unusable_payloads(raw: object) -> None:
    """Anything that is neither a list nor a string yields no categories.

    :param object raw: Unusable categories payload.
    :return None: Checks the empty-list contract.
    """
    assert coerce_categories(raw) == []
