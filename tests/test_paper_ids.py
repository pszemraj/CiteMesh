"""Contracts for the arXiv-issued DOI rule in :mod:`citemesh.core.paper_ids`."""

from __future__ import annotations

import pytest

from citemesh.core.paper_ids import arxiv_identifier_from_doi


@pytest.mark.parametrize(
    ("doi", "expected"),
    [
        ("10.48550/arXiv.2410.05192", "arxiv:2410.05192"),
        ("https://doi.org/10.48550/arXiv.2410.05192", "arxiv:2410.05192"),
        ("DOI:10.48550/ARXIV.2410.05192", "arxiv:2410.05192"),
        ("2410.05192v2", None),
        ("arXiv:2410.05192V2", None),
        ("https://doi.org/10.1000/ABC.Def", None),
        ("", None),
    ],
)
def test_arxiv_issued_doi_is_recognized_in_every_doi_spelling(
    doi: str, expected: str | None
) -> None:
    """Every recognized DOI spelling of an arXiv-issued DOI names one arXiv ID.

    :param str doi: Identifier candidate.
    :param str | None expected: Canonical arXiv ID, or ``None``.
    :return None: Checks normalization inside the rule rather than at callers.
    """
    assert arxiv_identifier_from_doi(doi) == expected
