"""Pytest shared fixtures and constants for consolidated tests."""

from __future__ import annotations

import pytest

from tests._helpers import get_paper_id_normalization_cases


@pytest.fixture
def paper_id_normalization_cases() -> list[tuple[str, str]]:
    """Provide shared normalization fixtures.

    :return list[tuple[str, str]]: Raw paper IDs and expected normalized IDs.
    """
    return get_paper_id_normalization_cases()
