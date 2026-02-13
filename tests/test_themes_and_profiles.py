"""Tests for theme detection and embedding model profiles."""

from __future__ import annotations

import pytest

from citemesh.data.model_profiles import get_embedding_model_profile
from citemesh.visualization.themes import get_theme


def test_get_theme_auto_prefers_colorfgbg_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """COLORFGBG should drive auto theme when parseable."""
    monkeypatch.setenv("COLORFGBG", "15;0")
    monkeypatch.delenv("DARKMODE", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    assert get_theme("auto").name == "dark"

    monkeypatch.setenv("COLORFGBG", "0;15")
    assert get_theme("auto").name == "light"


def test_get_theme_auto_falls_back_to_darkmode(monkeypatch: pytest.MonkeyPatch) -> None:
    """DARKMODE should be used when COLORFGBG is unavailable."""
    monkeypatch.delenv("COLORFGBG", raising=False)
    monkeypatch.setenv("DARKMODE", "1")
    assert get_theme("auto").name == "dark"


def test_get_theme_unknown_defaults_to_light() -> None:
    """Unknown theme keys should default to light palette."""
    assert get_theme("not-a-theme").name == "light"


def test_model_profiles_match_expected_formatters() -> None:
    """Gemma profile should format query/document strings as configured."""
    gemma = get_embedding_model_profile("google/embeddinggemma-300m")
    assert gemma.name == "google/embeddinggemma"
    assert gemma.float16_supported is False
    assert gemma.preferred_torch_dtype == "bfloat16"
    assert gemma.use_cuda_autocast is True
    assert gemma.compile_inner_transformer is True
    assert gemma.available_truncate_dims == (768, 512, 256, 128)
    assert gemma.recommended_truncate_dim == 256
    assert gemma.format_query("  attention  ").startswith(
        "task: search result | query:"
    )
    assert (
        gemma.format_document({"title": " Title ", "abstract": " Abstract "})
        == "title: Title | text: Abstract"
    )

    default = get_embedding_model_profile("all-MiniLM-L6-v2")
    assert default.name == "default"
    assert default.preferred_torch_dtype is None
    assert default.use_cuda_autocast is False
    assert default.compile_inner_transformer is False
    assert default.available_truncate_dims is None
    assert default.recommended_truncate_dim is None
    assert default.format_query("plain") == "plain"
    assert default.format_document({"title": "T", "abstract": ""}) == "T"
