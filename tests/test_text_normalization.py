"""Text normalization applied before embedding and in exports."""

from __future__ import annotations

import pytest

from citemesh.core.text_normalization import normalize_text
from citemesh.data.model_profiles import (
    compose_title_abstract_text,
    get_embedding_model_profile,
)
from citemesh.strategies.embedding.text import EmbeddingTask, format_embedding_metadata

WRAPPED_TITLE = "Megalodon: Efficient LLM Pretraining with Unlimited\n  Context Length"
CLEAN_TITLE = "Megalodon: Efficient LLM Pretraining with Unlimited Context Length"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (WRAPPED_TITLE, CLEAN_TITLE),
        ("a​b­c\x07d", "abcd"),
        ("“quoted” — dash…", '"quoted" - dash...'),
        ("x  y\t\r\nz", "x y z"),
        ("O(n²) $\\times$ α, Erdős, 日本語", None),
        ("  ", ""),
        # Pins the policy beyond the fingerprint probes: compatibility characters
        # survive (NFC, not NFKC), minus signs fold to "-", accents compose. A
        # change here changes the embedded text, so the probes must change too.
        (
            "Scaling x² ﬁne ＡＢＣ at 10⁻³, W⁻¹ − x₋₁ \U0001f642 é",
            "Scaling x² ﬁne ＡＢＣ at 10-³, W-¹ - x-₁ \U0001f642 é",
        ),
        ("caf\udce9 ok", "caf? ok"),
    ],
)
def test_normalize_text_cleans_layout_and_invisibles_but_keeps_content(
    raw: str, expected: str | None
) -> None:
    """Hard wraps and invisible characters go; scripts, math and accents stay."""
    out = normalize_text(raw)
    assert out == (raw if expected is None else expected)
    assert normalize_text(out) == out


@pytest.mark.parametrize("task", list(EmbeddingTask))
def test_wrapped_and_clean_text_produce_identical_model_input(
    task: EmbeddingTask,
) -> None:
    """A hard-wrapped corpus row and a clean provider record encode the same text."""
    profile = get_embedding_model_profile("unsloth/embeddinggemma-300m")
    wrapped = format_embedding_metadata(
        profile=profile,
        metadata={"title": WRAPPED_TITLE, "abstract": "First line\n  second line."},
        task=task,
    )
    clean = format_embedding_metadata(
        profile=profile,
        metadata={"title": CLEAN_TITLE, "abstract": "First line second line."},
        task=task,
    )
    assert wrapped == clean
    assert compose_title_abstract_text(
        {"title": WRAPPED_TITLE, "abstract": " x\n y "}
    ) == (f"{CLEAN_TITLE}. x y")
