"""Tests for CLI output path and metadata normalization helpers."""

from pathlib import Path

from citemesh.cli import canonicalize_paper_id_for_metadata, resolve_output_paths


def test_resolve_output_paths_preserves_dotted_basename_for_multi_export() -> None:
    """Unrecognized dotted suffixes should be preserved in output basenames."""
    paths = resolve_output_paths(
        base_output_path=Path("out/arxiv-2508.14040-example"),
        selected_formats=["png", "html", "json"],
        explicit_output=True,
    )

    assert paths["png"] == Path("out/arxiv-2508.14040-example.png")
    assert paths["html"] == Path("out/arxiv-2508.14040-example.html")
    assert paths["json"] == Path("out/arxiv-2508.14040-example.json")


def test_resolve_output_paths_replaces_known_suffix_for_single_export() -> None:
    """Known export suffixes should be replaced for single-format outputs."""
    paths = resolve_output_paths(
        base_output_path=Path("reports/example.graphml"),
        selected_formats=["png"],
        explicit_output=True,
    )

    assert paths["png"] == Path("reports/example.png")


def test_resolve_output_paths_keeps_matching_suffix_for_single_export() -> None:
    """Single-format exports should keep explicit output path when extension matches."""
    paths = resolve_output_paths(
        base_output_path=Path("reports/example.plotly.html"),
        selected_formats=["plotly"],
        explicit_output=True,
    )

    assert paths["plotly"] == Path("reports/example.plotly.html")


def test_canonicalize_paper_id_for_metadata_normalizes_arxiv_url() -> None:
    """Metadata should display concise canonical IDs for arXiv URLs."""
    assert (
        canonicalize_paper_id_for_metadata("https://arxiv.org/abs/2508.14040")
        == "arxiv:2508.14040"
    )
