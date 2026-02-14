"""Tests for CLI output path and metadata normalization helpers."""

from pathlib import Path

import networkx as nx

from citemesh.cli import canonicalize_paper_id_for_metadata, resolve_output_paths
from citemesh.visualization import generate_output_path
from tests.conftest import get_paper_id_normalization_cases


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


def test_canonicalize_paper_id_for_metadata_normalizes_arxiv_urls() -> None:
    """Metadata should display concise canonical IDs for arXiv URLs."""
    for raw_id, expected in get_paper_id_normalization_cases():
        if raw_id.startswith("http://") or raw_id.startswith("https://"):
            assert canonicalize_paper_id_for_metadata(raw_id) == expected


def test_generate_output_path_includes_seed_suffix_for_collision_safety() -> None:
    """Same title with different seed IDs should map to different output folders."""
    graph = nx.Graph()
    graph.add_node("seed-a", title="A Survey of Transformers")
    graph.add_node("seed-b", title="A Survey of Transformers")

    path_a = generate_output_path(graph, seed_id="seed-a", output_dir=Path("out"))
    path_b = generate_output_path(graph, seed_id="seed-b", output_dir=Path("out"))

    assert path_a.parent != path_b.parent
    assert path_a.parent.name.startswith("a-survey-of-transformers-")
    assert path_b.parent.name.startswith("a-survey-of-transformers-")
