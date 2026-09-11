"""Export format tables and output-path resolution for CLI runs.

Owns the supported export formats, their file extensions and exporter method
names, and the resolvers that turn a requested base path into concrete
artifact paths for single runs and dashboard collections.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from citemesh.visualization import generate_output_path
from citemesh.visualization.dashboard.package import (
    DASHBOARD_COLLECTION_FILENAME,
    DASHBOARD_PACKAGE_FILENAME,
    LEGACY_DASHBOARD_MANIFEST_FILENAME,
)

__all__ = [
    "DASHBOARD_COLLECTION_FILENAME",
    "DASHBOARD_PACKAGE_FILENAME",
    "EXPORT_EXTENSIONS",
    "EXPORT_FORMATS",
    "KNOWN_EXPORT_SUFFIXES",
    "LEGACY_DASHBOARD_MANIFEST_FILENAME",
    "resolve_dashboard_collection_outputs",
    "resolve_graph_config_path",
    "resolve_output_paths",
]


EXPORT_FORMATS = (
    "png",
    "html",
    "plotly",
    "dashboard",
    "json",
    "csv",
    "bibtex",
    "graphml",
)
EXPORT_EXTENSIONS: dict[str, str] = {
    "png": ".png",
    "html": ".html",
    "plotly": ".plotly.html",
    "dashboard": ".dashboard.html",
    "json": ".json",
    "csv": ".csv",
    "bibtex": ".bib",
    "graphml": ".graphml",
}
KNOWN_EXPORT_SUFFIXES: list[str] = sorted(
    EXPORT_EXTENSIONS.values(), key=len, reverse=True
)
# Table-driven export dispatch: format → GraphExporter method name.
# ``png`` is handled separately (uses ``visualize_graph``, not the exporter).
_EXPORTER_METHOD: dict[str, str] = {
    "html": "to_interactive_html",
    "plotly": "to_plotly_html",
    "dashboard": "to_dashboard_html",
    "json": "to_json",
    "csv": "to_csv",
    "bibtex": "to_bibtex",
    "graphml": "to_graphml",
}
_THEME_AWARE_FORMATS: frozenset = frozenset({"html", "plotly", "dashboard"})

# Verify dispatch coverage at import time — a new EXPORT_FORMATS entry without
# a dispatch mapping will fail fast here rather than silently skip at runtime.
assert set(_EXPORTER_METHOD) | {"png"} == set(EXPORT_FORMATS), (
    f"Export dispatch gap: covered={sorted(set(_EXPORTER_METHOD) | {'png'})}, "
    f"declared={sorted(EXPORT_FORMATS)}"
)


def resolve_output_paths(
    base_output_path: Path,
    selected_formats: list[str],
    explicit_output: bool,
    strategy: str,
) -> dict[str, Path]:
    """
    Resolve final output paths for selected export formats.

    :param Path base_output_path: Path provided by the user or auto-generated filename.
    :param List[str] selected_formats: Export formats selected for this run.
    :param bool explicit_output: True when the user provided ``--output``.
    :param str strategy: Active strategy name used for multi-format directory outputs.
    :return Dict[str, Path]: Mapping of export format -> resolved output path.
    """
    if explicit_output and base_output_path.is_dir():
        basename = strategy or "graph"
        return {
            fmt: base_output_path / f"{basename}{EXPORT_EXTENSIONS[fmt]}"
            for fmt in selected_formats
        }

    base_str = str(base_output_path)
    stripped_base = _strip_known_export_suffix(base_str)
    has_known_suffix = stripped_base != base_str

    output_paths: dict[str, Path] = {}
    if explicit_output and len(selected_formats) > 1:
        if "dashboard" in selected_formats and base_str.lower().endswith(
            EXPORT_EXTENSIONS["dashboard"]
        ):
            for fmt in selected_formats:
                if fmt == "dashboard":
                    output_paths[fmt] = base_output_path
                else:
                    output_paths[fmt] = Path(stripped_base + EXPORT_EXTENSIONS[fmt])
            return output_paths

        output_dir = Path(stripped_base) if has_known_suffix else base_output_path
        basename = strategy or "graph"
        for fmt in selected_formats:
            output_paths[fmt] = output_dir / f"{basename}{EXPORT_EXTENSIONS[fmt]}"
        return output_paths

    if explicit_output and len(selected_formats) == 1:
        fmt = selected_formats[0]
        desired_ext = EXPORT_EXTENSIONS[fmt]

        if base_str.lower().endswith(desired_ext):
            output_paths[fmt] = base_output_path
            return output_paths

        if has_known_suffix:
            output_paths[fmt] = Path(stripped_base + desired_ext)
            return output_paths

        output_paths[fmt] = Path(base_str + desired_ext)
        return output_paths

    for fmt in selected_formats:
        output_paths[fmt] = Path(stripped_base + EXPORT_EXTENSIONS[fmt])

    return output_paths


def _is_standalone_dashboard_output(
    base_output_path: Path,
    selected_formats: list[str],
    explicit_output: bool,
) -> bool:
    """Return whether dashboard export should remain a standalone HTML artifact.

    :param Path base_output_path: User-provided or generated base output path.
    :param List[str] selected_formats: Requested export formats.
    :param bool explicit_output: Whether ``--output`` was provided.
    :return bool: ``True`` when an explicit standalone dashboard path was
        requested, even if additional sibling exports were also selected.
    """
    return (
        explicit_output
        and "dashboard" in selected_formats
        and str(base_output_path).lower().endswith(EXPORT_EXTENSIONS["dashboard"])
        and not base_output_path.is_dir()
    )


def _resolve_dashboard_collection_root(
    base_output_path: Path,
    *,
    explicit_output: bool,
) -> Path:
    """Resolve root directory for shared dashboard collection artifacts.

    :param Path base_output_path: User-provided or generated base output path.
    :param bool explicit_output: Whether ``--output`` was provided.
    :return Path: Collection root directory containing shared dashboard shell.
    """
    if not explicit_output:
        return Path("out")
    if base_output_path.is_dir():
        return base_output_path
    base_str = str(base_output_path)
    stripped_base = _strip_known_export_suffix(base_str)
    if stripped_base != base_str:
        return Path(stripped_base)
    return base_output_path


def resolve_dashboard_collection_outputs(
    *,
    base_output_path: Path,
    selected_formats: list[str],
    explicit_output: bool,
    strategy: str,
    graph: nx.Graph,
    seed_id: str,
) -> tuple[dict[str, Path], Path]:
    """Resolve shared dashboard files and per-seed result artifacts.

    Dashboard state lives in one collection package at the collection root. A
    seed-specific result directory always retains the graph JSON and its build
    sidecar, alongside any additional requested formats.

    :param Path base_output_path: User-provided or generated base output path.
    :param List[str] selected_formats: Requested export formats.
    :param bool explicit_output: Whether ``--output`` was provided.
    :param str strategy: Active strategy name.
    :param nx.Graph graph: Built graph used for run-specific output naming.
    :param str seed_id: Seed node identifier.
    :return tuple[Dict[str, Path], Path]: Resolved output paths and package path.
    """
    collection_root = _resolve_dashboard_collection_root(
        base_output_path,
        explicit_output=explicit_output,
    )
    run_formats = list(
        dict.fromkeys(
            ["json", *(fmt for fmt in selected_formats if fmt != "dashboard")]
        )
    )
    run_base_output_path = generate_output_path(
        graph,
        seed_id,
        output_dir=collection_root,
        strategy=strategy,
    )
    output_paths = resolve_output_paths(
        base_output_path=run_base_output_path,
        selected_formats=run_formats,
        explicit_output=False,
        strategy=strategy,
    )
    output_paths["dashboard"] = collection_root / DASHBOARD_COLLECTION_FILENAME
    return output_paths, collection_root / DASHBOARD_PACKAGE_FILENAME


def _strip_known_export_suffix(filename: str) -> str:
    """Strip a known export suffix from a filename-like token.

    :param str filename: Candidate filename token.
    :return str: Filename with trailing known export suffix removed.
    """
    lowered = filename.lower()
    for suffix in KNOWN_EXPORT_SUFFIXES:
        if lowered.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def resolve_graph_config_path(output_paths: dict[str, Path], strategy: str) -> Path:
    """Resolve sidecar graph-config output path for a build run.

    :param Dict[str, Path] output_paths: Resolved export artifact paths.
    :param str strategy: Active strategy name.
    :return Path: Graph-config JSON output path.
    """
    if not output_paths:
        return Path(f"{strategy or 'graph'}.config.json")

    anchor_path = next(iter(output_paths.values()))
    stem = _strip_known_export_suffix(anchor_path.name)
    if not stem:
        stem = strategy or "graph"
    return anchor_path.parent / f"{stem}.config.json"


def _dashboard_optional_result_paths(output_paths: dict[str, Path]) -> set[Path]:
    """Return every optional export path owned by one collection result.

    :param Dict[str, Path] output_paths: Resolved collection output paths.
    :return Set[Path]: Known optional paths beside the mandatory graph JSON.
    """
    json_path = output_paths["json"]
    basename = _strip_known_export_suffix(json_path.name)
    return {
        json_path.parent / f"{basename}{EXPORT_EXTENSIONS[fmt]}"
        for fmt in EXPORT_FORMATS
        if fmt not in {"dashboard", "json"}
    }
