"""Visualization and export helpers."""

import os

import matplotlib

# Static exports only ever savefig; pin a headless backend so macOS does not
# select the main-thread-only MacOSX GUI backend. A user-set MPLBACKEND wins.
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg", force=False)

from .export import GraphExporter  # noqa: E402
from .render import compute_layout, generate_output_path, visualize_graph  # noqa: E402
from .themes import THEMES, get_theme  # noqa: E402

__all__ = [
    "generate_output_path",
    "visualize_graph",
    "compute_layout",
    "GraphExporter",
    "get_theme",
    "THEMES",
]
