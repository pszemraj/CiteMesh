"""Visualization and export helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._lazy import install_lazy_exports

if TYPE_CHECKING:
    from .export import GraphExporter
    from .render import compute_layout, generate_output_path, visualize_graph
    from .themes import THEMES, get_theme

__all__ = [
    "generate_output_path",
    "visualize_graph",
    "compute_layout",
    "GraphExporter",
    "get_theme",
    "THEMES",
]

# Lazy so importing ``citemesh.visualization.dashboard`` (or any sibling
# submodule) does not drag matplotlib in through the package ``__init__``.
__getattr__, __dir__ = install_lazy_exports(
    globals(),
    {
        "generate_output_path": (".render", "generate_output_path"),
        "visualize_graph": (".render", "visualize_graph"),
        "compute_layout": (".render", "compute_layout"),
        "GraphExporter": (".export", "GraphExporter"),
        "get_theme": (".themes", "get_theme"),
        "THEMES": (".themes", "THEMES"),
    },
)
