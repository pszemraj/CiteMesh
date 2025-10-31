"""
Theme definitions for CiteMesh visualizations and exports.

Provides a small registry of color palettes that keep the look-and-feel
consistent across static and interactive outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from matplotlib import colors as mpl_colors


@dataclass(frozen=True)
class Theme:
    """Visual theme configuration."""

    name: str
    background: str
    node_color_old: Tuple[float, float, float]
    node_color_new: Tuple[float, float, float]
    seed_color: Tuple[float, float, float]
    edge_color: Tuple[float, float, float]
    text_color: str

    def interpolate(self, norm: float) -> Tuple[float, float, float]:
        """Linear interpolation between old and new node colors."""
        r = (
            self.node_color_old[0]
            + (self.node_color_new[0] - self.node_color_old[0]) * norm
        )
        g = (
            self.node_color_old[1]
            + (self.node_color_new[1] - self.node_color_old[1]) * norm
        )
        b = (
            self.node_color_old[2]
            + (self.node_color_new[2] - self.node_color_old[2]) * norm
        )
        return (r, g, b)


THEMES: Dict[str, Theme] = {
    "light": Theme(
        name="light",
        background="#fafafa",
        node_color_old=(0.72, 0.83, 0.89),
        node_color_new=(0.45, 0.64, 0.61),
        seed_color=(0.87, 0.27, 0.27),
        edge_color=(0.5, 0.5, 0.5),
        text_color="#2d3748",
    ),
    "dark": Theme(
        name="dark",
        background="#1a1a1a",
        node_color_old=(0.4, 0.6, 0.8),
        node_color_new=(0.2, 0.8, 0.7),
        seed_color=(0.95, 0.4, 0.35),
        edge_color=(0.6, 0.6, 0.6),
        text_color="#e0e0e0",
    ),
    "solarized": Theme(
        name="solarized",
        background="#002b36",
        node_color_old=(0.51, 0.58, 0.59),
        node_color_new=(0.15, 0.63, 0.6),
        seed_color=(0.86, 0.2, 0.18),
        edge_color=(0.36, 0.43, 0.45),
        text_color="#93a1a1",
    ),
}


def _to_rgb_tuple(color: str) -> Tuple[float, float, float]:
    """Convert matplotlib color specification to RGB tuple (0-1)."""
    rgb = mpl_colors.to_rgb(color)
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def get_theme(name: str) -> Theme:
    """
    Retrieve a Theme by name, with support for 'auto'.

    Args:
        name: Theme identifier.

    Returns:
        Theme instance. Defaults to 'light' when name is unknown.
    """
    if not name:
        return THEMES["light"]

    if name == "auto":
        # Heuristic: pick dark theme when default matplotlib facecolor is dark.
        from matplotlib import rcParams

        facecolor = rcParams.get("figure.facecolor", "#ffffff")
        if not isinstance(facecolor, str):
            facecolor = mpl_colors.to_hex(facecolor)

        r, g, b = _to_rgb_tuple(facecolor)
        brightness = (0.299 * r) + (0.587 * g) + (0.114 * b)
        return THEMES["dark"] if brightness < 0.5 else THEMES["light"]

    return THEMES.get(name, THEMES["light"])
