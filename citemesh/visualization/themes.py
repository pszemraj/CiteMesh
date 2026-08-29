"""
Theme definitions for CiteMesh visualizations and exports.

Provides a small registry of color palettes that keep the look-and-feel
consistent across static and interactive outputs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Tuple


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
        """Linear interpolation between old and new node colors.

        :param float norm: Normalized position in [0, 1].
        :return Tuple[float, float, float]: Interpolated RGB color tuple.
        """
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


def get_theme(name: str) -> Theme:
    """
    Retrieve a Theme by name, with support for 'auto'.

    :param str name: Theme identifier.
    :return Theme: Theme instance. Defaults to 'light' when name is unknown.
    """
    if not name:
        return THEMES["light"]

    if name == "auto":
        return _detect_terminal_theme()

    return THEMES.get(name, THEMES["light"])


def _detect_terminal_theme() -> Theme:
    """Detect host appearance from explicit environment and system indicators.

    :return Theme: Best-effort inferred terminal theme.
    """
    colorfgbg = os.environ.get("COLORFGBG")
    if colorfgbg:
        parts = colorfgbg.split(";")
        if len(parts) >= 2:
            for token in reversed(parts):
                try:
                    bg = int(token)
                    if bg < 8:
                        return THEMES["dark"]
                    return THEMES["light"]
                except ValueError:
                    continue

    if os.environ.get("DARKMODE") == "1":
        return THEMES["dark"]

    system_theme = _detect_macos_theme()
    if system_theme is not None:
        return system_theme

    if os.environ.get("TERM_PROGRAM") == "iTerm.app":
        return THEMES["light"]

    return THEMES["light"]


def _detect_macos_theme() -> Theme | None:
    """Read the active macOS interface appearance when available.

    macOS omits ``AppleInterfaceStyle`` for the light appearance and writes
    ``Dark`` when dark appearance is active. Failures other than that normal
    missing-key case fall back to the cross-platform terminal heuristics.

    :return Theme | None: Active macOS theme, or ``None`` off macOS/on failure.
    """
    if sys.platform != "darwin":
        return None

    try:
        result = subprocess.run(
            ["defaults", "read", "-g", "AppleInterfaceStyle"],
            capture_output=True,
            check=False,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode == 0:
        return (
            THEMES["dark"]
            if result.stdout.strip().casefold() == "dark"
            else THEMES["light"]
        )
    if "does not exist" in result.stderr.casefold():
        return THEMES["light"]
    return None
