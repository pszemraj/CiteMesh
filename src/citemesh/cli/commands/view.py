"""``citemesh view``: open a saved dashboard or HTML export in a browser."""

from __future__ import annotations

import webbrowser
from pathlib import Path

from citemesh.visualization.dashboard.package import DASHBOARD_COLLECTION_FILENAME

from ..console import logger


def _run_view_command(path: Path, browser: str | None) -> int:
    """Open a saved HTML export using the selected browser.

    :param Path path: HTML file or directory containing the saved dashboard.
    :param Optional[str] browser: Browser name, or ``None`` for the system default.
    :return int: Zero if a browser opened, otherwise one with an error message.
    """
    target = path.expanduser()
    try:
        if target.is_dir():
            target = target / DASHBOARD_COLLECTION_FILENAME
        target = target.resolve(strict=True)
        if not target.is_file() or target.suffix.lower() not in {".html", ".htm"}:
            logger.error(
                "Expected a saved HTML file or collection directory: %s. "
                "Open a dashboard and use Add Results to import graph JSON.",
                target,
            )
            return 1
    except OSError as exc:
        logger.error("Cannot read saved results at %s: %s", target, exc)
        return 1

    try:
        open_tab = (
            webbrowser.get(browser).open_new_tab if browser else webbrowser.open_new_tab
        )
        opened = open_tab(target.as_uri())
    except (webbrowser.Error, OSError) as exc:
        logger.error("Could not launch browser: %s. Open %s manually.", exc, target)
        return 1
    if not opened:
        logger.error(
            "Could not open a browser. Open %s manually or choose one with --browser NAME.",
            target,
        )
        return 1
    logger.info("Opened %s", target)
    return 0
