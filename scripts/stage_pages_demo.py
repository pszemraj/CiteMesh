#!/usr/bin/env python3
"""Stage the demo HTML and JavaScript separately for social preview crawlers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def stage_demo(source: Path, destination: Path) -> None:
    """Move embedded scripts and graph data into a hosted JavaScript file."""
    scripts: list[str] = []

    def extract_script(match: re.Match[str]) -> str:
        attributes, content = match.groups()
        if 'type="application/json"' in attributes:
            script_id = re.search(r'\bid="([^"]+)"', attributes)
            assert script_id is not None
            scripts.append(
                f"document.getElementById({json.dumps(script_id.group(1))})"
                f".textContent = {json.dumps(content)};"
            )
            return f"<script{attributes}></script>"
        scripts.append(content)
        return ""

    html = re.sub(
        r"<script([^>]*)>(.*?)</script>",
        extract_script,
        source.read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )
    html = html.replace("</body>", '<script src="dashboard.js"></script>\n</body>')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "index.html").write_text(html, encoding="utf-8")
    (destination / "dashboard.js").write_text("\n;\n".join(scripts), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the source dashboard and destination directory."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="Standalone demo dashboard HTML")
    parser.add_argument("destination", type=Path, help="Pages staging directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Stage the Pages entry point and its companion script."""
    args = parse_args(argv)
    try:
        stage_demo(args.source, args.destination)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
