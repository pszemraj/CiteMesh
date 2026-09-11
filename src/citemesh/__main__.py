"""Module entry point for ``python -m citemesh``."""

from __future__ import annotations

from .cli import main


def run() -> int:
    """Run the CLI using argparse's default process-argv handling.

    :return int: Process-style exit code from :func:`citemesh.cli.main`.
    """

    return main()


if __name__ == "__main__":
    raise SystemExit(run())
