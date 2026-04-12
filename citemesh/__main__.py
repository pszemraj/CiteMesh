"""Module entry point for ``python -m citemesh``."""

import sys

from .cli import main


def run() -> int:
    """Run the CLI using explicit process argv tokens.

    :return int: Process-style exit code from :func:`citemesh.cli.main`.
    """

    return main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(run())
