"""Rich consoles and one-shot logging setup for the CiteMesh CLI.

Owns the shared stderr/stdout consoles and their width resolution, the CLI
logger, and :func:`_configure_logging`, which installs handlers exactly once
per process.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from citemesh._runtime import stderr_isatty, stdout_isatty
from citemesh.progress import set_progress_console

DEFAULT_LOG_WIDTH = 0

REDIRECTED_LOG_WIDTH = 140

LOG_LEVEL_CHOICES = ("debug", "info", "warning", "error")


def _resolve_console_width(log_width: int, *, interactive: bool) -> int | None:
    """Resolve the configured Rich console width for a target stream.

    :param int log_width: Requested Rich console width in columns.
    :param bool interactive: Whether the target stream is attached to a TTY.
    :return Optional[int]: Explicit column width or ``None`` for auto sizing.
    """
    resolved_width = int(log_width)
    if resolved_width > 0:
        return resolved_width
    if interactive:
        return None
    return REDIRECTED_LOG_WIDTH


def _size_console(console: Console, log_width: int, *, interactive: bool) -> Console:
    """Resize a console in place for the requested log width.

    ``Console.width`` accepts ``None`` to restore Rich's automatic sizing, so
    reassigning it is equivalent to rebuilding the console -- width is the only
    constructor argument that differs between import time and configuration
    time. Resizing keeps one console per stream, so a handler or progress bar
    already holding a reference keeps writing to the configured console.

    :param Console console: Console to resize in place.
    :param int log_width: Requested Rich console width in columns.
    :param bool interactive: Whether the console's stream is attached to a TTY.
    :return Console: The same console, resized.
    """
    console.width = _resolve_console_width(log_width, interactive=interactive)
    return console


log_console = _size_console(
    Console(stderr=True), DEFAULT_LOG_WIDTH, interactive=stderr_isatty()
)
output_console = _size_console(
    Console(), DEFAULT_LOG_WIDTH, interactive=stdout_isatty()
)

_LOGGING_CONFIGURED = False

logger = logging.getLogger("citemesh.cli")


def _configure_logging(
    *,
    log_level: str = "info",
    log_width: int = DEFAULT_LOG_WIDTH,
    log_file: str | None = None,
) -> None:
    """Configure CLI logging once at runtime.

    :param str log_level: Log level token.
    :param int log_width: Rich console width; non-positive values use stream defaults.
    :param str | None log_file: Optional plain-text log file path.
    :return None: Installs logging handlers and resizes the consoles once.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    level_name = str(log_level).strip().lower()
    if level_name not in LOG_LEVEL_CHOICES:
        level_name = "info"
    resolved_level = getattr(logging, level_name.upper(), logging.INFO)
    _size_console(log_console, log_width, interactive=stderr_isatty())
    _size_console(output_console, log_width, interactive=stdout_isatty())
    # Progress bars share the logging console so records render above a live bar.
    set_progress_console(log_console)
    console_handler = RichHandler(
        console=log_console,
        show_time=False,
        show_path=False,
        rich_tracebacks=False,
        markup=False,
    )
    console_handler.setLevel(resolved_level)
    handlers: list[logging.Handler] = [console_handler]
    if log_file is not None:
        resolved_log_file = Path(log_file).expanduser()
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            resolved_log_file,
            mode="w",
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=resolved_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )
    # Keep third-party HTTP logs concise without import-time side effects.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("semanticscholar").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    _LOGGING_CONFIGURED = True
