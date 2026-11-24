"""Rich-backed progress helpers shared by the CLI and library modules.

Progress displays render on the same stderr console the CLI gives its logging
handler, so log records print above a live bar instead of tearing through it.
Modules below the CLI import from here rather than constructing their own
console, which keeps a single owner for stream, width, and TTY policy.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from time import strftime
from typing import TypeVar

from rich.console import Console
from rich.errors import LiveError
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Column
from rich.text import Text

from ._runtime import stderr_isatty

T = TypeVar("T")

CONSOLE_TIME_FORMAT = "%H:%M:%S"

_progress_console: Console | None = None
_progress_enabled = True
_active_task: ContextVar[ProgressTask | None] = ContextVar(
    "progress_task", default=None
)
_active_retry_progress: ContextVar[RetryProgress | None] = ContextVar(
    "retry_progress", default=None
)


def set_progress_console(console: Console) -> None:
    """Route progress rendering through a caller-owned console.

    :param Console console: Console shared with the logging handler.
    :return None: Rebinds the module-level console.
    """
    global _progress_console
    _progress_console = console


def set_progress_enabled(enabled: bool) -> None:
    """Enable or suppress progress displays configured by the CLI.

    :param bool enabled: Whether interactive progress displays may render.
    :return None: Stores the process-level display policy.
    """
    global _progress_enabled
    _progress_enabled = bool(enabled)


def progress_enabled(*, require_tty: bool = False) -> bool:
    """Check whether verbosity allows progress, optionally requiring a terminal."""
    return _progress_enabled and (not require_tty or stderr_isatty())


def _resolve_console() -> Console:
    """Return the progress console, defaulting to plain stderr.

    :return Console: Console used for progress rendering.
    """
    if _progress_console is not None:
        return _progress_console
    return Console(stderr=True)


def _columns(width: int, total: int | None) -> Sequence[ProgressColumn]:
    """Keep counts visible and show an ETA only when the total is known."""
    compact = width < 80
    columns: list[ProgressColumn] = []
    if width >= 60:
        columns.append(
            TextColumn("{task.fields[updated]}", style="log.time", markup=False)
        )
    columns.append(
        TextColumn(
            "{task.description}",
            style="progress.description",
            markup=False,
            table_column=Column(ratio=1 if compact else None, no_wrap=True),
        )
    )
    columns.append(
        TextColumn(
            "{task.fields[retry_status]}",
            markup=False,
            table_column=Column(no_wrap=True),
        )
    )
    if not compact:
        columns.append(BarColumn(bar_width=None, table_column=Column(ratio=1)))
    columns.append(MofNCompleteColumn(table_column=Column(no_wrap=True)))
    if not compact:
        columns.append(TextColumn("{task.fields[unit]}", style="dim", markup=False))
    if not compact or total is None:
        columns.append(TimeElapsedColumn())
    if total is not None:
        columns.extend([TextColumn("ETA", style="dim"), TimeRemainingColumn()])
    return columns


class ProgressTask:
    """Handle for advancing a single task on a live progress display."""

    def __init__(
        self, progress: Progress, task_id: TaskID, description_column: ProgressColumn
    ) -> None:
        """Bind a handle to one task of a running display.

        :param Progress progress: Live progress display owning the task.
        :param TaskID task_id: Identifier of the task being advanced.
        :param ProgressColumn description_column: Label column sharing room with retries.
        :return None: Stores the display and task identity.
        """
        self._progress = progress
        self._task_id = task_id
        self._pending_advance = 0
        self._last_update = progress.get_time()
        self._description_column = description_column.get_table_column()
        self._description_ratio = self._description_column.ratio

    def _set_retry_status(self, status: str) -> None:
        """Update retry text in the task row without changing its count."""
        self._description_column.ratio = 1 if status else self._description_ratio
        self._progress.update(
            self._task_id,
            retry_status=status,
            refresh=self._progress.live.is_started,
        )

    @property
    def n(self) -> int:
        """Return the number of completed steps.

        :return int: Steps completed so far.
        """
        for task in self._progress.tasks:
            if task.id == self._task_id:
                return int(task.completed) + self._pending_advance
        return 0

    def update(self, advance: int = 1) -> None:
        """Advance the task by a number of steps.

        :param int advance: Steps to add to the completed count.
        :return None: Updates the live display.
        """
        self._pending_advance += advance
        # Rich keeps at most 1,000 speed samples. Per-row bursts would discard
        # the time spent encoding and leave an ETA based on bookkeeping speed.
        if self._progress.get_time() - self._last_update >= 0.5:
            self._flush()

    def _flush(self) -> None:
        """Report accumulated progress as one speed sample.

        :return None: Advances the display and resets the reporting interval.
        """
        if self._pending_advance:
            self._progress.advance(self._task_id, self._pending_advance)
            if self._pending_advance > 0:
                self._progress.update(
                    self._task_id, updated=strftime(CONSOLE_TIME_FORMAT)
                )
            self._pending_advance = 0
            self._last_update = self._progress.get_time()


class RetryProgress:
    """Show section retries in an active bar or a transient terminal line."""

    def __init__(self, section: str) -> None:
        """Name the section; defer all rendering until its first retry."""
        self.section = section
        self._retries = 0
        self._delay = 0.0
        self._live: Live | None = None
        self._task: ProgressTask | None = None
        self._token: Token[RetryProgress | None] | None = None

    def update(self, retries: int, *, delay: float = 0.0) -> bool:
        """Show the running count and optional backoff; return whether displayed."""
        self._retries, self._delay = retries, delay
        if not retries or not progress_enabled(require_tty=True):
            return False
        console = _resolve_console()
        if not console.is_terminal or console.is_dumb_terminal:
            return False
        if self._token is None:
            self._token = _active_retry_progress.set(self)
        status = f"{retries} {'retry' if retries == 1 else 'retries'}"
        if delay:
            status += (
                f" · retry in {delay:g}s" if console.width >= 80 else f" · {delay:g}s"
            )
        task = _active_task.get()
        if self._task is not None and self._task is not task:
            self._task._set_retry_status("")
        self._task = task
        if task is not None:
            self._stop_live()
            task._set_retry_status(status)
        else:
            prefix = (
                f"{strftime(CONSOLE_TIME_FORMAT)} Semantic Scholar · "
                if console.width >= 60
                else ""
            )
            suffix = f" · {status}"
            label = Text(self.section)
            label.truncate(
                max(0, console.width - Text(prefix + suffix).cell_len),
                overflow="ellipsis",
            )
            line = Text.assemble(prefix, label, suffix, no_wrap=True)
            if self._live is None:
                self._live = Live(
                    line, console=console, transient=True, auto_refresh=False
                )
                try:
                    self._live.start(refresh=True)
                except LiveError:
                    self._live = None
                    self.close()
                    return False
            else:
                self._live.update(line, refresh=True)
        return True

    def _stop_live(self) -> None:
        """Clear the standalone line before another display takes its place."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def close(self) -> None:
        """Clear the retry line or bar text when the section ends."""
        self._stop_live()
        if self._task is not None:
            self._task._set_retry_status("")
            self._task = None
        if self._token is not None:
            _active_retry_progress.reset(self._token)
            self._token = None


@contextmanager
def progress_task(
    *,
    total: int | None,
    description: str,
    unit: str = "items",
    enabled: bool | None = None,
) -> Iterator[ProgressTask]:
    """Run a live progress display for a manually advanced task.

    :param Optional[int] total: Expected step count, or ``None`` when unknown.
    :param str description: Label shown ahead of the bar.
    :param str unit: Noun describing the counted items.
    :param Optional[bool] enabled: Force display on or off; ``None`` shows the
        bar only when stderr is interactive.
    :return Iterator[ProgressTask]: Handle used to advance the task.
    """
    show = enabled is not False and progress_enabled(require_tty=enabled is None)
    console = _resolve_console()
    width = console.width
    columns = _columns(width, total)
    progress = Progress(
        *columns,
        console=console,
        disable=not show,
        expand=True,
    )
    retry_progress = _active_retry_progress.get() if show else None
    if retry_progress is not None:
        retry_progress._stop_live()
    with progress:
        task_id = progress.add_task(
            description,
            total=total,
            unit=unit,
            updated=strftime(CONSOLE_TIME_FORMAT),
            retry_status="",
        )
        task = ProgressTask(progress, task_id, columns[1 if width >= 60 else 0])
        token = _active_task.set(task) if show else None
        try:
            if retry_progress is not None:
                retry_progress.update(
                    retry_progress._retries, delay=retry_progress._delay
                )
            yield task
        finally:
            task._flush()
            if token is not None:
                _active_task.reset(token)


def progress_iterator(
    iterable: Iterable[T],
    *,
    description: str,
    unit: str = "items",
    total: int | None = None,
    enabled: bool | None = None,
) -> Generator[T, None, None]:
    """Yield items from an iterable while rendering a progress bar.

    Closing the returned generator early tears down the display, so callers may
    stop consuming it without leaving a live bar behind.

    :param Iterable[T] iterable: Source items to yield.
    :param str description: Label shown ahead of the bar.
    :param str unit: Noun describing the counted items.
    :param Optional[int] total: Expected item count; inferred from ``iterable``
        when it reports a length.
    :param Optional[bool] enabled: Force display on or off; ``None`` shows the
        bar only when stderr is interactive.
    :return Generator[T, None, None]: Items from ``iterable``, unchanged.
    """
    resolved_total = total
    if resolved_total is None:
        try:
            resolved_total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            resolved_total = None

    with progress_task(
        total=resolved_total,
        description=description,
        unit=unit,
        enabled=enabled,
    ) as task:
        for item in iterable:
            yield item
            task.update(1)
