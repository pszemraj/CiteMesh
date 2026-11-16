"""Rich-backed progress helpers shared by the CLI and library modules.

Progress bars render on the same stderr console the CLI gives its logging
handler, so log records print above a live bar instead of tearing through it.
Modules below the CLI import from here rather than constructing their own
console, which keeps a single owner for stream, width, and TTY policy.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from time import strftime
from typing import TypeVar

from rich.console import Console
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

from ._runtime import stderr_isatty

T = TypeVar("T")

CONSOLE_TIME_FORMAT = "%H:%M:%S"

_progress_console: Console | None = None
_progress_enabled = True


def set_progress_console(console: Console) -> None:
    """Route progress rendering through a caller-owned console.

    :param Console console: Console shared with the logging handler.
    :return None: Rebinds the module-level console.
    """
    global _progress_console
    _progress_console = console


def set_progress_enabled(enabled: bool) -> None:
    """Enable or suppress progress displays configured by the CLI.

    :param bool enabled: Whether interactive progress bars may render.
    :return None: Stores the process-level display policy.
    """
    global _progress_enabled
    _progress_enabled = bool(enabled)


def progress_enabled() -> bool:
    """Return whether the CLI currently permits progress displays.

    :return bool: ``True`` when the active verbosity allows progress output.
    """
    return _progress_enabled


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

    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        """Bind a handle to one task of a running display.

        :param Progress progress: Live progress display owning the task.
        :param TaskID task_id: Identifier of the task being advanced.
        :return None: Stores the display and task identity.
        """
        self._progress = progress
        self._task_id = task_id
        self._pending_advance = 0
        self._last_update = progress.get_time()

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
    show = _progress_enabled and (stderr_isatty() if enabled is None else bool(enabled))
    console = _resolve_console()
    progress = Progress(
        *_columns(console.width, total),
        console=console,
        disable=not show,
        expand=True,
    )
    with progress:
        task_id = progress.add_task(
            description,
            total=total,
            unit=unit,
            updated=strftime(CONSOLE_TIME_FORMAT),
        )
        task = ProgressTask(progress, task_id)
        try:
            yield task
        finally:
            task._flush()


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
