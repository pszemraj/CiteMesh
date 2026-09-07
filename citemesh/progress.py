"""Rich-backed progress helpers shared by the CLI and library modules.

Progress bars render on the same stderr console the CLI gives its logging
handler, so log records print above a live bar instead of tearing through it.
Modules below the CLI import from here rather than constructing their own
console, which keeps a single owner for stream, width, and TTY policy.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, Optional, Sequence, TypeVar

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

_progress_console: Optional[Console] = None


def set_progress_console(console: Console) -> None:
    """Route progress rendering through a caller-owned console.

    :param Console console: Console shared with the logging handler.
    :return None: Rebinds the module-level console.
    """
    global _progress_console
    _progress_console = console


def _resolve_console() -> Console:
    """Return the progress console, defaulting to plain stderr.

    :return Console: Console used for progress rendering.
    """
    if _progress_console is not None:
        return _progress_console
    return Console(stderr=True)


def _columns(unit: str) -> Sequence[ProgressColumn]:
    """Build the shared column layout for a progress display.

    ``unit`` is passed as a task field rather than interpolated into a column
    template so a caller-supplied label can never be parsed as console markup.

    :param str unit: Noun describing the counted items.
    :return Sequence[ProgressColumn]: Ordered progress columns.
    """
    return (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, table_column=Column(ratio=1)),
        MofNCompleteColumn(table_column=Column(no_wrap=True)),
        TextColumn("{task.fields[unit]}", style="dim"),
        TimeElapsedColumn(),
        TextColumn("ETA", style="dim"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[postfix]}", style="dim"),
    )


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

    @property
    def n(self) -> int:
        """Return the number of completed steps.

        :return int: Steps completed so far.
        """
        for task in self._progress.tasks:
            if task.id == self._task_id:
                return int(task.completed)
        return 0

    def update(self, advance: int = 1) -> None:
        """Advance the task by a number of steps.

        :param int advance: Steps to add to the completed count.
        :return None: Updates the live display.
        """
        self._progress.advance(self._task_id, advance)

    def set_postfix_str(self, text: str) -> None:
        """Set the trailing status text shown after the timer.

        :param str text: Status text to display.
        :return None: Updates the live display.
        """
        self._progress.update(self._task_id, postfix=str(text))


@contextmanager
def progress_task(
    *,
    total: Optional[int],
    description: str,
    unit: str = "items",
    enabled: Optional[bool] = None,
) -> Iterator[ProgressTask]:
    """Run a live progress display for a manually advanced task.

    :param Optional[int] total: Expected step count, or ``None`` when unknown.
    :param str description: Label shown ahead of the bar.
    :param str unit: Noun describing the counted items.
    :param Optional[bool] enabled: Force display on or off; ``None`` shows the
        bar only when stderr is interactive.
    :return Iterator[ProgressTask]: Handle used to advance the task.
    """
    show = stderr_isatty() if enabled is None else bool(enabled)
    progress = Progress(
        *_columns(unit),
        console=_resolve_console(),
        disable=not show,
        expand=True,
    )
    with progress:
        task_id = progress.add_task(
            description,
            total=total,
            unit=unit,
            postfix="",
        )
        yield ProgressTask(progress, task_id)


def progress_iterator(
    iterable: Iterable[T],
    *,
    description: str,
    unit: str = "items",
    total: Optional[int] = None,
    enabled: Optional[bool] = None,
) -> Iterator[T]:
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
    :return Iterator[T]: Items from ``iterable``, unchanged.
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
