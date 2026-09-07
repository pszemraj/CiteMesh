"""Tests for the Rich-backed progress helpers."""

from __future__ import annotations

import io
from typing import Any, Iterator

import pytest
from rich.console import Console
from rich.progress import Progress

from citemesh import progress as progress_module


@pytest.fixture
def progress_console(monkeypatch: pytest.MonkeyPatch) -> Console:
    """Route progress rendering through a recording in-memory console.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to restore the global.
    :return Console: Console whose ``export_text`` holds the rendered frames.
    """
    console = Console(
        file=io.StringIO(),
        force_terminal=True,
        no_color=True,
        record=True,
        width=80,
    )
    # Registering the global with monkeypatch first makes the public setter's
    # mutation revert at teardown, which the setter itself cannot undo.
    monkeypatch.setattr(progress_module, "_progress_console", None)
    progress_module.set_progress_console(console)
    return console


@pytest.fixture
def recorded_displays(monkeypatch: pytest.MonkeyPatch) -> list[Progress]:
    """Capture the progress displays the module constructs.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to restore the class.
    :return list[Progress]: Displays appended in construction order.
    """
    displays: list[Progress] = []

    class _RecordingProgress(progress_module.Progress):
        """Progress display that registers itself on construction."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Build a normal display and record it for later inspection.

            :param Any args: Positional arguments forwarded to ``Progress``.
            :param Any kwargs: Keyword arguments forwarded to ``Progress``.
            :return None: Appends the new display to the captured list.
            """
            super().__init__(*args, **kwargs)
            displays.append(self)

    monkeypatch.setattr(progress_module, "Progress", _RecordingProgress)
    return displays


def _unsized_letters() -> Iterator[str]:
    """Yield letters from a source that reports no length.

    :return Iterator[str]: Two letters, leaving the total unknown.
    """
    yield "a"
    yield "b"


def test_progress_iterator_with_zero_total_renders_a_bounded_bar(
    progress_console: Console,
    recorded_displays: list[Progress],
) -> None:
    """An empty source should render ``0/0`` rather than divide by zero."""
    items = list(
        progress_module.progress_iterator(
            [],
            description="scan",
            unit="papers",
            enabled=True,
        )
    )

    task = recorded_displays[0].tasks[0]
    assert items == []
    assert task.total == 0
    assert task.percentage == 0.0
    assert "0/0" in progress_console.export_text()


def test_progress_iterator_without_a_total_renders_indeterminate_counts(
    progress_console: Console,
    recorded_displays: list[Progress],
) -> None:
    """An unsized source should keep the total unknown and count against ``?``."""
    items = list(
        progress_module.progress_iterator(
            _unsized_letters(),
            description="scan",
            enabled=True,
        )
    )

    task = recorded_displays[0].tasks[0]
    assert items == ["a", "b"]
    assert task.total is None
    assert "2/?" in progress_console.export_text()


def test_progress_iterator_disabled_yields_items_without_rendering(
    progress_console: Console,
    recorded_displays: list[Progress],
) -> None:
    """A disabled bar should emit nothing while still yielding every item."""
    items = list(
        progress_module.progress_iterator(
            [1, 2, 3],
            description="scan",
            enabled=False,
        )
    )

    display = recorded_displays[0]
    assert items == [1, 2, 3]
    assert progress_console.export_text() == ""
    assert display.live.is_started is False
    assert display.tasks[0].completed == 3


def test_progress_iterator_close_after_break_stops_display(
    recorded_displays: list[Progress],
    progress_console: Console,
) -> None:
    """Closing a generator abandoned mid-iteration should stop the display."""
    iterator = progress_module.progress_iterator(
        [1, 2, 3, 4],
        description="scan",
        enabled=True,
    )
    for value in iterator:
        if value == 2:
            break

    display = recorded_displays[0]
    # A retained reference keeps the generator alive, so ``break`` alone leaves
    # the display running until the caller closes it.
    assert display.live.is_started is True

    iterator.close()

    assert display.live.is_started is False
    assert display.tasks[0].completed == 1


def test_progress_iterator_break_without_a_retained_reference_stops_display(
    recorded_displays: list[Progress],
    progress_console: Console,
) -> None:
    """Abandoning the generator inline should stop the display at finalization."""
    # Nothing binds the generator, so CPython finalizes it as the loop exits and
    # the context manager tears the display down without an explicit close.
    for value in progress_module.progress_iterator(
        [1, 2, 3, 4],
        description="scan",
        enabled=True,
    ):
        if value == 2:
            break

    display = recorded_displays[0]
    assert display.live.is_started is False
    assert display.tasks[0].completed == 1
