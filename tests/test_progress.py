"""Tests for the Rich-backed progress helpers."""

from __future__ import annotations

import io
from functools import partial
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


def test_dataset_progress_keeps_total_and_eta_visible_at_80_columns(
    monkeypatch: pytest.MonkeyPatch,
    progress_console: Console,
) -> None:
    """Dataset counts and a measured ETA should fit a standard terminal width.

    :param pytest.MonkeyPatch monkeypatch: Supplies a deterministic progress clock.
    :param Console progress_console: Recording console with an 80-column width.
    :return None: Checks the rendered count and ETA share an untruncated line.
    """
    clock = [0.0]
    monkeypatch.setattr(
        progress_module,
        "Progress",
        partial(Progress, get_time=lambda: clock[0], auto_refresh=False),
    )
    with progress_module.progress_task(
        total=3156800,
        description="Calibrating dataset",
        unit="papers",
        enabled=True,
    ) as task:
        clock[0] = 1.0
        task.update(1000)
        clock[0] = 2.0
        task.update(1000)

    lines = progress_console.export_text().splitlines()
    assert any(
        "Calibrating dataset" in line
        and "2000/3156800 papers" in line
        and "ETA 0:52:35" in line
        and len(line) <= 80
        for line in lines
    )


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


def test_progress_eta_retains_slow_work_after_a_fast_update_burst(
    monkeypatch: pytest.MonkeyPatch,
    progress_console: Console,
) -> None:
    """Rapid row updates must not evict the time spent on a slow batch.

    :param pytest.MonkeyPatch monkeypatch: Supplies a deterministic progress clock.
    :param Console progress_console: Console used for the progress display.
    :return None: Checks the estimate retains slow work and final counts are exact.
    """
    clock = [0.0]
    display = Progress(
        console=progress_console, get_time=lambda: clock[0], auto_refresh=False
    )
    monkeypatch.setattr(progress_module, "Progress", lambda *args, **kwargs: display)

    with progress_module.progress_task(
        total=40000, description="Hydrating dataset", enabled=True
    ) as task:
        clock[0] = 0.5
        task.update(1000)
        clock[0] = 10.5
        task.update(1000)
        for _ in range(5000):
            clock[0] += 0.00001
            task.update(1)
        assert task.n == 7000
        clock[0] = 11.0
        task.update(0)
        assert 50 < display.tasks[0].time_remaining < 70
        task.update(3)

    assert display.tasks[0].completed == 7003


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
