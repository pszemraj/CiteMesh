"""Tests for the Rich-backed progress helpers."""

from __future__ import annotations

import io
from collections.abc import Iterator
from functools import partial
from typing import Any

import pytest
from rich.console import Console
from rich.errors import LiveError
from rich.live import Live
from rich.progress import Progress
from rich.text import Text

from citemesh import progress as progress_module


@pytest.fixture
def progress_console(monkeypatch: pytest.MonkeyPatch) -> Console:
    """Route progress rendering through a recording in-memory console.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to restore the global.
    :return Console: Console recording rendered output, including live redraws.
    """
    monkeypatch.setenv("TERM", "xterm-256color")
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
    monkeypatch.setattr(progress_module, "_progress_enabled", True)
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


@pytest.fixture
def retry_displays(monkeypatch: pytest.MonkeyPatch) -> list[Live]:
    """Record standalone retry displays without replacing their rendering."""
    displays: list[Live] = []

    def record_live(
        renderable: Text, *, console: Console, transient: bool, auto_refresh: bool
    ) -> Live:
        """Construct the requested live line and retain it for assertions."""
        live = Live(
            renderable,
            console=console,
            transient=transient,
            auto_refresh=auto_refresh,
        )
        displays.append(live)
        return live

    monkeypatch.setattr(progress_module, "Live", record_live)
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


@pytest.mark.parametrize("width", [40, 60, 80])
def test_dataset_progress_keeps_total_and_eta_visible(
    monkeypatch: pytest.MonkeyPatch,
    progress_console: Console,
    recorded_displays: list[Progress],
    width: int,
) -> None:
    """Full dataset counts and ETA survive narrow terminals."""
    progress_console.width = width
    clock = [0.0]
    monkeypatch.setattr(
        progress_module,
        "Progress",
        partial(
            progress_module.Progress, get_time=lambda: clock[0], auto_refresh=False
        ),
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

    # Live redraws are terminal replacements, not separate exported text lines.
    progress_console.export_text(clear=True)
    progress_console.print(recorded_displays[0].get_renderable())
    lines = progress_console.export_text().splitlines()
    line = next(line for line in lines if "2000/3156800" in line)
    assert "ETA 0:52:35" in line
    assert len(line) <= width
    if width >= 60:
        assert "Calibrating dataset" in line
    if width >= 80:
        assert "2000/3156800 papers" in line


def test_progress_timestamp_changes_only_when_work_advances(
    monkeypatch: pytest.MonkeyPatch,
    progress_console: Console,
    recorded_displays: list[Progress],
) -> None:
    """Redraws and zero-count updates must not look like productive work."""
    clock = [0.0]
    wall_clock = ["12:00:00"]
    monkeypatch.setattr(progress_module, "strftime", lambda _format: wall_clock[0])
    monkeypatch.setattr(
        progress_module,
        "Progress",
        partial(
            progress_module.Progress, get_time=lambda: clock[0], auto_refresh=False
        ),
    )
    with progress_module.progress_task(
        total=2, description="Encoding papers", unit="papers", enabled=True
    ) as task:
        display = recorded_displays[0]
        assert display.tasks[0].fields["updated"] == "12:00:00"
        clock[0], wall_clock[0] = 10.0, "12:00:10"
        task.update(1)
        assert display.tasks[0].fields["updated"] == "12:00:10"
        clock[0], wall_clock[0] = 20.0, "12:00:20"
        task.update(0)
        display.refresh()
        assert display.tasks[0].fields["updated"] == "12:00:10"
        task.update(-1)
        assert display.tasks[0].fields["updated"] == "12:00:10"
        clock[0], wall_clock[0] = 30.0, "12:00:30"
        task.update(2)

    progress_console.export_text(clear=True)
    progress_console.print(display.get_renderable())
    assert "12:00:30" in progress_console.export_text()


def test_progress_labels_preserve_brackets_as_literal_text(
    progress_console: Console,
    recorded_displays: list[Progress],
) -> None:
    """Labels containing Rich-style tags must not silently lose their brackets."""
    with progress_module.progress_task(
        total=2,
        description="Encoding [bold]data[/bold]",
        unit="[red]rows[/red]",
        enabled=True,
    ) as task:
        task.update(1)

    progress_console.export_text(clear=True)
    progress_console.print(recorded_displays[0].get_renderable())
    output = progress_console.export_text()
    assert "Encoding [bold]data[/bold]" in output
    assert "[red]rows[/red]" in output


@pytest.mark.parametrize("width", [40, 60, 80])
def test_progress_iterator_without_a_total_renders_indeterminate_counts(
    progress_console: Console,
    recorded_displays: list[Progress],
    width: int,
) -> None:
    """Unsized sources show elapsed time and a count, without an empty ETA label."""
    progress_console.width = width
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
    output = progress_console.export_text()
    assert "2/?" in output
    assert "ETA" not in output
    assert "0:00:" in output


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


@pytest.mark.parametrize(
    ("interactive", "allowed", "enabled", "expected"),
    [
        (False, True, True, True),
        (True, True, False, False),
        (True, True, None, True),
        (False, True, None, False),
        (True, False, True, False),
    ],
)
def test_progress_display_policy_preserves_items_and_counts(
    monkeypatch: pytest.MonkeyPatch,
    progress_console: Console,
    recorded_displays: list[Progress],
    interactive: bool,
    allowed: bool,
    enabled: bool | None,
    expected: bool,
) -> None:
    """Verbosity gates every display while automatic display also requires a TTY."""
    monkeypatch.setattr(progress_module, "stderr_isatty", lambda: interactive)
    progress_module.set_progress_enabled(allowed)

    assert progress_module.progress_enabled() is allowed
    assert progress_module.progress_enabled(require_tty=True) is (
        allowed and interactive
    )

    items = list(
        progress_module.progress_iterator(
            [1, 2, 3], description="scan", enabled=enabled
        )
    )

    display = recorded_displays[0]
    assert items == [1, 2, 3]
    assert bool(progress_console.export_text()) is expected
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


def test_retry_line_starts_on_first_retry_and_clears_on_close(
    monkeypatch: pytest.MonkeyPatch,
    progress_console: Console,
    retry_displays: list[Live],
    recorded_displays: list[Progress],
) -> None:
    """Retries replace one transient text line without constructing a bar."""
    monkeypatch.setattr(progress_module, "stderr_isatty", lambda: True)
    monkeypatch.setattr(progress_module, "strftime", lambda _format: "12:04:31")
    reporter = progress_module.RetryProgress("focal references")
    try:
        assert reporter.update(0) is False
        assert retry_displays == []
        assert reporter.update(1) is True
        assert "1 retry" in progress_console.export_text()
        assert reporter.update(12, delay=3.5) is True
        assert len(retry_displays) == 1
        live = retry_displays[0]
        assert live.is_started
        assert live.transient
        rendered = "".join(
            segment.text for segment in progress_console.render(live.renderable)
        )
        assert rendered.rstrip("\n") == (
            "12:04:31 Semantic Scholar · focal references · 12 retries · retry in 3.5s"
        )
        reporter.update(12)
        assert "retry in" not in "".join(
            segment.text for segment in progress_console.render(live.renderable)
        )
        assert recorded_displays == []
    finally:
        reporter.close()
    assert not live.is_started
    assert progress_module._active_retry_progress.get() is None


@pytest.mark.parametrize(("interactive", "allowed"), [(False, True), (True, False)])
def test_retry_display_requires_terminal_and_progress_policy(
    monkeypatch: pytest.MonkeyPatch,
    progress_console: Console,
    retry_displays: list[Live],
    recorded_displays: list[Progress],
    interactive: bool,
    allowed: bool,
) -> None:
    """Redirected output and disabled progress never create retry displays."""
    monkeypatch.setattr(progress_module, "stderr_isatty", lambda: interactive)
    progress_module.set_progress_enabled(allowed)
    reporter = progress_module.RetryProgress("focal references")
    with progress_module.progress_task(
        total=3, description="Hydrating reference lists"
    ):
        assert reporter.update(12, delay=2) is False
    reporter.close()
    assert retry_displays == []
    assert recorded_displays[0].disable
    assert progress_console.export_text() == ""


@pytest.mark.parametrize("width", [40, 60, 80])
def test_retry_line_keeps_count_and_wait_visible_on_narrow_terminals(
    monkeypatch: pytest.MonkeyPatch,
    progress_console: Console,
    retry_displays: list[Live],
    width: int,
) -> None:
    """Compact the section label before clipping retry counts or backoff waits."""
    monkeypatch.setattr(progress_module, "stderr_isatty", lambda: True)
    progress_console.width = width
    reporter = progress_module.RetryProgress("reference recovery metadata")
    try:
        assert reporter.update(12, delay=4) is True
        rendered = "".join(
            segment.text
            for segment in progress_console.render(retry_displays[0].renderable)
        )
        assert "12 retries" in rendered
        assert rendered.rstrip().endswith("4s")
        assert len(rendered.rstrip()) <= width
    finally:
        reporter.close()


def test_retry_line_unavailable_does_not_interrupt_work(
    monkeypatch: pytest.MonkeyPatch,
    progress_console: Console,
) -> None:
    """A console occupied by another thread's Live can keep its existing display."""
    monkeypatch.setattr(progress_module, "stderr_isatty", lambda: True)

    def reject_start(self: Live, *, refresh: bool = False) -> None:
        """Reproduce Rich 13.7 rejecting a second active live display."""
        raise LiveError("Only one live display may be active at once")

    monkeypatch.setattr(Live, "start", reject_start)
    reporter = progress_module.RetryProgress("focal metadata")
    assert reporter.update(1, delay=4) is False
    reporter.close()
    assert progress_module._active_retry_progress.get() is None
    assert progress_console.export_text() == ""


@pytest.mark.parametrize("width", [40, 60, 80, 120])
def test_retry_count_uses_active_bar_and_clears_at_section_end(
    monkeypatch: pytest.MonkeyPatch,
    progress_console: Console,
    retry_displays: list[Live],
    recorded_displays: list[Progress],
    width: int,
) -> None:
    """A section with a bar retains its count and never starts another Live."""
    monkeypatch.setattr(progress_module, "stderr_isatty", lambda: True)
    progress_console.width = width
    reporter = progress_module.RetryProgress("reference hydration")
    with progress_module.progress_task(
        total=2000, description="Hydrating reference lists"
    ) as task:
        try:
            task.update(1)
            assert reporter.update(12, delay=2) is True
            display = recorded_displays[0]
            rendered = "".join(
                segment.text
                for segment in progress_console.render(display.get_renderable())
            )
            assert "12 retries" in rendered
            assert "2s" in rendered
            assert "0/2000" in rendered
            assert "ETA" in rendered
            assert task.n == 1
            assert retry_displays == []
        finally:
            reporter.close()
        assert display.tasks[0].description == "Hydrating reference lists"
        assert display.tasks[0].fields["retry_status"] == ""
    assert progress_module._active_task.get() is None


def test_retry_line_yields_to_a_later_bar(
    monkeypatch: pytest.MonkeyPatch,
    progress_console: Console,
    retry_displays: list[Live],
    recorded_displays: list[Progress],
) -> None:
    """Starting a bar after a retry line stops the line before opening the bar."""
    monkeypatch.setattr(progress_module, "stderr_isatty", lambda: True)
    reporter = progress_module.RetryProgress("selected metadata")
    try:
        reporter.update(1)
        with progress_module.progress_task(
            total=1000, description="Fetching paper metadata"
        ):
            assert not retry_displays[0].is_started
            assert recorded_displays[0].tasks[0].fields["retry_status"] == "1 retry"
            reporter.update(2)
            assert len(retry_displays) == 1
        reporter.update(3)
        assert len(retry_displays) == 2
        assert retry_displays[1].is_started
    finally:
        reporter.close()
    assert all(not live.is_started for live in retry_displays)
