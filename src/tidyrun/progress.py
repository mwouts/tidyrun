"""Progress reporting for plan materialization and execution."""

from __future__ import annotations

from collections.abc import Callable

ProgressCallback = Callable[[str], None]

_BAR_WIDTH = 24


class ProgressReporter:
    """Emit progress messages for a fixed-size batch of steps.

    With no callback, messages are rendered as an in-place progress bar on
    stdout.  With a callback, each message is passed to it as a plain string.
    A disabled reporter ignores every call, so callers never need to guard.
    """

    def __init__(
        self,
        enabled: bool,
        callback: ProgressCallback | None,
        phase: str,
        total: int,
    ) -> None:
        self.enabled = enabled
        self.callback = callback
        self.phase = phase
        self.total = total
        self.done = 0
        self.inline = callback is None
        self._last_render_length = 0

    def _emit(self, message: str) -> None:
        if self.inline:
            padding = max(0, self._last_render_length - len(message))
            print(f"\r{message}{' ' * padding}", end="", flush=True)
            self._last_render_length = len(message)
            return

        assert self.callback is not None
        self.callback(message)

    def _finish_inline(self) -> None:
        if self.inline:
            print()
            self._last_render_length = 0

    def _bar(self) -> str:
        if self.total <= 0:
            return "#" * _BAR_WIDTH
        filled = min(_BAR_WIDTH, int((self.done / self.total) * _BAR_WIDTH))
        return ("#" * filled) + ("-" * (_BAR_WIDTH - filled))

    def info(self, message: str) -> None:
        if not self.enabled:
            return
        if self.inline:
            if message.startswith("starting"):
                self._emit(
                    f"[{self.phase}] [{self._bar()}] {self.done}/{self.total} starting"
                )
                return
            if message == "done":
                self._emit(
                    f"[{self.phase}] [{self._bar()}] {self.done}/{self.total} done"
                )
                self._finish_inline()
                return

        self._emit(f"[{self.phase}] {message}")

    def step(self, job_id: str, *, skipped: bool = False) -> None:
        if not self.enabled:
            return
        self.done += 1
        status = "skipped" if skipped else "completed"
        if self.inline:
            self._emit(
                f"[{self.phase}] [{self._bar()}] {self.done}/{self.total} {status}: {job_id}"
            )
            return

        self._emit(f"[{self.phase}] [{self.done}/{self.total}] {status}: {job_id}")
