"""Clock abstractions for time injection and deterministic testing.

Enforces TDD Rule 3 from plan.md:
"Unit tests never touch the network or the clock. Time comes from an injected Clock."
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Protocol for abstracting time queries and sleeps."""

    def now(self) -> datetime:
        """Return the current time as a timezone-aware UTC datetime."""
        ...

    def sleep(self, seconds: float) -> None:
        """Sleep or advance time by `seconds`."""
        ...


class SystemClock:
    """Real system wall clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FakeClock:
    """Controllable in-memory clock for deterministic, fast tests.

    Records all sleep calls and advances instantaneous time without wall-clock delay.
    """

    def __init__(self, initial_time: datetime | None = None) -> None:
        if initial_time is None:
            self._current_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        else:
            if initial_time.tzinfo is None:
                raise ValueError("Initial time must be timezone-aware")
            self._current_time = initial_time.astimezone(UTC)
        self.sleep_log: list[float] = []

    def now(self) -> datetime:
        return self._current_time

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(f"Cannot sleep negative duration: {seconds}")
        self.sleep_log.append(seconds)
        from datetime import timedelta

        self._current_time += timedelta(seconds=seconds)

    def advance(self, seconds: float) -> None:
        """Explicitly advance clock without logging as a sleep."""
        from datetime import timedelta

        self._current_time += timedelta(seconds=seconds)

    def advance_to(self, target_time: datetime) -> None:
        """Advance clock forward to an explicit timestamp."""
        if target_time.tzinfo is None:
            raise ValueError("Target time must be timezone-aware")
        target_utc = target_time.astimezone(UTC)
        if target_utc < self._current_time:
            msg = f"Cannot advance clock backward: {self._current_time} to {target_utc}"
            raise ValueError(msg)
        delta = (target_utc - self._current_time).total_seconds()
        self.advance(delta)
