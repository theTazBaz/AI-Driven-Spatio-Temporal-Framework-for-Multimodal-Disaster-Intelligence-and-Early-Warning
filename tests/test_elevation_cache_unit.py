"""Unit tests - resumable elevation cache (no network)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from disaster_intel.adapters import SourceError
from disaster_intel.clock import FakeClock
from disaster_intel.state.elevation_cache import CachedElevation

pytestmark = pytest.mark.unit


class Scripted:
    """Elevation = 10 x latitude; optionally fails on its Nth call (like a 429)."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[int] = []
        self.fail_on_call = fail_on_call

    def elevations(self, points: Sequence[tuple[float, float]]) -> list[float | None]:
        self.calls.append(len(points))
        if self.fail_on_call == len(self.calls):
            raise SourceError("HTTP 429")
        return [10.0 * lat for lat, _ in points]


POINTS = [(12.0 + i * 0.01, 75.0) for i in range(250)]


def test_fetches_in_chunks_pauses_between_them_and_returns_in_order(tmp_path: Path) -> None:
    inner, clock = Scripted(), FakeClock()
    cache = CachedElevation(inner, tmp_path / "e.json", chunk=100, pause_s=12.0, clock=clock)
    result = cache.elevations(POINTS)
    assert inner.calls == [100, 100, 50]
    assert clock.sleep_log == [12.0, 12.0]
    assert result == [10.0 * lat for lat, _ in POINTS]


def test_a_failure_keeps_earlier_chunks_and_the_rerun_only_fetches_the_rest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "e.json"
    failing = Scripted(fail_on_call=2)
    with pytest.raises(SourceError):
        CachedElevation(failing, path, chunk=100, clock=FakeClock()).elevations(POINTS)
    assert failing.calls == [100, 100]  # second call failed; first chunk was saved

    healthy = Scripted()
    result = CachedElevation(healthy, path, chunk=100, clock=FakeClock()).elevations(POINTS)
    assert healthy.calls == [100, 50]  # 100 points came from disk
    assert result == [10.0 * lat for lat, _ in POINTS]


def test_a_complete_cache_makes_no_requests_and_duplicates_are_fetched_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "e.json"
    CachedElevation(Scripted(), path, clock=FakeClock()).elevations(POINTS)
    again = Scripted()
    assert len(CachedElevation(again, path, clock=FakeClock()).elevations(POINTS)) == 250
    assert again.calls == []
    dup = Scripted()
    CachedElevation(dup, tmp_path / "d.json", clock=FakeClock()).elevations(POINTS[:5] * 3)
    assert dup.calls == [5]


def test_a_corrupt_cache_file_is_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "e.json"
    path.write_text("{not json", encoding="utf-8")
    inner = Scripted()
    CachedElevation(inner, path, clock=FakeClock()).elevations(POINTS[:3])
    assert inner.calls == [3]
