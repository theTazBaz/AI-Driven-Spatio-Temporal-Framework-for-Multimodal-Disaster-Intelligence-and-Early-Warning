"""Resumable, throttled elevation fetching (Open-Meteo's free tier is rate limited).

Fetches ``chunk`` points at a time, sleeps ``pause_s`` between chunks, and saves every fetched
elevation to disk after EACH chunk. If the API answers 429 (per-minute / hourly / daily limit)
the error propagates, but nothing fetched so far is lost: run the same command again later and
only the missing points are requested.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

from disaster_intel.clock import Clock, SystemClock
from disaster_intel.state.static_layers import ElevationSource

DEFAULT_ELEVATION_CACHE = Path(".cache") / "elevation_points.json"


def point_key(lat: float, lon: float) -> str:
    return f"{lat:.5f},{lon:.5f}"


class CachedElevation:
    def __init__(
        self,
        inner: ElevationSource,
        path: Path | str = DEFAULT_ELEVATION_CACHE,
        chunk: int = 100,
        pause_s: float = 12.0,
        clock: Clock | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        if chunk < 1:
            raise ValueError("chunk must be >= 1")
        self._inner = inner
        self._path = Path(path)
        self._chunk = chunk
        self._pause_s = pause_s
        self._clock: Clock = clock or SystemClock()
        self._progress = progress
        self.points_fetched = 0

    def _load(self) -> dict[str, float | None]:
        if not self._path.is_file():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}  # a corrupt cache is just a cache miss
        return {str(k): (None if v is None else float(v)) for k, v in data.items()}

    def _save(self, cache: dict[str, float | None]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        tmp.replace(self._path)

    def elevations(self, points: Sequence[tuple[float, float]]) -> list[float | None]:
        cache = self._load()
        keys = [point_key(lat, lon) for lat, lon in points]
        todo = {k: p for k, p in zip(keys, points, strict=True) if k not in cache}
        pending = list(todo.items())
        for start in range(0, len(pending), self._chunk):
            if start:
                self._clock.sleep(self._pause_s)
            batch = pending[start : start + self._chunk]
            got = self._inner.elevations([p for _, p in batch])  # may raise; cache is on disk
            for (key, _), value in zip(batch, got, strict=True):
                cache[key] = value if value is None or math.isfinite(value) else None
            self.points_fetched += len(batch)
            self._save(cache)
            if self._progress is not None:
                self._progress(min(start + self._chunk, len(pending)), len(pending))
        return [cache[k] for k in keys]
