"""Increment 9 demo: one Western Ghats cell's state building up hour by hour in Redis.

    docker compose up -d --wait
    uv run python scripts/hot_demo.py              # Redis + ClickHouse (full demo)
    uv run python scripts/hot_demo.py --hot-only   # Redis only (skips the rebuild step)

A synthetic 8-day monsoon spell is streamed hour by hour (with a 12 h sensor outage) through
a TeeSink into ClickHouse (the record) and Redis (the hot tier). Prints the live windows at
three moments, a duplicate-replay check, hot read/write latency, and finally wipes Redis and
rebuilds it from ClickHouse to show the state is identical.
Everything runs in a throw-away database and key prefix that are removed at the end.
"""

from __future__ import annotations

import argparse
import math
import time
import uuid
from datetime import UTC, datetime, timedelta

import h3

from disaster_intel.clock import FakeClock
from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.storage import (
    ColdStore,
    HotStore,
    TeeSink,
    rebuild_hot_from_cold,
    summarize_latencies,
)
from disaster_intel.windows import HOUR, format_windows

CATALOG = VariableCatalog.default()
START = datetime(2024, 7, 14, tzinfo=UTC)
CELL = h3.latlng_to_cell(12.42, 75.74, 7)  # Madikeri, Kodagu
HOURS = 8 * 24


def obs(variable: str, ts: datetime, value: float) -> Observation:
    return Observation.from_raw(
        cell=CELL,
        source="open-meteo",
        variable=variable,
        raw_value=round(value, 2),
        raw_unit=CATALOG[variable].canonical_unit,
        timestamp=ts,
    )


def readings_for_hour(hour: int) -> list[Observation]:
    """What the feed delivers at hour ``hour`` of the spell (nothing during the outage)."""
    if 100 <= hour < 112:
        return []
    ts = START + hour * HOUR
    storm = max(0.0, math.sin((hour - 96) / 60 * math.pi)) if 96 <= hour < 156 else 0.0
    return [
        obs("precipitation", ts, 0.4 + 14.0 * storm),
        obs("soil_moisture_surface", ts, 0.25 + 0.012 * hour / 8),
        obs("discharge", ts, 40.0 + 260.0 * storm),
    ]


def show(hot: HotStore, clock: FakeClock, label: str) -> None:
    table = hot.windows(CELL)
    live = {k: v for k, v in table.items() if any(w.value is not None for w in v.values())}
    print(f"\n=== {label}: now = {clock.now():%Y-%m-%d %H:%M} UTC (value (coverage)) ===")
    print(format_windows(live))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hot-only", action="store_true", help="skip ClickHouse")
    args = parser.parse_args()

    clock = FakeClock(START)
    hot = HotStore.connect(clock=clock, prefix=f"demo:{uuid.uuid4().hex[:8]}:")
    cold: ColdStore | None = None
    database = f"demo_{uuid.uuid4().hex[:8]}"
    if not args.hot_only:
        cold = ColdStore.connect(database=database)
        cold.migrate()
    sink = TeeSink(cold, hot) if cold is not None else hot

    write_ms: list[float] = []
    read_ms: list[float] = []
    try:
        for hour in range(1, HOURS + 1):
            clock.sleep(3600)  # the clock reaches START + hour h
            batch = readings_for_hour(hour)
            if batch:
                started = time.perf_counter()
                sink.write(batch)
                write_ms.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            hot.windows(CELL)
            read_ms.append((time.perf_counter() - started) * 1000)
            if hour in (90, 120, HOURS):
                labels = {90: "before the storm", 120: "during the storm (gap in 72h)"}
                show(hot, clock, labels.get(hour, "end of the spell"))

        replay = readings_for_hour(HOURS)
        print(f"\nReplaying the last hour's readings: (stored, duplicates) = {hot.write(replay)}")

        late = obs("precipitation", START + 150 * HOUR - timedelta(minutes=20), 60.0)
        stored_late = sink.write([late])
        print(f"A 60 mm report from hour 150 arrives late: (stored, duplicates) = {stored_late}")
        show(hot, clock, "end of the spell after the late 60 mm report")

        print("\n" + summarize_latencies(write_ms).format("write per hour (sink)"))
        print(summarize_latencies(read_ms).format("hot read (1 cell, all windows)"))
        mem = hot.memory_usage(CELL)
        print(f"Redis memory for this cell: {mem:,} bytes (budget 8,192)")

        if cold is not None:
            before = hot.dump_cell(CELL)
            report = rebuild_hot_from_cold(hot, cold, [CELL], wipe=True)
            print("\n" + report.summary())
            same = hot.dump_cell(CELL) == before
            print(f"State after wipe + rebuild identical to live state: {same}")
            assert same
    finally:
        hot.clear()
        if cold is not None:
            cold.client.command(f"DROP DATABASE IF EXISTS {database}")


if __name__ == "__main__":
    main()
