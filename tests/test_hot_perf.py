"""Increment 9 budget + latency test (plan.md sections 9 and 11): real Redis.

Run with ``uv run pytest -m perf -s tests/test_hot_perf.py``. It loads a full 7 days x 6
hourly variables for a sample of cells, then reports:

* memory per cell (``MEMORY USAGE``) and the projection for all cells,
* hot WRITE latency (one cell's hourly update, 6 readings) and READ latency
  (``windows`` for one cell, all variables, all five windows) as average, p50, p95, p99.

Limits follow plan.md: "set after first measurement, only tightened". Memory per cell is the
plan's 8 KB; read/write p95 is the plan's proposed 5 ms on a dev machine.
"""

from __future__ import annotations

import math
import os
import random
import socket
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
import redis

from disaster_intel.clock import FakeClock
from disaster_intel.grid import CoastalGrid
from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.storage import HotStore, summarize_latencies

pytestmark = pytest.mark.perf

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
CATALOG = VariableCatalog.default()
END = datetime(2024, 7, 10, 12, 0, tzinfo=UTC)
HOURLY_VARIABLES = (
    "precipitation",
    "soil_moisture_surface",
    "soil_moisture_rootzone",
    "temperature",
    "wind_speed",
    "pressure",
)
SAMPLE_CELLS = 300
LATENCY_SAMPLES = 200

# ── Budget (tighten only) - FIRST RUN: replace with measured value + headroom ──
MAX_BYTES_PER_CELL = 8 * 1024  # plan.md section 11
MAX_READ_P95_MS = 5.0  # plan.md section 15
MAX_WRITE_P95_MS = 5.0


def full_week(cell: str, index: int, rng: random.Random) -> list[Observation]:
    out: list[Observation] = []
    for variable in HOURLY_VARIABLES:
        unit = CATALOG[variable].canonical_unit
        base = 10.0 + (index % 37) * 0.25
        for hour in range(168):
            value = base + 3.0 * math.sin(hour / 24 * 2 * math.pi) + rng.gauss(0, 0.8)
            out.append(
                Observation.from_raw(
                    cell=cell,
                    source="open-meteo",
                    variable=variable,
                    raw_value=round(value, 2),  # real feeds are reported to 2 decimals
                    raw_unit=unit,
                    timestamp=END - timedelta(hours=hour),
                )
            )
    return out


@pytest.fixture
def hot() -> Iterator[HotStore]:
    try:
        socket.create_connection((REDIS_HOST, REDIS_PORT), timeout=2.0).close()
    except OSError:
        pytest.fail(f"Redis {REDIS_HOST}:{REDIS_PORT} unreachable. Run 'docker compose up -d'.")
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    store = HotStore(client, clock=FakeClock(END), prefix=f"perf:{uuid.uuid4().hex[:10]}:")
    try:
        yield store
    finally:
        store.clear()


def test_memory_per_cell_and_hot_read_write_latency(hot: HotStore) -> None:
    cells = random.Random(1).sample(CoastalGrid(use_cache=True).cells, SAMPLE_CELLS)
    rng = random.Random(2024)
    for index, cell in enumerate(cells):
        hot.write(full_week(cell, index, rng))

    sizes = [hot.memory_usage(c) or 0 for c in cells]
    mean_bytes = sum(sizes) / len(sizes)
    worst = max(sizes)

    write_ms: list[float] = []
    read_ms: list[float] = []
    for index, cell in enumerate(cells[:LATENCY_SAMPLES]):
        # a live hourly update: the next hour's six readings for one cell
        hot_clock = hot._clock
        assert isinstance(hot_clock, FakeClock)
        nxt = END + timedelta(hours=1 + index % 3)
        batch = [
            Observation.from_raw(
                cell=cell,
                source="open-meteo",
                variable=v,
                raw_value=round(rng.uniform(5, 30), 2),
                raw_unit=CATALOG[v].canonical_unit,
                timestamp=nxt,
            )
            for v in HOURLY_VARIABLES
        ]
        started = time.perf_counter()
        hot.write(batch)
        write_ms.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        hot.windows(cell)
        read_ms.append((time.perf_counter() - started) * 1000)

    write_report = summarize_latencies(write_ms)
    read_report = summarize_latencies(read_ms)
    print(
        f"\n[hot memory] mean={mean_bytes:,.0f} B/cell worst={worst:,} B/cell "
        f"-> {mean_bytes * 5930 / 1e6:.1f} MB for 5,930 cells "
        f"(budget {MAX_BYTES_PER_CELL:,} B/cell)"
        f"\n[hot] {write_report.format('write (1 cell, 6 readings)')}"
        f"\n[hot] {read_report.format('read  (1 cell, 9 vars x 5 windows)')}"
    )

    assert worst <= MAX_BYTES_PER_CELL
    assert read_report.p95_ms <= MAX_READ_P95_MS
    assert write_report.p95_ms <= MAX_WRITE_P95_MS
