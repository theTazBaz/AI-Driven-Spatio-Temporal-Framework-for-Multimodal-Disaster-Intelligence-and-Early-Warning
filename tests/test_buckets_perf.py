"""Increment 8 budget + latency test (plan.md section 11): hourly_buckets for one synthetic day.

Run with ``uv run pytest -m perf -s``. The limits below are deliberately generous because
this is the FIRST run; plan.md says "set after first measurement, only tightened", so copy
the printed numbers into the constants (plus headroom) after your first run.
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

from disaster_intel.grid import CoastalGrid
from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.pipeline import calculate_percentiles
from disaster_intel.storage import ColdStore

pytestmark = pytest.mark.perf

CH_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
CATALOG = VariableCatalog.default()
DAY = datetime(2024, 7, 1, tzinfo=UTC)
HOURLY_VARIABLES = (
    "precipitation",
    "soil_moisture_surface",
    "soil_moisture_rootzone",
    "temperature",
    "wind_speed",
    "pressure",
)

# ── Budget (tighten only) - FIRST RUN: replace with measured value + headroom ──
MAX_BUCKET_BYTES_PER_ROW = 24.0
MAX_WINDOWS_P95_MS = 250.0


def synthetic_day(cells: tuple[str, ...]) -> Iterator[list[Observation]]:
    rng = random.Random(2024)
    for index, cell in enumerate(cells):
        batch: list[Observation] = []
        for variable in HOURLY_VARIABLES:
            unit = CATALOG[variable].canonical_unit
            base = 10.0 + (index % 37) * 0.25
            for hour in range(24):
                value = base + 3.0 * math.sin(hour / 24 * 2 * math.pi) + rng.gauss(0, 0.8)
                batch.append(
                    Observation.from_raw(
                        cell=cell,
                        source="open-meteo",
                        variable=variable,
                        raw_value=round(value, 2),
                        raw_unit=unit,
                        timestamp=DAY + timedelta(hours=hour),
                    )
                )
        yield batch


@pytest.fixture
def store() -> Iterator[ColdStore]:
    try:
        socket.create_connection((CH_HOST, CH_PORT), timeout=2.0).close()
    except OSError:
        pytest.fail(f"ClickHouse {CH_HOST}:{CH_PORT} unreachable. Run 'docker compose up -d'.")
    database = f"perf_{uuid.uuid4().hex[:12]}"
    cold = ColdStore.connect(host=CH_HOST, port=CH_PORT, database=database)
    cold.migrate()
    try:
        yield cold
    finally:
        cold.client.command(f"DROP DATABASE IF EXISTS {database}")


def test_bucket_table_budget_and_window_read_latency(store: ColdStore) -> None:
    cells = CoastalGrid(use_cache=True).cells
    pending: list[Observation] = []
    for per_cell in synthetic_day(cells):
        pending.extend(per_cell)
        if len(pending) >= 10_000:
            store.write(pending)
            pending = []
    store.write(pending)
    store.optimize()
    store.client.command(f"OPTIMIZE TABLE {store.database}.hourly_buckets FINAL")

    rows, compressed = store.client.query(
        "SELECT sum(rows), sum(data_compressed_bytes) FROM system.parts "
        f"WHERE database = '{store.database}' AND table = 'hourly_buckets' AND active"
    ).result_rows[0]
    expected_rows = len(cells) * len(HOURLY_VARIABLES) * 24  # one source -> one bucket per hour
    per_row = compressed / rows

    sample = random.Random(1).sample(cells, 100)
    as_of = DAY + timedelta(hours=24)
    latencies_ms: list[float] = []
    for cell in sample:
        started = time.perf_counter()
        store.windows(cell, as_of)
        latencies_ms.append((time.perf_counter() - started) * 1000)
    p50, p95, p99 = calculate_percentiles(latencies_ms, [50, 95, 99])

    print(
        f"\n[buckets] rows={int(rows):,} compressed={int(compressed):,} B -> {per_row:.2f} B/row, "
        f"{compressed / len(cells):.0f} B/cell-day"
        f"\n[windows] read latency p50={p50:.1f} ms p95={p95:.1f} ms p99={p99:.1f} ms"
    )

    assert int(rows) == expected_rows
    assert per_row <= MAX_BUCKET_BYTES_PER_ROW
    assert p95 <= MAX_WINDOWS_P95_MS
