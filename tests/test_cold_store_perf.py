"""Increment 7 storage-budget test (plan.md §11, 'Budget Enforcement Tests').

Loads one synthetic day for EVERY cell of the study region and checks, from
``system.parts``, that the cold store stays inside its byte budget.

plan.md: "set after first measurement, only tightened". The limits below are the
first measurement plus headroom; lower them whenever a run shows more room.
"""

from __future__ import annotations

import math
import os
import random
import socket
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from disaster_intel.grid import CoastalGrid
from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.storage import ColdStore

pytestmark = pytest.mark.perf

CH_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
CATALOG = VariableCatalog.default()
DAY = datetime(2024, 7, 1, tzinfo=UTC)

# The hourly gridded variables the Open-Meteo adapter delivers for every cell.
HOURLY_VARIABLES = (
    "precipitation",
    "soil_moisture_surface",
    "soil_moisture_rootzone",
    "temperature",
    "wind_speed",
    "pressure",
)

# ── Budget (tighten only) ────────────────────────────────────────────
MAX_COMPRESSED_BYTES_PER_ROW = 5.0  # first measurement: 3.31 B/row
MAX_COMPRESSED_BYTES_PER_CELL_DAY = 800.0  # first measurement: 477 B


def synthetic_day(cells: tuple[str, ...]) -> Iterator[list[Observation]]:
    """One day, every cell, every hourly variable.

    A diurnal curve plus seeded per-reading noise rounded to 2 decimals (like the API's
    precision). Noise matters: a noiseless sine compresses to ~0.2 B/row and would make
    the budget meaningless.
    """
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


def test_one_synthetic_day_for_all_cells_fits_the_byte_budget(store: ColdStore) -> None:
    cells = CoastalGrid(use_cache=True).cells
    pending: list[Observation] = []
    for per_cell in synthetic_day(cells):
        pending.extend(per_cell)
        if len(pending) >= 10_000:
            store.write(pending)
            pending = []
    store.write(pending)
    store.optimize()

    stats = store.storage_stats()
    expected_rows = len(cells) * len(HOURLY_VARIABLES) * 24
    bytes_per_cell_day = stats.compressed_bytes / len(cells)
    print(
        f"\n[budget] rows={stats.rows:,} compressed={stats.compressed_bytes:,} B "
        f"uncompressed={stats.uncompressed_bytes:,} B "
        f"-> {stats.bytes_per_row:.2f} B/row, {bytes_per_cell_day:.0f} B/cell-day"
    )

    assert stats.rows == expected_rows
    assert store.count() == expected_rows
    assert stats.bytes_per_row <= MAX_COMPRESSED_BYTES_PER_ROW
    assert bytes_per_cell_day <= MAX_COMPRESSED_BYTES_PER_CELL_DAY
