"""Increment 8 demo: build up one Western Ghats cell's rolling windows, hour by hour.

    docker compose up -d --wait
    uv run python scripts/windows_demo.py

Loads a synthetic 8-day monsoon spell (with a data gap and a late arrival) into a throw-away
ClickHouse database, prints the windows at several times, then drops the database.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime, timedelta

import h3

from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.storage import ColdStore
from disaster_intel.windows import HOUR, format_windows

CATALOG = VariableCatalog.default()
START = datetime(2024, 7, 14, tzinfo=UTC)
CELL = h3.latlng_to_cell(12.42, 75.74, 7)  # Madikeri, Kodagu


def obs(variable: str, ts: datetime, value: float) -> Observation:
    return Observation.from_raw(
        cell=CELL,
        source="open-meteo",
        variable=variable,
        raw_value=round(value, 2),
        raw_unit=CATALOG[variable].canonical_unit,
        timestamp=ts,
    )


def monsoon_spell() -> list[Observation]:
    rows: list[Observation] = []
    for hour in range(1, 8 * 24 + 1):
        ts = START + hour * HOUR
        if 100 <= hour < 112:
            continue  # 12-hour sensor outage -> coverage < 100%
        storm = max(0.0, math.sin((hour - 96) / 60 * math.pi)) if 96 <= hour < 156 else 0.0
        rows.append(obs("precipitation", ts, 0.4 + 14.0 * storm))
        rows.append(obs("soil_moisture_surface", ts, 0.25 + 0.012 * hour / 8))
        rows.append(obs("discharge", ts, 40.0 + 260.0 * storm))
    return rows


def show(store: ColdStore, label: str, as_of: datetime) -> None:
    print(f"\n=== {label}: as_of = {as_of:%Y-%m-%d %H:%M} UTC  (value (coverage)) ===")
    table = store.windows(CELL, as_of)
    live = {k: v for k, v in table.items() if any(w.value is not None for w in v.values())}
    print(format_windows(live))


def main() -> None:
    database = f"demo_{uuid.uuid4().hex[:8]}"
    store = ColdStore.connect(database=database)
    store.migrate()
    try:
        store.write(monsoon_spell())
        show(store, "before the storm", START + 90 * HOUR)
        show(store, "during the storm (gap in the 72h window)", START + 120 * HOUR)
        peak = START + 8 * 24 * HOUR
        show(store, "end of series", peak)

        late = obs("precipitation", START + 150 * HOUR - timedelta(minutes=20), 60.0)
        store.write([late])  # a delayed gauge report fixes the right bucket
        show(store, "end of series after a late 60 mm report", peak)
        show(store, "rollback: the state as it was at hour 90 (unchanged)", START + 90 * HOUR)
    finally:
        store.client.command(f"DROP DATABASE IF EXISTS {database}")


if __name__ == "__main__":
    main()
