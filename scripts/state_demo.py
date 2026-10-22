"""Increment 10 demo: one Western Ghats cell's state VECTOR building up hour by hour.

    docker compose up -d --wait
    uv run python scripts/state_demo.py

Uses synthetic terrain for the static layers (offline) and a synthetic 8-day monsoon spell.
Streams it through a TeeSink (ClickHouse + Redis), prints the full state vector at three
moments, shows hot vs cold agreeing, then shows the past (T) recomputed from ClickHouse
after more data arrived. Everything runs in a throw-away database and key prefix.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime

import h3

from disaster_intel.clock import FakeClock
from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.state import (
    FeatureSpec,
    FunctionElevation,
    StateAssembler,
    StateVector,
    build_static_layers,
    synthetic_ghats_elevation,
)
from disaster_intel.storage import ColdStore, HotStore, TeeSink
from disaster_intel.windows import HOUR

CATALOG = VariableCatalog.default()
SPEC = FeatureSpec.v1()
START = datetime(2024, 7, 14, tzinfo=UTC)
CELL = h3.latlng_to_cell(12.42, 75.74, 7)  # Madikeri, Kodagu
HOURS = 8 * 24


def obs(variable: str, ts: datetime, value: float) -> Observation:
    return Observation.from_raw(
        cell=CELL,
        source="open-meteo",
        variable=variable,
        raw_value=round(value, 3),
        raw_unit=CATALOG[variable].canonical_unit,
        timestamp=ts,
    )


def readings_for_hour(hour: int) -> list[Observation]:
    ts = START + hour * HOUR
    storm = max(0.0, math.sin((hour - 96) / 60 * math.pi)) if 96 <= hour < 156 else 0.0
    out = [
        obs("precipitation", ts, 0.4 + 14.0 * storm),
        obs("soil_moisture_rootzone", ts, 0.28 + 0.1 * min(1.0, hour / 150)),
    ]
    if hour < 100 or hour >= 112:  # the discharge gauge is offline for 12 hours
        out.append(obs("discharge", ts, 40.0 + 260.0 * storm))
    return out


def show(vector: StateVector, label: str) -> None:
    print(f"\n=== {label}: as_of={vector.as_of:%Y-%m-%d %H:%M} UTC  tier={vector.tier} ===")
    for name, value, cov in zip(SPEC.names, vector.values, vector.coverage, strict=True):
        if "seismic" in name and math.isnan(value):
            continue  # keep the printout short: no earthquakes in this spell
        text = "NaN (missing)" if math.isnan(value) else f"{value:12.4f}"
        extra = "" if cov is None else f"   coverage {cov:4.0%}"
        print(f"  {name:<34}{text:>14}{extra}")


def _static_layers():  # type: ignore[no-untyped-def]
    """Real layers from .cache if built and containing CELL; otherwise synthetic terrain."""
    from disaster_intel.state.static_layers import DEFAULT_STATIC_PATH, StaticLayers

    if DEFAULT_STATIC_PATH.is_file():
        real = StaticLayers.load(DEFAULT_STATIC_PATH)
        if CELL in real and "SYNTHETIC" not in real.meta.get("elevation", ""):
            print(f"[static layers: REAL, {real.meta.get('elevation')}]")
            return real
    print("[static layers: SYNTHETIC terrain]")
    return build_static_layers([CELL], FunctionElevation(synthetic_ghats_elevation))


def main() -> None:
    clock = FakeClock(START)
    hot = HotStore.connect(clock=clock, prefix=f"demo:{uuid.uuid4().hex[:8]}:")
    database = f"demo_{uuid.uuid4().hex[:8]}"
    cold = ColdStore.connect(database=database)
    cold.migrate()
    layers = _static_layers()
    assembler = StateAssembler(layers, hot=hot, cold=cold, clock=clock)
    sink = TeeSink(cold, hot)
    try:
        saved_t: StateVector | None = None
        for hour in range(1, HOURS + 1):
            clock.sleep(3600)
            sink.write(readings_for_hour(hour))
            if hour in (90, 125, HOURS):
                state = assembler.get_state(CELL)
                show(state, {90: "before the storm", 125: "mid-storm"}.get(hour, "end of spell"))
                if hour == 125:
                    saved_t = state
        print(f"\nhot vs cold agree at the end: {assembler.compare_tiers(CELL).agrees}")
        assert saved_t is not None
        replay = assembler.get_state(CELL, as_of=saved_t.as_of)
        same = replay.same_values(saved_t, rel_tol=1e-9, abs_tol=1e-9)
        print(f"state at {saved_t.as_of:%m-%d %H:%M} recomputed 67 hours later ({replay.tier}) "
              f"equals the live one: {same}")
        assert same
        print("\nJSON of the vector (NaN -> null):\n" + saved_t.to_json()[:240] + " ...")
    finally:
        hot.clear()
        cold.client.command(f"DROP DATABASE IF EXISTS {database}")


if __name__ == "__main__":
    main()
