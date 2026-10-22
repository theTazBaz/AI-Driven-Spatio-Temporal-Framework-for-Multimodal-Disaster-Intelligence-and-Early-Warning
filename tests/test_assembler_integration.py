"""Increment 10 integration tests - real Redis + real ClickHouse.

    docker compose up -d --wait
    uv run pytest -m integration tests/test_assembler_integration.py

Test list (plan.md, Increment 10):
 1. ``get_state`` agrees between the hot tier (Redis) and the cold tier (ClickHouse buckets),
    value by value and coverage by coverage.
 2. Deterministic rollback: the state at past time T, computed days later from ClickHouse,
    equals the state computed live at T from Redis.
 3. After a Redis wipe, ``compare_tiers`` notices; a rebuild from cold repairs it.
 4. Static layers built for real study-region cells flow into the vector.
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
import redis

from disaster_intel.clock import FakeClock
from disaster_intel.grid import CoastalGrid
from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.state import (
    FeatureSpec,
    FunctionElevation,
    StateAssembler,
    StaticLayers,
    build_static_layers,
    synthetic_ghats_elevation,
)
from disaster_intel.storage import ColdStore, HotStore, TeeSink, rebuild_hot_from_cold
from disaster_intel.windows import HOUR, variable_names

pytestmark = pytest.mark.integration

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
CH_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
CATALOG = VariableCatalog.default()
SPEC = FeatureSpec.v1()
END = datetime(2024, 7, 10, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def cells() -> tuple[str, ...]:
    return CoastalGrid(use_cache=True).cells


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(END)


@pytest.fixture
def hot(clock: FakeClock) -> Iterator[HotStore]:
    try:
        socket.create_connection((REDIS_HOST, REDIS_PORT), timeout=2.0).close()
    except OSError:
        pytest.fail(f"Redis {REDIS_HOST}:{REDIS_PORT} unreachable. Run 'docker compose up -d'.")
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    store = HotStore(client, clock=clock, prefix=f"test:{uuid.uuid4().hex[:10]}:")
    try:
        yield store
    finally:
        store.clear()


@pytest.fixture
def cold() -> Iterator[ColdStore]:
    try:
        socket.create_connection((CH_HOST, CH_PORT), timeout=2.0).close()
    except OSError:
        pytest.fail(f"ClickHouse {CH_HOST}:{CH_PORT} unreachable. Run 'docker compose up -d'.")
    database = f"test_{uuid.uuid4().hex[:12]}"
    store = ColdStore.connect(host=CH_HOST, port=CH_PORT, database=database)
    store.migrate()
    try:
        yield store
    finally:
        store.client.command(f"DROP DATABASE IF EXISTS {database}")


def make_obs(cell: str, variable: str, ts: datetime, value: float) -> Observation:
    return Observation.from_raw(
        cell=cell,
        source="open-meteo",
        variable=variable,
        raw_value=value,
        raw_unit=CATALOG[variable].canonical_unit,
        timestamp=ts,
    )


def random_series(cell: str, seed: int, end: datetime = END, days: int = 9) -> list[Observation]:
    rng = random.Random(seed)
    out: list[Observation] = []
    for variable in variable_names(CATALOG):
        for hour in range(24 * days + 1):
            if rng.random() < 0.3:
                continue
            ts = end - hour * HOUR
            if rng.random() < 0.3:
                ts -= timedelta(minutes=rng.randrange(1, 60))
            value = float(rng.randint(1, 4)) if variable == "cap_severity" else rng.uniform(0, 90)
            out.append(make_obs(cell, variable, ts, value))
    rng.shuffle(out)
    return out


def static_for(cells: list[str]) -> StaticLayers:
    return build_static_layers(cells, FunctionElevation(synthetic_ghats_elevation))


# ── 1. Hot == cold on the real stores ────────────────────────────────


def test_get_state_agrees_between_redis_and_clickhouse(
    hot: HotStore, cold: ColdStore, clock: FakeClock, cells: tuple[str, ...]
) -> None:
    chosen = [cells[0], cells[len(cells) // 2], cells[-1]]
    sink = TeeSink(cold, hot)
    for i, cell in enumerate(chosen):
        sink.write(random_series(cell, seed=100 + i))
    assembler = StateAssembler(static_for(chosen), hot=hot, cold=cold, clock=clock)
    for cell in chosen:
        h = assembler.get_state(cell, tier="hot")
        c = assembler.get_state(cell, tier="cold")
        assert h.same_values(c, rel_tol=1e-9, abs_tol=1e-9), cell
        assert assembler.compare_tiers(cell).agrees
        assert not all(math.isnan(v) for v in h.values)


# ── 2. Deterministic rollback ────────────────────────────────────────


def test_state_at_past_time_computed_later_equals_live_state_at_that_time(
    hot: HotStore, cold: ColdStore, clock: FakeClock, cells: tuple[str, ...]
) -> None:
    cell = cells[len(cells) // 3]
    sink = TeeSink(cold, hot)
    sink.write(random_series(cell, seed=7))
    assembler = StateAssembler(static_for([cell]), hot=hot, cold=cold, clock=clock)
    live = assembler.get_state(cell)
    assert live.tier == "hot"

    clock.sleep(30 * 3600)  # 30 hours later, new weather arrives
    sink.write(random_series(cell, seed=8, end=END + 30 * HOUR, days=1))
    replay = assembler.get_state(cell, as_of=END)
    assert replay.tier == "cold"
    assert replay.same_values(live, rel_tol=1e-9, abs_tol=1e-9)


# ── 3. A wiped Redis is detected and repaired ────────────────────────


def test_compare_tiers_detects_a_wiped_hot_tier_and_rebuild_repairs_it(
    hot: HotStore, cold: ColdStore, clock: FakeClock, cells: tuple[str, ...]
) -> None:
    cell = cells[5]
    TeeSink(cold, hot).write(random_series(cell, seed=3))
    assembler = StateAssembler(static_for([cell]), hot=hot, cold=cold, clock=clock)
    assert assembler.compare_tiers(cell).agrees
    hot.clear()
    assert not assembler.compare_tiers(cell).agrees
    rebuild_hot_from_cold(hot, cold, [cell])
    assert assembler.compare_tiers(cell).agrees


# ── 4. Static layers reach the vector ────────────────────────────────


def test_static_layers_for_real_cells_appear_in_the_vector(
    hot: HotStore, clock: FakeClock, cells: tuple[str, ...]
) -> None:
    chosen = list(cells[:8])
    layers = static_for(chosen)
    assembler = StateAssembler(layers, hot=hot, clock=clock)
    for cell in chosen:
        vector = assembler.get_state(cell)
        assert vector.get(SPEC, "elevation_m") == layers.get(cell).elevation_m
        assert vector.get(SPEC, "slope_deg") == layers.get(cell).slope_deg
        assert math.isnan(vector.get(SPEC, "population_density_km2"))  # no raster -> unknown
        assert math.isnan(vector.get(SPEC, "precipitation__24h"))  # nothing streamed -> NaN
