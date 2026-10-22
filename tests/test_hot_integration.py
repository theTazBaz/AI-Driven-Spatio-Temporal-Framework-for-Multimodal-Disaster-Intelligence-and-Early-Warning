"""Increment 9 integration tests - real Redis (and real ClickHouse for the rebuild).

    docker compose up -d --wait          # needs the redis service recreated, see README
    uv run pytest -m integration tests/test_hot_integration.py

Test list (plan.md, Increment 9):
 1. Hot windows equal the oracle after replaying a stream (real Redis, bytes round trip).
 2. Expired buckets are evicted (physically gone, and the key has a TTL).
 3. Applying one observation twice changes nothing (byte-identical hash).
 4. Wiping Redis and rebuilding from the cold tier gives IDENTICAL state, and the rebuilt
    hot windows agree with ClickHouse's windows (the dual-tier contract of Increment 10).
 5. Redis is configured NOT to silently evict state (policy must be noeviction).
Each test gets its own key prefix, removed afterwards.
"""

from __future__ import annotations

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
from disaster_intel.storage import ColdStore, HotStore, rebuild_hot_from_cold, to_float32
from disaster_intel.windows import HOUR, WindowValue, reference_windows, variable_names

pytestmark = pytest.mark.integration

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
CH_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
CATALOG = VariableCatalog.default()
ALL_VARIABLES = variable_names(CATALOG)
END = datetime(2024, 7, 10, 12, 0, tzinfo=UTC)

Windows = dict[str, dict[str, WindowValue]]


@pytest.fixture
def raw_redis() -> redis.Redis:
    try:
        socket.create_connection((REDIS_HOST, REDIS_PORT), timeout=2.0).close()
    except OSError:
        pytest.fail(f"Redis {REDIS_HOST}:{REDIS_PORT} unreachable. Run 'docker compose up -d'.")
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(END)


@pytest.fixture
def hot(raw_redis: redis.Redis, clock: FakeClock) -> Iterator[HotStore]:
    store = HotStore(raw_redis, clock=clock, prefix=f"test:{uuid.uuid4().hex[:10]}:")
    try:
        yield store
    finally:
        store.clear()


@pytest.fixture(scope="module")
def cells() -> tuple[str, ...]:
    return CoastalGrid(use_cache=True).cells


def make_obs(
    cell: str, variable: str, ts: datetime, value: float, source: str = "open-meteo"
) -> Observation:
    return Observation.from_raw(
        cell=cell,
        source=source,
        variable=variable,
        raw_value=value,
        raw_unit=CATALOG[variable].canonical_unit,
        timestamp=ts,
    )


def as_stored(o: Observation) -> Observation:
    return Observation(
        cell=o.cell,
        source=o.source,
        variable=o.variable,
        value=to_float32(o.value),
        unit=o.unit,
        timestamp=o.timestamp,
        idempotency_key=o.idempotency_key,
    )


def random_series(cell: str, seed: int, days: int = 9) -> list[Observation]:
    rng = random.Random(seed)
    out: list[Observation] = []
    for variable in ALL_VARIABLES:
        for hour in range(24 * days + 1):
            if rng.random() < 0.3:
                continue
            ts = END - timedelta(hours=hour)
            if rng.random() < 0.3:
                ts -= timedelta(minutes=rng.randrange(1, 60), microseconds=rng.randrange(0, 5))
            value = float(rng.randint(1, 4)) if variable == "cap_severity" else rng.uniform(0, 100)
            out.append(make_obs(cell, variable, ts, value))
    return out


def assert_windows_equal(got: Windows, want: Windows) -> None:
    assert got.keys() == want.keys()
    for variable in want:
        for window, expected in want[variable].items():
            actual = got[variable][window]
            where = f"{variable}/{window}"
            assert (actual.observed_hours, actual.hours) == (
                expected.observed_hours,
                expected.hours,
            ), where
            if expected.value is None:
                assert actual.value is None, where
            else:
                assert actual.value == pytest.approx(expected.value, rel=1e-9, abs=1e-9), where


# ── 1. Oracle agreement on real Redis ────────────────────────────────


def test_hot_windows_equal_the_oracle_after_replaying_a_stream(
    hot: HotStore, cells: tuple[str, ...]
) -> None:
    chosen = (cells[0], cells[len(cells) // 2], cells[-1])
    arrivals = [o for i, c in enumerate(chosen) for o in random_series(c, seed=40 + i)]
    random.Random(1).shuffle(arrivals)
    for i in range(0, len(arrivals), 500):
        hot.write(arrivals[i : i + 500])
    stored = [as_stored(o) for o in arrivals]
    for cell in chosen:
        assert_windows_equal(hot.windows(cell), reference_windows(stored, cell, END, CATALOG))


# ── 2. Eviction ──────────────────────────────────────────────────────


def test_expired_records_are_physically_removed_and_the_key_expires(
    hot: HotStore,
    clock: FakeClock,
    raw_redis: redis.Redis,
    cells: tuple[str, ...],
) -> None:
    cell = cells[0]
    hot.write([make_obs(cell, "precipitation", END - h * HOUR, 1.0) for h in range(0, 100)])
    assert len(hot.dump_cell(cell)[("precipitation", "open-meteo")]) == 100
    ttl = raw_redis.ttl(hot.key(cell))
    assert 0 < int(ttl) <= (168 + 24) * 3600

    clock.sleep(6 * 24 * 3600)
    assert hot.evict_expired([cell]) > 0
    kept = hot.dump_cell(cell)[("precipitation", "open-meteo")]
    assert 0 < len(kept) < 100
    clock.sleep(20 * 24 * 3600)
    hot.evict_expired([cell])
    assert raw_redis.exists(hot.key(cell)) == 0  # emptied -> Redis deletes the hash


# ── 3. Idempotency ───────────────────────────────────────────────────


def test_applying_an_observation_twice_leaves_the_stored_bytes_identical(
    hot: HotStore,
    raw_redis: redis.Redis,
    cells: tuple[str, ...],
) -> None:
    cell = cells[0]
    obs = [make_obs(cell, v, END - timedelta(minutes=20), 3.5) for v in ALL_VARIABLES[:4]]
    assert hot.write(obs) == (4, 0)
    before = raw_redis.hgetall(hot.key(cell))
    assert hot.write(obs) == (0, 4)
    assert raw_redis.hgetall(hot.key(cell)) == before


# ── 4. Wipe and rebuild from the real cold tier ──────────────────────


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


def test_wipe_and_rebuild_from_cold_gives_identical_state_and_matches_cold_windows(
    hot: HotStore, cold: ColdStore, cells: tuple[str, ...]
) -> None:
    chosen = (cells[0], cells[len(cells) // 2])
    arrivals = [o for i, c in enumerate(chosen) for o in random_series(c, seed=70 + i, days=12)]
    random.Random(3).shuffle(arrivals)
    for i in range(0, len(arrivals), 1000):
        batch = arrivals[i : i + 1000]
        cold.write(batch)  # the record, 12 days deep
        hot.write(batch)  # the cache, last 7 days only
    live = {c: hot.dump_cell(c) for c in chosen}
    assert all(live.values())

    report = rebuild_hot_from_cold(hot, cold, chosen, wipe=True)

    assert report.cleared_keys == len(chosen)
    assert {c: hot.dump_cell(c) for c in chosen} == live  # identical state
    for cell in chosen:  # and the two tiers answer the same question the same way
        assert_windows_equal(hot.windows(cell), cold.windows(cell, END))


# ── 5. Redis must not silently evict state ───────────────────────────


def test_redis_is_configured_not_to_evict_keys_silently(raw_redis: redis.Redis) -> None:
    policy = raw_redis.config_get("maxmemory-policy")["maxmemory-policy"]
    assert policy == "noeviction", (
        f"maxmemory-policy is {policy!r}: Redis would silently drop cells' state under memory "
        "pressure and every window read would be wrong. Set it to noeviction in "
        "docker-compose.yml and recreate the container (docker compose up -d redis)."
    )
