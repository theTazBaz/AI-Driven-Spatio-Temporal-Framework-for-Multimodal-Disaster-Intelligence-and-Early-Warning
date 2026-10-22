"""Increment 8 integration tests - real ClickHouse (docker compose up -d --wait).

Test list (plan.md, Increment 8):
 1. ClickHouse windows equal the slow Python oracle on random series, for every variable
    and every window, at aligned, unaligned, early and far-future as-of times.
 2. The bucket table equals plain-Python bucketization of what arrived.
 3. A late observation fixes the right buckets (and only those).
 4. Duplicates and revised values never double-count (why buckets are recomputed, not summed).
 5. Gaps report coverage; empty windows are None.
 6. Windows at past time T computed today equal those computed live at T (rollback).
Plus: hour-boundary semantics, source filter, backfill mode (maintain_buckets=False).
"""

from __future__ import annotations

import os
import random
import socket
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from disaster_intel.grid import CoastalGrid
from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.storage import ColdStore, to_float32
from disaster_intel.windows import (
    HOUR,
    WINDOW_HOURS,
    WindowValue,
    bucket_end,
    bucketize,
    reference_windows,
    variable_names,
)

pytestmark = pytest.mark.integration

CH_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
CATALOG = VariableCatalog.default()
ALL_VARIABLES = variable_names(CATALOG)
T0 = datetime(2024, 7, 1, tzinfo=UTC)
DAYS = 10
END = T0 + timedelta(days=DAYS)

Windows = dict[str, dict[str, WindowValue]]


@pytest.fixture
def store() -> Iterator[ColdStore]:
    try:
        socket.create_connection((CH_HOST, CH_PORT), timeout=2.0).close()
    except OSError:
        pytest.fail(f"ClickHouse {CH_HOST}:{CH_PORT} unreachable. Run 'docker compose up -d'.")
    database = f"test_{uuid.uuid4().hex[:12]}"
    cold = ColdStore.connect(host=CH_HOST, port=CH_PORT, database=database)
    cold.migrate()
    try:
        yield cold
    finally:
        cold.client.command(f"DROP DATABASE IF EXISTS {database}")


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
    """What the cold store hands back: the same record with a Float32-quantised value."""
    return Observation(
        cell=o.cell,
        source=o.source,
        variable=o.variable,
        value=to_float32(o.value),
        unit=o.unit,
        timestamp=o.timestamp,
        idempotency_key=o.idempotency_key,
    )


def random_series(cell: str, seed: int) -> list[Observation]:
    """Every catalog variable: ~70% of hours present, 30% of readings off the hour mark."""
    rng = random.Random(seed)
    out: list[Observation] = []
    for variable in ALL_VARIABLES:
        for hour in range(24 * DAYS + 1):
            if rng.random() < 0.3:
                continue  # a gap
            ts = T0 + timedelta(hours=hour)
            if rng.random() < 0.3:
                ts -= timedelta(minutes=rng.randrange(1, 60), microseconds=rng.randrange(0, 5))
            value = float(rng.randint(1, 4)) if variable == "cap_severity" else rng.uniform(0, 100)
            out.append(make_obs(cell, variable, ts, value))
    return out


def assert_windows_equal(got: Windows, want: Windows) -> None:
    assert got.keys() == want.keys()
    for variable in want:
        assert got[variable].keys() == want[variable].keys()
        for window, expected in want[variable].items():
            actual = got[variable][window]
            where = f"{variable}/{window}"
            assert (actual.observed_hours, actual.hours) == (
                expected.observed_hours,
                expected.hours,
            ), where
            assert actual.coverage == expected.coverage, where
            if expected.value is None:
                assert actual.value is None, where
            else:
                assert actual.value == pytest.approx(expected.value, rel=1e-9, abs=1e-9), where


def write_in_batches(store: ColdStore, rows: list[Observation], size: int = 1000) -> None:
    for i in range(0, len(rows), size):
        store.write(rows[i : i + size])


# ── 1. Oracle agreement ──────────────────────────────────────────────

AS_OFS = {
    "end_aligned": END,
    "end_plus_37min": END + timedelta(minutes=37),
    "mid_unaligned": END - timedelta(days=3, minutes=12),
    "one_hour_in": T0 + HOUR,
    "before_any_data": T0 - timedelta(days=1),
    "far_future": END + timedelta(days=30),
}


@pytest.mark.parametrize("as_of", AS_OFS.values(), ids=AS_OFS.keys())
def test_clickhouse_windows_equal_the_python_oracle_on_random_series(
    store: ColdStore, cells: tuple[str, ...], as_of: datetime
) -> None:
    chosen = (cells[0], cells[len(cells) // 2], cells[-1])
    arrivals = [o for i, c in enumerate(chosen) for o in random_series(c, seed=100 + i)]
    random.Random(5).shuffle(arrivals)  # arrival order must not matter
    write_in_batches(store, arrivals)
    stored = [as_stored(o) for o in arrivals]

    for cell in chosen:
        assert_windows_equal(
            store.windows(cell, as_of), reference_windows(stored, cell, as_of, CATALOG)
        )


def test_every_variable_and_window_is_exercised(store: ColdStore, cells: tuple[str, ...]) -> None:
    write_in_batches(store, random_series(cells[0], seed=1))
    table = store.windows(cells[0], END)
    assert list(table) == ALL_VARIABLES
    for variable in ALL_VARIABLES:
        assert set(table[variable]) == set(WINDOW_HOURS)
        assert table[variable]["7d"].value is not None, variable
        assert 0.0 < table[variable]["7d"].coverage <= 1.0, variable


def test_bucket_table_equals_python_bucketization(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    """Pins the SQL bucket expression against bucket_end() on real, off-the-hour data."""
    arrivals = random_series(cells[0], seed=9)
    write_in_batches(store, arrivals)
    expected = bucketize([as_stored(o) for o in arrivals])
    got = store.read_buckets(cells[0], T0 - 2 * HOUR, END + 2 * HOUR)
    assert got.keys() == expected.keys()
    for variable, by_bucket in expected.items():
        assert got[variable].keys() == by_bucket.keys(), variable
        for label, want in by_bucket.items():
            have = got[variable][label]
            assert (have.n, have.minimum, have.maximum) == (want.n, want.minimum, want.maximum)
            assert have.total == pytest.approx(want.total, rel=1e-9, abs=1e-9)


# ── 2. Hour-boundary semantics ───────────────────────────────────────


def test_reading_exactly_on_the_hour_belongs_to_that_hours_window(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    cell = cells[0]
    ten = T0 + timedelta(hours=10)
    store.write(
        [
            make_obs(cell, "precipitation", ten, 3.0),
            make_obs(cell, "precipitation", ten + timedelta(microseconds=1), 100.0),
        ]
    )
    assert store.windows(cell, ten)["precipitation"]["1h"].value == 3.0
    assert store.windows(cell, ten + timedelta(minutes=59))["precipitation"]["1h"].value == 3.0
    assert store.windows(cell, ten + HOUR)["precipitation"]["1h"].value == 100.0
    assert store.windows(cell, ten)["precipitation"]["7d"].value == 3.0


# ── 3. Late observations ─────────────────────────────────────────────


def test_a_late_observation_fixes_the_right_buckets_and_only_those(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    cell = cells[0]
    base = [make_obs(cell, "precipitation", T0 + h * HOUR, 1.0) for h in range(1, 24 * 5 + 1)]
    store.write(base)
    end = T0 + 24 * 5 * HOUR
    before_table = store.windows(cell, end)
    before_buckets = store.read_buckets(cell, T0, end)

    late_ts = end - timedelta(hours=30, minutes=10)  # inside 72h and 7d, outside 24h
    late = make_obs(cell, "precipitation", late_ts, 50.0)
    store.write([late])  # arrives long after the hours around it were written

    after_table = store.windows(cell, end)
    rain_before, rain_after = before_table["precipitation"], after_table["precipitation"]
    for window in ("1h", "6h", "24h"):
        assert rain_after[window] == rain_before[window], window  # untouched
    for window in ("72h", "7d"):
        old, new = rain_before[window].value, rain_after[window].value
        assert old is not None
        assert new == pytest.approx(old + 50.0), window
    assert rain_after["72h"].observed_hours == rain_before["72h"].observed_hours  # same hour slot

    changed = {
        label
        for label, bucket in store.read_buckets(cell, T0, end)["precipitation"].items()
        if bucket != before_buckets["precipitation"].get(label)
    }
    assert changed == {bucket_end(late_ts)}  # exactly one bucket was recomputed

    oracle = reference_windows([as_stored(o) for o in [*base, late]], cell, end, CATALOG)
    assert_windows_equal(after_table, oracle)


def test_a_late_observation_in_an_empty_hour_raises_coverage(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    cell = cells[0]
    end = T0 + 24 * HOUR
    store.write([make_obs(cell, "precipitation", T0 + h * HOUR, 1.0) for h in range(13, 25)])
    assert store.windows(cell, end)["precipitation"]["24h"].coverage == 12 / 24
    store.write([make_obs(cell, "precipitation", T0 + 5 * HOUR, 2.0)])  # fills a gap, late
    assert store.windows(cell, end)["precipitation"]["24h"].coverage == 13 / 24


# ── 4. Duplicates and revisions never double-count ───────────────────


def test_duplicate_inserts_are_not_double_counted(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    cell = cells[0]
    obs = make_obs(cell, "precipitation", T0 + HOUR, 4.0)
    store.client.command(f"SYSTEM STOP MERGES {store.database}.observations")
    store.write([obs])
    store.write([obs])  # at-least-once delivery after a crash
    store.write([obs, obs])
    assert store.raw_count() == 3  # one row per insert (a block dedups itself, parts do not)...
    window = store.windows(cell, T0 + HOUR)["precipitation"]["1h"]
    assert window.value == 4.0  # ...but the window counts the observation once


def test_a_revised_value_replaces_the_old_one(store: ColdStore, cells: tuple[str, ...]) -> None:
    cell = cells[0]
    store.client.command(f"SYSTEM STOP MERGES {store.database}.observations")
    store.write([make_obs(cell, "precipitation", T0 + HOUR, 1.0)])
    store.write([make_obs(cell, "precipitation", T0 + HOUR, 2.5)])
    assert store.windows(cell, T0 + HOUR)["precipitation"]["1h"].value == 2.5


# ── 5. Gaps, sources, unknown cells ──────────────────────────────────


def test_gaps_report_coverage_and_empty_windows_are_none(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    cell = cells[0]
    end = T0 + 24 * HOUR
    store.write([make_obs(cell, "precipitation", T0 + h * HOUR, 2.0) for h in range(1, 13)])
    rain = store.windows(cell, end)["precipitation"]
    assert rain["1h"] == WindowValue(value=None, observed_hours=0, hours=1)  # nothing in last hour
    assert rain["6h"].value is None
    assert rain["24h"] == WindowValue(value=24.0, observed_hours=12, hours=24)
    assert rain["24h"].coverage == 0.5
    assert store.windows(cell, end)["temperature"]["24h"].value is None  # never 0


def test_unknown_cell_has_empty_windows_with_zero_coverage(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    table = store.windows(cells[0], END)
    assert all(
        wv.value is None and wv.coverage == 0.0 for pw in table.values() for wv in pw.values()
    )


def test_source_filter_selects_one_product(store: ColdStore, cells: tuple[str, ...]) -> None:
    cell = cells[0]
    store.write(
        [
            make_obs(cell, "precipitation", T0 + HOUR, 1.0, source="open-meteo"),
            make_obs(cell, "precipitation", T0 + HOUR, 7.0, source="imerg"),
        ]
    )
    at = T0 + HOUR
    assert store.windows(cell, at, source="imerg")["precipitation"]["1h"].value == 7.0
    assert store.windows(cell, at, source="open-meteo")["precipitation"]["1h"].value == 1.0
    assert store.windows(cell, at)["precipitation"]["1h"].value == 8.0  # merged: see windows.py


# ── 6. Deterministic rollback ────────────────────────────────────────


def test_windows_at_past_time_T_computed_today_equal_those_computed_live_at_T(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    cell = cells[0]
    rng = random.Random(21)
    live: dict[datetime, Windows] = {}
    for hour in range(0, 24 * 5):
        now = T0 + hour * HOUR
        batch = [
            make_obs(cell, "precipitation", now, rng.uniform(0, 20)),
            make_obs(cell, "soil_moisture_surface", now, rng.uniform(0.1, 0.5)),
            make_obs(cell, "discharge", now - timedelta(minutes=25), rng.uniform(10, 900)),
        ]
        store.write(batch)  # observations stream in, in order
        if hour % 6 == 0:
            live[now] = store.windows(cell, now)  # what a live reader saw at T

    # the stream keeps going: more hours, another cell, then merges happen
    for hour in range(24 * 5, 24 * 6):
        store.write([make_obs(cell, "precipitation", T0 + hour * HOUR, rng.uniform(0, 20))])
    store.write([make_obs(cells[1], "precipitation", T0 + HOUR, 9.0)])

    for moment, seen in live.items():
        assert store.windows(cell, moment) == seen, moment
    store.optimize()
    for moment, seen in live.items():
        assert store.windows(cell, moment) == seen, moment


# ── Backfill mode ────────────────────────────────────────────────────


def test_backfill_without_maintenance_then_rebuild_equals_maintained_buckets(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    arrivals = random_series(cells[0], seed=77)
    bulk = ColdStore(store.client, database=store.database, maintain_buckets=False)
    write_in_batches(bulk, arrivals)
    assert bulk.read_buckets(cells[0], T0 - 2 * HOUR, END + 2 * HOUR) == {}  # nothing yet

    bulk.rebuild_buckets()
    oracle = reference_windows([as_stored(o) for o in arrivals], cells[0], END, CATALOG)
    assert_windows_equal(bulk.windows(cells[0], END), oracle)


def test_rebuild_is_idempotent(store: ColdStore, cells: tuple[str, ...]) -> None:
    write_in_batches(store, random_series(cells[0], seed=3))
    first = store.windows(cells[0], END)
    store.rebuild_buckets()
    store.rebuild_buckets()
    assert store.windows(cells[0], END) == first


def test_migration_2_is_recorded(store: ColdStore) -> None:
    rows = store.client.query(
        f"SELECT version FROM {store.database}.schema_migrations ORDER BY version"
    ).result_rows
    assert [int(r[0]) for r in rows] == [1, 2]
    assert store.migrate() == []
