"""Increment 7 integration tests — real ClickHouse (docker compose up -d).

Test list (plan.md, Increment 7):
 1. Write-then-read equals the reference copy kept at arrival (cold-correctness metric).
 2. Duplicate inserts collapse after deduplication — and reads MUST dedup explicitly,
    because ReplacingMergeTree only dedups when parts merge (pinned with merges stopped).
 3. ClickHouse parent-cell results equal the Python H3 library on 1,000 random cells.
 4. Re-running migrations is idempotent.
Plus: range/variable filters, out-of-order (late) arrival, revised value keeps the latest.

Each test gets its own throw-away database, so tests never see each other's data.
"""

from __future__ import annotations

import os
import random
import socket
import uuid
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import h3
import pytest

from disaster_intel.grid import CoastalGrid
from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.pipeline import InMemorySink
from disaster_intel.storage import ColdStore, to_float32

pytestmark = pytest.mark.integration

CH_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
CATALOG = VariableCatalog.default()
T0 = datetime(2024, 7, 1, 0, 0, tzinfo=UTC)
VARIABLES = ["precipitation", "soil_moisture_surface", "temperature", "pressure", "discharge"]


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


def make_obs(cell: str, variable: str, ts: datetime, value: float, source: str = "open-meteo"):  # type: ignore[no-untyped-def]
    return Observation.from_raw(
        cell=cell,
        source=source,
        variable=variable,
        raw_value=value,
        raw_unit=CATALOG[variable].canonical_unit,
        timestamp=ts,
    )


def random_observations(cells: tuple[str, ...], n: int, seed: int = 7) -> list[Observation]:
    rng = random.Random(seed)
    chosen: dict[tuple[str, str, datetime], Observation] = {}
    while len(chosen) < n:
        cell = rng.choice(cells)
        variable = rng.choice(VARIABLES)
        ts = T0 + timedelta(hours=rng.randrange(0, 24 * 40))
        chosen[(cell, variable, ts)] = make_obs(cell, variable, ts, rng.uniform(0, 100))
    return list(chosen.values())


def as_tuple(o: Observation) -> tuple[str, str, str, str, datetime, float]:
    return (o.cell, o.variable, o.source, o.unit, o.timestamp, o.value)


# ── 1. Cold-correctness ──────────────────────────────────────────────


def test_write_then_read_equals_reference_copy(store: ColdStore, cells: tuple[str, ...]) -> None:
    reference = InMemorySink()  # the copy kept at arrival
    arrivals = random_observations(cells, 3000)
    for i in range(0, len(arrivals), 1000):  # three separate batches = three parts
        batch = arrivals[i : i + 1000]
        reference.write(batch)
        store.write(batch)

    stored = sorted(as_tuple(o) for o in store.read_all())
    expected = sorted(
        (o.cell, o.variable, o.source, o.unit, o.timestamp, to_float32(o.value))
        for o in reference.all_observations()
    )
    assert stored == expected  # 100% identical records (value is float32 by design)
    assert {o.idempotency_key for o in store.read_all()} == {
        o.idempotency_key for o in reference.all_observations()
    }


def test_cold_correctness_metric_is_100_percent(store: ColdStore, cells: tuple[str, ...]) -> None:
    arrivals = random_observations(cells, 500)
    store.write(arrivals)
    report = store.cold_correctness(arrivals)
    assert report.total == 500
    assert report.matched == 500
    assert report.rate == 100.0


def test_cold_correctness_metric_detects_corruption(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    arrivals = random_observations(cells, 50)
    store.write(arrivals[:-1])  # one record never arrived
    report = store.cold_correctness(arrivals)
    assert report.matched == 49
    assert report.missing == 1
    assert report.rate < 100.0


# ── 2. Deduplication ─────────────────────────────────────────────────


def test_duplicate_inserts_collapse_after_deduplication(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    obs = make_obs(cells[0], "precipitation", T0, 4.0)
    table = f"{store.database}.observations"
    store.client.command(f"SYSTEM STOP MERGES {table}")  # make the pre-merge state deterministic

    store.write([obs])
    store.write([obs])

    assert store.raw_count() == 2  # pins ReplacingMergeTree: duplicates exist until a merge
    assert store.count() == 1  # ...so every read must deduplicate explicitly
    assert len(store.read_cell(obs.cell)) == 1
    assert len(store.read_all()) == 1

    store.client.command(f"SYSTEM START MERGES {table}")
    store.optimize()
    assert store.raw_count() == 1  # and the merge really does collapse them physically


def test_duplicates_inside_one_batch_also_collapse(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    obs = make_obs(cells[0], "pressure", T0, 1000.0)
    store.write([obs, obs, obs])
    assert store.count() == 1


def test_revised_value_for_same_key_keeps_the_latest_insert(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    table = f"{store.database}.observations"
    store.client.command(f"SYSTEM STOP MERGES {table}")
    store.write([make_obs(cells[0], "precipitation", T0, 1.0)])
    store.write([make_obs(cells[0], "precipitation", T0, 2.0)])
    assert [o.value for o in store.read_cell(cells[0])] == [2.0]


def test_same_instant_from_different_sources_are_not_duplicates(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    store.write(
        [
            make_obs(cells[0], "precipitation", T0, 1.0, source="open-meteo"),
            make_obs(cells[0], "precipitation", T0, 1.2, source="imerg"),
        ]
    )
    assert store.count() == 2


# ── 3. H3 parent cells ───────────────────────────────────────────────


@pytest.mark.parametrize("parent_res", [4, 5, 6])
def test_clickhouse_parent_cells_equal_python_h3_on_1000_random_cells(
    store: ColdStore, cells: tuple[str, ...], parent_res: int
) -> None:
    sample = random.Random(11).sample(cells, 1000)
    from_clickhouse = store.parent_cells(sample, parent_res)
    from_python = {c: h3.cell_to_parent(c, parent_res) for c in sample}
    assert from_clickhouse == from_python


def test_stored_rows_roll_up_to_the_same_parents_as_python_h3(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    arrivals = random_observations(cells, 800)
    store.write(arrivals)
    expected = Counter(h3.cell_to_parent(o.cell, 5) for o in arrivals)
    assert store.count_by_parent(5) == dict(expected)


# ── 4. Migrations ────────────────────────────────────────────────────


def test_rerunning_migrations_is_idempotent(store: ColdStore, cells: tuple[str, ...]) -> None:
    def schema() -> str:
        rows = store.client.query(f"SHOW CREATE TABLE {store.database}.observations").result_rows
        return str(rows[0][0])

    store.write(random_observations(cells, 100))
    before_schema, before_count = schema(), store.count()

    assert store.migrate() == []  # nothing new to apply
    assert store.migrate() == []
    assert schema() == before_schema
    assert store.count() == before_count  # data untouched


def test_fresh_database_applies_every_migration_once(store: ColdStore) -> None:
    applied = store.client.query(
        f"SELECT version FROM {store.database}.schema_migrations ORDER BY version"
    ).result_rows
    assert [int(r[0]) for r in applied] == sorted({int(r[0]) for r in applied})
    assert len(applied) >= 1


# ── Reads ────────────────────────────────────────────────────────────


def test_late_and_out_of_order_arrivals_are_read_back_in_time_order(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    hours = list(range(24))
    random.Random(3).shuffle(hours)
    store.write([make_obs(cells[0], "precipitation", T0 + timedelta(hours=h), h) for h in hours])
    got = [o.timestamp for o in store.read_cell(cells[0])]
    assert got == [T0 + timedelta(hours=h) for h in range(24)]


def test_read_cell_filters_by_variable_and_half_open_time_range(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    rows = [
        make_obs(cells[0], v, T0 + timedelta(hours=h), float(h))
        for v in ("precipitation", "temperature")
        for h in range(24)
    ]
    rows.append(make_obs(cells[1], "precipitation", T0, 9.0))  # other cell: never returned
    store.write(rows)

    got = store.read_cell(
        cells[0],
        variable="precipitation",
        start=T0 + timedelta(hours=6),
        end=T0 + timedelta(hours=12),
    )
    assert [o.value for o in got] == [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]  # [start, end)
    assert {o.variable for o in got} == {"precipitation"}
    assert len(store.read_cell(cells[0])) == 48


def test_unknown_cell_reads_empty(store: ColdStore, cells: tuple[str, ...]) -> None:
    assert store.read_cell(cells[0]) == []
    assert store.count() == 0


def test_microsecond_and_pre_epoch_free_timestamps_round_trip(
    store: ColdStore, cells: tuple[str, ...]
) -> None:
    ts = datetime(2021, 7, 22, 13, 45, 12, 654321, tzinfo=UTC)
    obs = make_obs(cells[0], "discharge", ts, 812.5)
    store.write([obs])
    (got,) = store.read_cell(cells[0])
    assert got.timestamp == ts
    assert got.idempotency_key == obs.idempotency_key
