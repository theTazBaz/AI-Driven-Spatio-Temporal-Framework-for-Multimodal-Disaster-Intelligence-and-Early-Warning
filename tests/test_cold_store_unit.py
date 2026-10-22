"""Increment 7 unit tests — ClickHouse cold store (no network, no containers).

Test list (plan.md, Increment 7), unit-testable half:
 1. Row mapping is lossless apart from the documented float32 quantisation.
 2. The idempotency key is recomputed on read and equals the key made at arrival.
 3. Bad records are rejected *before* anything is inserted.
 4. Writes are batched.
 5. Every migration statement is idempotent (IF NOT EXISTS) and migrate() twice applies nothing.
 6. Schema uses compact types and codecs (budget design).
 7. Every read deduplicates explicitly (FINAL) — ReplacingMergeTree only dedups on merge.
 8. Regression: every adapter variable's unit equals the catalog's canonical unit.

The database-backed half (cold-correctness, dedup, H3 parent cells, real idempotent
migrations) lives in tests/test_cold_store_integration.py.
"""

from __future__ import annotations

import math
import os
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from disaster_intel.adapters.open_meteo import OPEN_METEO_VARIABLE_MAP
from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.storage import (
    COLUMNS,
    ColdStore,
    get_migrations,
    observation_to_row,
    resolve_credentials,
    row_to_observation,
    to_float32,
)

pytestmark = pytest.mark.unit

CELL = "872a1072bffffff"
T0 = datetime(2024, 7, 1, 6, 0, tzinfo=UTC)
CATALOG = VariableCatalog.default()


def make_obs(
    variable: str = "precipitation",
    value: float = 1.5,
    cell: str = CELL,
    source: str = "open-meteo",
    ts: datetime = T0,
) -> Observation:
    return Observation.from_raw(
        cell=cell,
        source=source,
        variable=variable,
        raw_value=value,
        raw_unit=CATALOG[variable].canonical_unit,
        timestamp=ts,
    )


class FakeClient:
    """Records calls; emulates just enough of the schema_migrations table."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.queries: list[str] = []
        self.inserts: list[tuple[str, list[list[Any]]]] = []
        self.applied: list[int] = []
        self.count_result = 0

    def command(self, cmd: str, parameters: Any = None) -> Any:
        self.commands.append(cmd)

    def query(self, query: str, parameters: Any = None) -> Any:
        self.queries.append(query)
        if "schema_migrations" in query:
            return SimpleNamespace(result_rows=[(v,) for v in self.applied])
        if query.startswith("SELECT count()"):
            return SimpleNamespace(result_rows=[(self.count_result,)])
        return SimpleNamespace(result_rows=[])

    def insert(
        self,
        table: str,
        data: list[list[Any]],
        column_names: list[str] | None = None,
        database: str | None = None,
    ) -> Any:
        if table.endswith("schema_migrations"):
            self.applied.extend(int(r[0]) for r in data)
        else:
            self.inserts.append((table, data))


# ── 1. Row mapping ───────────────────────────────────────────────────


def test_to_float32_quantises_like_clickhouse() -> None:
    assert to_float32(0.5) == 0.5
    assert to_float32(0.1) != 0.1
    assert abs(to_float32(0.1) - 0.1) < 1e-8


def test_to_float32_rejects_values_float32_cannot_hold() -> None:
    with pytest.raises(ValueError):
        to_float32(1e39)


def test_observation_to_row_uses_column_order() -> None:
    row = observation_to_row(make_obs(value=2.5), CATALOG)
    assert COLUMNS == ("cell", "variable", "source", "ts", "value")
    assert row == [int(CELL, 16), "precipitation", "open-meteo", T0, 2.5]


def test_row_round_trip_preserves_every_field() -> None:
    original = make_obs(variable="temperature", value=27.25)
    restored = row_to_observation(observation_to_row(original, CATALOG), CATALOG)
    assert restored.cell == original.cell
    assert restored.variable == original.variable
    assert restored.source == original.source
    assert restored.unit == original.unit
    assert restored.timestamp == original.timestamp
    assert restored.value == to_float32(original.value)


def test_idempotency_key_is_recomputed_and_matches_arrival_key() -> None:
    """The key is not stored (8 incompressible bytes/row); it must be derivable."""
    for variable in ("precipitation", "pressure", "discharge"):
        original = make_obs(variable=variable)
        restored = row_to_observation(observation_to_row(original, CATALOG), CATALOG)
        assert restored.idempotency_key == original.idempotency_key


def test_microsecond_timestamps_survive() -> None:
    ts = datetime(2024, 7, 1, 6, 0, 0, 123456, tzinfo=UTC)
    restored = row_to_observation(observation_to_row(make_obs(ts=ts), CATALOG), CATALOG)
    assert restored.timestamp == ts


# ── 3. Validation happens before any insert ─────────────────────────


def test_unknown_variable_rejected() -> None:
    obs = make_obs()
    obs.variable = "not_a_variable"
    with pytest.raises(ValueError, match="not_a_variable"):
        observation_to_row(obs, CATALOG)


def test_non_canonical_unit_rejected() -> None:
    obs = make_obs()
    obs.unit = "inches"
    with pytest.raises(ValueError, match="unit"):
        observation_to_row(obs, CATALOG)


def test_invalid_h3_cell_rejected() -> None:
    with pytest.raises(ValueError, match="cell"):
        observation_to_row(make_obs(cell="not-a-cell"), CATALOG)


def test_bad_record_in_batch_means_nothing_is_inserted() -> None:
    client = FakeClient()
    store = ColdStore(client, database="t")
    bad = make_obs()
    bad.unit = "inches"
    with pytest.raises(ValueError):
        store.write([make_obs(), bad, make_obs(value=3.0)])
    assert client.inserts == []


# ── 4. Batching ──────────────────────────────────────────────────────


def test_write_splits_into_batches() -> None:
    client = FakeClient()
    store = ColdStore(client, database="t", batch_size=3)
    observations = [make_obs(value=float(i)) for i in range(7)]
    stored, duplicates = store.write(observations)
    assert (stored, duplicates) == (7, 0)
    assert [len(data) for _, data in client.inserts] == [3, 3, 1]


def test_write_empty_is_a_noop() -> None:
    client = FakeClient()
    assert ColdStore(client, database="t").write([]) == (0, 0)
    assert client.inserts == []


# ── 5. Migrations ────────────────────────────────────────────────────


def test_migration_versions_are_unique_and_increasing() -> None:
    versions = [m.version for m in get_migrations("t")]
    assert versions == sorted(set(versions))
    assert versions[0] == 1


def test_every_migration_statement_is_idempotent() -> None:
    for migration in get_migrations("t"):
        for statement in migration.statements:
            assert "IF NOT EXISTS" in statement.upper(), statement


def test_second_migrate_applies_nothing() -> None:
    client = FakeClient()
    store = ColdStore(client, database="t")
    first = store.migrate()
    second = store.migrate()
    assert first == [m.version for m in get_migrations("t")]
    assert second == []


@pytest.mark.parametrize("name", ["x; DROP TABLE y", "a b", "", "1abc", "a-b", "a" * 100])
def test_database_name_is_validated(name: str) -> None:
    with pytest.raises(ValueError, match="database"):
        ColdStore(FakeClient(), database=name)


# ── 6. Schema design (storage budget) ────────────────────────────────


def _observations_ddl() -> str:
    statements = [s for m in get_migrations("t") for s in m.statements]
    return next(s for s in statements if "t.observations" in s and "CREATE TABLE" in s)


def test_schema_uses_replacing_merge_tree_sorted_for_dedup_and_range_scans() -> None:
    ddl = _observations_ddl()
    assert "ReplacingMergeTree" in ddl
    assert "ORDER BY (cell, variable, source, ts)" in ddl
    assert "PARTITION BY toYYYYMM(ts)" in ddl


def test_schema_uses_compact_types_and_codecs() -> None:
    ddl = _observations_ddl()
    ddl = " ".join(ddl.split())  # collapse column-alignment whitespace
    assert "cell UInt64" in ddl
    assert "variable LowCardinality(String)" in ddl
    assert "source LowCardinality(String)" in ddl
    assert "value Float32" in ddl
    assert "Float64" not in ddl
    assert ddl.count("ZSTD") >= 3  # cell, ts, value
    assert "DoubleDelta" in ddl  # timestamps
    assert "idempotency_key" not in ddl  # derivable; would cost 8 incompressible bytes/row


# ── 7. Reads always deduplicate ──────────────────────────────────────


def test_every_read_query_deduplicates_with_final() -> None:
    client = FakeClient()
    store = ColdStore(client, database="t")
    store.count()
    store.read_cell(CELL)
    store.read_cell(CELL, variable="precipitation", start=T0, end=T0)
    store.read_all()
    store.count_by_parent(5)
    assert client.queries, "no queries were issued"
    for query in client.queries:
        assert "FINAL" in query, query


def test_raw_count_is_the_only_query_without_final() -> None:
    client = FakeClient()
    ColdStore(client, database="t").raw_count()
    assert "FINAL" not in client.queries[0]


# ── 8. Regression: catalog units agree with the adapter ─────────────


def test_every_adapter_unit_equals_catalog_canonical_unit() -> None:
    """Regression: catalog said wind_speed is m/s while Open-Meteo emits km/h."""
    for variable, unit in OPEN_METEO_VARIABLE_MAP.values():
        assert CATALOG[variable].canonical_unit == unit, variable


def test_observations_built_from_adapter_units_carry_the_catalog_unit() -> None:
    """Regression: convert_unit lower-cased pass-through units, so pressure became 'hpa'."""
    for variable, unit in OPEN_METEO_VARIABLE_MAP.values():
        obs = Observation.from_raw(
            cell=CELL, source="open-meteo", variable=variable, raw_value=1.0, raw_unit=unit,
            timestamp=T0,
        )  # fmt: skip
        assert obs.unit == CATALOG[variable].canonical_unit, variable


def test_float32_helper_is_finite_for_catalog_scale_values() -> None:
    for value in (0.0, -40.0, 1013.25, 12345.678, 1e-6):
        assert math.isfinite(to_float32(value))


# ── 9. Connection credentials (docker-compose creates this user) ─────


def test_credentials_default_to_the_compose_dev_user() -> None:
    assert resolve_credentials({}) == ("disaster", "disaster")


def test_credentials_can_be_overridden_from_the_environment() -> None:
    env = {"CLICKHOUSE_USER": "alice", "CLICKHOUSE_PASSWORD": "s3cret"}
    assert resolve_credentials(env) == ("alice", "s3cret")


# ── 10. Regression: driver timezone behaviour must never shift timestamps ─


@pytest.fixture
def india_local_time(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Process-local timezone = IST (UTC+05:30), like the WSL machine this bug appeared on."""
    monkeypatch.setenv("TZ", "IST-5:30")  # POSIX form: needs no tz database
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_naive_datetime_from_driver_is_read_as_utc_not_local_time(india_local_time: Any) -> None:
    """Regression: a driver returning a naive UTC wall-clock time was shifted by -05:30."""
    assert os.environ["TZ"] == "IST-5:30"
    naive_utc = datetime(2024, 7, 1, 6, 0)  # what the column holds, tz stripped by the driver
    row = [int(CELL, 16), "precipitation", "open-meteo", naive_utc, 1.5]
    assert row_to_observation(row, CATALOG).timestamp == T0


def test_integer_microsecond_timestamps_are_accepted_too() -> None:
    micros = int(T0.timestamp()) * 1_000_000 + 123_456
    row = [int(CELL, 16), "precipitation", "open-meteo", micros, 1.5]
    expected = datetime(2024, 7, 1, 6, 0, 0, 123_456, tzinfo=UTC)
    assert row_to_observation(row, CATALOG).timestamp == expected
