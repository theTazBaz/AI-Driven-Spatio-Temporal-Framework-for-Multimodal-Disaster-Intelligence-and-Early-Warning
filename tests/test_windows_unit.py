"""Increment 8 unit tests - hourly buckets and rolling windows (no network, no containers).

Test list (plan.md, Increment 8), unit-testable half:
 1. Bucket/window conventions: hour-ending buckets, window = last N buckets ending at as_of.
 2. The slow Python ``reference_windows`` oracle agrees with the bucket combine path on
    random data (hypothesis) - the same combine the Redis hot tier will reuse in Increment 9.
 3. Gaps report a coverage fraction; an empty window is None (never 0).
 4. Late / duplicate / revised data: every write recomputes the touched buckets from the
    deduplicated observations (the SQL itself is exercised in the integration tests).
 5. Reads of the bucket table deduplicate with FINAL.
 6. Migration 2 schema.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import h3
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from disaster_intel.models import AggMethod, Observation, VariableCatalog
from disaster_intel.storage import ColdStore, get_migrations
from disaster_intel.windows import (
    BUCKET_END_SQL,
    HOUR,
    WINDOW_HOURS,
    Bucket,
    WindowValue,
    bucket_end,
    bucketize,
    combine_buckets,
    floor_hour,
    format_windows,
    reference_windows,
    variable_names,
    windows_from_buckets,
)

pytestmark = pytest.mark.unit

CELL = "872a1072bffffff"
T0 = datetime(2024, 7, 1, 0, 0, tzinfo=UTC)
CATALOG = VariableCatalog.default()


def obs(
    variable: str = "precipitation",
    value: float = 1.0,
    ts: datetime = T0,
    source: str = "open-meteo",
    cell: str = CELL,
) -> Observation:
    return Observation(
        cell=cell,
        source=source,
        variable=variable,
        value=value,
        unit=CATALOG[variable].canonical_unit,
        timestamp=ts,
        idempotency_key="unused",
    )


# ── 1. Conventions ───────────────────────────────────────────────────


def test_window_lengths_are_1h_6h_24h_72h_7d() -> None:
    assert dict(WINDOW_HOURS) == {"1h": 1, "6h": 6, "24h": 24, "72h": 72, "7d": 168}


def test_bucket_end_is_hour_ending() -> None:
    ten = T0 + timedelta(hours=10)
    assert bucket_end(ten) == ten  # exactly on the hour stays in its own bucket
    assert bucket_end(ten - timedelta(microseconds=1)) == ten
    assert bucket_end(ten + timedelta(microseconds=1)) == ten + HOUR
    assert bucket_end(ten + timedelta(minutes=59)) == ten + HOUR


def test_floor_hour() -> None:
    ten = T0 + timedelta(hours=10)
    assert floor_hour(ten) == ten
    assert floor_hour(ten + timedelta(minutes=37, seconds=5)) == ten
    assert floor_hour(ten - timedelta(microseconds=1)) == ten - HOUR


def test_helpers_normalise_time_zones_and_reject_naive_datetimes() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    assert bucket_end(datetime(2024, 7, 1, 5, 30, tzinfo=ist)) == T0
    with pytest.raises(ValueError, match="Naive"):
        floor_hour(datetime(2024, 7, 1))
    with pytest.raises(ValueError, match="Naive"):
        bucket_end(datetime(2024, 7, 1))


def test_sql_bucket_expression_matches_python_bucket_end() -> None:
    """The SQL formula, evaluated here in Python, equals bucket_end() (SQL itself: integration)."""
    assert "3599999999" in BUCKET_END_SQL
    assert "3600000000" in BUCKET_END_SQL
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    for micros in (0, 1, 3_599_999_999, 3_600_000_000, 3_600_000_001, 1_719_792_000_123_456):
        ts = epoch + timedelta(microseconds=micros)
        sql_seconds = (micros + 3_599_999_999) // 3_600_000_000 * 3600
        assert bucket_end(ts) == epoch + timedelta(seconds=sql_seconds)


# ── Bucket ───────────────────────────────────────────────────────────


def test_bucket_of_and_merge() -> None:
    a = Bucket.of([1.0, 5.0])
    b = Bucket.of([3.0])
    assert a == Bucket(n=2, total=6.0, minimum=1.0, maximum=5.0)
    assert a.merge(b) == Bucket(n=3, total=9.0, minimum=1.0, maximum=5.0)


def test_empty_bucket_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Bucket.of([])


def test_bucketize_groups_by_variable_and_hour_ending_label() -> None:
    rows = [
        obs(ts=T0 + timedelta(minutes=30), value=1.0),  # -> bucket 01:00
        obs(ts=T0 + timedelta(hours=1), value=2.0),  # -> bucket 01:00 (exactly on the hour)
        obs(ts=T0 + timedelta(hours=1, minutes=1), value=4.0),  # -> bucket 02:00
        obs(variable="temperature", ts=T0 + timedelta(hours=1), value=30.0),
    ]
    got = bucketize(rows)
    assert got["precipitation"][T0 + HOUR] == Bucket(n=2, total=3.0, minimum=1.0, maximum=2.0)
    assert got["precipitation"][T0 + 2 * HOUR] == Bucket(n=1, total=4.0, minimum=4.0, maximum=4.0)
    assert got["temperature"][T0 + HOUR].n == 1


def test_bucketize_can_filter_one_source() -> None:
    rows = [obs(source="open-meteo", value=1.0), obs(source="imerg", value=9.0)]
    assert bucketize(rows, source="imerg")["precipitation"][T0].total == 9.0
    assert bucketize(rows)["precipitation"][T0].total == 10.0


# ── 2/3. Combining buckets, coverage, gaps ───────────────────────────


def buckets_for_hours(hours: list[int], value: float = 2.0) -> dict[datetime, Bucket]:
    return {T0 + h * HOUR: Bucket.of([value]) for h in hours}


@pytest.mark.parametrize(
    ("agg", "expected"),
    [(AggMethod.SUM, 12.0), (AggMethod.AVG, 4.0), (AggMethod.MAX, 7.0), (AggMethod.MIN, 1.0)],
)
def test_combine_uses_the_variables_aggregation(agg: AggMethod, expected: float) -> None:
    buckets = {
        T0 + 1 * HOUR: Bucket.of([1.0, 4.0]),
        T0 + 2 * HOUR: Bucket.of([7.0]),
    }
    assert combine_buckets(buckets, T0 + 2 * HOUR, 6, agg).value == expected


def test_average_is_weighted_by_observation_count_not_by_bucket() -> None:
    buckets = {T0 + HOUR: Bucket.of([0.0, 0.0, 0.0]), T0 + 2 * HOUR: Bucket.of([8.0])}
    assert combine_buckets(buckets, T0 + 2 * HOUR, 6, AggMethod.AVG).value == 2.0  # not 4.0


def test_window_covers_exactly_the_last_n_buckets_ending_at_as_of() -> None:
    end = T0 + 10 * HOUR
    buckets = buckets_for_hours(list(range(0, 11)))
    sums = {h: combine_buckets(buckets, end, h, AggMethod.SUM).value for h in (1, 6, 10, 11)}
    assert sums == {1: 2.0, 6: 12.0, 10: 20.0, 11: 22.0}  # bucket T0 (hour 0) is the 11th back


def test_as_of_inside_an_hour_is_floored() -> None:
    buckets = buckets_for_hours([10])
    inside = combine_buckets(buckets, T0 + 10 * HOUR + timedelta(minutes=59), 1, AggMethod.SUM)
    assert inside.value == 2.0
    before = combine_buckets(buckets, T0 + 10 * HOUR - timedelta(minutes=1), 1, AggMethod.SUM)
    assert before.value is None  # the 10:00 bucket is not complete yet at 09:59


def test_gaps_report_coverage_fraction() -> None:
    buckets = buckets_for_hours([h for h in range(1, 25) if h % 4 != 0])  # 18 of 24 hours
    result = combine_buckets(buckets, T0 + 24 * HOUR, 24, AggMethod.SUM)
    assert result.observed_hours == 18
    assert result.hours == 24
    assert result.coverage == 0.75
    assert result.value == 36.0


def test_empty_window_is_none_not_zero() -> None:
    result = combine_buckets({}, T0 + 24 * HOUR, 24, AggMethod.SUM)
    assert result == WindowValue(value=None, observed_hours=0, hours=24)
    assert result.coverage == 0.0


def test_window_must_span_at_least_one_hour() -> None:
    with pytest.raises(ValueError, match="at least one hour"):
        combine_buckets({}, T0, 0, AggMethod.SUM)


def test_windows_cover_every_catalog_variable_even_without_data() -> None:
    table = windows_from_buckets({}, T0)
    assert list(table) == variable_names(CATALOG)
    assert all(set(per_window) == set(WINDOW_HOURS) for per_window in table.values())
    assert all(wv.value is None for per_window in table.values() for wv in per_window.values())


# ── 2. The oracle ────────────────────────────────────────────────────


def test_oracle_includes_a_reading_exactly_at_as_of_and_excludes_one_microsecond_later() -> None:
    end = T0 + 10 * HOUR
    rows = [obs(ts=end, value=3.0), obs(ts=end + timedelta(microseconds=1), value=100.0)]
    one_hour = reference_windows(rows, CELL, end)["precipitation"]["1h"]
    assert one_hour.value == 3.0
    later = reference_windows(rows, CELL, end + HOUR)["precipitation"]["1h"]
    assert later.value == 100.0  # ...and it arrives in the next hour's bucket


def test_oracle_ignores_other_cells_and_filters_sources() -> None:
    other_cell = h3.latlng_to_cell(12.9, 74.8, 7)
    rows = [
        obs(value=1.0),
        obs(value=10.0, cell=other_cell),
        obs(value=100.0, source="imerg"),
    ]
    end = T0
    assert reference_windows(rows, CELL, end)["precipitation"]["1h"].value == 101.0
    assert reference_windows(rows, CELL, end, source="imerg")["precipitation"]["1h"].value == 100.0
    assert reference_windows([], CELL, end)["precipitation"]["1h"].value is None


VARIABLES = [
    "precipitation",
    "soil_moisture_surface",
    "temperature",
    "wind_speed",
    "discharge",
    "cap_severity",
]


@st.composite
def random_observation(draw: st.DrawFn) -> Observation:
    minutes = draw(st.integers(0, 60 * 24 * 9))
    micro = draw(st.sampled_from([0, 0, 0, 1]))  # hit hour boundaries and 1 us past them
    return obs(
        variable=draw(st.sampled_from(VARIABLES)),
        source=draw(st.sampled_from(["open-meteo", "imerg"])),
        value=draw(st.floats(-1000, 1000, allow_nan=False, allow_infinity=False)),
        ts=T0 + timedelta(minutes=minutes, microseconds=micro),
    )


def same(a: dict[str, dict[str, WindowValue]], b: dict[str, dict[str, WindowValue]]) -> None:
    assert a.keys() == b.keys()
    for variable in a:
        assert a[variable].keys() == b[variable].keys()
        for window, x in a[variable].items():
            y = b[variable][window]
            assert (x.observed_hours, x.hours) == (y.observed_hours, y.hours)
            if x.value is None:
                assert y.value is None
            else:
                assert y.value == pytest.approx(x.value, rel=1e-9, abs=1e-9)


@settings(max_examples=150, deadline=None)
@given(
    rows=st.lists(random_observation(), max_size=120),
    as_of_minutes=st.integers(0, 60 * 24 * 11),
    source=st.sampled_from([None, "open-meteo", "imerg"]),
)
def test_bucket_combine_equals_the_slow_oracle_on_random_data(
    rows: list[Observation], as_of_minutes: int, source: str | None
) -> None:
    as_of = T0 + timedelta(minutes=as_of_minutes)
    expected = reference_windows(rows, CELL, as_of, source=source)
    actual = windows_from_buckets(bucketize(rows, source=source), as_of)
    same(actual, expected)


def test_format_windows_renders_value_and_coverage() -> None:
    rows = [obs(ts=T0 + HOUR, value=2.0)]
    text = format_windows(windows_from_buckets(bucketize(rows), T0 + HOUR))
    assert "precipitation" in text
    assert "2.00 (100%)" in text  # 1h window fully covered
    assert "(4%)" in text  # 1 of 24 hours


# ── ColdStore with a fake client ─────────────────────────────────────


class RecordingClient:
    """Records every statement; answers bucket queries from canned rows."""

    def __init__(self, bucket_rows: list[tuple[Any, ...]] | None = None) -> None:
        self.commands: list[str] = []
        self.queries: list[tuple[str, Any]] = []
        self.inserts: list[tuple[str, list[list[Any]]]] = []
        self.bucket_rows = bucket_rows or []
        self.applied: list[int] = []

    def command(self, cmd: str, parameters: Any = None) -> Any:
        self.commands.append(cmd)

    def query(self, query: str, parameters: Any = None) -> Any:
        self.queries.append((query, parameters))
        if "schema_migrations" in query:
            return SimpleNamespace(result_rows=[(v,) for v in self.applied])
        if "hourly_buckets" in query:
            return SimpleNamespace(result_rows=self.bucket_rows)
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


def micros(ts: datetime) -> int:
    return (ts - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)


def epoch_seconds(ts: datetime) -> int:
    return int(ts.timestamp())


# ── 4. Writes recompute the touched buckets ──────────────────────────


def test_write_recomputes_touched_buckets_from_deduplicated_observations() -> None:
    client = RecordingClient()
    store = ColdStore(client, database="t")
    ts = T0 + timedelta(hours=6)
    store.write([obs(ts=ts, value=1.0), obs(variable="temperature", ts=ts, value=20.0)])
    (statement,) = client.commands
    assert statement.startswith("INSERT INTO t.hourly_buckets")
    assert "FROM t.observations FINAL" in statement  # duplicates/revisions collapse first
    assert f"cell IN ({int(CELL, 16)})" in statement
    assert "variable IN ('precipitation', 'temperature')" in statement
    assert "GROUP BY cell, variable, source, bucket" in statement
    assert BUCKET_END_SQL in statement


def test_refresh_range_is_the_half_open_span_of_the_touched_buckets() -> None:
    client = RecordingClient()
    store = ColdStore(client, database="t")
    first = T0 + timedelta(hours=6)  # bucket 06:00 holds (05:00, 06:00]
    last = T0 + timedelta(hours=8, minutes=15)  # bucket 09:00 holds (08:00, 09:00]
    store.write([obs(ts=first), obs(ts=last)])
    (statement,) = client.commands
    assert f"ts > fromUnixTimestamp64Micro({micros(T0 + 5 * HOUR)}, 'UTC')" in statement
    assert f"ts <= fromUnixTimestamp64Micro({micros(T0 + 9 * HOUR)}, 'UTC')" in statement


def test_a_late_observation_refreshes_only_its_own_hour() -> None:
    client = RecordingClient()
    store = ColdStore(client, database="t")
    late = T0 + timedelta(hours=3, minutes=20)  # bucket 04:00
    store.write([obs(ts=late)])
    (statement,) = client.commands
    assert f"ts > fromUnixTimestamp64Micro({micros(T0 + 3 * HOUR)}, 'UTC')" in statement
    assert f"ts <= fromUnixTimestamp64Micro({micros(T0 + 4 * HOUR)}, 'UTC')" in statement


def test_empty_write_does_not_touch_buckets() -> None:
    client = RecordingClient()
    ColdStore(client, database="t").write([])
    assert client.commands == []


def test_bad_record_means_no_bucket_refresh_either() -> None:
    client = RecordingClient()
    bad = obs()
    bad.unit = "inches"
    with pytest.raises(ValueError):
        ColdStore(client, database="t").write([obs(), bad])
    assert client.commands == []
    assert client.inserts == []


def test_maintain_buckets_false_skips_refresh_until_rebuild() -> None:
    client = RecordingClient()
    store = ColdStore(client, database="t", maintain_buckets=False)
    store.write([obs()])
    assert client.commands == []
    store.rebuild_buckets()
    (statement,) = client.commands
    assert statement.startswith("INSERT INTO t.hourly_buckets")
    assert "FINAL WHERE 1 GROUP BY" in statement


def test_refresh_is_split_so_no_statement_lists_thousands_of_cells() -> None:
    client = RecordingClient()
    store = ColdStore(client, database="t")
    cells = sorted(h3.grid_disk(CELL, 30))  # 2,791 distinct res-7 cells
    assert len(cells) > 2_000
    store.write([obs(cell=c) for c in cells])
    assert len(client.commands) == 2
    assert all(len(c) < 100_000 for c in client.commands)


def test_write_still_batches_inserts_independently_of_buckets() -> None:
    client = RecordingClient()
    store = ColdStore(client, database="t", batch_size=3)
    store.write([obs(value=float(i)) for i in range(7)])
    assert [len(data) for _, data in client.inserts] == [3, 3, 1]
    assert len(client.commands) == 1  # one refresh for the whole write, not one per batch


# ── 5. Bucket reads deduplicate and parse ────────────────────────────


def test_read_buckets_deduplicates_with_final_and_filters_the_range() -> None:
    client = RecordingClient()
    ColdStore(client, database="t").read_buckets(CELL, T0, T0 + 24 * HOUR)
    ((sql, params),) = client.queries
    assert "FROM t.hourly_buckets FINAL" in sql
    assert f"cell = {int(CELL, 16)}" in sql
    assert f"bucket > toDateTime({epoch_seconds(T0)}, 'UTC')" in sql
    assert f"bucket <= toDateTime({epoch_seconds(T0 + 24 * HOUR)}, 'UTC')" in sql
    assert params is None


def test_read_buckets_can_filter_one_source() -> None:
    client = RecordingClient()
    ColdStore(client, database="t").read_buckets(CELL, T0, T0 + HOUR, source="imerg")
    ((sql, params),) = client.queries
    assert "source = {source:String}" in sql
    assert params == {"source": "imerg"}


def test_read_buckets_rejects_an_invalid_cell() -> None:
    with pytest.raises(ValueError, match="cell"):
        ColdStore(RecordingClient(), database="t").read_buckets("nope", T0, T0 + HOUR)


def test_windows_combines_the_buckets_the_database_returns() -> None:
    end = T0 + 48 * HOUR
    rows = [
        ("precipitation", epoch_seconds(end), 1, 5.0, 5.0, 5.0),
        ("precipitation", epoch_seconds(end - 3 * HOUR), 2, 4.0, 1.0, 3.0),
        ("precipitation", epoch_seconds(end - 30 * HOUR), 1, 10.0, 10.0, 10.0),
        ("discharge", epoch_seconds(end - HOUR), 1, 80.0, 80.0, 80.0),
    ]
    client = RecordingClient(rows)
    store = ColdStore(client, database="t")

    table = store.windows(CELL, end + timedelta(minutes=20))  # as_of is floored to 48:00

    rain = table["precipitation"]
    assert rain["1h"] == WindowValue(value=5.0, observed_hours=1, hours=1)
    assert rain["6h"] == WindowValue(value=9.0, observed_hours=2, hours=6)
    assert rain["24h"].value == 9.0
    assert rain["72h"] == WindowValue(value=19.0, observed_hours=3, hours=72)
    assert table["discharge"]["6h"].value == 80.0  # MAX
    assert table["temperature"]["7d"] == WindowValue(value=None, observed_hours=0, hours=168)
    ((sql, _),) = client.queries
    assert f"bucket > toDateTime({epoch_seconds(end - 168 * HOUR)}, 'UTC')" in sql
    assert f"bucket <= toDateTime({epoch_seconds(end)}, 'UTC')" in sql


def test_windows_accepts_custom_window_lengths() -> None:
    client = RecordingClient()
    table = ColdStore(client, database="t").windows(CELL, T0, window_hours={"2h": 2})
    assert set(table["precipitation"]) == {"2h"}
    ((sql, _),) = client.queries
    assert f"bucket > toDateTime({epoch_seconds(T0 - 2 * HOUR)}, 'UTC')" in sql


# ── 6. Migration 2 ───────────────────────────────────────────────────


def test_migration_2_creates_a_versioned_bucket_table() -> None:
    migrations = get_migrations("t")
    assert [m.version for m in migrations] == [1, 2]
    ddl = " ".join(migrations[1].statements[0].split())
    assert "CREATE TABLE IF NOT EXISTS t.hourly_buckets" in ddl
    assert "ENGINE = ReplacingMergeTree(version)" in ddl
    assert "ORDER BY (cell, variable, source, bucket)" in ddl
    assert "n_obs UInt32" in ddl
    assert "sum_value Float64" in ddl  # sums must not lose precision across 168 buckets
    assert "min_value Float32" in ddl
    assert "max_value Float32" in ddl
    assert "t.observations" not in ddl


def test_migrating_a_database_that_already_has_version_1_applies_only_version_2() -> None:
    client = RecordingClient()
    client.applied = [1]
    assert ColdStore(client, database="t").migrate() == [2]
    assert any("hourly_buckets" in c for c in client.commands)
