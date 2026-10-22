"""Increment 9 unit tests - Redis hot tier (no network, no containers; FakeRedis).

Test list (plan.md, Increment 9), unit-testable half:
 1. Blob encoding round-trips exactly (float32, microsecond timestamps, sorted).
 2. Hot windows equal the slow Python oracle after replaying a (shuffled) stream.
    Hypothesis differential test over random series, as-of times and sources.
 3. Expired observations are evicted: lazily on write, by sweep, and refused on read.
 4. Applying one observation twice changes nothing (and a revised value replaces).
 5. Wiping Redis and rebuilding from a cold reader gives IDENTICAL state.
 6. Write path cost: bounded round trips, validated before any I/O, TTL set.
 7. Latency reporting: average, p50, p95, p99.
The real-Redis half (latency, memory per cell, eviction policy) is in test_hot_integration.py
and test_hot_perf.py.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from disaster_intel.clock import FakeClock
from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.storage import (
    HotFormatError,
    HotRangeError,
    HotStore,
    TeeSink,
    rebuild_hot_from_cold,
    summarize_latencies,
    to_float32,
)
from disaster_intel.storage.hot import (
    bucket_end_micros,
    decode_records,
    encode_records,
    make_field,
)
from disaster_intel.windows import (
    HOUR,
    WINDOW_HOURS,
    WindowValue,
    bucket_end,
    reference_windows,
    variable_names,
)
from fake_redis import FakeRedis

pytestmark = pytest.mark.unit

CELL = "872a1072bffffff"
OTHER_CELL = "872a10729ffffff"
NOW = datetime(2024, 7, 10, 12, 0, tzinfo=UTC)
CATALOG = VariableCatalog.default()
ALL_VARIABLES = variable_names(CATALOG)
Windows = dict[str, dict[str, WindowValue]]


def make_obs(
    variable: str = "precipitation",
    value: float = 1.5,
    ts: datetime = NOW,
    cell: str = CELL,
    source: str = "open-meteo",
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


def new_store(now: datetime = NOW) -> tuple[HotStore, FakeRedis, FakeClock]:
    redis_ = FakeRedis()
    clock = FakeClock(now)
    return HotStore(redis_, clock=clock), redis_, clock


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
            if expected.value is None:
                assert actual.value is None, where
            else:
                assert actual.value == pytest.approx(expected.value, rel=1e-9, abs=1e-9), where


# ── 1. Encoding ──────────────────────────────────────────────────────


def test_encode_decode_round_trip_is_exact_for_float32_and_microseconds() -> None:
    base = int(NOW.timestamp()) * 1_000_000
    records = {base + 123_456: to_float32(0.1), base - 7: to_float32(99.25), base: 0.0}
    assert decode_records(encode_records(records)) == records


def test_single_record_round_trips() -> None:
    assert decode_records(encode_records({5: 2.5})) == {5: 2.5}


def test_encoding_rejects_empty_and_decoding_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        encode_records({})
    with pytest.raises(HotFormatError):
        decode_records(b"")
    with pytest.raises(HotFormatError):
        decode_records(b"\x09whatever")  # unknown format version
    with pytest.raises(HotFormatError):
        decode_records(b"\x01not-zlib")


def test_hourly_series_compresses_to_about_four_bytes_per_observation() -> None:
    rng = random.Random(1)
    base = int(NOW.timestamp()) * 1_000_000
    records = {base + h * 3_600_000_000: to_float32(rng.uniform(0, 100)) for h in range(168)}
    assert len(encode_records(records)) < 168 * 5.5  # float32 noise dominates; ts ~ free


def _us(ts: datetime) -> int:
    return (ts - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)


def test_bucket_end_micros_matches_windows_bucket_end() -> None:
    rng = random.Random(3)
    for _ in range(500):
        ts = NOW + timedelta(microseconds=rng.randrange(-(10**10), 10**10))
        assert bucket_end_micros(_us(ts)) == _us(bucket_end(ts))


def test_field_names_reject_the_separator() -> None:
    assert make_field("precipitation", "open-meteo") == "precipitation|open-meteo"
    with pytest.raises(ValueError):
        make_field("precipitation", "a|b")


# ── 2. Oracle agreement ──────────────────────────────────────────────


def random_series(cell: str, seed: int, end: datetime, days: int = 10) -> list[Observation]:
    rng = random.Random(seed)
    out: list[Observation] = []
    for variable in ALL_VARIABLES:
        for hour in range(24 * days + 1):
            if rng.random() < 0.3:
                continue
            ts = end - timedelta(hours=hour)
            if rng.random() < 0.3:
                ts -= timedelta(minutes=rng.randrange(1, 60), microseconds=rng.randrange(0, 5))
            value = float(rng.randint(1, 4)) if variable == "cap_severity" else rng.uniform(0, 100)
            source = "open-meteo" if rng.random() < 0.8 else "imerg"
            out.append(make_obs(variable, value, ts, cell, source))
    return out


def test_hot_windows_equal_the_oracle_after_replaying_a_shuffled_stream() -> None:
    hot, _, _ = new_store()
    arrivals = random_series(CELL, 7, NOW) + random_series(OTHER_CELL, 8, NOW)
    random.Random(2).shuffle(arrivals)
    for i in range(0, len(arrivals), 250):
        hot.write(arrivals[i : i + 250])
    stored = [as_stored(o) for o in arrivals]
    for cell in (CELL, OTHER_CELL):
        assert_windows_equal(hot.windows(cell), reference_windows(stored, cell, NOW, CATALOG))
        assert_windows_equal(
            hot.windows(cell, source="imerg"),
            reference_windows(stored, cell, NOW, CATALOG, source="imerg"),
        )


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    minutes_later=st.integers(0, 600),
    batch=st.integers(1, 400),
)
def test_hot_equals_oracle_for_random_streams_and_as_of_times(
    seed: int, minutes_later: int, batch: int
) -> None:
    hot, _, clock = new_store()
    arrivals = random_series(CELL, seed, NOW, days=9)
    random.Random(seed).shuffle(arrivals)
    for i in range(0, len(arrivals), batch):
        hot.write(arrivals[i : i + batch])
    clock.sleep(minutes_later * 60)  # time passes with no writes; windows just slide
    as_of = clock.now()
    stored = [as_stored(o) for o in arrivals]
    assert_windows_equal(hot.windows(CELL), reference_windows(stored, CELL, as_of, CATALOG))


def test_short_windows_may_be_read_slightly_in_the_past() -> None:
    hot, _, _ = new_store()
    obs = [make_obs("precipitation", 2.0, NOW - h * HOUR) for h in range(0, 100)]
    hot.write(obs)
    past = NOW - timedelta(hours=60)
    got = hot.windows(CELL, past, window_hours={"6h": 6, "24h": 24})
    want = reference_windows(
        [as_stored(o) for o in obs], CELL, past, CATALOG, {"6h": 6, "24h": 24}
    )
    assert_windows_equal(got, want)


def test_empty_cell_has_none_values_and_zero_coverage() -> None:
    hot, _, _ = new_store()
    table = hot.windows(CELL)
    assert all(w.value is None and w.coverage == 0.0 for pw in table.values() for w in pw.values())


def test_reading_exactly_on_the_hour_belongs_to_that_hours_window() -> None:
    hot, _, _ = new_store()
    ten = NOW.replace(hour=10)
    hot.write(
        [
            make_obs("precipitation", 3.0, ten),
            make_obs("precipitation", 100.0, ten + timedelta(microseconds=1)),
        ]
    )
    # Same convention as the cold tier: B holds B-1h < ts <= B.
    assert hot.windows(CELL, ten, window_hours={"1h": 1})["precipitation"]["1h"].value == 3.0
    nxt = ten + HOUR
    assert hot.windows(CELL, nxt, window_hours={"1h": 1})["precipitation"]["1h"].value == 100.0


# ── 3. Eviction ──────────────────────────────────────────────────────


def test_observations_older_than_seven_days_are_rejected_and_counted() -> None:
    hot, _, _ = new_store()
    too_old = NOW - timedelta(days=7, hours=2)
    assert hot.write([make_obs(ts=too_old)]) == (0, 0)
    assert hot.rejected_stale == 1
    assert hot.dump_cell(CELL) == {}


def test_the_oldest_retained_bucket_is_exactly_168_hours_back() -> None:
    hot, _, _ = new_store()
    edge = NOW - 168 * HOUR  # bucket_end == horizon -> expired
    inside = edge + timedelta(microseconds=1)  # bucket_end == horizon + 1h -> kept
    hot.write([make_obs(ts=edge, value=1.0), make_obs(ts=inside, value=2.0)])
    kept = hot.dump_cell(CELL)[("precipitation", "open-meteo")]
    assert [v for _, v in kept] == [2.0]


def test_a_write_prunes_expired_records_from_the_fields_it_touches() -> None:
    hot, _, clock = new_store()
    hot.write([make_obs(ts=NOW - HOUR * h, value=float(h)) for h in range(0, 100)])
    clock.sleep(5 * 24 * 3600)  # five days later the oldest ~3 days have expired
    hot.write([make_obs(ts=clock.now(), value=-1.0)])
    kept = hot.dump_cell(CELL)[("precipitation", "open-meteo")]
    horizon = _us(hot.horizon())
    assert all(bucket_end_micros(t) > horizon for t, _ in kept)
    assert kept[-1][1] == -1.0
    assert len(kept) < 100


def test_evict_expired_sweeps_fields_nobody_wrote_to() -> None:
    hot, redis_, clock = new_store()
    hot.write(
        [
            make_obs("precipitation", 1.0, NOW),
            make_obs("pressure", 1000.0, NOW - timedelta(hours=100)),
        ]
    )
    clock.sleep(3 * 24 * 3600)  # pressure@-100h is now ~172h old: expired; precipitation not
    removed = hot.evict_expired([CELL])
    assert removed == 1
    assert set(hot.dump_cell(CELL)) == {("precipitation", "open-meteo")}
    clock.sleep(30 * 24 * 3600)
    assert hot.evict_expired([CELL]) == 1
    assert hot.key(CELL) not in redis_.hashes  # Redis drops an emptied hash


def test_reading_beyond_retained_range_fails_loudly_instead_of_truncating() -> None:
    hot, _, _ = new_store()
    with pytest.raises(HotRangeError):
        hot.windows(CELL, NOW - timedelta(hours=1))  # a 7d window would need 169 h of history
    hot.windows(CELL, NOW)  # fine
    hot.windows(CELL, NOW + timedelta(hours=5))  # the future is fine (just empty)


def test_retention_cannot_be_shorter_than_the_longest_window() -> None:
    with pytest.raises(ValueError):
        HotStore(FakeRedis(), retention_hours=100)


def test_keys_get_a_ttl_longer_than_retention() -> None:
    hot, redis_, _ = new_store()
    hot.write([make_obs()])
    assert redis_.ttls[hot.key(CELL)] == (168 + 24) * 3600


# ── 4. Idempotency ───────────────────────────────────────────────────


def test_applying_one_observation_twice_changes_nothing() -> None:
    hot, redis_, _ = new_store()
    obs = make_obs("precipitation", 4.0, NOW - timedelta(minutes=20))
    assert hot.write([obs]) == (1, 0)
    snapshot = dict(redis_.hashes[hot.key(CELL)])
    windows_before = hot.windows(CELL)
    assert hot.write([obs]) == (0, 1)
    assert hot.write([obs, obs]) == (0, 2)
    assert redis_.hashes[hot.key(CELL)] == snapshot  # byte-identical
    assert hot.windows(CELL) == windows_before
    assert hot.windows(CELL)["precipitation"]["1h"].value == 4.0


def test_duplicates_inside_one_batch_are_counted_once_in_state() -> None:
    hot, _, _ = new_store()
    obs = make_obs("precipitation", 4.0)
    assert hot.write([obs, obs, obs]) == (1, 2)
    assert hot.windows(CELL)["precipitation"]["24h"].value == 4.0


def test_a_revised_value_replaces_the_old_one() -> None:
    hot, _, _ = new_store()
    hot.write([make_obs("precipitation", 1.0)])
    assert hot.write([make_obs("precipitation", 2.5)]) == (1, 0)
    assert hot.windows(CELL)["precipitation"]["1h"].value == 2.5


def test_two_sources_at_the_same_instant_are_different_facts() -> None:
    hot, _, _ = new_store()
    hot.write([make_obs(value=1.0, source="open-meteo"), make_obs(value=7.0, source="imerg")])
    assert hot.windows(CELL)["precipitation"]["1h"].value == 8.0
    assert hot.windows(CELL, source="imerg")["precipitation"]["1h"].value == 7.0


def test_idempotent_for_random_replays() -> None:
    hot, redis_, _ = new_store()
    arrivals = random_series(CELL, 11, NOW)
    hot.write(arrivals)
    first = dict(redis_.hashes[hot.key(CELL)])
    random.Random(4).shuffle(arrivals)
    hot.write(arrivals)  # the whole stream again, different order
    assert redis_.hashes[hot.key(CELL)] == first


# ── 5. Rebuild from cold ─────────────────────────────────────────────


class FakeCold:
    """Stands in for ColdStore.read_cell: float32-quantised, start inclusive, sorted."""

    def __init__(self, observations: list[Observation]) -> None:
        self.rows = [as_stored(o) for o in observations]
        self.calls = 0

    def read_cell(
        self,
        cell: str,
        variable: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Observation]:
        self.calls += 1
        return sorted(
            (
                o
                for o in self.rows
                if o.cell == cell
                and (variable is None or o.variable == variable)
                and (start is None or o.timestamp >= start)
                and (end is None or o.timestamp < end)
            ),
            key=lambda o: o.timestamp,
        )


def test_wiping_redis_and_rebuilding_from_cold_gives_identical_state() -> None:
    hot, redis_, _ = new_store()
    arrivals = random_series(CELL, 21, NOW, days=12) + random_series(OTHER_CELL, 22, NOW, days=12)
    random.Random(9).shuffle(arrivals)
    hot.write(arrivals)  # the live path (old data is rejected as stale on the way in)
    live = {c: hot.dump_cell(c) for c in (CELL, OTHER_CELL)}
    live_windows = {c: hot.windows(c) for c in (CELL, OTHER_CELL)}
    assert all(live.values())

    cold = FakeCold(arrivals)  # cold keeps everything, including the old data
    report = rebuild_hot_from_cold(hot, cold, [CELL, OTHER_CELL], wipe=True)

    assert report.cleared_keys == 2
    assert report.cells == 2
    assert {c: hot.dump_cell(c) for c in (CELL, OTHER_CELL)} == live
    assert {c: hot.windows(c) for c in (CELL, OTHER_CELL)} == live_windows
    assert len(redis_.hashes) == 2


def test_rebuild_is_idempotent_and_reports_duplicates() -> None:
    hot, _, _ = new_store()
    cold = FakeCold(random_series(CELL, 5, NOW))
    first = rebuild_hot_from_cold(hot, cold, [CELL])
    state = hot.dump_cell(CELL)
    second = rebuild_hot_from_cold(hot, cold, [CELL])
    assert hot.dump_cell(CELL) == state
    assert first.stored > 0 and first.duplicates == 0
    assert second.stored == 0 and second.duplicates == second.observations_read


def test_rebuild_reads_only_the_retained_range_and_batches_writes() -> None:
    hot, redis_, _ = new_store()
    cells = [CELL, OTHER_CELL, "872a10728ffffff"]
    obs = [make_obs(ts=NOW - timedelta(days=d), cell=c) for c in cells for d in (0, 3, 9)]
    cold = FakeCold(obs)
    seen: list[tuple[int, int]] = []
    report = rebuild_hot_from_cold(
        hot, cold, cells, cells_per_write=2, progress=lambda a, b: seen.append((a, b))
    )
    assert cold.calls == 3
    assert report.observations_read == 6  # the 9-day-old readings are never even read
    assert report.stored == 6
    assert seen == [(2, 3), (3, 3)]
    assert redis_.round_trips == 4  # two writes x (read pipeline + write pipeline)


def test_rebuild_without_wipe_keeps_unrelated_keys() -> None:
    hot, _, _ = new_store()
    hot.write([make_obs(cell=OTHER_CELL)])
    rebuild_hot_from_cold(hot, FakeCold([make_obs()]), [CELL])
    assert set(hot.cells_present()) == {CELL, OTHER_CELL}


# ── 6. Write path ────────────────────────────────────────────────────


def test_bad_record_means_nothing_is_written() -> None:
    hot, redis_, _ = new_store()
    bad = make_obs()
    bad.unit = "inches"
    with pytest.raises(ValueError, match="unit"):
        hot.write([make_obs(), bad])
    assert redis_.hashes == {} and redis_.round_trips == 0


def test_unknown_variable_and_invalid_cell_are_rejected() -> None:
    hot, _, _ = new_store()
    bad = make_obs()
    bad.variable = "nope"
    with pytest.raises(ValueError, match="nope"):
        hot.write([bad])
    with pytest.raises(ValueError, match="cell"):
        hot.windows("not-a-cell")


def test_a_batch_costs_two_round_trips_regardless_of_size() -> None:
    hot, redis_, _ = new_store()
    obs = [
        make_obs(v, 1.0, NOW - h * HOUR, c)
        for c in (CELL, OTHER_CELL)
        for v in ("precipitation", "pressure")
        for h in range(30)
    ]
    hot.write(obs)
    assert redis_.round_trips == 2


def test_empty_write_is_a_noop() -> None:
    hot, redis_, _ = new_store()
    assert hot.write([]) == (0, 0)
    assert redis_.round_trips == 0


def test_a_whole_cell_is_read_with_one_hash_key() -> None:
    hot, redis_, _ = new_store()
    hot.write([make_obs(v, 1.0, NOW - h * HOUR) for v in ALL_VARIABLES[:6] for h in range(20)])
    assert list(redis_.hashes) == [hot.key(CELL)]
    assert len(redis_.hashes[hot.key(CELL)]) == 6  # one field per (variable, source)


def test_clear_only_removes_this_stores_prefix() -> None:
    redis_ = FakeRedis()
    a = HotStore(redis_, clock=FakeClock(NOW), prefix="a:")
    b = HotStore(redis_, clock=FakeClock(NOW), prefix="b:")
    a.write([make_obs()])
    b.write([make_obs()])
    assert a.clear() == 1
    assert a.cells_present() == [] and b.cells_present() == [CELL]


def test_memory_usage_reports_none_for_absent_cell() -> None:
    hot, _, _ = new_store()
    assert hot.memory_usage(CELL) is None
    hot.write([make_obs()])
    assert (hot.memory_usage(CELL) or 0) > 0


# ── tee sink ─────────────────────────────────────────────────────────


class Recorder:
    def __init__(self, name: str, log: list[str], fail: bool = False) -> None:
        self.name, self._log, self._fail = name, log, fail

    def write(self, observations: Any) -> tuple[int, int]:
        self._log.append(self.name)
        if self._fail:
            raise RuntimeError("redis down")
        return len(observations), 0


def test_tee_writes_primary_first_and_returns_its_result() -> None:
    log: list[str] = []
    tee = TeeSink(Recorder("cold", log), Recorder("hot", log))
    assert tee.write([make_obs(), make_obs(value=2.0)]) == (2, 0)
    assert log == ["cold", "hot"]
    assert tee.name == "cold+hot"


def test_tee_secondary_failure_propagates_after_the_primary_is_safe() -> None:
    log: list[str] = []
    tee = TeeSink(Recorder("cold", log), Recorder("hot", log, fail=True))
    with pytest.raises(RuntimeError):
        tee.write([make_obs()])
    assert log == ["cold", "hot"]


# ── 7. Latency reporting ─────────────────────────────────────────────


def test_latency_report_has_average_and_percentiles() -> None:
    report = summarize_latencies([float(x) for x in range(1, 101)])
    assert report.count == 100
    assert report.mean_ms == pytest.approx(50.5)
    assert report.p50_ms == pytest.approx(50.5)
    assert report.p95_ms == pytest.approx(95.05)
    assert report.p99_ms == pytest.approx(99.01)
    assert "p95=95.050 ms" in report.format("hot read")


def test_latency_report_of_nothing_is_zeros() -> None:
    assert summarize_latencies([]).count == 0
    assert WINDOW_HOURS["7d"] == 168
