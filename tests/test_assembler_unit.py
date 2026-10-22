"""Increment 10 unit tests - StateAssembler (no network, no containers).

Test list (plan.md, Increment 10), assembler half:
 1. Vector length and order match the FeatureSpec; every value lands in its own slot.
 2. A missing sensor is NaN (never 0), with coverage 0.0; zero rain is 0.0, not NaN.
 3. compound saturation inside a real vector = soil_moisture_rootzone(24h) x precipitation(72h).
 4. ``get_state`` agrees between the hot tier (Redis, FakeRedis here) and the cold tier (a
    reference-oracle stand-in here; real ClickHouse in test_assembler_integration.py).
 5. Tier choice: now -> hot; the past -> cold; hot cannot answer the past; the future is refused.
 6. Deterministic rollback: the state at past T computed LATER equals the live state at T.
 7. A vector built under another spec cannot be loaded.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import pytest

from disaster_intel.clock import FakeClock
from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.state import (
    FeatureSpec,
    SpecMismatchError,
    StateAssembler,
    StateVector,
    StaticFeatures,
    StaticLayers,
)
from disaster_intel.state.spec import V1_STATIC, feature_name
from disaster_intel.storage import HotRangeError, HotStore, to_float32
from disaster_intel.windows import HOUR, WindowValue, reference_windows, variable_names
from fake_redis import FakeRedis

pytestmark = pytest.mark.unit

CELL = "872a1072bffffff"
OTHER_CELL = "872a10729ffffff"
NOW = datetime(2024, 7, 10, 12, 0, tzinfo=UTC)
CATALOG = VariableCatalog.default()
SPEC = FeatureSpec.v1()
STATIC = StaticLayers(
    {CELL: StaticFeatures(elevation_m=123.0, slope_deg=4.5, population_density_km2=250.0)}
)


def make_obs(variable: str, value: float, ts: datetime, cell: str = CELL) -> Observation:
    return Observation.from_raw(
        cell=cell,
        source="open-meteo",
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


class OracleCold:
    """Stands in for ColdStore: windows come straight from the slow oracle (windows.py)."""

    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def write(self, observations: Sequence[Observation]) -> None:
        self.observations.extend(as_stored(o) for o in observations)

    def windows(
        self,
        cell: str,
        as_of: datetime,
        source: str | None = None,
        window_hours: Mapping[str, int] | None = None,
    ) -> dict[str, dict[str, WindowValue]]:
        if window_hours is None:
            return reference_windows(self.observations, cell, as_of, CATALOG, source=source)
        return reference_windows(
            self.observations, cell, as_of, CATALOG, window_hours=window_hours, source=source
        )


class World:
    """A hot tier + cold tier fed identically, and an assembler over them."""

    def __init__(self, now: datetime = NOW, static: StaticLayers = STATIC) -> None:
        self.clock = FakeClock(now)
        self.hot = HotStore(FakeRedis(), clock=self.clock)
        self.cold = OracleCold()
        self.assembler = StateAssembler(static, hot=self.hot, cold=self.cold, clock=self.clock)

    def feed(self, observations: Sequence[Observation]) -> None:
        self.cold.write(observations)
        self.hot.write(observations)


def hourly(variable: str, value: float, hours: int, end: datetime = NOW) -> list[Observation]:
    return [make_obs(variable, value, end - i * HOUR) for i in range(hours)]


# ── 1. Layout ────────────────────────────────────────────────────────


def test_vector_length_and_order_match_the_spec() -> None:
    world = World()
    world.feed(hourly("precipitation", 1.0, 30))
    vector = world.assembler.get_state(CELL)
    assert len(vector) == len(SPEC) == 29
    assert len(vector.coverage) == 29
    assert (vector.spec_version, vector.spec_fingerprint) == (SPEC.version, SPEC.fingerprint)
    mapping = vector.as_mapping(SPEC)
    assert list(mapping) == list(SPEC.names)


def test_dynamic_values_land_in_their_own_slots_with_coverage() -> None:
    world = World()
    world.feed(hourly("precipitation", 1.0, 30))  # now, now-1h, ..., now-29h
    v = world.assembler.get_state(CELL).as_mapping(SPEC)
    assert v["precipitation__1h"] == 1.0
    assert v["precipitation__6h"] == 6.0
    assert v["precipitation__24h"] == 24.0
    assert v["precipitation__72h"] == 30.0  # only 30 hours exist
    assert v["precipitation__7d"] == 30.0
    vector = world.assembler.get_state(CELL)
    assert vector.coverage[SPEC.index("precipitation__72h")] == pytest.approx(30 / 72)
    assert vector.coverage[SPEC.index("precipitation__24h")] == 1.0


def test_static_values_are_filled_and_unknown_cells_get_nan() -> None:
    world = World()
    known = world.assembler.get_state(CELL).as_mapping(SPEC)
    assert (known["elevation_m"], known["slope_deg"], known["population_density_km2"]) == (
        123.0,
        4.5,
        250.0,
    )
    other = world.assembler.get_state(OTHER_CELL).as_mapping(SPEC)
    assert all(math.isnan(other[name]) for name in V1_STATIC)


# ── 2. Missing is NaN, zero is zero ──────────────────────────────────


def test_a_missing_sensor_is_nan_with_zero_coverage_never_zero() -> None:
    world = World()
    world.feed(hourly("precipitation", 1.0, 10))  # no discharge at all
    vector = world.assembler.get_state(CELL)
    for window in ("1h", "6h", "24h", "72h", "7d"):
        i = SPEC.index(feature_name("discharge", window))
        assert math.isnan(vector.values[i])
        assert vector.coverage[i] == 0.0


def test_zero_rain_is_a_measurement_not_missing() -> None:
    world = World()
    world.feed(hourly("precipitation", 0.0, 10))
    assert world.assembler.get_state(CELL).get(SPEC, "precipitation__24h") == 0.0


# ── 3. Compound saturation in a real vector ──────────────────────────


def test_compound_saturation_is_soil_moisture_times_rolling_rainfall() -> None:
    world = World()
    world.feed(hourly("precipitation", 1.0, 80) + hourly("soil_moisture_rootzone", 0.4, 80))
    vector = world.assembler.get_state(CELL)
    soil = vector.get(SPEC, "soil_moisture_rootzone__24h")
    rain = vector.get(SPEC, "precipitation__72h")
    assert soil == pytest.approx(0.4, rel=1e-6)
    assert rain == 72.0
    assert vector.get(SPEC, "compound_saturation") == pytest.approx(soil * rain)


def test_compound_is_zero_with_zero_rain_and_nan_without_rain_data() -> None:
    world = World()
    world.feed(hourly("precipitation", 0.0, 80) + hourly("soil_moisture_rootzone", 0.4, 80))
    assert world.assembler.get_state(CELL).get(SPEC, "compound_saturation") == 0.0

    soil_only = World()
    soil_only.feed(hourly("soil_moisture_rootzone", 0.4, 80))
    assert math.isnan(soil_only.assembler.get_state(CELL).get(SPEC, "compound_saturation"))


def test_compound_rises_as_rain_and_soil_moisture_rise() -> None:
    def compound(rain: float, soil: float) -> float:
        w = World()
        w.feed(hourly("precipitation", rain, 80) + hourly("soil_moisture_rootzone", soil, 80))
        return w.assembler.get_state(CELL).get(SPEC, "compound_saturation")

    assert compound(2.0, 0.3) > compound(1.0, 0.3)
    assert compound(1.0, 0.4) > compound(1.0, 0.3)


# ── 4. Hot and cold agree ────────────────────────────────────────────


def random_stream(seed: int, cells: Sequence[str] = (CELL,)) -> list[Observation]:
    rng = random.Random(seed)
    out: list[Observation] = []
    for cell in cells:
        for variable in variable_names(CATALOG):
            for hour in range(24 * 9):
                if rng.random() < 0.35:
                    continue
                ts = NOW - hour * HOUR
                if rng.random() < 0.3:
                    ts -= timedelta(minutes=rng.randrange(1, 60), microseconds=rng.randrange(5))
                value = rng.uniform(0, 80)
                if variable == "cap_severity":
                    value = float(rng.randint(1, 4))
                out.append(make_obs(variable, value, ts, cell))
    rng.shuffle(out)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_get_state_agrees_between_hot_and_cold(seed: int) -> None:
    world = World()
    world.feed(random_stream(seed, (CELL, OTHER_CELL)))
    for cell in (CELL, OTHER_CELL):
        hot = world.assembler.get_state(cell, tier="hot")
        cold = world.assembler.get_state(cell, tier="cold")
        assert (hot.tier, cold.tier) == ("hot", "cold")
        assert hot.same_values(cold, rel_tol=1e-9, abs_tol=1e-9)
        assert world.assembler.compare_tiers(cell).agrees


def test_compare_tiers_flags_a_wiped_hot_tier() -> None:
    world = World()
    world.feed(hourly("precipitation", 1.0, 30))
    assert world.assembler.compare_tiers(CELL).agrees
    world.hot.clear()  # what a Redis restart looks like
    report = world.assembler.compare_tiers(CELL)
    assert not report.agrees
    assert "precipitation__24h" in report.differing


# ── 5. Tier choice and guards ────────────────────────────────────────


def test_auto_uses_hot_for_the_current_hour_and_cold_for_anything_earlier() -> None:
    world = World()
    world.feed(hourly("precipitation", 1.0, 100))
    world.clock.advance(30 * 60)  # now = 12:30
    assert world.assembler.get_state(CELL).tier == "hot"
    assert world.assembler.get_state(CELL, NOW + timedelta(minutes=10)).tier == "hot"  # 12:10
    # The hot tier keeps exactly 7 days ending at the current hour; a window ending in the
    # previous hour would reach one hour too far back, so the cold tier answers.
    assert world.assembler.get_state(CELL, NOW - timedelta(minutes=10)).tier == "cold"
    assert world.assembler.get_state(CELL, NOW - 5 * HOUR).tier == "cold"


def test_hot_cannot_answer_the_past_and_does_not_guess() -> None:
    world = World()
    with pytest.raises(HotRangeError):
        world.assembler.get_state(CELL, NOW - 5 * HOUR, tier="hot")
    hot_only = StateAssembler(STATIC, hot=world.hot, clock=world.clock)
    with pytest.raises(HotRangeError):
        hot_only.get_state(CELL, NOW - 5 * HOUR)


def test_cold_only_assembler_always_uses_cold() -> None:
    world = World()
    world.feed(hourly("precipitation", 1.0, 30))
    cold_only = StateAssembler(STATIC, cold=world.cold, clock=world.clock)
    assert cold_only.get_state(CELL).tier == "cold"
    with pytest.raises(ValueError):
        cold_only.get_state(CELL, tier="hot")


def test_assembler_needs_at_least_one_tier() -> None:
    with pytest.raises(ValueError):
        StateAssembler(STATIC)


def test_future_naive_and_invalid_inputs_are_refused() -> None:
    world = World()
    with pytest.raises(ValueError, match="future"):
        world.assembler.get_state(CELL, NOW + HOUR)
    with pytest.raises(ValueError, match="Naive"):
        world.assembler.get_state(CELL, datetime(2024, 7, 10, 11, 0))
    with pytest.raises(ValueError, match="Invalid H3"):
        world.assembler.get_state("not-a-cell")


def test_get_states_uses_one_consistent_instant() -> None:
    world = World()
    world.feed(random_stream(9, (CELL, OTHER_CELL)))
    states = world.assembler.get_states([CELL, OTHER_CELL])
    assert {s.as_of for s in states} == {NOW}
    assert [s.cell for s in states] == [CELL, OTHER_CELL]


# ── 6. Deterministic rollback ────────────────────────────────────────


def test_state_at_past_time_computed_later_equals_the_live_state_at_that_time() -> None:
    world = World(now=NOW)
    world.feed([o for o in random_stream(21) if o.timestamp <= NOW])
    live_at_t = world.assembler.get_state(CELL)  # computed live, from the hot tier, at T
    assert live_at_t.tier == "hot"

    # 40 hours pass and newer data arrives; it must not leak into the state at T.
    world.clock.sleep(40 * 3600)
    world.feed(
        [
            make_obs(variable, 55.0, NOW + (i + 1) * HOUR)
            for i in range(40)
            for variable in ("precipitation", "discharge")
        ]
    )

    replay = world.assembler.get_state(CELL, as_of=NOW)  # computed today for past T
    assert replay.tier == "cold"
    assert replay.same_values(live_at_t, rel_tol=1e-9, abs_tol=1e-9)
    assert replay.to_json() != ""  # serialisable


# ── 7. Another spec cannot load this vector ──────────────────────────


def test_a_vector_from_another_spec_cannot_be_loaded() -> None:
    small = FeatureSpec(
        version=2,
        dynamic=(("precipitation", "1h"), ("precipitation", "24h")),
        static=V1_STATIC,
        derived=(),
        definitions={
            "precipitation__1h": "SUM precipitation 1h",
            "precipitation__24h": "SUM precipitation 24h",
            **{name: f"static layer {name}" for name in V1_STATIC},
        },
    )
    world = World()
    world.feed(hourly("precipitation", 1.0, 30))
    v2 = StateAssembler(STATIC, hot=world.hot, clock=world.clock, spec=small).get_state(CELL)
    assert len(v2) == 5
    assert StateVector.from_json(v2.to_json(), small).same_values(v2)
    with pytest.raises(SpecMismatchError):
        StateVector.from_json(v2.to_json(), SPEC)
    v1 = world.assembler.get_state(CELL)
    with pytest.raises(SpecMismatchError):
        StateVector.from_json(v1.to_json(), small)
