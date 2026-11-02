"""Increment 12 tests - RedpandaBus, Bus Switch, and Contract Verification.

TDD Test Suite (from plan.md lines 312-319):
1. Same bus contract suite passes unchanged on RedpandaBus.
2. Configuration switch: `bus: memory` <-> `bus: redpanda` via get_bus and env vars.
3. Killing and restarting a consumer loses nothing and duplicates are dropped by the sink.
4. Topic retention policy is strictly 24-72 hours (storage budget enforcement).
5. Message serialization preserves keys, headers, timestamps, and Observation types.
6. Throughput smoke test sustains expected message processing rate.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from disaster_intel.bus import (
    Bus,
    InMemoryBus,
    Message,
    RedpandaBus,
    get_bus,
)
from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.pipeline import InMemorySink, run_replay_fixture
from test_bus import BaseBusContractTests

pytestmark = pytest.mark.unit

CATALOG = VariableCatalog.default()
CELL = "8760b2494ffffff"
T0 = datetime(2024, 7, 10, 12, 0, tzinfo=UTC)


def make_observation(val: float = 12.5, offset_sec: int = 0) -> Observation:
    return Observation.from_raw(
        cell=CELL,
        source="open-meteo",
        variable="precipitation",
        raw_value=val,
        raw_unit="mm",
        timestamp=T0 + timedelta(seconds=offset_sec),
    )


# ── 1. Bus Switch & Configuration Tests ──────────────────────────────


def test_bus_switch_returns_memory_bus() -> None:
    bus = get_bus("memory")
    assert isinstance(bus, InMemoryBus)


def test_bus_switch_returns_redpanda_bus() -> None:
    bus = get_bus("redpanda", test_mode=True)
    assert isinstance(bus, RedpandaBus)


def test_bus_switch_via_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISASTER_INTEL_BUS", "memory")
    assert isinstance(get_bus(), InMemoryBus)

    monkeypatch.setenv("DISASTER_INTEL_BUS", "redpanda")
    bus = get_bus(test_mode=True)
    assert isinstance(bus, RedpandaBus)


def test_bus_switch_rejects_unknown_bus_type() -> None:
    with pytest.raises(ValueError, match="Unknown bus type: 'nats'"):
        get_bus("nats")


# ── 2. Topic Retention Policy (24-72 hours) ──────────────────────────


def test_topic_retention_accepts_valid_range() -> None:
    bus_24 = RedpandaBus(retention_hours=24, test_mode=True)
    assert bus_24.retention_hours == 24
    assert bus_24.retention_ms == 24 * 3600 * 1000

    bus_72 = RedpandaBus(retention_hours=72, test_mode=True)
    assert bus_72.retention_hours == 72
    assert bus_72.retention_ms == 72 * 3600 * 1000


@pytest.mark.parametrize("invalid_hours", [0, 12, 23, 73, 168])
def test_topic_retention_rejects_out_of_budget_hours(invalid_hours: int) -> None:
    with pytest.raises(ValueError, match=r"retention.*24.*72"):
        RedpandaBus(retention_hours=invalid_hours, test_mode=True)


# ── 3. Serialization and Deserialization Fidelity ────────────────────


def test_message_serialization_roundtrip_observation() -> None:
    bus = RedpandaBus(test_mode=True)
    obs = make_observation(val=45.2, offset_sec=30)
    topic = "telemetry.open-meteo"

    offset = bus.publish(
        topic=topic,
        key=obs.cell,
        payload=obs,
        headers={"custom-header": "v1"},
    )
    assert offset == 0

    msgs = bus.poll(topic=topic, group_id="obs_verifier", max_messages=1)
    assert len(msgs) == 1
    m = msgs[0]

    assert m.topic == topic
    assert m.key == obs.cell
    assert m.offset == 0
    assert m.headers.get("custom-header") == "v1"

    # Reconstructed payload equals original Observation
    assert isinstance(m.payload, Observation)
    assert m.payload.cell == obs.cell
    assert m.payload.variable == obs.variable
    assert m.payload.value == obs.value
    assert m.payload.idempotency_key == obs.idempotency_key


def test_message_serialization_roundtrip_dict_and_primitives() -> None:
    bus = RedpandaBus(test_mode=True)
    topic = "telemetry.alerts"

    dict_payload = {"alert_id": "ALT-123", "severity": 4, "tags": ["flood", "warning"]}
    bus.publish(topic=topic, key="k1", payload=dict_payload)

    msgs = bus.poll(topic=topic, group_id="dict_verifier", max_messages=1)
    assert len(msgs) == 1
    assert msgs[0].payload == dict_payload


# ── 4. Consumer Restart, Crash Recovery & Deduplication ──────────────


def test_killing_and_restarting_consumer_loses_nothing() -> None:
    bus = RedpandaBus(test_mode=True)
    topic = "telemetry.stream"
    group = "sink_worker"
    sink = InMemorySink()

    # Publish 10 observations
    obs_list = [make_observation(val=float(i), offset_sec=i) for i in range(10)]
    for obs in obs_list:
        bus.publish(topic=topic, key=obs.cell, payload=obs)

    # Worker 1 reads 5 messages with auto_commit=False, processes, and commits
    worker1_msgs = bus.poll(topic=topic, group_id=group, max_messages=5, auto_commit=False)
    assert len(worker1_msgs) == 5
    stored, dups = sink.write([m.payload for m in worker1_msgs])
    assert stored == 5 and dups == 0
    # Commit up to offset 4
    bus.commit(topic=topic, group_id=group, offset=4)

    # Simulate crash of Worker 1 before it could read next batch.
    # Worker 2 starts up with the same group_id
    worker2_msgs = bus.poll(topic=topic, group_id=group, max_messages=10, auto_commit=True)
    assert len(worker2_msgs) == 5
    assert [m.offset for m in worker2_msgs] == [5, 6, 7, 8, 9]

    # Worker 2 writes remaining to sink
    stored2, dups2 = sink.write([m.payload for m in worker2_msgs])
    assert stored2 == 5 and dups2 == 0
    assert sink.count() == 10

    # If Worker 2 restarts and seeks back to offset 3 (simulating replay of batch)
    bus.seek(topic=topic, group_id=group, offset=3)
    replayed = bus.poll(topic=topic, group_id=group, max_messages=10)
    assert len(replayed) == 7  # offsets 3, 4, 5, 6, 7, 8, 9
    stored_replay, dups_replay = sink.write([m.payload for m in replayed])
    # All replayed observations are recognized as duplicates and safely dropped!
    assert stored_replay == 0
    assert dups_replay == 7
    assert sink.count() == 10


# ── 5. Throughput Smoke Test ─────────────────────────────────────────


def test_throughput_smoke_test_sustains_message_rate() -> None:
    bus = RedpandaBus(test_mode=True)
    topic = "telemetry.perf"
    n_messages = 2000

    messages = [(f"cell_{i % 50}", {"i": i, "val": i * 0.1}) for i in range(n_messages)]

    # Measure publish batch throughput
    t_start = time.perf_counter()
    offsets = bus.publish_batch(topic=topic, messages=messages)
    t_pub = time.perf_counter() - t_start

    assert len(offsets) == n_messages
    assert offsets[-1] == n_messages - 1
    pub_rate = n_messages / max(1e-6, t_pub)
    assert pub_rate > 1000.0, f"Publish throughput too low: {pub_rate:.1f} msg/sec"

    # Measure poll throughput
    t_start_poll = time.perf_counter()
    consumed: list[Message] = []
    while len(consumed) < n_messages:
        batch = bus.poll(topic=topic, group_id="perf_reader", max_messages=500)
        if not batch:
            break
        consumed.extend(batch)
    t_poll = time.perf_counter() - t_start_poll

    assert len(consumed) == n_messages
    poll_rate = n_messages / max(1e-6, t_poll)
    assert poll_rate > 1000.0, f"Poll throughput too low: {poll_rate:.1f} msg/sec"


# ── 6. Full Contract Suite Inheritance for RedpandaBus ───────────────


class TestRedpandaBusContract(BaseBusContractTests):
    """Verify that RedpandaBus satisfies all requirements of BaseBusContractTests."""

    def create_bus(self, max_queue_size: int | None = None) -> Bus:
        return RedpandaBus(max_queue_size=max_queue_size, test_mode=True)


def test_pipeline_replay_with_redpanda_bus() -> None:
    bus = RedpandaBus(test_mode=True)
    sink = InMemorySink()
    fixture = Path(__file__).parent / "fixtures" / "open_meteo_sample.json"
    metrics = run_replay_fixture(fixture_path=fixture, bus=bus, sink=sink)

    assert metrics.completion_rate == 100.0
    assert metrics.raw_readings_received == 24
    assert metrics.observations_stored == metrics.observations_produced
    assert sink.count() == metrics.observations_stored
