"""Increment 6 — End-to-End Ingestion Pipeline and Objective 1 Metrics.

Connects:
Adapters -> SpatialDiscretizer -> Streaming Bus -> Observation Sink

Tracks Objective 1 metrics:
- Latency percentiles (p50, p95, p99 in milliseconds)
- Completion rate (% of attempted records successfully stored)
- Source vs parse failure attribution
- Idempotency deduplication rate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from disaster_intel.adapters.base import SourceAdapter
from disaster_intel.adapters.open_meteo import OpenMeteoAdapter
from disaster_intel.bus import Bus, InMemoryBus, get_bus
from disaster_intel.clock import Clock, FakeClock, SystemClock
from disaster_intel.discretizers import SpatialDiscretizer
from disaster_intel.grid import CoastalGrid
from disaster_intel.models import Observation, RawReading

if TYPE_CHECKING:
    from collections.abc import Sequence


def calculate_percentiles(values: Sequence[float], percentiles: Sequence[float]) -> list[float]:
    """Calculate percentiles using linear interpolation."""
    if not values:
        return [0.0] * len(percentiles)

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    results: list[float] = []

    for p in percentiles:
        if p <= 0:
            results.append(sorted_vals[0])
        elif p >= 100:
            results.append(sorted_vals[-1])
        else:
            k = (n - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                results.append(sorted_vals[int(k)])
            else:
                d0 = sorted_vals[int(f)] * (c - k)
                d1 = sorted_vals[int(c)] * (k - f)
                results.append(round(d0 + d1, 3))

    return results


class ObservationSink(Protocol):
    """Protocol for observation storage sinks (in-memory, ClickHouse, Redis)."""

    name: str

    def write(self, observations: Sequence[Observation]) -> tuple[int, int]:
        """Write observations to sink. Returns (stored_count, duplicates_count)."""
        ...

    def contains(self, idempotency_key: str) -> bool:
        """Check if observation with given key already exists."""
        ...

    def count(self) -> int:
        """Return total count of unique observations stored."""
        ...

    def get_by_cell(self, cell: str) -> list[Observation]:
        """Retrieve observations for an H3 cell."""
        ...


class InMemorySink:
    """In-memory observation sink enforcing idempotency deduplication."""

    name: str = "in_memory_sink"

    def __init__(self) -> None:
        self._store: dict[str, Observation] = {}
        self._cell_index: dict[str, list[Observation]] = defaultdict(list)

    def write(self, observations: Sequence[Observation]) -> tuple[int, int]:
        stored = 0
        duplicates = 0
        for obs in observations:
            if obs.idempotency_key in self._store:
                duplicates += 1
            else:
                self._store[obs.idempotency_key] = obs
                self._cell_index[obs.cell].append(obs)
                stored += 1
        return stored, duplicates

    def contains(self, idempotency_key: str) -> bool:
        return idempotency_key in self._store

    def count(self) -> int:
        return len(self._store)

    def get_by_cell(self, cell: str) -> list[Observation]:
        return list(self._cell_index.get(cell, []))

    def all_observations(self) -> list[Observation]:
        return list(self._store.values())


@dataclass
class IngestionMetrics:
    """Comprehensive metrics tracking Objective 1 performance and reliability."""

    raw_readings_received: int = 0
    observations_produced: int = 0
    observations_stored: int = 0
    duplicates_dropped: int = 0
    source_errors: int = 0
    parse_errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        if self.raw_readings_received == 0:
            return 100.0
        # Completion is calculated against raw readings processed
        successful = self.raw_readings_received - self.source_errors - self.parse_errors
        return max(0.0, min(100.0, round((successful / self.raw_readings_received) * 100.0, 2)))

    @property
    def has_failures(self) -> bool:
        return self.source_errors > 0 or self.parse_errors > 0

    @property
    def p50_latency_ms(self) -> float:
        return calculate_percentiles(self.latencies_ms, [50])[0]

    @property
    def p95_latency_ms(self) -> float:
        return calculate_percentiles(self.latencies_ms, [95])[0]

    @property
    def p99_latency_ms(self) -> float:
        return calculate_percentiles(self.latencies_ms, [99])[0]

    def summary(self) -> str:
        p50, p95, p99 = calculate_percentiles(self.latencies_ms, [50, 95, 99])
        return (
            "\n"
            "+---------------------------------------------------------+\n"
            "|           OBJECTIVE 1 INGESTION METRICS SUMMARY         |\n"
            "+---------------------------------------------------------+\n"
            f"|  Raw Readings Received    : {self.raw_readings_received:<27} |\n"
            f"|  Observations Discretized : {self.observations_produced:<27} |\n"
            f"|  Observations Stored      : {self.observations_stored:<27} |\n"
            f"|  Duplicates Dropped       : {self.duplicates_dropped:<27} |\n"
            f"|  Completion Rate          : {str(self.completion_rate) + ' %':<27} |\n"
            f"|  Source Errors            : {self.source_errors:<27} |\n"
            f"|  Parse Errors             : {self.parse_errors:<27} |\n"
            "+---------------------------------------------------------+\n"
            f"|  Latency p50 (median)     : {str(p50) + ' ms':<27} |\n"
            f"|  Latency p95              : {str(p95) + ' ms':<27} |\n"
            f"|  Latency p99              : {str(p99) + ' ms':<27} |\n"
            "+---------------------------------------------------------+"
        )


class IngestionPipeline:
    """Streaming ingestion pipeline orchestrating ingestion, discretization, bus, and sink."""

    def __init__(
        self,
        adapter: SourceAdapter,
        discretizer: SpatialDiscretizer,
        bus: Bus,
        sink: ObservationSink,
        clock: Clock | None = None,
    ) -> None:
        self.adapter = adapter
        self.discretizer = discretizer
        self.bus = bus
        self.sink = sink
        self.clock = clock or SystemClock()
        self.metrics = IngestionMetrics()
        self.consumer_group = f"{sink.name}_pipeline"

    def process_raw_items(self, items: Sequence[Any]) -> IngestionMetrics:
        """Process a batch of raw readings or payloads through the pipeline."""
        self.metrics.raw_readings_received += len(items)
        now_dt = self.clock.now()
        published_topics: set[str] = set()

        for item in items:
            if not isinstance(item, RawReading):
                self.metrics.parse_errors += 1
                continue

            try:
                obs_list = self.discretizer.discretize(item)
            except Exception:
                self.metrics.parse_errors += 1
                continue

            self.metrics.observations_produced += len(obs_list)
            topic = f"telemetry.{item.source}"
            published_topics.add(topic)

            # Publish observations to bus
            for obs in obs_list:
                self.bus.publish(
                    topic=topic,
                    key=obs.cell,
                    payload=obs,
                )

        # Consume from bus and store in sink for all published topics
        if not published_topics:
            published_topics.add(f"telemetry.{self.adapter.source_name}")

        for topic in published_topics:
            self._drain_topic_to_sink(topic, now_dt)

        return self.metrics

    def run_once(self) -> IngestionMetrics:
        """Execute a single ingestion cycle using the configured adapter."""
        # Check if adapter has fetch_readings or direct fetch method
        readings = self.adapter.fetch_readings() if hasattr(self.adapter, "fetch_readings") else []

        return self.process_raw_items(readings)

    def _drain_topic_to_sink(self, topic: str, arrival_time: datetime) -> None:
        """Drain all pending messages from a bus topic into the sink."""
        while True:
            messages = self.bus.poll(
                topic=topic,
                group_id=self.consumer_group,
                max_messages=5000,
                auto_commit=True,
            )
            if not messages:
                break

            obs_batch: list[Observation] = []
            for msg in messages:
                if isinstance(msg.payload, Observation):
                    obs = msg.payload
                elif isinstance(msg.payload, dict):
                    obs = Observation.from_dict(msg.payload)
                else:
                    continue

                obs_batch.append(obs)
                # Track end-to-end ingestion latency in milliseconds
                latency = (arrival_time - obs.timestamp).total_seconds() * 1000.0
                self.metrics.latencies_ms.append(max(0.0, latency))

            stored, dups = self.sink.write(obs_batch)
            self.metrics.observations_stored += stored
            self.metrics.duplicates_dropped += dups


def run_replay_fixture(
    fixture_path: Path | str,
    grid: CoastalGrid | None = None,
    clock: Clock | None = None,
    bus: Bus | str | None = None,
    sink: ObservationSink | None = None,
) -> IngestionMetrics:
    """Replay a recorded JSON fixture through the complete pipeline."""
    target_grid = grid or CoastalGrid(use_cache=True)
    path = Path(fixture_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fixture file not found: {path}")

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    temp_clock = clock or SystemClock()
    adapter = OpenMeteoAdapter(clock=temp_clock)
    raw_readings = adapter.parse_payload(payload)

    if clock is None:
        if raw_readings:
            max_ts = max(r.timestamp for r in raw_readings)
            clock = FakeClock(max_ts + timedelta(seconds=1.0))
        else:
            clock = SystemClock()

    if isinstance(bus, Bus):
        active_bus = bus
    elif isinstance(bus, str):
        active_bus = get_bus(bus)
    else:
        active_bus = InMemoryBus()

    active_sink = sink or InMemorySink()
    discretizer = SpatialDiscretizer(grid=target_grid)

    pipeline = IngestionPipeline(
        adapter=adapter,
        discretizer=discretizer,
        bus=active_bus,
        sink=active_sink,
        clock=clock,
    )

    metrics = pipeline.process_raw_items(raw_readings)
    return metrics


def main() -> None:
    """CLI entry point for streaming ingestion and fixture replay."""
    parser = argparse.ArgumentParser(
        description="AI-Driven Multimodal Disaster Intelligence Ingestion Pipeline",
    )
    parser.add_argument(
        "--replay",
        type=str,
        help="Path to recorded JSON fixture to replay offline through the pipeline",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live ingestion against Open-Meteo API for study region",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=10,
        help="Max unique weather grid nodes to query in live mode (default: 10)",
    )
    parser.add_argument(
        "--bus",
        choices=["memory", "redpanda"],
        default=None,
        help="Streaming bus to use (default: $DISASTER_INTEL_BUS or memory)",
    )

    args = parser.parse_args()

    if args.replay:
        print(f"Replaying fixture: {args.replay}...")
        metrics = run_replay_fixture(args.replay, bus=args.bus)
        print(metrics.summary())
    elif args.live:
        print("Starting live ingestion cycle...")
        grid = CoastalGrid(use_cache=True)
        discretizer = SpatialDiscretizer(grid=grid)
        bus = get_bus(args.bus) if args.bus else InMemoryBus()
        sink = InMemorySink()
        clock = SystemClock()
        adapter = OpenMeteoAdapter(clock=clock)

        pipeline = IngestionPipeline(
            adapter=adapter,
            discretizer=discretizer,
            bus=bus,
            sink=sink,
            clock=clock,
        )

        nodes_to_query = list(discretizer.fan_out.unique_nodes[: args.max_nodes])
        print(f"Fetching live weather for {len(nodes_to_query)} grid nodes...")
        readings = adapter.fetch_grid_nodes(coords=nodes_to_query, past_days=1, forecast_days=0)
        print(f"Retrieved {len(readings)} live raw readings. Ingesting through pipeline...")

        metrics = pipeline.process_raw_items(readings)
        print(metrics.summary())
        print(f"\nSuccessfully stored {sink.count()} unique observations in sink.")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
