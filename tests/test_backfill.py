"""Increment 13 tests — Historical Backfill Mode and Phase 1 Evaluation Report.

TDD Test Suite (from plan.md lines 331-346 and Section 15):
1. Replaying same window twice -> identical cold-store contents (100% idempotency).
2. Replay runs well faster than real time (speedup > 100x).
3. Report contains every Obj 1 & 2 metric specified in Definition of Done.
4. Sanity check: 72h rainfall in monsoon window is well above quiet baseline.
5. Deterministic rollback: state at past time T equals live state computed at T.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from disaster_intel.backfill import (
    BENCHMARK_WINDOWS,
    BackfillEngine,
    generate_historical_spell,
)
from disaster_intel.clock import FakeClock
from disaster_intel.evaluation import (
    Phase1EvaluationReport,
    Phase1Evaluator,
)
from disaster_intel.pipeline import InMemorySink
from disaster_intel.state import (
    FunctionElevation,
    StateAssembler,
    build_static_layers,
)
from disaster_intel.storage import HotStore
from disaster_intel.windows import WindowValue, reference_windows
from fake_redis import FakeRedis

pytestmark = pytest.mark.unit

MADIKERI_CELL = "872a1072bffffff"
SHIRUR_CELL = "8760b526dffffff"


class TestHistoricalBackfill:
    """Verifies historical backfill mode mechanics, idempotency, and throughput."""

    def test_replaying_same_window_twice_gives_identical_cold_store_contents(self) -> None:
        """Replaying the same historical window twice must yield identical store contents.

        Second run must store 0 new records and drop 100% as duplicates.
        """
        sink = InMemorySink()
        engine = BackfillEngine(sink=sink)
        window = BENCHMARK_WINDOWS["shirur_landslide_2024"]

        # Run 1: initial backfill
        result1 = engine.run(window=window, cells=[SHIRUR_CELL])
        assert result1.records_stored > 0
        assert result1.duplicates_dropped == 0
        initial_count = sink.count()
        first_records = {k: v.value for k, v in sink._store.items()}

        # Run 2: identical repeat backfill
        result2 = engine.run(window=window, cells=[SHIRUR_CELL])
        assert result2.records_stored == 0
        assert result2.duplicates_dropped == result1.records_stored
        assert sink.count() == initial_count

        # Content must be 100% identical
        second_records = {k: v.value for k, v in sink._store.items()}
        assert first_records == second_records

    def test_replay_runs_well_faster_than_real_time(self) -> None:
        """A multi-day historical window must replay orders of magnitude faster than real time."""
        sink = InMemorySink()
        engine = BackfillEngine(sink=sink)
        window = BENCHMARK_WINDOWS["shirur_landslide_2024"]

        # 5 days = 120 simulated hours
        result = engine.run(window=window, cells=[SHIRUR_CELL, MADIKERI_CELL])
        assert result.wall_time_s >= 0.0
        # Simulated time is at least 96 hours; wall clock is under 5 seconds
        simulated_seconds = (window.end - window.start).total_seconds()
        assert simulated_seconds >= 4 * 86400
        assert result.speedup_factor > 100.0  # Must be well over 100x real-time speedup

    def test_heavy_monsoon_rainfall_well_above_quiet_period_baseline(self) -> None:
        """Sanity check: 72h rainfall in monsoon is far higher than quiet baseline."""
        monsoon_window = BENCHMARK_WINDOWS["shirur_landslide_2024"]
        quiet_window = BENCHMARK_WINDOWS["quiet_period_baseline"]

        monsoon_obs = generate_historical_spell(monsoon_window, [SHIRUR_CELL])
        quiet_obs = generate_historical_spell(quiet_window, [SHIRUR_CELL])

        monsoon_rain = sum(
            o.value for o in monsoon_obs if o.variable == "precipitation" and o.cell == SHIRUR_CELL
        )
        quiet_rain = sum(
            o.value for o in quiet_obs if o.variable == "precipitation" and o.cell == SHIRUR_CELL
        )

        # Monsoonal rainfall during Shirur landslide should be massive (>150 mm)
        # while quiet winter period in January should be near 0 mm
        assert monsoon_rain > 150.0
        assert quiet_rain < 10.0
        assert monsoon_rain > (quiet_rain * 20.0 + 50.0)


class TestPhase1EvaluationReport:
    """Verifies Phase 1 evaluation report generation and metric completeness."""

    def test_report_contains_every_obj_1_and_obj_2_metric(self) -> None:
        """The evaluation report must contain all metrics from plan.md Section 15."""
        evaluator = Phase1Evaluator()
        report: Phase1EvaluationReport = evaluator.evaluate()

        # Objective 1 metrics
        assert report.ingestion_p50_ms >= 0.0
        assert report.ingestion_p95_ms >= 0.0
        assert report.ingestion_p99_ms >= 0.0
        assert report.ingestion_completion_rate >= 99.0
        assert hasattr(report, "source_errors")
        assert hasattr(report, "parse_errors")
        assert report.source_errors == 0
        assert report.parse_errors == 0

        # Objective 2 metrics
        assert report.hot_p95_read_ms >= 0.0
        assert report.hot_p95_write_ms >= 0.0
        assert report.hot_p95_read_ms < 5.0  # Phase 1 target: < 5 ms
        assert report.cold_tier_correctness_pct == 100.0
        assert report.window_correctness_pct == 100.0
        assert report.deterministic_rollback_agrees is True

        # Storage budgets
        assert report.embedding_bytes_per_cell <= 64
        assert report.redis_kb_per_cell <= 8.0
        assert 24 <= report.bus_retention_hours <= 72
        assert report.raw_scene_cleanup_verified is True

        # Text representations
        summary = report.format_table()
        assert "OBJECTIVE 1" in summary
        assert "OBJECTIVE 2" in summary
        assert "STORAGE BUDGETS" in summary
        assert "Deterministic rollback" in summary

    def test_deterministic_rollback_during_historical_replay(self) -> None:
        """State at past time T recomputed after further simulation equals live state at T."""
        clock = FakeClock(datetime(2024, 7, 14, tzinfo=UTC))
        hot = HotStore(FakeRedis(), clock=clock)

        class MockCold:
            def __init__(self) -> None:
                self.observations: list[Any] = []

            def write(self, obs: list[Any]) -> None:
                self.observations.extend(obs)

            def windows(
                self,
                cell: str,
                as_of: datetime,
                source: str | None = None,
                window_hours: Mapping[str, int] | None = None,
            ) -> dict[str, dict[str, WindowValue]]:
                from disaster_intel.models import VariableCatalog

                cat = VariableCatalog.default()
                if window_hours is None:
                    return reference_windows(self.observations, cell, as_of, cat, source=source)
                return reference_windows(
                    self.observations,
                    cell,
                    as_of,
                    cat,
                    source=source,
                    window_hours=window_hours,
                )

        cold = MockCold()
        static = build_static_layers([SHIRUR_CELL], FunctionElevation(lambda lat, lon: 34.0))
        assembler = StateAssembler(static, hot=hot, cold=cold, clock=clock)

        window = BENCHMARK_WINDOWS["shirur_landslide_2024"]
        obs_all = generate_historical_spell(window, [SHIRUR_CELL])

        # Step 1: Feed up to halfway through the spell
        cutoff = window.start + (window.end - window.start) / 2
        batch1 = [o for o in obs_all if o.timestamp <= cutoff]
        batch2 = [o for o in obs_all if o.timestamp > cutoff]

        clock.advance_to(cutoff)
        hot.write(batch1)
        cold.write(batch1)

        live_state_at_t = assembler.get_state(SHIRUR_CELL)

        # Step 2: Feed the rest of the spell into future
        clock.advance_to(window.end)
        hot.write(batch2)
        cold.write(batch2)

        # Step 3: Rollback query at past T
        past_state = assembler.get_state(SHIRUR_CELL, as_of=cutoff)

        assert past_state.tier == "cold"
        assert past_state.same_values(live_state_at_t, rel_tol=1e-4, abs_tol=1e-4)
