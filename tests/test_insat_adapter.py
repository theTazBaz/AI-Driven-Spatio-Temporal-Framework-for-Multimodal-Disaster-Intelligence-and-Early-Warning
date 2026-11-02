"""Increment 16 tests — ISRO INSAT-3D/3DR Geostationary Rapid-Scan Ingestion.

TDD Test Suite (plan.md Section 10.7):
1. Temperature conversions and radiometric physics (Kelvin <-> Celsius).
2. Cold-top convective threshold detection (Tb <= 210 K / -63°C for severe cumulonimbus).
3. Convective cooling rate calculation (dT_B / dt <= -10 K/15min for convective updrafts).
4. Stateful multi-scan tracking computes cooling rates across sequential 15-minute passes.
5. Error handling and attribution: corrupt coordinates or unphysical temperatures skipped.
6. Golden fixture parse: Recorded INSAT-3D rapid-scan fixture parses into RawReadings.
7. Pipeline integration: IngestionPipeline processes INSAT readings end-to-end to sink.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from disaster_intel.adapters.base import ParseError
from disaster_intel.adapters.insat import (
    COLD_TOP_THRESHOLD_K,
    RAPID_COOLING_THRESHOLD_K_PER_15MIN,
    INSATAdapter,
    calculate_cooling_rate,
    celsius_to_kelvin,
    is_deep_convection,
    is_rapid_cooling,
    kelvin_to_celsius,
)
from disaster_intel.bus import InMemoryBus
from disaster_intel.discretizers import SpatialDiscretizer
from disaster_intel.grid import CoastalGrid
from disaster_intel.models import VariableCatalog
from disaster_intel.pipeline import IngestionPipeline, InMemorySink

FIXTURES_DIR = Path(__file__).parent / "fixtures"
T0 = datetime(2024, 7, 16, 12, 0, tzinfo=UTC)


# ── 1. Temperature Conversions and Physics ───────────────────────────


@pytest.mark.unit
class TestTemperaturePhysics:
    """Verify temperature conversions and physical bounds."""

    def test_celsius_to_kelvin(self) -> None:
        assert celsius_to_kelvin(0.0) == pytest.approx(273.15, rel=1e-9)
        assert celsius_to_kelvin(-63.15) == pytest.approx(210.0, rel=1e-9)
        assert celsius_to_kelvin(25.0) == pytest.approx(298.15, rel=1e-9)

    def test_kelvin_to_celsius(self) -> None:
        assert kelvin_to_celsius(273.15) == pytest.approx(0.0, rel=1e-9)
        assert kelvin_to_celsius(210.0) == pytest.approx(-63.15, rel=1e-9)
        assert kelvin_to_celsius(300.0) == pytest.approx(26.85, rel=1e-9)


# ── 2. Cold-Top Convective Threshold Detection ──────────────────────


@pytest.mark.unit
class TestColdTopThreshold:
    """Verify detection of deep convective cloud tops (Tb <= 210 K)."""

    def test_deep_convection_detected(self) -> None:
        # Severe cumulonimbus storm tops: 195 K to 208 K
        assert is_deep_convection(195.0) is True
        assert is_deep_convection(205.0) is True
        assert is_deep_convection(209.5) is True

    def test_warm_surface_or_low_cloud_not_deep_convection(self) -> None:
        # Clear land/sea surface: ~295 K; low stratus: ~275 K
        assert is_deep_convection(295.0) is False
        assert is_deep_convection(275.0) is False
        assert is_deep_convection(230.0) is False

    def test_boundary_conditions(self) -> None:
        assert is_deep_convection(COLD_TOP_THRESHOLD_K) is True
        assert is_deep_convection(COLD_TOP_THRESHOLD_K + 0.1) is False


# ── 3. Convective Cooling Rate Calculation ──────────────────────────


@pytest.mark.unit
class TestConvectiveCoolingRate:
    """Verify convective cooling rate (dT_B / dt) calculation."""

    def test_rapid_cooling_detected(self) -> None:
        # 15 K drop in 15 minutes = -15 K/15min <= -10 K/15min
        rate = calculate_cooling_rate(tb_current=210.0, tb_previous=225.0, dt_minutes=15.0)
        assert rate == pytest.approx(-15.0, rel=1e-9)
        assert is_rapid_cooling(rate) is True

    def test_gradual_change_not_rapid_cooling(self) -> None:
        # 2 K drop in 15 minutes = -2 K/15min > -10 K/15min
        rate = calculate_cooling_rate(tb_current=220.0, tb_previous=222.0, dt_minutes=15.0)
        assert rate == pytest.approx(-2.0, rel=1e-9)
        assert is_rapid_cooling(rate) is False

    def test_rate_normalized_for_different_scan_intervals(self) -> None:
        # 20 K drop over 30 minutes = -10 K/15min equivalent
        rate = calculate_cooling_rate(tb_current=210.0, tb_previous=230.0, dt_minutes=30.0)
        assert rate == pytest.approx(-10.0, rel=1e-9)
        assert is_rapid_cooling(rate) is True

    def test_threshold_boundary(self) -> None:
        assert is_rapid_cooling(RAPID_COOLING_THRESHOLD_K_PER_15MIN) is True
        assert is_rapid_cooling(RAPID_COOLING_THRESHOLD_K_PER_15MIN + 0.1) is False


# ── 4. Stateful Multi-Scan Tracking ──────────────────────────────────


@pytest.mark.unit
class TestStatefulScanTracking:
    """Verify that INSATAdapter computes cooling rates across sequential scans."""

    def test_sequential_scans_compute_cooling_rate(self) -> None:
        adapter = INSATAdapter()
        coord_lat, coord_lon = 13.0, 74.8

        # Scan 1 at T0: Cloud top at 235 K
        scan1 = {
            "timestamp": T0.isoformat(),
            "readings": [{"lat": coord_lat, "lon": coord_lon, "cloud_top_tb": 235.0}],
        }
        readings1 = adapter.parse_payload(scan1)
        # Scan 1 has no previous scan -> cooling rate is 0.0
        tb_r1 = next(r for r in readings1 if r.variable == "insat_cloud_top_tb")
        cr_r1 = next(r for r in readings1 if r.variable == "insat_convective_cooling_rate")
        assert tb_r1.value == 235.0
        assert cr_r1.value == 0.0

        # Scan 2 at T0 + 15 min: Rapid cloud-top ascent, Tb drops to 215 K (-20 K)
        scan2 = {
            "timestamp": (T0 + timedelta(minutes=15)).isoformat(),
            "readings": [{"lat": coord_lat, "lon": coord_lon, "cloud_top_tb": 215.0}],
        }
        readings2 = adapter.parse_payload(scan2)
        tb_r2 = next(r for r in readings2 if r.variable == "insat_cloud_top_tb")
        cr_r2 = next(r for r in readings2 if r.variable == "insat_convective_cooling_rate")
        assert tb_r2.value == 215.0
        assert cr_r2.value == pytest.approx(-20.0, rel=1e-5)


# ── 5. Error Handling and Attribution ────────────────────────────────


@pytest.mark.unit
class TestINSATErrorHandling:
    """Verify graceful handling of missing or corrupted data."""

    def test_missing_timestamp_raises_parse_error(self) -> None:
        adapter = INSATAdapter()
        payload: dict[str, Any] = {"readings": [{"lat": 13.0, "lon": 74.8, "cloud_top_tb": 220.0}]}
        with pytest.raises(ParseError, match=r"[Tt]imestamp"):
            adapter.parse_payload(payload)
        assert adapter.metrics.parse_errors == 1

    def test_missing_readings_raises_parse_error(self) -> None:
        adapter = INSATAdapter()
        payload = {"timestamp": T0.isoformat()}
        with pytest.raises(ParseError, match=r"[Rr]eadings|[Pp]ixels"):
            adapter.parse_payload(payload)
        assert adapter.metrics.parse_errors == 1

    def test_unphysical_temperatures_skipped_with_counter(self) -> None:
        adapter = INSATAdapter()
        payload = {
            "timestamp": T0.isoformat(),
            "readings": [
                {"lat": 13.0, "lon": 74.8, "cloud_top_tb": 220.0},  # valid
                {"lat": 13.0, "lon": 74.8, "cloud_top_tb": 50.0},  # unphysically cold (< 140 K)
                {"lat": 13.0, "lon": 74.8, "cloud_top_tb": 450.0},  # unphysically hot (> 350 K)
                {"lat": 13.0, "lon": 74.8, "cloud_top_tb": float("nan")},  # NaN
            ],
        }
        readings = adapter.parse_payload(payload)
        assert len(readings) == 2  # 1 Tb + 1 cooling_rate for the single valid reading
        assert adapter.metrics.records_skipped["unphysical_temperature"] == 2
        assert adapter.metrics.records_skipped["invalid_value"] == 1


# ── 6. Golden Fixture Parse ──────────────────────────────────────────


@pytest.mark.unit
class TestINSATGoldenParse:
    """Verify that recorded INSAT-3D rapid-scan fixture parses into RawReadings."""

    def test_golden_fixture_parse(self) -> None:
        fixture_path = FIXTURES_DIR / "insat_sample.json"
        raw_payload = json.loads(fixture_path.read_text(encoding="utf-8"))

        adapter = INSATAdapter()
        readings = adapter.parse_payload(raw_payload)

        assert len(readings) > 0
        variables = {r.variable for r in readings}
        assert variables == {"insat_cloud_top_tb", "insat_convective_cooling_rate"}

        for r in readings:
            assert r.source == "insat-3d"
            assert r.timestamp.tzinfo == UTC
            if r.variable == "insat_cloud_top_tb":
                assert 150.0 <= r.value <= 330.0
                assert r.unit == "K"
            elif r.variable == "insat_convective_cooling_rate":
                assert -60.0 <= r.value <= 60.0
                assert r.unit == "K/15min"


# ── 7. Pipeline Integration ──────────────────────────────────────────


@pytest.mark.unit
class TestINSATPipelineIntegration:
    """Verify that INSAT data flows through IngestionPipeline into sink."""

    def test_pipeline_ingests_insat_to_sink(self) -> None:
        grid = CoastalGrid(use_cache=True)
        discretizer = SpatialDiscretizer(grid=grid)
        bus = InMemoryBus()
        sink = InMemorySink()
        adapter = INSATAdapter(grid=grid)

        fixture_path = FIXTURES_DIR / "insat_sample.json"
        raw_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        readings = adapter.parse_payload(raw_payload)

        pipeline = IngestionPipeline(
            adapter=adapter,
            discretizer=discretizer,
            bus=bus,
            sink=sink,
        )

        metrics = pipeline.process_raw_items(readings)
        assert metrics.completion_rate == 100.0
        assert metrics.raw_readings_received == len(readings)
        assert metrics.observations_stored == len(readings)
        assert sink.count() == len(readings)

        # Verify catalog registration
        catalog = VariableCatalog.default()
        for cell in grid.cells:
            for obs in sink.get_by_cell(cell):
                assert obs.variable in catalog
                assert obs.unit == catalog[obs.variable].canonical_unit
