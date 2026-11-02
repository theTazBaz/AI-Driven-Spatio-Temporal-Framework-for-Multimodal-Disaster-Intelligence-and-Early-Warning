"""Increment 14 tests — Sentinel-1 C-SAR Radar Adapter and All-Weather Water Masking.

TDD Test Suite (plan.md Section 10.5):
1. Linear amplitude converted bit-exactly to decibels: sigma0 = 10 * log10(amplitude^2).
2. Synthetic radar patch correctly classifies calm open water (<= -18 dB) vs rough terrain.
3. Spatial pooling aggregates radar pixels into H3 Res-7 cells, calculating valid water fraction
   and mean backscatter.
4. Missing/corrupted polarizations attributed to SourceError or ParseError.
5. Golden fixture parse: Recorded Sentinel-1 GRD fixture parses into RawReadings.
6. End-to-end pipeline integration: Sentinel-1 -> Discretizer -> Bus -> Sink.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from disaster_intel.adapters.base import ParseError
from disaster_intel.adapters.sentinel1 import (
    Sentinel1Adapter,
    Sentinel1Pixel,
    amplitude_to_db,
    classify_water,
    db_to_amplitude,
)
from disaster_intel.bus import InMemoryBus
from disaster_intel.discretizers import SpatialDiscretizer
from disaster_intel.grid import CoastalGrid
from disaster_intel.models import VariableCatalog
from disaster_intel.pipeline import IngestionPipeline, InMemorySink

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MANGALORE_LAT = 12.9141
MANGALORE_LON = 74.8560
T0 = datetime(2024, 7, 16, 0, 45, tzinfo=UTC)


# ── 1. Radiometric Calibration (Amplitude -> dB) ─────────────────────


@pytest.mark.unit
class TestRadiometricCalibration:
    """Verify bit-exact linear amplitude to backscatter sigma0 (dB) conversion."""

    def test_amplitude_to_db_exact_values(self) -> None:
        assert amplitude_to_db(1.0) == 0.0
        assert amplitude_to_db(0.1) == pytest.approx(-20.0, rel=1e-9)
        assert amplitude_to_db(0.01) == pytest.approx(-40.0, rel=1e-9)
        assert amplitude_to_db(10.0) == pytest.approx(20.0, rel=1e-9)

    def test_amplitude_to_db_formula_identity(self) -> None:
        """sigma0 = 10 * log10(amplitude^2) == 20 * log10(amplitude)."""
        test_amplitudes = [0.05, 0.08, 0.1258925, 0.25, 0.5, 0.8, 1.2]
        for amp in test_amplitudes:
            from_power = 10.0 * math.log10(amp**2)
            from_function = amplitude_to_db(amp)
            assert from_function == pytest.approx(from_power, rel=1e-12)

    def test_water_threshold_boundary_amplitude(self) -> None:
        """The -18 dB threshold corresponds to amplitude 10^(-18/20) ~ 0.12589254."""
        boundary_amp = 10.0 ** (-18.0 / 20.0)
        assert amplitude_to_db(boundary_amp) == pytest.approx(-18.0, rel=1e-9)

    def test_non_positive_amplitude_raises_or_returns_neg_inf(self) -> None:
        with pytest.raises(ValueError, match=r"[Pp]ositive|[Nn]on-positive|[Gg]reater than zero"):
            amplitude_to_db(0.0)
        with pytest.raises(ValueError, match=r"[Pp]ositive|[Nn]on-positive|[Gg]reater than zero"):
            amplitude_to_db(-0.5)

    def test_db_to_amplitude_round_trip(self) -> None:
        for db in [-30.0, -24.5, -18.0, -12.0, -6.0, 0.0, 3.5]:
            amp = db_to_amplitude(db)
            assert amplitude_to_db(amp) == pytest.approx(db, rel=1e-9)


# ── 2. Specular Water Thresholding ───────────────────────────────────


@pytest.mark.unit
class TestWaterClassification:
    """Calm open water reflects radar specularly away (sigma0_VH <= -18 dB)."""

    def test_open_water_classified_correctly(self) -> None:
        # Typical open water backscatter: -20 dB to -26 dB
        assert classify_water(-24.0) is True
        assert classify_water(-20.5) is True
        assert classify_water(-18.1) is True

    def test_rough_vegetated_terrain_classified_as_land(self) -> None:
        # Volumetric vegetation scatter: -8 dB to -14 dB
        assert classify_water(-14.0) is False
        assert classify_water(-10.0) is False
        assert classify_water(-7.5) is False

    def test_threshold_boundary(self) -> None:
        assert classify_water(-18.0) is True
        assert classify_water(-17.999) is False

    def test_custom_threshold(self) -> None:
        assert classify_water(-16.0, threshold_db=-15.0) is True
        assert classify_water(-16.0, threshold_db=-17.0) is False


# ── 3. Spatial Pooling into H3 Res-7 Cells ───────────────────────────


@pytest.mark.unit
class TestSpatialPooling:
    """Verify pooling radar pixels into H3 Res-7 cells."""

    def test_aggregate_pixels_computes_fraction_and_mean_backscatter(self) -> None:
        grid = CoastalGrid(use_cache=True)
        adapter = Sentinel1Adapter(grid=grid)

        # Generate 10 synthetic pixels around Mangalore inside a single H3 Res-7 cell
        cell = grid.latlng_to_cell(MANGALORE_LAT, MANGALORE_LON)
        center_lat, center_lon = grid.cell_to_latlng(cell)

        # 6 water pixels (-22 dB -> amp ~ 0.0794) and 4 land pixels (-10 dB -> amp ~ 0.3162)
        pixels = [
            Sentinel1Pixel(
                lat=center_lat + 0.001 * i, lon=center_lon + 0.001 * i, vh_amplitude=0.0794
            )
            for i in range(6)
        ] + [
            Sentinel1Pixel(
                lat=center_lat - 0.001 * i, lon=center_lon - 0.001 * i, vh_amplitude=0.3162
            )
            for i in range(1, 5)
        ]

        aggregates = adapter.aggregate_pixels(pixels, timestamp=T0)
        assert len(aggregates) == 1
        agg = aggregates[0]
        assert agg.cell == cell
        assert agg.pixel_count == 10
        assert agg.water_pixel_count == 6
        assert agg.water_fraction == pytest.approx(0.6, rel=1e-5)
        # Expected mean backscatter: (6 * -22.0024 + 4 * -10.0017) / 10 = -17.2021 dB
        expected_mean = (6 * amplitude_to_db(0.0794) + 4 * amplitude_to_db(0.3162)) / 10.0
        assert agg.mean_vh_db == pytest.approx(expected_mean, rel=1e-3)


# ── 4. Error Handling and Attribution ────────────────────────────────


@pytest.mark.unit
class TestSentinel1ErrorHandling:
    """Verify corrupt or missing data is handled gracefully and attributed correctly."""

    def test_missing_timestamp_raises_parse_error(self) -> None:
        adapter = Sentinel1Adapter()
        payload: dict[str, Any] = {
            "scene_id": "S1_TEST",
            "pixels": [{"lat": 13.0, "lon": 74.8, "vh_amplitude": 0.1}],
        }
        with pytest.raises(ParseError, match=r"[Tt]imestamp"):
            adapter.parse_payload(payload)
        assert adapter.metrics.parse_errors == 1

    def test_missing_pixels_field_raises_parse_error(self) -> None:
        adapter = Sentinel1Adapter()
        payload = {"scene_id": "S1_TEST", "timestamp": "2024-07-16T00:45:00Z"}
        with pytest.raises(ParseError, match=r"[Pp]ixels|[Rr]eadings"):
            adapter.parse_payload(payload)
        assert adapter.metrics.parse_errors == 1

    def test_empty_pixels_list_raises_parse_error(self) -> None:
        adapter = Sentinel1Adapter()
        payload = {"scene_id": "S1_TEST", "timestamp": "2024-07-16T00:45:00Z", "pixels": []}
        with pytest.raises(ParseError, match=r"[Ee]mpty"):
            adapter.parse_payload(payload)
        assert adapter.metrics.parse_errors == 1

    def test_corrupt_pixel_skipped_with_counter(self) -> None:
        adapter = Sentinel1Adapter()
        payload = {
            "scene_id": "S1_TEST",
            "timestamp": "2024-07-16T00:45:00Z",
            "pixels": [
                {"lat": 13.0, "lon": 74.8, "vh_amplitude": 0.1},
                {"lat": 999.0, "lon": 74.8, "vh_amplitude": 0.1},  # invalid lat
                {"lat": 13.0, "lon": 74.8, "vh_amplitude": -0.5},  # negative amp
                {"lat": 13.0, "lon": 74.8, "vh_amplitude": float("nan")},  # NaN amp
            ],
        }
        readings = adapter.parse_payload(payload)
        assert len(readings) == 2  # 1 water_fraction + 1 vh_backscatter for the valid pixel
        assert adapter.metrics.records_skipped["invalid_coordinate"] == 1
        assert adapter.metrics.records_skipped["invalid_amplitude"] == 2


# ── 5. Golden Fixture Parse ──────────────────────────────────────────


@pytest.mark.unit
class TestSentinel1GoldenParse:
    """Verify that recorded Sentinel-1 GRD fixture parses into RawReadings."""

    def test_golden_fixture_parse(self) -> None:
        fixture_path = FIXTURES_DIR / "sentinel1_sample.json"
        raw_payload = json.loads(fixture_path.read_text(encoding="utf-8"))

        adapter = Sentinel1Adapter()
        readings = adapter.parse_payload(raw_payload)

        assert len(readings) > 0
        variables = {r.variable for r in readings}
        assert variables == {"sar_water_fraction", "sar_vh_backscatter_db"}

        for r in readings:
            assert r.source == "sentinel-1"
            assert r.timestamp.tzinfo == UTC
            if r.variable == "sar_water_fraction":
                assert 0.0 <= r.value <= 1.0
                assert r.unit == "fraction"
            elif r.variable == "sar_vh_backscatter_db":
                assert -50.0 <= r.value <= 10.0
                assert r.unit == "dB"


# ── 6. End-to-End Pipeline Integration ───────────────────────────────


@pytest.mark.unit
class TestSentinel1PipelineIntegration:
    """Verify that Sentinel-1 data flows end-to-end through IngestionPipeline."""

    def test_pipeline_ingests_sentinel1_to_sink(self) -> None:
        grid = CoastalGrid(use_cache=True)
        discretizer = SpatialDiscretizer(grid=grid)
        bus = InMemoryBus()
        sink = InMemorySink()
        adapter = Sentinel1Adapter(grid=grid)

        fixture_path = FIXTURES_DIR / "sentinel1_sample.json"
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

        # Verify observations in sink have canonical units in catalog
        catalog = VariableCatalog.default()
        for cell in grid.cells:
            cell_obs = sink.get_by_cell(cell)
            for obs in cell_obs:
                assert obs.variable in catalog
                assert obs.unit == catalog[obs.variable].canonical_unit
