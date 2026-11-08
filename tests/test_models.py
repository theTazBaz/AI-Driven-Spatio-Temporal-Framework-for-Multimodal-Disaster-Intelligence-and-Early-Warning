"""Increment 2 tests — Data Model: RawReading, Observation, and VariableCatalog.

TDD Test Suite (from plan.md):
1. Naive datetimes rejected.
2. NaN and out-of-range coordinates rejected.
3. Everything normalised to UTC.
4. Unit conversions (K->C, mm/hr->mm).
5. Stable idempotency key.
6. JSON round-trip identical.
7. Catalog covers all required variables.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timezone

import pytest

from disaster_intel.models import (
    AggMethod,
    CAPSeverity,
    GridNodeGeometry,
    Observation,
    PointGeometry,
    PolygonGeometry,
    RawReading,
    VariableCatalog,
)

# ── Validation tests ────────────────────────────────────────────────


@pytest.mark.unit
class TestRawReadingValidation:
    """RawReading must reject invalid inputs at construction time."""

    def test_naive_datetime_rejected(self) -> None:
        """A datetime without tzinfo must raise ValueError."""
        naive_dt = datetime(2024, 7, 15, 12, 0, 0)
        with pytest.raises(ValueError, match=r"[Nn]aive|timezone"):
            RawReading(
                source="open-meteo",
                variable="precipitation",
                value=5.2,
                timestamp=naive_dt,
                geometry=PointGeometry(lat=13.0, lon=74.8),
                unit="mm",
            )

    def test_nan_value_rejected(self) -> None:
        """NaN values must not be accepted."""
        with pytest.raises(ValueError, match=r"[Nn]a[Nn]|finite"):
            RawReading(
                source="open-meteo",
                variable="precipitation",
                value=float("nan"),
                timestamp=datetime(2024, 7, 15, 12, 0, 0, tzinfo=UTC),
                geometry=PointGeometry(lat=13.0, lon=74.8),
                unit="mm",
            )

    def test_inf_value_rejected(self) -> None:
        """Infinity values must not be accepted."""
        with pytest.raises(ValueError, match=r"finite|[Ii]nf"):
            RawReading(
                source="open-meteo",
                variable="temperature",
                value=float("inf"),
                timestamp=datetime(2024, 7, 15, 12, 0, 0, tzinfo=UTC),
                geometry=PointGeometry(lat=13.0, lon=74.8),
                unit="celsius",
            )

    def test_latitude_out_of_range_rejected(self) -> None:
        """Latitude outside [-90, 90] must raise ValueError."""
        with pytest.raises(ValueError, match=r"[Ll]at"):
            PointGeometry(lat=91.0, lon=74.8)

    def test_longitude_out_of_range_rejected(self) -> None:
        """Longitude outside [-180, 180] must raise ValueError."""
        with pytest.raises(ValueError, match=r"[Ll]on"):
            PointGeometry(lat=13.0, lon=181.0)

    def test_negative_latitude_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"[Ll]at"):
            PointGeometry(lat=-91.0, lon=0.0)

    def test_negative_longitude_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"[Ll]on"):
            PointGeometry(lat=0.0, lon=-181.0)

    def test_empty_source_rejected(self) -> None:
        """Source must be a non-empty string."""
        with pytest.raises(ValueError, match=r"[Ss]ource"):
            RawReading(
                source="",
                variable="precipitation",
                value=5.0,
                timestamp=datetime(2024, 7, 15, 12, 0, 0, tzinfo=UTC),
                geometry=PointGeometry(lat=13.0, lon=74.8),
                unit="mm",
            )

    def test_empty_variable_rejected(self) -> None:
        """Variable name must be a non-empty string."""
        with pytest.raises(ValueError, match=r"[Vv]ariable"):
            RawReading(
                source="open-meteo",
                variable="",
                value=5.0,
                timestamp=datetime(2024, 7, 15, 12, 0, 0, tzinfo=UTC),
                geometry=PointGeometry(lat=13.0, lon=74.8),
                unit="mm",
            )


# ── UTC normalisation ───────────────────────────────────────────────


@pytest.mark.unit
class TestUTCNormalisation:
    """All timestamps must be stored as UTC regardless of input timezone."""

    def test_utc_passthrough(self) -> None:
        utc_dt = datetime(2024, 7, 15, 12, 0, 0, tzinfo=UTC)
        reading = RawReading(
            source="open-meteo",
            variable="precipitation",
            value=5.0,
            timestamp=utc_dt,
            geometry=PointGeometry(lat=13.0, lon=74.8),
            unit="mm",
        )
        assert reading.timestamp.tzinfo == UTC

    def test_ist_converted_to_utc(self) -> None:
        """IST (UTC+5:30) should be normalised to UTC."""
        from datetime import timedelta

        ist = timezone(timedelta(hours=5, minutes=30))
        ist_dt = datetime(2024, 7, 15, 17, 30, 0, tzinfo=ist)  # 12:00 UTC
        reading = RawReading(
            source="open-meteo",
            variable="precipitation",
            value=5.0,
            timestamp=ist_dt,
            geometry=PointGeometry(lat=13.0, lon=74.8),
            unit="mm",
        )
        assert reading.timestamp.tzinfo == UTC
        assert reading.timestamp.hour == 12
        assert reading.timestamp.minute == 0


# ── Unit conversions ────────────────────────────────────────────────


@pytest.mark.unit
class TestUnitConversions:
    """Observation must normalise physical units to canonical form."""

    def test_kelvin_to_celsius(self) -> None:
        """300K should convert to 26.85C."""
        obs = Observation.from_raw(
            cell="8760112eaffffff",
            source="open-meteo",
            variable="temperature",
            raw_value=300.0,
            raw_unit="kelvin",
            timestamp=datetime(2024, 7, 15, 12, 0, 0, tzinfo=UTC),
        )
        assert abs(obs.value - 26.85) < 0.01

    def test_celsius_passthrough(self) -> None:
        """Celsius values should pass through unchanged."""
        obs = Observation.from_raw(
            cell="8760112eaffffff",
            source="open-meteo",
            variable="temperature",
            raw_value=26.85,
            raw_unit="celsius",
            timestamp=datetime(2024, 7, 15, 12, 0, 0, tzinfo=UTC),
        )
        assert abs(obs.value - 26.85) < 0.01

    def test_mm_per_hour_to_mm(self) -> None:
        """10 mm/hr over 1 hour -> 10 mm accumulated."""
        obs = Observation.from_raw(
            cell="8760112eaffffff",
            source="open-meteo",
            variable="precipitation",
            raw_value=10.0,
            raw_unit="mm/hr",
            timestamp=datetime(2024, 7, 15, 12, 0, 0, tzinfo=UTC),
            interval_hours=1.0,
        )
        assert abs(obs.value - 10.0) < 0.01

    def test_mm_passthrough(self) -> None:
        """Already in mm, should pass through."""
        obs = Observation.from_raw(
            cell="8760112eaffffff",
            source="open-meteo",
            variable="precipitation",
            raw_value=5.2,
            raw_unit="mm",
            timestamp=datetime(2024, 7, 15, 12, 0, 0, tzinfo=UTC),
        )
        assert abs(obs.value - 5.2) < 0.01


# ── Idempotency key ────────────────────────────────────────────────


@pytest.mark.unit
class TestIdempotencyKey:
    """Observation.idempotency_key must be stable and deterministic."""

    def _make_obs(self) -> Observation:
        return Observation.from_raw(
            cell="8760112eaffffff",
            source="open-meteo",
            variable="temperature",
            raw_value=26.85,
            raw_unit="celsius",
            timestamp=datetime(2024, 7, 15, 12, 0, 0, tzinfo=UTC),
        )

    def test_key_is_stable(self) -> None:
        """Same inputs must always produce the same key."""
        obs1 = self._make_obs()
        obs2 = self._make_obs()
        assert obs1.idempotency_key == obs2.idempotency_key

    def test_key_is_string(self) -> None:
        obs = self._make_obs()
        assert isinstance(obs.idempotency_key, str)
        assert len(obs.idempotency_key) > 0

    def test_different_cell_different_key(self) -> None:
        """Different cells must yield different keys."""
        obs1 = self._make_obs()
        obs2 = Observation.from_raw(
            cell="87601128dffffff",
            source="open-meteo",
            variable="temperature",
            raw_value=26.85,
            raw_unit="celsius",
            timestamp=datetime(2024, 7, 15, 12, 0, 0, tzinfo=UTC),
        )
        assert obs1.idempotency_key != obs2.idempotency_key

    def test_different_time_different_key(self) -> None:
        obs1 = self._make_obs()
        obs2 = Observation.from_raw(
            cell="8760112eaffffff",
            source="open-meteo",
            variable="temperature",
            raw_value=26.85,
            raw_unit="celsius",
            timestamp=datetime(2024, 7, 15, 13, 0, 0, tzinfo=UTC),
        )
        assert obs1.idempotency_key != obs2.idempotency_key


# ── JSON round-trip ─────────────────────────────────────────────────


@pytest.mark.unit
class TestJSONRoundTrip:
    """Observation -> JSON -> Observation must be lossless."""

    def test_observation_json_round_trip(self) -> None:
        obs = Observation.from_raw(
            cell="8760112eaffffff",
            source="open-meteo",
            variable="temperature",
            raw_value=26.85,
            raw_unit="celsius",
            timestamp=datetime(2024, 7, 15, 12, 0, 0, tzinfo=UTC),
        )
        json_str = obs.to_json()
        json.loads(json_str)
        restored = Observation.from_json(json_str)

        assert restored.cell == obs.cell
        assert restored.source == obs.source
        assert restored.variable == obs.variable
        assert abs(restored.value - obs.value) < 1e-9
        assert restored.timestamp == obs.timestamp
        assert restored.idempotency_key == obs.idempotency_key

    def test_raw_reading_json_round_trip_point(self) -> None:
        reading = RawReading(
            source="usgs",
            variable="seismic_magnitude",
            value=4.5,
            timestamp=datetime(2024, 7, 15, 12, 0, 0, tzinfo=UTC),
            geometry=PointGeometry(lat=13.0, lon=74.8),
            unit="richter",
        )
        json_str = reading.to_json()
        restored = RawReading.from_json(json_str)
        assert restored.source == reading.source
        assert restored.variable == reading.variable
        assert abs(restored.value - reading.value) < 1e-9
        assert restored.timestamp == reading.timestamp
        assert isinstance(restored.geometry, PointGeometry)
        assert abs(restored.geometry.lat - 13.0) < 1e-9


# ── Geometry types ──────────────────────────────────────────────────


@pytest.mark.unit
class TestGeometryTypes:
    """Validate the three geometry types: Point, GridNode, Polygon."""

    def test_point_geometry(self) -> None:
        pt = PointGeometry(lat=13.0108, lon=74.7943)
        assert pt.kind == "point"
        assert abs(pt.lat - 13.0108) < 1e-9

    def test_grid_node_geometry(self) -> None:
        gn = GridNodeGeometry(lat=13.0, lon=74.8, resolution_deg=0.1)
        assert gn.kind == "grid_node"
        assert abs(gn.resolution_deg - 0.1) < 1e-9

    def test_polygon_geometry(self) -> None:
        coords = [(12.8, 74.8), (13.0, 74.8), (13.0, 75.0), (12.8, 75.0)]
        pg = PolygonGeometry(vertices=coords)
        assert pg.kind == "polygon"
        assert len(pg.vertices) == 4

    def test_polygon_minimum_3_vertices(self) -> None:
        with pytest.raises(ValueError, match=r"[Vv]ert"):
            PolygonGeometry(vertices=[(0.0, 0.0), (1.0, 1.0)])


# ── Variable Catalog ────────────────────────────────────────────────


@pytest.mark.unit
class TestVariableCatalog:
    """VariableCatalog must cover all required variables from the plan."""

    REQUIRED_VARIABLES: tuple[str, ...] = (
        "precipitation",
        "soil_moisture_surface",
        "soil_moisture_rootzone",
        "temperature",
        "wind_speed",
        "pressure",
        "discharge",
        "seismic_magnitude",
        "cap_severity",
    )

    def test_catalog_covers_all_required_variables(self) -> None:
        catalog = VariableCatalog.default()
        for var_name in self.REQUIRED_VARIABLES:
            assert var_name in catalog, f"Missing variable: {var_name}"

    def test_each_variable_has_canonical_unit(self) -> None:
        catalog = VariableCatalog.default()
        for var_name in self.REQUIRED_VARIABLES:
            var = catalog[var_name]
            assert var.canonical_unit, f"{var_name} missing canonical_unit"

    def test_each_variable_has_aggregation_method(self) -> None:
        catalog = VariableCatalog.default()
        for var_name in self.REQUIRED_VARIABLES:
            var = catalog[var_name]
            assert isinstance(var.agg_method, AggMethod)

    def test_precipitation_is_sum(self) -> None:
        catalog = VariableCatalog.default()
        assert catalog["precipitation"].agg_method == AggMethod.SUM

    def test_temperature_is_avg(self) -> None:
        catalog = VariableCatalog.default()
        assert catalog["temperature"].agg_method == AggMethod.AVG

    def test_seismic_magnitude_is_max(self) -> None:
        catalog = VariableCatalog.default()
        assert catalog["seismic_magnitude"].agg_method == AggMethod.MAX

    def test_cap_severity_is_max(self) -> None:
        catalog = VariableCatalog.default()
        assert catalog["cap_severity"].agg_method == AggMethod.MAX

    def test_unknown_variable_raises_key_error(self) -> None:
        catalog = VariableCatalog.default()
        with pytest.raises(KeyError):
            _ = catalog["nonexistent_var"]


# ── CAP Severity ordinal ────────────────────────────────────────────


@pytest.mark.unit
class TestCAPSeverity:
    """CAP severity values must be ordered: Minor < Moderate < Severe < Extreme."""

    def test_severity_ordering(self) -> None:
        assert CAPSeverity.MINOR < CAPSeverity.MODERATE
        assert CAPSeverity.MODERATE < CAPSeverity.SEVERE
        assert CAPSeverity.SEVERE < CAPSeverity.EXTREME

    def test_severity_from_string(self) -> None:
        assert CAPSeverity.from_str("Minor") == CAPSeverity.MINOR
        assert CAPSeverity.from_str("extreme") == CAPSeverity.EXTREME


# ── Data Origin Provenance ──────────────────────────────────────────


@pytest.mark.unit
class TestDataOriginProvenance:
    """Verify data origin tagging and schema provenance tracking."""

    def test_inferred_data_origins(self) -> None:
        from disaster_intel.models import infer_data_origin

        assert infer_data_origin("open-meteo") == "REANALYSIS_ERA5"
        assert infer_data_origin("sentinel-1") == "SATELLITE_SAR"
        assert infer_data_origin("insat-3d") == "SATELLITE_INSAT"
        assert infer_data_origin("glofas") == "HYDROLOGY_GLOFAS"
        assert infer_data_origin("usgs") == "SEISMIC_USGS"
        assert infer_data_origin("gdacs") == "CIVIL_ALERT_GDACS"
        assert infer_data_origin("gsi") == "GROUND_TRUTH_GSI"
        assert infer_data_origin("synthetic") == "SYNTHETIC_STRESS_BURST"
        assert infer_data_origin("custom-station") == "REAL_IN_SITU"

    def test_explicit_origin_override(self) -> None:
        from disaster_intel.models import infer_data_origin

        assert infer_data_origin("open-meteo", "CUSTOM_ORIGIN") == "CUSTOM_ORIGIN"

    def test_observation_provenance_round_trip(self) -> None:
        obs = Observation.from_raw(
            cell="8760112eaffffff",
            source="sentinel-1",
            variable="sar_water_fraction",
            raw_value=0.45,
            raw_unit="fraction",
            timestamp=datetime(2024, 7, 13, 0, 48, 0, tzinfo=UTC),
        )
        assert obs.data_origin == "SATELLITE_SAR"

        d = obs.to_dict()
        assert d["data_origin"] == "SATELLITE_SAR"

        restored = Observation.from_dict(d)
        assert restored.data_origin == "SATELLITE_SAR"
