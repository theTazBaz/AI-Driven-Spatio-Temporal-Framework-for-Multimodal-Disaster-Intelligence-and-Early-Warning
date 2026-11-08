"""Unit tests for Increment R6 — Real GloFAS and USGS Adapters."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from disaster_intel.adapters.base import ParseError
from disaster_intel.adapters.gdacs import GDACSAdapter, parse_gdacs_xml
from disaster_intel.adapters.glofas import GloFASAdapter
from disaster_intel.adapters.usgs import USGSAdapter
from disaster_intel.discretizers import PolygonDiscretizer
from disaster_intel.models import CAPSeverity, PointGeometry, PolygonGeometry


@pytest.mark.unit
class TestGloFASAdapter:
    """Verify Copernicus GloFAS river discharge parsing."""

    def test_parse_valid_glofas_payload(self) -> None:
        adapter = GloFASAdapter()
        payload = {
            "latitude": 14.81,
            "longitude": 74.13,
            "daily": {
                "time": ["2024-07-18", "2024-07-19", "2024-07-20"],
                "river_discharge": [14.24, 15.22, 15.48],
            },
        }
        readings = adapter.parse_payload(payload)
        assert len(readings) == 3

        r0 = readings[0]
        assert r0.source == "glofas"
        assert r0.variable == "discharge"
        assert r0.value == pytest.approx(14.24)
        assert r0.unit == "m3/s"
        assert r0.timestamp == datetime(2024, 7, 18, 12, 0, tzinfo=UTC)

    def test_skips_null_or_nan_discharge(self) -> None:
        adapter = GloFASAdapter()
        payload = {
            "latitude": 14.81,
            "longitude": 74.13,
            "daily": {
                "time": ["2024-07-18", "2024-07-19", "2024-07-20"],
                "river_discharge": [14.24, None, float("nan")],
            },
        }
        readings = adapter.parse_payload(payload)
        assert len(readings) == 1
        assert readings[0].value == pytest.approx(14.24)
        assert adapter.metrics.records_skipped["null_discharge"] == 2

    def test_rejects_non_dict_payload(self) -> None:
        adapter = GloFASAdapter()
        with pytest.raises(ParseError):
            adapter.parse_payload(["not", "a", "dict"])


@pytest.mark.unit
class TestUSGSAdapter:
    """Verify USGS earthquake GeoJSON parsing."""

    def test_parse_valid_usgs_geojson(self) -> None:
        adapter = USGSAdapter()
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "mag": 5.0,
                        "place": "49 km NE of Mulugu, India",
                        "time": 1701388800000,
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [80.3296, 18.4471, 10.0],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {
                        "mag": 4.4,
                        "place": "22 km NNW of Anjangaon, India",
                        "time": 1701475200000,
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [77.2016, 21.343, 10.0],
                    },
                },
            ],
        }
        readings = adapter.parse_payload(payload)
        assert len(readings) == 2

        r0 = readings[0]
        assert r0.source == "usgs"
        assert r0.variable == "seismic_magnitude"
        assert r0.value == pytest.approx(5.0)
        assert r0.unit == "Mw"
        assert isinstance(r0.geometry, PointGeometry)
        assert r0.geometry.lat == pytest.approx(18.4471)
        assert r0.geometry.lon == pytest.approx(80.3296)

    def test_skips_events_with_missing_mag_or_coords(self) -> None:
        adapter = USGSAdapter()
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "properties": {"mag": None, "time": 1701388800000},
                    "geometry": {"coordinates": [80.0, 15.0, 5.0]},
                },
                {
                    "properties": {"mag": 3.8, "time": 1701388800000},
                    "geometry": {"coordinates": []},
                },
            ],
        }
        readings = adapter.parse_payload(payload)
        assert len(readings) == 0
        assert adapter.metrics.records_skipped["null_magnitude"] == 1
        assert adapter.metrics.records_skipped["missing_coordinates"] == 1

    def test_rejects_malformed_payload(self) -> None:
        adapter = USGSAdapter()
        with pytest.raises(ParseError):
            adapter.parse_payload("malformed_json_string")


@pytest.mark.unit
class TestGDACSAdapter:
    """Verify GDACS civil alert RSS/XML feed parsing and discretization."""

    SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:gdacs="http://www.gdacs.org" xmlns:georss="http://www.georss.org/georss">
  <channel>
    <title>GDACS Alerts</title>
    <item>
      <title>Green flood alert in India</title>
      <guid>FL1104248</guid>
      <pubDate>Wed, 28 Oct 2026 12:25:05 GMT</pubDate>
      <gdacs:country>India</gdacs:country>
      <gdacs:eventtype>FL</gdacs:eventtype>
      <gdacs:alertlevel>Green</gdacs:alertlevel>
      <gdacs:alertscore>1.0</gdacs:alertscore>
      <georss:point>15.445 80.166</georss:point>
      <gdacs:bbox>76.166 84.166 11.445 19.445</gdacs:bbox>
    </item>
    <item>
      <title>Red notification for tropical cyclone</title>
      <guid>TC1001346</guid>
      <pubDate>Sun, 01 Nov 2026 02:23:45 GMT</pubDate>
      <gdacs:country>India</gdacs:country>
      <gdacs:eventtype>TC</gdacs:eventtype>
      <gdacs:alertlevel>Red</gdacs:alertlevel>
      <gdacs:alertscore>3.0</gdacs:alertscore>
      <georss:point>15.8 80.2</georss:point>
      <gdacs:bbox>69.407 95.407 0.670 26.670</gdacs:bbox>
    </item>
  </channel>
</rss>
"""

    def test_parse_gdacs_xml_structure(self) -> None:
        alerts = parse_gdacs_xml(self.SAMPLE_XML, country_filter="India")
        assert len(alerts) == 2

        a0 = alerts[0]
        assert a0.alert_id == "FL1104248"
        assert a0.event_type == "FL"
        assert a0.alert_level == "Green"
        assert a0.alert_score == pytest.approx(1.0)
        assert a0.point is not None
        assert a0.point[0] == pytest.approx(15.445)
        assert a0.point[1] == pytest.approx(80.166)
        assert a0.bbox == (76.166, 84.166, 11.445, 19.445)

        geom = a0.to_geometry()
        assert isinstance(geom, PolygonGeometry)
        assert len(geom.vertices) == 5

    def test_adapter_emits_cap_severity_readings(self) -> None:
        adapter = GDACSAdapter()
        readings = adapter.parse_payload(self.SAMPLE_XML)
        assert len(readings) == 2

        r_flood = readings[0]
        assert r_flood.source == "gdacs"
        assert r_flood.variable == "cap_severity"
        assert r_flood.value == float(CAPSeverity.MINOR.value)
        assert r_flood.unit == "ordinal"

        r_cyclone = readings[1]
        assert r_cyclone.value == float(CAPSeverity.EXTREME.value)

    def test_polygon_discretization_of_gdacs_alert(self) -> None:
        adapter = GDACSAdapter()
        discretizer = PolygonDiscretizer()
        readings = adapter.parse_payload(self.SAMPLE_XML)

        observations = discretizer.discretize(readings[0])
        assert len(observations) > 0
        for obs in observations:
            assert obs.variable == "cap_severity"
            assert obs.value == float(CAPSeverity.MINOR.value)

    def test_rejects_malformed_xml(self) -> None:
        adapter = GDACSAdapter()
        with pytest.raises(ParseError):
            adapter.parse_payload("<broken>xml<not_closed>")


class TestGDACSLive:
    """Live test for GDACS RSS alerts."""

    @pytest.mark.live
    def test_fetch_live_gdacs_alerts(self) -> None:
        adapter = GDACSAdapter()
        # Fetch global or India alerts live
        readings = adapter.fetch_live_alerts(country=None)
        assert len(readings) > 0
        assert all(r.source == "gdacs" for r in readings)
        assert all(r.variable == "cap_severity" for r in readings)
