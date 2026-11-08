"""Increment R6 — USGS Earthquake Hazards Seismic Adapter.

Ingests real seismic hypocenters and moment magnitudes (Mw) from the USGS
Earthquake Hazards API. Discretizes earthquake epicenters onto Western Ghats H3 cells.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import httpx

from disaster_intel.adapters.base import (
    AdapterMetrics,
    ParseError,
    SourceAdapter,
    SourceError,
)
from disaster_intel.clock import Clock, SystemClock
from disaster_intel.models import PointGeometry, RawReading


class USGSAdapter(SourceAdapter):
    """Source adapter for USGS seismic hazard event feeds."""

    source_name: str = "usgs"
    URL: str = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def __init__(
        self,
        clock: Clock | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.clock = clock or SystemClock()
        self._client = client
        self.metrics = AdapterMetrics()

    def fetch_events(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        min_magnitude: float = 2.5,
    ) -> list[RawReading]:
        """Fetch seismic events within bbox (west, south, east, north) and date range."""
        west, south, east, north = bbox
        params = {
            "format": "geojson",
            "starttime": start_date,
            "endtime": end_date,
            "minmagnitude": str(min_magnitude),
            "minlatitude": str(south),
            "maxlatitude": str(north),
            "minlongitude": str(west),
            "maxlongitude": str(east),
        }
        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=30.0)

        try:
            self.metrics.requests_made += 1
            try:
                resp = client.get(self.URL, params=params)
            except httpx.TransportError as exc:
                self.metrics.retries += 1
                raise SourceError(f"USGS network error: {exc}") from exc

            if resp.status_code != 200:
                raise SourceError(f"USGS HTTP error {resp.status_code}: {resp.text}")

            return self.parse_payload(resp.json())
        finally:
            if owns_client:
                client.close()

    def parse_payload(self, payload: Any) -> list[RawReading]:
        """Parse USGS GeoJSON payload into RawReading records."""
        if not isinstance(payload, dict):
            self.metrics.record_parse_error()
            raise ParseError("Expected GeoJSON FeatureCollection dict")

        features = payload.get("features", [])
        readings: list[RawReading] = []

        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [])

            if len(coords) < 2:
                self.metrics.record_skip("missing_coordinates")
                continue

            lon = float(coords[0])
            lat = float(coords[1])
            mag = props.get("mag")
            time_epoch_ms = props.get("time")

            if mag is None or not math.isfinite(mag):
                self.metrics.record_skip("null_magnitude")
                continue
            if time_epoch_ms is None:
                self.metrics.record_skip("null_time")
                continue

            ts = datetime.fromtimestamp(time_epoch_ms / 1000.0, tz=UTC)
            pt = PointGeometry(lat=lat, lon=lon)

            readings.append(
                RawReading(
                    source=self.source_name,
                    variable="seismic_magnitude",
                    value=float(mag),
                    timestamp=ts,
                    geometry=pt,
                    unit="Mw",
                )
            )

        self.metrics.readings_produced += len(readings)
        return readings
