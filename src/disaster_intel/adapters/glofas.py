"""Increment R6 — Copernicus GloFAS River Discharge Adapter.

Ingests real physical river discharge (m^3/s) from the ECMWF Copernicus Emergency
Management Service (EMS) Global Flood Awareness System (GloFAS) via Open-Meteo Flood API.
Provides hydrological streamflow telemetry across Western Ghats river basins.
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


class GloFASAdapter(SourceAdapter):
    """Source adapter for Copernicus GloFAS river discharge telemetry."""

    source_name: str = "glofas"
    URL: str = "https://flood-api.open-meteo.com/v1/flood"

    def __init__(
        self,
        clock: Clock | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.clock = clock or SystemClock()
        self._client = client
        self.metrics = AdapterMetrics()

    def fetch_discharge(
        self,
        coords: list[tuple[float, float]],
        start_date: str,
        end_date: str,
    ) -> list[RawReading]:
        """Fetch real daily river discharge (m^3/s) for coordinates."""
        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=30.0)
        readings: list[RawReading] = []

        try:
            for lat, lon in coords:
                params: dict[str, str | float] = {
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "river_discharge",
                    "start_date": start_date,
                    "end_date": end_date,
                }
                self.metrics.requests_made += 1
                try:
                    resp = client.get(self.URL, params=params)
                except httpx.TransportError as exc:
                    self.metrics.retries += 1
                    raise SourceError(f"GloFAS network error: {exc}") from exc

                if resp.status_code != 200:
                    raise SourceError(f"GloFAS HTTP error {resp.status_code}: {resp.text}")

                payload = resp.json()
                readings.extend(self._parse_location(payload, lat, lon))
            return readings
        finally:
            if owns_client:
                client.close()

    def _parse_location(self, data: dict[str, Any], lat: float, lon: float) -> list[RawReading]:
        daily = data.get("daily", {})
        times = daily.get("time", [])
        discharges = daily.get("river_discharge", [])

        if len(times) != len(discharges):
            raise ParseError("GloFAS times and discharge arrays have mismatched lengths")

        geom = PointGeometry(lat=lat, lon=lon)
        out: list[RawReading] = []

        for t_str, q in zip(times, discharges, strict=True):
            if q is None or not math.isfinite(q):
                self.metrics.record_skip("null_discharge")
                continue

            ts = datetime.fromisoformat(f"{t_str}T12:00:00+00:00").replace(tzinfo=UTC)
            out.append(
                RawReading(
                    source=self.source_name,
                    variable="discharge",
                    value=float(q),
                    timestamp=ts,
                    geometry=geom,
                    unit="m3/s",
                )
            )

        self.metrics.readings_produced += len(out)
        return out

    def parse_payload(self, payload: Any) -> list[RawReading]:
        """Parse raw GloFAS API dictionary payload."""
        if not isinstance(payload, dict):
            self.metrics.record_parse_error()
            raise ParseError("Expected dict payload for GloFAS")
        lat = float(payload.get("latitude", 0.0))
        lon = float(payload.get("longitude", 0.0))
        return self._parse_location(payload, lat, lon)
