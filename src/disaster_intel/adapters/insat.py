"""Increment 16 — ISRO INSAT-3D/3DR Geostationary Rapid-Scan Ingestion.

Provides:
- Ingestion of 15-minute geostationary thermal infrared (TIR-1, ~10.8 µm) readings.
- Cloud-top brightness temperature (Tb) in Kelvin.
- Cold-top convective storm threshold detection (Tb <= 210 K / -63°C).
- Convective cooling rate (dT_B / dt <= -10 K/15min) as early flash flood / cloudburst indicator.
- Stateful tracking across sequential 15-minute scans.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from disaster_intel.adapters.base import (
    AdapterMetrics,
    ParseError,
    SourceAdapter,
)
from disaster_intel.clock import Clock, SystemClock
from disaster_intel.grid import CoastalGrid
from disaster_intel.models import PointGeometry, RawReading

COLD_TOP_THRESHOLD_K: float = 210.0
RAPID_COOLING_THRESHOLD_K_PER_15MIN: float = -10.0
MIN_VALID_TEMPERATURE_K: float = 140.0
MAX_VALID_TEMPERATURE_K: float = 350.0


def celsius_to_kelvin(celsius: float) -> float:
    """Convert Celsius temperature to Kelvin."""
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert Kelvin temperature to Celsius."""
    return kelvin - 273.15


def is_deep_convection(tb_kelvin: float, threshold_k: float = COLD_TOP_THRESHOLD_K) -> bool:
    """Check if cloud-top temperature indicates severe deep convection (cumulonimbus)."""
    return tb_kelvin <= threshold_k


def is_rapid_cooling(
    rate_k_per_15min: float,
    threshold: float = RAPID_COOLING_THRESHOLD_K_PER_15MIN,
) -> bool:
    """Check if cooling rate indicates rapid convective updraft intensification."""
    return rate_k_per_15min <= threshold


def calculate_cooling_rate(
    tb_current: float,
    tb_previous: float,
    dt_minutes: float = 15.0,
) -> float:
    """Calculate normalized convective cooling rate in Kelvin per 15 minutes.

    Formula:
        rate = (tb_current - tb_previous) * (15.0 / dt_minutes)
    """
    if dt_minutes <= 0.0:
        raise ValueError(f"Time delta must be positive, got {dt_minutes}")
    return (tb_current - tb_previous) * (15.0 / dt_minutes)


@dataclass(frozen=True, slots=True)
class INSATReading:
    """Geo-located INSAT rapid-scan infrared observation."""

    lat: float
    lon: float
    cloud_top_tb: float
    previous_tb: float | None = None
    cooling_rate: float | None = None


class INSATAdapter(SourceAdapter):
    """Source adapter for ISRO INSAT-3D/3DR geostationary meteorological rapid-scans."""

    source_name: str = "insat-3d"
    resolution_deg: float = 0.04  # ~4 km at nadir

    def __init__(
        self,
        grid: CoastalGrid | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.grid = grid or CoastalGrid(use_cache=True)
        self.clock = clock or SystemClock()
        self.metrics = AdapterMetrics()
        # Stateful history: (lat_round, lon_round) -> (timestamp, tb_k)
        self._history: dict[tuple[float, float], tuple[datetime, float]] = {}

    def parse_payload(self, payload: Any) -> list[RawReading]:
        """Parse raw INSAT rapid-scan JSON payload into domain RawReading objects."""
        if not isinstance(payload, dict):
            self.metrics.record_parse_error()
            raise ParseError("Payload must be a JSON dictionary")

        raw_ts = payload.get("timestamp")
        if not raw_ts or not isinstance(raw_ts, str):
            self.metrics.record_parse_error()
            raise ParseError("Payload missing valid ISO-8601 'timestamp' field")

        try:
            ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
        except Exception as e:
            self.metrics.record_parse_error()
            raise ParseError(f"Invalid timestamp format '{raw_ts}': {e}") from e

        readings_data = payload.get("readings") or payload.get("pixels") or payload.get("data")
        if readings_data is None:
            self.metrics.record_parse_error()
            raise ParseError("Payload missing 'readings' or 'pixels' array")

        if not isinstance(readings_data, list) or len(readings_data) == 0:
            self.metrics.record_parse_error()
            raise ParseError("INSAT payload contains empty or invalid readings list")

        interval_min = float(payload.get("interval_minutes", 15.0))
        readings: list[RawReading] = []

        for item in readings_data:
            if not isinstance(item, dict):
                self.metrics.record_skip("malformed_item")
                continue

            lat = item.get("lat")
            lon = item.get("lon")
            tb_raw = item.get("cloud_top_tb") or item.get("tb")

            if lat is None or lon is None or tb_raw is None:
                self.metrics.record_skip("missing_fields")
                continue

            try:
                p_lat = float(lat)
                p_lon = float(lon)
                p_tb = float(tb_raw)
            except (ValueError, TypeError):
                self.metrics.record_skip("invalid_types")
                continue

            if not (-90.0 <= p_lat <= 90.0 and -180.0 <= p_lon <= 180.0):
                self.metrics.record_skip("invalid_coordinate")
                continue

            if not math.isfinite(p_tb):
                self.metrics.record_skip("invalid_value")
                continue

            # Convert to Kelvin based on explicit unit if provided, defaulting to Kelvin
            unit_str = str(item.get("unit", "K")).strip().lower()
            tb_k = celsius_to_kelvin(p_tb) if unit_str in ("celsius", "c", "°c") else p_tb

            # Validate physical temperature range for terrestrial troposphere (140 K - 350 K)
            if not (MIN_VALID_TEMPERATURE_K <= tb_k <= MAX_VALID_TEMPERATURE_K):
                self.metrics.record_skip("unphysical_temperature")
                continue

            # Compute convective cooling rate
            coord_key = (round(p_lat, 4), round(p_lon, 4))
            cooling_rate: float

            if "cooling_rate" in item and item["cooling_rate"] is not None:
                cooling_rate = float(item["cooling_rate"])
            elif "previous_tb" in item and item["previous_tb"] is not None:
                prev_tb = float(item["previous_tb"])
                prev_k = celsius_to_kelvin(prev_tb) if prev_tb <= 60.0 else prev_tb
                cooling_rate = calculate_cooling_rate(tb_k, prev_k, interval_min)
            elif coord_key in self._history:
                prev_ts, prev_k = self._history[coord_key]
                dt_sec = (ts - prev_ts).total_seconds()
                dt_m = dt_sec / 60.0
                if 5.0 <= dt_m <= 60.0:
                    cooling_rate = calculate_cooling_rate(tb_k, prev_k, dt_m)
                else:
                    cooling_rate = 0.0
            else:
                cooling_rate = 0.0

            # Update scan history
            self._history[coord_key] = (ts, tb_k)

            geom = PointGeometry(lat=p_lat, lon=p_lon)

            # 1. Cloud-top brightness temperature
            readings.append(
                RawReading(
                    source=self.source_name,
                    variable="insat_cloud_top_tb",
                    value=tb_k,
                    timestamp=ts,
                    geometry=geom,
                    unit="K",
                )
            )

            # 2. Convective cooling rate
            readings.append(
                RawReading(
                    source=self.source_name,
                    variable="insat_convective_cooling_rate",
                    value=cooling_rate,
                    timestamp=ts,
                    geometry=geom,
                    unit="K/15min",
                )
            )

        self.metrics.readings_produced += len(readings)
        return readings
