"""Increment 14 — Sentinel-1 C-SAR Radar Adapter and All-Weather Water Masking.

Provides:
- Linear amplitude to decibel backscatter calibration:
  sigma0 = 10 * log10(amplitude^2) = 20 * log10(amplitude)
- Specular water thresholding: calm open water backscatter <= -18 dB
- Spatial pooling: aggregation of radar pixels into H3 Res-7 cells
- Adapter producing RawReadings for sar_water_fraction and sar_vh_backscatter_db
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from disaster_intel.adapters.base import (
    AdapterMetrics,
    ParseError,
    SourceAdapter,
    SourceError,
)
from disaster_intel.clock import Clock, SystemClock
from disaster_intel.grid import CoastalGrid
from disaster_intel.models import PointGeometry, RawReading

if TYPE_CHECKING:
    from collections.abc import Sequence

WATER_THRESHOLD_DB: float = -18.0


def amplitude_to_db(amplitude: float) -> float:
    """Convert linear radar amplitude to backscatter sigma0 in decibels (dB).

    Formula:
        sigma0 = 10 * log10(amplitude^2) = 20 * log10(amplitude)

    Raises:
        ValueError: If amplitude is non-positive or non-finite.
    """
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError(f"Amplitude must be a positive finite float, got {amplitude}")
    return 20.0 * math.log10(amplitude)


def db_to_amplitude(db: float) -> float:
    """Convert backscatter sigma0 in decibels (dB) back to linear amplitude.

    Formula:
        amplitude = 10^(db / 20)
    """
    if not math.isfinite(db):
        raise ValueError(f"dB value must be a finite float, got {db}")
    return float(10.0 ** (db / 20.0))


def classify_water(vh_db: float, threshold_db: float = WATER_THRESHOLD_DB) -> bool:
    """Classify a radar pixel as open water based on specular reflection.

    Calm open water reflects the C-band radar beam specularly away from the receiver,
    yielding very low cross-polarization backscatter (typically <= -18 dB).
    Rough or vegetated terrain produces diffuse scattering (typically > -18 dB).
    """
    return vh_db <= threshold_db


@dataclass(frozen=True, slots=True)
class Sentinel1Pixel:
    """Single geo-located radar pixel reading."""

    lat: float
    lon: float
    vh_amplitude: float
    vv_amplitude: float | None = None


@dataclass(frozen=True, slots=True)
class Sentinel1CellAggregate:
    """Aggregated radar statistics for an H3 Res-7 cell."""

    cell: str
    timestamp: datetime
    water_fraction: float
    mean_vh_db: float
    pixel_count: int
    water_pixel_count: int


class Sentinel1Adapter(SourceAdapter):
    """Source adapter for Sentinel-1 C-Band Synthetic Aperture Radar (SAR) GRD products."""

    source_name: str = "sentinel-1"

    def __init__(
        self,
        grid: CoastalGrid | None = None,
        water_threshold_db: float = WATER_THRESHOLD_DB,
        clock: Clock | None = None,
    ) -> None:
        self.grid = grid or CoastalGrid(use_cache=True)
        self.water_threshold_db = water_threshold_db
        self.clock = clock or SystemClock()
        self.metrics = AdapterMetrics()

    def aggregate_pixels(
        self,
        pixels: Sequence[Sentinel1Pixel],
        timestamp: datetime,
    ) -> list[Sentinel1CellAggregate]:
        """Group radar pixels by H3 Res-7 cell and compute water fraction and mean backscatter."""
        cell_pixels: dict[str, list[float]] = defaultdict(list)
        cell_water: dict[str, int] = defaultdict(int)

        for p in pixels:
            if not (-90.0 <= p.lat <= 90.0 and -180.0 <= p.lon <= 180.0):
                self.metrics.record_skip("invalid_coordinate")
                continue
            if not math.isfinite(p.vh_amplitude) or p.vh_amplitude <= 0.0:
                self.metrics.record_skip("invalid_amplitude")
                continue

            vh_db = amplitude_to_db(p.vh_amplitude)
            cell = self.grid.latlng_to_cell(p.lat, p.lon)
            cell_pixels[cell].append(vh_db)
            if classify_water(vh_db, self.water_threshold_db):
                cell_water[cell] += 1

        aggregates: list[Sentinel1CellAggregate] = []
        for cell, dbs in cell_pixels.items():
            count = len(dbs)
            water_count = cell_water[cell]
            water_frac = water_count / count if count > 0 else 0.0
            mean_db = sum(dbs) / count if count > 0 else 0.0

            aggregates.append(
                Sentinel1CellAggregate(
                    cell=cell,
                    timestamp=timestamp,
                    water_fraction=water_frac,
                    mean_vh_db=mean_db,
                    pixel_count=count,
                    water_pixel_count=water_count,
                )
            )

        return aggregates

    def parse_payload(self, payload: Any) -> list[RawReading]:
        """Parse raw Sentinel-1 scene JSON payload into domain RawReading objects."""
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

        pixel_data = payload.get("pixels") if "pixels" in payload else payload.get("readings")
        if pixel_data is None:
            self.metrics.record_parse_error()
            raise ParseError("Payload missing 'pixels' or 'readings' array")

        if not isinstance(pixel_data, list) or len(pixel_data) == 0:
            self.metrics.record_parse_error()
            raise ParseError("Sentinel-1 payload contains empty or invalid pixel list")

        parsed_pixels: list[Sentinel1Pixel] = []
        for item in pixel_data:
            if not isinstance(item, dict):
                self.metrics.record_skip("malformed_pixel")
                continue
            lat = item.get("lat")
            lon = item.get("lon")
            vh_amp = item.get("vh_amplitude")
            vv_amp = item.get("vv_amplitude")

            if lat is None or lon is None or vh_amp is None:
                self.metrics.record_skip("missing_pixel_fields")
                continue

            try:
                p_lat = float(lat)
                p_lon = float(lon)
                p_vh = float(vh_amp)
                p_vv = float(vv_amp) if vv_amp is not None else None
            except (ValueError, TypeError):
                self.metrics.record_skip("invalid_pixel_types")
                continue

            parsed_pixels.append(
                Sentinel1Pixel(
                    lat=p_lat,
                    lon=p_lon,
                    vh_amplitude=p_vh,
                    vv_amplitude=p_vv,
                )
            )

        if not parsed_pixels:
            self.metrics.record_parse_error()
            raise ParseError("No valid pixels found in Sentinel-1 payload")

        aggregates = self.aggregate_pixels(parsed_pixels, ts)

        readings: list[RawReading] = []
        for agg in aggregates:
            cell_lat, cell_lon = self.grid.cell_to_latlng(agg.cell)
            geom = PointGeometry(lat=cell_lat, lon=cell_lon)

            # 1. sar_water_fraction reading
            readings.append(
                RawReading(
                    source=self.source_name,
                    variable="sar_water_fraction",
                    value=agg.water_fraction,
                    timestamp=agg.timestamp,
                    geometry=geom,
                    unit="fraction",
                )
            )

            # 2. sar_vh_backscatter_db reading
            readings.append(
                RawReading(
                    source=self.source_name,
                    variable="sar_vh_backscatter_db",
                    value=agg.mean_vh_db,
                    timestamp=agg.timestamp,
                    geometry=geom,
                    unit="dB",
                )
            )

        self.metrics.readings_produced += len(readings)
        return readings

    @classmethod
    def query_cdse_catalog(
        cls,
        bbox: tuple[float, float, float, float],
        start_date: str | datetime,
        end_date: str | datetime,
        product_type: str = "S1A_IW_GRDH",
        limit: int = 10,
        timeout: float = 20.0,
    ) -> list[dict[str, Any]]:
        """Query Copernicus Data Space Ecosystem (CDSE) OData catalog for real Sentinel-1 orbits.

        Args:
            bbox: (min_lat, min_lon, max_lat, max_lon) in WGS84 degrees.
            start_date: Start ISO-8601 timestamp string or datetime.
            end_date: End ISO-8601 timestamp string or datetime.
            product_type: Prefix of product (default S1A_IW_GRDH).
            limit: Maximum number of products to return.
            timeout: HTTP request timeout in seconds.

        Returns:
            List of product metadata dicts with 'id', 'name', 'content_date', and 'footprint'.
        """
        import httpx

        min_lat, min_lon, max_lat, max_lon = bbox
        if isinstance(start_date, datetime):
            start_str = start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        else:
            start_str = start_date if "Z" in start_date else f"{start_date}T00:00:00.000Z"

        if isinstance(end_date, datetime):
            end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        else:
            end_str = end_date if "Z" in end_date else f"{end_date}T23:59:59.000Z"

        polygon_wkt = (
            f"POLYGON(({min_lon:.4f} {min_lat:.4f}, "
            f"{max_lon:.4f} {min_lat:.4f}, "
            f"{max_lon:.4f} {max_lat:.4f}, "
            f"{min_lon:.4f} {max_lat:.4f}, "
            f"{min_lon:.4f} {min_lat:.4f}))"
        )

        filter_expr = (
            f"startswith(Name,'{product_type}') and "
            f"ContentDate/Start gt {start_str} and "
            f"ContentDate/Start lt {end_str} and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon_wkt}')"
        )

        url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        params = {
            "$filter": filter_expr,
            "$top": str(limit),
            "$select": "Id,Name,ContentDate,GeoFootprint",
        }

        try:
            resp = httpx.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            products: list[dict[str, Any]] = []
            for item in data.get("value", []):
                products.append(
                    {
                        "id": item.get("Id"),
                        "name": item.get("Name"),
                        "content_date": item.get("ContentDate", {}),
                        "footprint": item.get("GeoFootprint", {}),
                    }
                )
            return products
        except Exception as e:
            raise SourceError(f"Failed to query CDSE catalogue: {e}") from e

    def process_geotiff(
        self,
        geotiff_path: Path | str,
        timestamp: datetime | None = None,
        vh_band: int = 1,
        vv_band: int | None = None,
        ephemeral: bool = False,
        step: int = 4,
    ) -> list[RawReading]:
        """Process a GeoTIFF scene, extract backscatter and water fraction, and pool to H3 cells.

        Enforces the Zero Disk Retention Rule: If ephemeral=True, the raw GeoTIFF is deleted
        immediately after cell aggregation.
        """
        import rasterio

        path = Path(geotiff_path)
        if not path.exists():
            raise FileNotFoundError(f"GeoTIFF file not found: {path}")

        ts = timestamp or self.clock.now()
        pixels: list[Sentinel1Pixel] = []

        try:
            with rasterio.open(path) as src:
                vh_data = src.read(vh_band)
                vv_data = src.read(vv_band) if vv_band is not None else None
                height, width = vh_data.shape

                # Subsample regularly to avoid memory saturation on gigabyte rasters
                for r in range(0, height, step):
                    for c in range(0, width, step):
                        vh_val = float(vh_data[r, c])
                        if not math.isfinite(vh_val) or vh_val <= 0.0:
                            continue

                        lon, lat = src.xy(r, c)
                        vv_val = float(vv_data[r, c]) if vv_data is not None else None
                        pixels.append(
                            Sentinel1Pixel(
                                lat=lat,
                                lon=lon,
                                vh_amplitude=vh_val,
                                vv_amplitude=vv_val,
                            )
                        )
        finally:
            if ephemeral and path.exists():
                path.unlink()

        if not pixels:
            return []

        aggregates = self.aggregate_pixels(pixels, ts)
        readings: list[RawReading] = []
        for agg in aggregates:
            cell_lat, cell_lon = self.grid.cell_to_latlng(agg.cell)
            geom = PointGeometry(lat=cell_lat, lon=cell_lon)
            readings.append(
                RawReading(
                    source=self.source_name,
                    variable="sar_water_fraction",
                    value=agg.water_fraction,
                    timestamp=agg.timestamp,
                    geometry=geom,
                    unit="fraction",
                )
            )
            readings.append(
                RawReading(
                    source=self.source_name,
                    variable="sar_vh_backscatter_db",
                    value=agg.mean_vh_db,
                    timestamp=agg.timestamp,
                    geometry=geom,
                    unit="dB",
                )
            )

        self.metrics.readings_produced += len(readings)
        return readings
