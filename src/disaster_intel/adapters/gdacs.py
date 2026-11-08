"""GDACS Civil Protection and Disaster Alert Adapter.

Ingests real multi-hazard alerts from GDACS:
- Live RSS/CAP XML feed: https://www.gdacs.org/xml/rss.xml
- Automated parsing of disaster types: Cyclones (TC), Floods (FL), Earthquakes (EQ), Wildfires (WF)
- Severity mapped to CAPSeverity (Green -> MINOR, Orange -> MODERATE, Red -> SEVERE/EXTREME)
- Bounding polygon extraction for PolygonDiscretizer into H3 Res-7 cells.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from disaster_intel.adapters.base import (
    AdapterMetrics,
    ParseError,
    SourceAdapter,
    SourceError,
)
from disaster_intel.clock import Clock, SystemClock
from disaster_intel.grid import CoastalGrid
from disaster_intel.models import (
    CAPSeverity,
    Geometry,
    PointGeometry,
    PolygonGeometry,
    RawReading,
)

if TYPE_CHECKING:
    pass

GDACS_RSS_URL: str = "https://www.gdacs.org/xml/rss.xml"

GDACS_NAMESPACES: dict[str, str] = {
    "gdacs": "http://www.gdacs.org",
    "georss": "http://www.georss.org/georss",
    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
}


def alert_level_to_cap(level: str, score: float = 1.0) -> CAPSeverity:
    """Map GDACS alert level ('Green', 'Orange', 'Red') and score to CAPSeverity."""
    lvl = level.strip().lower()
    if lvl == "red":
        return CAPSeverity.EXTREME if score >= 2.5 else CAPSeverity.SEVERE
    mapping = {
        "green": CAPSeverity.MINOR,
        "orange": CAPSeverity.MODERATE,
    }
    return mapping.get(lvl, CAPSeverity.MINOR)


@dataclass(frozen=True, slots=True)
class GDACSAlert:
    """Parsed alert item from GDACS."""

    alert_id: str
    title: str
    event_type: str
    alert_level: str
    alert_score: float
    country: str
    timestamp: datetime
    point: tuple[float, float] | None  # (lat, lon)
    bbox: tuple[float, float, float, float] | None  # (min_lon, max_lon, min_lat, max_lat)

    def to_geometry(self) -> Geometry:
        """Convert alert spatial extent to PolygonGeometry or PointGeometry."""

        def wrap_lon(val: float) -> float:
            return ((val + 180.0) % 360.0) - 180.0

        def clamp_lat(val: float) -> float:
            return max(-90.0, min(90.0, val))

        if self.bbox is not None:
            min_lon, max_lon, min_lat, max_lat = self.bbox
            c_min_lat = clamp_lat(min_lat)
            c_max_lat = clamp_lat(max_lat)
            w_min_lon = wrap_lon(min_lon)
            w_max_lon = wrap_lon(max_lon)
            # 5-vertex closed polygon: (lat, lon) pairs
            vertices = [
                (c_min_lat, w_min_lon),
                (c_min_lat, w_max_lon),
                (c_max_lat, w_max_lon),
                (c_max_lat, w_min_lon),
                (c_min_lat, w_min_lon),
            ]
            return PolygonGeometry(vertices=vertices)
        elif self.point is not None:
            return PointGeometry(lat=clamp_lat(self.point[0]), lon=wrap_lon(self.point[1]))
        else:
            raise ValueError(f"Alert {self.alert_id} has neither bbox nor point coordinates")


def parse_gdacs_xml(
    xml_content: str | bytes,
    country_filter: str | None = None,
) -> list[GDACSAlert]:
    """Parse raw GDACS RSS XML feed into structured GDACSAlert objects."""
    try:
        root = ET.fromstring(xml_content)
    except Exception as e:
        raise ParseError(f"Malformed GDACS XML: {e}") from e

    channel = root.find("channel")
    if channel is None:
        raise ParseError("GDACS RSS missing channel element")

    alerts: list[GDACSAlert] = []
    cf_lower = country_filter.lower() if country_filter else None

    for item in channel.findall("item"):
        guid_elem = item.find("guid")
        alert_id = (
            guid_elem.text.strip() if guid_elem is not None and guid_elem.text else "unknown"
        )

        title_elem = item.find("title")
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

        country_elem = item.find("gdacs:country", GDACS_NAMESPACES)
        country = (
            country_elem.text.strip() if country_elem is not None and country_elem.text else ""
        )

        if cf_lower and cf_lower not in country.lower():
            continue

        etype_elem = item.find("gdacs:eventtype", GDACS_NAMESPACES)
        event_type = (
            etype_elem.text.strip() if etype_elem is not None and etype_elem.text else "OTHER"
        )

        level_elem = item.find("gdacs:alertlevel", GDACS_NAMESPACES)
        alert_level = (
            level_elem.text.strip() if level_elem is not None and level_elem.text else "Green"
        )

        score_elem = item.find("gdacs:alertscore", GDACS_NAMESPACES)
        try:
            alert_score = (
                float(score_elem.text) if score_elem is not None and score_elem.text else 1.0
            )
        except ValueError:
            alert_score = 1.0

        pubdate_elem = item.find("pubDate")
        if pubdate_elem is not None and pubdate_elem.text:
            try:
                # Standard RFC 822/2822: e.g. 'Wed, 28 Oct 2026 12:25:05 GMT'
                ts = datetime.strptime(pubdate_elem.text.strip(), "%a, %d %b %Y %H:%M:%S %Z")
                ts = ts.replace(tzinfo=UTC)
            except Exception:
                ts = datetime.now(tz=UTC)
        else:
            ts = datetime.now(tz=UTC)

        # Spatial point
        point_elem = item.find("georss:point", GDACS_NAMESPACES)
        point: tuple[float, float] | None = None
        if point_elem is not None and point_elem.text:
            parts = point_elem.text.strip().split()
            if len(parts) == 2:
                try:
                    point = (float(parts[0]), float(parts[1]))
                except ValueError:
                    point = None

        # Spatial bbox
        bbox_elem = item.find("gdacs:bbox", GDACS_NAMESPACES)
        bbox: tuple[float, float, float, float] | None = None
        if bbox_elem is not None and bbox_elem.text:
            bparts = bbox_elem.text.strip().split()
            if len(bparts) == 4:
                try:
                    bbox = (
                        float(bparts[0]),
                        float(bparts[1]),
                        float(bparts[2]),
                        float(bparts[3]),
                    )
                except ValueError:
                    bbox = None

        if point is None and bbox is None:
            continue

        alerts.append(
            GDACSAlert(
                alert_id=alert_id,
                title=title,
                event_type=event_type,
                alert_level=alert_level,
                alert_score=alert_score,
                country=country,
                timestamp=ts,
                point=point,
                bbox=bbox,
            )
        )

    return alerts


class GDACSAdapter(SourceAdapter):
    """Source adapter for Global Disaster Alert and Coordination System (GDACS)."""

    source_name: str = "gdacs"

    def __init__(
        self,
        grid: CoastalGrid | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.grid = grid or CoastalGrid(use_cache=True)
        self.clock = clock or SystemClock()
        self.metrics = AdapterMetrics()

    def fetch_live_alerts(
        self,
        country: str | None = "India",
        timeout: float = 20.0,
    ) -> list[RawReading]:
        """Fetch live civil protection alerts from GDACS RSS feed."""
        import httpx

        try:
            resp = httpx.get(GDACS_RSS_URL, timeout=timeout)
            resp.raise_for_status()
            return self.parse_xml(resp.text, country_filter=country)
        except Exception as e:
            raise SourceError(f"Failed to fetch live GDACS feed: {e}") from e

    def parse_xml(
        self,
        xml_text: str | bytes,
        country_filter: str | None = None,
    ) -> list[RawReading]:
        """Parse XML feed into RawReadings."""
        alerts = parse_gdacs_xml(xml_text, country_filter=country_filter)
        readings: list[RawReading] = []

        for alert in alerts:
            cap_enum = alert_level_to_cap(alert.alert_level, alert.alert_score)
            geom = alert.to_geometry()

            readings.append(
                RawReading(
                    source=self.source_name,
                    variable="cap_severity",
                    value=float(cap_enum.value),
                    timestamp=alert.timestamp,
                    geometry=geom,
                    unit="ordinal",
                )
            )

        self.metrics.readings_produced += len(readings)
        return readings

    def parse_payload(self, payload: Any) -> list[RawReading]:
        """Parse dictionary or raw string payload into domain RawReadings."""
        if isinstance(payload, (str, bytes)):
            return self.parse_xml(payload)

        if isinstance(payload, dict):
            raw_xml = payload.get("xml")
            if raw_xml:
                return self.parse_xml(raw_xml)

            # JSON representation
            raw_alerts = payload.get("alerts") or payload.get("items")
            if not isinstance(raw_alerts, list):
                self.metrics.record_parse_error()
                raise ParseError("Payload missing valid 'alerts' or 'items' list")

            readings: list[RawReading] = []
            for item in raw_alerts:
                if not isinstance(item, dict):
                    self.metrics.record_skip("malformed_alert")
                    continue

                level = str(item.get("alert_level", "Green"))
                score = float(item.get("alert_score", 1.0))
                cap_enum = alert_level_to_cap(level, score)

                raw_ts = item.get("timestamp")
                if raw_ts:
                    try:
                        ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
                        ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
                    except Exception:
                        ts = self.clock.now()
                else:
                    ts = self.clock.now()

                geom_dict = item.get("geometry")
                if not isinstance(geom_dict, dict):
                    self.metrics.record_skip("missing_geometry")
                    continue

                try:
                    if geom_dict.get("kind") == "polygon":
                        vertices = [(float(v[0]), float(v[1])) for v in geom_dict["vertices"]]
                        geom: Geometry = PolygonGeometry(vertices=vertices)
                    elif geom_dict.get("kind") == "point":
                        geom = PointGeometry(
                            lat=float(geom_dict["lat"]), lon=float(geom_dict["lon"])
                        )
                    else:
                        self.metrics.record_skip("unknown_geometry_kind")
                        continue
                except (ValueError, KeyError, TypeError):
                    self.metrics.record_skip("invalid_geometry")
                    continue

                readings.append(
                    RawReading(
                        source=self.source_name,
                        variable="cap_severity",
                        value=float(cap_enum.value),
                        timestamp=ts,
                        geometry=geom,
                        unit="ordinal",
                    )
                )

            self.metrics.readings_produced += len(readings)
            return readings

        self.metrics.record_parse_error()
        raise ParseError(f"Unsupported payload type: {type(payload)}")
