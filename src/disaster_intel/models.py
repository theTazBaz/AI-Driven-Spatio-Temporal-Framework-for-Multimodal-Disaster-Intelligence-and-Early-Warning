"""Increment 2 — Data model for multimodal disaster telemetry.

Core types:
- Geometry      : PointGeometry | GridNodeGeometry | PolygonGeometry
- RawReading    : A single parsed reading from any source adapter (before discretisation)
- Observation   : A per-cell, unit-normalised, idempotency-keyed record (after discretisation)
- Variable      : Metadata for one physical quantity (canonical unit, aggregation method)
- VariableCatalog: Registry of every tracked variable
- CAPSeverity   : Ordered enum for Common Alerting Protocol severity levels
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from datetime import UTC, datetime
from enum import IntEnum, unique
from typing import Any, Literal

# ── Geometry types ──────────────────────────────────────────────────


class PointGeometry:
    """A single geographic point (earthquake epicentre, rain gauge, etc.)."""

    kind: Literal["point"] = "point"

    __slots__ = ("lat", "lon")

    def __init__(self, *, lat: float, lon: float) -> None:
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude {lat} out of range [-90, 90]")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude {lon} out of range [-180, 180]")
        self.lat = lat
        self.lon = lon

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "lat": self.lat, "lon": self.lon}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PointGeometry:
        return cls(lat=d["lat"], lon=d["lon"])


class GridNodeGeometry:
    """A node on a regular lat/lon grid (e.g. Open-Meteo's 0.1 degree grid)."""

    kind: Literal["grid_node"] = "grid_node"

    __slots__ = ("lat", "lon", "resolution_deg")

    def __init__(self, *, lat: float, lon: float, resolution_deg: float) -> None:
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude {lat} out of range [-90, 90]")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude {lon} out of range [-180, 180]")
        if resolution_deg <= 0.0:
            raise ValueError(f"Resolution must be positive, got {resolution_deg}")
        self.lat = lat
        self.lon = lon
        self.resolution_deg = resolution_deg

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "lat": self.lat,
            "lon": self.lon,
            "resolution_deg": self.resolution_deg,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GridNodeGeometry:
        return cls(lat=d["lat"], lon=d["lon"], resolution_deg=d["resolution_deg"])


class PolygonGeometry:
    """An arbitrary polygon (CAP alert zones, administrative boundaries, etc.)."""

    kind: Literal["polygon"] = "polygon"

    __slots__ = ("vertices",)

    def __init__(self, *, vertices: list[tuple[float, float]]) -> None:
        if len(vertices) < 3:
            raise ValueError(f"Polygon requires at least 3 vertices, got {len(vertices)}")
        for lat, lon in vertices:
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(f"Vertex latitude {lat} out of range [-90, 90]")
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(f"Vertex longitude {lon} out of range [-180, 180]")
        self.vertices = vertices

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "vertices": self.vertices}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolygonGeometry:
        return cls(vertices=[tuple(v) for v in d["vertices"]])


# Union type for all geometries
Geometry = PointGeometry | GridNodeGeometry | PolygonGeometry

_GEOMETRY_CONSTRUCTORS: dict[
    str, type[PointGeometry] | type[GridNodeGeometry] | type[PolygonGeometry]
] = {
    "point": PointGeometry,
    "grid_node": GridNodeGeometry,
    "polygon": PolygonGeometry,
}


def _geometry_from_dict(d: dict[str, Any]) -> Geometry:
    kind = d["kind"]
    constructor = _GEOMETRY_CONSTRUCTORS.get(kind)
    if constructor is None:
        raise ValueError(f"Unknown geometry kind: {kind}")
    return constructor.from_dict(d)


# ── CAP Severity ────────────────────────────────────────────────────


@unique
class CAPSeverity(IntEnum):
    """Common Alerting Protocol severity levels, ordered for MAX aggregation."""

    MINOR = 1
    MODERATE = 2
    SEVERE = 3
    EXTREME = 4

    @classmethod
    def from_str(cls, s: str) -> CAPSeverity:
        return cls[s.strip().upper()]


# ── Aggregation method ──────────────────────────────────────────────


@unique
class AggMethod(IntEnum):
    """How hourly bucket values are combined into rolling windows."""

    SUM = 1
    AVG = 2
    MAX = 3
    MIN = 4


# ── Variable ────────────────────────────────────────────────────────


class Variable:
    """Metadata for a single physical quantity tracked by the framework."""

    __slots__ = ("agg_method", "canonical_unit", "description", "name")

    def __init__(
        self,
        *,
        name: str,
        canonical_unit: str,
        agg_method: AggMethod,
        description: str = "",
    ) -> None:
        self.name = name
        self.canonical_unit = canonical_unit
        self.agg_method = agg_method
        self.description = description

    def __repr__(self) -> str:
        return f"Variable({self.name!r}, unit={self.canonical_unit!r}, agg={self.agg_method.name})"


# ── Variable Catalog ────────────────────────────────────────────────


class VariableCatalog:
    """Registry of all tracked physical variables.

    Usage:
        catalog = VariableCatalog.default()
        var = catalog["precipitation"]
    """

    def __init__(self, variables: list[Variable]) -> None:
        self._variables: dict[str, Variable] = {v.name: v for v in variables}

    def __contains__(self, name: str) -> bool:
        return name in self._variables

    def __getitem__(self, name: str) -> Variable:
        return self._variables[name]

    def __len__(self) -> int:
        return len(self._variables)

    @classmethod
    def default(cls) -> VariableCatalog:
        """Return the default catalog covering all Phase 1 variables."""
        return cls(
            [
                Variable(
                    name="precipitation",
                    canonical_unit="mm",
                    agg_method=AggMethod.SUM,
                    description="Accumulated precipitation",
                ),
                Variable(
                    name="soil_moisture_surface",
                    canonical_unit="m3/m3",
                    agg_method=AggMethod.AVG,
                    description="Volumetric soil moisture, surface layer (0-7 cm)",
                ),
                Variable(
                    name="soil_moisture_rootzone",
                    canonical_unit="m3/m3",
                    agg_method=AggMethod.AVG,
                    description="Volumetric soil moisture, root-zone layer (0-100 cm)",
                ),
                Variable(
                    name="temperature",
                    canonical_unit="celsius",
                    agg_method=AggMethod.AVG,
                    description="Surface air temperature at 2m",
                ),
                Variable(
                    name="wind_speed",
                    canonical_unit="km/h",
                    agg_method=AggMethod.MAX,
                    description="Wind speed at 10m",
                ),
                Variable(
                    name="pressure",
                    canonical_unit="hPa",
                    agg_method=AggMethod.AVG,
                    description="Mean sea-level atmospheric pressure",
                ),
                Variable(
                    name="discharge",
                    canonical_unit="m3/s",
                    agg_method=AggMethod.MAX,
                    description="River discharge (Open-Meteo flood API / GloFAS)",
                ),
                Variable(
                    name="seismic_magnitude",
                    canonical_unit="richter",
                    agg_method=AggMethod.MAX,
                    description="Earthquake magnitude (Richter / moment magnitude)",
                ),
                Variable(
                    name="cap_severity",
                    canonical_unit="ordinal",
                    agg_method=AggMethod.MAX,
                    description="CAP alert severity level (Minor/Moderate/Severe/Extreme)",
                ),
            ]
        )


# ── Unit conversion ─────────────────────────────────────────────────


@functools.cache
def _canonical_units() -> dict[str, str]:
    return {v.name: v.canonical_unit for v in VariableCatalog.default()._variables.values()}


def _keep_canonical_spelling(variable: str, raw_lower: str) -> str:
    """'hpa' -> 'hPa': a unit equal to the canonical one (ignoring case) is the canonical one."""
    canonical = _canonical_units().get(variable)
    return canonical if canonical is not None and canonical.lower() == raw_lower else raw_lower


def convert_unit(
    value: float,
    raw_unit: str,
    variable: str,
    interval_hours: float = 1.0,
) -> tuple[float, str]:
    """Convert a raw value to its canonical unit.

    Returns (converted_value, canonical_unit).
    """
    raw_lower = raw_unit.lower().strip()

    if variable == "temperature":
        if raw_lower == "kelvin" or raw_lower == "k":
            return (value - 273.15, "celsius")
        if raw_lower in ("celsius", "c", "°c"):
            return (value, "celsius")
        # Fahrenheit as a future extension
        if raw_lower in ("fahrenheit", "f", "°f"):
            return ((value - 32.0) * 5.0 / 9.0, "celsius")
        return (value, raw_lower)

    if variable == "precipitation":
        if raw_lower == "mm/hr" or raw_lower == "mm/h":
            return (value * interval_hours, "mm")
        if raw_lower == "mm":
            return (value, "mm")
        return (value, raw_lower)

    # For all other variables, pass through (keeping the catalog's spelling, e.g. 'hPa')
    return (value, _keep_canonical_spelling(variable, raw_lower))


# ── RawReading ──────────────────────────────────────────────────────


class RawReading:
    """A single parsed reading from a source adapter, before spatial discretisation.

    Validation rules (TDD, plan.md):
    - Naive datetimes are rejected.
    - NaN/Inf values are rejected.
    - Timestamps are normalised to UTC.
    """

    __slots__ = ("geometry", "source", "timestamp", "unit", "value", "variable")

    def __init__(
        self,
        *,
        source: str,
        variable: str,
        value: float,
        timestamp: datetime,
        geometry: Geometry,
        unit: str,
    ) -> None:
        if not source:
            raise ValueError("Source must be a non-empty string")
        if not variable:
            raise ValueError("Variable must be a non-empty string")
        if not math.isfinite(value):
            raise ValueError(f"Value must be finite, got {value}")
        if timestamp.tzinfo is None:
            raise ValueError(
                "Naive datetime (no timezone) is not allowed. "
                "All timestamps must be timezone-aware."
            )

        self.source = source
        self.variable = variable
        self.value = value
        self.timestamp = timestamp.astimezone(UTC)
        self.geometry = geometry
        self.unit = unit

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "variable": self.variable,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "geometry": self.geometry.to_dict(),
            "unit": self.unit,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RawReading:
        return cls(
            source=d["source"],
            variable=d["variable"],
            value=d["value"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            geometry=_geometry_from_dict(d["geometry"]),
            unit=d["unit"],
        )

    @classmethod
    def from_json(cls, s: str) -> RawReading:
        return cls.from_dict(json.loads(s))


# ── Observation ─────────────────────────────────────────────────────


def compute_idempotency_key(cell: str, variable: str, source: str, timestamp: datetime) -> str:
    """Deterministic key: hash of (cell, variable, source, UTC timestamp).

    Shared by ``Observation.from_raw`` and the cold store, which does not persist
    the key (it is fully derivable from the stored columns) and recomputes it on read.
    """
    key_input = f"{cell}|{variable}|{source}|{timestamp.astimezone(UTC).isoformat()}"
    return hashlib.sha256(key_input.encode("utf-8")).hexdigest()[:16]


class Observation:
    """A per-cell, unit-normalised, idempotency-keyed record.

    This is the universal record that flows through the bus and into storage.
    """

    __slots__ = (
        "cell",
        "idempotency_key",
        "source",
        "timestamp",
        "unit",
        "value",
        "variable",
    )

    def __init__(
        self,
        *,
        cell: str,
        source: str,
        variable: str,
        value: float,
        unit: str,
        timestamp: datetime,
        idempotency_key: str,
    ) -> None:
        self.cell = cell
        self.source = source
        self.variable = variable
        self.value = value
        self.unit = unit
        self.timestamp = timestamp
        self.idempotency_key = idempotency_key

    @classmethod
    def from_reading(
        cls,
        reading: RawReading,
        cell: str,
        interval_hours: float = 1.0,
    ) -> Observation:
        """Create an Observation directly from a RawReading and an H3 cell ID."""
        return cls.from_raw(
            cell=cell,
            source=reading.source,
            variable=reading.variable,
            raw_value=reading.value,
            raw_unit=reading.unit,
            timestamp=reading.timestamp,
            interval_hours=interval_hours,
        )

    @classmethod
    def from_raw(
        cls,
        reading_or_cell: RawReading | str | None = None,
        cell: str | None = None,
        *,
        source: str = "",
        variable: str = "",
        raw_value: float = 0.0,
        raw_unit: str = "",
        timestamp: datetime | None = None,
        interval_hours: float = 1.0,
        **kwargs: Any,
    ) -> Observation:
        """Create an Observation by converting raw values to canonical units.

        Supports:
        - Observation.from_raw(reading, cell)
        - Observation.from_raw(cell=..., source=..., variable=..., raw_value=..., timestamp=...)
        """
        if isinstance(reading_or_cell, RawReading):
            target_cell = cell or kwargs.get("cell", "")
            return cls.from_reading(
                reading=reading_or_cell,
                cell=target_cell,
                interval_hours=interval_hours,
            )

        target_cell = str(reading_or_cell or cell or kwargs.get("cell", ""))
        src = source or kwargs.get("source", "")
        var = variable or kwargs.get("variable", "")
        val = (
            raw_value if ("raw_value" in kwargs or raw_value != 0.0) else kwargs.get("value", 0.0)
        )
        unit_str = raw_unit or kwargs.get("unit", "")
        ts = timestamp or kwargs.get("timestamp")

        if not math.isfinite(val):
            raise ValueError(f"Value must be finite, got {val}")
        if ts is None or ts.tzinfo is None:
            raise ValueError("Naive datetime is not allowed")

        ts_utc = ts.astimezone(UTC)
        converted_value, canonical_unit = convert_unit(val, unit_str, var, interval_hours)

        key = compute_idempotency_key(target_cell, var, src, ts_utc)

        return cls(
            cell=target_cell,
            source=src,
            variable=var,
            value=converted_value,
            unit=canonical_unit,
            timestamp=ts_utc,
            idempotency_key=key,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "source": self.source,
            "variable": self.variable,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "idempotency_key": self.idempotency_key,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Observation:
        return cls(
            cell=d["cell"],
            source=d["source"],
            variable=d["variable"],
            value=d["value"],
            unit=d["unit"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            idempotency_key=d["idempotency_key"],
        )

    @classmethod
    def from_json(cls, s: str) -> Observation:
        return cls.from_dict(json.loads(s))

    def __repr__(self) -> str:
        return (
            f"Observation(cell={self.cell!r}, var={self.variable!r}, "
            f"value={self.value}, t={self.timestamp.isoformat()})"
        )
