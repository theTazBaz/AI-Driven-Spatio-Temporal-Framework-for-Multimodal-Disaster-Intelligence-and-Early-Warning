"""Telemetry source adapters package."""

from disaster_intel.adapters.base import (
    AdapterError,
    AdapterMetrics,
    HttpResponse,
    HttpTransport,
    HttpxTransport,
    MockHttpTransport,
    ParseError,
    SourceAdapter,
    SourceError,
)
from disaster_intel.adapters.open_meteo import OpenMeteoAdapter
from disaster_intel.adapters.sentinel1 import (
    WATER_THRESHOLD_DB,
    Sentinel1Adapter,
    Sentinel1CellAggregate,
    Sentinel1Pixel,
    amplitude_to_db,
    classify_water,
    db_to_amplitude,
)

__all__ = [
    "WATER_THRESHOLD_DB",
    "AdapterError",
    "AdapterMetrics",
    "HttpResponse",
    "HttpTransport",
    "HttpxTransport",
    "MockHttpTransport",
    "OpenMeteoAdapter",
    "ParseError",
    "Sentinel1Adapter",
    "Sentinel1CellAggregate",
    "Sentinel1Pixel",
    "SourceAdapter",
    "SourceError",
    "amplitude_to_db",
    "classify_water",
    "db_to_amplitude",
]
