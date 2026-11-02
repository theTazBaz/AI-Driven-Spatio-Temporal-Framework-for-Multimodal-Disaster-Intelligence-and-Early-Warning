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
from disaster_intel.adapters.insat import (
    COLD_TOP_THRESHOLD_K,
    RAPID_COOLING_THRESHOLD_K_PER_15MIN,
    INSATAdapter,
    calculate_cooling_rate,
    celsius_to_kelvin,
    is_deep_convection,
    is_rapid_cooling,
    kelvin_to_celsius,
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
    "COLD_TOP_THRESHOLD_K",
    "RAPID_COOLING_THRESHOLD_K_PER_15MIN",
    "WATER_THRESHOLD_DB",
    "AdapterError",
    "AdapterMetrics",
    "HttpResponse",
    "HttpTransport",
    "HttpxTransport",
    "INSATAdapter",
    "MockHttpTransport",
    "OpenMeteoAdapter",
    "ParseError",
    "Sentinel1Adapter",
    "Sentinel1CellAggregate",
    "Sentinel1Pixel",
    "SourceAdapter",
    "SourceError",
    "amplitude_to_db",
    "calculate_cooling_rate",
    "celsius_to_kelvin",
    "classify_water",
    "db_to_amplitude",
    "is_deep_convection",
    "is_rapid_cooling",
    "kelvin_to_celsius",
]
