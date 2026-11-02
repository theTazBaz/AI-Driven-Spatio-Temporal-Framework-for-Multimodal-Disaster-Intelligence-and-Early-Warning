"""Message streaming bus package."""

from __future__ import annotations

import os
from typing import Any

from disaster_intel.bus.base import (
    BufferFullError,
    Bus,
    BusError,
    Message,
    TopicNotFoundError,
)
from disaster_intel.bus.memory import InMemoryBus
from disaster_intel.bus.redpanda import (
    DEFAULT_RETENTION_HOURS,
    MAX_RETENTION_HOURS,
    MIN_RETENTION_HOURS,
    FakeKafkaTransport,
    RealKafkaTransport,
    RedpandaBus,
)


def get_bus(bus_type: str | None = None, **kwargs: Any) -> Bus:
    """Return configured Bus instance ('memory' or 'redpanda').

    Defaults to env var DISASTER_INTEL_BUS, or 'memory'.
    """
    selected = (bus_type or os.environ.get("DISASTER_INTEL_BUS") or "memory").strip().lower()
    if selected == "memory":
        return InMemoryBus(**kwargs)
    if selected == "redpanda":
        return RedpandaBus(**kwargs)
    raise ValueError(f"Unknown bus type: '{bus_type}'. Supported: 'memory', 'redpanda'")


__all__ = [
    "DEFAULT_RETENTION_HOURS",
    "MAX_RETENTION_HOURS",
    "MIN_RETENTION_HOURS",
    "BufferFullError",
    "Bus",
    "BusError",
    "FakeKafkaTransport",
    "InMemoryBus",
    "Message",
    "RealKafkaTransport",
    "RedpandaBus",
    "TopicNotFoundError",
    "get_bus",
]
