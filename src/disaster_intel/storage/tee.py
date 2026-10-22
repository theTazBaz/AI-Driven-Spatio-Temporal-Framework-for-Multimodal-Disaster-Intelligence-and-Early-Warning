"""Increment 9 - dual-write sink: the cold tier is the record, the hot tier a cache of it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from disaster_intel.models import Observation


class WritableStore(Protocol):
    name: str

    def write(self, observations: Sequence[Observation]) -> tuple[int, int]: ...


class TeeSink:
    """Write each batch to ``primary`` (ClickHouse) first, then to every ``secondary`` (Redis).

    Order matters: if a secondary fails, the exception propagates, but the record is already
    safe in the primary and a retry is harmless (every write is idempotent). The hot tier can
    always be rebuilt from the primary (``rebuild_hot_from_cold``).
    Return value is the primary's ``(stored, duplicates)`` so pipeline metrics stay unchanged.
    """

    def __init__(self, primary: WritableStore, *secondaries: WritableStore) -> None:
        self.primary = primary
        self.secondaries = secondaries
        self.name = "+".join([primary.name, *(s.name for s in secondaries)])

    def write(self, observations: Sequence[Observation]) -> tuple[int, int]:
        result = self.primary.write(observations)
        for secondary in self.secondaries:
            secondary.write(observations)
        return result
