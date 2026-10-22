"""Increment 9 - Redis hot tier (``HotStore``).

The hot tier answers "what is the state of this cell *now*?" in a millisecond or two. The
cold tier (ClickHouse, Increments 7-8) is the permanent record; the hot tier is a rebuildable
cache of its last seven days.

Design (and why it differs from the plan's one-line sketch)
-----------------------------------------------------------
plan.md pictures a "7-day bucket ring, ~6.7 KB/cell". That figure is 168 hours x 10 variables
x 4 bytes, i.e. ONE float per bucket. A single float cannot answer SUM, AVG, MAX and MIN, and
-- more importantly -- it cannot make "applying one observation twice changes nothing" true:
an aggregate (n, sum, min, max) cannot tell a duplicate from a new reading, nor undo a revised
value. So the hot tier keeps the *observations themselves*, compressed, and computes the
window aggregates at read time with the same bucket conventions the cold tier uses
(``disaster_intel.windows``). That gives:

* **Idempotent apply**   an observation is identified by (variable, source, ts); applying it
  again overwrites the same slot. A revised value replaces the old one (as in ClickHouse's
  ``ReplacingMergeTree``).
* **Oracle equality**    windows follow the same conventions as ``ColdStore.windows`` and
  ``reference_windows`` (``disaster_intel.windows``), pinned by differential tests.
* **Rebuildable**        wiping Redis and replaying the last 7 days from the cold tier gives
  byte-identical state (``rebuild.py``).

Layout
------
One Redis hash per H3 cell: key ``ws:v1:<cell>``. One hash field per (variable, source):
``"<variable>|<source>"`` -> a zlib-compressed, time-sorted column block (delta-coded
timestamps + float32 values, ~4 bytes per observation after compression). All of a cell's
state is therefore one ``HGETALL``.

Retention
---------
An observation is kept while ``bucket_end(ts) > floor_hour(now) - retention_hours``.
Eviction is lazy (every write prunes the fields it touches), plus ``evict_expired`` for a
sweep, plus a TTL on the key so abandoned cells disappear. ``windows()`` refuses (raises
``HotRangeError``) any ``as_of`` whose window would reach back past the retained range, so a
caller can never silently get a truncated answer; for the past, use ``ColdStore.windows``.

Concurrency
-----------
Writes are read-modify-write per cell and assume ONE writer per cell (the bus already orders
messages per cell key, so a single consumer owns each cell). Readers are always safe: every
write to a cell is a single atomic ``HSET``.
"""

from __future__ import annotations

import math
import struct
import zlib
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import h3

from disaster_intel.clock import Clock, SystemClock
from disaster_intel.models import AggMethod, Observation, VariableCatalog
from disaster_intel.pipeline import calculate_percentiles
from disaster_intel.storage.cold import observation_to_row
from disaster_intel.windows import (
    HOUR,
    WINDOW_HOURS,
    WindowValue,
    floor_hour,
    variable_names,
)

KEY_PREFIX = "ws:v1:"
RETENTION_HOURS = 168  # 7 days: the longest rolling window
TTL_SLACK_HOURS = 24
_FORMAT_VERSION = 1
_HOUR_US = 3_600_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ONE_MICROSECOND = timedelta(microseconds=1)
_FIELD_SEPARATOR = "|"


class HotRangeError(ValueError):
    """The requested window reaches back further than the hot tier retains."""


class HotFormatError(ValueError):
    """A stored blob is not in a format this code version understands."""


# ── Redis protocol (the slice of redis-py that HotStore uses) ────────


class RedisPipeline(Protocol):
    def hmget(self, name: str, keys: Any, *args: Any) -> Any: ...

    def hgetall(self, name: str) -> Any: ...

    def hset(self, name: str, key: Any = None, value: Any = None, mapping: Any = None) -> Any: ...

    def hdel(self, name: str, *keys: Any) -> Any: ...

    def expire(self, name: str, time: Any) -> Any: ...

    def execute(self) -> list[Any]: ...


class RedisClient(Protocol):
    def pipeline(self, transaction: bool = True) -> Any: ...

    def hgetall(self, name: str) -> Any: ...

    def delete(self, *names: str) -> Any: ...

    def scan_iter(self, match: Any = None, count: Any = None) -> Iterator[Any]: ...

    def memory_usage(self, key: str) -> Any: ...


# ── Pure encoding (unit-tested without Redis) ────────────────────────


def _micros(ts: datetime) -> int:
    if ts.tzinfo is None:
        raise ValueError("Naive datetime is not allowed")
    return (ts.astimezone(UTC) - _EPOCH) // _ONE_MICROSECOND


def _from_micros(micros: int) -> datetime:
    return _EPOCH + timedelta(microseconds=micros)


def bucket_end_micros(ts_micros: int) -> int:
    """Hour-ending bucket label of a timestamp, in epoch microseconds (matches bucket_end())."""
    return -(-ts_micros // _HOUR_US) * _HOUR_US


def encode_records(records: Mapping[int, float]) -> bytes:
    """``{ts_micros: value}`` -> compressed blob (sorted by time, delta-coded, float32)."""
    if not records:
        raise ValueError("Refusing to encode an empty record set")
    stamps = sorted(records)
    n = len(stamps)
    deltas = [stamps[i] - stamps[i - 1] for i in range(1, n)]
    payload = (
        struct.pack("<Iq", n, stamps[0])
        + struct.pack(f"<{n - 1}q", *deltas)
        + struct.pack(f"<{n}f", *(records[t] for t in stamps))
    )
    return bytes([_FORMAT_VERSION]) + zlib.compress(payload, 6)


def decode_records(blob: bytes) -> dict[int, float]:
    """Inverse of ``encode_records``. Values come back as float32-quantised floats."""
    if not blob or blob[0] != _FORMAT_VERSION:
        raise HotFormatError(f"Unsupported hot-tier blob format {blob[:1]!r}")
    try:
        payload = zlib.decompress(blob[1:])
        n, first = struct.unpack_from("<Iq", payload, 0)
        offset = struct.calcsize("<Iq")
        deltas = struct.unpack_from(f"<{n - 1}q", payload, offset)
        offset += 8 * (n - 1)
        values = struct.unpack_from(f"<{n}f", payload, offset)
    except (zlib.error, struct.error) as exc:
        raise HotFormatError(f"Corrupt hot-tier blob: {exc}") from exc
    stamps = [first]
    for delta in deltas:
        stamps.append(stamps[-1] + delta)
    return dict(zip(stamps, values, strict=True))


def make_field(variable: str, source: str) -> str:
    if _FIELD_SEPARATOR in variable or _FIELD_SEPARATOR in source:
        raise ValueError(f"'{_FIELD_SEPARATOR}' is not allowed in variable/source names")
    return f"{variable}{_FIELD_SEPARATOR}{source}"


def split_field(field: bytes | str) -> tuple[str, str]:
    text = field.decode("utf-8") if isinstance(field, bytes) else field
    variable, _, source = text.partition(_FIELD_SEPARATOR)
    return variable, source


def _reduce_values(agg: AggMethod, values: Sequence[float]) -> float:
    if agg is AggMethod.SUM:
        return math.fsum(values)
    if agg is AggMethod.AVG:
        return math.fsum(values) / len(values)
    if agg is AggMethod.MAX:
        return max(values)
    if agg is AggMethod.MIN:
        return min(values)
    raise ValueError(f"Unsupported aggregation {agg!r}")


# ── Latency reporting (plan: "average, p50, p95, p99") ───────────────


@dataclass(frozen=True)
class LatencyReport:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def format(self, label: str) -> str:
        return (
            f"{label}: n={self.count} avg={self.mean_ms:.3f} ms p50={self.p50_ms:.3f} ms "
            f"p95={self.p95_ms:.3f} ms p99={self.p99_ms:.3f} ms"
        )


def summarize_latencies(values_ms: Sequence[float]) -> LatencyReport:
    if not values_ms:
        return LatencyReport(0, 0.0, 0.0, 0.0, 0.0)
    p50, p95, p99 = calculate_percentiles(values_ms, [50, 95, 99])
    return LatencyReport(
        count=len(values_ms),
        mean_ms=sum(values_ms) / len(values_ms),
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
    )


# ── The store ────────────────────────────────────────────────────────

CellState = dict[tuple[str, str], list[tuple[int, float]]]


class HotStore:
    """Redis-backed rolling 7-day state, one hash per H3 cell.

    Mirrors the ``ObservationSink`` surface that matters to the pipeline (``name`` and
    ``write``) and ``ColdStore.windows`` (same signature and return shape), so the state
    assembler in Increment 10 can treat both tiers alike.
    """

    name: str = "redis_hot_store"

    def __init__(
        self,
        client: RedisClient,
        clock: Clock | None = None,
        catalog: VariableCatalog | None = None,
        prefix: str = KEY_PREFIX,
        retention_hours: int = RETENTION_HOURS,
    ) -> None:
        if retention_hours < max(WINDOW_HOURS.values()):
            raise ValueError(
                f"retention_hours must be >= the longest window ({max(WINDOW_HOURS.values())})"
            )
        self._client = client
        self._clock: Clock = clock or SystemClock()
        self._catalog = catalog or VariableCatalog.default()
        self._prefix = prefix
        self._retention_hours = retention_hours
        self.rejected_stale = 0  # observations dropped because they were already expired

    @classmethod
    def connect(
        cls,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        **kwargs: Any,
    ) -> HotStore:
        """Connect to the Redis that docker-compose publishes on localhost:6379."""
        import redis

        client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        return cls(client, **kwargs)

    # ── helpers ──────────────────────────────────────────────────────

    @property
    def client(self) -> RedisClient:
        return self._client

    @property
    def retention_hours(self) -> int:
        return self._retention_hours

    def key(self, cell: str) -> str:
        return f"{self._prefix}{cell}"

    def horizon(self) -> datetime:
        """Observations are kept while ``bucket_end(ts) > horizon()``."""
        return floor_hour(self._clock.now()) - self._retention_hours * HOUR

    @property
    def _ttl_seconds(self) -> int:
        return (self._retention_hours + TTL_SLACK_HOURS) * 3600

    @staticmethod
    def _check_cell(cell: str) -> None:
        if not isinstance(cell, str) or not h3.is_valid_cell(cell):
            raise ValueError(f"Invalid H3 cell {cell!r}")

    # ── writes ───────────────────────────────────────────────────────

    def write(self, observations: Sequence[Observation]) -> tuple[int, int]:
        """Apply observations. Returns ``(stored, duplicates)``.

        ``stored`` counts new readings and revised values; ``duplicates`` counts readings
        identical to what is already held (same variable, source, timestamp AND value).
        Already-expired readings are dropped and counted in ``rejected_stale``.
        All observations are validated before any Redis call, as in ``ColdStore.write``.
        """
        validated: list[tuple[str, str, str, int, float]] = []
        for obs in observations:
            row = observation_to_row(obs, self._catalog)  # raises on unknown/invalid input
            field = make_field(obs.variable, obs.source)
            validated.append((obs.cell, obs.variable, field, _micros(obs.timestamp), row[4]))

        horizon_us = _micros(self.horizon())
        grouped: dict[str, dict[str, dict[int, float]]] = {}
        duplicates = 0
        for cell, _variable, field, ts_us, value in validated:
            if bucket_end_micros(ts_us) <= horizon_us:
                self.rejected_stale += 1
                continue
            slot = grouped.setdefault(cell, {}).setdefault(field, {})
            if slot.get(ts_us) == value:
                duplicates += 1
            slot[ts_us] = value  # last write wins
        if not grouped:
            return 0, duplicates

        cells = list(grouped)
        read = self._client.pipeline(transaction=False)
        for cell in cells:
            read.hmget(self.key(cell), list(grouped[cell]))
        existing_per_cell = read.execute()

        write = self._client.pipeline(transaction=False)
        stored = 0
        for cell, existing_blobs in zip(cells, existing_per_cell, strict=True):
            updates: dict[str, bytes] = {}
            for (field, incoming), blob in zip(grouped[cell].items(), existing_blobs, strict=True):
                merged = decode_records(blob) if blob else {}
                for ts_us, value in incoming.items():
                    if ts_us in merged and merged[ts_us] == value:
                        duplicates += 1
                    else:
                        stored += 1
                    merged[ts_us] = value
                kept = {t: v for t, v in merged.items() if bucket_end_micros(t) > horizon_us}
                updates[field] = encode_records(kept)
            write.hset(self.key(cell), mapping=updates)
            write.expire(self.key(cell), self._ttl_seconds)
        write.execute()
        return stored, duplicates

    def evict_expired(self, cells: Iterable[str]) -> int:
        """Sweep ``cells``: drop expired records (and emptied fields). Returns records removed."""
        cell_list = list(cells)
        for cell in cell_list:
            self._check_cell(cell)
        if not cell_list:
            return 0
        horizon_us = _micros(self.horizon())
        read = self._client.pipeline(transaction=False)
        for cell in cell_list:
            read.hgetall(self.key(cell))
        snapshots = read.execute()

        write = self._client.pipeline(transaction=False)
        removed = 0
        for cell, raw in zip(cell_list, snapshots, strict=True):
            for field, blob in (raw or {}).items():
                records = decode_records(blob)
                kept = {t: v for t, v in records.items() if bucket_end_micros(t) > horizon_us}
                if len(kept) == len(records):
                    continue
                removed += len(records) - len(kept)
                name = field.decode("utf-8") if isinstance(field, bytes) else field
                if kept:
                    write.hset(self.key(cell), mapping={name: encode_records(kept)})
                else:
                    write.hdel(self.key(cell), name)
        write.execute()
        return removed

    def clear(self) -> int:
        """Delete every key under this store's prefix (tests, rebuilds). Returns keys deleted."""
        deleted = 0
        batch: list[str] = []
        for raw_key in self._client.scan_iter(match=f"{self._prefix}*", count=1000):
            batch.append(raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key)
            if len(batch) >= 1000:
                deleted += int(self._client.delete(*batch))
                batch = []
        if batch:
            deleted += int(self._client.delete(*batch))
        return deleted

    # ── reads ────────────────────────────────────────────────────────

    def windows(
        self,
        cell: str,
        as_of: datetime | None = None,
        source: str | None = None,
        window_hours: Mapping[str, int] | None = None,
    ) -> dict[str, dict[str, WindowValue]]:
        """Rolling windows for every catalog variable of one cell (same shape as the cold tier).

        ``as_of`` defaults to the clock's now. Raises ``HotRangeError`` if the window would
        reach back beyond the retained range (use ``ColdStore.windows`` for that).
        """
        self._check_cell(cell)
        hours = dict(WINDOW_HOURS if window_hours is None else window_hours)
        end = floor_hour(self._clock.now() if as_of is None else as_of)
        if end - max(hours.values()) * HOUR < self.horizon():
            raise HotRangeError(
                f"Window ending {end.isoformat()} reaches before the hot tier's retained range "
                f"(starts after {self.horizon().isoformat()}); read the cold tier instead"
            )
        raw = self._client.hgetall(self.key(cell)) or {}
        return self._windows_from_raw(raw, end, hours, source)

    def _windows_from_raw(
        self,
        raw: Mapping[Any, bytes],
        end: datetime,
        hours: Mapping[str, int],
        source: str | None,
    ) -> dict[str, dict[str, WindowValue]]:
        """Windows straight from the integer bucket labels (no per-bucket objects).

        Same conventions as ``windows.py`` -- hour-ending buckets, window = buckets in
        ``(end - hours, end]``, coverage = distinct non-empty buckets -- and pinned against
        ``reference_windows`` by the differential tests. Plain integer math keeps a full
        read (9 variables x 5 windows) around a millisecond of CPU.
        """
        end_us = _micros(end)
        lo = end_us - max(hours.values()) * _HOUR_US
        per_variable: dict[str, list[tuple[int, float]]] = {}
        for field, blob in raw.items():
            variable, field_source = split_field(field)
            if variable not in self._catalog or (source is not None and field_source != source):
                continue
            for ts_us, value in decode_records(blob).items():
                label = bucket_end_micros(ts_us)
                if lo < label <= end_us:
                    per_variable.setdefault(variable, []).append((label, value))

        result: dict[str, dict[str, WindowValue]] = {}
        for name in variable_names(self._catalog):
            agg = self._catalog[name].agg_method
            pairs = sorted(per_variable.get(name, ()))
            labels = [label for label, _ in pairs]
            per_window: dict[str, WindowValue] = {}
            for window, window_len in hours.items():
                chosen = pairs[bisect_right(labels, end_us - window_len * _HOUR_US) :]
                if not chosen:
                    per_window[window] = WindowValue(
                        value=None, observed_hours=0, hours=window_len
                    )
                    continue
                per_window[window] = WindowValue(
                    value=_reduce_values(agg, [v for _, v in chosen]),
                    observed_hours=len({label for label, _ in chosen}),
                    hours=window_len,
                )
            result[name] = per_window
        return result

    def dump_cell(self, cell: str) -> CellState:
        """Everything held for one cell: ``{(variable, source): [(ts_micros, value), ...]}``.

        Used by tests to prove two hot states are identical (e.g. live vs rebuilt).
        """
        self._check_cell(cell)
        raw = self._client.hgetall(self.key(cell)) or {}
        state: CellState = {}
        for field, blob in raw.items():
            records = decode_records(blob)
            state[split_field(field)] = sorted(records.items())
        return state

    def cells_present(self) -> list[str]:
        """Cells that currently have a key (sorted)."""
        prefix_len = len(self._prefix)
        found = []
        for raw_key in self._client.scan_iter(match=f"{self._prefix}*", count=1000):
            text = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
            found.append(text[prefix_len:])
        return sorted(found)

    # ── metrics ──────────────────────────────────────────────────────

    def memory_usage(self, cell: str) -> int | None:
        """Bytes Redis reports for the cell's key (``MEMORY USAGE``); None if absent."""
        self._check_cell(cell)
        used = self._client.memory_usage(self.key(cell))
        return None if used is None else int(used)
