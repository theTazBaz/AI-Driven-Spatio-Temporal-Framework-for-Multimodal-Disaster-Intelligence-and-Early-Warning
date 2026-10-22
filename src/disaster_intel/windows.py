"""Increment 8 - hourly buckets and rolling windows (pure logic, no database).

Conventions (pinned by tests, shared by the cold tier now and the Redis hot tier in Inc 9):

* **Hour-ending buckets.** Bucket ``B`` (an hour-aligned UTC instant) holds every observation
  with ``B - 1h < ts <= B``. A reading stamped exactly 10:00 therefore belongs to the 10:00
  bucket, which matches Open-Meteo's backward-looking hourly values.
* **A window is the last ``hours`` buckets ending at ``as_of`` floored to the hour**:
  buckets ``as_of_h - hours < B <= as_of_h``. Every observation inside has ``ts <= as_of``,
  so a window computed today for a past time T equals the one computed live at T.
* **Aggregation** per variable follows ``Variable.agg_method`` (SUM / AVG / MAX / MIN). AVG is
  the mean of the observations (sum / count), not the mean of bucket means.
* **Coverage** is the fraction of the window's hourly buckets that hold >= 1 observation.
  A window with no observations has ``value=None`` (never 0) and ``coverage=0.0``.

``reference_windows`` is the slow, obviously-correct oracle the database is tested against.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

from disaster_intel.models import AggMethod, Observation, VariableCatalog

HOUR = timedelta(hours=1)
_HOUR_US = 3_600_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ONE_MICROSECOND = timedelta(microseconds=1)

WINDOW_HOURS: Mapping[str, int] = MappingProxyType(
    {"1h": 1, "6h": 6, "24h": 24, "72h": 72, "7d": 168}
)

# The same bucketing, in ClickHouse SQL, for observations.ts (DateTime64(6, 'UTC')).
# Integer arithmetic on purpose: ceil(micros / 1h) * 1h. Tests pin it against bucket_end().
BUCKET_END_SQL = (
    "toDateTime(toUInt32(intDiv(toUnixTimestamp64Micro(ts) + 3599999999, 3600000000) * 3600),"
    " 'UTC')"
)


# ── Time helpers ─────────────────────────────────────────────────────


def _micros(ts: datetime) -> int:
    if ts.tzinfo is None:
        raise ValueError("Naive datetime is not allowed")
    return (ts.astimezone(UTC) - _EPOCH) // _ONE_MICROSECOND


def _from_micros(micros: int) -> datetime:
    return _EPOCH + timedelta(microseconds=micros)


def floor_hour(ts: datetime) -> datetime:
    """Largest hour boundary <= ts (UTC)."""
    return _from_micros(_micros(ts) // _HOUR_US * _HOUR_US)


def bucket_end(ts: datetime) -> datetime:
    """The hour-ending bucket label of ``ts``: smallest hour boundary >= ts (UTC)."""
    return _from_micros(-(-_micros(ts) // _HOUR_US) * _HOUR_US)


# ── Buckets ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Bucket:
    """Aggregate state of one (cell, variable, hour): enough to answer SUM/AVG/MAX/MIN."""

    n: int
    total: float
    minimum: float
    maximum: float

    @classmethod
    def of(cls, values: Iterable[float]) -> Bucket:
        vals = list(values)
        if not vals:
            raise ValueError("A bucket needs at least one value")
        return cls(n=len(vals), total=math.fsum(vals), minimum=min(vals), maximum=max(vals))

    def merge(self, other: Bucket) -> Bucket:
        return Bucket(
            n=self.n + other.n,
            total=math.fsum((self.total, other.total)),
            minimum=min(self.minimum, other.minimum),
            maximum=max(self.maximum, other.maximum),
        )


def bucketize(
    observations: Iterable[Observation], source: str | None = None
) -> dict[str, dict[datetime, Bucket]]:
    """Group the observations of ONE cell into ``{variable: {bucket_end: Bucket}}``.

    Sources are merged unless ``source`` filters to one. Used by the Redis hot tier (Inc 9)
    and by tests that compare the database's bucket table with plain Python.
    """
    grouped: dict[str, dict[datetime, list[float]]] = {}
    for obs in observations:
        if source is not None and obs.source != source:
            continue
        grouped.setdefault(obs.variable, {}).setdefault(bucket_end(obs.timestamp), []).append(
            obs.value
        )
    return {
        variable: {b: Bucket.of(vals) for b, vals in by_bucket.items()}
        for variable, by_bucket in grouped.items()
    }


# ── Windows ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class WindowValue:
    """One rolling-window result: the aggregate plus how much of the window was observed."""

    value: float | None  # None when no observation fell inside the window
    observed_hours: int  # hourly buckets that hold >= 1 observation
    hours: int  # window length in hourly buckets

    @property
    def coverage(self) -> float:
        return self.observed_hours / self.hours


def _reduce(agg: AggMethod, parts: Sequence[Bucket]) -> float:
    if agg is AggMethod.SUM:
        return math.fsum(p.total for p in parts)
    if agg is AggMethod.AVG:
        return math.fsum(p.total for p in parts) / sum(p.n for p in parts)
    if agg is AggMethod.MAX:
        return max(p.maximum for p in parts)
    if agg is AggMethod.MIN:
        return min(p.minimum for p in parts)
    raise ValueError(f"Unsupported aggregation {agg!r}")


def combine_buckets(
    buckets: Mapping[datetime, Bucket], as_of: datetime, hours: int, agg: AggMethod
) -> WindowValue:
    """Combine the <= ``hours`` hourly buckets that end at ``floor_hour(as_of)``."""
    if hours < 1:
        raise ValueError("A window must span at least one hour")
    end = floor_hour(as_of)
    parts = [b for i in range(hours) if (b := buckets.get(end - i * HOUR)) is not None]
    if not parts:
        return WindowValue(value=None, observed_hours=0, hours=hours)
    return WindowValue(value=_reduce(agg, parts), observed_hours=len(parts), hours=hours)


def variable_names(catalog: VariableCatalog) -> list[str]:
    """Catalog variable names in registration order (the catalog has no public iterator)."""
    return list(catalog._variables)


def windows_from_buckets(
    buckets_by_variable: Mapping[str, Mapping[datetime, Bucket]],
    as_of: datetime,
    catalog: VariableCatalog | None = None,
    window_hours: Mapping[str, int] = WINDOW_HOURS,
) -> dict[str, dict[str, WindowValue]]:
    """``{variable: {window_name: WindowValue}}`` for EVERY catalog variable."""
    cat = catalog or VariableCatalog.default()
    return {
        name: {
            window: combine_buckets(
                buckets_by_variable.get(name, {}), as_of, hours, cat[name].agg_method
            )
            for window, hours in window_hours.items()
        }
        for name in variable_names(cat)
    }


def reference_windows(
    observations: Iterable[Observation],
    cell: str,
    as_of: datetime,
    catalog: VariableCatalog | None = None,
    window_hours: Mapping[str, int] = WINDOW_HOURS,
    source: str | None = None,
) -> dict[str, dict[str, WindowValue]]:
    """The oracle: windows straight from raw observations, no buckets involved.

    Deliberately naive (a full scan per variable per window) so it is easy to trust.
    """
    cat = catalog or VariableCatalog.default()
    end = floor_hour(as_of)
    mine = [o for o in observations if o.cell == cell and (source is None or o.source == source)]
    result: dict[str, dict[str, WindowValue]] = {}
    for name in variable_names(cat):
        agg = cat[name].agg_method
        per_window: dict[str, WindowValue] = {}
        for window, hours in window_hours.items():
            start = end - hours * HOUR
            inside = [o for o in mine if o.variable == name and start < o.timestamp <= end]
            if not inside:
                per_window[window] = WindowValue(value=None, observed_hours=0, hours=hours)
                continue
            values = [o.value for o in inside]
            observed = len({bucket_end(o.timestamp) for o in inside})
            if agg is AggMethod.SUM:
                value = math.fsum(values)
            elif agg is AggMethod.AVG:
                value = math.fsum(values) / len(values)
            elif agg is AggMethod.MAX:
                value = max(values)
            else:
                value = min(values)
            per_window[window] = WindowValue(value=value, observed_hours=observed, hours=hours)
        result[name] = per_window
    return result


def format_windows(table: Mapping[str, Mapping[str, WindowValue]]) -> str:
    """Plain-ASCII table for the demo / CLI: value (coverage %) per variable and window."""
    windows = list(next(iter(table.values())))
    lines = ["variable".ljust(24) + "".join(w.rjust(16) for w in windows)]
    for variable, per_window in table.items():
        cells = []
        for w in windows:
            wv = per_window[w]
            text = "-" if wv.value is None else f"{wv.value:.2f} ({wv.coverage:.0%})"
            cells.append(text.rjust(16))
        lines.append(variable.ljust(24) + "".join(cells))
    return "\n".join(lines)
