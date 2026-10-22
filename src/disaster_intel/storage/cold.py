"""Increment 7 — ClickHouse cold store (``ColdStore``).

The cold store is the permanent, compressed record of every ``Observation``
(Objective 2, cold tier). Two rules shape the API:

1. **Reads always deduplicate.** ``ReplacingMergeTree`` only removes duplicates when
   background merges happen, so a plain ``SELECT`` can return the same observation twice.
   Every read here uses ``FINAL``; ``raw_count()`` is the single deliberate exception and
   exists so a test can pin the pre-merge behaviour.
2. **What is derivable is not stored.** The unit comes from the ``VariableCatalog`` and the
   idempotency key is recomputed from (cell, variable, source, ts); see ``migrations.py``.

Floating-point note: values are stored as ``Float32`` (plan.md storage budget), so a value
read back equals ``to_float32(original)``, not necessarily the original float64.
"""

from __future__ import annotations

import math
import os
import struct
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import h3

from disaster_intel.models import (
    Observation,
    VariableCatalog,
    compute_idempotency_key,
)
from disaster_intel.storage.migrations import (
    bootstrap_statements,
    get_migrations,
    validate_identifier,
)
from disaster_intel.windows import (
    BUCKET_END_SQL,
    HOUR,
    WINDOW_HOURS,
    Bucket,
    WindowValue,
    bucket_end,
    floor_hour,
    windows_from_buckets,
)

COLUMNS: tuple[str, ...] = ("cell", "variable", "source", "ts", "value")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ONE_MICROSECOND = timedelta(microseconds=1)
_REFRESH_CELLS_PER_STATEMENT = 2_000  # keeps each statement far below max_query_size (256 KB)


class ClickHouseClient(Protocol):
    """The slice of ``clickhouse_connect``'s client that ``ColdStore`` uses."""

    def command(self, cmd: str, parameters: Any = None) -> Any: ...

    def query(self, query: str, parameters: Any = None) -> Any: ...

    def insert(
        self,
        table: str,
        data: Any,
        column_names: Any = None,
        database: str | None = None,
    ) -> Any: ...


# ── Pure conversions (unit-tested without a database) ───────────────


def resolve_credentials(environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    """ClickHouse (user, password) from CLICKHOUSE_USER / CLICKHOUSE_PASSWORD.

    The defaults match the local-development user that ``docker-compose.yml`` creates.
    They are NOT secrets; override them through the environment anywhere else.
    """
    env = os.environ if environ is None else environ
    return env.get("CLICKHOUSE_USER", "disaster"), env.get("CLICKHOUSE_PASSWORD", "disaster")


def to_float32(value: float) -> float:
    """Round a float to what a ClickHouse ``Float32`` column will hand back.

    Raises ``ValueError`` if the value does not fit. (Some Python versions raise
    ``OverflowError`` from ``struct.pack``, others silently round to infinity; both are
    rejected so an ``inf`` can never reach the database.)
    """
    try:
        rounded = float(struct.unpack("f", struct.pack("f", value))[0])
    except OverflowError as exc:
        raise ValueError(f"Value {value!r} does not fit in Float32") from exc
    if math.isfinite(value) and not math.isfinite(rounded):
        raise ValueError(f"Value {value!r} does not fit in Float32")
    return rounded


def observation_to_row(obs: Observation, catalog: VariableCatalog) -> list[Any]:
    """Validate ``obs`` and convert it to a row in ``COLUMNS`` order."""
    if obs.variable not in catalog:
        raise ValueError(f"Unknown variable {obs.variable!r}; register it in VariableCatalog")
    canonical = catalog[obs.variable].canonical_unit
    if obs.unit != canonical:
        raise ValueError(
            f"Observation unit {obs.unit!r} for {obs.variable!r} is not the canonical "
            f"unit {canonical!r}; the cold store does not persist units"
        )
    if not isinstance(obs.cell, str) or not h3.is_valid_cell(obs.cell):
        raise ValueError(f"Invalid H3 cell {obs.cell!r}")
    if obs.timestamp.tzinfo is None:
        raise ValueError("Naive datetime is not allowed")
    return [
        int(obs.cell, 16),
        obs.variable,
        obs.source,
        obs.timestamp.astimezone(UTC),
        to_float32(obs.value),
    ]


def _to_utc(ts: datetime | int) -> datetime:
    """Normalise whatever the driver hands back for ``ts`` to an aware UTC datetime.

    * ``int``            microseconds since the Unix epoch (what ``_select`` asks for);
    * naive ``datetime`` the column is UTC, so the wall-clock value IS UTC. Calling
                         ``astimezone`` on it would wrongly treat it as local time (a
                         -05:30 shift on an India-time machine);
    * aware ``datetime`` converted to UTC.
    """
    if isinstance(ts, int):
        return _EPOCH + timedelta(microseconds=ts)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def row_to_observation(row: Sequence[Any], catalog: VariableCatalog) -> Observation:
    """Inverse of ``observation_to_row``; recomputes unit and idempotency key."""
    cell_int, variable, source, ts, value = row
    cell = format(int(cell_int), "x")
    timestamp = _to_utc(ts)
    return Observation(
        cell=cell,
        source=str(source),
        variable=str(variable),
        value=float(value),
        unit=catalog[str(variable)].canonical_unit,
        timestamp=timestamp,
        idempotency_key=compute_idempotency_key(cell, str(variable), str(source), timestamp),
    )


def _micros(dt: datetime) -> int:
    return (dt.astimezone(UTC) - _EPOCH) // _ONE_MICROSECOND


# ── Result types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class StorageStats:
    """Physical size of the observations table (active parts only)."""

    rows: int
    compressed_bytes: int
    uncompressed_bytes: int

    @property
    def bytes_per_row(self) -> float:
        return self.compressed_bytes / self.rows if self.rows else 0.0

    @property
    def uncompressed_bytes_per_row(self) -> float:
        return self.uncompressed_bytes / self.rows if self.rows else 0.0


@dataclass(frozen=True)
class CorrectnessReport:
    """Cold-correctness metric: stored records vs the copy kept at arrival."""

    total: int  # records in the arrival-time reference
    matched: int  # present in cold store with identical content
    missing: int  # in reference, absent from cold store
    mismatched: int  # key present, but content differs

    @property
    def rate(self) -> float:
        return 100.0 if self.total == 0 else round(self.matched / self.total * 100.0, 4)


# ── The store ────────────────────────────────────────────────────────


class ColdStore:
    """ClickHouse-backed permanent observation store.

    Mirrors the in-memory sink used in Increment 6 (``write``, ``count``, ``get_by_cell``)
    except ``contains(idempotency_key)``: the key is deliberately not stored, so use
    ``read_cell`` / ``cold_correctness`` instead. ``write`` cannot tell duplicates from new
    rows at insert time, so it returns ``(rows_written, 0)``; duplicates collapse on read.
    """

    name: str = "clickhouse_cold_store"

    def __init__(
        self,
        client: ClickHouseClient,
        database: str = "disaster_intel",
        catalog: VariableCatalog | None = None,
        batch_size: int = 10_000,
        maintain_buckets: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._maintain_buckets = maintain_buckets
        self._client = client
        self._database = validate_identifier(database)
        self._catalog = catalog or VariableCatalog.default()
        self._batch_size = batch_size

    @classmethod
    def connect(
        cls,
        host: str = "localhost",
        port: int = 8123,
        username: str | None = None,
        password: str | None = None,
        database: str = "disaster_intel",
        **kwargs: Any,
    ) -> ColdStore:
        """Connect over ClickHouse's HTTP interface (the port docker-compose publishes)."""
        import clickhouse_connect

        env_user, env_password = resolve_credentials()
        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=env_user if username is None else username,
            password=env_password if password is None else password,
        )
        return cls(client, database=database, **kwargs)

    # ── properties ───────────────────────────────────────────────────

    @property
    def client(self) -> ClickHouseClient:
        return self._client

    @property
    def database(self) -> str:
        return self._database

    @property
    def _table(self) -> str:
        return f"{self._database}.observations"

    # ── migrations ───────────────────────────────────────────────────

    def migrate(self) -> list[int]:
        """Apply pending migrations; returns the versions applied (``[]`` on a re-run)."""
        for statement in bootstrap_statements(self._database):
            self._client.command(statement)
        rows = self._client.query(
            f"SELECT version FROM {self._database}.schema_migrations ORDER BY version"
        ).result_rows
        applied = {int(r[0]) for r in rows}
        newly_applied: list[int] = []
        for migration in get_migrations(self._database):
            if migration.version in applied:
                continue
            for statement in migration.statements:
                self._client.command(statement)
            self._client.insert(
                f"{self._database}.schema_migrations",
                [[migration.version, migration.name]],
                column_names=["version", "name"],
            )
            newly_applied.append(migration.version)
        return newly_applied

    # ── writes ───────────────────────────────────────────────────────

    def write(self, observations: Sequence[Observation]) -> tuple[int, int]:
        """Insert observations in batches. All are validated before any is inserted.

        Unless ``maintain_buckets=False`` (bulk backfill: call ``rebuild_buckets()`` once at
        the end instead), every hourly bucket the write touched is then recomputed.
        """
        rows = [observation_to_row(o, self._catalog) for o in observations]
        for start in range(0, len(rows), self._batch_size):
            self._client.insert(
                self._table,
                rows[start : start + self._batch_size],
                column_names=list(COLUMNS),
            )
        if rows and self._maintain_buckets:
            self._refresh_buckets(rows)
        return len(rows), 0

    def optimize(self) -> None:
        """Force a merge so duplicates are physically removed (tests, maintenance)."""
        self._client.command(f"OPTIMIZE TABLE {self._table} FINAL")

    # ── hourly buckets & rolling windows (Increment 8) ───────────────

    @property
    def _bucket_table(self) -> str:
        return f"{self._database}.hourly_buckets"

    def _bucket_insert_sql(self, where: str) -> str:
        """Recompute buckets from the DEDUPLICATED observations and insert them.

        ``version`` is the server clock in nanoseconds, so a later recomputation always wins
        in the ReplacingMergeTree (and reads use FINAL).
        """
        return (
            f"INSERT INTO {self._bucket_table} "
            "(cell, variable, source, bucket, n_obs, sum_value, min_value, max_value, version) "
            f"SELECT cell, variable, source, {BUCKET_END_SQL} AS bucket, count(), sum(value), "
            "min(value), max(value), toUInt64(toUnixTimestamp64Nano(now64(9))) "
            f"FROM {self._table} FINAL WHERE {where} "
            "GROUP BY cell, variable, source, bucket"
        )

    def _refresh_buckets(self, rows: Sequence[Sequence[Any]]) -> None:
        """Recompute every bucket that ``rows`` can have changed (late data, duplicates, edits)."""
        cells = sorted({int(r[0]) for r in rows})
        variables = sorted({validate_identifier(str(r[1]), "variable") for r in rows})
        ends = [bucket_end(r[3]) for r in rows]
        lo, hi = _micros(min(ends) - HOUR), _micros(max(ends))  # (lo, hi]: ts of those buckets
        quoted = ", ".join(f"'{v}'" for v in variables)
        for i in range(0, len(cells), _REFRESH_CELLS_PER_STATEMENT):
            chunk = ", ".join(str(c) for c in cells[i : i + _REFRESH_CELLS_PER_STATEMENT])
            where = (
                f"cell IN ({chunk}) AND variable IN ({quoted}) "
                f"AND ts > fromUnixTimestamp64Micro({lo}, 'UTC') "
                f"AND ts <= fromUnixTimestamp64Micro({hi}, 'UTC')"
            )
            self._client.command(self._bucket_insert_sql(where))

    def rebuild_buckets(self) -> None:
        """Recompute ALL buckets (after a ``maintain_buckets=False`` backfill, or to repair)."""
        self._client.command(self._bucket_insert_sql("1"))

    def read_buckets(
        self,
        cell: str,
        start: datetime,
        end: datetime,
        source: str | None = None,
    ) -> dict[str, dict[datetime, Bucket]]:
        """``{variable: {bucket_end: Bucket}}`` for ``start < bucket <= end`` (hour-aligned).

        Sources are merged unless ``source`` selects one.
        """
        if not h3.is_valid_cell(cell):
            raise ValueError(f"Invalid H3 cell {cell!r}")
        lo, hi = _micros(start) // 1_000_000, _micros(end) // 1_000_000
        where = [
            f"cell = {int(cell, 16)}",
            f"bucket > toDateTime({lo}, 'UTC')",
            f"bucket <= toDateTime({hi}, 'UTC')",
        ]
        params: dict[str, Any] = {}
        if source is not None:
            where.append("source = {source:String}")
            params["source"] = source
        sql = (
            "SELECT variable, toUnixTimestamp(bucket) AS b, sum(n_obs), sum(sum_value), "
            "min(min_value), max(max_value) "
            f"FROM {self._bucket_table} FINAL WHERE {' AND '.join(where)} "
            "GROUP BY variable, b"
        )
        result = self._client.query(sql, parameters=params or None)
        buckets: dict[str, dict[datetime, Bucket]] = {}
        for variable, b, n, total, minimum, maximum in result.result_rows:
            buckets.setdefault(str(variable), {})[_EPOCH + timedelta(seconds=int(b))] = Bucket(
                n=int(n), total=float(total), minimum=float(minimum), maximum=float(maximum)
            )
        return buckets

    def windows(
        self,
        cell: str,
        as_of: datetime,
        source: str | None = None,
        window_hours: Mapping[str, int] | None = None,
    ) -> dict[str, dict[str, WindowValue]]:
        """Rolling windows (1h/6h/24h/72h/7d) for every catalog variable of one cell.

        ``{variable: {window: WindowValue(value, observed_hours, hours)}}``; ``.coverage`` is
        the observed fraction of the window. See ``disaster_intel.windows`` for the exact
        bucket/window conventions.
        """
        hours = dict(WINDOW_HOURS if window_hours is None else window_hours)
        end = floor_hour(as_of)
        buckets = self.read_buckets(cell, end - max(hours.values()) * HOUR, end, source)
        return windows_from_buckets(buckets, end, self._catalog, hours)

    # ── reads (all deduplicate with FINAL) ───────────────────────────

    def count(self) -> int:
        """Number of unique observations."""
        return int(
            self._client.query(f"SELECT count() FROM {self._table} FINAL").result_rows[0][0]
        )

    def raw_count(self) -> int:
        """Physical row count INCLUDING not-yet-merged duplicates. For tests/diagnostics only."""
        return int(self._client.query(f"SELECT count() FROM {self._table}").result_rows[0][0])

    def read_cell(
        self,
        cell: str,
        variable: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Observation]:
        """Observations for one cell, oldest first. ``start`` inclusive, ``end`` exclusive."""
        if not h3.is_valid_cell(cell):
            raise ValueError(f"Invalid H3 cell {cell!r}")
        where = [f"cell = {int(cell, 16)}"]
        params: dict[str, Any] = {}
        if variable is not None:
            where.append("variable = {variable:String}")
            params["variable"] = variable
        if start is not None:
            where.append(f"ts >= fromUnixTimestamp64Micro({_micros(start)}, 'UTC')")
        if end is not None:
            where.append(f"ts < fromUnixTimestamp64Micro({_micros(end)}, 'UTC')")
        return self._select(" AND ".join(where), params)

    def get_by_cell(self, cell: str) -> list[Observation]:
        return self.read_cell(cell)

    def read_all(self) -> list[Observation]:
        """Every unique observation (small data sets / tests; do not use on a year of data)."""
        return self._select("1", {})

    def _select(self, where: str, params: dict[str, Any]) -> list[Observation]:
        sql = (
            "SELECT cell, variable, source, toUnixTimestamp64Micro(ts) AS ts_us, value "
            f"FROM {self._table} FINAL "
            f"WHERE {where} ORDER BY ts, variable, source, cell"
        )
        result = self._client.query(sql, parameters=params or None)
        return [row_to_observation(r, self._catalog) for r in result.result_rows]

    # ── H3 in the database ───────────────────────────────────────────

    def parent_cells(self, cells: Sequence[str], parent_res: int) -> dict[str, str]:
        """ClickHouse's ``h3ToParent`` for each cell (compared against ``h3`` in tests)."""
        if not cells:
            return {}
        ints = ", ".join(str(int(c, 16)) for c in cells)
        sql = (
            "SELECT h3ToString(c), h3ToString(h3ToParent(c, "
            f"{int(parent_res)})) FROM (SELECT arrayJoin([{ints}]) AS c)"
        )
        return {str(a): str(b) for a, b in self._client.query(sql).result_rows}

    def count_by_parent(self, parent_res: int) -> dict[str, int]:
        """Unique observations rolled up to their H3 parent cell at ``parent_res``."""
        sql = (
            f"SELECT h3ToString(h3ToParent(cell, {int(parent_res)})) AS parent, count() "
            f"FROM {self._table} FINAL GROUP BY parent"
        )
        return {str(p): int(n) for p, n in self._client.query(sql).result_rows}

    # ── metrics ──────────────────────────────────────────────────────

    def storage_stats(self) -> StorageStats:
        """Bytes on disk for the observations table, from ``system.parts``."""
        sql = (
            "SELECT sum(rows), sum(data_compressed_bytes), sum(data_uncompressed_bytes) "
            "FROM system.parts "
            f"WHERE database = '{self._database}' AND table = 'observations' AND active"
        )
        rows, compressed, uncompressed = self._client.query(sql).result_rows[0]
        return StorageStats(int(rows or 0), int(compressed or 0), int(uncompressed or 0))

    def cold_correctness(self, reference: Iterable[Observation]) -> CorrectnessReport:
        """Compare the store with a copy kept at arrival (Objective 2 correctness metric).

        A record matches when its idempotency key exists in the store and every stored
        field equals the reference (value compared after Float32 quantisation).
        """
        stored = {o.idempotency_key: o for o in self.read_all()}
        seen: Counter[str] = Counter()
        for ref in reference:
            seen["total"] += 1
            got = stored.get(ref.idempotency_key)
            if got is None:
                seen["missing"] += 1
            elif (
                got.cell,
                got.variable,
                got.source,
                got.unit,
                got.timestamp,
                got.value,
            ) == (
                ref.cell,
                ref.variable,
                ref.source,
                ref.unit,
                ref.timestamp,
                to_float32(ref.value),
            ):
                seen["matched"] += 1
            else:
                seen["mismatched"] += 1
        return CorrectnessReport(
            total=seen["total"],
            matched=seen["matched"],
            missing=seen["missing"],
            mismatched=seen["mismatched"],
        )
