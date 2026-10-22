"""Increment 7 — ClickHouse schema migrations.

Every statement is ``IF NOT EXISTS`` *and* every applied version is recorded in
``schema_migrations``, so running migrations again is a no-op (plan.md Increment 7).

Schema decisions (plan.md §11 storage budget):

* ``cell UInt64``            H3 index as an integer: 8 B instead of a 15-char string, and
                             ClickHouse's ``h3*`` functions take UInt64 directly.
* ``LowCardinality(String)`` for ``variable`` / ``source`` (a handful of distinct values).
* ``ts DateTime64(6,'UTC')`` microsecond-exact so the idempotency key can be recomputed.
* ``value Float32``          half the bytes of Float64 (plan.md: "Float32 values").
* No ``unit`` column         units are a function of ``variable`` (VariableCatalog).
* No ``idempotency_key``     it is a hash of (cell, variable, source, ts): derivable, and an
                             8-byte hash is incompressible, which would dwarf the other columns.
* ``ORDER BY (cell, variable, source, ts)`` doubles as the ReplacingMergeTree dedup key, and
  keeps each series contiguous so Delta/DoubleDelta/Gorilla + ZSTD compress it well.
  (plan.md lists (cell, variable, time); ``source`` is added because two sources reporting
  the same variable at the same instant are different facts, not duplicates.)
* ``PARTITION BY toYYYYMM(ts)`` monthly parts: cheap historical backfills and drops.

Migration 2 (Increment 8) adds ``hourly_buckets``: one pre-aggregated row per
(cell, variable, source, hour-ending bucket) so a rolling window is a combine of <= 168 rows
instead of a scan of raw observations. It is a ``ReplacingMergeTree(version)``: ``ColdStore``
recomputes every bucket a write touched from ``observations FINAL`` and inserts it with a
newer ``version``, so duplicates, revised values and late arrivals can never double-count
(a plain SummingMergeTree materialized view would add a duplicate insert's value twice).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def validate_identifier(name: str, what: str = "database") -> str:
    """Reject anything that is not a plain SQL identifier (the name is interpolated)."""
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        raise ValueError(f"Invalid {what} name: {name!r}")
    return name


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


def get_migrations(database: str) -> tuple[Migration, ...]:
    """All migrations for ``database``, in the order they must be applied."""
    db = validate_identifier(database)
    return (
        Migration(
            version=1,
            name="create_observations",
            statements=(
                f"""
CREATE TABLE IF NOT EXISTS {db}.observations
(
    cell     UInt64                    CODEC(Delta, ZSTD(3)),
    variable LowCardinality(String),
    source   LowCardinality(String),
    ts       DateTime64(6, 'UTC')      CODEC(DoubleDelta, ZSTD(3)),
    value    Float32                   CODEC(Gorilla, ZSTD(3))
)
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (cell, variable, source, ts)
""".strip(),
            ),
        ),
        Migration(
            version=2,
            name="create_hourly_buckets",
            statements=(
                f"""
CREATE TABLE IF NOT EXISTS {db}.hourly_buckets
(
    cell      UInt64               CODEC(Delta, ZSTD(3)),
    variable  LowCardinality(String),
    source    LowCardinality(String),
    bucket    DateTime('UTC')      CODEC(DoubleDelta, ZSTD(3)),
    n_obs     UInt32               CODEC(ZSTD(3)),
    sum_value Float64              CODEC(Gorilla, ZSTD(3)),
    min_value Float32              CODEC(Gorilla, ZSTD(3)),
    max_value Float32              CODEC(Gorilla, ZSTD(3)),
    version   UInt64               CODEC(DoubleDelta, ZSTD(3))
)
ENGINE = ReplacingMergeTree(version)
PARTITION BY toYYYYMM(bucket)
ORDER BY (cell, variable, source, bucket)
""".strip(),
            ),
        ),
    )


def bootstrap_statements(database: str) -> tuple[str, ...]:
    """Statements run before any migration: the database and the version-tracking table."""
    db = validate_identifier(database)
    return (
        f"CREATE DATABASE IF NOT EXISTS {db}",
        f"""
CREATE TABLE IF NOT EXISTS {db}.schema_migrations
(
    version    UInt32,
    name       String,
    applied_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY version
""".strip(),
    )
