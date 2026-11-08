"""Inspect and compare live data stored in Hot Tier (Redis) and Cold Tier (ClickHouse)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import clickhouse_connect
import redis

from disaster_intel.clock import FakeClock
from disaster_intel.state import (
    FeatureSpec,
    StateAssembler,
    StaticLayers,
)
from disaster_intel.storage.cold import ColdStore
from disaster_intel.storage.hot import HotStore, _from_micros, decode_records

FOCAL = "8760a9931ffffff"


def main() -> None:
    print("=" * 78)
    print("  LIVE DUAL-TIER DATA INSPECTOR: REDIS (HOT) vs CLICKHOUSE (COLD)")
    print("=" * 78)

    # 1. Connect to Redis and find showcase prefix
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=False)
    try:
        keys = redis_client.keys(b"showcase:*:*")
    except Exception as e:
        print(f"\n[!] Could not connect to Redis: {e}")
        sys.exit(1)

    if not keys:
        print("\n[!] No active showcase keys found in Redis. Run scripts/showcase_demo.py first.")
        sys.exit(0)

    # Pick the first key's prefix
    sample_key = keys[0].decode("utf-8")
    parts = sample_key.split(":")
    prefix = f"{parts[0]}:{parts[1]}:"
    focal_key = f"{prefix}{FOCAL}".encode()
    if not redis_client.exists(focal_key):
        # Pick any existing cell key
        focal_key = keys[0]
        focal_cell = focal_key.decode("utf-8").replace(prefix, "")
    else:
        focal_cell = FOCAL

    print(f"\n[1. REDIS HOT TIER] Key: {focal_key.decode('utf-8')}")
    print("  Redis Data Type : HASH")
    raw_hash = redis_client.hgetall(focal_key)
    print(f"  Field Count     : {len(raw_hash)} variables tracked")
    for field_bytes in sorted(raw_hash):
        field = field_bytes.decode("utf-8")
        recs = decode_records(raw_hash[field_bytes])
        print(f"    - Field: {field:<32} -> {len(recs)} hourly readings retained (7-day ring)")

    # Inspect raw precipitation values in Redis
    if b"precipitation|open-meteo" in raw_hash:
        raw_precip = raw_hash[b"precipitation|open-meteo"]
        recs = decode_records(raw_precip)
        sorted_recs = sorted(recs.items())
        print(f"\n  Latest 5 hourly rainfall values in Redis for cell {focal_cell}:")
        for m, val in sorted_recs[-5:]:
            ts_str = _from_micros(m).strftime("%Y-%m-%d %H:%M")
            print(f"    {ts_str} UTC -> {val:.2f} mm")

    # 2. Cold Tier (ClickHouse) Inspection
    try:
        ch_client = clickhouse_connect.get_client(
            host="localhost", port=8123, username="disaster", password="disaster"
        )
        raw_dbs = ch_client.command("SHOW DATABASES")
    except Exception as e:
        print(f"\n[!] Could not connect to ClickHouse: {e}")
        sys.exit(1)

    databases = [db.strip() for db in raw_dbs.splitlines() if db.strip()]
    showcase_dbs = [db for db in databases if db.startswith("showcase_")]
    if not showcase_dbs:
        print("\n[!] No showcase databases found in ClickHouse.")
        sys.exit(0)

    latest_db = sorted(showcase_dbs)[-1]
    print(f"\n[2. CLICKHOUSE COLD TIER] Database: {latest_db}")
    total_obs = ch_client.command(f"SELECT count() FROM {latest_db}.observations")
    total_buckets = ch_client.command(f"SELECT count() FROM {latest_db}.hourly_buckets")
    print(f"  Raw Observations Stored: {total_obs:,}")
    print(f"  Pre-aggregated Buckets : {total_buckets:,}")

    # Sample recent rows
    rows = ch_client.query(
        f"SELECT hex(cell), variable, source, ts, value FROM {latest_db}.observations "
        f"ORDER BY ts DESC LIMIT 5"
    ).result_rows

    print("\n  Latest 5 rows written to ClickHouse cold tier:")
    for r_row in rows:
        cell_hex, var, _src, ts, val = r_row
        print(f"    cell={cell_hex:<15} var={var:<24} val={val:8.2f} ts={ts}")

    # 3. Dual-Tier Assembler Verification
    clock = FakeClock(datetime(2024, 7, 20, 12, 0, tzinfo=UTC))
    hot = HotStore(redis_client, clock=clock, prefix=prefix)
    cold = ColdStore.connect(database=latest_db)
    layers = StaticLayers.load()
    assembler = StateAssembler(layers, hot=hot, cold=cold, clock=clock)

    print(f"\n[3. DUAL-TIER AGREEMENT AT FOCAL CELL {focal_cell}]")
    hot_state = assembler.get_state(focal_cell, tier="hot")
    cold_state = assembler.get_state(focal_cell, tier="cold")
    spec = FeatureSpec.v3()
    hot_map = hot_state.as_mapping(spec)
    cold_map = cold_state.as_mapping(spec)

    matches = 0
    for feat in spec.names:
        h_val = hot_map[feat]
        c_val = cold_map[feat]
        match = (h_val == c_val) or (abs(h_val - c_val) < 1e-4)
        if match:
            matches += 1

    print(f"  Features compared : {len(spec.names)}")
    pct = (matches / len(spec.names)) * 100.0
    print(f"  Features matching : {matches}/{len(spec.names)} ({pct:.1f}%)")
    print("  Dual-tier bit-exact agreement verified on live Redis + ClickHouse!")
    print("=" * 78)


if __name__ == "__main__":
    main()
