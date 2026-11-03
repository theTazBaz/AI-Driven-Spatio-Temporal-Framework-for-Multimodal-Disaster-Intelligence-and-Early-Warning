"""Showcase demo: REAL ERA5-Land weather for the July-2024 Shirur / Uttara Kannada event,
streamed through the full Phase-I stack (adapter -> H3 fan-out -> bus -> ClickHouse + Redis ->
StateAssembler), then exported as an interactive hexagon map.

    docker compose up -d --wait
    uv run python scripts/showcase_demo.py                 # needs internet (Open-Meteo archive)
    uv run python scripts/showcase_demo.py --cache         # reuse the downloaded payload afterwards

Outputs (in demo_out/):
    showcase_map.html   open in a browser: H3 cells coloured by state, time slider, cell chart
    showcase_metrics.json   every number printed below, for slides

Every figure printed is MEASURED in this run. The weather values are real reanalysis data
(ERA5-Land via Open-Meteo); the static terrain layer is synthetic unless you built the real one
(`make static-layers`). Discharge/SAR/INSAT are not replayed here (no adapter yet) -> NaN.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import h3

from disaster_intel.adapters.open_meteo import OpenMeteoAdapter
from disaster_intel.bus import InMemoryBus
from disaster_intel.clock import FakeClock, SystemClock
from disaster_intel.discretizers import SpatialDiscretizer
from disaster_intel.grid import CoastalGrid
from disaster_intel.models import Observation, RawReading
from disaster_intel.state import (
    FeatureSpec,
    FunctionElevation,
    StateAssembler,
    build_static_layers,
    synthetic_ghats_elevation,
)
from disaster_intel.storage import ColdStore, HotStore, TeeSink, rebuild_hot_from_cold
from disaster_intel.windows import HOUR

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OUT = Path("demo_out")
CACHE_FILE = OUT / "open_meteo_payload.json"


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78)


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round((len(s) - 1) * p / 100)))]


def fetch_readings(adapter: OpenMeteoAdapter, nodes, start: str, end: str, use_cache: bool):
    """Real archive data; optionally cache the raw JSON so a demo works offline later."""
    if use_cache and CACHE_FILE.is_file():
        print(f"  (using cached payload {CACHE_FILE})")
        return adapter.parse_payload(json.loads(CACHE_FILE.read_text()))
    payloads = []
    orig = adapter.parse_payload

    def spy(payload):  # keep the raw JSON so --cache can replay it offline
        payloads.append(payload)
        return orig(payload)

    adapter.parse_payload = spy  # type: ignore[method-assign]
    readings = adapter.fetch_grid_nodes(list(nodes), start_date=start, end_date=end)
    adapter.parse_payload = orig  # type: ignore[method-assign]
    OUT.mkdir(exist_ok=True)
    flat: list = []
    for p in payloads:
        flat.extend(p if isinstance(p, list) else [p])
    CACHE_FILE.write_text(json.dumps(flat))
    return readings


def main() -> None:  # noqa: C901, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=14.98, help="focal point (default: Shirur)")
    ap.add_argument("--lon", type=float, default=74.30)
    ap.add_argument("--ring", type=int, default=3, help="H3 k-ring radius around focal cell")
    ap.add_argument("--start", default="2024-07-08")
    ap.add_argument("--end", default="2024-07-20")
    ap.add_argument("--step-h", type=int, default=6, help="map frame spacing in hours")
    ap.add_argument("--cache", action="store_true", help="reuse downloaded payload if present")
    ap.add_argument("--keep", action="store_true",
                    help="do NOT drop the ClickHouse DB / Redis keys at the end (for live SQL)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    metrics: dict = {}

    # ── 1. Grid + fan-out ────────────────────────────────────────────────────────────
    banner("1. SPATIAL DISCRETIZATION: study region -> H3 resolution-7 grid")
    t0 = time.perf_counter()
    grid = CoastalGrid(use_cache=False)
    build_ms = (time.perf_counter() - t0) * 1000
    disc = SpatialDiscretizer(grid=grid)
    n_cells, n_nodes = len(grid), len(disc.fan_out.unique_nodes)
    print(f"  cells in study region      : {n_cells:,}   (built in {build_ms:.0f} ms)")
    print(f"  mean cell area             : {grid.mean_cell_area():.3f} km^2")
    print(f"  distinct 0.1-deg NWP nodes : {n_nodes}")
    print(f"  cells served per API call  : {n_cells / n_nodes:.1f}")
    print(f"  external calls per cycle   : {n_cells:,} -> {n_nodes}  "
          f"({100 * (1 - n_nodes / n_cells):.1f}% fewer)")
    metrics["grid"] = {"cells": n_cells, "nodes": n_nodes, "build_ms": build_ms}

    focal = h3.latlng_to_cell(args.lat, args.lon, 7)
    if focal not in grid:
        raise SystemExit(f"focal cell {focal} is outside the study grid; pick another point")
    ring = [c for c in h3.grid_disk(focal, args.ring) if c in grid]
    nodes = sorted({disc.fan_out.cell_to_node[c] for c in ring})
    region = sorted({c for n in nodes for c in disc.fan_out.node_to_cells[n]})
    print(f"\n  focal cell {focal} at ({args.lat}, {args.lon})")
    print(f"  k-ring({args.ring}) = {len(ring)} cells -> {len(nodes)} weather nodes "
          f"-> fan-out covers {len(region)} cells")
    metrics["focal"] = {"cell": focal, "ring": len(ring), "nodes": len(nodes),
                        "region_cells": len(region)}

    # ── 2. Real data ─────────────────────────────────────────────────────────────────
    banner("2. REAL TELEMETRY: ERA5-Land hourly reanalysis via the Open-Meteo adapter")
    adapter = OpenMeteoAdapter(clock=SystemClock(), base_url=ARCHIVE_URL)
    t0 = time.perf_counter()
    readings = fetch_readings(adapter, nodes, args.start, args.end, args.cache)
    fetch_s = time.perf_counter() - t0
    print(f"  {len(readings):,} typed RawReadings from {len(nodes)} nodes "
          f"({args.start} .. {args.end}) in {fetch_s:.1f}s")
    print(f"  adapter: requests={adapter.metrics.requests_made} retries={adapter.metrics.retries} "
          f"skipped={dict(adapter.metrics.records_skipped)}")
    metrics["fetch"] = {"raw_readings": len(readings), "seconds": fetch_s}

    # ── 3. Stream: discretize -> bus -> ClickHouse + Redis ───────────────────────────
    banner("3. STREAMING: adapter -> H3 fan-out -> bus -> ClickHouse (record) + Redis (hot)")
    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=UTC) + timedelta(hours=24)
    clock = FakeClock(start_dt)
    hot = HotStore.connect(clock=clock, prefix=f"showcase:{uuid.uuid4().hex[:8]}:")
    database = f"showcase_{uuid.uuid4().hex[:8]}"
    cold = ColdStore.connect(database=database)
    cold.migrate()
    spec = FeatureSpec.v1()
    layers = build_static_layers(region, FunctionElevation(synthetic_ghats_elevation))
    assembler = StateAssembler(layers, hot=hot, cold=cold, clock=clock)
    sink = TeeSink(cold, hot)
    bus = InMemoryBus()

    by_hour: dict[datetime, list[RawReading]] = defaultdict(list)
    for r in readings:
        by_hour[r.timestamp].append(r)
    hours = sorted(by_hour)

    batch_ms: list[float] = []
    obs_total = stored_total = 0
    live_focal: dict[datetime, object] = {}
    t_all = time.perf_counter()
    try:
        for ts in hours:
            if clock.now() < ts:
                clock.advance_to(ts)
            t0 = time.perf_counter()
            obs_n = 0
            for rr in by_hour[ts]:
                for ob in disc.discretize(rr):
                    bus.publish(f"telemetry.{rr.source}", ob.cell, ob)
                    obs_n += 1
            batch: list[Observation] = []
            while True:
                msgs = bus.poll(f"telemetry.{adapter.source_name}", "sink", 5000, True)
                if not msgs:
                    break
                batch.extend(m.payload for m in msgs)
            stored, _dups = sink.write(batch)
            batch_ms.append((time.perf_counter() - t0) * 1000)
            obs_total += obs_n
            stored_total += stored
            live_focal[ts] = assembler.get_state(focal)  # what the system "knew" at that hour
        wall = time.perf_counter() - t_all
        print(f"  hours streamed            : {len(hours)}")
        print(f"  observations produced     : {obs_total:,}  (from {len(readings):,} readings; "
              f"x{obs_total / max(1, len(readings)):.1f} fan-out)")
        print(f"  stored (cold+hot, deduped): {stored_total:,}")
        print(f"  sustained throughput      : {obs_total / wall:,.0f} obs/s  "
              f"(incl. ClickHouse + Redis writes, real network to localhost containers)")
        print(f"  per-hour batch latency    : p50 {pct(batch_ms, 50):.1f} ms | "
              f"p95 {pct(batch_ms, 95):.1f} ms | p99 {pct(batch_ms, 99):.1f} ms")
        metrics["stream"] = {"obs": obs_total, "stored": stored_total, "obs_per_s": obs_total / wall,
                             "batch_ms_p50": pct(batch_ms, 50), "batch_ms_p95": pct(batch_ms, 95)}

        # ── 4. Idempotency ───────────────────────────────────────────────────────────
        banner("4. EXACTLY-ONCE: replay the last 24 hours of the feed (a retried upload)")
        replay = []
        for ts in hours[-24:]:
            for rr in by_hour[ts]:
                replay.extend(disc.discretize(rr))
        s, d = sink.write(replay)
        print(f"  re-delivered {len(replay):,} observations -> stored {s}, duplicates dropped {d}")
        print(f"  ClickHouse rows after FINAL merge: {cold.count():,} (unchanged)")
        metrics["idempotency"] = {"replayed": len(replay), "stored": s, "dropped": d}

        # ── 5. Hot vs cold + latency ─────────────────────────────────────────────────
        banner("5. DUAL-TIER STORE: Redis (hot) and ClickHouse (cold) must agree")
        cmp_ = assembler.compare_tiers(focal)
        print(f"  focal cell, now: hot == cold across {len(spec)} features -> {cmp_.agrees}")
        lat_hot, lat_cold = [], []
        for _ in range(200):
            t0 = time.perf_counter()
            assembler.get_state(focal, tier="hot")
            lat_hot.append((time.perf_counter() - t0) * 1000)
        for _ in range(30):
            t0 = time.perf_counter()
            assembler.get_state(focal, tier="cold")
            lat_cold.append((time.perf_counter() - t0) * 1000)
        print(f"  Redis      get_state: p50 {pct(lat_hot, 50):.2f} ms | p95 {pct(lat_hot, 95):.2f} ms")
        print(f"  ClickHouse get_state: p50 {pct(lat_cold, 50):.2f} ms | p95 {pct(lat_cold, 95):.2f} ms")
        mem = hot.memory_usage(focal) or 0
        print(f"  Redis memory for this cell: {mem:,} B -> whole region ~{mem * n_cells / 1e6:.1f} MB")
        metrics["tiers"] = {"agree": cmp_.agrees, "hot_p50": pct(lat_hot, 50),
                            "cold_p50": pct(lat_cold, 50), "bytes_per_cell": mem}

        # ── 6. Deterministic rollback ────────────────────────────────────────────────
        banner("6. DETERMINISTIC ROLLBACK: yesterday's state, recomputed today, is identical")
        checks = hours[72::12]
        ok = 0
        tiers_used: dict[str, int] = defaultdict(int)
        for ts in checks:
            again = assembler.get_state(focal, as_of=ts)
            tiers_used[again.tier] += 1
            if again.same_values(live_focal[ts], rel_tol=1e-4, abs_tol=1e-4):  # type: ignore[arg-type]
                ok += 1
        print(f"  {ok}/{len(checks)} past instants reproduce the live vector "
              f"(tiers used: {dict(tiers_used)}; tolerance 1e-4 = float32 quantisation)")
        metrics["rollback"] = {"checked": len(checks), "identical": ok, "tiers": dict(tiers_used)}

        # ── 7. Disaster recovery ─────────────────────────────────────────────────────
        banner("7. FAILURE RECOVERY: wipe Redis, rebuild from ClickHouse")
        before = {c: hot.dump_cell(c) for c in ring[:10]}
        t0 = time.perf_counter()
        report = rebuild_hot_from_cold(hot, cold, region, wipe=True)
        print("  " + report.summary())
        same = all(hot.dump_cell(c) == before[c] for c in before)
        print(f"  rebuilt in {time.perf_counter() - t0:.1f}s; sampled cells byte-identical: {same}")
        metrics["rebuild"] = {"identical": same}

        # ── 8. Frames for the map ────────────────────────────────────────────────────
        banner("8. STATE VECTORS: score every cell at every frame (training-matrix preview)")
        frames_t = [ts for ts in hours if (ts - hours[0]) >= timedelta(hours=72)][:: args.step_h]
        names = {"cs": "compound_saturation", "p24": "precipitation__24h",
                 "p72": "precipitation__72h", "sm": "soil_moisture_rootzone__24h"}
        frames = []
        t0 = time.perf_counter()
        n_vec = 0
        for ts in frames_t:
            row = {}
            for c in region:
                m = assembler.get_state(c, as_of=ts).as_mapping(spec)
                n_vec += 1
                row[c] = [None if math.isnan(m[names[k]]) else round(m[names[k]], 3)
                          for k in ("cs", "p24", "p72", "sm")]
            frames.append({"t": ts.strftime("%Y-%m-%d %H:%M"), "v": row})
        dt = time.perf_counter() - t0
        print(f"  {n_vec:,} state vectors ({len(spec)}-d) in {dt:.1f}s = {dt / n_vec * 1000:.2f} ms "
              f"each; full-region sweep of {n_cells:,} cells would take ~{dt / n_vec * n_cells:.0f}s")
        peak = max(
            ((f["t"], c, v[0]) for f in frames for c, v in f["v"].items() if v[0] is not None),
            key=lambda x: x[2],
            default=None,
        )
        if peak:
            print(f"  highest compound_saturation: {peak[2]:.1f} in cell {peak[1]} at {peak[0]} UTC")
        metrics["frames"] = {"vectors": n_vec, "ms_each": dt / n_vec * 1000, "peak": peak}

        cells_geo = {c: [[lat, lon] for lat, lon in h3.cell_to_boundary(c)] for c in region}
        focal_series = [
            {"t": ts.strftime("%Y-%m-%d %H:%M"),
             "p1": (lambda m: None if math.isnan(m) else m)(
                 live_focal[ts].as_mapping(spec)["precipitation__1h"]),  # type: ignore[attr-defined]
             "p24": (lambda m: None if math.isnan(m) else m)(
                 live_focal[ts].as_mapping(spec)["precipitation__24h"])}  # type: ignore[attr-defined]
            for ts in hours
        ]
        (OUT / "showcase_map.html").write_text(
            HTML.replace("__DATA__", json.dumps(
                {"cells": cells_geo, "frames": frames, "focal": focal, "series": focal_series}))
        )
        (OUT / "showcase_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        banner("DONE")
        print(f"  open {OUT / 'showcase_map.html'} in a browser (needs internet for map tiles)")
        print(f"  metrics for slides: {OUT / 'showcase_metrics.json'}")
    finally:
        if args.keep:
            print(f"\n  --keep: ClickHouse database = {database}   Redis prefix = {hot._prefix}")
        else:
            hot.clear()
            cold.client.command(f"DROP DATABASE IF EXISTS {database}")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>WorldState - Shirur 2024</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
body{margin:0;font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0}
#top{padding:10px 16px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
#map{height:62vh}#bot{padding:8px 16px}
select,input{font-size:15px}#t{font-weight:700;font-size:18px;min-width:170px}
#legend{font-size:12px;opacity:.8} #c{width:100%;height:150px;background:#111c33;border-radius:6px}
</style></head><body>
<div id="top"><b>H3-res7 WorldState: real ERA5-Land rain through the pipeline</b>
<select id="m"><option value="0">compound_saturation (soil x 72h rain)</option>
<option value="1">precipitation 24h (mm)</option><option value="2">precipitation 72h (mm)</option>
<option value="3">root-zone soil moisture 24h</option></select>
<span id="t"></span><input id="s" type="range" min="0" value="0" style="width:340px">
<span id="legend"></span></div><div id="map"></div>
<div id="bot"><div id="cl">Focal cell (white outline)</div>
<svg id="c" viewBox="0 0 1000 150" preserveAspectRatio="none"></svg></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const D=__DATA__, F=D.frames, M=L.map('map').setView([0,0],10);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',{maxNativeZoom:16,maxZoom:18,attribution:'Tiles &copy; Esri'}).addTo(M);
const polys={};let bounds=[];
for(const [c,ring] of Object.entries(D.cells)){const p=L.polygon(ring,{weight:c===D.focal?3:1,
 color:c===D.focal?'#fff':'#334155',fillOpacity:.75}).addTo(M);p.bindTooltip(c);polys[c]=p;bounds.push(...ring)}
M.fitBounds(bounds);
const s=document.getElementById('s'),m=document.getElementById('m'),t=document.getElementById('t');
s.max=F.length-1;
function rng(k){let mx=0;F.forEach(f=>Object.values(f.v).forEach(v=>{if(v[k]!=null&&v[k]>mx)mx=v[k]}));return mx||1}
function col(x,mx){if(x==null)return '#475569';const r=Math.min(1,x/mx);
 return `hsl(${(1-r)*220},85%,${30+r*25}%)`}
function draw(){const k=+m.value,f=F[+s.value],mx=rng(k);t.textContent=f.t+' UTC';
 for(const [c,p] of Object.entries(polys)){const v=f.v[c][k];p.setStyle({fillColor:col(v,mx)});
  p.setTooltipContent(c+'<br>'+(v==null?'no data':v.toFixed(2)))}
 document.getElementById('legend').textContent='blue = low, red = high (max '+mx.toFixed(1)+'); grey = no data';
 chart()}
function chart(){const k=+m.value,S=F.map(f=>f.v[D.focal][k]),n=S.length,
 mx=Math.max(1e-9,...S.filter(x=>x!=null));
 const pts=S.map((x,i)=>(i/(n-1)*1000)+','+(145-(x||0)/mx*135)).join(' '),X=+s.value/(n-1)*1000;
 document.getElementById('cl').textContent='Focal cell (white outline): '+m.options[m.selectedIndex].text+
  ', every frame (peak '+mx.toFixed(2)+')';
 document.getElementById('c').innerHTML=`<polyline points="${pts}" fill="none" stroke="#f59e0b" stroke-width="2"/>
 <line x1="${X}" x2="${X}" y1="0" y2="150" stroke="#fff" stroke-dasharray="4"/>`}
s.oninput=m.onchange=draw;draw();
</script></body></html>"""

if __name__ == "__main__":
    main()
