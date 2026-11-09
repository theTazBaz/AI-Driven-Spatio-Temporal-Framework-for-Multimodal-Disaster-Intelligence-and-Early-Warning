"""Historical disaster replay pre-computation and caching engine.

Generates complete replay bundles (H3 cells, geomorphology, temporal frames,
and multi-horizon time series) for all verified historical disaster events:
1. 2024 Shirur NH-66 Hill Collapse and Mudslide (Uttara Kannada)
2. 2018 Kodagu Cloudburst Landslides (Western Ghats)
3. 2021 Uttara Kannada Gangavali River Flood Inundation
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import h3

from disaster_intel.adapters.open_meteo import OpenMeteoAdapter
from disaster_intel.clock import FakeClock
from disaster_intel.discretizers import SpatialDiscretizer
from disaster_intel.grid import CoastalGrid
from disaster_intel.ground_truth.catalog import GroundTruthCatalog
from disaster_intel.models import Observation, RawReading
from disaster_intel.state import FeatureSpec, StateAssembler, StaticLayers
from disaster_intel.storage import FakeRedis, HotStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

REPLAY_DIR = PROJECT_ROOT / "data" / "historical" / "replays"


@dataclass(frozen=True)
class EventConfig:
    event_id: str
    name: str
    district: str
    hazard_type: str
    focal_lat: float
    focal_lon: float
    start_date: str
    end_date: str
    payload_file: Path
    description: str


EVENTS: list[EventConfig] = [
    EventConfig(
        event_id="shirur_2024",
        name="2024 Shirur NH-66 Hill Collapse",
        district="Uttara Kannada (Ankola Taluk)",
        hazard_type="landslide",
        focal_lat=14.98,
        focal_lon=74.30,
        start_date="2024-07-08",
        end_date="2024-07-20",
        payload_file=PROJECT_ROOT / "data" / "historical" / "shirur_2024_payload.json",
        description=(
            "Catastrophic slope shear failure along NH-66 corridor following 72h continuous "
            "monsoon deluge exceeding 450mm cumulative precipitation."
        ),
    ),
    EventConfig(
        event_id="kodagu_2018",
        name="2018 Kodagu Cloudburst Landslides",
        district="Kodagu (Madikeri / Somwarpet)",
        hazard_type="landslide",
        focal_lat=12.42,
        focal_lon=75.74,
        start_date="2018-08-08",
        end_date="2018-08-20",
        payload_file=PROJECT_ROOT / "data" / "historical" / "kodagu_2018_payload.json",
        description=(
            "Extreme cloudburst event triggering over 200 simultaneous slope collapses "
            "documented in official GSI National Landslide Susceptibility Mapping (NLSM)."
        ),
    ),
    EventConfig(
        event_id="uttara_kannada_2021",
        name="2021 Uttara Kannada Gangavali Flood",
        district="Uttara Kannada (Karwar / Ankola)",
        hazard_type="flood",
        focal_lat=14.815,
        focal_lon=74.132,
        start_date="2021-07-18",
        end_date="2021-07-26",
        payload_file=PROJECT_ROOT / "data" / "historical" / "uttara_kannada_2021_payload.json",
        description=(
            "Severe riverine flash flood and estuarine inundation along the Gangavali and Kali "
            "river basins following sustained Western Ghats downpours."
        ),
    ),
]


def generate_replay_for_event(
    event_cfg: EventConfig,
    grid: CoastalGrid,
    disc: SpatialDiscretizer,
    layers: StaticLayers,
    spec: FeatureSpec,
    gt_catalog: GroundTruthCatalog,
    step_h: int = 6,
) -> dict[str, Any]:
    """Precompute replay frames and temporal metrics for a single historical disaster."""
    print(f"\n[+] Precomputing replay for: {event_cfg.name} ({event_cfg.event_id})")
    t_start = time.perf_counter()

    if not event_cfg.payload_file.exists():
        raise FileNotFoundError(f"Missing historical payload: {event_cfg.payload_file}")

    with open(event_cfg.payload_file, encoding="utf-8") as f:
        payload_data = json.load(f)

    # Determine focal cell
    focal_cell = grid.latlng_to_cell(event_cfg.focal_lat, event_cfg.focal_lon)
    if focal_cell not in grid:
        # Snap to nearest cell in grid if exact point is slightly off
        min_dist = float("inf")
        nearest_cell = next(iter(grid.cells))
        for c in grid.cells:
            c_lat, c_lon = grid.cell_to_latlng(c)
            d = (c_lat - event_cfg.focal_lat) ** 2 + (c_lon - event_cfg.focal_lon) ** 2
            if d < min_dist:
                min_dist = d
                nearest_cell = c
        focal_cell = nearest_cell

    # Parse raw readings
    adapter = OpenMeteoAdapter(
        clock=FakeClock(datetime.fromisoformat(event_cfg.start_date).replace(tzinfo=UTC))
    )
    readings = adapter.parse_payload(payload_data)

    # Organize readings by timestamp
    by_hour: dict[datetime, list[RawReading]] = defaultdict(list)
    for r in readings:
        by_hour[r.timestamp].append(r)
    hours = sorted(by_hour.keys())

    # Find cells covered by this event
    covered_cells: set[str] = set()
    for r in readings:
        for obs in disc.discretize(r):
            covered_cells.add(obs.cell)

    if not covered_cells:
        covered_cells = {focal_cell}

    sorted_cells = sorted(covered_cells)

    # Stream into hot store using FakeRedis
    start_dt = hours[0]
    clock = FakeClock(start_dt)
    fake_redis = FakeRedis()
    hot = HotStore(fake_redis, clock=clock, prefix=f"replay:{event_cfg.event_id}:")
    assembler = StateAssembler(layers, hot=hot, clock=clock)

    metric_keys = ["p1", "p6", "p24", "p72", "p7d", "sm", "cs"]
    metric_feature_map = {
        "p1": "precipitation__1h",
        "p6": "precipitation__6h",
        "p24": "precipitation__24h",
        "p72": "precipitation__72h",
        "p7d": "precipitation__7d",
        "sm": "soil_moisture_rootzone__24h",
        "cs": "compound_saturation",
    }

    frames: list[dict[str, Any]] = []
    timeseries_data: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # Pre-calculate cell geomorphology metadata
    cells_meta: dict[str, dict[str, Any]] = {}
    gt_event = gt_catalog.get_event(event_cfg.event_id)
    positive_cells_list = list(gt_event.positive_cells)

    for c in sorted_cells:
        centroid = list(grid.cell_to_latlng(c))
        poly = [[lat, lon] for lat, lon in h3.cell_to_boundary(c)]
        feat = layers.get(c)
        elev = round(float(feat.elevation_m), 1) if not math.isnan(feat.elevation_m) else 0.0
        slope = round(float(feat.slope_deg), 1) if not math.isnan(feat.slope_deg) else 0.0
        gsi_idx = (
            round(float(feat.gsi_susceptibility_index), 3)
            if not math.isnan(feat.gsi_susceptibility_index)
            else 0.0
        )

        d_lat = math.radians(centroid[0] - event_cfg.focal_lat)
        d_lon = math.radians(centroid[1] - event_cfg.focal_lon)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(event_cfg.focal_lat))
            * math.cos(math.radians(centroid[0]))
            * math.sin(d_lon / 2) ** 2
        )
        dist_km = round(6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a))), 1)

        is_gt_positive = c in gt_event.positive_cells

        cells_meta[c] = {
            "poly": poly,
            "lat": round(centroid[0], 4),
            "lon": round(centroid[1], 4),
            "elev": elev,
            "slope": slope,
            "gsi_idx": gsi_idx,
            "dist_km": dist_km,
            "is_gt_positive": is_gt_positive,
        }

    # Simulate hour-by-hour stream
    # Record frames every step_h hours once 72h warm-up period is reached
    warmup_period = timedelta(hours=72)
    obs_count = 0

    for ts in hours:
        clock.advance_to(ts)
        batch: list[Observation] = []
        for rr in by_hour[ts]:
            batch.extend(disc.discretize(rr))
        obs_count += len(batch)
        hot.write(batch)

        # Record timeseries for focal and first 5 cells
        if ts.hour % 2 == 0 or ts == hours[-1]:
            for inspect_c in [focal_cell, *sorted_cells[:5]]:
                m_curr = assembler.get_state(inspect_c, tier="hot").as_mapping(spec)
                p1_val = m_curr.get("precipitation__1h", 0.0)
                p24_val = m_curr.get("precipitation__24h", 0.0)
                sm_val = m_curr.get("soil_moisture_rootzone__24h", 0.0)
                cs_val = m_curr.get("compound_saturation", 0.0)
                timeseries_data[inspect_c].append(
                    {
                        "t": ts.strftime("%Y-%m-%d %H:%M"),
                        "p1": round(0.0 if math.isnan(p1_val) else float(p1_val), 2),
                        "p24": round(0.0 if math.isnan(p24_val) else float(p24_val), 2),
                        "sm": round(0.0 if math.isnan(sm_val) else float(sm_val), 3),
                        "cs": round(0.0 if math.isnan(cs_val) else float(cs_val), 2),
                    }
                )

        if (ts - start_dt) >= warmup_period and ts.hour % step_h == 0:
            ts_ist = ts + timedelta(hours=5, minutes=30)
            f_row: dict[str, list[float | None]] = {}
            for c in sorted_cells:
                m = assembler.get_state(c, tier="hot").as_mapping(spec)
                f_row[c] = [
                    None
                    if math.isnan(m[metric_feature_map[k]])
                    else round(float(m[metric_feature_map[k]]), 3)
                    for k in metric_keys
                ]
            frames.append(
                {
                    "t": ts.strftime("%Y-%m-%d %H:%M"),
                    "t_ist": ts_ist.strftime("%d %b %H:%M"),
                    "v": f_row,
                }
            )

    elapsed_s = time.perf_counter() - t_start
    print(
        f"    Generated {len(frames)} frames across {len(sorted_cells)} cells "
        f"({obs_count} obs) in {elapsed_s:.2f}s"
    )

    bundle = {
        "event_id": event_cfg.event_id,
        "name": event_cfg.name,
        "district": event_cfg.district,
        "hazard_type": event_cfg.hazard_type,
        "description": event_cfg.description,
        "focal": focal_cell,
        "focal_coords": [event_cfg.focal_lat, event_cfg.focal_lon],
        "start_date": event_cfg.start_date,
        "end_date": event_cfg.end_date,
        "cells": cells_meta,
        "frames": frames,
        "positive_cells": positive_cells_list,
        "timeseries": timeseries_data,
        "benchmarks": {
            "cells_count": len(sorted_cells),
            "observations_fanned_out": obs_count,
            "stream_duration_sec": round(elapsed_s, 2),
            "frames_count": len(frames),
            "step_hours": step_h,
        },
    }

    return bundle


def generate_all_replays() -> None:
    """Precompute and write JSON replay files for all events."""
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    grid = CoastalGrid(use_cache=True)
    disc = SpatialDiscretizer(grid=grid)
    layers = StaticLayers.load()
    spec = FeatureSpec.v1()
    gt_catalog = GroundTruthCatalog(grid=grid)

    for cfg in EVENTS:
        out_file = REPLAY_DIR / f"{cfg.event_id}.json"
        bundle = generate_replay_for_event(
            cfg, grid=grid, disc=disc, layers=layers, spec=spec, gt_catalog=gt_catalog
        )
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(bundle, f)
        print(f"  -> Saved {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_all_replays()
