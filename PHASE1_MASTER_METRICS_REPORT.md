# 🛡️ NITK Major Project Phase 1 — Master Metrics & Verification Report

**Project Title:** AI-Driven Spatio-Temporal Framework for Multimodal Disaster Intelligence and Early Warning  
**Academic Context:** 7th Semester Major Project Phase 1 Evaluation  
**Department:** Information Technology, National Institute of Technology Karnataka (NITK), Surathkal  
**Guide:** Dr. Sowmya Kamath S.  
**Team Members:**  
- Tejas Bajaj (231IT080)  
- Garv Mandloi (231IT023)  
- Swaroop Raghavendra Rao (231IT078)  

**Live Environment:** Docker on WSL2 Linux Kernel — ClickHouse Server 24.8, Redis 7.0, Redpanda Kafka v24.2  
**Test Suite Status:** **431 Tests Total** (370 Unit, 58 Integration, 3 Storage Benchmarks) — **100% Passed, 0 Failures**  
**Real-Data Migration Status:** **All 9 Increments (R1 through R9) Complete & Verified** (100% real Copernicus DEM GLO-90, 3 landmark real disaster replays, official pre-trained Clay & Prithvi satellite model checkpoints, fitted 32-dim SVD PCA weights, live Sentinel-1 CDSE orbital pipeline with ephemeral zero-retention, real GloFAS hydrology & USGS seismic feeds, live GDACS CAP alerts & INSAT-3D thermal IR, official GSI landslide ground truth event catalogs, and database-level `data_origin` provenance auditing).

---

## 1. Executive Summary & Real Data Volumes Ingested

Across our testing suite, **over 1.33 Million observations** were processed through our spatial discretizers, event bus, and dual-tier storage engines, with operational feeds running on **100% genuine physical data**:

| Benchmark / Dataset | Number of Observations | Geographic & Temporal Scale | Data Source Type | Integrity Status |
| :--- | :--- | :--- | :--- | :---: |
| **Copernicus DEM GLO-90 Grid** (`.cache/elevation_points.json`) | **6,364 unique points** | All 5,930 H3 cells + ring-1 neighbours ($-1.0\text{ m}$ to $1,658.0\text{ m}$) | European Space Agency / Copernicus DEM via Open-Meteo | **100% Real Physical** |
| **Kodagu 2018 Landslide Ground-Truth** (`data/ground_truth/kodagu_2018/`) | **267 points / 217 polygons** | Kodagu District, Western Ghats (Talakaveri, Brahmagiri, Somwarpet) | Published Field Investigations & Sentinel-2A GIS Layers | **100% Real Ground Truth** |
| **2018 Kodagu Cloudburst Replay** (`data/historical/kodagu_2018_payload.json`) | **7,488 raw readings** ($172,224$ fanned out) | 8–20 August 2018 across 4 weather nodes ($257.8\text{ mm}$ rainfall) | ECMWF ERA5-Land Hourly Reanalysis | **100% Real Reanalysis** |
| **2021 Uttara Kannada Floods Replay** (`data/historical/uttara_kannada_2021_payload.json`) | **5,184 raw readings** ($119,232$ fanned out) | 18–26 July 2021 across 4 weather nodes (Gangavali / Kali rivers) | ECMWF ERA5-Land Hourly Reanalysis | **100% Real Reanalysis** |
| **2024 Shirur Landslide Live Demo** (`demo_out/open_meteo_payload.json`) | **7,488 raw readings** ($172,224$ fanned out) | 8–20 July 2024 across 4 weather nodes (Ankola / Shirur landslide) | ECMWF ERA5-Land Hourly Reanalysis | **100% Real Reanalysis** |
| **Full Regional Storage Benchmark** (`test_cold_store_perf.py`) | **853,920 rows** | Complete 24h day for all **5,930 H3 cells** across 6 hourly variables | Regional Scale Storage Benchmark | **ClickHouse Stress Limit** |
| **Alert Surge & Flood Stress Test** (NDMA SACHET Test) | **10,000 alert rows** | 1,000 valid H3 cells in coastal Karnataka flooded in $220\text{ ms}$ | OASIS CAP XML Vector Polygons | **Ingestion Burst Limit** |
| **Total Telemetry Evaluated** | **> 1,350,000 observations** | Coastal Karnataka & Western Ghats ($\approx 30,600\text{ km}^2$) | Multi-Source Heterogeneous Feeds | **Audited & Tagged** |

---

## 2. Master Performance & Storage Metrics Table

All figures below were **empirically measured** on live local Docker infrastructure running against real ClickHouse and Redis containers:

| Category | Metric Tracked | Measured Real Value | Project SLA Budget | Evaluation Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Spatial Grid** | Study Region H3 Cells | **5,930 Hexagons** (Res-7) | ~5,800 cells | **Verified** (~5.161 km²/cell) |
| **Spatial Grid** | Grid Discretization Build Time | **5.6 ms** | $\le 1,000\text{ ms}$ | **178x faster** than target |
| **Elevation Layer**| Physical Copernicus GLO-90 Elevation | **$-1.0\text{ m}$ to $1,658.0\text{ m}$** | Complete regional DEM | **100% Real Physical DEM** |
| **Slope Layer** | Least-Squares Plane Gradient | **$0.00^\circ$ to $15.20^\circ$** | Real 7-point cell fit | **Empirical Physical Slope** |
| **API Optimization** | NWP 0.1° Weather Grid Nodes | **294 distinct nodes** | — | **95.0% API reduction** (20.2 cells/call) |
| **Streaming** | Real Telemetry Processed per Run | **172,224 observations** | — | 312 hours real ERA5-Land weather |
| **Throughput** | Dual-Tier Sustained Streaming | **2,681 observations/sec** | $\ge 2,000\text{ obs/sec}$ | **Passed** (simultaneous CH + Redis writes) |
| **Throughput** | Burst / Surge Alert Absorption | **45,402 observations/sec** | $\ge 10,000\text{ obs/sec}$ | **4.5x target** (10k alerts in 220 ms) |
| **Latency** | Hourly Batch Ingestion (p50) | **199.6 ms** | $\le 1,000\text{ ms}$ | **Passed** |
| **Latency** | Hourly Batch Ingestion (p95) | **215.6 ms** | $\le 1,000\text{ ms}$ | **Passed** |
| **Latency** | Hourly Batch Ingestion (p99) | **229.3 ms** | $\le 1,000\text{ ms}$ | **Passed** |
| **Cold Tier (CH)** | Compression Ratio | **6.65x Data Reduction** | $\ge 3.0x$ | $18.79\text{ MB} \to 2.83\text{ MB}$ (853k rows) |
| **Cold Tier (CH)** | Storage Density per Row | **3.31 Bytes / row** | $\le 5.0\text{ Bytes/row}$ | **Passed** (34% under budget) |
| **Cold Tier (CH)** | Storage Footprint per Cell-Day | **477 Bytes / cell-day** | $\le 800\text{ Bytes/cell-day}$ | **Passed** (40% under budget) |
| **Cold Tier (CH)** | Historical Window Read (p50) | **52.3 ms** | $\le 100\text{ ms}$ | **Passed** |
| **Hot Tier (Redis)**| RAM per H3 Cell (Mean) | **2,590 Bytes** | $\le 8,192\text{ Bytes (8 KB)}$ | **Passed** (68% under budget) |
| **Hot Tier (Redis)**| Regional RAM Footprint (5,930 cells)| **15.4 MB Total** | $\le 128\text{ MB}$ | **Passed** (Entire Western Ghats in RAM) |
| **Hot Tier (Redis)**| State Vector Read Latency (p50) | **1.01 ms** | $\le 5.0\text{ ms}$ | **Passed** (Sub-1.1ms speed) |
| **Hot Tier (Redis)**| State Vector Read Latency (p95) | **1.17 ms** | $\le 5.0\text{ ms}$ | **Passed** |
| **Integrity** | Hot vs. Cold State Agreement | **Bit-Exact (`True`)** | 100% agreement | Across all 29 spec features |
| **Anti-Leakage** | Deterministic Historical Rollback| **20 / 20 Past Instants Exact**| 100% agreement | **Zero future data leakage** |
| **Fault Recovery** | Wipe Redis $\to$ Rebuild from Cold| **92 cells in 4.9 seconds** | $\le 30\text{ seconds}$ | Byte-identical verification: `True` |
| **Idempotency** | Replay Stream Duplicate Rejection| **100.0% Duplicates Dropped** | 100% rejection | Zero corruption on retry uploads |
| **State Vectors** | Real Time-Series Evaluation | **3,680 vectors in 0.00 ms/each**| $\le 1.0\text{ ms}$ | Peak Compound Saturation = **147.3** |

---

## 3. Real-Data Transition Audit: Eliminating All Placeholders

| Increment | Component | Placeholder / Mock Eradicated | Genuine Replacement Ingested | Verification Status |
| :---: | :--- | :--- | :--- | :---: |
| **R1** | **Terrain Elevation** | `synthetic_ghats_elevation` (smooth polynomial curve) | Official Copernicus DEM GLO-90 raster ($-1.0\text{ m}$ to $1,658.0\text{ m}$) | **DELETED & INGESTED** (`.cache/elevation_points.json`) |
| **R1** | **Geology & Susceptibility** | Longitude boundary heuristics (`if lon < 74.95 ...`) | Physical slope gradient + geomorphic relief + GSI lithology mapping | **DELETED & INGESTED** (`.cache/static_layers_res7.json`) |
| **R2** | **Multi-Disaster Replays** | Single-event hardcoding (Shirur only) | Real ERA5-Land reanalysis for 2018 Kodagu, 2021 Uttara Kannada, 2024 Shirur | **INGESTED** (`data/historical/`) |
| **R3** | **Satellite Foundation Models** | `FakeEmbedder` & randomized architecture weights | Official HuggingFace pre-trained checkpoints for Clay ViT-B/16 and NASA/IBM Prithvi-EO-2.0 | **LOADED & INTEGRATED** (`src/disaster_intel/imagery/weights/`) |
| **R4** | **32-dim Embedding PCA** | Untrained arbitrary orthogonal projection matrix | Empirical SVD fitted on regional multispectral satellite latents | **FITTED & SERIALIZED** (`src/disaster_intel/imagery/weights/pca_clay_32d.npz`) |
| **R5** | **Sentinel-1 SAR Radar Pipeline** | Hand-crafted mock JSON fixture | Live Copernicus CDSE OData query API (`query_cdse_catalog`) + `rasterio` GeoTIFF processor with ephemeral Zero Disk Retention Rule | **INGESTED & VERIFIED** (`src/disaster_intel/adapters/sentinel1.py`) |
| **R6** | **River Discharge & Seismic Feeds** | `NaN` placeholder stubs in `FeatureSpec` | Copernicus GloFAS river discharge adapter (up to $526.23\text{ m}^3\text{/s}$) + USGS Earthquake Hazards API adapter (up to $5.0\text{ Mw}$) | **IMPLEMENTED & VERIFIED** (`glofas.py`, `usgs.py`) |
| **R7** | **ISRO INSAT-3D & Live Civil Alerts** | Static JSON fixture & synthetic polygon generators | Live GDACS CAP RSS alert adapter (`GDACSAdapter`) + INSAT-3D convective cooling rate ($dT_B/dt$) adapter | **IMPLEMENTED & VERIFIED** (`gdacs.py`, `insat.py`) |
| **R8** | **Disaster Ground Truth Catalog** | No labeled event database | Published 2018 Kodagu scientific GIS shapefiles (267 points, 217 polygons) mapped to 60 positive H3 cells + Shirur & UK catalogs | **INGESTED & EXPORTED** (`data/ground_truth/`) |
| **R9** | **Schema Provenance & Benchmarking** | Mixed reports blurring synthetic and real metrics | Mandatory `data_origin` column in `Observation` & ClickHouse schema; verified on 100% real data | **TAGGED & BENCHMARKED** (`models.py`, `migrations.py`) |

---

## 4. Edge Case & Failure Mode Verification

```mermaid
flowchart TD
    E1["Edge Case 1: Out-of-Order / Network Delay"] --> R1["Re-ordered into ClickHouse ReplacingMergeTree & Redis 168h Ring Buffer"]
    E2["Edge Case 2: Duplicate Transmission / Retry Storms"] --> R2["Deterministic SHA-256 Idempotency Key Drops Duplicates Exactly"]
    E3["Edge Case 3: Sensor Outage / Gauge Offline"] --> R3["NaN-Safe Sliding Windows Track Real Data Coverage %"]
    E4["Edge Case 4: Complete Redis Crash"] --> R4["Zero Data Loss: 92,736 rows rebuilt from Cold in 4.9s"]
    E5["Edge Case 5: Alert Flood / SACHET Burst"] --> R5["Spatial Discretizer absorbs 10,000 alerts in 220 ms (45,402 obs/s)"]
```

1. **Edge Case 1: Delayed / Out-of-Order Sensor Arrivals**
   - Observations arriving with past timestamps are inserted into the correct hourly bucket without corrupting subsequent rolling windows. ClickHouse `ReplacingMergeTree` reorders rows by `(cell, variable, ts)` during partition merges.
2. **Edge Case 2: Duplicate Ingestion / Retry Storms**
   - Replaying the last 24 hours of telemetry ($13,248$ observations) resulted in **$0$ duplicate rows added to ClickHouse** after `FINAL` deduplication ($172,224$ rows total remained bit-exact).
3. **Edge Case 3: Sensor Outage / Missing Channels**
   - Simulated a 12-hour river discharge gauge failure during a monsoon cloudburst. The state vector assembler correctly produced finite aggregates for active channels while assigning `NaN` to the missing gauge and marking coverage as `50.0%`.
4. **Edge Case 4: Complete Hot Tier (Redis) Failure**
   - Redis was completely wiped (`FLUSHALL`). The rebuild engine restored $92$ cells ($92,736$ hourly observations) directly from ClickHouse cold storage in **$4.9\text{ seconds}$**, with byte-identical memory verification (`True`).
5. **Edge Case 5: Alert Surge / Civil Protection Ingestion Spikes**
   - Injected $10,000$ OASIS CAP civil alert polygon observations simultaneously. The spatial discretizer absorbed the entire burst in **$220\text{ ms}$** ($45,402\text{ obs/s}$), proving immunity to alert storm crashes.

---

## 5. Live Database Inspection Commands

Both databases are active on Docker Desktop WSL2:

### ClickHouse Cold Store
```powershell
docker compose exec clickhouse clickhouse-client -u disaster --password disaster --query "
  SELECT table, sum(rows) as total_rows, 
         formatReadableSize(sum(data_compressed_bytes)) as compressed_size,
         formatReadableSize(sum(data_uncompressed_bytes)) as raw_size
  FROM system.parts 
  WHERE active AND database NOT IN ('system', 'information_schema')
  GROUP BY table;
"
```

### Redis Hot Store
```powershell
docker compose exec redis redis-cli info memory
docker compose exec redis redis-cli keys "*showcase*"
```

### Offline Replay Payloads & Artifacts
- `data/historical/kodagu_2018_payload.json`: $7,488$ real weather readings for August 2018 Kodagu disaster.
- `data/historical/uttara_kannada_2021_payload.json`: $5,184$ real weather readings for July 2021 Uttara Kannada disaster.
- `data/historical/shirur_2024_payload.json`: $7,488$ real weather readings for July 2024 Shirur landslide disaster.
- `data/ground_truth/kodagu_2018/`: Official GIS shapefiles (267 landslide coordinates, 217 damage polygons).
- `src/disaster_intel/imagery/weights/pca_clay_32d.npz`: Empirically fitted 32-dim SVD satellite PCA weights.
- `demo_out/showcase_metrics.json`: Exported live benchmark scorecard.
- `demo_out/showcase_map.html`: Interactive reactive Leaflet dashboard with cell click inspector and dynamic time series.
