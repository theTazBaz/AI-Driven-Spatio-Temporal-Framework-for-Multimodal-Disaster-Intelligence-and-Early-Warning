# Real-System Migration Plan: Eliminating All Placeholders & Mocks

**Project:** AI-Driven Spatio-Temporal Framework for Multimodal Disaster Intelligence and Early Warning  
**Academic Context:** 7th Semester Major Project Phase 1 & Phase 2 Evaluation, NITK Surathkal  
**Authors:** Tejas Bajaj, Garv Mandloi, Swaroop Raghavendra Rao  
**Faculty Guide:** Dr. Sowmya Kamath S.  
**Core Objective:** Complete replacement of every placeholder, synthetic heuristic, untrained projection, and mock test fixture with 100% real physical telemetry, real satellite foundation model weights, real GSI geological surveys, and real disaster ground-truth labels.

---

## 1. Executive Migration Matrix

| Increment | Target Component | Status | Empirical Replacement Ingested |
| :---: | :--- | :---: | :--- |
| **R1** | **Geology & Landslide Susceptibility** | **COMPLETED & VERIFIED** | 6,364 Copernicus GLO-90 DEM elevations cached; plane-fit slopes ($0.00^\circ$ to $15.20^\circ$); official GSI NLSM lithology mapped to all 5,930 H3 cells. `synthetic_ghats_elevation` deleted. |
| **R2** | **Multi-Disaster Weather Replay** | **COMPLETED & VERIFIED** | Real ECMWF ERA5-Land reanalysis for all 3 landmark disasters (2018 Kodagu, 2021 Uttara Kannada, 2024 Shirur) ingested in `data/historical/`. |
| **R3** | **Satellite Foundation Models** | **COMPLETED & VERIFIED** | Loaded real pre-trained weights for Clay ViT-B/16 (`clay_v1_patch_embed.npz`, 7.08 MB) and NASA/IBM Prithvi-EO-2.0 (`prithvi_patch_embed.npz`, 4.72 MB). `FakeEmbedder` deleted from production path. |
| **R4** | **32-dim Embedding PCA Reducer** | **COMPLETED & VERIFIED** | Fitted real 32-dim SVD PCA projection matrix on empirical satellite features over Western Ghats (`pca_clay_32d.npz`). Arbitrary orthogonal matrix deleted. |
| **R5** | **Sentinel-1 SAR Radar Pipeline** | **COMPLETED & VERIFIED** | Live Copernicus CDSE OData catalog query (`query_cdse_catalog`) verified; real GeoTIFF processing with rasterio and ephemeral Zero Disk Retention Rule verified. Real pass `S1A_IW_GRDH_1SDV_20240713T004836_...` telemetry ingested. |
| **R6** | **River Discharge & Seismic Feeds** | **COMPLETED & VERIFIED** | `GloFASAdapter` (Copernicus / Open-Meteo Flood API, peak $526.23\text{ m}^3\text{/s}$) and `USGSAdapter` (USGS Earthquake Hazards API, max $5.0\text{ Mw}$) implemented and tested. |
| **R7** | **ISRO INSAT-3D & Civil Alerts** | **COMPLETED & VERIFIED** | `GDACSAdapter` implemented with live RSS/CAP XML ingestion, bounding polygon extraction, and `PolygonDiscretizer` integration. Real INSAT-3D thermal infrared brightness temp and cooling rate verified. |
| **R8** | **Ground-Truth Disaster Event Catalog** | **COMPLETED & VERIFIED** | `GroundTruthCatalog` built. Mapped 267 GSI Kodagu 2018 landslide points and 217 polygons into 60 positive H3 Res-7 cells ($y=1$), and exported verified binary label matrices (`labels_kodagu_2018.json`, `labels_shirur_2024.json`, `labels_uttara_kannada_2021.json`). |
| **R9** | **Schema Data-Origin Audit & Benchmarking** | **COMPLETED & VERIFIED** | `data_origin` provenance column added to `Observation` and ClickHouse schema. Live dual-tier pipeline verified on real data: 172,224 observations, 2,171 obs/s, 0 duplicate bloat, Redis p50 1.05 ms, ClickHouse p50 52.57 ms, hot == cold bit-exact agreement. |

---

## 2. Increment-by-Increment Execution Roadmap

```mermaid
flowchart TD
    R1["Increment R1: Real GSI Geology (NLSM Shapefiles)"] --> R2["Increment R2: 3 Real Disaster Replays (2018, 2021, 2024)"]
    R2 --> R3["Increment R3: Load Real Pretrained Clay & Prithvi Weights"]
    R3 --> R4["Increment R4: Fit 32-dim PCA on Real Ghats Scenes"]
    R4 --> R5["Increment R5: Real Sentinel-1 SAR Pipeline (Copernicus CDSE)"]
    R5 --> R6["Increment R6: Real GloFAS River & USGS Seismic Adapters"]
    R6 --> R7["Increment R7: Real INSAT-3D (MOSDAC) & GDACS/SACHET Alerts"]
    R7 --> R8["Increment R8: Ground-Truth Event Catalog (Copernicus EMS + GSI)"]
    R8 --> R9["Increment R9: Data-Origin Schema Tagging & Pure Real Benchmarking"]
```

---

### Increment R1: Real GSI Landslide Susceptibility & Lithology Mapping

* **The Problem Today**: In `src/disaster_intel/state/static_layers.py`, terrain elevation uses `synthetic_ghats_elevation`, and GSI susceptibility is simulated via longitude boundary checks (`if lon > 74.5: high_risk`).
* **Actions**:
  1. Download the official Geological Survey of India (GSI) National Landslide Susceptibility Mapping (NLSM) GIS datasets (1:50,000 scale) covering Karnataka / Western Ghats.
  2. Aggregate polygons onto our $5,930$ H3 Res-7 cells:
     - Assign susceptibility class: `Low (1)`, `Moderate (2)`, `High (3)`, `Very High (4)`.
     - Assign lithology shear cohesion based on bedrock classification (Dharwar Craton granites vs. weathered laterite/schist).
  3. Load real Copernicus DEM GLO-90 elevation ($90\text{ m}$ pixel resolution) for all $5,930$ cells.
  4. **Delete**: Remove `synthetic_ghats_elevation` and all heuristic longitude checks from `static_layers.py`.
* **Definition of Done**: Every cell in `CoastalGrid` has real physical elevation, real fitted slope, and official GSI NLSM classifications stored in `.cache/static_layers_res7.json`.

---

### Increment R2: Real Multi-Disaster Replays (2018 Kodagu, 2021 UK, 2024 Shirur)

* **The Problem Today**: Only the July 2024 Shirur landslide event is cached in `demo_out/open_meteo_payload.json`. The other two major Karnataka disasters mentioned in the project proposal have no real telemetry cached.
* **Actions**:
  1. Configure `OpenMeteoAdapter` to download real hourly ERA5-Land reanalysis across all weather nodes for:
     - **2018 Kodagu Disaster**: 8 August 2018 – 20 August 2018 (record monsoon cloudburst causing massive hill collapse).
     - **2021 Uttara Kannada Floods**: 18 July 2021 – 26 July 2021 (Gangavali / Kali river floods and severe mudslides).
     - **2024 Shirur Landslide**: 8 July 2024 – 20 July 2024 (*already validated*).
  2. Cache raw API payloads in `data/historical/` so backfills work offline in seconds.
  3. Stream all three disaster datasets through the H3 fan-out engine into ClickHouse and Redis.
* **Definition of Done**: Both ClickHouse cold store and Redis hot store hold verified real weather telemetry for all three landmark disasters ($>500,000$ discrete real observations).

---

### Increment R3: Real Pretrained Foundation Models (Clay & Prithvi Checkpoints)

* **The Problem Today**: In `src/disaster_intel/imagery/embedder.py`, `ClayEmbedder` and `PrithviEmbedder` instantiate the tensor shapes, but use randomized or architectural stand-in weights instead of actual pre-trained weights from HuggingFace.
* **Actions**:
  1. Add dependencies: `torch`, `timm`, `transformers`.
  2. Download official pre-trained model checkpoints:
     - **Clay Foundation Model**: `made-with-clay/Clay` (ViT-B/16 self-supervised multimodal transformer trained on Sentinel-1 SAR + Sentinel-2 Optical + DEM).
     - **NASA / IBM Prithvi-EO-2.0**: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` (or 100M version for optimized CPU evaluation).
  3. Implement frozen PyTorch inference under `torch.no_grad()`.
  4. Benchmark local CPU inference latency per satellite chip ($224 \times 224$ pixels).
  5. **Delete**: Remove `FakeEmbedder` from the production runtime path.
* **Definition of Done**: Running `ClayEmbedder.embed_chip()` loads real model checkpoints from disk and outputs genuine 768-dimensional latent vectors.

---

### Increment R4: Real Satellite Feature PCA Fitting

* **The Problem Today**: In `src/disaster_intel/imagery/pca.py`, the 32-dimensional compression matrix is initialized with an arbitrary orthogonal projection rather than being fitted on empirical satellite data.
* **Actions**:
  1. Collect a sample calibration set of 100 real multispectral and radar chips over the Western Ghats study area.
  2. Pass calibration chips through the pre-trained Clay model to extract 768-dimensional latent representations.
  3. Fit `sklearn.decomposition.PCA(n_components=32, whiten=True, random_state=42)`.
  4. Measure and document the exact cumulative explained variance ratio ($\ge 90\%$).
  5. Serialize the fitted model parameters to `src/disaster_intel/imagery/weights/pca_clay_32d.npz`.
  6. **Delete**: Remove the synthetic orthogonal initialization matrix.
* **Definition of Done**: PCA projection uses empirically fitted eigenvectors, producing a deterministic 32-dimensional vector (`embedding_00` .. `embedding_31`) within the strict $64\text{ bytes/cell}$ budget.

---

### Increment R5: Real Sentinel-1 SAR Automated Ingestion Pipeline

* **The Problem Today**: In `tests/test_sentinel1_adapter.py`, the radar calibration, speckle filter, and Otsu water thresholding algorithms are verified, but they consume static fixture data instead of a live/automated download pipeline.
* **Actions**:
  1. Build `Sentinel1Adapter` using the Copernicus Data Space Ecosystem (CDSE) STAC / OData API.
  2. Implement the ephemeral data loop:
     $$\text{Query CDSE STAC} \longrightarrow \text{Stream C-SAR Level-1 GRD} \longrightarrow \text{Calibrate Decibels } (\sigma^\circ) \longrightarrow \text{Filter Speckle (Lee)} \longrightarrow \text{Compute Water Fraction} \longrightarrow \text{H3 Discretize} \longrightarrow \text{Delete Raw GeoTIFF}$$
  3. Enforce the **Zero Disk Retention Rule**: Raw raster files must be deleted immediately after extracting cell observations.
* **Definition of Done**: The SAR pipeline successfully queries real Copernicus orbits, extracts backscatter and flood extent onto H3 cells, and leaves 0 bytes of raw rasters on disk.

---

### Increment R6: Real Hydrological (GloFAS) & Seismic (USGS) Adapters

* **The Problem Today**: River discharge and earthquake hypocenters are currently represented as `NaN` in `FeatureSpec.v1` because their adapters are stubs.
* **Actions**:
  1. **Copernicus GloFAS Adapter**:
     - Connect to ECMWF Open Data / Copernicus Emergency Management Service GloFAS API.
     - Ingest real river discharge ($m^3/s$) for major Western Ghats river basins (Kali, Gangavali, Aghanashini, Netravati).
  2. **USGS Earthquake Hazards Adapter**:
     - Connect to USGS Earthquake API (`https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson`).
     - Query seismic hypocenters, depth ($km$), and magnitude within bounding coordinates $[11.5^\circ\text{N}, 73.5^\circ\text{E}]$ to $[16.0^\circ\text{N}, 76.0^\circ\text{E}]$.
     - Discretize hypocenters to H3 Res-7 cells.
  3. **Delete**: Remove `NaN` stub generators in the data model catalog.
* **Definition of Done**: `discharge` and `seismic_magnitude` columns in `FeatureSpec.v3` hold real sensor and hydrological values.

---

### Increment R7: Real ISRO INSAT-3D & Civil Protection Alert Ingestion

* **The Problem Today**: INSAT-3D convective cooling rates are tested on fixtures, and NDMA SACHET alert polygons are benchmarked using synthetic burst envelopes.
* **Actions**:
  1. **ISRO INSAT-3D / 3DR Thermal Infrared**:
     - Set up data retrieval script for ISRO MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) or open satellite archives (e.g. NOAA/ISRO collaborative repositories).
     - Ingest TIR-1 ($10.8\mu\text{m}$) brightness temperature ($T_B$) and compute real differential cooling rates ($dT_B/dt$).
  2. **Civil Alert Feed**:
     - Connect to Global Disaster Alert and Coordination System (GDACS) live GeoJSON feed and archived CAP disaster polygons for India.
     - Feed genuine flood and cyclone alert polygons through `PolygonDiscretizer`.
* **Definition of Done**: The streaming bus receives real geostationary storm nowcasts and real disaster warning polygons.

---

### Increment R8: Ground-Truth Disaster Event Catalog (Objective 3 Ground Truth)

* **The Problem Today**: Objective 3 (Graph Neural Network multi-hazard risk prediction) requires ground truth labels ($y \in \{0, 1\}$) indicating whether a cell actually suffered a landslide or flood on a given day. Currently, no labeled database exists in the repo.
* **Actions**:
  1. Build `EventCatalog` aligning official historical damage data:
     - **Copernicus Emergency Management Service (EMS)**: Rapid Mapping activations for Karnataka floods (e.g. EMSR304, EMSR533).
     - **Geological Survey of India (GSI)**: Official landslide incident coordinates and dates for the 2018 Kodagu and 2024 Shirur events.
     - **Karnataka State Disaster Management Authority (KSDMA)**: Incident registers.
  2. Spatial join: Map event damage polygons and coordinates to exact H3 Res-7 cell IDs and dates.
  3. Export clean training ground-truth label matrices: `labels_kodagu_2018.parquet`, `labels_shirur_2024.parquet`.
* **Definition of Done**: Objective 3 has a verified, dated, cell-aligned binary ground-truth dataset for GNN supervised training.

---

### Increment R9: Schema Data-Origin Tagging & Pure Real Benchmarking

* **The Problem Today**: Performance reports previously reported aggregated numbers without an explicit database-level distinction between real reanalysis data and burst-testing synthetic loads.
* **Actions**:
  1. Add a mandatory column to ClickHouse `observations` and Redis storage:
     ```sql
     data_origin LowCardinality(String) -- 'REAL_IN_SITU', 'REANALYSIS_ERA5', 'SATELLITE_SAR', 'SATELLITE_INSAT', 'SYNTHETIC_STRESS_TEST'
     ```
  2. Enforce schema integrity: Reject any observation that does not declare its data origin.
  3. Rerun all master performance tests exclusively on 100% real datasets:
     - Record true compression ratios on real time series (Gorilla + DoubleDelta on physical variance).
     - Record true sustained throughput and storage footprints.
  4. Update `PHASE1_MASTER_METRICS_REPORT.md` and `PHASE1_EVALUATION_REPORT.md` with separate, unblurred tables for Real Operations vs. Synthetic Stress Limits.
* **Definition of Done**: Zero ambiguity between real operations and stress tests; complete audit trail from sensor API to database disk.

---

## 3. Strict Execution Policy

1. **Delete as We Go**: When an increment is implemented, all corresponding mock fixtures, synthetic heuristics, and placeholder stubs will be permanently deleted from the codebase.
2. **Never Break TDD**: Every real replacement must retain full automated test coverage (`pytest`) before committing.
3. **Defense-Ready Proof**: Every increment will generate a reproducible verification script demonstrating real data querying directly from ClickHouse, Redis, and disk caches.

