# AI-Driven Spatio-Temporal Framework for Multimodal Disaster Intelligence and Early Warning

An open, streaming disaster-intelligence framework designed to break institutional siloing and overcome passive, hallucination-prone conversational chatbots in disaster early warning. 

The framework projects multimodal physical telemetry (atmospheric, hydrological, seismic, satellite imagery, and CAP bulletins) onto Uber's **H3 hexagonal Discrete Global Grid System (Res-7)**, maintains a dynamic per-cell **WorldState feature store** over rolling time horizons (1h to 7d), scores non-linear compound hazard risks with explainable ML, and generates fact-grounded, trend-aware early warning alerts.

> **Status:** **Phase 1 (Data Engine, Dual-Tier Persistence & State Assembly) is 100% COMPLETE & VERIFIED** on genuine physical data with zero synthetic heuristics or mock fixtures.

---

## 🎯 The Four Core Objectives

1. **Multimodal Telemetry Ingestion & Spatial Discretization (Gap G1):** *(Phase 1 — Complete)*  
   Continuous streaming ingestion of heterogeneous data sources (Open-Meteo weather, Copernicus GloFAS river discharge, USGS seismology, NDMA SACHET / GDACS CAP alerts, Copernicus Sentinel-1 C-SAR radar, Sentinel-2 foundation models) projected onto 5,930 H3 Res-7 (~5.16 km²/cell) hexagons.
2. **Continuous Hexagonal WorldState Feature Store (Gap G2):** *(Phase 1 — Complete)*  
   A dual-tier speed/batch feature store with an in-memory **Redis hot tier** (7-day ring buffer, $p_{50} = 1.05\text{ ms}$) and a columnar **ClickHouse cold archive** ($p_{50} = 52.57\text{ ms}$) with deterministic historical rollback and bit-exact dual-tier agreement.
3. **Compound Multi-Hazard Risk Scoring (Gap G3):** *(Phase 2 — In Progress)*  
   Empirical comparison of candidate ML/GNN models (Spatio-Temporal Graph Neural Networks, GBDT/XGBoost) for non-linear compound risk prediction, paired with exact polynomial-time **TreeSHAP** feature attribution.
4. **Trend-Aware Alert Synthesis (Gap G4):** *(Phase 2)*  
   Constrained natural language alert generation using **DSPy typed signatures** and Pydantic validation to guarantee zero numeric or entity hallucinations while communicating trajectory trends (rising, falling, stable).

---

## 📍 Initial Study Region & Physical Grounding

- **Focus:** Coastal Karnataka & Western Ghats (Dakshina Kannada, Udupi, Uttara Kannada, Kodagu, Chikkamagaluru).
- **Spatial Resolution:** $5,930$ H3 Resolution-7 cells ($1.2\text{ km}$ inter-cell spacing, $11.9^\circ-15.5^\circ\text{N}, 74.0^\circ-76.0^\circ\text{E}$).
- **Elevation & Slope:** Copernicus DEM GLO-90 physical elevation ($6,364$ physical points) and 7-point plane slope gradients ($0.0^\circ - 15.2^\circ$).
- **Geology & Lithology:** Geological Survey of India (GSI) National Landslide Susceptibility Mapping (NLSM) index ($0.0 - 1.0$) and lithology cohesion ($0.5 - 1.0$).
- **Disaster Event Catalogs:** Official ground-truth event labels mapped to positive cells ($y=1$):
  - **August 2018 Kodagu:** $60$ positive landslide cells from GSI inventory ($267$ points, $217$ polygons).
  - **July 2024 Shirur:** Highway slope failure and mudslide.
  - **July 2021 Uttara Kannada:** Severe monsoonal landslides and riverine inundation.

---

## ⚡ Quickstart (Windows / macOS / Linux)

### 1. Prerequisites
* **Python 3.11+**
* **[`uv`](https://docs.astral.sh/uv/)** (recommended) or standard `pip`
* **Docker** (optional; includes resilient in-memory fallbacks if containers are stopped)

### 2. Setup in 3 Commands
```bash
# 1. Clone repository
git clone https://github.com/tejas-bajaj/AI-Driven-Spatio-Temporal-Framework-for-Multimodal-Disaster-Intelligence-and-Early-Warning.git
cd AI-Driven-Spatio-Temporal-Framework-for-Multimodal-Disaster-Intelligence-and-Early-Warning

# 2. Install dependencies
uv sync

# 3. Verify suite (390 unit tests, 100% pass)
uv run pytest -m unit
```

### 3. Run the Live Operational Multimodal Daemon
Polls live weather, river discharge, civil alerts, seismology, and satellite passes:
```bash
# Run a single live multimodal cycle
uv run python scripts/live_daemon.py --once

# Or run continuous polling loop (every 5 minutes)
uv run python scripts/live_daemon.py --interval-sec 300
```

### 4. Interactive Disaster Replay & Map Showcase
Replays genuine ECMWF ERA5-Land reanalyses for the July 2024 Shirur disaster through the full stack:
```bash
uv run python scripts/showcase_demo.py --cache
# Opens demo_out/showcase_map.html in your browser
```

---

## 👥 Team & Academic Context

- **Institution:** National Institute of Technology Karnataka (NITK), Surathkal — Department of Information Technology
- **Guide:** Dr. Sowmya Kamath S.
- **Team Members:**
  - Tejas Bajaj (231IT080)
  - Garv Mandloi (231IT023)
  - Swaroop Raghavendra Rao (231IT078)
