# AI-Driven Spatio-Temporal Framework for Multimodal Disaster Intelligence and Early Warning

An open, streaming disaster-intelligence framework designed to break institutional siloing and overcome passive, hallucination-prone conversational chatbots in disaster early warning. 

The framework projects multimodal physical telemetry (atmospheric, hydrological, seismic, satellite imagery, and CAP bulletins) onto Uber's **H3 hexagonal Discrete Global Grid System (Res-7)**, maintains a dynamic per-cell **WorldState feature store** over rolling time horizons (1h to 7d), scores non-linear compound hazard risks with explainable ML, and generates fact-grounded, trend-aware early warning alerts.

---

## 🎯 The Four Core Objectives

1. **Multimodal Telemetry Ingestion & Spatial Discretization (Gap G1):**  
   Continuous streaming ingestion of heterogeneous data sources (Open-Meteo, USGS, GDACS, NDMA SACHET, GPM IMERG, SMAP, Sentinel/NISAR) projected onto a unified H3 Res-7 (~5.16 km²/cell) spatial index.
2. **Continuous Hexagonal WorldState Feature Store (Gap G2):**  
   A dual-tier speed/batch feature store with an in-memory **Redis hot tier** (1h, 6h, 24h, 72h, 7d rolling windows) and a columnar **ClickHouse cold archive** with deterministic historical rollback.
3. **Compound Multi-Hazard Risk Scoring (Gap G3):**  
   Empirical comparison of candidate ML models (e.g., GBDT/XGBoost vs. CNN-LSTM/TFT) for non-linear compound risk prediction, paired with exact polynomial-time **TreeSHAP** feature attribution. *(Phase 2)*
4. **Trend-Aware Alert Synthesis (Gap G4):**  
   Constrained natural language alert generation using **DSPy typed signatures** and Pydantic validation to guarantee zero numeric or entity hallucinations while communicating trajectory trends (rising, falling, stable). *(Phase 2)*

---

## 📍 Initial Study Region

- **Focus:** Coastal Karnataka & Western Ghats (Dakshina Kannada, Udupi, Uttara Kannada, Kodagu, Chikkamagaluru).
- **Scale:** ~5,800 H3 Resolution-7 cells.
- **Hazards:** Monsoonal flash floods, riverine flooding, and compound slope-failure landslides.

---

## 🛠️ Implementation Plan & Engineering Guardrails

For the detailed 14-increment TDD roadmap (Increments 0 to 13), architectural contracts, storage budgets, and evaluation metrics for Phase 1, see:

👉 **[Phase 1 Design and TDD Plan (`plan.md`)](plan.md)**  
👉 **[Component & Increment Explanation Guide (`INCREMENTS_EXPLANATION.md`)](INCREMENTS_EXPLANATION.md)**

---

## ⚡ Quickstart (Windows / macOS / Linux)

Anyone can clone and run this project in under a minute with zero platform-specific dependencies:

### 1. Prerequisites
* **Python 3.11+**
* **[`uv`](https://docs.astral.sh/uv/)** (recommended) or standard `pip`
* **Docker** (optional, only needed for containerized ClickHouse, Redis, and Redpanda)

### 2. Setup in 3 Commands
```bash
# 1. Clone repository
git clone https://github.com/tejas-bajaj/AI-Driven-Spatio-Temporal-Framework-for-Multimodal-Disaster-Intelligence-and-Early-Warning.git
cd AI-Driven-Spatio-Temporal-Framework-for-Multimodal-Disaster-Intelligence-and-Early-Warning

# 2. Install dependencies (instant lockfile sync)
uv sync
# (Or using standard pip: pip install -e ".[dev]")

# 3. Verify suite (runs 100% offline in < 2 seconds)
uv run pytest -m unit

# 4. Replay recorded telemetry fixture through ingestion pipeline
uv run ingest --replay tests/fixtures/open_meteo_sample.json
```

### 3. Start Local Storage Infrastructure (needed for Increment 7+ integration tests)
```bash
docker compose up -d --wait
uv run pytest -m integration      # ClickHouse cold store (Increment 7), service health
uv run pytest -m perf -s          # storage budgets; prints bytes per row, hot memory + latency
uv run python scripts/hot_demo.py # Increment 9: Redis hot tier demo (wipe + rebuild from ClickHouse)
make rebuild-hot                  # repopulate Redis from ClickHouse
```

---

## 👥 Team & Academic Context

- **Institution:** National Institute of Technology Karnataka (NITK), Surathkal — Department of Information Technology
- **Guide:** Dr. Sowmya Kamath S.
- **Team Members:**
  - Tejas Bajaj (231IT080)
  - Garv Mandloi (231IT023)
  - Swaroop Raghavendra Rao (231IT078)
