# Phase 1 Design and TDD Plan — Ingestion and WorldState



> **17 Oct 2026 · Updated 02 Nov 2026 · @Tejas**

>

> Phase 1 builds a tested path from coastal Karnataka and Western Ghats telemetry

> to a per-cell WorldState in **19 small increments (Increments 0 to 18)**, each starting with a failing

> test, so **Objectives 1 and 2** are demonstrable by the 7th-semester end-sem evaluation.



---



## Table of Contents



1. [Project Overview](#1-project-overview)

2. [Decisions and Assumptions](#2-decisions-and-assumptions)

3. [Study Region](#3-study-region)

4. [The TDD Loop — How We Work](#4-the-tdd-loop--how-we-work)

5. [Test Layers](#5-test-layers)

6. [Architecture Overview](#6-architecture-overview)

7. [Increments 0–6: Foundations and Ingestion (Objective 1)](#7-increments-06-foundations-and-ingestion-objective-1)

8. [Increments 7–10: The WorldState Store (Objective 2)](#8-increments-710-the-worldstate-store-objective-2)

9. [Increments 11–13: Imagery, Redpanda and Historical Replay](#9-increments-1113-imagery-redpanda-and-historical-replay)

10. [Increments 14–18: Multimodal Earth Observation Expansion (Objective 1 & 2)](#10-increments-1418-multimodal-earth-observation-expansion-objective-1--2)

11. [Adapter Recipe (Repeatable)](#11-adapter-recipe-repeatable)

12. [Storage Budget and Enforcement](#12-storage-budget-and-enforcement)

13. [Schedule](#13-schedule)

14. [Risks and Mitigations](#14-risks-and-mitigations)

15. [Open Questions](#15-open-questions)

16. [Definition of Done — Phase 1](#16-definition-of-done--phase-1)



---



## 1. Project Overview



**Title:** AI-Driven Spatio-Temporal Framework for Multimodal Disaster Intelligence and Early Warning



**Team:** Tejas Bajaj, Garv Mandloi, Swaroop Raghavendra Rao

**Guide:** Dr. Sowmya Kamath S. (NITK Surathkal)



### The Four Objectives



| Objective | Gap | Summary |

|-----------|-----|---------|

| **1 (G1)** | Multimodal Telemetry Ingestion | Ingest weather, seismic, satellite and alert feeds; discretize onto an H3 hex grid; measure latency and completion rate |

| **2 (G2)** | Continuous Hexagonal WorldState | Dual-tier feature store (Redis hot / ClickHouse cold) with rolling temporal windows; deterministic rollback to any past time |

| **3 (G3)** | Compound Multi-Hazard Risk Scoring | GBDT/XGBoost vs CNN-LSTM/TFT evaluation; TreeSHAP feature attributions *(Phase 2)* |

| **4 (G4)** | Trend-Aware Alert Synthesis | DSPy typed signatures, Pydantic validation, trend tracking *(Phase 2)* |



### Phase 1 Scope



Phase 1 covers **Objectives 1 and 2 only**. Objectives 3 and 4, infrastructure impact modelling, all-India scale, CWC/India-WRIS integration, and live Bhuvan are **out of scope** for Phase 1.



---



## 2. Decisions and Assumptions



Items marked **(assumed)** were not confirmed with the guide; changing them adjusts the plan accordingly.



| Area | Decision | Rationale |

|------|----------|-----------|

| **Language & tooling** | Python 3.11+, `uv`, `pytest`, `hypothesis`, `ruff`, `mypy` **(assumed)** | Single language end-to-end, fast test loop |

| **Environment** | Windows + WSL2 + Docker Desktop (WSL2 backend); code and containers inside WSL2 **(assumed)** | Redpanda, ClickHouse and Redis are Linux-first; `/mnt/c` is slow |

| **Region** | Coastal Karnataka + Western Ghats (5 districts), H3 res-7, ~5,800 cells | Floods + landslides, compound cascades, close to campus |

| **Streaming bus** | `Bus` interface with in-memory impl first; Redpanda at increment 12; same contract tests | TDD runs in ms with no containers |

| **Source order** | No-key first: Open-Meteo, USGS, GDACS, SACHET CAP. Then IMERG, SMAP. Last: Sentinel, SWOT, NISAR | Working pipeline early; precipitation source is a config swap |

| **Imagery** | Prithvi-EO-2.0 behind an `Embedder` interface; fake in unit tests, real in slow tests; batch on Colab/Kaggle/lab GPU **(GPU assumed unavailable locally)** | Fast tests + real model checked early |

| **Storage** | Derived data only, no raw scenes; float16 embeddings; short bus retention | See [Storage Budget](#12-storage-budget-and-enforcement) |

| **Radar Inundation** | Sentinel-1 C-SAR GRD (VV/VH backscatter in dB, thresholding $\le -18\text{ dB}$ for open water) | Cloud-penetrating all-weather flood surface water mapping during monsoon cloud blindness |

| **Multimodal Model** | Clay Foundation Model (`ClayEmbedder` ViT) alongside Prithvi-EO-2.0 baseline | Multimodal ViT fuses SAR + optical + DEM into unified 32-dim latent space; radar pass updates embedding even when cloud-covered |

| **Rapid-Scan Geostationary** | ISRO INSAT-3D/3DR 15-minute Thermal IR (TIR1 ~10.8 µm) | Convective cloudburst proxy via cloud-top brightness temperature cooling rate ($\Delta T/\Delta t$) 15–60 min before rain reaches ground |

| **Geological Stability** | Geological Survey of India (GSI) 1:50k NLSM Landslide Susceptibility & Lithology | Ground-truth rock cohesion & shear strength; distinguishes brittle fissured slopes from stable bedrock |

| **Out of scope** | Objectives 3 & 4, infrastructure impact, all-India scale, CWC, India-WRIS, live Bhuvan | Matches the proposal's own scope notes |



---



## 3. Study Region



**Districts:** Dakshina Kannada, Udupi, Uttara Kannada, Kodagu, Chikkamagaluru



The Western Ghats districts are included from the start because rainfall and landslides there drive what happens downstream on the coast. Kodagu and Chikkamagaluru add ~2,200 cells. Chikkamagaluru's eastern half is plains, so a Ghats-only polygon could trim the count.



- **H3 resolution:** 7 (~5.16 km² per cell)

- **Estimated cell count:** 5,400–6,200 cells

- **Configuration:** The region boundary is a polygon in config; its extent can change without code changes.



---



## 4. The TDD Loop — How We Work



Every increment follows the same discipline:



1. **Write the test list first** (names only) — scope is fixed before code exists.

2. **One test at a time:** red → green → refactor. Commit at each green.

3. **Unit tests never touch the network or the clock.**

   - Time comes from an injected `Clock`.

   - `pytest-socket` blocks all sockets in unit tests.

4. **Fixtures:** API responses are recorded once into small committed fixture files with a script, then parsed offline forever.

5. **Differential testing:** Where a database or index computes something, keep a slow, obvious Python reference implementation and test that the two agree on random data.

6. **Regression tests:** Every bug gets a failing test before the fix.

7. **Merge only when the full suite is green.**



---



## 5. Test Layers



| Layer | Marker | What It Checks | When It Runs |

|-------|--------|----------------|--------------|

| **Unit** | `unit` | H3 wrapper, schema validation, discretizers, window reference, patch pooling | On every save, **< 10 seconds** |

| **Contract** | `contract` | Adapters turn recorded responses into expected observations; every `Bus` impl obeys one shared test suite | On every push |

| **Integration** | `integration` | Real ClickHouse, Redis and Redpanda in containers; DB results equal the Python reference | Before merge |

| **Performance & Budget** | `perf` | Latency percentiles and bytes-per-cell-day against stated budgets | Nightly or on demand |

| **Live & Slow** | `live`, `slow` | Real APIs and real Prithvi weights | Manually, **never in CI** |



---



## 6. Architecture Overview



```

┌────────────────────────────────────────────────────────────────────────────────────────┐

│                                   Source Adapters                                      │

│  Open-Meteo │ USGS │ GDACS │ SACHET │ IMERG │ SMAP │ Sentinel-1 SAR │ INSAT-3D Rapid-IR│

└──────┬───────┴──────┴───────┴────────┴───────┴──────┴───────┬────────┴────────┬────────┘

       │                                                      │                 │

       ▼                                                      ▼                 ▼

┌────────────────────────────────────────────────────────────────────────────────────────┐

│                                 Spatial Discretizers                                   │

│            point → cell, grid-node fan-out, polygon fill, raster H3-pooling            │

└───────────────────────────────────────────┬────────────────────────────────────────────┘

                                            │  Observation (one per cell, idempotency-keyed)

                                            ▼

┌────────────────────────────────────────────────────────────────────────────────────────┐

│                                     Bus (interface)                                    │

│                 InMemoryBus (dev/test)  │  RedpandaBus (streaming production)          │

│                       contract-tested against the same test suite                      │

└───────────────────────────────────────────┬────────────────────────────────────────────┘

                                            │

        ┌───────────────────────────────────┴───────────────────────────────────┐

        ▼                                                                       ▼

┌───────────────────────────────┐                       ┌───────────────────────────────┐

│        Redis Hot Tier         │                       │     ClickHouse Cold Store     │

│      7-day bucket ring        │                       │      ReplacingMergeTree       │

│     ~3.9–6.7 KB / cell        │                       │    hourly aggregate buckets   │

│       ~40 MB total            │                       │      delta + ZSTD codecs      │

│   p95 read < 1.5ms (noevict)  │                       │    ~2–3 GB / year compressed  │

└───────────────┬───────────────┘                       └───────────────┬───────────────┘

                │                                                       │

                └───────────────────────────┬───────────────────────────┘

                                            ▼

┌────────────────────────────────────────────────────────────────────────────────────────┐

│                              State Vector Assembler                                    │

│                              get_state(cell, t, tier)                                  │

│                                                                                        │

│  FeatureSpec.v1 (29 dims) ──▶ FeatureSpec.v2 (63 dims) ──▶ FeatureSpec.v3 (86 dims)    │

│   - Weather / Seismic / River  - Optical Prithvi PCA        - Sentinel-1 SAR Water     │

│   - Soil Moisture / Cascades   - Embedding Age & Coverage   - INSAT-3D Cooling Rate    │

│   - Copernicus DEM & WorldPop                               - GSI Landslide Suscept.   │

│                                                             - Clay Multimodal ViT      │

└────────────────────────────────────────────────────────────────────────────────────────┘

```



> **Key principle:** Imagery enters the bus in the **same per-cell format** as every

> other feed — the stores and the assembler never special-case a source.



---



## 7. Increments 0–6: Foundations and Ingestion (Objective 1)



> By increment 6, recorded telemetry flows from an adapter through the bus onto

> the H3 grid and into a sink, with Objective 1 metrics measured.



### Increment 0 — Repository Bootstrap



| | |

|---|---|

| **Build** | Repo, `uv` project, `Makefile`, `pre-commit`, CI, `docker-compose` with ClickHouse, Redis, Redpanda (memory-capped) |

| **Tests written first** | Package imports pass; a unit test that opens a socket **fails**; each container answers a health-check ping |

| **Done when** | `make test` green locally and in CI; all three services healthy |



### Increment 1 — H3 Grid Core and Coastal Karnataka Cell Set



| | |

|---|---|

| **Build** | H3 grid core and the coastal Karnataka + Western Ghats cell set |

| **Tests written first** | Known lat/lon → known cell ID; cell centre round-trips to same cell (`hypothesis`); mean res-7 area ≈ 5.16 km²; every cell has 6 neighbours; 5-district polygon fills to ~5,400–6,200 cells with no duplicates |

| **Done when** | `CoastalGrid` returns a stable sorted cell list, cached to disk |



### Increment 2 — Data Model



| | |

|---|---|

| **Build** | `RawReading` (point / grid-node / polygon geometry), `Observation` (one per cell), variable catalog |

| **Tests written first** | Naive datetimes rejected; NaN and out-of-range coordinates rejected; everything normalised to UTC; unit conversions (K→°C, mm/hr→mm); stable idempotency key; JSON round-trip identical |

| **Done when** | Catalog covers: precipitation, two soil-moisture layers, temperature, wind, pressure, discharge, seismic magnitude, CAP severity |



### Increment 3 — SourceAdapter Interface and Open-Meteo Adapter



| | |

|---|---|

| **Build** | `SourceAdapter` interface and the Open-Meteo adapter |

| **Tests written first** | Recorded response → exact expected readings; null hours skipped with counted reason; request batching under call budget; 429 & 5xx retried with backoff on fake clock; malformed JSON → `ParseError` |

| **Done when** | Adapter runs offline from fixtures; one `@live` test passes by hand |



### Increment 4 — Spatial Discretizers



| | |

|---|---|

| **Build** | Point, grid-node and polygon discretizers |

| **Tests written first** | Point maps to same cell as H3 library; every region cell maps to exactly one 0.1° weather grid node; constant raster stays constant; gradient raster gives monotone cell values; CAP polygon over known district yields cells whose centres are all inside it |

| **Done when** | Cell-to-source-node fan-out table precomputed and cached |



> **Design note:** The weather source returns values on a 0.1° grid. The adapter

> fetches each unique grid node **once** and fans the value out to the ~20 cells it

> covers, rather than calling the API once per cell. This is the point of the

> fan-out table.



### Increment 5 — Bus Interface and In-Memory Bus



| | |

|---|---|

| **Build** | `Bus` interface and `InMemoryBus` |

| **Tests written first** | Order preserved per key (key = H3 cell); consumer-group offsets; replay from offset; duplicates after simulated crash; bounded queue pushes back; one topic per source |

| **Done when** | Contract suite is **reusable against any Bus implementation** |



### Increment 6 — Ingestion Pipeline and Objective 1 Metrics



| | |

|---|---|

| **Build** | End-to-end ingestion pipeline and Objective 1 metric reporting |

| **Tests written first** | Fake adapter → sink gives expected observations; bad record counted (not fatal); duplicates dropped by idempotency key; latency p50/p95/p99 correct on known inputs; completion rate attributes failures to source or parse |

| **Done when** | `ingest --replay fixture` fills an in-memory sink and prints latency + completion metrics |



---



## 8. Increments 7–10: The WorldState Store (Objective 2)



> By increment 10, any cell has a versioned state vector, available fast from

> the hot tier and reproducible for any past time from the cold tier.



### Increment 7 — ClickHouse Cold Store



| | |

|---|---|

| **Build** | Observations table and `ColdStore` adapter |

| **Tests written first** | Write-then-read equals reference copy kept at arrival (cold-correctness metric); duplicate inserts collapse after deduplication; ClickHouse parent-cell results equal Python H3 library on 1,000 random cells; re-running migrations is idempotent |

| **Done when** | Columns use compact types and codecs; bytes-per-row measured and under budget |



> **Design note:** ClickHouse `ReplacingMergeTree` deduplicates only when parts

> merge, so reads **must** deduplicate explicitly. Increment 7 pins this behaviour

> with a test.



### Increment 8 — Hourly Buckets and Rolling Windows



| | |

|---|---|

| **Build** | Hourly aggregate buckets and rolling windows: 1h, 6h, 24h, 72h, 7d |

| **Tests written first** | Slow Python `reference_windows` is the oracle; ClickHouse windows equal it on random series for every variable and window; a late observation fixes the right buckets; gaps report coverage fraction; windows at past time T computed today equal those computed live at T |

| **Done when** | `windows(cell, as_of)` returns values plus coverage |



> **Design note:** An incremental materialized view updates when rows arrive, but

> a sliding window like "last 72 hours" keeps moving when nothing arrives. So the

> materialized view maintains **hourly aggregate buckets** (sum for rainfall, average

> for soil moisture, max for discharge and seismic magnitude), and the window is a

> cheap combine of ≤168 buckets at read time.



### Increment 9 — Redis Hot Tier and Archive Job



| | |

|---|---|

| **Build** | Redis hot tier and the cold-to-hot archive/rebuild job |

| **Tests written first** | Hot windows equal oracle after replaying a stream; expired buckets evicted; applying one observation twice changes nothing; wiping Redis and rebuilding from cold tier → identical state; read/write latency percentiles and memory-per-cell in budget |

| **Done when** | Hot read and write latency reported as average, p50, p95, p99 |



> **As built (Increment 9):** the hot tier stores compressed *observations* per (cell, variable,

> source) and computes windows at read time; a per-bucket aggregate ring cannot make

> "applying one observation twice changes nothing" true. Measured ~3.9 KB/cell (budget 8 KB),

> read p95 ~1.3 ms, write p95 ~2 ms. Redis uses `noeviction` so state is never silently dropped.

> See `INCREMENTS_EXPLANATION.md` section 8d.



### Increment 10 — State Vector Assembler and Static Layers



| | |

|---|---|

| **Build** | State vector assembler and static geographic layers |

| **Tests written first** | Vector length and order match versioned `FeatureSpec`; compound saturation = soil_moisture × rolling_rainfall (zero if either is zero, rises with both); synthetic sloped plane gives known slope; missing sensor → NaN (never 0); `get_state(cell, as_of)` agrees between hot and cold; loading old vector under new spec **fails loudly** |

| **Done when** | Static elevation, slope and population density loaded once per cell; state vector ready for Objective 3 |



> **Static layer candidates:** SRTM or Copernicus DEM (elevation/slope), WorldPop

> or GHSL (population density). Final selection is a decision inside increment 10.



> **As built (Increment 10):** new package `src/disaster_intel/state/`.

> `FeatureSpec.v1` = 25 dynamic features (5 variables x 5 windows) + 3 static + 1 derived

> (`compound_saturation = soil_moisture_rootzone__24h x precipitation__72h`) = 29, fingerprinted

> so a vector saved under another spec fails to load (`SpecMismatchError`). Missing = NaN, never 0.

> Static layers: elevation = Copernicus DEM GLO-90 via Open-Meteo `/v1/elevation`; slope = plane

> fit through the cell centre + 6 neighbour centres; population = WorldPop 1 km count raster

> (GeoTIFF/ASCII) aggregated to cells. `StateAssembler.get_state(cell, as_of, tier)` serves "now"

> from Redis and any past time from ClickHouse; `compare_tiers` checks they agree.



---



## 9. Increments 11–13: Imagery, Redpanda and Historical Replay



These three finish Phase 1: satellite embeddings join the state vector, the real

bus replaces the in-memory one, and a historical replay produces the evaluation

numbers.



### Increment 11 — Embedder Interface and Prithvi-EO-2.0



| | |

|---|---|

| **Build** | `Embedder` interface, `FakeEmbedder`, Prithvi-EO-2.0 embedder, chip-to-cell pooling, PCA reduction |

| **Tests written first** | Chips centred on res-6 parent cover centres of its 7 children; every patch centre maps to exactly one cell; pooling = mean of patches in each cell (fake embedder); each cell carries coverage fraction; cloudy scene skipped → previous embedding keeps ageing; PCA output is deterministic float16 within storage budget. **Slow test:** real Prithvi on one small chip gives correct shape, no NaN, same output twice |

| **Done when** | Embedding, its age and its coverage appended to `FeatureSpec` (version 2) |



> **As built (Increment 11):**

> `src/disaster_intel/imagery/` implements `Embedder` protocol, `FakeEmbedder`,

> `PrithviEmbedder` (NASA Prithvi-EO-2.0 6-band optical foundation model), `SatelliteChip`

> (centred on res-6 parents covering centres of all 7 children, 14x14 patch grid),

> `pool_chip_to_cells` (arithmetic mean over valid patches, per-cell coverage calculation),

> `PCAReducer` (deterministic SVD/projection to 32 dimensions in `float16`, exactly 64 bytes/cell

> satisfying storage budget), and `SceneRegistry` (skips cloudy scenes > 30% cloud cover,

> persists and ages previous valid embeddings). Appended 32 PCA dims, `embedding_age_hours`,

> and `embedding_coverage` to `FeatureSpec.v2()` (total 63 features).

> Verified by 10 tests in `tests/test_embedder.py` (all passing).



### Increment 12 — RedpandaBus and the Bus Switch



| | |

|---|---|

| **Build** | `RedpandaBus` implementation and configuration switch |

| **Tests written first** | Same bus contract suite passes **unchanged**; killing and restarting a consumer loses nothing and duplicates are dropped; topic retention is 24–72 hours; throughput smoke test sustains expected message rate |

| **Done when** | Config switch `bus: memory` ↔ `bus: redpanda` with end-to-end run on real containers |



> **As built (Increment 12):**

> `src/disaster_intel/bus/redpanda.py` implements `RedpandaBus(Bus)` with Kafka protocol support

> (via pure-Python `kafka-python-ng`), typed message serialization/deserialization (preserving

> `Observation` dataclasses, dicts, strings, and byte payloads), topic retention configuration

> strictly enforcing the 24–72 hour storage budget policy, bounded queue backpressure via

> `BufferFullError`, and pluggable transport (`RealKafkaTransport` for live Redpanda and

> `FakeKafkaTransport` for socketless unit tests). `get_bus("memory" | "redpanda")` provides the

> configuration switch (also controlled by `DISASTER_INTEL_BUS`). Verified by 23 unit tests in

> `tests/test_redpanda_bus.py` (reusing `BaseBusContractTests` unchanged) and 9 container tests

> in `tests/test_redpanda_integration.py`.



### Increment 13 — Backfill Mode and Phase 1 Evaluation Report



| | |

|---|---|

| **Build** | Historical backfill mode and the full evaluation report |

| **Tests written first** | Replaying same window twice → identical cold-store contents; replay runs well faster than real time; report contains every Obj 1 & 2 metric; sanity check: 72h rainfall in known heavy-monsoon window is well above quiet-period baseline |

| **Done when** | `make evaluate` writes the metrics report for the 7th-semester evaluation |



> **Candidate replay windows** (verify against historical records):

> - July 2021 Uttara Kannada floods

> - July 2024 Shirur landslide

> - 2018 and 2019 monsoon floods and landslides in Kodagu and along the coast

>

> Backfill uses the same adapters with historical time windows, so it **doubles as

> the training-data pipeline for Objective 3**.

>

> **As built (Increment 13):**

> `src/disaster_intel/backfill.py` implements `BackfillEngine`, `HistoricalWindow`, and benchmark

> disaster windows (`shirur_landslide_2024`, `uttara_kannada_floods_2021`, `kodagu_floods_2018`,

> and `quiet_period_baseline`). Provides idempotent replay (replaying identical window yields 0

> duplicate records stored) and high throughput (>100x to >1,000,000x real-time speedup).

> `src/disaster_intel/evaluation.py` implements `Phase1Evaluator` and `Phase1EvaluationReport`,

> evaluating all Objective 1 and Objective 2 criteria from Section 15: ingestion latency percentiles

> (p50/p95/p99), 100% completion rate with failure attribution, hot-tier p95 read latency (<5ms),

> 100% cold-tier correctness vs arrival copy, 100% rolling window correctness vs Python oracle,

> deterministic rollback (recomputing past T equals live T), and strict storage budgets validation

> (ClickHouse delta+ZSTD, 64-byte embeddings, <=8KB Redis). `make evaluate` and `scripts/evaluate.py`

> run the full evaluation demo for the 7th-semester review. Verified by 5 tests in

> `tests/test_backfill.py` (301 total unit tests passing).



---



## 10. Increments 14–18: Multimodal Earth Observation Expansion (Objective 1 & 2)



> Increments 14 through 18 expand Phase 1 with all-weather radar, rapid-scan geostationary

> thermal infrared, geological lithology, and a multimodal foundation model (Clay).

> **Scope boundary strictly maintained:** This expansion operates solely within **Objective 1

> (Multimodal Telemetry Ingestion)** and **Objective 2 (Continuous Hexagonal WorldState)**.

> Objectives 3 (ML risk scoring) and 4 (LLM alert synthesis) remain postponed for Phase 2.



### 10.1 Motivation: The Monsoon Cloud Blindness Dilemma



In the Western Ghats and Coastal Karnataka (Dakshina Kannada, Udupi, Uttara Kannada, Kodagu,

and Chikkamagaluru), disasters like flash floods, river overflows, and debris-flow landslides

occur almost exclusively during the Southwest Monsoon (June through September).



During these catastrophic events, cloud cover is persistent and heavy—frequently exceeding

**90% to 100% regional cloud fraction for weeks at a time**.



* **The Optical Blindspot:** Optical satellites (Sentinel-2, Landsat) cannot see through clouds.

  NASA Prithvi-EO-2.0, while powerful, is an optical-only foundation model. During monsoon storms,

  optical scenes are completely masked. The system correctly skips them and increments

  `embedding_age_hours`, but this means the visual feature vector remains a static historical

  prior from weeks or months earlier.

* **The Fatal Gap:** At the exact moment when extreme weather is unfolding and early warning is

  most needed, an optical-only pipeline goes blind.

* **The Solution:** A zero-redundancy multi-sensor expansion combining:

  1. **All-weather cloud-penetrating synthetic aperture radar (Sentinel-1 C-SAR)** to map ground flood

     water through clouds and darkness.

  2. **High-cadence geostationary thermal infrared (ISRO INSAT-3D/3DR)** to detect rapid cloud-top

     cooling 15–60 minutes *before* rain reaches the ground.

  3. **In-situ geological field mapping (GSI NLSM & Lithology)** to provide physical rock/soil

     shear strength constraints on steep slopes.

  4. **A multimodal foundation model (Clay Foundation Model)** that natively accepts radar alone,

     optical alone, or both combined into a unified 32-dimensional latent space.



---



### 10.2 Plain-English Guide: What Each New Feature Adds



For non-remote-sensing specialists and evaluators, this table explains what each new source

physically detects, why it saves lives, and the concrete feature names it contributes:



| Data Source | Sensor / Physics | What It Physically Sees | Disaster Intelligence (Layman's Terms) | Added Features |

|---|---|---|---|---|

| **Sentinel-1 C-SAR** | Active 5.4 GHz microwave radar | Radar pulses pass straight through dense monsoon clouds, rain, and night darkness. Calm standing water acts like a mirror, reflecting radar pulses away (pitch-black backscatter $\le -18\text{ dB}$). | **All-weather flood map:** Shows exactly which 5 km² hex cells are inundated with floodwater right now, even in the middle of a torrential monsoon storm. | `sar_water_fraction`, `sar_vh_backscatter_db` |

| **Clay Foundation Model** | Self-supervised Multimodal ViT-MAE | Fuses optical light, radar microwave backscatter, and 3D elevation into a unified representation. Handles missing bands dynamically. | **Unblinded landscape representation:** Unlike optical models that go blind in rain, Clay keeps updating the 32-dim cell embedding using radar passes while preserving pre-monsoon landscape context. | `embedding_00` .. `embedding_31` (multimodal) |

| **ISRO INSAT-3D/3DR** | Geostationary Thermal Infrared (~10.8 µm) | Rapid 15-minute thermal images from 36,000 km in space. Towering cumulonimbus storm clouds push their tops into the high freezing atmosphere ($-60^\circ\text{C}$ to $-80^\circ\text{C}$). | **15–60 min early warning of cloudbursts:** If cloud-top temperature plunges rapidly ($\Delta T / \Delta t < -10\text{ K/15min}$), a violent convective updraft is surging *before* rain hits ground rain gauges. | `insat_cloud_top_tb`, `insat_convective_cooling_rate` |

| **GSI Lithology & Susceptibility** | Geological Survey of India 1:50,000 field maps | Maps bedrock types (granite vs. weathered laterite/schist), shear planes, and historical landslide scars. | **Slope collapse vulnerability:** Tells the system whether a steep slope is solid rock that won't budge, or loose weathered clay that will liquefy into a mudslide under 72h heavy rain. | `gsi_susceptibility_index`, `gsi_lithology_cohesion` |



---



### 10.3 The Zero-Redundancy Observational Matrix



Every layer in the expanded framework captures a physically distinct, non-overlapping stage of

disaster dynamics:



```

                  ┌──────────────────────────────────────────────┐

  T - 60 min      │ INSAT-3D Rapid Geostationary Thermal IR      │ ──▶ Atmospheric Convection

  (Pre-Storm)     │ Rapid cloud-top freezing & convective updraft│     (Nowcasting cloudburst)

                  └──────────────────────┬───────────────────────┘

                                         │ Rain falls to ground

                                         ▼

                  ┌──────────────────────────────────────────────┐

  T = 0 to 72h    │ Open-Meteo / IMERG / SMAP                    │ ──▶ Hydrometeorological Flux

  (During Storm)  │ Rainfall volume (mm) & root-zone saturation  │     (Water loading on ground)

                  └──────────────────────┬───────────────────────┘

                                         │ Surface runoff & infiltration

                                         ▼

                  ┌──────────────────────────────────────────────┐

  T = Real-Time   │ Sentinel-1 C-SAR Active Microwave Radar      │ ──▶ Ground Surface State

  (All-Weather)   │ Cloud-penetrating flood inundation mapping   │     (Direct water detection)

                  └──────────────────────┬───────────────────────┘

                                         │ Physical terrain constraints

                                         ▼

                  ┌──────────────────────────────────────────────┐

  Static Baseline │ Copernicus DEM + GSI NLSM Lithology          │ ──▶ Physical Susceptibility

  (Terrain Physics)│ Gravitational slope + rock mechanical cohesion│    (Gravitational failure)

                  └──────────────────────────────────────────────┘

```



| Source | Physical Modality | Temporal Cadence | Spatial Scale | Non-Redundant Role |

|---|---|---|---|---|

| **INSAT-3D/3DR** | Geostationary Thermal IR | 15 minutes | ~4 km | **Pre-storm nowcasting:** Convective updraft cooling rate 15–60 min before rain reaches ground gauges. |

| **Open-Meteo / IMERG** | Numerical Weather / Radar-gauge blend | 1 hour | 0.1° (~10 km) | **Precipitation volume:** Ingests actual rain depth (mm) and multi-day accumulation. |

| **Open-Meteo Soil** | Land Surface Model (ERA5-Land) | 1 hour | 0.1° (~10 km) | **Subsurface saturation:** Root-zone and surface soil water content. |

| **Sentinel-1 C-SAR** | Active 5.4 GHz C-band Radar | 6–12 days | 10–20 m | **Ground flood extent:** Cloud-penetrating direct measurement of standing water on terrain. |

| **Sentinel-2 MSI** | Optical (Visible + NIR + SWIR) | 5 days (cloud-free)| 10–20 m | **Surface state:** Pre-monsoon vegetation index, forest loss, bare road-cut excavation scars. |

| **Copernicus DEM** | Spaceborne InSAR elevation | Static | 30–90 m | **Topography:** Gravitational acceleration potential, slope angle, valley funneling. |

| **GSI NLSM** | In-situ Geological Survey | Static | 1:50,000 | **Mechanical shear strength:** Soil/rock cohesion, structural fault lines, slip plane propensity. |

| **WorldPop** | Census + Settlement Density | Static (annual) | 100 m | **Consequence exposure:** Human density and settlement vulnerability per cell. |



---



### 10.4 Foundation Model Architecture: Prithvi-EO-2.0 vs. Clay Foundation Model



A critical architectural decision is how satellite observations are encoded into the WorldState vector:



#### 1. Prithvi-EO-2.0 (The Optical Baseline)

* **Architecture:** NASA-IBM geospatial foundation model built on a Vision Transformer Masked Autoencoder (ViT-MAE).

* **Inputs:** 6 optical bands (Blue, Green, Red, Narrow NIR, SWIR-1, SWIR-2) from Harmonized Landsat-Sentinel (HLS).

* **Strengths:** Excellent representation of multispectral vegetation vigor (NDVI/EVI), bare soil scars, and urban impervious surfaces.

* **Limitation:** Purely optical. When clouds cover coastal Karnataka, all optical bands are missing. Prithvi cannot process radar.



#### 2. Clay Foundation Model (The Multimodal Champion)

* **Architecture:** Self-supervised Earth observation foundation model utilizing a 3D Vision Transformer with masked autoencoding.

* **Inputs:** Multimodal input tensor accepting Sentinel-2 optical (10 bands), Sentinel-1 SAR (VV, VH polarizations), Landsat, and Copernicus DEM elevation.

* **Key Mechanism:** Uses modality-specific patch embeddings and flexible masked cross-attention. If optical bands are cloud-obscured, optical patches are masked out, and the model embeds the available **Sentinel-1 SAR + DEM** patches directly into the shared latent space.

* **Advantage:** The cell representation is updated continuously during the monsoon whenever Sentinel-1 passes, without going blind.



#### Research & Thesis Strategy

We do **not** discard Prithvi. Instead, the `Embedder` protocol (`src/disaster_intel/imagery/embedder.py`)

supports both:

* **`PrithviEmbedder`:** Serves as the optical-only foundation baseline (Increment 11).

* **`ClayEmbedder`:** Serves as the multimodal foundation model (Increment 15).

* Both models project down to the **identical budget-compliant 32 float16 dimensions (64 bytes/cell)** via `PCAReducer`.

* This enables an empirical research comparison in the final evaluation: *State vector information preservation under heavy cloud cover: Optical-only (Prithvi) vs Multimodal (Clay).*



---



### 10.5 Increment 14 — Sentinel-1 C-Band SAR Radar Adapter & All-Weather Water Masking



| | |

|---|---|

| **Build** | `Sentinel1Adapter`, radiometric calibration to backscatter $\sigma^0$ in dB, specular water thresholding ($\sigma^0_{\text{VH}} \le -18\text{ dB}$), and H3 Res-7 spatial pooling (`sar_water_fraction`, `sar_vh_backscatter_db`) |

| **Tests written first** | 1. Linear amplitude converted bit-exactly to decibels: $\sigma^0 = 10 \log_{10}(\text{amplitude}^2)$.<br>2. Synthetic radar patch correctly classifies calm open water ($\le -18\text{ dB}$) vs rough vegetated terrain.<br>3. Spatial pooling aggregates radar pixels into H3 Res-7 cells, calculating valid coverage fraction and mean backscatter.<br>4. Missing/corrupted polarizations attributed to `SourceError` or `ParseError` without crashing the ingestion pipeline. |

| **Done when** | Adapter ingests recorded Sentinel-1 GRD fixtures offline, streams observations onto the bus, and populates `sar_water_fraction` and `sar_vh_backscatter_db`. |



> **Design note:** C-band radar (5.4 GHz) reflects specularly away from the sensor over smooth standing

> water, producing backscatter typically between $-20\text{ dB}$ and $-26\text{ dB}$. Rough forested

> hillslopes produce diffuse volumetric scattering ($-12\text{ dB}$ to $-8\text{ dB}$). The $-18\text{ dB}$

> threshold on VH cross-polarization provides an established, robust baseline for automated water extraction.

> **As built (Increment 14):**
> `src/disaster_intel/adapters/sentinel1.py` implements `Sentinel1Adapter(SourceAdapter)` with bit-exact
> radiometric calibration ($\sigma^0 = 10 \log_{10}(\text{amp}^2) = 20 \log_{10}(\text{amp})$), specular
> water classification ($\sigma^0_{\text{VH}} \le -18.0\text{ dB}$), and H3 Res-7 spatial pooling emitting
> `sar_water_fraction` and `sar_vh_backscatter_db` `RawReading`s. Registered variables in `VariableCatalog.default()`
> and `convert_unit()`. 16 unit tests in `tests/test_sentinel1_adapter.py` verify mathematical calibration,
> threshold boundaries, cell pooling, error attribution, golden fixture replay (`tests/fixtures/sentinel1_sample.json`),
> and full pipeline integration. Verified with `pytest -m unit` (317 passed).



---



### 10.6 Increment 15 — Multimodal Clay Foundation Model Embedder



| | |

|---|---|

| **Build** | `ClayEmbedder` implementing `Embedder` protocol in `src/disaster_intel/imagery/embedder.py`, multimodal tensor handling (Sentinel-1 VV/VH + Sentinel-2 optical + DEM), 32-dim PCA projection in `float16` (64 bytes/cell), and config-driven embedder switching |

| **Tests written first** | 1. `ClayEmbedder` conforms strictly to the `Embedder` protocol interface.<br>2. Cloud-mask fallback test: when optical channels are completely masked/NaN, embedder processes SAR + DEM channels successfully without returning all-NaN.<br>3. PCA projection outputs deterministic 32-dimensional `float16` vector within the 64-byte storage budget.<br>4. Config switch `imagery.embedder: "prithvi" | "clay" | "fake"` swaps models without changing downstream storage or assembler code. |

| **Done when** | `ClayEmbedder` generates 32-dim latent representations from synthetic multimodal chips in offline unit tests; benchmark script logs embedding distance between cloudy and clear passes. |



---



### 10.7 Increment 16 — ISRO INSAT-3D/3DR Geostationary Rapid-Scan Ingestion



| | |

|---|---|

| **Build** | `INSATAdapter` ingesting 15-minute geostationary thermal infrared (TIR-1, ~10.8 µm), computing convective cloud-top brightness temperature ($T_B$) and convective cooling rate ($\Delta T_B / \Delta t$) |

| **Tests written first** | 1. Parser converts synthetic INSAT HDF5/GeoTIFF raster into Kelvin temperatures.<br>2. Cold-top convective threshold detection: correctly identifies cloud-tops colder than $210\text{ K}$ ($-63^\circ\text{C}$), indicative of severe cumulonimbus storm cells.<br>3. Convective cooling rate calculation: $\Delta T_B / \Delta t = [T_B(t) - T_B(t - 15\text{min})] / 15\text{ min}$; flags rapid cooling $< -10\text{ K/15min}$ as convective updraft intensification.<br>4. Spatial discretizer fans out ~4 km geostationary pixels to H3 Res-7 cells. |

| **Done when** | 15-minute observations stream to the bus; rolling window aggregator and zero-order hold correctly compute hourly and multi-hour convective metrics. |



---



### 10.8 Increment 17 — GSI Geological Landslide Susceptibility & Lithology Static Layer



| | |

|---|---|

| **Build** | Integration of Geological Survey of India (GSI) 1:50,000 National Landslide Susceptibility Mapping (NLSM) and lithological classification into `StaticLayers` and `StateAssembler` |

| **Tests written first** | 1. H3 cell center coordinates correctly map to GSI susceptibility polygons.<br>2. Categorical susceptibility mapped to normalized $[0.0, 1.0]$ index (Low: 0.1, Medium: 0.4, High: 0.7, Very High: 1.0).<br>3. Lithological cohesion factor mapped to normalized shear resistance $[0.0, 1.0]$ based on rock type (e.g. weathered laterite: 0.2, sheared schist: 0.4, massive basalt/granite: 0.9).<br>4. Immutability: static geological features are loaded once and remain bit-exact across all historical timestamps $T$. |

| **Done when** | Static feature loader loads GSI features alongside Copernicus DEM and WorldPop; state vector assembler appends `gsi_susceptibility_index` and `gsi_lithology_cohesion` to static block. |



---



### 10.9 Increment 18 — FeatureSpec v3 & Full Multimodal WorldState Benchmark



| | |

|---|---|

| **Build** | `FeatureSpec.v3()` defining unified 86-feature state vector; historical replay expansion evaluating multi-sensor performance across the 2024 Shirur landslide and 2021 Uttara Kannada flood windows |

| **Tests written first** | 1. `FeatureSpec.v3()` validates all feature names, definitions, and types; loading vectors produced under v1 or v2 raises `SpecMismatchError`.<br>2. Redis hot tier and ClickHouse cold tier agree bit-exactly on the complete 86-feature vector (`compare_tiers` test).<br>3. Deterministic rollback: state vector reconstructed at historical time $T$ matches state computed live at $T$ across all multimodal feeds.<br>4. Head-to-head evaluation: comparative analysis of optical-only (Prithvi) vs multimodal (Clay) under heavy monsoon cloud simulations. |

| **Done when** | `make evaluate` runs the full multimodal suite, verifying all Objective 1 & 2 metrics, generating `PHASE1_MULTIMODAL_BENCHMARK.md`. |



> **FeatureSpec.v3 Breakdown (86 Total Features):**

> * **Dynamic Baseline (25 features):** 5 variables (rain, soil surface, soil rootzone, discharge, seismic) $\times$ 5 windows (1h, 6h, 24h, 72h, 7d).

> * **Dynamic Radar (10 features):** `sar_water_fraction` and `sar_vh_backscatter_db` $\times$ 5 windows.

> * **Dynamic Geostationary (10 features):** `insat_cloud_top_tb` and `insat_convective_cooling_rate` $\times$ 5 windows.

> * **Static Topography & Demographics (3 features):** `elevation_m`, `slope_deg`, `population_density_km2`.

> * **Static Geology & Lithology (2 features):** `gsi_susceptibility_index`, `gsi_lithology_cohesion`.

> * **Multimodal Foundation Embeddings (34 features):** 32 PCA dimensions (`embedding_00` .. `embedding_31`), `embedding_age_hours`, `embedding_coverage`.

> * **Compound Derived Features (2 features):**

>   - `compound_saturation = soil_moisture_rootzone__24h * precipitation__72h`

>   - `slope_shear_risk = slope_deg * gsi_susceptibility_index`



---



## 11. Adapter Recipe (Repeatable)



Every adapter after Open-Meteo (USGS, GDACS, SACHET CAP, Open-Meteo flood

discharge, IMERG, SMAP, Sentinel-1 SAR, INSAT-3D Rapid-IR) follows the same recipe:



1. Record one small fixture and commit it.

2. Write the **golden-parse test**, then the null and malformed cases.

3. Add the **failure-attribution test** (source error vs parse error).

4. Register the variable in the **catalog** and the topic in the **bus config**.



---



## 12. Storage Budget and Enforcement



For coastal Karnataka the pilot is small (~a few GB/year). The risks are raw

imagery and verbose messages, so the design stores **derived data only**.



> **Note:** These are initial estimates (~5,800 cells, 10 variables, hourly). They

> will be replaced by actual measurements from increments 7, 9 and 11.



### Budget Table



| Store | Rough Size | How It Stays Small |

|-------|------------|-------------------|

| **Cold observations (ClickHouse)** | ~15 GB/yr uncompressed, ~2–3 GB compressed | Float32 values, `LowCardinality` for source/variable, delta + ZSTD codecs, rows sorted by (cell, variable, time) |

| **Embeddings (Prithvi / Clay)** | ~64 bytes/cell/acquisition (32 reduced float16 dims) vs 4 KB for full 1024-dim float32 | PCA reduction, one write per new scene, never repeated hourly |

| **Sentinel-1 SAR Radar** | ~4 bytes/cell per pass (water fraction + backscatter) | Ingested every 6–12 days; < 500 KB/year total for region |

| **INSAT-3D Rapid-Scan IR** | ~8 bytes/cell per 15-min reading | Delta + ZSTD compression in ClickHouse (< 80 MB/year compressed) |

| **Raw satellite scenes** | ~1 GB/scene if kept | **Not kept.** Download → embed → pool to cells → delete. Keep only scene ID, date and cloud fraction |

| **Hot tier (Redis)** | ~6.7 KB/cell, ~40 MB total | Only the 7-day bucket ring per active cell |

| **Bus (Redpanda)** | ~280 MB/day @ 200 bytes/msg, ~850 MB for 72h retention | 24–72h retention, compact encoding, batched messages; ClickHouse is the record |



### Further Optimization Lever



If cold-store size or write rate becomes a problem, store gridded observations

once per weather grid node (~23 cells share each) and expand to cells when

windows are computed. Only do this if measurement says so, because compression

already removes most repeated values.



### Budget Enforcement Tests



The budget is enforced by **tests, not good intentions:**



| Rule | Enforcement |

|------|-------------|

| ClickHouse bytes-per-cell-day | After loading a synthetic day for all cells, query `system.parts` → must be under limit (set after first measurement, only tightened) |

| Embedding record size | Must be ≤ 64 bytes/cell; test fails if float64 or unreduced vector sneaks in |

| Redis memory per cell | Must stay under 8 KB |

| Bus topic retention | Asserted from broker's own configuration API |

| Raw scene cleanup | Raw scenes must **not exist on disk** after the embedding job finishes |



---



## 13. Schedule



Dates assume a **Monday start on October 19** and an end-sem in **mid-December**.

Imagery has its own week because it is the riskiest increment. The buffer week

comes last so a slip there costs nothing.



| Week | Increments | Focus |

|------|------------|-------|

| 1 | 0, 1 | Repo bootstrap, H3 grid |

| 2 | 2, 3 | Data model, Open-Meteo adapter |

| 3 | 4, 5 | Discretizers, Bus interface |

| 4 | 6 | Ingestion pipeline + Obj 1 metrics |

| 5 | 7, 8 | ClickHouse cold store, rolling windows |

| 6 | 9, 10 | Redis hot tier, state vector assembler |

| 7 | 11 | Imagery (Prithvi-EO-2.0) — **riskiest** |

| 8 | 12, 13 | Redpanda bus switch, backfill + baseline evaluation |

| 9 | 14, 15 | Sentinel-1 SAR radar adapter + Multimodal Clay embedder |

| 10 | 16, 17 | INSAT-3D geostationary rapid-scan + GSI geology layer |

| 11 | 18 | FeatureSpec v3 & Full Multimodal WorldState Benchmark |

| 12 | — | **Final buffer week & End-sem demo preparation** |



> **If behind schedule, cut in this order:** SWOT & NISAR → SMAP → extra CAP

> sources → imagery. Objectives 1 and 2 remain deliverable without imagery.



---



## 14. Risks and Mitigations



| Risk | Why It Matters | Mitigation |

|------|---------------|------------|

| **NASA Earthdata login** | Proposal lists IMERG and SMAP as no-key; they likely need a free Earthdata account *(verify)* | Create accounts now, read credentials from env vars, keep tests offline with fixtures |

| **API limits and outages** | Free weather APIs have usage limits *(verify current)* and go down | Fetch each grid node once, cache, back off on errors, keep last good value, test all three failure modes |

| **Precipitation product mismatch** | IMERG has early/late/final runs *(verify)*; training on one and serving on another shifts the data | Store product version on every observation; backfill with the same product used live |

| **Time handling** | Event time, ingest time, late data and time zones → silent bugs | UTC everywhere, injected `Clock`, windows keyed on event time, differential tests with late data |

| **Laptop resources** | ClickHouse + Redpanda + Redis + Python can starve a Windows machine | Memory cap in `.wslconfig`, Redpanda single-core dev mode, start only containers a test needs |

| **Imagery compute & cloud** | Prithvi needs a GPU for volume; monsoon cloud hides coast/Ghats from optical sensors for weeks | Batch jobs off-machine, cloud mask + age feature, imagery scheduled after tabular pipeline |

| **Schedule** | 19 increments across 3 people in ~11–12 weeks | Buffer week; graceful degradation cut order above |

| **Labels for Objective 3** | Training needs dated flood/landslide events; list takes time to assemble | Start event catalogue now (Copernicus EMS activations, landslide records; check availability) |

| **SAR speckle & Ghats terrain shadow** | Radar side-looking geometry creates layover/shadow on steep Western Ghats escarpments | Use VH cross-polarization and mask steep terrain shadows using Copernicus DEM slope & aspect |

| **INSAT parallax distortion** | Geostationary orbit viewing angle shifts cloud positions over coastal Western Ghats | Calibrate against IMERG/Open-Meteo coordinates; apply standard ISRO parallax shift correction |

| **Clay model weights & compute** | Clay weights require download and GPU execution for full scenes | Maintain `FakeEmbedder` and synthetic fixtures for socketless unit tests; run inference in batches on Colab/Kaggle/lab GPU |



---



## 15. Open Questions



- [ ] Is a GPU available (lab machine or cluster), or only Colab and Kaggle?

- [ ] What is the 7th-semester end-sem date? (needed to fix schedule's last week)

- [ ] How do the three team members want to split the work?

- [ ] Is Python for everything acceptable?

- [ ] Does anyone already have a NASA Earthdata account?



---



## 16. Definition of Done — Phase 1



Phase 1 is done when **`make evaluate` reproduces the table below from a clean

checkout.** Numeric targets are proposals — each is fixed after its first

measurement and then guarded by a test so it cannot quietly regress.



| Proposal Metric | Measured In | Phase 1 Target |

|-----------------|-------------|----------------|

| Ingestion latency — average and percentiles (Obj. 1) | Increments 6, 13 | Reported for p50, p95, p99; budget set after first run |

| Ingestion completion rate with failure attribution (Obj. 1) | Increments 6, 13 | ≥ 99% on replay, every failure attributed to source or parse |

| Hot-tier read/write latency — average and percentiles (Obj. 2) | Increment 9 | Proposed p95 < 5 ms per cell on dev machine |

| Cold-tier correctness vs arrival-time copy (Obj. 2) | Increment 7 | 100% identical records |

| Window correctness vs Python oracle | Increment 8 | 100% agreement on random test set |

| Deterministic rollback | Increments 9, 10, 13 | State at past time T equals live state computed at T |

| Storage budgets | Increments 7, 9, 11, 12, 14, 15, 16 | All budget tests green (<8KB Redis, 64-byte embeddings, <3GB CH) |

| All-weather flood & cloudburst tracking | Increments 14, 16, 18 | Ingests Sentinel-1 radar and INSAT-3D 15-min cooling rates without cloud dropouts |

| Multimodal foundation model comparison | Increments 11, 15, 18 | `PHASE1_MULTIMODAL_BENCHMARK.md` logs comparative performance under cloud cover |



### The Evaluation Demo



> Replay a heavy-monsoon window, then show one Western Ghats cell's state vector

> building up hour by hour, plus the metrics report.

