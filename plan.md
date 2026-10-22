# Phase 1 Design and TDD Plan — Ingestion and WorldState

> **17 Oct 2026 · @Tejas**
>
> Phase 1 builds a tested path from coastal Karnataka and Western Ghats telemetry
> to a per-cell WorldState in **14 small increments**, each starting with a failing
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
10. [Adapter Recipe (Repeatable)](#10-adapter-recipe-repeatable)
11. [Storage Budget and Enforcement](#11-storage-budget-and-enforcement)
12. [Schedule](#12-schedule)
13. [Risks and Mitigations](#13-risks-and-mitigations)
14. [Open Questions](#14-open-questions)
15. [Definition of Done — Phase 1](#15-definition-of-done--phase-1)

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
| **Storage** | Derived data only, no raw scenes; float16 embeddings; short bus retention | See [Storage Budget](#11-storage-budget-and-enforcement) |
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
┌────────────────────────────────────────────────────────────────────┐
│                         Source Adapters                             │
│  Open-Meteo │ USGS │ GDACS │ SACHET │ IMERG │ SMAP │ Sentinel...  │
└──────┬───────┴──────┴───────┴────────┴───────┴──────┴──────────────┘
       │  RawReading (point / grid-node / polygon geometry)
       ▼
┌──────────────────────┐
│  Spatial Discretizer  │  point → cell, grid-node fan-out, polygon fill
└──────┬───────────────┘
       │  Observation (one per cell, idempotency-keyed)
       ▼
┌──────────────────────┐
│    Bus (interface)    │  InMemoryBus  │  RedpandaBus
│  one topic per source │  contract-tested against the same suite
└──────┬───────────────┘
       │
  ┌────┴─────────────────────────────────┐
  ▼                                      ▼
┌─────────────────────┐    ┌──────────────────────────────┐
│  Redis Hot Tier      │    │  ClickHouse Cold Store        │
│  7-day bucket ring   │    │  ReplacingMergeTree           │
│  ~6.7 KB / cell      │    │  hourly aggregate buckets     │
│  ~40 MB total        │    │  delta + ZSTD codecs          │
└─────────┬───────────┘    │  ~2–3 GB / year compressed    │
          │                 └──────────────┬───────────────┘
          └────────┬───────────────────────┘
                   ▼
         ┌─────────────────────┐
         │  State Assembler     │
         │  get_state(cell, t)  │
         │  versioned FeatureSpec│
         └─────────────────────┘
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

### Increment 12 — RedpandaBus and the Bus Switch

| | |
|---|---|
| **Build** | `RedpandaBus` implementation and configuration switch |
| **Tests written first** | Same bus contract suite passes **unchanged**; killing and restarting a consumer loses nothing and duplicates are dropped; topic retention is 24–72 hours; throughput smoke test sustains expected message rate |
| **Done when** | Config switch `bus: memory` ↔ `bus: redpanda` with end-to-end run on real containers |

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

---

## 10. Adapter Recipe (Repeatable)

Every adapter after Open-Meteo (USGS, GDACS, SACHET CAP, Open-Meteo flood
discharge, IMERG, SMAP) follows the same recipe:

1. Record one small fixture and commit it.
2. Write the **golden-parse test**, then the null and malformed cases.
3. Add the **failure-attribution test** (source error vs parse error).
4. Register the variable in the **catalog** and the topic in the **bus config**.

---

## 11. Storage Budget and Enforcement

For coastal Karnataka the pilot is small (~a few GB/year). The risks are raw
imagery and verbose messages, so the design stores **derived data only**.

> **Note:** These are initial estimates (~5,800 cells, 10 variables, hourly). They
> will be replaced by actual measurements from increments 7, 9 and 11.

### Budget Table

| Store | Rough Size | How It Stays Small |
|-------|------------|-------------------|
| **Cold observations (ClickHouse)** | ~15 GB/yr uncompressed, ~2–3 GB compressed | Float32 values, `LowCardinality` for source/variable, delta + ZSTD codecs, rows sorted by (cell, variable, time) |
| **Embeddings** | ~64 bytes/cell/acquisition (32 reduced float16 dims) vs 4 KB for full 1024-dim float32 | PCA reduction, one write per new scene, never repeated hourly |
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

## 12. Schedule

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
| 8 | 12, 13 | Redpanda bus switch, backfill + evaluation |
| 9 | — | **Buffer week** |

> **If behind schedule, cut in this order:** SWOT & NISAR → SMAP → extra CAP
> sources → imagery. Objectives 1 and 2 remain deliverable without imagery.

---

## 13. Risks and Mitigations

| Risk | Why It Matters | Mitigation |
|------|---------------|------------|
| **NASA Earthdata login** | Proposal lists IMERG and SMAP as no-key; they likely need a free Earthdata account *(verify)* | Create accounts now, read credentials from env vars, keep tests offline with fixtures |
| **API limits and outages** | Free weather APIs have usage limits *(verify current)* and go down | Fetch each grid node once, cache, back off on errors, keep last good value, test all three failure modes |
| **Precipitation product mismatch** | IMERG has early/late/final runs *(verify)*; training on one and serving on another shifts the data | Store product version on every observation; backfill with the same product used live |
| **Time handling** | Event time, ingest time, late data and time zones → silent bugs | UTC everywhere, injected `Clock`, windows keyed on event time, differential tests with late data |
| **Laptop resources** | ClickHouse + Redpanda + Redis + Python can starve a Windows machine | Memory cap in `.wslconfig`, Redpanda single-core dev mode, start only containers a test needs |
| **Imagery compute & cloud** | Prithvi needs a GPU for volume; monsoon cloud hides coast/Ghats from optical sensors for weeks | Batch jobs off-machine, cloud mask + age feature, imagery scheduled after tabular pipeline |
| **Schedule** | 14 increments across 3 people in ~8 weeks | Buffer week; graceful degradation cut order above |
| **Labels for Objective 3** | Training needs dated flood/landslide events; list takes time to assemble | Start event catalogue now (Copernicus EMS activations, landslide records; check availability) |

---

## 14. Open Questions

- [ ] Is a GPU available (lab machine or cluster), or only Colab and Kaggle?
- [ ] What is the 7th-semester end-sem date? (needed to fix schedule's last week)
- [ ] How do the three team members want to split the work?
- [ ] Is Python for everything acceptable?
- [ ] Does anyone already have a NASA Earthdata account?

---

## 15. Definition of Done — Phase 1

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
| Storage budgets | Increments 7, 9, 11, 12 | All budget tests green |

### The Evaluation Demo

> Replay a heavy-monsoon window, then show one Western Ghats cell's state vector
> building up hour by hour, plus the metrics report.
