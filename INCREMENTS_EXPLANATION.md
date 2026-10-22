# AI-Driven Spatio-Temporal Framework for Multimodal Disaster Intelligence
## Increments & Architecture Explanation Guide

This document provides a comprehensive, function-by-function and module-by-module explanation of everything built in the project so far. It explains **what** each component does, **why** it was designed that way, and **how** it connects to the broader disaster intelligence pipeline.

---

## Table of Contents
1. [Core Design Philosophy & TDD Discipline](#1-core-design-philosophy--tdd-discipline)
2. [Increment 0: Project Scaffold & Infrastructure Config](#2-increment-0-project-scaffold--infrastructure-config)
3. [Increment 1: H3 Spatial Grid (`CoastalGrid`)](#3-increment-1-h3-spatial-grid-coastalgrid)
4. [Increment 2: Multimodal Telemetry Data Model (`models.py`)](#4-increment-2-multimodal-telemetry-data-model-modelspy)
5. [Increment 3: Telemetry Adapters & Deterministic Clock (`clock.py`, `adapters/`)](#5-increment-3-telemetry-adapters--deterministic-clock-clockpy-adapters)
6. [Increment 4: Spatial Discretizers & Fan-Out Table (`discretizers.py`)](#6-increment-4-spatial-discretizers--fan-out-table-discretizerspy)
7. [Increment 5: Streaming Bus Interface & In-Memory Bus (`bus/`)](#7-increment-5-streaming-bus-interface--in-memory-bus-bus)
8. [Current System Status & Test Suite Summary](#8-current-system-status--test-suite-summary)
9. [Upcoming Increments Roadmap (Increments 6 to 14)](#9-upcoming-increments-roadmap-increments-6-to-14)

---

## 1. Core Design Philosophy & TDD Discipline

Every single feature in this project adheres to strict **Test-Driven Development (TDD)** and clean hexagonal architecture:

1. **Tests Are Written First (RED)**: We write unit tests specifying edge cases and constraints before creating the implementation. Code is only written to make tests pass (GREEN), then refactored.
2. **Network Isolation**: Unit tests never touch the network (`pytest-socket` blocks all socket creation during `@pytest.mark.unit`). External API calls run only through recorded JSON fixtures or injected mock transports.
3. **Deterministic Clock**: Unit tests never sleep or rely on system wall clocks; time is provided by an injected `Clock` protocol (`FakeClock`), allowing instantaneous testing of retry backoffs.
4. **Idempotency & Deduplication**: Every telemetry reading generates a deterministic hash-based key, ensuring retries and duplicate transmissions never corrupt historical aggregates.

---

## 2. Increment 0: Project Scaffold & Infrastructure Config

### Purpose
Establish a production-grade Python package foundation with strict quality gates, dependency pinning, and container orchestration for hot/cold storage and message streaming.

### Key Components
* **[`pyproject.toml`](pyproject.toml)**: Configures `hatchling` build system, Python >= 3.11, strict ruff linter (`select = ["E", "F", "W", "I", "UP", "B", "SIM", "RUF"]`), strict mypy type checking, and pytest marker definitions (`unit`, `contract`, `integration`, `perf`, `live`, `slow`).
* **[`docker-compose.yml`](docker-compose.yml)**: Defines three core services:
  * **ClickHouse** (`clickhouse/clickhouse-server:24.3`): Columnar analytical database with `ReplacingMergeTree` for permanent, compressed observation storage.
  * **Redis** (`redis:7-alpine`): In-memory store holding a 7-day rolling window per cell for sub-millisecond AI model inferences.
  * **Redpanda** (`vectorized/redpanda:v23.3`): High-throughput, Kafka-compatible message bus for event ingestion.
* **[`study_region.json`](src/disaster_intel/config/study_region.json)**: Exact latitude/longitude polygon boundary enclosing Coastal Karnataka (Uttara Kannada, Udupi, Dakshina Kannada) and Western Ghats catchments (Shivamogga, Kodagu).

---

## 3. Increment 1: H3 Spatial Grid (`CoastalGrid`)

### Purpose
Solves **Gap G1 (Spatial Resolution Mismatch)**. Satellite and weather models use rectangular lat/lon grids with distortion towards the poles, while ground stations are scattered points. Uber's **H3 hexagonal hierarchical spatial index** provides equal-area, distortion-minimized cells where every cell has exactly 6 equidistant neighbors.

### Key Class: `CoastalGrid` ([`src/disaster_intel/grid.py`](src/disaster_intel/grid.py))
* **Resolution 7 Standard**: Each cell has an area of **~5.16 km²** (diameter ~2.6 km). This resolution matches the physical scale of localized flash floods and landslide hillslopes while remaining computationally tractable.
* **`self.cells`**: Precomputes and sorts a deterministic tuple of **5,930 H3 cells** completely covering the study region.
* **`latlng_to_cell(lat, lng)`**: Maps any GPS point to its enclosing H3 cell in $O(1)$ time.
* **`cell_to_latlng(cell)`**: Returns the geographic centroid `(lat, lon)` of an H3 hexagon.
* **`neighbors(cell)`**: Returns the 6 adjacent cells (1-ring neighbor set) used by spatial convolution and graph neural networks.
* **Disk Caching**: Cell computation is serialized to `.cache/coastal_grid_res7.json` on disk, eliminating re-computation overhead on future startups.

---

## 4. Increment 2: Multimodal Telemetry Data Model (`models.py`)

### Purpose
Provides strong typing, unit normalization, coordinate validation, and deduplication keys before any data enters storage or pipelines.

### Key Classes ([`src/disaster_intel/models.py`](src/disaster_intel/models.py))

#### 1. Geometry Types
* **`PointGeometry(lat, lon)`**: Represents a localized observation (e.g. USGS earthquake epicenter, rain gauge). Rejects out-of-range coordinates (`[-90, 90]`, `[-180, 180]`).
* **`GridNodeGeometry(lat, lon, resolution_deg)`**: Represents regular raster weather nodes (e.g. Open-Meteo 0.1° grid).
* **`PolygonGeometry(vertices)`**: Represents regional warning areas (e.g. NDMA SACHET alert boundaries). Enforces $\ge 3$ vertices.

#### 2. Telemetry Records
* **`RawReading`**:
  * An unprocessed record straight from an external source adapter (before spatial discretization).
  * Enforces **strict UTC normalization** (rejects naive datetimes; converts non-UTC timezones to UTC).
  * Rejects `NaN`, `+Inf`, and `-Inf` values at instantiation.
* **`Observation`**:
  * The canonical, per-cell, post-discretization record.
  * Carries an **H3 cell ID**, canonical physical unit, UTC timestamp, and source attribution.
  * **Deterministic Idempotency Key**: Generated via SHA-256 hash of `"{cell}|{variable}|{source}|{timestamp_iso}"`. If the same observation is sent twice (e.g. network retry after crash), the cold store recognizes the identical key and deduplicates it.

#### 3. `VariableCatalog` & Unit Conversion
* **`VariableCatalog`**: Registry defining every supported telemetry stream with its canonical storage unit and aggregation method:
  * `precipitation`: `mm`, Aggregation = `SUM`
  * `soil_moisture_surface` (0-7 cm): `m3/m3`, Aggregation = `MEAN`
  * `soil_moisture_rootzone` (7-28 cm): `m3/m3`, Aggregation = `MEAN`
  * `temperature`: `celsius`, Aggregation = `MEAN`
  * `wind_speed`: `km/h`, Aggregation = `MAX`
  * `pressure`: `hPa`, Aggregation = `MEAN`
  * `discharge`: `m3/s`, Aggregation = `MAX`
  * `seismic_magnitude`: `magnitude`, Aggregation = `MAX`
  * `cap_severity`: `index`, Aggregation = `MAX`
* **`convert_unit(value, from_unit, variable, interval_hours)`**:
  * Automatically converts alternative incoming units into canonical units:
    * Kelvin $\rightarrow$ Celsius ($K - 273.15$)
    * Fahrenheit $\rightarrow$ Celsius ($(F - 32) \times \frac{5}{9}$)
    * Precipitation rate $\text{mm/hr} \rightarrow \text{mm}$ (scaled by interval)
    * Soil moisture percentage $\% \rightarrow \text{m}^3/\text{m}^3$ (divided by 100)

---

## 5. Increment 3: Telemetry Adapters & Deterministic Clock (`clock.py`, `adapters/`)

### Purpose
Isolate all external API interactions behind a clean interface, implement rate-limiting protection, request batching, failure attribution, and ensure unit tests run without real time delays or network requests.

### Key Components

#### 1. Injected Clock Abstraction ([`src/disaster_intel/clock.py`](src/disaster_intel/clock.py))
* **`Clock` (Protocol)**: Defines `now() -> datetime` and `sleep(seconds: float) -> None`.
* **`SystemClock`**: Production clock utilizing real wall time (`datetime.now(UTC)` and `time.sleep`).
* **`FakeClock`**: Deterministic test clock. Calling `fake_clock.sleep(10.0)` does not pause execution for 10 seconds; instead, it advances internal simulated time instantly and records the sleep duration in `sleep_log`. This allows testing 10-second exponential backoffs in microseconds!

#### 2. Base Adapter & Metrics ([`src/disaster_intel/adapters/base.py`](src/disaster_intel/adapters/base.py))
* **`AdapterMetrics`**: Tracks performance and attributes errors:
  * `requests_made`, `readings_produced`, `retries`.
  * `records_skipped`: Granular map of reasons (e.g. `{"null_hour": 6}`).
  * **Failure Attribution**: Distinguishes upstream network/source outages (`SourceError`, e.g. 503 or 429) from data corruption (`ParseError`, e.g. malformed JSON).
* **`HttpTransport` & `MockHttpTransport`**: Transport protocol allowing tests to inject mock HTTP responses without network sockets.

#### 3. Open-Meteo Weather Adapter ([`src/disaster_intel/adapters/open_meteo.py`](src/disaster_intel/adapters/open_meteo.py))
* **Multi-Location Request Batching**: Free weather APIs have rate limits and call budgets. The adapter batches multiple coordinates into single requests (e.g. 50 locations per call).
* **Exponential Backoff**: If an HTTP `429` (Too Many Requests) or `5xx` (Server Error) occurs, the adapter retries with exponential backoff ($t_{\text{sleep}} = t_{\text{initial}} \times 2^{\text{attempt}}$) via the injected clock.
* **Null-Hour Skipping**: If a specific timestamp has missing data in the weather model (`null`), it is skipped safely and incremented in `metrics.records_skipped["null_hour"]` rather than crashing the pipeline.
* **Live Test Verified**: Tested and verified live against Open-Meteo's API for Mangalore (retrieving 144 hourly readings).

---

## 6. Increment 4: Spatial Discretizers & Fan-Out Table (`discretizers.py`)

### Purpose
Solve the mapping between incoming continuous spatial data and discrete H3 resolution-7 cells. Specifically optimizes raster weather ingestion so we **do not** call the weather API 5,930 times for 5,930 cells.

### Key Classes ([`src/disaster_intel/discretizers.py`](src/disaster_intel/discretizers.py))

#### 1. `FanOutTable` (95% Call Reduction)
* **The Problem**: Open-Meteo data is produced on a 0.1° grid (~11 km spacing). Our study region contains 5,930 H3 res-7 cells (~2.6 km diameter). Querying Open-Meteo once for every individual cell would waste thousands of redundant API requests.
* **The Solution**: 
  * Every H3 cell's centroid is deterministically rounded to its nearest 0.1° grid node.
  * The **5,930 cells** collapse into exactly **294 unique grid nodes** (fan-out factor of ~20.2 cells per node).
  * We fetch each grid node **once**, and the fan-out table replicates the value out to all ~20 overlapping cells.
  * The table is precomputed and saved to `.cache/fan_out_res7_0.1deg.json`.

#### 2. `PointDiscretizer`
* Maps single point readings (e.g. earthquake magnitudes, river discharge gauges) to the enclosing cell via `grid.latlng_to_cell(lat, lon)` and creates an `Observation`.

#### 3. `PolygonDiscretizer` & `point_in_polygon`
* Implements the ray-casting algorithm to test polygon containment.
* Uses `h3.polygon_to_cells` to identify all cells covered by regional warning polygons (NDMA SACHET CAP flood/cyclone alerts), creating an `Observation` for every enclosed cell.

#### 4. `SpatialDiscretizer`
* Unified router: inspects any incoming `RawReading.geometry.kind`:
  * `"point"` $\rightarrow$ dispatches to `PointDiscretizer`
  * `"grid_node"` $\rightarrow$ dispatches to `GridNodeDiscretizer` (fanning out via `FanOutTable`)
  * `"polygon"` $\rightarrow$ dispatches to `PolygonDiscretizer`

---

## 7. Increment 5: Streaming Bus Interface & In-Memory Bus (`bus/`)

### Purpose
Provides asynchronous decoupling between telemetry ingestion and downstream storage engines (ClickHouse, Redis). A unified streaming contract ensures messages are ingested reliably, ordered per cell, replayable on crash, and backpressured when buffers fill.

### Key Classes ([`src/disaster_intel/bus/`](src/disaster_intel/bus/))

#### 1. `Bus` Contract ([`src/disaster_intel/bus/base.py`](src/disaster_intel/bus/base.py))
* **`Message`**: Immutable, slotted data structure containing `topic`, `key` (H3 cell ID), `payload`, integer `offset`, UTC `timestamp`, and `headers`.
* **`publish(topic, key, payload)`**: Appends a message and returns its strictly monotonic integer offset.
* **`poll(topic, group_id, max_messages, auto_commit)`**: Consumes messages starting from the consumer group's uncommitted read position.
* **`commit(topic, group_id, offset)`**: Advances the persistent consumer offset for that specific group.
* **`seek(topic, group_id, offset)`**: Allows moving the consumer group cursor backwards to any past offset for deterministic historical replaying.

#### 2. `InMemoryBus` ([`src/disaster_intel/bus/memory.py`](src/disaster_intel/bus/memory.py))
* **Thread-Safe**: Uses re-entrant locks (`threading.RLock`) to support multi-threaded producer and consumer pipelines.
* **Per-Key Order Preservation**: Sequential observations for the same H3 cell are guaranteed to arrive in the exact order they were published.
* **Independent Consumer Groups**: Different downstream workers (e.g., `group_id="clickhouse_cold_sink"` vs `group_id="redis_hot_sink"`) read from the same topic at their own independent speeds without interfering with each other's offsets.
* **Backpressure (`BufferFullError`)**: Bounded queue protection (`max_queue_size`) ensures high-velocity telemetry floods do not exhaust system memory.
* **Reusable Contract Test Suite (`tests/test_bus.py`)**: Defines `BaseBusContractTests` which tests `InMemoryBus` today and will verify `RedpandaBus` (Kafka) in container tests with identical behavior.

---

---

## 8. Increment 6: Ingestion Pipeline & Objective 1 Metrics (`pipeline.py`)

### Purpose
Connects all components developed in Increments 2–5 into an end-to-end telemetry processing engine and fulfills **Objective 1** of the project:
* Ingest multimodal environmental data with rigorous tracking of p50, p95, and p99 ingestion latency.
* Guarantee high completion rate ($\ge 98\%$) with precise failure attribution (distinguishing remote source errors from local payload parse failures).
* Enforce idempotent deduplication at the storage boundary.
* Provide an offline fixture replay CLI (`ingest --replay`) and a live API ingestion pipeline (`ingest --live`).

### End-to-End Data Flow

```text
[Raw Telemetry Payload] 
        │
        ▼
[SourceAdapter] ──(parses & validates)──► [RawReading Stream]
                                                 │
                                                 ▼
                                      [SpatialDiscretizer]
                                                 │ (maps point/polygon/grid to H3 res-7)
                                                 ▼
                                        [Observation Stream]
                                                 │
                                                 ▼
                                        [Streaming Bus]
                                     (telemetry.{source} topic)
                                                 │
                                                 ▼
                                         [Pipeline Drain]
                                                 │ (tracks arrival time & latency)
                                                 ▼
                                        [Observation Sink]
                                      (enforces idempotency)
```

### Key Classes & Functions ([`src/disaster_intel/pipeline.py`](src/disaster_intel/pipeline.py))

#### 1. `calculate_percentiles(values, percentiles)`
* Calculates exact mathematical percentiles ($p_{50}$, $p_{95}$, $p_{99}$) using linear interpolation.
* Used to accurately measure latency distribution across thousands of processed observations.

#### 2. `ObservationSink` Protocol & `InMemorySink`
* **`ObservationSink`**: Storage abstraction adhered to by both in-memory testing sinks and production stores (ClickHouse cold store, Redis hot store).
* **`InMemorySink`**:
  * Stores observations keyed by `idempotency_key` (`sha256(source + variable + cell + timestamp)`).
  * Returns `(stored_count, duplicates_count)`.
  * Indexes observations by H3 cell for fast spatial querying.

#### 3. `IngestionMetrics`
* Real-time metrics collector tracking:
  * `raw_readings_received`: Total incoming raw readings.
  * `observations_produced`: Discretized cell observations generated.
  * `observations_stored`: Unique observations successfully committed.
  * `duplicates_dropped`: Duplicate records dropped via idempotency keys.
  * `completion_rate`: Percent of attempted records successfully parsed and stored.
  * `source_errors` vs `parse_errors`: Clear attribution of remote upstream failures vs local data format errors.
  * `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`: Real latency percentiles in milliseconds.
  * `summary()`: Renders a clean ASCII summary table compatible with all operating systems and console encodings.

#### 4. `IngestionPipeline`
* Coordinates the complete flow:
  1. Accepts batches of `RawReading` items or fetches them from an adapter.
  2. Discretizes readings across the coastal H3 grid.
  3. Publishes observations to per-source bus topics (`telemetry.<source>`).
  4. Consumes published messages, computes end-to-end latency relative to clock time, and persists unique observations into the sink.

#### 5. CLI (`ingest`)
* Configured in `pyproject.toml` under `[project.scripts]`.
* Replay mode: `uv run ingest --replay tests/fixtures/open_meteo_sample.json` (runs 100% offline).
* Live mode: `uv run ingest --live --max-nodes 10` (fetches real weather data across study region grid nodes).

---

## 8b. Increment 7: ClickHouse Cold Store (`storage/`)

### Purpose
First half of Objective 2: the permanent, compressed, deduplicated record of every `Observation`, from which any past state can later be reproduced (Increments 8-10).

### Key Components ([`src/disaster_intel/storage/`](src/disaster_intel/storage/))
* **`migrations.py`**: versioned, idempotent DDL. Every statement is `IF NOT EXISTS` and applied versions are recorded in `schema_migrations`, so `migrate()` a second time returns `[]`.
* **`observations` table**: `ReplacingMergeTree`, `PARTITION BY toYYYYMM(ts)`, `ORDER BY (cell, variable, source, ts)`.
  * `cell UInt64` (H3 index as integer), `variable`/`source` `LowCardinality(String)`, `ts DateTime64(6,'UTC')`, `value Float32`, with Delta / DoubleDelta / Gorilla + ZSTD codecs.
  * **No `unit` column** (a function of `variable` via `VariableCatalog`) and **no `idempotency_key` column** (a hash of the other columns; recomputed on read). Both save bytes.
* **`ColdStore`** ([`cold.py`](src/disaster_intel/storage/cold.py)): `migrate()`, `write()`, `read_cell()`, `read_all()`, `count()`, `parent_cells()`, `count_by_parent()`, `storage_stats()`, `cold_correctness()`.
* **Reads always deduplicate (`FINAL`)**: `ReplacingMergeTree` only removes duplicates when parts merge. `raw_count()` is the one deliberate non-deduplicated query; a test pins the pre-merge behaviour with `SYSTEM STOP MERGES`.
* **Float32**: values read back equal `to_float32(original)`; the cold-correctness metric compares after that quantisation.

### Bugs found and fixed on the way (regression tests added)
* Catalog said `wind_speed` is `m/s` while the Open-Meteo adapter emits `km/h` -> catalog now `km/h`.
* `convert_unit` lower-cased pass-through units, so pressure observations carried `hpa` instead of `hPa` -> canonical spelling is kept.

### Tests
* `tests/test_cold_store_unit.py` (unit, fake client): row mapping, validation-before-insert, batching, idempotent migrations, schema/codecs, every read uses `FINAL`.
* `tests/test_cold_store_integration.py` (real ClickHouse): cold-correctness = 100%, duplicates collapse, ClickHouse `h3ToParent` equals Python `h3` on 1,000 random cells (res 4/5/6), migrations idempotent.
* `tests/test_cold_store_perf.py`: one synthetic day for all 5,930 cells x 6 hourly variables; budget measured from `system.parts` (first measurement 3.31 B/row compressed).

---

## 8c. Increment 8: Hourly Buckets & Rolling Windows (`windows.py`, `storage/`)

### Purpose
Second step of Objective 2: turn the raw cold-store observations into rolling-window features (1h, 6h, 24h, 72h, 7d) that are correct, reproducible for any past time, and cheap to read.

### Key Components
* **[`windows.py`](src/disaster_intel/windows.py)** (pure Python, no database): `bucket_end`, `floor_hour`, `Bucket`, `bucketize`, `combine_buckets`, `windows_from_buckets`, and the slow oracle `reference_windows`. The Redis hot tier (Increment 9) reuses `Bucket` / `combine_buckets`.
* **Migration 2**: `hourly_buckets` (`ReplacingMergeTree(version)`), one row per (cell, variable, source, hour-ending bucket) with `n_obs`, `sum_value`, `min_value`, `max_value`.
* **`ColdStore`**: `write()` recomputes every bucket it touched from `observations FINAL`; `read_buckets()`, `windows(cell, as_of)`, `rebuild_buckets()`; `maintain_buckets=False` for bulk backfill.

### Conventions (pinned by tests)
* Bucket `B` holds observations with `B - 1h < ts <= B` (hour-ending). A window is the last N buckets ending at `as_of` floored to the hour, so a window computed today for time T equals the one computed live at T.
* SUM for rainfall, AVG (sum / count) for soil moisture/temperature/pressure, MAX for wind/discharge/seismic/CAP.
* `WindowValue(value, observed_hours, hours)`; `coverage = observed_hours / hours`; no data means `value=None`, never 0.

### Deviation from plan.md (and why)
plan.md suggests an incremental materialized view. A plain MV adds every *insert*, so an at-least-once duplicate or a revised value would be counted twice (ReplacingMergeTree only dedups on merge). Buckets are instead recomputed from the deduplicated observations for exactly the (cells, variables, hours) a write touched, which also handles late arrivals. The "cheap combine of <= 168 buckets at read time" design is unchanged.

### Tests
* `tests/test_windows_unit.py`: conventions, coverage, hypothesis differential test (buckets vs oracle), refresh SQL, FINAL on reads, migration 2.
* `tests/test_windows_integration.py`: ClickHouse windows == oracle for every variable/window, late observation, duplicates/revisions, rollback equality, backfill mode.
* `tests/test_buckets_perf.py`: bucket-table bytes/row and `windows()` read latency.

---

## 8d. Increment 9: Redis Hot Tier & Rebuild Job (`storage/hot.py`, `rebuild.py`, `tee.py`)

### Purpose
Third step of Objective 2: the *current* state of every cell, readable in about a millisecond, and provably rebuildable from the cold tier.

### Key Components
* **[`hot.py`](src/disaster_intel/storage/hot.py)** - `HotStore`: one Redis hash per H3 cell (`ws:v1:<cell>`), one field per (variable, source) holding a zlib-compressed, time-sorted, delta-coded block of float32 observations. `write()`, `windows()` (same signature/shape as `ColdStore.windows`), `evict_expired()`, `clear()`, `dump_cell()`, `memory_usage()`; plus `summarize_latencies()` (average, p50, p95, p99).
* **[`rebuild.py`](src/disaster_intel/storage/rebuild.py)** - `rebuild_hot_from_cold()` and a CLI (`make rebuild-hot`).
* **[`tee.py`](src/disaster_intel/storage/tee.py)** - `TeeSink`: writes the cold tier first (the record), then the hot tier.

### Deviation from plan.md (and why)
plan.md sketches a "bucket ring, ~6.7 KB/cell" (one float per hourly bucket). One float cannot answer SUM/AVG/MAX/MIN, and an aggregate cannot detect a duplicate or undo a revised value, so "applying one observation twice changes nothing" could not hold. The hot tier therefore stores the **observations** (compressed) and computes window aggregates at read time with the same conventions as the cold tier. Measured: **~3.9 KB/cell (~23 MB for 5,930 cells)**, hot read p95 ~1.3 ms, write p95 ~2 ms (budget: 8 KB, 5 ms).

### Conventions (pinned by tests)
* An observation is identified by (variable, source, ts); re-applying overwrites the same slot (idempotent); a revised value replaces the old one (like `ReplacingMergeTree`).
* Kept while `bucket_end(ts) > floor_hour(now) - 168 h`. Eviction is lazy on write, plus `evict_expired` and a key TTL.
* `windows()` raises `HotRangeError` rather than returning a truncated window; the past is the cold tier's job.
* Redis runs with `maxmemory-policy noeviction` (a test enforces it): silent LRU eviction would make window reads wrong. docker-compose changed accordingly.
* Single writer per cell (the bus already orders by cell key); readers are always safe.

### Tests
* `tests/test_hot_unit.py` (unit, `tests/fake_redis.py`): encoding, hypothesis differential test vs `reference_windows`, eviction, idempotency, rebuild, round-trip count, tee sink, latency report.
* `tests/test_hot_integration.py` (real Redis; the rebuild test also needs ClickHouse): oracle agreement, physical eviction + TTL, byte-identical re-apply, wipe-and-rebuild equals live state and equals ClickHouse windows, eviction-policy guard.
* `tests/test_hot_perf.py`: memory per cell and hot read/write latency percentiles.

---

## 9. Current System Status & Test Suite Summary

All quality gates pass with zero warnings or errors:

* **Unit Test Suite**: **189 passed** in **~6 seconds** (`pytest -m unit`).
* **Linter**: `ruff check src tests` (0 errors, 100% clean).
* **Formatter**: `ruff format --check src tests` (100% compliant across 22 source files).
* **Type Checker**: `mypy src tests` (Strict typing verified across 22 source files).
* **Live Integration**: `@pytest.mark.live` verified against Open-Meteo live API.
* **CLI Verified**: `ingest --replay` verified on recorded telemetry fixtures.

---

## 10. Upcoming Increments Roadmap (Increments 7 to 14)

### Increment 7: ClickHouse Cold Store (Objective 2) - DONE (see section 8b)

### Increment 8: Hourly Buckets & Rolling Windows - DONE (see section 8c)
* Incremental materialized views maintaining hourly aggregates (`SUM` for rain, `AVG` for soil moisture, `MAX` for wind/discharge).
* Rolling windows: 1h, 6h, 24h, 72h, 7d.

### Increment 9: Redis Hot Tier & Rebuild Job - DONE (see section 8d)

### Increment 10: State Vector Assembler & Static Layers - DONE (see plan.md 'As built (Increment 10)')

### Increments 11–14: Multimodal Fusion, Foundation Model & Warning Engine
* Cross-attention spatio-temporal fusion, NASA Prithvi geospatial backbone, and early warning lead-time benchmarking.

