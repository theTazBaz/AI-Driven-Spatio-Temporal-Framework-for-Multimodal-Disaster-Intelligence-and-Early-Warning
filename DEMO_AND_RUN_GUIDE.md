# Phase I: Run Guide (WSL) and Demo Playbook

Covers: (A) prerequisites on WSL + Docker, (B) fresh `git clone` to running system, (C) what to demo and in what order, (D) metrics: what each means, where it comes from, how far to trust it, (E) things you must NOT show, (F) viva Q&A.

> **Honesty note.** I read your repo and report and checked the new script for syntax and against your APIs, but I could not execute anything against Docker or Open-Meteo from my sandbox. Do a full dry run (Section B, then C) at least a day before. Expected failure points are listed in B.9.

---

## A. Prerequisites on Windows + WSL

### A.1 WSL2 with Ubuntu (PowerShell as Administrator)

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
wsl --set-default-version 2
```

Reboot if asked, then open "Ubuntu 24.04" and create a user. Check from PowerShell: `wsl -l -v` must show `VERSION 2`.

### A.2 Give WSL enough memory

Your compose caps are ClickHouse 1 GB + Redpanda 512 MB + Redis 160 MB, so give WSL at least 4 GB. Create `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=6GB
processors=4
swap=2GB
```

Then in PowerShell: `wsl --shutdown`, and reopen Ubuntu.

### A.3 Docker: choose ONE option

**Option 1 (easiest): Docker Desktop for Windows**
1. Install Docker Desktop, enable "Use WSL 2 based engine".
2. Settings → Resources → WSL Integration → switch on **Ubuntu-24.04** → Apply & restart.
3. Inside Ubuntu: `docker version` and `docker compose version` must both print client and server info.

**Option 2: Docker Engine installed natively inside WSL (no Docker Desktop)**

```bash
# enable systemd in WSL
echo -e "[boot]\nsystemd=true" | sudo tee /etc/wsl.conf
# (PowerShell)  wsl --shutdown   then reopen Ubuntu

# install Docker Engine
sudo apt-get update && sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
 | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker
sudo systemctl enable --now docker
docker run --rm hello-world
```

### A.4 Tools inside Ubuntu

```bash
sudo apt-get update && sudo apt-get install -y git make curl build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc          # or open a new terminal
uv --version
```

Ubuntu 24.04 ships Python 3.12 (project needs ≥ 3.11). If `python3 --version` is older: `uv python install 3.12`.

> **Work inside the Linux filesystem (`~/...`), never under `/mnt/c/...`.** Docker volumes and Python installs are many times slower across the Windows mount, and that would distort your latency numbers.

---

## B. Fresh clone → running system

### B.1 Clone

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/tejas-bajaj/AI-Driven-Spatio-Temporal-Framework-for-Multimodal-Disaster-Intelligence-and-Early-Warning.git
cd AI-Driven-Spatio-Temporal-Framework-for-Multimodal-Disaster-Intelligence-and-Early-Warning
```

### B.2 Install dependencies (note the `--extra dev`)

```bash
uv sync --extra dev
```

Your README says plain `uv sync`, but `pytest`, `ruff` and `mypy` live in the `dev` optional-dependency group, which plain `uv sync` does not install. Fix the README too.

### B.3 Put the new demo script in place

Copy `showcase_demo.py` (delivered with this guide) to `scripts/showcase_demo.py`, then commit it.

### B.4 Offline sanity check (no Docker)

```bash
uv run pytest -m unit          # expect 371 passed in ~8 s
uv run ruff check src tests
uv run mypy src
uv run ingest --replay tests/fixtures/open_meteo_sample.json
```

Read the replay output with care: completion rate, duplicates and error counters are meaningful, but **ignore its latency lines** (see Section E).

### B.5 Start the infrastructure

```bash
docker compose up -d --wait      # first run pulls 3 images, a few minutes
docker compose ps                # all three must say "healthy"
```

Quick probes:

```bash
curl -s localhost:8123/ping                         # ClickHouse -> "Ok."
docker compose exec redis redis-cli ping            # PONG
docker compose exec redis redis-cli config get maxmemory-policy   # noeviction
docker compose exec redpanda rpk cluster health     # Healthy: true
```

Browser UIs you can show: ClickHouse SQL playground at `http://localhost:8123/play` (user `disaster`, password `disaster`).

### B.6 Containerised tests

```bash
uv run pytest -m integration                # real ClickHouse + Redis + Redpanda
uv run pytest -m perf -s                    # prints bytes/row, Redis bytes/cell, latencies
```

Save the `-s` output to a file now: it is your evidence for the cold-tier and hot-tier numbers:

```bash
uv run pytest -m perf -s 2>&1 | tee demo_out_perf.txt
```

### B.7 Existing demos

```bash
uv run python scripts/windows_demo.py   # ClickHouse windows, gap, late arrival, rollback
uv run python scripts/hot_demo.py       # Redis hot tier, latency, wipe + rebuild
uv run python scripts/state_demo.py     # full state vector, hot==cold, rollback
```

### B.8 The showcase (real data, map)

```bash
uv run python scripts/showcase_demo.py --keep      # needs internet; downloads once
# later runs, even offline:
uv run python scripts/showcase_demo.py --cache --keep
```

It writes `demo_out/showcase_map.html` (open from WSL with `explorer.exe "$(wslpath -w demo_out/showcase_map.html)"`) and `demo_out/showcase_metrics.json`. `--keep` leaves the ClickHouse database and Redis keys alive so you can query them live; the script prints their names. When finished: `docker compose down -v`.

### B.9 If something breaks

| Symptom | Fix |
|---|---|
| `docker: command not found` in WSL | Docker Desktop → WSL Integration not enabled for your distro (A.3) |
| `permission denied` on docker socket | `sudo usermod -aG docker $USER && newgrp docker` |
| ClickHouse unhealthy / exits | WSL memory too low (A.2); `docker compose logs clickhouse` |
| Port 8123/6379/9092 already in use | stop the other service, or `docker compose down` and retry |
| `Authentication failed` for ClickHouse | credentials default to `disaster`/`disaster`; if you changed env vars, export `CLICKHOUSE_USER` / `CLICKHOUSE_PASSWORD` |
| Showcase: `0 observations` / KeyError | Open-Meteo returned coordinates that map to no fan-out node: lower `--ring`, or check the printed node list |
| Showcase: HTTP 429 | free-tier rate limit; wait a minute (the adapter already backs off), then rerun; use `--cache` afterwards |
| Map is grey/blank | needs internet for Leaflet and tiles (unpkg + OpenStreetMap); open it once while online beforehand |
| `focal cell ... outside the study grid` | the lat/lon you gave is outside the polygon; use the defaults |

---

## C. The demo (about 15 minutes)

**The thesis to repeat three times:** *Phase I builds the substrate that makes compound-risk modelling possible: every observation from every source lands in one H3 cell, in one time convention, with one auditable state vector, fast enough for hourly scoring on a laptop, and reproducible for any past instant.* The point of Phase I is not "we predict floods"; it is "we can now ask a model a clean, leak-free, comparable question about any cell at any hour."

Run these on one screen: terminal left, browser right. Open `demo_out/showcase_map.html` before you start.

### Act 0: The problem (1 min, slides only)

Show the cascade: several days of rain → saturated soil → discharge surge → landslide. Then: IMD, CWC, GSI and NDMA each publish separately, so no one system holds the cross-variable state in which the cascade is visible. "Our Phase I answer is the shared state."

### Act 1: One grid for everything (2 min): showcase step 1

What appears: 5,930 cells, 5.16 km² each, 294 weather nodes, "20.2 cells per API call, 95% fewer calls".

Say:
- **Why H3 res 7?** Fine enough to separate valley floors from steep slopes (res 6 ≈ 36 km² would merge them); coarser than the 0.1° weather inputs, so res 8 would only create cells with no independent data.
- **Why hexagons?** Six equidistant neighbours: no diagonal-vs-edge bias when modelling spread, and clean `grid_disk` neighbourhoods, which Objective 3's graph variant will use.
- **Why fan-out matters:** a 0.1° weather node covers ~20 cells, so we fetch each node once instead of 5,930 times. This is the difference between fitting inside a free API's rate limit and not.

### Act 2: Real telemetry, typed and attributed (2 min): showcase step 2

What appears: N typed `RawReading`s from real ERA5-Land data (the same land-surface reanalysis that Open-Meteo serves), for 8–20 July 2024 around Shirur, with retry/skip counters.

Say:
- Every reading is validated at construction: UTC only, no NaN/inf, canonical units. Garbage is rejected at the boundary, not discovered inside a model.
- Failures are *attributed*: `SourceError` (upstream down) vs `ParseError` (bad payload) are counted separately, so an operator can tell a dead API from corrupt data. (Be ready to say the 0/0 on clean data only shows clean data is clean; the mechanism itself is proven by fault-injection unit tests: show one: `uv run pytest tests/test_adapters.py -v`.)
- This is **real data, not synthetic**, which is the main upgrade over the report's evaluation. Say plainly it is reanalysis, not a live feed.

### Act 3: Streaming into two stores (3 min): showcase steps 3 and 4

What appears: hours streamed, observations produced vs readings in (fan-out multiplier ≈ 20×), stored count, sustained obs/s, per-hour batch latency p50/p95/p99. Then the replay: re-deliver the last 24 h → "stored 0, duplicates dropped N".

Say:
- Path: adapter → discretizer → bus (keyed by H3 cell so per-cell order is preserved) → `TeeSink` → ClickHouse first (system of record), then Redis (hot).
- **This throughput number is the strong one** because it includes real ClickHouse and Redis writes, unlike the in-memory 1.6×10⁵ obs/s in the report. Quote it as "measured with real containers on localhost, single core"; the report's in-memory figure then becomes "the processing ceiling without I/O".
- **Idempotency:** each observation's key is `SHA256(cell | variable | source | event time)`, computed from event time, not arrival time. A retried upload, a replayed backfill or a duplicated bus message maps to the same key and is stored once. Combined with at-least-once delivery that gives effectively-once storage. The replay line is the proof.
- Why two stores: ClickHouse is cheap, columnar and immutable (history, training sets, audits); Redis answers "what is the state of this cell right now" in under a millisecond. This is the speed-layer / batch-layer pattern.

### Act 4: The two tiers agree (2 min): showcase step 5

What appears: `hot == cold across 29 features -> True`, hot vs cold `get_state` latency, bytes per cell and projected region memory.

Say:
- Two stores are only safe if they compute *identical* windows. Both use the same rule (hour-ending buckets, SUM for rain, AVG for soil, MAX/MIN for triggers), and both are tested against an independent naive oracle with property-based tests (Hypothesis randomises missing data, duplicates, revisions, out-of-order arrival).
- Redis runs `noeviction`: a state store that silently evicts cells would return wrong windows, so it refuses writes instead. Show: `docker compose exec redis redis-cli config get maxmemory-policy`.
- Memory: ~3.9 KB/cell → ~23 MB for the whole region. "The state of Karnataka's coast fits in a phone's RAM."

### Act 5: Time travel and failure recovery (3 min): showcase steps 6 and 7

What appears: "k/k past instants reproduce the live vector (tiers used: hot and cold)". Then Redis wiped and rebuilt from ClickHouse, with a byte-identical check.

Say:
- **Rollback is the core scientific property.** A model trained on the state at time T must see only what was known at T. Because windows are keyed on event time with fixed bucket boundaries, the state for T recomputed days later (served from ClickHouse once T is beyond Redis's 168 h horizon) equals what the live system saw. That is what rules out leakage in Objective 3's training set. Tolerance 1e-4 is only float32 quantisation in the cold tier.
- **Recovery:** Redis is disposable by design; flush it and `rebuild_hot_from_cold` restores identical state. That is why ClickHouse is written first.

### Act 6: The map (3 min): browser

Open `showcase_map.html`.
1. Drag the slider from 11 July onward with metric **precipitation 24 h**: rain arrives as a spatial pattern, not a point reading. Cells sharing a weather node move together; that is the fan-out made visible, and an honest limit of 0.1° inputs.
2. Switch to **root-zone soil moisture 24 h**: it rises slowly and keeps rising after rain stops. That memory is why we need multi-horizon windows (1 h to 7 d): antecedent wetness decides whether the next burst runs off.
3. Switch to **compound_saturation** (= root-zone soil moisture 24 h × rainfall 72 h): it keeps climbing after the 1-hour intensity peaks, because the 72 h accumulation lags. Say: *this is exactly why the compound term has to be read with its trend, which Objective 4 adds.*
4. Point at the white-outlined focal cell and the chart below: live state hour by hour, 1 h vs 24 h rainfall.
5. Cells that stay grey/`no data` are shown as missing, never as zero: "zero rain is a measurement, a dead gauge is not."

Do **not** say the map "detects the landslide". It shows the antecedent conditions a model will learn from. Without labels there is no detection claim.

### Act 7: Live queries against the stores (1 min, optional wow)

With `--keep`, use the database name the script printed (`showcase_xxxxxxxx`):

```bash
docker compose exec clickhouse clickhouse-client --user disaster --password disaster -q \
 "SELECT variable, count() n, round(min(value),2) lo, round(max(value),2) hi
  FROM showcase_XXXXXXXX.observations FINAL GROUP BY variable ORDER BY variable"

docker compose exec clickhouse clickhouse-client --user disaster --password disaster -q \
 "SELECT toUInt64(cell) c, round(sum(value),1) mm FROM showcase_XXXXXXXX.observations FINAL
  WHERE variable='precipitation' GROUP BY c ORDER BY mm DESC LIMIT 5"

docker compose exec redis redis-cli --scan --pattern 'showcase:*' | head -3
docker compose exec redis redis-cli memory usage "$(docker compose exec -T redis redis-cli --scan --pattern 'showcase:*' | head -1)"
```

Meaning: the archive is plain SQL and H3-native (ClickHouse has `h3ToParent` etc.), so anyone can audit a claim; Redis holds one compact hash per cell.

### Act 8: Close (1 min)

Slide: Phase I delivers objectives 1 and 2; the state vector (29/63/86-d, versioned and fingerprinted so a model can never silently read shifted columns) is the contract Objectives 3 and 4 consume. Then your validity list (E.2) and the plan.

---

## D. Metrics: what to show and what each one means

| # | Metric | Value (report) | How to produce live | What it means / why it matters | Caveat to state |
|---|---|---|---|---|---|
| 1 | Cells / area | 5,930; 5.16 km² | showcase step 1 | Spatial resolution of every later claim | |
| 2 | API-call reduction | 5,930 → 294 (−95.0%, 20.2×) | showcase step 1 | Feasible on free-tier rate limits | Fan-out gives all cells of a node the *same* weather values |
| 3 | Ingestion throughput | ~1.6×10⁵ obs/s in-memory; run showcase for the real-container figure | showcase step 3 | Headroom: region needs only tens of thousands obs/h | In-memory figure excludes network/disk; prefer the container figure |
| 4 | Per-batch / per-reading latency | 0.10 ms p50 in-process; showcase prints per-hour batch p50/p95/p99 | showcase step 3 | Processing is not the bottleneck | Not end-to-end; excludes Redpanda hop (say so) |
| 5 | Duplicate rejection | 100% (142,320/142,320) | showcase step 4 | Effectively-once storage | |
| 6 | Hot read latency | 0.73 ms full read; 0.49 ms/state | `pytest -m perf -s`, showcase step 5 | Whole region refresh ≈ 3 s | single Redis node |
| 7 | Hot memory | 3.9 KB/cell → ~23 MB | perf test, showcase step 5 | Region state fits in RAM | |
| 8 | Cold bytes/row | 3.31 B vs 5 B budget → ~1.7 GB/yr (10 vars, hourly) | `pytest -m perf -s` | Cheap long-term archive | Extrapolated from synthetic noise-added data |
| 9 | Hot == cold, oracle agreement | all variables/horizons | showcase step 5; `pytest -m unit tests/test_windows_unit.py` | Windows are trustworthy | Differential tests, not an external dataset |
| 10 | Rollback determinism | k/k identical (tol 1e-4) | showcase step 6 | No temporal leakage in training data | |
| 11 | Rebuild from cold | identical state | showcase step 7 | Redis is disposable | |
| 12 | Quality gates | 371 unit tests in ~8 s; ruff and strict mypy clean | `uv run pytest -m unit`, `ruff`, `mypy` | Engineering rigour | Tests prove correctness of mechanism, not hazard detection |
| 13 | Embedding budget | 64 B/cell/acquisition (768-d → 32 float16 = 48×) | `uv run pytest tests/test_embedder.py tests/test_clay_embedder.py -v` | Satellite features fit the memory budget | **Stand-in back-ends, not pretrained weights**: shows interface/budget only |

Lead with rows 3, 5, 9, 10: speed, correctness, reproducibility. Those are the claims that survive questioning.

**The metrics you cannot show yet** (say this proactively): hazard-detection accuracy, precision/recall/AUC, calibration, lead time, SHAP attribution quality, alert hallucination rate. They belong to Objectives 3 and 4, and the report already schedules them (Dec–May). Evaluators respect a clearly drawn boundary.

---

## E. Do NOT show, and fix before the evaluation

**E.1 Things that will embarrass you if shown unqualified**
1. `make evaluate` / `scripts/evaluate.py` and `PHASE1_EVALUATION_REPORT.md`, `PHASE1_MULTIMODAL_BENCHMARK.md`: your own Appendix I says latency (31/46/48 ms) is generated as `14 + hash mod 35`, Redis memory (6.70 KB) is a constant, hot latencies come from a fake Redis, and the focal cells resolve to Hyderabad and New York. Never put these numbers on a slide. Fix the harness or delete the claims before the committee opens the repo.
2. The latency block in `ingest --replay`: it compares fixture timestamps with a fake clock set to the last timestamp + 1 s, so it measures a clock offset, not processing time.
3. SAR/INSAT "case study" values (196 K, −20.5 dB): static template text. Table 4.3 (function-level threshold checks) is the honest framing.
4. Any claim about Clay vs Prithvi, or embedding "information content".
5. "Shirur landslide detected": the coordinates need re-verification and there are no labels.

**E.2 Validity gaps to state yourselves** (your report already does; say it first)
- ERA5-Land reanalysis at 0.1° replaces a live feed; point extremes are smoothed.
- Static terrain in the showcase is synthetic unless you ran `make static-layers` (real Copernicus DEM via Open-Meteo, 12 s between requests, slow) and the GSI layer is a coarse placeholder.
- Discharge, seismic, SACHET/GDACS, IMERG, SMAP, SWOT, NISAR adapters are pending, so those state features are NaN, shown honestly as missing.
- Single-node Redis and ClickHouse; no availability or scaling claim.

**E.3 Cheap fixes worth doing this week**
- README: `uv sync --extra dev`; "< 2 seconds" → "~8 seconds"; ~5,800 → 5,930 cells.
- Fix the focal-cell constants in `evaluation.py` (derive from `h3.latlng_to_cell`), or remove the module from the demo path.
- Commit `bench.py` (the report cites it as "supplied with this report" but it is not in the repo).
- Add `demo_out/` to `.gitignore`.

---

## F. Likely questions and strong answers

1. **"Is this real data?"** The showcase replays real ERA5-Land hourly reanalysis through the real pipeline. It is not a live feed, and the earlier benchmark windows were synthetic; we say so in Chapter 4.
2. **"What does this predict?"** Nothing yet, by design. Phase I produces the leak-free per-cell state a model needs; Objective 3 compares XGBoost/LightGBM, a CNN/LSTM and a small TFT on it, split by event and region.
3. **"Why not just use ERA5 files and pandas?"** Continuous streaming: late/duplicate/revised data, hot reads in under a millisecond, deterministic rollback and a rebuild path. A batch notebook gives none of these.
4. **"How do you prevent label leakage?"** Event-time bucketing with fixed boundaries; the state at T is a pure function of observations with event time ≤ T, verified by the rollback test; planned splits by event and region.
5. **"Why NaN, not zero?"** Zero rain is a measurement; a dead gauge is not. Each feature also carries a coverage fraction, so a model can weight partial windows.
6. **"Why a dual tier? Why not ClickHouse alone?"** ClickHouse is excellent for history but not for sub-millisecond per-cell reads; Redis alone would be volatile. Writes go cold first, so hot is always rebuildable.
7. **"What if a message arrives out of order?"** It lands in its correct historical bucket by event time; property tests randomise arrival order.
8. **"Why hexagons?"** Equidistant neighbours, hierarchical parents (res 6/7 chips for imagery), native H3 functions in ClickHouse.
9. **"How would this scale to all of India?"** Honest answer: not claimed. The design is partitionable by H3 parent cell (bus key is the cell; ClickHouse is ordered by cell), but we have not measured it.
10. **"What is novel?"** Not any single component. The combination: compound-hazard-ready, streaming, CPU-only, India-focused state on H3, with deterministic replay and an evaluation that audits its own numbers.

---

## G. Pre-demo checklist (night before)

- [ ] `docker compose down -v && docker compose up -d --wait` from scratch, all healthy
- [ ] `uv run pytest -m unit` → 371 passed; `-m integration` → pass
- [ ] `uv run pytest -m perf -s | tee demo_out_perf.txt` saved
- [ ] `uv run python scripts/showcase_demo.py` once online (creates `demo_out/open_meteo_payload.json`), then `--cache --keep` as the live run
- [ ] `showcase_map.html` opens, tiles load; screenshots saved as a backup
- [ ] Terminal font large, `clear` between acts, WSL window not on `/mnt/c`
- [ ] Backup: screen recording of the full run, in case the venue Wi-Fi or Docker fails
- [ ] Slides contain no number from Section E.1
