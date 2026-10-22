.PHONY: test test-unit test-contract test-integration test-perf test-all lint fmt typecheck up down clean demo-hot rebuild-hot demo-state static-layers static-layers-offline

# ── Testing ──────────────────────────────────────────────────────────
test: test-unit  ## Default: run unit tests only (< 10 s)

test-unit:
	uv run pytest -m unit -x

test-contract:
	uv run pytest -m contract -x

test-integration:
	uv run pytest -m integration -x

test-perf:
	uv run pytest -m perf -x

test-all:
	uv run pytest -x

# ── Code quality ─────────────────────────────────────────────────────
lint: typecheck  ## Lint and type-check
	uv run ruff check src tests

fmt:  ## Auto-format and fix
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy src

# ── Hot tier (Increment 9) ───────────────────────────────────────────
demo-hot:  ## Build one cell's state hour by hour in Redis; wipe + rebuild from ClickHouse
	uv run python scripts/hot_demo.py

rebuild-hot:  ## Repopulate Redis from ClickHouse (all cells; wipes the hot tier first)
	uv run python -m disaster_intel.storage.rebuild --clear

# ── State vector assembler (Increment 10) ──────────────────────────
demo-state:  ## One cell's state VECTOR building hour by hour; hot == cold; rollback
	uv run python scripts/state_demo.py

static-layers:  ## Build elevation/slope (Copernicus DEM via Open-Meteo); add POP=path for WorldPop
	uv run python -m disaster_intel.state.static_layers build $(if $(POP),--population $(POP),)

static-layers-offline:  ## Same, with SYNTHETIC terrain (no network; for demos only)
	uv run python -m disaster_intel.state.static_layers build --source synthetic

# ── Infrastructure ───────────────────────────────────────────────────
up:  ## Start ClickHouse, Redis, Redpanda
	docker compose up -d --wait

down:  ## Stop services (keep data)
	docker compose down

clean:  ## Stop services and delete volumes
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
