"""Increment 12 integration tests — real Redpanda container.

    docker compose up -d redpanda --wait
    uv run pytest -m integration tests/test_redpanda_integration.py

Test list (plan.md lines 312-319):
1. Base bus contract tests pass against real containerized Redpanda.
2. End-to-end fixture replay through real Redpanda into sink.
3. Retention policy applied to real Redpanda topics.
"""

from __future__ import annotations

import os
import socket
import uuid
from pathlib import Path

import pytest

from disaster_intel.bus import (
    Bus,
    RedpandaBus,
)
from disaster_intel.pipeline import InMemorySink, run_replay_fixture
from test_bus import BaseBusContractTests

pytestmark = pytest.mark.integration

REDPANDA_HOST = os.environ.get("REDPANDA_HOST", "localhost")
REDPANDA_PORT = int(os.environ.get("REDPANDA_PORT", "9092"))


def _is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


@pytest.fixture(autouse=True)
def _require_redpanda() -> None:
    if not _is_port_open(REDPANDA_HOST, REDPANDA_PORT):
        pytest.skip(
            f"Redpanda broker at {REDPANDA_HOST}:{REDPANDA_PORT} is not reachable. "
            "Run 'docker compose up -d redpanda' first."
        )


class TestRealRedpandaContract(BaseBusContractTests):
    """Verify that real containerized Redpanda satisfies the Bus contract."""

    def create_bus(self, max_queue_size: int | None = None) -> Bus:
        client_id = f"test_client_{uuid.uuid4().hex[:8]}"
        return RedpandaBus(
            bootstrap_servers=f"{REDPANDA_HOST}:{REDPANDA_PORT}",
            client_id=client_id,
            max_queue_size=max_queue_size,
            test_mode=False,
        )


def test_real_redpanda_pipeline_replay() -> None:
    client_id = f"pipeline_client_{uuid.uuid4().hex[:8]}"
    bus = RedpandaBus(
        bootstrap_servers=f"{REDPANDA_HOST}:{REDPANDA_PORT}",
        client_id=client_id,
        test_mode=False,
    )
    sink = InMemorySink()
    fixture = Path(__file__).parent / "fixtures" / "open_meteo_sample.json"

    metrics = run_replay_fixture(fixture_path=fixture, bus=bus, sink=sink)
    assert metrics.completion_rate == 100.0
    assert metrics.raw_readings_received == 24
    assert metrics.observations_stored == metrics.observations_produced
    assert sink.count() == metrics.observations_stored
