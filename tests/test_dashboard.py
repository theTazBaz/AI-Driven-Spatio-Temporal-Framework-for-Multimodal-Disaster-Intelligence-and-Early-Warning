"""Unit tests for Disaster Intelligence Tactical Command-Center API."""

from __future__ import annotations

import pytest
from starlette.requests import Request

from disaster_intel.dashboard.server import (
    get_cell_timeseries,
    get_ground_truth,
    get_replay,
    get_status,
    inspect_cell,
    list_events,
    load_cached_replays,
    serve_dashboard,
)


@pytest.fixture(autouse=True)
def ensure_replays_loaded() -> None:
    load_cached_replays()


@pytest.mark.unit
def test_dashboard_homepage() -> None:
    """Test that serve_dashboard returns the Tactical Command-Center HTML dashboard."""
    # Create mock request
    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}
    request = Request(scope)
    html = serve_dashboard(request)
    assert "DISASTER INTEL // TACTICAL RADAR" in html
    assert "NITK SURATHKAL" in html
    assert "HISTORICAL REPLAY" in html
    assert "86-DIMENSIONAL WORLDSTATE VECTOR" in html


@pytest.mark.unit
def test_api_status() -> None:
    """Test that get_status reports grid and storage tier telemetry."""
    data = get_status()
    assert data["status"] == "OPERATIONAL"
    assert data["grid"]["total_cells"] == 5930
    assert data["features"]["v3_dims"] == 86
    assert "shirur_2024" in data["loaded_replays"]


@pytest.mark.unit
def test_api_events() -> None:
    """Test that list_events lists the three verified historical disaster events."""
    events = list_events()
    assert len(events) >= 3
    event_ids = {e["event_id"] for e in events}
    assert "shirur_2024" in event_ids
    assert "kodagu_2018" in event_ids
    assert "uttara_kannada_2021" in event_ids


@pytest.mark.unit
def test_api_replay_shirur() -> None:
    """Test retrieving the 2024 Shirur replay bundle."""
    data = get_replay("shirur_2024")
    assert data["event_id"] == "shirur_2024"
    assert len(data["cells"]) > 0
    assert len(data["frames"]) > 0
    assert data["focal"] in data["cells"]


@pytest.mark.unit
def test_api_replay_kodagu() -> None:
    """Test retrieving the 2018 Kodagu cloudburst replay bundle."""
    data = get_replay("kodagu_2018")
    assert data["event_id"] == "kodagu_2018"
    assert len(data["cells"]) > 0
    assert len(data["frames"]) > 0


@pytest.mark.unit
def test_api_replay_uttara_kannada() -> None:
    """Test retrieving the 2021 Uttara Kannada flood replay bundle."""
    data = get_replay("uttara_kannada_2021")
    assert data["event_id"] == "uttara_kannada_2021"
    assert len(data["cells"]) > 0
    assert len(data["frames"]) > 0


@pytest.mark.unit
def test_api_replay_404() -> None:
    """Test requesting a non-existent replay event raises 404 HTTPException."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        get_replay("non_existent_disaster_9999")
    assert exc_info.value.status_code == 404


@pytest.mark.unit
def test_api_cell_inspector() -> None:
    """Test inspecting a cell returns geomorphology and complete 86-D WorldState vector."""
    cell_id = "8760a9931ffffff"
    data = inspect_cell(cell_id)
    assert data["cell_id"] == cell_id
    assert "geomorphology" in data
    assert "elevation_m" in data["geomorphology"]
    assert "slope_deg" in data["geomorphology"]
    assert "gsi_susceptibility_index" in data["geomorphology"]

    # Verify 86-dimensional WorldState vector
    vector = data["world_state_vector"]
    assert vector["total_dimensions"] == 86
    assert len(vector["features"]) == 86
    feature_names = {f["feature_name"] for f in vector["features"]}
    assert "compound_saturation" in feature_names
    assert "slope_shear_risk" in feature_names
    assert "elevation_m" in feature_names
    assert "embedding_00" in feature_names


@pytest.mark.unit
def test_api_ground_truth_kodagu() -> None:
    """Test returning official GSI 2018 Kodagu landslide survey GeoJSON."""
    geojson = get_ground_truth("kodagu_2018")
    assert geojson["type"] == "FeatureCollection"
    assert geojson["total_features"] > 200
    kinds = {f["properties"]["kind"] for f in geojson["features"]}
    assert "gsi_point" in kinds
    assert "gsi_polygon" in kinds


@pytest.mark.unit
def test_api_ground_truth_shirur() -> None:
    """Test returning 2024 Shirur landslide epicenter and debris corridor."""
    geojson = get_ground_truth("shirur_2024")
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) >= 2


@pytest.mark.unit
def test_api_timeseries() -> None:
    """Test returning historical hourly time series for a cell."""
    data = get_cell_timeseries("shirur_2024", "8760a9931ffffff")
    assert isinstance(data, list)
    assert len(data) > 0
    assert "p1" in data[0]
    assert "cs" in data[0]
