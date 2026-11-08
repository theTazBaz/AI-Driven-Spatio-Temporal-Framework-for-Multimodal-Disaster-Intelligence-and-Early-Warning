"""Unit tests for Increment R8 — Ground-Truth Disaster Event Catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from disaster_intel.grid import CoastalGrid
from disaster_intel.ground_truth.catalog import GroundTruthCatalog

pytestmark = pytest.mark.unit


class TestGroundTruthCatalog:
    """Verify ground truth damage mapping and binary label generation for Objective 3."""

    def test_load_kodagu_2018_from_gsi_shapefiles(self) -> None:
        catalog = GroundTruthCatalog()
        event = catalog.load_kodagu_2018()

        assert event.event_id == "kodagu_2018"
        assert event.hazard_type == "landslide"
        assert event.metadata["point_records"] == 267
        assert event.metadata["polygon_records"] == 217

        # 60 unique H3 Res-7 cells confirmed in Kodagu
        assert len(event.positive_cells) == 60
        assert event.metadata["unique_positive_cells"] == 60

        # Verify a known positive cell from Talakaveri temple landslide (12.3862, 75.4932)
        grid = CoastalGrid(use_cache=True)
        talakaveri_cell = grid.latlng_to_cell(12.3862, 75.4932)
        assert event.is_cell_positive(talakaveri_cell)

    def test_load_shirur_2024(self) -> None:
        catalog = GroundTruthCatalog()
        event = catalog.load_shirur_2024()

        assert event.event_id == "shirur_2024"
        assert event.hazard_type == "landslide"
        assert event.metadata["taluk"] == "Ankola"

        grid = CoastalGrid(use_cache=True)
        shirur_cell = grid.latlng_to_cell(14.2818, 74.6186)
        assert event.is_cell_positive(shirur_cell)
        assert len(event.positive_cells) >= 1

    def test_load_uttara_kannada_2021(self) -> None:
        catalog = GroundTruthCatalog()
        event = catalog.load_uttara_kannada_2021()

        assert event.event_id == "uttara_kannada_2021"
        assert event.hazard_type == "flood"
        assert len(event.positive_cells) > 0

    def test_get_labels_dict_matches_grid_cardinality(self) -> None:
        catalog = GroundTruthCatalog()
        labels = catalog.get_labels_dict("kodagu_2018")

        assert len(labels) == len(catalog.grid.cells)
        # All values must be 0 or 1
        assert set(labels.values()) == {0, 1}
        # Exactly 60 positive cells
        assert sum(labels.values()) == 60

    def test_export_labels_json(self, tmp_path: Path) -> None:
        catalog = GroundTruthCatalog()
        out_file = tmp_path / "test_labels_kodagu.json"

        result_path = catalog.export_labels("kodagu_2018", out_file)
        assert result_path.exists()

        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["event_id"] == "kodagu_2018"
        assert data["positive_cells_count"] == 60
        assert data["total_cells"] == len(catalog.grid.cells)
        assert len(data["labels"]) == len(catalog.grid.cells)
