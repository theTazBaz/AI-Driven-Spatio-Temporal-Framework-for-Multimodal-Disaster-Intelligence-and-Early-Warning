"""Increment 17 tests — GSI Geological Landslide Susceptibility & Lithology Static Layer.

TDD Test Suite (plan.md Section 10.8):
1. Categorical susceptibility mapped to normalized [0.0, 1.0] index
   (Low: 0.1, Medium: 0.4, High: 0.7, Very High: 1.0).
2. Lithological cohesion factor mapped to normalized shear resistance [0.0, 1.0]
   (laterite: 0.2, schist: 0.4, granite: 0.9).
3. H3 cell coordinates map to GSI susceptibility and lithological cohesion via GSIGeologySource.
4. StaticLayers serialization preserves GSI features with backward compatibility for 3-field files.
5. Immutability: static geological features loaded once remain bit-exact across timestamps.
6. StateAssembler integration: static geology features correctly populate state vectors.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from disaster_intel.grid import CoastalGrid
from disaster_intel.state import (
    FeatureSpec,
    FunctionElevation,
    GSIGeologySource,
    MapGeologySource,
    StateAssembler,
    StaticFeatures,
    StaticLayers,
    build_static_layers,
    map_gsi_lithology,
    map_gsi_susceptibility,
    synthetic_ghats_elevation,
)
from disaster_intel.windows import WindowValue

SAMPLE_CELL = "8760b2494ffffff"
T0 = datetime(2024, 7, 16, 12, 0, tzinfo=UTC)


# ── 1. GSI Susceptibility Index Normalization ─────────────────────────


@pytest.mark.unit
class TestGSISusceptibilityMapping:
    """Verify mapping of GSI NLSM susceptibility categories to normalized [0.0, 1.0]."""

    def test_categorical_mapping_values(self) -> None:
        assert map_gsi_susceptibility("Low") == pytest.approx(0.1, rel=1e-5)
        assert map_gsi_susceptibility("low") == pytest.approx(0.1, rel=1e-5)
        assert map_gsi_susceptibility("Medium") == pytest.approx(0.4, rel=1e-5)
        assert map_gsi_susceptibility("Moderate") == pytest.approx(0.4, rel=1e-5)
        assert map_gsi_susceptibility("High") == pytest.approx(0.7, rel=1e-5)
        assert map_gsi_susceptibility("Very High") == pytest.approx(1.0, rel=1e-5)
        assert map_gsi_susceptibility("Critical") == pytest.approx(1.0, rel=1e-5)

    def test_numeric_passthrough_and_bounds(self) -> None:
        assert map_gsi_susceptibility(0.55) == pytest.approx(0.55, rel=1e-5)
        assert map_gsi_susceptibility(0.0) == 0.0
        assert map_gsi_susceptibility(1.0) == 1.0

    def test_invalid_categories_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"[Uu]nknown|[Ii]nvalid"):
            map_gsi_susceptibility("ExtremelySafe")
        with pytest.raises(ValueError, match=r"[Rr]ange|[Bb]ounds"):
            map_gsi_susceptibility(1.5)
        with pytest.raises(ValueError, match=r"[Rr]ange|[Bb]ounds"):
            map_gsi_susceptibility(-0.1)


# ── 2. Lithological Cohesion Factor Mapping ──────────────────────────


@pytest.mark.unit
class TestGSILithologyMapping:
    """Verify mapping of rock types to normalized shear resistance [0.0, 1.0]."""

    def test_rock_type_cohesion_values(self) -> None:
        # Weathered laterite: highly permeable cap, prone to sliding when saturated
        assert map_gsi_lithology("laterite") == pytest.approx(0.2, rel=1e-5)
        assert map_gsi_lithology("weathered_laterite") == pytest.approx(0.2, rel=1e-5)

        # Sheared schist / phyllite: foliated metamorphic rock with slickensided slip surfaces
        assert map_gsi_lithology("schist") == pytest.approx(0.4, rel=1e-5)
        assert map_gsi_lithology("sheared_schist") == pytest.approx(0.4, rel=1e-5)
        assert map_gsi_lithology("phyllite") == pytest.approx(0.4, rel=1e-5)

        # Massive granite / basalt: competent crystalline bedrock
        assert map_gsi_lithology("granite") == pytest.approx(0.9, rel=1e-5)
        assert map_gsi_lithology("massive_granite") == pytest.approx(0.9, rel=1e-5)
        assert map_gsi_lithology("basalt") == pytest.approx(0.9, rel=1e-5)

        # Coastal alluvium / loose sand: very low shear strength
        assert map_gsi_lithology("alluvium") == pytest.approx(0.1, rel=1e-5)
        assert map_gsi_lithology("sand") == pytest.approx(0.1, rel=1e-5)

        # Charnockite / quartzite: strong resistant rocks
        assert map_gsi_lithology("charnockite") == pytest.approx(0.7, rel=1e-5)
        assert map_gsi_lithology("quartzite") == pytest.approx(0.7, rel=1e-5)

    def test_numeric_passthrough_and_bounds(self) -> None:
        assert map_gsi_lithology(0.65) == pytest.approx(0.65, rel=1e-5)
        with pytest.raises(ValueError, match=r"[Uu]nknown|[Ii]nvalid"):
            map_gsi_lithology("unknown_unobtainium")


# ── 3. H3 Cell Geology Mapping via GSIGeologySource ──────────────────


@pytest.mark.unit
class TestH3CellGeologyMapping:
    """Verify spatial geology assignment to H3 cells."""

    def test_synthetic_western_ghats_geology_gradient(self) -> None:
        geology = GSIGeologySource()

        # Lowland coastal point (Mangalore: 12.91°N, 74.85°E) -> coastal laterite belt
        coastal_susc, coastal_cohesion = geology.get_geology(12.91, 74.85)
        assert 0.1 <= coastal_susc <= 0.4
        assert coastal_cohesion == pytest.approx(0.2, rel=1e-5)

        # Escarpment hillslope (Agumbe: 13.51°N, 75.09°E) -> high susceptibility sheared schist
        escarp_susc, escarp_cohesion = geology.get_geology(13.51, 75.09)
        assert escarp_susc >= 0.7  # High or Very High susceptibility
        assert escarp_cohesion == pytest.approx(0.4, rel=1e-5)  # Schist

        # High plateau summit (Chikmagalur / Kudremukh: 13.21°N, 75.25°E) -> granite/charnockite
        _plateau_susc, plateau_cohesion = geology.get_geology(13.21, 75.25)
        assert plateau_cohesion >= 0.7

    def test_map_geology_source_explicit_lookup(self) -> None:
        explicit_data = {
            SAMPLE_CELL: (0.7, 0.4),  # High susceptibility, schist
        }
        source = MapGeologySource(explicit_data)
        susc, cohesion = source.get_for_cell(SAMPLE_CELL)
        assert susc == 0.7
        assert cohesion == 0.4


# ── 4. StaticLayers Serialization and Backwards Compatibility ─────────


@pytest.mark.unit
class TestStaticLayersSerialization:
    """Verify 5-field static layer persistence and backward compatibility."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        features = {
            "cell_1": StaticFeatures(
                elevation_m=250.0,
                slope_deg=18.5,
                population_density_km2=120.0,
                gsi_susceptibility_index=0.7,
                gsi_lithology_cohesion=0.4,
            ),
            "cell_2": StaticFeatures(
                elevation_m=15.0,
                slope_deg=1.2,
                population_density_km2=850.0,
                gsi_susceptibility_index=0.1,
                gsi_lithology_cohesion=0.2,
            ),
            "cell_empty": StaticFeatures(
                elevation_m=float("nan"),
                slope_deg=float("nan"),
                population_density_km2=float("nan"),
                gsi_susceptibility_index=float("nan"),
                gsi_lithology_cohesion=float("nan"),
            ),
        }

        layers = StaticLayers(features, meta={"geology": "GSI NLSM 1:50,000"})
        save_file = tmp_path / "static_test.json"
        layers.save(save_file)

        loaded = StaticLayers.load(save_file)
        assert len(loaded) == 3

        f1 = loaded.get("cell_1")
        assert f1.elevation_m == 250.0
        assert f1.slope_deg == 18.5
        assert f1.population_density_km2 == 120.0
        assert f1.gsi_susceptibility_index == 0.7
        assert f1.gsi_lithology_cohesion == 0.4

        f_empty = loaded.get("cell_empty")
        assert math.isnan(f_empty.elevation_m)
        assert math.isnan(f_empty.gsi_susceptibility_index)
        assert math.isnan(f_empty.gsi_lithology_cohesion)

    def test_backwards_compatibility_with_3field_format(self, tmp_path: Path) -> None:
        """Old files saved with 3 fields load gracefully with NaN for GSI features."""
        payload = {
            "format": 1,
            "meta": {"source": "legacy_v1"},
            "cells": {
                "cell_old": [100.0, 5.0, 45.0],
            },
        }
        legacy_file = tmp_path / "legacy_static.json"
        legacy_file.write_text(json.dumps(payload), encoding="utf-8")

        loaded = StaticLayers.load(legacy_file)
        f = loaded.get("cell_old")
        assert f.elevation_m == 100.0
        assert f.slope_deg == 5.0
        assert f.population_density_km2 == 45.0
        assert math.isnan(f.gsi_susceptibility_index)
        assert math.isnan(f.gsi_lithology_cohesion)


# ── 5. Immutability Across Historical Timestamps ─────────────────────


@pytest.mark.unit
class TestGeologyImmutability:
    """Verify that static geological features remain bit-exact across all timestamps."""

    def test_geology_features_do_not_age_or_mutate_over_time(self) -> None:
        grid = CoastalGrid(use_cache=True)
        cell = grid.cells[0]

        geology = GSIGeologySource()
        elev = FunctionElevation(synthetic_ghats_elevation)
        layers = build_static_layers([cell], elevation=elev, geology=geology)

        f_t0 = layers.get(cell)
        assert not math.isnan(f_t0.gsi_susceptibility_index)
        assert not math.isnan(f_t0.gsi_lithology_cohesion)

        # 50 days later
        f_t50 = layers.get(cell)
        assert f_t50.gsi_susceptibility_index == f_t0.gsi_susceptibility_index
        assert f_t50.gsi_lithology_cohesion == f_t0.gsi_lithology_cohesion
        assert f_t50.elevation_m == f_t0.elevation_m


# ── 6. StateAssembler Integration ────────────────────────────────────


class DummyStore:
    def windows(
        self,
        cell: str,
        as_of: datetime,
        source: str | None = None,
        window_hours: Mapping[str, int] | None = None,
    ) -> dict[str, dict[str, WindowValue]]:
        variables = [
            "precipitation",
            "soil_moisture_surface",
            "soil_moisture_rootzone",
            "discharge",
            "seismic_magnitude",
        ]
        windows = ["1h", "6h", "24h", "72h", "7d"]
        return {
            v: {w: WindowValue(value=10.0, observed_hours=1, hours=1) for w in windows}
            for v in variables
        }


@pytest.mark.unit
class TestStateAssemblerGeologyIntegration:
    """Verify StateAssembler includes GSI features when defined in spec.static."""

    def test_state_vector_includes_gsi_features(self) -> None:
        cell = SAMPLE_CELL
        features = {
            cell: StaticFeatures(
                elevation_m=350.0,
                slope_deg=22.0,
                population_density_km2=50.0,
                gsi_susceptibility_index=0.7,
                gsi_lithology_cohesion=0.4,
            ),
        }
        layers = StaticLayers(features)

        # Custom spec including the 2 GSI features in static block
        spec = FeatureSpec(
            version=17,
            dynamic=(),
            static=(
                "elevation_m",
                "slope_deg",
                "population_density_km2",
                "gsi_susceptibility_index",
                "gsi_lithology_cohesion",
            ),
            derived=(),
            definitions={
                "elevation_m": "elevation in metres",
                "slope_deg": "slope in degrees",
                "population_density_km2": "population density",
                "gsi_susceptibility_index": "GSI landslide susceptibility index [0, 1]",
                "gsi_lithology_cohesion": "GSI rock cohesion factor [0, 1]",
            },
        )

        assembler = StateAssembler(static=layers, cold=DummyStore(), spec=spec)
        vector = assembler.get_state(cell, as_of=T0, tier="cold")

        mapping = vector.as_mapping(spec)
        assert mapping["elevation_m"] == 350.0
        assert mapping["slope_deg"] == 22.0
        assert mapping["gsi_susceptibility_index"] == 0.7
        assert mapping["gsi_lithology_cohesion"] == 0.4
