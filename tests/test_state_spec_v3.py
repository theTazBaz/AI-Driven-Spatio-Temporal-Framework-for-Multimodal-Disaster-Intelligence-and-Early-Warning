"""Increment 18 tests — FeatureSpec v3 (86-dimensional WorldState) & Multimodal Assembly.

TDD Test Suite (plan.md Section 10.9):
1. FeatureSpec.v3() defines exactly 86 features across 4 modalities:
   - 45 Dynamic (5 baseline + 2 SAR + 2 INSAT) x 5 windows (1h, 6h, 24h, 72h, 7d)
   - 5 Static (Copernicus DEM, slope, WorldPop, GSI susceptibility, GSI lithology)
   - 34 Imagery (32-dim multimodal PCA embedding, age, coverage)
   - 2 Compound derived (compound_saturation, slope_shear_risk)
2. SpecMismatchError on cross-spec loading (v1 <-> v3, v2 <-> v3, truncated, fingerprint mismatch).
3. Compound interaction terms logic:
   - compound_saturation = soil_moisture_rootzone__24h * precipitation__72h
   - slope_shear_risk = slope_deg * gsi_susceptibility_index
4. Serialization & NaN preservation (null in JSON, never coerced to 0).
5. StateAssembler integration with FeatureSpec.v3():
   - Complete 86-feature vector populated across dynamic, static, imagery, and derived blocks.
   - Dual-tier agreement (hot == cold) across all 86 dimensions.
   - Deterministic rollback at historical timestamp T.
6. Historical spell replay across Shirur Landslide (2024) and Uttara Kannada Floods (2021).
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from disaster_intel.clock import FakeClock
from disaster_intel.models import Observation, VariableCatalog
from disaster_intel.state import (
    FeatureSpec,
    SpecMismatchError,
    StateAssembler,
    StateVector,
    StaticFeatures,
    StaticLayers,
)
from disaster_intel.state.spec import (
    V1_WINDOWS,
    V3_DYNAMIC_VARIABLES,
    V3_IMAGERY,
    V3_STATIC,
    feature_name,
)
from disaster_intel.storage import FakeRedis, HotStore
from disaster_intel.windows import WindowValue, reference_windows

pytestmark = pytest.mark.unit

NOW = datetime(2024, 7, 16, 12, 0, tzinfo=UTC)
SHIRUR_CELL = "8760b526dffffff"
CATALOG = VariableCatalog.default()
SPEC_V3 = FeatureSpec.v3()
NAN = float("nan")


def make_v3_vector(
    spec: FeatureSpec = SPEC_V3, values: tuple[float, ...] | None = None
) -> StateVector:
    vals = values if values is not None else tuple(float(i) for i in range(len(spec)))
    return StateVector(
        cell=SHIRUR_CELL,
        as_of=NOW,
        spec_version=spec.version,
        spec_fingerprint=spec.fingerprint,
        values=vals,
        coverage=tuple(None for _ in vals),
    )


def make_obs(
    cell: str,
    source: str,
    variable: str,
    value: float,
    unit: str,
    ts: datetime,
) -> Observation:
    return Observation.from_raw(
        cell=cell,
        source=source,
        variable=variable,
        raw_value=value,
        raw_unit=unit,
        timestamp=ts,
    )


# ── 1. Layout & Completeness (86 features) ───────────────────────────


@pytest.mark.unit
class TestFeatureSpecV3Layout:
    """Verify exact 86-feature dimensionality and layout structure."""

    def test_v3_has_86_features_in_exact_order(self) -> None:
        names = SPEC_V3.names
        assert len(SPEC_V3) == 86
        assert len(names) == 86

        # 1. Dynamic block (45 features)
        expected_dynamic = [feature_name(v, w) for v in V3_DYNAMIC_VARIABLES for w in V1_WINDOWS]
        assert len(expected_dynamic) == 45
        assert list(names[:45]) == expected_dynamic

        # Verify new SAR & INSAT variables in dynamic block
        assert "sar_water_fraction__1h" in names[:45]
        assert "sar_vh_backscatter_db__24h" in names[:45]
        assert "insat_cloud_top_tb__1h" in names[:45]
        assert "insat_convective_cooling_rate__6h" in names[:45]

        # 2. Static block (5 features)
        assert list(names[45:50]) == list(V3_STATIC)
        assert names[45] == "elevation_m"
        assert names[46] == "slope_deg"
        assert names[47] == "population_density_km2"
        assert names[48] == "gsi_susceptibility_index"
        assert names[49] == "gsi_lithology_cohesion"

        # 3. Imagery block (34 features)
        assert list(names[50:84]) == list(V3_IMAGERY)
        assert names[50] == "embedding_00"
        assert names[81] == "embedding_31"
        assert names[82] == "embedding_age_hours"
        assert names[83] == "embedding_coverage"

        # 4. Derived block (2 features)
        assert list(names[84:86]) == ["compound_saturation", "slope_shear_risk"]

        # All names are unique
        assert len(set(names)) == len(names)

    def test_all_features_have_definitions(self) -> None:
        assert len(SPEC_V3.definitions) == 86
        for name in SPEC_V3.names:
            assert name in SPEC_V3.definitions
            assert len(SPEC_V3.definitions[name]) > 5

    def test_fingerprint_changes_with_version_and_content(self) -> None:
        spec_v1 = FeatureSpec.v1()
        spec_v2 = FeatureSpec.v2()
        assert SPEC_V3.fingerprint != spec_v1.fingerprint
        assert SPEC_V3.fingerprint != spec_v2.fingerprint

        # Determinism
        assert FeatureSpec.v3().fingerprint == SPEC_V3.fingerprint

        # Mutation of definition changes fingerprint
        mod_defs = dict(SPEC_V3.definitions)
        mod_defs["slope_shear_risk"] = "modified definition"
        mutated = dataclasses.replace(SPEC_V3, definitions=mod_defs)
        assert mutated.fingerprint != SPEC_V3.fingerprint


# ── 2. Cross-Spec Failure (Fails Loudly) ──────────────────────────────


@pytest.mark.unit
class TestCrossSpecFailsLoudly:
    """Verify that StateVector loading under mismatched spec raises SpecMismatchError."""

    def test_loading_v1_vector_under_v3_fails(self) -> None:
        v1_spec = FeatureSpec.v1()
        v1_vec = StateVector(
            cell=SHIRUR_CELL,
            as_of=NOW,
            spec_version=v1_spec.version,
            spec_fingerprint=v1_spec.fingerprint,
            values=tuple(float(i) for i in range(len(v1_spec))),
            coverage=tuple(None for _ in range(len(v1_spec))),
        )
        json_str = v1_vec.to_json()
        with pytest.raises(SpecMismatchError, match="Vector was built under spec v1"):
            StateVector.from_json(json_str, SPEC_V3)

    def test_loading_v2_vector_under_v3_fails(self) -> None:
        v2_spec = FeatureSpec.v2()
        v2_vec = StateVector(
            cell=SHIRUR_CELL,
            as_of=NOW,
            spec_version=v2_spec.version,
            spec_fingerprint=v2_spec.fingerprint,
            values=tuple(float(i) for i in range(len(v2_spec))),
            coverage=tuple(None for _ in range(len(v2_spec))),
        )
        json_str = v2_vec.to_json()
        with pytest.raises(SpecMismatchError, match="Vector was built under spec v2"):
            StateVector.from_json(json_str, SPEC_V3)

    def test_loading_v3_vector_under_v1_or_v2_fails(self) -> None:
        v3_vec = make_v3_vector()
        json_str = v3_vec.to_json()
        with pytest.raises(SpecMismatchError, match="Vector was built under spec v3"):
            StateVector.from_json(json_str, FeatureSpec.v1())
        with pytest.raises(SpecMismatchError, match="Vector was built under spec v3"):
            StateVector.from_json(json_str, FeatureSpec.v2())

    def test_truncated_vector_fails(self) -> None:
        payload = make_v3_vector().to_dict()
        payload["values"] = payload["values"][:45]  # Truncate to dynamic only
        payload["coverage"] = payload["coverage"][:45]
        with pytest.raises(SpecMismatchError, match="Vector has 45 values; spec v3 defines 86"):
            StateVector.from_dict(payload, SPEC_V3)


# ── 3. Compound Interaction Terms ─────────────────────────────────────


@pytest.mark.unit
class TestCompoundDerivedFeatures:
    """Verify both compound interaction terms."""

    def test_compound_saturation_logic(self) -> None:
        # Product of rootzone soil moisture (24h) and cumulative rain (72h)
        assert SPEC_V3.derive(
            {
                "soil_moisture_rootzone__24h": 0.45,
                "precipitation__72h": 120.0,
                "slope_deg": 10.0,
                "gsi_susceptibility_index": 0.4,
            }
        )[0] == pytest.approx(54.0)

        # Zero if either input is 0.0
        assert (
            SPEC_V3.derive(
                {
                    "soil_moisture_rootzone__24h": 0.0,
                    "precipitation__72h": 120.0,
                    "slope_deg": 10.0,
                    "gsi_susceptibility_index": 0.4,
                }
            )[0]
            == 0.0
        )

        # NaN if either is missing
        assert math.isnan(
            SPEC_V3.derive(
                {
                    "soil_moisture_rootzone__24h": NAN,
                    "precipitation__72h": 120.0,
                    "slope_deg": 10.0,
                    "gsi_susceptibility_index": 0.4,
                }
            )[0]
        )

    def test_slope_shear_risk_logic(self) -> None:
        # Product of slope angle in degrees and GSI landslide susceptibility index [0, 1]
        # Agumbe escarpment: 28.0 deg slope on High susceptibility (0.7) -> 19.6
        derived = SPEC_V3.derive(
            {
                "soil_moisture_rootzone__24h": 0.35,
                "precipitation__72h": 50.0,
                "slope_deg": 28.0,
                "gsi_susceptibility_index": 0.7,
            }
        )
        assert derived[1] == pytest.approx(19.6)

        # Flat coastal terrain (0.0 deg) -> risk is 0.0 regardless of susceptibility
        flat = SPEC_V3.derive(
            {
                "soil_moisture_rootzone__24h": 0.35,
                "precipitation__72h": 50.0,
                "slope_deg": 0.0,
                "gsi_susceptibility_index": 1.0,
            }
        )
        assert flat[1] == 0.0

        # Missing GSI layer or DEM -> NaN
        missing = SPEC_V3.derive(
            {
                "soil_moisture_rootzone__24h": 0.35,
                "precipitation__72h": 50.0,
                "slope_deg": 28.0,
                "gsi_susceptibility_index": NAN,
            }
        )
        assert math.isnan(missing[1])


# ── 4. Serialization & NaN Preservation ──────────────────────────────


@pytest.mark.unit
class TestV3Serialization:
    """Verify strict JSON serialization with null for NaN."""

    def test_nan_preservation_and_same_values(self) -> None:
        vals = [float(i) for i in range(86)]
        vals[10] = NAN  # sar_water_fraction__1h
        vals[48] = NAN  # gsi_susceptibility_index
        vec = make_v3_vector(values=tuple(vals))

        text = vec.to_json()
        assert "NaN" not in text
        data = json.loads(text)
        assert data["values"][10] is None
        assert data["values"][48] is None

        restored = StateVector.from_json(text, SPEC_V3)
        assert math.isnan(restored.values[10])
        assert math.isnan(restored.values[48])
        assert restored.same_values(vec)


# ── 5. StateAssembler Integration with FeatureSpec.v3() ──────────────


class MemoryColdStore:
    """In-memory cold store satisfying WindowSource."""

    def __init__(self, obs_list: Sequence[Observation] | None = None) -> None:
        self.obs: list[Observation] = list(obs_list) if obs_list else []

    def write(self, batch: Sequence[Observation]) -> None:
        self.obs.extend(batch)

    def windows(
        self,
        cell: str,
        as_of: datetime,
        source: str | None = None,
        window_hours: Mapping[str, int] | None = None,
    ) -> dict[str, dict[str, WindowValue]]:
        if window_hours is None:
            return reference_windows(self.obs, cell, as_of, CATALOG, source=source)
        return reference_windows(
            self.obs, cell, as_of, CATALOG, source=source, window_hours=window_hours
        )


class MockClayImageryLookup:
    """Simulates Clay foundation model embedding source."""

    def __init__(self, embedding_vector: Sequence[float] | None = None) -> None:
        self.embedding = (
            list(embedding_vector) if embedding_vector else [0.05 * (i % 7) for i in range(32)]
        )

    def get_features(self, cell: str, as_of: datetime) -> Mapping[str, float]:
        feat: dict[str, float] = {f"embedding_{i:02d}": self.embedding[i] for i in range(32)}
        feat["embedding_age_hours"] = 1.5
        feat["embedding_coverage"] = 0.95
        return feat


@pytest.mark.unit
class TestStateAssemblerV3Integration:
    """Verify StateAssembler assembles the complete 86-feature vector."""

    def test_state_assembler_v3_all_modalities(self) -> None:
        clock = FakeClock(NOW)
        cell = SHIRUR_CELL

        # 1. Observations spanning Baseline, SAR, and INSAT
        obs = [
            make_obs(cell, "open-meteo", "precipitation", 25.0, "mm", NOW),
            make_obs(cell, "open-meteo", "soil_moisture_rootzone", 0.42, "m3/m3", NOW),
            make_obs(cell, "sentinel-1", "sar_water_fraction", 0.35, "fraction", NOW),
            make_obs(cell, "sentinel-1", "sar_vh_backscatter_db", -19.5, "dB", NOW),
            make_obs(cell, "insat-3d", "insat_cloud_top_tb", 198.0, "K", NOW),
            make_obs(cell, "insat-3d", "insat_convective_cooling_rate", -14.0, "K/15min", NOW),
        ]

        # Hot and cold stores
        fake_redis = FakeRedis()
        hot = HotStore(fake_redis, clock=clock)
        hot.write(obs)

        cold = MemoryColdStore(obs)

        # Static layers with GSI geology
        features = {
            cell: StaticFeatures(
                elevation_m=420.0,
                slope_deg=26.5,
                population_density_km2=65.0,
                gsi_susceptibility_index=0.7,
                gsi_lithology_cohesion=0.4,
            ),
        }
        static = StaticLayers(features)

        # Imagery lookup
        imagery = MockClayImageryLookup()

        assembler = StateAssembler(
            static=static,
            hot=hot,
            cold=cold,
            imagery=imagery,
            spec=SPEC_V3,
            clock=clock,
        )

        vector = assembler.get_state(cell, as_of=NOW, tier="hot")
        assert len(vector) == 86
        m = vector.as_mapping(SPEC_V3)

        # Check Dynamic
        assert m["precipitation__1h"] == pytest.approx(25.0)
        assert m["sar_water_fraction__1h"] == pytest.approx(0.35)
        assert m["sar_vh_backscatter_db__1h"] == pytest.approx(-19.5)
        assert m["insat_cloud_top_tb__1h"] == pytest.approx(198.0)
        assert m["insat_convective_cooling_rate__1h"] == pytest.approx(-14.0)

        # Check Static
        assert m["elevation_m"] == 420.0
        assert m["slope_deg"] == 26.5
        assert m["population_density_km2"] == 65.0
        assert m["gsi_susceptibility_index"] == 0.7
        assert m["gsi_lithology_cohesion"] == 0.4

        # Check Imagery (Clay)
        assert m["embedding_00"] == pytest.approx(0.0)
        assert m["embedding_01"] == pytest.approx(0.05)
        assert m["embedding_age_hours"] == 1.5
        assert m["embedding_coverage"] == 0.95

        # Check Compound
        assert m["compound_saturation"] == pytest.approx(0.42 * 25.0)
        assert m["slope_shear_risk"] == pytest.approx(26.5 * 0.7)

    def test_dual_tier_agreement_hot_equals_cold_86_features(self) -> None:
        clock = FakeClock(NOW)
        cell = SHIRUR_CELL
        obs = [
            make_obs(cell, "open-meteo", "precipitation", 45.0, "mm", NOW),
            make_obs(cell, "sentinel-1", "sar_water_fraction", 0.12, "fraction", NOW),
            make_obs(cell, "insat-3d", "insat_cloud_top_tb", 205.0, "K", NOW),
        ]
        hot = HotStore(FakeRedis(), clock=clock)
        hot.write(obs)
        cold = MemoryColdStore(obs)
        static = StaticLayers(
            {
                cell: StaticFeatures(
                    elevation_m=150.0,
                    slope_deg=12.0,
                    population_density_km2=200.0,
                    gsi_susceptibility_index=0.4,
                    gsi_lithology_cohesion=0.4,
                )
            }
        )
        imagery = MockClayImageryLookup()

        assembler = StateAssembler(
            static=static, hot=hot, cold=cold, imagery=imagery, spec=SPEC_V3, clock=clock
        )
        comparison = assembler.compare_tiers(cell, as_of=NOW, rel_tol=1e-3, abs_tol=1e-3)
        assert comparison.agrees, f"Mismatches: {comparison.differing}"

    def test_deterministic_rollback_across_timestamps(self) -> None:
        """State vector computed at historical T equals live state computed at T."""
        t_historical = NOW - timedelta(hours=48)
        clock = FakeClock(t_historical)
        cell = SHIRUR_CELL

        obs_historical = [
            make_obs(cell, "open-meteo", "precipitation", 60.0, "mm", t_historical),
            make_obs(cell, "sentinel-1", "sar_water_fraction", 0.5, "fraction", t_historical),
        ]
        obs_future = [
            make_obs(cell, "open-meteo", "precipitation", 10.0, "mm", NOW),
        ]

        cold = MemoryColdStore(obs_historical + obs_future)
        static = StaticLayers(
            {
                cell: StaticFeatures(
                    elevation_m=150.0,
                    slope_deg=12.0,
                    population_density_km2=200.0,
                    gsi_susceptibility_index=0.4,
                    gsi_lithology_cohesion=0.4,
                )
            }
        )

        assembler = StateAssembler(static=static, cold=cold, spec=SPEC_V3, clock=clock)
        live_past_state = assembler.get_state(cell, as_of=t_historical, tier="cold")

        # Advance clock to NOW and request past timestamp t_historical
        clock.advance_to(NOW)
        replayed_state = assembler.get_state(cell, as_of=t_historical, tier="cold")

        assert live_past_state.same_values(replayed_state)


# ── 6. Historical Replay & Foundation Model Comparison ────────────────


@pytest.mark.unit
class TestHistoricalReplayAndFoundationComparison:
    """Multi-sensor historical scenario replay and optical vs multimodal foundation comparison."""

    def test_shirur_landslide_2024_multimodal_state_evolution(self) -> None:
        """Simulate multi-sensor disaster dynamics leading to July 16, 2024 Shirur landslide."""
        t_peak = datetime(2024, 7, 16, 9, 0, tzinfo=UTC)
        clock = FakeClock(t_peak)
        cell = SHIRUR_CELL

        # Multi-sensor observations simulating the hours before the slope collapse:
        # 1. Open-Meteo: torrential precipitation accumulating over 72h (320 mm)
        # 2. Soil moisture: saturated rootzone (0.48 m3/m3)
        # 3. INSAT-3D/3DR: rapid convective cloud-top freezing (196 K, -16 K/15min cooling rate)
        # 4. Sentinel-1 SAR: saturated ground water fraction (0.40) & backscatter drop (-20.5 dB)
        obs = [
            make_obs(cell, "open-meteo", "precipitation", 320.0, "mm", t_peak),
            make_obs(cell, "open-meteo", "soil_moisture_rootzone", 0.48, "m3/m3", t_peak),
            make_obs(cell, "insat-3d", "insat_cloud_top_tb", 196.0, "K", t_peak),
            make_obs(cell, "insat-3d", "insat_convective_cooling_rate", -16.0, "K/15min", t_peak),
            make_obs(cell, "sentinel-1", "sar_water_fraction", 0.40, "fraction", t_peak),
            make_obs(cell, "sentinel-1", "sar_vh_backscatter_db", -20.5, "dB", t_peak),
        ]

        # Static layers: Western Ghats high-relief escarpment with GSI High susceptibility
        # Weathered laterite cap over sheared chlorite schist (low cohesion 0.35)
        static = StaticLayers(
            {
                cell: StaticFeatures(
                    elevation_m=340.0,
                    slope_deg=31.5,
                    population_density_km2=45.0,
                    gsi_susceptibility_index=0.85,
                    gsi_lithology_cohesion=0.35,
                )
            }
        )

        cold = MemoryColdStore(obs)
        imagery = MockClayImageryLookup()
        assembler = StateAssembler(
            static=static, cold=cold, imagery=imagery, spec=SPEC_V3, clock=clock
        )

        state = assembler.get_state(cell, as_of=t_peak, tier="cold")
        assert len(state) == 86
        m = state.as_mapping(SPEC_V3)

        # Compound saturation: 0.48 rootzone * 320.0 mm rain = 153.6
        assert m["compound_saturation"] == pytest.approx(153.6)

        # Slope shear risk: 31.5 deg slope * 0.85 GSI susceptibility = 26.775
        assert m["slope_shear_risk"] == pytest.approx(26.775)

        # Convective storm signal captured in 1h window
        assert m["insat_cloud_top_tb__1h"] <= 210.0  # Cold top threshold
        assert m["insat_convective_cooling_rate__1h"] <= -10.0  # Rapid cooling threshold

        # SAR ground saturation captured
        assert m["sar_water_fraction__1h"] >= 0.30
        assert m["sar_vh_backscatter_db__1h"] <= -18.0

    def test_uttara_kannada_floods_2021_radar_flood_inundation(self) -> None:
        """Simulate low-lying Kali/Gangavali river basin inundation during July 2021 floods."""
        t_flood = datetime(2021, 7, 23, 15, 0, tzinfo=UTC)
        clock = FakeClock(t_flood)
        flood_cell = "8760b2494ffffff"

        obs = [
            make_obs(flood_cell, "open-meteo", "discharge", 1450.0, "m3/s", t_flood),
            make_obs(flood_cell, "sentinel-1", "sar_water_fraction", 0.78, "fraction", t_flood),
            make_obs(flood_cell, "sentinel-1", "sar_vh_backscatter_db", -23.5, "dB", t_flood),
        ]

        # Coastal floodplain: low elevation (12 m), low slope (1.5 deg), alluvium
        static = StaticLayers(
            {
                flood_cell: StaticFeatures(
                    elevation_m=12.0,
                    slope_deg=1.5,
                    population_density_km2=680.0,
                    gsi_susceptibility_index=0.1,  # Low landslide susceptibility
                    gsi_lithology_cohesion=0.1,  # Alluvium
                )
            }
        )

        cold = MemoryColdStore(obs)
        assembler = StateAssembler(static=static, cold=cold, spec=SPEC_V3, clock=clock)
        state = assembler.get_state(flood_cell, as_of=t_flood, tier="cold")
        m = state.as_mapping(SPEC_V3)

        assert m["discharge__1h"] == pytest.approx(1450.0)
        assert m["sar_water_fraction__1h"] == pytest.approx(0.78)
        assert m["sar_vh_backscatter_db__1h"] == pytest.approx(-23.5)
        # Flat land -> slope shear risk negligible
        assert m["slope_shear_risk"] == pytest.approx(0.15)

    def test_optical_prithvi_vs_multimodal_clay_under_cloud_blindness(self) -> None:
        """Verify Clay preserves valid representations under clouds whereas optical goes blind."""
        from disaster_intel.imagery.chips import create_chip_for_parent
        from disaster_intel.imagery.embedder import ClayEmbedder, PrithviEmbedder

        chip = create_chip_for_parent(
            "8660b2497ffffff", height_px=32, width_px=32, patch_size_px=16
        )

        # 1. Optical-only Prithvi: accepts 6 optical bands. When obscured by clouds (NaNs):
        prithvi = PrithviEmbedder(embed_dim=768, patch_size=16, in_channels=6)
        optical_cloudy = np.full((6, 32, 32), np.nan, dtype=np.float32)
        prithvi_out = prithvi.embed_chip(chip, image_data=optical_cloudy)
        # Purely optical model outputs NaNs when optical bands are missing
        assert np.isnan(prithvi_out).all()

        # 2. Multimodal Clay: accepts 9 bands (SAR + Optical + DEM).
        # When optical bands (indices 2:8) are obscured by monsoon clouds:
        clay = ClayEmbedder(embed_dim=768, patch_size=16, in_channels=9)
        multimodal_data = np.zeros((9, 32, 32), dtype=np.float32)
        multimodal_data[0:2] = 0.5  # SAR active
        multimodal_data[2:8] = np.nan  # Optical cloud-blinded
        multimodal_data[8] = 0.3  # DEM topography active

        cloud_mask = np.ones((2, 2), dtype=np.uint8)  # 100% cloudy patches
        clay_out = clay.embed_chip(chip, image_data=multimodal_data, cloud_mask=cloud_mask)

        # Clay produces 100% finite embeddings from radar and topography
        assert not np.isnan(clay_out).any()
        assert clay_out.shape == (2, 2, 768)
