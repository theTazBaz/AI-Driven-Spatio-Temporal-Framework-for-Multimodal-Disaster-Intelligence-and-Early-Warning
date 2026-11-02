"""Increment 11 — Tests for Embedder interface, Prithvi-EO-2.0, chip-to-cell pooling,
PCA reduction, cloud masking, and FeatureSpec v2.

TDD requirements from plan.md:
1. Chips centred on res-6 parent cover centres of its 7 children.
2. Every patch centre maps to exactly one cell.
3. Pooling = mean of patches in each cell (fake embedder).
4. Each cell carries coverage fraction.
5. Cloudy scene skipped -> previous embedding keeps ageing.
6. PCA output is deterministic float16 within storage budget (64 bytes/cell).
7. Slow test: real Prithvi on one small chip gives correct shape, no NaN, same output twice.
8. FeatureSpec v2: embedding, age, and coverage appended to FeatureSpec.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import h3
import numpy as np
import pytest

from disaster_intel.grid import CoastalGrid
from disaster_intel.imagery import (
    CellEmbedding,
    FakeEmbedder,
    Patch,
    PCAReducer,
    PrithviEmbedder,
    SatelliteScene,
    SceneRegistry,
    create_chip_for_parent,
    pool_chip_to_cells,
)
from disaster_intel.state.assembler import StateAssembler
from disaster_intel.state.spec import (
    FeatureSpec,
    SpecMismatchError,
    StateVector,
)
from disaster_intel.state.static_layers import StaticFeatures
from disaster_intel.windows import WindowValue

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def grid() -> CoastalGrid:
    return CoastalGrid(use_cache=True)


@pytest.fixture(scope="module")
def sample_parent(grid: CoastalGrid) -> str:
    cell = grid.cells[len(grid.cells) // 2]
    return str(h3.cell_to_parent(cell, 6))


# ── Test 1: Chips centred on res-6 parent cover centres of 7 children ──


@pytest.mark.unit
def test_chip_centred_on_res6_parent_covers_all_7_children(grid: CoastalGrid) -> None:
    # Test across multiple distinct parents in coastal Karnataka
    tested_parents = list({h3.cell_to_parent(c, 6) for c in grid.cells[:100]})[:10]
    for parent in tested_parents:
        chip = create_chip_for_parent(parent, chip_span_km=8.0)
        children = h3.cell_to_children(parent, 7)
        assert len(children) == 7

        # Verify chip covers all 7 child centers
        assert chip.covers_res6_children(), f"Chip for parent {parent} did not cover all children"
        for child in children:
            child_lat, child_lon = h3.cell_to_latlng(child)
            assert chip.min_lat <= child_lat <= chip.max_lat
            assert chip.min_lon <= child_lon <= chip.max_lon


# ── Test 2: Every patch centre maps to exactly one cell ────────────────


@pytest.mark.unit
def test_every_patch_centre_maps_to_exactly_one_cell(sample_parent: str) -> None:
    chip = create_chip_for_parent(sample_parent, height_px=224, width_px=224, patch_size_px=16)
    expected_patches = (224 // 16) * (224 // 16)  # 14 * 14 = 196
    assert len(chip.patches) == expected_patches

    seen_patches: set[tuple[int, int]] = set()
    for patch in chip.patches:
        assert isinstance(patch, Patch)
        assert (patch.row, patch.col) not in seen_patches
        seen_patches.add((patch.row, patch.col))

        # Maps to a valid H3 resolution-7 cell
        assert patch.h3_cell is not None
        assert h3.is_valid_cell(patch.h3_cell)
        assert h3.get_resolution(patch.h3_cell) == 7

        # Center matches h3.latlng_to_cell
        expected_cell = h3.latlng_to_cell(patch.center_lat, patch.center_lon, 7)
        assert patch.h3_cell == expected_cell


# ── Test 3: Pooling = mean of patches in each cell (fake embedder) ─────


@pytest.mark.unit
def test_spatial_pooling_computes_exact_mean_of_cell_patches(sample_parent: str) -> None:
    chip = create_chip_for_parent(sample_parent, height_px=224, width_px=224, patch_size_px=16)
    embed_dim = 64
    embedder = FakeEmbedder(embed_dim=embed_dim, mode="deterministic")

    patch_embeddings = embedder.embed_chip(chip)
    assert patch_embeddings.shape == (14, 14, embed_dim)

    pooled = pool_chip_to_cells(chip, patch_embeddings)
    children = set(h3.cell_to_children(sample_parent, 7))

    # All 7 children are covered by patches
    for child in children:
        assert child in pooled
        cell_embed = pooled[child]
        assert isinstance(cell_embed, CellEmbedding)

        # Manually compute mean over patches mapped to this child
        child_patches = [p for p in chip.patches if p.h3_cell == child]
        assert len(child_patches) == cell_embed.total_patches
        assert cell_embed.valid_patches == len(child_patches)
        assert cell_embed.coverage == 1.0

        manual_sum = np.zeros(embed_dim, dtype=np.float64)
        for p in child_patches:
            manual_sum += patch_embeddings[p.row, p.col]
        expected_mean = manual_sum / len(child_patches)

        np.testing.assert_allclose(cell_embed.embedding, expected_mean, rtol=1e-5, atol=1e-5)


# ── Test 4: Each cell carries coverage fraction ───────────────────────


@pytest.mark.unit
def test_cell_coverage_fraction_with_cloud_mask(sample_parent: str) -> None:
    chip = create_chip_for_parent(sample_parent, height_px=224, width_px=224, patch_size_px=16)
    embed_dim = 32
    embedder = FakeEmbedder(embed_dim=embed_dim, mode="constant", constant_val=2.5)
    patch_embeddings = embedder.embed_chip(chip)

    # Cloud mask: 1 = cloudy (invalid), 0 = clear (valid)
    # Mask out rows >= 7
    cloud_mask = np.zeros((14, 14), dtype=np.uint8)
    cloud_mask[7:, :] = 1

    pooled = pool_chip_to_cells(chip, patch_embeddings, cloud_mask=cloud_mask)

    for cell, cell_embed in pooled.items():
        cell_patches = [p for p in chip.patches if p.h3_cell == cell]
        valid_patches = [p for p in cell_patches if cloud_mask[p.row, p.col] == 0]

        assert cell_embed.total_patches == len(cell_patches)
        assert cell_embed.valid_patches == len(valid_patches)
        expected_coverage = len(valid_patches) / len(cell_patches)
        assert math.isclose(cell_embed.coverage, expected_coverage, rel_tol=1e-6)

        if len(valid_patches) > 0:
            # Constant value 2.5 was preserved by mean
            np.testing.assert_allclose(cell_embed.embedding, 2.5, rtol=1e-5)
        else:
            # Completely cloudy cell has NaN embedding and 0 coverage
            assert np.isnan(cell_embed.embedding).all()
            assert cell_embed.coverage == 0.0


# ── Test 5: Cloudy scene skipped -> previous embedding keeps ageing ───


@pytest.mark.unit
def test_cloudy_scene_skipped_and_previous_embedding_ages(sample_parent: str) -> None:
    registry = SceneRegistry(cloud_threshold=0.3)
    t0 = datetime(2024, 7, 10, 10, 0, tzinfo=UTC)

    # Scene 1: Clear acquisition (cloud fraction 0.10 <= 0.30)
    chip1 = create_chip_for_parent(sample_parent)
    embedder = FakeEmbedder(embed_dim=32, mode="constant", constant_val=1.0)
    emb1 = embedder.embed_chip(chip1)
    pooled1 = pool_chip_to_cells(chip1, emb1)

    scene1 = SatelliteScene(
        scene_id="SCENE_001",
        acquisition_time=t0,
        cloud_fraction=0.10,
        cell_embeddings=pooled1,
    )
    accepted1 = registry.register_scene(scene1)
    assert accepted1 is True

    # At t0 + 2h: embedding is present and age is 2.0 hours
    t_check1 = t0 + timedelta(hours=2)
    sample_child = h3.cell_to_children(sample_parent, 7)[0]
    features1 = registry.get_features(sample_child, as_of=t_check1)
    assert features1["embedding_age_hours"] == 2.0
    assert features1["embedding_coverage"] == 1.0
    assert not math.isnan(features1["embedding_00"])

    # Scene 2: Monsoonal cloudy acquisition at t0 + 24h (cloud fraction 0.85 > 0.30)
    t_cloud = t0 + timedelta(hours=24)
    scene2 = SatelliteScene(
        scene_id="SCENE_002_CLOUDY",
        acquisition_time=t_cloud,
        cloud_fraction=0.85,
        cell_embeddings={},
    )
    accepted2 = registry.register_scene(scene2)
    assert accepted2 is False  # SKIPPED!

    # At t0 + 48h: previous embedding from SCENE_001 is still active, ageing to 48.0 hours
    t_check2 = t0 + timedelta(hours=48)
    features2 = registry.get_features(sample_child, as_of=t_check2)
    assert features2["embedding_age_hours"] == 48.0
    assert features2["embedding_00"] == features1["embedding_00"]
    assert features2["embedding_coverage"] == 1.0


# ── Test 6: PCA output deterministic float16 within storage budget ────


@pytest.mark.unit
def test_pca_output_deterministic_float16_storage_budget() -> None:
    rng = np.random.default_rng(12345)
    in_dim = 768  # Standard ViT / Prithvi embedding dimension
    n_components = 32  # Target storage budget dimensions

    # Calibration dataset
    x_train = rng.standard_normal((100, in_dim)).astype(np.float32)

    reducer = PCAReducer(n_components=n_components, seed=42)
    reducer.fit(x_train)

    # Test sample vector
    x_test = rng.standard_normal((1, in_dim)).astype(np.float32)

    out1 = reducer.transform(x_test)
    out2 = reducer.transform(x_test)

    # 1. Output shape is (1, 32)
    assert out1.shape == (1, 32)

    # 2. Output dtype is float16
    assert out1.dtype == np.float16

    # 3. Storage budget: 32 dimensions x 2 bytes = 64 bytes
    assert out1.nbytes == 64, f"Expected 64 bytes per cell embedding, got {out1.nbytes}"

    # 4. Deterministic: exact bit-for-bit equality
    assert np.array_equal(out1, out2)

    # 5. No NaN or Inf
    assert not np.isnan(out1).any()
    assert not np.isinf(out1).any()


# ── Test 7: FeatureSpec v2 layout and fingerprint ─────────────────────


@pytest.mark.unit
def test_feature_spec_v2_appends_imagery_and_changes_fingerprint() -> None:
    v1 = FeatureSpec.v1()
    v2 = FeatureSpec.v2()

    # V1 is preserved
    assert v1.version == 1
    assert len(v1) == 29
    assert len(v1.imagery) == 0

    # V2 has version 2 and 63 features
    assert v2.version == 2
    assert len(v2) == 29 + 34  # 63 features
    assert len(v2.imagery) == 34
    assert v2.fingerprint != v1.fingerprint

    # Imagery features order: embedding_00..embedding_31, age, coverage
    expected_imagery = (
        *(f"embedding_{i:02d}" for i in range(32)),
        "embedding_age_hours",
        "embedding_coverage",
    )
    assert v2.imagery == expected_imagery

    # All features in v2 have definitions
    for name in v2.names:
        assert name in v2.definitions

    # SpecMismatchError when loading v1 vector under v2 spec
    dummy_v1_values = tuple(1.0 for _ in range(len(v1)))
    dummy_v1_cov = tuple(1.0 for _ in range(len(v1)))
    v1_vector = StateVector(
        cell="876015ad0ffffff",
        as_of=datetime(2024, 7, 10, 12, 0, tzinfo=UTC),
        spec_version=v1.version,
        spec_fingerprint=v1.fingerprint,
        values=dummy_v1_values,
        coverage=dummy_v1_cov,
    )
    saved_json = v1_vector.to_json()

    with pytest.raises(SpecMismatchError):
        StateVector.from_json(saved_json, v2)


# ── Test 8: StateAssembler integrates imagery with FeatureSpec v2 ─────


class DummyWindowSource:
    def windows(
        self,
        cell: str,
        as_of: datetime,
        source: str | None = None,
        window_hours: Mapping[str, int] | None = None,
    ) -> dict[str, dict[str, WindowValue]]:
        wh = window_hours or {"1h": 1, "6h": 6, "24h": 24, "72h": 72, "7d": 168}
        vars_list = [
            "precipitation",
            "soil_moisture_surface",
            "soil_moisture_rootzone",
            "discharge",
            "seismic_magnitude",
        ]
        return {
            v: {w: WindowValue(value=1.5, observed_hours=wh[w], hours=wh[w]) for w in wh}
            for v in vars_list
        }


class DummyStaticLookup:
    def get(self, cell: str) -> StaticFeatures:
        return StaticFeatures(
            elevation_m=25.0,
            slope_deg=3.2,
            population_density_km2=150.0,
        )


@pytest.mark.unit
def test_state_assembler_assembles_v2_with_imagery(sample_parent: str) -> None:
    spec = FeatureSpec.v2()
    registry = SceneRegistry(cloud_threshold=0.3)
    t0 = datetime(2024, 7, 10, 10, 0, tzinfo=UTC)

    # Register a clear scene
    chip = create_chip_for_parent(sample_parent)
    embedder = FakeEmbedder(embed_dim=32, mode="constant", constant_val=0.42)
    patch_emb = embedder.embed_chip(chip)
    pooled = pool_chip_to_cells(chip, patch_emb)

    scene = SatelliteScene(
        scene_id="SCENE_CLEAR_01",
        acquisition_time=t0,
        cloud_fraction=0.05,
        cell_embeddings=pooled,
    )
    registry.register_scene(scene)

    assembler = StateAssembler(
        static=DummyStaticLookup(),
        cold=DummyWindowSource(),
        imagery=registry,
        spec=spec,
    )

    sample_child = h3.cell_to_children(sample_parent, 7)[0]
    when = t0 + timedelta(hours=5)
    vector = assembler.get_state(sample_child, as_of=when, tier="cold")

    assert vector.spec_version == 2
    assert vector.spec_fingerprint == spec.fingerprint
    assert len(vector) == 63

    # Check mapping
    mapping = vector.as_mapping(spec)
    assert mapping["elevation_m"] == 25.0
    assert mapping["embedding_age_hours"] == 5.0
    assert mapping["embedding_coverage"] == 1.0
    assert math.isclose(mapping["embedding_00"], 0.42, rel_tol=1e-3)


@pytest.mark.unit
def test_state_assembler_v2_handles_missing_imagery(sample_parent: str) -> None:
    # Assembler with no imagery source attached
    spec = FeatureSpec.v2()
    assembler = StateAssembler(
        static=DummyStaticLookup(),
        cold=DummyWindowSource(),
        imagery=None,
        spec=spec,
    )
    sample_child = h3.cell_to_children(sample_parent, 7)[0]
    when = datetime(2024, 7, 10, 15, 0, tzinfo=UTC)
    vector = assembler.get_state(sample_child, as_of=when, tier="cold")

    mapping = vector.as_mapping(spec)
    assert math.isnan(mapping["embedding_age_hours"])
    assert math.isnan(mapping["embedding_00"])
    assert math.isnan(mapping["embedding_31"])
    assert math.isnan(mapping["embedding_coverage"])


# ── Test 9: Slow test: PrithviEmbedder on small chip ──────────────────


@pytest.mark.slow
def test_prithvi_embedder_small_chip_slow(sample_parent: str) -> None:
    """Slow test verifying real PrithviEmbedder produces correct shape, no NaNs,

    and deterministic output. Runs when PyTorch is installed and --run-slow is selected.
    """
    chip = create_chip_for_parent(sample_parent, height_px=224, width_px=224, patch_size_px=16)
    embedder = PrithviEmbedder()

    # Synthetic 6-band optical input (B02, B03, B04, B8A, B11, B12)
    # Shape: (6, 224, 224)
    rng = np.random.default_rng(42)
    image_bands = rng.uniform(0.0, 1.0, size=(6, 224, 224)).astype(np.float32)

    out1 = embedder.embed_chip(chip, image_bands)
    out2 = embedder.embed_chip(chip, image_bands)

    # 1. Correct patch grid shape: (14, 14, embed_dim)
    assert out1.shape == (14, 14, embedder.embed_dim)

    # 2. No NaN or Inf
    assert not np.isnan(out1).any()
    assert not np.isinf(out1).any()

    # 3. Same output twice (deterministic)
    np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-5)
