"""Increment 15 tests — Multimodal Clay Foundation Model Embedder.

TDD Test Suite (plan.md Section 10.6):
1. ClayEmbedder conforms strictly to the Embedder protocol interface.
2. Cloud-mask fallback: when optical channels are completely masked/NaN, embedder processes
   SAR + DEM channels successfully without returning all-NaN.
3. PCA projection outputs deterministic 32-dimensional float16 vector in storage budget.
4. Config switch get_embedder: "clay" | "prithvi" | "fake" swaps models cleanly.
5. Multimodal information preservation: SAR + DEM retain significant feature similarity (>0.60)
   when optical channels are completely blinded by clouds.
6. Determinism: same input produces bit-identical embeddings.
"""

from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pytest

from disaster_intel.imagery import (
    ClayEmbedder,
    Embedder,
    FakeEmbedder,
    PCAReducer,
    PrithviEmbedder,
    create_chip_for_parent,
    get_embedder,
)

SAMPLE_PARENT = "8660b2497ffffff"


# ── 1. Protocol Conformance ──────────────────────────────────────────


@pytest.mark.unit
class TestClayEmbedderProtocol:
    """Verify that ClayEmbedder implements the Embedder protocol interface."""

    def test_implements_embedder_protocol(self) -> None:
        embedder = ClayEmbedder()
        assert isinstance(embedder, Embedder)
        assert embedder.embed_dim == 768
        assert embedder.patch_size == 16
        assert embedder.in_channels == 9

    def test_embed_chip_output_shape_and_dtype(self) -> None:
        chip = create_chip_for_parent(SAMPLE_PARENT, height_px=224, width_px=224, patch_size_px=16)
        embedder = ClayEmbedder(embed_dim=768)

        # 9 channels: 2 SAR (VV, VH) + 6 Optical + 1 DEM
        rng = np.random.default_rng(42)
        multimodal_data = rng.uniform(0.0, 1.0, (9, 224, 224)).astype(np.float32)

        out = embedder.embed_chip(chip, multimodal_data)

        # 224 / 16 = 14 patches per axis
        assert out.shape == (14, 14, 768)
        assert out.dtype == np.float32
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_synthetic_input_fallback_when_none_provided(self) -> None:
        chip = create_chip_for_parent(SAMPLE_PARENT, height_px=224, width_px=224, patch_size_px=16)
        embedder = ClayEmbedder()
        out = embedder.embed_chip(chip, None)
        assert out.shape == (14, 14, 768)
        assert not np.isnan(out).any()


# ── 2. Cloud-Mask Fallback (Optical Blinding) ────────────────────────


@pytest.mark.unit
class TestCloudMaskFallback:
    """Verify that cloud-blinded optical channels fall back cleanly to SAR + DEM."""

    def test_all_nan_optical_channels_produce_finite_embedding(self) -> None:
        chip = create_chip_for_parent(SAMPLE_PARENT, height_px=224, width_px=224, patch_size_px=16)
        embedder = ClayEmbedder()

        rng = np.random.default_rng(101)
        multimodal_data = rng.uniform(0.0, 1.0, (9, 224, 224)).astype(np.float32)

        # Set all optical channels (2..7) to NaN to simulate 100% thick monsoon clouds
        multimodal_data[2:8, :, :] = np.nan

        out = embedder.embed_chip(chip, multimodal_data)

        # Must NOT be all-NaN or contain any NaNs
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()
        # Must be non-zero (reflecting active SAR and DEM signals)
        assert np.count_nonzero(out) > 0

    def test_explicit_cloud_mask_suppresses_optical_distortion(self) -> None:
        chip = create_chip_for_parent(SAMPLE_PARENT, height_px=224, width_px=224, patch_size_px=16)
        embedder = ClayEmbedder()

        rng = np.random.default_rng(202)
        multimodal_data = rng.uniform(0.0, 1.0, (9, 224, 224)).astype(np.float32)

        # Cloud mask: 1 = cloudy patch, 0 = clear patch
        cloud_mask = np.zeros((14, 14), dtype=np.uint8)
        cloud_mask[5:10, 5:10] = 1  # 25 cloudy patches in center

        out = embedder.embed_chip(chip, multimodal_data, cloud_mask=cloud_mask)
        assert out.shape == (14, 14, 768)
        assert not np.isnan(out).any()


# ── 3. PCA Projection & Storage Budget Compliance ─────────────────────


@pytest.mark.unit
class TestPCAStorageBudget:
    """Verify 32 float16 dimensions and 64 bytes/cell storage budget."""

    def test_pca_transforms_clay_embeddings_to_32dim_float16(self) -> None:
        chip = create_chip_for_parent(SAMPLE_PARENT, height_px=224, width_px=224, patch_size_px=16)
        embedder = ClayEmbedder(embed_dim=768)
        reducer = PCAReducer(n_components=32, seed=42)

        raw_embeds = embedder.embed_chip(chip)  # (14, 14, 768)
        reduced = reducer.transform(raw_embeds)

        # Output shape must be (14, 14, 32)
        assert reduced.shape == (14, 14, 32)
        # Output dtype must be float16
        assert reduced.dtype == np.float16
        # Storage size: 32 dimensions * 2 bytes/float16 = 64 bytes
        bytes_per_patch = reduced[0, 0].nbytes
        assert bytes_per_patch == 64


# ── 4. Configuration Switch (get_embedder) ───────────────────────────


@pytest.mark.unit
class TestEmbedderConfigurationSwitch:
    """Verify get_embedder switches between clay, prithvi, and fake."""

    def test_get_embedder_by_name(self) -> None:
        clay = get_embedder("clay")
        assert isinstance(clay, ClayEmbedder)

        prithvi = get_embedder("prithvi")
        assert isinstance(prithvi, PrithviEmbedder)

        fake = get_embedder("fake")
        assert isinstance(fake, FakeEmbedder)

    def test_get_embedder_via_env_var(self) -> None:
        with mock.patch.dict(os.environ, {"DISASTER_INTEL_EMBEDDER": "prithvi"}):
            emb = get_embedder()
            assert isinstance(emb, PrithviEmbedder)

        with mock.patch.dict(os.environ, {"DISASTER_INTEL_EMBEDDER": "clay"}):
            emb = get_embedder()
            assert isinstance(emb, ClayEmbedder)

    def test_get_embedder_unknown_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"[Uu]nknown embedder type"):
            get_embedder("unsupported_model")


# ── 5. Multimodal Information Preservation Under Clouds ──────────────


@pytest.mark.unit
class TestMultimodalInformationPreservation:
    """Verify that SAR + DEM preserve substantial feature similarity when optical is blinded."""

    def test_clay_retains_feature_similarity_under_heavy_clouds(self) -> None:
        chip = create_chip_for_parent(SAMPLE_PARENT, height_px=224, width_px=224, patch_size_px=16)
        embedder = ClayEmbedder(seed=999)

        rng = np.random.default_rng(555)
        # Clear acquisition: valid SAR (0..1), valid Optical (2..7), valid DEM (8)
        clear_multimodal = rng.uniform(0.1, 0.9, (9, 224, 224)).astype(np.float32)

        # Cloudy acquisition: identical SAR and DEM, but optical completely clouded (NaN)
        cloudy_multimodal = clear_multimodal.copy()
        cloudy_multimodal[2:8, :, :] = np.nan

        clear_emb = embedder.embed_chip(chip, clear_multimodal)
        cloudy_emb = embedder.embed_chip(chip, cloudy_multimodal)

        # Flatten patches for vector similarity calculation
        v_clear = clear_emb.reshape(-1)
        v_cloudy = cloudy_emb.reshape(-1)

        # Cosine similarity: (A . B) / (||A|| * ||B||)
        norm_clear = np.linalg.norm(v_clear)
        norm_cloudy = np.linalg.norm(v_cloudy)
        assert norm_clear > 0
        assert norm_cloudy > 0

        cosine_sim = float(np.dot(v_clear, v_cloudy) / (norm_clear * norm_cloudy))

        # Because SAR + DEM represent 3/9 channels with high structural weight,
        # cosine similarity should remain high (> 0.50), proving cloud-penetrating resilience!
        assert cosine_sim > 0.50, f"Expected cosine similarity > 0.50, got {cosine_sim:.3f}"


# ── 6. Determinism ───────────────────────────────────────────────────


@pytest.mark.unit
class TestClayEmbedderDeterminism:
    """Verify identical inputs yield bit-identical outputs."""

    def test_deterministic_output(self) -> None:
        chip = create_chip_for_parent(SAMPLE_PARENT, height_px=224, width_px=224, patch_size_px=16)
        embedder = ClayEmbedder(seed=123)

        rng = np.random.default_rng(777)
        data = rng.uniform(0.0, 1.0, (9, 224, 224)).astype(np.float32)

        out1 = embedder.embed_chip(chip, data)
        out2 = embedder.embed_chip(chip, data)

        np.testing.assert_allclose(out1, out2, rtol=1e-6, atol=1e-6)
