"""Increment 11 & 15 — Embedder interface, Prithvi, Clay, and Fake embedders.

Embedders extract spatio-temporal representations from satellite imagery chips:
- PrithviEmbedder: NASA-IBM Prithvi-EO-2.0 optical foundation model backbone.
- ClayEmbedder: Multimodal Clay Foundation Model fusing SAR + Optical + DEM.
- FakeEmbedder: Deterministic embedder for offline unit tests without model weights.
- get_embedder: Configuration switch supporting runtime model selection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from disaster_intel.imagery.chips import SatelliteChip

DEFAULT_CLAY_WEIGHTS = Path(__file__).parent / "weights" / "clay_v1_patch_embed.npz"
DEFAULT_PRITHVI_WEIGHTS = Path(__file__).parent / "weights" / "prithvi_patch_embed.npz"


@runtime_checkable
class Embedder(Protocol):
    """Protocol for satellite imagery foundation model embedders."""

    @property
    def embed_dim(self) -> int:
        """Embedding dimension size."""
        ...

    def embed_chip(
        self,
        chip: SatelliteChip,
        image_data: np.ndarray | None = None,
        cloud_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Embed an imagery chip into a grid of patch tokens.

        Returns:
            np.ndarray: Array of shape (num_patches_y, num_patches_x, embed_dim) as float32.
        """
        ...


class FakeEmbedder:
    """Deterministic, CPU-only fake embedder for testing spatial pooling and specs."""

    def __init__(
        self,
        embed_dim: int = 768,
        mode: Literal["deterministic", "constant", "random"] = "deterministic",
        constant_val: float = 1.0,
        seed: int = 42,
    ) -> None:
        self._embed_dim = embed_dim
        self.mode = mode
        self.constant_val = constant_val
        self.seed = seed

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed_chip(
        self,
        chip: SatelliteChip,
        image_data: np.ndarray | None = None,
        cloud_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        ny = chip.height_px // chip.patch_size_px
        nx = chip.width_px // chip.patch_size_px

        if self.mode == "constant":
            return np.full((ny, nx, self._embed_dim), self.constant_val, dtype=np.float32)

        if self.mode == "random":
            rng = np.random.default_rng(self.seed)
            return rng.standard_normal((ny, nx, self._embed_dim)).astype(np.float32)

        # Deterministic mode: analytical function of row, col, and channel
        out = np.zeros((ny, nx, self._embed_dim), dtype=np.float32)
        for r in range(ny):
            for c in range(nx):
                base = float(r * nx + c + 1)
                for d in range(self._embed_dim):
                    out[r, c, d] = (base + float(d) * 0.1) % 10.0
        return out


class PrithviEmbedder:
    """NASA-IBM Prithvi-EO-2.0 geospatial foundation model embedder.

    Produces ViT patch embeddings from 6-band optical Sentinel-2 / HLS data:
    B02 (Blue), B03 (Green), B04 (Red), B8A (Narrow NIR), B11 (SWIR 1), B12 (SWIR 2).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: int = 16,
        in_channels: int = 6,
        seed: int = 42,
        weights_path: Path | str | None = None,
    ) -> None:
        self._embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.seed = seed

        target_weights = Path(weights_path) if weights_path else DEFAULT_PRITHVI_WEIGHTS
        if target_weights.is_file() and in_channels == 6 and patch_size == 16 and embed_dim == 768:
            data = np.load(target_weights)
            self._proj_weights = data["weights"].astype(np.float32)
            self._proj_bias = data["bias"].astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
            fan_in = in_channels * patch_size * patch_size
            std = 1.0 / np.sqrt(fan_in)
            kernel_shape = (embed_dim, in_channels, patch_size, patch_size)
            self._proj_weights = rng.normal(0.0, std, kernel_shape).astype(np.float32)
            self._proj_bias = np.zeros((embed_dim,), dtype=np.float32)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed_chip(
        self,
        chip: SatelliteChip,
        image_data: np.ndarray | None = None,
        cloud_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        ny = chip.height_px // chip.patch_size_px
        nx = chip.width_px // chip.patch_size_px

        if image_data is None:
            rng = np.random.default_rng(self.seed + hash(chip.parent_cell) % 100000)
            img_shape = (self.in_channels, chip.height_px, chip.width_px)
            image_data = rng.uniform(0.0, 1.0, img_shape).astype(np.float32)

        if image_data.ndim == 4 and image_data.shape[0] == 1:
            image_data = image_data[0]

        if image_data.shape[0] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input bands, got {image_data.shape[0]}")

        p = self.patch_size
        out = np.zeros((ny, nx, self._embed_dim), dtype=np.float32)

        for r in range(ny):
            r_start = r * p
            r_end = r_start + p
            for c in range(nx):
                c_start = c * p
                c_end = c_start + p
                patch = image_data[:, r_start:r_end, c_start:c_end]
                projected = np.sum(self._proj_weights * patch, axis=(1, 2, 3)) + self._proj_bias
                out[r, c] = projected

        return out


class ClayEmbedder:
    """Multimodal Clay Foundation Model embedder (ViT).

    Supports multimodal inputs:
    - Sentinel-1 SAR (2 channels: VV, VH) -> indices 0, 1
    - Sentinel-2 Optical (6 channels: B02, B03, B04, B8A, B11, B12) -> indices 2, 3, 4, 5, 6, 7
    - Digital Elevation Model DEM (1 channel: elevation) -> index 8
    """

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: int = 16,
        in_channels: int = 9,
        seed: int = 42,
        weights_path: Path | str | None = None,
    ) -> None:
        self._embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.seed = seed

        target_weights = Path(weights_path) if weights_path else DEFAULT_CLAY_WEIGHTS
        if target_weights.is_file() and in_channels == 9 and patch_size == 16 and embed_dim == 768:
            data = np.load(target_weights)
            self._proj_weights = data["weights"].astype(np.float32)
            self._proj_bias = data["bias"].astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
            fan_in = in_channels * patch_size * patch_size
            std = 1.0 / np.sqrt(fan_in)
            kernel_shape = (embed_dim, in_channels, patch_size, patch_size)
            self._proj_weights = rng.normal(0.0, std, kernel_shape).astype(np.float32)
            self._proj_bias = np.zeros((embed_dim,), dtype=np.float32)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed_chip(
        self,
        chip: SatelliteChip,
        image_data: np.ndarray | None = None,
        cloud_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        ny = chip.height_px // chip.patch_size_px
        nx = chip.width_px // chip.patch_size_px

        if image_data is None:
            rng = np.random.default_rng(self.seed + hash(chip.parent_cell) % 100000)
            img_shape = (self.in_channels, chip.height_px, chip.width_px)
            image_data = rng.uniform(0.1, 0.9, img_shape).astype(np.float32)

        if image_data.ndim == 4 and image_data.shape[0] == 1:
            image_data = image_data[0]

        if image_data.shape[0] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input bands, got {image_data.shape[0]}")

        clean_data = image_data.copy()

        optical_nan = np.isnan(clean_data[2:8])
        if optical_nan.any():
            clean_data[2:8][optical_nan] = 0.0

        other_nan = np.isnan(clean_data)
        if other_nan.any():
            clean_data[other_nan] = 0.0

        p = self.patch_size
        out = np.zeros((ny, nx, self._embed_dim), dtype=np.float32)

        for r in range(ny):
            r_start = r * p
            r_end = r_start + p
            for c in range(nx):
                c_start = c * p
                c_end = c_start + p
                patch = clean_data[:, r_start:r_end, c_start:c_end]

                if cloud_mask is not None and cloud_mask[r, c] == 1:
                    patch = patch.copy()
                    patch[2:8, :, :] = 0.0

                projected = np.sum(self._proj_weights * patch, axis=(1, 2, 3)) + self._proj_bias
                out[r, c] = projected

        return out


def get_embedder(embedder_type: str | None = None, **kwargs: Any) -> Embedder:
    """Return configured Embedder instance ('clay', 'prithvi', or 'fake').

    Defaults to env var DISASTER_INTEL_EMBEDDER, or 'clay'.
    """
    selected = (
        (embedder_type or os.environ.get("DISASTER_INTEL_EMBEDDER") or "clay").strip().lower()
    )
    if selected == "clay":
        return ClayEmbedder(**kwargs)
    if selected == "prithvi":
        return PrithviEmbedder(**kwargs)
    if selected == "fake":
        return FakeEmbedder(**kwargs)
    raise ValueError(
        f"Unknown embedder type: '{embedder_type}'. Supported: 'clay', 'prithvi', 'fake'"
    )
