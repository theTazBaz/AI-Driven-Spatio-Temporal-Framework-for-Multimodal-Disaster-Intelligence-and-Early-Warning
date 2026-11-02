"""Increment 11 — PCA dimensionality reduction for satellite embeddings.

Reduces high-dimensional embeddings (e.g. 768 from Prithvi-EO-2.0) down to 32 float16
dimensions (~64 bytes per cell per acquisition), enforcing the Phase 1 storage budget.
Outputs are deterministic bit-for-bit.
"""

from __future__ import annotations

from typing import cast

import numpy as np


class PCAReducer:
    """Deterministic Principal Component Analysis reducer outputting float16 arrays."""

    def __init__(self, n_components: int = 32, seed: int = 42) -> None:
        self.n_components = n_components
        self.seed = seed
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> PCAReducer:
        """Fit PCA components using SVD on centered training data."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.mean_ = np.mean(x, axis=0, keepdims=True).astype(np.float32)
        x_centered = x - self.mean_

        # SVD on centered data
        _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
        self.components_ = vt[: self.n_components].astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project input embeddings to 32 dimensions and cast to float16."""
        if self.components_ is None or self.mean_ is None:
            # If not explicitly fit, create deterministic orthonormal projection
            in_dim = x.shape[-1]
            rng = np.random.default_rng(self.seed)
            mat = rng.standard_normal((in_dim, self.n_components)).astype(np.float32)
            q, _ = np.linalg.qr(mat)
            self.components_ = q.T  # (n_components, in_dim)
            self.mean_ = np.zeros((1, in_dim), dtype=np.float32)

        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        x_centered = x_flat - self.mean_
        projected = np.dot(x_centered, self.components_.T)
        out = projected.reshape(*orig_shape[:-1], self.n_components)
        return cast(np.ndarray, out.astype(np.float16))

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
