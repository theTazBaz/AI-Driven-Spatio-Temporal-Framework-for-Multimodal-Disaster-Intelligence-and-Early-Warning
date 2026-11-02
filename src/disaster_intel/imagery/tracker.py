"""Increment 11 — SatelliteScene and SceneRegistry for imagery lifecycle tracking.

Tracks satellite acquisitions over time:
- Cloudy scenes (cloud_fraction > threshold) are skipped to prevent corrupting state.
- Previous valid embeddings persist and keep ageing (embedding_age_hours increases).
- Interfaces with StateAssembler as an ImageryLookup source.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from disaster_intel.imagery.pca import PCAReducer
from disaster_intel.imagery.pooling import CellEmbedding

NAN = float("nan")
NUM_EMBEDDING_DIMS = 32


@dataclass(frozen=True)
class SatelliteScene:
    """Metadata and cell embeddings from a single satellite acquisition pass."""

    scene_id: str
    acquisition_time: datetime
    cloud_fraction: float
    cell_embeddings: Mapping[str, CellEmbedding]

    def is_cloudy(self, threshold: float = 0.3) -> bool:
        return self.cloud_fraction > threshold


@dataclass(frozen=True)
class _CellRecord:
    acquisition_time: datetime
    values: tuple[float, ...]  # 32 dimensions
    coverage: float


class SceneRegistry:
    """Lifecycle registry managing satellite acquisitions and per-cell ageing."""

    def __init__(
        self,
        cloud_threshold: float = 0.3,
        pca: PCAReducer | None = None,
    ) -> None:
        self.cloud_threshold = cloud_threshold
        self._pca = pca or PCAReducer(n_components=NUM_EMBEDDING_DIMS)
        # Mapping: cell -> list of historical records sorted by acquisition_time
        self._records: dict[str, list[_CellRecord]] = {}

    def register_scene(self, scene: SatelliteScene) -> bool:
        """Register a new satellite scene pass.

        Returns:
            bool: True if scene was accepted, False if skipped due to cloud cover.
        """
        if scene.is_cloudy(self.cloud_threshold):
            return False

        for cell, cell_emb in scene.cell_embeddings.items():
            if cell_emb.coverage <= 0.0 or np.isnan(cell_emb.embedding).all():
                continue

            raw_emb = cell_emb.embedding
            if raw_emb.shape[-1] != NUM_EMBEDDING_DIMS:
                reduced = self._pca.transform(raw_emb)
            else:
                reduced = raw_emb.astype(np.float16)

            record = _CellRecord(
                acquisition_time=scene.acquisition_time,
                values=tuple(float(v) for v in reduced.flatten()),
                coverage=cell_emb.coverage,
            )
            history = self._records.setdefault(cell, [])
            history.append(record)
            history.sort(key=lambda r: r.acquisition_time)

        return True

    def get_features(self, cell: str, as_of: datetime) -> dict[str, float]:
        """Query imagery features for cell as of a given timestamp."""
        out: dict[str, float] = {}
        history = self._records.get(cell, [])

        # Find the most recent record with acquisition_time <= as_of
        valid_records = [r for r in history if r.acquisition_time <= as_of]

        if not valid_records:
            for i in range(NUM_EMBEDDING_DIMS):
                out[f"embedding_{i:02d}"] = NAN
            out["embedding_age_hours"] = NAN
            out["embedding_coverage"] = NAN
            return out

        latest = valid_records[-1]
        age_hours = (as_of - latest.acquisition_time).total_seconds() / 3600.0

        for i, val in enumerate(latest.values):
            out[f"embedding_{i:02d}"] = val
        out["embedding_age_hours"] = max(0.0, age_hours)
        out["embedding_coverage"] = latest.coverage

        return out

    def get_imagery(self, cell: str, as_of: datetime) -> Mapping[str, float] | None:
        """ImageryLookup protocol implementation for StateAssembler."""
        features = self.get_features(cell, as_of)
        if math.isnan(features["embedding_age_hours"]):
            return None
        return features
