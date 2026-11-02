"""Increment 11 — Satellite imagery embeddings and geospatial foundation model module."""

from disaster_intel.imagery.chips import (
    KM_PER_DEG_LAT,
    Patch,
    SatelliteChip,
    create_chip_for_parent,
)
from disaster_intel.imagery.embedder import (
    Embedder,
    FakeEmbedder,
    PrithviEmbedder,
)
from disaster_intel.imagery.pca import PCAReducer
from disaster_intel.imagery.pooling import (
    CellEmbedding,
    pool_chip_to_cells,
)
from disaster_intel.imagery.tracker import (
    NUM_EMBEDDING_DIMS,
    SatelliteScene,
    SceneRegistry,
)

__all__ = [
    "KM_PER_DEG_LAT",
    "NUM_EMBEDDING_DIMS",
    "CellEmbedding",
    "Embedder",
    "FakeEmbedder",
    "PCAReducer",
    "Patch",
    "PrithviEmbedder",
    "SatelliteChip",
    "SatelliteScene",
    "SceneRegistry",
    "create_chip_for_parent",
    "pool_chip_to_cells",
]
