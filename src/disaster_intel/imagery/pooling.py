"""Increment 11 — Spatial pooling from satellite chip patches to H3 cells.

Aggregates patch tokens into cell-level representations via arithmetic mean over
valid (non-cloudy) patches. Computes per-cell coverage fraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from disaster_intel.imagery.chips import Patch, SatelliteChip


@dataclass(frozen=True)
class CellEmbedding:
    """Cell-level pooled satellite embedding representation."""

    cell: str
    embedding: np.ndarray
    coverage: float
    valid_patches: int
    total_patches: int


def pool_chip_to_cells(
    chip: SatelliteChip,
    patch_embeddings: np.ndarray,
    cloud_mask: np.ndarray | None = None,
) -> dict[str, CellEmbedding]:
    """Pool patch embeddings to H3 resolution-7 cells via arithmetic mean.

    Args:
        chip: SatelliteChip containing the patch grid and H3 cell mappings.
        patch_embeddings: Array of shape (ny, nx, embed_dim).
        cloud_mask: Optional binary mask of shape (ny, nx) where != 0 indicates cloud (invalid).

    Returns:
        dict[str, CellEmbedding]: Mapping from H3 cell ID to pooled CellEmbedding.
    """
    embed_dim = patch_embeddings.shape[-1]
    patches_by_cell: dict[str, list[Patch]] = {}
    for patch in chip.patches:
        patches_by_cell.setdefault(patch.h3_cell, []).append(patch)

    out: dict[str, CellEmbedding] = {}
    for cell, patches in patches_by_cell.items():
        total = len(patches)
        valid: list[Patch] = []
        for p in patches:
            if cloud_mask is None or cloud_mask[p.row, p.col] == 0:
                valid.append(p)

        valid_count = len(valid)
        coverage = valid_count / total if total > 0 else 0.0

        if valid_count > 0:
            accum = np.zeros(embed_dim, dtype=np.float64)
            for p in valid:
                accum += patch_embeddings[p.row, p.col]
            mean_emb = (accum / valid_count).astype(np.float32)
        else:
            mean_emb = np.full(embed_dim, np.nan, dtype=np.float32)

        out[cell] = CellEmbedding(
            cell=cell,
            embedding=mean_emb,
            coverage=coverage,
            valid_patches=valid_count,
            total_patches=total,
        )

    return out
