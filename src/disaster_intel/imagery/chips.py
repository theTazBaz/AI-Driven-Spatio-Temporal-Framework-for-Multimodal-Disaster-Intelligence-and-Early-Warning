"""Increment 11 — Satellite chip geometry and H3 cell discretization.

Chips are centred on H3 resolution-6 parent cells, covering the geographic extent
and centers of all 7 resolution-7 child cells. Each chip is subdivided into a grid
of patches (e.g. 14x14 patches for 224x224 chip with 16x16 patch size).
Every patch centre maps to exactly one H3 resolution-7 cell.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import h3

KM_PER_DEG_LAT = 111.32


@dataclass(frozen=True)
class Patch:
    """A single spatial patch within a satellite chip."""

    row: int
    col: int
    center_lat: float
    center_lon: float
    h3_cell: str


@dataclass(frozen=True)
class SatelliteChip:
    """A square satellite imagery chip centred on a resolution-6 parent cell."""

    parent_cell: str
    center_lat: float
    center_lon: float
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    height_px: int = 224
    width_px: int = 224
    patch_size_px: int = 16
    patches: tuple[Patch, ...] = ()

    def covers_res6_children(self) -> bool:
        """Check if all 7 children of parent_cell have their centres within this chip."""
        children = h3.cell_to_children(self.parent_cell, 7)
        return self.covers_cells(children)

    def covers_cells(self, cells: Sequence[str]) -> bool:
        """Check if all given cell centres fall within the geographic bounds of this chip."""
        for c in cells:
            clat, clon = h3.cell_to_latlng(c)
            if not (self.min_lat <= clat <= self.max_lat and self.min_lon <= clon <= self.max_lon):
                return False
        return True


def create_chip_for_parent(
    parent_cell: str,
    chip_span_km: float = 8.0,
    height_px: int = 224,
    width_px: int = 224,
    patch_size_px: int = 16,
) -> SatelliteChip:
    """Create a SatelliteChip centred on an H3 resolution-6 parent cell.

    At 8.0 km span, the chip comfortably encloses the ~2.52 km radius of the 7 child cells.
    """
    center_lat, center_lon = h3.cell_to_latlng(parent_cell)

    half_km = chip_span_km / 2.0
    half_lat = half_km / KM_PER_DEG_LAT
    deg_lon_km = KM_PER_DEG_LAT * math.cos(math.radians(center_lat))
    half_lon = half_km / deg_lon_km

    min_lat = center_lat - half_lat
    max_lat = center_lat + half_lat
    min_lon = center_lon - half_lon
    max_lon = center_lon + half_lon

    num_patches_y = height_px // patch_size_px
    num_patches_x = width_px // patch_size_px

    patches: list[Patch] = []
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    for r in range(num_patches_y):
        patch_lat = max_lat - ((r + 0.5) / num_patches_y) * lat_span
        for c in range(num_patches_x):
            patch_lon = min_lon + ((c + 0.5) / num_patches_x) * lon_span
            cell_id = h3.latlng_to_cell(patch_lat, patch_lon, 7)
            patches.append(
                Patch(
                    row=r,
                    col=c,
                    center_lat=patch_lat,
                    center_lon=patch_lon,
                    h3_cell=str(cell_id),
                )
            )

    return SatelliteChip(
        parent_cell=parent_cell,
        center_lat=center_lat,
        center_lon=center_lon,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        height_px=height_px,
        width_px=width_px,
        patch_size_px=patch_size_px,
        patches=tuple(patches),
    )
