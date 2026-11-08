"""Increment 4 — Spatial Discretizers for Multimodal Telemetry.

Discretizes incoming telemetry geometries into H3 resolution-7 cells:
- PointDiscretizer: USGS earthquakes, gauge stations, localized points.
- GridNodeDiscretizer: 0.1 deg weather grid nodes (Open-Meteo, GloFAS) fanning out to ~20 cells.
- PolygonDiscretizer: NDMA SACHET CAP warning polygons, regional alerts.
- SpatialDiscretizer: Unified router dispatching on reading.geometry.kind.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import h3

from disaster_intel.grid import CoastalGrid
from disaster_intel.models import (
    GridNodeGeometry,
    Observation,
    PointGeometry,
    PolygonGeometry,
    RawReading,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_CACHE_DIR = Path(".cache")


def point_in_polygon(
    lat: float,
    lon: float,
    polygon: Sequence[tuple[float, float]],
    tolerance: float = 1e-7,
) -> bool:
    """Ray-casting algorithm to determine if point (lat, lon) is inside a polygon.

    Polygon is a sequence of (latitude, longitude) vertices.
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    p1_lat, p1_lon = polygon[0]
    for i in range(1, n + 1):
        p2_lat, p2_lon = polygon[i % n]
        if (p1_lon > lon) != (p2_lon > lon):
            lat_intersect = (p2_lat - p1_lat) * (lon - p1_lon) / (p2_lon - p1_lon) + p1_lat
            if lat <= lat_intersect + tolerance:
                inside = not inside
        p1_lat, p1_lon = p2_lat, p2_lon

    return inside


class PointDiscretizer:
    """Discretizes geographic points to exact H3 resolution-7 cells."""

    def __init__(self, grid: CoastalGrid | None = None) -> None:
        self.grid = grid or CoastalGrid(use_cache=True)

    def discretize_point(self, lat: float, lon: float) -> str:
        """Map latitude and longitude to H3 cell ID."""
        return self.grid.latlng_to_cell(lat, lon)

    def discretize(self, reading: RawReading) -> list[Observation]:
        """Convert a point RawReading to an Observation."""
        if not isinstance(reading.geometry, PointGeometry):
            raise ValueError(f"Expected PointGeometry, got {type(reading.geometry).__name__}")

        cell = self.discretize_point(reading.geometry.lat, reading.geometry.lon)
        return [Observation.from_reading(reading, cell)]


class FanOutTable:
    """Cell-to-source-node fan-out table for regular raster weather grids.

    Maps 0.1 degree grid nodes to the ~15-25 H3 res-7 cells they cover.
    """

    def __init__(
        self,
        cell_to_node: dict[str, tuple[float, float]],
        node_to_cells: dict[tuple[float, float], tuple[str, ...]],
        resolution_deg: float = 0.1,
    ) -> None:
        self.cell_to_node = cell_to_node
        self.node_to_cells = node_to_cells
        self.resolution_deg = resolution_deg
        self.unique_nodes: tuple[tuple[float, float], ...] = tuple(sorted(node_to_cells.keys()))

    @classmethod
    def from_grid(
        cls,
        grid: CoastalGrid,
        resolution_deg: float = 0.1,
        cache_path: Path | str | None = None,
        use_cache: bool = True,
        force_recompute: bool = False,
    ) -> FanOutTable:
        """Compute or load the fan-out table for the given grid."""
        resolved_cache = (
            Path(cache_path)
            if cache_path is not None
            else (DEFAULT_CACHE_DIR / f"fan_out_res{grid.resolution}_{resolution_deg}deg.json")
            if use_cache
            else None
        )

        if use_cache and not force_recompute and resolved_cache and resolved_cache.is_file():
            try:
                return cls.from_cache(resolved_cache)
            except Exception:
                # Fall back to recompute on corrupt cache
                pass

        cell_to_node: dict[str, tuple[float, float]] = {}
        node_to_cells_builder: dict[tuple[float, float], list[str]] = defaultdict(list)

        for cell in grid.cells:
            cell_lat, cell_lon = grid.cell_to_latlng(cell)
            node_lat = round(round(cell_lat / resolution_deg) * resolution_deg, 4)
            node_lon = round(round(cell_lon / resolution_deg) * resolution_deg, 4)
            node_coord = (node_lat, node_lon)

            cell_to_node[cell] = node_coord
            node_to_cells_builder[node_coord].append(cell)

        node_to_cells: dict[tuple[float, float], tuple[str, ...]] = {
            k: tuple(sorted(v)) for k, v in node_to_cells_builder.items()
        }

        instance = cls(
            cell_to_node=cell_to_node,
            node_to_cells=node_to_cells,
            resolution_deg=resolution_deg,
        )

        if use_cache and resolved_cache is not None:
            instance.save_cache(resolved_cache)

        return instance

    @classmethod
    def from_cache(cls, cache_path: Path | str) -> FanOutTable:
        """Load fan-out table from disk cache."""
        path = Path(cache_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        resolution_deg = float(data["resolution_deg"])
        cell_to_node = {k: (float(v[0]), float(v[1])) for k, v in data["cell_to_node"].items()}
        node_to_cells = {
            (float(k.split(",")[0]), float(k.split(",")[1])): tuple(v)
            for k, v in data["node_to_cells"].items()
        }

        return cls(
            cell_to_node=cell_to_node,
            node_to_cells=node_to_cells,
            resolution_deg=resolution_deg,
        )

    def save_cache(self, cache_path: Path | str) -> None:
        """Serialize fan-out table to disk cache."""
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "resolution_deg": self.resolution_deg,
            "cell_to_node": {k: [v[0], v[1]] for k, v in self.cell_to_node.items()},
            "node_to_cells": {f"{k[0]},{k[1]}": list(v) for k, v in self.node_to_cells.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)


class GridNodeDiscretizer:
    """Fans out 0.1 degree grid node readings to all enclosed H3 cells."""

    def __init__(
        self,
        grid: CoastalGrid | None = None,
        fan_out: FanOutTable | None = None,
    ) -> None:
        self.grid = grid or CoastalGrid(use_cache=True)
        self.fan_out = fan_out or FanOutTable.from_grid(grid=self.grid, resolution_deg=0.1)

    def discretize(self, reading: RawReading) -> list[Observation]:
        """Fan out a grid node reading to all overlapping cells in the fan-out table."""
        if not isinstance(reading.geometry, GridNodeGeometry):
            raise ValueError(f"Expected GridNodeGeometry, got {type(reading.geometry).__name__}")

        res = reading.geometry.resolution_deg or self.fan_out.resolution_deg
        node_lat = round(round(reading.geometry.lat / res) * res, 4)
        node_lon = round(round(reading.geometry.lon / res) * res, 4)
        node_coord = (node_lat, node_lon)
        cells = self.fan_out.node_to_cells.get(node_coord, ())

        return [Observation.from_reading(reading, cell) for cell in cells]


class PolygonDiscretizer:
    """Discretizes regional warning polygons (NDMA SACHET CAP alerts) into H3 cells."""

    def __init__(self, grid: CoastalGrid | None = None) -> None:
        self.grid = grid or CoastalGrid(use_cache=True)

    def discretize(self, reading: RawReading) -> list[Observation]:
        """Fill a polygon geometry with H3 cells and create Observations."""
        if not isinstance(reading.geometry, PolygonGeometry):
            raise ValueError(f"Expected PolygonGeometry, got {type(reading.geometry).__name__}")

        lats = [v[0] for v in reading.geometry.vertices]
        lons = [v[1] for v in reading.geometry.vertices]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # For subcontinental/oceanic polygons (>10 deg), evaluate directly against grid cells
        # to prevent h3.polygon_to_cells from generating tens of millions of out-of-bounds cells.
        if (max_lat - min_lat) > 10.0 or (max_lon - min_lon) > 10.0:
            if max_lat < 11.9 or min_lat > 15.5 or max_lon < 74.0 or min_lon > 76.0:
                return []
            matched = []
            for c in self.grid.cells:
                clat, clon = self.grid.cell_to_latlng(c)
                if (
                    min_lat <= clat <= max_lat
                    and min_lon <= clon <= max_lon
                    and point_in_polygon(clat, clon, reading.geometry.vertices)
                ):
                    matched.append(c)
            return [Observation.from_reading(reading, c) for c in sorted(matched)]

        poly = h3.LatLngPoly(reading.geometry.vertices)
        cells = h3.polygon_to_cells(poly, self.grid.resolution)

        # Filter to cells inside our study grid
        region_cells = [c for c in cells if c in self.grid]
        target_cells = region_cells if region_cells else list(cells)

        return [Observation.from_reading(reading, c) for c in sorted(target_cells)]


class SpatialDiscretizer:
    """Unified discretizer that automatically dispatches by geometry kind."""

    def __init__(
        self,
        grid: CoastalGrid | None = None,
        fan_out: FanOutTable | None = None,
    ) -> None:
        self.grid = grid or CoastalGrid(use_cache=True)
        self.fan_out = fan_out or FanOutTable.from_grid(grid=self.grid, resolution_deg=0.1)
        self.point_discretizer = PointDiscretizer(grid=self.grid)
        self.grid_node_discretizer = GridNodeDiscretizer(grid=self.grid, fan_out=self.fan_out)
        self.polygon_discretizer = PolygonDiscretizer(grid=self.grid)

    def discretize(self, reading: RawReading) -> list[Observation]:
        """Dispatch discretization based on reading geometry kind."""
        kind = reading.geometry.kind
        if kind == "point":
            return self.point_discretizer.discretize(reading)
        elif kind == "grid_node":
            return self.grid_node_discretizer.discretize(reading)
        elif kind == "polygon":
            return self.polygon_discretizer.discretize(reading)
        else:
            raise ValueError(f"Unsupported geometry kind: {kind}")
