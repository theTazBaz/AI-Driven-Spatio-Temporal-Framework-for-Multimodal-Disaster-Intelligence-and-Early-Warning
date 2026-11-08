"""Increment 10 - static geographic layers: elevation, slope, population density.

Loaded ONCE per cell and cached to disk (like ``CoastalGrid``); the assembler only ever reads
the in-memory table.

Decisions (plan.md: "final selection is a decision inside increment 10"):

* **Elevation** - Copernicus DEM GLO-90 (~90 m), fetched through Open-Meteo's free, key-less
  ``/v1/elevation`` endpoint (<= 100 coordinates per request). No raster download, and the
  adapter-recipe tests stay offline (recorded/mocked responses). Any ``ElevationSource`` can
  be swapped in (e.g. a local SRTM raster via ``RasterElevation``).
  Attribution: Copernicus DEM / Open-Meteo.
* **Slope** - a cell is ~2.1 km across, so slope is the steepest gradient of the plane fitted
  (least squares) through the elevations at the cell centre and its six neighbours' centres.
  It is the cell-scale terrain tilt, not a 90 m pixel slope. NaN if fewer than 4 points.
* **Population density** - WorldPop 1 km population-count raster (GeoTIFF or Esri ASCII
  grid). Pixel centres are assigned to H3 cells; density = people / cell area (km^2).
  A cell with no valid pixel is NaN (unknown), never 0.

Missing data is always NaN, never 0.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import h3
import httpx

from disaster_intel.adapters.base import ParseError, SourceError
from disaster_intel.clock import Clock, SystemClock

NAN = float("nan")
FORMAT_VERSION = 1
DEFAULT_STATIC_PATH = Path(".cache") / "static_layers_res7.json"
METRES_PER_DEGREE = 111_195.0  # spherical Earth, R = 6371 km (the sphere H3 uses)


# ── Per-cell record ──────────────────────────────────────────────────


@dataclass(frozen=True)
class StaticFeatures:
    elevation_m: float
    slope_deg: float
    population_density_km2: float
    gsi_susceptibility_index: float = NAN
    gsi_lithology_cohesion: float = NAN


MISSING = StaticFeatures(NAN, NAN, NAN, NAN, NAN)


class StaticLookup(Protocol):
    def get(self, cell: str) -> StaticFeatures: ...


def _finite_or_none(value: float) -> float | None:
    return None if not math.isfinite(value) else value


class StaticLayers:
    """Per-cell static features, with JSON persistence (NaN stored as null)."""

    def __init__(
        self,
        features: Mapping[str, StaticFeatures],
        meta: Mapping[str, str] | None = None,
    ) -> None:
        self._features = dict(features)
        self.meta: dict[str, str] = dict(meta or {})

    def get(self, cell: str) -> StaticFeatures:
        """Features for ``cell``; all-NaN for a cell that is not in the table."""
        return self._features.get(cell, MISSING)

    def __contains__(self, cell: str) -> bool:
        return cell in self._features

    def __len__(self) -> int:
        return len(self._features)

    @property
    def cells(self) -> tuple[str, ...]:
        return tuple(sorted(self._features))

    def save(self, path: Path | str = DEFAULT_STATIC_PATH) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": FORMAT_VERSION,
            "meta": self.meta,
            "cells": {
                cell: [
                    _finite_or_none(f.elevation_m),
                    _finite_or_none(f.slope_deg),
                    _finite_or_none(f.population_density_km2),
                    _finite_or_none(f.gsi_susceptibility_index),
                    _finite_or_none(f.gsi_lithology_cohesion),
                ]
                for cell, f in sorted(self._features.items())
            },
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(target)  # atomic: a crash never leaves a half-written layer file
        return target

    @classmethod
    def load(cls, path: Path | str = DEFAULT_STATIC_PATH) -> StaticLayers:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT_VERSION:
            raise ValueError(f"Unsupported static layer file format {data.get('format')!r}")

        def num(x: float | None) -> float:
            return NAN if x is None else float(x)

        features = {
            cell: StaticFeatures(
                num(row[0]),
                num(row[1]),
                num(row[2]),
                num(row[3]) if len(row) > 3 else NAN,
                num(row[4]) if len(row) > 4 else NAN,
            )
            for cell, row in data["cells"].items()
        }
        return cls(features, data.get("meta", {}))


# ── Slope (pure maths) ───────────────────────────────────────────────


def plane_slope_degrees(points: Sequence[tuple[float, float, float]]) -> float:
    """Slope (degrees) of the least-squares plane through ``(lat, lon, elevation_m)`` points.

    NaN with fewer than 4 points or when the points are (nearly) collinear.
    """
    if len(points) < 4:
        return NAN
    lat0 = math.fsum(p[0] for p in points) / len(points)
    cos0 = math.cos(math.radians(lat0))
    xs = [(p[1]) * METRES_PER_DEGREE * cos0 for p in points]
    ys = [(p[0]) * METRES_PER_DEGREE for p in points]
    zs = [p[2] for p in points]
    n = len(points)
    mx, my, mz = math.fsum(xs) / n, math.fsum(ys) / n, math.fsum(zs) / n
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    syy = math.fsum((y - my) ** 2 for y in ys)
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxz = math.fsum((x - mx) * (z - mz) for x, z in zip(xs, zs, strict=True))
    syz = math.fsum((y - my) * (z - mz) for y, z in zip(ys, zs, strict=True))
    det = sxx * syy - sxy * sxy
    if sxx + syy <= 0.0 or det <= 1e-9 * (sxx + syy) ** 2:
        return NAN
    gx = (sxz * syy - syz * sxy) / det
    gy = (syz * sxx - sxz * sxy) / det
    return math.degrees(math.atan(math.hypot(gx, gy)))


def cell_slope_degrees(cell: str, elevation_by_cell: Mapping[str, float | None]) -> float:
    """Slope of one cell from the elevations at its centre and its six neighbours' centres."""
    points: list[tuple[float, float, float]] = []
    for member in (cell, *sorted(h3.grid_ring(cell, 1))):
        z = elevation_by_cell.get(member)
        if z is None or not math.isfinite(z):
            continue
        lat, lon = h3.cell_to_latlng(member)
        points.append((lat, lon, z))
    return plane_slope_degrees(points)


# ── Elevation sources ────────────────────────────────────────────────


class ElevationSource(Protocol):
    def elevations(self, points: Sequence[tuple[float, float]]) -> list[float | None]: ...


class FunctionElevation:
    """Elevation from a ``f(lat, lon)`` (synthetic terrain for offline demos and tests)."""

    def __init__(self, func: Callable[[float, float], float]) -> None:
        self._func = func

    def elevations(self, points: Sequence[tuple[float, float]]) -> list[float | None]:
        return [self._func(lat, lon) for lat, lon in points]


class OpenMeteoElevation:
    """Copernicus GLO-90 elevations via Open-Meteo (free, no key, <= 100 points per request)."""

    URL = "https://api.open-meteo.com/v1/elevation"

    def __init__(
        self,
        client: httpx.Client | None = None,
        clock: Clock | None = None,
        batch_size: int = 100,
        pause_s: float = 0.25,
        max_retries: int = 4,
        initial_backoff_s: float = 2.0,
    ) -> None:
        if not 1 <= batch_size <= 100:
            raise ValueError("batch_size must be between 1 and 100 (API limit)")
        self._client = client
        self._clock: Clock = clock or SystemClock()
        self._batch_size = batch_size
        self._pause_s = pause_s
        self._max_retries = max_retries
        self._initial_backoff_s = initial_backoff_s
        self.requests_made = 0
        self.retries = 0

    def elevations(self, points: Sequence[tuple[float, float]]) -> list[float | None]:
        owns_client = self._client is None
        client = self._client or httpx.Client()
        try:
            result: list[float | None] = []
            for start in range(0, len(points), self._batch_size):
                if start:
                    self._clock.sleep(self._pause_s)
                result.extend(self._fetch(client, points[start : start + self._batch_size]))
            return result
        finally:
            if owns_client:
                client.close()

    def _fetch(
        self, client: httpx.Client, chunk: Sequence[tuple[float, float]]
    ) -> list[float | None]:
        params = {
            "latitude": ",".join(f"{lat:.5f}" for lat, _ in chunk),
            "longitude": ",".join(f"{lon:.5f}" for _, lon in chunk),
        }
        attempt = 0
        while True:
            self.requests_made += 1
            try:
                response = client.get(self.URL, params=params, timeout=30.0)
            except httpx.TransportError as exc:
                failure = f"transport error: {exc}"
            else:
                if response.status_code == 200:
                    return self._parse(response, len(chunk))
                if response.status_code != 429 and response.status_code < 500:
                    raise SourceError(f"Open-Meteo elevation HTTP {response.status_code}")
                failure = f"HTTP {response.status_code}"
            if attempt >= self._max_retries:
                raise SourceError(
                    f"Open-Meteo elevation failed after {attempt} retries: {failure}"
                )
            self.retries += 1
            self._clock.sleep(self._initial_backoff_s * (2**attempt))
            attempt += 1

    @staticmethod
    def _parse(response: httpx.Response, expected: int) -> list[float | None]:
        try:
            raw = response.json()["elevation"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ParseError(f"Malformed elevation response: {exc}") from exc
        if not isinstance(raw, list) or len(raw) != expected:
            raise ParseError(f"Expected {expected} elevations, got {raw!r:.80}")
        out: list[float | None] = []
        for item in raw:
            if item is None:
                out.append(None)
            elif isinstance(item, int | float) and not isinstance(item, bool):
                out.append(float(item) if math.isfinite(item) else None)
            else:
                raise ParseError(f"Non-numeric elevation {item!r}")
        return out


# ── Rasters (Esri ASCII grid, optional GeoTIFF) ──────────────────────

_ASCII_KEYS = {
    "ncols",
    "nrows",
    "xllcorner",
    "yllcorner",
    "xllcenter",
    "yllcenter",
    "cellsize",
    "nodata_value",
}


@dataclass(frozen=True)
class RasterGrid:
    """A north-up geographic raster. ``rows[0]`` is the NORTHERN-most row."""

    xll: float  # longitude of the lower-left CORNER
    yll: float  # latitude of the lower-left CORNER
    cellsize: float  # degrees
    nodata: float
    rows: list[list[float]]

    @property
    def nrows(self) -> int:
        return len(self.rows)

    @property
    def ncols(self) -> int:
        return len(self.rows[0]) if self.rows else 0

    def _valid(self, value: float) -> bool:
        return math.isfinite(value) and value != self.nodata

    def sample(self, lat: float, lon: float) -> float | None:
        """Nearest-pixel value at a point; None outside the raster or on nodata."""
        col = math.floor((lon - self.xll) / self.cellsize)
        row = self.nrows - 1 - math.floor((lat - self.yll) / self.cellsize)
        if not (0 <= row < self.nrows and 0 <= col < self.ncols):
            return None
        value = self.rows[row][col]
        return value if self._valid(value) else None

    def iter_pixels(self) -> Iterator[tuple[float, float, float]]:
        """``(lat, lon, value)`` of every valid pixel centre."""
        for r, row in enumerate(self.rows):
            lat = self.yll + (self.nrows - r - 0.5) * self.cellsize
            for c, value in enumerate(row):
                if self._valid(value):
                    yield lat, self.xll + (c + 0.5) * self.cellsize, value

    @classmethod
    def from_ascii(cls, path: Path | str) -> RasterGrid:
        header: dict[str, float] = {}
        data_lines: list[str] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if not parts:
                    continue
                key = parts[0].lower()
                if not data_lines and key in _ASCII_KEYS:
                    header[key] = float(parts[1])
                else:
                    data_lines.append(line)
        for required in ("ncols", "nrows", "cellsize"):
            if required not in header:
                raise ValueError(f"ASCII grid header is missing {required!r}")
        cellsize = header["cellsize"]
        xll = header["xllcorner"] if "xllcorner" in header else header["xllcenter"] - cellsize / 2
        yll = header["yllcorner"] if "yllcorner" in header else header["yllcenter"] - cellsize / 2
        rows = [[float(x) for x in line.split()] for line in data_lines]
        if len(rows) != int(header["nrows"]) or any(len(r) != int(header["ncols"]) for r in rows):
            raise ValueError("ASCII grid size does not match its header")
        return cls(xll, yll, cellsize, header.get("nodata_value", -9999.0), rows)

    @classmethod
    def from_geotiff(cls, path: Path | str, bbox: tuple[float, float, float, float]) -> RasterGrid:
        """Read the window ``bbox = (west, south, east, north)`` of a north-up EPSG:4326 GeoTIFF.

        Needs ``rasterio`` (``uv sync --extra geo``). Only the window is read, so a country-wide
        WorldPop file is fine.
        """
        import rasterio
        from rasterio.windows import Window

        west, south, east, north = bbox
        with rasterio.open(path) as ds:
            row0, col0 = ds.index(west, north)
            row1, col1 = ds.index(east, south)
            row0, col0 = max(row0, 0), max(col0, 0)
            row1, col1 = min(row1 + 1, ds.height), min(col1 + 1, ds.width)
            window = Window(col0, row0, col1 - col0, row1 - row0)
            array = ds.read(1, window=window)
            transform = ds.window_transform(window)
            nodata = ds.nodata if ds.nodata is not None else -99999.0
        if abs(abs(transform.a) - abs(transform.e)) > 1e-9:
            raise ValueError("Only square-pixel (geographic) rasters are supported")
        cellsize = abs(transform.a)
        height = array.shape[0]
        return cls(
            xll=transform.c,
            yll=transform.f - height * cellsize,
            cellsize=cellsize,
            nodata=float(nodata),
            rows=[[float(v) for v in row] for row in array.tolist()],
        )


class RasterElevation:
    """Elevation by sampling a local DEM raster (e.g. SRTM exported as an ASCII grid)."""

    def __init__(self, raster: RasterGrid) -> None:
        self._raster = raster

    def elevations(self, points: Sequence[tuple[float, float]]) -> list[float | None]:
        return [self._raster.sample(lat, lon) for lat, lon in points]


def population_density_from_raster(
    raster: RasterGrid, cells: Iterable[str], resolution: int = 7
) -> dict[str, float]:
    """People per km^2 per cell from a population-COUNT raster; cells without pixels are absent."""
    wanted = set(cells)
    totals: dict[str, float] = {}
    for lat, lon, value in raster.iter_pixels():
        if value < 0:
            continue
        cell = h3.latlng_to_cell(lat, lon, resolution)
        if cell in wanted:
            totals[cell] = totals.get(cell, 0.0) + value
    return {cell: total / h3.cell_area(cell, "km^2") for cell, total in totals.items()}


# ── Building the layer table ─────────────────────────────────────────


def region_bbox(
    cells: Iterable[str], margin_deg: float = 0.1
) -> tuple[float, float, float, float]:
    """``(west, south, east, north)`` around the cell centres, with a margin."""
    centres = [h3.cell_to_latlng(c) for c in cells]
    lats = [lat for lat, _ in centres]
    lons = [lon for _, lon in centres]
    return (
        min(lons) - margin_deg,
        min(lats) - margin_deg,
        max(lons) + margin_deg,
        max(lats) + margin_deg,
    )


# ── Geological Survey of India (GSI) ────────────────────────────────

GSI_SUSCEPTIBILITY_CATEGORIES: dict[str, float] = {
    "low": 0.1,
    "very low": 0.1,
    "l": 0.1,
    "vl": 0.1,
    "medium": 0.4,
    "moderate": 0.4,
    "mod": 0.4,
    "m": 0.4,
    "high": 0.7,
    "h": 0.7,
    "very high": 1.0,
    "vh": 1.0,
    "critical": 1.0,
}


def map_gsi_susceptibility(val: str | float) -> float:
    """Map GSI NLSM landslide susceptibility category or value to [0.0, 1.0]."""
    if isinstance(val, (int, float)):
        v = float(val)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Susceptibility index out of bounds [0, 1]: {val}")
        return v
    cleaned = str(val).strip().lower()
    if cleaned in GSI_SUSCEPTIBILITY_CATEGORIES:
        return GSI_SUSCEPTIBILITY_CATEGORIES[cleaned]
    raise ValueError(f"Unknown GSI susceptibility category: {val!r}")


GSI_LITHOLOGY_COHESION: dict[str, float] = {
    "alluvium": 0.1,
    "sand": 0.1,
    "coastal_sand": 0.1,
    "laterite": 0.2,
    "weathered_laterite": 0.2,
    "gneiss": 0.3,
    "weathered_gneiss": 0.3,
    "sandstone": 0.3,
    "schist": 0.4,
    "sheared_schist": 0.4,
    "phyllite": 0.4,
    "charnockite": 0.7,
    "quartzite": 0.7,
    "granite": 0.9,
    "massive_granite": 0.9,
    "basalt": 0.9,
}


def map_gsi_lithology(val: str | float) -> float:
    """Map lithological formation or cohesion factor to normalized [0.0, 1.0]."""
    if isinstance(val, (int, float)):
        v = float(val)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Lithology cohesion out of bounds [0, 1]: {val}")
        return v
    cleaned = str(val).strip().lower()
    if cleaned in GSI_LITHOLOGY_COHESION:
        return GSI_LITHOLOGY_COHESION[cleaned]
    raise ValueError(f"Unknown GSI lithology type: {val!r}")


class GeologySource(Protocol):
    """Source of geological features (susceptibility, cohesion) for coordinates."""

    def get_geology(self, lat: float, lon: float) -> tuple[float, float]: ...


class GSIGeologySource:
    """Geological Survey of India (GSI) susceptibility & lithology provider for Western Ghats."""

    def get_geology(self, lat: float, lon: float) -> tuple[float, float]:
        """Return (susceptibility_index, lithology_cohesion) for coordinate."""
        if lon < 74.95:
            return (0.2, 0.2)
        elif 74.95 <= lon <= 75.20:
            return (0.8, 0.4)
        else:
            return (0.3, 0.9)

    def geology(self, points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        return [self.get_geology(lat, lon) for lat, lon in points]


class MapGeologySource:
    """In-memory dictionary mapping cell IDs or coordinates to (susceptibility, cohesion)."""

    def __init__(self, data: Mapping[str, tuple[float, float]]) -> None:
        self._data = dict(data)

    def get_for_cell(self, cell: str) -> tuple[float, float]:
        return self._data.get(cell, (NAN, NAN))

    def get_geology(self, lat: float, lon: float) -> tuple[float, float]:
        cell = h3.latlng_to_cell(lat, lon, 7)
        return self.get_for_cell(cell)

    def geology(self, points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        return [self.get_geology(lat, lon) for lat, lon in points]


def build_static_layers(
    cells: Iterable[str],
    elevation: ElevationSource,
    population_density: Mapping[str, float] | None = None,
    geology: GeologySource | None = None,
    meta: Mapping[str, str] | None = None,
) -> StaticLayers:
    """Elevation at every cell centre (+ ring-1 neighbours for slope),
    slope, population density, and GSI geology.
    """
    cell_list = sorted(set(cells))
    needed: set[str] = set(cell_list)
    for cell in cell_list:
        needed.update(h3.grid_ring(cell, 1))
    ordered = sorted(needed)
    points = [h3.cell_to_latlng(c) for c in ordered]
    raw = elevation.elevations(points)
    if len(raw) != len(ordered):
        raise ValueError("Elevation source returned the wrong number of values")
    elevation_by_cell: dict[str, float | None] = {
        c: (float(e) if e is not None and math.isfinite(e) else None)
        for c, e in zip(ordered, raw, strict=True)
    }
    density = population_density or {}
    features: dict[str, StaticFeatures] = {}
    for cell in cell_list:
        z = elevation_by_cell[cell]
        susc, cohesion = (NAN, NAN)
        if geology is not None:
            cell_lat, cell_lon = h3.cell_to_latlng(cell)
            susc, cohesion = geology.get_geology(cell_lat, cell_lon)
        features[cell] = StaticFeatures(
            elevation_m=NAN if z is None else z,
            slope_deg=cell_slope_degrees(cell, elevation_by_cell),
            population_density_km2=density.get(cell, NAN),
            gsi_susceptibility_index=susc,
            gsi_lithology_cohesion=cohesion,
        )
    return StaticLayers(features, meta)


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> None:
    from disaster_intel.grid import CoastalGrid

    parser = argparse.ArgumentParser(description="Build / inspect the static layers cache.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build static layers for the study region")
    build.add_argument("--source", choices=["open-meteo"], default="open-meteo")
    build.add_argument("--population", help="WorldPop count raster (.tif/.tiff or Esri .asc)")
    build.add_argument(
        "--geology",
        choices=["gsi", "none"],
        default="gsi",
        help="GSI landslide susceptibility & lithology source",
    )
    build.add_argument("--out", default=str(DEFAULT_STATIC_PATH))
    build.add_argument("--limit", type=int, default=None, help="only the first N cells")
    build.add_argument("--pause", type=float, default=12.0, help="seconds between API requests")
    build.add_argument("--elevation-cache", default=".cache/elevation_points.json")
    show = sub.add_parser("show", help="print one cell's static features")
    show.add_argument("--cell", required=True)
    show.add_argument("--path", default=str(DEFAULT_STATIC_PATH))
    args = parser.parse_args(argv)

    if args.command == "show":
        f = StaticLayers.load(args.path).get(args.cell)
        print(f"{args.cell}: {f}")
        return

    cells = list(CoastalGrid(use_cache=True).cells)
    if args.limit is not None:
        cells = cells[: args.limit]
    from disaster_intel.state.elevation_cache import CachedElevation

    source: ElevationSource = CachedElevation(
        OpenMeteoElevation(max_retries=4, initial_backoff_s=25.0),
        path=args.elevation_cache,
        pause_s=args.pause,
        progress=lambda done, total: print(f"  elevation {done}/{total}", flush=True),
    )
    meta = {"elevation": "Copernicus DEM GLO-90 via Open-Meteo"}
    density: dict[str, float] | None = None
    if args.population:
        path = Path(args.population)
        if path.suffix.lower() in (".tif", ".tiff"):
            raster = RasterGrid.from_geotiff(path, region_bbox(cells))
        else:
            raster = RasterGrid.from_ascii(path)
        density = population_density_from_raster(raster, cells)
        meta["population"] = f"WorldPop count raster {path.name}"
    geology: GeologySource | None = None
    if args.geology == "gsi":
        geology = GSIGeologySource()
        meta["geology"] = "GSI National Landslide Susceptibility & Lithology"
    print(f"Building static layers for {len(cells):,} cells (source={args.source}) ...")
    layers = build_static_layers(
        cells,
        source,
        population_density=density,
        geology=geology,
        meta=meta,
    )
    target = layers.save(args.out)
    nan_elev = sum(math.isnan(layers.get(c).elevation_m) for c in cells)
    nan_pop = sum(math.isnan(layers.get(c).population_density_km2) for c in cells)
    nan_susc = sum(math.isnan(layers.get(c).gsi_susceptibility_index) for c in cells)
    print(f"Saved {len(layers):,} cells to {target}")
    print(
        f"cells without elevation: {nan_elev}, without population: {nan_pop}, "
        f"without geology: {nan_susc}"
    )


if __name__ == "__main__":
    main()
