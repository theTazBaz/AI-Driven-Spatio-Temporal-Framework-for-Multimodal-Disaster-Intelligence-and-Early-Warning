"""Increment 10 unit tests - static layers (elevation, slope, population density).

Test list (plan.md, Increment 10), static half:
 1. A synthetic sloped plane gives the known slope (pure maths and through real H3 cells).
 2. Missing elevation -> NaN (never 0); too few points or a degenerate fit -> NaN.
 3. Layers are built ONCE per cell and survive a save/load round trip (NaN <-> null).
 4. Esri ASCII rasters parse and sample correctly; population density = people / cell area.
 5. The Open-Meteo elevation source batches (<= 100 points), retries 429/5xx on a fake clock,
    and maps bad payloads to ParseError / SourceError (adapter recipe).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h3
import httpx
import pytest

from disaster_intel.adapters import ParseError, SourceError
from disaster_intel.clock import FakeClock
from disaster_intel.state import (
    FunctionElevation,
    OpenMeteoElevation,
    RasterElevation,
    RasterGrid,
    StaticFeatures,
    StaticLayers,
    build_static_layers,
    cell_slope_degrees,
    plane_slope_degrees,
    population_density_from_raster,
    synthetic_ghats_elevation,
)
from disaster_intel.state.static_layers import METRES_PER_DEGREE

pytestmark = pytest.mark.unit

CENTRE = h3.latlng_to_cell(13.0, 75.0, 7)
PLANE_SLOPE = math.degrees(math.atan(0.1))  # a 10 % grade


def east_metres(lat: float, lon: float, lat0: float, lon0: float) -> float:
    return (lon - lon0) * METRES_PER_DEGREE * math.cos(math.radians(lat0))


def cell_east_metres(cell: str, lat0: float, lon0: float) -> float:
    lat, lon = h3.cell_to_latlng(cell)
    return east_metres(lat, lon, lat0, lon0)


# ── 1. Slope ─────────────────────────────────────────────────────────


def test_plane_slope_recovers_a_known_grade_in_any_direction() -> None:
    lats, lons = (12.99, 13.0, 13.01), (74.99, 75.0, 75.01)
    east = [(la, lo, 0.1 * east_metres(la, lo, 13.0, 75.0)) for la in lats for lo in lons]
    assert plane_slope_degrees(east) == pytest.approx(PLANE_SLOPE, abs=0.01)

    north = [(la, lo, 0.1 * (la - 13.0) * METRES_PER_DEGREE) for la in lats for lo in lons]
    assert plane_slope_degrees(north) == pytest.approx(PLANE_SLOPE, abs=0.01)

    both = [
        (la, lo, 0.2 * (la - 13.0) * METRES_PER_DEGREE + 0.1 * east_metres(la, lo, 13.0, 75.0))
        for la in lats
        for lo in lons
    ]
    assert plane_slope_degrees(both) == pytest.approx(
        math.degrees(math.atan(math.hypot(0.1, 0.2))), abs=0.01
    )


def test_flat_ground_has_zero_slope() -> None:
    flat = [(la, lo, 250.0) for la in (12.99, 13.0, 13.01) for lo in (74.99, 75.0, 75.01)]
    assert plane_slope_degrees(flat) == pytest.approx(0.0, abs=1e-9)


def test_too_few_or_collinear_points_give_nan_not_zero() -> None:
    assert math.isnan(plane_slope_degrees([(13.0, 75.0, 1.0)] * 3))
    line = [(13.0, 75.0 + i * 0.01, 10.0 * i) for i in range(6)]
    assert math.isnan(plane_slope_degrees(line))


def test_cell_slope_through_real_h3_neighbours_matches_the_plane() -> None:
    lat0, lon0 = h3.cell_to_latlng(CENTRE)
    ring = [CENTRE, *h3.grid_ring(CENTRE, 1)]
    elevations: dict[str, float | None] = {c: 0.1 * cell_east_metres(c, lat0, lon0) for c in ring}
    assert cell_slope_degrees(CENTRE, elevations) == pytest.approx(PLANE_SLOPE, abs=0.02)


def test_cell_slope_tolerates_one_missing_neighbour_but_not_four() -> None:
    lat0, lon0 = h3.cell_to_latlng(CENTRE)
    ring = sorted(h3.grid_ring(CENTRE, 1))
    elevations: dict[str, float | None] = {
        c: 0.1 * cell_east_metres(c, lat0, lon0) for c in [CENTRE, *ring]
    }
    elevations[ring[0]] = None
    assert cell_slope_degrees(CENTRE, elevations) == pytest.approx(PLANE_SLOPE, abs=0.02)
    for c in ring[:4]:
        elevations[c] = None
    assert math.isnan(cell_slope_degrees(CENTRE, elevations))  # only 3 points left


# ── 2./3. Building and persisting the table ──────────────────────────


def patch_of_cells(n: int = 12) -> list[str]:
    cells = sorted(h3.grid_disk(CENTRE, 2))
    return cells[:n]


def test_build_gives_elevation_slope_and_nan_population_by_default() -> None:
    lat0, lon0 = h3.cell_to_latlng(CENTRE)
    source = FunctionElevation(lambda la, lo: 0.1 * east_metres(la, lo, lat0, lon0))
    layers = build_static_layers(patch_of_cells(), source)
    for cell in patch_of_cells():
        f = layers.get(cell)
        assert f.slope_deg == pytest.approx(PLANE_SLOPE, abs=0.05)
        assert f.elevation_m == pytest.approx(0.1 * cell_east_metres(cell, lat0, lon0), abs=1e-6)
        assert math.isnan(f.population_density_km2)  # unknown, NOT zero


def test_build_requests_each_needed_point_once_including_ring_neighbours() -> None:
    calls: list[int] = []

    class Counting:
        def elevations(self, points: Sequence[tuple[float, float]]) -> list[float | None]:
            calls.append(len(points))
            return [100.0] * len(points)

    cells = patch_of_cells()
    build_static_layers(cells, Counting())
    needed = set(cells)
    for c in cells:
        needed.update(h3.grid_ring(c, 1))
    assert calls == [len(needed)]


def test_missing_elevation_is_nan_and_population_is_used() -> None:
    cells = patch_of_cells(3)

    class Holes:
        def elevations(self, points: Sequence[tuple[float, float]]) -> list[float | None]:
            return [None] * len(points)

    layers = build_static_layers(cells, Holes(), {cells[0]: 123.5})
    assert math.isnan(layers.get(cells[0]).elevation_m)
    assert math.isnan(layers.get(cells[0]).slope_deg)
    assert layers.get(cells[0]).population_density_km2 == 123.5
    assert math.isnan(layers.get(cells[1]).population_density_km2)


def test_unknown_cell_is_all_nan() -> None:
    f = StaticLayers({}).get(CENTRE)
    assert all(math.isnan(v) for v in (f.elevation_m, f.slope_deg, f.population_density_km2))


def test_save_load_round_trip_keeps_nan_as_null(tmp_path: Path) -> None:
    layers = StaticLayers(
        {
            CENTRE: StaticFeatures(12.5, 3.25, float("nan")),
            "872a10729ffffff": StaticFeatures(float("nan"), float("nan"), 80.0),
        },
        meta={"elevation": "test"},
    )
    path = layers.save(tmp_path / "sub" / "static.json")
    assert "NaN" not in path.read_text(encoding="utf-8")
    assert not list(path.parent.glob("*.tmp"))  # atomic write left no temp file
    loaded = StaticLayers.load(path)
    assert loaded.meta == {"elevation": "test"}
    assert loaded.cells == layers.cells
    a = loaded.get(CENTRE)
    assert (a.elevation_m, a.slope_deg) == (12.5, 3.25)
    assert math.isnan(a.population_density_km2)
    assert math.isnan(loaded.get("872a10729ffffff").elevation_m)


def test_load_rejects_an_unknown_file_format(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"format": 99, "cells": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        StaticLayers.load(path)


def test_synthetic_terrain_rises_inland() -> None:
    assert synthetic_ghats_elevation(13.0, 74.4) == 0.0
    assert synthetic_ghats_elevation(13.0, 75.8) == pytest.approx(1200.0)
    assert synthetic_ghats_elevation(13.0, 75.1) < synthetic_ghats_elevation(13.0, 75.4)


# ── 4. Rasters and population density ────────────────────────────────

ASCII = """ncols 4
nrows 3
xllcorner 74.0
yllcorner 12.0
cellsize 0.5
NODATA_value -9999
1 2 3 4
5 -9999 7 8
9 10 11 12
"""


def test_ascii_grid_parses_with_north_row_first_and_samples_by_position(tmp_path: Path) -> None:
    path = tmp_path / "g.asc"
    path.write_text(ASCII, encoding="utf-8")
    grid = RasterGrid.from_ascii(path)
    assert (grid.nrows, grid.ncols, grid.cellsize) == (3, 4, 0.5)
    assert grid.sample(13.25, 74.25) == 1.0  # top-left pixel
    assert grid.sample(12.25, 75.75) == 12.0  # bottom-right pixel
    assert grid.sample(12.75, 74.75) is None  # the nodata pixel
    assert grid.sample(11.0, 74.25) is None  # outside the raster
    assert len(list(grid.iter_pixels())) == 11
    assert RasterElevation(grid).elevations([(13.25, 74.25), (0.0, 0.0)]) == [1.0, None]


def test_ascii_grid_with_xllcenter_and_wrong_size_is_handled(tmp_path: Path) -> None:
    centre = tmp_path / "c.asc"
    centre.write_text(
        "ncols 2\nnrows 1\nxllcenter 74.25\nyllcenter 12.25\ncellsize 0.5\n7 8\n", encoding="utf-8"
    )
    grid = RasterGrid.from_ascii(centre)
    assert (grid.xll, grid.yll) == (74.0, 12.0)
    assert grid.nodata == -9999.0  # default when the header omits it
    bad = tmp_path / "b.asc"
    bad.write_text("ncols 3\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 1\n1 2\n", "utf-8")
    with pytest.raises(ValueError):
        RasterGrid.from_ascii(bad)


def test_population_density_is_people_per_km2_of_the_cell() -> None:
    lat0, lon0 = h3.cell_to_latlng(CENTRE)
    size, half = 0.001, 0.05
    n = int(2 * half / size)
    yll, xll = lat0 - half, lon0 - half
    people_per_pixel = 2.0
    raster = RasterGrid(xll, yll, size, -99999.0, [[people_per_pixel] * n for _ in range(n)])

    density = population_density_from_raster(raster, [CENTRE])
    inside = sum(
        1
        for r in range(n)
        for c in range(n)
        if h3.latlng_to_cell(yll + (n - r - 0.5) * size, xll + (c + 0.5) * size, 7) == CENTRE
    )
    area = h3.cell_area(CENTRE, "km^2")
    assert density[CENTRE] == pytest.approx(inside * people_per_pixel / area)
    # sanity on the physics: 2 people per ~0.012 km^2 pixel is ~166 people/km^2
    assert density[CENTRE] == pytest.approx(166.0, rel=0.06)


def test_cells_without_pixels_are_absent_so_the_table_makes_them_nan() -> None:
    raster = RasterGrid(0.0, 0.0, 1.0, -9999.0, [[1.0]])
    assert population_density_from_raster(raster, [CENTRE]) == {}


def test_negative_population_pixels_are_ignored() -> None:
    lat0, lon0 = h3.cell_to_latlng(CENTRE)
    raster = RasterGrid(lon0 - 0.005, lat0 - 0.005, 0.01, -9999.0, [[-5.0]])
    assert population_density_from_raster(raster, [CENTRE]) == {}


# ── 5. Open-Meteo elevation source (mock transport: no sockets) ──────


def points(n: int) -> list[tuple[float, float]]:
    return [(12.0 + i * 0.01, 75.0) for i in range(n)]


def make_source(
    transport: httpx.MockTransport, clock: FakeClock | None = None, **kwargs: Any
) -> OpenMeteoElevation:
    client = httpx.Client(transport=transport)
    return OpenMeteoElevation(client=client, clock=clock or FakeClock(), **kwargs)


def test_open_meteo_elevation_batches_100_points_and_pauses_between_requests() -> None:
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        lats = request.url.params["latitude"].split(",")
        seen.append(len(lats))
        assert len(request.url.params["longitude"].split(",")) == len(lats)
        return httpx.Response(200, json={"elevation": [float(i) for i in range(len(lats))]})

    clock = FakeClock(datetime(2024, 1, 1, tzinfo=UTC))
    source = make_source(httpx.MockTransport(handler), clock, pause_s=0.5)
    result = source.elevations(points(250))
    assert seen == [100, 100, 50]
    assert len(result) == 250
    assert clock.sleep_log == [0.5, 0.5]  # between batches only


def test_open_meteo_elevation_formats_coordinates_to_5_decimals() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.url.params)
        return httpx.Response(200, json={"elevation": [10.0, 20.0]})

    make_source(httpx.MockTransport(handler)).elevations([(12.5, 75.25), (13.123456789, 74.0)])
    assert captured["latitude"] == "12.50000,13.12346"
    assert captured["longitude"] == "75.25000,74.00000"


def test_open_meteo_elevation_retries_429_with_backoff_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"elevation": [5.0]})

    clock = FakeClock()
    source = make_source(httpx.MockTransport(handler), clock, initial_backoff_s=2.0)
    assert source.elevations(points(1)) == [5.0]
    assert clock.sleep_log == [2.0, 4.0]
    assert source.retries == 2


def test_open_meteo_elevation_gives_up_after_max_retries_as_a_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    source = make_source(httpx.MockTransport(handler), max_retries=2)
    with pytest.raises(SourceError):
        source.elevations(points(1))
    assert source.requests_made == 3


def test_open_meteo_elevation_does_not_retry_a_400() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": True, "reason": "bad"})

    source = make_source(httpx.MockTransport(handler))
    with pytest.raises(SourceError):
        source.elevations(points(1))
    assert source.requests_made == 1


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        '{"nope": []}',
        '{"elevation": [1.0]}',  # wrong length for 2 points
        '{"elevation": ["x", 2.0]}',
        '{"elevation": "abc"}',
    ],
)
def test_open_meteo_elevation_malformed_payloads_raise_parse_error(body: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    with pytest.raises(ParseError):
        make_source(httpx.MockTransport(handler)).elevations(points(2))


def test_open_meteo_elevation_keeps_null_as_none_never_zero() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"elevation": [None, 7.0]})

    assert make_source(httpx.MockTransport(handler)).elevations(points(2)) == [None, 7.0]


def test_batch_size_above_the_api_limit_is_refused() -> None:
    with pytest.raises(ValueError):
        OpenMeteoElevation(batch_size=101)
