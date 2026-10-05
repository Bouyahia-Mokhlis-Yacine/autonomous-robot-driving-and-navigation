"""Phase 14: path-clearance measurement (diagnostic only).

Measures how close a ``Path`` comes to OCCUPIED geometry in a planning
map, in METERS (not grid-cell counts). Used to distinguish, with actual
numbers, which stage of the existing pipeline -- raw A*, line-of-sight
simplification, or gradient-descent smoothing (see
``planning/path_smoothing.py``, all UNMODIFIED) -- is responsible for a
path coming close to the robot's own inflation buffer.

Nothing here is called by ``planning`` or ``navigation`` at runtime; this
is read-only instrumentation, imported only by diagnostic tooling
(``simulation/diagnostics.py``, tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from common.geometry import distance
from common.messages import OccupancyGrid, Path
from mapping.occupancy_grid import OccupancyGridMap


@dataclass(frozen=True)
class ClearanceMeasurement:
    """Result of sampling a path's distance to the nearest OCCUPIED cell.

    ``min_clearance_m``/``mean_clearance_m`` are in meters.
    ``min_clearance_m == search_radius_m`` means no occupied cell was
    found within ``search_radius_m`` of ANY sampled point -- i.e. "at
    least this far", not an exact value (see ``measure_path_clearance``).
    """

    min_clearance_m: float
    mean_clearance_m: float
    min_clearance_world: tuple[float, float]
    min_clearance_grid: tuple[int, int]
    min_clearance_segment_index: int
    path_length_m: float
    num_samples: int
    search_radius_m: float


def _nearest_occupied_distance_m(
    grid: OccupancyGrid, x: float, y: float, search_radius_m: float
) -> float:
    """Euclidean distance (meters) from world point ``(x, y)`` to the
    nearest OCCUPIED cell's center, searched within a local
    ``search_radius_m`` bounding box (same bounding-box-then-exact
    pattern as ``PersistentWorldModel._points_world_by_state``) --
    never scans the whole grid. Returns ``search_radius_m`` itself (a
    safe lower bound, "at least this far") if no occupied cell is found
    in range, rather than an unbounded/infinite value.
    """
    resolution = grid.resolution
    radius_cells = int(math.ceil(search_radius_m / resolution))
    center_row = int(math.floor((y - grid.origin_y) / resolution))
    center_col = int(math.floor((x - grid.origin_x) / resolution))

    row_lo = max(0, center_row - radius_cells)
    row_hi = min(grid.height, center_row + radius_cells + 1)
    col_lo = max(0, center_col - radius_cells)
    col_hi = min(grid.width, center_col + radius_cells + 1)
    if row_hi <= row_lo or col_hi <= col_lo:
        return search_radius_m

    window = grid.cells[row_lo:row_hi, col_lo:col_hi]
    occ_rows, occ_cols = np.nonzero(window == OccupancyGrid.OCCUPIED)
    if occ_rows.size == 0:
        return search_radius_m

    occ_x = grid.origin_x + (occ_cols.astype(np.float64) + col_lo + 0.5) * resolution
    occ_y = grid.origin_y + (occ_rows.astype(np.float64) + row_lo + 0.5) * resolution
    dists = np.hypot(occ_x - x, occ_y - y)
    return float(np.min(dists))


def measure_path_clearance(
    path: Path,
    grid_map: OccupancyGridMap,
    sample_step_m: float | None = None,
    search_radius_m: float = 2.0,
) -> ClearanceMeasurement:
    """Sample ``path`` at ``sample_step_m`` intervals (default: the
    grid's own resolution) and report the minimum/mean distance to the
    nearest OCCUPIED cell, in meters. Deterministic: pure function of
    ``path``/``grid_map``, no randomness, no live clock.
    """
    grid = grid_map.grid
    step = sample_step_m if sample_step_m is not None else grid.resolution

    if path.is_empty():
        return ClearanceMeasurement(
            min_clearance_m=math.inf, mean_clearance_m=math.inf,
            min_clearance_world=(math.nan, math.nan), min_clearance_grid=(-1, -1),
            min_clearance_segment_index=-1, path_length_m=0.0, num_samples=0,
            search_radius_m=search_radius_m,
        )
    if len(path.points) == 1:
        p = path.points[0]
        d = _nearest_occupied_distance_m(grid, p.x, p.y, search_radius_m)
        row, col = grid_map.world_to_grid(p.x, p.y)
        return ClearanceMeasurement(
            min_clearance_m=d, mean_clearance_m=d, min_clearance_world=(p.x, p.y),
            min_clearance_grid=(row, col), min_clearance_segment_index=0,
            path_length_m=0.0, num_samples=1, search_radius_m=search_radius_m,
        )

    min_clearance = math.inf
    min_world = (math.nan, math.nan)
    min_grid = (-1, -1)
    min_segment = -1
    total = 0.0
    count = 0
    path_length = 0.0

    for seg_idx, (a, b) in enumerate(zip(path.points, path.points[1:])):
        seg_len = distance(a.x, a.y, b.x, b.y)
        path_length += seg_len
        n_samples = max(1, int(math.ceil(seg_len / step))) if seg_len > 0.0 else 0
        start_i = 0 if seg_idx == 0 else 1  # don't re-sample the point shared with the previous segment
        for i in range(start_i, n_samples + 1):
            t = i / n_samples if n_samples > 0 else 0.0
            x = a.x + t * (b.x - a.x)
            y = a.y + t * (b.y - a.y)
            d = _nearest_occupied_distance_m(grid, x, y, search_radius_m)
            total += d
            count += 1
            if d < min_clearance:
                min_clearance = d
                min_world = (x, y)
                min_grid = grid_map.world_to_grid(x, y)
                min_segment = seg_idx

    return ClearanceMeasurement(
        min_clearance_m=min_clearance,
        mean_clearance_m=(total / count if count else math.inf),
        min_clearance_world=min_world,
        min_clearance_grid=min_grid,
        min_clearance_segment_index=min_segment,
        path_length_m=path_length,
        num_samples=count,
        search_radius_m=search_radius_m,
    )
