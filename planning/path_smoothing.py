"""Post-process a raw A* path: dedupe, collapse collinear runs, shortcut
via line-of-sight, and optionally smooth -- while guaranteeing the
result stays collision-free.

Never trusts smoothing blindly: ``smooth_path`` verifies its output
against the planning map and falls back to the (already collision-free)
input if smoothing would cut through an obstacle.

Phase 15 -- clearance-aware line-of-sight: ``has_line_of_sight`` and
``simplify_line_of_sight`` accept an optional ``clearance_m`` (meters,
default ``0.0`` -- every pre-Phase-15 caller is completely unaffected).
This is an ADDITIONAL path-smoothing margin on top of the planning
grid's own inflation (``mapping.inflation``, unmodified) -- it never
changes what "collision" means, only how much extra room a LOS shortcut
must leave before ``simplify_line_of_sight`` is willing to take it over
the original, already-safe A* waypoints. Motivated by Phase 14's
measurement: LOS shortcuts were found to leave as little as 0.03-0.04m
of clearance from occupied geometry -- enough to be technically
collision-free in the inflated grid, but tight enough that ordinary
localization/discretization variance could put the robot's own cell
back inside the inflation buffer (Phase 12.5/13). Gradient smoothing
(``smooth_path``) is deliberately NOT given this parameter -- its
existing collision-only safety fallback is unchanged, per the approved,
narrowly-scoped Phase 15 design.
"""

from __future__ import annotations

import math

import numpy as np

from common.messages import OccupancyGrid, Path, PathPoint
from mapping.occupancy_grid import OccupancyGridMap
from mapping.ray_casting import bresenham_line


def remove_duplicate_points(path: Path, tolerance: float = 1e-9) -> Path:
    """Drop consecutive points that are (near-)identical to the previous kept point."""
    if not path.points:
        return Path(points=[])

    result = [path.points[0]]
    for point in path.points[1:]:
        last = result[-1]
        if abs(point.x - last.x) > tolerance or abs(point.y - last.y) > tolerance:
            result.append(point)
    return Path(points=result)


def remove_collinear_points(path: Path, tolerance: float = 1e-6) -> Path:
    """Drop interior points that lie on the straight segment between their neighbors.

    Compares each candidate against the *last kept* point (not just the
    previous original point), so an arbitrarily long straight run
    collapses to its two endpoints.
    """
    if len(path.points) <= 2:
        return Path(points=list(path.points))

    result = [path.points[0]]
    for i in range(1, len(path.points) - 1):
        a = result[-1]
        b = path.points[i]
        c = path.points[i + 1]
        cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        if abs(cross) > tolerance:
            result.append(b)
    result.append(path.points[-1])
    return Path(points=result)


def _shift_mask(mask: np.ndarray, d_row: int, d_col: int) -> np.ndarray:
    """Return ``mask`` shifted by (d_row, d_col), zero-filled at the edges.

    Duplicated (not imported) from ``mapping.inflation``'s private helper
    of the same name -- deliberately, to avoid this module importing a
    private (underscore) symbol from another module for one tiny,
    self-contained array op, matching this project's own established
    convention for exactly this situation (see
    ``slam/scan_matching_pipeline.py``'s module docstring).
    """
    rows, cols = mask.shape
    result = np.zeros_like(mask)
    row_lo, row_hi = max(0, d_row), min(rows, rows + d_row)
    col_lo, col_hi = max(0, d_col), min(cols, cols + d_col)
    if row_hi <= row_lo or col_hi <= col_lo:
        return result
    result[row_lo:row_hi, col_lo:col_hi] = mask[
        row_lo - d_row : row_hi - d_row, col_lo - d_col : col_hi - d_col
    ]
    return result


def _clearance_blocked_mask(grid: OccupancyGrid, clearance_m: float) -> np.ndarray:
    """Boolean mask, same shape as ``grid.cells``: True for every OCCUPIED
    cell plus every cell within ``clearance_m`` meters of one (disc
    expansion, same technique as ``mapping.inflation.inflate_occupancy_grid``
    -- but computed here as a local, throwaway array; the actual grid is
    never mutated or replaced). ``clearance_m <= 0.0`` reduces to exactly
    the OCCUPIED mask, i.e. this function is a no-op extension for the
    pre-Phase-15 default.
    """
    occupied_mask = grid.cells == OccupancyGrid.OCCUPIED
    if clearance_m <= 0.0 or not np.any(occupied_mask):
        return occupied_mask

    radius_cells = int(math.ceil(clearance_m / grid.resolution))
    blocked = occupied_mask.copy()
    for d_row in range(-radius_cells, radius_cells + 1):
        for d_col in range(-radius_cells, radius_cells + 1):
            if d_row == 0 and d_col == 0:
                continue
            if math.hypot(d_row * grid.resolution, d_col * grid.resolution) > clearance_m:
                continue
            blocked |= _shift_mask(occupied_mask, d_row, d_col)
    return blocked


def has_line_of_sight(
    grid_map: OccupancyGridMap,
    a: PathPoint,
    b: PathPoint,
    clearance_m: float = 0.0,
    _blocked_mask: np.ndarray | None = None,
) -> bool:
    """True if the straight segment a->b crosses no occupied grid cell,
    and (Phase 15, ``clearance_m > 0``) stays at least ``clearance_m``
    meters from any occupied cell -- an ADDITIONAL path-smoothing margin
    on top of the grid's own inflation, not a change to collision
    checking itself. Default ``clearance_m=0.0`` is exactly this
    function's pre-Phase-15 behavior (every existing caller unaffected).

    ``_blocked_mask`` (internal use only): a precomputed
    ``_clearance_blocked_mask`` result, reused across many calls (see
    ``simplify_line_of_sight``) so the O(radius^2) mask build isn't
    repeated per candidate segment. External callers should not pass it.
    """
    grid = grid_map.grid
    mask = _blocked_mask if _blocked_mask is not None else _clearance_blocked_mask(grid, clearance_m)
    r0, c0 = grid_map.world_to_grid(a.x, a.y)
    r1, c1 = grid_map.world_to_grid(b.x, b.y)
    for row, col in bresenham_line(r0, c0, r1, c1):
        if not grid_map.in_bounds(row, col):
            return False
        if mask[row, col]:
            return False
    return True


def simplify_line_of_sight(path: Path, grid_map: OccupancyGridMap, clearance_m: float = 0.0) -> Path:
    """Greedy "string pulling": from each point, jump to the farthest point
    with a clear line of sight (respecting ``clearance_m``, Phase 15),
    skipping unnecessary intermediate waypoints.

    Collision-free by construction -- every kept segment was explicitly
    verified against ``grid_map``. If no shortcut past the very next
    waypoint satisfies ``clearance_m``, this simply keeps that next
    waypoint (the original, already-safe A* point) unchanged -- the
    clearance margin can only ever make simplification more conservative,
    never cause it to fail or drop a point that was required for safety.
    """
    points = path.points
    if len(points) <= 2:
        return Path(points=list(points))

    blocked_mask = _clearance_blocked_mask(grid_map.grid, clearance_m)
    result = [points[0]]
    i = 0
    last_index = len(points) - 1
    while i < last_index:
        j = last_index
        while j > i + 1 and not has_line_of_sight(grid_map, points[i], points[j], _blocked_mask=blocked_mask):
            j -= 1
        result.append(points[j])
        i = j
    return Path(points=result)


def _path_is_collision_free(path: Path, grid_map: OccupancyGridMap) -> bool:
    if not path.points:
        return True
    for point in path.points:
        row, col = grid_map.world_to_grid(point.x, point.y)
        if not grid_map.in_bounds(row, col) or grid_map.get_cell(row, col) == OccupancyGrid.OCCUPIED:
            return False
    for a, b in zip(path.points, path.points[1:]):
        if not has_line_of_sight(grid_map, a, b):
            return False
    return True


def smooth_path(
    path: Path,
    grid_map: OccupancyGridMap,
    weight_data: float = 0.5,
    weight_smooth: float = 0.2,
    tolerance: float = 1e-6,
    max_iterations: int = 200,
) -> Path:
    """Gradient-descent waypoint smoothing (endpoints fixed), with a safety fallback.

    Each interior point is pulled toward both its original position
    (``weight_data``) and the midpoint of its smoothed neighbors
    (``weight_smooth``), iterated until the per-iteration change drops
    below ``tolerance``. If the smoothed result would cross an obstacle,
    the original (already collision-free) path is returned unchanged --
    smoothing never trades safety for a shorter-looking path.
    """
    points = path.points
    n = len(points)
    if n <= 2:
        return Path(points=list(points))

    original = [[p.x, p.y] for p in points]
    smoothed = [list(coords) for coords in original]

    change = tolerance + 1.0
    iterations = 0
    while change >= tolerance and iterations < max_iterations:
        change = 0.0
        iterations += 1
        for i in range(1, n - 1):
            for dim in (0, 1):
                old_val = smoothed[i][dim]
                smoothed[i][dim] += weight_data * (original[i][dim] - smoothed[i][dim])
                smoothed[i][dim] += weight_smooth * (
                    smoothed[i - 1][dim] + smoothed[i + 1][dim] - 2.0 * smoothed[i][dim]
                )
                change += abs(old_val - smoothed[i][dim])

    smoothed_path = Path(points=[PathPoint(x=x, y=y) for x, y in smoothed])

    if _path_is_collision_free(smoothed_path, grid_map):
        return smoothed_path
    return Path(points=list(points))


def process_path(
    path: Path, grid_map: OccupancyGridMap, simplify: bool = True, smooth: bool = False,
    clearance_m: float = 0.0,
) -> Path:
    """Full post-processing pipeline: dedupe -> collinear reduction -> (simplify) -> (smooth).

    ``clearance_m`` (Phase 15): forwarded to ``simplify_line_of_sight``
    unchanged -- see that function and the module docstring. Default
    ``0.0`` preserves this function's exact pre-Phase-15 behavior.
    """
    result = remove_duplicate_points(path)
    result = remove_collinear_points(result)
    if simplify:
        result = simplify_line_of_sight(result, grid_map, clearance_m=clearance_m)
    if smooth:
        result = smooth_path(result, grid_map)
    return result
