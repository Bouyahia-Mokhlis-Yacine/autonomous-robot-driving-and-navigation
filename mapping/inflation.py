"""Obstacle inflation: turn a raw occupancy grid into a planning map.

Expands every occupied cell by ``inflation_radius`` so the planner
(Phase 6 A*) can treat the robot as a point and simply reject occupied
cells, instead of doing per-node footprint collision checks itself.

Always operates on a copy -- the raw occupancy grid (built by
``mapping.occupancy_grid`` from real sensor data) must never be mutated
by planning-time inflation, per the project's engineering constraints.
"""

from __future__ import annotations

import math

import numpy as np

from common.messages import OccupancyGrid


def compute_inflation_radius(robot_width: float, robot_length: float, safety_margin: float) -> float:
    """Robot's bounding-circle radius (half its footprint diagonal) plus a safety margin.

    Using the half-diagonal (not just half-width) ensures the circle
    covers the robot regardless of heading, since the inflated grid does
    not account for orientation -- the planner treats the robot as a
    point once cells are inflated.
    """
    if robot_width <= 0.0 or robot_length <= 0.0:
        raise ValueError("robot_width and robot_length must be positive")
    if safety_margin < 0.0:
        raise ValueError("safety_margin must be >= 0")
    half_diagonal = math.hypot(robot_width, robot_length) / 2.0
    return half_diagonal + safety_margin


def _shift_mask(mask: np.ndarray, d_row: int, d_col: int) -> np.ndarray:
    """Return ``mask`` shifted by (d_row, d_col), zero-filled at the edges."""
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


def inflate_occupancy_grid(grid: OccupancyGrid, inflation_radius: float) -> OccupancyGrid:
    """Return a new ``OccupancyGrid`` with obstacles expanded by ``inflation_radius`` meters.

    Any cell whose center lies within ``inflation_radius`` of an occupied
    cell's center becomes occupied in the result, regardless of its
    original free/unknown state. The input grid is never modified.
    """
    if inflation_radius < 0.0:
        raise ValueError("inflation_radius must be >= 0")

    new_cells = grid.cells.copy()
    occupied_mask = grid.cells == OccupancyGrid.OCCUPIED

    if inflation_radius == 0.0 or not np.any(occupied_mask):
        return OccupancyGrid(
            resolution=grid.resolution,
            width=grid.width,
            height=grid.height,
            origin_x=grid.origin_x,
            origin_y=grid.origin_y,
            cells=new_cells,
        )

    radius_cells = int(math.ceil(inflation_radius / grid.resolution))
    inflated_mask = occupied_mask.copy()

    for d_row in range(-radius_cells, radius_cells + 1):
        for d_col in range(-radius_cells, radius_cells + 1):
            distance = math.hypot(d_row * grid.resolution, d_col * grid.resolution)
            if distance > inflation_radius:
                continue
            if d_row == 0 and d_col == 0:
                continue
            inflated_mask |= _shift_mask(occupied_mask, d_row, d_col)

    new_cells[inflated_mask] = OccupancyGrid.OCCUPIED

    return OccupancyGrid(
        resolution=grid.resolution,
        width=grid.width,
        height=grid.height,
        origin_x=grid.origin_x,
        origin_y=grid.origin_y,
        cells=new_cells,
    )


def inflate_from_footprint(
    grid: OccupancyGrid, robot_width: float, robot_length: float, safety_margin: float
) -> OccupancyGrid:
    """Convenience wrapper: compute the inflation radius from the footprint, then inflate."""
    radius = compute_inflation_radius(robot_width, robot_length, safety_margin)
    return inflate_occupancy_grid(grid, radius)
