"""2D occupancy grid map: world<->grid transforms and LiDAR scan integration.

Deliberately independent of planning: this module only maintains free /
occupied / unknown cell state from sensor data. ``planning.astar`` must
treat the resulting grid as an opaque input, never the reverse.
"""

from __future__ import annotations

import math

import numpy as np

from common.messages import LaserScan, OccupancyGrid, Pose2D
from mapping.ray_casting import bresenham_line


class OccupancyGridMap:
    """Owns an ``OccupancyGrid`` message and the logic to update it from scans."""

    def __init__(
        self,
        resolution: float,
        width_m: float,
        height_m: float,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ) -> None:
        if resolution <= 0.0:
            raise ValueError("resolution must be > 0")
        if width_m <= 0.0 or height_m <= 0.0:
            raise ValueError("width_m and height_m must be positive")

        width_cells = max(1, round(width_m / resolution))
        height_cells = max(1, round(height_m / resolution))
        cells = np.full((height_cells, width_cells), OccupancyGrid.UNKNOWN, dtype=np.int8)

        self.grid = OccupancyGrid(
            resolution=resolution,
            width=width_cells,
            height=height_cells,
            origin_x=origin_x,
            origin_y=origin_y,
            cells=cells,
        )

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """World (x, y) -> grid (row, col) containing that point. Not bounds-checked."""
        col = int(math.floor((x - self.grid.origin_x) / self.grid.resolution))
        row = int(math.floor((y - self.grid.origin_y) / self.grid.resolution))
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Grid (row, col) -> world coordinates of that cell's center."""
        x = self.grid.origin_x + (col + 0.5) * self.grid.resolution
        y = self.grid.origin_y + (row + 0.5) * self.grid.resolution
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.grid.height and 0 <= col < self.grid.width

    def get_cell(self, row: int, col: int) -> int:
        if not self.in_bounds(row, col):
            raise IndexError(f"Cell ({row}, {col}) is outside the grid")
        return int(self.grid.cells[row, col])

    def set_cell(self, row: int, col: int, value: int) -> None:
        if not self.in_bounds(row, col):
            raise IndexError(f"Cell ({row}, {col}) is outside the grid")
        self.grid.cells[row, col] = value

    def update_from_scan(self, pose: Pose2D, scan: LaserScan) -> None:
        """Integrate one LiDAR scan: mark traversed cells free, hit cells occupied.

        For each ray, the endpoint is marked OCCUPIED only if the ray
        actually hit something (range < range_max); a max-range return
        means "no obstacle observed" and its endpoint cell is marked
        FREE. Cells outside the grid are silently skipped. Unknown cells
        that no ray touches are left untouched.
        """
        origin_row, origin_col = self.world_to_grid(pose.x, pose.y)

        for i in range(scan.num_rays):
            measured_range = scan.ranges[i]
            is_hit = measured_range < scan.range_max
            angle = pose.theta + scan.angle_at(i)

            end_x = pose.x + measured_range * math.cos(angle)
            end_y = pose.y + measured_range * math.sin(angle)
            end_row, end_col = self.world_to_grid(end_x, end_y)

            traced_cells = bresenham_line(origin_row, origin_col, end_row, end_col)

            for row, col in traced_cells[:-1]:
                if self.in_bounds(row, col):
                    self.grid.cells[row, col] = OccupancyGrid.FREE

            last_row, last_col = traced_cells[-1]
            if self.in_bounds(last_row, last_col):
                self.grid.cells[last_row, last_col] = (
                    OccupancyGrid.OCCUPIED if is_hit else OccupancyGrid.FREE
                )
