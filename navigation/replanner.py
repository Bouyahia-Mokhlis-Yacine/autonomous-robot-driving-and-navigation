"""Dynamic obstacle layer, planning-map combination, and replan triggers.

Maintains the three-layer picture from the project spec:

    1. global/static map       -- the persistent SLAM map (mapping.occupancy_grid)
    2. local/dynamic layer     -- DynamicObstacleLayer, rebuilt from only the
                                   most recent scan(s), never accumulated
    3. combined planning map   -- combine_planning_map(static, dynamic),
                                   which the caller then inflates (mapping.inflation)
                                   before handing to the planner

The dynamic layer is deliberately short-lived and separate from the
global map so a moving obstacle that leaves the scene does not
permanently corrupt the persistent SLAM map, per project constraints.

``Replanner`` aggregates all Phase 9 + Phase 10 replan triggers: a
blocked path (Phase 9 baseline), a periodic timer, and the robot
deviating too far from its path. A changed goal is not handled here --
``NavigationManager.set_goal`` always plans immediately, so that trigger
needs no separate check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from common.geometry import point_to_segment_distance
from common.messages import LaserScan, OccupancyGrid, Path, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from mapping.ray_casting import bresenham_line
from planning.path_smoothing import has_line_of_sight


def path_is_blocked(path: Path, grid_map: OccupancyGridMap) -> bool:
    """True if any waypoint or connecting segment of ``path`` is now occupied."""
    if path.is_empty():
        return False

    for point in path.points:
        row, col = grid_map.world_to_grid(point.x, point.y)
        if not grid_map.in_bounds(row, col) or grid_map.get_cell(row, col) == OccupancyGrid.OCCUPIED:
            return True

    for a, b in zip(path.points, path.points[1:]):
        if not has_line_of_sight(grid_map, a, b):
            return True

    return False


def path_deviation(pose: Pose2D, path: Path) -> float:
    """Distance from ``pose`` to the nearest point on ``path`` (0 if empty)."""
    if path.is_empty():
        return 0.0
    if len(path.points) == 1:
        p = path.points[0]
        return point_to_segment_distance(pose.x, pose.y, p.x, p.y, p.x, p.y)
    return min(
        point_to_segment_distance(pose.x, pose.y, a.x, a.y, b.x, b.y)
        for a, b in zip(path.points, path.points[1:])
    )


def obstacle_ahead(pose: Pose2D, grid_map: OccupancyGridMap, lookahead: float) -> bool:
    """True if an occupied cell (or the map boundary) lies within ``lookahead``
    meters directly ahead of ``pose`` along its current heading."""
    end_x = pose.x + lookahead * math.cos(pose.theta)
    end_y = pose.y + lookahead * math.sin(pose.theta)
    r0, c0 = grid_map.world_to_grid(pose.x, pose.y)
    r1, c1 = grid_map.world_to_grid(end_x, end_y)

    for row, col in bresenham_line(r0, c0, r1, c1):
        if not grid_map.in_bounds(row, col):
            return True  # the world boundary is a real wall in this simulator
        if grid_map.get_cell(row, col) == OccupancyGrid.OCCUPIED:
            return True
    return False


class DynamicObstacleLayer:
    """A short-lived occupancy grid built from only the most recent scan.

    ``update`` fully resets the grid before integrating the new scan, so
    stale detections never linger once an obstacle moves away.
    """

    def __init__(
        self, resolution: float, width_m: float, height_m: float, origin_x: float = 0.0, origin_y: float = 0.0
    ) -> None:
        self._resolution = resolution
        self._width_m = width_m
        self._height_m = height_m
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._grid_map = OccupancyGridMap(resolution, width_m, height_m, origin_x, origin_y)

    def update(self, pose: Pose2D, scan: LaserScan) -> None:
        self._grid_map = OccupancyGridMap(
            self._resolution, self._width_m, self._height_m, self._origin_x, self._origin_y
        )
        self._grid_map.update_from_scan(pose, scan)

    def get_grid(self) -> OccupancyGrid:
        return self._grid_map.grid


def combine_planning_map(static_grid: OccupancyGrid, dynamic_grid: OccupancyGrid) -> OccupancyGrid:
    """Merge the static and dynamic layers: occupied wins, then free, then unknown.

    Both grids must share resolution, dimensions, and origin (the normal
    case, since both are built over the same world).
    """
    if (
        static_grid.resolution != dynamic_grid.resolution
        or static_grid.width != dynamic_grid.width
        or static_grid.height != dynamic_grid.height
        or static_grid.origin_x != dynamic_grid.origin_x
        or static_grid.origin_y != dynamic_grid.origin_y
    ):
        raise ValueError("static_grid and dynamic_grid must share resolution, dimensions, and origin")

    occupied_mask = (static_grid.cells == OccupancyGrid.OCCUPIED) | (dynamic_grid.cells == OccupancyGrid.OCCUPIED)
    free_mask = (~occupied_mask) & (
        (static_grid.cells == OccupancyGrid.FREE) | (dynamic_grid.cells == OccupancyGrid.FREE)
    )

    combined = np.full_like(static_grid.cells, OccupancyGrid.UNKNOWN)
    combined[free_mask] = OccupancyGrid.FREE
    combined[occupied_mask] = OccupancyGrid.OCCUPIED

    return OccupancyGrid(
        resolution=static_grid.resolution,
        width=static_grid.width,
        height=static_grid.height,
        origin_x=static_grid.origin_x,
        origin_y=static_grid.origin_y,
        cells=combined,
    )


@dataclass
class ReplannerConfig:
    replan_interval: float          # seconds, periodic replan timer
    deviation_threshold: float      # meters, forces a replan if exceeded

    def __post_init__(self) -> None:
        if self.replan_interval <= 0.0:
            raise ValueError("replan_interval must be > 0")
        if self.deviation_threshold <= 0.0:
            raise ValueError("deviation_threshold must be > 0")


class Replanner:
    """Aggregates the replan triggers: blocked path, deviation, periodic timer."""

    def __init__(self, config: ReplannerConfig) -> None:
        self.config = config
        self._time_since_last_replan = 0.0

    def reset_timer(self) -> None:
        """Call whenever a fresh path is committed (new goal or successful replan)."""
        self._time_since_last_replan = 0.0

    def should_replan(self, pose: Pose2D, path: Path, grid_map: OccupancyGridMap, dt: float) -> tuple[bool, str]:
        self._time_since_last_replan += dt

        if path_is_blocked(path, grid_map):
            return True, "path is blocked"

        deviation = path_deviation(pose, path)
        if deviation > self.config.deviation_threshold:
            return True, f"robot deviated {deviation:.2f} m from path"

        if self._time_since_last_replan >= self.config.replan_interval:
            return True, "periodic replan interval expired"

        return False, ""
