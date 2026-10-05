"""Direction-aware motion safety: is it clear to move the way we are about to?

Each motion mode protects the part of the world it moves into, using the
robot's footprint (robot.yaml ``robot.length``/``width``), not a single ray:

    FORWARD           a zone in front of the front edge:
                      L/2 < x <= L/2 + stop_distance,  |y| <= W/2 + side_margin
    REVERSE           the same zone behind the rear edge
    ROTATE_IN_PLACE   a circle of radius (footprint half-diagonal +
                      rotation_margin) -- what the corners sweep while turning

(robot frame: x = FRONT, y = LEFT, origin = footprint centre. Scan points
are shifted by the LiDAR mount, robot.yaml ``robot.lidar_x``/``lidar_y``;
0 deg of the sensor = front, per the driver's convention and the 2026-09-29
check.)

Evidence, checked every control tick:
    - the LATEST RAW LiDAR SCAN. Its points are already in the robot frame
      (bin angle + range), so no pose estimate is involved -- a localization
      error cannot hide an obstacle -- and it is fresh: a new obstacle counts
      without waiting to be confirmed in the map. ``min_scan_points`` points
      are needed inside a zone (a single outlier does not stop the robot).
    - the CONFIRMED MAP (the raw, uninflated world-model grid) near the
      robot, transformed with the current pose: remembered geometry, e.g.
      something in a LiDAR blind spot. One cell is enough.
Points INSIDE the footprint are ignored: nothing real can be there (they
are the robot itself, or map/pose noise), and counting them would freeze
the robot permanently.

This does not replace ``navigation.replanner.obstacle_ahead`` (the planning
map ray that triggers a stop-and-replan); NavigationManager runs both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from common.messages import LaserScan, OccupancyGrid, Pose2D
from control.path_follower import MotionMode
from mapping.occupancy_grid import OccupancyGridMap


@dataclass(frozen=True)
class MotionSafetyConfig:
    robot_length: float
    robot_width: float
    stop_distance: float          # depth of the front/rear zones (m)
    side_margin: float = 0.05
    rotation_margin: float = 0.03
    min_scan_points: int = 2
    lidar_x: float = 0.0          # LiDAR mount in the robot frame (robot.yaml robot.lidar_x/lidar_y)
    lidar_y: float = 0.0

    def __post_init__(self) -> None:
        if self.robot_length <= 0 or self.robot_width <= 0 or self.stop_distance <= 0:
            raise ValueError("robot size and stop_distance must be > 0")
        if self.side_margin < 0 or self.rotation_margin < 0 or self.min_scan_points < 1:
            raise ValueError("margins must be >= 0 and min_scan_points >= 1")

    @property
    def rotation_radius(self) -> float:
        return math.hypot(self.robot_length, self.robot_width) / 2.0 + self.rotation_margin


def motion_safety_config(robot_cfg) -> MotionSafetyConfig:
    """From a ``common.config.RobotConfig`` (robot.yaml)."""
    s = robot_cfg.safety
    return MotionSafetyConfig(robot_length=robot_cfg.robot.length, robot_width=robot_cfg.robot.width,
                              stop_distance=s.emergency_stop_distance, side_margin=s.side_margin,
                              rotation_margin=s.rotation_margin, min_scan_points=s.min_scan_points,
                              lidar_x=robot_cfg.robot.lidar_x, lidar_y=robot_cfg.robot.lidar_y)


class MotionSafety:
    def __init__(self, config: MotionSafetyConfig,
                 scan_provider: Callable[[], LaserScan | None] | None = None) -> None:
        self.config = config
        self._scan_provider = scan_provider
        self.last_block: str | None = None   # human-readable reason of the most recent block

    # ------------------------------------------------------------- public ---

    def is_clear(self, mode: MotionMode, pose: Pose2D, raw_grid_map: OccupancyGridMap | None) -> bool:
        """True if moving in ``mode`` is clear. STOPPED is always clear."""
        if mode is MotionMode.STOPPED:
            return True
        c = self.config
        for source, points, needed in (("scan", self._scan_points(), c.min_scan_points),
                                       ("map", self._map_points(pose, raw_grid_map), 1)):
            if points.size == 0:
                continue
            inside = self._in_zone(mode, points)
            count = int(np.count_nonzero(inside))
            if count >= needed:
                x, y = points[inside][np.argmin(np.hypot(*points[inside].T))]
                self.last_block = (f"{mode.value} blocked by {count} {source} point(s); nearest at "
                                   f"x {x:+.2f} m, y {y:+.2f} m (robot frame)")
                return False
        return True

    def modes_to_check(self, mode: MotionMode, linear: float, angular: float) -> set[MotionMode]:
        """The follower's mode plus whatever the actual (acceleration-limited)
        command still does -- e.g. still rolling forward while the mode
        already says REVERSE."""
        modes = {mode}
        if linear > 1e-3:
            modes.add(MotionMode.FORWARD)
        elif linear < -1e-3:
            modes.add(MotionMode.REVERSE)
        elif abs(angular) > 1e-3:
            modes.add(MotionMode.ROTATE_IN_PLACE)
        modes.discard(MotionMode.STOPPED)
        return modes

    # ------------------------------------------------------------ helpers ---

    def _in_zone(self, mode: MotionMode, pts: np.ndarray) -> np.ndarray:
        c = self.config
        x, y = pts[:, 0], pts[:, 1]
        hl, hw = c.robot_length / 2.0, c.robot_width / 2.0
        in_footprint = (np.abs(x) <= hl) & (np.abs(y) <= hw)
        lateral = np.abs(y) <= hw + c.side_margin
        if mode is MotionMode.FORWARD:
            return lateral & (x > hl) & (x <= hl + c.stop_distance)
        if mode is MotionMode.REVERSE:
            return lateral & (x < -hl) & (x >= -hl - c.stop_distance)
        if mode is MotionMode.ROTATE_IN_PLACE:
            return (np.hypot(x, y) <= c.rotation_radius) & ~in_footprint
        return np.zeros(len(pts), dtype=bool)

    def _reach(self) -> float:
        c = self.config
        return max(c.rotation_radius, math.hypot(c.robot_length / 2.0 + c.stop_distance,
                                                 c.robot_width / 2.0 + c.side_margin))

    def _scan_points(self) -> np.ndarray:
        """Latest raw scan hits in the robot frame (shifted by the LiDAR mount), within reach."""
        scan = self._scan_provider() if self._scan_provider is not None else None
        if scan is None:
            return np.empty((0, 2))
        r = np.asarray(scan.ranges, dtype=np.float64)
        a = scan.angle_min + np.arange(r.size) * scan.angle_increment
        c = self.config
        mount = math.hypot(c.lidar_x, c.lidar_y)
        ok = np.isfinite(r) & (r >= scan.range_min) & (r < scan.range_max) & (r <= self._reach() + mount)
        return np.column_stack((r[ok] * np.cos(a[ok]) + c.lidar_x, r[ok] * np.sin(a[ok]) + c.lidar_y))

    def _map_points(self, pose: Pose2D, raw_grid_map: OccupancyGridMap | None) -> np.ndarray:
        """Confirmed occupied cells near the robot, in the robot frame."""
        if raw_grid_map is None:
            return np.empty((0, 2))
        grid = raw_grid_map.grid
        reach = self._reach() + grid.resolution
        r0, c0 = raw_grid_map.world_to_grid(pose.x - reach, pose.y - reach)
        r1, c1 = raw_grid_map.world_to_grid(pose.x + reach, pose.y + reach)
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(grid.height - 1, r1), min(grid.width - 1, c1)
        if r1 < r0 or c1 < c0:
            return np.empty((0, 2))
        rows, cols = np.nonzero(grid.cells[r0:r1 + 1, c0:c1 + 1] == OccupancyGrid.OCCUPIED)
        if rows.size == 0:
            return np.empty((0, 2))
        wx = grid.origin_x + (cols + c0 + 0.5) * grid.resolution - pose.x
        wy = grid.origin_y + (rows + r0 + 0.5) * grid.resolution - pose.y
        cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
        return np.column_stack((cos_t * wx + sin_t * wy, -sin_t * wx + cos_t * wy))
