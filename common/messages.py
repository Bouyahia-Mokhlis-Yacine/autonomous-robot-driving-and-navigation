"""ROS-2-style message dataclasses shared across all subsystems.

These types are the explicit contracts between modules (sensors, SLAM,
mapping, planning, control, robot). Keeping them here means every module
depends on a stable, dependency-free data layer instead of on each other's
internals, so any component (e.g. SimulatedLiDAR) can later be swapped for
a hardware equivalent (e.g. RPLiDARDriver) without touching consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np


@dataclass
class LaserScan:
    """A single 2D LiDAR scan, analogous to sensor_msgs/LaserScan.

    Ranges are ordered from ``angle_min`` to ``angle_max`` in steps of
    ``angle_increment``. A range value equal to ``range_max`` means "no
    obstacle detected" (max-range return), matching ROS conventions.

    ``scan_time``/``time_increment`` mirror sensor_msgs/LaserScan's
    fields of the same names: the duration of one full acquisition
    (seconds) and the time between adjacent angular bins. ``0.0`` means
    "not provided" (an instantaneous/idealized scan, or a source that
    does not report timing -- the real ``RPLidarDriver`` currently
    leaves both at 0.0). Nothing in SLAM/mapping reads them; they are
    informational metadata so consumers and diagnostics can tell an
    instantaneous scan from one acquired over a rotation. Unlike ROS,
    bin 0 is not necessarily the first-acquired bin -- see
    ``simulation.realistic_lidar`` for the acquisition order.
    """

    timestamp: float
    angle_min: float
    angle_increment: float
    ranges: list[float]
    range_min: float
    range_max: float
    scan_time: float = 0.0
    time_increment: float = 0.0

    def __post_init__(self) -> None:
        if self.angle_increment == 0.0 and len(self.ranges) > 1:
            raise ValueError("angle_increment must be nonzero for a multi-ray scan")
        if self.range_min < 0.0:
            raise ValueError("range_min must be >= 0")
        if self.range_max <= self.range_min:
            raise ValueError("range_max must be > range_min")
        if self.scan_time < 0.0 or self.time_increment < 0.0:
            raise ValueError("scan_time and time_increment must be >= 0")

    @property
    def num_rays(self) -> int:
        return len(self.ranges)

    def angle_at(self, index: int) -> float:
        """Angle (radians) of the ray at ``index``."""
        return self.angle_min + index * self.angle_increment


@dataclass
class Pose2D:
    """Robot pose in the 2D world frame."""

    x: float
    y: float
    theta: float
    timestamp: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.theta)


@dataclass
class Goal:
    """A user- or system-requested navigation destination."""

    x: float
    y: float


@dataclass
class PathPoint:
    """A single waypoint in world coordinates.

    ``direction`` is how the robot must travel TOWARD this point:
    ``0`` = either (the path follower picks forward, reverse or a turn in
    place -- what the grid planner always produces), ``+1`` = forward only,
    ``-1`` = reverse only."""

    x: float
    y: float
    direction: int = 0

    FORWARD: ClassVar[int] = 1
    REVERSE: ClassVar[int] = -1
    EITHER: ClassVar[int] = 0


@dataclass
class Path:
    """An ordered sequence of waypoints from start to goal."""

    points: list[PathPoint] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def is_empty(self) -> bool:
        return len(self.points) == 0


@dataclass
class VelocityCommand:
    """A commanded body-frame velocity for a differential/skid-steer robot."""

    linear: float
    angular: float


@dataclass
class OccupancyGrid:
    """A 2D occupancy grid, analogous to nav_msgs/OccupancyGrid.

    ``cells`` uses the standard occupancy-grid convention:
        -1  -> unknown
         0  -> free
         100 -> occupied

    The grid origin (``origin_x``, ``origin_y``) is the world-coordinate
    position of cell ``(0, 0)``'s lower-left corner.
    """

    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    cells: np.ndarray  # shape (height, width), dtype=int8

    UNKNOWN: ClassVar[int] = -1
    FREE: ClassVar[int] = 0
    OCCUPIED: ClassVar[int] = 100

    def __post_init__(self) -> None:
        expected_shape = (self.height, self.width)
        if self.cells.shape != expected_shape:
            raise ValueError(
                f"cells shape {self.cells.shape} does not match "
                f"(height, width) = {expected_shape}"
            )
