"""Top-level SLAM interface: LaserScan -> Pose2D + OccupancyGrid.

This is the only SLAM-related type the rest of the project should ever
import. Planning, control, and navigation depend on ``SLAM.update`` /
``SLAM.get_map`` and nothing about the chosen ``SLAMBackend``.
"""

from __future__ import annotations

from common.messages import LaserScan, OccupancyGrid, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from slam.backend import SLAMBackend


class SLAM:
    """Combines a pose-estimation backend with occupancy-grid mapping."""

    def __init__(self, backend: SLAMBackend, occupancy_grid_map: OccupancyGridMap) -> None:
        self._backend = backend
        self._map = occupancy_grid_map
        self._latest_pose: Pose2D | None = None

    def update(self, scan: LaserScan) -> Pose2D:
        """Estimate the current pose from ``scan`` and fold it into the map."""
        pose = self._backend.estimate_pose(scan)
        self._map.update_from_scan(pose, scan)
        self._latest_pose = pose
        return pose

    def get_map(self) -> OccupancyGrid:
        return self._map.grid

    def get_pose(self) -> Pose2D | None:
        """Most recent pose from ``update``, or None before the first call."""
        return self._latest_pose
