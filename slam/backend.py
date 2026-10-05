"""Pluggable pose-estimation backends behind a single ``SLAMBackend`` protocol.

Per project constraints, this does not implement a SLAM optimizer
(scan matching, pose-graph optimization, loop closure) from scratch. A
real backend should wrap a proven library or algorithm (e.g. scan
matching, Cartographer, BreezySLAM) behind ``estimate_pose`` — nothing
outside ``slam/`` needs to change when that swap happens, since
``slam.slam.SLAM`` only depends on this protocol.

Three backends are provided:

- ``GroundTruthBackend``: simulation-only "cheat" localization that reads
  the simulator's exact pose. This is explicitly not SLAM; it exists so
  the rest of the navigation stack (mapping, planning, control) can be
  built and tested before a real localization algorithm is wired in.
- ``OdometryBackend``: open-loop dead reckoning via ``pose.integrate_pose``.
  No LiDAR-based correction, so it drifts over time. Originally meant as
  a stand-in for real wheel-encoder odometry -- the robot this project
  targets has no encoders, so this is no longer that stand-in; it
  remains available as a documented fallback/motion-prior source should
  one ever be needed (e.g. from an IMU), but is not driven by anything
  in the current hardware.
- ``LidarScanMatchingBackend``: the real backend for this robot. Wraps
  BreezySLAM (github.com/simondlevy/BreezySLAM), a proven, ROS-free 2D
  LiDAR SLAM library purpose-built to run *without* wheel odometry --
  exactly this project's situation. Estimates pose by scan-matching
  against BreezySLAM's own internal map, which exists purely as its
  scratch space for that matching; the occupancy grid the rest of this
  project actually plans against is still built independently by
  ``mapping.occupancy_grid.OccupancyGridMap`` from the poses this
  backend returns, unchanged from how ``GroundTruthBackend`` and
  ``OdometryBackend`` already work with it.
"""

from __future__ import annotations

import math
from typing import Callable, Protocol

from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser

from common.config import LidarConfig, SlamConfig
from common.geometry import wrap_angle
from common.messages import LaserScan, Pose2D, VelocityCommand
from slam.pose import integrate_pose


class SLAMBackend(Protocol):
    """Contract every localization backend must satisfy."""

    def estimate_pose(self, scan: LaserScan) -> Pose2D:
        """Return the current best pose estimate, given the latest scan."""
        ...


class GroundTruthBackend:
    """Simulation-mode localization: returns the simulator's true pose.

    ``pose_provider`` is typically ``lambda: environment.robot_pose``.
    Ignores the scan entirely -- this is not SLAM, it is a placeholder
    for early pipeline development, as permitted by the project spec.
    """

    def __init__(self, pose_provider: Callable[[], Pose2D]) -> None:
        self._pose_provider = pose_provider

    def estimate_pose(self, scan: LaserScan) -> Pose2D:
        return self._pose_provider()


class OdometryBackend:
    """Open-loop dead-reckoning localization from commanded velocities.

    Call ``integrate(cmd, dt)`` once per control tick to advance the
    internal pose estimate; ``estimate_pose`` then returns that estimate
    (the scan argument is unused -- there is no LiDAR-based correction).
    """

    def __init__(self, initial_pose: Pose2D) -> None:
        self._pose = initial_pose

    def integrate(self, cmd: VelocityCommand, dt: float) -> Pose2D:
        self._pose = integrate_pose(self._pose, cmd, dt)
        return self._pose

    def estimate_pose(self, scan: LaserScan) -> Pose2D:
        return self._pose


class LidarScanMatchingBackend:
    """Real localization: LiDAR scan matching via BreezySLAM, no odometry input.

    Reports pose in a world frame with the origin at wherever the robot
    was when this backend was constructed (BreezySLAM itself works in an
    internal frame centered on its scratch map; this class subtracts
    that fixed offset so callers see an ordinary world-frame ``Pose2D``,
    consistent with the other backends).

    A scan with too few valid (in-range) points is not fed to BreezySLAM
    -- scan matching on a near-empty scan is more likely to produce a
    bad jump than a useful correction -- and the previous pose estimate
    is returned unchanged instead.
    """

    def __init__(self, lidar_config: LidarConfig, slam_config: SlamConfig, min_valid_points: int = 10) -> None:
        num_rays = int(round((lidar_config.angle_max - lidar_config.angle_min) / lidar_config.angular_resolution)) + 1
        laser = Laser(
            scan_size=num_rays,
            scan_rate_hz=lidar_config.scan_rate,
            detection_angle_degrees=math.degrees(lidar_config.angle_max - lidar_config.angle_min),
            distance_no_detection_mm=lidar_config.range_max * 1000.0,
        )
        self._slam = RMHC_SLAM(
            laser,
            slam_config.map_size_pixels,
            slam_config.map_size_meters,
            hole_width_mm=slam_config.hole_width_mm,
            sigma_xy_mm=slam_config.sigma_xy_mm,
            sigma_theta_degrees=slam_config.sigma_theta_degrees,
            max_search_iter=slam_config.max_search_iter,
        )
        # BreezySLAM starts the robot at the center of its internal map
        # (see SinglePositionSLAM.__init__: init_coord_mm = 500 * map_size_meters);
        # subtracting that offset makes our reported pose start at (0, 0).
        self._origin_offset_mm = 500.0 * slam_config.map_size_meters
        self._min_valid_points = min_valid_points
        self._last_pose = Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=0.0)

    def estimate_pose(self, scan: LaserScan) -> Pose2D:
        distances_mm: list[float] = []
        angles_deg: list[float] = []
        for i in range(scan.num_rays):
            r = scan.ranges[i]
            if r >= scan.range_max:
                continue  # no detection; only real hits are sent, matching BreezySLAM's own RPLiDAR example
            distances_mm.append(r * 1000.0)
            angles_deg.append(math.degrees(scan.angle_at(i)))

        if len(distances_mm) < self._min_valid_points:
            return self._last_pose

        self._slam.update(distances_mm, scan_angles_degrees=angles_deg)
        x_mm, y_mm, theta_deg = self._slam.getpos()

        pose = Pose2D(
            x=(x_mm - self._origin_offset_mm) / 1000.0,
            y=(y_mm - self._origin_offset_mm) / 1000.0,
            theta=wrap_angle(math.radians(theta_deg)),
            timestamp=scan.timestamp,
        )
        self._last_pose = pose
        return pose
