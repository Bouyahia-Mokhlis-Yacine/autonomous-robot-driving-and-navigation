"""Ray-casting 2D LiDAR simulator.

Casts rays from the robot pose against an ``Environment``'s obstacles and
its outer world boundary (the room's walls also return finite ranges,
they are not "open space"). Each ray reports the nearest intersection, or
``range_max`` if nothing is within sensor range. Output is a
``LaserScan`` — the same message type a real RPLiDAR driver will produce
(Phase 12), so downstream mapping/SLAM code is sensor-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from common.config import LidarConfig
from common.geometry import Point, ray_box_intersection
from common.messages import LaserScan, Pose2D
from simulation.environment import Environment


@dataclass
class SimulatedLidar:
    config: LidarConfig

    def __post_init__(self) -> None:
        c = self.config
        if c.range_min < 0.0:
            raise ValueError("range_min must be >= 0")
        if c.range_max <= c.range_min:
            raise ValueError("range_max must be > range_min")
        if c.angular_resolution <= 0.0:
            raise ValueError("angular_resolution must be > 0")
        if c.angle_max <= c.angle_min:
            raise ValueError("angle_max must be > angle_min")
        if c.scan_rate <= 0.0:
            raise ValueError("scan_rate must be > 0")

    def _ray_angles(self) -> list[float]:
        c = self.config
        num_rays = int(math.floor((c.angle_max - c.angle_min) / c.angular_resolution)) + 1
        return [c.angle_min + i * c.angular_resolution for i in range(num_rays)]

    def _cast_ray(self, environment: Environment, origin: Point, direction: Point) -> float:
        """Nearest hit distance along ``direction`` from ``origin``, world units."""
        world_box_min = (0.0, 0.0)
        world_box_max = (environment.width, environment.height)
        boundary_hit = ray_box_intersection(origin, direction, world_box_min, world_box_max)
        # The origin is inside the world by construction (collision-checked
        # robot pose), so the boundary's exit distance is the wall's range.
        best = boundary_hit[1] if boundary_hit is not None else self.config.range_max

        for obstacle in environment.all_obstacles():
            half_w, half_h = obstacle.width / 2.0, obstacle.height / 2.0
            box_min = (obstacle.x - half_w, obstacle.y - half_h)
            box_max = (obstacle.x + half_w, obstacle.y + half_h)
            hit = ray_box_intersection(origin, direction, box_min, box_max)
            if hit is None:
                continue
            t_enter, t_exit = hit
            candidate = t_enter if t_enter >= 0.0 else (0.0 if t_exit >= 0.0 else None)
            if candidate is not None and candidate < best:
                best = candidate

        return best

    def scan(self, environment: Environment, pose: Pose2D, timestamp: float) -> LaserScan:
        """Simulate one LiDAR sweep from ``pose`` within ``environment``."""
        origin = (pose.x, pose.y)
        ranges: list[float] = []
        for local_angle in self._ray_angles():
            world_angle = pose.theta + local_angle
            direction = (math.cos(world_angle), math.sin(world_angle))
            distance = self._cast_ray(environment, origin, direction)
            ranges.append(self.config.range_max if distance > self.config.range_max else max(distance, 0.0))

        return LaserScan(
            timestamp=timestamp,
            angle_min=self.config.angle_min,
            angle_increment=self.config.angular_resolution,
            ranges=ranges,
            range_min=self.config.range_min,
            range_max=self.config.range_max,
        )
