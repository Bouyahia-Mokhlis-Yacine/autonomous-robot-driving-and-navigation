"""2D rectangular simulation world: bounds, obstacles, robot, collisions.

This module owns only world state and physical collision detection. It
must not contain any planning, path-following, or navigation-state logic
— those live in ``planning``, ``control``, and ``navigation``. The
simulator exposes obstacle geometry so ``simulation.simulated_lidar``
(Phase 2) can ray-cast against it, but never reaches into planner
internals itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from common.geometry import polygons_intersect, rectangle_corners
from common.messages import Pose2D
from simulation.obstacles import DynamicObstacle, RectangleObstacle


@dataclass
class Environment:
    """A rectangular world of size (width x height) meters, origin at (0, 0).

    The robot footprint is modeled as a rectangle (``robot_length`` along
    its forward/local-x axis, ``robot_width`` along local-y), matching
    ``config/robot.yaml``'s ``length``/``width`` fields.
    """

    width: float
    height: float
    robot_length: float
    robot_width: float
    robot_pose: Pose2D
    static_obstacles: list[RectangleObstacle] = field(default_factory=list)
    dynamic_obstacles: list[DynamicObstacle] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("World width and height must be positive")
        if self.robot_length <= 0.0 or self.robot_width <= 0.0:
            raise ValueError("Robot footprint dimensions must be positive")

    def add_static_obstacle(self, obstacle: RectangleObstacle) -> None:
        self.static_obstacles.append(obstacle)

    def add_dynamic_obstacle(self, obstacle: DynamicObstacle) -> None:
        self.dynamic_obstacles.append(obstacle)

    def all_obstacles(self) -> list[RectangleObstacle]:
        """All obstacles (static + dynamic) as a flat list, for sensors/mapping."""
        return [*self.static_obstacles, *self.dynamic_obstacles]

    def robot_footprint_corners(self, pose: Pose2D | None = None) -> list[tuple[float, float]]:
        p = pose if pose is not None else self.robot_pose
        return rectangle_corners(p.x, p.y, self.robot_length, self.robot_width, p.theta)

    def is_in_bounds(self, pose: Pose2D | None = None) -> bool:
        """True if the robot footprint at ``pose`` lies entirely within the world."""
        for x, y in self.robot_footprint_corners(pose):
            if not (0.0 <= x <= self.width and 0.0 <= y <= self.height):
                return False
        return True

    def check_collision(self, pose: Pose2D | None = None) -> bool:
        """True if the robot footprint at ``pose`` is out of bounds or overlaps an obstacle."""
        if not self.is_in_bounds(pose):
            return True
        footprint = self.robot_footprint_corners(pose)
        for obstacle in self.all_obstacles():
            if polygons_intersect(footprint, obstacle.corners()):
                return True
        return False

    def try_move_robot(self, pose: Pose2D) -> bool:
        """Attempt to move the robot to ``pose``.

        Returns True and updates ``robot_pose`` if the destination is
        collision-free; otherwise leaves ``robot_pose`` unchanged and
        returns False. This is physical collision handling, not path
        planning: it does not search for an alternate pose.
        """
        if self.check_collision(pose):
            return False
        self.robot_pose = pose
        return True

    def step(self, dt: float) -> None:
        """Advance dynamic obstacles by ``dt`` seconds, bouncing off world walls."""
        if dt < 0.0:
            raise ValueError("dt must be >= 0")
        for obstacle in self.dynamic_obstacles:
            obstacle.step(dt)
            half_w, half_h = obstacle.width / 2.0, obstacle.height / 2.0

            if obstacle.x - half_w < 0.0:
                obstacle.x = half_w
                obstacle.velocity_x = abs(obstacle.velocity_x)
            elif obstacle.x + half_w > self.width:
                obstacle.x = self.width - half_w
                obstacle.velocity_x = -abs(obstacle.velocity_x)

            if obstacle.y - half_h < 0.0:
                obstacle.y = half_h
                obstacle.velocity_y = abs(obstacle.velocity_y)
            elif obstacle.y + half_h > self.height:
                obstacle.y = self.height - half_h
                obstacle.velocity_y = -abs(obstacle.velocity_y)
