"""Rectangular obstacle shapes for the 2D simulation world.

Obstacles are axis-aligned rectangles defined in world coordinates. This
module holds only geometry + (for dynamic obstacles) motion integration —
no planning, no collision response. Collision checks against the robot
footprint live in ``simulation.environment``.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.geometry import Point, rectangle_corners


@dataclass
class RectangleObstacle:
    """A static axis-aligned rectangular obstacle."""

    x: float       # center, world meters
    y: float       # center, world meters
    width: float   # extent along world x, meters
    height: float  # extent along world y, meters

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Obstacle width and height must be positive")

    def corners(self) -> list[Point]:
        return rectangle_corners(self.x, self.y, self.width, self.height, 0.0)


@dataclass
class DynamicObstacle(RectangleObstacle):
    """A rectangular obstacle that moves at a constant velocity.

    Boundary handling (e.g. bouncing off world walls) is the
    ``Environment``'s responsibility, since only it knows the world
    bounds; this class only integrates position from velocity.
    """

    velocity_x: float = 0.0
    velocity_y: float = 0.0

    def step(self, dt: float) -> None:
        if dt < 0.0:
            raise ValueError("dt must be >= 0")
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt
