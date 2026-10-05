"""Pure, GUI-toolkit-independent geometry: screen<->world coordinate
conversion and drawing-primitive computation.

Nothing in this module imports a GUI library or touches navigation
state -- it only computes numbers from messages (Pose2D, LaserScan,
Path). That keeps the screen<->world transform (explicitly required to
be tested) testable headlessly, and keeps ``visualization.gui`` a thin
wiring layer instead of where the actual math lives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from common.messages import LaserScan, Path, Pose2D


@dataclass(frozen=True)
class ViewTransform:
    """Maps world meters <-> canvas pixels.

    World y grows upward; screen/canvas y grows downward, so the
    transform flips y about ``canvas_height``. ``world_origin_*`` is the
    world point that maps to the canvas's bottom-left corner.
    """

    world_origin_x: float
    world_origin_y: float
    scale: float          # pixels per meter
    canvas_height: int    # pixels

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("scale must be > 0")
        if self.canvas_height <= 0:
            raise ValueError("canvas_height must be > 0")


def world_to_screen(x: float, y: float, view: ViewTransform) -> tuple[int, int]:
    px = (x - view.world_origin_x) * view.scale
    py = view.canvas_height - (y - view.world_origin_y) * view.scale
    return int(round(px)), int(round(py))


def screen_to_world(px: float, py: float, view: ViewTransform) -> tuple[float, float]:
    x = px / view.scale + view.world_origin_x
    y = (view.canvas_height - py) / view.scale + view.world_origin_y
    return x, y


def heading_line_endpoints(pose: Pose2D, length: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """World-frame (start, end) points for a robot heading indicator."""
    end_x = pose.x + length * math.cos(pose.theta)
    end_y = pose.y + length * math.sin(pose.theta)
    return (pose.x, pose.y), (end_x, end_y)


def scan_hit_points_world(pose: Pose2D, scan: LaserScan) -> list[tuple[float, float]]:
    """World-frame points for each ray that actually hit something.

    A ray counts as a hit iff ``range_min <= r < range_max`` -- matching
    the exact hit criterion used everywhere else in this project
    (``slam.scan_preprocessing.preprocess_scan``,
    ``mapping.occupancy_grid.OccupancyGridMap.update_from_scan``,
    ``mapping.persistent_world_model.PersistentWorldModel.integrate``).
    Max-range rays (``r >= range_max``, "nothing detected") and invalid
    rays (``r < range_min``) are both excluded -- this project's own
    Phase 5/6/7 test scans routinely fill non-hit bearings with an
    explicit below-range_min sentinel (not ``range_max``), which the
    original range_max-only check here would have silently mis-rendered
    as real hits at range 0.
    """
    points: list[tuple[float, float]] = []
    for i in range(scan.num_rays):
        r = scan.ranges[i]
        if r >= scan.range_max or r < scan.range_min:
            continue
        angle = pose.theta + scan.angle_at(i)
        points.append((pose.x + r * math.cos(angle), pose.y + r * math.sin(angle)))
    return points


def path_screen_points(path: Path, view: ViewTransform) -> list[tuple[int, int]]:
    """Convenience: every path waypoint converted to screen coordinates."""
    return [world_to_screen(p.x, p.y, view) for p in path.points]


# ---------------------------------------------------- orientation check view --
# Robot frame (used everywhere in this project): +x = robot FRONT, +y = robot
# LEFT, angles counter-clockwise from +x. These helpers only DRAW that
# convention; they never change it.

def robot_to_world(pose: Pose2D, x_robot: float, y_robot: float) -> tuple[float, float]:
    """A point given in the robot frame -> the world (map) frame."""
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    return pose.x + c * x_robot - s * y_robot, pose.y + s * x_robot + c * y_robot


def world_to_robot(pose: Pose2D, x: float, y: float) -> tuple[float, float]:
    """A world (map) point -> the robot frame (x FRONT, y LEFT)."""
    dx, dy = x - pose.x, y - pose.y
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    return c * dx + s * dy, -s * dx + c * dy


def footprint_corners_world(pose: Pose2D, length: float, width: float) -> list[tuple[float, float]]:
    """The robot's rectangle (centred on the pose) in the world frame:
    front-left, front-right, rear-right, rear-left."""
    hl, hw = length / 2.0, width / 2.0
    return [robot_to_world(pose, x, y) for x, y in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def direction_label(bearing: float) -> str:
    """Robot-frame bearing (radians, CCW from FRONT) -> FRONT / LEFT / REAR / RIGHT
    (90-degree sectors centred on each direction)."""
    deg = math.degrees(math.atan2(math.sin(bearing), math.cos(bearing)))
    if -45.0 <= deg <= 45.0:
        return "FRONT"
    if 45.0 < deg < 135.0:
        return "LEFT"
    if -135.0 < deg < -45.0:
        return "RIGHT"
    return "REAR"


@dataclass(frozen=True)
class ForwardUpView:
    """World (map) frame -> canvas with map +x pointing UP and map +y pointing
    LEFT, centred on (center_x, center_y). The SLAM map frame is born at the
    robot's startup pose, so at startup the robot's FRONT points up and its
    LEFT points left -- the same picture as the orientation-check view."""

    center_x: float
    center_y: float
    scale: float   # pixels per metre
    side: int      # canvas is side x side pixels

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        half = self.side / 2.0
        return (int(round(half - (y - self.center_y) * self.scale)),
                int(round(half - (x - self.center_x) * self.scale)))

    def to_world(self, px: float, py: float) -> tuple[float, float]:
        half = self.side / 2.0
        return self.center_x + (half - py) / self.scale, self.center_y + (half - px) / self.scale


def cells_forward_up(cells):
    """Grid array (row = y ascending, col = x ascending) -> image array for a
    ForwardUpView: row 0 = largest x (top), col 0 = largest y (left)."""
    return cells[::-1, ::-1].swapaxes(0, 1)
