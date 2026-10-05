"""Phase 13: recovery from the Phase 12.5 root cause -- the robot's own
current cell becoming OCCUPIED in the *inflated* planning map while the
*raw* (uninflated) world-model map still says that same cell is FREE.

This is not a collision and not a genuine "no path exists" situation: it
means the robot's estimated position is merely inside its own configured
inflation/safety buffer around a confirmed obstacle. ``GlobalPlanner``
correctly refuses to plan from an occupied start cell (that refusal is
NOT weakened here -- see ``planning/astar.py``, unmodified); this module
detects the specific condition and picks a short, safety-checked reverse
target so ``NavigationManager`` can physically move the robot out of its
own buffer before re-attempting normal planning.

Deliberately narrow: the only recovery motion considered is straight-line
reverse along the robot's current heading (``pose.theta + pi``), checked
for a clear line of sight against the RAW grid over a short, configured
distance via the existing ``planning.path_smoothing.has_line_of_sight``
(no new raycasting logic). If that direction is not safe, this module
reports "no safe recovery" rather than choosing a blind or arbitrary
alternative -- the caller then falls through to normal planning, which
will correctly re-report ``BLOCKED`` (see ``navigation_manager.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from common.geometry import wrap_angle
from common.messages import OccupancyGrid, PathPoint, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from planning.path_smoothing import has_line_of_sight


@dataclass
class RecoveryConfig:
    """Phase 13 tuning parameters. Conservative initial defaults, not
    claimed to be optimal -- consistent with every other "initial,
    reasoned default for empirical tuning" in this project."""

    # Well under the project-wide max_linear_velocity (0.50 m/s) -- also
    # independently clamped by whatever RobotInterface actually executes
    # the command (e.g. SimulatedRobot), so this is a deliberate choice,
    # not the only safety backstop.
    linear_speed: float = 0.15
    # How far behind the robot must be clear (raw grid) for the reverse
    # direction to be considered safe -- and, since speed is held
    # constant for the whole maneuver, this doubles as the recovery's
    # intended travel distance (linear_speed * max_duration, below).
    # Deliberately set comfortably ABOVE this project's default
    # inflation radius (~0.57m for config/robot.yaml's dimensions, see
    # mapping.inflation.compute_inflation_radius): backing up LESS than
    # one inflation radius can leave the robot still within the same
    # obstacle's buffer, where the very next replanned (taut,
    # corner-hugging) path immediately re-approaches the same boundary
    # -- observed empirically as a slow repeating recovery cycle at the
    # smaller 0.3m default. Backing up past the full buffer radius gives
    # the replanner genuine room to find a different approach.
    probe_distance: float = 0.8
    max_duration: float = 6.0  # seconds, hard cap on one recovery attempt (0.15 m/s * 6s ~= probe_distance)
    max_attempts: int = 3      # consecutive attempts before giving up (falls through to BLOCKED)
    # Shallow buffer intrusion (see ``buffer_escape_point``): if a buffer-free
    # cell lies within this distance, plan from it instead of reversing.
    # Should equal the robot's SAFETY MARGIN (config/robot.yaml
    # safety.inflation_radius, 0.25 m) -- the part of the inflation radius
    # beyond the robot's own bounding circle -- so an escape is only ever
    # taken while the bounding circle itself is still clear of obstacles.
    # 0 disables it (every start inside the buffer reverses, as before).
    max_escape_distance: float = 0.25

    def __post_init__(self) -> None:
        if self.linear_speed <= 0.0:
            raise ValueError("linear_speed must be > 0")
        if self.probe_distance <= 0.0:
            raise ValueError("probe_distance must be > 0")
        if self.max_duration <= 0.0:
            raise ValueError("max_duration must be > 0")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.max_escape_distance < 0.0:
            raise ValueError("max_escape_distance must be >= 0")


def inflated_start_cell_blocked(
    pose: Pose2D, raw_grid_map: OccupancyGridMap, planning_grid_map: OccupancyGridMap
) -> bool:
    """True iff the cell containing ``pose`` is OCCUPIED in the inflated
    planning map but FREE in the raw (uninflated) world-model map --
    the exact Phase 12.5 failure condition.

    Deliberately conservative: returns False (do not trigger recovery)
    if the two maps don't share the same resolution/dimensions (so a
    single (row, col) pair can't safely address both), or if that cell
    is out of bounds in either map -- "if any of these conditions cannot
    be established reliably, do not trigger this recovery," per the
    Phase 13 spec.
    """
    if (
        raw_grid_map.grid.resolution != planning_grid_map.grid.resolution
        or raw_grid_map.grid.width != planning_grid_map.grid.width
        or raw_grid_map.grid.height != planning_grid_map.grid.height
        or raw_grid_map.grid.origin_x != planning_grid_map.grid.origin_x
        or raw_grid_map.grid.origin_y != planning_grid_map.grid.origin_y
    ):
        return False

    row, col = raw_grid_map.world_to_grid(pose.x, pose.y)
    if not raw_grid_map.in_bounds(row, col):
        return False

    raw_state = raw_grid_map.get_cell(row, col)
    planning_state = planning_grid_map.get_cell(row, col)
    return planning_state == OccupancyGrid.OCCUPIED and raw_state == OccupancyGrid.FREE


def reverse_target_point(pose: Pose2D, distance: float) -> PathPoint:
    """The world point ``distance`` meters directly behind ``pose``
    (i.e. along ``pose.theta + pi``)."""
    angle = wrap_angle(pose.theta + math.pi)
    return PathPoint(x=pose.x + distance * math.cos(angle), y=pose.y + distance * math.sin(angle))


def recovery_direction_is_safe(pose: Pose2D, raw_grid_map: OccupancyGridMap, distance: float) -> bool:
    """True iff the straight reverse segment from ``pose`` out to
    ``reverse_target_point`` crosses no RAW-occupied cell -- i.e. it is
    genuinely safe to move into, not merely outside the inflation
    buffer. Reuses the existing line-of-sight check rather than a new
    raycasting implementation."""
    target = reverse_target_point(pose, distance)
    origin = PathPoint(x=pose.x, y=pose.y)
    return has_line_of_sight(raw_grid_map, origin, target)


def buffer_escape_point(
    pose: Pose2D, raw_grid_map: OccupancyGridMap, planning_grid_map: OccupancyGridMap, max_distance: float
) -> PathPoint | None:
    """Nearest cell centre OUTSIDE the inflation buffer that the robot can
    drive straight to, for a start pose that is only shallowly inside it.

    Why: the planner's paths run along the inflated boundary, so ordinary
    tracking error (a few cm) regularly puts the robot's own cell inside the
    buffer while its real footprint is still well clear of any obstacle.
    Reversing ~0.8 m (``RECOVERING``) in that situation does not change the
    route geometry -- the next plan hugs the same boundary and the robot
    returns to the same spot. Planning from the nearest buffer-free cell
    instead is how a planner leaves its own inflation: the robot simply moves
    the few centimetres out as the first part of the new path.

    A candidate must be: non-occupied in the planning (inflated) map, FREE
    (observed, not unknown) in the raw map, within ``max_distance`` of
    ``pose``, and reachable by a straight segment that crosses no
    raw-occupied cell. Returns the nearest such cell centre, or ``None``
    (the caller then falls back to the reverse recovery). Deterministic.
    """
    if max_distance <= 0.0:
        return None
    raw, plan = raw_grid_map.grid, planning_grid_map.grid
    if (raw.resolution, raw.width, raw.height, raw.origin_x, raw.origin_y) != (
        plan.resolution, plan.width, plan.height, plan.origin_x, plan.origin_y
    ):
        return None
    res = raw.resolution
    r0, c0 = raw_grid_map.world_to_grid(pose.x, pose.y)
    reach = int(math.ceil(max_distance / res)) + 1
    rows = np.arange(max(0, r0 - reach), min(raw.height, r0 + reach + 1))
    cols = np.arange(max(0, c0 - reach), min(raw.width, c0 + reach + 1))
    if rows.size == 0 or cols.size == 0:
        return None
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    xs = raw.origin_x + (cc + 0.5) * res
    ys = raw.origin_y + (rr + 0.5) * res
    dist = np.hypot(xs - pose.x, ys - pose.y)
    ok = (
        (plan.cells[rr, cc] != OccupancyGrid.OCCUPIED)
        & (raw.cells[rr, cc] == OccupancyGrid.FREE)
        & (dist <= max_distance)
    )
    origin = PathPoint(x=pose.x, y=pose.y)
    for i in np.argsort(dist[ok], kind="stable"):
        candidate = PathPoint(x=float(xs[ok][i]), y=float(ys[ok][i]))
        if has_line_of_sight(raw_grid_map, origin, candidate):
            return candidate
    return None
