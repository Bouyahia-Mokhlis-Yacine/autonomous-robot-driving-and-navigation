"""High-level global planning: (start pose, goal) -> world-frame Path.

Composes ``planning.astar`` and ``planning.path`` over a caller-supplied
``OccupancyGridMap`` (normally the Phase 5 inflated planning map, not the
raw SLAM map -- but this module does not inflate anything itself; that
separation is deliberate, per "Do not couple the grid to A*").
"""

from __future__ import annotations

from typing import Callable

from common.messages import Goal, Path, PathPoint, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from planning.astar import astar
from planning.path import grid_path_to_world_path


class GlobalPlanner:
    """Plans a collision-free grid path and converts it to world coordinates.

    ``unknown_cost_multiplier`` (Phase 11): forwarded to ``planning.astar.astar``
    unchanged -- see that module for the full explanation. Default is
    ``1.0`` (UNKNOWN costs the same as FREE, this class's pre-Phase-11
    behavior), matching ``astar()``'s own default so every existing
    caller that doesn't pass this parameter sees identical behavior to
    before. The risk-averse default lives in configuration
    (``common.config.PlanningConfig.unknown_cost_multiplier``), not
    here -- a caller wanting that behavior passes it explicitly.
    """

    def __init__(self, allow_diagonal: bool = True, unknown_cost_multiplier: float = 1.0) -> None:
        self.allow_diagonal = allow_diagonal
        self.unknown_cost_multiplier = unknown_cost_multiplier

    def plan(self, grid_map: OccupancyGridMap, start: Pose2D, goal: Goal, deadline: float | None = None,
             should_abort: Callable[[], bool] | None = None) -> Path | None:
        """Return a world-frame ``Path`` from ``start`` to ``goal``, or None.

        ``deadline``/``should_abort``: forwarded to ``astar``, whose
        ``PlanningTimeout`` propagates (a timed-out search is not "no path").

        None means no path could be produced -- either the start/goal
        cell is out of bounds or occupied, or the goal is genuinely
        unreachable from the start under the current planning map.
        Distinguishing those cases (for user-facing goal validation vs.
        a BLOCKED navigation state) is the caller's responsibility.
        """
        start_cell = grid_map.world_to_grid(start.x, start.y)
        goal_cell = grid_map.world_to_grid(goal.x, goal.y)

        try:
            cells = astar(
                grid_map.grid, start_cell, goal_cell, allow_diagonal=self.allow_diagonal,
                unknown_cost_multiplier=self.unknown_cost_multiplier, deadline=deadline, should_abort=should_abort,
            )
        except ValueError:
            return None

        if cells is None:
            return None

        path = grid_path_to_world_path(grid_map, cells)
        # The final waypoint is the goal cell's *center*, which can be up
        # to half a cell away from the actually-requested Goal(x, y) --
        # significant at coarse resolutions, and it's what goal-tolerance
        # checks compare against. Snap it to the exact goal (the cell was
        # already confirmed free by A*, so this stays collision-free).
        path.points[-1] = PathPoint(x=goal.x, y=goal.y)
        return path
