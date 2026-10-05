"""Goal validation: is a requested ``Goal`` even worth attempting to plan to?

Kept separate from ``GlobalPlanner`` so a clearly-invalid goal (out of
bounds, sitting inside an obstacle) can be rejected immediately with a
specific reason, before spending a full A* search to discover the same
thing indirectly as "no path found".
"""

from __future__ import annotations

from common.messages import Goal, OccupancyGrid
from mapping.occupancy_grid import OccupancyGridMap


def validate_goal(grid_map: OccupancyGridMap, goal: Goal) -> tuple[bool, str]:
    """Return (is_valid, reason). ``reason`` is empty when valid."""
    row, col = grid_map.world_to_grid(goal.x, goal.y)
    if not grid_map.in_bounds(row, col):
        return False, f"goal ({goal.x}, {goal.y}) is outside the map bounds"
    if grid_map.get_cell(row, col) == OccupancyGrid.OCCUPIED:
        return False, f"goal ({goal.x}, {goal.y}) is occupied"
    return True, ""
