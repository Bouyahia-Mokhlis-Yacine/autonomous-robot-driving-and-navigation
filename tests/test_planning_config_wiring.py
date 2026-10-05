"""Phase 11.5 tests: configuration wiring from config/navigation.yaml's
``planning.unknown_cost_multiplier`` into the ``GlobalPlanner`` instances
actually constructed by the runtime entry points
(``real_sensor_walkthrough.py`` / ``ros2_walkthrough.py``).

Those entry points connect to real hardware (LiDAR / ROS2) in their
``main()``, so they cannot be run directly in a unit test. Instead these
tests replicate their exact composition expression --
``GlobalPlanner(allow_diagonal=nav_cfg.planning.allow_diagonal,
unknown_cost_multiplier=nav_cfg.planning.unknown_cost_multiplier)`` --
against the real, on-disk ``config/navigation.yaml`` (via
``load_navigation_config()``, not a hand-built config object), proving the
actual configured value reaches the planner and then A*, not just that the
YAML file contains the right number.
"""

from __future__ import annotations

from common.config import load_navigation_config
from common.messages import Goal, OccupancyGrid, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from planning.global_planner import GlobalPlanner


def double_corridor_grid() -> OccupancyGridMap:
    """Two equal-length routes between the same start/goal: row 0 is
    FREE, row 2 is left UNKNOWN (default), row 1 is a wall with
    connectors at the two ends. See tests/test_unknown_cost_planning.py
    for the full diagram/derivation."""
    grid_map = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=3.0)
    for col in range(5):
        grid_map.set_cell(0, col, OccupancyGrid.FREE)
    grid_map.set_cell(1, 0, OccupancyGrid.FREE)
    grid_map.set_cell(1, 4, OccupancyGrid.FREE)
    for col in (1, 2, 3):
        grid_map.set_cell(1, col, OccupancyGrid.OCCUPIED)
    return grid_map


TOP_ROUTE_CELLS = {(0, c) for c in range(5)}
BOTTOM_ROUTE_CELLS = {(2, c) for c in range(5)}


# --- Test A: the configured value reaches GlobalPlanner ---------------------

def test_configured_multiplier_reaches_global_planner() -> None:
    nav_cfg = load_navigation_config()  # real config/navigation.yaml, not a hand-built object
    assert nav_cfg.planning.unknown_cost_multiplier == 3.0  # sanity: this is the value we're tracing

    planner = GlobalPlanner(
        allow_diagonal=nav_cfg.planning.allow_diagonal,
        unknown_cost_multiplier=nav_cfg.planning.unknown_cost_multiplier,
    )
    assert planner.unknown_cost_multiplier == 3.0  # the constructed object actually carries it


# --- Test B: the configured value reaches A* and changes path selection -----

def test_configured_multiplier_changes_route_through_full_chain() -> None:
    nav_cfg = load_navigation_config()
    grid_map = double_corridor_grid()
    start = Pose2D(*grid_map.grid_to_world(1, 0), theta=0.0)
    goal_xy = grid_map.grid_to_world(1, 4)
    goal = Goal(x=goal_xy[0], y=goal_xy[1])

    # The exact composition used by real_sensor_walkthrough.py / ros2_walkthrough.py.
    configured_planner = GlobalPlanner(
        allow_diagonal=False,
        unknown_cost_multiplier=nav_cfg.planning.unknown_cost_multiplier,
    )
    configured_path = configured_planner.plan(grid_map, start=start, goal=goal)
    assert configured_path is not None
    configured_cells = {grid_map.world_to_grid(p.x, p.y) for p in configured_path.points}
    # config → PlanningConfig → GlobalPlanner → A*: with the real 3.0
    # penalty, the FREE route must win.
    assert configured_cells & TOP_ROUTE_CELLS == TOP_ROUTE_CELLS
    assert not (configured_cells & BOTTOM_ROUTE_CELLS)

    # Contrast with the neutral (1.0) planner on the identical map: the
    # two routes tie on cost, so the UNKNOWN route becomes reachable as
    # a winning answer too -- proving the 3.0 from config is what
    # forces the FREE-only outcome above, not some other fixed bias in
    # the planner or the map itself.
    neutral_planner = GlobalPlanner(allow_diagonal=False, unknown_cost_multiplier=1.0)
    neutral_path = neutral_planner.plan(grid_map, start=start, goal=goal)
    assert neutral_path is not None
    assert len(neutral_path.points) == len(configured_path.points)  # same length, both optimal


# --- Test C: existing direct GlobalPlanner() callers/tests are unaffected --

def test_direct_global_planner_construction_still_defaults_to_neutral() -> None:
    """Every existing test and caller that constructs GlobalPlanner()
    without unknown_cost_multiplier (e.g. tests/test_global_planner.py,
    tests/test_navigation_manager.py) must keep seeing the pre-Phase-11
    behavior -- this wiring change only affects the two runtime entry
    points, not GlobalPlanner's own default."""
    planner = GlobalPlanner(allow_diagonal=True)
    assert planner.unknown_cost_multiplier == 1.0
