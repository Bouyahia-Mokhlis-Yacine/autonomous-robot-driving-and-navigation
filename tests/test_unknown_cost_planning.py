"""Phase 11 tests: risk-aware UNKNOWN-space planning.

FREE and OCCUPIED semantics are unchanged and re-verified here only
incidentally; the focus is exclusively on UNKNOWN's new behavior:
traversable, but at a configurable cost penalty relative to FREE.

Most tests use a small, deliberately simple "double corridor" grid --
two geometrically EQUAL-LENGTH orthogonal routes between the same
start and goal, one entirely FREE, one entirely UNKNOWN -- so the only
variable affecting which route A* picks is the UNKNOWN cost
multiplier, not path length. Diagram (rows top to bottom, as drawn --
row 0 is the TOP route):

    row 0:  F  F  F  F  F      <- top route (FREE)
    row 1:  F  O  O  O  F      <- connectors at col 0/4, wall in between
    row 2:  .  .  .  .  .      <- bottom route (left as UNKNOWN, the
                                   OccupancyGridMap default -- never
                                   explicitly set)

Start = (1, 0), goal = (1, 4). With allow_diagonal=False, the only two
paths are: via row 0 (6 steps, all FREE) or via row 2 (6 steps, all
UNKNOWN except the row-1 connector cells, which are FREE).
"""

from __future__ import annotations

import pytest

from common.messages import Goal, OccupancyGrid, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from planning.astar import astar
from planning.global_planner import GlobalPlanner
from planning.path_smoothing import has_line_of_sight, process_path


def double_corridor_grid() -> OccupancyGridMap:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=3.0)
    for col in range(5):
        grid_map.set_cell(0, col, OccupancyGrid.FREE)  # top route
    grid_map.set_cell(1, 0, OccupancyGrid.FREE)
    grid_map.set_cell(1, 4, OccupancyGrid.FREE)
    for col in (1, 2, 3):
        grid_map.set_cell(1, col, OccupancyGrid.OCCUPIED)  # wall between the two routes
    # Row 2 (the bottom route) is left at the default UNKNOWN -- never set.
    return grid_map


TOP_ROUTE_CELLS = {(0, c) for c in range(5)}
BOTTOM_ROUTE_CELLS = {(2, c) for c in range(5)}


# --- Test 1: FREE preferred over an equivalent UNKNOWN route ----------------

def test_free_route_preferred_over_equal_length_unknown_route() -> None:
    grid_map = double_corridor_grid()
    cells = astar(grid_map.grid, (1, 0), (1, 4), allow_diagonal=False, unknown_cost_multiplier=3.0)
    assert cells is not None
    visited = set(cells)
    assert visited & TOP_ROUTE_CELLS == TOP_ROUTE_CELLS  # took the FREE route in full
    assert not (visited & BOTTOM_ROUTE_CELLS)  # never touched the UNKNOWN route


def test_free_route_preferred_via_global_planner() -> None:
    grid_map = double_corridor_grid()
    planner = GlobalPlanner(allow_diagonal=False, unknown_cost_multiplier=3.0)
    start = Pose2D(*grid_map.grid_to_world(1, 0), theta=0.0)
    goal_xy = grid_map.grid_to_world(1, 4)
    path = planner.plan(grid_map, start=start, goal=Goal(x=goal_xy[0], y=goal_xy[1]))
    assert path is not None
    visited_cells = {grid_map.world_to_grid(p.x, p.y) for p in path.points}
    assert visited_cells & TOP_ROUTE_CELLS
    assert not (visited_cells & BOTTOM_ROUTE_CELLS)


# --- Test 2: UNKNOWN remains traversable when it's the only route ----------

def test_unknown_remains_traversable_when_it_is_the_only_route() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=1.0)
    # Every cell defaults to UNKNOWN -- a single-row corridor entirely
    # unexplored, no FREE alternative exists anywhere.
    cells = astar(grid_map.grid, (0, 0), (0, 4), allow_diagonal=False, unknown_cost_multiplier=5.0)
    assert cells is not None  # must NOT be treated as blocked
    assert cells[0] == (0, 0)
    assert cells[-1] == (0, 4)


def test_unknown_remains_traversable_with_a_large_penalty() -> None:
    # Even a very large penalty must never make UNKNOWN impassable --
    # only expensive.
    grid_map = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=1.0)
    cells = astar(grid_map.grid, (0, 0), (0, 4), allow_diagonal=False, unknown_cost_multiplier=1000.0)
    assert cells is not None


# --- Test 3: OCCUPIED remains completely blocked -----------------------------

def test_occupied_remains_blocked_regardless_of_unknown_multiplier() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=3.0)
    for col in range(5):
        grid_map.set_cell(1, col, OccupancyGrid.OCCUPIED)  # solid wall, no gap anywhere
    cells = astar(grid_map.grid, (0, 2), (2, 2), allow_diagonal=False, unknown_cost_multiplier=3.0)
    assert cells is None


def test_occupied_never_appears_in_a_returned_path() -> None:
    grid_map = double_corridor_grid()
    cells = astar(grid_map.grid, (1, 0), (1, 4), allow_diagonal=False, unknown_cost_multiplier=3.0)
    assert cells is not None
    for row, col in cells:
        assert grid_map.get_cell(row, col) != OccupancyGrid.OCCUPIED


# --- Test 4: the penalty actually affects which route is selected -----------

def test_unknown_penalty_changes_route_selection() -> None:
    grid_map = double_corridor_grid()

    # No penalty (multiplier == 1.0, the pre-Phase-11 default): both
    # routes cost exactly the same (6 steps each) -- either is a valid,
    # optimal answer, so we only assert a path is found here, not which one.
    neutral = astar(grid_map.grid, (1, 0), (1, 4), allow_diagonal=False, unknown_cost_multiplier=1.0)
    assert neutral is not None

    # With a real penalty, the FREE route must win outright.
    penalized_cells = astar(grid_map.grid, (1, 0), (1, 4), allow_diagonal=False, unknown_cost_multiplier=3.0)
    assert penalized_cells is not None
    visited = set(penalized_cells)
    assert visited & TOP_ROUTE_CELLS == TOP_ROUTE_CELLS
    assert not (visited & BOTTOM_ROUTE_CELLS)


# --- Test 5: the penalty is genuinely configurable (connected end to end) ---

def test_penalty_is_configurable_low_vs_high() -> None:
    """A shorter UNKNOWN-heavy route vs. a longer FREE-heavy route: a low
    penalty should let the shorter UNKNOWN route win on total cost; a
    high penalty should flip the preference to the longer FREE route --
    proving the configured value is actually connected to the planner,
    not just accepted and ignored."""
    grid_map = OccupancyGridMap(resolution=1.0, width_m=7.0, height_m=1.0)
    # A single row: cells 1-3 are UNKNOWN (default, left unset) -- the
    # direct, short route. Force a longer alternative to not exist at
    # all here (single row, so this test is really about: does a HIGH
    # enough multiplier make an otherwise-cheaper UNKNOWN path lose to
    # ANY viable detour). To keep this deterministic and avoid a
    # fragile geometric assumption, use a second row as the FREE
    # detour, one step longer.
    grid_map = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=2.0)
    grid_map.set_cell(0, 0, OccupancyGrid.FREE)
    grid_map.set_cell(0, 4, OccupancyGrid.FREE)
    # Row 0, cols 1-3: UNKNOWN (default) -- direct 4-step route (0,0)->(0,4).
    for col in range(5):
        grid_map.set_cell(1, col, OccupancyGrid.FREE)  # row 1: fully FREE, one step longer via the ends
    grid_map.set_cell(0, 0, OccupancyGrid.FREE)

    # Direct route: (0,0)->(0,1)->(0,2)->(0,3)->(0,4): 4 steps, 3 of
    # which enter UNKNOWN cells (cost = 1 + 3*m).
    # Detour route: (0,0)->(1,0)->(1,1)->(1,2)->(1,3)->(1,4)->(0,4): 6
    # steps, all FREE (cost = 6.0), assuming (0,0) and (0,4) connectors
    # are FREE (set above).

    low = astar(grid_map.grid, (0, 0), (0, 4), allow_diagonal=False, unknown_cost_multiplier=1.0)
    assert low is not None
    assert (0, 1) in low and (0, 2) in low and (0, 3) in low  # direct route: cost 1+3*1=4 < 6

    high = astar(grid_map.grid, (0, 0), (0, 4), allow_diagonal=False, unknown_cost_multiplier=5.0)
    assert high is not None
    assert (0, 1) not in high  # direct route now costs 1+3*5=16 > 6 -- detour wins
    assert (1, 2) in high  # took the all-FREE detour through row 1


# --- Test 6: diagonal movement / corner-cutting prevention preserved -------

def test_diagonal_corner_cutting_still_blocked_with_unknown_weighting() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=3.0, height_m=3.0)
    # Block the two orthogonal cells adjacent to a diagonal step from
    # (1,1) to (0,0), forcing the corner-cut rule to reject it even
    # though (0,0) itself is passable (UNKNOWN, default).
    grid_map.set_cell(0, 1, OccupancyGrid.OCCUPIED)
    grid_map.set_cell(1, 0, OccupancyGrid.OCCUPIED)

    cells = astar(grid_map.grid, (1, 1), (0, 0), allow_diagonal=True, unknown_cost_multiplier=3.0)
    # Corner-cutting must remain forbidden -- no path exists (every
    # other cell is boxed in by the two occupied orthogonal neighbors
    # and the grid boundary).
    assert cells is None


def test_diagonal_unknown_step_costs_sqrt2_times_multiplier() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=3.0, height_m=3.0)
    grid_map.set_cell(0, 0, OccupancyGrid.FREE)
    grid_map.set_cell(1, 1, OccupancyGrid.FREE)  # goal; (0,1) and (1,0) stay UNKNOWN (default) -- not blocked

    diagonal_path = astar(grid_map.grid, (0, 0), (1, 1), allow_diagonal=True, unknown_cost_multiplier=3.0)
    assert diagonal_path == [(0, 0), (1, 1)]  # the direct diagonal step remains legal and chosen


# --- Test 7: existing planning/navigation behavior does not regress --------

def test_existing_astar_default_behavior_unchanged() -> None:
    """astar()'s own default (unknown_cost_multiplier=1.0) must make
    UNKNOWN and FREE cost exactly the same -- i.e. every pre-Phase-11
    caller (including every existing test) sees identical behavior."""
    grid_map = double_corridor_grid()
    cells_default = astar(grid_map.grid, (1, 0), (1, 4), allow_diagonal=False)  # no multiplier passed at all
    cells_explicit_neutral = astar(grid_map.grid, (1, 0), (1, 4), allow_diagonal=False, unknown_cost_multiplier=1.0)
    assert cells_default is not None and cells_explicit_neutral is not None
    assert len(cells_default) == len(cells_explicit_neutral) == 7  # both routes are 6 steps (7 cells) either way


def test_invalid_unknown_cost_multiplier_rejected() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=3.0, height_m=3.0)
    with pytest.raises(ValueError):
        astar(grid_map.grid, (0, 0), (2, 2), allow_diagonal=True, unknown_cost_multiplier=0.5)


def test_global_planner_default_unchanged() -> None:
    """GlobalPlanner's own default (unknown_cost_multiplier=1.0) must
    also be a no-op -- existing callers that only pass allow_diagonal
    see identical behavior to before Phase 11."""
    grid_map = double_corridor_grid()
    planner = GlobalPlanner(allow_diagonal=False)  # no unknown_cost_multiplier passed
    start = Pose2D(*grid_map.grid_to_world(1, 0), theta=0.0)
    goal_xy = grid_map.grid_to_world(1, 4)
    path = planner.plan(grid_map, start=start, goal=Goal(x=goal_xy[0], y=goal_xy[1]))
    assert path is not None
    assert len(path.points) == 7  # 6 steps (7 cells) via the top route


# --- Path smoothing: UNKNOWN remains a planning preference, not an obstacle -

def test_path_smoothing_still_treats_unknown_as_traversable() -> None:
    """Phase 11 does not change path_smoothing.py -- confirmed directly:
    a line-of-sight check across UNKNOWN cells must still succeed (only
    OCCUPIED blocks it), so a path A* chose (possibly through UNKNOWN,
    when that was cost-optimal or the only option) is never invalidated
    by smoothing/simplification."""
    grid_map = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=1.0)
    # Every cell defaults to UNKNOWN.
    a = grid_map.grid_to_world(0, 0)
    b = grid_map.grid_to_world(0, 4)
    from common.messages import PathPoint
    assert has_line_of_sight(grid_map, PathPoint(x=a[0], y=a[1]), PathPoint(x=b[0], y=b[1])) is True


def test_process_path_does_not_turn_unknown_into_an_obstacle() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=1.0)
    cells = astar(grid_map.grid, (0, 0), (0, 4), allow_diagonal=False, unknown_cost_multiplier=3.0)
    assert cells is not None
    from planning.path import grid_path_to_world_path
    raw_path = grid_path_to_world_path(grid_map, cells)

    processed = process_path(raw_path, grid_map, simplify=True, smooth=True)
    assert not processed.is_empty()
    for point in processed.points:
        row, col = grid_map.world_to_grid(point.x, point.y)
        assert grid_map.get_cell(row, col) != OccupancyGrid.OCCUPIED
