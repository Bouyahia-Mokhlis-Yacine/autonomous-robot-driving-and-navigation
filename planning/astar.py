"""Grid-based A* search over an ``OccupancyGrid``.

Operates purely in grid (row, col) index space and grid-unit costs --
no knowledge of world coordinates, robots, or the planner that calls it.
``mapping`` produces the grid this module consumes; this module must
never reach back into mapping/inflation internals (kept decoupled per
project constraints).

A cell is passable if it is not ``OCCUPIED``. Both ``FREE`` and
``UNKNOWN`` cells are treated as traversable -- inflation is what
actually keeps the robot away from real obstacles and their safety
margins; A* itself does not additionally forbid unexplored space.

Phase 11 -- risk-aware UNKNOWN traversal: entering an ``UNKNOWN`` cell
costs ``unknown_cost_multiplier`` times as much as entering a ``FREE``
one (same multiplier applied to both orthogonal and diagonal step
costs, preserving their relative geometric ratio). Default is ``1.0``
-- i.e. UNKNOWN costs exactly the same as FREE, identical to this
module's behavior before Phase 11 -- so every existing caller that
doesn't pass this new parameter is completely unaffected. A caller
that wants the risk-averse behavior (this project's
``planning.global_planner.GlobalPlanner``, configured from
``common.config.PlanningConfig.unknown_cost_multiplier``) passes a
value ``> 1.0`` explicitly.

Why the existing octile/Manhattan heuristic remains admissible
unchanged: it is built from the MINIMUM possible per-step cost (1.0
orthogonal, sqrt(2) diagonal -- i.e. the FREE-cell cost). Since
``unknown_cost_multiplier`` can only ever raise a step's actual cost
(enforced: it must be ``>= 1.0``, never lower), the true cost of any
path can never be less than what the heuristic already assumes -- the
heuristic never overestimates, so A* remains correct without any
change to it. This is deliberate, not an oversight: multiplying the
heuristic itself by the penalty (as a first instinct might suggest)
would be wrong, since not every remaining cell toward the goal is
actually UNKNOWN.
"""

from __future__ import annotations

import heapq
import math
import time

from typing import Callable

from common.messages import OccupancyGrid

Cell = tuple[int, int]

_ORTHOGONAL_OFFSETS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL_OFFSETS: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_SQRT2 = math.sqrt(2.0)


class PlanningTimeout(Exception):
    """A* ran past its ``deadline`` -- the request is abandoned, which is
    neither "unreachable" (None) nor an invalid request (ValueError)."""


def _in_bounds(cell: Cell, grid: OccupancyGrid) -> bool:
    row, col = cell
    return 0 <= row < grid.height and 0 <= col < grid.width


def _is_passable(cell: Cell, grid: OccupancyGrid) -> bool:
    row, col = cell
    return bool(grid.cells[row, col] != OccupancyGrid.OCCUPIED)


def _heuristic(a: Cell, b: Cell, allow_diagonal: bool) -> float:
    d_row = abs(a[0] - b[0])
    d_col = abs(a[1] - b[1])
    if not allow_diagonal:
        return float(d_row + d_col)
    # Octile distance: admissible & consistent for 8-connected movement
    # with orthogonal cost 1 and diagonal cost sqrt(2).
    return (d_row + d_col) + (_SQRT2 - 2.0) * min(d_row, d_col)


def astar(
    grid: OccupancyGrid, start: Cell, goal: Cell, allow_diagonal: bool = False,
    unknown_cost_multiplier: float = 1.0, deadline: float | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> list[Cell] | None:
    """Shortest (cost-minimizing) grid-cell path from ``start`` to
    ``goal``, or None if unreachable.

    ``deadline`` (optional, a ``time.monotonic()`` value): raise
    ``PlanningTimeout`` once the search is still running past it --
    bounds the worst case (an unreachable goal searches every reachable
    cell). None (the default) never times out. ``should_abort``
    (optional): polled at the same points; True also raises
    ``PlanningTimeout`` (the request was superseded).

    ``unknown_cost_multiplier`` (Phase 11): the cost of entering an
    UNKNOWN cell is this many times a FREE cell's cost (default 1.0 =
    identical cost, i.e. this module's pre-Phase-11 behavior). Must be
    ``>= 1.0`` -- UNKNOWN must never be cheaper than FREE (that would
    perversely route the planner *toward* unexplored space) and this
    is also what keeps the existing distance heuristic admissible (see
    module docstring); raises ``ValueError`` otherwise. OCCUPIED
    remains completely impassable regardless of this parameter.

    Raises ``ValueError`` if ``start``/``goal`` are out of bounds or
    occupied -- that is an invalid request, distinct from a valid,
    in-bounds, free goal that simply cannot be reached (which returns
    None).
    """
    if unknown_cost_multiplier < 1.0:
        raise ValueError("unknown_cost_multiplier must be >= 1.0 (UNKNOWN must not be cheaper than FREE)")

    for name, cell in (("start", start), ("goal", goal)):
        if not _in_bounds(cell, grid):
            raise ValueError(f"{name} cell {cell} is out of grid bounds")
        if not _is_passable(cell, grid):
            raise ValueError(f"{name} cell {cell} is occupied")

    if start == goal:
        return [start]

    # Same search as the straightforward form (per-cell tuples, a visited
    # set, _neighbors/_heuristic calls), on flat cell indices and
    # precomputed lists instead: ~3-4x faster, identical result. Heap
    # entries are (f, g, row * width + col), which order exactly like
    # (f, g, (row, col)) since col < width, so ties break identically;
    # step costs and the heuristic are the same float expressions.
    # tests/test_astar_equivalence.py checks it against the original.
    height, width = grid.height, grid.width
    passable = (grid.cells != OccupancyGrid.OCCUPIED).ravel().tolist()
    unknown = (grid.cells == OccupancyGrid.UNKNOWN).ravel().tolist()
    orth_cost = (1.0 * 1.0, 1.0 * unknown_cost_multiplier)          # [FREE, UNKNOWN]
    diag_cost = (_SQRT2 * 1.0, _SQRT2 * unknown_cost_multiplier)
    octile = _SQRT2 - 2.0
    goal_row, goal_col = goal
    goal_idx = goal_row * width + goal_col
    start_idx = start[0] * width + start[1]
    orthogonal = [(d_row, d_col, d_row * width + d_col) for d_row, d_col in _ORTHOGONAL_OFFSETS]
    diagonal = [(d_row, d_col, d_row * width + d_col) for d_row, d_col in _DIAGONAL_OFFSETS] if allow_diagonal else []

    g_score = [math.inf] * (height * width)
    g_score[start_idx] = 0.0
    came_from: dict[int, int] = {}
    closed = bytearray(height * width)
    open_heap: list[tuple[float, float, int]] = [(_heuristic(start, goal, allow_diagonal), 0.0, start_idx)]
    heappop, heappush = heapq.heappop, heapq.heappush
    expansions = 0

    while open_heap:
        _, g, current = heappop(open_heap)
        if closed[current]:
            continue
        closed[current] = 1

        if current == goal_idx:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return [divmod(idx, width) for idx in path]

        expansions += 1
        if not expansions & 1023:
            if deadline is not None and time.monotonic() > deadline:
                raise PlanningTimeout(f"A* exceeded its deadline after {expansions} expansions")
            if should_abort is not None and should_abort():
                raise PlanningTimeout(f"A* aborted after {expansions} expansions")

        row, col = divmod(current, width)
        for d_row, d_col, d_idx in orthogonal:
            n_row, n_col = row + d_row, col + d_col
            if not (0 <= n_row < height and 0 <= n_col < width):
                continue
            neighbor = current + d_idx
            if not passable[neighbor] or closed[neighbor]:
                continue
            tentative_g = g + orth_cost[unknown[neighbor]]
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                dr, dc = abs(n_row - goal_row), abs(n_col - goal_col)
                h = (dr + dc) + octile * min(dr, dc) if allow_diagonal else float(dr + dc)
                heappush(open_heap, (tentative_g + h, tentative_g, neighbor))
        for d_row, d_col, d_idx in diagonal:
            n_row, n_col = row + d_row, col + d_col
            if not (0 <= n_row < height and 0 <= n_col < width):
                continue
            neighbor = current + d_idx
            # Forbid cutting a corner between two blocked orthogonal cells
            # (the robot cannot physically pass through that gap).
            if (not passable[neighbor] or closed[neighbor]
                    or not passable[current + d_row * width] or not passable[current + d_col]):
                continue
            tentative_g = g + diag_cost[unknown[neighbor]]
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                dr, dc = abs(n_row - goal_row), abs(n_col - goal_col)
                heappush(open_heap, (tentative_g + ((dr + dc) + octile * min(dr, dc)), tentative_g, neighbor))

    return None
