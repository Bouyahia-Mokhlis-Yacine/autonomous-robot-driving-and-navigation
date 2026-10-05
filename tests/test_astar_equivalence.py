"""The optimized ``planning.astar.astar`` must return exactly what the original
implementation returned (same cells, same tie-breaking), on random grids and
on a real room map recorded from the RPLiDAR on the Pi. ``_reference_astar``
below is the original code, verbatim apart from its name."""

from __future__ import annotations

import heapq
import math
import time
from pathlib import Path

import numpy as np
import pytest

from common.messages import OccupancyGrid
from planning.astar import (
    _DIAGONAL_OFFSETS,
    _ORTHOGONAL_OFFSETS,
    _SQRT2,
    Cell,
    PlanningTimeout,
    _heuristic,
    _in_bounds,
    _is_passable,
    astar,
)

FIXTURE = Path(__file__).parent / "fixtures" / "live_room_map_2026-09-28.npz"


def _step_cost_multiplier(cell: Cell, grid: OccupancyGrid, unknown_cost_multiplier: float) -> float:
    """Cost multiplier for MOVING INTO ``cell``: ``unknown_cost_multiplier``
    if it is UNKNOWN, ``1.0`` otherwise (FREE, or any other non-OCCUPIED
    value -- OCCUPIED cells never reach here, filtered by
    ``_is_passable`` before this is called)."""
    row, col = cell
    if grid.cells[row, col] == OccupancyGrid.UNKNOWN:
        return unknown_cost_multiplier
    return 1.0


def _neighbors(
    cell: Cell, grid: OccupancyGrid, allow_diagonal: bool, unknown_cost_multiplier: float
) -> list[tuple[Cell, float]]:
    row, col = cell
    results: list[tuple[Cell, float]] = []

    for d_row, d_col in _ORTHOGONAL_OFFSETS:
        neighbor = (row + d_row, col + d_col)
        if _in_bounds(neighbor, grid) and _is_passable(neighbor, grid):
            cost = 1.0 * _step_cost_multiplier(neighbor, grid, unknown_cost_multiplier)
            results.append((neighbor, cost))

    if allow_diagonal:
        for d_row, d_col in _DIAGONAL_OFFSETS:
            neighbor = (row + d_row, col + d_col)
            if not _in_bounds(neighbor, grid) or not _is_passable(neighbor, grid):
                continue
            # Forbid cutting a corner between two blocked orthogonal cells
            # (the robot cannot physically pass through that gap).
            orth_a, orth_b = (row + d_row, col), (row, col + d_col)
            if not _is_passable(orth_a, grid) or not _is_passable(orth_b, grid):
                continue
            cost = _SQRT2 * _step_cost_multiplier(neighbor, grid, unknown_cost_multiplier)
            results.append((neighbor, cost))

    return results


def _reconstruct_path(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path



def _reference_astar(
    grid: OccupancyGrid, start: Cell, goal: Cell, allow_diagonal: bool = False,
    unknown_cost_multiplier: float = 1.0,
) -> list[Cell] | None:
    if unknown_cost_multiplier < 1.0:
        raise ValueError("unknown_cost_multiplier must be >= 1.0 (UNKNOWN must not be cheaper than FREE)")

    for name, cell in (("start", start), ("goal", goal)):
        if not _in_bounds(cell, grid):
            raise ValueError(f"{name} cell {cell} is out of grid bounds")
        if not _is_passable(cell, grid):
            raise ValueError(f"{name} cell {cell} is occupied")

    if start == goal:
        return [start]

    g_score: dict[Cell, float] = {start: 0.0}
    came_from: dict[Cell, Cell] = {}
    open_heap: list[tuple[float, float, Cell]] = [(_heuristic(start, goal, allow_diagonal), 0.0, start)]
    visited: set[Cell] = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return _reconstruct_path(came_from, current)

        for neighbor, step_cost in _neighbors(current, grid, allow_diagonal, unknown_cost_multiplier):
            if neighbor in visited:
                continue
            tentative_g = g + step_cost
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + _heuristic(neighbor, goal, allow_diagonal)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    return None


def _grid(cells: np.ndarray) -> OccupancyGrid:
    h, w = cells.shape
    return OccupancyGrid(resolution=0.05, width=w, height=h, origin_x=0.0, origin_y=0.0, cells=cells.astype(np.int8))


def _pick_free(rng, cells, n):
    free = np.argwhere(cells != OccupancyGrid.OCCUPIED)
    return [tuple(int(v) for v in free[i]) for i in rng.choice(len(free), n, replace=False)]


@pytest.mark.parametrize("seed", range(40))
def test_matches_reference_on_random_grids(seed):
    rng = np.random.default_rng(seed)
    h, w = rng.integers(5, 45, size=2)
    cells = rng.choice([OccupancyGrid.FREE, OccupancyGrid.UNKNOWN, OccupancyGrid.OCCUPIED], size=(h, w),
                       p=rng.dirichlet([2, 2, 1]))
    grid = _grid(cells)
    if (cells != OccupancyGrid.OCCUPIED).sum() < 2:
        pytest.skip("no free cells")
    for allow_diagonal in (False, True):
        for multiplier in (1.0, 3.0, 7.5):
            for _ in range(4):
                start, goal = _pick_free(rng, cells, 2)
                expected = _reference_astar(grid, start, goal, allow_diagonal, multiplier)
                assert astar(grid, start, goal, allow_diagonal, multiplier) == expected


def test_matches_reference_on_open_grid_with_many_ties():
    """All-FREE grid: every equal-cost route ties, so tie-breaking decides the path."""
    grid = _grid(np.zeros((30, 30)))
    for allow_diagonal in (False, True):
        for start, goal in (((0, 0), (29, 29)), ((5, 20), (25, 3)), ((15, 0), (15, 29))):
            assert astar(grid, start, goal, allow_diagonal, 3.0) == _reference_astar(grid, start, goal, allow_diagonal, 3.0)


def test_matches_reference_on_the_recorded_room_map():
    data = np.load(FIXTURE)
    cells = data["planning_cells"]
    grid = _grid(cells)
    rng = np.random.default_rng(1)
    known_free = np.argwhere(cells == OccupancyGrid.FREE)
    for i in rng.choice(len(known_free), 6, replace=False)[::2]:
        start = tuple(int(v) for v in known_free[i])
        goal = tuple(int(v) for v in known_free[rng.integers(len(known_free))])
        assert astar(grid, start, goal, True, 3.0) == _reference_astar(grid, start, goal, True, 3.0)


def test_same_errors_as_reference():
    cells = np.zeros((5, 5))
    cells[2, 2] = OccupancyGrid.OCCUPIED
    grid = _grid(cells)
    for start, goal in (((0, 0), (2, 2)), ((2, 2), (0, 0)), ((0, 0), (9, 9)), ((-1, 0), (0, 0))):
        with pytest.raises(ValueError):
            astar(grid, start, goal, True, 3.0)
    with pytest.raises(ValueError):
        astar(grid, (0, 0), (4, 4), True, 0.5)
    assert astar(grid, (1, 1), (1, 1)) == [(1, 1)]


def test_deadline_aborts_a_full_map_search():
    """Unreachable goal on a large open grid: without a deadline A* searches
    every cell; with one it gives up promptly by raising PlanningTimeout."""
    cells = np.full((400, 400), OccupancyGrid.UNKNOWN)
    cells[380:385, 380:385] = OccupancyGrid.OCCUPIED
    cells[382, 382] = OccupancyGrid.FREE          # enclosed goal
    grid = _grid(cells)
    started = time.monotonic()
    with pytest.raises(PlanningTimeout):
        astar(grid, (10, 10), (382, 382), True, 3.0, deadline=time.monotonic() + 0.05)
    assert time.monotonic() - started < 0.5


def test_deadline_does_not_change_a_result_found_in_time():
    grid = _grid(np.zeros((40, 40)))
    assert astar(grid, (0, 0), (39, 39), True, 3.0, deadline=time.monotonic() + 30.0) == astar(grid, (0, 0), (39, 39), True, 3.0)
