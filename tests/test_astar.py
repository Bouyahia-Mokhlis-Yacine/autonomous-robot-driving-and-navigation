"""Phase 6 tests: A* on deterministic known maps."""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.messages import OccupancyGrid
from planning.astar import astar


def make_grid(rows: int, cols: int, occupied: set[tuple[int, int]] | None = None) -> OccupancyGrid:
    cells = np.full((rows, cols), OccupancyGrid.FREE, dtype=np.int8)
    for r, c in occupied or set():
        cells[r, c] = OccupancyGrid.OCCUPIED
    return OccupancyGrid(resolution=1.0, width=cols, height=rows, origin_x=0.0, origin_y=0.0, cells=cells)


def path_is_connected(path: list[tuple[int, int]], allow_diagonal: bool) -> bool:
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        if allow_diagonal:
            if max(dr, dc) != 1:
                return False
        else:
            if dr + dc != 1:
                return False
    return True


def path_avoids_cells(path: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    return not any(cell in occupied for cell in path)


def test_start_equals_goal_returns_single_cell() -> None:
    grid = make_grid(5, 5)
    path = astar(grid, (2, 2), (2, 2), allow_diagonal=False)
    assert path == [(2, 2)]


def test_empty_grid_shortest_4_connected_path_length() -> None:
    grid = make_grid(5, 5)
    path = astar(grid, (0, 0), (4, 4), allow_diagonal=False)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    # Manhattan distance is 8 -> 9 cells on the shortest 4-connected path.
    assert len(path) == 9
    assert path_is_connected(path, allow_diagonal=False)


def test_empty_grid_shortest_8_connected_path_length() -> None:
    grid = make_grid(5, 5)
    path = astar(grid, (0, 0), (4, 4), allow_diagonal=True)
    assert path is not None
    # A pure diagonal line is optimal: 5 cells (0,0)..(4,4).
    assert len(path) == 5
    assert path_is_connected(path, allow_diagonal=True)


def test_wall_with_gap_forces_detour_through_gap() -> None:
    occupied = {(2, c) for c in range(5) if c != 2}  # wall across row 2, gap at col 2
    grid = make_grid(5, 5, occupied)
    path = astar(grid, (0, 2), (4, 2), allow_diagonal=False)
    assert path is not None
    assert (2, 2) in path  # must pass through the gap
    assert path_avoids_cells(path, occupied)


def test_fully_enclosed_goal_is_unreachable_returns_none() -> None:
    occupied = {(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)}
    grid = make_grid(5, 5, occupied)
    path = astar(grid, (0, 0), (2, 2), allow_diagonal=False)  # (2,2) is boxed in
    assert path is None


def test_occupied_start_raises() -> None:
    grid = make_grid(5, 5, {(0, 0)})
    with pytest.raises(ValueError):
        astar(grid, (0, 0), (4, 4), allow_diagonal=False)


def test_occupied_goal_raises() -> None:
    grid = make_grid(5, 5, {(4, 4)})
    with pytest.raises(ValueError):
        astar(grid, (0, 0), (4, 4), allow_diagonal=False)


def test_out_of_bounds_start_raises() -> None:
    grid = make_grid(5, 5)
    with pytest.raises(ValueError):
        astar(grid, (-1, 0), (4, 4), allow_diagonal=False)


def test_out_of_bounds_goal_raises() -> None:
    grid = make_grid(5, 5)
    with pytest.raises(ValueError):
        astar(grid, (0, 0), (5, 5), allow_diagonal=False)


def test_path_never_crosses_occupied_cells() -> None:
    occupied = {(2, 1), (2, 2), (2, 3), (1, 3), (3, 3)}
    grid = make_grid(6, 6, occupied)
    path = astar(grid, (0, 0), (5, 5), allow_diagonal=True)
    assert path is not None
    assert path_avoids_cells(path, occupied)
    assert path_is_connected(path, allow_diagonal=True)


def test_diagonal_move_forbidden_through_blocked_corner() -> None:
    # Obstacles at (1,0) and (0,1) block the diagonal gap between (0,0) and (1,1).
    occupied = {(1, 0), (0, 1)}
    grid = make_grid(3, 3, occupied)
    path = astar(grid, (0, 0), (1, 1), allow_diagonal=True)
    # The direct diagonal cut is forbidden; (0,0)/(1,1) are otherwise isolated
    # from the rest of the grid by the two blocked orthogonal cells, so there
    # is no valid route at all.
    assert path is None


def test_8_connected_path_no_longer_than_4_connected_for_same_map() -> None:
    occupied = {(2, c) for c in range(6) if c != 3}
    grid = make_grid(6, 6, occupied)
    path_4 = astar(grid, (0, 0), (5, 5), allow_diagonal=False)
    path_8 = astar(grid, (0, 0), (5, 5), allow_diagonal=True)
    assert path_4 is not None and path_8 is not None

    def path_length(path: list[tuple[int, int]]) -> float:
        total = 0.0
        for (r0, c0), (r1, c1) in zip(path, path[1:]):
            total += math.hypot(r1 - r0, c1 - c0)
        return total

    assert path_length(path_8) <= path_length(path_4) + 1e-9
