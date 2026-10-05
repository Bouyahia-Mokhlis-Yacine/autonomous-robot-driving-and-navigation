"""Phase 7 tests: path post-processing (dedupe, collinear reduction,
line-of-sight simplification, safe smoothing)."""

from __future__ import annotations

import numpy as np
import pytest

from common.messages import OccupancyGrid, Path, PathPoint
from mapping.occupancy_grid import OccupancyGridMap
from planning.astar import astar
from planning.path import path_length
from planning.path_smoothing import (
    _path_is_collision_free,
    has_line_of_sight,
    process_path,
    remove_collinear_points,
    remove_duplicate_points,
    simplify_line_of_sight,
    smooth_path,
)


def make_grid_map(size: int = 20, resolution: float = 1.0, occupied: set[tuple[int, int]] | None = None) -> OccupancyGridMap:
    grid_map = OccupancyGridMap(resolution=resolution, width_m=size * resolution, height_m=size * resolution)
    for r, c in occupied or set():
        grid_map.set_cell(r, c, OccupancyGrid.OCCUPIED)
    return grid_map


def pp(x: float, y: float) -> PathPoint:
    return PathPoint(x=x, y=y)


# ---------------------------------------------------------------- dedupe ---


def test_remove_duplicate_points_collapses_repeats() -> None:
    path = Path(points=[pp(0, 0), pp(0, 0), pp(1, 0), pp(1, 0), pp(1, 0), pp(2, 0)])
    result = remove_duplicate_points(path)
    assert [(p.x, p.y) for p in result.points] == [(0, 0), (1, 0), (2, 0)]


def test_remove_duplicate_points_empty_and_single() -> None:
    assert remove_duplicate_points(Path(points=[])).points == []
    result = remove_duplicate_points(Path(points=[pp(1, 1)]))
    assert [(p.x, p.y) for p in result.points] == [(1, 1)]


# -------------------------------------------------------- collinear reduction ---


def test_remove_collinear_points_collapses_straight_run() -> None:
    path = Path(points=[pp(0, 0), pp(1, 0), pp(2, 0), pp(3, 0), pp(4, 0)])
    result = remove_collinear_points(path)
    assert [(p.x, p.y) for p in result.points] == [(0, 0), (4, 0)]


def test_remove_collinear_points_keeps_corner() -> None:
    path = Path(points=[pp(0, 0), pp(1, 0), pp(2, 0), pp(2, 1), pp(2, 2)])
    result = remove_collinear_points(path)
    assert [(p.x, p.y) for p in result.points] == [(0, 0), (2, 0), (2, 2)]


def test_remove_collinear_points_diagonal_run() -> None:
    path = Path(points=[pp(0, 0), pp(1, 1), pp(2, 2), pp(3, 3)])
    result = remove_collinear_points(path)
    assert [(p.x, p.y) for p in result.points] == [(0, 0), (3, 3)]


def test_remove_collinear_points_short_paths_unchanged() -> None:
    assert remove_collinear_points(Path(points=[])).points == []
    single = remove_collinear_points(Path(points=[pp(1, 1)]))
    assert [(p.x, p.y) for p in single.points] == [(1, 1)]
    two = remove_collinear_points(Path(points=[pp(1, 1), pp(2, 2)]))
    assert [(p.x, p.y) for p in two.points] == [(1, 1), (2, 2)]


# ------------------------------------------------------------ line of sight ---


def test_line_of_sight_clear_in_empty_grid() -> None:
    grid_map = make_grid_map()
    assert has_line_of_sight(grid_map, pp(0.5, 0.5), pp(15.5, 15.5)) is True


def test_line_of_sight_blocked_by_obstacle() -> None:
    grid_map = make_grid_map(occupied={(5, 5)})
    assert has_line_of_sight(grid_map, pp(0.5, 5.5), pp(10.5, 5.5)) is False


def test_simplify_line_of_sight_collapses_staircase_in_open_space() -> None:
    # A grid-following staircase path with no obstacles should collapse to
    # start and end only, since the direct line is clear.
    staircase = Path(points=[pp(0.5, 0.5), pp(1.5, 0.5), pp(1.5, 1.5), pp(2.5, 1.5), pp(2.5, 2.5)])
    grid_map = make_grid_map()
    result = simplify_line_of_sight(staircase, grid_map)
    assert len(result.points) == 2
    assert (result.points[0].x, result.points[0].y) == (0.5, 0.5)
    assert (result.points[-1].x, result.points[-1].y) == (2.5, 2.5)


def test_simplify_line_of_sight_keeps_necessary_waypoint_around_obstacle() -> None:
    occupied = {(5, c) for c in range(20) if c != 10}  # wall with a gap at col 10
    grid_map = make_grid_map(occupied=occupied)
    path = Path(points=[pp(0.5, 0.5), pp(10.5, 4.5), pp(10.5, 5.5), pp(10.5, 6.5), pp(15.5, 15.5)])

    result = simplify_line_of_sight(path, grid_map)

    # Must still pass through the gap; cannot shortcut straight through the wall.
    for a, b in zip(result.points, result.points[1:]):
        assert has_line_of_sight(grid_map, a, b)


def test_simplify_line_of_sight_short_paths_unchanged() -> None:
    grid_map = make_grid_map()
    empty = simplify_line_of_sight(Path(points=[]), grid_map)
    assert empty.points == []
    two = simplify_line_of_sight(Path(points=[pp(0.5, 0.5), pp(1.5, 1.5)]), grid_map)
    assert len(two.points) == 2


# ------------------------------------------------------------------- smoothing ---


def test_smooth_path_keeps_endpoints_fixed() -> None:
    grid_map = make_grid_map()
    path = Path(points=[pp(0.0, 0.0), pp(5.0, 3.0), pp(10.0, 0.0)])
    result = smooth_path(path, grid_map)
    assert result.points[0].x == pytest.approx(0.0)
    assert result.points[0].y == pytest.approx(0.0)
    assert result.points[-1].x == pytest.approx(10.0)
    assert result.points[-1].y == pytest.approx(0.0)


def test_smooth_path_result_is_always_collision_free() -> None:
    # A path that hugs a concave obstacle corner -- naive smoothing would
    # want to cut the corner straight through the obstacle.
    occupied = {(r, c) for r in range(4, 8) for c in range(4, 8)}
    grid_map = make_grid_map(size=20, occupied=occupied)
    path = Path(
        points=[
            pp(2.5, 2.5),
            pp(3.5, 3.5),
            pp(3.5, 8.5),
            pp(8.5, 8.5),
            pp(8.5, 3.5),
            pp(12.5, 2.5),
        ]
    )
    result = smooth_path(path, grid_map, max_iterations=500)
    assert _path_is_collision_free(result, grid_map)


def test_smooth_path_short_paths_unchanged() -> None:
    grid_map = make_grid_map()
    single = smooth_path(Path(points=[pp(1, 1)]), grid_map)
    assert [(p.x, p.y) for p in single.points] == [(1, 1)]
    two = smooth_path(Path(points=[pp(1, 1), pp(2, 2)]), grid_map)
    assert [(p.x, p.y) for p in two.points] == [(1, 1), (2, 2)]


# ---------------------------------------------------------------- full pipeline ---


def test_process_path_on_astar_output_stays_collision_free_and_no_longer() -> None:
    occupied = {(5, c) for c in range(20) if c != 10}
    grid_map = make_grid_map(occupied=occupied)

    raw_cells = astar(grid_map.grid, (0, 0), (19, 19), allow_diagonal=True)
    assert raw_cells is not None
    raw_points = [pp(*grid_map.grid_to_world(r, c)) for r, c in raw_cells]
    raw_path = Path(points=raw_points)

    processed = process_path(raw_path, grid_map, simplify=True, smooth=False)

    assert _path_is_collision_free(processed, grid_map)
    assert path_length(processed) <= path_length(raw_path) + 1e-9
    # Endpoints preserved.
    assert (processed.points[0].x, processed.points[0].y) == (raw_points[0].x, raw_points[0].y)
    assert (processed.points[-1].x, processed.points[-1].y) == (raw_points[-1].x, raw_points[-1].y)
    # Simplification should meaningfully reduce waypoint count on this map.
    assert len(processed.points) < len(raw_path.points)


def test_process_path_with_smoothing_still_collision_free() -> None:
    occupied = {(5, c) for c in range(20) if c != 10}
    grid_map = make_grid_map(occupied=occupied)

    raw_cells = astar(grid_map.grid, (0, 0), (19, 19), allow_diagonal=True)
    assert raw_cells is not None
    raw_path = Path(points=[pp(*grid_map.grid_to_world(r, c)) for r, c in raw_cells])

    processed = process_path(raw_path, grid_map, simplify=True, smooth=True)
    assert _path_is_collision_free(processed, grid_map)
