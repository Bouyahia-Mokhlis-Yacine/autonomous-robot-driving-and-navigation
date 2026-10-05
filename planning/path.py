"""Conversion between A*'s grid-cell paths and the world-frame ``Path`` message."""

from __future__ import annotations

from common.geometry import distance
from common.messages import Path, PathPoint
from mapping.occupancy_grid import OccupancyGridMap
from planning.astar import Cell


def grid_path_to_world_path(grid_map: OccupancyGridMap, cells: list[Cell]) -> Path:
    """Convert a list of (row, col) grid cells into world-frame waypoints.

    Each waypoint is the world-coordinate center of its grid cell.
    """
    points = []
    for row, col in cells:
        x, y = grid_map.grid_to_world(row, col)
        points.append(PathPoint(x=x, y=y))
    return Path(points=points)


def path_length(path: Path) -> float:
    """Total geometric length of ``path`` in meters."""
    total = 0.0
    for a, b in zip(path.points, path.points[1:]):
        total += distance(a.x, a.y, b.x, b.y)
    return total
