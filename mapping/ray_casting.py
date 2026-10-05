"""Grid-index ray tracing (Bresenham's line algorithm).

Pure integer grid geometry with no knowledge of occupancy values, world
units, or sensors — reusable anywhere a straight line of grid cells
between two cells is needed (occupancy updates here, potentially
visualization overlays later).
"""

from __future__ import annotations


def bresenham_line(row0: int, col0: int, row1: int, col1: int) -> list[tuple[int, int]]:
    """Grid cells (row, col) on the line from (row0, col0) to (row1, col1), inclusive."""
    cells: list[tuple[int, int]] = []
    row, col = row0, col0
    d_row = abs(row1 - row0)
    d_col = abs(col1 - col0)
    step_row = 1 if row1 >= row0 else -1
    step_col = 1 if col1 >= col0 else -1

    if d_col > d_row:
        error = d_col / 2.0
        while col != col1:
            cells.append((row, col))
            error -= d_row
            if error < 0:
                row += step_row
                error += d_col
            col += step_col
    else:
        error = d_row / 2.0
        while row != row1:
            cells.append((row, col))
            error -= d_col
            if error < 0:
                col += step_col
                error += d_row
            row += step_row

    cells.append((row, col))
    return cells
