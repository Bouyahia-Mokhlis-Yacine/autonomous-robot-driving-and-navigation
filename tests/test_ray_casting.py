"""Tests for the pure grid-index Bresenham line tracer."""

from __future__ import annotations

from mapping.ray_casting import bresenham_line


def test_horizontal_line() -> None:
    cells = bresenham_line(0, 0, 0, 5)
    assert cells == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]


def test_vertical_line() -> None:
    cells = bresenham_line(0, 0, 5, 0)
    assert cells == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]


def test_diagonal_line() -> None:
    cells = bresenham_line(0, 0, 3, 3)
    assert cells == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_single_cell_when_start_equals_end() -> None:
    assert bresenham_line(4, 4, 4, 4) == [(4, 4)]


def test_line_endpoints_are_always_included() -> None:
    cells = bresenham_line(2, 1, -3, 7)
    assert cells[0] == (2, 1)
    assert cells[-1] == (-3, 7)


def test_line_is_connected_no_gaps() -> None:
    cells = bresenham_line(0, 0, 4, 9)
    for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
        assert abs(r1 - r0) <= 1
        assert abs(c1 - c0) <= 1
