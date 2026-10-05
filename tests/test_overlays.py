"""Phase 11 tests: screen<->world coordinate conversion and pure overlay
geometry (no GUI toolkit, no display required)."""

from __future__ import annotations

import math

import pytest

from common.messages import LaserScan, Path, PathPoint, Pose2D
from visualization.overlays import (
    ViewTransform,
    heading_line_endpoints,
    path_screen_points,
    scan_hit_points_world,
    screen_to_world,
    world_to_screen,
)


def make_view(**overrides) -> ViewTransform:
    defaults = dict(world_origin_x=0.0, world_origin_y=0.0, scale=10.0, canvas_height=200)
    defaults.update(overrides)
    return ViewTransform(**defaults)


def test_view_transform_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        make_view(scale=0.0)
    with pytest.raises(ValueError):
        make_view(canvas_height=0)


def test_world_origin_maps_to_bottom_left_corner() -> None:
    view = make_view()
    assert world_to_screen(0.0, 0.0, view) == (0, 200)


def test_world_y_up_maps_to_screen_y_down() -> None:
    view = make_view()
    # Moving up in world y (away from origin) should move UP on screen (smaller py).
    px_low, py_low = world_to_screen(0.0, 1.0, view)
    px_high, py_high = world_to_screen(0.0, 5.0, view)
    assert py_high < py_low


def test_scale_converts_meters_to_pixels() -> None:
    view = make_view(scale=10.0)
    px, py = world_to_screen(2.0, 0.0, view)
    assert px == 20  # 2 m * 10 px/m


def test_nonzero_world_origin_shifts_the_frame() -> None:
    view = make_view(world_origin_x=5.0, world_origin_y=5.0)
    assert world_to_screen(5.0, 5.0, view) == (0, 200)


def test_round_trip_world_to_screen_to_world() -> None:
    view = make_view(world_origin_x=-3.0, world_origin_y=2.0, scale=17.0, canvas_height=350)
    for x, y in [(0.0, 0.0), (12.3, -4.5), (-3.0, 2.0), (100.0, 50.0)]:
        px, py = world_to_screen(x, y, view)
        back_x, back_y = screen_to_world(px, py, view)
        assert back_x == pytest.approx(x, abs=1.0 / view.scale)  # sub-pixel rounding tolerance
        assert back_y == pytest.approx(y, abs=1.0 / view.scale)


def test_round_trip_screen_to_world_to_screen_exact() -> None:
    # Going screen -> world -> screen has no rounding step in between,
    # so this direction should be exact.
    view = make_view(scale=12.5, canvas_height=480)
    for px, py in [(0, 0), (100, 200), (479, 1)]:
        x, y = screen_to_world(px, py, view)
        back_px, back_py = world_to_screen(x, y, view)
        assert back_px == pytest.approx(px, abs=1e-6)
        assert back_py == pytest.approx(py, abs=1e-6)


def test_heading_line_points_in_pose_direction() -> None:
    pose = Pose2D(x=1.0, y=1.0, theta=0.0, timestamp=0.0)
    start, end = heading_line_endpoints(pose, length=2.0)
    assert start == (1.0, 1.0)
    assert end[0] == pytest.approx(3.0)
    assert end[1] == pytest.approx(1.0, abs=1e-9)


def test_heading_line_respects_rotated_pose() -> None:
    pose = Pose2D(x=0.0, y=0.0, theta=math.pi / 2.0, timestamp=0.0)
    _, end = heading_line_endpoints(pose, length=1.0)
    assert end[0] == pytest.approx(0.0, abs=1e-9)
    assert end[1] == pytest.approx(1.0)


def test_scan_hit_points_excludes_max_range_rays() -> None:
    pose = Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=0.0)
    scan = LaserScan(
        timestamp=0.0, angle_min=0.0, angle_increment=math.pi / 2.0,
        ranges=[1.0, 5.0, 2.0], range_min=0.1, range_max=5.0,
    )
    points = scan_hit_points_world(pose, scan)
    assert len(points) == 2  # the range==range_max ray (index 1) is excluded
    assert points[0] == pytest.approx((1.0, 0.0), abs=1e-9)  # angle 0, range 1.0
    assert points[1][0] == pytest.approx(-2.0)  # angle pi, range 2.0
    assert points[1][1] == pytest.approx(0.0, abs=1e-9)


def test_scan_hit_points_empty_scan() -> None:
    pose = Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=0.0)
    scan = LaserScan(timestamp=0.0, angle_min=0.0, angle_increment=1.0, ranges=[], range_min=0.1, range_max=5.0)
    assert scan_hit_points_world(pose, scan) == []


def test_path_screen_points_converts_every_waypoint() -> None:
    view = make_view()
    path = Path(points=[PathPoint(0.0, 0.0), PathPoint(1.0, 1.0), PathPoint(2.0, 0.0)])
    screen_points = path_screen_points(path, view)
    assert screen_points == [world_to_screen(p.x, p.y, view) for p in path.points]
    assert len(screen_points) == 3


# ------------------------------------------------ orientation check helpers --

from visualization.overlays import direction_label, footprint_corners_world, robot_to_world, world_to_robot  # noqa: E402


def test_robot_frame_convention_front_is_plus_x_left_is_plus_y() -> None:
    pose = Pose2D(1.0, 2.0, 0.0)
    assert robot_to_world(pose, 1.0, 0.0) == pytest.approx((2.0, 2.0))   # FRONT
    assert robot_to_world(pose, 0.0, 1.0) == pytest.approx((1.0, 3.0))   # LEFT


def test_robot_frame_rotates_with_heading_and_round_trips() -> None:
    pose = Pose2D(0.5, -0.5, math.pi / 2)                               # facing +y
    assert robot_to_world(pose, 1.0, 0.0) == pytest.approx((0.5, 0.5))   # FRONT is now +y
    assert robot_to_world(pose, 0.0, 1.0) == pytest.approx((-0.5, -0.5))  # LEFT is now -x
    assert world_to_robot(pose, *robot_to_world(pose, 0.3, -0.7)) == pytest.approx((0.3, -0.7))


def test_footprint_corners_order_and_size() -> None:
    fl, fr, rr, rl = footprint_corners_world(Pose2D(0.0, 0.0, 0.0), length=0.5, width=0.4)
    assert fl == pytest.approx((0.25, 0.2)) and fr == pytest.approx((0.25, -0.2))
    assert rr == pytest.approx((-0.25, -0.2)) and rl == pytest.approx((-0.25, 0.2))


@pytest.mark.parametrize("deg, label", [(0, "FRONT"), (44, "FRONT"), (90, "LEFT"), (-90, "RIGHT"),
                                        (180, "REAR"), (-179, "REAR"), (135, "REAR"), (-46, "RIGHT")])
def test_direction_label(deg: float, label: str) -> None:
    assert direction_label(math.radians(deg)) == label


from visualization.overlays import ForwardUpView, cells_forward_up  # noqa: E402


def test_forward_up_view_puts_plus_x_up_and_plus_y_left() -> None:
    v = ForwardUpView(center_x=1.0, center_y=2.0, scale=100.0, side=400)
    assert v.to_screen(1.0, 2.0) == (200, 200)
    assert v.to_screen(2.0, 2.0) == (200, 100)    # +x (FRONT at startup) is UP
    assert v.to_screen(1.0, 3.0) == (100, 200)    # +y (LEFT) is LEFT
    assert v.to_world(*v.to_screen(0.37, 2.81)) == pytest.approx((0.37, 2.81), abs=0.01)


def test_cells_forward_up_orientation() -> None:
    import numpy as np
    cells = np.zeros((3, 4), dtype=int)       # 3 rows (y), 4 cols (x)
    cells[0, 3] = 1                           # smallest y, largest x -> top-right
    cells[2, 0] = 2                           # largest y, smallest x -> bottom-left
    img = cells_forward_up(cells)
    assert img.shape == (4, 3)
    assert img[0, 2] == 1 and img[3, 0] == 2
