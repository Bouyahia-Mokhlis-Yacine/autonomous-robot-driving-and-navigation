"""Small, dependency-free 2D geometry helpers shared across subsystems."""

from __future__ import annotations

import math


def wrap_angle(angle: float) -> float:
    """Wrap an angle (radians) into the range (-pi, pi]."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def angle_diff(target: float, current: float) -> float:
    """Shortest signed angular difference target - current, in (-pi, pi]."""
    return wrap_angle(target - current)


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def normalize(x: float, y: float) -> tuple[float, float]:
    """Return a unit vector in the direction of (x, y).

    Returns (0.0, 0.0) for a zero-length input rather than raising, since
    callers (e.g. controllers) generally want a safe no-op direction.
    """
    length = math.hypot(x, y)
    if length == 0.0:
        return (0.0, 0.0)
    return (x / length, y / length)


def clamp(value: float, low: float, high: float) -> float:
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    return max(low, min(high, value))


Point = tuple[float, float]


def rectangle_corners(
    center_x: float, center_y: float, length: float, width: float, theta: float
) -> list[Point]:
    """Corners of a rectangle centered at (center_x, center_y).

    ``length`` is the extent along the rectangle's local x-axis (forward
    direction for a robot footprint), ``width`` along its local y-axis,
    rotated by ``theta`` radians. Returns 4 corners in order (used for the
    Separating Axis Theorem collision check below).
    """
    half_l = length / 2.0
    half_w = width / 2.0
    local_corners = (
        (half_l, half_w),
        (half_l, -half_w),
        (-half_l, -half_w),
        (-half_l, half_w),
    )
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return [
        (center_x + lx * cos_t - ly * sin_t, center_y + lx * sin_t + ly * cos_t)
        for lx, ly in local_corners
    ]


def _polygon_axes(corners: list[Point]) -> list[Point]:
    """Outward edge-normal axes of a convex polygon, for SAT."""
    axes = []
    n = len(corners)
    for i in range(n):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % n]
        edge_x, edge_y = x2 - x1, y2 - y1
        # Perpendicular to the edge.
        normal = (-edge_y, edge_x)
        axes.append(normalize(*normal))
    return axes


def _project(corners: list[Point], axis: Point) -> tuple[float, float]:
    dots = [x * axis[0] + y * axis[1] for x, y in corners]
    return min(dots), max(dots)


def point_to_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Shortest distance from point (px, py) to the segment [(ax,ay), (bx,by)]."""
    seg_dx, seg_dy = bx - ax, by - ay
    seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
    if seg_len_sq == 0.0:
        return distance(px, py, ax, ay)
    t = ((px - ax) * seg_dx + (py - ay) * seg_dy) / seg_len_sq
    t = clamp(t, 0.0, 1.0)
    closest_x, closest_y = ax + t * seg_dx, ay + t * seg_dy
    return distance(px, py, closest_x, closest_y)


def polygons_intersect(poly_a: list[Point], poly_b: list[Point]) -> bool:
    """True if two convex polygons overlap, via the Separating Axis Theorem.

    Used to test a (possibly rotated) robot footprint rectangle against
    axis-aligned obstacle rectangles; works for any convex polygons.
    """
    for axis in _polygon_axes(poly_a) + _polygon_axes(poly_b):
        if axis == (0.0, 0.0):
            continue
        min_a, max_a = _project(poly_a, axis)
        min_b, max_b = _project(poly_b, axis)
        if max_a < min_b or max_b < min_a:
            return False  # Found a separating axis: no collision.
    return True


def ray_box_intersection(
    origin: Point, direction: Point, box_min: Point, box_max: Point
) -> tuple[float, float] | None:
    """Slab-method ray/axis-aligned-box intersection.

    Returns ``(t_enter, t_exit)`` — the ray parameter at which it enters
    and exits the box — or ``None`` if the ray never touches the box.
    ``t_enter`` may be negative (origin already inside the box, or the box
    is only intersected by the backward ray); callers decide how to
    interpret that based on whether the origin is expected to start
    inside (world boundary) or outside (a solid obstacle).
    """
    t_min, t_max = -math.inf, math.inf
    for origin_c, dir_c, lo, hi in (
        (origin[0], direction[0], box_min[0], box_max[0]),
        (origin[1], direction[1], box_min[1], box_max[1]),
    ):
        if dir_c == 0.0:
            if origin_c < lo or origin_c > hi:
                return None
            continue
        t1 = (lo - origin_c) / dir_c
        t2 = (hi - origin_c) / dir_c
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None
    return (t_min, t_max)


def offset_pose(pose, dx: float, dy: float):
    """The pose of a point rigidly attached at ``(dx, dy)`` in ``pose``'s own
    frame (x forward, y left), with the same heading -- e.g. the LiDAR pose
    from the robot pose (its mount offset), or the robot pose from the LiDAR
    pose (the negated offset)."""
    from common.messages import Pose2D
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    return Pose2D(x=pose.x + c * dx - s * dy, y=pose.y + s * dx + c * dy, theta=pose.theta, timestamp=pose.timestamp)
