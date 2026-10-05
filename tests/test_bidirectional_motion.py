"""Bidirectional path following (FORWARD / REVERSE / ROTATE_IN_PLACE), its
direction-aware motion safety, and the per-motion motor duty floors."""

from __future__ import annotations

import dataclasses
import math

import pytest

from common.config import load_hardware_config, load_robot_config
from common.messages import Goal, LaserScan, OccupancyGrid, Path, PathPoint, Pose2D, VelocityCommand
from control.path_follower import MotionMode, PathFollower, PathFollowerConfig, path_follower_config
from hardware.motor_driver import GpioDiffDriveRobot, RecordingMotor, motion_min_duty
from mapping.occupancy_grid import OccupancyGridMap
from navigation.motion_safety import MotionSafety, MotionSafetyConfig
from navigation.navigation_manager import NavigationManager
from navigation.replanner import Replanner, ReplannerConfig
from planning.global_planner import GlobalPlanner
from slam.pose import integrate_pose
from tests.test_navigation_manager import FakeRobot

F, R, T, S = MotionMode.FORWARD, MotionMode.REVERSE, MotionMode.ROTATE_IN_PLACE, MotionMode.STOPPED


def cfg(**over) -> PathFollowerConfig:
    base = dict(max_linear_velocity=0.4, max_angular_velocity=1.5, max_linear_acceleration=100.0,
                max_angular_acceleration=100.0, goal_tolerance=0.1, heading_gain=2.0, waypoint_tolerance=0.2,
                turn_in_place_enter=math.radians(30), turn_in_place_exit=math.radians(10),
                allow_reverse=True, max_reverse_distance=1.5)
    base.update(over)
    return PathFollowerConfig(**base)


def at(deg: float, x: float = 0.0, y: float = 0.0) -> Pose2D:
    return Pose2D(x, y, math.radians(deg))


def path(*pts, direction: int = 0) -> Path:
    return Path([PathPoint(x, y, direction) for x, y in pts])


def drive(follower: PathFollower, pose: Pose2D, p: Path, dt: float = 0.05, steps: int = 800,
          reverse_allowed: bool = True) -> tuple[Pose2D, list[MotionMode]]:
    modes = []
    for _ in range(steps):
        if follower.is_goal_reached(pose, p):
            break
        cmd = follower.compute_command(pose, p, dt, reverse_allowed=reverse_allowed)
        modes.append(follower.mode)
        pose = integrate_pose(pose, cmd, dt)
    return pose, modes


def collapse(modes: list[MotionMode]) -> list[MotionMode]:
    out = []
    for m in modes:
        if not out or out[-1] is not m:
            out.append(m)
    return out


# ------------------------------------------------------------ path follower --

def test_small_heading_error_drives_forward() -> None:
    f = PathFollower(cfg())
    cmd = f.compute_command(at(15), path((2.0, 0.0)), 0.1)
    assert f.mode is F and cmd.linear > 0.3 and cmd.angular < 0     # steering back toward the target


def test_zero_heading_error_does_not_turn() -> None:
    f = PathFollower(cfg())
    cmd = f.compute_command(at(0), path((2.0, 0.0)), 0.1)
    assert f.mode is F and cmd.angular == pytest.approx(0.0, abs=1e-9)


def test_large_heading_error_turns_in_place() -> None:
    f = PathFollower(cfg())
    cmd = f.compute_command(at(0), path((0.0, 2.0)), 0.1)    # 90 deg to the left
    assert f.mode is T and cmd.linear == 0.0 and cmd.angular > 0


def test_turn_ends_only_below_the_exit_threshold() -> None:
    f = PathFollower(cfg())
    p = path((0.0, 2.0))
    f.compute_command(at(0), p, 0.1)
    for deg, expected in ((50, T), (65, T), (75, T), (79.5, T), (81, F)):   # errors 40, 25, 15, 10.5, 9 deg
        f.compute_command(at(deg), p, 0.1)
        assert f.mode is expected, deg


def test_hysteresis_no_forward_turn_chattering() -> None:
    f = PathFollower(cfg())
    p = path((2.0, 0.0))
    f.compute_command(at(0), p, 0.1)
    assert f.mode is F
    f.compute_command(at(25), p, 0.1)          # below ENTER: keeps driving
    assert f.mode is F
    f.compute_command(at(35), p, 0.1)          # above ENTER: turn in place
    assert f.mode is T
    f.compute_command(at(20), p, 0.1)          # above EXIT: keeps turning
    assert f.mode is T


def test_goal_behind_is_reached_by_reversing_without_a_u_turn() -> None:
    f = PathFollower(cfg())
    end, modes = drive(f, at(0), path((-1.0, 0.0)))
    assert f.is_goal_reached(end, path((-1.0, 0.0)))
    assert collapse(modes) == [R]
    assert abs(end.theta) < math.radians(5)    # still facing the same way


def test_reverse_segment_produces_negative_speed() -> None:
    f = PathFollower(cfg())
    cmd = f.compute_command(at(0), path((-1.0, 0.1)), 0.1)
    assert f.mode is R and cmd.linear < 0


def test_explicit_forward_waypoint_behind_turns_around_instead_of_reversing() -> None:
    f = PathFollower(cfg())
    end, modes = drive(f, at(0), path((-1.0, 0.0), direction=PathPoint.FORWARD))
    assert collapse(modes)[:2] == [T, F]
    assert abs(abs(end.theta) - math.pi) < math.radians(12)


def test_explicit_reverse_waypoint_ahead_turns_then_reverses() -> None:
    f = PathFollower(cfg())
    _, modes = drive(f, at(0), path((1.0, 0.0), direction=PathPoint.REVERSE))
    assert collapse(modes)[:2] == [T, R]
    assert F not in modes


def test_long_stretch_behind_turns_around_instead_of_reversing() -> None:
    f = PathFollower(cfg(max_reverse_distance=1.5))
    f.compute_command(at(0), path((-3.0, 0.0)), 0.1)
    assert f.mode is T


def test_reverse_refused_by_caller_turns_toward_the_front() -> None:
    f = PathFollower(cfg())
    cmd = f.compute_command(at(0), path((-1.0, 0.0)), 0.1, reverse_allowed=False)
    assert f.mode is T and cmd.linear == 0.0


def test_direction_change_mid_path_and_waypoint_progression_while_reversing() -> None:
    # Out 1 m ahead, then back past the start: forward, then reverse -- no U-turn.
    f = PathFollower(cfg())
    p = path((1.0, 0.0), (0.4, 0.0), (-0.3, 0.0))
    end, modes = drive(f, at(0), p)
    assert f.is_goal_reached(end, p)
    assert collapse(modes) == [F, R]
    assert f._local_planner.index == 2          # advanced through every waypoint while reversing


def test_legacy_config_is_unchanged_forward_only() -> None:
    f = PathFollower(dataclasses.replace(cfg(), turn_in_place_enter=None, allow_reverse=False))
    cmd = f.compute_command(at(0), path((-1.0, 0.0)), 0.1)
    assert cmd.linear >= 0.0                    # never reverses


def test_robot_yaml_enables_bidirectional_mode() -> None:
    c = path_follower_config(load_robot_config())
    assert c.bidirectional and c.allow_reverse
    assert math.degrees(c.turn_in_place_enter) == pytest.approx(30) and math.degrees(c.turn_in_place_exit) == pytest.approx(10)


def test_path_point_direction_defaults_to_either() -> None:
    assert PathPoint(1.0, 2.0).direction == PathPoint.EITHER == 0


# ------------------------------------------------------------ motion safety --

L, W = 0.5, 0.4
SAFETY = MotionSafetyConfig(robot_length=L, robot_width=W, stop_distance=0.3, side_margin=0.05,
                            rotation_margin=0.03, min_scan_points=2)


def scan_with(*points_robot_frame) -> LaserScan:
    ranges = [8.0] * 361
    for x, y in points_robot_frame:
        i = round((math.atan2(y, x) + math.pi) / math.radians(1.0))
        ranges[i] = math.hypot(x, y)
    return LaserScan(timestamp=0.0, angle_min=-math.pi, angle_increment=math.radians(1.0),
                     ranges=ranges, range_min=0.15, range_max=8.0)


def safety(*pts) -> MotionSafety:
    s = scan_with(*pts)
    return MotionSafety(SAFETY, scan_provider=lambda: s)


def test_obstacle_ahead_blocks_forward_only() -> None:
    s = safety((0.40, 0.0), (0.40, 0.03))
    assert not s.is_clear(F, at(0), None)
    assert s.is_clear(R, at(0), None)


def test_obstacle_behind_blocks_reverse_only() -> None:
    s = safety((-0.40, 0.0), (-0.40, -0.03))
    assert not s.is_clear(R, at(0), None)
    assert s.is_clear(F, at(0), None)


def test_clear_rear_allows_reverse() -> None:
    assert safety((1.5, 0.0), (1.5, 0.1), (-1.5, 0.0), (-1.5, 0.1)).is_clear(R, at(0), None)


def test_rear_check_covers_the_footprint_width_not_one_ray() -> None:
    # Behind the rear-LEFT corner, not straight behind.
    assert not safety((-0.35, 0.22), (-0.36, 0.23)).is_clear(R, at(0), None)
    assert safety((-0.35, 0.40), (-0.36, 0.41)).is_clear(R, at(0), None)     # beside, outside the corridor


def test_turn_in_place_respects_the_swept_footprint() -> None:
    # Half-diagonal 0.32 m + 0.03 m: a point 0.34 m away beside the corner blocks a
    # turn but is outside the forward zone; 0.40 m is clear.
    def polar(r, deg):
        return r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg))
    near = safety(polar(0.34, 60), polar(0.34, 62))      # beside the front-left corner; two scan bins
    assert not near.is_clear(T, at(0), None)
    assert near.is_clear(F, at(0), None)
    assert safety(polar(0.40, 60), polar(0.40, 62)).is_clear(T, at(0), None)


def test_single_scan_outlier_does_not_block_but_a_map_cell_does() -> None:
    assert safety((0.40, 0.0)).is_clear(F, at(0), None)
    grid = OccupancyGridMap(resolution=0.05, width_m=4.0, height_m=4.0, origin_x=-2.0, origin_y=-2.0)
    r, c = grid.world_to_grid(0.40, 0.0)
    grid.grid.cells[r, c] = OccupancyGrid.OCCUPIED
    assert not safety().is_clear(F, at(0), grid)
    # The map is used through the pose: the same cell is BEHIND a robot facing -x.
    assert not safety().is_clear(R, at(180), grid) and safety().is_clear(F, at(180), grid)


def test_points_inside_the_footprint_are_ignored() -> None:
    s = safety((0.1, 0.1), (0.12, 0.1), (-0.2, -0.15), (-0.21, -0.15))
    assert s.is_clear(F, at(0), None) and s.is_clear(R, at(0), None) and s.is_clear(T, at(0), None)


# --------------------------------------------------- NavigationManager level --

def manager(scan: LaserScan | None, pose: Pose2D, obstacle_stop_distance: float = 0.3):
    robot = FakeRobot(pose)
    holder = {"scan": scan}
    m = NavigationManager(
        global_planner=GlobalPlanner(allow_diagonal=True), path_follower=PathFollower(cfg()), robot=robot,
        replanner=Replanner(ReplannerConfig(replan_interval=100.0, deviation_threshold=100.0)),
        obstacle_stop_distance=obstacle_stop_distance,
        motion_safety=MotionSafety(SAFETY, scan_provider=lambda: holder["scan"]))
    return m, robot, holder


def free_grid() -> OccupancyGridMap:
    g = OccupancyGridMap(resolution=0.05, width_m=10.0, height_m=10.0, origin_x=-5.0, origin_y=-5.0)
    g.grid.cells[:] = OccupancyGrid.FREE
    return g


def test_safety_stop_works_forward_and_reverse() -> None:
    grid = free_grid()
    for goal, blocker in (((1.5, 0.0), [(0.4, 0.0), (0.4, 0.03)]),      # forward, obstacle ahead
                          ((-1.0, 0.0), [(-0.4, 0.0), (-0.4, 0.03)])):  # reverse wanted, obstacle behind
        m, robot, holder = manager(scan_with(*blocker), at(0))
        m.set_goal(Goal(*goal), grid, grid)
        cmd = m.update(grid, 0.1, grid)
        if goal[0] > 0:
            assert cmd == VelocityCommand(0.0, 0.0) and m.safety_stops == 1
        else:
            # Scenario D: reverse refused -> the follower turns toward the front instead.
            assert m.get_motion_mode() is T and cmd.linear == 0.0
        # Remove the obstacle: motion resumes by itself.
        holder["scan"] = scan_with()
        cmd = m.update(grid, 0.1, grid)
        assert cmd.linear != 0.0 or cmd.angular != 0.0


def test_scenario_d_obstacle_behind_and_turning_blocked_stops_safely() -> None:
    grid = free_grid()
    blocker = [(-0.4, 0.0), (-0.4, 0.03), (0.0, 0.3), (0.01, 0.3)]   # behind, and beside (inside the turning circle)
    m, robot, _ = manager(scan_with(*blocker), at(0))
    m.set_goal(Goal(-1.0, 0.0), grid, grid)
    for _ in range(5):
        assert m.update(grid, 0.1, grid) == VelocityCommand(0.0, 0.0)
    assert robot.last_command == VelocityCommand(0.0, 0.0)


def test_scenario_e_obstacle_ahead_does_not_stop_reversing_away() -> None:
    grid = free_grid()
    r, c = grid.world_to_grid(0.45, 0.0)            # a wall cell just ahead (confirmed map)
    grid.grid.cells[r - 3:r + 4, c] = OccupancyGrid.OCCUPIED
    m, robot, _ = manager(scan_with((0.45, 0.0), (0.45, 0.05)), at(0))
    m.set_goal(Goal(-1.0, 0.0), grid, grid)
    cmd = VelocityCommand(0.0, 0.0)
    for _ in range(3):
        cmd = m.update(grid, 0.1, grid)
    assert m.get_motion_mode() is R and cmd.linear < 0


# --------------------------------------------------------- motor duty floors --

def test_turn_in_place_and_reverse_use_their_own_minimum_duty() -> None:
    motors = dataclasses.replace(load_hardware_config().motors, min_duty=0.3, min_reverse_duty=0.35, min_turn_duty=0.5)
    limits = load_robot_config().limits
    assert motion_min_duty(0.0, 0.6, motors) == 0.5
    assert motion_min_duty(-0.1, 0.0, motors) == 0.35
    assert motion_min_duty(0.1, 0.2, motors) == 0.3
    r = GpioDiffDriveRobot(motors, limits, motor_factory=RecordingMotor, hold_pins_low=lambda pins: True)
    r.set_velocity(VelocityCommand(0.0, 0.6))                 # left back, right forward
    assert r.last_duty == pytest.approx((-0.5, 0.5))
    r.set_velocity(VelocityCommand(-0.05, 0.0))
    assert r.last_duty == pytest.approx((-0.35, -0.35))
    r.set_velocity(VelocityCommand(0.05, 0.0))
    assert r.last_duty == pytest.approx((0.3, 0.3))
    r.close()


def test_duty_floors_default_to_min_duty() -> None:
    m = load_hardware_config().motors
    assert m.min_reverse_duty == m.min_duty and m.min_turn_duty == m.min_duty


def test_recovery_reverse_is_also_guarded_by_rear_clearance() -> None:
    from navigation.navigation_manager import NavigationState
    grid = free_grid()
    m, robot, holder = manager(scan_with(), at(0))
    m.set_goal(Goal(1.5, 0.0), grid, grid)
    m.state = NavigationState.RECOVERING                  # as if _plan had entered the reverse recovery
    cmd = m.update(grid, 0.1, grid)
    assert cmd.linear < 0                                  # rear clear: recovery reverses
    holder["scan"] = scan_with((-0.4, 0.0), (-0.4, 0.03))
    m.state = NavigationState.RECOVERING
    assert m.update(grid, 0.1, grid) == VelocityCommand(0.0, 0.0) and m.state is not NavigationState.RECOVERING


def test_side_target_prefers_turning_the_front_toward_it() -> None:
    f = PathFollower(cfg())
    for side in (88.0, 92.0, 98.0):             # the rear saves < 20 deg of turning: still turn the FRONT
        f.reset()
        f.compute_command(at(0), path((1.0 * math.cos(math.radians(side)), 1.0 * math.sin(math.radians(side)))), 0.1)
        assert f.mode is T and not f._rotate_to_rear, side
    f.reset()
    f.compute_command(at(0), path((math.cos(math.radians(125)), math.sin(math.radians(125)))), 0.1)
    assert f.mode is T and f._rotate_to_rear      # 55 deg vs 125 deg: the rear is clearly closer


# ------------------------------------------ regression: 120 deg left-right loop --

@pytest.mark.parametrize("keep_motion", [False, True])
@pytest.mark.parametrize("deg", [100.0, 120.0, 150.0, -120.0])
@pytest.mark.parametrize("dist", [1.0, 2.5])
def test_goal_well_off_the_front_never_turns_back_and_forth(deg: float, dist: float, keep_motion: bool) -> None:
    """Robot 2026-09-30: a goal ~120 deg off made the robot turn ~30 deg left,
    then right, forever -- the reverse-eligibility stretch was measured along
    the robot's CURRENT rear. Replans every 1 s (a fresh path from the robot,
    as the planner produces), with a full reset (sync) or keeping the motion
    (async periodic replan). 0.2 m/s / 0.6 rad/s as on the floor."""
    c = cfg(max_linear_velocity=0.2, max_reverse_velocity=0.2, max_angular_velocity=0.6,
            max_linear_acceleration=0.5, max_angular_acceleration=2.0)
    f = PathFollower(c)
    a = math.radians(deg)
    goal = (dist * math.cos(a), dist * math.sin(a))
    full = [PathPoint(goal[0] * k / 5, goal[1] * k / 5) for k in range(1, 6)]
    p = Path(list(full))
    pose, signs = at(0), []
    for tick in range(600):                                  # 60 s at 10 Hz
        if tick and tick % 10 == 0:                          # periodic replan: path restarts at the robot
            remaining = [q for q in full if math.hypot(q.x - pose.x, q.y - pose.y) > 0.05 and
                         (q.x - pose.x) * goal[0] + (q.y - pose.y) * goal[1] > 0] or [full[-1]]
            p = Path([PathPoint(pose.x, pose.y)] + remaining)
            f.reset(keep_motion=keep_motion)
        if f.is_goal_reached(pose, p):
            break
        cmd = f.compute_command(pose, p, 0.1)
        if f.mode is T and abs(cmd.angular) > 1e-6 and (not signs or signs[-1] != (cmd.angular > 0)):
            signs.append(cmd.angular > 0)
        pose = integrate_pose(pose, cmd, 0.1)
    assert f.is_goal_reached(pose, p), (deg, dist, keep_motion, pose)
    assert len(signs) <= 1, f"turn direction reversed {len(signs) - 1} times"
