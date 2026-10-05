"""Differential/skid-steer path-following controller.

Input: current ``Pose2D`` + a global ``Path``.
Output: ``VelocityCommand(linear, angular)`` and the ``MotionMode`` it
represents (``PathFollower.mode``).

Two behaviors, chosen by ``PathFollowerConfig.turn_in_place_enter``:

LEGACY (``turn_in_place_enter=None``, the dataclass default): heading-
proportional control with a cosine speed profile -- forward speed tapers to
zero as the heading error approaches +-90 deg, so the robot drives wide
forward arcs for moderate errors, turns in place only beyond 90 deg, and
never reverses.

BIDIRECTIONAL (``turn_in_place_enter`` set; config/robot.yaml ``control:``):
three explicit motion modes, decided per tick against the CURRENT local
target (the next path point, not the final goal):
    FORWARD          the target is within ``turn_in_place_enter`` of the
                     robot's FRONT: drive forward, steering out the error.
    REVERSE          the target is within ``turn_in_place_enter`` of the
                     robot's REAR, reversing is allowed (config, the
                     waypoint's ``direction``, and ``reverse_allowed`` -- the
                     caller's rear-clearance verdict), and the path left to
                     cover in that direction is at most
                     ``max_reverse_distance``: drive backward, steering out
                     the error (the rear is the "nose").
    ROTATE_IN_PLACE  otherwise: linear 0, turn toward whichever alignment
                     (front or, if reversing is eligible, rear) needs less
                     turning -- the rear only if it saves ``reverse_bias`` --
                     until the error is below ``turn_in_place_exit``.
Hysteresis: FORWARD/REVERSE continue until the error exceeds ENTER; a turn
continues until it is below EXIT -- no forward/turn chattering.

Acceleration-rate limiting applies to every mode, so a new target (e.g.
after replanning) never commands an instantaneous velocity jump.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from common.geometry import angle_diff, clamp, distance
from common.messages import Path, PathPoint, Pose2D, VelocityCommand
from control.local_planner import LocalPlanner


class MotionMode(Enum):
    STOPPED = "STOPPED"
    FORWARD = "FORWARD"
    REVERSE = "REVERSE"
    ROTATE_IN_PLACE = "ROTATE_IN_PLACE"


@dataclass
class PathFollowerConfig:
    max_linear_velocity: float
    max_angular_velocity: float
    max_linear_acceleration: float
    max_angular_acceleration: float
    goal_tolerance: float
    heading_gain: float
    waypoint_tolerance: float | None = None  # defaults to goal_tolerance if omitted
    # --- Bidirectional motion (None = LEGACY forward-only cosine profile). ---
    turn_in_place_enter: float | None = None   # radians
    turn_in_place_exit: float = math.radians(10.0)
    min_turn_velocity: float = 0.3             # rad/s floor while turning in place (avoids stalling near EXIT)
    allow_reverse: bool = False
    max_reverse_velocity: float | None = None  # m/s; None = max_linear_velocity
    max_reverse_distance: float = 1.5          # m of path covered backward in one go
    reverse_bias: float = math.radians(20.0)   # turn to face the target with the REAR only if that saves this much turning

    def __post_init__(self) -> None:
        if self.max_linear_velocity <= 0.0 or self.max_angular_velocity <= 0.0:
            raise ValueError("max velocities must be > 0")
        if self.max_linear_acceleration <= 0.0 or self.max_angular_acceleration <= 0.0:
            raise ValueError("max accelerations must be > 0")
        if self.goal_tolerance <= 0.0:
            raise ValueError("goal_tolerance must be > 0")
        if self.turn_in_place_enter is not None:
            if not 0.0 < self.turn_in_place_exit < self.turn_in_place_enter < math.pi / 2:
                raise ValueError("need 0 < turn_in_place_exit < turn_in_place_enter < 90 deg")
            if self.min_turn_velocity < 0.0 or self.max_reverse_distance < 0.0:
                raise ValueError("min_turn_velocity and max_reverse_distance must be >= 0")
        if self.max_reverse_velocity is not None and self.max_reverse_velocity <= 0.0:
            raise ValueError("max_reverse_velocity must be > 0")

    @property
    def bidirectional(self) -> bool:
        return self.turn_in_place_enter is not None


def path_follower_config(robot_cfg) -> PathFollowerConfig:
    """``PathFollowerConfig`` from a ``common.config.RobotConfig`` (robot.yaml)."""
    limits, control = robot_cfg.limits, robot_cfg.control
    enter = control.turn_in_place_enter_deg
    return PathFollowerConfig(
        max_linear_velocity=limits.max_linear_velocity,
        max_angular_velocity=limits.max_angular_velocity,
        max_linear_acceleration=limits.max_linear_acceleration,
        max_angular_acceleration=limits.max_angular_acceleration,
        goal_tolerance=robot_cfg.safety.goal_tolerance,
        heading_gain=control.heading_gain,
        waypoint_tolerance=control.waypoint_tolerance,
        turn_in_place_enter=None if enter is None else math.radians(enter),
        turn_in_place_exit=math.radians(control.turn_in_place_exit_deg),
        min_turn_velocity=min(control.min_turn_velocity, limits.max_angular_velocity),
        allow_reverse=control.allow_reverse,
        max_reverse_velocity=(None if control.max_reverse_velocity is None
                              else min(control.max_reverse_velocity, limits.max_linear_velocity)),
        max_reverse_distance=control.max_reverse_distance,
        reverse_bias=math.radians(control.reverse_bias_deg),
    )


class PathFollower:
    """Converts (pose, path) into a rate- and velocity-limited VelocityCommand."""

    def __init__(self, config: PathFollowerConfig) -> None:
        self.config = config
        self._local_planner = LocalPlanner(
            waypoint_tolerance=config.waypoint_tolerance or config.goal_tolerance
        )
        self._last_command = VelocityCommand(linear=0.0, angular=0.0)
        self._last_target: PathPoint | None = None
        self.mode = MotionMode.STOPPED
        self._rotate_to_rear = False       # ROTATE_IN_PLACE: aligning the rear (True) or the front
        self.mode_changes: list[tuple[MotionMode, MotionMode]] = []  # (from, to), for diagnostics

    def reset(self, keep_motion: bool = False) -> None:
        """Call when a new path is assigned (e.g. after replanning).

        ``keep_motion=True`` -- a periodic replan (same goal, fresher path)
        while the robot keeps following: only the waypoint index restarts. The
        motion mode, the chosen turn direction and the velocity ramp carry
        over, so a turn in place is not re-decided (or cut short at ENTER
        instead of EXIT) and the speed is not ramped from zero every replan."""
        self._local_planner.reset()
        self._last_target = None
        if keep_motion:
            return
        self._last_command = VelocityCommand(linear=0.0, angular=0.0)
        self._set_mode(MotionMode.STOPPED)

    def halt(self) -> None:
        """The caller stopped the robot (e.g. a safety check): the next command
        ramps up from zero, and the motion mode is decided afresh."""
        self._last_command = VelocityCommand(linear=0.0, angular=0.0)
        self._set_mode(MotionMode.STOPPED)

    @property
    def last_command(self) -> VelocityCommand:
        return self._last_command

    def get_local_target(self) -> PathPoint | None:
        """The waypoint targeted by the most recent ``compute_command`` call.

        A cached read, not a recomputation -- recomputing via the
        internal ``LocalPlanner`` would risk advancing its waypoint index
        a second time for the same tick (e.g. if a GUI reads this
        alongside the controller that actually drives the robot).
        """
        return self._last_target

    def is_goal_reached(self, pose: Pose2D, path: Path) -> bool:
        if path.is_empty():
            return True
        goal = path.points[-1]
        return distance(pose.x, pose.y, goal.x, goal.y) <= self.config.goal_tolerance

    def compute_command(self, pose: Pose2D, path: Path, dt: float, reverse_allowed: bool = True) -> VelocityCommand:
        """``reverse_allowed``: the caller's verdict that reversing is currently
        safe (e.g. rear clearance); combined with the config and the path."""
        if dt < 0.0:
            raise ValueError("dt must be >= 0")

        if path.is_empty() or self.is_goal_reached(pose, path):
            desired = VelocityCommand(linear=0.0, angular=0.0)
            self._set_mode(MotionMode.STOPPED)
        else:
            target = self._local_planner.select_target(pose, path)
            assert target is not None  # path is non-empty here
            self._last_target = target
            if self.config.bidirectional:
                desired = self._bidirectional(pose, path, target, reverse_allowed)
            else:
                desired = self._legacy(pose, target)

        limited = self._apply_acceleration_limits(desired, dt)
        self._last_command = limited
        return limited

    # ------------------------------------------------------------------ modes --

    def _legacy(self, pose: Pose2D, target: PathPoint) -> VelocityCommand:
        desired_heading = math.atan2(target.y - pose.y, target.x - pose.x)
        heading_error = angle_diff(desired_heading, pose.theta)
        angular = clamp(self.config.heading_gain * heading_error,
                        -self.config.max_angular_velocity, self.config.max_angular_velocity)
        # Full speed when facing the target head-on, tapering to zero
        # (and never reversing) as the heading error approaches +-90 deg
        # or beyond -- the robot turns in place instead of backing up.
        speed_scale = max(0.0, math.cos(heading_error))
        self._set_mode(MotionMode.FORWARD if speed_scale > 0.0 else MotionMode.ROTATE_IN_PLACE)
        return VelocityCommand(linear=self.config.max_linear_velocity * speed_scale, angular=angular)

    def _bidirectional(self, pose: Pose2D, path: Path, target: PathPoint, reverse_allowed: bool) -> VelocityCommand:
        c = self.config
        bearing = math.atan2(target.y - pose.y, target.x - pose.x)
        err_front = angle_diff(bearing, pose.theta)
        err_rear = angle_diff(bearing, pose.theta + math.pi)
        forward_ok = target.direction != PathPoint.REVERSE
        reverse_ok = (c.allow_reverse and reverse_allowed and target.direction != PathPoint.FORWARD
                      and self._reverse_stretch(pose, path) <= c.max_reverse_distance)
        if not forward_ok and not reverse_ok:
            # The path demands reverse but reversing is not possible right now:
            # hold still rather than drive forward against the path's intent.
            self._set_mode(MotionMode.STOPPED)
            return VelocityCommand(0.0, 0.0)
        enter, exit_ = c.turn_in_place_enter, c.turn_in_place_exit

        mode = self.mode
        if mode is MotionMode.FORWARD and not (forward_ok and abs(err_front) <= enter):
            mode = MotionMode.STOPPED
        elif mode is MotionMode.REVERSE and not (reverse_ok and abs(err_rear) <= enter):
            mode = MotionMode.STOPPED
        elif mode is MotionMode.ROTATE_IN_PLACE:
            if self._rotate_to_rear and not reverse_ok:
                self._rotate_to_rear = False
            if not self._rotate_to_rear and not forward_ok:
                self._rotate_to_rear = True
            aligned_err = err_rear if self._rotate_to_rear else err_front
            if abs(aligned_err) < exit_:
                mode = MotionMode.REVERSE if self._rotate_to_rear else MotionMode.FORWARD

        if mode is MotionMode.STOPPED:  # fresh decision
            if forward_ok and abs(err_front) <= enter:
                mode = MotionMode.FORWARD
            elif reverse_ok and abs(err_rear) <= enter:
                mode = MotionMode.REVERSE
            else:
                mode = MotionMode.ROTATE_IN_PLACE
                # Turn toward the alignment that needs less turning; forward is
                # preferred unless the rear saves at least reverse_bias.
                self._rotate_to_rear = reverse_ok and (not forward_ok or abs(err_rear) + c.reverse_bias < abs(err_front))
        self._set_mode(mode)

        if mode is MotionMode.FORWARD:
            return VelocityCommand(c.max_linear_velocity * math.cos(err_front),
                                   clamp(c.heading_gain * err_front, -c.max_angular_velocity, c.max_angular_velocity))
        if mode is MotionMode.REVERSE:
            v = c.max_reverse_velocity or c.max_linear_velocity
            return VelocityCommand(-v * math.cos(err_rear),
                                   clamp(c.heading_gain * err_rear, -c.max_angular_velocity, c.max_angular_velocity))
        err = err_rear if self._rotate_to_rear else err_front
        speed = clamp(c.heading_gain * abs(err), c.min_turn_velocity, c.max_angular_velocity)
        return VelocityCommand(0.0, math.copysign(speed, err))

    def _reverse_stretch(self, pose: Pose2D, path: Path) -> float:
        """Path length the robot would cover backward before it has to turn
        anyway: from the robot through the upcoming waypoints while each
        segment keeps within ``turn_in_place_enter`` of the direction toward
        the current local target -- the direction it would reverse in.

        Deliberately independent of the robot's CURRENT heading. Measuring
        against the current rear made the stretch 0 m while the rear was still
        turning toward the target (so reversing looked eligible) and the full
        length once aligned (too long) -- the turn then reversed direction,
        and every periodic replan chose the rear again: the left-right turning
        loop seen on the robot with a goal ~120 deg off (2026-09-30)."""
        c = self.config
        points = path.points[self._local_planner.index:]
        if not points:
            return 0.0
        direction = math.atan2(points[0].y - pose.y, points[0].x - pose.x)
        length, px, py = 0.0, pose.x, pose.y
        for point in points:
            seg = distance(px, py, point.x, point.y)
            if seg > 1e-9 and abs(angle_diff(math.atan2(point.y - py, point.x - px), direction)) > c.turn_in_place_enter:
                break
            length += seg
            if length > c.max_reverse_distance:
                break
            px, py = point.x, point.y
        return length

    def _set_mode(self, mode: MotionMode) -> None:
        if mode is not self.mode:
            self.mode_changes.append((self.mode, mode))
            self.mode = mode

    def _apply_acceleration_limits(self, desired: VelocityCommand, dt: float) -> VelocityCommand:
        max_dv = self.config.max_linear_acceleration * dt
        max_domega = self.config.max_angular_acceleration * dt

        linear = clamp(desired.linear, self._last_command.linear - max_dv, self._last_command.linear + max_dv)
        angular = clamp(
            desired.angular, self._last_command.angular - max_domega, self._last_command.angular + max_domega
        )

        linear = clamp(linear, -self.config.max_linear_velocity, self.config.max_linear_velocity)
        angular = clamp(angular, -self.config.max_angular_velocity, self.config.max_angular_velocity)
        return VelocityCommand(linear=linear, angular=angular)
