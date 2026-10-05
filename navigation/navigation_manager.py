"""Central navigation state machine.

Owns the goal -> plan -> move -> (replan) -> goal-reached lifecycle.
Nothing else in the project should drive the robot directly once a
``NavigationManager`` is in charge -- GUI code (Phase 11) reads its state
and path for display and calls ``set_goal``/``cancel_goal``/
``emergency_stop`` only, it never owns navigation state itself, per
project constraints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from common.logger import get_logger
from common.messages import Goal, Path, PathPoint, Pose2D, VelocityCommand
import math

from control.path_follower import MotionMode, PathFollower
from mapping.occupancy_grid import OccupancyGridMap
from navigation.goal_manager import validate_goal
from navigation.motion_safety import MotionSafety
from navigation.recovery import (
    RecoveryConfig,
    buffer_escape_point,
    inflated_start_cell_blocked,
    recovery_direction_is_safe,
)
from navigation.replanner import Replanner, obstacle_ahead, path_is_blocked
from planning.async_planner import OK, TIMEOUT, PlanExecutor, PlanRequest, PlanResult
from planning.global_planner import GlobalPlanner
from planning.path_smoothing import process_path
from robot.robot_interface import RobotInterface

logger = get_logger(__name__)


class LocalizationValidity(Protocol):
    """Anything that can say whether the current pose estimate is safe to
    navigate on -- in practice ``slam.localization_health.LocalizationHealthMonitor``."""

    def is_localization_valid(self) -> bool:
        ...


@dataclass(frozen=True)
class _PendingPlan:
    request_id: int
    goal: Goal
    submitted_at: float        # time.monotonic()
    keep_following: bool       # True: a periodic replan -- the current path stays in use meanwhile
    resubmits: int = 0


# A plan that came back already blocked on the (newer) current map is
# re-requested at most this often before giving up (BLOCKED).
_MAX_STALE_RESUBMITS = 2
# Extra wait beyond the planner's own time budget before the manager itself
# declares a timeout (process hand-over, pickling, a restart).
_PLAN_TIMEOUT_GRACE_S = 0.5


class NavigationState(Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    MOVING = "MOVING"
    REPLANNING = "REPLANNING"
    RECOVERING = "RECOVERING"
    GOAL_REACHED = "GOAL_REACHED"
    BLOCKED = "BLOCKED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    LOCALIZATION_LOST = "LOCALIZATION_LOST"


class NavigationManager:
    def __init__(
        self,
        global_planner: GlobalPlanner,
        path_follower: PathFollower,
        robot: RobotInterface,
        replanner: Replanner,
        obstacle_stop_distance: float,
        recovery_config: RecoveryConfig | None = None,
        path_clearance_margin: float = 0.0,
        localization: LocalizationValidity | None = None,
        plan_executor: PlanExecutor | None = None,
        planning_timeout_s: float = 2.0,
        motion_safety: MotionSafety | None = None,
    ) -> None:
        """``obstacle_stop_distance`` is the reactive "stop now, then
        replan" lookahead distance -- typically ``config.robot.safety.
        emergency_stop_distance`` -- not to be confused with the
        ``EMERGENCY_STOP`` *state*, which is only entered via an explicit
        ``emergency_stop()`` call.

        ``recovery_config`` (Phase 13, optional, appended last so every
        existing positional call site is unaffected): tuning for the
        ``RECOVERING`` state -- see ``navigation.recovery``. Defaults to
        ``RecoveryConfig()`` if omitted.

        ``path_clearance_margin`` (Phase 15, optional): forwarded to
        ``planning.path_smoothing.process_path`` as ``clearance_m`` --
        see that module's docstring. Default ``0.0`` matches
        ``process_path``'s own default (no extra margin), so every
        existing caller that doesn't pass this is unaffected; the
        configured, risk-averse value
        (``common.config.PlanningConfig.path_clearance_margin``) lives
        in configuration, not here -- a caller wanting it passes it
        explicitly, per this project's established Phase 11 pattern.

        ``localization`` (optional): a localization-health source. When
        given, every active state (planning, moving, replanning,
        recovering) first checks ``is_localization_valid()``; if it is
        not, the robot is stopped and the manager waits in
        ``LOCALIZATION_LOST`` -- it never steers, plans, or declares the
        goal reached from a pose that is not trustworthy. Once valid
        again it replans from the recovered pose and resumes. Omitting it
        (the default) keeps the original behavior exactly.

        ``plan_executor`` (optional, e.g. ``planning.async_planner.
        ProcessPlanExecutor``): plan ASYNCHRONOUSLY, so ``update()`` -- the
        control tick -- never runs the search. Omitted (the default), every
        plan is computed inline exactly as before. With it:
            - a request goes to the executor; until its result arrives the
              robot is stopped with no path (new goal, blocked/deviated
              path, reactive stop, after recovery or localization loss) --
              except for a periodic replan, which keeps following the still
              valid current path meanwhile;
            - a result is used only if it answers the CURRENT request (a new
              goal, cancel, STOP or localization loss invalidates older ones)
              and is still clear on the CURRENT planning map (else it is
              re-requested, at most ``_MAX_STALE_RESUBMITS`` times);
            - no path, an error, or no answer within ``planning_timeout_s``
              (+ a small grace) -> stop, no path, BLOCKED.
        ``planning_timeout_s`` is also the planner's own search budget.

        ``motion_safety`` (optional, ``navigation.motion_safety``): every tick
        before a path-following command is sent, check that the zone the
        robot is about to move into (front / rear / turning circle, by the
        follower's motion mode) is clear in the latest raw scan and the
        confirmed map. Reversing is offered to the follower only while the
        rear is clear; any other blocked motion stops the robot (it resumes
        by itself once the zone clears, and periodic replanning continues).
        Omitted: no such check (original behavior)."""
        if obstacle_stop_distance <= 0.0:
            raise ValueError("obstacle_stop_distance must be > 0")
        if planning_timeout_s <= 0.0:
            raise ValueError("planning_timeout_s must be > 0")

        self._global_planner = global_planner
        self._path_follower = path_follower
        self._robot = robot
        self._replanner = replanner
        self._obstacle_stop_distance = obstacle_stop_distance
        self._recovery_config = recovery_config or RecoveryConfig()
        self._path_clearance_margin = path_clearance_margin
        self._localization = localization
        self._plan_executor = plan_executor
        self._planning_timeout_s = planning_timeout_s
        self._motion_safety = motion_safety
        self.safety_stops = 0               # ticks on which motion_safety stopped the robot
        self._safety_blocked = False
        self._pending: _PendingPlan | None = None
        self._plan_request_id = 0
        self.last_plan_latency_s: float | None = None   # async mode: submit -> result applied

        self.state = NavigationState.IDLE
        self._goal: Goal | None = None
        self._path: Path | None = None
        # The pose at which the reactive obstacle-ahead stop (below) most
        # recently fired -- see its use in update() for why this is needed.
        self._reactive_stop_awaiting_follow = False
        # Phase 13 RECOVERING state (see navigation.recovery and update()).
        self._recovery_attempts = 0
        self._recovery_elapsed = 0.0

    def _set_state(self, new_state: NavigationState) -> None:
        if new_state is not self.state:
            logger.info("Navigation state: %s -> %s", self.state.value, new_state.value)
        self.state = new_state

    def get_goal(self) -> Goal | None:
        return self._goal

    def is_localization_valid(self) -> bool:
        """True when no localization source was configured (original behavior)."""
        return self._localization is None or self._localization.is_localization_valid()

    def _enter_localization_lost(self) -> VelocityCommand:
        if self.state is not NavigationState.LOCALIZATION_LOST:
            logger.warning("Localization not valid; stopping until it recovers (goal kept: %s)", self._goal)
        self._pending = None  # a plan from the untrusted pose must not be used; replanned on recovery
        self._robot.stop()
        self._recovery_elapsed = 0.0
        self._set_state(NavigationState.LOCALIZATION_LOST)
        return VelocityCommand(linear=0.0, angular=0.0)

    def get_path(self) -> Path | None:
        return self._path

    def get_local_target(self) -> PathPoint | None:
        """The waypoint the path follower is currently steering toward, for display."""
        return self._path_follower.get_local_target()

    def is_planning(self) -> bool:
        """Async mode: a plan request is outstanding."""
        return self._pending is not None

    def get_recovery_attempts(self) -> int:
        """Consecutive RECOVERING attempts since the last successful plan
        or new goal -- for diagnostics/display (Phase 13)."""
        return self._recovery_attempts

    def get_recovery_elapsed(self) -> float:
        """Seconds elapsed in the current RECOVERING maneuver (0.0 outside
        that state) -- for diagnostics/display (Phase 13)."""
        return self._recovery_elapsed

    def set_goal(self, goal: Goal, grid_map: OccupancyGridMap, raw_grid_map: OccupancyGridMap | None = None) -> None:
        """Accept and validate a new goal, then attempt to plan to it.

        ``raw_grid_map`` (Phase 13, optional): the UNINFLATED world-model
        grid, for RECOVERING detection -- see ``_plan``. Omitting it (the
        default) makes this call behave exactly as before Phase 13:
        recovery can never trigger without it.
        """
        if self.state is NavigationState.EMERGENCY_STOP:
            logger.warning("Ignoring new goal while EMERGENCY_STOP is active; call clear_emergency_stop() first")
            return

        self._pending = None  # whatever was being planned is for the previous goal
        valid, reason = validate_goal(grid_map, goal)
        if not valid:
            logger.warning("Goal rejected: %s", reason)
            self._goal = None
            self._path = None
            self._robot.stop()
            self._set_state(NavigationState.BLOCKED)
            return

        self._goal = goal
        self._reactive_stop_awaiting_follow = False
        self._recovery_attempts = 0
        if not self.is_localization_valid():
            # Keep the goal; plan once localization is valid (see update()).
            self._path = None
            self._enter_localization_lost()
            return
        self._set_state(NavigationState.PLANNING)
        self._plan(grid_map, raw_grid_map)

    def cancel_goal(self) -> None:
        """Clear the current goal/path and stop, returning to IDLE (unless e-stopped)."""
        self._pending = None
        self._goal = None
        self._path = None
        self._reactive_stop_awaiting_follow = False
        self._recovery_attempts = 0
        self._recovery_elapsed = 0.0
        self._robot.stop()
        if self.state is not NavigationState.EMERGENCY_STOP:
            self._set_state(NavigationState.IDLE)

    def emergency_stop(self) -> None:
        logger.warning("EMERGENCY STOP triggered")
        self._pending = None
        self._robot.stop()
        self._set_state(NavigationState.EMERGENCY_STOP)

    def clear_emergency_stop(self) -> None:
        if self.state is NavigationState.EMERGENCY_STOP:
            self._set_state(NavigationState.IDLE)

    def _plan(self, grid_map: OccupancyGridMap, raw_grid_map: OccupancyGridMap | None = None,
              keep_following: bool = False) -> None:
        """``keep_following`` (async mode only): a periodic replan -- keep the
        current path in use until the new one arrives."""
        assert self._goal is not None
        start_pose = self._robot.get_pose()

        # Phase 13: the robot's own cell is OCCUPIED only because of the
        # inflation/safety buffer (raw map still says FREE) -- not a
        # genuine collision and not "no path exists" (GlobalPlanner
        # hasn't even been asked yet). Attempt a short, safety-checked
        # reverse instead of immediately reporting BLOCKED, but only if
        # attempts remain and a safe reverse direction actually exists;
        # otherwise fall through to the normal plan() call below, which
        # will itself correctly report BLOCKED (this never weakens a
        # genuine no-path situation -- see navigation.recovery).
        plan_start = start_pose
        in_buffer = raw_grid_map is not None and inflated_start_cell_blocked(start_pose, raw_grid_map, grid_map)
        if in_buffer:
            # Shallow intrusion (normal tracking error along a boundary-
            # hugging path): plan from the nearest buffer-free cell instead
            # of reversing -- see navigation.recovery.buffer_escape_point.
            escape = buffer_escape_point(
                start_pose, raw_grid_map, grid_map, self._recovery_config.max_escape_distance
            )
            if escape is not None:
                logger.info(
                    "Start (%.3f, %.3f) is inside the inflation buffer; planning from the nearest "
                    "buffer-free cell (%.3f, %.3f)", start_pose.x, start_pose.y, escape.x, escape.y,
                )
                plan_start = Pose2D(x=escape.x, y=escape.y, theta=start_pose.theta, timestamp=start_pose.timestamp)
                in_buffer = False
        if in_buffer:
            if self._recovery_attempts < self._recovery_config.max_attempts and recovery_direction_is_safe(
                start_pose, raw_grid_map, self._recovery_config.probe_distance
            ):
                self._recovery_attempts += 1
                self._recovery_elapsed = 0.0
                logger.warning(
                    "Start cell occupied only in the inflated map at (%.3f, %.3f); "
                    "attempting recovery %d/%d",
                    start_pose.x, start_pose.y, self._recovery_attempts, self._recovery_config.max_attempts,
                )
                self._set_state(NavigationState.RECOVERING)
                return

        if self._plan_executor is not None:
            self._submit_plan(grid_map, plan_start, keep_following)
            return

        raw_path = self._global_planner.plan(grid_map, plan_start, self._goal)

        if raw_path is None:
            logger.warning("No path found from %s to goal %s", start_pose, self._goal)
            self._path = None
            self._robot.stop()
            self._set_state(NavigationState.BLOCKED)
            return

        self._recovery_attempts = 0
        self._path = process_path(
            raw_path, grid_map, simplify=True, smooth=True, clearance_m=self._path_clearance_margin
        )
        self._path_follower.reset()
        self._replanner.reset_timer()
        self._set_state(NavigationState.MOVING)

    # ------------------------------------------------ async planning (optional) --

    def _submit_plan(self, grid_map: OccupancyGridMap, plan_start: Pose2D, keep_following: bool,
                     resubmits: int = 0) -> None:
        assert self._plan_executor is not None and self._goal is not None
        if not keep_following:
            # Never keep driving (or later resume) a path planned for an older
            # goal / pose / map while the new one is computed.
            self._path = None
            self._robot.stop()
        self._plan_request_id += 1
        self._pending = _PendingPlan(self._plan_request_id, self._goal, time.monotonic(), keep_following, resubmits)
        self._plan_executor.submit(PlanRequest(
            request_id=self._plan_request_id, grid_map=grid_map, start=plan_start, goal=self._goal,
            planner=self._global_planner, clearance_m=self._path_clearance_margin,
            time_budget_s=self._planning_timeout_s,
        ))

    def _poll_plan(self, grid_map: OccupancyGridMap) -> None:
        """Apply a finished plan for the current request, or time it out."""
        assert self._plan_executor is not None
        for result in self._plan_executor.poll():
            if self._pending is not None and result.request_id == self._pending.request_id:
                self._apply_plan_result(result, grid_map)
        pending = self._pending
        if pending is not None and time.monotonic() - pending.submitted_at > self._planning_timeout_s + _PLAN_TIMEOUT_GRACE_S:
            self._apply_plan_result(PlanResult(pending.request_id, TIMEOUT, None,
                                               time.monotonic() - pending.submitted_at, "no answer from the planner"),
                                    grid_map)

    def _apply_plan_result(self, result: PlanResult, grid_map: OccupancyGridMap) -> None:
        pending = self._pending
        assert pending is not None
        self._pending = None
        self.last_plan_latency_s = time.monotonic() - pending.submitted_at
        if pending.goal != self._goal:  # cannot happen (goal changes invalidate), but never apply a foreign plan
            return
        if result.status == OK and result.path is not None:
            if path_is_blocked(result.path, grid_map):
                # Planned on an older map; the current one blocks it.
                if pending.resubmits < _MAX_STALE_RESUBMITS:
                    logger.info("Plan #%d is blocked on the current map; replanning", result.request_id)
                    self._set_state(NavigationState.REPLANNING)
                    self._submit_plan(grid_map, self._robot.get_pose(), keep_following=False,
                                      resubmits=pending.resubmits + 1)
                    return
                logger.warning("Plans keep arriving blocked on the current map; giving up (BLOCKED)")
            else:
                self._recovery_attempts = 0
                self._path = result.path
                # A periodic replan (kept following meanwhile) keeps the
                # follower's motion mode and ramp; anything else starts afresh.
                self._path_follower.reset(keep_motion=pending.keep_following)
                self._replanner.reset_timer()
                self._set_state(NavigationState.MOVING)
                return
        else:
            logger.warning("Planning %s after %.2f s for goal %s%s", result.status.replace("_", " "),
                           self.last_plan_latency_s, self._goal, f" ({result.detail})" if result.detail else "")
        self._path = None
        self._robot.stop()
        self._set_state(NavigationState.BLOCKED)

    def update(
        self, grid_map: OccupancyGridMap, dt: float, raw_grid_map: OccupancyGridMap | None = None
    ) -> VelocityCommand:
        """Advance the state machine by one control tick.

        Commands the robot (via ``set_velocity``/``stop``) as a side
        effect and returns the same command, for logging/telemetry.

        ``raw_grid_map`` (Phase 13, optional): forwarded to ``_plan`` for
        RECOVERING detection -- see ``set_goal``. Omitting it disables
        recovery entirely (every pre-Phase-13 caller is unaffected).
        """
        if self.state in (
            NavigationState.IDLE,
            NavigationState.GOAL_REACHED,
            NavigationState.BLOCKED,
            NavigationState.EMERGENCY_STOP,
        ):
            self._robot.stop()
            return VelocityCommand(linear=0.0, angular=0.0)

        # Localization safety: never act on an untrustworthy pose.
        if not self.is_localization_valid():
            return self._enter_localization_lost()
        if self.state is NavigationState.LOCALIZATION_LOST:
            # Valid again: replan from the recovered pose (the old path was
            # computed from, and followed with, a pose we no longer trust).
            # Drive again only from the next tick, like every other replan.
            logger.info("Localization recovered; replanning from the recovered pose")
            self._robot.stop()
            self._set_state(NavigationState.REPLANNING)
            self._plan(grid_map, raw_grid_map)
            return VelocityCommand(linear=0.0, angular=0.0)

        if self._pending is not None:
            self._poll_plan(grid_map)
            if self._pending is not None and not self._pending.keep_following:
                self._robot.stop()  # waiting for a plan: stay stopped
                return VelocityCommand(linear=0.0, angular=0.0)
            if self.state is NavigationState.BLOCKED:
                return VelocityCommand(linear=0.0, angular=0.0)

        if self.state is NavigationState.RECOVERING:
            return self._update_recovery(grid_map, raw_grid_map, dt)

        # MOVING or REPLANNING (mid-tick after a triggered replan attempt).
        if self._path is None:
            self._set_state(NavigationState.BLOCKED)
            self._robot.stop()
            return VelocityCommand(linear=0.0, angular=0.0)

        pose = self._robot.get_pose()

        if self._path_follower.is_goal_reached(pose, self._path):
            logger.info("Goal reached at (%.3f, %.3f)", pose.x, pose.y)
            self._pending = None
            self._robot.stop()
            self._set_state(NavigationState.GOAL_REACHED)
            return VelocityCommand(linear=0.0, angular=0.0)

        travel = self._travel_pose(pose)
        if travel is not None and obstacle_ahead(self._reactive_check_pose(travel, grid_map, raw_grid_map),
                                                 grid_map, self._obstacle_stop_distance):
            # Immediate reactive safety: the FIRST time this fires at a
            # given pose, stop this tick unconditionally, then replan, and
            # do not resume driving in the same tick even if replanning
            # succeeds -- wait one tick before ever moving again.
            #
            # But if it still trips on a later tick and the path-following
            # branch below has NOT run since that stop (so the robot has
            # only ever been commanded zero -- it cannot have moved or
            # turned), stopping again achieves nothing and would freeze the
            # robot forever: a zero command never changes heading, so the
            # same situation reproduces the same result on every tick. In
            # that case fall through to normal path-following so the path
            # follower can steer (and, if needed, turn in place) away from
            # the obstacle using the already-fresh replanned path --
            # acceleration-rate-limited from zero (path_follower.reset()
            # already ran inside _plan()), so this is still a gentle,
            # rate-limited resumption, not an unsafe lurch. The next time
            # the check trips after that follow tick, it stops again.
            #
            # This is tracked with a flag, not by comparing poses: the
            # original exact-pose test (< 1e-6) never matched with a real
            # sensor, because the estimated pose of a STATIONARY robot
            # jitters by millimetres from scan to scan -- so it stopped on
            # every tick, path-following never ran, and the robot crawled
            # (realistic stress run E_blackout_recovery, seed 7).
            if not self._reactive_stop_awaiting_follow:
                logger.warning(
                    "Obstacle within %.2f m ahead at (%.3f, %.3f); stopping and replanning",
                    self._obstacle_stop_distance, pose.x, pose.y,
                )
                self._robot.stop()
                self._reactive_stop_awaiting_follow = True
                self._set_state(NavigationState.REPLANNING)
                self._plan(grid_map, raw_grid_map)
                return VelocityCommand(linear=0.0, angular=0.0)
        else:
            self._reactive_stop_awaiting_follow = False

        should_replan, reason = self._replanner.should_replan(pose, self._path, grid_map, dt)
        if should_replan and self._plan_executor is not None:
            periodic = reason.startswith("periodic")
            if not (periodic and self._pending is not None):  # one periodic request at a time
                logger.info("Replanning: %s", reason)
                self._set_state(NavigationState.REPLANNING)
                self._plan(grid_map, raw_grid_map, keep_following=periodic)
                if self._pending is None or not self._pending.keep_following:
                    # blocked / deviated path (or recovery): wait stopped for the new plan
                    self._robot.stop()
                    return VelocityCommand(linear=0.0, angular=0.0)
                self._set_state(NavigationState.MOVING)  # periodic: keep following meanwhile
        elif should_replan:
            logger.info("Replanning: %s", reason)
            self._robot.stop()  # do not continue driving a stale/blocked path while replanning
            self._set_state(NavigationState.REPLANNING)
            self._plan(grid_map, raw_grid_map)
            if self.state is not NavigationState.MOVING:
                return VelocityCommand(linear=0.0, angular=0.0)
            pose = self._robot.get_pose()

        safety = self._motion_safety
        reverse_ok = safety is None or safety.is_clear(MotionMode.REVERSE, pose, raw_grid_map)
        cmd = self._path_follower.compute_command(pose, self._path, dt, reverse_allowed=reverse_ok)
        if safety is not None:
            for mode in safety.modes_to_check(self._path_follower.mode, cmd.linear, cmd.angular):
                if not safety.is_clear(mode, pose, raw_grid_map):
                    if not self._safety_blocked:
                        logger.warning("Motion safety: %s -- stopping", safety.last_block)
                    self._safety_blocked = True
                    self.safety_stops += 1
                    self._robot.stop()
                    self._path_follower.halt()
                    return VelocityCommand(linear=0.0, angular=0.0)
            if self._safety_blocked:
                logger.info("Motion safety: clear again")
            self._safety_blocked = False
        self._robot.set_velocity(cmd)
        self._reactive_stop_awaiting_follow = False  # path-following has now run since any reactive stop
        return cmd

    def get_motion_mode(self) -> MotionMode:
        """The path follower's current motion mode (for display/diagnostics)."""
        return self._path_follower.mode

    def _travel_pose(self, pose: Pose2D) -> Pose2D | None:
        """The pose whose heading is the direction of travel, for the reactive
        obstacle-ahead ray: the robot's heading when driving forward, reversed
        when reversing, None while turning in place (the ray says nothing
        about a turn; motion_safety checks the turning circle). The
        original forward-only follower always travels along its heading."""
        follower = self._path_follower
        if not follower.config.bidirectional:
            return pose
        last = follower.last_command
        if follower.mode is MotionMode.REVERSE or last.linear < -1e-3:
            return Pose2D(x=pose.x, y=pose.y, theta=pose.theta + math.pi, timestamp=pose.timestamp)
        if follower.mode is MotionMode.ROTATE_IN_PLACE and abs(last.linear) <= 1e-3:
            return None
        return pose

    def _reactive_check_pose(
        self, pose: Pose2D, grid_map: OccupancyGridMap, raw_grid_map: OccupancyGridMap | None
    ) -> Pose2D:
        """Where the reactive obstacle-ahead ray starts. Normally the robot's
        pose. But when the robot is only shallowly inside the inflation
        buffer (its own cell inflated, raw FREE -- the situation ``_plan``
        resolves by planning from the nearest buffer-free cell), a ray from
        the robot's own cell starts inside the buffer and reports "obstacle
        ahead" whatever the heading. Measured in the doorway scenario: every
        such stop had no raw obstacle within the stop distance of the
        bumper. So the ray starts from that same buffer-free point, along
        the robot's heading: heading back into the buffer (e.g. toward the
        wall) still stops the robot; continuing out of it does not."""
        if raw_grid_map is None or not inflated_start_cell_blocked(pose, raw_grid_map, grid_map):
            return pose
        escape = buffer_escape_point(pose, raw_grid_map, grid_map, self._recovery_config.max_escape_distance)
        if escape is None:
            return pose
        return Pose2D(x=escape.x, y=escape.y, theta=pose.theta, timestamp=pose.timestamp)

    def _update_recovery(
        self, grid_map: OccupancyGridMap, raw_grid_map: OccupancyGridMap | None, dt: float
    ) -> VelocityCommand:
        """One tick of the bounded RECOVERING maneuver (Phase 13): reverse
        in a straight line at ``self._recovery_config.linear_speed`` for
        at most ``max_duration`` seconds, then stop and re-attempt
        planning. See ``navigation.recovery`` for the safety check
        already performed before this state was ever entered (in
        ``_plan``) -- this method does not re-probe mid-maneuver, per the
        approved design (stop -> re-evaluate -> resume, not a
        continuously-replanned local controller)."""
        self._recovery_elapsed += dt
        if self._recovery_elapsed >= self._recovery_config.max_duration:
            self._robot.stop()
            self._recovery_elapsed = 0.0
            self._set_state(NavigationState.REPLANNING)
            self._plan(grid_map, raw_grid_map)
            return VelocityCommand(linear=0.0, angular=0.0)

        if self._motion_safety is not None and not self._motion_safety.is_clear(
                MotionMode.REVERSE, self._robot.get_pose(), raw_grid_map):
            # The recovery reverse was checked on the map when it started; the
            # live scan now shows something behind -- end it here, replan.
            logger.warning("Recovery reverse stopped by motion safety: %s", self._motion_safety.last_block)
            self.safety_stops += 1
            self._robot.stop()
            self._recovery_elapsed = 0.0
            self._set_state(NavigationState.REPLANNING)
            self._plan(grid_map, raw_grid_map)
            return VelocityCommand(linear=0.0, angular=0.0)
        cmd = VelocityCommand(linear=-self._recovery_config.linear_speed, angular=0.0)
        self._robot.set_velocity(cmd)
        return cmd
