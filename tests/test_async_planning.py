"""Asynchronous global planning (NavigationManager + planning.async_planner).

Regression for the first real-sensor test on the Pi: A* ran inside the
control tick and an unreachable goal froze the control loop for 4.8 s.
Planning now runs off the control loop; these tests pin down that
    - an unreachable goal cannot freeze update() / the control loop,
    - STOP and localization-loss handling still run while a plan is pending,
    - a planning timeout leaves the robot stopped (BLOCKED, no path),
    - a finished plan is used only for the current goal/request and only if
      it is still clear on the current map,
    - a new goal that fails never leaves the previous path active.
"""

from __future__ import annotations

import time
import types
from pathlib import Path

import numpy as np
import pytest

import navigation.navigation_manager as nm_module
from common.messages import Goal, OccupancyGrid, Pose2D, VelocityCommand
from control.path_follower import PathFollower, PathFollowerConfig
from mapping.occupancy_grid import OccupancyGridMap
from navigation.navigation_manager import NavigationManager, NavigationState
from navigation.replanner import Replanner, ReplannerConfig
from planning.async_planner import NO_PATH, OK, PlanResult, ProcessPlanExecutor, compute_plan
from planning.global_planner import GlobalPlanner
from tests.test_navigation_manager import FakeRobot

ZERO = VelocityCommand(linear=0.0, angular=0.0)
ROOM_MAP = Path(__file__).parent / "fixtures" / "live_room_map_2026-09-28.npz"


class FakeExecutor:
    """Holds requests until the test delivers a result (or never does)."""

    def __init__(self) -> None:
        self.submitted = []
        self._ready = []

    def submit(self, request) -> None:
        self.submitted.append(request)

    def poll(self):
        out, self._ready = self._ready, []
        return out

    def deliver(self, result: PlanResult) -> None:
        self._ready.append(result)

    def solve(self, request=None) -> PlanResult:
        """Compute the real plan for a request (default: the newest) and queue it."""
        result = compute_plan(request or self.submitted[-1])
        self.deliver(result)
        return result

    def close(self) -> None:
        pass


class Health:
    def __init__(self) -> None:
        self.valid = True

    def is_localization_valid(self) -> bool:
        return self.valid


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(nm_module, "time", types.SimpleNamespace(monotonic=c.monotonic))
    return c


def make(executor, pose=Pose2D(1.0, 1.0, 0.0), replan_interval=100.0, timeout=2.0, health=None):
    robot = FakeRobot(pose)
    follower = PathFollower(PathFollowerConfig(
        max_linear_velocity=1.0, max_angular_velocity=2.0, max_linear_acceleration=5.0,
        max_angular_acceleration=5.0, goal_tolerance=0.15, heading_gain=2.0))
    manager = NavigationManager(
        global_planner=GlobalPlanner(allow_diagonal=True), path_follower=follower, robot=robot,
        replanner=Replanner(ReplannerConfig(replan_interval=replan_interval, deviation_threshold=100.0)),
        obstacle_stop_distance=0.05, localization=health, plan_executor=executor, planning_timeout_s=timeout)
    return manager, robot


def grid(size=10.0, res=0.5) -> OccupancyGridMap:
    return OccupancyGridMap(resolution=res, width_m=size, height_m=size)


def block_cells(g: OccupancyGridMap, cells) -> OccupancyGridMap:
    for r, c in cells:
        g.grid.cells[r, c] = OccupancyGrid.OCCUPIED
    return g


# --------------------------------------------------------------- basic flow --

def test_goal_waits_stopped_for_the_plan_then_moves(clock) -> None:
    ex = FakeExecutor()
    manager, robot = make(ex)
    g = grid()
    manager.set_goal(Goal(8.0, 8.0), g)
    assert manager.state is NavigationState.PLANNING and manager.is_planning()
    assert manager.get_path() is None and len(ex.submitted) == 1
    for _ in range(5):                               # control ticks while the planner works
        assert manager.update(g, 0.1) == ZERO
        assert robot.last_command == ZERO
    ex.solve()
    cmd = manager.update(g, 0.1)
    assert manager.state is NavigationState.MOVING and not manager.is_planning()
    assert manager.get_path() is not None and cmd.linear > 0.0


def test_sync_mode_is_unchanged_without_an_executor() -> None:
    manager, _ = make(None)
    manager.set_goal(Goal(8.0, 8.0), grid())
    assert manager.state is NavigationState.MOVING and manager.get_path() is not None
    assert not manager.is_planning()


# ------------------------------------------ result must match current request --

def test_result_for_a_superseded_goal_is_ignored(clock) -> None:
    ex = FakeExecutor()
    manager, robot = make(ex)
    g = grid()
    manager.set_goal(Goal(8.0, 8.0), g)
    first = ex.submitted[-1]
    manager.set_goal(Goal(2.0, 8.0), g)              # new goal before the first plan came back
    ex.solve(first)                                  # the OLD plan arrives now
    assert manager.update(g, 0.1) == ZERO
    assert manager.get_path() is None and manager.is_planning() and robot.last_command == ZERO
    ex.solve()                                       # the current one
    manager.update(g, 0.1)
    assert manager.state is NavigationState.MOVING
    assert (manager.get_path().points[-1].x, manager.get_path().points[-1].y) == (2.0, 8.0)


def test_result_after_cancel_is_ignored(clock) -> None:
    ex = FakeExecutor()
    manager, robot = make(ex)
    g = grid()
    manager.set_goal(Goal(8.0, 8.0), g)
    manager.cancel_goal()
    ex.solve()
    assert manager.update(g, 0.1) == ZERO
    assert manager.state is NavigationState.IDLE and manager.get_path() is None and robot.last_command == ZERO


def test_plan_blocked_on_the_current_map_is_not_used(clock) -> None:
    """The map changed while planning: a result that crosses a new obstacle is
    re-requested; if every answer is blocked, the robot stays stopped (BLOCKED)."""
    ex = FakeExecutor()
    manager, robot = make(ex)
    old_map = grid()
    manager.set_goal(Goal(8.0, 1.0), old_map)
    stale = ex.solve()                                # planned on the old map
    cells = {old_map.world_to_grid(p.x, p.y) for p in stale.path.points}
    new_map = block_cells(grid(), [old_map.world_to_grid(5.0, 1.0)] + [(r, c) for r, c in cells if 6 <= c <= 12])
    for attempt in range(3):
        assert manager.update(new_map, 0.1) == ZERO
        assert manager.get_path() is None or manager.state is NavigationState.BLOCKED
        if manager.state is NavigationState.BLOCKED:
            break
        assert manager.is_planning() and len(ex.submitted) == attempt + 2   # re-requested on the new map
        ex.deliver(PlanResult(ex.submitted[-1].request_id, OK, stale.path, 0.01))  # still the stale path
    assert manager.state is NavigationState.BLOCKED and manager.get_path() is None
    assert robot.last_command == ZERO


# ------------------------------------------------------- failures are safe --

def test_planning_timeout_leaves_the_robot_stopped(clock) -> None:
    ex = FakeExecutor()                               # never answers
    manager, robot = make(ex, timeout=2.0)
    g = grid()
    manager.set_goal(Goal(8.0, 8.0), g)
    for _ in range(24):                               # 2.4 s of ticks: waiting, stopped
        clock.now += 0.1
        assert manager.update(g, 0.1) == ZERO
    assert manager.is_planning()
    clock.now += 0.2                                  # past budget + grace
    assert manager.update(g, 0.1) == ZERO
    assert manager.state is NavigationState.BLOCKED and not manager.is_planning()
    assert manager.get_path() is None and robot.last_command == ZERO
    ex.solve()                                        # a very late answer changes nothing
    assert manager.update(g, 0.1) == ZERO and manager.state is NavigationState.BLOCKED


def test_failed_new_goal_never_leaves_the_old_path_active(clock) -> None:
    ex = FakeExecutor()
    manager, robot = make(ex)
    g = grid()
    manager.set_goal(Goal(8.0, 8.0), g)
    ex.solve()
    manager.update(g, 0.1)
    assert manager.state is NavigationState.MOVING and robot.last_command.linear > 0.0
    manager.set_goal(Goal(8.0, 2.0), g)               # new goal: old path dropped at once
    assert manager.get_path() is None and robot.last_command == ZERO
    ex.deliver(PlanResult(ex.submitted[-1].request_id, NO_PATH, None, 0.3))
    assert manager.update(g, 0.1) == ZERO
    assert manager.state is NavigationState.BLOCKED and manager.get_path() is None
    for _ in range(3):
        assert manager.update(g, 0.1) == ZERO


# --------------------------------------- safety still runs while planning --

def test_emergency_stop_while_planning(clock) -> None:
    ex = FakeExecutor()
    manager, robot = make(ex)
    g = grid()
    manager.set_goal(Goal(8.0, 8.0), g)
    manager.emergency_stop()
    assert manager.state is NavigationState.EMERGENCY_STOP and not manager.is_planning()
    ex.solve()
    assert manager.update(g, 0.1) == ZERO
    assert manager.state is NavigationState.EMERGENCY_STOP and manager.get_path() is None


def test_localization_loss_while_planning(clock) -> None:
    ex = FakeExecutor()
    health = Health()
    manager, robot = make(ex, health=health)
    g = grid()
    manager.set_goal(Goal(8.0, 8.0), g)
    request_before_loss = ex.submitted[-1]
    health.valid = False
    assert manager.update(g, 0.1) == ZERO
    assert manager.state is NavigationState.LOCALIZATION_LOST and not manager.is_planning()
    ex.solve(request_before_loss)                     # planned from the untrusted pose: must be ignored
    assert manager.update(g, 0.1) == ZERO and manager.get_path() is None
    health.valid = True
    assert manager.update(g, 0.1) == ZERO             # recovered: replans from the recovered pose
    assert manager.is_planning() and len(ex.submitted) == 2
    ex.solve()
    manager.update(g, 0.1)
    assert manager.state is NavigationState.MOVING


def test_periodic_replan_keeps_following_but_a_blocked_path_stops(clock) -> None:
    ex = FakeExecutor()
    manager, robot = make(ex, replan_interval=0.3)
    g = grid()
    manager.set_goal(Goal(8.0, 1.0), g)
    first = ex.solve()
    manager.update(g, 0.1)
    for _ in range(4):                                # periodic replan triggers; still driving
        cmd = manager.update(g, 0.1)
    assert manager.is_planning() and cmd.linear > 0.0 and manager.get_path() is first.path
    submitted = len(ex.submitted)
    manager.update(g, 0.1)
    assert len(ex.submitted) == submitted             # no second periodic request while one is pending
    assert len(first.path.points) == 2                # a straight line: block its middle
    blocked = block_cells(grid(), [g.world_to_grid(x, 1.0) for x in (4.0, 4.5, 5.0)])
    assert manager.update(blocked, 0.1) == ZERO       # path blocked: stop and wait for the new plan
    assert manager.get_path() is None and manager.is_planning()


# ------------------------------------------------ real worker process --

def _room_planning_map():
    data = np.load(ROOM_MAP)
    m = OccupancyGridMap(resolution=float(data["resolution"]), width_m=20.0, height_m=20.0,
                         origin_x=float(data["origin_x"]), origin_y=float(data["origin_y"]))
    m.grid.cells[:] = data["planning_cells"]
    return m


def _unreachable_goal(m: OccupancyGridMap) -> Goal:
    """A free cell walled in, far out in unexplored space: A* has to search
    everything reachable before it can say "no path"."""
    r, c = 20, 20
    m.grid.cells[r - 2:r + 3, c - 2:c + 3] = OccupancyGrid.OCCUPIED
    m.grid.cells[r, c] = OccupancyGrid.FREE
    return Goal(x=m.grid.origin_x + (c + 0.5) * m.grid.resolution, y=m.grid.origin_y + (r + 0.5) * m.grid.resolution)


def _free_start(m: OccupancyGridMap) -> Pose2D:
    free = np.argwhere(m.grid.cells == OccupancyGrid.FREE)
    r, c = free[len(free) // 2]
    return Pose2D(x=m.grid.origin_x + (c + 0.5) * m.grid.resolution, y=m.grid.origin_y + (r + 0.5) * m.grid.resolution,
                  theta=0.0)


@pytest.fixture(scope="module")
def process_executor():
    ex = ProcessPlanExecutor()
    yield ex
    ex.close()


def _drive_until_settled(manager, robot, m, deadline_s):
    longest, end = 0.0, time.monotonic() + deadline_s
    while time.monotonic() < end and manager.is_planning():
        started = time.perf_counter()
        manager.update(m, 0.1)
        longest = max(longest, time.perf_counter() - started)
        if manager.is_planning():
            assert robot.last_command == ZERO         # never moves while the goal is being planned
        time.sleep(0.02)
    return longest


def test_unreachable_goal_cannot_freeze_the_control_tick(process_executor) -> None:
    m = _room_planning_map()
    goal = _unreachable_goal(m)
    manager, robot = make(process_executor, pose=_free_start(m), timeout=10.0)
    started = time.perf_counter()
    manager.set_goal(goal, m)
    assert time.perf_counter() - started < 0.1        # set_goal only hands the request over
    longest = _drive_until_settled(manager, robot, m, 15.0)
    assert manager.state is NavigationState.BLOCKED and manager.get_path() is None
    assert longest < 0.05, f"a control tick took {longest * 1000:.0f} ms while planning"


def test_real_planner_timeout_rejects_promptly(process_executor) -> None:
    m = _room_planning_map()
    goal = _unreachable_goal(m)
    manager, robot = make(process_executor, pose=_free_start(m), timeout=0.15)
    started = time.monotonic()
    manager.set_goal(goal, m)
    _drive_until_settled(manager, robot, m, 5.0)
    assert manager.state is NavigationState.BLOCKED and robot.last_command == ZERO
    assert time.monotonic() - started < 0.15 + 0.5 + 0.3


def test_reachable_goal_through_the_worker_process(process_executor) -> None:
    m = _room_planning_map()
    start = _free_start(m)
    free = np.argwhere(m.grid.cells == OccupancyGrid.FREE)
    r, c = free[0]
    goal = Goal(x=m.grid.origin_x + (c + 0.5) * m.grid.resolution, y=m.grid.origin_y + (r + 0.5) * m.grid.resolution)
    manager, robot = make(process_executor, pose=start, timeout=10.0)
    manager.set_goal(goal, m)
    _drive_until_settled(manager, robot, m, 15.0)
    assert manager.state is NavigationState.MOVING
    sync_manager, _ = make(None, pose=start)
    sync_manager.set_goal(goal, m)
    assert manager.get_path() == sync_manager.get_path()   # the same plan the inline code makes


# ------------------------------------------------- runtime: STOP while planning --

def test_stop_is_processed_while_a_plan_is_outstanding(monkeypatch) -> None:
    from network.client import OperatorClient
    from robot_runtime import build_runtime
    from runtime.backends import LiveBackend
    from runtime.config import load_runtime_config
    from tests.test_live_backend_motors import SimulatedSerialLidar

    def fake_start(self, rplidar_cfg, nav_cfg):
        from real_world_model_walkthrough import run_threads
        self._driver = SimulatedSerialLidar()
        self._threads = run_threads(self._driver, self.processor, self._stop_event)

    monkeypatch.setattr(LiveBackend, "_start_lidar", fake_start)
    config = load_runtime_config()
    config.live.motors, config.live.motors_dry_run, config.live.motor_guard = "gpio", True, False
    config.live.map_size_m, config.live.planning_timeout_s = 14.0, 30.0
    runtime = build_runtime("live", config, host="127.0.0.1", port=0)
    silent = FakeExecutor()                           # a plan that never comes back
    runtime.backend.nav_manager._plan_executor = silent
    runtime.start()
    client = OperatorClient("127.0.0.1", runtime.server.port, heartbeat_s=0.2).start()
    try:
        assert client.wait_for(lambda c: c.state is not None and c.state["health"]["state"] == "LOCALIZED", 15)
        client.resume()
        assert client.wait_for(lambda c: not c.state["safety"]["stop_latched"], 3)
        client.set_goal(1.0, 1.2)
        assert client.wait_for(lambda c: c.state["stats"]["planning"], 3)
        ticks = runtime.tick_count
        time.sleep(0.5)
        assert runtime.tick_count >= ticks + 3        # the control loop keeps running meanwhile
        client.stop()
        assert client.wait_for(lambda c: c.state["safety"]["stop_latched"]
                               and c.state["nav"]["state"] == "EMERGENCY_STOP" and not c.state["stats"]["planning"], 2)
        assert runtime.backend._inner_robot.last_duty == (0.0, 0.0)
        assert runtime.max_tick_s < 0.1
    finally:
        client.close()
        runtime.shutdown()
