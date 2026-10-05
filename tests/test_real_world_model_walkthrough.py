"""real_world_model_walkthrough.py: the live-sensor composition, exercised
without hardware (the real recorded fixture and a fake driver)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from common.config import load_navigation_config, load_robot_config
from common.messages import LaserScan
from real_world_model_walkthrough import LiveProcessor, build_stack, run_threads
from slam.localization_health import LocalizationHealth
from slam.scan_matching_pipeline import BootstrapStatus
from visualization.gui_state import LatestStateBox

NAV_CFG = load_navigation_config()
ROBOT_CFG = load_robot_config()
FIXTURE = Path(__file__).parent / "fixtures" / "real_scans_stationary.json"


def real_scans() -> list[LaserScan]:
    return [LaserScan(**raw) for raw in json.loads(FIXTURE.read_text())]


def test_stack_is_configured_like_the_simulator_without_a_command_prior() -> None:
    wm, pipeline, health = build_stack(NAV_CFG, ROBOT_CFG)
    assert wm.max_range_policy == NAV_CFG.mapping.max_range_policy
    cfg = pipeline._pipeline_config
    assert cfg.time_aware == NAV_CFG.localization.time_aware_validation
    assert cfg.max_linear_velocity == ROBOT_CFG.limits.max_linear_velocity
    assert pipeline._motion_prior is None            # sensor-only: nothing is commanded
    assert health.config.lost_after_s == NAV_CFG.localization.lost_after_s


def test_real_recorded_scans_bootstrap_and_localize() -> None:
    """The 25 real stationary scans (~2.2 s) through the exact live path."""
    box = LatestStateBox()
    proc = LiveProcessor(NAV_CFG, ROBOT_CFG, box)
    assert proc.health.health is LocalizationHealth.INITIALIZING
    results = [proc.process(scan) for scan in real_scans()]
    assert results[-1].bootstrap_status is BootstrapStatus.COMPLETE
    assert proc.health.health is LocalizationHealth.LOCALIZED
    assert proc.stats.processed == 25 and proc.stats.accepted == len(proc.trajectory) > 0
    state = box.get()
    assert state is not None and state.bootstrap_status is BootstrapStatus.COMPLETE
    assert len(state.current_scan_points_world) > 100
    # Stationary sensor: the estimate must stay near the origin it defined.
    assert max(abs(x) for x, _ in proc.trajectory) < 0.1 and max(abs(y) for _, y in proc.trajectory) < 0.1


class FakeDriver:
    """Yields scans like RPLidarDriver.iter_scans(), optionally paced."""

    def __init__(self, scans: list[LaserScan], period: float) -> None:
        self.scans, self.period = scans, period

    def iter_scans(self):
        for scan in self.scans:
            if self.period:
                time.sleep(self.period)
            yield scan


def _run(driver, proc) -> None:
    stop = threading.Event()
    reader, worker = run_threads(driver, proc, stop)
    reader.join(timeout=10.0)
    time.sleep(0.3)                                  # let the worker drain the last scan
    stop.set()
    worker.join(timeout=2.0)
    assert not reader.is_alive() and not worker.is_alive()


def test_threads_process_a_paced_stream_and_stop_cleanly() -> None:
    proc = LiveProcessor(NAV_CFG, ROBOT_CFG, LatestStateBox())
    _run(FakeDriver(real_scans(), period=0.09), proc)   # the sensor's real ~11 Hz
    s = proc.stats
    assert s.received == 25
    assert s.processed + s.dropped == s.received       # every scan accounted for
    assert s.processed >= 20                           # ~15 ms processing keeps up with 90 ms scans


def test_a_burst_faster_than_processing_is_dropped_not_queued() -> None:
    proc = LiveProcessor(NAV_CFG, ROBOT_CFG, LatestStateBox())
    burst = real_scans() * 4                              # 100 scans, no pacing at all
    burst = [LaserScan(**{**scan.__dict__, "timestamp": 1000.0 + 0.001 * i}) for i, scan in enumerate(burst)]
    _run(FakeDriver(burst, period=0.0), proc)
    s = proc.stats
    assert s.received == 100
    assert s.processed + s.dropped == s.received
    assert s.dropped > 0                                  # latest-only: the reader never waits for SLAM


# ------------------------------------------------------- navigation view --

def _free_goal(session, near=(0.0, 0.0), min_d=0.3, max_d=2.0):
    """A cell that is not OCCUPIED in the planning (inflated) map -- FREE or UNKNOWN, both of which
    the planner may route through -- between min_d and max_d from ``near``."""
    import numpy as np
    from common.messages import Goal, OccupancyGrid
    g = session.planning_map.grid
    rows, cols = np.nonzero(g.cells != OccupancyGrid.OCCUPIED)
    xs = g.origin_x + (cols + 0.5) * g.resolution
    ys = g.origin_y + (rows + 0.5) * g.resolution
    d = np.hypot(xs - near[0], ys - near[1])
    order = np.flatnonzero((d > min_d) & (d < max_d))
    # The recorded room is cramped (objects ~0.5-0.75 m away, inflation 0.57 m),
    # so most nearby cells are unreachable: pick one A* can actually reach.
    start = session.robot.get_pose()
    for i in order[np.argsort(d[order])]:
        goal = Goal(float(xs[i]), float(ys[i]))
        if session.nav_manager._global_planner.plan(session.planning_map, start, goal) is not None:
            return goal
    raise AssertionError("no reachable cell in the recorded room")


def _session_after_fixture():
    from common.messages import Pose2D
    from real_world_model_walkthrough import NavigationSession
    from robot.sensor_only_robot import SensorOnlyRobot

    robot = SensorOnlyRobot(Pose2D(0.0, 0.0, 0.0))
    proc = LiveProcessor(NAV_CFG, ROBOT_CFG, LatestStateBox(), robot=robot)
    session = NavigationSession(NAV_CFG, ROBOT_CFG, proc, robot)
    return proc, session, robot


def _open_room_scans(n: int = 30) -> list[LaserScan]:
    """Stationary realistic-sensor scans from the middle of the simulator's
    10 x 8 m room (the recorded fixture room is too cramped to plan in)."""
    from simulation.realistic_lidar import RealisticLidar
    from simulation.scenario import default_scenario
    env = default_scenario(ROBOT_CFG.robot.width, ROBOT_CFG.robot.length).environment
    pose = env.robot_pose
    lidar = RealisticLidar(NAV_CFG.lidar, seed=1, start_time=1000.0)
    return [r.scan for r in lidar.acquire_until(env, lambda t: pose, 1000.0 + n / 11.0) if r.scan is not None]


def test_clicked_target_is_planned_on_the_live_world_model() -> None:
    from navigation.navigation_manager import NavigationState
    from navigation.world_model_bridge import build_planning_map

    proc, session, robot = _session_after_fixture()
    for scan in _open_room_scans():
        proc.process(scan)
    assert proc.health.is_localization_valid()
    session.planning_map = build_planning_map(proc.world_model, ROBOT_CFG)   # what the 0.3 s rebuild does
    goal = _free_goal(session, min_d=1.0, max_d=2.5)
    session.set_goal(goal.x, goal.y)                                         # what a left-click does
    assert session.nav_manager.state is NavigationState.MOVING
    path = session.nav_manager.get_path()
    assert path is not None and len(path.points) >= 2
    from navigation.world_model_bridge import robot_pose_from_lidar
    # nav pose = live SLAM estimate (the LiDAR's), moved to the footprint centre by the mount
    assert robot.get_pose() == robot_pose_from_lidar(proc.pipeline.state.last_good_pose, proc.lidar_offset)


def test_click_passes_the_raw_map_to_the_navigation_manager() -> None:
    """NavigationGUI's own click handler omits the raw map; without it a start
    inside the inflation buffer (common in a cramped room) can never use the
    buffer escape or recovery. The live view's click must pass it."""
    proc, session, robot = _session_after_fixture()
    calls = []
    session.nav_manager.set_goal = lambda goal, grid, raw=None: calls.append((goal, grid, raw))
    session.set_goal(1.0, 2.0)
    (goal, grid, raw), = calls
    assert (goal.x, goal.y) == (1.0, 2.0)
    assert grid is session.planning_map
    assert raw is proc.world_model.occupancy_grid


def test_target_clicked_while_bootstrapping_waits_for_localization() -> None:
    from navigation.navigation_manager import NavigationState

    proc, session, robot = _session_after_fixture()
    proc.process(real_scans()[0])                             # still bootstrapping
    session.nav_manager.set_goal(_free_goal(session, min_d=0.3, max_d=3.0), session.planning_map)
    assert session.nav_manager.state is NavigationState.LOCALIZATION_LOST
    assert session.nav_manager.get_path() is None


# ------------------------------------------------------- live-map rules --

def test_limit_range_drops_everything_beyond_the_limit() -> None:
    from real_world_model_walkthrough import limit_range
    scan = LaserScan(timestamp=0.0, angle_min=0.0, angle_increment=0.1,
                     ranges=[0.1, 1.0, 4.99, 5.0, 7.5, 8.0], range_min=0.15, range_max=8.0)
    out = limit_range(scan, 5.0)
    assert out.range_max == 5.0
    assert out.ranges == [0.1, 1.0, 4.99, 5.0, 5.0, 5.0]   # >= 5 m -> "no return"; invalid 0.1 untouched
    assert limit_range(scan, None) is scan and limit_range(scan, 9.0) is scan


def test_processor_applies_the_range_limit_before_slam() -> None:
    box = LatestStateBox()
    proc = LiveProcessor(NAV_CFG, ROBOT_CFG, box, max_range=1.0)
    for scan in real_scans():
        proc.process(scan)
    import math
    pose = box.get().robot_pose
    assert all(math.hypot(x - pose.x, y - pose.y) < 1.0 + 1e-6 for x, y in box.get().current_scan_points_world)


def test_inflation_scale_shrinks_only_the_live_planning_buffer() -> None:
    from mapping.inflation import compute_inflation_radius
    from navigation.world_model_bridge import build_planning_map

    full = compute_inflation_radius(ROBOT_CFG.robot.width, ROBOT_CFG.robot.length, ROBOT_CFG.safety.inflation_radius)
    proc, session, robot = _session_after_fixture()
    for scan in real_scans():
        proc.process(scan)
    # Scale 1.0 reproduces the shared bridge exactly.
    assert (session.build_planning_map().grid.cells == build_planning_map(proc.world_model, ROBOT_CFG).grid.cells).all()

    from real_world_model_walkthrough import NavigationSession
    small = NavigationSession(NAV_CFG, ROBOT_CFG, proc, robot, inflation_scale=0.1)
    assert abs(small.inflation_radius - 0.1 * full) < 1e-12
    inflated_full = (session.build_planning_map().grid.cells == 100).sum()
    inflated_small = (small.build_planning_map().grid.cells == 100).sum()
    assert inflated_small < inflated_full
    assert load_robot_config().safety.inflation_radius == ROBOT_CFG.safety.inflation_radius  # config untouched
