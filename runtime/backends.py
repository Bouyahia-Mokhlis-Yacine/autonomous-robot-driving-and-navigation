"""Robot backends for ``runtime.core.RobotRuntime``: thin adapters over the
EXISTING stacks, so the runtime (and the network API on top of it) is the same
whether the robot is simulated or real.

    SimBackend   ``simulation.closed_loop.ClosedLoopSimulation`` in realistic
                 mode (RPLiDAR model, asynchronous SLAM with latency, health
                 monitor, commanded-motion prediction, delay compensation,
                 doorway/reactive-stop fixes) -- the stress-tested pipeline --
                 stepped in real time, with no built-in goal.
    LiveBackend  real RPLiDAR -> ``real_world_model_walkthrough`` components
                 (``LiveProcessor``: reader + latest-only SLAM worker threads,
                 ``NavigationSession``: planning map + NavigationManager).
                 Robot = ``SensorOnlyRobot`` (sensor-only) or, with
                 ``live.motors: gpio``, ``hardware.motor_driver.GpioDiffDriveRobot``
                 (L298N via the Pi's GPIO), always behind
                 ``SafetyGatedRobot`` (STOP latch) and ``CommandRecorder``
                 (commanded motion -> ICP seed + delay compensation, as in
                 the simulator). Motors also start the separate motor guard
                 process and latch STOP at startup.

Nothing here plans, localizes or maps on its own: every decision is made by the
existing ScanMatchingPipeline / PersistentWorldModel /
LocalizationHealthMonitor / NavigationManager.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
import time
from typing import Callable, Protocol

from common.config import HardwareConfig, NavigationConfig, RobotConfig
from common.messages import Goal, LaserScan, OccupancyGrid, Pose2D, VelocityCommand
from mapping.inflation import compute_inflation_radius
from navigation.navigation_manager import NavigationManager
from robot.robot_interface import RobotInterface
from robot.sensor_only_robot import SensorOnlyRobot
from runtime.config import LiveSettings
from common.geometry import offset_pose
from slam.localization_health import LocalizationHealthMonitor

ZERO = VelocityCommand(linear=0.0, angular=0.0)
logger = logging.getLogger("autonomous_navigation.runtime.backends")


class SafetyGatedRobot:
    """``RobotInterface`` wrapper that refuses to move while ``blocked()`` is
    true (latched STOP): ``set_velocity`` then sends a stop instead. ``stop``
    always passes through. Independent of NavigationManager's own state
    machine, so a latched STOP holds even if something upstream misbehaves.
    Thread-safe for this use: ``stop`` may be called from the network thread."""

    def __init__(self, inner: RobotInterface, blocked: Callable[[], bool]) -> None:
        self.inner = inner
        self._blocked = blocked
        self._lock = threading.Lock()

    def set_velocity(self, cmd: VelocityCommand) -> None:
        with self._lock:
            if self._blocked():
                self.inner.stop()
            else:
                self.inner.set_velocity(cmd)

    def stop(self) -> None:
        with self._lock:
            self.inner.stop()

    def get_pose(self) -> Pose2D:
        return self.inner.get_pose()

    @property
    def last_command(self) -> VelocityCommand:
        return getattr(self.inner, "last_command", ZERO)


class CommandRecorder:
    """Records every command actually handed to the robot (time.time() clock,
    the same clock the RPLiDAR driver stamps scans with) into a
    ``slam.motion_model.CommandHistory``: the commanded-motion prior for ICP
    and the source for delay compensation. A stop is recorded as zero."""

    def __init__(self, inner, history) -> None:
        self.inner = inner
        self.history = history

    def set_velocity(self, cmd: VelocityCommand) -> None:
        self.inner.set_velocity(cmd)
        self.history.record(time.time(), getattr(self.inner, "last_command", cmd))

    def stop(self) -> None:
        self.inner.stop()
        self.history.record(time.time(), ZERO)

    def get_pose(self) -> Pose2D:
        return self.inner.get_pose()

    @property
    def last_command(self) -> VelocityCommand:
        return getattr(self.inner, "last_command", ZERO)


class _BufferClearCounter(logging.Filter):
    """Counts the rplidar library's "Too many measurments in the input buffer
    ... Clearing buffer" events (serial backlog > 500 measurements, i.e. scan
    data discarded because the reader thread fell behind). Logging is unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.clears = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING and "Too many measurments" in str(record.msg):
            self.clears += 1
        return True


class Backend(Protocol):
    name: str
    nav_manager: NavigationManager
    health: LocalizationHealthMonitor

    def tick(self, dt: float) -> None: ...
    def stop_now(self) -> None: ...
    def set_goal(self, x: float, y: float) -> None: ...
    def pose(self) -> Pose2D: ...
    def ground_truth_pose(self) -> Pose2D | None: ...
    def latest_scan(self) -> tuple[LaserScan | None, Pose2D | None]: ...
    def world_grid(self) -> OccupancyGrid: ...
    def planning_grid(self) -> OccupancyGrid: ...
    def map_version(self) -> int: ...
    def stats(self) -> dict: ...
    def close(self) -> None: ...


class SimBackend:
    name = "sim"
    requires_resume_at_start = False

    def __init__(self, nav_cfg: NavigationConfig, robot_cfg: RobotConfig, dt: float, seed: int | None = 42) -> None:
        from simulation.closed_loop import ClosedLoopSimulation
        from simulation.scenario import default_scenario
        from simulation.sensor_simulation_config import load_sensor_simulation_config

        scenario = default_scenario(robot_cfg.robot.width, robot_cfg.robot.length)
        sensor = load_sensor_simulation_config(mode="realistic", seed=seed)
        self.sim = ClosedLoopSimulation(
            environment=scenario.environment, ground_truth_start=scenario.ground_truth_start,
            goal=None, nav_cfg=nav_cfg, robot_cfg=robot_cfg, dt=dt,
            max_simulation_time=float("inf"), sensor_simulation=sensor,
        )
        self.nav_manager = self.sim.nav_manager
        self.health = self.sim.localization_health

    def tick(self, dt: float) -> None:
        self.sim.step()  # one control period of simulated time (sim.dt == control period)

    def stop_now(self) -> None:
        # The simulated "motor command" is the command the physics applies next
        # substep; zero it now (atomic assignment), before the next tick sees
        # the latched stop and puts NavigationManager in EMERGENCY_STOP.
        self.sim._current_cmd = ZERO

    def set_goal(self, x: float, y: float) -> None:
        self.nav_manager.set_goal(Goal(x=x, y=y), self.sim.planning_map, self.sim.world_model.occupancy_grid)

    def pose(self) -> Pose2D:
        return self.sim.nav_robot.get_pose()

    def ground_truth_pose(self) -> Pose2D | None:
        return self.sim._ground_truth_to_slam(self.sim.environment.robot_pose)  # display only

    def latest_scan(self) -> tuple[LaserScan | None, Pose2D | None]:
        done = self.sim.last_completed
        return (done.scan, done.result.pose) if done is not None else (None, None)

    def world_grid(self) -> OccupancyGrid:
        return self.sim.world_model.get_grid()

    def planning_grid(self) -> OccupancyGrid:
        return self.sim.planning_map.grid

    def map_version(self) -> int:
        return self.sim.processor.metrics.scans_processed

    def stats(self) -> dict:
        m = self.sim.processor.metrics
        return {"scans_received": m.scans_received, "scans_processed": m.scans_processed,
                "scans_dropped": m.scans_dropped, "sim_time": round(self.sim._sim_time, 2),
                "collision": self.sim.metrics.collision}

    def close(self) -> None:
        pass


class LiveBackend:
    name = "live"

    def __init__(self, nav_cfg: NavigationConfig, robot_cfg: RobotConfig, hw_cfg: HardwareConfig,
                 live: LiveSettings, stop_latched: Callable[[], bool], port: str | None = None) -> None:
        from real_world_model_walkthrough import LiveProcessor, NavigationSession
        from slam.motion_model import CommandHistory
        from visualization.gui_state import LatestStateBox

        nav_cfg = dataclasses.replace(nav_cfg, mapping=dataclasses.replace(
            nav_cfg.mapping, width=live.map_size_m, height=live.map_size_m))
        self._guard = None
        self.motors_enabled = live.motors == "gpio"
        self.requires_resume_at_start = self.motors_enabled
        # Obstacle inflation: an absolute radius (live.inflation_radius_m) or a
        # fraction of robot.yaml's (half-diagonal + safety margin).
        full_radius = compute_inflation_radius(robot_cfg.robot.width, robot_cfg.robot.length,
                                               robot_cfg.safety.inflation_radius)
        inflation_scale = (live.inflation_radius_m / full_radius) if live.inflation_radius_m > 0 else live.inflation_scale
        self.inflation_radius = inflation_scale * full_radius
        half_diagonal = math.hypot(robot_cfg.robot.width, robot_cfg.robot.length) / 2.0
        logger.info("Obstacle inflation %.3f m (robot half-width %.2f m, half-diagonal %.2f m, config full %.2f m)",
                    self.inflation_radius, robot_cfg.robot.width / 2.0, half_diagonal, full_radius)
        if self.motors_enabled:
            from hardware.motor_driver import GpioDiffDriveRobot
            if hw_cfg.motors is None:
                raise ValueError("live.motors is 'gpio' but config/hardware.yaml has no motors: section")
            if self.inflation_radius < half_diagonal:
                logger.warning("Inflation %.2f m is below the robot's half-diagonal %.2f m WITH MOTORS: a corner "
                               "can clip an obstacle while turning near it. Drive slowly and supervise.",
                               self.inflation_radius, half_diagonal)
            self._inner_robot = GpioDiffDriveRobot(hw_cfg.motors, robot_cfg.limits, dry_run=live.motors_dry_run)
            if live.motor_guard and not self._inner_robot.dry_run:
                from hardware.motor_guard import spawn_motor_guard
                self._guard = spawn_motor_guard([p for pair in hw_cfg.motors.pins.values() for p in pair])
        else:
            self._inner_robot = SensorOnlyRobot(initial_pose=Pose2D(0.0, 0.0, 0.0))
        limits = robot_cfg.limits
        self.command_history = CommandHistory(limits.max_linear_velocity, limits.max_angular_velocity)
        self.delay_compensation = nav_cfg.localization.delay_compensation and self.motors_enabled
        self.robot = SafetyGatedRobot(CommandRecorder(self._inner_robot, self.command_history), stop_latched)
        motion_prior = self.command_history if (self.motors_enabled and nav_cfg.localization.use_commanded_motion) else None
        self.processor = LiveProcessor(nav_cfg, robot_cfg, LatestStateBox(), robot=self._inner_robot,
                                       max_range=live.max_range_m or None, forget_after=live.forget_after_s or None,
                                       motion_prior=motion_prior)
        self._plan_executor = None
        if live.async_planning:
            from planning.async_planner import ProcessPlanExecutor
            self._plan_executor = ProcessPlanExecutor()
        self.session = NavigationSession(nav_cfg, robot_cfg, self.processor, self.robot,
                                         inflation_scale=inflation_scale, plan_executor=self._plan_executor,
                                         planning_timeout_s=live.planning_timeout_s)
        self.nav_manager = self.session.nav_manager
        self.health = self.processor.health

        rplidar_cfg = hw_cfg.rplidar if port is None else dataclasses.replace(hw_cfg.rplidar, port=port)
        self._stop_event = threading.Event()
        self._clear_counter = _BufferClearCounter()
        logging.getLogger("autonomous_navigation.hardware.rplidar_driver").addFilter(self._clear_counter)
        self._threads: tuple = ()
        self._driver = None
        try:
            self._start_lidar(rplidar_cfg, nav_cfg)
        except Exception:
            self._release_motors()
            self._close_planner()
            raise

    def _start_lidar(self, rplidar_cfg, nav_cfg: NavigationConfig) -> None:
        from hardware.rplidar_driver import RPLidarDriver
        from real_world_model_walkthrough import run_threads

        self._driver = RPLidarDriver(hardware_config=rplidar_cfg, lidar_config=nav_cfg.lidar)
        self._driver.connect()
        self._threads = run_threads(self._driver, self.processor, self._stop_event)

    def tick(self, dt: float) -> None:
        if self.delay_compensation and self.processor.last_accepted is not None:
            # As in ClosedLoopSimulation: the newest SLAM pose describes the
            # robot at its scan's time; move it to now with the commands sent since.
            anchor_pose, anchor_time = self.processor.last_accepted
            self._inner_robot.set_pose(self.command_history.propagate(anchor_pose, anchor_time, time.time()))
        self.session.step(dt)

    def stop_now(self) -> None:
        self.robot.stop()

    def set_goal(self, x: float, y: float) -> None:
        self.session.set_goal(x, y)

    def pose(self) -> Pose2D:
        return self._inner_robot.get_pose()

    def ground_truth_pose(self) -> Pose2D | None:
        return None

    def latest_scan(self) -> tuple[LaserScan | None, Pose2D | None]:
        """The newest scan and the pose it is drawn from: the LiDAR's own
        pose (robot pose shifted by the mount), so displays place it right."""
        return self.processor.last_scan, offset_pose(self._inner_robot.get_pose(), *self.processor.lidar_offset)

    def world_grid(self) -> OccupancyGrid:
        return self.processor.world_model.get_grid()

    def planning_grid(self) -> OccupancyGrid:
        return self.session.planning_map.grid

    def map_version(self) -> int:
        return self.processor.stats.processed

    def stats(self) -> dict:
        s = self.processor.stats
        out = {"scans_received": s.received, "scans_processed": s.processed, "scans_dropped": s.dropped,
               "scans_accepted": s.accepted, "motors": "gpio" if self.motors_enabled else "none"}
        latency = self.nav_manager.last_plan_latency_s
        out["last_plan_ms"] = None if latency is None else round(latency * 1000.0, 1)
        out["planning"] = self.nav_manager.is_planning()
        out["lidar_buffer_clears"] = self._clear_counter.clears
        if self.motors_enabled:
            left, right = self._inner_robot.last_duty
            out.update(duty_left=round(left, 3), duty_right=round(right, 3),
                       motor_watchdog_trips=self._inner_robot.watchdog_trips,
                       motors_dry_run=self._inner_robot.dry_run)
        return out

    def close(self) -> None:
        self._stop_event.set()
        self.robot.stop()
        for thread in self._threads:
            thread.join(timeout=2.0)
        logging.getLogger("autonomous_navigation.hardware.rplidar_driver").removeFilter(self._clear_counter)
        if self._driver is not None:
            self._driver.disconnect()
        self._release_motors()
        self._close_planner()

    def _close_planner(self) -> None:
        if self._plan_executor is not None:
            self._plan_executor.close()
            self._plan_executor = None

    def _release_motors(self) -> None:
        if self.motors_enabled:
            self._inner_robot.close()     # stops, then releases the pins
        if self._guard is not None:
            self._guard.stop()            # normal shutdown: motors already stopped by us
            self._guard = None
