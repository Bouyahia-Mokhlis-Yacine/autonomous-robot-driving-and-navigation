"""Real-sensor walkthrough for the CURRENT localization/mapping stack.

Live RPLiDAR -> ScanMatchingPipeline (ICP, time-aware motion validation)
-> PersistentWorldModel (tentative/confirmed map, max-range policy)
-> LocalizationHealthMonitor -> NavigationGUI (default: light map, click a
target, A* plan) or WorldModelGUI (``--viewer world-model``: dark map viewer).

Navigation view: left-click sets a target, right-click cancels. The planner,
path smoothing, replanning, reactive stop and localization-loss stop are the
existing ``NavigationManager``, configured exactly as ``ClosedLoopSimulation``
configures it (planning map = inflated world model, rebuilt every 0.3 s; raw
map passed in so the doorway buffer escape applies; health monitor attached).
Its "robot" is a ``SensorOnlyRobot``: pose = the live SLAM estimate, velocity
commands go nowhere. Carry the sensor along the green path; GOAL_REACHED fires
when you physically arrive.

The same composition as ``visualize_replay.py`` (world model as both the
pipeline's reference provider and its integrator, ``build_visualization_state``
into a ``LatestStateBox`` read by ``WorldModelGUI``), with the fixture replaced
by the real driver, and the pipeline/world-model/health configuration built
exactly as ``simulation/closed_loop.py`` builds it from config/navigation.yaml
and config/robot.yaml. ``real_sensor_walkthrough.py`` (the older BreezySLAM
stack) is untouched and still available.

SENSOR-ONLY: no motors, no navigation, no velocity commands. Carry the sensor
(or push the robot) by hand. Consequences, by design of the existing pipeline:
    - no commanded-motion prior: ICP is seeded from the scan-derived velocity
      estimate (clamped to the robot's velocity limits); there is nothing to
      delay-compensate either, so the pose shown is the last accepted SLAM pose;
    - hand motion is still validated against the robot's limits
      (0.5 m/s, 1.5 rad/s, +50% margin): moving or turning the sensor faster
      than the robot could will be rejected as implausible motion;
    - bootstrap: keep the sensor STILL for the first ~1 s, until the status
      panel says "Bootstrap: COMPLETE" (the pipeline assumes it is stationary
      until then and builds its first confirmed map at the origin).

Threads (the pattern real_sensor_walkthrough.py established on this hardware):
a reader thread drains the serial stream continuously into a single
latest-scan slot, and a processing thread takes whatever is newest when it is
free (latest-only: the policy the simulator's realistic mode models). A slow
processing step can therefore never back up the serial buffer.

Run:
    conda activate AI-env
    python real_world_model_walkthrough.py                         # port/baud from config/hardware.yaml
    python real_world_model_walkthrough.py --port /dev/ttyUSB1
    python real_world_model_walkthrough.py --record /tmp/walk.json # also save every received scan
    python real_world_model_walkthrough.py --viewer world-model    # the dark map-only viewer
    python real_world_model_walkthrough.py --max-range 5 --forget-after 2   # the defaults (0 disables either)
    python real_world_model_walkthrough.py --map-size 20                    # larger area, smaller on screen

Live-map rules (defaults requested for hand-held testing, both optional):
    --max-range 5.0     readings farther than 5 m are dropped (treated as "no
                        return", which the max-range policy then ignores)
    --forget-after 2.0  occupied geometry not HIT for 2 s is removed from the map
                        (PersistentWorldModel occupied_timeout). Removes stale
                        clutter/noise, but also forgets walls behind you or
                        occluded ones, so the map only holds recently seen
                        geometry and drift is not corrected against older map.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import threading
import time
from dataclasses import dataclass, field

from common.config import (
    NavigationConfig,
    RobotConfig,
    load_hardware_config,
    load_navigation_config,
    load_robot_config,
)
from common.logger import configure_logging, get_logger
from common.messages import Goal, LaserScan, Pose2D
from control.path_follower import PathFollower, path_follower_config
from navigation.motion_safety import MotionSafety, motion_safety_config
from mapping.occupancy_grid import OccupancyGridMap
from mapping.persistent_world_model import PersistentWorldModel
from navigation.navigation_manager import NavigationManager
from navigation.replanner import Replanner, ReplannerConfig
from mapping.inflation import compute_inflation_radius, inflate_occupancy_grid
from navigation.world_model_bridge import apply_accepted_pose, robot_pose_from_lidar
from slam.motion_model import mounted_prior
from planning.global_planner import GlobalPlanner
from robot.sensor_only_robot import SensorOnlyRobot
from slam.localization_health import LocalizationHealthConfig, LocalizationHealthMonitor
from slam.scan_matching_pipeline import BootstrapConfig, PipelineConfig, PipelineResult, ScanMatchingPipeline
# Tk-free: this module's processing classes run headless on the Pi (runtime/backends.py LiveBackend).
# The Tk viewers live in visualization/live_walkthrough_gui.py and are imported only by main().
from visualization.gui_state import GuiDataSource, LatestStateBox, build_visualization_state

logger = get_logger(__name__)

PLANNING_REBUILD_INTERVAL = 0.3  # seconds; same as simulation/closed_loop.py and real_sensor_walkthrough.py


def limit_range(scan: LaserScan, max_range: float | None) -> LaserScan:
    """Drop everything at or beyond ``max_range``: those bins become the
    scan's no-return value (the new ``range_max``), exactly as the driver
    already reports readings beyond the sensor's configured range."""
    if max_range is None or max_range >= scan.range_max:
        return scan
    ranges = [r if r < max_range else max_range for r in scan.ranges]
    return dataclasses.replace(scan, ranges=ranges, range_max=max_range)


def build_stack(
    nav_cfg: NavigationConfig, robot_cfg: RobotConfig, occupied_timeout: float | None = None,
    motion_prior=None,
) -> tuple[PersistentWorldModel, ScanMatchingPipeline, LocalizationHealthMonitor]:
    """World model + pipeline + health monitor, configured exactly as
    ``ClosedLoopSimulation`` configures them -- minus the commanded-motion
    prior, since nothing is commanded in a sensor-only run."""
    m, lh, limits = nav_cfg.mapping, nav_cfg.localization, robot_cfg.limits
    world_model = PersistentWorldModel(
        resolution=m.resolution, width_m=m.width, height_m=m.height,
        origin_x=-m.width / 2.0, origin_y=-m.height / 2.0,
        max_range_policy=m.max_range_policy, max_range_clear_distance=m.max_range_clear_distance,
        occupied_timeout=occupied_timeout,
    )
    map_filter = dict(min_integration_translation=lh.map_update_min_translation_m,
                      min_integration_rotation=math.radians(lh.map_update_min_rotation_deg))
    pipeline_config = PipelineConfig(**map_filter)
    if lh.time_aware_validation:
        pipeline_config = PipelineConfig(
            **map_filter,
            max_linear_velocity=limits.max_linear_velocity,
            max_angular_velocity=limits.max_angular_velocity,
            velocity_margin=lh.velocity_margin,
            translation_tolerance=lh.translation_tolerance_m,
            rotation_tolerance=math.radians(lh.rotation_tolerance_deg),
        )
    pipeline = ScanMatchingPipeline(
        reference_provider=world_model, world_model=world_model, pipeline_config=pipeline_config,
        bootstrap_config=BootstrapConfig(min_duration=lh.bootstrap_duration_s, timeout=lh.bootstrap_timeout_s),
        # Commanded body motion re-expressed for the off-centre LiDAR (the pipeline tracks the sensor).
        motion_prior=mounted_prior(motion_prior, robot_cfg.robot.lidar_x, robot_cfg.robot.lidar_y),
    )
    health = LocalizationHealthMonitor(LocalizationHealthConfig(
        degraded_after_s=lh.degraded_after_s, lost_after_s=lh.lost_after_s,
        recovery_confirmations=lh.recovery_confirmations,
    ))
    return world_model, pipeline, health


@dataclass
class WalkthroughStats:
    received: int = 0
    processed: int = 0
    dropped: int = 0          # overwritten in the latest-scan slot before processing got to them
    accepted: int = 0
    rejections: dict = field(default_factory=dict)
    last_processing_ms: float = 0.0


class LiveProcessor:
    """One scan in -> pipeline, health monitor, trajectory, GUI state out.
    Hardware-free, so it is testable with recorded scans."""

    def __init__(self, nav_cfg: NavigationConfig, robot_cfg: RobotConfig, state_box: LatestStateBox,
                 robot: SensorOnlyRobot | None = None, max_range: float | None = None,
                 forget_after: float | None = None, motion_prior=None) -> None:
        self.world_model, self.pipeline, self.health = build_stack(nav_cfg, robot_cfg, occupied_timeout=forget_after,
                                                                   motion_prior=motion_prior)
        # (pose, the instant it describes) of the newest accepted result -- for delay compensation.
        self.last_accepted: tuple[Pose2D, float] | None = None
        self.max_range = max_range
        self.state_box = state_box
        self.robot = robot  # when navigating: receives every ACCEPTED pose (apply_accepted_pose)
        self.lidar_offset = (robot_cfg.robot.lidar_x, robot_cfg.robot.lidar_y)
        self.last_scan: LaserScan | None = None
        self.trajectory: list[tuple[float, float]] = []
        self.stats = WalkthroughStats()

    def process(self, scan: LaserScan) -> PipelineResult:
        scan = limit_range(scan, self.max_range)
        start = time.perf_counter()
        result = self.pipeline.process_scan(scan)
        self.stats.last_processing_ms = (time.perf_counter() - start) * 1000.0
        self.stats.processed += 1
        if result.accepted:
            self.stats.accepted += 1
            self.trajectory.append((result.pose.x, result.pose.y))
        elif result.rejection_reason:
            self.stats.rejections[result.rejection_reason] = self.stats.rejections.get(result.rejection_reason, 0) + 1
        if self.robot is not None:
            apply_accepted_pose(self.robot, result, self.lidar_offset)
        if result.accepted:
            # The robot (footprint centre) pose: delay compensation propagates it with body commands.
            self.last_accepted = (robot_pose_from_lidar(result.pose, self.lidar_offset),
                                  scan.timestamp - 0.5 * scan.scan_time)
        self.last_scan = scan
        self.health.on_result(result, scan.timestamp)
        self.health.update(scan.timestamp)
        self.state_box.set(build_visualization_state(
            self.world_model, result, scan, tuple(self.trajectory), processing_time_ms=self.stats.last_processing_ms,
        ))
        return result


def run_threads(driver, processor: LiveProcessor, stop_event: threading.Event,
                recorded: list[LaserScan] | None = None) -> tuple[threading.Thread, threading.Thread]:
    """Start the reader (serial -> latest slot) and processing (latest slot ->
    pipeline) threads. ``driver`` needs only ``iter_scans()``."""
    slot: list[LaserScan | None] = [None]
    slot_lock = threading.Lock()
    new_scan = threading.Event()

    def read() -> None:
        try:
            for scan in driver.iter_scans():
                if stop_event.is_set():
                    break
                processor.stats.received += 1
                if recorded is not None:
                    recorded.append(scan)
                with slot_lock:
                    if slot[0] is not None:
                        processor.stats.dropped += 1  # previous one never got processed
                    slot[0] = scan
                new_scan.set()
        except Exception:
            if not stop_event.is_set():
                logger.exception("Scan reader thread stopped unexpectedly")
        finally:
            new_scan.set()

    def process() -> None:
        while not stop_event.is_set():
            if not new_scan.wait(timeout=0.5):
                continue
            new_scan.clear()
            with slot_lock:
                scan, slot[0] = slot[0], None
            if scan is None:
                continue
            try:
                processor.process(scan)
            except Exception:
                logger.exception("Processing failed for scan at t=%.3f", scan.timestamp)

    reader = threading.Thread(target=read, name="rplidar-reader", daemon=True)
    worker = threading.Thread(target=process, name="slam-worker", daemon=True)
    reader.start()
    worker.start()
    return reader, worker


def build_navigation(nav_cfg: NavigationConfig, robot_cfg: RobotConfig, robot: SensorOnlyRobot,
                     health: LocalizationHealthMonitor, plan_executor=None,
                     planning_timeout_s: float = 2.0, scan_provider=None) -> NavigationManager:
    """The existing NavigationManager, wired as ``ClosedLoopSimulation`` wires it
    (``plan_executor``: plan asynchronously -- see NavigationManager;
    ``scan_provider``: the newest raw scan, for direction-aware motion safety)."""
    follower = PathFollower(path_follower_config(robot_cfg))
    return NavigationManager(
        global_planner=GlobalPlanner(
            allow_diagonal=nav_cfg.planning.allow_diagonal,
            unknown_cost_multiplier=nav_cfg.planning.unknown_cost_multiplier,
        ),
        path_follower=follower,
        robot=robot,
        replanner=Replanner(ReplannerConfig(
            replan_interval=nav_cfg.planning.replan_interval,
            deviation_threshold=robot_cfg.safety.path_deviation_threshold,
        )),
        obstacle_stop_distance=robot_cfg.safety.emergency_stop_distance,
        path_clearance_margin=nav_cfg.planning.path_clearance_margin,
        localization=health,
        plan_executor=plan_executor,
        planning_timeout_s=planning_timeout_s,
        motion_safety=MotionSafety(motion_safety_config(robot_cfg), scan_provider=scan_provider),
    )


class NavigationSession:
    """Planning map + NavigationManager tick for the navigation view. Runs on
    the GUI thread (``step``), reading the world model the worker thread
    updates; the planning map is replaced (never mutated) on each rebuild."""

    def __init__(self, nav_cfg: NavigationConfig, robot_cfg: RobotConfig, processor: LiveProcessor,
                 robot: SensorOnlyRobot, inflation_scale: float = 1.0, plan_executor=None,
                 planning_timeout_s: float = 2.0) -> None:
        if inflation_scale <= 0.0:
            raise ValueError("inflation_scale must be > 0")
        self.robot_cfg = robot_cfg
        self.processor = processor
        self.robot = robot
        # config/robot.yaml's radius (robot half-diagonal + safety margin), scaled
        # for this live tool only; 1.0 = exactly what build_planning_map uses.
        self.inflation_radius = inflation_scale * compute_inflation_radius(
            robot_cfg.robot.width, robot_cfg.robot.length, robot_cfg.safety.inflation_radius)
        self.nav_manager = build_navigation(nav_cfg, robot_cfg, robot, processor.health,
                                            plan_executor=plan_executor, planning_timeout_s=planning_timeout_s,
                                            scan_provider=lambda: processor.last_scan)
        self.planning_map: OccupancyGridMap = self.build_planning_map()
        self._last_rebuild = time.monotonic()

    def step(self, dt: float) -> None:
        self.processor.health.update(time.time())
        if time.monotonic() - self._last_rebuild >= PLANNING_REBUILD_INTERVAL:
            self.planning_map = self.build_planning_map()
            self._last_rebuild = time.monotonic()
        self.nav_manager.update(self.planning_map, dt, self.processor.world_model.occupancy_grid)

    def build_planning_map(self) -> OccupancyGridMap:
        """``navigation.world_model_bridge.build_planning_map`` with this
        session's inflation radius (same grid layout, same inflation code)."""
        source = self.processor.world_model.get_grid()
        planning_map = OccupancyGridMap(
            resolution=source.resolution, width_m=source.width * source.resolution,
            height_m=source.height * source.resolution, origin_x=source.origin_x, origin_y=source.origin_y,
        )
        planning_map.grid = inflate_occupancy_grid(source, self.inflation_radius)
        return planning_map

    def set_goal(self, x: float, y: float) -> None:
        """A clicked target. Passes the RAW map too (NavigationGUI's own click
        handler does not), so a start inside the inflation buffer -- common in
        a cramped room -- gets the buffer escape / recovery instead of BLOCKED."""
        self.nav_manager.set_goal(Goal(x=x, y=y), self.planning_map, self.processor.world_model.occupancy_grid)

    def data_source(self) -> GuiDataSource:
        return GuiDataSource(
            get_static_map=lambda: self.processor.world_model.occupancy_grid,
            get_planning_map=lambda: self.planning_map,
            get_pose=self.robot.get_pose,
            get_scan=lambda: self.processor.last_scan,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live RPLiDAR -> ScanMatchingPipeline -> PersistentWorldModel viewer.")
    parser.add_argument("--port", default=None, help="serial port (default: config/hardware.yaml)")
    parser.add_argument("--record", default=None, help="save every received scan to this JSON file on exit")
    parser.add_argument("--viewer", choices=("navigation", "world-model"), default="navigation",
                        help="navigation: light map + click a target to plan (default); world-model: dark map viewer")
    parser.add_argument("--map-size", type=float, default=12.0,
                        help="side of the square live map in metres, centred on the start point (default 12: "
                             "6 m each way, enough for the 5 m range limit)")
    parser.add_argument("--inflation-scale", type=float, default=0.1,
                        help="obstacle inflation as a fraction of config/robot.yaml's (0.57 m for this robot). "
                             "Default 0.1 for hand-held viewing; below ~0.5 the buffer is smaller than the robot "
                             "itself, so planned paths are NOT safe for a real robot. Use 1.0 before any motors.")
    parser.add_argument("--scale", type=float, default=None,
                        help="pixels per metre (default: as large as fits the screen)")
    parser.add_argument("--max-range", type=float, default=5.0,
                        help="drop readings farther than this (m); 0 = keep the sensor's configured range")
    parser.add_argument("--forget-after", type=float, default=2.0,
                        help="remove map obstacles not seen for this long (s); 0 = never forget")
    args = parser.parse_args()

    from visualization.live_walkthrough_gui import HealthAwareWorldModelGUI, LiveNavigationGUI, fit_scale

    robot_cfg, nav_cfg = load_robot_config(), load_navigation_config()
    nav_cfg = dataclasses.replace(
        nav_cfg, mapping=dataclasses.replace(nav_cfg.mapping, width=args.map_size, height=args.map_size))
    scale = args.scale or fit_scale(args.map_size)
    rplidar_cfg = load_hardware_config().rplidar
    if args.port:
        rplidar_cfg = dataclasses.replace(rplidar_cfg, port=args.port)
    configure_logging(nav_cfg.logging_level)

    from hardware.rplidar_driver import RPLidarDriver  # needs pyserial/rplidar; imported only for real runs

    state_box = LatestStateBox()
    nav_robot = SensorOnlyRobot(initial_pose=Pose2D(x=0.0, y=0.0, theta=0.0))
    processor = LiveProcessor(nav_cfg, robot_cfg, state_box,
                              robot=nav_robot if args.viewer == "navigation" else None,
                              max_range=args.max_range or None, forget_after=args.forget_after or None)
    logger.info("Live-map rules: max range %s, forget unseen obstacles after %s",
                f"{args.max_range} m" if args.max_range else "off (sensor range)",
                f"{args.forget_after} s" if args.forget_after else "never")
    recorded: list[LaserScan] | None = [] if args.record else None
    stop_event = threading.Event()

    logger.info("Connecting to RPLiDAR on %s @ %d baud ...", rplidar_cfg.port, rplidar_cfg.baudrate)
    driver = RPLidarDriver(hardware_config=rplidar_cfg, lidar_config=nav_cfg.lidar)
    driver.connect()
    reader = worker = None
    try:
        logger.info("Device: %s  health: %s", driver.get_device_info(), driver.get_health())
        logger.warning("SENSOR-ONLY: no motors. Keep the sensor still until 'Bootstrap: COMPLETE', then carry it.")
        reader, worker = run_threads(driver, processor, stop_event, recorded)

        if args.viewer == "navigation":
            session = NavigationSession(nav_cfg, robot_cfg, processor, nav_robot, inflation_scale=args.inflation_scale)
            half_width = robot_cfg.robot.width / 2.0
            log = logger.warning if session.inflation_radius < half_width else logger.info
            log("Obstacle inflation %.3f m (%.0f%% of config %.3f m)%s", session.inflation_radius,
                100 * args.inflation_scale, session.inflation_radius / args.inflation_scale,
                f" -- SMALLER THAN THE ROBOT'S HALF-WIDTH ({half_width:.2f} m): paths may pass where "
                "the real robot cannot. Fine for sensor-only viewing; use --inflation-scale 1 before motors."
                if session.inflation_radius < half_width else "")
            gui = LiveNavigationGUI(session, scale=scale, dt=0.1)
            gui.root.title("LIVE RPLiDAR navigation -- click a target (sensor-only, no motors)")
        else:
            gui = HealthAwareWorldModelGUI(state_box, processor, scale=scale, dt=0.1,
                                           world_size_hint_m=nav_cfg.mapping.width)
            gui.root.title("World model -- LIVE RPLiDAR (sensor-only, no motors)")

        def on_close() -> None:
            logger.info("Window closed; stopping threads and the RPLiDAR motor ...")
            stop_event.set()
            gui.close()

        gui.root.protocol("WM_DELETE_WINDOW", on_close)
        gui.run()
    finally:
        stop_event.set()
        # Same shutdown order as real_sensor_walkthrough.py: let the reader
        # leave iter_scans() (next scan is ~0.1 s away) before closing the port.
        for thread in (reader, worker):
            if thread is not None:
                thread.join(timeout=2.0)
        driver.disconnect()  # stops the scan and the motor
        s = processor.stats
        logger.info("Scans received %d, processed %d, dropped %d, accepted %d, rejected %s; "
                    "localization losses %d", s.received, s.processed, s.dropped, s.accepted,
                    s.rejections, processor.health.loss_events)
        if recorded is not None:
            with open(args.record, "w", encoding="utf-8") as f:
                json.dump([dataclasses.asdict(scan) for scan in recorded], f)
            logger.info("Saved %d scans to %s", len(recorded), args.record)


if __name__ == "__main__":
    main()
