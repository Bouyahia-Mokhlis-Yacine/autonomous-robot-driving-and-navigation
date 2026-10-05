"""Real-sensor, no-motors walkthrough.

Connects to the actual RPLiDAR, builds a live occupancy map using
LiDAR-only localization (no wheel encoders -- Phase 13), and shows it in
the real navigation GUI (Phase 11): left-click a destination to plan a
path to it with the exact same stack (SLAM, inflation, A*, dynamic-layer
replanning) the real robot will eventually use.

NO MOTORS ARE DRIVEN. There is no robot attached to this script's
``NavigationManager`` -- ``robot/sensor_only_robot.py`` stands in for
one. Carry the sensor by hand toward a clicked goal; because pose comes
from real scan-matching against your real motion, ``GOAL_REACHED`` will
genuinely fire when you physically arrive.

Scan acquisition runs in a dedicated background thread that continuously
drains the sensor's serial stream at its own pace, independent of how
long GUI/planning processing takes on the main thread -- otherwise any
per-tick processing slower than the sensor's native rate lets the serial
input buffer overflow (observed in practice: constant "too many
measurements" warnings and degraded localization when scans were read
synchronously inside the GUI tick).

Run:
    conda activate AI-env
    python real_sensor_walkthrough.py
"""

from __future__ import annotations

import threading
import time

from common.config import load_hardware_config, load_navigation_config, load_robot_config
from common.logger import configure_logging, get_logger
from common.messages import LaserScan, Pose2D
from control.path_follower import PathFollower, PathFollowerConfig
from hardware.rplidar_driver import RPLidarDriver
from mapping.inflation import compute_inflation_radius, inflate_occupancy_grid
from mapping.occupancy_grid import OccupancyGridMap
from navigation.navigation_manager import NavigationManager
from navigation.replanner import DynamicObstacleLayer, Replanner, ReplannerConfig, combine_planning_map
from planning.global_planner import GlobalPlanner
from robot.sensor_only_robot import SensorOnlyRobot
from slam.backend import LidarScanMatchingBackend
from slam.slam import SLAM
from visualization.gui import GuiDataSource, NavigationGUI

logger = get_logger(__name__)

WORLD_SIZE_M = 20.0  # map extent; sensor starts at the center
MAP_RESOLUTION = 0.05
PLANNING_REBUILD_INTERVAL = 0.3  # seconds; inflate/combine don't need to run every tick


def main() -> None:
    robot_cfg = load_robot_config()
    nav_cfg = load_navigation_config()
    hw_cfg = load_hardware_config()
    configure_logging(nav_cfg.logging_level)

    origin = -WORLD_SIZE_M / 2.0

    static_map = OccupancyGridMap(
        resolution=MAP_RESOLUTION, width_m=WORLD_SIZE_M, height_m=WORLD_SIZE_M,
        origin_x=origin, origin_y=origin,
    )
    slam = SLAM(
        backend=LidarScanMatchingBackend(nav_cfg.lidar, nav_cfg.slam),
        occupancy_grid_map=static_map,
    )
    dynamic_layer = DynamicObstacleLayer(
        resolution=MAP_RESOLUTION, width_m=WORLD_SIZE_M, height_m=WORLD_SIZE_M,
        origin_x=origin, origin_y=origin,
    )
    planning_map = OccupancyGridMap(
        resolution=MAP_RESOLUTION, width_m=WORLD_SIZE_M, height_m=WORLD_SIZE_M,
        origin_x=origin, origin_y=origin,
    )
    inflation_radius = compute_inflation_radius(
        robot_cfg.robot.width, robot_cfg.robot.length, robot_cfg.safety.inflation_radius
    )

    robot = SensorOnlyRobot(initial_pose=Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=0.0))
    follower = PathFollower(PathFollowerConfig(
        max_linear_velocity=robot_cfg.limits.max_linear_velocity,
        max_angular_velocity=robot_cfg.limits.max_angular_velocity,
        max_linear_acceleration=robot_cfg.limits.max_linear_acceleration,
        max_angular_acceleration=robot_cfg.limits.max_angular_acceleration,
        goal_tolerance=robot_cfg.safety.goal_tolerance,
        heading_gain=robot_cfg.control.heading_gain,
        waypoint_tolerance=robot_cfg.control.waypoint_tolerance,
    ))
    replanner = Replanner(ReplannerConfig(
        replan_interval=nav_cfg.planning.replan_interval,
        deviation_threshold=robot_cfg.safety.path_deviation_threshold,
    ))
    nav_manager = NavigationManager(
        global_planner=GlobalPlanner(
            allow_diagonal=nav_cfg.planning.allow_diagonal,
            unknown_cost_multiplier=nav_cfg.planning.unknown_cost_multiplier,
        ),
        path_follower=follower,
        robot=robot,
        replanner=replanner,
        obstacle_stop_distance=robot_cfg.safety.emergency_stop_distance,
        path_clearance_margin=nav_cfg.planning.path_clearance_margin,
    )

    logger.info("Connecting to RPLiDAR on %s @ %d baud ...", hw_cfg.rplidar.port, hw_cfg.rplidar.baudrate)
    driver = RPLidarDriver(hardware_config=hw_cfg.rplidar, lidar_config=nav_cfg.lidar)
    driver.connect()
    logger.warning("NO MOTORS ARE DRIVEN -- carry the sensor by hand to reach a clicked destination.")

    # --- background scan reader: keeps the serial buffer drained regardless
    # of how long GUI/planning processing takes on the main thread ---
    latest_scan_box: list[LaserScan | None] = [None]
    stop_event = threading.Event()

    def read_scans() -> None:
        try:
            for scan in driver.iter_scans():
                latest_scan_box[0] = scan
                if stop_event.is_set():
                    break
        except Exception:
            if not stop_event.is_set():
                logger.exception("Scan reader thread stopped unexpectedly")

    reader_thread = threading.Thread(target=read_scans, daemon=True)
    reader_thread.start()

    last_processed_timestamp: float | None = None
    last_tick_time = time.time()
    last_planning_rebuild = 0.0

    def step(_nominal_dt: float) -> None:
        nonlocal last_tick_time, last_processed_timestamp, last_planning_rebuild
        now = time.time()
        dt = max(1e-3, now - last_tick_time)
        last_tick_time = now

        scan = latest_scan_box[0]
        if scan is None or scan.timestamp == last_processed_timestamp:
            return  # nothing new since the last tick; just let the GUI redraw as-is
        last_processed_timestamp = scan.timestamp

        pose = slam.update(scan)  # grows the persistent map AND estimates pose
        robot.set_pose(pose)

        if now - last_planning_rebuild >= PLANNING_REBUILD_INTERVAL:
            dynamic_layer.update(pose, scan)
            combined = combine_planning_map(static_map.grid, dynamic_layer.get_grid())
            planning_map.grid = inflate_occupancy_grid(combined, inflation_radius)
            last_planning_rebuild = now

        nav_manager.update(planning_map, dt)

    data_source = GuiDataSource(
        get_static_map=lambda: static_map,
        get_planning_map=lambda: planning_map,
        get_pose=robot.get_pose,
        get_scan=lambda: latest_scan_box[0],
    )

    gui = NavigationGUI(nav_manager, data_source, scale=30.0, dt=0.1, step_callback=step)
    gui.root.title("Real-sensor walkthrough (no motors -- carry the sensor to the goal)")

    def on_close() -> None:
        logger.info("Window closed; stopping scan reader and disconnecting RPLiDAR...")
        stop_event.set()
        reader_thread.join(timeout=2.0)
        driver.disconnect()
        gui.close()

    gui.root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        gui.run()
    finally:
        stop_event.set()
        reader_thread.join(timeout=2.0)
        driver.disconnect()


if __name__ == "__main__":
    main()
