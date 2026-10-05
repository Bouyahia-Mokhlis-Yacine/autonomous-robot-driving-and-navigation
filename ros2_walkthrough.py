"""ROS2 real-sensor, no-motors walkthrough (rplidar_ros as scan source).

Same purpose and same mapping/localization stack as
``real_sensor_walkthrough.py`` -- live map, click a destination, get a
planned path, carry the sensor there by hand, no motors driven -- just
using ROS2's ``rplidar_ros`` driver (via ``ros2_integration/map_bridge.py``)
as the scan *source* instead of this project's own
``hardware/rplidar_driver.py``.

An earlier version of this script also used ROS2's ``slam_toolbox`` for
mapping and localization. Removed after live testing showed it wasn't
doing the job: ``slam_toolbox`` only inserts new data into its map when
it detects the *sensor itself* has moved -- with the sensor sitting
still while objects around it changed, its map stayed frozen for
20+ minutes with byte-identical content, confirmed via ``/map`` and tf
inspection.
This project's own mapping (``mapping.occupancy_grid.OccupancyGridMap.
update_from_scan``) has no such motion-gating -- it ray-traces whatever
the current scan says on every call, so it reflects a moved object on
the very next scan regardless of whether the sensor moved. Localization
uses the same Phase 13 ``LidarScanMatchingBackend`` (BreezySLAM, no
wheel odometry) ``real_sensor_walkthrough.py`` already uses.

``rplidar_ros`` remains in use as the scan source (it's been reliable);
only the mapping/localization layer changed.

NO MOTORS ARE DRIVEN.
"""

from __future__ import annotations

import time

import rclpy

from common.config import load_navigation_config, load_robot_config
from common.logger import configure_logging, get_logger
from common.messages import Pose2D
from control.path_follower import PathFollower, PathFollowerConfig
from mapping.inflation import compute_inflation_radius, inflate_occupancy_grid
from mapping.occupancy_grid import OccupancyGridMap
from navigation.navigation_manager import NavigationManager
from navigation.replanner import DynamicObstacleLayer, Replanner, ReplannerConfig, combine_planning_map
from planning.global_planner import GlobalPlanner
from robot.sensor_only_robot import SensorOnlyRobot
from ros2_integration.map_bridge import ROS2ScanBridge, spin_bridge_in_background
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

    rclpy.init()
    bridge = ROS2ScanBridge()
    spin_bridge_in_background(bridge)
    logger.info("ROS2 bridge started; waiting for /scan from rplidar_ros...")
    logger.warning("NO MOTORS ARE DRIVEN -- carry the sensor by hand to reach a clicked destination.")

    last_processed_timestamp: float | None = None
    last_tick_time = time.time()
    last_planning_rebuild = 0.0

    def step(_nominal_dt: float) -> None:
        nonlocal last_tick_time, last_processed_timestamp, last_planning_rebuild
        now = time.time()
        dt = max(1e-3, now - last_tick_time)
        last_tick_time = now

        scan = bridge.get_scan()
        if scan is None or scan.timestamp == last_processed_timestamp:
            return  # nothing new since the last tick; just let the GUI redraw as-is
        last_processed_timestamp = scan.timestamp

        pose = slam.update(scan)  # re-traces the current scan into the map AND estimates pose
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
        get_scan=bridge.get_scan,
    )

    gui = NavigationGUI(nav_manager, data_source, scale=30.0, dt=0.1, step_callback=step)
    gui.root.title("ROS2 real-sensor walkthrough (no motors)")

    def shutdown() -> None:
        logger.info("Shutting down ROS2 bridge...")
        try:
            bridge.destroy_node()
        except Exception:
            logger.exception("Error destroying bridge node")
        if rclpy.ok():
            rclpy.shutdown()

    def on_close() -> None:
        shutdown()
        gui.close()

    gui.root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        gui.run()
    finally:
        shutdown()


if __name__ == "__main__":
    main()
