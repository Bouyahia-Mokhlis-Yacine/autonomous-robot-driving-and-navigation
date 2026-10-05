"""ROS2 bridge: turns ROS2's live ``/scan`` (published by ``rplidar_ros``)
into this project's own ``LaserScan`` type.

Only usable inside the ROS2 container (``ros2_integration/Dockerfile.nav``)
-- rclpy is not a dependency of the rest of this project, and nothing
outside ``ros2_integration/`` imports this module.

This bridge deliberately does NOT use ``slam_toolbox``'s ``/map`` or its
``map``->``laser`` tf (an earlier version of this file did). Diagnosed
live against the real sensor: ``slam_toolbox`` only inserts new data
into its map when it detects the *sensor itself* has moved -- it is
built to map a space as a robot travels through it, not to react to a
stationary sensor's surroundings changing. For this project's walkthrough
tools (testing before any motor exists, typically with the sensor mostly
stationary while things around it change), that is the wrong tool.

Mapping and localization here use this project's own, already-tested
stack instead (``slam.slam.SLAM`` + ``slam.backend.LidarScanMatchingBackend``
+ ``mapping.occupancy_grid.OccupancyGridMap``, wired up in
``ros2_walkthrough.py`` exactly as ``real_sensor_walkthrough.py`` already
does) -- ``OccupancyGridMap.update_from_scan()`` has no motion-gating at
all, it just ray-traces whatever the current scan says on every call, so
it naturally reflects a moved object on the very next scan regardless of
whether the sensor moved. ``rplidar_ros`` remains the scan *source*
(it's been reliable), only the mapping/localization layer changed.

The node is spun on a dedicated background thread (mirroring the
scan-reader thread in real_sensor_walkthrough.py) so it never blocks the
GUI's Tk main loop; ``get_scan`` is a plain attribute read of state the
callback thread writes, which is safe under the GIL for this
single-writer/single-reader pattern (no lock needed).
"""

from __future__ import annotations

import math
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan as RosLaserScan

from common.messages import LaserScan


class ROS2ScanBridge(Node):
    def __init__(self) -> None:
        super().__init__("nav_bridge")
        # Default (volatile/best-effort-compatible) QoS matches rplidar_ros's
        # typical sensor-data publisher.
        self.create_subscription(RosLaserScan, "/scan", self._on_scan, 10)
        self._scan: LaserScan | None = None

    def _on_scan(self, msg: RosLaserScan) -> None:
        # ROS reports out-of-range/no-detection as inf/nan; this project's
        # own LaserScan convention (matching hardware/rplidar_driver.py and
        # simulation.simulated_lidar) uses range_max as that sentinel.
        ranges = [
            r if (math.isfinite(r) and msg.range_min <= r <= msg.range_max) else msg.range_max
            for r in msg.ranges
        ]
        self._scan = LaserScan(
            timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment,
            ranges=ranges,
            range_min=msg.range_min,
            range_max=msg.range_max,
        )

    def get_scan(self) -> LaserScan | None:
        """Latest scan, or None before the first ``/scan`` message arrives."""
        return self._scan


def spin_bridge_in_background(bridge: ROS2ScanBridge) -> threading.Thread:
    """Spin ``bridge`` on a dedicated daemon thread; the caller's main
    thread (running the Tk GUI) is left free."""

    def _spin() -> None:
        rclpy.spin(bridge)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    return thread
