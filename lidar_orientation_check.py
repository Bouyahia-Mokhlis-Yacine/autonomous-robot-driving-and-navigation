"""Physical check of the LiDAR's angle direction and mounting yaw (sensor only,
no motors). Run on the Pi with the runtime STOPPED (it opens the serial port):

    python lidar_orientation_check.py [seconds]

Clear ~1 m around the robot, then hold ONE object (a box, a leg) about 0.5 m
from the robot. Every second this prints where the nearest object is, in the
frame the whole stack uses (x = robot forward, y = robot LEFT, angles
counter-clockwise):

    1. object straight in FRONT of the robot -> must print bearing ~0 deg, AHEAD
    2. object on the robot's LEFT side       -> must print bearing ~+90 deg, LEFT

If (2) prints RIGHT (~-90 deg), scans are MIRRORED: the RPLiDAR reports angles
clockwise and hardware/rplidar_driver.py does not negate them. If (1) is off
by a roughly constant angle, the sensor's 0 deg is not the robot's forward
direction (mounting yaw; not configurable yet). Either must be fixed before
the robot drives: SLAM would turn the opposite way to the wheels.
"""

import math
import sys
import time

sys.path.insert(0, ".")
from common.config import load_hardware_config, load_navigation_config
from hardware.rplidar_driver import RPLidarDriver

MIN_RANGE_M = 0.20   # ignore the robot's own parts right next to the sensor


def main() -> None:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    nav = load_navigation_config()
    driver = RPLidarDriver(hardware_config=load_hardware_config().rplidar, lidar_config=nav.lidar)
    driver.connect()
    t_end, last_print = time.monotonic() + duration, 0.0
    try:
        for scan in driver.iter_scans():
            if time.monotonic() > t_end:
                break
            if time.monotonic() - last_print < 1.0:
                continue
            hits = [(r, scan.angle_at(i)) for i, r in enumerate(scan.ranges)
                    if MIN_RANGE_M <= r < scan.range_max]
            if not hits:
                continue
            r, a = min(hits)
            deg = math.degrees(a)
            side = "AHEAD" if abs(deg) < 30 else "BEHIND" if abs(deg) > 150 else "LEFT" if deg > 0 else "RIGHT"
            print(f"nearest object {r:.2f} m at bearing {deg:+6.1f} deg -> {side:6s} "
                  f"(x {r * math.cos(a):+.2f} m, y {r * math.sin(a):+.2f} m)", flush=True)
            last_print = time.monotonic()
    finally:
        driver.disconnect()


if __name__ == "__main__":
    main()
