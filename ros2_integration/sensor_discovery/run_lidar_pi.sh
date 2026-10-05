#!/bin/bash
# Launches the RPLiDAR driver container on the Pi. Run this ON THE PI.
#
# Unlike run_container.sh (a generic dev shell), this execs the compiled
# rplidar_composition node binary DIRECTLY instead of going through
# `ros2 run rplidar_ros rplidar_composition`. That distinction matters for
# `docker stop`:
#
# `ros2 run` is a Python wrapper that launches the actual node as a CHILD
# process via subprocess.Popen (see ros2run/api/__init__.py) -- it does not
# register its own SIGTERM handler and does not relay signals to the
# child. When `docker stop` sends SIGTERM to PID 1 (the wrapper) inside
# the container, it's silently ignored -- an unhandled signal sent to
# PID 1 of a PID namespace is dropped by the kernel unless the process
# installs its own handler. Docker's grace period then expires and it
# sends SIGKILL, which tears down the whole container PID namespace --
# including the never-signaled rplidar_composition child -- before it
# gets any chance to run its normal shutdown path. That's how the LiDAR
# motor was left spinning after a clean-looking `docker stop` on
# 2026-09-23 (see the rplidar-motor-no-auto-stop project memory).
#
# rclcpp nodes (rplidar_composition included) install their own SIGINT and
# SIGTERM handlers for a clean rclcpp::shutdown() -- but only if the
# signal reaches the node process itself. Exec'ing the binary directly
# makes it PID 1, so it receives the signal straight from `docker stop`
# with no wrapper in between. Combined with Dockerfile.pi's
# `STOPSIGNAL SIGINT` (matching the one signal every ROS2 tool in this
# stack is built around), `docker stop lidar-pi` now reaches the node the
# same way an interactive Ctrl+C always did, and it can flush its own
# motor-stop logic before exiting.
#
# --device /dev/ttyUSB0: the RPLiDAR's USB-serial device. Check
# `ls /dev/ttyUSB* /dev/ttyACM*` on the Pi if it stops landing on
# ttyUSB0 specifically.

set -euo pipefail

docker run --rm -d \
  --name lidar-pi \
  --network host \
  --device /dev/ttyUSB0 \
  gesture-robot-ros2:jazzy \
  bash -c 'source /opt/ros/jazzy/setup.bash && exec /opt/ros/jazzy/lib/rplidar_ros/rplidar_composition --ros-args \
    -p serial_port:=/dev/ttyUSB0 \
    -p serial_baudrate:=256000 \
    -p frame_id:=laser \
    -p angle_compensate:=true'

echo "lidar-pi started. Stop it with: docker stop lidar-pi"
