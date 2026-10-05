#!/bin/bash
# Launches the ROS2 dev container for the LiDAR/parking work (Raspberry Pi
# side). Run this ON THE PI, not the laptop.
#
# Image: gesture-robot-ros2:jazzy -- a custom image built FROM
# ros:jazzy-ros-base with rplidar_ros already installed (see Dockerfile.pi
# in this folder). Build it once with:
#   docker build -t gesture-robot-ros2:jazzy -f Dockerfile.pi .
# Using the bare ros:jazzy-ros-base image and apt-installing rplidar_ros
# on every single `docker run` (the original approach) meant a real,
# if small, delay on every launch -- baking it in makes every run
# instant. The Pi only ever needs rplidar_ros -- rviz2/slam_toolbox stay
# on the laptop by design (see Dockerfile.robot there), so this image
# stays small (899MB vs. the laptop's 2.05GB).
#
# No `docker compose` plugin is installed on the Pi (and installing one
# needs sudo, which needs a password we don't have non-interactively) --
# plain `docker run` is simpler here anyway for a single container.
#
# --network host: ROS2 nodes discover each other over DDS, which relies on
# UDP multicast -- bridge/NAT networking (Docker's default) breaks that
# discovery between containers on different machines (Pi vs. laptop).
# Host networking sidesteps it entirely, at the cost of not isolating the
# container's network namespace -- fine for a single-purpose robot.
#
# --device /dev/gpiochip0 (and gpiochip4, a symlink alias to it on this Pi):
# needed once the motor-control node (wrapping robot/motor_controller.py)
# runs inside this container instead of directly on the host. Not used yet
# -- the Flask-based motor server (interfaces/3_laptop_with_robot_camera/
# pi_server.py) still owns the GPIO pins for now; don't run both against
# the same pins at once (see the earlier "GPIO busy" incident notes).
#
# --device /dev/ttyUSB0: the RPLiDAR's USB-serial device -- only passed
# through if it's actually plugged into the Pi right now (checked below),
# since the LiDAR has been moved back and forth between the Pi and the
# laptop during bring-up/testing. Check `ls /dev/ttyUSB* /dev/ttyACM*` on
# the Pi if it stops landing on ttyUSB0 specifically.
#
# -v: mounts a persistent workspace directory on the Pi's filesystem into
# the container, so code/build artifacts survive container restarts and
# are editable from outside the container too (e.g. over the same ssh
# session used for everything else in this project).
#
# Usage:
#   ./run_container.sh            # interactive shell inside the container
#   ./run_container.sh ros2 topic list   # run a one-off command instead

set -euo pipefail

WORKSPACE_DIR="${HOME}/ros2_ws"
mkdir -p "${WORKSPACE_DIR}"

# -i always (so piped/scripted input still works); -t only when stdin is
# actually a terminal -- `docker run -t` without a real TTY attached (e.g.
# this script invoked non-interactively, such as over `ssh pi cmd` or from
# another script) fails outright with "the input device is not a TTY"
# instead of just behaving like a plain (non-interactive) run.
DOCKER_TTY_FLAGS="-i"
if [ -t 0 ]; then
  DOCKER_TTY_FLAGS="-it"
fi

# Default to an interactive shell when called with no arguments.
if [ $# -eq 0 ]; then
  set -- bash
fi

LIDAR_DEVICE_FLAGS=()
if [ -e /dev/ttyUSB0 ]; then
  LIDAR_DEVICE_FLAGS=(--device /dev/ttyUSB0)
fi

# NOTE: this used to be `bash -c "source ... && ${*:-bash}"`, which is
# broken for any multi-word command -- `$*` joins args with spaces and
# loses their quoting, so e.g. `apt-get install -y foo` (3+ separate
# words) got flattened into one string, re-split wrong by the inner
# `bash -c`, and only "apt-get" (no args) actually ran, which is why an
# install command silently printed apt-get's bare usage text instead of
# installing anything (caught on the laptop side, same bug here).
# Passing "$@" as separate trailing arguments to a FIXED -c script (which
# re-expands them via its own "$@") preserves each argument/word
# correctly, including ones with spaces, quotes, or pipes.
docker run --rm ${DOCKER_TTY_FLAGS} \
  --name gesture-robot-ros2 \
  --network host \
  --device /dev/gpiochip0 \
  --device /dev/gpiochip4 \
  "${LIDAR_DEVICE_FLAGS[@]}" \
  -v "${WORKSPACE_DIR}:/root/ros2_ws" \
  -w /root/ros2_ws \
  gesture-robot-ros2:jazzy \
  bash -c 'source /opt/ros/jazzy/setup.bash && exec "$@"' bash "$@"
