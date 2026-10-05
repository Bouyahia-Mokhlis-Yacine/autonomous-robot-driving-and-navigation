#!/bin/bash
# Launches the ROS2 dev container for the LiDAR/parking work (laptop side:
# slam_toolbox + Nav2 -- the heavy-compute half of the split, mirroring
# how gesture recognition is already offloaded to the laptop instead of
# running on the Pi). Run this ON THE LAPTOP. For the Pi side, see
# run_container.sh in this same folder.
#
# Image: gesture-robot-ros2:jazzy -- a custom image built FROM
# ros:jazzy-ros-base (see Dockerfile.robot in this folder) with
# rplidar_ros, rviz2 AND slam_toolbox already installed. Build it once
# with:
#   docker build -t gesture-robot-ros2:jazzy -f Dockerfile.robot .
# Using the bare ros:jazzy-ros-base image and apt-installing these on
# every single `docker run` (the original approach) meant re-downloading
# 1-2GB+ of packages EVERY time -- painfully slow for repeated bring-up
# testing. Nav2 is NOT baked in yet (not needed until the actual
# navigation phase) -- add it to the Dockerfile and rebuild when that
# phase starts.
#
# Same --network host reasoning as the Pi side: ROS2/DDS discovery relies
# on UDP multicast, which doesn't cross Docker's default bridge network --
# host networking lets this container and the Pi's container find each
# other over Wi-Fi as if they were both just processes on the LAN.
#
# --device /dev/ttyUSB0: the RPLiDAR A2M12's USB-serial adapter (a
# Silicon Labs CP210x bridge) -- only passed through if it's actually
# plugged into THIS machine right now (checked below), since the LiDAR
# has been moved back and forth between the laptop (bring-up/testing) and
# the Pi (actual robot use) during development. When it's on the Pi
# instead, this container just subscribes to /scan etc. over the network
# and doesn't need the device at all. Check `ls /dev/ttyUSB*` if it stops
# landing on ttyUSB0 specifically.
#
# Usage:
#   ./run_container_laptop.sh                  # interactive shell inside the container
#   ./run_container_laptop.sh ros2 topic list   # one-off command instead

set -euo pipefail

WORKSPACE_DIR="${HOME}/ros2_ws"
mkdir -p "${WORKSPACE_DIR}"

# -i always; -t only when stdin is an actual terminal (see run_container.sh
# for why -- fails outright with "the input device is not a TTY" otherwise
# when this script is invoked non-interactively, e.g. from another script).
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
# installing anything. Passing "$@" as separate trailing arguments to a
# FIXED -c script (which re-expands them via its own "$@") preserves each
# argument/word correctly, including ones with spaces, quotes, or pipes.
docker run --rm ${DOCKER_TTY_FLAGS} \
  --name gesture-robot-ros2-laptop \
  --network host \
  "${LIDAR_DEVICE_FLAGS[@]}" \
  -v "${WORKSPACE_DIR}:/root/ros2_ws" \
  -w /root/ros2_ws \
  gesture-robot-ros2:jazzy \
  bash -c 'source /opt/ros/jazzy/setup.bash && exec "$@"' bash "$@"
