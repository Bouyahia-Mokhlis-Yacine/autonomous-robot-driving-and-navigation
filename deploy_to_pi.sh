#!/usr/bin/env bash
# Copy this project to the Raspberry Pi and set it up for the real-robot test.
#
#   bash deploy_to_pi.sh                     # default target pi@raspberrypi.local
#   PI=<user>@<pi-host> bash deploy_to_pi.sh
#
# What it does (it NEVER drives the motors):
#   1. rsync the project to ~/autonomous_navigation on the Pi
#   2. create ~/navenv (Python venv with --system-site-packages, so the Pi's
#      apt-installed python3-lgpio / gpiozero are reused if present) and pip
#      install the runtime dependencies
#   3. report: GPIO/serial access, anything else holding the LiDAR port or the
#      motor pins (gesture project's pi_server.py, old ROS2 containers)
#   4. run the test suite and the SLAM timing benchmark ON THE PI
# Needs passwordless SSH (ssh-copy-id) and no sudo.
set -euo pipefail

PI="${PI:-pi@raspberrypi.local}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="autonomous_navigation"

echo "== 1/4 copying project to $PI:~/$REMOTE_DIR"
rsync -az --delete \
  --exclude '__pycache__' --exclude '.pytest_cache' --exclude '*.pyc' \
  "$HERE/" "$PI:$REMOTE_DIR/"

echo "== 2/4 Python environment on the Pi"
ssh "$PI" bash -s <<'REMOTE'
set -euo pipefail
cd ~/autonomous_navigation
python3 --version
if [ ! -d ~/navenv ]; then
  python3 -m venv --system-site-packages ~/navenv
fi
. ~/navenv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy scipy pyyaml pillow pyserial "rplidar==0.9.2" pytest py-spy
python - <<'PY'
import importlib
for mod in ("gpiozero", "lgpio"):
    try:
        importlib.import_module(mod)
        print(f"  {mod}: available")
    except Exception as err:  # noqa: BLE001
        print(f"  {mod}: MISSING ({err.__class__.__name__})")
PY
if ! python -c "import gpiozero, lgpio" 2>/dev/null; then
  echo "  installing gpiozero + lgpio with pip (needs the system liblgpio; if this fails:"
  echo "  sudo apt install python3-gpiozero python3-lgpio   -- then rerun this script)"
  pip install --quiet gpiozero lgpio || true
fi
python -c "import numpy, scipy, yaml, serial, rplidar; print('  core dependencies OK')"
REMOTE

echo "== 3/4 hardware access and conflicts on the Pi"
ssh "$PI" bash -s <<'REMOTE'
echo "  groups: $(id -nG)"
for g in dialout gpio; do id -nG | grep -qw "$g" && echo "  in group $g: yes" || echo "  in group $g: NO"; done
ls -l /dev/ttyUSB* 2>/dev/null || echo "  no /dev/ttyUSB* -- is the RPLiDAR plugged into the Pi (USB + DC power)?"
ls -l /dev/gpiochip* 2>/dev/null | head -3
echo "  processes that would conflict (must not run during the test):"
pgrep -af "pi_server.py|pi_app.py|run_pipeline.py|robot_runtime.py|motor_guard" || echo "    none"
if command -v docker >/dev/null; then
  docker ps --format '    container {{.Names}} ({{.Image}}) {{.Status}}' 2>/dev/null || true
fi
if command -v fuser >/dev/null && ls /dev/ttyUSB0 >/dev/null 2>&1; then
  fuser -v /dev/ttyUSB0 2>&1 | sed 's/^/    /' || echo "    /dev/ttyUSB0 free"
fi
REMOTE

echo "== 4/4 tests and SLAM benchmark on the Pi"
ssh "$PI" bash -s <<'REMOTE'
set -e
cd ~/autonomous_navigation
. ~/navenv/bin/activate
python -m pytest -q -p no:cacheprovider \
  tests/test_motor_driver.py tests/test_network_protocol.py tests/test_robot_runtime.py \
  tests/test_real_world_model_walkthrough.py tests/test_headless_imports.py \
  tests/test_world_model_integrate_equivalence.py tests/test_live_backend_motors.py \
  tests/test_async_planning.py tests/test_astar_equivalence.py tests/test_motor_watchdog.py tests/test_map_update_filter.py tests/test_real_lidar_robustness.py tests/test_bidirectional_motion.py tests/test_lidar_mount.py 2>&1 | tail -3
python benchmark_slam_processing.py --duration 20
(vcgencmd measure_temp && vcgencmd get_throttled) 2>/dev/null || true
REMOTE

echo
echo "Setup done. Next, on the Pi (ssh $PI), robot ON BLOCKS first:"
echo "  cd ~/autonomous_navigation && . ~/navenv/bin/activate"
echo "  python motor_test.py"
echo "Then the full system:"
echo "  Pi:     python robot_runtime.py --backend live --motors gpio"
echo "  laptop: python operator_gui.py --host ${PI#*@}"
