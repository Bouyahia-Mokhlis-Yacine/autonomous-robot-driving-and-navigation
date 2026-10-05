# Autonomous LiDAR Navigation for a Raspberry Pi Mobile Robot

Autonomous navigation for a four-wheel skid-steer robot driven by a Raspberry
Pi 4, using a 2D RPLiDAR as its only navigation sensor (no wheel encoders, no
IMU). An operator clicks a destination on a map displayed on a laptop; the
robot localizes itself from the LiDAR, plans a collision-free path, follows it,
replans when the path is blocked, and stops safely at the goal or whenever
navigation becomes unsafe.

Developed during an internship at the Embedded Systems Laboratory, Research
Center in Industrial Technologies (CRTI).

- Demonstration video: [INSERT AUTONOMOUS NAVIGATION YOUTUBE LINK]
- Internship report: [INSERT INTERNSHIP REPORT LINK]

## Pipeline

```
RPLiDAR
   ↓
Scan processing (361 rays, 1°)
   ↓
ICP localization  ← persistent world model (reference)
   ↓
Persistent world model (occupancy grid)
   ↓
Obstacle inflation
   ↓
A* planning
   ↓
Path smoothing
   ↓
Path following + motion safety
   ↓
Motor control (GPIO PWM → L298N)
```

## Features

- **Localization without odometry:** trimmed point-to-point ICP against the map,
  seeded with the commanded motion, with a physical-plausibility check and a
  localization-health monitor that stops the robot when the pose is not trusted.
- **Persistent world model:** 5 cm occupancy grid where new obstacles are
  confirmed after 0.5 s and removed only when the LiDAR sees through them.
- **Planning:** 8-connected A* on an inflated map (unknown space costs more),
  in a separate process with a 2 s budget, followed by path simplification and
  smoothing.
- **Path following:** forward, reverse and rotate-in-place modes with hysteresis
  and acceleration limits.
- **Safety:** raw-scan obstacle checks before every motion, latched STOP,
  operator heartbeat, motor watchdog, and a separate guard process that cuts the
  motor pins if the program dies.
- **Operator GUI:** live map, planning map, robot, LiDAR points, path, goal and
  navigation/localization state; click to set a goal, STOP/RESUME buttons.
- **Simulation:** 2D simulator with an ideal and a realistic RPLiDAR model fitted
  to real recordings; the real navigation code runs unchanged inside it.

## Hardware

| Component | Role |
|---|---|
| Raspberry Pi 4 Model B | onboard computer (runs the whole navigation stack) |
| RPLiDAR (A2 family) | 2D laser scanner, USB serial at 256000 baud, ~12 revolutions/s, used up to 8 m |
| 2 × L298N H-bridges | one per side, ENA/ENB tied to +5 V, motor battery on the 12 V input, common ground with the Pi |
| 4 DC gear motors | skid-steer drive (no encoders) |
| Laptop | operator GUI |

Motor wiring (BCM numbering, `config/hardware.yaml`):

| Motor | Forward pin | Backward pin |
|---|---|---|
| Front-left | GPIO 12 | GPIO 17 |
| Back-left | GPIO 25 | GPIO 26 |
| Front-right | GPIO 23 | GPIO 19 |
| Back-right | GPIO 21 | GPIO 20 |

ROS 2 was used only for the first-time discovery of the RPLiDAR
(`ros2_integration/sensor_discovery/`, see `docs/technical.md`); it is not
needed to run the project.

## Installation

Python 3.11.

```bash
pip install -r requirements.txt
```

On the Raspberry Pi, also install the GPIO libraries (needed only to drive the
motors):

```bash
pip install gpiozero lgpio
```

`deploy_to_pi.sh` copies the project to the Pi over SSH, creates a virtual
environment, installs the dependencies, checks serial/GPIO permissions, and runs
the hardware-related tests and a timing benchmark on the Pi (it never moves the
motors):

```bash
PI=<user>@<pi-host> bash deploy_to_pi.sh
```

## Running

All commands are run from the repository root.

**Simulation (laptop only)**

```bash
python simulate_navigation.py --sensor realistic --seed 42   # closed-loop simulation
python robot_runtime.py                                      # terminal 1: simulated robot + real navigation stack
python operator_gui.py                                       # terminal 2: operator GUI (left-click = goal)
```

**Real robot**

```bash
# on the Pi: sensor only (map and localization, no motors)
python robot_runtime.py --backend live
# on the Pi: robot on blocks, check each motor and the watchdog
python motor_test.py
# on the Pi: full navigation with motors (STOP is latched until RESUME)
python robot_runtime.py --backend live --motors gpio
# on the laptop
python operator_gui.py --host <pi-address>
```

`python lidar_orientation_check.py` (on the Pi, runtime stopped) checks the LiDAR
angle direction. Keep the robot still for the first ~2 s after start-up while
the initial map is built.

**Tests**

```bash
pytest
```

## Architecture

| Package | Role |
|---|---|
| `hardware/` | RPLiDAR driver, GPIO motor driver with watchdog, motor guard process |
| `slam/` | scan preprocessing, ICP registration, scan-matching pipeline, localization health |
| `mapping/` | occupancy grid, persistent world model, ray casting, inflation |
| `planning/` | A*, asynchronous planner, path post-processing |
| `control/` | path follower, waypoint selection, differential-drive conversion |
| `navigation/` | navigation state machine, replanning, motion safety |
| `runtime/`, `network/` | 10 Hz robot runtime, TCP/JSON protocol |
| `visualization/` | operator GUI and map rendering |
| `simulation/` | 2D world, LiDAR models, closed-loop simulation |
| `config/` | robot, navigation, hardware, runtime and simulation parameters |

Modules communicate through simple message dataclasses (`common/messages.py`),
so the simulated and real sensor, and the sensor-only and motorized robot, are
interchangeable. See `docs/technical.md` for the algorithms and parameters.

## Testing and results

The project has an automated test suite of 817 test cases covering
localization, mapping, planning, control, the runtime protocol, the motor driver
and watchdogs, closed-loop simulation scenarios and replays of real LiDAR
recordings (`tests/fixtures/`).

| Test | Result |
|---|---|
| Simulation (realistic sensor, 5 seeds per scenario) | goal reached in every navigation scenario, no collision |
| Stationary localization, live on the Pi (60 s) | 0.6 cm, 0.9° |
| Stationary recording replayed for 5 min | max. 5.0 cm, 1.8° |
| 91 s live run on the Pi | 12.25 scans/s with none dropped; no missed 10 Hz control tick; ~8 ms per scan |

## Limitation

Heading accuracy degrades during fast in-place rotations: the heading error can
reach several degrees after a fast full turn, and it is not detected by the
localization-health monitor.

## Repository structure

```
autonomous_navigation/
├── config/              YAML parameters
├── common/              messages, geometry, configuration loading
├── hardware/            LiDAR and motor drivers
├── slam/ mapping/ planning/ control/ navigation/
├── runtime/ network/    robot runtime and operator protocol
├── visualization/       operator GUI
├── simulation/          simulator and LiDAR models
├── ros2_integration/    ROS 2 tools used for first-time LiDAR discovery
├── tests/               automated tests and recorded LiDAR data
├── docs/technical.md    technical notes
├── robot_runtime.py     runtime (Pi or simulation)
├── operator_gui.py      operator GUI (laptop)
├── motor_test.py        motor bench test
└── deploy_to_pi.sh      deployment to the Raspberry Pi
```

## Status

This project is complete. (Still can  be improved tho)
