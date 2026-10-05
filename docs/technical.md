# Technical notes

This document gives more detail than the README on the localization, mapping,
planning, safety and validation of the navigation stack. All parameters live
in `config/*.yaml`.

## 1. LiDAR scans

`hardware/rplidar_driver.py` wraps the `rplidar` serial library (256000 baud)
and converts each revolution into a `LaserScan` of **361 rays over [−π, π],
1° apart**:

- the nearest measurement in each 1° bin is kept;
- readings outside 0.15–8.0 m are discarded, and a bin without a return is
  reported as `range_max` (8.0 m = "no return");
- the sensor reports angles clockwise, so they are negated
  (`angles_clockwise: true`) to get the convention used everywhere:
  `+x` forward, `+y` left, angles counter-clockwise;
- a reader thread keeps only the latest scan, so slow processing never fills
  the serial buffer.

`lidar_orientation_check.py` verifies the angle direction on the robot.

## 2. Localization

### Scan registration (`slam/scan_registration.py`)

Deterministic **trimmed point-to-point ICP** between the scan's hit points
(robot frame) and confirmed world-model cells within 8 m:

1. transform the scan with the current estimate and find nearest neighbours
   with a SciPy `cKDTree`; ignore pairs farther than 0.3 m;
2. keep the best 80 % of pairs by distance (at least 3 pairs);
3. reject the fit if the kept points cover fewer than 3 of 8 angular sectors;
4. solve the rigid transform in closed form with an SVD (Kabsch), re-fitting
   the absolute pose each iteration;
5. stop when the update is below 1 mm and 0.05°, or after 50 iterations.

Failures return the seed pose with an explicit reason
(`insufficient_correspondences`, `insufficient_spatial_spread`,
`degenerate_geometry`, ...).

### Pipeline (`slam/scan_matching_pipeline.py`)

| Step | Behaviour |
|---|---|
| Bootstrap | The first scan defines the origin. For ≥ 2 s the robot must stay still and every scan is integrated at the origin, until the confirmed map is usable (fails after 5 s). |
| Seed | Last accepted pose moved by the commanded motion (`slam/motion_model.py`), or constant-velocity extrapolation without motors. |
| Plausibility gate | Reject if the motion exceeds `1.5 × v_max × Δt + 5 cm` or `1.5 × ω_max × Δt + 3°`. |
| Map-update filter | After bootstrap, integrate a scan only after ≥ 5 cm or ≥ 2° of motion (prevents map/pose drift of a stationary robot). |
| Failure handling | A rejected scan keeps the last pose; 5 consecutive failures mean "lost". |
| Delay compensation | Each 10 Hz control tick, the pose is propagated from the scan's capture time with the commanded motion. |

### Localization health (`slam/localization_health.py`)

INITIALIZING (bootstrap) → LOCALIZED → DEGRADED (no accepted scan for 0.6 s,
still usable) → LOST (pipeline lost or no accepted scan for 1.2 s; navigation
stops). Two consecutive accepted scans are required to leave LOST.

## 3. World model and inflation

`mapping/persistent_world_model.py`: 0.05 m cells over 20 × 20 m, origin at
(−10, −10) m.

| Cell state | Published as | Rule |
|---|---|---|
| UNKNOWN | −1 | never observed |
| FREE | 0 | crossed by a ray |
| TENTATIVE | −1 | hit once; confirmed if hit again ≥ 0.5 s later |
| CONFIRMED | 100 | cleared only when a ray passes through it |

Occluded or out-of-range cells keep their state, and rays without a return
are not used as free-space evidence (`max_range_policy: ignore`). Integration
is vectorized with NumPy.

`mapping/inflation.py` grows occupied cells by 0.20 m (`config/runtime.yaml`)
into a separate planning map, rebuilt every 0.3 s.

## 4. Planning and path following

- **A\*** (`planning/astar.py`): 8-connected grid, octile heuristic, unknown
  cells cost ×3, no diagonal corner cutting. Invalid requests and unreachable
  goals are reported separately. The search runs in a worker process
  (`planning/async_planner.py`) with a 2 s budget.
- **Path post-processing** (`planning/path_smoothing.py`): duplicate and
  collinear point removal, line-of-sight shortcuts with 0.10 m extra
  clearance, gradient smoothing with a collision fallback.
- **Path following** (`control/path_follower.py`): FORWARD when the target is
  within 30° of the front, REVERSE (≤ 0.25 m/s, ≤ 1.5 m of path) when within
  30° of the rear and the rear is clear, otherwise ROTATE_IN_PLACE until the
  error is below 10°. Heading gain 2.0, acceleration limits 0.5 m/s² and
  2.0 rad/s², goal tolerance 0.15 m.
- **Navigation manager** (`navigation/navigation_manager.py`): IDLE, PLANNING,
  MOVING, REPLANNING, RECOVERING, GOAL_REACHED, BLOCKED, EMERGENCY_STOP,
  LOCALIZATION_LOST. Replanning on a blocked path, a deviation > 0.30 m, and
  every 1 s.

## 5. Safety

| Layer | Behaviour |
|---|---|
| Motion safety (`navigation/motion_safety.py`) | Each tick, the zone the robot is about to enter (front/rear box, or turning circle) is checked on the latest raw scan and on the confirmed map. |
| Reactive stop | Obstacle within 0.30 m along the heading on the planning map → stop and replan. |
| Localization gate | Stop when health is not LOCALIZED/DEGRADED. |
| Inflation-buffer escape | Plan from the nearest buffer-free cell, or reverse a short checked distance. |
| Latched STOP | Operator STOP latches on the robot; latched at start-up with motors. |
| Operator loss | No heartbeat for 1.5 s → goal cancelled. |
| Motor watchdog | `GpioDiffDriveRobot` stops if no command arrives within 0.5 s. |
| Motor guard | `hardware/motor_guard.py`, a separate process, sets the motor pins low if the runtime dies. |

## 6. Runtime protocol

`network/protocol.py`: TCP (port 8765), one JSON object per line.

- Operator → robot: `hello`, `heartbeat`, `set_goal`, `cancel_goal`, `stop`,
  `resume`. Commands carry a per-session sequence number; duplicate,
  out-of-order, stale (> 2 s) or malformed commands are rejected.
- Robot → operator: `welcome`, `state` (10 Hz: pose, health, navigation state,
  scan, statistics), `map` (1 Hz when changed, zlib + base64), `ack`.

## 7. Simulation

`simulation/` contains a 2D world with rectangular obstacles and collision
detection, and two LiDAR models (`config/lidar_simulation.yaml`):

- **ideal:** perfect, instantaneous scans at 10 Hz;
- **realistic:** fitted to real recordings — ~11 Hz with jitter, 14.5 % missed
  returns, range noise with outliers, partial scans, motion distortion within
  a revolution, and simulated localization processing delay.

The real localization, planning and control code runs unchanged in the
closed-loop simulator (`simulation/closed_loop.py`).

## 8. Validation summary

| Test | Result |
|---|---|
| Simulation, realistic sensor, 5 seeds per scenario | goal reached in every navigation scenario, no collision; max. position error 0.07–0.09 m |
| Sensor blackout / slow processing (simulation) | safe stop while lost, goal reached after recovery |
| Stationary, live on the Pi (60 s) | 0.6 cm, 0.9° |
| Stationary recording replayed for 5 min | max. 5.0 cm, 1.8°, no growth |
| 2 s sensor blackout (real data) | LOST after 0.4 s, LOCALIZED 0.1 s after scans return |
| 91 s live run on the Pi | 12.25 scans/s, 0 dropped; no missed control tick (max. 76 ms of 100 ms); ~8 ms per scan |
| Automated tests | 817 test cases (`pytest`) |

**Limitation:** heading accuracy degrades during fast in-place rotations
(up to ~9° in simulation at 1.5 rad/s; 2–10° after a full turn on real data
without motion information). The error is not flagged by the health monitor.

## 9. ROS 2 (first-time sensor discovery)

ROS 2 was used only to discover and verify the RPLiDAR before the project's
own driver was written; the navigation stack does not depend on it.
`ros2_integration/sensor_discovery/` contains the Docker setup (ROS 2 Jazzy
with `rplidar_ros`, host networking):

```bash
docker build -t gesture-robot-ros2:jazzy -f Dockerfile.pi .   # once, on the Pi
./run_lidar_pi.sh                                              # rplidar_composition on /dev/ttyUSB0, 256000 baud
./run_container_laptop.sh ros2 topic echo /scan --once         # inspect the scan
```

The driver reported firmware 1.32, hardware revision 6 and health "good" in
Sensitivity scan mode. `ros2_walkthrough.py` (with `ros2_integration/map_bridge.py`
and `Dockerfile.nav`) displays the `/scan` topic as a map in the project's GUI.
