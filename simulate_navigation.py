"""Phase 12: software-only, closed-loop navigation simulation.

No physical LiDAR needed. Ground truth exists only inside the simulator
(``simulation.environment.Environment``); the navigation stack (scan
matching, ``PersistentWorldModel``, risk-aware ``GlobalPlanner``,
``NavigationManager``) receives nothing but synthetic LiDAR scans and
drives a simulated robot exactly as it would drive a real one -- see
``simulation/closed_loop.py`` for the full architecture and the
ground-truth isolation guarantee.

Runtime note: ``PersistentWorldModel.integrate()`` (existing, unmodified
mapping code) costs roughly 0.15-0.25s per simulated scan at this
project's real sensor configuration on this machine, so simulated time
costs very roughly 2.5x that much in wall-clock time -- the default
45s of simulated time takes on the order of two minutes to run.

Sensor realism (``config/lidar_simulation.yaml``): ``--sensor ideal``
is the original perfect/instantaneous/synchronous simulator;
``--sensor realistic`` uses the RPLiDAR A2M12 model with asynchronous,
latency-bearing SLAM processing, and additionally prints the sensor/
processing diagnostics. Without ``--sensor``, the file's ``mode`` decides
(ideal by default).

Run:
    conda activate AI-env
    python simulate_navigation.py                        # ideal (original)
    python simulate_navigation.py --sensor realistic --seed 42
"""

from __future__ import annotations

import argparse

from common.config import load_navigation_config, load_robot_config
from common.logger import configure_logging, get_logger
from simulation.closed_loop import ClosedLoopSimulation
from simulation.scenario import default_scenario
from simulation.sensor_run_log import format_sensor_report
from simulation.sensor_simulation_config import load_sensor_simulation_config

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Software-only closed-loop navigation simulation.")
    parser.add_argument("--duration", type=float, default=45.0, help="max simulation time, seconds")
    parser.add_argument("--dt", type=float, default=0.1, help="fixed simulation timestep, seconds")
    parser.add_argument("--sensor", choices=("ideal", "realistic"), default=None,
                        help="sensor model (default: 'mode' in config/lidar_simulation.yaml)")
    parser.add_argument("--seed", type=int, default=None, help="override the config's random seed")
    args = parser.parse_args()
    sensor_cfg = load_sensor_simulation_config(
        mode=args.sensor, seed=args.seed if args.seed is not None else "from_file",
    )

    robot_cfg = load_robot_config()
    nav_cfg = load_navigation_config()
    configure_logging(nav_cfg.logging_level)

    scenario = default_scenario(robot_cfg.robot.width, robot_cfg.robot.length)
    sim = ClosedLoopSimulation(
        environment=scenario.environment,
        ground_truth_start=scenario.ground_truth_start,
        goal=scenario.slam_frame_goal,
        nav_cfg=nav_cfg,
        robot_cfg=robot_cfg,
        dt=args.dt,
        max_simulation_time=args.duration,
        sensor_simulation=sensor_cfg,
    )
    metrics = sim.run()

    def fmt(value: float | None, unit: str) -> str:
        return f"{value:.3f} {unit}" if value is not None else "n/a"

    print(f"Simulation complete (sensor: {sensor_cfg.mode}"
          + (f", seed {sensor_cfg.seed})" if sensor_cfg.is_realistic else ")"))
    print()
    print(f"Goal reached: {metrics.goal_reached}")
    print(f"Final navigation state: {metrics.final_nav_state}")
    print(f"Collision: {metrics.collision}")
    print(f"Simulation time: {metrics.simulation_time:.2f} s ({metrics.ticks} ticks)")
    bootstrap_note = f" (at {metrics.bootstrap_duration:.2f} s)" if metrics.bootstrap_duration is not None else ""
    print(f"Bootstrap completed: {metrics.bootstrap_completed}{bootstrap_note}")
    print(f"Final position error: {fmt(metrics.final_position_error, 'm')}")
    print(f"Final heading error: {fmt(metrics.final_heading_error, 'rad')}")
    print(f"Mean localization error: {fmt(metrics.mean_position_error, 'm')}")
    print(f"Maximum localization error: {metrics.max_position_error:.3f} m")
    print(f"Ground-truth distance traveled: {metrics.ground_truth_distance:.3f} m")
    print(f"Planned path length: {fmt(metrics.planned_path_length, 'm')}")

    report = sim.sensor_report()
    if report is not None:
        print()
        print(format_sensor_report(report))


if __name__ == "__main__":
    main()
