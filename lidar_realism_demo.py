"""Visual demo: the ideal simulated LiDAR vs. the realistic RPLiDAR model.

Runs the SAME scenario twice -- once with the original ideal sensor, once
with the realistic A2M12 model + asynchronous SLAM latency
(config/lidar_simulation.yaml) -- records every tick, then opens a
side-by-side replay (``visualization.sensor_comparison_viewer``).

Default: a fixed drive script (forward, turn left, forward, turn right,
through the doorway), so both runs have the SAME ground-truth trajectory
and every difference on screen comes from the sensor/processing model.
``--closed-loop`` instead lets the navigation stack drive to the goal in
both runs (trajectories then diverge, which is itself informative).

Both simulations run to completion first: expect ~1 minute before the
window opens (the ideal run integrates every scan, ~0.2 s wall-clock each).

Run:
    conda activate AI-env
    python lidar_realism_demo.py
    python lidar_realism_demo.py --turn-rate 0.4           # gentler turns
    python lidar_realism_demo.py --closed-loop --duration 40
    python lidar_realism_demo.py --no-gui                  # just print the comparison
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from common.config import load_navigation_config, load_robot_config
from common.logger import configure_logging, get_logger
from common.messages import Pose2D, VelocityCommand
from simulation.closed_loop import ClosedLoopSimulation
from simulation.scenario import default_scenario
from simulation.sensor_run_log import format_sensor_report
from simulation.sensor_simulation_config import load_sensor_simulation_config
from visualization.overlays import scan_hit_points_world
from visualization.sensor_comparison_viewer import SensorComparisonViewer, SensorDemoFrame

logger = get_logger(__name__)

EMPTY = np.empty((0, 2))


def drive_script(speed: float, turn_rate: float) -> list[tuple[float, VelocityCommand]]:
    """Forward ~0.75 m, left 90 deg, forward ~1.2 m, right 90 deg, forward
    ~1.5 m through the doorway (SLAM frame: wall at x=1.5, gap y in [0.5, 2.7])."""
    quarter = (math.pi / 2.0) / turn_rate
    return [
        (0.75 / speed, VelocityCommand(linear=speed, angular=0.0)),
        (quarter, VelocityCommand(linear=0.0, angular=turn_rate)),
        (1.2 / speed, VelocityCommand(linear=speed, angular=0.0)),
        (quarter, VelocityCommand(linear=0.0, angular=-turn_rate)),
        (1.5 / speed, VelocityCommand(linear=speed, angular=0.0)),
    ]


def run_and_record(sim: ClosedLoopSimulation, label: str) -> list[SensorDemoFrame]:
    processed: dict = {}
    original = sim.pipeline.process_scan

    def capture(scan):
        result = original(scan)
        processed["scan"], processed["result"] = scan, result
        processed["count"] = processed.get("count", 0) + 1
        return result

    sim.pipeline.process_scan = capture  # observe only; the pipeline itself is untouched

    frames: list[SensorDemoFrame] = []
    last_grid_version = None
    grid_copy = None
    offset = sim.ground_truth_start
    while sim._sim_time < sim.max_simulation_time - 1e-9:
        sim.step()
        m = sim.metrics
        gt = sim._ground_truth_to_slam(sim.environment.robot_pose)
        est = sim.nav_robot.get_pose()

        # Copy the (mutable, shared) grid only when a scan was integrated.
        version = processed.get("count", 0)
        if version != last_grid_version or grid_copy is None:
            g = sim.world_model.get_grid()
            grid_copy = type(g)(resolution=g.resolution, width=g.width, height=g.height,
                                origin_x=g.origin_x, origin_y=g.origin_y, cells=g.cells.copy())
            last_grid_version = version

        if sim.realistic and sim.last_scan_record is not None:
            hits = sim.last_scan_record.delivered_hit_points_world()
            raw = hits - np.array([offset.x, offset.y]) if hits.size else EMPTY
        elif not sim.realistic and "scan" in processed:
            raw = np.array(scan_hit_points_world(gt, processed["scan"])) if processed else EMPTY
        else:
            raw = EMPTY
        if "scan" in processed:
            pts = scan_hit_points_world(processed["result"].pose, processed["scan"])
            proc_pts = np.array(pts) if pts else EMPTY
        else:
            proc_pts = EMPTY
        health = sim.localization_health.health.value
        localized = sim.localization_health.is_localization_valid() or not m.bootstrap_completed

        err = math.hypot(gt.x - est.x, gt.y - est.y)
        heading_err = math.degrees(abs(math.remainder(gt.theta - est.theta, 2 * math.pi)))
        lines = [
            f"t={m.simulation_time:6.2f}s  nav={m.final_nav_state}",
            f"pose error {err:5.3f} m   heading error {heading_err:5.1f} deg",
            f"localization: {health:<12s} pose age {sim.localization_health.pose_age():4.2f}s  "
            f"losses {sim.localization_health.loss_events}",
        ]
        if sim.realistic:
            p = sim.processor
            pm = p.metrics
            rec = sim.last_scan_record
            state = (f"BUSY on scan @{p.in_flight_scan_timestamp:.2f}s, done @{p.in_flight_finish_time:.2f}s"
                     if p.busy else "idle")
            lines += [
                f"SLAM: {state}",
                f"queue={p.queue_size}  received={pm.scans_received}  processed={pm.scans_processed}  "
                f"dropped={pm.scans_dropped}",
            ]
            if rec is not None:
                valid = sum(1 for r in rec.scan.ranges if r < rec.scan.range_max) if rec.scan else 0
                lines.append(f"last revolution: {valid}/361 bins valid" + ("  PARTIAL" if rec.is_partial else "")
                             + ("  (swallowed)" if rec.scan is None else ""))
            if sim.last_completed is not None:
                c = sim.last_completed
                lines.append(f"newest scan used was acquired {m.simulation_time - c.scan.timestamp + c.scan.scan_time:4.2f}s ago"
                             + ("  (pose propagated to now)" if sim.delay_compensation else ""))
        else:
            lines += ["SLAM: synchronous, every scan, zero latency", "", "", ""]
        frames.append(SensorDemoFrame(
            time=m.simulation_time, grid=grid_copy, ground_truth=gt, estimated=est, raw_hits=raw,
            processed_scan_points=proc_pts, status_lines=tuple(lines), localized=localized,
        ))
        if len(frames) % 50 == 0:
            logger.info("%s: recorded %.1f s", label, m.simulation_time)
        if m.goal_reached or m.collision:
            break
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Ideal vs. realistic LiDAR, side by side.")
    parser.add_argument("--closed-loop", action="store_true", help="let navigation drive to the goal instead of the script")
    parser.add_argument("--duration", type=float, default=None, help="simulated seconds (default: script length + margin)")
    parser.add_argument("--speed", type=float, default=0.3, help="scripted forward speed, m/s")
    parser.add_argument("--turn-rate", type=float, default=0.8, help="scripted turn rate, rad/s")
    parser.add_argument("--seed", type=int, default=None, help="override the config's seed")
    parser.add_argument("--no-gui", action="store_true", help="only print the comparison")
    args = parser.parse_args()

    robot_cfg = load_robot_config()
    nav_cfg = load_navigation_config()
    configure_logging("WARNING")

    script = None if args.closed_loop else drive_script(args.speed, args.turn_rate)
    duration = args.duration or (40.0 if args.closed_loop else sum(d for d, _ in script) + 3.0)

    runs = []
    for mode in ("ideal", "realistic"):
        sensor = load_sensor_simulation_config(mode=mode, seed=args.seed if args.seed is not None else "from_file")
        scenario = default_scenario(robot_cfg.robot.width, robot_cfg.robot.length)
        sim = ClosedLoopSimulation(
            environment=scenario.environment, ground_truth_start=scenario.ground_truth_start,
            goal=scenario.slam_frame_goal, nav_cfg=nav_cfg, robot_cfg=robot_cfg,
            max_simulation_time=duration, sensor_simulation=sensor, drive_script=script,
        )
        print(f"running {mode} simulation ({duration:.0f} s simulated)...", flush=True)
        frames = run_and_record(sim, mode)
        runs.append((mode, sim, frames))

    for mode, sim, _ in runs:
        m = sim.metrics
        print(f"\n=== {mode.upper()} ===")
        print(f"mean pose error {m.mean_position_error:.3f} m, max {m.max_position_error:.3f} m, "
              f"final {m.final_position_error:.3f} m; travelled {m.ground_truth_distance:.2f} m; "
              f"nav state {m.final_nav_state}; collision {m.collision}")
        if sim.realistic:
            print(format_sensor_report(sim.sensor_report()))

    if args.no_gui:
        return
    titles = {"ideal": "IDEAL simulated LiDAR", "realistic": "REALISTIC RPLiDAR A2M12 model"}
    viewer = SensorComparisonViewer(
        [(titles[mode], frames) for mode, _, frames in runs],
        scale=46.0,
        view_bounds=(-5.2, -4.2, 5.2, 4.2),  # the whole 10 x 8 m room, SLAM frame
    )
    viewer.run()


if __name__ == "__main__":
    main()
