"""Localization stress-test matrix for the realistic sensor simulation.

Runs a fixed set of scenarios (scripted motion at several turn rates,
navigation-driven runs, an induced sensor blackout, processing overload)
and reports, per scenario, the metrics that decide whether the robot
software copes with realistic sensing: position/heading error against
ground truth, localization-loss events, scan acceptance, processing rate,
dropped scans and latency. Ground truth is used ONLY here, for evaluation.

Scripted scenarios (``drive_script``) measure localization alone -- the
navigation manager is bypassed, so they show whether SLAM keeps track, not
whether navigation reacts safely. The ``nav_*`` scenarios exercise the full
stack including the localization-health response.

Run:
    conda activate AI-env
    python localization_stress_test.py                 # all scenarios, realistic sensor
    python localization_stress_test.py D_fast_turn nav_goal_behind
    python localization_stress_test.py --sensor ideal  # same matrix, ideal sensor
    python localization_stress_test.py --json out.json
    python localization_stress_test.py --localization delay_compensation=false   # A/B a feature
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from common.config import load_navigation_config, load_robot_config
from common.geometry import distance, wrap_angle
from common.messages import Goal, LaserScan, VelocityCommand
from simulation.closed_loop import ClosedLoopSimulation
from simulation.scenario import default_scenario
from simulation.sensor_processing import ProcessingConfig
from simulation.sensor_simulation_config import SensorSimulationConfig, load_sensor_simulation_config

V = VelocityCommand


@dataclass
class Scenario:
    name: str
    description: str
    duration: float
    drive_script: list[tuple[float, VelocityCommand]] | None = None
    goal: Goal | None = None
    processing: ProcessingConfig | None = None
    # (start, end) simulated-time window during which every scan reaching
    # SLAM is replaced by an empty one (sensor blackout / induced failure).
    blackout: tuple[float, float] | None = None


def spin(rate: float, angle: float = 2.0 * math.pi) -> list[tuple[float, VelocityCommand]]:
    return [(angle / rate, V(0.0, rate))]


def path_script(turn_rate: float, speed: float = 0.3) -> list[tuple[float, VelocityCommand]]:
    quarter = (math.pi / 2.0) / turn_rate
    return [(0.75 / speed, V(speed, 0.0)), (quarter, V(0.0, turn_rate)), (1.2 / speed, V(speed, 0.0)),
            (quarter, V(0.0, -turn_rate)), (1.5 / speed, V(speed, 0.0))]


SCENARIOS = [
    Scenario("A_straight", "forward 1.0 m then back 3.0 m at 0.4 m/s", 14.0,
             drive_script=[(2.5, V(0.4, 0.0)), (7.5, V(-0.4, 0.0))]),
    Scenario("B_slow_turn", "spin 360 deg in place at 0.3 rad/s", 25.0, drive_script=spin(0.3)),
    Scenario("C_medium_turn", "spin 360 deg in place at 0.8 rad/s", 12.0, drive_script=spin(0.8)),
    Scenario("D_fast_turn", "spin 360 deg in place at 1.5 rad/s (max)", 10.0, drive_script=spin(1.5)),
    Scenario("D2_fast_path", "drive + 90 deg turns at 1.5 rad/s through the doorway", 20.0,
             drive_script=path_script(1.5)),
    Scenario("nav_goal", "navigate to the default goal through the doorway", 60.0,
             goal=Goal(3.0, 0.0)),
    Scenario("nav_goal_behind", "navigate to a goal behind the robot (180 deg turn first)", 40.0,
             goal=Goal(-3.0, -1.0)),
    Scenario("E_blackout_recovery", "navigate; sensor blackout 4.0-5.5 s; must stop, recover, resume", 60.0,
             goal=Goal(3.0, 0.0), blackout=(4.0, 5.5)),
    Scenario("F_overload", "navigate with SLAM processing 0.4-0.6 s per scan (overloaded)", 60.0,
             goal=Goal(3.0, 0.0), processing=ProcessingConfig(delay_min_s=0.4, delay_max_s=0.6)),
]


@dataclass
class ScenarioResult:
    name: str
    sensor: str
    sim_time: float
    goal_reached: bool
    collision: bool
    final_nav_state: str
    mean_position_error: float
    max_position_error: float
    final_position_error: float
    max_heading_error_deg: float
    final_heading_error_deg: float
    loss_events: int
    time_not_localized: float
    max_time_not_localized: float
    distance_driven_while_not_localized: float
    scans_accepted: int
    scans_rejected: int
    rejection_reasons: dict
    scans_generated: int
    scans_processed: int
    scans_dropped: int
    processing_rate_hz: float
    mean_processing_time_s: float | None
    max_processing_time_s: float | None
    max_queue_size: int
    mean_latency_s: float | None
    ground_truth_distance: float
    extra: dict = field(default_factory=dict)


def run_scenario(scn: Scenario, sensor: SensorSimulationConfig, localization_overrides: dict | None = None) -> ScenarioResult:
    robot_cfg = load_robot_config()
    nav_cfg = load_navigation_config()
    if localization_overrides:
        nav_cfg = dataclasses.replace(nav_cfg, localization=dataclasses.replace(nav_cfg.localization, **localization_overrides))
    if scn.processing is not None and sensor.is_realistic:
        sensor = dataclasses.replace(sensor, processing=scn.processing)
    base = default_scenario(robot_cfg.robot.width, robot_cfg.robot.length)
    sim = ClosedLoopSimulation(
        environment=base.environment, ground_truth_start=base.ground_truth_start,
        goal=scn.goal or base.slam_frame_goal, nav_cfg=nav_cfg, robot_cfg=robot_cfg,
        max_simulation_time=scn.duration, sensor_simulation=sensor, drive_script=scn.drive_script,
    )

    reasons: collections.Counter = collections.Counter()
    accepted = 0
    process = sim.pipeline.process_scan

    def observed(scan: LaserScan):
        nonlocal accepted
        if scn.blackout is not None and scn.blackout[0] <= scan.timestamp < scn.blackout[1]:
            scan = dataclasses.replace(scan, ranges=[scan.range_max] * scan.num_rays)
        result = process(scan)
        if result.accepted:
            accepted += 1
        else:
            reasons[result.rejection_reason] += 1
        return result

    sim.pipeline.process_scan = observed

    max_heading = 0.0
    not_loc_time = 0.0
    run_not_loc = 0.0
    max_run_not_loc = 0.0
    dist_not_loc = 0.0
    prev_gt = sim.environment.robot_pose
    while sim._sim_time < scn.duration - 1e-9:
        sim.step()
        gt = sim._ground_truth_to_slam(sim.environment.robot_pose)
        est = sim.nav_robot.get_pose()
        max_heading = max(max_heading, abs(wrap_angle(gt.theta - est.theta)))
        cur_gt = sim.environment.robot_pose
        # Health as navigation sees it (slam.localization_health), after bootstrap.
        if sim.metrics.bootstrap_completed and not sim.localization_health.is_localization_valid():
            not_loc_time += sim.dt
            run_not_loc += sim.dt
            max_run_not_loc = max(max_run_not_loc, run_not_loc)
            dist_not_loc += distance(prev_gt.x, prev_gt.y, cur_gt.x, cur_gt.y)
        else:
            run_not_loc = 0.0
        prev_gt = cur_gt
        if sim.metrics.goal_reached or sim.metrics.collision:
            break

    m = sim.metrics
    rep = sim.sensor_report() or {}
    if sim.realistic:
        pm = sim.processor.metrics
        generated, processed, dropped = rep["scans_generated"], pm.scans_processed, pm.scans_dropped
        proc_times = pm.processing_times
        max_q = pm.max_queue_size
        latency = pm.average_processing_latency
    else:
        generated = processed = accepted + sum(reasons.values())
        dropped, proc_times, max_q, latency = 0, [], 0, 0.0
    return ScenarioResult(
        name=scn.name, sensor=sensor.mode, sim_time=m.simulation_time, goal_reached=m.goal_reached,
        collision=m.collision, final_nav_state=m.final_nav_state,
        mean_position_error=m.mean_position_error or 0.0, max_position_error=m.max_position_error,
        final_position_error=m.final_position_error or 0.0,
        max_heading_error_deg=math.degrees(max_heading),
        final_heading_error_deg=math.degrees(m.final_heading_error or 0.0),
        loss_events=sim.localization_health.loss_events, time_not_localized=not_loc_time, max_time_not_localized=max_run_not_loc,
        distance_driven_while_not_localized=dist_not_loc,
        scans_accepted=accepted, scans_rejected=sum(reasons.values()), rejection_reasons=dict(reasons),
        scans_generated=generated, scans_processed=processed, scans_dropped=dropped,
        processing_rate_hz=processed / m.simulation_time if m.simulation_time else 0.0,
        mean_processing_time_s=float(np.mean(proc_times)) if proc_times else None,
        max_processing_time_s=float(np.max(proc_times)) if proc_times else None,
        max_queue_size=max_q, mean_latency_s=latency, ground_truth_distance=m.ground_truth_distance,
    )


def print_result(r: ScenarioResult, description: str) -> None:
    ms = lambda v: "n/a" if v is None else f"{v * 1000:.0f} ms"  # noqa: E731
    print(f"--- {r.name} [{r.sensor}] -- {description}")
    print(f"  t={r.sim_time:.1f}s goal_reached={r.goal_reached} collision={r.collision} "
          f"nav={r.final_nav_state} travelled={r.ground_truth_distance:.2f} m")
    print(f"  position error: mean {r.mean_position_error:.3f} m, max {r.max_position_error:.3f} m, "
          f"final {r.final_position_error:.3f} m | heading error: max {r.max_heading_error_deg:.1f} deg, "
          f"final {r.final_heading_error_deg:.1f} deg")
    print(f"  localization LOST (health monitor): {r.loss_events} events, {r.time_not_localized:.1f} s total "
          f"(longest {r.max_time_not_localized:.1f} s), robot moved {r.distance_driven_while_not_localized:.2f} m "
          f"while not localized")
    print(f"  scans: generated {r.scans_generated}, processed {r.scans_processed} "
          f"({r.processing_rate_hz:.1f} Hz), dropped {r.scans_dropped}, accepted {r.scans_accepted}, "
          f"rejected {r.scans_rejected} {r.rejection_reasons}")
    print(f"  SLAM processing: mean {ms(r.mean_processing_time_s)}, max {ms(r.max_processing_time_s)}, "
          f"arrival->result {ms(r.mean_latency_s)}, max queue {r.max_queue_size}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Localization stress-test matrix.")
    parser.add_argument("scenarios", nargs="*", help="subset of scenario names (default: all)")
    parser.add_argument("--sensor", choices=("ideal", "realistic"), default="realistic")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", default=None, help="write results to this JSON file")
    parser.add_argument("--localization", action="append", default=[], metavar="KEY=VALUE",
                        help="override a navigation.yaml localization setting, e.g. delay_compensation=false")
    args = parser.parse_args()
    overrides = {}
    for item in args.localization:
        key, _, value = item.partition("=")
        overrides[key] = {"true": True, "false": False}.get(value.lower(), None)
        if overrides[key] is None:
            overrides[key] = float(value) if "." in value else int(value)
    logging.disable(logging.WARNING)

    sensor = load_sensor_simulation_config(mode=args.sensor, seed=args.seed if args.seed is not None else "from_file")
    wanted = set(args.scenarios)
    unknown = wanted - {s.name for s in SCENARIOS}
    if unknown:
        raise SystemExit(f"unknown scenario(s): {sorted(unknown)}; choose from {[s.name for s in SCENARIOS]}")
    results = []
    for scn in SCENARIOS:
        if wanted and scn.name not in wanted:
            continue
        r = run_scenario(scn, sensor, overrides)
        print_result(r, scn.description)
        results.append(dataclasses.asdict(r))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
