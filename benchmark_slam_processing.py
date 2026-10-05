"""Wall-clock benchmark of one SLAM processing step, broken down by stage.

Replays a deterministic realistic-sensor run (slow in-place spin) with every scan
processed (instantaneous processing model, so SLAM sees ~11 scans/s and
there are plenty of samples), and times each stage of
``ScanMatchingPipeline.process_scan`` with ``time.perf_counter`` by
wrapping the stage functions from outside (the pipeline is not modified):

    preprocessing   slam.scan_preprocessing.preprocess_scan
    reference query PersistentWorldModel.get_confirmed_occupied_points_world
    registration    slam.scan_registration.register (ICP)
    ray tracing     mapping.ray_casting.bresenham_line calls inside integrate()
    map integration PersistentWorldModel.integrate() minus its ray tracing
    other           everything else in process_scan (seed prediction, gating,
                    bookkeeping)
    total           the whole process_scan call

Numbers are for THIS machine; the realistic simulation's processing-delay
model (config/lidar_simulation.yaml) should be set from them.

Run:
    conda activate AI-env
    python benchmark_slam_processing.py                # ~300 processed scans
    python benchmark_slam_processing.py --duration 60 --json bench.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from collections import defaultdict

import numpy as np

import mapping.persistent_world_model as pwm_module
import slam.scan_matching_pipeline as pipeline_module
from common.config import load_navigation_config, load_robot_config
from common.messages import VelocityCommand
from simulation.closed_loop import ClosedLoopSimulation
from simulation.scenario import default_scenario
from simulation.sensor_processing import ProcessingConfig
from simulation.sensor_simulation_config import load_sensor_simulation_config

STAGES = ("preprocessing", "reference_query", "registration", "ray_tracing", "map_integration", "other", "total")


def summarize(samples: list[float]) -> dict:
    a = np.asarray(samples) * 1000.0
    return {
        "n": int(a.size), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max()),
    }


def run_benchmark(duration: float, seed: int) -> dict[str, list[float]]:
    robot_cfg = load_robot_config()
    nav_cfg = load_navigation_config()
    sensor = load_sensor_simulation_config(mode="realistic", seed=seed)
    sensor = dataclasses.replace(sensor, processing=ProcessingConfig.instantaneous())
    scenario = default_scenario(robot_cfg.robot.width, robot_cfg.robot.length)
    # A slow in-place spin: deterministic, never collides, and sweeps the
    # whole room, so every scan is a realistic full-room observation.
    sim = ClosedLoopSimulation(
        environment=scenario.environment, ground_truth_start=scenario.ground_truth_start,
        goal=scenario.slam_frame_goal, nav_cfg=nav_cfg, robot_cfg=robot_cfg,
        max_simulation_time=duration, sensor_simulation=sensor,
        drive_script=[(duration, VelocityCommand(linear=0.0, angular=0.3))],
    )

    per_scan: dict[str, float] = defaultdict(float)
    samples: dict[str, list[float]] = defaultdict(list)

    def timed(name, fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                per_scan[name] += time.perf_counter() - t0
        return wrapper

    # Module-level names the pipeline / world model actually call.
    orig_pre, orig_reg = pipeline_module.preprocess_scan, pipeline_module.register
    orig_bres = getattr(pwm_module, "bresenham_line", None)
    pipeline_module.preprocess_scan = timed("preprocessing", orig_pre)
    pipeline_module.register = timed("registration", orig_reg)
    if orig_bres is not None:
        pwm_module.bresenham_line = timed("ray_tracing", orig_bres)
    wm = sim.world_model
    wm.get_confirmed_occupied_points_world = timed("reference_query", wm.get_confirmed_occupied_points_world)
    wm.integrate = timed("integrate", wm.integrate)
    process = sim.pipeline.process_scan

    def measured(scan):
        per_scan.clear()
        t0 = time.perf_counter()
        result = process(scan)
        total = time.perf_counter() - t0
        integrate = per_scan.get("integrate", 0.0)
        tracing = per_scan.get("ray_tracing", 0.0)
        samples["total"].append(total)
        for name in ("preprocessing", "reference_query", "registration"):
            samples[name].append(per_scan.get(name, 0.0))
        samples["ray_tracing"].append(tracing)
        samples["map_integration"].append(integrate - tracing)
        samples["other"].append(total - integrate - sum(per_scan.get(n, 0.0) for n in
                                                          ("preprocessing", "reference_query", "registration")))
        return result

    sim.pipeline.process_scan = measured
    try:
        sim.run()
    finally:
        pipeline_module.preprocess_scan, pipeline_module.register = orig_pre, orig_reg
        if orig_bres is not None:
            pwm_module.bresenham_line = orig_bres
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SLAM processing per scan, by stage.")
    parser.add_argument("--duration", type=float, default=30.0, help="simulated seconds (~11 scans each)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    samples = run_benchmark(args.duration, args.seed)
    stats = {name: summarize(samples[name]) for name in STAGES if samples.get(name)}
    total_mean = stats["total"]["mean_ms"]
    print(f"SLAM processing per scan ({stats['total']['n']} scans, realistic sensor, this machine)")
    print(f"{'stage':18s} {'mean':>9s} {'median':>9s} {'p95':>9s} {'max':>9s} {'share':>7s}")
    for name in STAGES:
        if name not in stats:
            continue
        s = stats[name]
        share = "" if name == "total" else f"{100 * s['mean_ms'] / total_mean:6.1f}%"
        print(f"{name:18s} {s['mean_ms']:8.2f}ms {s['median_ms']:8.2f}ms {s['p95_ms']:8.2f}ms "
              f"{s['max_ms']:8.2f}ms {share:>7s}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"stats": stats, "samples_s": {k: list(v) for k, v in samples.items()}}, f, indent=2)


if __name__ == "__main__":
    main()
