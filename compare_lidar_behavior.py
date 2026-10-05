"""Compare a real RPLiDAR recording against the simulated sensors.

Computes the same statistics (``simulation.lidar_statistics``) for:
    REAL       -- a recorded scan sequence (default: the 25-scan stationary
                  fixture, tests/fixtures/real_scans_stationary.json)
    REALISTIC  -- the RPLiDAR model (config/lidar_simulation.yaml), sensor
                  held stationary in a calibration scene
    IDEAL      -- the original SimulatedLidar at navigation.yaml's scan_rate
and prints them side by side with the REALISTIC-minus-REAL difference.

This is a TUNING AID, not a validation: it reports how far apart the
numbers are, never that the simulation "matches reality". Geometry-bound
statistics (valid readings per scan, range distribution, per-bin jitter)
depend on the scene, and the simulator cannot know the real room -- the
default "cluttered" scene only roughly imitates the recording's short
median range (~0.75 m). Scene-independent statistics (scan timing, dropout
rate in typically-valid directions, partial-scan rate) are the ones to
tune against. The per-bin stationary jitter is averaged over windows the
same length as the real recording, since that statistic depends on how
many scans it is computed from.

Run:
    conda activate AI-env
    python compare_lidar_behavior.py
    python compare_lidar_behavior.py --num-scans 500 --seed 7 --scene default
    python compare_lidar_behavior.py --real path/to/other_recording.json --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.config import load_navigation_config, load_robot_config
from common.messages import LaserScan, Pose2D
from simulation.environment import Environment
from simulation.lidar_statistics import scan_statistics
from simulation.obstacles import RectangleObstacle
from simulation.realistic_lidar import RealisticLidar
from simulation.scenario import default_scenario
from simulation.sensor_simulation_config import load_sensor_simulation_config
from simulation.simulated_lidar import SimulatedLidar

DEFAULT_REAL = Path(__file__).resolve().parent / "tests" / "fixtures" / "real_scans_stationary.json"


def cluttered_scene() -> Environment:
    """Small furnished room; median range ~1 m, like the recording's 0.75 m."""
    env = Environment(width=3.0, height=2.4, robot_length=0.3, robot_width=0.25, robot_pose=Pose2D(1.1, 1.0, 0.3))
    env.add_static_obstacle(RectangleObstacle(x=2.1, y=1.5, width=0.4, height=0.6))
    env.add_static_obstacle(RectangleObstacle(x=0.5, y=0.3, width=0.6, height=0.3))
    return env


def scene(name: str) -> Environment:
    if name == "cluttered":
        return cluttered_scene()
    robot = load_robot_config().robot
    return default_scenario(robot.width, robot.length).environment


def load_real(path: Path) -> list[LaserScan]:
    with path.open("r", encoding="utf-8") as f:
        return [LaserScan(**raw) for raw in json.load(f)]


def simulate_realistic(env: Environment, num_scans: int, seed: int | None) -> list[LaserScan]:
    cfg = load_sensor_simulation_config(mode="realistic", seed=seed)
    lidar = RealisticLidar(load_navigation_config().lidar, cfg.lidar, seed=cfg.seed)
    pose = env.robot_pose
    scans: list[LaserScan] = []
    horizon = 0.0
    while len(scans) < num_scans:
        horizon += 1.0
        scans += [r.scan for r in lidar.acquire_until(env, lambda t: pose, horizon) if r.scan is not None]
    return scans[:num_scans]


def simulate_ideal(env: Environment, num_scans: int) -> list[LaserScan]:
    lidar_cfg = load_navigation_config().lidar
    lidar = SimulatedLidar(lidar_cfg)
    return [lidar.scan(env, env.robot_pose, timestamp=(k + 1) / lidar_cfg.scan_rate) for k in range(num_scans)]


def windowed_jitter(scans: list[LaserScan], window: int) -> dict:
    """Per-bin stationary std, averaged over consecutive windows of ``window`` scans."""
    medians, p90s = [], []
    for start in range(0, len(scans) - window + 1, window):
        j = scan_statistics(scans[start:start + window]).get("stationary_bin_std_m", {})
        if j.get("median") is not None:
            medians.append(j["median"])
            p90s.append(j["p90"])
    return {
        "median": float(np.mean(medians)) if medians else None,
        "p90": float(np.mean(p90s)) if p90s else None,
        "windows": len(medians),
    }


ROWS = [
    # label, getter(stats) -> (mean, std) | value, display scale, unit
    ("scan interval", lambda s: (s["scan_interval_s"]["mean"], s["scan_interval_s"]["std"]), 1000, "ms"),
    ("scan interval min", lambda s: s["scan_interval_s"]["min"], 1000, "ms"),
    ("scan interval max", lambda s: s["scan_interval_s"]["max"], 1000, "ms"),
    ("scan frequency", lambda s: s["scan_frequency_hz"], 1, "Hz"),
    ("valid readings/scan", lambda s: (s["valid_per_scan"]["mean"], s["valid_per_scan"]["std"]), 1, ""),
    ("missing readings/scan", lambda s: (s["missing_per_scan"]["mean"], s["missing_per_scan"]["std"]), 1, ""),
    ("partial-scan fraction", lambda s: s["partial_fraction"], 100, "%"),
    ("dropout, typical bins", lambda s: s.get("dropout_rate_in_typical_bins"), 100, "%"),
    ("typically-valid bins", lambda s: s.get("typically_valid_bins"), 1, ""),
    ("range", lambda s: (s["range_m"]["mean"], s["range_m"]["std"]), 1, "m"),
    ("range median", lambda s: s["range_m"]["median"], 1, "m"),
    ("stationary bin std median", lambda s: s["jitter"]["median"], 1000, "mm"),
    ("stationary bin std p90", lambda s: s["jitter"]["p90"], 1000, "mm"),
]


def _cell(value, scale: float) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, tuple):
        mean, std = value
        if mean is None:
            return "n/a"
        return f"{mean * scale:8.2f} ± {std * scale:6.2f}"
    return f"{value * scale:8.2f}"


def _mean(value):
    return value[0] if isinstance(value, tuple) else value


def print_table(real: dict, realistic: dict, ideal: dict) -> None:
    header = f"{'metric':27s} {'REAL':>19s} {'REALISTIC SIM':>19s} {'IDEAL SIM':>19s} {'SIM-REAL':>10s}"
    print(header)
    print("-" * len(header))
    for label, get, scale, unit in ROWS:
        values = []
        for stats in (real, realistic, ideal):
            try:
                values.append(get(stats))
            except (KeyError, TypeError):
                values.append(None)
        r, s = _mean(values[0]), _mean(values[1])
        diff = f"{(s - r) * scale:+9.2f}" if r is not None and s is not None else "      n/a"
        name = f"{label} [{unit}]" if unit else label
        print(f"{name:27s} {_cell(values[0], scale):>19s} {_cell(values[1], scale):>19s} "
              f"{_cell(values[2], scale):>19s} {diff:>10s}")


def print_histograms(real: dict, realistic: dict) -> None:
    print()
    print("scan interval histogram (fraction of intervals per 10 ms bin)")
    rh = real["interval_histogram_ms"]
    sh = realistic["interval_histogram_ms"]
    rn = max(1, sum(c for _, _, c in rh))
    sn = max(1, sum(c for _, _, c in sh))
    print(f"  {'bin [ms]':>11s}  {'REAL':<26s}  {'REALISTIC SIM':<26s}")
    for (lo, hi, rc), (_, _, sc) in zip(rh, sh):
        rf, sf = rc / rn, sc / sn
        print(f"  {lo:4.0f}-{hi:<4.0f}  {rf:5.2f} {'#' * int(round(rf * 20)):<20s}  {sf:5.2f} {'#' * int(round(sf * 20)):<20s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare real RPLiDAR scans with the simulated sensors.")
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL, help="real scan recording (JSON list of LaserScan)")
    parser.add_argument("--num-scans", type=int, default=300, help="simulated scans per model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene", choices=("cluttered", "default"), default="cluttered")
    parser.add_argument("--json", type=Path, default=None, help="also write all statistics to this file")
    args = parser.parse_args()

    real_scans = load_real(args.real)
    env = scene(args.scene)
    realistic_scans = simulate_realistic(env, args.num_scans, args.seed)
    ideal_scans = simulate_ideal(scene(args.scene), args.num_scans)

    window = len(real_scans)
    stats = {}
    for name, scans in (("real", real_scans), ("realistic", realistic_scans), ("ideal", ideal_scans)):
        s = scan_statistics(scans)
        s["jitter"] = windowed_jitter(scans, window)
        stats[name] = s

    print(f"REAL: {args.real.name} ({len(real_scans)} scans)   "
          f"SIM: {args.num_scans} stationary scans, scene '{args.scene}', seed {args.seed}")
    print("Descriptive comparison only -- similar numbers do not prove the simulation is realistic.")
    print("Scene-dependent rows: valid/missing readings, typically-valid bins, range, bin std.")
    print()
    print_table(stats["real"], stats["realistic"], stats["ideal"])
    print_histograms(stats["real"], stats["realistic"])

    if args.json is not None:
        args.json.write_text(json.dumps(stats, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
