"""Headless robot runtime -- runs on the Raspberry Pi (or locally with --backend sim).

    LiDAR (real, or the realistic simulator) -> latest-only SLAM worker ->
    ScanMatchingPipeline -> PersistentWorldModel -> LocalizationHealthMonitor ->
    NavigationManager -> RobotInterface (safety-gated; sensor-only or GPIO motors)
                                   ^
    operator commands (TCP) -------+   state/map snapshots -> operator GUI (TCP)

The runtime is the ONLY authority for map, pose, goal and path. The operator
GUI (operator_gui.py) only displays what it publishes and sends commands.
Safety policy (STOP latch, operator loss, stale/duplicate commands):
config/runtime.yaml and runtime/core.py. Protocol + coordinate conventions:
network/protocol.py.

Run:
    conda activate AI-env
    python robot_runtime.py                          # simulated robot (default), listens on :8765
    python robot_runtime.py --backend live           # real RPLiDAR (on the Pi), sensor-only
    python robot_runtime.py --backend live --motors gpio   # + wheel motors (STOP latched until RESUME)
    python robot_runtime.py --backend live --motors gpio --max-speed 0.2 --max-turn 0.6   # first floor tests
    python robot_runtime.py --port 9000 --seed 7 --config config/runtime.yaml
Stop with Ctrl+C (or SIGTERM): STOP is latched, the robot command zeroed, the
LiDAR disconnected.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import signal
import threading
import time

from common.config import load_hardware_config, load_navigation_config, load_robot_config
from common.logger import configure_logging, get_logger
from runtime.backends import LiveBackend, SimBackend
from runtime.config import load_runtime_config
from runtime.core import RobotRuntime

logger = get_logger(__name__)


def build_runtime(backend_name: str, config, seed: int | None = 42, lidar_port: str | None = None,
                  host: str | None = None, port: int | None = None,
                  max_speed: float | None = None, max_turn: float | None = None) -> RobotRuntime:
    robot_cfg, nav_cfg = load_robot_config(), load_navigation_config()
    limits = robot_cfg.limits
    for value, name, cap in ((max_speed, "max_speed", limits.max_linear_velocity),
                             (max_turn, "max_turn", limits.max_angular_velocity)):
        if value is not None and not 0.0 < value <= cap:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, {cap}] (robot.yaml limit); got {value}")
    if max_speed is not None or max_turn is not None:
        # Only ever LOWERS robot.yaml's limits; used everywhere they are (path
        # follower, command clamp, motor mapping, SLAM motion-plausibility gate).
        robot_cfg = dataclasses.replace(robot_cfg, limits=dataclasses.replace(
            limits, max_linear_velocity=max_speed or limits.max_linear_velocity,
            max_angular_velocity=max_turn or limits.max_angular_velocity))
        logger.warning("Speed limits lowered for this run: %.2f m/s, %.2f rad/s",
                       robot_cfg.limits.max_linear_velocity, robot_cfg.limits.max_angular_velocity)
    dt = 1.0 / config.control.rate_hz
    holder: dict = {}
    if backend_name == "sim":
        backend = SimBackend(nav_cfg, robot_cfg, dt=dt, seed=seed)
    elif backend_name == "live":
        backend = LiveBackend(nav_cfg, robot_cfg, load_hardware_config(), config.live,
                              stop_latched=lambda: holder["runtime"].is_stop_latched(), port=lidar_port)
    else:
        raise ValueError(f"unknown backend {backend_name!r}")
    runtime = RobotRuntime(backend, config, host=host, port=port)
    holder["runtime"] = runtime
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless robot runtime (Pi side).")
    parser.add_argument("--backend", choices=("sim", "live"), default="sim")
    parser.add_argument("--config", default=None, help="runtime config (default config/runtime.yaml)")
    parser.add_argument("--host", default=None, help="listen address (default from config)")
    parser.add_argument("--port", type=int, default=None, help="listen port (default from config)")
    parser.add_argument("--seed", type=int, default=42, help="simulator seed (sim backend)")
    parser.add_argument("--lidar-port", default=None, help="RPLiDAR serial port (live backend; default config)")
    parser.add_argument("--motors", choices=("none", "gpio"), default=None,
                        help="live backend wheel motors (default: config live.motors)")
    parser.add_argument("--motors-dry-run", action="store_true", help="gpio motor code path without touching GPIO")
    parser.add_argument("--max-speed", type=float, default=None,
                        help="m/s; LOWER robot.yaml's max_linear_velocity for this run (e.g. 0.2 for first floor tests)")
    parser.add_argument("--max-turn", type=float, default=None,
                        help="rad/s; LOWER robot.yaml's max_angular_velocity for this run")
    args = parser.parse_args()

    configure_logging("INFO")
    logging.getLogger("autonomous_navigation.navigation.navigation_manager").setLevel(logging.WARNING)
    config = load_runtime_config(args.config)
    if args.motors is not None:
        config.live.motors = args.motors
    if args.motors_dry_run:
        config.live.motors_dry_run = True
    runtime = build_runtime(args.backend, config, seed=args.seed, lidar_port=args.lidar_port,
                            host=args.host, port=args.port, max_speed=args.max_speed, max_turn=args.max_turn)
    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())

    port = runtime.start()
    s = config.safety
    logger.info("Robot runtime up: backend=%s, operators connect to port %d; operator-loss policy=%s, "
                "STOP latched at start=%s", args.backend, port, s.operator_loss_policy, runtime.is_stop_latched())
    if args.backend == "live" and config.live.motors == "gpio":
        logger.warning("MOTORS ENABLED%s. The robot will not move until an operator presses RESUME and "
                       "sets a goal. Keep the physical e-stop within reach.",
                       " (dry-run: GPIO untouched)" if config.live.motors_dry_run else "")
    try:
        last = time.monotonic()
        while not done.wait(1.0):
            if time.monotonic() - last >= 10.0:
                last = time.monotonic()
                logger.info("alive: ticks=%d overruns=%d max_tick=%.1f ms operators=%d nav=%s health=%s stop=%s",
                            runtime.tick_count, runtime.tick_overruns, runtime.max_tick_s * 1000,
                            runtime.server.operators_present(), runtime.backend.nav_manager.state.value,
                            runtime.backend.health.health.value, runtime.is_stop_latched())
    finally:
        logger.info("Shutting down: latching STOP, stopping the robot, closing connections ...")
        runtime.shutdown()
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
