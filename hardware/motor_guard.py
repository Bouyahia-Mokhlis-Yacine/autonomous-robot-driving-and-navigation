"""Motor guard: a SEPARATE process that forces the motor pins low if the robot
runtime dies without stopping the motors.

Why a separate process: the motors are driven by software PWM inside the
runtime's own process. If that process is killed hard (``kill -9``, out of
memory, interpreter crash), every in-process safety measure dies with it and
the pins can be left at whatever level they had -- a motor could keep
running. When the runtime's process ends, the kernel releases its GPIO lines,
so this guard can claim them, drive every IN pin LOW (L298N: both inputs low
= motor off) and HOLD them there.

Lifecycle:
    - ``spawn_motor_guard(pins)`` (the runtime, at startup) starts
      ``python -m hardware.motor_guard --parent-pid <runtime pid> --pins ...``
    - the guard polls the parent every ``--poll-s``
    - normal shutdown: the runtime stops its motors itself and then
      terminates the guard (``MotorGuardHandle.stop()``) -- no action taken
    - abnormal death: the guard claims the pins, drives them low, logs it, and
      keeps holding them until it is stopped (``pkill -f hardware.motor_guard``)
      -- a new runtime cannot claim the pins while it holds them, which is
      the intended fail-safe.
    - whenever the guard exits, it leaves the pins as outputs driven LOW
      (``hardware.pin_hold``), never as floating inputs.

Needs gpiozero on the robot. Without it (a laptop) the guard still runs and
reports what it would do, so the lifecycle is testable anywhere.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger("autonomous_navigation.hardware.motor_guard")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parent_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def drive_pins_low(pins: list[int]) -> list:
    """Claim each pin as an output held LOW. Returns the device objects
    (they must stay referenced to keep the pins claimed)."""
    try:
        from gpiozero import DigitalOutputDevice
    except (ImportError, NotImplementedError):
        logger.error("gpiozero not available: would drive pins %s LOW", pins)
        return []
    devices = []
    for pin in pins:
        try:
            devices.append(DigitalOutputDevice(pin, initial_value=False))
        except Exception:  # noqa: BLE001 -- keep going: every other pin still matters
            logger.exception("Could not claim GPIO %d to force it low", pin)
    return devices


def guard(parent_pid: int, pins: list[int], poll_s: float = 0.05,
          on_parent_death: Callable[[list[int]], object] = drive_pins_low,
          running: Callable[[], bool] = lambda: True, is_child: bool = False) -> bool:
    """Wait for the watched runtime to die; then force the pins low. Returns
    True if it engaged, False if asked to stop (``running()`` False) first.
    ``is_child``: this process was spawned by the runtime, so being re-parented
    (``getppid`` changes) also means the runtime is gone -- even if its pid
    was already reused."""
    while running():
        if not parent_alive(parent_pid) or (is_child and os.getppid() != parent_pid):
            logger.critical("Robot runtime (pid %d) died without stopping the motors -- forcing pins %s LOW",
                            parent_pid, pins)
            held = on_parent_death(pins)
            _keep = held  # noqa: F841 -- keep the claimed devices alive while holding
            while running():
                time.sleep(0.5)
            return True
        time.sleep(poll_s)
    return False


class MotorGuardHandle:
    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process

    def stop(self) -> None:
        """Normal shutdown: the runtime already stopped the motors itself."""
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()


def spawn_motor_guard(pins: list[int], poll_s: float = 0.05) -> MotorGuardHandle:
    cmd = [sys.executable, "-m", "hardware.motor_guard", "--parent-pid", str(os.getpid()),
           "--poll-s", str(poll_s), "--pins", *map(str, pins)]
    process = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
    logger.info("Motor guard started (pid %d) watching runtime pid %d, pins %s", process.pid, os.getpid(), pins)
    return MotorGuardHandle(process)


def main() -> None:
    parser = argparse.ArgumentParser(description="Force motor pins low if the robot runtime dies.")
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--pins", type=int, nargs="+", required=True)
    parser.add_argument("--poll-s", type=float, default=0.05)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [motor-guard] %(message)s")

    stop = {"requested": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C in the runtime's terminal must not kill the guard
    try:
        engaged = guard(args.parent_pid, args.pins, args.poll_s, running=lambda: not stop["requested"], is_child=True)
        if engaged:
            logger.info("Guard released on request")
    finally:
        # Whatever happened, never leave the pins floating (gpiozero resets
        # released pins to inputs; the L298N can read those as HIGH).
        from hardware.pin_hold import hold_pins_low
        hold_pins_low(args.pins)


if __name__ == "__main__":
    main()
