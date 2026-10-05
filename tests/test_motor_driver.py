"""hardware/motor_driver.py (velocity -> L298N duty, safety) and
hardware/motor_guard.py (separate-process pin guard), plus how the live
backend's robot chain behaves -- all without GPIO (RecordingMotor)."""

from __future__ import annotations

import dataclasses
import math
import os
import subprocess
import sys
import time

import pytest

from common.config import MotionLimits, MotorConfig, load_hardware_config, load_robot_config
from common.messages import VelocityCommand as V
from hardware.motor_driver import GpioDiffDriveRobot, RecordingMotor, body_to_duty
from hardware.motor_guard import guard, parent_alive

CFG = load_hardware_config().motors
LIMITS = load_robot_config().limits


def robot(cfg: MotorConfig = CFG, limits: MotionLimits = LIMITS) -> GpioDiffDriveRobot:
    return GpioDiffDriveRobot(cfg, limits, motor_factory=RecordingMotor)


def sides(r: GpioDiffDriveRobot) -> tuple[float, float]:
    v = r.motor_values()
    assert v["front_left"] == v["back_left"] and v["front_right"] == v["back_right"]  # skid-steer pairs
    return v["front_left"], v["front_right"]


def test_config_matches_the_robot_wiring() -> None:
    assert CFG.pins == {"front_left": [12, 17], "back_left": [25, 26], "front_right": [23, 19], "back_right": [21, 20]}
    r = robot()
    assert {n: m.pins for n, m in r._motors.items()} == {n: tuple(p) for n, p in CFG.pins.items()}
    r.close()


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(CFG, pins={**CFG.pins, "back_right": [17, 21]})   # a pin used twice
    with pytest.raises(ValueError):
        dataclasses.replace(CFG, min_duty=0.9, max_duty=0.5)


def test_forward_backward_and_turn_directions() -> None:
    r = robot()
    r.set_velocity(V(0.2, 0.0))
    left, right = sides(r)
    assert left > 0 and right > 0 and left == right
    r.set_velocity(V(-0.2, 0.0))
    left, right = sides(r)
    assert left < 0 and right < 0
    r.set_velocity(V(0.0, 1.0))           # +angular = turn LEFT (counter-clockwise)
    left, right = sides(r)
    assert left < 0 < right
    r.set_velocity(V(0.0, -1.0))
    left, right = sides(r)
    assert right < 0 < left
    r.set_velocity(V(0.2, 0.5))           # arc left: right side faster
    left, right = sides(r)
    assert 0 < left < right
    r.close()


def test_duty_is_capped_but_the_turn_radius_is_kept() -> None:
    left, right = body_to_duty(0.5, 1.5, CFG)                   # far beyond max_duty
    assert max(abs(left), abs(right)) == pytest.approx(CFG.max_duty)
    raw_l = 0.5 - 1.5 * CFG.track_width_m / 2
    raw_r = 0.5 + 1.5 * CFG.track_width_m / 2
    assert left / right == pytest.approx(raw_l / raw_r)          # same ratio -> same curvature


def test_deadband_trim_and_inversion() -> None:
    r = robot(dataclasses.replace(CFG, right_trim=0.8, invert_left=True))
    r.set_velocity(V(0.01 * CFG.max_wheel_speed_mps, 0.0))       # tiny: below zero_threshold -> stop
    assert sides(r) == (0.0, 0.0)
    r.set_velocity(V(0.05, 0.0))                                 # small: raised to min_duty (not stalling)
    left, right = sides(r)
    assert left == pytest.approx(-CFG.min_duty)                  # inverted side
    assert right == pytest.approx(CFG.min_duty)                  # trim applied, then raised to min_duty
    r.set_velocity(V(0.5, 0.0))
    left, right = sides(r)
    assert right == pytest.approx(CFG.max_duty * 0.8)            # the stronger side trimmed down
    r.close()


def test_commands_are_clamped_and_invalid_ones_stop_the_robot() -> None:
    r = robot()
    r.set_velocity(V(9.0, -9.0))
    assert r.last_command == V(LIMITS.max_linear_velocity, -LIMITS.max_angular_velocity)
    r.set_velocity(V(0.2, 0.0))
    for bad in (V(math.nan, 0.0), V(0.1, math.inf), V(-math.inf, 0.0)):
        r.set_velocity(bad)
        assert sides(r) == (0.0, 0.0)
        r.set_velocity(V(0.2, 0.0))
    assert r.rejected_commands == 3
    r.close()


def test_watchdog_stops_the_motors_when_commands_stop_arriving() -> None:
    limits = dataclasses.replace(LIMITS, command_timeout=0.2)
    r = robot(limits=limits)
    r.set_velocity(V(0.3, 0.0))
    assert sides(r) != (0.0, 0.0)
    time.sleep(0.1)
    assert sides(r) != (0.0, 0.0)                                # still fresh
    time.sleep(0.25)
    assert sides(r) == (0.0, 0.0) and r.watchdog_trips == 1
    r.close()


def test_regular_commands_keep_it_running() -> None:
    limits = dataclasses.replace(LIMITS, command_timeout=0.2)
    r = robot(limits=limits)
    for _ in range(10):
        r.set_velocity(V(0.3, 0.0))
        time.sleep(0.05)
    assert sides(r) != (0.0, 0.0) and r.watchdog_trips == 0
    r.close()


def test_close_stops_and_releases_every_motor() -> None:
    r = robot()
    r.set_velocity(V(0.3, 0.3))
    r.close()
    assert all(v == 0.0 for v in r.motor_values().values())
    assert all(m.closed for m in r._motors.values())
    r.set_velocity(V(0.3, 0.0))                                  # after close: ignored
    assert all(v == 0.0 for v in r.motor_values().values())


def test_without_gpiozero_it_falls_back_to_dry_run() -> None:
    r = GpioDiffDriveRobot(CFG, LIMITS)                          # no factory; gpiozero absent on the laptop
    try:
        import gpiozero  # noqa: F401
        pytest.skip("gpiozero installed here")
    except ImportError:
        assert r.dry_run
    finally:
        r.close()


def test_pose_comes_from_slam_not_from_the_motors() -> None:
    from common.messages import Pose2D
    r = robot()
    r.set_velocity(V(0.3, 0.0))
    assert r.get_pose() == Pose2D(0.0, 0.0, 0.0)                  # no odometry invented
    r.set_pose(Pose2D(1.0, 2.0, 0.5))
    assert r.get_pose() == Pose2D(1.0, 2.0, 0.5)
    r.close()


# ------------------------------------------------------ live robot chain --

def test_stop_latch_gates_the_real_motor_driver_and_commands_are_recorded() -> None:
    from runtime.backends import CommandRecorder, SafetyGatedRobot
    from slam.motion_model import CommandHistory

    motors = robot()
    history = CommandHistory(LIMITS.max_linear_velocity, LIMITS.max_angular_velocity)
    latched = {"on": False}
    chain = SafetyGatedRobot(CommandRecorder(motors, history), lambda: latched["on"])
    chain.set_velocity(V(0.3, 0.0))
    assert sides(motors)[0] > 0 and history.command_at(time.time()) == (0.3, 0.0)
    latched["on"] = True
    chain.set_velocity(V(0.3, 0.0))                              # refused: stop sent instead
    assert sides(motors) == (0.0, 0.0) and history.command_at(time.time()) == (0.0, 0.0)
    latched["on"] = False
    chain.set_velocity(V(0.2, 0.0))
    assert sides(motors)[0] > 0
    motors.close()


def test_runtime_latches_stop_at_startup_when_motors_are_attached() -> None:
    from runtime.config import RuntimeConfig
    from runtime.core import RobotRuntime
    from tests.test_robot_runtime import FakeBackend

    backend = FakeBackend()
    backend.requires_resume_at_start = True
    runtime = RobotRuntime(backend, RuntimeConfig(), host="127.0.0.1", port=0)
    assert runtime.is_stop_latched()
    runtime.tick(0.1)
    assert backend.nav_manager.state.value == "EMERGENCY_STOP"


# ---------------------------------------------------------- motor guard --

def test_guard_engages_when_the_runtime_dies() -> None:
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    engaged = []
    try:
        assert parent_alive(parent.pid)
        parent.kill()
        parent.wait()
        stop_after = time.monotonic() + 1.0
        # The guard checks the watched pid; os.getppid() is this test process, so
        # only the dead-parent path can trigger here.
        result = guard(parent.pid, [17, 12], poll_s=0.01, on_parent_death=engaged.append,
                       running=lambda: time.monotonic() < stop_after)
        assert result is True and engaged == [[17, 12]]
    finally:
        if parent.poll() is None:
            parent.kill()


def test_guard_does_nothing_while_the_runtime_is_alive_and_stops_on_request() -> None:
    engaged = []
    stop_after = time.monotonic() + 0.3
    result = guard(os.getpid(), [17, 12], poll_s=0.01, on_parent_death=engaged.append,
                   running=lambda: time.monotonic() < stop_after)
    assert result is False and engaged == []


def test_spawned_guard_process_exits_cleanly_on_normal_shutdown() -> None:
    from hardware.motor_guard import spawn_motor_guard
    handle = spawn_motor_guard([17, 12, 19, 23, 26, 25, 20, 21], poll_s=0.05)
    time.sleep(0.5)
    assert handle.process.poll() is None                         # watching
    handle.stop()
    assert handle.process.poll() is not None                     # gone, did not engage


def test_spawned_guard_engages_after_the_runtime_is_killed_hard() -> None:
    """The failure the guard exists for: the runtime process is SIGKILLed (no
    cleanup runs). A child 'runtime' spawns the real guard subprocess and is
    killed -9; the guard must notice and engage (on a laptop without gpiozero
    it logs what it would drive LOW)."""
    import signal
    code = ("import sys, time; sys.path.insert(0, '.'); "
            "from hardware.motor_guard import spawn_motor_guard; "
            "h = spawn_motor_guard([17, 12, 19, 23, 26, 25, 20, 21], poll_s=0.02); "
            "print(h.process.pid, flush=True); time.sleep(30)")
    fake_runtime = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
    guard_pid = int(fake_runtime.stdout.readline())
    time.sleep(0.3)
    fake_runtime.kill()                                            # kill -9: no cleanup at all
    fake_runtime.wait()
    deadline = time.monotonic() + 3.0
    log = ""
    import fcntl
    fl = fcntl.fcntl(fake_runtime.stderr, fcntl.F_GETFL)
    fcntl.fcntl(fake_runtime.stderr, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    while time.monotonic() < deadline and "forcing pins" not in log:
        time.sleep(0.05)
        try:
            log += fake_runtime.stderr.read() or ""
        except (TypeError, OSError):
            pass
    try:
        assert "died without stopping the motors -- forcing pins [17, 12, 19, 23, 26, 25, 20, 21] LOW" in log
        assert parent_alive(guard_pid)                             # still holding the pins
    finally:
        os.kill(guard_pid, signal.SIGTERM)


# ------------------------------------------- pins are left driven LOW, not floating --

def test_close_leaves_every_motor_pin_held_low() -> None:
    held = []
    r = GpioDiffDriveRobot(CFG, LIMITS, motor_factory=RecordingMotor, hold_pins_low=lambda pins: held.append(pins))
    r.set_velocity(V(0.2, 0.0))
    r.close()
    assert len(held) == 1
    assert sorted(held[0]) == sorted(p for pair in CFG.pins.values() for p in pair)
    assert r.motor_values() == {name: 0.0 for name in r.motor_values()}


def test_dry_run_close_touches_no_pins() -> None:
    held = []
    r = GpioDiffDriveRobot(CFG, LIMITS, dry_run=True, hold_pins_low=lambda pins: held.append(pins))
    r.close()
    assert held == []


def test_hold_pins_low_runs_pinctrl_output_low() -> None:
    from hardware.pin_hold import hold_pins_low
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    assert hold_pins_low([17, 12, 26], run=run, which=lambda _: "/usr/bin/pinctrl")
    assert calls == [["/usr/bin/pinctrl", "set", "17,12,26", "op", "dl"]]


def test_hold_pins_low_reports_failure_without_raising() -> None:
    from hardware.pin_hold import hold_pins_low
    assert not hold_pins_low([17], which=lambda _: None)
    assert not hold_pins_low([17], run=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "denied"),
                             which=lambda _: "/usr/bin/pinctrl")

    def boom(cmd, **kw):
        raise OSError("no")
    assert not hold_pins_low([17], run=boom, which=lambda _: "/usr/bin/pinctrl")
