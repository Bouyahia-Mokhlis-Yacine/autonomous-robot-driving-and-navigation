"""Motor-command watchdog below the Python control loop (hardware.motor_protocol).

Logic tests drive ``MotorControllerLogic`` (the microcontroller firmware's
reference logic) with an explicit clock. Process tests run the fake
controller (``hardware.fake_motor_controller``) and a Pi-side sender as two
separate OS processes connected by a pipe -- the sender is then frozen
(SIGSTOP), kept busy in a 4.8 s computation, killed, or its link closed, and
the controller must stop the motors on its own within the 300 ms timeout.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from common.config import load_hardware_config, load_robot_config
from common.messages import VelocityCommand
from hardware.motor_link import MicrocontrollerDiffDriveRobot, MotorCommandLink
from hardware.motor_protocol import (
    ARMED,
    DISARMED,
    STOPPED,
    TRIPPED,
    MotorControllerLogic,
    crc8,
    encode_motion,
    encode_stop,
    parse_status,
)

ROOT = Path(__file__).resolve().parent.parent


class Host:
    """A Pi-side frame source with its own clock (ms) and sequence counter."""

    def __init__(self, session: int = 0x1234, clock_offset_ms: int = 5_000_000) -> None:
        self.session, self.seq, self.offset = session, 0, clock_offset_ms

    def motion(self, now_ms: int, left: int, right: int, seq: int | None = None, host_ms: int | None = None) -> bytes:
        if seq is None:
            self.seq += 1
            seq = self.seq
        return encode_motion(self.session, seq, now_ms + self.offset if host_ms is None else host_ms, left, right)

    def stop(self, now_ms: int) -> bytes:
        self.seq += 1
        return encode_stop(self.session, self.seq, now_ms + self.offset)


def armed(logic: MotorControllerLogic, host: Host, now_ms: int = 0) -> MotorControllerLogic:
    assert logic.feed_line(host.motion(now_ms, 0, 0), now_ms)
    assert logic.state == ARMED
    return logic


# ------------------------------------------------------------ logic: basics --

def test_power_up_needs_a_zero_command_before_any_motion() -> None:
    logic, host = MotorControllerLogic(), Host()
    assert logic.state == DISARMED
    assert not logic.feed_line(host.motion(0, 500, 500), 0)
    assert logic.duty == (0, 0) and logic.state == DISARMED
    armed(logic, host, 10)
    assert logic.feed_line(host.motion(20, 500, -300), 20) and logic.duty == (500, -300)


def test_continuous_valid_commands_keep_the_watchdog_alive() -> None:
    logic, host = MotorControllerLogic(), Host()
    armed(logic, host)
    for t in range(100, 10_000, 100):                 # 10 Hz for 10 s
        logic.tick(t - 1)
        assert logic.state == ARMED
        assert logic.feed_line(host.motion(t, 400, 400), t)
    assert logic.counters.trips == 0 and logic.duty == (400, 400)


def test_command_timeout_stops_the_motors() -> None:
    logic, host = MotorControllerLogic(timeout_ms=300), Host()
    armed(logic, host)
    assert logic.feed_line(host.motion(100, 600, 600), 100)
    logic.tick(400)                                   # exactly 300 ms: still fine
    assert logic.state == ARMED and logic.duty == (600, 600)
    logic.tick(401)
    assert logic.state == TRIPPED and logic.duty == (0, 0) and logic.trip_ms == 401


def test_stop_frame_stops_immediately_and_motion_needs_rearming() -> None:
    logic, host = MotorControllerLogic(), Host()
    armed(logic, host)
    logic.feed_line(host.motion(50, 700, 700), 50)
    logic.feed_line(host.stop(60), 60)
    assert logic.state == STOPPED and logic.duty == (0, 0)
    assert not logic.feed_line(host.motion(70, 700, 700), 70) and logic.duty == (0, 0)
    assert logic.feed_line(host.motion(80, 0, 0), 80) and logic.state == ARMED
    assert logic.feed_line(host.motion(90, 200, 200), 90) and logic.duty == (200, 200)


def test_a_stop_frame_is_obeyed_even_if_stale() -> None:
    logic, host = MotorControllerLogic(), Host()
    armed(logic, host)
    logic.feed_line(host.motion(50, 700, 700), 50)
    old_stop = encode_stop(host.session, 1, 0)        # old seq, ancient timestamp
    logic.feed_line(old_stop, 60)
    assert logic.state == STOPPED and logic.duty == (0, 0)


def test_after_a_trip_motion_resumes_only_after_a_zero_command() -> None:
    logic, host = MotorControllerLogic(), Host()
    armed(logic, host)
    logic.feed_line(host.motion(100, 500, 500), 100)
    logic.tick(1000)
    assert logic.state == TRIPPED
    for t in (1010, 1110, 1210):                      # the unfrozen host sends motion again
        assert not logic.feed_line(host.motion(t, 500, 500), t)
        assert logic.duty == (0, 0) and logic.state == TRIPPED
    assert logic.feed_line(host.motion(1300, 0, 0), 1300) and logic.state == ARMED
    assert logic.feed_line(host.motion(1400, 500, 500), 1400) and logic.duty == (500, 500)


# ------------------------------------------- logic: nothing invalid refreshes --

def _malformed_frames(host: Host, now: int) -> list[bytes]:
    good = host.motion(now, 300, 300, seq=10_000)
    body = good.split(b"*")[0]
    return [
        b"",
        b"garbage\n",
        b"M,1234,5,6,7,8\n",                                       # no CRC
        body + b"*00\n" if crc8(body) != 0 else body + b"*01\n",   # wrong CRC
        good[:-5] + b"\n",                                         # truncated
        encode_motion(host.session, 10_001, now + host.offset, 1001, 0),  # out of range
        b"M,1234,x,y,1,2*" + f"{crc8(b'M,1234,x,y,1,2'):02X}".encode() + b"\n",
        b"M,1234,10002*" + f"{crc8(b'M,1234,10002'):02X}".encode() + b"\n",  # wrong field count
        b"X" * 200 + b"\n",
    ]


def test_malformed_commands_do_not_refresh_the_watchdog() -> None:
    logic, host = MotorControllerLogic(), Host()
    armed(logic, host)
    assert logic.feed_line(host.motion(100, 300, 300), 100)
    for t in range(110, 400, 10):
        for frame in _malformed_frames(host, t):
            assert not logic.feed_line(frame, t)
    logic.tick(401)
    assert logic.state == TRIPPED and logic.duty == (0, 0)
    assert logic.counters.malformed > 0 and logic.counters.accepted == 2


def test_duplicate_and_old_commands_do_not_refresh_the_watchdog() -> None:
    logic, host = MotorControllerLogic(), Host()
    armed(logic, host)
    frame = host.motion(100, 300, 300)
    assert logic.feed_line(frame, 100)
    last = host.seq
    for t in range(150, 400, 50):
        assert not logic.feed_line(frame, t)                                   # exact duplicate
        assert not logic.feed_line(host.motion(t, 300, 300, seq=last - 1), t)  # older seq
    logic.tick(401)
    assert logic.state == TRIPPED and logic.counters.stale_seq == 10


def test_late_delivered_commands_are_rejected() -> None:
    """A USB backlog flushed after a stall: frames with fresh sequence numbers
    but old timestamps must not refresh the watchdog or re-arm."""
    logic, host = MotorControllerLogic(max_age_ms=150), Host()
    armed(logic, host)
    for t in range(100, 600, 100):
        assert logic.feed_line(host.motion(t, 300, 300), t)
    backlog = [host.motion(t, 0 if t == 600 else 300, 300 if t != 600 else 0) for t in range(600, 1500, 100)]
    logic.tick(1500)                                  # the stall tripped the watchdog
    assert logic.state == TRIPPED
    for frame in backlog:                             # all delivered at once, 0.1-0.9 s late
        logic.feed_line(frame, 1500)
    late = [f for f, t in zip(backlog, range(600, 1500, 100)) if 1500 - t > 150]
    assert logic.state == TRIPPED and logic.duty == (0, 0)
    assert logic.counters.late >= len(late) - 1       # the zero "arming" frame among them too
    assert logic.feed_line(host.motion(1510, 0, 0), 1510) and logic.state == ARMED   # a FRESH zero re-arms


def test_clock_drift_does_not_cause_false_late_rejections() -> None:
    """MCU clock 0.05 % slower than the Pi's for an hour at 10 Hz."""
    logic, host = MotorControllerLogic(), Host()
    armed(logic, host)
    for i in range(1, 36_000):
        mcu = i * 100
        pi = int(mcu * 1.0005)
        assert logic.feed_line(host.motion(mcu, 200, 200, host_ms=pi + host.offset), mcu), i


def test_a_second_sender_is_ignored_while_armed_and_a_new_session_must_arm() -> None:
    logic, first, second = MotorControllerLogic(), Host(session=0x1111), Host(session=0x2222)
    armed(logic, first)
    assert logic.feed_line(first.motion(100, 300, 300), 100)
    assert not logic.feed_line(second.motion(110, 0, 0), 110) and logic.counters.wrong_session == 1
    logic.tick(1000)                                  # first sender died -> trip
    assert not logic.feed_line(second.motion(1010, 300, 300), 1010)
    assert logic.feed_line(second.motion(1020, 0, 0), 1020) and logic.state == ARMED   # restarted Pi process
    assert logic.feed_line(second.motion(1030, 300, 300), 1030)


def test_sequence_numbers_wrap_around() -> None:
    logic, host = MotorControllerLogic(), Host()
    host.seq = 2**32 - 3
    armed(logic, host)
    for t in range(100, 700, 100):
        assert logic.feed_line(host.motion(t, 100, 100), t)
    assert logic.last_seq < 10


# ----------------------------------------------------- Pi-side robot adapter --

def test_robot_adapter_sends_shaped_duty_and_sees_trips() -> None:
    frames: list[bytes] = []
    clock = [100.0]
    link = MotorCommandLink(frames.append, clock=lambda: clock[0], session=0x0ABC)
    robot = MicrocontrollerDiffDriveRobot(link, load_hardware_config().motors, load_robot_config().limits)
    logic = MotorControllerLogic()
    robot.arm()
    robot.set_velocity(VelocityCommand(0.2, 0.0))
    for i, frame in enumerate(frames):
        logic.feed_line(frame, 10 * i)
    expected = tuple(int(round(d * 1000)) for d in robot.last_duty)
    assert logic.state == ARMED and logic.duty == expected and expected[0] > 0
    robot.stop()
    logic.feed_line(frames[-1], 30)
    assert logic.duty == (0, 0)
    link.handle_line(logic.status_line())
    assert link.controller_tripped() and parse_status(logic.status_line())["state"] == STOPPED
    robot.set_velocity(VelocityCommand(float("nan"), 0.0))
    assert robot.rejected_commands == 1 and frames[-1].startswith(b"S,")


# ------------------------------------------------ separate OS processes --

SENDER = r"""
import os, sys, time
from hardware.motor_link import MotorCommandLink
fd, mode, after = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
link = MotorCommandLink(lambda b: os.write(fd, b))
link.send_duty(0.0, 0.0)                       # arm
n = 0
while True:
    link.send_duty(0.5, 0.5)
    n += 1
    if n == after:
        if mode == "busy":                     # Python alive, control loop stuck in a 4.8 s computation
            end = time.monotonic() + 4.8
            x = 0
            while time.monotonic() < end:
                x += 1
        elif mode == "crash":
            os._exit(1)
        elif mode == "unplug":
            os.close(fd)
            time.sleep(60)
    time.sleep(0.1)
"""


class Rig:
    """fake controller process <-pipe- sender process; events read here."""

    def __init__(self, mode: str = "normal", after: int = 10 ** 9) -> None:
        cmd_r, cmd_w = os.pipe()
        self.ev_r, ev_w = os.pipe()
        stop_r, self.stop_w = os.pipe()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        self.controller = subprocess.Popen(
            [sys.executable, "-m", "hardware.fake_motor_controller", "--read-fd", str(cmd_r),
             "--events-fd", str(ev_w), "--stop-fd", str(stop_r)],
            pass_fds=(cmd_r, ev_w, stop_r), cwd=ROOT, env=env)
        self.sender = subprocess.Popen([sys.executable, "-c", SENDER, str(cmd_w), mode, str(after)],
                                       pass_fds=(cmd_w,), cwd=ROOT, env=env)
        for fd in (cmd_r, cmd_w, ev_w, stop_r):
            os.close(fd)
        self._buf = b""
        self.events: list[tuple[float, str, int, int, float | None]] = []

    def read_events(self, until: float) -> None:
        while time.monotonic() < until:
            ready, _, _ = select.select([self.ev_r], [], [], max(0.0, until - time.monotonic()))
            if not ready:
                break
            chunk = os.read(self.ev_r, 4096)
            if not chunk:
                break
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                t, state, left, right, last = line.decode().split()
                self.events.append((float(t), state, int(left), int(right), None if last == "-" else float(last)))

    def wait_for(self, predicate, timeout: float):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for event in self.events:
                if predicate(event):
                    return event
            self.read_events(min(end, time.monotonic() + 0.05))
        for event in self.events:
            if predicate(event):
                return event
        return None

    def close(self) -> None:
        for proc in (self.sender,):
            try:
                proc.send_signal(signal.SIGCONT)
            except ProcessLookupError:
                pass
            proc.kill()
            proc.wait(timeout=5)
        os.close(self.stop_w)
        self.controller.wait(timeout=5)
        os.close(self.ev_r)


@pytest.fixture
def rig_factory():
    rigs = []

    def make(mode="normal", after=10 ** 9):
        rigs.append(Rig(mode, after))
        return rigs[-1]

    yield make
    for rig in rigs:
        rig.close()


def _driving(rig) -> tuple:
    event = rig.wait_for(lambda e: e[1] == ARMED and e[2] == 500, 10.0)
    assert event is not None, "controller never started driving"
    return event


def _trip(rig, timeout: float = 3.0):
    return rig.wait_for(lambda e: e[1] == TRIPPED, timeout)


def _latency(trip) -> float:
    t, _, left, right, last_valid = trip
    assert (left, right) == (0, 0)
    return t - last_valid


def test_process_continuous_commands_keep_driving(rig_factory) -> None:
    rig = rig_factory()
    _driving(rig)
    assert _trip(rig, 1.5) is None                    # 1.5 s of 10 Hz commands: never trips


def test_process_frozen_python_sigstop_stops_the_motors(rig_factory) -> None:
    rig = rig_factory()
    _driving(rig)
    time.sleep(0.35)
    rig.sender.send_signal(signal.SIGSTOP)            # the whole Pi-side process frozen
    trip = _trip(rig)
    assert trip is not None
    assert 0.300 <= _latency(trip) <= 0.360, _latency(trip)
    rig.sender.send_signal(signal.SIGCONT)            # it wakes up and sends motion again ...
    time.sleep(0.6)
    rig.read_events(time.monotonic() + 0.1)
    assert rig.events[-1][1] == TRIPPED and rig.events[-1][2:4] == (0, 0)   # ... but stays stopped


def test_process_busy_control_loop_stops_the_motors(rig_factory) -> None:
    """Python alive but the loop stuck (the 4.8 s A* freeze seen on the Pi)."""
    rig = rig_factory("busy", after=5)
    _driving(rig)
    trip = _trip(rig)
    assert trip is not None and 0.300 <= _latency(trip) <= 0.360, trip


def test_process_runtime_crash_stops_the_motors(rig_factory) -> None:
    rig = rig_factory("crash", after=5)
    _driving(rig)
    trip = _trip(rig)
    assert trip is not None and 0.300 <= _latency(trip) <= 0.360, trip


def test_process_killed_runtime_stops_the_motors(rig_factory) -> None:
    rig = rig_factory()
    _driving(rig)
    time.sleep(0.25)
    rig.sender.kill()                                 # SIGKILL: no cleanup code runs at all
    trip = _trip(rig)
    assert trip is not None and 0.300 <= _latency(trip) <= 0.360, trip


def test_process_unplugged_link_stops_the_motors(rig_factory) -> None:
    rig = rig_factory("unplug", after=5)
    _driving(rig)
    trip = _trip(rig)
    assert trip is not None and 0.300 <= _latency(trip) <= 0.360, trip
