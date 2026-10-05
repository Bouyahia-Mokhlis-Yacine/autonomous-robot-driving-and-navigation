"""Stand-in for the motor microcontroller: ``MotorControllerLogic`` in its own
OS process, reading command frames from a file descriptor (a pipe or pty in
place of the USB serial port) on its own clock.

It is deliberately a separate process: the point of the watchdog is that it
keeps working when the Pi-side Python process is frozen (SIGSTOP), busy in a
long computation, crashed, or disconnected. Every state/duty change is
written to ``events_fd`` as a line
``<time.monotonic()> <state> <left> <right> <last_valid_monotonic|->`` --
CLOCK_MONOTONIC is system-wide on Linux, so tests can measure latencies
across processes.

    python -m hardware.fake_motor_controller --read-fd N --events-fd M [--stop-fd K]
"""

from __future__ import annotations

import argparse
import os
import select
import time

from hardware.motor_protocol import MotorControllerLogic


def run_fake_controller(read_fd: int, events_fd: int, stop_fd: int | None = None, status_fd: int | None = None,
                        timeout_ms: int = 300, loop_ms: float = 1.0) -> None:
    """Runs until ``stop_fd`` becomes readable (or closes). EOF on
    ``read_fd`` (link unplugged) is not special: frames simply stop, and the
    watchdog does the rest -- as on the real controller."""
    logic = MotorControllerLogic(timeout_ms=timeout_ms)
    start = time.monotonic()

    def now_ms() -> int:
        return int((time.monotonic() - start) * 1000.0)

    def to_monotonic(ms: int | None) -> float | None:
        return None if ms is None else start + ms / 1000.0

    buffer = b""
    link_open = True
    last_report = None
    last_status = 0.0
    watch = [read_fd] + ([stop_fd] if stop_fd is not None else [])
    while True:
        readable, _, _ = select.select(watch if link_open else watch[1:], [], [], loop_ms / 1000.0)
        if stop_fd is not None and stop_fd in readable:
            return
        if read_fd in readable:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                link_open = False          # EOF: nothing more will arrive
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                logic.feed_line(line, now_ms())
            if len(buffer) > 256:          # garbage without newlines: drop it
                buffer = b""
        logic.tick(now_ms())
        report = (logic.state, logic.duty)
        if report != last_report:
            last_report = report
            last_valid = to_monotonic(logic.last_valid_ms)
            os.write(events_fd, (f"{time.monotonic():.6f} {logic.state} {logic.duty[0]} {logic.duty[1]} "
                                 f"{'-' if last_valid is None else f'{last_valid:.6f}'}\n").encode())
        if status_fd is not None and time.monotonic() - last_status >= 0.1:
            last_status = time.monotonic()
            try:
                os.write(status_fd, logic.status_line())
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--read-fd", type=int, required=True)
    parser.add_argument("--events-fd", type=int, required=True)
    parser.add_argument("--stop-fd", type=int, default=None)
    parser.add_argument("--status-fd", type=int, default=None)
    parser.add_argument("--timeout-ms", type=int, default=300)
    args = parser.parse_args()
    run_fake_controller(args.read_fd, args.events_fd, args.stop_fd, args.status_fd, args.timeout_ms)


if __name__ == "__main__":
    main()
