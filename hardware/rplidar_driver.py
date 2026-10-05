"""Real RPLiDAR driver behind the same ``LaserScan`` abstraction as
``simulation.simulated_lidar.SimulatedLidar``.

Wraps the third-party ``rplidar`` package (a proven implementation of the
binary serial protocol) rather than reimplementing it -- consistent with
this project's approach to SLAM (Phase 4): don't hand-roll something a
maintained library already does correctly.

Environment-specific workaround
--------------------------------
RPLiDAR A2/A3 units use 256000 baud, which is not a "standard" POSIX
termios baud rate. On this development machine (and apparently others --
this is a known class of issue), pyserial's normal baud-rate setup fails
for such rates with ``termios.error: (22, 'Invalid argument')``: it
passes its internal ``BOTHER`` sentinel through the plain
``termios.tcsetattr`` call before it ever reaches the real custom-baud
ioctl, and that plain call rejects the sentinel. The fix used here: let
pyserial open and configure the port at a placeholder *standard* baud
rate (so its normal, well-tested flag setup succeeds), then patch only
the speed fields directly via a raw ``TCSETS2``/termios2 ioctl. Verified
against a real, connected RPLiDAR (256000 baud) in this environment.

Binning raw measurements into a LaserScan
------------------------------------------
Each RPLiDAR revolution yields measurements at irregular angles, not the
evenly-spaced grid ``LaserScan`` assumes (``angle_min + i *
angle_increment``, matching how every other part of this project reads
scans). Each measurement is binned into the nearest angular slot at
``angular_resolution``; a bin with no measurement reports ``range_max``
-- the same "nothing detected here" convention ``SimulatedLidar`` uses,
so mapping code needs no special-casing per sensor source.

Angle direction: the RPLiDAR reports angles increasing CLOCKWISE seen from
above; this project's angles are counter-clockwise (+x = robot FRONT, +y =
robot LEFT). With ``RPLidarHardwareConfig.angles_clockwise`` (config/
hardware.yaml, default true) every raw angle is negated before binning.
Confirmed on the robot 2026-09-29 with ``operator_gui.py --orientation-check``:
without it, objects on the robot's left appeared on its right.
"""

from __future__ import annotations

import fcntl
import math
import struct
import time
from dataclasses import dataclass
from typing import Iterator

import serial
from rplidar import RPLidar as _RPLidarBackend

from common.config import LidarConfig, RPLidarHardwareConfig
from common.logger import get_logger
from common.messages import LaserScan

logger = get_logger(__name__)

# --- Linux termios2 custom-baud-rate workaround (see module docstring) ---
_TCGETS2 = 0x802C542A
_TCSETS2 = 0x402C542B
_BOTHER = 0o010000
_CBAUD = 0o010017
_TERMIOS2_FMT = "=4I1B19B2I"  # iflag,oflag,cflag,lflag; line; cc[19]; ispeed,ospeed

_STANDARD_BAUDRATES = {
    50, 75, 110, 134, 150, 200, 300, 600, 1200, 1800, 2400, 4800, 9600,
    19200, 38400, 57600, 115200, 230400, 460800, 500000, 576000, 921600,
    1000000, 1152000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000,
}


def _force_custom_baudrate(serial_port: serial.Serial, baudrate: int) -> None:
    """Override only the speed fields of an already-open port via termios2.

    Leaves every other flag pyserial already configured (parity, stop
    bits, raw mode, CLOCAL/CREAD, ...) untouched.
    """
    fd = serial_port.fileno()
    buf = fcntl.ioctl(fd, _TCGETS2, b"\0" * struct.calcsize(_TERMIOS2_FMT))
    fields = list(struct.unpack(_TERMIOS2_FMT, buf))
    fields[2] = (fields[2] & ~_CBAUD) | _BOTHER  # c_cflag
    fields[-2] = baudrate  # c_ispeed
    fields[-1] = baudrate  # c_ospeed
    fcntl.ioctl(fd, _TCSETS2, struct.pack(_TERMIOS2_FMT, *fields))


class RPLidarDriver:
    """Streams ``LaserScan`` messages from a real RPLiDAR over serial."""

    def __init__(self, hardware_config: RPLidarHardwareConfig, lidar_config: LidarConfig) -> None:
        self.hardware_config = hardware_config
        self.lidar_config = lidar_config
        self._backend: _RPLidarBackend | None = None

        span = lidar_config.angle_max - lidar_config.angle_min
        if span <= 0.0:
            raise ValueError("lidar_config.angle_max must be > angle_min")
        self._num_bins = int(round(span / lidar_config.angular_resolution)) + 1

    def connect(self) -> None:
        """Open the serial port, apply the real baud rate, and start the motor."""
        if self._backend is not None:
            return

        port = self.hardware_config.port
        baudrate = self.hardware_config.baudrate
        timeout = self.hardware_config.timeout

        backend = _RPLidarBackend.__new__(_RPLidarBackend)  # bypass __init__'s auto-connect
        backend.port = port
        backend.baudrate = baudrate
        backend.timeout = timeout
        backend.motor_running = None
        backend.logger = logger
        backend._serial_port = None

        placeholder_baud = baudrate if baudrate in _STANDARD_BAUDRATES else 115200
        try:
            backend._serial_port = serial.Serial(
                port, placeholder_baud,
                parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=timeout,
            )
        except serial.SerialException as err:
            raise ConnectionError(f"Failed to open {port}: {err}") from err

        if baudrate not in _STANDARD_BAUDRATES:
            _force_custom_baudrate(backend._serial_port, baudrate)

        backend.start_motor()
        time.sleep(0.3)  # let the motor spin up before the first command
        self._backend = backend
        logger.info("Connected to RPLiDAR on %s at %d baud", port, baudrate)

    def disconnect(self) -> None:
        if self._backend is None:
            return
        try:
            self._backend.stop()
        finally:
            try:
                self._backend.stop_motor()
            finally:
                self._backend.disconnect()
                self._backend = None
                logger.info("Disconnected from RPLiDAR")

    def __enter__(self) -> "RPLidarDriver":
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.disconnect()

    def get_device_info(self) -> dict:
        self._require_connected()
        return self._backend.get_info()

    def get_health(self) -> tuple[str, int]:
        self._require_connected()
        return self._backend.get_health()

    def iter_scans(self) -> Iterator[LaserScan]:
        """Yield one ``LaserScan`` per revolution until the caller stops iterating."""
        self._require_connected()
        for raw_scan in self._backend.iter_scans(
            min_len=self.hardware_config.min_measurements_per_scan
        ):
            yield self._bin_scan(raw_scan, timestamp=time.time())

    def _bin_scan(self, raw_scan: list[tuple[float, float, float]], timestamp: float) -> LaserScan:
        cfg = self.lidar_config
        ranges = [cfg.range_max] * self._num_bins

        for _quality, angle_deg, distance_mm in raw_scan:
            distance_m = distance_mm / 1000.0
            if distance_m < cfg.range_min or distance_m > cfg.range_max:
                continue  # invalid or out-of-range; leave the bin at range_max

            angle_rad = math.radians(angle_deg)  # RPLiDAR reports [0, 360) deg, clockwise
            if self.hardware_config.angles_clockwise:
                angle_rad = -angle_rad  # -> counter-clockwise (+y = robot LEFT), this project's convention
            if angle_rad > math.pi:
                angle_rad -= 2.0 * math.pi  # fold into (-pi, pi] to match angle_min/angle_max
            elif angle_rad <= -math.pi:
                angle_rad += 2.0 * math.pi

            bin_index = round((angle_rad - cfg.angle_min) / cfg.angular_resolution)
            if 0 <= bin_index < self._num_bins and distance_m < ranges[bin_index]:
                ranges[bin_index] = distance_m  # keep the nearest hit if multiple land in one bin

        return LaserScan(
            timestamp=timestamp,
            angle_min=cfg.angle_min,
            angle_increment=cfg.angular_resolution,
            ranges=ranges,
            range_min=cfg.range_min,
            range_max=cfg.range_max,
        )

    def _require_connected(self) -> None:
        if self._backend is None:
            raise RuntimeError("RPLidarDriver is not connected; call connect() first")
