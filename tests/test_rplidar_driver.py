"""Phase 12 tests: RPLidarDriver's pure logic (scan binning, connection
guards). The actual serial connection is hardware-dependent and is
verified separately against a real connected RPLiDAR, not here."""

from __future__ import annotations

import math

import pytest

from common.config import LidarConfig, RPLidarHardwareConfig, load_hardware_config
from hardware.rplidar_driver import RPLidarDriver, _STANDARD_BAUDRATES


def make_driver(angles_clockwise: bool = False, **lidar_overrides) -> RPLidarDriver:
    hardware_config = RPLidarHardwareConfig(
        port="/dev/ttyUSB0", baudrate=256000, timeout=3.0, min_measurements_per_scan=5,
        angles_clockwise=angles_clockwise,
    )
    lidar_defaults = dict(
        range_min=0.15, range_max=8.0, angle_min=-math.pi, angle_max=math.pi,
        angular_resolution=math.radians(1.0), scan_rate=10.0,
    )
    lidar_defaults.update(lidar_overrides)
    return RPLidarDriver(hardware_config, LidarConfig(**lidar_defaults))


def test_num_bins_computed_from_angular_span() -> None:
    driver = make_driver(angle_min=-math.pi, angle_max=math.pi, angular_resolution=math.pi / 2.0)
    assert driver._num_bins == 5  # -180,-90,0,90,180


def test_rejects_invalid_angle_span() -> None:
    with pytest.raises(ValueError):
        make_driver(angle_min=1.0, angle_max=1.0)


def test_operations_require_connection_first() -> None:
    driver = make_driver()
    with pytest.raises(RuntimeError):
        driver.get_device_info()
    with pytest.raises(RuntimeError):
        driver.get_health()
    with pytest.raises(RuntimeError):
        next(driver.iter_scans())


def test_disconnect_without_connect_is_a_safe_no_op() -> None:
    driver = make_driver()
    driver.disconnect()  # must not raise


def test_bin_scan_places_measurement_in_nearest_bin() -> None:
    driver = make_driver(angle_min=-math.pi, angle_max=math.pi, angular_resolution=math.pi / 2.0)
    # angle 0 deg, distance 1500mm -> bin index 2 (angle 0 rad)
    raw_scan = [(15.0, 0.0, 1500.0)]
    scan = driver._bin_scan(raw_scan, timestamp=1.0)
    assert scan.ranges[2] == pytest.approx(1.5)
    # untouched bins remain at range_max (no detection).
    assert scan.ranges[0] == pytest.approx(8.0)
    assert scan.ranges[1] == pytest.approx(8.0)


def test_bin_scan_folds_angles_past_180_degrees() -> None:
    driver = make_driver(angle_min=-math.pi, angle_max=math.pi, angular_resolution=math.pi / 2.0)
    # angle 270 deg == -90 deg -> bin index 1
    raw_scan = [(15.0, 270.0, 2000.0)]
    scan = driver._bin_scan(raw_scan, timestamp=1.0)
    assert scan.ranges[1] == pytest.approx(2.0)


def test_bin_scan_discards_out_of_range_measurements() -> None:
    driver = make_driver(range_min=0.15, range_max=8.0)
    too_close = [(15.0, 0.0, 50.0)]     # 0.05m < range_min
    too_far = [(15.0, 0.0, 9000.0)]     # 9.0m > range_max
    for raw in (too_close, too_far):
        scan = driver._bin_scan(raw, timestamp=0.0)
        assert all(r == pytest.approx(8.0) for r in scan.ranges)


def test_bin_scan_keeps_nearest_when_multiple_hits_share_a_bin() -> None:
    driver = make_driver(angle_min=-math.pi, angle_max=math.pi, angular_resolution=math.pi / 2.0)
    raw_scan = [(15.0, 0.5, 3000.0), (15.0, 0.2, 1000.0), (15.0, 0.1, 2000.0)]  # all fold to bin 2
    scan = driver._bin_scan(raw_scan, timestamp=0.0)
    assert scan.ranges[2] == pytest.approx(1.0)


def test_bin_scan_output_matches_laserscan_contract() -> None:
    driver = make_driver()
    scan = driver._bin_scan([], timestamp=42.0)
    assert scan.timestamp == 42.0
    assert scan.num_rays == driver._num_bins
    assert scan.angle_min == pytest.approx(-math.pi)
    assert scan.range_max == pytest.approx(8.0)


def test_standard_baudrates_include_common_rates() -> None:
    assert 115200 in _STANDARD_BAUDRATES
    assert 9600 in _STANDARD_BAUDRATES
    assert 256000 not in _STANDARD_BAUDRATES  # exactly the rate needing the ioctl workaround


def test_load_hardware_config() -> None:
    config = load_hardware_config()
    assert config.rplidar.port == "/dev/ttyUSB0"
    assert config.rplidar.baudrate == 256000
    assert config.rplidar.timeout == pytest.approx(3.0)
    assert config.rplidar.min_measurements_per_scan == 5


# ------------------------------------------ clockwise raw angles (the real sensor) --

@pytest.mark.parametrize("raw_deg, expected_deg", [
    (0.0, 0.0),        # FRONT stays FRONT
    (90.0, -90.0),     # 90 deg clockwise = robot RIGHT = -y
    (270.0, 90.0),     # 270 deg clockwise = robot LEFT = +y
    (180.0, 180.0),    # REAR: folds to +pi (last bin), as before
    (10.0, -10.0),
])
def test_clockwise_raw_angles_are_mirrored_into_the_ccw_robot_frame(raw_deg: float, expected_deg: float) -> None:
    driver = make_driver(angles_clockwise=True)
    scan = driver._bin_scan([(15.0, raw_deg, 1000.0)], timestamp=1.0)
    hits = [i for i, r in enumerate(scan.ranges) if r < scan.range_max]
    assert len(hits) == 1
    assert math.degrees(scan.angle_at(hits[0])) == pytest.approx(expected_deg, abs=1e-6)


def test_config_file_declares_the_sensor_clockwise() -> None:
    assert load_hardware_config().rplidar.angles_clockwise is True
