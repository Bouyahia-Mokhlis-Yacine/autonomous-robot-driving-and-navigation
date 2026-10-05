"""Closed-loop integration of the realistic sensor model
(``ClosedLoopSimulation(sensor_simulation=...)``): ideal mode is unchanged,
realistic mode is asynchronous with observable latency, and runs are
deterministic for a seed. Short simulated durations keep these fast --
realistic mode only integrates ~4-5 scans/s, so it is cheaper than ideal."""

from __future__ import annotations

import dataclasses

import pytest

from common.config import load_navigation_config, load_robot_config
from common.messages import VelocityCommand
from simulation.closed_loop import ClosedLoopSimulation
from simulation.scenario import default_scenario
from simulation.sensor_simulation_config import (
    IDEAL,
    REALISTIC,
    SensorSimulationConfig,
    load_sensor_simulation_config,
)

ROBOT_CFG = load_robot_config()
NAV_CFG = load_navigation_config()


def make_sim(sensor: SensorSimulationConfig | None, duration: float, drive_script=None,
             nav_cfg=NAV_CFG) -> ClosedLoopSimulation:
    scenario = default_scenario(ROBOT_CFG.robot.width, ROBOT_CFG.robot.length)
    return ClosedLoopSimulation(
        environment=scenario.environment,
        ground_truth_start=scenario.ground_truth_start,
        goal=scenario.slam_frame_goal,
        nav_cfg=nav_cfg,
        robot_cfg=ROBOT_CFG,
        max_simulation_time=duration,
        sensor_simulation=sensor,
        drive_script=drive_script,
    )


def test_config_file_loads_both_modes() -> None:
    cfg = load_sensor_simulation_config()
    assert cfg.mode in (IDEAL, REALISTIC)
    assert load_sensor_simulation_config(mode=REALISTIC, seed=7).seed == 7
    realistic = load_sensor_simulation_config(mode=REALISTIC)
    assert realistic.is_realistic
    assert realistic.lidar.scan_rate_hz == pytest.approx(11.05)
    assert 0.0 < realistic.processing.delay_min_s <= realistic.processing.delay_max_s
    with pytest.raises(ValueError):
        SensorSimulationConfig(mode="perfect")


def test_ideal_mode_is_exactly_the_original_simulation() -> None:
    original = make_sim(None, 2.0).run()
    ideal = make_sim(load_sensor_simulation_config(mode=IDEAL), 2.0).run()
    assert dataclasses.asdict(original) == dataclasses.asdict(ideal)


def test_realistic_mode_processes_asynchronously_with_latency() -> None:
    sensor = load_sensor_simulation_config(mode=REALISTIC, seed=42)
    sim = make_sim(sensor, 4.0)
    metrics = sim.run()
    report = sim.sensor_report()

    assert report["scans_received"] > 35                # ~11 Hz over 4 s
    # Measured processing (~12-24 ms) keeps up with the sensor: nothing dropped.
    assert report["scans_processed"] >= report["scans_received"] - 1
    assert report["scans_dropped"] == 0
    assert sensor.processing.delay_min_s <= report["mean_processing_time_s"] <= sensor.processing.delay_max_s
    assert report["average_processing_latency_s"] >= sensor.processing.delay_min_s
    assert report["mean_acquisition_duration_s"] == pytest.approx(0.0905, abs=0.003)
    assert report["mean_scan_age_at_use_s"] > report["average_processing_latency_s"]
    assert report["scans"]["scan_frequency_hz"] == pytest.approx(11.0, abs=0.6)
    # Bootstrap is only known once a scan's processing has COMPLETED.
    assert metrics.bootstrap_completed
    assert metrics.bootstrap_duration > sim.sensor_log.delivered_scans[0].timestamp
    assert sim.last_completed.finish_time > sim.last_completed.scan.timestamp


def _nav_cfg(delay_compensation: bool):
    return dataclasses.replace(NAV_CFG, localization=dataclasses.replace(
        NAV_CFG.localization, delay_compensation=delay_compensation))


def test_without_compensation_navigation_only_sees_a_new_pose_when_processing_completes() -> None:
    sim = make_sim(load_sensor_simulation_config(mode=REALISTIC, seed=1), 3.0,
                   drive_script=[(5.0, VelocityCommand(linear=0.2, angular=0.0))], nav_cfg=_nav_cfg(False))
    pose_changes = []
    last = sim.nav_robot.get_pose()
    for _ in range(30):
        sim.step()
        now = sim.nav_robot.get_pose()
        if now != last:
            pose_changes.append(sim._sim_time)
            assert sim.last_completed is not None
            assert sim.last_completed.finish_time <= sim._sim_time + 1e-9
            assert sim._sim_time - sim.last_completed.finish_time < sim.dt + 1e-9
        last = now
    assert pose_changes


def test_with_compensation_the_pose_is_propagated_between_completions() -> None:
    """Every tick the navigation pose = last accepted SLAM pose moved forward,
    with the commanded motion, from its scan's capture time to now."""
    from simulation.closed_loop import scan_pose_time

    sim = make_sim(load_sensor_simulation_config(mode=REALISTIC, seed=1), 4.0,
                   drive_script=[(5.0, VelocityCommand(linear=0.2, angular=0.0))], nav_cfg=_nav_cfg(True))
    moved_without_new_scan = 0
    last_pose, last_completed = sim.nav_robot.get_pose(), None
    while sim._sim_time < 4.0 - 1e-9:
        sim.step()
        pose = sim.nav_robot.get_pose()
        if sim._script_start_time is not None and sim.last_completed is last_completed and pose != last_pose:
            moved_without_new_scan += 1
        if sim._pose_anchor is not None:
            anchor_pose, anchor_time = sim._pose_anchor
            expected = sim.command_history.propagate(anchor_pose, anchor_time, sim._sim_time)
            assert (pose.x, pose.y, pose.theta) == pytest.approx((expected.x, expected.y, expected.theta))
            if sim.last_completed is not None and sim.last_completed.result.accepted:
                assert anchor_time == pytest.approx(scan_pose_time(sim.last_completed.scan))
        last_pose, last_completed = pose, sim.last_completed
    assert moved_without_new_scan > 0


def test_overloaded_processing_drops_scans_and_stays_controlled() -> None:
    """Test F in miniature: SLAM slower than the sensor (0.4-0.6 s per scan)."""
    from simulation.sensor_processing import ProcessingConfig

    sensor = dataclasses.replace(load_sensor_simulation_config(mode=REALISTIC, seed=42),
                                 processing=ProcessingConfig(delay_min_s=0.4, delay_max_s=0.6))
    sim = make_sim(sensor, 6.0)
    sim.run()
    report = sim.sensor_report()
    assert report["scans_dropped"] > report["scans_processed"]  # most scans cannot be processed
    assert report["max_queue_size"] == 1                        # latest_only never lets a backlog form
    assert 0.4 <= report["mean_processing_time_s"] <= 0.6
    assert sim.metrics.max_position_error < 0.3                 # delay compensation keeps the pose usable
    assert not sim.metrics.collision


def test_realistic_runs_are_deterministic_for_a_seed() -> None:
    def once(seed: int):
        sim = make_sim(load_sensor_simulation_config(mode=REALISTIC, seed=seed), 3.0)
        metrics = sim.run()
        return dataclasses.asdict(metrics), [s.ranges for s in sim.sensor_log.delivered_scans], \
            [s.timestamp for s in sim.sensor_log.delivered_scans]

    a, b, c = once(42), once(42), once(43)
    assert a == b
    assert a[2] != c[2]


def test_drive_script_gives_both_modes_the_same_trajectory() -> None:
    script = [(1.0, VelocityCommand(linear=0.2, angular=0.0)), (1.0, VelocityCommand(linear=0.0, angular=0.5))]
    ideal = make_sim(load_sensor_simulation_config(mode=IDEAL), 4.0, drive_script=script)
    realistic = make_sim(load_sensor_simulation_config(mode=REALISTIC, seed=3), 4.0, drive_script=script)
    ideal.run()
    realistic.run()
    # Script starts at each mode's own bootstrap completion, so compare the
    # displacement it produced, not absolute time.
    assert ideal.metrics.ground_truth_distance == pytest.approx(0.2, abs=0.03)
    assert realistic.metrics.ground_truth_distance == pytest.approx(0.2, abs=0.03)
    assert ideal.nav_manager.get_goal() is None and realistic.nav_manager.get_goal() is None
