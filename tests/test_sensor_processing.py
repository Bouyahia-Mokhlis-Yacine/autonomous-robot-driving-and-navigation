"""Tests for ``simulation.sensor_processing`` (asynchronous scan arrival ->
SLAM processing, in simulated time): latency, queue/drop policies, event
ordering, and determinism."""

from __future__ import annotations

import pytest

from common.messages import LaserScan
from simulation.sensor_processing import (
    ProcessingConfig,
    ProcessingDelayModel,
    QueuePolicy,
    ScanProcessingSimulator,
)


def scan_at(t: float) -> LaserScan:
    return LaserScan(timestamp=t, angle_min=0.0, angle_increment=0.1, ranges=[1.0, 1.0], range_min=0.1, range_max=5.0)


def run(config: ProcessingConfig, arrivals: list[float], until: float, seed: int = 0):
    """Submit scans arriving at ``arrivals``; advance in 10 ms ticks. Returns
    the simulator, the order scans were processed in, and the completions."""
    sim = ScanProcessingSimulator(config, seed=seed)
    for t in arrivals:
        sim.submit(scan_at(t))
    processed: list[float] = []
    completed = []
    t = 0.0
    while t < until:
        t = round(t + 0.01, 10)
        completed += sim.advance(t, lambda s: processed.append(s.timestamp) or s.timestamp)
    return sim, processed, completed


FAST_SENSOR = [round(0.09 * k, 10) for k in range(1, 23)]  # ~11 Hz for 2 s


# 0.205 s never coincides with a 0.09 s arrival grid, so no timeline below
# depends on how exact ties are broken.
def fixed(policy: QueuePolicy, delay: float = 0.205, capacity: int = 1) -> ProcessingConfig:
    return ProcessingConfig(ProcessingDelayModel.UNIFORM, delay, delay, policy, capacity)


# ----------------------------------------------------------------- latency ---

def test_arrival_and_completion_are_different_times_and_follow_the_model() -> None:
    cfg = ProcessingConfig(ProcessingDelayModel.UNIFORM, 0.15, 0.25, QueuePolicy.LATEST_ONLY, 1)
    sim, _, completed = run(cfg, FAST_SENSOR, until=3.0, seed=1)
    assert completed
    for c in completed:
        assert c.finish_time > c.arrival_time
        assert 0.15 <= c.finish_time - c.start_time <= 0.25
        assert c.start_time >= c.arrival_time
        assert c.result == c.scan.timestamp
    times = [c.finish_time - c.start_time for c in completed]
    assert max(times) - min(times) > 0.02  # really drawn from a range, not constant
    assert sim.metrics.average_processing_latency >= 0.15


def test_results_become_available_only_at_the_completion_instant() -> None:
    sim = ScanProcessingSimulator(fixed(QueuePolicy.QUEUE_ALL, delay=0.25))
    sim.submit(scan_at(0.25))
    assert sim.advance(0.49, lambda s: s.timestamp) == []   # arrived, still processing
    assert sim.busy and sim.in_flight_finish_time == 0.5
    done = sim.advance(0.5, lambda s: s.timestamp)
    assert [(c.arrival_time, c.finish_time) for c in done] == [(0.25, 0.5)]


def test_instantaneous_config_reproduces_synchronous_processing() -> None:
    sim, processed, completed = run(ProcessingConfig.instantaneous(), FAST_SENSOR, until=3.0)
    assert processed == FAST_SENSOR
    assert all(c.finish_time == c.arrival_time for c in completed)
    assert sim.metrics.scans_dropped == 0


def test_processing_is_serial_and_in_order() -> None:
    sim, processed, completed = run(fixed(QueuePolicy.QUEUE_ALL), FAST_SENSOR, until=10.0)
    assert processed == sorted(processed)
    for a, b in zip(completed, completed[1:]):
        assert b.start_time >= a.finish_time  # one scan at a time


# ------------------------------------------------------------ queue policy ---

def conservation(sim: ScanProcessingSimulator) -> None:
    m = sim.metrics
    in_flight = 1 if sim.busy else 0
    assert m.scans_received == m.scans_processed + m.scans_dropped + sim.queue_size + in_flight


def test_queue_all_never_drops_and_the_queue_grows_when_sensor_outpaces_processing() -> None:
    sim, processed, completed = run(fixed(QueuePolicy.QUEUE_ALL), FAST_SENSOR, until=2.0)
    assert sim.metrics.scans_dropped == 0
    assert sim.metrics.max_queue_size > 5
    assert completed[-1].queue_wait > completed[0].queue_wait  # latency grows without bound
    conservation(sim)


def test_latest_only_always_processes_the_newest_waiting_scan() -> None:
    sim, processed, completed = run(fixed(QueuePolicy.LATEST_ONLY), FAST_SENSOR, until=3.0)
    assert sim.metrics.max_queue_size == 1
    assert sim.metrics.scans_dropped > 0
    for c in completed[1:]:
        # Whatever was picked up is the most recent scan that had arrived by then.
        newest_arrived = max(t for t in FAST_SENSOR if t <= c.start_time + 1e-12)
        assert c.scan.timestamp == newest_arrived
    conservation(sim)


def test_drop_newest_keeps_the_oldest_waiting_scans() -> None:
    sim, processed, _ = run(fixed(QueuePolicy.DROP_NEWEST, capacity=2), FAST_SENSOR, until=5.0)
    assert sim.metrics.max_queue_size == 2
    assert sim.metrics.scans_dropped > 0
    # 0.09 runs until 0.295 while 0.18/0.27 fill the queue; 0.18 starts at
    # 0.295, 0.36 refills it, so 0.45 is the first arrival to find it full.
    assert processed[:4] == [0.09, 0.18, 0.27, 0.36]
    assert 0.45 not in processed
    conservation(sim)


def test_drop_oldest_evicts_the_oldest_waiting_scan() -> None:
    sim, processed, _ = run(fixed(QueuePolicy.DROP_OLDEST, capacity=2), FAST_SENSOR, until=5.0)
    assert sim.metrics.max_queue_size == 2
    assert sim.metrics.scans_dropped > 0
    # Same timeline as above, but 0.45 evicts the oldest waiting scan (0.27).
    assert processed[:3] == [0.09, 0.18, 0.36]
    assert 0.27 not in processed
    assert processed[-1] == FAST_SENSOR[-1]
    conservation(sim)


def test_policy_and_model_accept_their_yaml_string_names() -> None:
    cfg = ProcessingConfig(delay_model="uniform", queue_policy="drop_oldest", queue_capacity=3)
    assert cfg.delay_model is ProcessingDelayModel.UNIFORM
    assert cfg.queue_policy is QueuePolicy.DROP_OLDEST
    with pytest.raises(ValueError):
        ProcessingConfig(queue_policy="newest_wins")
    with pytest.raises(ValueError):
        ProcessingConfig(delay_min_s=0.3, delay_max_s=0.1)


def test_out_of_order_submission_is_rejected() -> None:
    sim = ScanProcessingSimulator()
    sim.submit(scan_at(0.2))
    with pytest.raises(ValueError):
        sim.submit(scan_at(0.1))


def test_same_seed_same_delays() -> None:
    cfg = ProcessingConfig()
    a = run(cfg, FAST_SENSOR, until=3.0, seed=42)[2]
    b = run(cfg, FAST_SENSOR, until=3.0, seed=42)[2]
    c = run(cfg, FAST_SENSOR, until=3.0, seed=7)[2]
    assert [x.finish_time for x in a] == [x.finish_time for x in b]
    assert [x.finish_time for x in a] != [x.finish_time for x in c]
