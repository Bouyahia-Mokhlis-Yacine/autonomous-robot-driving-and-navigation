"""Top-level simulation clock/loop driving an ``Environment``.

Phase 1 scope: advance dynamic obstacles over time. Later phases attach
sensor sampling (simulated LiDAR, Phase 2) at each tick without this
module needing to know about planning or control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simulation.environment import Environment


@dataclass
class Simulator:
    environment: Environment
    time: float = field(default=0.0, init=False)

    def step(self, dt: float) -> None:
        """Advance the simulation by ``dt`` seconds."""
        self.environment.step(dt)
        self.time += dt

    def run(self, duration: float, dt: float) -> None:
        """Step the simulation forward for ``duration`` seconds in increments of ``dt``."""
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        steps = int(round(duration / dt))
        for _ in range(steps):
            self.step(dt)
