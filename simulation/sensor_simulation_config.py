"""Loader for ``config/lidar_simulation.yaml`` (ideal vs. realistic sensor).

Lives in ``simulation`` rather than ``common.config`` because the typed
result is built from simulation-only classes (``RealisticLidarConfig``,
``ProcessingConfig``), and ``common`` must not depend on ``simulation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from simulation.realistic_lidar import RealisticLidarConfig
from simulation.sensor_processing import ProcessingConfig

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "lidar_simulation.yaml"

IDEAL = "ideal"
REALISTIC = "realistic"


@dataclass
class SensorSimulationConfig:
    mode: str = REALISTIC
    seed: int | None = 42
    physics_dt: float = 0.01
    lidar: RealisticLidarConfig = field(default_factory=RealisticLidarConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    def __post_init__(self) -> None:
        if self.mode not in (IDEAL, REALISTIC):
            raise ValueError(f"mode must be '{IDEAL}' or '{REALISTIC}', got {self.mode!r}")
        if self.physics_dt <= 0.0:
            raise ValueError("physics_dt must be > 0")

    @property
    def is_realistic(self) -> bool:
        return self.mode == REALISTIC


def load_sensor_simulation_config(
    path: Path | str | None = None, mode: str | None = None, seed: int | None | str = "from_file"
) -> SensorSimulationConfig:
    """``mode``/``seed`` override the file's values when given (``seed=None``
    explicitly means nondeterministic)."""
    resolved = Path(path) if path is not None else _DEFAULT_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    realistic = data.get("realistic", {})
    return SensorSimulationConfig(
        mode=mode if mode is not None else data.get("mode", REALISTIC),
        seed=data.get("seed") if seed == "from_file" else seed,
        physics_dt=float(data.get("physics_dt", 0.01)),
        lidar=RealisticLidarConfig(**realistic.get("sensor", {})),
        processing=ProcessingConfig(**realistic.get("processing", {})),
    )
