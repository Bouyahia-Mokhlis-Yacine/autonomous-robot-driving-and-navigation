"""Typed loader for config/runtime.yaml (see that file for every field's meaning)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

_DEFAULT = Path(__file__).resolve().parent.parent / "config" / "runtime.yaml"

OPERATOR_LOSS_POLICIES = ("stop", "continue")


@dataclass
class NetworkSettings:
    host: str = "0.0.0.0"
    port: int = 8765
    state_rate_hz: float = 10.0
    map_rate_hz: float = 1.0
    send_timeout_s: float = 1.0


@dataclass
class SafetySettings:
    operator_timeout_s: float = 1.5
    operator_loss_policy: str = "stop"
    max_command_state_age_s: float = 2.0
    start_stopped: bool = False

    def __post_init__(self) -> None:
        if self.operator_loss_policy not in OPERATOR_LOSS_POLICIES:
            raise ValueError(f"operator_loss_policy must be one of {OPERATOR_LOSS_POLICIES}")
        if self.operator_timeout_s <= 0 or self.max_command_state_age_s <= 0:
            raise ValueError("timeouts must be > 0")


@dataclass
class ControlSettings:
    rate_hz: float = 10.0


@dataclass
class LiveSettings:
    map_size_m: float = 20.0
    max_range_m: float = 0.0
    forget_after_s: float = 0.0
    inflation_scale: float = 1.0
    inflation_radius_m: float = 0.0  # > 0: absolute obstacle inflation (m), overrides inflation_scale
    motors: str = "none"         # none | gpio
    motor_guard: bool = True     # separate process forcing the pins low if the runtime dies
    motors_dry_run: bool = False # gpio backend without touching GPIO (logic check)
    async_planning: bool = True  # plan in a separate process, never inside the control tick
    planning_timeout_s: float = 2.0  # a plan request not answered within this -> stop, BLOCKED

    def __post_init__(self) -> None:
        if self.motors not in ("none", "gpio"):
            raise ValueError("live.motors must be 'none' or 'gpio'")
        if self.inflation_radius_m < 0.0 or self.inflation_scale <= 0.0:
            raise ValueError("inflation_radius_m must be >= 0 and inflation_scale > 0")


@dataclass
class RuntimeConfig:
    network: NetworkSettings = field(default_factory=NetworkSettings)
    safety: SafetySettings = field(default_factory=SafetySettings)
    control: ControlSettings = field(default_factory=ControlSettings)
    live: LiveSettings = field(default_factory=LiveSettings)


def _section(cls, data: dict | None):
    data = data or {}
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_runtime_config(path: Path | str | None = None) -> RuntimeConfig:
    resolved = Path(path) if path is not None else _DEFAULT
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return RuntimeConfig(
        network=_section(NetworkSettings, data.get("network")),
        safety=_section(SafetySettings, data.get("safety")),
        control=_section(ControlSettings, data.get("control")),
        live=_section(LiveSettings, data.get("live")),
    )
