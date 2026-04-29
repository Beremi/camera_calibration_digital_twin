"""Uncertain servo model for simulation corruption and estimator mismatch."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(slots=True)
class ServoCorruptionConfig:
    delay_s: float = 0.0
    gain_error: float = 1.0
    bias: float = 0.0
    lag_time_constant_s: float = 0.05
    rate_limit_per_s: float = 4.0
    deadband: float = 0.0
    backlash: float = 0.0
    process_noise_std: float = 0.0
    drop_probability: float = 0.0
    jitter_std_s: float = 0.0


@dataclass(slots=True)
class ServoStepResult:
    commanded: float
    effective_command: float
    target: float
    realized_position: float
    realized_velocity: float
    dropped: bool


class UncertainServoModel:
    """Small but testable corruption model matching the implementation playbook."""

    def __init__(
        self,
        config: ServoCorruptionConfig,
        *,
        initial_position: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.position = float(initial_position)
        self.velocity = 0.0
        self._delay_buffer: deque[tuple[float, float]] = deque()
        self._backlash_sign = 0.0

    def _deadband(self, value: float) -> float:
        if abs(value) <= float(self.config.deadband):
            return 0.0
        return float(value)

    def _backlash_term(self, commanded_delta: float) -> float:
        if abs(commanded_delta) < 1e-12:
            return self._backlash_sign * float(self.config.backlash)
        self._backlash_sign = math.copysign(1.0, commanded_delta)
        return self._backlash_sign * float(self.config.backlash)

    def step(self, *, command: float, dt_s: float) -> ServoStepResult:
        dt_s = max(float(dt_s), 1e-9)
        effective_time = max(
            0.0,
            dt_s + (0.0 if self.config.jitter_std_s <= 0.0 else float(self.rng.normal(scale=self.config.jitter_std_s))),
        )
        self._delay_buffer.append((float(self.config.delay_s) + effective_time, float(command)))
        effective_command = float(command)
        carried = deque()
        while self._delay_buffer:
            remaining_delay_s, candidate_command = self._delay_buffer.popleft()
            remaining_delay_s -= dt_s
            if remaining_delay_s <= 0.0:
                effective_command = candidate_command
            else:
                carried.append((remaining_delay_s, candidate_command))
        self._delay_buffer = carried
        dropped = bool(self.rng.random() < float(self.config.drop_probability))
        if dropped:
            effective_command = self.position
        target = self._deadband(effective_command)
        target = float(self.config.gain_error) * target + float(self.config.bias) + self._backlash_term(target - self.position)
        desired_velocity = (target - self.position) / max(float(self.config.lag_time_constant_s), 1e-6)
        desired_velocity = float(np.clip(desired_velocity, -self.config.rate_limit_per_s, self.config.rate_limit_per_s))
        process_noise = 0.0 if self.config.process_noise_std <= 0.0 else float(self.rng.normal(scale=self.config.process_noise_std))
        self.velocity = desired_velocity + process_noise
        self.position = float(self.position + self.velocity * dt_s)
        return ServoStepResult(
            commanded=float(command),
            effective_command=float(effective_command),
            target=float(target),
            realized_position=float(self.position),
            realized_velocity=float(self.velocity),
            dropped=dropped,
        )
