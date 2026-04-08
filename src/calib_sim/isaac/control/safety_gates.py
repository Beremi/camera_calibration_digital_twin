"""Uncertainty- and visibility-aware control gates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GateDecision:
    allow_motion: bool
    reason: str
    gain_scale: float


@dataclass(slots=True)
class SafetyGates:
    max_position_radius_95_m: float = 0.10
    max_innovation_norm: float = 0.25
    max_anchor_lost_steps: int = 5

    def evaluate(
        self,
        *,
        position_radius_95_m: float,
        innovation_norm: float,
        anchor_visible: bool,
        anchor_lost_steps: int,
        saturation_fraction: float = 0.0,
    ) -> GateDecision:
        if not anchor_visible and anchor_lost_steps > self.max_anchor_lost_steps:
            return GateDecision(False, "anchor_lost", 0.0)
        if position_radius_95_m > self.max_position_radius_95_m:
            return GateDecision(False, "covariance_high", 0.0)
        if innovation_norm > self.max_innovation_norm:
            return GateDecision(False, "innovation_spike", 0.0)
        if saturation_fraction > 0.5:
            return GateDecision(True, "saturation_guard", 0.5)
        return GateDecision(True, "nominal", 1.0)
