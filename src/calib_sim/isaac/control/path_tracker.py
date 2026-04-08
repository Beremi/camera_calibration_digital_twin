"""Anchored-frame waypoint path tracker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calib_sim.isaac.control.safety_gates import SafetyGates
from calib_sim.isaac.control.waypoint_manager import WaypointManager


@dataclass(slots=True)
class PathTracker:
    waypoint_manager: WaypointManager
    safety_gates: SafetyGates
    proportional_gain: float = 1.0
    max_command_abs: float = 0.25
    anchor_lost_steps: int = 0

    def command(
        self,
        *,
        current_position_world_m: np.ndarray,
        position_radius_95_m: float,
        innovation_norm: float,
        anchor_visible: bool,
    ) -> tuple[float, float, float]:
        if not anchor_visible:
            self.anchor_lost_steps += 1
        else:
            self.anchor_lost_steps = 0
        decision = self.safety_gates.evaluate(
            position_radius_95_m=position_radius_95_m,
            innovation_norm=innovation_norm,
            anchor_visible=anchor_visible,
            anchor_lost_steps=self.anchor_lost_steps,
        )
        waypoint = self.waypoint_manager.current_waypoint()
        if waypoint is None or not decision.allow_motion:
            return (0.0, 0.0, 0.0)
        error = np.asarray(waypoint.position_world_m, dtype=np.float64) - np.asarray(current_position_world_m, dtype=np.float64)
        raw = decision.gain_scale * float(self.proportional_gain) * error
        clamped = np.clip(raw, -self.max_command_abs, self.max_command_abs)
        return float(clamped[0]), float(clamped[1]), float(clamped[2])
