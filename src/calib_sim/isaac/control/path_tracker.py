"""Anchored-frame waypoint path tracker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calib_sim.isaac.control.safety_gates import SafetyGates
from calib_sim.isaac.control.waypoint_manager import WaypointManager


@dataclass(slots=True)
class PathCommand:
    waypoint_index: int
    desired_position_world_m: tuple[float, float, float]
    tracking_error_world_m: tuple[float, float, float]
    command_delta_world_m: tuple[float, float, float]
    allow_motion: bool
    safety_reason: str


@dataclass(slots=True)
class PathTracker:
    waypoint_manager: WaypointManager
    safety_gates: SafetyGates
    proportional_gain: float = 1.0
    max_command_abs: float = 0.25
    anchor_lost_steps: int = 0

    def compute_control(
        self,
        *,
        current_position_world_m: np.ndarray,
        position_radius_95_m: float,
        innovation_norm: float,
        anchor_visible: bool,
    ) -> PathCommand:
        current_position_world_m = np.asarray(current_position_world_m, dtype=np.float64).reshape(3)
        self.waypoint_manager.update(current_position_world_m)
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
        if waypoint is None:
            return PathCommand(
                waypoint_index=int(self.waypoint_manager.current_index),
                desired_position_world_m=tuple(float(value) for value in current_position_world_m),
                tracking_error_world_m=(0.0, 0.0, 0.0),
                command_delta_world_m=(0.0, 0.0, 0.0),
                allow_motion=False,
                safety_reason="path_complete",
            )
        target = np.asarray(waypoint.position_world_m, dtype=np.float64)
        error = target - current_position_world_m
        raw = decision.gain_scale * float(self.proportional_gain) * error
        clamped = np.clip(raw, -self.max_command_abs, self.max_command_abs)
        if not decision.allow_motion:
            clamped = np.zeros(3, dtype=np.float64)
        return PathCommand(
            waypoint_index=int(self.waypoint_manager.current_index),
            desired_position_world_m=tuple(float(value) for value in target),
            tracking_error_world_m=tuple(float(value) for value in error),
            command_delta_world_m=tuple(float(value) for value in clamped),
            allow_motion=bool(decision.allow_motion),
            safety_reason=decision.reason,
        )

    def command(
        self,
        *,
        current_position_world_m: np.ndarray,
        position_radius_95_m: float,
        innovation_norm: float,
        anchor_visible: bool,
    ) -> tuple[float, float, float]:
        decision = self.compute_control(
            current_position_world_m=current_position_world_m,
            position_radius_95_m=position_radius_95_m,
            innovation_norm=innovation_norm,
            anchor_visible=anchor_visible,
        )
        return decision.command_delta_world_m


__all__ = ["PathCommand", "PathTracker"]
