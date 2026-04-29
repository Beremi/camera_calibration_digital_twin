"""Anchored-frame waypoint sequencing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Waypoint:
    position_world_m: tuple[float, float, float]
    tolerance_m: float = 0.02


@dataclass(slots=True)
class WaypointManager:
    waypoints: tuple[Waypoint, ...]
    current_index: int = 0
    reached_indices: list[int] = field(default_factory=list)

    def current_waypoint(self) -> Waypoint | None:
        if self.current_index >= len(self.waypoints):
            return None
        return self.waypoints[self.current_index]

    def update(self, current_position_world_m: np.ndarray) -> bool:
        waypoint = self.current_waypoint()
        if waypoint is None:
            return True
        error = np.linalg.norm(np.asarray(current_position_world_m, dtype=np.float64).reshape(3) - np.asarray(waypoint.position_world_m, dtype=np.float64))
        if error <= float(waypoint.tolerance_m):
            self.reached_indices.append(self.current_index)
            self.current_index += 1
        return self.current_index >= len(self.waypoints)
