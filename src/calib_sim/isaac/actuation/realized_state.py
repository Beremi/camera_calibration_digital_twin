"""Realized actuator state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RealizedJointState:
    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    internal_state: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "joint_names": list(self.joint_names),
            "positions": [float(value) for value in self.positions],
            "velocities": [float(value) for value in self.velocities],
            "internal_state": dict(self.internal_state),
        }
