"""Vectorized command interface over multiple uncertain servos."""

from __future__ import annotations

from dataclasses import dataclass

from calib_sim.isaac.actuation.realized_state import RealizedJointState
from calib_sim.isaac.actuation.servo_model import ServoCorruptionConfig, UncertainServoModel


@dataclass(slots=True)
class ServoCommandInterface:
    joint_names: tuple[str, ...]
    models: tuple[UncertainServoModel, ...]

    @classmethod
    def from_shared_config(cls, joint_names: tuple[str, ...], config: ServoCorruptionConfig) -> "ServoCommandInterface":
        return cls(joint_names=joint_names, models=tuple(UncertainServoModel(config) for _ in joint_names))

    def apply(self, *, commands: tuple[float, ...], dt_s: float) -> RealizedJointState:
        if len(commands) != len(self.models):
            raise ValueError("commands must match the number of servo models.")
        positions: list[float] = []
        velocities: list[float] = []
        internal_state: dict[str, dict[str, float | bool]] = {}
        for joint_name, model, command in zip(self.joint_names, self.models, commands):
            result = model.step(command=float(command), dt_s=float(dt_s))
            positions.append(result.realized_position)
            velocities.append(result.realized_velocity)
            internal_state[joint_name] = {
                "effective_command": result.effective_command,
                "target": result.target,
                "dropped": result.dropped,
            }
        return RealizedJointState(
            joint_names=self.joint_names,
            positions=tuple(positions),
            velocities=tuple(velocities),
            internal_state=internal_state,
        )
