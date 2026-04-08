"""State and snapshot containers for Isaac estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _matrix_json(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix, dtype=np.float64)]


@dataclass(slots=True)
class MotionState:
    rotation_wi: np.ndarray
    position_world_m: np.ndarray
    velocity_world_mps: np.ndarray
    gyro_bias_rps: np.ndarray
    accel_bias_mps2: np.ndarray


@dataclass(slots=True)
class FilterStateSnapshot:
    timestamp_s: float
    sim_time_s: float
    rotation_wi: np.ndarray
    position_world_m: np.ndarray
    velocity_world_mps: np.ndarray
    gyro_bias_rps: np.ndarray
    accel_bias_mps2: np.ndarray
    covariance: np.ndarray
    innovation_diagnostics: dict[str, float] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "rotation_wi": _matrix_json(self.rotation_wi),
            "position_world_m": [float(value) for value in np.asarray(self.position_world_m, dtype=np.float64)],
            "velocity_world_mps": [float(value) for value in np.asarray(self.velocity_world_mps, dtype=np.float64)],
            "gyro_bias_rps": [float(value) for value in np.asarray(self.gyro_bias_rps, dtype=np.float64)],
            "accel_bias_mps2": [float(value) for value in np.asarray(self.accel_bias_mps2, dtype=np.float64)],
            "covariance": _matrix_json(self.covariance),
            "innovation_diagnostics": {str(key): float(value) for key, value in self.innovation_diagnostics.items()},
        }


@dataclass(slots=True)
class SmootherStateSnapshot:
    timestamp_s: float
    sim_time_s: float
    active_tag_poses: dict[int, np.ndarray]
    cloned_positions_world_m: tuple[np.ndarray, ...]
    covariance: np.ndarray

    def as_json(self) -> dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "active_tag_poses": {str(tag_id): _matrix_json(pose) for tag_id, pose in self.active_tag_poses.items()},
            "cloned_positions_world_m": [
                [float(value) for value in np.asarray(position, dtype=np.float64)]
                for position in self.cloned_positions_world_m
            ],
            "covariance": _matrix_json(self.covariance),
        }


@dataclass(slots=True)
class UncertaintySnapshot:
    timestamp_s: float
    sim_time_s: float
    pose_covariance: np.ndarray
    velocity_covariance: np.ndarray
    bias_covariance: np.ndarray
    position_radius_95_m: float
    velocity_radius_95_mps: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "sim_time_s": float(self.sim_time_s),
            "pose_covariance": _matrix_json(self.pose_covariance),
            "velocity_covariance": _matrix_json(self.velocity_covariance),
            "bias_covariance": _matrix_json(self.bias_covariance),
            "position_radius_95_m": float(self.position_radius_95_m),
            "velocity_radius_95_mps": float(self.velocity_radius_95_mps),
            "diagnostics": {str(key): float(value) for key, value in self.diagnostics.items()},
        }
