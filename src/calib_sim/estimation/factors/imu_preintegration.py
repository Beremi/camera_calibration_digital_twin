"""Discrete IMU integration helpers matched to repo conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from calib_sim.common.inertial import integrate_interval_constant_world_acceleration

@dataclass(slots=True)
class ImuSample:
    """One IMU packet expressed in the body frame."""

    timestamp_s: float
    dt_s: float
    accel_body_mps2: np.ndarray
    gyro_body_rps: np.ndarray


@dataclass(slots=True)
class PreintegratedImuDelta:
    """Discrete preintegration result using the repo's interval semantics."""

    delta_time_s: float
    delta_rotation_cw: np.ndarray
    delta_velocity_world_mps: np.ndarray
    delta_position_world_m: np.ndarray


def preintegrate_imu_samples(
    samples: Sequence[ImuSample],
    *,
    initial_rotation_cw: Sequence[Sequence[float]] | None = None,
    initial_velocity_world_mps: Sequence[float] | None = None,
    gravity_world_mps2: Sequence[float] = (0.0, 0.0, -9.81),
    accel_bias_mps2: Sequence[float] = (0.0, 0.0, 0.0),
    gyro_bias_rps: Sequence[float] = (0.0, 0.0, 0.0),
) -> PreintegratedImuDelta:
    start_rotation = np.eye(3, dtype=np.float64) if initial_rotation_cw is None else np.asarray(initial_rotation_cw, dtype=np.float64).reshape(3, 3)
    start_velocity = np.zeros(3, dtype=np.float64) if initial_velocity_world_mps is None else np.asarray(initial_velocity_world_mps, dtype=np.float64).reshape(3)
    integrated = integrate_interval_constant_world_acceleration(
        start_position_world_m=np.zeros(3, dtype=np.float64),
        start_rotation_wi=start_rotation,
        start_velocity_world_mps=start_velocity,
        gyro_packets_body_rps=[sample.gyro_body_rps for sample in samples],
        accel_packets_body_mps2=[sample.accel_body_mps2 for sample in samples],
        dt_packets_s=[sample.dt_s for sample in samples],
        gravity_world_mps2=gravity_world_mps2,
        accel_bias_mps2=accel_bias_mps2,
        gyro_bias_rps=gyro_bias_rps,
    )

    return PreintegratedImuDelta(
        delta_time_s=float(integrated.delta_time_s),
        delta_rotation_cw=integrated.final_rotation_wi @ start_rotation.T,
        delta_velocity_world_mps=integrated.final_velocity_world_mps - start_velocity,
        delta_position_world_m=integrated.final_position_world_m,
    )


def integrate_discrete_imu_sequence(
    samples: Sequence[ImuSample],
    *,
    initial_position_world_m: Sequence[float] = (0.0, 0.0, 0.0),
    initial_rotation_cw: Sequence[Sequence[float]] | None = None,
    initial_velocity_world_mps: Sequence[float] = (0.0, 0.0, 0.0),
    gravity_world_mps2: Sequence[float] = (0.0, 0.0, -9.81),
    accel_bias_mps2: Sequence[float] = (0.0, 0.0, 0.0),
    gyro_bias_rps: Sequence[float] = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray | float]:
    initial_position_world_m = np.asarray(initial_position_world_m, dtype=np.float64).reshape(3)
    rotation_cw = np.eye(3, dtype=np.float64) if initial_rotation_cw is None else np.asarray(initial_rotation_cw, dtype=np.float64).reshape(3, 3)
    velocity_world_mps = np.asarray(initial_velocity_world_mps, dtype=np.float64).reshape(3)
    integrated = integrate_interval_constant_world_acceleration(
        start_position_world_m=initial_position_world_m,
        start_rotation_wi=rotation_cw,
        start_velocity_world_mps=velocity_world_mps,
        gyro_packets_body_rps=[sample.gyro_body_rps for sample in samples],
        accel_packets_body_mps2=[sample.accel_body_mps2 for sample in samples],
        dt_packets_s=[sample.dt_s for sample in samples],
        gravity_world_mps2=gravity_world_mps2,
        accel_bias_mps2=accel_bias_mps2,
        gyro_bias_rps=gyro_bias_rps,
    )

    return {
        "delta_time_s": float(integrated.delta_time_s),
        "final_position_world_m": integrated.final_position_world_m,
        "final_rotation_cw": integrated.final_rotation_wi,
        "final_velocity_world_mps": integrated.final_velocity_world_mps,
    }
