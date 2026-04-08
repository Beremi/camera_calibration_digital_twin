"""Shared inertial-model helpers used by the simulator and estimators.

This module is the canonical home for the repo's IMU convention:

    tilde_f = R_IW (a_W - g_W) + b_a + n_a
    tilde_omega = omega + b_g + n_g

The helpers below intentionally keep the discrete transition semantics explicit
so the simulator truth path, factor residuals, and tests stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


def rotation_matrix_from_rvec(rvec_rad: Sequence[float]) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec_rad, dtype=np.float64).reshape(3, 1))
    return rotation_matrix.astype(np.float64)


def rvec_from_rotation_matrix(rotation_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3))
    return rvec.reshape(3).astype(np.float64)


def camera_pose_from_imu_pose(
    *,
    imu_position_world_m: Sequence[float],
    imu_rotation_wi: Sequence[Sequence[float]],
    imu_translation_camera_m: Sequence[float],
    rotation_ci: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    imu_position_world_m = np.asarray(imu_position_world_m, dtype=np.float64).reshape(3)
    imu_rotation_wi = np.asarray(imu_rotation_wi, dtype=np.float64).reshape(3, 3)
    rotation_ci = np.asarray(rotation_ci, dtype=np.float64).reshape(3, 3)
    rotation_ic = rotation_ci.T
    translation_ic = -(rotation_ic @ np.asarray(imu_translation_camera_m, dtype=np.float64).reshape(3))
    camera_position_world_m = imu_position_world_m + imu_rotation_wi @ translation_ic
    camera_rotation_wc = imu_rotation_wi @ rotation_ic
    return camera_position_world_m.astype(np.float64), camera_rotation_wc.astype(np.float64)


def imu_pose_from_camera_pose(
    *,
    camera_position_world_m: Sequence[float],
    camera_rotation_wc: Sequence[Sequence[float]],
    imu_translation_camera_m: Sequence[float],
    rotation_ci: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    camera_position_world_m = np.asarray(camera_position_world_m, dtype=np.float64).reshape(3)
    camera_rotation_wc = np.asarray(camera_rotation_wc, dtype=np.float64).reshape(3, 3)
    rotation_ci = np.asarray(rotation_ci, dtype=np.float64).reshape(3, 3)
    imu_translation_camera_m = np.asarray(imu_translation_camera_m, dtype=np.float64).reshape(3)
    imu_position_world_m = camera_position_world_m + camera_rotation_wc @ imu_translation_camera_m
    imu_rotation_wi = camera_rotation_wc @ rotation_ci
    return imu_position_world_m.astype(np.float64), imu_rotation_wi.astype(np.float64)


def accelerometer_specific_force_body(
    *,
    rotation_wi: Sequence[Sequence[float]],
    acceleration_world_mps2: Sequence[float],
    gravity_world_mps2: Sequence[float] = (0.0, 0.0, -9.81),
    accel_bias_mps2: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    rotation_wi = np.asarray(rotation_wi, dtype=np.float64).reshape(3, 3)
    acceleration_world_mps2 = np.asarray(acceleration_world_mps2, dtype=np.float64).reshape(3)
    gravity_world_mps2 = np.asarray(gravity_world_mps2, dtype=np.float64).reshape(3)
    accel_bias_mps2 = np.asarray(accel_bias_mps2, dtype=np.float64).reshape(3)
    return (rotation_wi.T @ (acceleration_world_mps2 - gravity_world_mps2) + accel_bias_mps2).astype(np.float64)


def gyroscope_measurement_body(
    *,
    start_rotation_wi: Sequence[Sequence[float]],
    end_rotation_wi: Sequence[Sequence[float]],
    delta_time_s: float,
    gyro_bias_rps: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    start_rotation_wi = np.asarray(start_rotation_wi, dtype=np.float64).reshape(3, 3)
    end_rotation_wi = np.asarray(end_rotation_wi, dtype=np.float64).reshape(3, 3)
    gyro_bias_rps = np.asarray(gyro_bias_rps, dtype=np.float64).reshape(3)
    relative_rotation = start_rotation_wi.T @ end_rotation_wi
    return (rvec_from_rotation_matrix(relative_rotation) / max(float(delta_time_s), 1e-12) + gyro_bias_rps).astype(np.float64)


@dataclass(slots=True)
class AggregatedInertialTransition:
    delta_time_s: float
    final_rotation_wi: np.ndarray
    mean_world_acceleration_mps2: np.ndarray
    final_velocity_world_mps: np.ndarray
    final_position_world_m: np.ndarray


@dataclass(slots=True)
class InertialResidualComponents:
    position: np.ndarray
    rotation: np.ndarray
    velocity: np.ndarray

    def stacked(self) -> np.ndarray:
        return np.concatenate((self.position, self.rotation, self.velocity), axis=0).astype(np.float64)


def integrate_interval_constant_world_acceleration(
    *,
    start_position_world_m: Sequence[float],
    start_rotation_wi: Sequence[Sequence[float]],
    start_velocity_world_mps: Sequence[float],
    gyro_packets_body_rps: Sequence[Sequence[float]],
    accel_packets_body_mps2: Sequence[Sequence[float]],
    dt_packets_s: Sequence[float],
    gravity_world_mps2: Sequence[float] = (0.0, 0.0, -9.81),
    accel_bias_mps2: Sequence[float] = (0.0, 0.0, 0.0),
    gyro_bias_rps: Sequence[float] = (0.0, 0.0, 0.0),
) -> AggregatedInertialTransition:
    start_position_world_m = np.asarray(start_position_world_m, dtype=np.float64).reshape(3)
    start_rotation_wi = np.asarray(start_rotation_wi, dtype=np.float64).reshape(3, 3)
    start_velocity_world_mps = np.asarray(start_velocity_world_mps, dtype=np.float64).reshape(3)
    gravity_world_mps2 = np.asarray(gravity_world_mps2, dtype=np.float64).reshape(3)
    accel_bias_mps2 = np.asarray(accel_bias_mps2, dtype=np.float64).reshape(3)
    gyro_bias_rps = np.asarray(gyro_bias_rps, dtype=np.float64).reshape(3)
    gyro_packets = np.asarray(gyro_packets_body_rps, dtype=np.float64).reshape(-1, 3)
    accel_packets = np.asarray(accel_packets_body_mps2, dtype=np.float64).reshape(-1, 3)
    dt_packets = np.asarray(dt_packets_s, dtype=np.float64).reshape(-1)

    if gyro_packets.shape[0] != accel_packets.shape[0] or gyro_packets.shape[0] != dt_packets.shape[0]:
        raise ValueError("IMU packet arrays must have matching lengths.")
    if gyro_packets.shape[0] == 0:
        return AggregatedInertialTransition(
            delta_time_s=0.0,
            final_rotation_wi=start_rotation_wi.copy(),
            mean_world_acceleration_mps2=np.zeros(3, dtype=np.float64),
            final_velocity_world_mps=start_velocity_world_mps.copy(),
            final_position_world_m=start_position_world_m.copy(),
        )

    total_dt_s = float(np.sum(dt_packets))
    current_rotation_wi = start_rotation_wi.copy()
    accumulated_world_accel = np.zeros(3, dtype=np.float64)
    for gyro_body_rps, accel_body_mps2, dt_s in zip(gyro_packets, accel_packets, dt_packets):
        corrected_gyro = np.asarray(gyro_body_rps, dtype=np.float64).reshape(3) - gyro_bias_rps
        corrected_accel = np.asarray(accel_body_mps2, dtype=np.float64).reshape(3) - accel_bias_mps2
        current_rotation_wi = current_rotation_wi @ rotation_matrix_from_rvec(corrected_gyro * float(dt_s))
        accumulated_world_accel += (current_rotation_wi @ corrected_accel + gravity_world_mps2) * float(dt_s)
    mean_world_acceleration_mps2 = accumulated_world_accel / max(total_dt_s, 1e-12)
    final_velocity_world_mps = start_velocity_world_mps + mean_world_acceleration_mps2 * total_dt_s
    # The current async headline recordings store frame-to-frame velocity as
    # displacement / dt, so the position transition stays consistent when we
    # apply the final interval velocity over the full frame interval.
    final_position_world_m = start_position_world_m + final_velocity_world_mps * total_dt_s
    return AggregatedInertialTransition(
        delta_time_s=total_dt_s,
        final_rotation_wi=current_rotation_wi.astype(np.float64),
        mean_world_acceleration_mps2=mean_world_acceleration_mps2.astype(np.float64),
        final_velocity_world_mps=final_velocity_world_mps.astype(np.float64),
        final_position_world_m=final_position_world_m.astype(np.float64),
    )


def inertial_transition_residual_components(
    *,
    start_position_world_m: Sequence[float],
    start_rotation_wi: Sequence[Sequence[float]],
    start_velocity_world_mps: Sequence[float],
    end_position_world_m: Sequence[float],
    end_rotation_wi: Sequence[Sequence[float]],
    end_velocity_world_mps: Sequence[float],
    gyro_packets_body_rps: Sequence[Sequence[float]],
    accel_packets_body_mps2: Sequence[Sequence[float]],
    dt_packets_s: Sequence[float],
    gravity_world_mps2: Sequence[float] = (0.0, 0.0, -9.81),
    accel_bias_mps2: Sequence[float] = (0.0, 0.0, 0.0),
    gyro_bias_rps: Sequence[float] = (0.0, 0.0, 0.0),
) -> InertialResidualComponents:
    predicted = integrate_interval_constant_world_acceleration(
        start_position_world_m=start_position_world_m,
        start_rotation_wi=start_rotation_wi,
        start_velocity_world_mps=start_velocity_world_mps,
        gyro_packets_body_rps=gyro_packets_body_rps,
        accel_packets_body_mps2=accel_packets_body_mps2,
        dt_packets_s=dt_packets_s,
        gravity_world_mps2=gravity_world_mps2,
        accel_bias_mps2=accel_bias_mps2,
        gyro_bias_rps=gyro_bias_rps,
    )
    end_position_world_m = np.asarray(end_position_world_m, dtype=np.float64).reshape(3)
    end_rotation_wi = np.asarray(end_rotation_wi, dtype=np.float64).reshape(3, 3)
    end_velocity_world_mps = np.asarray(end_velocity_world_mps, dtype=np.float64).reshape(3)
    return InertialResidualComponents(
        position=(predicted.final_position_world_m - end_position_world_m).astype(np.float64),
        rotation=rvec_from_rotation_matrix(predicted.final_rotation_wi.T @ end_rotation_wi).astype(np.float64),
        velocity=(predicted.final_velocity_world_mps - end_velocity_world_mps).astype(np.float64),
    )
