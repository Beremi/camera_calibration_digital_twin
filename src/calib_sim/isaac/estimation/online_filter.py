"""Low-latency anchored visual-inertial filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calib_sim.isaac.estimation.imu_preintegration import preintegrate_imu_packets
from calib_sim.isaac.estimation.state_defs import FilterStateSnapshot, MotionState
from calib_sim.isaac.estimation.uncertainty import radius_95_from_covariance, sanitize_covariance
from calib_sim.isaac.logging.schemas import IsaacImuPacket


@dataclass(slots=True)
class AnchoredOnlineFilter:
    state: MotionState
    covariance: np.ndarray
    last_timestamp_s: float = 0.0
    last_innovation_norm: float = 0.0

    @classmethod
    def identity_initialized(cls) -> "AnchoredOnlineFilter":
        return cls(
            state=MotionState(
                rotation_wi=np.eye(3, dtype=np.float64),
                position_world_m=np.zeros(3, dtype=np.float64),
                velocity_world_mps=np.zeros(3, dtype=np.float64),
                gyro_bias_rps=np.zeros(3, dtype=np.float64),
                accel_bias_mps2=np.zeros(3, dtype=np.float64),
            ),
            covariance=np.eye(15, dtype=np.float64) * 1e-3,
            last_timestamp_s=0.0,
        )

    def propagate(self, packets: tuple[IsaacImuPacket, ...]) -> None:
        delta = preintegrate_imu_packets(
            packets=packets,
            start_rotation_wi=self.state.rotation_wi,
            start_position_world_m=self.state.position_world_m,
            start_velocity_world_mps=self.state.velocity_world_mps,
            accel_bias_mps2=self.state.accel_bias_mps2,
            gyro_bias_rps=self.state.gyro_bias_rps,
        )
        self.state.rotation_wi = delta.final_rotation_wi
        self.state.velocity_world_mps = delta.final_velocity_world_mps
        self.state.position_world_m = delta.final_position_world_m
        if packets:
            self.last_timestamp_s = float(packets[-1].timestamp_s)
        process_noise = np.eye(self.covariance.shape[0], dtype=np.float64) * max(delta.delta_time_s, 1e-6) * 1e-4
        self.covariance = sanitize_covariance(self.covariance + process_noise)

    def anchor_visual_update(
        self,
        *,
        measured_position_world_m: np.ndarray,
        measured_rotation_wi: np.ndarray,
        measurement_std_m: float = 0.01,
        rotation_gain: float = 0.5,
    ) -> None:
        measured_position_world_m = np.asarray(measured_position_world_m, dtype=np.float64).reshape(3)
        measured_rotation_wi = np.asarray(measured_rotation_wi, dtype=np.float64).reshape(3, 3)
        innovation = measured_position_world_m - self.state.position_world_m
        kalman_gain = 1.0 / max(1.0 + measurement_std_m * 100.0, 1e-6)
        self.state.position_world_m = self.state.position_world_m + kalman_gain * innovation
        self.state.rotation_wi = measured_rotation_wi.copy()
        self.last_innovation_norm = float(np.linalg.norm(innovation))
        self.covariance = sanitize_covariance(self.covariance * (1.0 - 0.25 * kalman_gain))
        self.covariance[:3, :3] = np.eye(3, dtype=np.float64) * max(measurement_std_m**2, 1e-9)
        self.covariance[3:6, 3:6] = np.eye(3, dtype=np.float64) * max(rotation_gain * measurement_std_m**2, 1e-9)

    def auxiliary_visual_update(self, *, relative_position_delta_m: np.ndarray, trust: float = 0.15) -> None:
        relative_position_delta_m = np.asarray(relative_position_delta_m, dtype=np.float64).reshape(3)
        self.state.position_world_m = self.state.position_world_m + float(trust) * relative_position_delta_m
        self.last_innovation_norm = float(np.linalg.norm(relative_position_delta_m))
        self.covariance = sanitize_covariance(self.covariance * max(1.0 - float(trust) * 0.1, 1e-6))

    def export_snapshot(self, *, sim_time_s: float) -> FilterStateSnapshot:
        return FilterStateSnapshot(
            timestamp_s=float(self.last_timestamp_s),
            sim_time_s=float(sim_time_s),
            rotation_wi=self.state.rotation_wi.copy(),
            position_world_m=self.state.position_world_m.copy(),
            velocity_world_mps=self.state.velocity_world_mps.copy(),
            gyro_bias_rps=self.state.gyro_bias_rps.copy(),
            accel_bias_mps2=self.state.accel_bias_mps2.copy(),
            covariance=self.covariance.copy(),
            innovation_diagnostics={
                "last_innovation_norm": float(self.last_innovation_norm),
                "position_radius_95_m": float(radius_95_from_covariance(self.covariance[:3, :3])),
            },
        )
