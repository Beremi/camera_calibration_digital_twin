"""IMU preintegration helpers reused by the anchored online filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calib_sim.common.inertial import integrate_interval_constant_world_acceleration
from calib_sim.isaac.logging.schemas import IsaacImuPacket


@dataclass(slots=True)
class PreintegratedImuDelta:
    delta_time_s: float
    final_rotation_wi: np.ndarray
    final_velocity_world_mps: np.ndarray
    final_position_world_m: np.ndarray


def preintegrate_imu_packets(
    *,
    packets: tuple[IsaacImuPacket, ...],
    start_rotation_wi: np.ndarray,
    start_position_world_m: np.ndarray,
    start_velocity_world_mps: np.ndarray,
    accel_bias_mps2: np.ndarray,
    gyro_bias_rps: np.ndarray,
) -> PreintegratedImuDelta:
    if not packets:
        return PreintegratedImuDelta(
            delta_time_s=0.0,
            final_rotation_wi=np.asarray(start_rotation_wi, dtype=np.float64),
            final_velocity_world_mps=np.asarray(start_velocity_world_mps, dtype=np.float64),
            final_position_world_m=np.asarray(start_position_world_m, dtype=np.float64),
        )
    timestamps = [float(packet.timestamp_s) for packet in packets]
    dt_packets = [max(timestamps[index] - timestamps[index - 1], 1e-6) for index in range(1, len(timestamps))]
    dt_packets.insert(0, dt_packets[0] if dt_packets else 1e-2)
    transition = integrate_interval_constant_world_acceleration(
        start_position_world_m=start_position_world_m,
        start_rotation_wi=start_rotation_wi,
        start_velocity_world_mps=start_velocity_world_mps,
        gyro_packets_body_rps=[(packet.wx, packet.wy, packet.wz) for packet in packets],
        accel_packets_body_mps2=[(packet.ax, packet.ay, packet.az) for packet in packets],
        dt_packets_s=dt_packets,
        accel_bias_mps2=accel_bias_mps2,
        gyro_bias_rps=gyro_bias_rps,
    )
    return PreintegratedImuDelta(
        delta_time_s=float(transition.delta_time_s),
        final_rotation_wi=np.asarray(transition.final_rotation_wi, dtype=np.float64),
        final_velocity_world_mps=np.asarray(transition.final_velocity_world_mps, dtype=np.float64),
        final_position_world_m=np.asarray(transition.final_position_world_m, dtype=np.float64),
    )
