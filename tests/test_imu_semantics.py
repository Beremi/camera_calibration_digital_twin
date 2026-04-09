"""Scientific IMU semantics checks for the Isaac scaffold."""

from __future__ import annotations

import numpy as np

from calib_sim.common.inertial import accelerometer_specific_force_body, gyroscope_measurement_body
from calib_sim.isaac.clocks import ScheduledSensorTick, TimestampTriplet
from calib_sim.isaac.estimation.imu_preintegration import preintegrate_imu_packets
from calib_sim.isaac.logging.schemas import IsaacImuPacket
from calib_sim.isaac.sensors import IsaacImuBinding, ImuSensorSpec


def test_accelerometer_specific_force_is_explicit_at_rest() -> None:
    specific_force = accelerometer_specific_force_body(
        rotation_wi=np.eye(3, dtype=np.float64),
        acceleration_world_mps2=(0.0, 0.0, 0.0),
    )
    assert np.allclose(specific_force, np.array([0.0, 0.0, 9.81], dtype=np.float64))


def test_gyroscope_measurement_uses_body_frame_angular_rate() -> None:
    end_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    gyro = gyroscope_measurement_body(
        start_rotation_wi=np.eye(3, dtype=np.float64),
        end_rotation_wi=end_rotation,
        delta_time_s=0.5,
    )
    assert np.allclose(gyro[:2], 0.0, atol=1e-8)
    assert abs(gyro[2] - np.pi) < 1e-6


def test_preintegration_preserves_stationary_state_for_specific_force_packets() -> None:
    packets = tuple(
        IsaacImuPacket(
            packet_index=index,
            timestamp_s=0.01 * (index + 1),
            sim_time_s=0.01 * (index + 1),
            dt_s=0.01,
            wx=0.0,
            wy=0.0,
            wz=0.0,
            ax=0.0,
            ay=0.0,
            az=9.81,
            imu_frame="I",
            imu_semantics="specific_force",
            noise_preset="ideal",
        )
        for index in range(4)
    )
    delta = preintegrate_imu_packets(
        packets=packets,
        start_rotation_wi=np.eye(3, dtype=np.float64),
        start_position_world_m=np.zeros(3, dtype=np.float64),
        start_velocity_world_mps=np.zeros(3, dtype=np.float64),
        accel_bias_mps2=np.zeros(3, dtype=np.float64),
        gyro_bias_rps=np.zeros(3, dtype=np.float64),
    )
    assert np.allclose(delta.final_position_world_m, 0.0, atol=1e-9)
    assert np.allclose(delta.final_velocity_world_mps, 0.0, atol=1e-9)
    assert np.allclose(delta.final_rotation_wi, np.eye(3), atol=1e-9)


def test_isaac_imu_binding_caps_unphysical_pose_differentiation_spikes() -> None:
    spec = ImuSensorSpec(
        name="imu",
        rate_hz=200.0,
        frame_id="I",
        prim_path="/World/Robot/imu",
        parent_prim_path="/World/Robot",
        synthetic_velocity_lowpass_alpha=1.0,
        synthetic_max_world_velocity_mps=2.0,
        synthetic_max_world_acceleration_mps2=20.0,
        synthetic_max_specific_force_mps2=25.0,
    )
    binding = IsaacImuBinding.create(spec)
    timestamps = TimestampTriplet(sim_time_s=0.0, sensor_time_s=0.0, host_time_s=0.0)

    packet0 = binding.sample(
        tick=ScheduledSensorTick(sample_index=0, sim_time_s=0.005, dt_s=0.005),
        timestamps=timestamps,
        position_world_m=np.zeros(3, dtype=np.float64),
        rotation_wi=np.eye(3, dtype=np.float64),
    )
    packet1 = binding.sample(
        tick=ScheduledSensorTick(sample_index=1, sim_time_s=0.010, dt_s=0.005),
        timestamps=timestamps,
        position_world_m=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        rotation_wi=np.eye(3, dtype=np.float64),
    )

    assert packet0 is not None
    assert packet1 is not None
    assert np.linalg.norm(binding.previous_velocity_world_mps) <= 2.0 + 1e-9
    assert np.linalg.norm(np.array([packet1.ax, packet1.ay, packet1.az], dtype=np.float64)) <= 25.0 + 1e-9
