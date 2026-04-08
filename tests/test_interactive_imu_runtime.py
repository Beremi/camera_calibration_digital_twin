"""Focused tests for the interactive IMU runtime helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from calib_sim.interactive.imu_runtime import (  # noqa: E402
    FixedRateImuScheduler,
    ImuBiasState,
    ImuNoisePreset,
    ImuRuntime,
    ImuTruthSample,
    checkpoint_01_compat_imu_runtime,
    describe_imu_measurement_convention,
    load_builtin_noise_presets,
    load_imu_noise_preset,
    load_timing_preset,
    load_vision_noise_preset,
)


def _zero_truth(sample_time_s: float, dt_s: float) -> ImuTruthSample:
    return ImuTruthSample(
        sim_time_s=sample_time_s,
        accel_body_mps2=np.zeros(3, dtype=np.float64),
        gyro_body_rps=np.zeros(3, dtype=np.float64),
    )


def test_builtin_noise_presets_load_and_parse() -> None:
    presets = load_builtin_noise_presets()
    assert set(presets) == {"imu_ideal", "imu_nominal_phone", "imu_stress_phone"}
    assert presets["imu_ideal"].mode == "checkpoint_01_compat"
    assert presets["imu_nominal_phone"].rate_hz == 58.236
    assert presets["imu_stress_phone"].rate_hz == 100.0


def test_visual_and_timing_presets_load() -> None:
    vision = load_vision_noise_preset("config/noise/vision_nominal.yaml")
    timing = load_timing_preset("config/noise/timing_nominal.yaml")
    timing_vi = load_timing_preset("config/noise/timing_vi_headline.yaml")
    assert vision.corner_noise_std_px > 0.0
    assert timing.physics_rate_hz == 240.0
    assert timing.imu_rate_hz == 58.236
    assert timing_vi.imu_rate_hz == 100.0


def test_imu_convention_descriptor_is_explicit_for_compat_and_async() -> None:
    compat = describe_imu_measurement_convention(compat_single_sample=True, read_gravity=True)
    async_mode = describe_imu_measurement_convention(compat_single_sample=False, read_gravity=True)

    assert compat["imu_measurement_convention"] == "body_specific_force_plus_bias"
    assert compat["imu_timestamp_semantics"] == "frame_locked_sim_time"
    assert compat["imu_sampling_mode"] == "end_sampled_interval"
    assert async_mode["imu_timestamp_semantics"] == "independent_sim_time"
    assert async_mode["imu_sampling_mode"] == "independent_fixed_rate"


def test_fixed_rate_scheduler_emits_expected_due_times() -> None:
    scheduler = FixedRateImuScheduler(rate_hz=100.0, start_time_s=0.0)
    first = scheduler.advance_to(0.009)
    assert first == []
    second = scheduler.advance_to(0.031)
    assert [round(item.sim_time_s, 3) for item in second] == [0.010, 0.020, 0.030]
    assert [item.sample_index for item in second] == [0, 1, 2]
    assert [round(item.dt_s, 3) for item in second] == [0.010, 0.010, 0.010]


def test_checkpoint_compat_runtime_emits_one_packet_per_frame() -> None:
    preset = load_imu_noise_preset("config/noise/imu_ideal.yaml")
    runtime = checkpoint_01_compat_imu_runtime(preset, rng=np.random.default_rng(7))
    first = runtime.advance_to(0.083333333, _zero_truth, frame_dt_s=0.083333333, frame_tick_index=1)
    second = runtime.advance_to(0.166666667, _zero_truth, frame_dt_s=0.083333334, frame_tick_index=2)
    assert len(first) == 1
    assert len(second) == 1
    assert first[0].is_checkpoint_compat is True
    assert first[0].tick_index == 1
    assert second[0].tick_index == 2
    assert np.allclose(first[0].accel_body_mps2, 0.0)
    assert np.allclose(first[0].gyro_body_rps, 0.0)


def test_bias_random_walk_is_reproducible_and_accumulates() -> None:
    preset = ImuNoisePreset(
        name="test",
        mode="async_realism",
        rate_hz=100.0,
        read_gravity=True,
        accel_noise_std_mps2=np.zeros(3, dtype=np.float64),
        gyro_noise_std_rps=np.zeros(3, dtype=np.float64),
        accel_bias_initial_mps2=np.zeros(3, dtype=np.float64),
        gyro_bias_initial_rps=np.zeros(3, dtype=np.float64),
        accel_bias_walk_std_mps2_per_sqrt_s=np.array([0.1, 0.1, 0.1], dtype=np.float64),
        gyro_bias_walk_std_rps_per_sqrt_s=np.array([0.01, 0.01, 0.01], dtype=np.float64),
    )
    state_a = ImuBiasState.from_preset(preset)
    state_b = ImuBiasState.from_preset(preset)
    rng_a = np.random.default_rng(1234)
    rng_b = np.random.default_rng(1234)
    state_a.step(dt_s=0.25, rng=rng_a, preset=preset)
    state_b.step(dt_s=0.25, rng=rng_b, preset=preset)
    assert np.allclose(state_a.accel_bias_mps2, state_b.accel_bias_mps2)
    assert np.allclose(state_a.gyro_bias_rps, state_b.gyro_bias_rps)
    assert np.linalg.norm(state_a.accel_bias_mps2) > 0.0
    assert np.linalg.norm(state_a.gyro_bias_rps) > 0.0


def test_runtime_can_emit_async_measurements_with_bias_and_noise() -> None:
    preset = load_imu_noise_preset("config/noise/imu_nominal_phone.yaml")
    runtime = ImuRuntime(preset=preset, rng=np.random.default_rng(9))
    packets = runtime.advance_to(0.07, _zero_truth)
    assert len(packets) == 4
    assert [packet.sample_index for packet in packets] == [0, 1, 2, 3]
    assert all(packet.dt_s > 0.0 for packet in packets)
    assert all(packet.sim_time_s > 0.0 for packet in packets)
    assert any(np.linalg.norm(packet.accel_body_mps2) > 0.0 for packet in packets)
    assert any(np.linalg.norm(packet.gyro_body_rps) > 0.0 for packet in packets)
