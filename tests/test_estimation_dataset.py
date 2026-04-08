"""Focused tests for batch estimation dataset and noise loaders."""

from __future__ import annotations

import json
import shutil
from dataclasses import fields
from pathlib import Path

import pytest

from calib_sim.estimation import (
    BatchCalibrationDataset,
    BatchCalibrationEvaluationData,
    BatchSchemaError,
    load_batch_dataset,
    load_device_config,
    load_evaluation_data,
    load_imu_noise_preset,
    load_timing_noise_preset,
    load_vision_noise_preset,
)
from calib_sim.estimation.reporting import _perturb_dataset


RUN_DIR = Path("output/interactive_runs/run_20260406_083611")


def _copy_minimal_run(src: Path, dst: Path, *, include_camera_gt: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["metadata.json", "samples.jsonl", "imu.csv"]:
        shutil.copy2(src / name, dst / name)
    if include_camera_gt:
        shutil.copy2(src / "camera_gt.csv", dst / "camera_gt.csv")


def test_load_batch_dataset_parses_current_checkpoint_without_gt_leak() -> None:
    dataset = load_batch_dataset(RUN_DIR)

    assert isinstance(dataset, BatchCalibrationDataset)
    assert dataset.run_dir == RUN_DIR.resolve()
    assert dataset.run_name == "tabletop_grab_challenge"
    assert dataset.recording_frame_rate_hz == pytest.approx(12.0)
    assert len(dataset.camera_frames) == 240
    assert len(dataset.imu_packets) == 240
    assert len(dataset.tag_detections) == 714
    assert dataset.camera_model.name == "pixel_9a_main"
    assert dataset.device_config.name == "pixel_9a_phone"
    assert dataset.device_config.imu.rate_hz == pytest.approx(58.236)
    assert dataset.imu_convention.imu_measurement_convention == "body_specific_force_plus_bias"
    assert dataset.imu_convention.imu_timestamp_semantics == "frame_locked_sim_time"
    assert dataset.imu_convention.imu_gravity_handling == "gravity_removed_from_specific_force"
    assert dataset.imu_convention.imu_sampling_mode == "end_sampled_interval"
    assert dataset.tag_catalog[155].size_m == pytest.approx(0.06)
    assert dataset.tag_catalog[201].size_m == pytest.approx(0.06)
    assert dataset.tag_catalog[0].size_m == pytest.approx(0.10)
    assert all(not hasattr(frame, "ground_truth") for frame in dataset.camera_frames)
    assert all(not hasattr(detection, "ground_truth") for detection in dataset.tag_detections)
    assert all("camera_gt" not in field.name for field in fields(BatchCalibrationDataset))


def test_load_batch_dataset_ignores_corrupted_camera_gt_file(tmp_path: Path) -> None:
    clean_run = tmp_path / "clean"
    dirty_run = tmp_path / "dirty"
    _copy_minimal_run(RUN_DIR, clean_run, include_camera_gt=True)
    _copy_minimal_run(RUN_DIR, dirty_run, include_camera_gt=True)
    (dirty_run / "camera_gt.csv").write_text(
        "\n".join(
            [
                "tick_index,sim_time_s,cx_world_m,cy_world_m,cz_world_m,vx_world_mps,vy_world_mps,vz_world_mps,r00,r01,r02,r10,r11,r12,r20,r21,r22,servo0_deg,servo1_deg,servo2_deg",
                "1,0.1,999,999,999,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    clean = load_batch_dataset(clean_run)
    dirty = load_batch_dataset(dirty_run)

    assert clean.camera_frames == dirty.camera_frames
    assert clean.tag_detections == dirty.tag_detections
    assert clean.imu_packets == dirty.imu_packets


def test_load_evaluation_data_uses_gt_and_requires_gt_file(tmp_path: Path) -> None:
    _copy_minimal_run(RUN_DIR, tmp_path / "run", include_camera_gt=True)
    evaluation = load_evaluation_data(tmp_path / "run")

    assert isinstance(evaluation, BatchCalibrationEvaluationData)
    assert len(evaluation.camera_truth) == 240
    assert len(evaluation.tag_truth_by_frame) > 0
    first_frame = evaluation.tag_truth_by_frame[1]
    assert len(first_frame) == 8
    assert {sample.tag_id for sample in first_frame} >= {155, 201}

    no_gt_run = tmp_path / "no_gt"
    _copy_minimal_run(RUN_DIR, no_gt_run, include_camera_gt=False)
    with pytest.raises(FileNotFoundError):
        load_evaluation_data(no_gt_run)


def test_load_batch_dataset_rejects_legacy_schema_without_modern_fields(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy_run"
    legacy.mkdir()
    (legacy / "metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": "2026-04-06T00:00:00+00:00",
                "config": {
                    "name": "legacy",
                    "device_config_path": "config/device/pixel_9a_phone.yaml",
                    "tags": [],
                    "primary_camera": {"width": 960, "height": 540, "fov_deg": 72.0},
                },
                "camera_model": {},
            }
        ),
        encoding="utf-8",
    )
    (legacy / "samples.jsonl").write_text(
        json.dumps(
            {
                "sim_time_s": 0.1,
                "tick_index": 1,
                "detections": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (legacy / "imu.csv").write_text(
        "tick_index,sim_time_s,ax_mps2,ay_mps2,az_mps2,gx_rps,gy_rps,gz_rps\n1,0.1,0,0,0,0,0,0\n",
        encoding="utf-8",
    )

    with pytest.raises(BatchSchemaError):
        load_batch_dataset(legacy)


def test_noise_preset_loaders_return_expected_nominal_values() -> None:
    imu = load_imu_noise_preset("imu_nominal_phone")
    imu_async_ideal = load_imu_noise_preset("config/noise/imu_async_ideal.yaml")
    vision = load_vision_noise_preset("vision_nominal")
    timing = load_timing_noise_preset("timing_nominal")
    timing_vi = load_timing_noise_preset("config/noise/timing_vi_headline.yaml")

    assert imu.name == "imu_nominal_phone"
    assert imu.accel_noise_std == pytest.approx((0.0047856453, 0.0047856453, 0.0047856453))
    assert imu.gyro_noise_std == pytest.approx((0.0012217305, 0.0012217305, 0.0012217305))
    assert imu_async_ideal.name == "imu_async_ideal"
    assert imu_async_ideal.accel_noise_std == pytest.approx((0.0, 0.0, 0.0))
    assert vision.corner_noise_std_px == pytest.approx(0.75)
    assert vision.detection_drop_probability == pytest.approx(0.05)
    assert vision.visibility_failure_probability == pytest.approx(0.03)
    assert vision.outlier_probability == pytest.approx(0.01)
    assert timing.imu_rate_hz == pytest.approx(58.236)
    assert timing.physics_rate_hz == pytest.approx(240.0)
    assert timing_vi.imu_rate_hz == pytest.approx(100.0)


def test_dataset_perturbation_respects_visibility_failures_and_outliers() -> None:
    dataset = load_batch_dataset(RUN_DIR)
    baseline_detection_count = len(dataset.tag_detections)

    outlier_preset = load_vision_noise_preset("vision_nominal")
    outlier_preset.outlier_probability = 1.0
    outlier_preset.detection_drop_probability = 0.0
    outlier_preset.visibility_failure_probability = 0.0
    perturbed_outliers = _perturb_dataset(
        dataset,
        vision_noise=outlier_preset,
        imu_noise=load_imu_noise_preset("imu_ideal"),
        seed=7,
    )
    baseline_first = dataset.tag_detections[0]
    outlier_first = perturbed_outliers.tag_detections[0]
    assert baseline_first.corners_xy_clockwise_px != outlier_first.corners_xy_clockwise_px

    visibility_preset = load_vision_noise_preset("vision_nominal")
    visibility_preset.corner_noise_std_px = 0.0
    visibility_preset.detection_drop_probability = 0.0
    visibility_preset.visibility_failure_probability = 0.8
    visibility_preset.outlier_probability = 0.0
    perturbed_visibility = _perturb_dataset(
        dataset,
        vision_noise=visibility_preset,
        imu_noise=load_imu_noise_preset("imu_ideal"),
        seed=11,
    )
    assert len(perturbed_visibility.tag_detections) < baseline_detection_count


def test_device_config_loader_parses_current_phone_preset() -> None:
    device = load_device_config("config/device/pixel_9a_phone.yaml")

    assert device.name == "pixel_9a_phone"
    assert device.camera.width == 1920
    assert device.camera.height == 1080
    assert device.imu.rate_hz == pytest.approx(58.236)
    assert device.mount.imu_translation_m == pytest.approx((0.0, 0.0, 0.01))
