"""Dataset loading utilities for batch calibration estimation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from calib_sim.estimation.types import (
    BatchCalibrationDataset,
    BatchCalibrationEvaluationData,
    CameraFrame,
    CameraModelSnapshot,
    CameraPoseEstimationBridgeSnapshot,
    CameraTruthSample,
    DeviceCameraSnapshot,
    DeviceConfigSnapshot,
    DeviceImuSnapshot,
    DeviceMountSnapshot,
    ImuConventionSnapshot,
    ImuPacket,
    TagDetectionObservation,
    TagSpec,
    TagTruthSample,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CURRENT_SCHEMA_VERSION = 1


class BatchDataError(ValueError):
    """Raised when a recording does not match the expected batch schema."""


class BatchSchemaError(BatchDataError):
    """Raised when a recording looks like an incompatible legacy schema."""


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BatchSchemaError(f"{context} must be a mapping.")
    return value


def _require_keys(mapping: dict[str, Any], *, context: str, required: Iterable[str]) -> None:
    missing = [key for key in required if key not in mapping]
    if missing:
        raise BatchSchemaError(f"{context} is missing required keys: {', '.join(missing)}")


def _float_tuple(values: Any, *, length: int, context: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise BatchSchemaError(f"{context} must contain exactly {length} numeric values.")
    try:
        return tuple(float(value) for value in values)
    except Exception as exc:  # pragma: no cover - defensive.
        raise BatchSchemaError(f"{context} must contain numeric values.") from exc


def _float_pair(values: Any, *, context: str) -> tuple[float, float]:
    pair = _float_tuple(values, length=2, context=context)
    return float(pair[0]), float(pair[1])


def _float_triplet(values: Any, *, context: str) -> tuple[float, float, float]:
    triplet = _float_tuple(values, length=3, context=context)
    return float(triplet[0]), float(triplet[1]), float(triplet[2])


def _float_quintet(values: Any, *, context: str) -> tuple[float, float, float, float, float]:
    quintet = _float_tuple(values, length=5, context=context)
    return tuple(float(value) for value in quintet)  # type: ignore[return-value]


def _camera_pose_bridge_from_mapping(mapping: dict[str, Any]) -> CameraPoseEstimationBridgeSnapshot:
    _require_keys(
        mapping,
        context="camera_pose_estimation",
        required=["pattern_half_extent_m", "measurement_unit", "measurement_order", "y_axis", "undistort_before_export"],
    )
    measurement_order = mapping["measurement_order"]
    if not isinstance(measurement_order, (list, tuple)) or not measurement_order:
        raise BatchSchemaError("camera_pose_estimation.measurement_order must be a non-empty list.")
    return CameraPoseEstimationBridgeSnapshot(
        pattern_half_extent_m=float(mapping["pattern_half_extent_m"]),
        measurement_unit=str(mapping["measurement_unit"]),
        measurement_order=tuple(str(item) for item in measurement_order),
        y_axis=str(mapping["y_axis"]),
        undistort_before_export=bool(mapping["undistort_before_export"]),
    )


def _camera_model_from_mapping(mapping: dict[str, Any]) -> CameraModelSnapshot:
    _require_keys(
        mapping,
        context="camera_model",
        required=[
            "name",
            "source_toml_path",
            "projection_model",
            "apply_lens_distortion_in_render",
            "output_resolution_px",
            "focal_length_mm",
            "focus_distance_diopters",
            "target_frames_per_second",
            "timestamp_source",
            "sensor_orientation_deg",
            "effective_intrinsics_px",
            "crop_rect_active_array_px",
            "distortion",
            "camera_pose_estimation",
        ],
    )
    output_resolution_px = mapping["output_resolution_px"]
    if not isinstance(output_resolution_px, (list, tuple)) or len(output_resolution_px) != 2:
        raise BatchSchemaError("camera_model.output_resolution_px must contain width and height.")
    intrinsics = _require_mapping(mapping["effective_intrinsics_px"], context="camera_model.effective_intrinsics_px")
    crop = _require_mapping(mapping["crop_rect_active_array_px"], context="camera_model.crop_rect_active_array_px")
    distortion = _require_mapping(mapping["distortion"], context="camera_model.distortion")
    return CameraModelSnapshot(
        name=str(mapping["name"]),
        source_toml_path=str(mapping["source_toml_path"]),
        projection_model=str(mapping["projection_model"]),
        apply_lens_distortion_in_render=bool(mapping["apply_lens_distortion_in_render"]),
        output_width_px=int(output_resolution_px[0]),
        output_height_px=int(output_resolution_px[1]),
        focal_length_mm=float(mapping["focal_length_mm"]),
        focus_distance_diopters=float(mapping["focus_distance_diopters"]),
        target_frames_per_second=float(mapping["target_frames_per_second"]),
        timestamp_source=str(mapping["timestamp_source"]),
        sensor_orientation_deg=int(mapping["sensor_orientation_deg"]),
        fx_px=float(intrinsics["fx"]),
        fy_px=float(intrinsics["fy"]),
        cx_px=float(intrinsics["cx"]),
        cy_px=float(intrinsics["cy"]),
        skew_px=float(intrinsics.get("skew", 0.0)),
        crop_x_px=float(crop["x"]),
        crop_y_px=float(crop["y"]),
        crop_width_px=float(crop["width"]),
        crop_height_px=float(crop["height"]),
        distortion_model=str(distortion["model"]),
        distortion_coefficients=(
            float(distortion["coefficients"][0]),
            float(distortion["coefficients"][1]),
            float(distortion["coefficients"][2]),
            float(distortion["coefficients"][3]),
            float(distortion["coefficients"][4]),
        ),
        pose_bridge=_camera_pose_bridge_from_mapping(_require_mapping(mapping["camera_pose_estimation"], context="camera_model.camera_pose_estimation")),
    )


def load_device_config(path: str | Path) -> DeviceConfigSnapshot:
    resolved_path = _resolve_repo_path(path)
    raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    raw = _require_mapping(raw, context="device config")
    _require_keys(raw, context="device config", required=["name", "namespace", "camera_model_toml", "camera_rgb", "imu", "mount"])
    camera_raw = _require_mapping(raw["camera_rgb"], context="device config.camera_rgb")
    imu_raw = _require_mapping(raw["imu"], context="device config.imu")
    mount_raw = _require_mapping(raw["mount"], context="device config.mount")
    return DeviceConfigSnapshot(
        name=str(raw["name"]),
        namespace=str(raw["namespace"]),
        camera_model_toml=str(raw["camera_model_toml"]),
        camera=DeviceCameraSnapshot(
            width=int(camera_raw["width"]),
            height=int(camera_raw["height"]),
            fps=float(camera_raw["fps"]),
            fx=float(camera_raw["fx"]),
            fy=float(camera_raw["fy"]),
            cx=float(camera_raw["cx"]),
            cy=float(camera_raw["cy"]),
            distortion_model=str(camera_raw["distortion_model"]),
            distortion_coefficients=(
                float(camera_raw["distortion_coefficients"][0]),
                float(camera_raw["distortion_coefficients"][1]),
                float(camera_raw["distortion_coefficients"][2]),
                float(camera_raw["distortion_coefficients"][3]),
                float(camera_raw["distortion_coefficients"][4]),
            ),
        ),
        imu=DeviceImuSnapshot(
            rate_hz=float(imu_raw["rate_hz"]),
            read_gravity=bool(imu_raw.get("read_gravity", True)),
            accel_noise_std=_float_triplet(imu_raw["accel_noise_std"], context="device config.imu.accel_noise_std"),
            gyro_noise_std=_float_triplet(imu_raw["gyro_noise_std"], context="device config.imu.gyro_noise_std"),
            accel_bias=_float_triplet(imu_raw.get("accel_bias", [0.0, 0.0, 0.0]), context="device config.imu.accel_bias"),
            gyro_bias=_float_triplet(imu_raw.get("gyro_bias", [0.0, 0.0, 0.0]), context="device config.imu.gyro_bias"),
        ),
        mount=DeviceMountSnapshot(
            camera_frame=str(mount_raw.get("camera_frame", "phone_camera")),
            imu_frame=str(mount_raw.get("imu_frame", "phone_imu")),
            imu_translation_m=_float_triplet(mount_raw.get("imu_translation_m", [0.0, 0.0, 0.0]), context="device config.mount.imu_translation_m"),
            imu_rpy_deg=_float_triplet(mount_raw.get("imu_rpy_deg", [0.0, 0.0, 0.0]), context="device config.mount.imu_rpy_deg"),
        ),
    )


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in run directory: {run_dir}")
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw = _require_mapping(raw, context="metadata.json")
    _require_keys(raw, context="metadata.json", required=["config", "camera_model"])
    config_raw = _require_mapping(raw["config"], context="metadata.json.config")
    _require_keys(config_raw, context="metadata.json.config", required=["name", "device_config_path", "primary_camera", "tags"])
    if "recording" in raw:
        recording_raw = _require_mapping(raw["recording"], context="metadata.json.recording")
        if "frame_rate_hz" not in recording_raw:
            raise BatchSchemaError("metadata.json.recording.frame_rate_hz is required for batch estimation runs.")
    return raw


def _load_tag_catalog(config_raw: dict[str, Any]) -> dict[int, TagSpec]:
    tags_raw = config_raw.get("tags")
    if not isinstance(tags_raw, list) or not tags_raw:
        raise BatchSchemaError("metadata.json.config.tags must be a non-empty list.")
    catalog: dict[int, TagSpec] = {}
    for entry in tags_raw:
        item = _require_mapping(entry, context="metadata.json.config.tags[]")
        _require_keys(item, context="metadata.json.config.tags[]", required=["tag_id", "family", "size_m", "position_m", "mount"])
        orientation_rpy_deg = item.get("orientation_rpy_deg")
        catalog[int(item["tag_id"])] = TagSpec(
            tag_id=int(item["tag_id"]),
            family=str(item["family"]),
            size_m=float(item["size_m"]),
            position_m=_float_triplet(item["position_m"], context="metadata.json.config.tags[].position_m"),
            mount=str(item["mount"]),
            orientation_rpy_deg=None if orientation_rpy_deg is None else _float_triplet(orientation_rpy_deg, context="metadata.json.config.tags[].orientation_rpy_deg"),
        )
    return catalog


def _imu_convention_from_metadata(metadata: dict[str, Any]) -> ImuConventionSnapshot:
    recording_raw = _require_mapping(metadata.get("recording", {}), context="metadata.json.recording") if "recording" in metadata else {}
    imu_runtime_raw = _require_mapping(metadata.get("imu_runtime", {}), context="metadata.json.imu_runtime") if "imu_runtime" in metadata else {}
    imu_packet_mode = str(recording_raw.get("imu_packet_mode", "compat")).lower()
    imu_mode = str(recording_raw.get("imu_mode", imu_runtime_raw.get("mode", "checkpoint_01_compat"))).lower()
    read_gravity = bool(imu_runtime_raw.get("read_gravity", True))
    measurement_convention = str(
        metadata.get(
            "imu_measurement_convention",
            "body_specific_force_plus_bias" if read_gravity else "body_linear_acceleration_plus_bias",
        )
    )
    timestamp_semantics_default = "independent_sim_time" if imu_packet_mode == "async" else "frame_locked_sim_time"
    gravity_handling_default = "gravity_removed_from_specific_force" if read_gravity else "gravity_retained_in_body_acceleration"
    sampling_mode_default = "end_sampled_interval" if imu_mode == "checkpoint_01_compat" else "independent_fixed_rate"
    return ImuConventionSnapshot(
        imu_measurement_convention=measurement_convention,
        imu_timestamp_semantics=str(metadata.get("imu_timestamp_semantics", timestamp_semantics_default)),
        imu_gravity_handling=str(metadata.get("imu_gravity_handling", gravity_handling_default)),
        imu_sampling_mode=str(metadata.get("imu_sampling_mode", sampling_mode_default)),
        accelerometer_measurement_equation=str(
            metadata.get(
                "accelerometer_measurement_equation",
                "tilde_f_k = R_IW_k (a_W_k - g_W) + b_a + n_a",
            )
        ),
        gyroscope_measurement_equation=str(
            metadata.get(
                "gyroscope_measurement_equation",
                "tilde_omega_k = omega_k + b_g + n_g",
            )
        ),
    )


def _camera_model_matches(reference: CameraModelSnapshot, candidate: CameraModelSnapshot) -> bool:
    return reference.summary() == candidate.summary()


def _parse_sample_row(
    row: dict[str, Any],
    *,
    camera_model_reference: CameraModelSnapshot,
    tag_catalog: dict[int, TagSpec],
    strict: bool,
) -> tuple[CameraFrame, tuple[TagDetectionObservation, ...], CameraModelSnapshot]:
    _require_keys(
        row,
        context="samples.jsonl row",
        required=[
            "sim_time_s",
            "tick_index",
            "recording_frame_index",
            "servo_positions_deg",
            "servo_targets_deg",
            "automation",
            "imu",
            "detections",
            "camera_model",
            "camera_pose_estimation",
            "files",
        ],
    )
    camera_model = _camera_model_from_mapping(_require_mapping(row["camera_model"], context="samples.jsonl row camera_model"))
    if strict and not _camera_model_matches(camera_model_reference, camera_model):
        raise BatchSchemaError("Per-sample camera_model snapshot does not match metadata.json camera_model.")
    servo_positions = _float_triplet(row["servo_positions_deg"], context="samples.jsonl row servo_positions_deg")
    servo_targets = _float_triplet(row["servo_targets_deg"], context="samples.jsonl row servo_targets_deg")
    camera_frame = CameraFrame(
        frame_index=int(row["recording_frame_index"]),
        recording_frame_index=int(row["recording_frame_index"]),
        tick_index=int(row["tick_index"]),
        timestamp_s=float(row["sim_time_s"]),
        servo_positions_deg=servo_positions,
        servo_targets_deg=servo_targets,
        auto_demo_enabled=bool(_require_mapping(row["automation"], context="samples.jsonl row automation").get("auto_demo_enabled", False)),
        camera_model_name=camera_model.name,
    )

    detections = []
    detections_raw = row["detections"]
    if not isinstance(detections_raw, list):
        raise BatchSchemaError("samples.jsonl row detections must be a list.")
    for item in detections_raw:
        detection = _require_mapping(item, context="samples.jsonl row detections[]")
        _require_keys(
            detection,
            context="samples.jsonl row detections[]",
            required=["family", "tag_id", "corners_xy_clockwise", "center_xy", "points5_xy", "quality"],
        )
        tag_id = int(detection["tag_id"])
        tag_spec = tag_catalog.get(tag_id)
        if tag_spec is None:
            raise BatchSchemaError(f"Detection references unknown tag_id={tag_id}.")
        corners = tuple(
            _float_pair(point, context=f"samples.jsonl row detections[].corners_xy_clockwise[{index}]")
            for index, point in enumerate(detection["corners_xy_clockwise"])
        )
        points5 = tuple(
            _float_pair(point, context=f"samples.jsonl row detections[].points5_xy[{index}]")
            for index, point in enumerate(detection["points5_xy"])
        )
        pose_rvec = detection.get("pose_camera_rvec")
        pose_tvec = detection.get("pose_camera_tvec")
        detections.append(
            TagDetectionObservation(
                frame_index=camera_frame.frame_index,
                recording_frame_index=camera_frame.recording_frame_index,
                tick_index=camera_frame.tick_index,
                timestamp_s=camera_frame.timestamp_s,
                tag_id=tag_id,
                family=str(detection["family"]),
                corners_xy_clockwise_px=corners,
                center_xy_px=_float_pair(detection["center_xy"], context="samples.jsonl row detections[].center_xy"),
                points5_xy_px=points5,
                physical_edge_length_m=float(tag_spec.size_m),
                pose_camera_rvec_rad=None if pose_rvec is None else _float_triplet(pose_rvec, context="samples.jsonl row detections[].pose_camera_rvec"),
                pose_camera_tvec_m=None if pose_tvec is None else _float_triplet(pose_tvec, context="samples.jsonl row detections[].pose_camera_tvec"),
                quality=dict(detection["quality"]),
                camera_model_name=camera_model.name,
            )
        )
    return camera_frame, tuple(detections), camera_model


def _load_samples(
    run_dir: Path,
    *,
    camera_model_reference: CameraModelSnapshot,
    tag_catalog: dict[int, TagSpec],
    strict: bool,
) -> tuple[tuple[CameraFrame, ...], tuple[TagDetectionObservation, ...], CameraModelSnapshot]:
    samples_path = run_dir / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples.jsonl in run directory: {run_dir}")
    camera_frames: list[CameraFrame] = []
    detections: list[TagDetectionObservation] = []
    last_camera_model: CameraModelSnapshot | None = None
    with samples_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row = _require_mapping(row, context=f"samples.jsonl line {line_number}")
            camera_frame, line_detections, camera_model = _parse_sample_row(
                row,
                camera_model_reference=camera_model_reference,
                tag_catalog=tag_catalog,
                strict=strict,
            )
            camera_frames.append(camera_frame)
            detections.extend(line_detections)
            last_camera_model = camera_model
    if last_camera_model is None:
        raise BatchSchemaError("samples.jsonl did not contain any records.")
    return tuple(camera_frames), tuple(detections), last_camera_model


def _load_imu_packets(run_dir: Path) -> tuple[ImuPacket, ...]:
    imu_path = run_dir / "imu.csv"
    if not imu_path.exists():
        raise FileNotFoundError(f"Missing imu.csv in run directory: {run_dir}")
    with imu_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise BatchSchemaError("imu.csv did not contain any packets.")
    packets = []
    for row in rows:
        _require_keys(row, context="imu.csv row", required=["tick_index", "sim_time_s", "ax_mps2", "ay_mps2", "az_mps2", "gx_rps", "gy_rps", "gz_rps"])
        packets.append(
            ImuPacket(
                tick_index=int(row["tick_index"]),
                timestamp_s=float(row["sim_time_s"]),
                accel_body_mps2=_float_triplet([row["ax_mps2"], row["ay_mps2"], row["az_mps2"]], context="imu.csv accel_body_mps2"),
                gyro_body_rps=_float_triplet([row["gx_rps"], row["gy_rps"], row["gz_rps"]], context="imu.csv gyro_body_rps"),
            )
        )
    return tuple(packets)


def load_batch_dataset(run_dir: str | Path, *, strict: bool = True) -> BatchCalibrationDataset:
    resolved_run_dir = _resolve_repo_path(run_dir)
    if not resolved_run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {resolved_run_dir}")
    metadata = _load_metadata(resolved_run_dir)
    config_raw = _require_mapping(metadata["config"], context="metadata.json.config")
    camera_model = _camera_model_from_mapping(_require_mapping(metadata["camera_model"], context="metadata.json.camera_model"))
    device_config = load_device_config(config_raw["device_config_path"])
    tag_catalog = _load_tag_catalog(config_raw)
    camera_frames, detections, last_camera_model = _load_samples(
        resolved_run_dir,
        camera_model_reference=camera_model,
        tag_catalog=tag_catalog,
        strict=strict,
    )
    if strict and not _camera_model_matches(camera_model, last_camera_model):
        raise BatchSchemaError("Camera model snapshot drifted across samples.jsonl records.")
    imu_packets = _load_imu_packets(resolved_run_dir)
    recording_raw = _require_mapping(metadata.get("recording", {}), context="metadata.json.recording") if "recording" in metadata else {}
    frame_rate_hz = float(recording_raw.get("frame_rate_hz", config_raw.get("stream_fps", 0.0)))
    if strict and frame_rate_hz <= 0.0:
        raise BatchSchemaError("Recording frame_rate_hz must be positive.")
    return BatchCalibrationDataset(
        run_dir=resolved_run_dir,
        run_name=str(config_raw["name"]),
        schema_version=int(metadata.get("schema_version", CURRENT_SCHEMA_VERSION)),
        config_name=str(config_raw["name"]),
        recording_frame_rate_hz=frame_rate_hz,
        camera_model=camera_model,
        device_config=device_config,
        imu_convention=_imu_convention_from_metadata(metadata),
        tag_catalog=tag_catalog,
        camera_frames=camera_frames,
        tag_detections=detections,
        imu_packets=imu_packets,
    )


def _camera_truth_rotation_matrix(row: dict[str, Any]) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return (
        (
            float(row["r00"]),
            float(row["r01"]),
            float(row["r02"]),
        ),
        (
            float(row["r10"]),
            float(row["r11"]),
            float(row["r12"]),
        ),
        (
            float(row["r20"]),
            float(row["r21"]),
            float(row["r22"]),
        ),
    )


def _rotation_matrix_to_xyz_deg(rotation_cw: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]) -> tuple[float, float, float]:
    matrix = np.asarray(rotation_cw, dtype=np.float64).reshape(3, 3)
    sy = float(np.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0]))
    singular = sy < 1e-9
    if not singular:
        rx = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        ry = float(np.arctan2(-matrix[2, 0], sy))
        rz = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:
        rx = float(np.arctan2(-matrix[1, 2], matrix[1, 1]))
        ry = float(np.arctan2(-matrix[2, 0], sy))
        rz = 0.0
    return (float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz)))


def load_evaluation_data(run_dir: str | Path) -> BatchCalibrationEvaluationData:
    resolved_run_dir = _resolve_repo_path(run_dir)
    camera_gt_path = resolved_run_dir / "camera_gt.csv"
    samples_path = resolved_run_dir / "samples.jsonl"
    metadata = _load_metadata(resolved_run_dir)
    if not camera_gt_path.exists():
        raise FileNotFoundError(f"Missing camera_gt.csv in run directory: {resolved_run_dir}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples.jsonl in run directory: {resolved_run_dir}")

    with camera_gt_path.open("r", encoding="utf-8", newline="") as handle:
        camera_rows = list(csv.DictReader(handle))
    if not camera_rows:
        raise BatchSchemaError("camera_gt.csv did not contain any samples.")
    camera_truth = []
    for row in camera_rows:
        _require_keys(
            row,
            context="camera_gt.csv row",
            required=[
                "tick_index",
                "sim_time_s",
                "cx_world_m",
                "cy_world_m",
                "cz_world_m",
                "vx_world_mps",
                "vy_world_mps",
                "vz_world_mps",
                "r00",
                "r01",
                "r02",
                "r10",
                "r11",
                "r12",
                "r20",
                "r21",
                "r22",
                "servo0_deg",
                "servo1_deg",
                "servo2_deg",
            ],
        )
        camera_truth.append(
            CameraTruthSample(
                frame_index=int(row["tick_index"]),
                recording_frame_index=int(row["tick_index"]),
                tick_index=int(row["tick_index"]),
                timestamp_s=float(row["sim_time_s"]),
                position_world_m=(float(row["cx_world_m"]), float(row["cy_world_m"]), float(row["cz_world_m"])),
                velocity_world_mps=(float(row["vx_world_mps"]), float(row["vy_world_mps"]), float(row["vz_world_mps"])),
                rotation_cw=_camera_truth_rotation_matrix(row),
                rotation_xyz_deg=_rotation_matrix_to_xyz_deg(_camera_truth_rotation_matrix(row)),
                servo_positions_deg=(float(row["servo0_deg"]), float(row["servo1_deg"]), float(row["servo2_deg"])),
            )
        )

    tag_truth_by_frame: dict[int, list[TagTruthSample]] = {}
    with samples_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = _require_mapping(json.loads(stripped), context=f"samples.jsonl line {line_number}")
            if "ground_truth" not in row:
                continue
            gt = _require_mapping(row["ground_truth"], context=f"samples.jsonl line {line_number} ground_truth")
            tags = gt.get("tags") or []
            if not isinstance(tags, list):
                raise BatchSchemaError("samples.jsonl ground_truth.tags must be a list.")
            samples_for_frame = tag_truth_by_frame.setdefault(int(row["recording_frame_index"]), [])
            for tag in tags:
                item = _require_mapping(tag, context=f"samples.jsonl line {line_number} ground_truth.tags[]")
                _require_keys(
                    item,
                    context="samples.jsonl ground_truth.tags[]",
                    required=[
                        "tag_id",
                        "family",
                        "size_m",
                        "visible",
                        "tag_center_world_m",
                        "camera_position_tag_m",
                        "camera_rotation_tc",
                        "rendered_image_points_px",
                    ],
                )
                rendered_points = tuple(
                    _float_pair(point, context="samples.jsonl ground_truth.tags[].rendered_image_points_px[]")
                    for point in item["rendered_image_points_px"]
                )
                ideal_points = item.get("ideal_image_points_px")
                image_plane_points = item.get("image_plane_points_m")
                samples_for_frame.append(
                    TagTruthSample(
                        frame_index=int(row["recording_frame_index"]),
                        recording_frame_index=int(row["recording_frame_index"]),
                        tick_index=int(row["tick_index"]),
                        timestamp_s=float(row["sim_time_s"]),
                        tag_id=int(item["tag_id"]),
                        family=str(item["family"]),
                        size_m=float(item["size_m"]),
                        visible=bool(item["visible"]),
                        tag_center_world_m=_float_triplet(item["tag_center_world_m"], context="samples.jsonl ground_truth.tags[].tag_center_world_m"),
                        camera_position_tag_m=_float_triplet(item["camera_position_tag_m"], context="samples.jsonl ground_truth.tags[].camera_position_tag_m"),
                        camera_rotation_tc=(
                            _float_triplet(item["camera_rotation_tc"][0:3], context="samples.jsonl ground_truth.tags[].camera_rotation_tc[0]"),
                            _float_triplet(item["camera_rotation_tc"][3:6], context="samples.jsonl ground_truth.tags[].camera_rotation_tc[1]"),
                            _float_triplet(item["camera_rotation_tc"][6:9], context="samples.jsonl ground_truth.tags[].camera_rotation_tc[2]"),
                        ),
                        rendered_image_points_px=rendered_points,
                        ideal_image_points_px=None
                        if ideal_points is None
                        else tuple(_float_pair(point, context="samples.jsonl ground_truth.tags[].ideal_image_points_px[]") for point in ideal_points),
                        image_plane_points_m=None
                        if image_plane_points is None
                        else tuple(_float_pair(point, context="samples.jsonl ground_truth.tags[].image_plane_points_m[]") for point in image_plane_points),
                    )
                )
    imu_runtime_raw = _require_mapping(metadata.get("imu_runtime", {}), context="metadata.json.imu_runtime") if "imu_runtime" in metadata else {}
    accel_bias_truth = None
    gyro_bias_truth = None
    if imu_runtime_raw:
        accel_bias_initial = imu_runtime_raw.get("accel_bias_initial_mps2", imu_runtime_raw.get("accel_bias_mps2"))
        gyro_bias_initial = imu_runtime_raw.get("gyro_bias_initial_rps", imu_runtime_raw.get("gyro_bias_rps"))
        accel_walk = imu_runtime_raw.get("accel_bias_walk_std_mps2_per_sqrt_s", [0.0, 0.0, 0.0])
        gyro_walk = imu_runtime_raw.get("gyro_bias_walk_std_rps_per_sqrt_s", [0.0, 0.0, 0.0])
        if accel_bias_initial is not None and np.allclose(np.asarray(accel_walk, dtype=np.float64), 0.0):
            accel_bias_truth = _float_triplet(accel_bias_initial, context="metadata.json.imu_runtime.accel_bias_initial_mps2")
        if gyro_bias_initial is not None and np.allclose(np.asarray(gyro_walk, dtype=np.float64), 0.0):
            gyro_bias_truth = _float_triplet(gyro_bias_initial, context="metadata.json.imu_runtime.gyro_bias_initial_rps")

    return BatchCalibrationEvaluationData(
        run_dir=resolved_run_dir,
        camera_truth=tuple(camera_truth),
        tag_truth_by_frame={key: tuple(value) for key, value in tag_truth_by_frame.items()},
        global_gyro_bias_rps_truth=gyro_bias_truth,
        global_accel_bias_mps2_truth=accel_bias_truth,
    )
