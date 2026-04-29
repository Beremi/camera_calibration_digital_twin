"""Tabletop auto-demo dataset export helpers for the slim tabletop branch."""

from __future__ import annotations

import csv
from dataclasses import fields
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from calib_sim.estimation.noise_models import load_imu_noise_preset
from calib_sim.interactive.camera_model import PhoneCameraModel, load_phone_camera_model
from calib_sim.isaac.app import IsaacAppBootstrapConfig, create_runtime, load_isaac_yaml
from calib_sim.isaac.demo_profiles import get_demo_profile
from calib_sim.isaac.frontend.apriltag_frontend import IsaacAprilTagFrontend
from calib_sim.isaac.logging.run_manifest import build_run_manifest
from calib_sim.isaac.logging.writer import IsaacRunWriter
from calib_sim.tag_service.detector import (
    rendered_apriltag_face_corners_m,
    rendered_apriltag_geometry_metadata,
)
from calib_sim.reporting.tabletop_localization import (
    _camera_frame_packet_from_row,
    calibrate_tag_measurement_noise,
    export_isaac_tag_pose_measurements,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = REPO_ROOT / "tabletop_autodemo_dataset"
DETECTOR_BACKENDS = ("new_pupil",)
DEFAULT_PHONE_PROFILE_PATH = REPO_ROOT / "config" / "camera" / "pixel_9a_main.toml"
PHONE_CLEAN_VARIANT_NAME = "phone_clean"
_DEFAULT_IMAGE_NOISE_POLICY = {
    "name": "pixel_9a_low_light_h264",
    "target_mean_luma": 127.0,
    "mean_luma_blend": 0.25,
    "channel_gain_std": 0.01,
    "blur_sigma": 0.45,
    "luma_residual_std": 0.82,
    "chroma_residual_std": 0.60,
    "shot_noise_std_scale": 0.018,
    "jpeg_quality": 94,
    "target_laplacian_variance": 24.0,
    "target_h264_block_boundary_ratio": 1.14,
    "source_capture": "Beremi/bayesian_camera_IMU_calibration@8ae5a1232616273187badb6afaf13cde67b6d616",
}
_DEFAULT_SEED = 7
_PHONE_CAPTURE_FRAME_WIDTH = 640
_PHONE_CAPTURE_FRAME_HEIGHT = 480
_PHONE_CAPTURE_ROTATION_DEGREES = 0
_PHONE_CAPTURE_EXPOSURE_TIME_NS = 20_002_889
_PHONE_CAPTURE_FRAME_DURATION_NS = 33_488_214
_PHONE_CAPTURE_ROLLING_SHUTTER_SKEW_NS = 10_681_500
_PHONE_CAPTURE_FOCUS_DISTANCE_DIOPTERS = 0.2651071
_PHONE_CAPTURE_VIDEO_BIT_RATE_BPS = 11_920_098
_PHONE_CAPTURE_START_NANOS = 1_154_000_000_000_000
_PHONE_CAPTURE_VIDEO_START_OFFSET_NS = 441_003_874
_PHONE_CAPTURE_SENSOR_REGISTRATION_OFFSET_NS = 361_269_336
_PHONE_CAPTURE_ACCEL_BIAS = (-0.021678425, 0.03811153, 0.030581525)
_PHONE_CAPTURE_GYRO_BIAS = (-5.239102e-4, 2.0142319e-4, -0.0018390116)
_PHONE_CAMERA_RESULTS_FIELDNAMES = [
    "frame_number",
    "sensor_timestamp_nanos",
    "relative_session_nanos",
    "exposure_time_nanos",
    "sensitivity_iso",
    "frame_duration_nanos",
    "rolling_shutter_skew_nanos",
    "lens_focus_distance_diopters",
    "lens_state",
    "af_mode",
    "af_state",
    "ae_state",
    "awb_state",
    "video_stabilization_mode",
    "optical_stabilization_mode",
    "zoom_ratio",
    "crop_left",
    "crop_top",
    "crop_right",
    "crop_bottom",
    "active_physical_camera_id",
]
_PHONE_FRAMES_FIELDNAMES = [
    "frame_index",
    "camera_timestamp_nanos",
    "relative_session_nanos",
    "width",
    "height",
    "rotation_degrees",
]
_PHONE_IMU_FIELDNAMES = [
    "elapsed_realtime_nanos",
    "sensor_type",
    "x",
    "y",
    "z",
    "accuracy",
    "bias_x",
    "bias_y",
    "bias_z",
]
_GT_TAG_PROJECTIONS_PATH = Path("gt") / "tag_image_projections.jsonl"
_GT_RENDER_GEOMETRY_PATH = Path("gt") / "tag_render_geometry.json"
_CAMERA_PROJECTION_POSE_TRACE_PATH = Path("gt") / "camera_projection_pose_trace.jsonl"
_CAMERA_PROJECTION_GT_PATH = Path("gt") / "camera_projection_gt.csv"
_CAMERA_PROJECTION_SYNC_DIAGNOSTICS_PATH = Path("analysis") / "camera_projection_sync_diagnostics.jsonl"
_CAMERA_PROJECTION_SYNC_SUMMARY_PATH = Path("analysis") / "camera_projection_sync_summary.json"
_DETECTION_EXPORT_CHUNK_FRAMES = 60
_PROJECTION_POSE_LAG_FRAMES = 1


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _extract_json_object_from_text(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    candidate_indices = [index for index, char in enumerate(text) if char == "{"]
    for index in candidate_indices:
        try:
            payload, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _reset_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _symlink_or_copy(source: Path, target: Path) -> None:
    source = source.resolve()
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source, target_is_directory=source.is_dir())
        return
    except OSError:
        pass
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _camera_rate_hz(variant_dir: Path) -> float:
    camera_payload = _load_json(variant_dir / "config_snapshot" / "camera.json")
    return float(camera_payload.get("rate_hz", 60.0))


def _imu_rate_hz(variant_dir: Path) -> float:
    imu_payload = _load_json(variant_dir / "config_snapshot" / "imu.json")
    return float(imu_payload.get("rate_hz", 200.0))


def _scene_loop_duration_s(scene_payload: dict[str, Any]) -> float:
    motion_program = scene_payload.get("motion_program", {})
    if not isinstance(motion_program, dict):
        return 60.0
    return max(float(motion_program.get("loop_duration_s", 60.0)), 1.0)


def _load_frame_packets(run_dir: Path) -> list[Any]:
    return [_camera_frame_packet_from_row(row) for row in _load_jsonl(run_dir / "raw" / "camera_frames.jsonl")]


def _tag_size_lookup(run_dir: Path) -> dict[int, float]:
    tag_gt = _load_json(run_dir / "gt" / "tag_gt.json")
    tags = tag_gt.get("tags", [])
    if not isinstance(tags, list):
        return {}
    return {int(item["tag_id"]): float(item["size_m"]) for item in tags if isinstance(item, dict)}


def _set_variant_scene_frontend_option(variant_dir: Path, *, key: str, value: Any) -> None:
    scene_snapshot_path = variant_dir / "config_snapshot" / "scene.json"
    if not scene_snapshot_path.exists():
        return
    scene_payload = _load_json(scene_snapshot_path)
    frontend_config = scene_payload.get("frontend")
    if not isinstance(frontend_config, dict):
        frontend_config = {}
        scene_payload["frontend"] = frontend_config
    frontend_config[str(key)] = value
    _write_json(scene_snapshot_path, scene_payload)


def _repo_relative_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _phone_capture_nanos_from_timestamp(timestamp_s: float) -> int:
    return int(_PHONE_CAPTURE_START_NANOS + _PHONE_CAPTURE_VIDEO_START_OFFSET_NS + round(float(timestamp_s) * 1.0e9))


def _phone_capture_relative_nanos(timestamp_s: float) -> int:
    return int(_PHONE_CAPTURE_VIDEO_START_OFFSET_NS + round(float(timestamp_s) * 1.0e9))


def _phone_iso_for_frame(frame_ordinal: int) -> int:
    # Smooth deterministic replay of the ISO range measured in the Pixel 9a sample capture.
    phase = float(frame_ordinal) * 0.173
    iso = 468.7601626 + 58.5751845 * np.sin(phase) + 16.0 * np.sin(phase * 0.37 + 1.2)
    return int(np.clip(round(float(iso)), 332, 533))


def _phone_capture_variant_names(root: Path) -> tuple[str, ...]:
    ordered = ["clean", PHONE_CLEAN_VARIANT_NAME, "noisy"]
    return tuple(name for name in ordered if (root / name).exists())


def _clear_backend_outputs(variant_dir: Path) -> None:
    for path in (
        variant_dir / "raw" / "detections.jsonl",
        variant_dir / "localization",
        variant_dir / "analysis" / "camera_projection_sync_diagnostics.jsonl",
        variant_dir / "analysis" / "camera_projection_sync_summary.json",
    ):
        if path.exists() or path.is_symlink():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()


def _remove_variant_video_outputs(variant_dir: Path) -> None:
    for path in (
        variant_dir / "mounted_video.mp4",
        variant_dir / "mounted_video_timestamps.csv",
        variant_dir / "gt_path.csv",
        variant_dir / "phone_capture",
    ):
        if path.exists() or path.is_symlink():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()


def _load_phone_model(phone_profile_path: str | Path | None) -> PhoneCameraModel:
    return load_phone_camera_model(phone_profile_path or DEFAULT_PHONE_PROFILE_PATH)


def _rotation_matrix_from_quaternion_wxyz(quaternion_wxyz: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _order_corners_clockwise_top_left_first(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(4, 2)
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    clockwise = points[np.argsort(angles)[::-1]]
    start_index = int(np.argmin(clockwise[:, 0] + clockwise[:, 1]))
    return np.roll(clockwise, -start_index, axis=0)


def _best_cyclic_corner_alignment(reference_xy: np.ndarray, candidate_xy: np.ndarray) -> tuple[np.ndarray, float]:
    reference = np.asarray(reference_xy, dtype=np.float64).reshape(4, 2)
    candidate = np.asarray(candidate_xy, dtype=np.float64).reshape(4, 2)
    best_candidate = candidate
    best_rmse = float("inf")
    for orientation_candidate in (candidate, candidate[::-1]):
        for shift in range(4):
            rotated = np.roll(orientation_candidate, shift=shift, axis=0)
            rmse = float(np.sqrt(np.mean(np.sum(np.square(reference - rotated), axis=1))))
            if rmse < best_rmse:
                best_candidate = rotated
                best_rmse = rmse
    return best_candidate, best_rmse


def _camera_model_for_projection(camera_payload: dict[str, Any], frame_row: dict[str, Any]) -> Any:
    intrinsics_snapshot = dict(frame_row.get("intrinsics_snapshot", {}))
    intrinsics_config = dict(camera_payload.get("intrinsics", {}))
    fx_px = float(intrinsics_snapshot.get("fx_px", intrinsics_config.get("fx_px", 0.0)))
    fy_px = float(intrinsics_snapshot.get("fy_px", intrinsics_config.get("fy_px", 0.0)))
    cx_px = float(intrinsics_snapshot.get("cx_px", intrinsics_config.get("cx_px", 0.0)))
    cy_px = float(intrinsics_snapshot.get("cy_px", intrinsics_config.get("cy_px", 0.0)))
    distortion = intrinsics_snapshot.get("distortion_coefficients", camera_payload.get("distortion_coefficients", (0.0, 0.0, 0.0, 0.0, 0.0)))
    distortion_coefficients = tuple(float(value) for value in list(distortion)[:5])
    width_px = int(frame_row.get("image_width_px", camera_payload.get("width_px", 0)) or camera_payload.get("width_px", 0))
    height_px = int(frame_row.get("image_height_px", camera_payload.get("height_px", 0)) or camera_payload.get("height_px", 0))
    return SimpleNamespace(
        fx_px=fx_px,
        fy_px=fy_px,
        cx_px=cx_px,
        cy_px=cy_px,
        distortion_coefficients=distortion_coefficients,
        apply_lens_distortion_in_render=any(abs(value) > 1e-12 for value in distortion_coefficients),
    )


def _rendered_frame_bounds(frame_row: dict[str, Any], camera_payload: dict[str, Any]) -> tuple[int, int]:
    width_px = int(frame_row.get("image_width_px", camera_payload.get("width_px", 0)) or camera_payload.get("width_px", 0))
    height_px = int(frame_row.get("image_height_px", camera_payload.get("height_px", 0)) or camera_payload.get("height_px", 0))
    return width_px, height_px


def _projection_pose_row(frame_rows: list[dict[str, Any]], row_index: int) -> dict[str, Any]:
    projection_index = max(int(row_index) - int(_PROJECTION_POSE_LAG_FRAMES), 0)
    return frame_rows[projection_index]


def _projection_pose_trace_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace_rows: list[dict[str, Any]] = []
    for row_index, frame_row in enumerate(frame_rows):
        projection_pose_row = _projection_pose_row(frame_rows, row_index)
        raw_extrinsics = dict(frame_row.get("extrinsics_snapshot", {}))
        projection_extrinsics = dict(projection_pose_row.get("extrinsics_snapshot", {}))
        trace_rows.append(
            {
                "frame_index": int(frame_row["frame_index"]),
                "timestamp_s": float(frame_row["timestamp_s"]),
                "sim_time_s": float(frame_row.get("sim_time_s", frame_row["timestamp_s"])),
                "sensor_time_s": float(frame_row.get("sensor_time_s", frame_row["timestamp_s"])),
                "raw_pose_frame_index": int(frame_row["frame_index"]),
                "raw_pose_timestamp_s": float(frame_row["timestamp_s"]),
                "raw_pose_sensor_time_s": float(frame_row.get("sensor_time_s", frame_row["timestamp_s"])),
                "projection_pose_frame_index": int(projection_pose_row["frame_index"]),
                "projection_pose_timestamp_s": float(projection_pose_row["timestamp_s"]),
                "projection_pose_sensor_time_s": float(
                    projection_pose_row.get("sensor_time_s", projection_pose_row["timestamp_s"])
                ),
                "projection_pose_lag_frames": int(frame_row["frame_index"]) - int(projection_pose_row["frame_index"]),
                "raw_position_world_m": [
                    float(value) for value in raw_extrinsics.get("position_world_m", [0.0, 0.0, 0.0])
                ],
                "raw_orientation_wxyz": [
                    float(value) for value in raw_extrinsics.get("orientation_wxyz", [1.0, 0.0, 0.0, 0.0])
                ],
                "projection_position_world_m": [
                    float(value) for value in projection_extrinsics.get("position_world_m", [0.0, 0.0, 0.0])
                ],
                "projection_orientation_wxyz": [
                    float(value) for value in projection_extrinsics.get("orientation_wxyz", [1.0, 0.0, 0.0, 0.0])
                ],
            }
        )
    return trace_rows


def _projection_camera_gt_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, frame_row in enumerate(frame_rows):
        projection_pose_row = _projection_pose_row(frame_rows, row_index)
        extrinsics = dict(projection_pose_row.get("extrinsics_snapshot", {}))
        position_world_m = list(extrinsics.get("position_world_m", [0.0, 0.0, 0.0]))
        orientation_wxyz = list(extrinsics.get("orientation_wxyz", [1.0, 0.0, 0.0, 0.0]))
        rows.append(
            {
                "frame_index": int(frame_row["frame_index"]),
                "projection_pose_frame_index": int(projection_pose_row["frame_index"]),
                "timestamp_s": float(frame_row["timestamp_s"]),
                "sim_time_s": float(frame_row.get("sim_time_s", frame_row["timestamp_s"])),
                "sensor_time_s": float(frame_row.get("sensor_time_s", frame_row["timestamp_s"])),
                "projection_pose_timestamp_s": float(projection_pose_row["timestamp_s"]),
                "projection_pose_sensor_time_s": float(
                    projection_pose_row.get("sensor_time_s", projection_pose_row["timestamp_s"])
                ),
                "px": float(position_world_m[0]),
                "py": float(position_world_m[1]),
                "pz": float(position_world_m[2]),
                "qw": float(orientation_wxyz[0]),
                "qx": float(orientation_wxyz[1]),
                "qy": float(orientation_wxyz[2]),
                "qz": float(orientation_wxyz[3]),
            }
        )
    return rows


def _usd_camera_points_to_repo_camera_frame(points_camera_usd: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera_usd, dtype=np.float64).reshape(-1, 3)
    # Isaac camera world poses are authored in USD camera axes:
    # +X right, +Y up, -Z forward. Our projection helper expects
    # +X right, +Y up, +Z forward, so only the forward axis sign flips.
    return np.column_stack((points[:, 0], points[:, 1], -points[:, 2])).astype(np.float64)


def _project_camera_points_to_pixels(camera_model: Any, camera_points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_camera = np.asarray(camera_points_m, dtype=np.float64).reshape(-1, 3)
    visible = np.asarray(points_camera[:, 2] > 0.05, dtype=bool)
    if bool(getattr(camera_model, "apply_lens_distortion_in_render", False)):
        points_camera_cv = np.column_stack(
            (points_camera[:, 0], -points_camera[:, 1], points_camera[:, 2])
        ).astype(np.float64)
        camera_matrix = np.array(
            [
                [float(camera_model.fx_px), float(getattr(camera_model, "skew_px", 0.0)), float(camera_model.cx_px)],
                [0.0, float(camera_model.fy_px), float(camera_model.cy_px)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.asarray(
            getattr(camera_model, "distortion_coefficients", (0.0, 0.0, 0.0, 0.0, 0.0)),
            dtype=np.float64,
        ).reshape(-1)
        projected_pixels, _ = cv2.projectPoints(
            points_camera_cv.reshape(-1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            camera_matrix,
            distortion,
        )
        return projected_pixels.reshape(-1, 2).astype(np.float64), visible

    z = np.maximum(points_camera[:, 2], 1e-6)
    u = float(camera_model.fx_px) * (points_camera[:, 0] / z) + float(camera_model.cx_px)
    v = float(camera_model.fy_px) * (-(points_camera[:, 1] / z)) + float(camera_model.cy_px)
    return np.column_stack((u, v)).astype(np.float64), visible


def _rendered_tag_world_corners(tag_row: dict[str, Any]) -> np.ndarray:
    rotation_wt = np.asarray(tag_row["rotation_wt"], dtype=np.float64).reshape(3, 3)
    position_world_m = np.asarray(tag_row["position_world_m"], dtype=np.float64).reshape(3)
    local_corners = rendered_apriltag_face_corners_m(float(tag_row["size_m"]))
    return (rotation_wt @ local_corners.T).T + position_world_m.reshape(1, 3)


def _project_rendered_tag_geometry(
    *,
    tag_row: dict[str, Any],
    image_frame_row: dict[str, Any],
    projection_pose_row: dict[str, Any],
    camera_payload: dict[str, Any],
) -> dict[str, Any]:
    world_corners = _rendered_tag_world_corners(tag_row)
    extrinsics = dict(projection_pose_row.get("extrinsics_snapshot", {}))
    camera_position_world_m = np.asarray(extrinsics.get("position_world_m", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
    camera_rotation_wc = _rotation_matrix_from_quaternion_wxyz(extrinsics.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0)))
    camera_model = _camera_model_for_projection(camera_payload, image_frame_row)
    width_px, height_px = _rendered_frame_bounds(image_frame_row, camera_payload)
    camera_corners_usd = (camera_rotation_wc.T @ (world_corners - camera_position_world_m.reshape(1, 3)).T).T
    camera_corners = _usd_camera_points_to_repo_camera_frame(camera_corners_usd)
    projected_corners_xy, _ = _project_camera_points_to_pixels(camera_model, camera_corners)
    ordered_corners_xy = _order_corners_clockwise_top_left_first(projected_corners_xy)
    face_center_world_m = np.mean(world_corners, axis=0)
    camera_center_usd = camera_rotation_wc.T @ (face_center_world_m - camera_position_world_m)
    camera_center = _usd_camera_points_to_repo_camera_frame(camera_center_usd.reshape(1, 3)).reshape(3)
    projected_center_xy, _ = _project_camera_points_to_pixels(camera_model, camera_center.reshape(1, 3))
    all_finite = bool(np.isfinite(ordered_corners_xy).all())
    center_finite = bool(np.isfinite(projected_center_xy).all())
    corners_in_front = bool(np.all(camera_corners[:, 2] > 0.05))
    center_in_front = bool(float(camera_center[2]) > 0.05)
    in_bounds = (
        (ordered_corners_xy[:, 0] >= 0.0)
        & (ordered_corners_xy[:, 0] < float(max(width_px, 1)))
        & (ordered_corners_xy[:, 1] >= 0.0)
        & (ordered_corners_xy[:, 1] < float(max(height_px, 1)))
    )
    any_corner_in_frame = bool(np.any(in_bounds)) if all_finite else False
    all_corners_in_frame = bool(np.all(in_bounds)) if all_finite else False
    fully_visible = bool(all_finite and center_finite and corners_in_front and center_in_front and all_corners_in_frame)
    partially_visible = bool(all_finite and corners_in_front and any_corner_in_frame and not all_corners_in_frame)
    return {
        "corners_xy": ordered_corners_xy,
        "center_xy": projected_center_xy.reshape(-1, 2)[0],
        "visibility_flags": {
            "corners_in_front": corners_in_front,
            "center_in_front": center_in_front,
            "any_corner_in_frame": any_corner_in_frame,
            "all_corners_in_frame": all_corners_in_frame,
            "fully_visible": fully_visible,
            "partially_visible": partially_visible,
        },
    }


def _compute_gt_tag_image_projections(variant_dir: Path) -> list[dict[str, Any]]:
    camera_payload = _load_json(variant_dir / "config_snapshot" / "camera.json")
    frame_rows = _load_jsonl(variant_dir / "raw" / "camera_frames.jsonl")
    tag_gt = _load_json(variant_dir / "gt" / "tag_gt.json")
    tags = [dict(item) for item in tag_gt.get("tags", []) if isinstance(item, dict)]
    geometry_metadata = rendered_apriltag_geometry_metadata()
    _write_json(
        variant_dir / _GT_RENDER_GEOMETRY_PATH,
        {
            **geometry_metadata,
            "projection_pose_lag_frames": int(_PROJECTION_POSE_LAG_FRAMES),
            "projection_pose_source": "previous_frame_extrinsics_snapshot",
        },
    )
    trace_rows = _projection_pose_trace_rows(frame_rows)
    _write_jsonl(variant_dir / _CAMERA_PROJECTION_POSE_TRACE_PATH, trace_rows)
    projection_camera_gt_rows = _projection_camera_gt_rows(frame_rows)
    _write_csv(
        variant_dir / _CAMERA_PROJECTION_GT_PATH,
        [
            "frame_index",
            "projection_pose_frame_index",
            "timestamp_s",
            "sim_time_s",
            "sensor_time_s",
            "projection_pose_timestamp_s",
            "projection_pose_sensor_time_s",
            "px",
            "py",
            "pz",
            "qw",
            "qx",
            "qy",
            "qz",
        ],
        projection_camera_gt_rows,
    )
    projection_rows: list[dict[str, Any]] = []
    updated_frame_rows: list[dict[str, Any]] = []
    for row_index, frame_row in enumerate(frame_rows):
        projection_pose_row = _projection_pose_row(frame_rows, row_index)
        fully_visible_tag_ids: list[int] = []
        for tag in tags:
            tag_id = int(tag["tag_id"])
            tag_size_m = float(tag["size_m"])
            projected = _project_rendered_tag_geometry(
                tag_row=tag,
                image_frame_row=frame_row,
                projection_pose_row=projection_pose_row,
                camera_payload=camera_payload,
            )
            visibility = dict(projected["visibility_flags"])
            fully_visible = bool(visibility.get("fully_visible", False))
            if fully_visible:
                fully_visible_tag_ids.append(tag_id)
            projection_rows.append(
                {
                    "frame_index": int(frame_row["frame_index"]),
                    "timestamp_s": float(frame_row["timestamp_s"]),
                    "sim_time_s": float(frame_row.get("sim_time_s", frame_row["timestamp_s"])),
                    "sensor_time_s": float(frame_row.get("sensor_time_s", frame_row["timestamp_s"])),
                    "projection_pose_frame_index": int(projection_pose_row["frame_index"]),
                    "projection_pose_timestamp_s": float(projection_pose_row["timestamp_s"]),
                    "tag_id": tag_id,
                    "is_anchor": bool(tag.get("is_anchor", False)),
                    "tag_size_m": tag_size_m,
                    "corners_xy": [[float(x), float(y)] for x, y in np.asarray(projected["corners_xy"], dtype=np.float64).tolist()],
                    "center_xy": [float(value) for value in np.asarray(projected["center_xy"], dtype=np.float64).tolist()],
                    "visibility_flags": {str(key): bool(value) for key, value in visibility.items()},
                }
            )
        updated_frame = dict(frame_row)
        updated_frame["visible_gt_tag_ids"] = [int(tag_id) for tag_id in sorted(fully_visible_tag_ids)]
        updated_frame_rows.append(updated_frame)
    _write_jsonl(variant_dir / "raw" / "camera_frames.jsonl", updated_frame_rows)
    _write_jsonl(variant_dir / _GT_TAG_PROJECTIONS_PATH, projection_rows)
    return projection_rows


def _frontend_from_scene_config(anchor_tag_id: int, frontend_config: dict[str, Any]) -> IsaacAprilTagFrontend:
    allowed_kwargs: dict[str, Any] = {}
    for item in fields(IsaacAprilTagFrontend):
        if not item.init or item.name == "anchor_tag_id":
            continue
        if item.name in frontend_config:
            allowed_kwargs[item.name] = frontend_config[item.name]
    return IsaacAprilTagFrontend(anchor_tag_id=int(anchor_tag_id), **allowed_kwargs)


def _backend_frontend(anchor_tag_id: int, scene_payload: dict[str, Any], backend: str) -> IsaacAprilTagFrontend:
    normalized = str(backend).strip().lower()
    if normalized != "new_pupil":
        raise ValueError(f"Unsupported detector backend: {backend}")
    frontend_config = dict(scene_payload.get("frontend", {}))
    frontend_config["detector_backend"] = "pupil_apriltags"
    frontend_config["roi_recovery_enabled"] = True
    return _frontend_from_scene_config(int(anchor_tag_id), frontend_config)


def _frontend_summary_from_detections(
    frame_rows: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
    *,
    anchor_tag_id: int,
) -> dict[str, Any]:
    frames_total = int(len(frame_rows))
    detections_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in detection_rows:
        detections_by_frame.setdefault(int(row["frame_index"]), []).append(row)
    anchor_visible_frames = sum(
        1
        for frame_index, rows in detections_by_frame.items()
        if any(int(row["tag_id"]) == int(anchor_tag_id) for row in rows)
    )
    detection_failures = sum(1 for row in frame_rows if int(row["frame_index"]) not in detections_by_frame)
    return {
        "frames_processed": frames_total,
        "mean_detections_per_frame": 0.0 if frames_total <= 0 else float(len(detection_rows) / frames_total),
        "anchor_visible_ratio": 0.0 if frames_total <= 0 else float(anchor_visible_frames / frames_total),
        "detection_failures": int(detection_failures),
    }


def _encode_video_from_variant_frames(variant_dir: Path) -> None:
    frame_rows = _load_jsonl(variant_dir / "raw" / "camera_frames.jsonl")
    frame_rows = sorted(frame_rows, key=lambda item: int(item.get("frame_index", 0)))
    if not frame_rows:
        raise ValueError(f"No camera frames found in {variant_dir / 'raw' / 'camera_frames.jsonl'}.")
    timestamp_rows: list[dict[str, Any]] = []
    first_image = cv2.imread(str(variant_dir / str(frame_rows[0]["rgb_path"])))
    if first_image is None:
        raise FileNotFoundError(f"Could not read first frame image for video export in {variant_dir}.")
    height_px, width_px = first_image.shape[:2]
    fps = _camera_rate_hz(variant_dir)
    writer = cv2.VideoWriter(
        str(variant_dir / "mounted_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width_px), int(height_px)),
    )
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {variant_dir / 'mounted_video.mp4'}.")
        for row in frame_rows:
            image_path = variant_dir / str(row["rgb_path"])
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise FileNotFoundError(f"Could not read frame image {image_path}.")
            if image_bgr.shape[1] != width_px or image_bgr.shape[0] != height_px:
                image_bgr = cv2.resize(image_bgr, (int(width_px), int(height_px)), interpolation=cv2.INTER_AREA)
            writer.write(image_bgr)
            timestamp_rows.append(
                {
                    "frame_index": int(row["frame_index"]),
                    "timestamp_s": float(row["timestamp_s"]),
                    "sim_time_s": float(row.get("sim_time_s", row["timestamp_s"])),
                    "sensor_time_s": float(row.get("sensor_time_s", row["timestamp_s"])),
                    "host_time_s": _float_or_none(row.get("host_time_s")),
                    "rgb_path": str(row["rgb_path"]),
                }
            )
    finally:
        writer.release()
    _write_csv(
        variant_dir / "mounted_video_timestamps.csv",
        ["frame_index", "timestamp_s", "sim_time_s", "sensor_time_s", "host_time_s", "rgb_path"],
        timestamp_rows,
    )


def _write_gt_path_export(variant_dir: Path) -> None:
    camera_gt_path = variant_dir / _CAMERA_PROJECTION_GT_PATH
    if not camera_gt_path.exists():
        camera_gt_path = variant_dir / "gt" / "camera_gt.csv"
    fieldnames, rows = _load_csv(camera_gt_path)
    if not rows:
        raise ValueError(f"No camera GT rows found in {camera_gt_path}.")
    _write_csv(variant_dir / "gt_path.csv", fieldnames, rows)


def write_dataset_convenience_exports(variant_dir: str | Path) -> None:
    resolved = Path(variant_dir).resolve()
    _encode_video_from_variant_frames(resolved)
    _write_gt_path_export(resolved)


def _draw_detection_label(
    image_bgr: np.ndarray,
    *,
    text: str,
    origin_xy: tuple[int, int],
    color_bgr: tuple[int, int, int],
) -> None:
    x_px, y_px = origin_xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x0 = int(max(x_px, 0))
    y0 = int(max(y_px - text_height - baseline - 6, 0))
    x1 = int(min(x0 + text_width + 10, image_bgr.shape[1] - 1))
    y1 = int(min(y0 + text_height + baseline + 8, image_bgr.shape[0] - 1))
    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), color_bgr, 1)
    cv2.putText(
        image_bgr,
        text,
        (x0 + 5, y1 - baseline - 4),
        font,
        scale,
        color_bgr,
        thickness,
        cv2.LINE_AA,
    )


def _draw_detection_overlay_frame(
    image_bgr: np.ndarray,
    *,
    frame_row: dict[str, Any],
    detection_rows: list[dict[str, Any]],
    backend: str,
) -> np.ndarray:
    output = image_bgr.copy()
    palette = [
        (0, 220, 255),
        (0, 160, 255),
        (60, 255, 120),
        (255, 170, 70),
        (255, 90, 220),
        (120, 220, 255),
    ]
    for detection_index, detection in enumerate(detection_rows):
        corners = np.asarray(detection.get("corners_xy", ()), dtype=np.float64).reshape(-1, 2)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            continue
        color = palette[int(detection.get("tag_id", detection_index)) % len(palette)]
        polygon = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [polygon], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
        for corner_index, corner_xy in enumerate(corners):
            corner = tuple(int(value) for value in np.round(corner_xy).tolist())
            cv2.circle(output, corner, 5, color, -1, cv2.LINE_AA)
            cv2.putText(
                output,
                str(corner_index),
                (corner[0] + 6, corner[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        center_xy = np.mean(corners, axis=0)
        center = tuple(int(value) for value in np.round(center_xy).tolist())
        cv2.circle(output, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
        score = detection.get("score")
        label = f"id {int(detection.get('tag_id', -1))}"
        if score not in (None, ""):
            label += f"  {float(score):.2f}"
        _draw_detection_label(
            output,
            text=label,
            origin_xy=(int(np.min(corners[:, 0])), int(np.min(corners[:, 1]))),
            color_bgr=color,
        )

    header = (
        f"{backend} detections | frame {int(frame_row.get('frame_index', 0))} | "
        f"{len(detection_rows)} tag(s)"
    )
    cv2.rectangle(output, (0, 0), (min(output.shape[1] - 1, 760), 42), (0, 0, 0), -1)
    cv2.putText(
        output,
        header,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def write_detection_overlay_video(
    variant_dir: str | Path,
    *,
    backend: str = "new_pupil",
    detections_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved = Path(variant_dir).resolve()
    backend_dir = resolved / "localization" / str(backend)
    backend_dir.mkdir(parents=True, exist_ok=True)
    output_path = backend_dir / "detections_overlay.mp4"
    frame_rows = sorted(
        _load_jsonl(resolved / "raw" / "camera_frames.jsonl"),
        key=lambda row: int(row.get("frame_index", 0)),
    )
    if not frame_rows:
        raise ValueError(f"No frame rows found for detection overlay video: {resolved}")
    detection_source_path = Path(detections_path).resolve() if detections_path is not None else backend_dir / "detections.jsonl"
    detection_rows = _load_jsonl(detection_source_path)
    detections_by_frame: dict[int, list[dict[str, Any]]] = {}
    for detection in detection_rows:
        detections_by_frame.setdefault(int(detection.get("frame_index", 0)), []).append(dict(detection))

    first_image = cv2.imread(str(resolved / str(frame_rows[0]["rgb_path"])), cv2.IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError(f"Could not read first frame for detection overlay: {resolved / str(frame_rows[0]['rgb_path'])}")
    height_px, width_px = first_image.shape[:2]
    fps = float(_camera_rate_hz(resolved))
    ffmpeg_path = shutil.which("ffmpeg")
    process: subprocess.Popen[bytes] | None = None
    writer: cv2.VideoWriter | None = None
    encoder = "opencv/mp4v"
    fallback = True
    if ffmpeg_path:
        cmd = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{int(width_px)}x{int(height_px)}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-bf",
            "0",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        process = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdin=subprocess.PIPE)
        encoder = "ffmpeg/libx264"
        fallback = False
    else:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (int(width_px), int(height_px)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open detection overlay video writer for {output_path}.")

    try:
        for frame_row in frame_rows:
            image_path = resolved / str(frame_row["rgb_path"])
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Could not read frame for detection overlay: {image_path}")
            if image_bgr.shape[1] != width_px or image_bgr.shape[0] != height_px:
                image_bgr = cv2.resize(image_bgr, (int(width_px), int(height_px)), interpolation=cv2.INTER_AREA)
            overlay = _draw_detection_overlay_frame(
                image_bgr,
                frame_row=frame_row,
                detection_rows=detections_by_frame.get(int(frame_row.get("frame_index", 0)), []),
                backend=str(backend),
            )
            if process is not None:
                if process.stdin is None:
                    raise RuntimeError("FFmpeg stdin closed before writing detection overlay frames.")
                process.stdin.write(overlay.tobytes())
            elif writer is not None:
                writer.write(overlay)
    finally:
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"FFmpeg detection overlay encode failed with exit code {return_code}: {output_path}")
        if writer is not None:
            writer.release()

    return {
        "path": str(output_path),
        "relative_path": f"localization/{backend}/detections_overlay.mp4",
        "backend": str(backend),
        "encoder": encoder,
        "fallback": bool(fallback),
        "width_px": int(width_px),
        "height_px": int(height_px),
        "fps": fps,
        "frame_count": int(len(frame_rows)),
        "detection_count": int(len(detection_rows)),
    }


def _phone_distortion_enabled(phone_model: PhoneCameraModel) -> bool:
    return bool(phone_model.apply_lens_distortion_in_render) and any(
        abs(float(value)) > 1e-12 for value in phone_model.distortion_coefficients
    )


def _phone_camera_intrinsics_payload(phone_model: PhoneCameraModel, *, include_distortion: bool) -> dict[str, Any]:
    return {
        "fx_px": float(phone_model.fx_px),
        "fy_px": float(phone_model.fy_px),
        "cx_px": float(phone_model.cx_px),
        "cy_px": float(phone_model.cy_px),
        "skew_px": float(phone_model.skew_px),
        "distortion_model": "opencv_brown_conrady" if bool(include_distortion) else "opencv_pinhole",
        "distortion_coefficients": [float(value) for value in (phone_model.distortion_coefficients if include_distortion else (0.0, 0.0, 0.0, 0.0, 0.0))],
        "encoded_rotation_degrees": int(phone_model.encoded_rotation_degrees),
    }


def _phone_effect_profile(phone_model: PhoneCameraModel, *, physical: bool, noisy: bool) -> dict[str, Any]:
    distortion_applied = bool(physical) and _phone_distortion_enabled(phone_model)
    return {
        "name": "pixel_9a_processed_video_v1" if bool(physical) else "ideal_pinhole_with_phone_sidecar",
        "source_toml_path": str(phone_model.source_toml_path),
        "output_resolution_px": [int(phone_model.output_width_px), int(phone_model.output_height_px)],
        "distortion_applied_to_rgb": bool(distortion_applied),
        "distortion_model": "opencv_brown_conrady" if bool(distortion_applied) else "opencv_pinhole",
        "distortion_coefficients": [
            float(value)
            for value in (
                phone_model.distortion_coefficients
                if distortion_applied
                else (0.0, 0.0, 0.0, 0.0, 0.0)
            )
        ],
        "source_distortion_metadata": {
            "model": str(phone_model.distortion_model),
            "coefficients": [float(value) for value in phone_model.distortion_coefficients],
            "apply_lens_distortion_in_render": bool(phone_model.apply_lens_distortion_in_render),
        },
        "rolling_shutter": {
            "approximation": "rotation_dominant_image_space_band_remap" if bool(physical) else "metadata_only",
            "skew_nanos": int(_PHONE_CAPTURE_ROLLING_SHUTTER_SKEW_NS),
            "exact_per_row_rendering": False,
        },
        "low_light_noise_applied": bool(noisy),
        "low_light_noise_policy": dict(_DEFAULT_IMAGE_NOISE_POLICY) if bool(noisy) else {"name": "none"},
    }


def _update_variant_phone_camera_metadata(variant_dir: Path, phone_model: PhoneCameraModel, *, include_distortion: bool) -> list[dict[str, Any]]:
    camera_path = variant_dir / "config_snapshot" / "camera.json"
    camera_payload = _load_json(camera_path)
    intrinsics_payload = _phone_camera_intrinsics_payload(phone_model, include_distortion=bool(include_distortion))
    camera_payload.update(
        {
            "source_toml_path": str(phone_model.source_toml_path),
            "width_px": int(phone_model.output_width_px),
            "height_px": int(phone_model.output_height_px),
            "rate_hz": float(phone_model.target_frames_per_second),
            "focal_length_mm": float(phone_model.focal_length_mm),
            "distortion_model": str(intrinsics_payload["distortion_model"]),
            "distortion_coefficients": list(intrinsics_payload["distortion_coefficients"]),
            "intrinsics": {
                "fx_px": float(phone_model.fx_px),
                "fy_px": float(phone_model.fy_px),
                "cx_px": float(phone_model.cx_px),
                "cy_px": float(phone_model.cy_px),
                "skew_px": float(phone_model.skew_px),
            },
            "phone_video_mode": {
                "encoded_rotation_degrees": int(phone_model.encoded_rotation_degrees),
                "crop_rect_active_array_px": {
                    "x": float(phone_model.crop_x_px),
                    "y": float(phone_model.crop_y_px),
                    "width": float(phone_model.crop_width_px),
                    "height": float(phone_model.crop_height_px),
                },
            },
        }
    )
    _write_json(camera_path, camera_payload)

    frame_rows = _load_jsonl(variant_dir / "raw" / "camera_frames.jsonl")
    updated_rows: list[dict[str, Any]] = []
    for row in frame_rows:
        updated = dict(row)
        updated["image_width_px"] = int(phone_model.output_width_px)
        updated["image_height_px"] = int(phone_model.output_height_px)
        intrinsics_snapshot = dict(updated.get("intrinsics_snapshot", {}))
        intrinsics_snapshot.update(intrinsics_payload)
        updated["intrinsics_snapshot"] = intrinsics_snapshot
        updated_rows.append(updated)
    _write_jsonl(variant_dir / "raw" / "camera_frames.jsonl", updated_rows)
    return updated_rows


def _resize_center_crop_to_resolution(
    image_bgr: np.ndarray,
    *,
    width_px: int,
    height_px: int,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    target_width = int(width_px)
    target_height = int(height_px)
    if image_bgr.shape[1] == target_width and image_bgr.shape[0] == target_height:
        return image_bgr
    source_height, source_width = image_bgr.shape[:2]
    target_aspect = float(target_width) / max(float(target_height), 1.0)
    source_aspect = float(source_width) / max(float(source_height), 1.0)
    if source_aspect > target_aspect:
        crop_width = max(int(round(float(source_height) * target_aspect)), 1)
        crop_x0 = max((source_width - crop_width) // 2, 0)
        cropped = image_bgr[:, crop_x0 : crop_x0 + crop_width]
    else:
        crop_height = max(int(round(float(source_width) / target_aspect)), 1)
        crop_y0 = max((source_height - crop_height) // 2, 0)
        cropped = image_bgr[crop_y0 : crop_y0 + crop_height, :]
    return cv2.resize(cropped, (target_width, target_height), interpolation=interpolation)


def _relative_rotation_vector_between_frames(frame_rows: list[dict[str, Any]], row_index: int) -> np.ndarray:
    if len(frame_rows) < 2:
        return np.zeros(3, dtype=np.float64)
    current = dict(frame_rows[row_index].get("extrinsics_snapshot", {}))
    neighbor_index = min(row_index + 1, len(frame_rows) - 1)
    if neighbor_index == row_index:
        neighbor_index = max(row_index - 1, 0)
    neighbor = dict(frame_rows[neighbor_index].get("extrinsics_snapshot", {}))
    current_q = current.get("orientation_wxyz")
    neighbor_q = neighbor.get("orientation_wxyz")
    if current_q is None or neighbor_q is None:
        return np.zeros(3, dtype=np.float64)
    current_rotation = _rotation_matrix_from_quaternion_wxyz(current_q)
    neighbor_rotation = _rotation_matrix_from_quaternion_wxyz(neighbor_q)
    relative = neighbor_rotation @ current_rotation.T
    rotation_vector, _ = cv2.Rodrigues(relative.astype(np.float64))
    if neighbor_index < row_index:
        rotation_vector = -rotation_vector
    return np.asarray(rotation_vector, dtype=np.float64).reshape(3)


def _rolling_shutter_warp_bgr(
    image_bgr: np.ndarray,
    *,
    frame_rows: list[dict[str, Any]],
    row_index: int,
    phone_model: PhoneCameraModel,
) -> np.ndarray:
    rotation_vector = _relative_rotation_vector_between_frames(frame_rows, row_index)
    if float(np.linalg.norm(rotation_vector)) <= 1e-12:
        return image_bgr
    current_time = float(frame_rows[row_index].get("timestamp_s", 0.0))
    if row_index + 1 < len(frame_rows):
        dt_s = abs(float(frame_rows[row_index + 1].get("timestamp_s", current_time)) - current_time)
    elif row_index > 0:
        dt_s = abs(current_time - float(frame_rows[row_index - 1].get("timestamp_s", current_time)))
    else:
        dt_s = 1.0 / max(float(phone_model.target_frames_per_second), 1.0)
    rolling_fraction = float((_PHONE_CAPTURE_ROLLING_SHUTTER_SKEW_NS * 1.0e-9) / max(dt_s, 1e-6))
    height_px, width_px = image_bgr.shape[:2]
    row_alpha = (np.arange(height_px, dtype=np.float32) / max(float(height_px - 1), 1.0)) - 0.5
    shift_x = np.clip(float(rotation_vector[1]) * float(phone_model.fx_px) * rolling_fraction * row_alpha, -8.0, 8.0)
    shift_y = np.clip(-float(rotation_vector[0]) * float(phone_model.fy_px) * rolling_fraction * row_alpha, -8.0, 8.0)
    grid_x, grid_y = np.meshgrid(np.arange(width_px, dtype=np.float32), np.arange(height_px, dtype=np.float32))
    map_x = grid_x - shift_x.reshape(-1, 1)
    map_y = grid_y - shift_y.reshape(-1, 1)
    return cv2.remap(
        image_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _apply_phone_physical_rgb_stream(variant_dir: Path, phone_model: PhoneCameraModel, *, include_rolling_shutter: bool) -> None:
    frame_rows = _load_jsonl(variant_dir / "raw" / "camera_frames.jsonl")
    remap_x = getattr(phone_model, "_remap_x")
    remap_y = getattr(phone_model, "_remap_y")
    apply_distortion = _phone_distortion_enabled(phone_model)
    for row_index, row in enumerate(frame_rows):
        image_path = variant_dir / str(row["rgb_path"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read RGB frame for phone postprocess: {image_path}")
        processed = _resize_center_crop_to_resolution(
            image_bgr,
            width_px=int(phone_model.output_width_px),
            height_px=int(phone_model.output_height_px),
            interpolation=cv2.INTER_CUBIC,
        )
        if apply_distortion:
            processed = cv2.remap(
                processed,
                remap_x,
                remap_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        if bool(include_rolling_shutter):
            processed = _rolling_shutter_warp_bgr(
                processed,
                frame_rows=frame_rows,
                row_index=row_index,
                phone_model=phone_model,
            )
        if not cv2.imwrite(str(image_path), processed):
            raise RuntimeError(f"Failed to write phone-processed RGB frame {image_path}.")


def apply_phone_physical_variant(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    phone_profile_path: str | Path | None = None,
    seed: int = _DEFAULT_SEED,
) -> dict[str, Any]:
    source_path = Path(source_dir).resolve()
    target_path = Path(target_dir).resolve()
    if target_path.exists():
        shutil.rmtree(target_path)
    shutil.copytree(source_path, target_path)
    _remove_variant_video_outputs(target_path)
    _clear_backend_outputs(target_path)
    phone_model = _load_phone_model(phone_profile_path)
    _update_variant_phone_camera_metadata(
        target_path,
        phone_model,
        include_distortion=_phone_distortion_enabled(phone_model),
    )
    _apply_phone_physical_rgb_stream(target_path, phone_model, include_rolling_shutter=True)
    _compute_gt_tag_image_projections(target_path)
    return {
        "variant_dir": str(target_path),
        "seed": int(seed),
        "phone_effect_profile": _phone_effect_profile(phone_model, physical=True, noisy=False),
    }


def _noisy_bgr_image(clean_bgr: np.ndarray, *, frame_index: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + int(frame_index) * 7919)
    static_rng = np.random.default_rng(int(seed) + 17)
    channel_gains = 1.0 + static_rng.normal(
        0.0,
        float(_DEFAULT_IMAGE_NOISE_POLICY["channel_gain_std"]),
        size=(1, 1, 3),
    )
    image = np.asarray(clean_bgr, dtype=np.float32) * channel_gains
    ycrcb = cv2.cvtColor(np.clip(image, 0.0, 255.0).astype(np.uint8), cv2.COLOR_BGR2YCrCb).astype(np.float32)
    blur_sigma = float(_DEFAULT_IMAGE_NOISE_POLICY["blur_sigma"])
    ycrcb[:, :, 0] = cv2.GaussianBlur(ycrcb[:, :, 0], (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    target_mean = float(_DEFAULT_IMAGE_NOISE_POLICY["target_mean_luma"])
    current_mean = float(np.mean(ycrcb[:, :, 0]))
    ycrcb[:, :, 0] += (target_mean - current_mean) * float(_DEFAULT_IMAGE_NOISE_POLICY["mean_luma_blend"])
    shot_noise_std = np.sqrt(np.clip(ycrcb[:, :, 0], 0.0, 255.0)) * float(_DEFAULT_IMAGE_NOISE_POLICY["shot_noise_std_scale"])
    ycrcb[:, :, 0] += rng.normal(0.0, float(_DEFAULT_IMAGE_NOISE_POLICY["luma_residual_std"]), size=ycrcb[:, :, 0].shape)
    ycrcb[:, :, 0] += rng.normal(0.0, shot_noise_std, size=ycrcb[:, :, 0].shape)
    for channel in (1, 2):
        chroma = ycrcb[:, :, channel]
        small = cv2.resize(chroma, (max(chroma.shape[1] // 2, 1), max(chroma.shape[0] // 2, 1)), interpolation=cv2.INTER_AREA)
        chroma = cv2.resize(small, (chroma.shape[1], chroma.shape[0]), interpolation=cv2.INTER_LINEAR)
        chroma += rng.normal(0.0, float(_DEFAULT_IMAGE_NOISE_POLICY["chroma_residual_std"]), size=chroma.shape)
        ycrcb[:, :, channel] = chroma
    image_u8 = cv2.cvtColor(np.clip(ycrcb, 0.0, 255.0).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    jpeg_quality = int(_DEFAULT_IMAGE_NOISE_POLICY["jpeg_quality"])
    success, encoded = cv2.imencode(".jpg", image_u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if success:
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is not None:
            image_u8 = decoded
    return image_u8


def _derive_noisy_rgb_stream(clean_dir: Path, noisy_dir: Path, *, seed: int) -> None:
    frame_rows = _load_jsonl(clean_dir / "raw" / "camera_frames.jsonl")
    for row in frame_rows:
        frame_index = int(row["frame_index"])
        clean_path = clean_dir / str(row["rgb_path"])
        noisy_path = noisy_dir / str(row["rgb_path"])
        clean_bgr = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        if clean_bgr is None:
            raise FileNotFoundError(f"Could not read clean RGB frame {clean_path}.")
        noisy_bgr = _noisy_bgr_image(clean_bgr, frame_index=frame_index, seed=seed)
        noisy_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(noisy_path), noisy_bgr):
            raise RuntimeError(f"Failed to write noisy RGB frame {noisy_path}.")


def _derive_noisy_imu_stream(clean_dir: Path, noisy_dir: Path, *, seed: int, preset_name: str) -> None:
    fieldnames, clean_rows = _load_csv(clean_dir / "raw" / "imu.csv")
    _gt_fieldnames, gt_rows = _load_csv(clean_dir / "gt" / "imu_gt.csv")
    if not clean_rows or not gt_rows:
        raise ValueError("Need both clean raw IMU and GT IMU rows to derive the noisy variant.")
    preset = load_imu_noise_preset(preset_name)
    rng = np.random.default_rng(int(seed) + 104729)
    accel_bias = np.asarray(preset.accel_bias_mps2, dtype=np.float64).reshape(3)
    gyro_bias = np.asarray(preset.gyro_bias_rps, dtype=np.float64).reshape(3)
    gt_times = np.asarray([float(row["timestamp_s"]) for row in gt_rows], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for clean_row in clean_rows:
        clean_timestamp_s = float(clean_row["timestamp_s"])
        gt_index = int(np.searchsorted(gt_times, clean_timestamp_s, side="left"))
        candidate_indices = [max(min(gt_index, len(gt_rows) - 1), 0)]
        if gt_index > 0:
            candidate_indices.append(gt_index - 1)
        gt_row = min(
            (gt_rows[index] for index in candidate_indices),
            key=lambda item: abs(float(item["timestamp_s"]) - clean_timestamp_s),
        )
        dt_s = float(clean_row.get("dt_s", 0.0) or 0.0)
        sqrt_dt = float(np.sqrt(max(dt_s, 1e-9)))
        accel_bias = accel_bias + rng.normal(
            0.0,
            np.asarray(preset.accel_bias_random_walk_std, dtype=np.float64) * sqrt_dt,
        )
        gyro_bias = gyro_bias + rng.normal(
            0.0,
            np.asarray(preset.gyro_bias_random_walk_std, dtype=np.float64) * sqrt_dt,
        )
        gt_accel = np.asarray(
            [
                float(gt_row.get(key, clean_row.get(key, 0.0) or 0.0))
                for key in ("ax", "ay", "az")
            ],
            dtype=np.float64,
        )
        gt_gyro = np.asarray(
            [
                float(gt_row.get(key, clean_row.get(key, 0.0) or 0.0))
                for key in ("wx", "wy", "wz")
            ],
            dtype=np.float64,
        )
        noisy_accel = gt_accel + accel_bias + rng.normal(0.0, np.asarray(preset.accel_noise_std, dtype=np.float64))
        noisy_gyro = gt_gyro + gyro_bias + rng.normal(0.0, np.asarray(preset.gyro_noise_std, dtype=np.float64))
        packet = dict(clean_row)
        packet.update(
            {
                "ax": float(noisy_accel[0]),
                "ay": float(noisy_accel[1]),
                "az": float(noisy_accel[2]),
                "wx": float(noisy_gyro[0]),
                "wy": float(noisy_gyro[1]),
                "wz": float(noisy_gyro[2]),
                "noise_preset": str(preset.name),
            }
        )
        rows.append(packet)
    _write_csv(noisy_dir / "raw" / "imu.csv", fieldnames, rows)


def derive_noisy_dataset_variant(
    clean_dir: str | Path,
    noisy_dir: str | Path,
    *,
    seed: int = _DEFAULT_SEED,
    imu_noise_preset: str = "imu_nominal_phone",
) -> dict[str, Any]:
    clean_path = Path(clean_dir).resolve()
    noisy_path = Path(noisy_dir).resolve()
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean dataset variant does not exist: {clean_path}")
    if noisy_path.exists():
        shutil.rmtree(noisy_path)
    shutil.copytree(clean_path, noisy_path)
    _remove_variant_video_outputs(noisy_path)
    _derive_noisy_rgb_stream(clean_path, noisy_path, seed=int(seed))
    _derive_noisy_imu_stream(clean_path, noisy_path, seed=int(seed), preset_name=str(imu_noise_preset))
    return {
        "clean_dir": str(clean_path),
        "noisy_dir": str(noisy_path),
        "imu_noise_preset": str(imu_noise_preset),
        "image_noise_policy": dict(_DEFAULT_IMAGE_NOISE_POLICY),
    }


def _write_phone_capture_video(
    variant_dir: Path,
    capture_dir: Path,
    *,
    fps: float,
    width_px: int,
    height_px: int,
) -> dict[str, Any]:
    frame_rows = sorted(
        _load_jsonl(variant_dir / "raw" / "camera_frames.jsonl"),
        key=lambda row: int(row.get("frame_index", 0)),
    )
    if not frame_rows:
        raise ValueError(f"No camera frame rows found for phone video export: {variant_dir}")
    capture_dir.mkdir(parents=True, exist_ok=True)
    video_path = capture_dir / "video.mp4"
    frame_tmp_dir = capture_dir / ".video_frames"
    _reset_dir(frame_tmp_dir)
    try:
        for output_index, row in enumerate(frame_rows):
            source_path = variant_dir / str(row["rgb_path"])
            image_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Could not read RGB frame for phone video: {source_path}")
            image_bgr = _resize_center_crop_to_resolution(
                image_bgr,
                width_px=int(width_px),
                height_px=int(height_px),
                interpolation=cv2.INTER_CUBIC,
            )
            target_path = frame_tmp_dir / f"frame_{output_index:06d}.png"
            if not cv2.imwrite(str(target_path), image_bgr):
                raise RuntimeError(f"Failed to stage phone video frame {target_path}.")
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            cmd = [
                ffmpeg_path,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                f"{float(fps):.6f}",
                "-i",
                str(frame_tmp_dir / "frame_%06d.png"),
                "-an",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-bf",
                "0",
                "-b:v",
                str(int(_PHONE_CAPTURE_VIDEO_BIT_RATE_BPS)),
                "-maxrate",
                str(int(_PHONE_CAPTURE_VIDEO_BIT_RATE_BPS)),
                "-bufsize",
                str(int(_PHONE_CAPTURE_VIDEO_BIT_RATE_BPS * 2)),
                "-movflags",
                "+faststart",
                str(video_path),
            ]
            completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, text=True, capture_output=True)
            if completed.returncode == 0 and video_path.exists():
                return {
                    "video_path": str(video_path),
                    "codec": "h264",
                    "encoder": "ffmpeg/libx264",
                    "profile": "high",
                    "width_px": int(width_px),
                    "height_px": int(height_px),
                    "pixel_format": "yuv420p",
                    "color_space": "bt709",
                    "bit_rate_bps": int(_PHONE_CAPTURE_VIDEO_BIT_RATE_BPS),
                    "b_frames": 0,
                    "fallback": False,
                }
        first_image = cv2.imread(str(frame_tmp_dir / "frame_000000.png"), cv2.IMREAD_COLOR)
        if first_image is None:
            raise FileNotFoundError("Could not read staged first frame for OpenCV fallback.")
        height_px, width_px = first_image.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (int(width_px), int(height_px)),
        )
        try:
            if not writer.isOpened():
                raise RuntimeError(f"Could not open fallback video writer for {video_path}.")
            for output_index in range(len(frame_rows)):
                image_bgr = cv2.imread(str(frame_tmp_dir / f"frame_{output_index:06d}.png"), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise FileNotFoundError(f"Could not read staged frame {output_index}.")
                writer.write(image_bgr)
        finally:
            writer.release()
        return {
            "video_path": str(video_path),
            "codec": "mp4v",
            "encoder": "opencv",
            "profile": None,
            "width_px": int(width_px),
            "height_px": int(height_px),
            "pixel_format": None,
            "color_space": None,
            "bit_rate_bps": None,
            "b_frames": None,
            "fallback": True,
        }
    finally:
        shutil.rmtree(frame_tmp_dir, ignore_errors=True)


def _write_phone_frames_csv(variant_dir: Path, capture_dir: Path) -> list[dict[str, Any]]:
    frame_rows = sorted(
        _load_jsonl(variant_dir / "raw" / "camera_frames.jsonl"),
        key=lambda row: int(row.get("frame_index", 0)),
    )
    rows: list[dict[str, Any]] = []
    for output_index, row in enumerate(frame_rows, start=1):
        timestamp_s = float(row.get("sensor_time_s", row.get("timestamp_s", 0.0)))
        rows.append(
            {
                "frame_index": int(output_index),
                "camera_timestamp_nanos": int(_phone_capture_nanos_from_timestamp(timestamp_s)),
                "relative_session_nanos": int(_phone_capture_relative_nanos(timestamp_s)),
                "width": int(_PHONE_CAPTURE_FRAME_WIDTH),
                "height": int(_PHONE_CAPTURE_FRAME_HEIGHT),
                "rotation_degrees": int(_PHONE_CAPTURE_ROTATION_DEGREES),
            }
        )
    _write_csv(capture_dir / "frames.csv", _PHONE_FRAMES_FIELDNAMES, rows)
    return rows


def _write_phone_camera_results_csv(variant_dir: Path, capture_dir: Path) -> list[dict[str, Any]]:
    frame_rows = sorted(
        _load_jsonl(variant_dir / "raw" / "camera_frames.jsonl"),
        key=lambda row: int(row.get("frame_index", 0)),
    )
    rows: list[dict[str, Any]] = []
    for output_index, row in enumerate(frame_rows):
        timestamp_s = float(row.get("sensor_time_s", row.get("timestamp_s", 0.0)))
        rows.append(
            {
                "frame_number": int(1024 + output_index),
                "sensor_timestamp_nanos": int(_phone_capture_nanos_from_timestamp(timestamp_s)),
                "relative_session_nanos": int(_phone_capture_relative_nanos(timestamp_s)),
                "exposure_time_nanos": int(_PHONE_CAPTURE_EXPOSURE_TIME_NS),
                "sensitivity_iso": int(_phone_iso_for_frame(output_index)),
                "frame_duration_nanos": int(_PHONE_CAPTURE_FRAME_DURATION_NS),
                "rolling_shutter_skew_nanos": int(_PHONE_CAPTURE_ROLLING_SHUTTER_SKEW_NS),
                "lens_focus_distance_diopters": float(_PHONE_CAPTURE_FOCUS_DISTANCE_DIOPTERS),
                "lens_state": "STATIONARY",
                "af_mode": "OFF",
                "af_state": "INACTIVE",
                "ae_state": "CONVERGED",
                "awb_state": "CONVERGED",
                "video_stabilization_mode": "OFF",
                "optical_stabilization_mode": "OFF",
                "zoom_ratio": 1.0,
                "crop_left": 0,
                "crop_top": 0,
                "crop_right": 4000,
                "crop_bottom": 3000,
                "active_physical_camera_id": 2,
            }
        )
    _write_csv(capture_dir / "camera_results.csv", _PHONE_CAMERA_RESULTS_FIELDNAMES, rows)
    return rows


def _write_phone_imu_csv(variant_dir: Path, capture_dir: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
    _fieldnames, imu_rows = _load_csv(variant_dir / "raw" / "imu.csv")
    rows: list[dict[str, Any]] = []
    for row in imu_rows:
        timestamp_s = float(row.get("sensor_time_s", row.get("timestamp_s", 0.0)) or 0.0)
        base_nanos = int(_phone_capture_nanos_from_timestamp(timestamp_s))
        accel_values = [float(row.get(key, 0.0) or 0.0) for key in ("ax", "ay", "az")]
        gyro_values = [float(row.get(key, 0.0) or 0.0) for key in ("wx", "wy", "wz")]
        rows.append(
            {
                "elapsed_realtime_nanos": int(base_nanos),
                "sensor_type": "accelerometer_uncalibrated",
                "x": float(accel_values[0]),
                "y": float(accel_values[1]),
                "z": float(accel_values[2]),
                "accuracy": 3,
                "bias_x": float(_PHONE_CAPTURE_ACCEL_BIAS[0]),
                "bias_y": float(_PHONE_CAPTURE_ACCEL_BIAS[1]),
                "bias_z": float(_PHONE_CAPTURE_ACCEL_BIAS[2]),
            }
        )
        rows.append(
            {
                "elapsed_realtime_nanos": int(base_nanos + 4_293_000),
                "sensor_type": "gyroscope_uncalibrated",
                "x": float(gyro_values[0]),
                "y": float(gyro_values[1]),
                "z": float(gyro_values[2]),
                "accuracy": 3,
                "bias_x": float(_PHONE_CAPTURE_GYRO_BIAS[0]),
                "bias_y": float(_PHONE_CAPTURE_GYRO_BIAS[1]),
                "bias_z": float(_PHONE_CAPTURE_GYRO_BIAS[2]),
            }
        )
    rows = sorted(rows, key=lambda item: int(item["elapsed_realtime_nanos"]))
    _write_csv(capture_dir / "imu.csv", _PHONE_IMU_FIELDNAMES, rows)
    rates: dict[str, float] = {}
    by_sensor: dict[str, list[int]] = {}
    for row in rows:
        by_sensor.setdefault(str(row["sensor_type"]), []).append(int(row["elapsed_realtime_nanos"]))
    for sensor_type, timestamps in by_sensor.items():
        if len(timestamps) >= 2:
            intervals_s = np.diff(np.asarray(timestamps, dtype=np.float64)) * 1.0e-9
            rates[sensor_type.replace("_uncalibrated", "")] = float(1.0 / max(float(np.mean(intervals_s)), 1e-9))
    return rows, rates


def _write_phone_session_json(
    variant_dir: Path,
    capture_dir: Path,
    *,
    phone_model: PhoneCameraModel,
    frames_rows: list[dict[str, Any]],
    camera_result_rows: list[dict[str, Any]],
    imu_rows: list[dict[str, Any]],
    observed_imu_rates_hz: dict[str, float],
    video_manifest: dict[str, Any],
) -> dict[str, Any]:
    session_id = f"sim_{variant_dir.name}_{_load_json(variant_dir / 'manifest.json').get('run_id', variant_dir.name)}"
    sample_counts = {
        "accelerometer": sum(1 for row in imu_rows if str(row["sensor_type"]).startswith("accelerometer")),
        "gyroscope": sum(1 for row in imu_rows if str(row["sensor_type"]).startswith("gyroscope")),
    }
    files = {
        "videoFileName": "video.mp4",
        "imuFileName": "imu.csv",
        "framesFileName": "frames.csv",
        "cameraResultsFileName": "camera_results.csv",
        "manifestFileName": "session.json",
        "videoBytes": int((capture_dir / "video.mp4").stat().st_size) if (capture_dir / "video.mp4").exists() else 0,
        "imuBytes": int((capture_dir / "imu.csv").stat().st_size) if (capture_dir / "imu.csv").exists() else 0,
        "framesBytes": int((capture_dir / "frames.csv").stat().st_size) if (capture_dir / "frames.csv").exists() else 0,
        "cameraResultsBytes": int((capture_dir / "camera_results.csv").stat().st_size) if (capture_dir / "camera_results.csv").exists() else 0,
    }
    payload = {
        "schemaVersion": 3,
        "sessionId": str(session_id),
        "sessionLabel": f"sim_{variant_dir.name}",
        "status": "COMPLETED",
        "startedAtUtc": "2026-03-27T18:52:49.129319Z",
        "completedAtUtc": "2026-03-27T18:52:57.899243Z",
        "monotonicSessionStartElapsedRealtimeNanos": int(_PHONE_CAPTURE_START_NANOS),
        "sensorRegistrationElapsedRealtimeNanos": int(_PHONE_CAPTURE_START_NANOS + _PHONE_CAPTURE_SENSOR_REGISTRATION_OFFSET_NS),
        "videoStartElapsedRealtimeNanos": int(_PHONE_CAPTURE_START_NANOS + _PHONE_CAPTURE_VIDEO_START_OFFSET_NS),
        "stopRequestedElapsedRealtimeNanos": int(
            (frames_rows[-1]["camera_timestamp_nanos"] if frames_rows else _PHONE_CAPTURE_START_NANOS)
            + _PHONE_CAPTURE_FRAME_DURATION_NS
        ),
        "imuStopElapsedRealtimeNanos": int(imu_rows[-1]["elapsed_realtime_nanos"]) if imu_rows else None,
        "videoFinalizeElapsedRealtimeNanos": int(
            (frames_rows[-1]["camera_timestamp_nanos"] if frames_rows else _PHONE_CAPTURE_START_NANOS)
            + 2 * _PHONE_CAPTURE_FRAME_DURATION_NS
        ),
        "failureMessage": None,
        "camera": {
            "resolutionPreset": "FHD",
            "targetFramesPerSecond": 30,
            "audioEnabled": False,
            "timestampSource": "REALTIME",
            "controlRequest": {
                "scientificCaptureMode": True,
                "targetFpsRange": "30-30",
                "videoStabilizationModeRequested": "OFF",
                "opticalStabilizationModeRequested": "OFF",
                "autoFocusModeRequested": "OFF",
                "focusDistanceDioptersRequested": float(_PHONE_CAPTURE_FOCUS_DISTANCE_DIOPTERS),
            },
            "staticMetadata": {
                "logicalCameraId": "0",
                "availablePhysicalCameraIds": [],
                "sensorOrientationDegrees": int(phone_model.sensor_orientation_deg),
                "activeArrayBounds": {"left": 0, "top": 0, "right": 4000, "bottom": 3000},
                "focalLengthsMillimeters": [float(phone_model.focal_length_mm)],
                "hyperfocalDistanceDiopters": float(_PHONE_CAPTURE_FOCUS_DISTANCE_DIOPTERS),
                "minimumFocusDistanceDiopters": 20.0,
                "lensIntrinsicCalibration": [2694.107, 2694.107, 2000.6089, 1507.567, 0.0],
                "lensDistortion": [float(value) for value in phone_model.distortion_coefficients],
                "availableVideoStabilizationModes": ["OFF", "ON", "PREVIEW_STABILIZATION"],
                "availableOpticalStabilizationModes": ["OFF", "ON"],
            },
        },
        "sensors": {
            "accelerometerEnabled": True,
            "gyroscopeEnabled": True,
            "samplingPreset": "GAME",
            "expectedRateHz": 50,
            "preferUncalibratedSensors": True,
        },
        "device": {
            "manufacturer": "Google",
            "brand": "google",
            "model": "Pixel 9a",
            "device": "tegu",
            "product": "tegu",
            "apiLevel": 36,
            "androidRelease": "16",
            "appVersionName": "0.1.0",
            "appVersionCode": 1,
        },
        "files": files,
        "sampleCounts": sample_counts,
        "frameCount": int(len(frames_rows)),
        "cameraResultCount": int(len(camera_result_rows)),
        "captureRootDescription": "simulated_phone_capture",
        "quality": {
            "frameTimestampsLogged": True,
            "cameraResultsLogged": True,
            "frameCount": int(len(frames_rows)),
            "cameraResultCount": int(len(camera_result_rows)),
            "matchedFrameResultCount": int(min(len(frames_rows), len(camera_result_rows))),
            "unmatchedFrameCount": int(max(len(frames_rows) - len(camera_result_rows), 0)),
            "unmatchedCameraResultCount": int(max(len(camera_result_rows) - len(frames_rows), 0)),
            "timestampSourceRealtime": True,
            "videoStabilizationOffObserved": True,
            "opticalStabilizationOffObserved": True,
            "focusDistanceStable": True,
            "focusDistanceMinDiopters": float(_PHONE_CAPTURE_FOCUS_DISTANCE_DIOPTERS),
            "focusDistanceMaxDiopters": float(_PHONE_CAPTURE_FOCUS_DISTANCE_DIOPTERS),
            "afLockedObserved": True,
            "aeConvergedObserved": True,
            "awbConvergedObserved": True,
            "observedImuRatesHz": observed_imu_rates_hz,
            "issues": ["Simulated V1 phone capture: rolling shutter is image-space approximate."],
        },
        "videoEncoding": dict(video_manifest),
        "notes": [
            "Generated by camera_calibration_digital_twin phone-matched export.",
            "Sidecar columns match the Pixel 9a sample capture shape.",
        ],
    }
    _write_json(capture_dir / "session.json", payload)
    payload["files"]["manifestBytes"] = int((capture_dir / "session.json").stat().st_size)
    _write_json(capture_dir / "session.json", payload)
    return payload


def write_phone_capture_mirror(
    variant_dir: str | Path,
    *,
    phone_profile_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved = Path(variant_dir).resolve()
    phone_model = _load_phone_model(phone_profile_path)
    capture_dir = resolved / "phone_capture"
    if capture_dir.exists():
        shutil.rmtree(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    video_manifest = _write_phone_capture_video(
        resolved,
        capture_dir,
        fps=float(phone_model.target_frames_per_second),
        width_px=int(phone_model.output_width_px),
        height_px=int(phone_model.output_height_px),
    )
    frames_rows = _write_phone_frames_csv(resolved, capture_dir)
    camera_result_rows = _write_phone_camera_results_csv(resolved, capture_dir)
    imu_rows, observed_rates = _write_phone_imu_csv(resolved, capture_dir)
    session_payload = _write_phone_session_json(
        resolved,
        capture_dir,
        phone_model=phone_model,
        frames_rows=frames_rows,
        camera_result_rows=camera_result_rows,
        imu_rows=imu_rows,
        observed_imu_rates_hz=observed_rates,
        video_manifest=video_manifest,
    )
    return {
        "path": str(capture_dir),
        "files": {
            "video_mp4": "phone_capture/video.mp4",
            "frames_csv": "phone_capture/frames.csv",
            "camera_results_csv": "phone_capture/camera_results.csv",
            "imu_csv": "phone_capture/imu.csv",
            "session_json": "phone_capture/session.json",
        },
        "video": dict(video_manifest),
        "session": {
            "schemaVersion": int(session_payload["schemaVersion"]),
            "sessionId": str(session_payload["sessionId"]),
            "frameCount": int(session_payload["frameCount"]),
            "cameraResultCount": int(session_payload["cameraResultCount"]),
            "sampleCounts": dict(session_payload["sampleCounts"]),
        },
    }


def _detector_metrics(
    frame_rows: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
    gt_projection_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_frames = int(len(frame_rows))
    detections_total = int(len(detection_rows))
    frames_with_detections = {int(row["frame_index"]) for row in detection_rows}
    best_detections_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for detection in detection_rows:
        key = (int(detection["frame_index"]), int(detection["tag_id"]))
        current = best_detections_by_key.get(key)
        if current is None or float(detection.get("score", 0.0) or 0.0) >= float(current.get("score", 0.0) or 0.0):
            best_detections_by_key[key] = detection
    detections_by_tag: dict[int, set[int]] = {}
    visible_by_tag: dict[int, int] = {}
    frames_with_gt_visible_tags: set[int] = set()
    matched_visible_pairs = 0
    corner_rmse_values_px: list[float] = []
    center_error_values_px: list[float] = []
    for gt_row in gt_projection_rows:
        frame_index = int(gt_row["frame_index"])
        tag_id = int(gt_row["tag_id"])
        visibility = dict(gt_row.get("visibility_flags", {}))
        if not bool(visibility.get("fully_visible", False)):
            continue
        frames_with_gt_visible_tags.add(frame_index)
        visible_by_tag[tag_id] = visible_by_tag.get(tag_id, 0) + 1
        detection = best_detections_by_key.get((frame_index, tag_id))
        if detection is None:
            continue
        matched_visible_pairs += 1
        detections_by_tag.setdefault(tag_id, set()).add(frame_index)
        detection_corners = np.asarray(detection.get("corners_xy", ()), dtype=np.float64).reshape(-1, 2)
        gt_corners = np.asarray(gt_row.get("corners_xy", ()), dtype=np.float64).reshape(-1, 2)
        if detection_corners.shape == (4, 2) and gt_corners.shape == (4, 2):
            aligned_gt_corners, corner_rmse_px = _best_cyclic_corner_alignment(detection_corners, gt_corners)
            corner_rmse_values_px.append(float(corner_rmse_px))
            detection_center_xy = np.mean(detection_corners, axis=0)
            gt_center_xy = np.asarray(gt_row.get("center_xy", ()), dtype=np.float64).reshape(-1)
            if gt_center_xy.shape == (2,):
                center_error_values_px.append(float(np.linalg.norm(detection_center_xy - gt_center_xy)))
    per_tag: dict[str, Any] = {}
    for tag_id in sorted(set(visible_by_tag) | set(detections_by_tag)):
        visible_frames = int(visible_by_tag.get(int(tag_id), 0))
        detected_frames = int(len(detections_by_tag.get(int(tag_id), set())))
        per_tag[str(int(tag_id))] = {
            "visible_frames": visible_frames,
            "detected_frames": detected_frames,
            "visible_hit_rate": None if visible_frames <= 0 else float(detected_frames / visible_frames),
        }
    visible_tag_observations_total = int(sum(visible_by_tag.values()))
    return {
        "frames_total": total_frames,
        "frames_with_detections": int(len(frames_with_detections)),
        "detection_frame_fraction": 0.0 if total_frames <= 0 else float(len(frames_with_detections) / total_frames),
        "frames_with_gt_visible_tags": int(len(frames_with_gt_visible_tags)),
        "gt_visible_frame_fraction": 0.0 if total_frames <= 0 else float(len(frames_with_gt_visible_tags) / total_frames),
        "detections_total": detections_total,
        "detections_per_frame": 0.0 if total_frames <= 0 else float(detections_total / total_frames),
        "visible_tag_observations_total": visible_tag_observations_total,
        "matched_visible_detections_total": int(matched_visible_pairs),
        "gt_visible_hit_rate": None
        if visible_tag_observations_total <= 0
        else float(matched_visible_pairs / visible_tag_observations_total),
        "per_tag_coverage": per_tag,
        "pixel_metrics": {
            "matched_detection_count": int(len(corner_rmse_values_px)),
            "corner_rmse_mean_px": None if not corner_rmse_values_px else float(np.mean(np.asarray(corner_rmse_values_px, dtype=np.float64))),
            "corner_rmse_p95_px": None if not corner_rmse_values_px else float(np.percentile(np.asarray(corner_rmse_values_px, dtype=np.float64), 95.0)),
            "center_error_mean_px": None if not center_error_values_px else float(np.mean(np.asarray(center_error_values_px, dtype=np.float64))),
            "center_error_p95_px": None if not center_error_values_px else float(np.percentile(np.asarray(center_error_values_px, dtype=np.float64), 95.0)),
        },
    }


def _best_detections_by_key(detection_rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    best_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for detection in detection_rows:
        key = (int(detection["frame_index"]), int(detection["tag_id"]))
        current = best_by_key.get(key)
        if current is None or float(detection.get("score", 0.0) or 0.0) >= float(current.get("score", 0.0) or 0.0):
            best_by_key[key] = detection
    return best_by_key


def _matched_detection_pixel_error_rows(
    detection_rows: list[dict[str, Any]],
    gt_projection_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_detections = _best_detections_by_key(detection_rows)
    for gt_row in gt_projection_rows:
        visibility = dict(gt_row.get("visibility_flags", {}))
        if not bool(visibility.get("fully_visible", False)):
            continue
        frame_index = int(gt_row["frame_index"])
        tag_id = int(gt_row["tag_id"])
        detection = best_detections.get((frame_index, tag_id))
        if detection is None:
            continue
        detection_corners = np.asarray(detection.get("corners_xy", ()), dtype=np.float64).reshape(-1, 2)
        gt_corners = np.asarray(gt_row.get("corners_xy", ()), dtype=np.float64).reshape(-1, 2)
        if detection_corners.shape != (4, 2) or gt_corners.shape != (4, 2):
            continue
        aligned_gt_corners, corner_rmse_px = _best_cyclic_corner_alignment(detection_corners, gt_corners)
        detection_center_xy = np.mean(detection_corners, axis=0)
        gt_center_xy = np.asarray(gt_row.get("center_xy", ()), dtype=np.float64).reshape(-1)
        center_error_px = None if gt_center_xy.shape != (2,) else float(np.linalg.norm(detection_center_xy - gt_center_xy))
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_s": float(gt_row["timestamp_s"]),
                "tag_id": tag_id,
                "corner_rmse_px": float(corner_rmse_px),
                "center_error_px": center_error_px,
                "detection_score": float(detection.get("score", 0.0) or 0.0),
                "projection_pose_frame_index": int(gt_row.get("projection_pose_frame_index", frame_index)),
                "projection_pose_timestamp_s": float(gt_row.get("projection_pose_timestamp_s", gt_row["timestamp_s"])),
                "detection_corners_xy": [[float(x), float(y)] for x, y in detection_corners.tolist()],
                "gt_corners_xy": [[float(x), float(y)] for x, y in np.asarray(aligned_gt_corners, dtype=np.float64).tolist()],
            }
        )
    return rows


def _percentile_or_none(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize_tag_pixel_errors(
    per_frame_rows: list[dict[str, Any]],
    *,
    coverage_by_tag: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    grouped_rows: dict[int, list[dict[str, Any]]] = {}
    for row in per_frame_rows:
        grouped_rows.setdefault(int(row["tag_id"]), []).append(dict(row))
    summary: dict[str, dict[str, Any]] = {}
    for tag_id in sorted(set(grouped_rows) | {int(tag) for tag in coverage_by_tag}):
        tag_rows = sorted(grouped_rows.get(int(tag_id), []), key=lambda item: int(item["frame_index"]))
        corner_values = [float(row["corner_rmse_px"]) for row in tag_rows]
        center_values = [
            float(row["center_error_px"])
            for row in tag_rows
            if row.get("center_error_px") is not None
        ]
        coverage = dict(coverage_by_tag.get(str(int(tag_id)), {}))
        summary[str(int(tag_id))] = {
            "visible_frames": int(coverage.get("visible_frames", 0) or 0),
            "detected_frames": int(coverage.get("detected_frames", 0) or 0),
            "visible_hit_rate": coverage.get("visible_hit_rate"),
            "matched_frames": int(len(tag_rows)),
            "corner_rmse_mean_px": _mean_or_none(corner_values),
            "corner_rmse_p95_px": _percentile_or_none(corner_values, 95.0),
            "corner_rmse_max_px": None if not corner_values else float(max(corner_values)),
            "center_error_mean_px": _mean_or_none(center_values),
            "center_error_p95_px": _percentile_or_none(center_values, 95.0),
        }
    return summary


def _format_optional_float(value: Any, *, precision: int = 3) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.{precision}f}"


def _draw_tag_corner_rmse_plot(
    *,
    rows: list[dict[str, Any]],
    variant_name: str,
    output_path: Path,
) -> None:
    width_px = 1400
    height_px = 900
    margin_left = 110
    margin_right = 260
    margin_top = 90
    margin_bottom = 100
    plot_left = margin_left
    plot_right = width_px - margin_right
    plot_top = margin_top
    plot_bottom = height_px - margin_bottom
    plot_width = max(plot_right - plot_left, 1)
    plot_height = max(plot_bottom - plot_top, 1)

    canvas = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"{variant_name.title()} Pupil Pixel Error by Tag",
        (margin_left, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "x-axis: frame index | y-axis: corner RMSE to rendered GT tag corners [px]",
        (margin_left, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    if not rows:
        cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (220, 220, 220), 1)
        cv2.putText(
            canvas,
            "No matched visible GT detections available.",
            (plot_left + 40, plot_top + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (70, 70, 70),
            2,
            cv2.LINE_AA,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), canvas):
            raise RuntimeError(f"Could not write plot image: {output_path}")
        return

    grouped_rows: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(int(row["tag_id"]), []).append(dict(row))
    for tag_rows in grouped_rows.values():
        tag_rows.sort(key=lambda item: int(item["frame_index"]))

    frame_values = [int(row["frame_index"]) for row in rows]
    error_values = [float(row["corner_rmse_px"]) for row in rows]
    min_frame = int(min(frame_values))
    max_frame = int(max(frame_values))
    min_error = 0.0
    max_error = float(max(error_values))
    if max_frame <= min_frame:
        max_frame = min_frame + 1
    if max_error <= min_error:
        max_error = min_error + 1.0
    else:
        max_error *= 1.1

    palette = [
        (230, 99, 71),
        (71, 99, 230),
        (34, 139, 34),
        (160, 82, 45),
        (148, 0, 211),
        (0, 140, 255),
        (255, 140, 0),
        (205, 92, 92),
        (0, 128, 128),
        (199, 21, 133),
    ]

    def _point(frame_index: int, error_px: float) -> tuple[int, int]:
        x_frac = (float(frame_index) - float(min_frame)) / max(float(max_frame - min_frame), 1.0)
        y_frac = (float(error_px) - float(min_error)) / max(float(max_error - min_error), 1e-9)
        x_px = int(round(plot_left + x_frac * plot_width))
        y_px = int(round(plot_bottom - y_frac * plot_height))
        return x_px, y_px

    for grid_idx in range(6):
        y_frac = grid_idx / 5.0
        y_px = int(round(plot_bottom - y_frac * plot_height))
        value = min_error + y_frac * (max_error - min_error)
        cv2.line(canvas, (plot_left, y_px), (plot_right, y_px), (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{value:.2f}",
            (18, y_px + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )
    for grid_idx in range(6):
        x_frac = grid_idx / 5.0
        x_px = int(round(plot_left + x_frac * plot_width))
        frame_value = min_frame + x_frac * (max_frame - min_frame)
        cv2.line(canvas, (x_px, plot_top), (x_px, plot_bottom), (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(int(round(frame_value))),
            (x_px - 12, plot_bottom + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (180, 180, 180), 1)
    cv2.putText(canvas, "Frame Index", (plot_left + plot_width // 2 - 50, height_px - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (55, 55, 55), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Corner RMSE [px]", (18, plot_top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (55, 55, 55), 2, cv2.LINE_AA)

    legend_y = plot_top + 20
    legend_x = plot_right + 24
    for color_index, (tag_id, tag_rows) in enumerate(sorted(grouped_rows.items())):
        color = palette[color_index % len(palette)]
        polyline = np.asarray([_point(int(row["frame_index"]), float(row["corner_rmse_px"])) for row in tag_rows], dtype=np.int32)
        if len(polyline) >= 2:
            cv2.polylines(canvas, [polyline.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
        for x_px, y_px in polyline.tolist():
            cv2.circle(canvas, (int(x_px), int(y_px)), 4, color, -1, cv2.LINE_AA)
        cv2.line(canvas, (legend_x, legend_y), (legend_x + 28, legend_y), color, 3, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"Tag {tag_id}",
            (legend_x + 40, legend_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
        legend_y += 28

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write plot image: {output_path}")


def write_tabletop_autodemo_pixel_precision_report(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).resolve()
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    variant_names = _phone_capture_variant_names(root)
    overall_rows: list[dict[str, Any]] = []
    per_tag_rows: list[dict[str, Any]] = []
    plot_paths: dict[str, Path] = {}
    csv_paths: dict[str, Path] = {}
    for variant_name in variant_names:
        backend_dir = root / variant_name / "localization" / "new_pupil"
        summary = _load_json(backend_dir / "backend_summary.json")
        detection_rows = _load_jsonl(backend_dir / "detections.jsonl")
        gt_projection_rows = _load_jsonl(root / variant_name / _GT_TAG_PROJECTIONS_PATH)
        per_frame_rows = _matched_detection_pixel_error_rows(detection_rows, gt_projection_rows)
        coverage_by_tag = dict(summary.get("detection_metrics", {}).get("per_tag_coverage", {}))
        per_tag_summary = _summarize_tag_pixel_errors(per_frame_rows, coverage_by_tag=coverage_by_tag)
        csv_path = analysis_dir / f"{variant_name}_tag_corner_rmse_by_frame.csv"
        csv_paths[variant_name] = csv_path
        _write_csv(
            csv_path,
            [
                "variant",
                "frame_index",
                "timestamp_s",
                "tag_id",
                "corner_rmse_px",
                "center_error_px",
                "detection_score",
                "projection_pose_frame_index",
                "projection_pose_timestamp_s",
            ],
            [
                {
                    "variant": variant_name,
                    "frame_index": int(row["frame_index"]),
                    "timestamp_s": float(row["timestamp_s"]),
                    "tag_id": int(row["tag_id"]),
                    "corner_rmse_px": float(row["corner_rmse_px"]),
                    "center_error_px": row.get("center_error_px"),
                    "detection_score": float(row.get("detection_score", 0.0) or 0.0),
                    "projection_pose_frame_index": int(row.get("projection_pose_frame_index", row["frame_index"])),
                    "projection_pose_timestamp_s": float(row.get("projection_pose_timestamp_s", row["timestamp_s"])),
                }
                for row in per_frame_rows
            ],
        )
        plot_path = analysis_dir / f"{variant_name}_tag_corner_rmse_by_frame.png"
        plot_paths[variant_name] = plot_path
        _draw_tag_corner_rmse_plot(rows=per_frame_rows, variant_name=variant_name, output_path=plot_path)

        pixel_metrics = dict(summary.get("detection_metrics", {}).get("pixel_metrics", {}))
        overall_rows.append(
            {
                "variant": variant_name,
                "matched_detections": int(pixel_metrics.get("matched_detection_count", 0) or 0),
                "corner_rmse_mean_px": pixel_metrics.get("corner_rmse_mean_px"),
                "corner_rmse_p95_px": pixel_metrics.get("corner_rmse_p95_px"),
                "corner_rmse_max_px": None if not per_frame_rows else float(max(float(row["corner_rmse_px"]) for row in per_frame_rows)),
                "center_error_mean_px": pixel_metrics.get("center_error_mean_px"),
                "center_error_p95_px": pixel_metrics.get("center_error_p95_px"),
                "gt_visible_hit_rate": summary.get("detection_metrics", {}).get("gt_visible_hit_rate"),
            }
        )
        for tag_id, metrics in sorted(per_tag_summary.items(), key=lambda item: int(item[0])):
            per_tag_rows.append(
                {
                    "variant": variant_name,
                    "tag_id": int(tag_id),
                    **metrics,
                }
            )

    _write_json(
        analysis_dir / "pixel_precision_summary.json",
        {
            "dataset_root": str(root),
            "overall": overall_rows,
            "per_tag": per_tag_rows,
            "plots": {variant: str(path.resolve()) for variant, path in plot_paths.items()},
            "csv": {variant: str(path.resolve()) for variant, path in csv_paths.items()},
        },
    )

    lines = [
        "# Tabletop Auto-Demo Pixel Precision Report",
        "",
        f"Dataset root: `{root}`",
        "",
        "Pixel precision here means per-tag corner RMSE in pixels between the best-scoring detection and the rendered GT tag corners.",
        "Only GT tags marked as fully visible are included in the matched-error curves and summary statistics below.",
        "",
        "## Overall Precision",
        "",
        "| Variant | Matched Detections | Corner RMSE Mean [px] | Corner RMSE P95 [px] | Corner RMSE Max [px] | Center Error Mean [px] | Center Error P95 [px] | GT-Visible Hit Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["matched_detections"]),
                    _format_optional_float(row["corner_rmse_mean_px"], precision=3),
                    _format_optional_float(row["corner_rmse_p95_px"], precision=3),
                    _format_optional_float(row["corner_rmse_max_px"], precision=3),
                    _format_optional_float(row["center_error_mean_px"], precision=3),
                    _format_optional_float(row["center_error_p95_px"], precision=3),
                    _format_optional_float(row["gt_visible_hit_rate"], precision=3),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-Tag Precision",
            "",
            "| Variant | Tag | Visible Frames | Detected Frames | Visible Hit Rate | Matched Frames | Corner RMSE Mean [px] | Corner RMSE P95 [px] | Corner RMSE Max [px] | Center Error Mean [px] | Center Error P95 [px] |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in per_tag_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["tag_id"]),
                    str(row["visible_frames"]),
                    str(row["detected_frames"]),
                    _format_optional_float(row["visible_hit_rate"], precision=3),
                    str(row["matched_frames"]),
                    _format_optional_float(row["corner_rmse_mean_px"], precision=3),
                    _format_optional_float(row["corner_rmse_p95_px"], precision=3),
                    _format_optional_float(row["corner_rmse_max_px"], precision=3),
                    _format_optional_float(row["center_error_mean_px"], precision=3),
                    _format_optional_float(row["center_error_p95_px"], precision=3),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Framewise Curves",
            "",
        ]
    )
    for variant_name in variant_names:
        title = str(variant_name).replace("_", " ").title()
        lines.extend(
            [
                f"- {title} CSV: [`analysis/{csv_paths[variant_name].name}`](analysis/{csv_paths[variant_name].name})",
            ]
        )
    lines.append("")
    for variant_name in variant_names:
        title = str(variant_name).replace("_", " ").title()
        lines.extend(
            [
                f"### {title}",
                "",
                f"![{title} per-tag corner RMSE](analysis/{plot_paths[variant_name].name})",
                "",
            ]
        )
    report_path = root / "pixel_precision_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_tabletop_autodemo_corner_dataset_report(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).resolve()
    variant_names = _phone_capture_variant_names(root)
    backend_summaries: dict[str, dict[str, Any]] = {}
    for variant_name in variant_names:
        backend_summaries[variant_name] = _load_json(
            root / variant_name / "localization" / "new_pupil" / "backend_summary.json"
        )

    def _metric(summary: dict[str, Any], *path: str) -> Any:
        value: Any = summary
        for item in path:
            if not isinstance(value, dict):
                return None
            value = value.get(item)
        return value

    def _format_optional(value: Any, *, precision: int = 3) -> str:
        if value in (None, ""):
            return "n/a"
        return f"{float(value):.{precision}f}"

    lines = [
        "# Tabletop Auto-Demo Corner Export",
        "",
        f"Dataset root: `{root}`",
        "",
        "This export keeps the image-plane AprilTag corner detections and GT projections only.",
        "Tag pose solving, localization, and meter-based reports are disabled.",
        "",
        "## Summary",
        "",
        "| Variant | Backend | Detections / Frame | GT-Visible Hit Rate | Corner RMSE Mean [px] | Corner RMSE P95 [px] |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for variant_name in variant_names:
        summary = backend_summaries[variant_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    variant_name,
                    "new_pupil",
                    _format_optional(_metric(summary, "detection_metrics", "detections_per_frame")),
                    _format_optional(_metric(summary, "detection_metrics", "gt_visible_hit_rate")),
                    _format_optional(_metric(summary, "detection_metrics", "pixel_metrics", "corner_rmse_mean_px")),
                    _format_optional(_metric(summary, "detection_metrics", "pixel_metrics", "corner_rmse_p95_px")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `clean/raw/detections.jsonl` and `clean/localization/new_pupil/detections.jsonl` contain the clean image-corner detections.",
            *(
                [
                    "- `phone_clean/raw/detections.jsonl` and `phone_clean/localization/new_pupil/detections.jsonl` contain the phone-clean image-corner detections."
                ]
                if PHONE_CLEAN_VARIANT_NAME in variant_names
                else []
            ),
            "- `noisy/raw/detections.jsonl` and `noisy/localization/new_pupil/detections.jsonl` contain the noisy image-corner detections.",
            "- `localization/new_pupil/detections_overlay.mp4` in each variant is a quick visual check video with detected tag quads and IDs.",
            "- Each variant's `gt/tag_image_projections.jsonl` contains the GT image-plane tag corners in that variant's pixel coordinate system.",
            "- `raw/camera_frames.jsonl` in each variant carries frame timestamps and image paths.",
            "",
        ]
    )
    report_path = root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _projection_sync_diagnostics(
    variant_dir: Path,
    *,
    detection_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    frame_rows = _load_jsonl(variant_dir / "raw" / "camera_frames.jsonl")
    camera_payload = _load_json(variant_dir / "config_snapshot" / "camera.json")
    tag_gt = _load_json(variant_dir / "gt" / "tag_gt.json")
    tag_rows_by_id = {
        int(tag["tag_id"]): dict(tag)
        for tag in tag_gt.get("tags", [])
        if isinstance(tag, dict)
    }
    frame_rows_by_index = {int(row["frame_index"]): dict(row) for row in frame_rows}
    detections_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in detection_rows:
        detections_by_frame.setdefault(int(row["frame_index"]), []).append(dict(row))

    diagnostic_rows: list[dict[str, Any]] = []
    best_lag_counts: dict[str, int] = {}
    lag_mean_rmse: dict[str, list[float]] = {"-1": [], "0": [], "1": []}
    sorted_frame_rows = [frame_rows_by_index[int(frame_index)] for frame_index in sorted(frame_rows_by_index)]
    row_index_by_frame_index = {int(row["frame_index"]): index for index, row in enumerate(sorted_frame_rows)}
    for frame_index, rows in sorted(detections_by_frame.items()):
        image_frame_row = frame_rows_by_index.get(int(frame_index))
        if image_frame_row is None:
            continue
        image_row_index = row_index_by_frame_index[int(frame_index)]
        lag_metrics: dict[str, Any] = {}
        best_lag: int | None = None
        best_rmse = float("inf")
        for lag in (-1, 0, 1):
            candidate_index = min(max(image_row_index + lag, 0), len(sorted_frame_rows) - 1)
            projection_pose_row = sorted_frame_rows[candidate_index]
            rmse_values: list[float] = []
            matched_tag_ids: list[int] = []
            for detection in rows:
                tag_id = int(detection["tag_id"])
                tag_row = tag_rows_by_id.get(int(tag_id))
                detection_corners = np.asarray(detection.get("corners_xy", ()), dtype=np.float64).reshape(-1, 2)
                if tag_row is None or detection_corners.shape != (4, 2):
                    continue
                projected = _project_rendered_tag_geometry(
                    tag_row=tag_row,
                    image_frame_row=image_frame_row,
                    projection_pose_row=projection_pose_row,
                    camera_payload=camera_payload,
                )
                if not bool(dict(projected["visibility_flags"]).get("fully_visible", False)):
                    continue
                _aligned, rmse = _best_cyclic_corner_alignment(
                    detection_corners,
                    np.asarray(projected["corners_xy"], dtype=np.float64).reshape(4, 2),
                )
                rmse_values.append(float(rmse))
                matched_tag_ids.append(int(tag_id))
            mean_rmse = None if not rmse_values else float(np.mean(np.asarray(rmse_values, dtype=np.float64)))
            lag_key = str(int(lag))
            lag_metrics[lag_key] = {
                "projection_pose_frame_index": int(projection_pose_row["frame_index"]),
                "mean_corner_rmse_px": mean_rmse,
                "matched_tag_ids": [int(tag_id) for tag_id in matched_tag_ids],
            }
            if mean_rmse is not None:
                lag_mean_rmse[lag_key].append(float(mean_rmse))
                if float(mean_rmse) < best_rmse:
                    best_rmse = float(mean_rmse)
                    best_lag = int(lag)
        best_lag_key = None if best_lag is None else str(int(best_lag))
        if best_lag_key is not None:
            best_lag_counts[best_lag_key] = best_lag_counts.get(best_lag_key, 0) + 1
        diagnostic_rows.append(
            {
                "frame_index": int(frame_index),
                "timestamp_s": float(image_frame_row["timestamp_s"]),
                "projection_pose_lag_frames_used": int(_PROJECTION_POSE_LAG_FRAMES),
                "best_lag_frames": None if best_lag is None else int(best_lag),
                "lag_metrics": lag_metrics,
            }
        )
    _write_jsonl(variant_dir / _CAMERA_PROJECTION_SYNC_DIAGNOSTICS_PATH, diagnostic_rows)
    summary = {
        "projection_pose_lag_frames_used": int(_PROJECTION_POSE_LAG_FRAMES),
        "best_lag_frame_counts": {str(key): int(value) for key, value in sorted(best_lag_counts.items())},
        "lag_mean_corner_rmse_px": {
            lag_key: (None if not values else float(np.mean(np.asarray(values, dtype=np.float64))))
            for lag_key, values in lag_mean_rmse.items()
        },
        "diagnostic_row_count": int(len(diagnostic_rows)),
        "diagnostic_path": str((variant_dir / _CAMERA_PROJECTION_SYNC_DIAGNOSTICS_PATH).resolve()),
    }
    _write_json(variant_dir / _CAMERA_PROJECTION_SYNC_SUMMARY_PATH, summary)
    return summary


def _flatten_backend_summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    detection = dict(payload.get("detection_metrics", {}))
    rows.append({"section": "detection", "metric": "frames_total", "value": detection.get("frames_total")})
    rows.append({"section": "detection", "metric": "frames_with_detections", "value": detection.get("frames_with_detections")})
    rows.append({"section": "detection", "metric": "frames_with_gt_visible_tags", "value": detection.get("frames_with_gt_visible_tags")})
    rows.append({"section": "detection", "metric": "detections_total", "value": detection.get("detections_total")})
    rows.append({"section": "detection", "metric": "detections_per_frame", "value": detection.get("detections_per_frame")})
    rows.append({"section": "detection", "metric": "gt_visible_hit_rate", "value": detection.get("gt_visible_hit_rate")})
    pixel_metrics = dict(detection.get("pixel_metrics", {}))
    for metric in ("matched_detection_count", "corner_rmse_mean_px", "corner_rmse_p95_px", "center_error_mean_px", "center_error_p95_px"):
        rows.append({"section": "pixel", "metric": metric, "value": pixel_metrics.get(metric)})
    measurement = dict(payload.get("measurement", {}))
    for mode in ("anchor_only", "multitag"):
        mode_payload = dict(measurement.get(mode, {}))
        for metric in ("mean_position_error_m", "p95_position_error_m", "mean_rotation_error_deg", "p95_rotation_error_deg", "sample_count"):
            rows.append({"section": mode, "metric": metric, "value": mode_payload.get(metric)})
    for tag_id, metrics in dict(detection.get("per_tag_coverage", {})).items():
        rows.append({"section": f"tag_{tag_id}", "metric": "visible_frames", "value": metrics.get("visible_frames")})
        rows.append({"section": f"tag_{tag_id}", "metric": "detected_frames", "value": metrics.get("detected_frames")})
        rows.append({"section": f"tag_{tag_id}", "metric": "visible_hit_rate", "value": metrics.get("visible_hit_rate")})
    return rows


def _materialize_backend_replay_run(variant_dir: Path, backend_dir: Path, detections_path: Path) -> Path:
    replay_dir = backend_dir / "_replay_run"
    _reset_dir(replay_dir)
    raw_dir = replay_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _symlink_or_copy(variant_dir / "raw" / "camera_frames.jsonl", raw_dir / "camera_frames.jsonl")
    _symlink_or_copy(variant_dir / "raw" / "imu.csv", raw_dir / "imu.csv")
    _symlink_or_copy(variant_dir / "raw" / "rgb", raw_dir / "rgb")
    _symlink_or_copy(detections_path, raw_dir / "detections.jsonl")
    _symlink_or_copy(variant_dir / "gt", replay_dir / "gt")
    _symlink_or_copy(variant_dir / "config_snapshot", replay_dir / "config_snapshot")
    _symlink_or_copy(variant_dir / "manifest.json", replay_dir / "manifest.json")
    return replay_dir


def _export_detection_chunk_worker(
    *,
    variant_dir_str: str,
    backend: str,
    scene_payload: dict[str, Any],
    anchor_tag_id: int,
    tag_size_by_id: dict[int, float],
    warmup_rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    output_path_str: str,
    error_path_str: str,
) -> None:
    variant_dir = Path(variant_dir_str)
    output_path = Path(output_path_str)
    error_path = Path(error_path_str)
    try:
        frontend = _backend_frontend(anchor_tag_id, scene_payload, backend)
        for frame_row in warmup_rows:
            frame_packet = _camera_frame_packet_from_row(frame_row)
            image_bgr = cv2.imread(str(variant_dir / str(frame_packet.rgb_path)), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(
                    f"Could not read image for warm-up frame {frame_packet.frame_index}: {variant_dir / frame_packet.rgb_path}"
                )
            frontend.process_bgr_frame(image_bgr, frame_packet=frame_packet, tag_size_by_id=tag_size_by_id)

        serialized_rows: list[dict[str, Any]] = []
        for frame_row in chunk_rows:
            frame_packet = _camera_frame_packet_from_row(frame_row)
            image_bgr = cv2.imread(str(variant_dir / str(frame_packet.rgb_path)), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(
                    f"Could not read image for frame {frame_packet.frame_index}: {variant_dir / frame_packet.rgb_path}"
                )
            pack = frontend.process_bgr_frame(
                image_bgr,
                frame_packet=frame_packet,
                tag_size_by_id=tag_size_by_id,
            )
            serialized_rows.extend(packet.as_json() for packet in pack.detections)
        _write_jsonl(output_path, serialized_rows)
        if error_path.exists():
            error_path.unlink()
        # `pupil_apriltags` can segfault during interpreter teardown even after a successful export.
        # Exit the worker process immediately once the chunk is safely written.
        os._exit(0)
    except BaseException:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        os._exit(1)


def _run_local_backend_detection_export(variant_dir: Path, *, backend: str, output_path: Path) -> dict[str, Any]:
    scene_payload = _load_json(variant_dir / "config_snapshot" / "scene.json")
    anchor_tag_id = int(scene_payload.get("anchor_tag_id", 0))
    tag_size_by_id = _tag_size_lookup(variant_dir)
    frame_rows = _load_jsonl(variant_dir / "raw" / "camera_frames.jsonl")
    export_manifest_path = output_path.with_name(f"{output_path.stem}_export_manifest.json")
    detection_rows: list[dict[str, Any]] = []
    resume_frame_index = 0
    if output_path.exists():
        existing_rows = _load_jsonl(output_path)
        detection_rows = list(existing_rows)
        if export_manifest_path.exists():
            manifest_payload = _load_json(export_manifest_path)
            if bool(manifest_payload.get("complete", False)) and int(manifest_payload.get("frame_count", -1)) == int(len(frame_rows)):
                return {
                    "detections": existing_rows,
                    "frontend_summary": _frontend_summary_from_detections(frame_rows, existing_rows, anchor_tag_id=anchor_tag_id),
                }
            resume_frame_index = min(
                int(manifest_payload.get("frames_processed", 0)),
                int(len(frame_rows)),
            )
        existing_frames = {int(row["frame_index"]) for row in existing_rows}
        if len(existing_frames) == len(frame_rows):
            _write_json(
                export_manifest_path,
                {
                    "backend": str(backend),
                    "complete": True,
                    "frame_count": int(len(frame_rows)),
                    "detection_count": int(len(existing_rows)),
                },
            )
            return {
                "detections": existing_rows,
                "frontend_summary": _frontend_summary_from_detections(frame_rows, existing_rows, anchor_tag_id=anchor_tag_id),
            }
        if resume_frame_index <= 0:
            resume_frame_index = min((max(existing_frames) + 1) if existing_frames else 0, int(len(frame_rows)))
    elif export_manifest_path.exists():
        export_manifest_path.unlink()

    chunk_size = max(int(_DETECTION_EXPORT_CHUNK_FRAMES), 1)
    overlap_frames = max(
        int(scene_payload.get("frontend", {}).get("temporal_max_gap_frames", 0)),
        int(scene_payload.get("frontend", {}).get("roi_recovery_max_gap_frames", 0)),
        int(scene_payload.get("frontend", {}).get("anchor_temporal_clean_max_gap_frames", 0)),
    )
    spawn_context = mp.get_context("fork")
    for chunk_start in range(resume_frame_index, len(frame_rows), chunk_size):
        warm_start = max(0, chunk_start - overlap_frames)
        chunk_frame_rows = frame_rows[chunk_start : chunk_start + chunk_size]
        chunk_output_path = output_path.with_name(f".{output_path.stem}_chunk_{chunk_start:06d}.jsonl")
        chunk_error_path = output_path.with_name(f".{output_path.stem}_chunk_{chunk_start:06d}.error.txt")
        if chunk_output_path.exists():
            chunk_output_path.unlink()
        if chunk_error_path.exists():
            chunk_error_path.unlink()
        worker = spawn_context.Process(
            target=_export_detection_chunk_worker,
            kwargs={
                "variant_dir_str": str(variant_dir),
                "backend": str(backend),
                "scene_payload": scene_payload,
                "anchor_tag_id": int(anchor_tag_id),
                "tag_size_by_id": dict(tag_size_by_id),
                "warmup_rows": list(frame_rows[warm_start:chunk_start]),
                "chunk_rows": list(chunk_frame_rows),
                "output_path_str": str(chunk_output_path),
                "error_path_str": str(chunk_error_path),
            },
        )
        worker.start()
        worker.join()
        if worker.exitcode != 0:
            details = chunk_error_path.read_text(encoding="utf-8") if chunk_error_path.exists() else ""
            raise RuntimeError(
                f"Detection export chunk starting at frame {chunk_start} failed with exit code {worker.exitcode}.\n{details}"
            )
        chunk_rows = _load_jsonl(chunk_output_path)
        chunk_output_path.unlink(missing_ok=True)
        chunk_error_path.unlink(missing_ok=True)
        _append_jsonl(output_path, chunk_rows)
        detection_rows.extend(chunk_rows)
        _write_json(
            export_manifest_path,
            {
                "backend": str(backend),
                "complete": False,
                "frame_count": int(len(frame_rows)),
                "frames_processed": int(min(chunk_start + chunk_size, len(frame_rows))),
                "detection_count": int(len(detection_rows)),
            },
        )
    _write_json(
        export_manifest_path,
        {
            "backend": str(backend),
            "complete": True,
            "frame_count": int(len(frame_rows)),
            "detection_count": int(len(detection_rows)),
        },
    )
    return {
        "detections": detection_rows,
        "frontend_summary": _frontend_summary_from_detections(frame_rows, detection_rows, anchor_tag_id=anchor_tag_id),
    }


def _generate_backend_corner_summary(
    variant_dir: Path,
    *,
    backend: str,
) -> dict[str, Any]:
    backend_dir = variant_dir / "localization" / str(backend)
    summary_path = backend_dir / "backend_summary.json"
    if summary_path.exists():
        return _load_json(summary_path)
    backend_dir.mkdir(parents=True, exist_ok=True)
    detections_path = backend_dir / "detections.jsonl"
    detector_result = _run_local_backend_detection_export(variant_dir, backend=str(backend), output_path=detections_path)
    _symlink_or_copy(detections_path, variant_dir / "raw" / "detections.jsonl")
    frame_rows = _load_jsonl(variant_dir / "raw" / "camera_frames.jsonl")
    detection_rows = _load_jsonl(detections_path)
    gt_projection_rows = _load_jsonl(variant_dir / _GT_TAG_PROJECTIONS_PATH)
    overlay_video = write_detection_overlay_video(
        variant_dir,
        backend=str(backend),
        detections_path=detections_path,
    )
    summary = {
        "backend": str(backend),
        "variant": variant_dir.name,
        "mode": "corners_only",
        "detection_metrics": _detector_metrics(frame_rows, detection_rows, gt_projection_rows),
        "frontend_summary": detector_result.get("frontend_summary", {}),
        "gt_basis": {
            "render_geometry_json": str((variant_dir / _GT_RENDER_GEOMETRY_PATH).resolve()),
            "camera_projection_pose_trace_jsonl": str((variant_dir / _CAMERA_PROJECTION_POSE_TRACE_PATH).resolve()),
            "camera_projection_gt_csv": str((variant_dir / _CAMERA_PROJECTION_GT_PATH).resolve()),
            "projection_pose_lag_frames": int(_PROJECTION_POSE_LAG_FRAMES),
        },
        "paths": {
            "camera_frames_jsonl": str((variant_dir / "raw" / "camera_frames.jsonl").resolve()),
            "raw_detections_jsonl": str((variant_dir / "raw" / "detections.jsonl").resolve()),
            "detections_jsonl": str(detections_path.resolve()),
            "detections_overlay_mp4": str(Path(overlay_video["path"]).resolve()),
            "gt_tag_projections_jsonl": str((variant_dir / _GT_TAG_PROJECTIONS_PATH).resolve()),
        },
        "detection_overlay_video": dict(overlay_video),
    }
    _write_json(summary_path, summary)
    _write_csv(
        backend_dir / "backend_summary.csv",
        ["section", "metric", "value"],
        _flatten_backend_summary_rows(summary),
    )
    return summary


def _generate_backend_localization(
    variant_dir: Path,
    *,
    backend: str,
) -> dict[str, Any]:
    backend_dir = variant_dir / "localization" / str(backend)
    if (backend_dir / "backend_summary.json").exists():
        return _load_json(backend_dir / "backend_summary.json")
    backend_dir.mkdir(parents=True, exist_ok=True)
    detections_path = backend_dir / "detections.jsonl"
    detector_result = _run_local_backend_detection_export(variant_dir, backend=str(backend), output_path=detections_path)
    replay_run_dir = _materialize_backend_replay_run(variant_dir, backend_dir, detections_path)
    measurement_manifest = export_isaac_tag_pose_measurements(replay_run_dir, output_dir=backend_dir)
    calibration = calibrate_tag_measurement_noise(
        replay_run_dir,
        output_dir=backend_dir,
        measurement_manifest=measurement_manifest,
    )
    frame_rows = _load_jsonl(variant_dir / "raw" / "camera_frames.jsonl")
    detection_rows = _load_jsonl(detections_path)
    gt_projection_rows = _load_jsonl(variant_dir / _GT_TAG_PROJECTIONS_PATH)
    sync_diagnostics = _projection_sync_diagnostics(variant_dir, detection_rows=detection_rows)
    overlay_video = write_detection_overlay_video(
        variant_dir,
        backend=str(backend),
        detections_path=detections_path,
    )
    summary = {
        "backend": str(backend),
        "variant": variant_dir.name,
        "detection_metrics": _detector_metrics(frame_rows, detection_rows, gt_projection_rows),
        "frontend_summary": detector_result.get("frontend_summary", {}),
        "measurement": dict(calibration.get("by_mode", {})),
        "gt_basis": {
            "render_geometry_json": str((variant_dir / _GT_RENDER_GEOMETRY_PATH).resolve()),
            "camera_projection_pose_trace_jsonl": str((variant_dir / _CAMERA_PROJECTION_POSE_TRACE_PATH).resolve()),
            "camera_projection_gt_csv": str((variant_dir / _CAMERA_PROJECTION_GT_PATH).resolve()),
            "projection_pose_lag_frames": int(_PROJECTION_POSE_LAG_FRAMES),
        },
        "projection_sync_diagnostics": dict(sync_diagnostics),
        "paths": {
            "detections_jsonl": str(detections_path.resolve()),
            "detections_overlay_mp4": str(Path(overlay_video["path"]).resolve()),
            "gt_tag_projections_jsonl": str((variant_dir / _GT_TAG_PROJECTIONS_PATH).resolve()),
            "anchor_measurements_jsonl": str((backend_dir / "camera_pose_measurements_anchor_only.jsonl").resolve()),
            "multitag_measurements_jsonl": str((backend_dir / "camera_pose_measurements_multitag.jsonl").resolve()),
            "noise_calibration_json": str((backend_dir / "camera_pose_noise_calibration.json").resolve()),
            "measurement_manifest": str(Path(measurement_manifest["manifest_path"]).resolve()),
            "camera_projection_sync_diagnostics_jsonl": str((variant_dir / _CAMERA_PROJECTION_SYNC_DIAGNOSTICS_PATH).resolve()),
            "camera_projection_sync_summary_json": str((variant_dir / _CAMERA_PROJECTION_SYNC_SUMMARY_PATH).resolve()),
        },
        "detection_overlay_video": dict(overlay_video),
    }
    _write_json(backend_dir / "backend_summary.json", summary)
    _write_csv(
        backend_dir / "backend_summary.csv",
        ["section", "metric", "value"],
        _flatten_backend_summary_rows(summary),
    )
    return summary


def _variant_manifest(
    *,
    variant_dir: Path,
    variant_name: str,
    source_capture_run_id: str,
    derived_from: str | None,
    image_noise_policy: dict[str, Any],
    imu_noise_preset: str,
    corners_only: bool,
    include_convenience_exports: bool,
    phone_effect_profile: dict[str, Any] | None = None,
    phone_capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "dataset_name": "tabletop_autodemo_dataset",
        "variant": str(variant_name),
        "source_capture_run_id": str(source_capture_run_id),
        "derived_from_variant": None if derived_from is None else str(derived_from),
        "export_mode": "corners_only" if bool(corners_only) else "full_localization",
        "sensor_rates_hz": {
            "camera": _camera_rate_hz(variant_dir),
            "imu": _imu_rate_hz(variant_dir),
        },
        "timestamp_files": {
            "camera_frames_jsonl": "raw/camera_frames.jsonl",
            "mounted_video_timestamps_csv": "mounted_video_timestamps.csv",
            "imu_csv": "raw/imu.csv",
            "camera_gt_csv": "gt/camera_gt.csv",
            "camera_projection_gt_csv": str(_CAMERA_PROJECTION_GT_PATH),
            "imu_gt_csv": "gt/imu_gt.csv",
            "gt_tag_projections_jsonl": str(_GT_TAG_PROJECTIONS_PATH),
            "camera_projection_pose_trace_jsonl": str(_CAMERA_PROJECTION_POSE_TRACE_PATH),
        },
        "gt_basis": {
            "tag_render_geometry_json": str(_GT_RENDER_GEOMETRY_PATH),
            "projection_pose_lag_frames": int(_PROJECTION_POSE_LAG_FRAMES),
            "projection_pose_source": "previous_frame_extrinsics_snapshot",
        },
        "image_noise_policy": dict(image_noise_policy),
        "imu_noise_preset": str(imu_noise_preset),
        "detector_comparison_set": list(DETECTOR_BACKENDS),
        "corner_detection_outputs": {
            str(backend): f"localization/{backend}/detections.jsonl" for backend in DETECTOR_BACKENDS
        },
        "corner_detection_overlay_videos": {
            str(backend): f"localization/{backend}/detections_overlay.mp4" for backend in DETECTOR_BACKENDS
        },
    }
    if phone_effect_profile is not None:
        payload["phone_effect_profile"] = dict(phone_effect_profile)
    if phone_capture is not None:
        payload["phone_capture"] = dict(phone_capture)
    if bool(include_convenience_exports):
        payload["convenience_exports"] = {
            "mounted_video_mp4": "mounted_video.mp4",
            "mounted_video_timestamps_csv": "mounted_video_timestamps.csv",
            "gt_path_csv": "gt_path.csv",
        }
    return payload


def _write_variant_manifest(
    variant_dir: Path,
    *,
    variant_name: str,
    source_capture_run_id: str,
    derived_from: str | None,
    image_noise_policy: dict[str, Any],
    imu_noise_preset: str,
    corners_only: bool,
    include_convenience_exports: bool,
    phone_effect_profile: dict[str, Any] | None = None,
    phone_capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _variant_manifest(
        variant_dir=variant_dir,
        variant_name=variant_name,
        source_capture_run_id=source_capture_run_id,
        derived_from=derived_from,
        image_noise_policy=image_noise_policy,
        imu_noise_preset=imu_noise_preset,
        corners_only=bool(corners_only),
        include_convenience_exports=bool(include_convenience_exports),
        phone_effect_profile=phone_effect_profile,
        phone_capture=phone_capture,
    )
    _write_json(variant_dir / "dataset_manifest.json", payload)
    return payload


def _capture_clean_master_run(
    output_dir: Path,
    *,
    profile_key: str,
    seed: int,
) -> dict[str, Any]:
    profile = get_demo_profile(profile_key)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    scene_config_path = REPO_ROOT / profile.scene_config_path
    robot_config_path = REPO_ROOT / profile.robot_config_path
    camera_config_path = REPO_ROOT / profile.camera_config_path
    imu_config_path = REPO_ROOT / profile.imu_config_path
    actuation_config_path = REPO_ROOT / "config/isaac/actuation/none.yaml"
    estimation_config_path = REPO_ROOT / profile.estimation_config_path
    control_config_path = REPO_ROOT / profile.control_config_path
    scene_payload = load_isaac_yaml(scene_config_path)
    scene_payload["observer_cameras"] = []
    scene_payload["observer_camera_prim_paths"] = []
    loop_duration_s = _scene_loop_duration_s(scene_payload)
    run_id = datetime.now(timezone.utc).strftime("tabletop_autodemo_capture_%Y%m%d_%H%M%S")
    bootstrap = IsaacAppBootstrapConfig(
        scene_config_path=str(scene_config_path),
        robot_config_path=str(robot_config_path),
        camera_config_path=str(camera_config_path),
        imu_config_path=str(imu_config_path),
        actuation_config_path=str(actuation_config_path),
        estimation_config_path=str(estimation_config_path),
        control_config_path=str(control_config_path),
        run_id=run_id,
        run_dir=str(output_dir),
        headless=True,
        width=int(load_isaac_yaml(camera_config_path).get("width_px", 1920)),
        height=int(load_isaac_yaml(camera_config_path).get("height_px", 1080)),
        streaming_backend="webrtc",
        seed=int(seed),
        duration_s=float(loop_duration_s),
        estimator_mode=profile.default_estimator_mode,
        controller_mode=profile.default_runtime_controller_mode,
        bootstrap_control_policy=profile.bootstrap_control_policy,
    )
    runtime = create_runtime(bootstrap)
    runtime.config.config_payloads["scene"] = scene_payload
    runtime.config.duration_s = float(loop_duration_s)
    writer = IsaacRunWriter(output_dir)
    runtime.config.run_id = run_id
    runtime.config.run_dir = str(output_dir)
    runtime.set_writer(writer, reset_recording_baseline=True)
    for name, payload in runtime.config.config_payloads.items():
        writer.write_config_snapshot(name, payload)
    writer.write_manifest(
        build_run_manifest(
            repo_root=REPO_ROOT,
            run_id=run_id,
            isaac_sim_version="interactive_demo",
            stage_usd_path=runtime.config.stage_path if Path(runtime.config.stage_path).exists() else f"programmatic:{profile.key}",
            robot_preset=runtime.config.robot_preset,
            anchor_tag_id=runtime.config.anchor_tag_id,
            estimator_mode=runtime.config.estimator_mode,
            controller_mode=runtime.config.controller_mode,
            bootstrap_control_policy=runtime.config.bootstrap_control_policy,
            noise_presets={"imu": "ideal", "vision": "clean", "actuation": "none"},
            random_seed=int(seed),
            controller_config=json.loads(json.dumps(runtime.config.config_payloads["control"])),
            estimator_config=json.loads(json.dumps(runtime.config.config_payloads["estimation"])),
            ros2_bridge_used=False,
        )
    )
    runtime.start()
    try:
        runtime.set_interactive_control_mode("auto_demo")
        runtime.set_imu_noise_preset("imu_ideal")
        runtime.set_vision_noise_mode("clean")
        runtime.set_actuation_config_payload(load_isaac_yaml(actuation_config_path))
        runtime.set_tag_detection_mode("off")
        summary = runtime.run_until_done()
    finally:
        runtime.shutdown()
    return {
        "run_id": run_id,
        "run_dir": str(output_dir),
        "duration_s": float(loop_duration_s),
        "summary": summary.as_json(),
    }


def _capture_clean_master_run_subprocess(
    output_dir: Path,
    *,
    profile_key: str,
    seed: int,
    corners_only: bool,
    skip_phone_export: bool,
    phone_profile_path: str | Path | None,
) -> dict[str, Any]:
    script_path = REPO_ROOT / "scripts" / "export_tabletop_autodemo_dataset.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing dataset export script: {script_path}")
    env = os.environ.copy()
    existing_pythonpath = str(env.get("PYTHONPATH", "")).strip()
    if existing_pythonpath:
        env["PYTHONPATH"] = f"src:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = "src"
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--output-root",
            str(output_dir),
            "--seed",
            str(int(seed)),
            "--capture-only",
            *(tuple() if bool(corners_only) else ("--full-analysis",)),
            *(("--skip-phone-export",) if bool(skip_phone_export) else tuple()),
            *(("--phone-profile", str(phone_profile_path)) if phone_profile_path is not None else tuple()),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Fresh clean capture subprocess failed.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError("Fresh clean capture subprocess returned no JSON payload.")
    payload = _extract_json_object_from_text(stdout)
    if payload is not None:
        return payload
    manifest_path = output_dir / "manifest.json"
    camera_frames_path = output_dir / "raw" / "camera_frames.jsonl"
    if manifest_path.exists() and camera_frames_path.exists():
        manifest_payload = _load_json(manifest_path)
        return {
            "run_id": str(manifest_payload.get("run_id", output_dir.name)),
            "run_dir": str(output_dir.resolve()),
            "duration_s": None,
            "summary": {
                "complete": True,
                "source": "recovered_from_capture_directory_without_stdout_json",
            },
        }
    raise RuntimeError(
        "Fresh clean capture subprocess did not return valid JSON.\n"
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )


def _stage_source_run_dir(source_run_path: Path, output_root: Path) -> Path:
    resolved_source = source_run_path.resolve()
    resolved_root = output_root.resolve()
    try:
        inside_root = resolved_source.is_relative_to(resolved_root)
    except AttributeError:  # pragma: no cover - Python < 3.9 fallback
        inside_root = str(resolved_source).startswith(str(resolved_root))
    if not inside_root:
        return resolved_source
    staging_dir = resolved_root.parent / f".{resolved_root.name}_source_staging"
    _reset_dir(staging_dir)
    shutil.copytree(resolved_source, staging_dir, dirs_exist_ok=True)
    return staging_dir


def _prepare_variant_base(
    source_run_dir: Path,
    variant_dir: Path,
    *,
    variant_name: str,
    source_capture_run_id: str,
    derived_from: str | None,
    image_noise_policy: dict[str, Any],
    imu_noise_preset: str,
    corners_only: bool,
    include_convenience_exports: bool,
) -> None:
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    shutil.copytree(source_run_dir, variant_dir)
    for observer_path in (
        variant_dir / "raw" / "observer_camera_frames.jsonl",
        variant_dir / "raw" / "observer_rgb",
    ):
        if observer_path.exists() or observer_path.is_symlink():
            if observer_path.is_dir() and not observer_path.is_symlink():
                shutil.rmtree(observer_path)
            else:
                observer_path.unlink()
    if bool(corners_only):
        _set_variant_scene_frontend_option(variant_dir, key="solve_tag_pose", value=False)
    _compute_gt_tag_image_projections(variant_dir)
    if bool(include_convenience_exports):
        write_dataset_convenience_exports(variant_dir)
    _write_variant_manifest(
        variant_dir,
        variant_name=variant_name,
        source_capture_run_id=source_capture_run_id,
        derived_from=derived_from,
        image_noise_policy=image_noise_policy,
        imu_noise_preset=imu_noise_preset,
        corners_only=bool(corners_only),
        include_convenience_exports=bool(include_convenience_exports),
    )


def _finalize_variant_exports(
    variant_dir: Path,
    *,
    variant_name: str,
    source_capture_run_id: str,
    derived_from: str | None,
    image_noise_policy: dict[str, Any],
    imu_noise_preset: str,
    corners_only: bool,
    include_convenience_exports: bool,
    phone_profile_path: str | Path | None,
    phone_effect_profile: dict[str, Any] | None,
    include_phone_capture: bool,
) -> dict[str, Any]:
    phone_capture = None
    if bool(include_convenience_exports):
        write_dataset_convenience_exports(variant_dir)
    if bool(include_phone_capture):
        phone_capture = write_phone_capture_mirror(variant_dir, phone_profile_path=phone_profile_path)
    return _write_variant_manifest(
        variant_dir,
        variant_name=variant_name,
        source_capture_run_id=source_capture_run_id,
        derived_from=derived_from,
        image_noise_policy=image_noise_policy,
        imu_noise_preset=imu_noise_preset,
        corners_only=bool(corners_only),
        include_convenience_exports=bool(include_convenience_exports),
        phone_effect_profile=phone_effect_profile,
        phone_capture=phone_capture,
    )


def _finalize_variant_in_place(
    variant_dir: Path,
    *,
    variant_name: str,
    source_capture_run_id: str,
    derived_from: str | None,
    image_noise_policy: dict[str, Any],
    imu_noise_preset: str,
    corners_only: bool,
    include_convenience_exports: bool,
) -> None:
    for observer_path in (
        variant_dir / "raw" / "observer_camera_frames.jsonl",
        variant_dir / "raw" / "observer_rgb",
    ):
        if observer_path.exists() or observer_path.is_symlink():
            if observer_path.is_dir() and not observer_path.is_symlink():
                shutil.rmtree(observer_path)
            else:
                observer_path.unlink()
    if bool(corners_only):
        _set_variant_scene_frontend_option(variant_dir, key="solve_tag_pose", value=False)
    _compute_gt_tag_image_projections(variant_dir)
    if bool(include_convenience_exports):
        write_dataset_convenience_exports(variant_dir)
    _write_variant_manifest(
        variant_dir,
        variant_name=variant_name,
        source_capture_run_id=source_capture_run_id,
        derived_from=derived_from,
        image_noise_policy=image_noise_policy,
        imu_noise_preset=imu_noise_preset,
        corners_only=bool(corners_only),
        include_convenience_exports=bool(include_convenience_exports),
    )


def write_tabletop_autodemo_dataset_report(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).resolve()
    variant_names = _phone_capture_variant_names(root)
    backend_summaries: dict[str, dict[str, dict[str, Any]]] = {variant_name: {} for variant_name in variant_names}
    for variant_name in variant_names:
        variant_dir = root / variant_name
        for backend in DETECTOR_BACKENDS:
            summary_path = variant_dir / "localization" / backend / "backend_summary.json"
            backend_summaries[variant_name][backend] = _load_json(summary_path)

    def _metric(summary: dict[str, Any], *path: str) -> Any:
        value: Any = summary
        for item in path:
            if not isinstance(value, dict):
                return None
            value = value.get(item)
        return value

    def _format_optional(value: Any, *, precision: int = 4) -> str:
        if value in (None, ""):
            return "n/a"
        return f"{float(value):.{precision}f}"

    def _pose_degradation() -> str:
        anchor_ratios: list[float] = []
        multitag_ratios: list[float] = []
        for backend in DETECTOR_BACKENDS:
            baseline_variant = PHONE_CLEAN_VARIANT_NAME if PHONE_CLEAN_VARIANT_NAME in backend_summaries else "clean"
            clean_anchor = _metric(backend_summaries[baseline_variant][backend], "measurement", "anchor_only", "mean_position_error_m")
            noisy_anchor = _metric(backend_summaries["noisy"][backend], "measurement", "anchor_only", "mean_position_error_m")
            clean_multi = _metric(backend_summaries[baseline_variant][backend], "measurement", "multitag", "mean_position_error_m")
            noisy_multi = _metric(backend_summaries["noisy"][backend], "measurement", "multitag", "mean_position_error_m")
            if clean_anchor not in (None, 0.0) and noisy_anchor is not None:
                anchor_ratios.append(float(noisy_anchor) / max(float(clean_anchor), 1e-9))
            if clean_multi not in (None, 0.0) and noisy_multi is not None:
                multitag_ratios.append(float(noisy_multi) / max(float(clean_multi), 1e-9))
        if not anchor_ratios or not multitag_ratios:
            return "insufficient data"
        return "anchor_only" if float(np.mean(anchor_ratios)) > float(np.mean(multitag_ratios)) else "multitag"

    def _sync_summary_line(variant_name: str, backend: str) -> str:
        summary = backend_summaries[variant_name][backend]
        sync_summary = dict(summary.get("projection_sync_diagnostics", {}))
        best_lag_counts = dict(sync_summary.get("best_lag_frame_counts", {}))
        lag_used = sync_summary.get("projection_pose_lag_frames_used")
        lag_minus_one = best_lag_counts.get("-1", 0)
        lag_zero = best_lag_counts.get("0", 0)
        lag_plus_one = best_lag_counts.get("1", 0)
        return (
            f"- `{variant_name}/{backend}` projection GT uses frame-index matching with a configured pose lag of "
            f"`{lag_used}` frame(s); best-fit lag counts were `-1:{lag_minus_one}`, `0:{lag_zero}`, `+1:{lag_plus_one}`."
        )

    lines = [
        "# Tabletop Auto-Demo Dataset Report",
        "",
        f"Dataset root: `{root}`",
        "",
        "## Variants",
        "",
        f"- Clean: [`clean/`](clean)",
        *(["- Phone clean: [`phone_clean/`](phone_clean)"] if PHONE_CLEAN_VARIANT_NAME in variant_names else []),
        f"- Noisy: [`noisy/`](noisy)",
        "- Pixel precision analysis: [`pixel_precision_report.md`](pixel_precision_report.md)",
        "",
        "## GT Basis",
        "",
        "- GT tag corners are projected from the rendered visible marker square, not the outer printed board.",
        "- The rendered tag face plane is used consistently for export GT, viewer overlays, and PnP local points.",
        "- Camera-pose GT is matched by frame index first and exported at `gt/camera_projection_gt.csv`.",
        "- Full projection basis metadata is exported per variant at `gt/tag_render_geometry.json` and `gt/camera_projection_pose_trace.jsonl`.",
        "",
        "## Localization Summary",
        "",
        "| Variant | Backend | Detections / Frame | GT-Visible Hit Rate | Corner RMSE Mean [px] | Corner RMSE P95 [px] | Anchor Mean / P95 Pos Error [m] | Multitag Mean / P95 Pos Error [m] |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant_name in variant_names:
        for backend in DETECTOR_BACKENDS:
            summary = backend_summaries[variant_name][backend]
            lines.append(
                "| "
                + " | ".join(
                    [
                        variant_name,
                        f"[{backend}]({variant_name}/localization/{backend})",
                        f"{float(_metric(summary, 'detection_metrics', 'detections_per_frame') or 0.0):.3f}",
                        _format_optional(_metric(summary, "detection_metrics", "gt_visible_hit_rate"), precision=3),
                        _format_optional(_metric(summary, "detection_metrics", "pixel_metrics", "corner_rmse_mean_px"), precision=3),
                        _format_optional(_metric(summary, "detection_metrics", "pixel_metrics", "corner_rmse_p95_px"), precision=3),
                        f"{_format_optional(_metric(summary, 'measurement', 'anchor_only', 'mean_position_error_m'), precision=4)} / {_format_optional(_metric(summary, 'measurement', 'anchor_only', 'p95_position_error_m'), precision=4)}",
                        f"{_format_optional(_metric(summary, 'measurement', 'multitag', 'mean_position_error_m'), precision=4)} / {_format_optional(_metric(summary, 'measurement', 'multitag', 'p95_position_error_m'), precision=4)}",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Per-Tag Coverage",
            "",
            "| Variant | Backend | Tag | Visible Frames | Detected Frames | Visible Hit Rate |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant_name in variant_names:
        for backend in DETECTOR_BACKENDS:
            summary = backend_summaries[variant_name][backend]
            per_tag = dict(_metric(summary, "detection_metrics", "per_tag_coverage") or {})
            for tag_id, metrics in sorted(per_tag.items(), key=lambda item: int(item[0])):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            variant_name,
                            backend,
                            str(tag_id),
                            str(metrics.get("visible_frames")),
                            str(metrics.get("detected_frames")),
                            "n/a"
                            if metrics.get("visible_hit_rate") is None
                            else f"{float(metrics['visible_hit_rate']):.3f}",
                        ]
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            "- Supported detector on clean data: `new_pupil`",
            *(["- Supported detector on phone-clean data: `new_pupil`"] if PHONE_CLEAN_VARIANT_NAME in variant_names else []),
            "- Supported detector on noisy data: `new_pupil`",
            "- GT tag image-plane reference is exported per variant at `gt/tag_image_projections.jsonl`.",
            _sync_summary_line("clean", "new_pupil"),
            *([_sync_summary_line(PHONE_CLEAN_VARIANT_NAME, "new_pupil")] if PHONE_CLEAN_VARIANT_NAME in variant_names else []),
            _sync_summary_line("noisy", "new_pupil"),
            f"- Pose mode that degrades most under noisy images: `{_pose_degradation()}`",
            "- Batch BI is intentionally excluded from the slim branch dataset export.",
            "",
        ]
    )
    report_path = root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def export_tabletop_autodemo_dataset(
    *,
    output_root: str | Path = DEFAULT_DATASET_ROOT,
    profile_key: str = "tabletop_replica",
    seed: int = _DEFAULT_SEED,
    source_run_dir: str | Path | None = None,
    capture_only: bool = False,
    corners_only: bool = False,
    skip_phone_export: bool = False,
    phone_profile_path: str | Path | None = DEFAULT_PHONE_PROFILE_PATH,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    phone_model = None if bool(skip_phone_export) else _load_phone_model(phone_profile_path)
    if bool(capture_only):
        _reset_dir(root)
        capture_payload = _capture_clean_master_run(root, profile_key=profile_key, seed=int(seed))
        source_capture_run_id = str(capture_payload["run_id"])
        _finalize_variant_in_place(
            root,
            variant_name="clean",
            source_capture_run_id=source_capture_run_id,
            derived_from=None,
            image_noise_policy={"name": "clean"},
            imu_noise_preset="imu_ideal",
            corners_only=bool(corners_only),
            include_convenience_exports=not bool(corners_only),
        )
        if not bool(skip_phone_export):
            _finalize_variant_exports(
                root,
                variant_name="clean",
                source_capture_run_id=source_capture_run_id,
                derived_from=None,
                image_noise_policy={"name": "clean"},
                imu_noise_preset="imu_ideal",
                corners_only=bool(corners_only),
                include_convenience_exports=not bool(corners_only),
                phone_profile_path=phone_profile_path,
                phone_effect_profile=_phone_effect_profile(phone_model, physical=False, noisy=False),
                include_phone_capture=True,
            )
        manifest = _load_json(root / "dataset_manifest.json") if (root / "dataset_manifest.json").exists() else {}
        payload = {
            "dataset_root": str(root),
            "clean_dir": str(root),
            "source_capture_run_id": source_capture_run_id,
            "capture": capture_payload,
            "capture_only": True,
            "export_mode": "corners_only" if bool(corners_only) else "full_localization",
            "phone_export_enabled": not bool(skip_phone_export),
            "phone_capture": manifest.get("phone_capture"),
        }
        _write_json(root / "dataset_export_manifest.json", payload)
        return payload

    clean_dir = root / "clean"
    phone_clean_dir = root / PHONE_CLEAN_VARIANT_NAME
    noisy_dir = root / "noisy"
    staged_source_dir: Path | None = None
    if source_run_dir is None:
        capture_source_dir = root.parent / f".{root.name}_capture_source"
        capture_payload = _capture_clean_master_run_subprocess(
            capture_source_dir,
            profile_key=profile_key,
            seed=int(seed),
            corners_only=bool(corners_only),
            skip_phone_export=bool(skip_phone_export),
            phone_profile_path=phone_profile_path,
        )
        staged_source_dir = capture_source_dir.resolve()
        source_capture_run_id = str(capture_payload["run_id"])
    else:
        source_run_path = Path(source_run_dir).resolve()
        staged_source_dir = _stage_source_run_dir(source_run_path, root)
        source_capture_run_id = str(_load_json(staged_source_dir / "manifest.json").get("run_id", staged_source_dir.name))
        capture_payload = {"run_id": source_capture_run_id, "run_dir": str(staged_source_dir), "duration_s": None, "summary": {}}

    try:
        _reset_dir(root)
        _prepare_variant_base(
            staged_source_dir,
            clean_dir,
            variant_name="clean",
            source_capture_run_id=source_capture_run_id,
            derived_from=None,
            image_noise_policy={"name": "clean"},
            imu_noise_preset="imu_ideal",
            corners_only=bool(corners_only),
            include_convenience_exports=not bool(corners_only),
        )
        if not bool(skip_phone_export):
            _finalize_variant_exports(
                clean_dir,
                variant_name="clean",
                source_capture_run_id=source_capture_run_id,
                derived_from=None,
                image_noise_policy={"name": "clean"},
                imu_noise_preset="imu_ideal",
                corners_only=bool(corners_only),
                include_convenience_exports=not bool(corners_only),
                phone_profile_path=phone_profile_path,
                phone_effect_profile=_phone_effect_profile(phone_model, physical=False, noisy=False),
                include_phone_capture=True,
            )
    finally:
        if staged_source_dir is not None and staged_source_dir.name in {
            f".{root.name}_capture_source",
            f".{root.name}_source_staging",
        }:
            shutil.rmtree(staged_source_dir, ignore_errors=True)
    if not bool(skip_phone_export):
        apply_phone_physical_variant(
            clean_dir,
            phone_clean_dir,
            phone_profile_path=phone_profile_path,
            seed=int(seed),
        )
        _finalize_variant_exports(
            phone_clean_dir,
            variant_name=PHONE_CLEAN_VARIANT_NAME,
            source_capture_run_id=source_capture_run_id,
            derived_from="clean",
            image_noise_policy={"name": "pixel_9a_physical_clean"},
            imu_noise_preset="imu_ideal",
            corners_only=bool(corners_only),
            include_convenience_exports=not bool(corners_only),
            phone_profile_path=phone_profile_path,
            phone_effect_profile=_phone_effect_profile(phone_model, physical=True, noisy=False),
            include_phone_capture=True,
        )
        derive_noisy_dataset_variant(phone_clean_dir, noisy_dir, seed=int(seed), imu_noise_preset="imu_nominal_phone")
        noisy_derived_from = PHONE_CLEAN_VARIANT_NAME
        noisy_phone_profile = _phone_effect_profile(phone_model, physical=True, noisy=True)
    else:
        derive_noisy_dataset_variant(clean_dir, noisy_dir, seed=int(seed), imu_noise_preset="imu_nominal_phone")
        noisy_derived_from = "clean"
        noisy_phone_profile = None
    if bool(corners_only):
        _set_variant_scene_frontend_option(noisy_dir, key="solve_tag_pose", value=False)
    _finalize_variant_exports(
        noisy_dir,
        variant_name="noisy",
        source_capture_run_id=source_capture_run_id,
        derived_from=noisy_derived_from,
        image_noise_policy=dict(_DEFAULT_IMAGE_NOISE_POLICY),
        imu_noise_preset="imu_nominal_phone",
        corners_only=bool(corners_only),
        include_convenience_exports=not bool(corners_only),
        phone_profile_path=phone_profile_path,
        phone_effect_profile=noisy_phone_profile,
        include_phone_capture=not bool(skip_phone_export),
    )
    variant_dirs = [("clean", clean_dir)]
    if not bool(skip_phone_export):
        variant_dirs.append((PHONE_CLEAN_VARIANT_NAME, phone_clean_dir))
    variant_dirs.append(("noisy", noisy_dir))
    if bool(corners_only):
        detection_summary: dict[str, dict[str, Any]] = {variant_name: {} for variant_name, _variant_dir in variant_dirs}
        for variant_name, variant_dir in variant_dirs:
            detection_summary[variant_name]["new_pupil"] = _generate_backend_corner_summary(
                variant_dir,
                backend="new_pupil",
            )
        report_path = write_tabletop_autodemo_corner_dataset_report(root)
        payload = {
            "dataset_root": str(root),
            "clean_dir": str(clean_dir),
            "phone_clean_dir": None if bool(skip_phone_export) else str(phone_clean_dir),
            "noisy_dir": str(noisy_dir),
            "variants": [variant_name for variant_name, _variant_dir in variant_dirs],
            "source_capture_run_id": source_capture_run_id,
            "capture_source": "fresh_simulation" if source_run_dir is None else "existing_source_run",
            "capture": capture_payload,
            "export_mode": "corners_only",
            "phone_export_enabled": not bool(skip_phone_export),
            "phone_profile_path": _repo_relative_path(phone_profile_path or DEFAULT_PHONE_PROFILE_PATH),
            "detection_summary": detection_summary,
            "report_path": str(report_path),
        }
    else:
        localization_summary: dict[str, dict[str, Any]] = {variant_name: {} for variant_name, _variant_dir in variant_dirs}
        for variant_name, variant_dir in variant_dirs:
            localization_summary[variant_name]["new_pupil"] = _generate_backend_localization(variant_dir, backend="new_pupil")
        pixel_precision_report_path = write_tabletop_autodemo_pixel_precision_report(root)
        report_path = write_tabletop_autodemo_dataset_report(root)
        payload = {
            "dataset_root": str(root),
            "clean_dir": str(clean_dir),
            "phone_clean_dir": None if bool(skip_phone_export) else str(phone_clean_dir),
            "noisy_dir": str(noisy_dir),
            "variants": [variant_name for variant_name, _variant_dir in variant_dirs],
            "source_capture_run_id": source_capture_run_id,
            "capture_source": "fresh_simulation" if source_run_dir is None else "existing_source_run",
            "capture": capture_payload,
            "export_mode": "full_localization",
            "phone_export_enabled": not bool(skip_phone_export),
            "phone_profile_path": _repo_relative_path(phone_profile_path or DEFAULT_PHONE_PROFILE_PATH),
            "localization_summary": localization_summary,
            "pixel_precision_report_path": str(pixel_precision_report_path),
            "report_path": str(report_path),
        }
    _write_json(root / "dataset_export_manifest.json", payload)
    return payload


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_PHONE_PROFILE_PATH",
    "DETECTOR_BACKENDS",
    "PHONE_CLEAN_VARIANT_NAME",
    "apply_phone_physical_variant",
    "derive_noisy_dataset_variant",
    "export_tabletop_autodemo_dataset",
    "write_detection_overlay_video",
    "write_phone_capture_mirror",
    "write_dataset_convenience_exports",
    "write_tabletop_autodemo_corner_dataset_report",
    "write_tabletop_autodemo_pixel_precision_report",
    "write_tabletop_autodemo_dataset_report",
]
