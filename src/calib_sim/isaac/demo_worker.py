"""Worker-process helpers shared by the slim tabletop standard app."""

from __future__ import annotations

import base64
from collections.abc import Mapping
import copy
from copy import deepcopy
from datetime import datetime, timezone
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np

from calib_sim.isaac.demo_profiles import DEFAULT_PROFILE_KEY, IsaacDemoProfile, get_demo_profile, list_demo_profiles
from calib_sim.isaac.demo_protocol import (
    empty_snapshot,
    initial_catalog,
    snapshot_with_error,
    validate_control_mode,
    validate_noise_modes,
)
from calib_sim.tag_service.detector import rendered_apriltag_local_corners_m


REPO_ROOT = Path(__file__).resolve().parents[3]
PRIMARY_STREAM_MAX_WIDTH_PX = 1280
OBSERVER_STREAM_MAX_WIDTH_PX = 512
PRIMARY_STREAM_JPEG_QUALITY = 90
OBSERVER_STREAM_JPEG_QUALITY = 76
SNAPSHOT_PUBLISH_INTERVAL_S = 0.125
FORCED_SNAPSHOT_PUBLISH_INTERVAL_S = 0.5


def _default_detector_mode_for_profile(profile: IsaacDemoProfile) -> str:
    del profile
    return "new_pupil"


def _repo_path(path_like: str | Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _encode_jpeg_data_url(
    image_rgb: np.ndarray | None,
    *,
    max_width_px: int | None = None,
    jpeg_quality: int = 80,
) -> str:
    if image_rgb is None:
        return ""
    image = np.asarray(image_rgb, dtype=np.uint8)
    if max_width_px is not None and int(max_width_px) > 0 and image.shape[1] > int(max_width_px):
        scale = float(max_width_px) / float(image.shape[1])
        resized_height = max(int(round(float(image.shape[0]) * scale)), 1)
        image = cv2.resize(image, (int(max_width_px), int(resized_height)), interpolation=cv2.INTER_AREA)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not success:
        return ""
    return f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


def _cached_frame_data_url(
    *,
    cache: dict[str, dict[str, Any]],
    stream_key: str,
    frame_index: int | None,
    image_rgb: np.ndarray | None,
    max_width_px: int,
    jpeg_quality: int,
) -> str:
    if image_rgb is None:
        cache.pop(stream_key, None)
        return ""
    normalized_frame_index = None if frame_index is None else int(frame_index)
    cached = cache.get(stream_key)
    if cached is not None and cached.get("frame_index") == normalized_frame_index:
        return str(cached.get("data_url", ""))
    data_url = _encode_jpeg_data_url(
        image_rgb,
        max_width_px=max_width_px,
        jpeg_quality=jpeg_quality,
    )
    cache[stream_key] = {"frame_index": normalized_frame_index, "data_url": data_url}
    return data_url


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quaternion_wxyz_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _project_tag_corners_for_display(
    *,
    camera_frame_packet: Any,
    tag_pose: Any,
    local_tag_points_m: np.ndarray,
) -> np.ndarray | None:
    if camera_frame_packet is None:
        return None
    points = np.asarray(local_tag_points_m, dtype=np.float64).reshape(-1, 3)
    if points.shape != (4, 3):
        return None
    intrinsics = dict(getattr(camera_frame_packet, "intrinsics_snapshot", {}) or {})
    fx = float(intrinsics.get("fx_px", 0.0))
    fy = float(intrinsics.get("fy_px", 0.0))
    cx = float(intrinsics.get("cx_px", 0.0))
    cy = float(intrinsics.get("cy_px", 0.0))
    if fx <= 0.0 or fy <= 0.0:
        return None
    distortion = np.asarray(
        intrinsics.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    extrinsics = dict(getattr(camera_frame_packet, "extrinsics_snapshot", {}) or {})
    position_world_m = np.asarray(extrinsics.get("position_world_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    orientation_wxyz = np.asarray(extrinsics.get("orientation_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64).reshape(4)
    rotation_wc = _quaternion_wxyz_to_rotation_matrix(orientation_wxyz)
    rotation_cw = rotation_wc.T
    translation_cw = -rotation_cw @ position_world_m
    rotation_wt = np.asarray(getattr(tag_pose, "rotation_wt", np.eye(3)), dtype=np.float64).reshape(3, 3)
    position_wt = np.asarray(getattr(tag_pose, "position_world_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    world_points = (rotation_wt @ points.T).T + position_wt.reshape(1, 3)
    camera_points = (rotation_cw @ (world_points - position_world_m.reshape(1, 3)).T).T
    if np.any(camera_points[:, 2] <= 1e-6):
        return None
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    rvec_cw, _ = cv2.Rodrigues(rotation_cw)
    projected_points, _ = cv2.projectPoints(
        world_points,
        rvec_cw,
        translation_cw.reshape(3, 1),
        camera_matrix,
        distortion,
    )
    projected = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
    width_px = int(getattr(camera_frame_packet, "image_width_px", 0))
    height_px = int(getattr(camera_frame_packet, "image_height_px", 0))
    if width_px > 0 and height_px > 0:
        margin_px = 96.0
        if (
            np.max(projected[:, 0]) < -margin_px
            or np.min(projected[:, 0]) > float(width_px) + margin_px
            or np.max(projected[:, 1]) < -margin_px
            or np.min(projected[:, 1]) > float(height_px) + margin_px
        ):
            return None
    return projected


def _anchor_display_source(detection: Any) -> str:
    if not bool(getattr(detection, "is_anchor", False)):
        return "detected_corners"
    flags = dict(getattr(detection, "visibility_flags", {}) or {})
    if bool(flags.get("anchor_temporal_degraded", False)):
        return "projected_anchor_fallback"
    return "detected_corners"


def _detection_measurement_source(detection: Any) -> str:
    source = str(getattr(detection, "measurement_source", "") or "").strip().lower()
    if source:
        return source
    backend = str(getattr(detection, "detector_backend", "") or "").strip().lower()
    if "temporal_track" in backend:
        return "temporal_tracked"
    if "retry_upsampled" in backend:
        return "retry_detected"
    if "rejected_candidate_match" in backend or "bright_quad_match" in backend:
        return "heuristic_fallback"
    return "native_detected" if backend else "unknown"


def _tabletop_frontend_config(runtime: Any) -> dict[str, Any]:
    scene_payload = getattr(getattr(runtime, "config", None), "config_payloads", {}).get("scene", {})
    frontend = scene_payload.get("frontend", {})
    return dict(frontend) if isinstance(frontend, Mapping) else {}


def _mapped_auxiliary_projected_fallback_tag_ids(runtime: Any) -> set[int]:
    frontend = _tabletop_frontend_config(runtime)
    raw_ids = frontend.get("auxiliary_projected_fallback_tag_ids", [155, 201])
    if not isinstance(raw_ids, (list, tuple)):
        return {155, 201}
    return {int(tag_id) for tag_id in raw_ids}


def _auxiliary_temporal_clean_max_gap_frames(runtime: Any) -> int:
    frontend = _tabletop_frontend_config(runtime)
    return max(int(frontend.get("auxiliary_temporal_clean_max_gap_frames", 1)), 0)


def _auxiliary_projected_persistence_max_gap_frames(runtime: Any) -> int:
    frontend = _tabletop_frontend_config(runtime)
    return max(int(frontend.get("auxiliary_projected_persistence_max_gap_frames", 2)), 0)


def _display_source_for_detection(runtime: Any, detection: Any) -> str:
    if bool(getattr(detection, "is_anchor", False)):
        return _anchor_display_source(detection)
    measurement_source = _detection_measurement_source(detection)
    if measurement_source != "temporal_tracked":
        return "detected_corners"
    temporal_gap_frames = int(getattr(detection, "temporal_gap_frames", 0))
    clean_gap_frames = _auxiliary_temporal_clean_max_gap_frames(runtime)
    if temporal_gap_frames <= clean_gap_frames:
        return "detected_corners"
    if int(getattr(detection, "tag_id", -1)) in _mapped_auxiliary_projected_fallback_tag_ids(runtime):
        return "projected_auxiliary_fallback"
    return "detected_corners"


class _NullIsaacRunWriter:
    def save_rgb_image(self, **_: Any) -> str:
        return ""

    def save_named_rgb_image(self, **_: Any) -> str:
        return ""

    def write_camera_frame(self, *_: Any, **__: Any) -> None:
        return None

    def write_observer_camera_frame(self, *_: Any, **__: Any) -> None:
        return None

    def write_detection(self, *_: Any, **__: Any) -> None:
        return None

    def write_estimator_input(self, *_: Any, **__: Any) -> None:
        return None

    def write_dropout_debug_frame(self, *_: Any, **__: Any) -> None:
        return None

    def write_dropout_debug_event(self, *_: Any, **__: Any) -> None:
        return None

    def write_imu_packet(self, *_: Any, **__: Any) -> None:
        return None

    def write_command(self, *_: Any, **__: Any) -> None:
        return None

    def write_controller_diagnostic(self, *_: Any, **__: Any) -> None:
        return None

    def write_realized_state(self, *_: Any, **__: Any) -> None:
        return None

    def write_gt(self, *_: Any, **__: Any) -> None:
        return None

    def write_tag_gt(self, *_: Any, **__: Any) -> None:
        return None

    def write_filter_state(self, *_: Any, **__: Any) -> None:
        return None

    def write_smoother_state(self, *_: Any, **__: Any) -> None:
        return None

    def write_uncertainty(self, *_: Any, **__: Any) -> None:
        return None

    def write_config_snapshot(self, *_: Any, **__: Any) -> None:
        return None

    def write_manifest(self, *_: Any, **__: Any) -> None:
        return None

    def finalize_summary(self, *_: Any, **__: Any) -> None:
        return None


def _actuation_config_path_for_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    mapping = {
        "none": "config/isaac/actuation/none.yaml",
        "servo_nominal": "config/isaac/actuation/servo_nominal.yaml",
        "servo_stress": "config/isaac/actuation/servo_stress.yaml",
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported actuation mode: {mode}") from exc


def _imu_preset_name_for_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    mapping = {
        "ideal": "imu_ideal",
        "nominal_phone": "imu_nominal_phone",
        "stress_phone": "imu_stress_phone",
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported IMU mode: {mode}") from exc


def _load_yaml_payload(path_like: str | Path) -> dict[str, Any]:
    import yaml

    resolved = _repo_path(path_like)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in {resolved}")
    return payload


def _projected_tag_overlay_entry(
    runtime: Any,
    *,
    tag_pose: Any,
    detection: Any | None,
    display_source: str,
) -> dict[str, Any] | None:
    camera_frame_packet = getattr(runtime, "_latest_display_camera_frame_packet", None)
    if camera_frame_packet is None:
        camera_frame_packet = runtime._latest_camera_frame_packet
    if tag_pose is None or camera_frame_packet is None:
        return None
    local_tag_points_m = None
    if detection is not None:
        local_tag_points_m = np.asarray(getattr(detection, "local_tag_points_m", ()), dtype=np.float64).reshape(-1, 3)
    if local_tag_points_m is None or local_tag_points_m.shape != (4, 3):
        local_tag_points_m = np.asarray(
            rendered_apriltag_local_corners_m(float(getattr(tag_pose, "size_m", 0.0))),
            dtype=np.float64,
        )
    projected_corners = _project_tag_corners_for_display(
        camera_frame_packet=camera_frame_packet,
        tag_pose=tag_pose,
        local_tag_points_m=local_tag_points_m,
    )
    if projected_corners is None:
        return None
    is_anchor = bool(getattr(tag_pose, "is_anchor", False))
    measurement_source = "projected_fallback" if detection is None else _detection_measurement_source(detection)
    detector_backend = (
        "anchor_map_projection"
        if is_anchor
        else ("mapped_tag_projection" if detection is None else str(getattr(detection, "detector_backend", "")))
    )
    return {
        "tag_id": int(getattr(tag_pose, "tag_id", getattr(runtime.config, "anchor_tag_id", 0))),
        "is_anchor": is_anchor,
        "corners_xy": tuple((float(x), float(y)) for x, y in projected_corners.tolist()),
        "display_source": str(display_source),
        "measurement_source": str(measurement_source),
        "detector_backend": str(detector_backend),
        "temporal_gap_frames": 0 if detection is None else int(getattr(detection, "temporal_gap_frames", 0)),
        "score": 0.0 if detection is None else float(getattr(detection, "score", 0.0)),
    }


def _projected_overlay_entries(runtime: Any) -> list[dict[str, Any]]:
    tag_pose_map = getattr(runtime, "_tag_pose_map", {}) or {}
    if not tag_pose_map:
        return []
    detections_by_id = {int(item.tag_id): item for item in getattr(runtime, "_last_detections", ())}
    camera_frame_packet = getattr(runtime, "_latest_display_camera_frame_packet", None)
    if camera_frame_packet is None:
        camera_frame_packet = runtime._latest_camera_frame_packet
    current_frame_index = None if camera_frame_packet is None else int(camera_frame_packet.frame_index)
    persistence_gap_frames = _auxiliary_projected_persistence_max_gap_frames(runtime)
    projected_entries: list[dict[str, Any]] = []
    for tag_id, tag_pose in sorted(tag_pose_map.items()):
        detection = detections_by_id.get(int(tag_id))
        if bool(getattr(tag_pose, "is_anchor", False)):
            if detection is not None and _display_source_for_detection(runtime, detection) == "detected_corners":
                continue
            projected = _projected_tag_overlay_entry(
                runtime,
                tag_pose=tag_pose,
                detection=detection,
                display_source="projected_anchor_fallback",
            )
            if projected is not None:
                projected_entries.append(projected)
            continue
        if int(tag_id) not in _mapped_auxiliary_projected_fallback_tag_ids(runtime):
            continue
        if detection is not None:
            if _display_source_for_detection(runtime, detection) != "projected_auxiliary_fallback":
                continue
            projected = _projected_tag_overlay_entry(
                runtime,
                tag_pose=tag_pose,
                detection=detection,
                display_source="projected_auxiliary_fallback",
            )
            if projected is not None:
                projected_entries.append(projected)
            continue
        last_seen_frame_index = getattr(runtime, "_last_detected_frame_by_tag", {}).get(int(tag_id))
        if (
            current_frame_index is None
            or last_seen_frame_index is None
            or int(current_frame_index) - int(last_seen_frame_index) > persistence_gap_frames
        ):
            continue
        projected = _projected_tag_overlay_entry(
            runtime,
            tag_pose=tag_pose,
            detection=None,
            display_source="projected_auxiliary_fallback",
        )
        if projected is not None:
            projected_entries.append(projected)
    return projected_entries


def _overlay_entries(runtime: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for detection in runtime._last_detections:
        display_source = _display_source_for_detection(runtime, detection)
        if str(display_source).startswith("projected_"):
            continue
        entries.append(
            {
                "tag_id": int(detection.tag_id),
                "is_anchor": bool(detection.is_anchor),
                "corners_xy": tuple((float(x), float(y)) for x, y in detection.corners_xy),
                "display_source": display_source,
                "measurement_source": _detection_measurement_source(detection),
                "detector_backend": str(detection.detector_backend),
                "score": float(detection.score),
            }
        )
    entries.extend(_projected_overlay_entries(runtime))
    return entries


def _draw_detection_overlay(image_rgb: np.ndarray | None, overlay_entries: list[dict[str, Any]], *, control_mode: str, noise_modes: dict[str, str]) -> np.ndarray | None:
    if image_rgb is None:
        return None
    canvas = np.asarray(image_rgb, dtype=np.uint8).copy()
    for entry in overlay_entries:
        corners = np.asarray(entry.get("corners_xy", ()), dtype=np.float32).reshape(-1, 2)
        if corners.shape[0] >= 4:
            display_source = str(entry.get("display_source", ""))
            projected_fallback = display_source.startswith("projected_")
            is_anchor = bool(entry.get("is_anchor", False))
            if projected_fallback:
                color = (76, 216, 136) if is_anchor else (110, 196, 255)
            else:
                color = (32, 232, 160) if is_anchor else (255, 189, 71)
            cv2.polylines(canvas, [corners.astype(np.int32)], isClosed=True, color=color, thickness=1 if projected_fallback else 2)
            origin = tuple(corners[0].astype(np.int32))
            cv2.putText(
                canvas,
                f"id {int(entry.get('tag_id', -1))}{' proj' if projected_fallback else ''}",
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    cv2.putText(
        canvas,
        (
            f"mode: {control_mode} | detector: {noise_modes.get('detector_mode', 'new_pupil')} "
            f"| imu: {noise_modes['imu_mode']} | vision: {noise_modes['vision_mode']} | actuation: {noise_modes['actuation_mode']}"
        ),
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (232, 242, 248),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _format_pose_text(translation_m: tuple[float, float, float] | None) -> str:
    if translation_m is None:
        return "n/a"
    values = np.asarray(translation_m, dtype=np.float64).reshape(3)
    return f"[{values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f}] m"


def _build_pattern_locations(runtime: Any) -> list[dict[str, Any]]:
    aux_summary = runtime._latest_auxiliary_update_summary
    aux_decisions = {}
    if aux_summary is not None:
        try:
            aux_decisions = {
                int(item["tag_id"]): dict(item)
                for item in aux_summary.as_json().get("decisions", [])
            }
        except Exception:
            aux_decisions = {}
    anchor_result = runtime._latest_anchor_update_result or {}
    items: list[dict[str, Any]] = []
    anchor_item_present = False
    projected_entries = {int(item["tag_id"]): dict(item) for item in _projected_overlay_entries(runtime)}
    for detection in runtime._last_detections:
        display_source = _display_source_for_detection(runtime, detection)
        if int(detection.tag_id) in projected_entries and str(display_source).startswith("projected_"):
            continue
        decision = aux_decisions.get(int(detection.tag_id), {})
        mapped_pose = runtime._tag_pose_map.get(int(detection.tag_id))
        world_pose_text = None
        if mapped_pose is not None:
            world_pose_text = _format_pose_text(mapped_pose.position_world_m)
        decision_reason = decision.get("reason")
        if bool(detection.is_anchor):
            anchor_item_present = True
            decision_reason = anchor_result.get("reason", decision_reason or "anchor_visible")
            if display_source == "projected_anchor_fallback":
                decision_reason = "projected_anchor_fallback"
        items.append(
            {
                "tag_id": int(detection.tag_id),
                "role": "anchor" if bool(detection.is_anchor) else "auxiliary",
                "score": float(detection.score),
                "pose_ready": bool(detection.pose_camera_tvec_m is not None),
                "camera_pose_text": _format_pose_text(detection.pose_camera_tvec_m),
                "world_pose": {"text": world_pose_text} if world_pose_text is not None else None,
                "mapped_pose": {"text": world_pose_text} if world_pose_text is not None else None,
                "decision_reason": str(decision_reason or ("accepted" if decision.get("accepted") else "visible")),
                "detector_backend": str(detection.detector_backend),
                "measurement_source": _detection_measurement_source(detection),
                "temporal_gap_frames": int(getattr(detection, "temporal_gap_frames", 0)),
                "display_source": display_source,
            }
        )
    for projected_entry in projected_entries.values():
        if bool(projected_entry.get("is_anchor")) and anchor_item_present:
            continue
        mapped_pose = runtime._tag_pose_map.get(int(projected_entry["tag_id"]))
        world_pose_text = None if mapped_pose is None else _format_pose_text(mapped_pose.position_world_m)
        items.append(
            {
                "tag_id": int(projected_entry["tag_id"]),
                "role": "anchor" if bool(projected_entry.get("is_anchor", False)) else "auxiliary",
                "score": float(projected_entry.get("score", 0.0)),
                "pose_ready": False,
                "camera_pose_text": "projected from map",
                "world_pose": {"text": world_pose_text} if world_pose_text is not None else None,
                "mapped_pose": {"text": world_pose_text} if world_pose_text is not None else None,
                "decision_reason": str(projected_entry["display_source"]),
                "detector_backend": str(projected_entry["detector_backend"]),
                "measurement_source": str(projected_entry.get("measurement_source", "projected_fallback")),
                "temporal_gap_frames": int(projected_entry.get("temporal_gap_frames", 0)),
                "display_source": str(projected_entry["display_source"]),
            }
        )
    return items


def _build_arm_joint_payload(runtime: Any) -> list[dict[str, Any]]:
    if runtime._robot_binding is None:
        return []
    joint_names = tuple(runtime._robot_binding.robot.joint_names)
    arm_joint_count = max(min(len(joint_names) - 2, len(joint_names)), 0)
    joint_positions = runtime._robot_binding.get_joint_positions()
    joint_limits = runtime._robot_binding.joint_limits_rad()
    lower = upper = None
    if joint_limits is not None:
        lower, upper = joint_limits
    if runtime._manual_joint_target_positions is not None:
        joint_targets = runtime._manual_joint_target_positions
    elif runtime._latest_command_packet is not None:
        joint_targets = np.asarray(runtime._latest_command_packet.desired_positions, dtype=np.float64).reshape(-1)
    else:
        joint_targets = joint_positions
    items: list[dict[str, Any]] = []
    for index in range(arm_joint_count):
        limit_min = -180.0 if lower is None or index >= len(lower) else float(np.degrees(lower[index]))
        limit_max = 180.0 if upper is None or index >= len(upper) else float(np.degrees(upper[index]))
        items.append(
            {
                "name": str(joint_names[index]),
                "label": f"Joint {index + 1}",
                "position_deg": float(np.degrees(joint_positions[index])),
                "target_deg": float(np.degrees(joint_targets[index])),
                "limit_min_deg": limit_min,
                "limit_max_deg": limit_max,
            }
        )
    return items


def _build_snapshot(
    runtime: Any,
    profile: IsaacDemoProfile,
    noise_modes: dict[str, str],
    recording_state: dict[str, Any],
    analysis_state: dict[str, Any],
    session_state: str,
    *,
    image_cache: dict[str, dict[str, Any]],
    session_error: str | None = None,
) -> dict[str, Any]:
    overlay_entries = _overlay_entries(runtime)
    primary_overlay = _draw_detection_overlay(
        runtime._latest_camera_image_rgb,
        overlay_entries,
        control_mode=str(runtime._interactive_control_mode or profile.default_control_mode),
        noise_modes=noise_modes,
    )
    observer_views = []
    for item in sorted(runtime._latest_observer_views.values(), key=lambda entry: str(entry.get("name", ""))):
        observer_name = str(item.get("name", "observer"))
        observer_frame_index = item.get("frame_index")
        observer_views.append(
            {
                "name": observer_name,
                "frame_index": observer_frame_index,
                "timestamp_s": item.get("timestamp_s"),
                "image_data_url": _cached_frame_data_url(
                    cache=image_cache,
                    stream_key=f"observer:{observer_name}",
                    frame_index=observer_frame_index,
                    image_rgb=item.get("image_rgb"),
                    max_width_px=OBSERVER_STREAM_MAX_WIDTH_PX,
                    jpeg_quality=OBSERVER_STREAM_JPEG_QUALITY,
                ),
                "source": "observer camera",
            }
        )
    measured_imu = runtime._latest_imu_packet
    truth = runtime._latest_imu_truth or {}
    estimator_mode = runtime.config.estimator_mode
    innovation_norm = None if runtime._filter is None else float(runtime._filter.last_innovation_norm)
    estimator_summary = "Waiting for filter state."
    if runtime._latest_filter_snapshot is not None:
        position = np.asarray(runtime._latest_filter_snapshot.position_world_m, dtype=np.float64).reshape(3)
        velocity = np.asarray(runtime._latest_filter_snapshot.velocity_world_mps, dtype=np.float64).reshape(3)
        estimator_summary = f"p=[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] m | v=[{velocity[0]:.3f}, {velocity[1]:.3f}, {velocity[2]:.3f}] m/s"
    snapshot = empty_snapshot(profile_key=profile.key)
    snapshot["session"] = {
        "state": session_state,
        "profile_key": profile.key,
        "profile_label": profile.label,
        "last_error": session_error,
    }
    snapshot["frames"] = {
        "primary": {
            "image_data_url": _cached_frame_data_url(
                cache=image_cache,
                stream_key="primary",
                frame_index=None if runtime._latest_camera_frame_packet is None else int(runtime._latest_camera_frame_packet.frame_index),
                image_rgb=primary_overlay,
                max_width_px=PRIMARY_STREAM_MAX_WIDTH_PX,
                jpeg_quality=PRIMARY_STREAM_JPEG_QUALITY,
            ),
            "frame_index": None if runtime._latest_camera_frame_packet is None else int(runtime._latest_camera_frame_packet.frame_index),
            "timestamp_s": None if runtime._latest_camera_frame_packet is None else float(runtime._latest_camera_frame_packet.timestamp_s),
        },
        "observers": observer_views,
    }
    snapshot["robot"] = {"arm_joints": _build_arm_joint_payload(runtime)}
    snapshot["control"] = {
        "control_mode": str(runtime._interactive_control_mode or profile.default_control_mode),
        "tracking_error_norm_m": None if runtime._latest_controller_diagnostic is None else float(runtime._latest_controller_diagnostic.tracking_error_norm_m),
        "safety_reason": None if runtime._latest_controller_diagnostic is None else str(runtime._latest_controller_diagnostic.safety_reason),
        "desired_position_world_m": None
        if runtime._latest_controller_diagnostic is None
        else list(float(value) for value in runtime._latest_controller_diagnostic.desired_position_world_m),
    }
    snapshot["estimation"] = {
        "estimator_mode": str(estimator_mode),
        "anchor_visible": bool(runtime._last_anchor_visible),
        "position_radius_95_m": None if runtime._latest_uncertainty is None else float(runtime._latest_uncertainty.position_radius_95_m),
        "innovation_norm": innovation_norm,
        "summary": estimator_summary,
    }
    snapshot["detections"] = {
        "count": int(len(runtime._last_detections)),
        "pattern_locations": _build_pattern_locations(runtime),
    }
    snapshot["imu"] = {
        "measured": {
            "accel_mps2": [0.0, 0.0, 0.0]
            if measured_imu is None
            else [float(measured_imu.ax), float(measured_imu.ay), float(measured_imu.az)],
            "gyro_rps": [0.0, 0.0, 0.0]
            if measured_imu is None
            else [float(measured_imu.wx), float(measured_imu.wy), float(measured_imu.wz)],
        },
        "truth": {
            "accel_mps2": [float(value) for value in np.asarray(truth.get("accel_mps2", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)],
            "gyro_rps": [float(value) for value in np.asarray(truth.get("gyro_rps", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)],
        },
    }
    snapshot["noise"] = dict(noise_modes)
    snapshot["recording"] = deepcopy(recording_state)
    snapshot["analysis"] = deepcopy(analysis_state)
    return snapshot


def _resolve_session_state(runtime: Any, requested_state: str) -> str:
    normalized = str(requested_state or "").strip().lower()
    if normalized == "error":
        return "error"
    primary_ready = any(
        candidate is not None
        for candidate in (
            getattr(runtime, "_latest_camera_image_rgb", None),
            getattr(runtime, "_latest_display_camera_image_rgb", None),
            getattr(runtime, "_latest_camera_frame_packet", None),
            getattr(runtime, "_latest_display_camera_frame_packet", None),
        )
    )
    observer_bindings = getattr(runtime, "_observer_camera_bindings", [])
    observer_ready = len(observer_bindings) == 0 or len(runtime._latest_observer_views) >= len(observer_bindings)
    joints_ready = runtime._latest_realized_joint_packet is not None
    if primary_ready and observer_ready and joints_ready:
        return "live"
    return "starting"


def _publish_snapshot(snapshot_queue: mp.Queue, snapshot: dict[str, Any]) -> None:
    try:
        while True:
            snapshot_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        snapshot_queue.put_nowait(snapshot)
    except queue.Full:
        pass


def _build_bootstrap(
    profile: IsaacDemoProfile,
    *,
    run_dir: Path,
    session_id: str,
    headless: bool,
    width: int,
    height: int,
    streaming_backend: str = "webrtc",
    webrtc_signal_port: int = 49110,
    webrtc_stream_port: int = 47990,
    webrtc_target_fps: int = 60,
):
    from calib_sim.isaac.app import IsaacAppBootstrapConfig

    return IsaacAppBootstrapConfig(
        scene_config_path=profile.scene_config_path,
        robot_config_path=profile.robot_config_path,
        camera_config_path=profile.camera_config_path,
        imu_config_path=profile.imu_config_path,
        actuation_config_path=profile.actuation_config_path,
        estimation_config_path=profile.estimation_config_path,
        control_config_path=profile.control_config_path,
        run_id=f"isaac_demo_session_{session_id}",
        run_dir=str(run_dir),
        headless=bool(headless),
        width=int(width),
        height=int(height),
        streaming_backend=str(streaming_backend),
        webrtc_signal_port=int(webrtc_signal_port),
        webrtc_stream_port=int(webrtc_stream_port),
        webrtc_target_fps=int(webrtc_target_fps),
        duration_s=3600.0,
        estimator_mode=profile.default_estimator_mode,
        controller_mode=profile.default_runtime_controller_mode,
        bootstrap_control_policy=profile.bootstrap_control_policy,
        promote_latest_complete=False,
    )


def _initialize_recording(runtime: Any, profile: IsaacDemoProfile, recording_state: dict[str, Any], noise_modes: dict[str, str]) -> None:
    from calib_sim.isaac.logging.run_manifest import build_run_manifest
    from calib_sim.isaac.logging.writer import IsaacRunWriter

    run_id = datetime.now(timezone.utc).strftime(f"isaac_demo_{profile.key}_%Y%m%d_%H%M%S")
    run_dir = _repo_path(Path("output") / "isaac_demo_runs" / profile.key / run_id)
    writer = IsaacRunWriter(run_dir)
    runtime.config.run_id = run_id
    runtime.config.run_dir = str(run_dir)
    runtime.set_writer(writer, reset_recording_baseline=True)
    for name, payload in runtime.config.config_payloads.items():
        writer.write_config_snapshot(name, payload)
    manifest = build_run_manifest(
        repo_root=REPO_ROOT,
        run_id=run_id,
        isaac_sim_version="interactive_demo",
        stage_usd_path=runtime.config.stage_path if Path(runtime.config.stage_path).exists() else f"programmatic:{profile.key}",
        robot_preset=runtime.config.robot_preset,
        anchor_tag_id=runtime.config.anchor_tag_id,
        estimator_mode=runtime.config.estimator_mode,
        controller_mode=runtime.config.controller_mode,
        bootstrap_control_policy=runtime.config.bootstrap_control_policy,
        noise_presets={
            "imu": noise_modes["imu_mode"],
            "vision": noise_modes["vision_mode"],
            "actuation": noise_modes["actuation_mode"],
        },
        random_seed=int(runtime.config.seed),
        controller_config=copy.deepcopy(runtime.config.config_payloads["control"]),
        estimator_config=copy.deepcopy(runtime.config.config_payloads["estimation"]),
        ros2_bridge_used=False,
    )
    writer.write_manifest(manifest)
    runtime._write_tag_ground_truth()
    recording_state.update(
        {
            "enabled": True,
            "active_run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "last_run_dir": recording_state.get("last_run_dir"),
            "output_root": "output/isaac_demo_runs",
        }
    )


def _stop_recording(runtime: Any, recording_state: dict[str, Any], null_writer: _NullIsaacRunWriter) -> None:
    if not recording_state.get("enabled") or not recording_state.get("active_run_dir"):
        return
    summary = runtime.summary()
    runtime.persist_summary(summary)
    last_run_dir = str(recording_state["active_run_dir"])
    recording_state.update(
        {
            "enabled": False,
            "active_run_dir": None,
            "last_run_dir": last_run_dir,
            "output_root": "output/isaac_demo_runs",
        }
    )
    runtime.set_writer(null_writer, reset_recording_baseline=True)


def _run_analysis_on_last_run(recording_state: dict[str, Any], analysis_state: dict[str, Any]) -> None:
    last_run_dir = recording_state.get("last_run_dir")
    if not last_run_dir:
        raise FileNotFoundError("No recorded Isaac demo run is available yet.")
    started_at = _iso_now()
    analysis_state.update(
        {
            "state": "running",
            "last_started_at_utc": started_at,
            "last_completed_at_utc": None,
            "last_run_dir": str(Path(last_run_dir)),
            "summary": None,
            "report_path": None,
            "error": None,
        }
    )
    analysis_state.update(
        {
            "state": "completed",
            "last_completed_at_utc": _iso_now(),
            "summary": "Offline report generation was removed from the slim tabletop branch.",
            "report_path": None,
            "error": None,
        }
    )


def demo_worker_main(
    *,
    command_queue: mp.Queue,
    snapshot_queue: mp.Queue,
    stop_event: mp.Event,
    profile_key: str,
    headless: bool,
    width: int,
    height: int,
    session_id: str,
) -> None:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    profile = get_demo_profile(profile_key)
    noise_modes = {
        "imu_mode": "ideal",
        "vision_mode": "clean",
        "actuation_mode": "none",
        "detector_mode": _default_detector_mode_for_profile(profile),
    }
    recording_state = {"enabled": False, "active_run_dir": None, "last_run_dir": None, "output_root": "output/isaac_demo_runs"}
    analysis_state = {
        "state": "idle",
        "last_started_at_utc": None,
        "last_completed_at_utc": None,
        "last_run_dir": None,
        "summary": None,
        "report_path": None,
        "error": None,
    }
    null_writer = _NullIsaacRunWriter()
    image_cache: dict[str, dict[str, Any]] = {}
    runtime = None
    try:
        from calib_sim.isaac.app import create_runtime

        browsing_run_dir = _repo_path(Path("/tmp") / "calib_sim_isaac_demo" / f"session_{session_id}")
        browsing_run_dir.mkdir(parents=True, exist_ok=True)
        bootstrap = _build_bootstrap(profile, run_dir=browsing_run_dir, session_id=session_id, headless=headless, width=width, height=height)
        runtime = create_runtime(bootstrap)
        runtime.set_writer(null_writer, reset_recording_baseline=True)
        runtime.start()
        runtime.set_interactive_control_mode(profile.default_control_mode)
        runtime.set_imu_noise_preset(_imu_preset_name_for_mode(noise_modes["imu_mode"]))
        runtime.set_vision_noise_mode(noise_modes["vision_mode"])
        runtime.set_actuation_config_payload(_load_yaml_payload(_actuation_config_path_for_mode(noise_modes["actuation_mode"])))
        runtime.set_tag_detection_mode(noise_modes["detector_mode"])
        last_publish_at = 0.0
        last_forced_publish_at = 0.0
        last_published_primary_frame_index: int | None = None
        last_published_observer_signature: tuple[tuple[str, int | None], ...] = ()
        last_published_session_state: str | None = None
        last_published_error: str | None = None
        session_state = "starting"
        session_error = None
        while not stop_event.is_set():
            while True:
                try:
                    command = command_queue.get_nowait()
                except queue.Empty:
                    break
                if not isinstance(command, Mapping):
                    continue
                command_type = str(command.get("type", "")).strip().lower()
                if command_type == "stop":
                    stop_event.set()
                    break
                if command_type == "launch_session":
                    requested_mode = command.get("control_mode", profile.default_control_mode)
                    runtime.set_interactive_control_mode(
                        validate_control_mode(str(requested_mode), allow_auto_path=bool(profile.allow_auto_path))
                    )
                    session_state = "starting"
                    session_error = None
                elif command_type == "set_control_mode":
                    runtime.set_interactive_control_mode(
                        validate_control_mode(str(command.get("control_mode", profile.default_control_mode)), allow_auto_path=bool(profile.allow_auto_path))
                    )
                elif command_type == "set_joint_targets":
                    runtime.set_manual_joint_targets_deg(np.asarray(command.get("joint_targets_deg", []), dtype=np.float64))
                elif command_type == "nudge_task_target":
                    runtime.nudge_manual_task_target(np.asarray(command.get("delta_world_m", [0.0, 0.0, 0.0]), dtype=np.float64))
                elif command_type == "set_noise_modes":
                    updates = validate_noise_modes(
                        imu_mode=command.get("imu_mode"),
                        vision_mode=command.get("vision_mode"),
                        actuation_mode=command.get("actuation_mode"),
                        detector_mode=command.get("detector_mode"),
                    )
                    if "imu_mode" in updates:
                        noise_modes["imu_mode"] = updates["imu_mode"]
                        runtime.set_imu_noise_preset(_imu_preset_name_for_mode(noise_modes["imu_mode"]))
                    if "vision_mode" in updates:
                        noise_modes["vision_mode"] = updates["vision_mode"]
                        runtime.set_vision_noise_mode(noise_modes["vision_mode"])
                    if "actuation_mode" in updates:
                        noise_modes["actuation_mode"] = updates["actuation_mode"]
                        runtime.set_actuation_config_payload(_load_yaml_payload(_actuation_config_path_for_mode(noise_modes["actuation_mode"])))
                    if "detector_mode" in updates:
                        noise_modes["detector_mode"] = updates["detector_mode"]
                        runtime.set_tag_detection_mode(noise_modes["detector_mode"])
                elif command_type == "set_recording":
                    if bool(command.get("enabled", False)):
                        if not recording_state["enabled"]:
                            _initialize_recording(runtime, profile, recording_state, noise_modes)
                    else:
                        _stop_recording(runtime, recording_state, null_writer)
                elif command_type == "run_analysis":
                    try:
                        _run_analysis_on_last_run(recording_state, analysis_state)
                    except Exception:
                        pass
                elif command_type == "home":
                    runtime.home()
                elif command_type == "refresh":
                    pass
            if stop_event.is_set():
                break
            runtime.step(1)
            session_state = _resolve_session_state(runtime, session_state)
            now = time.monotonic()
            if now - last_publish_at >= SNAPSHOT_PUBLISH_INTERVAL_S:
                primary_frame_index = (
                    None if runtime._latest_camera_frame_packet is None else int(runtime._latest_camera_frame_packet.frame_index)
                )
                observer_signature = tuple(
                    sorted(
                        (
                            str(item.get("name", "")),
                            None if item.get("frame_index") is None else int(item.get("frame_index")),
                        )
                        for item in runtime._latest_observer_views.values()
                    )
                )
                should_publish = (
                    primary_frame_index != last_published_primary_frame_index
                    or observer_signature != last_published_observer_signature
                    or session_state != last_published_session_state
                    or session_error != last_published_error
                    or now - last_forced_publish_at >= FORCED_SNAPSHOT_PUBLISH_INTERVAL_S
                )
                if should_publish:
                    _publish_snapshot(
                        snapshot_queue,
                        _build_snapshot(
                            runtime,
                            profile,
                            noise_modes,
                            recording_state,
                            analysis_state,
                            session_state,
                            image_cache=image_cache,
                            session_error=session_error,
                        ),
                    )
                    last_published_primary_frame_index = primary_frame_index
                    last_published_observer_signature = observer_signature
                    last_published_session_state = session_state
                    last_published_error = session_error
                    last_forced_publish_at = now
                last_publish_at = now
    except Exception as exc:
        _publish_snapshot(snapshot_queue, snapshot_with_error(str(exc), profile_key=profile.key))
    finally:
        if runtime is not None:
            if recording_state.get("enabled"):
                _stop_recording(runtime, recording_state, null_writer)
            runtime.shutdown()


class IsaacDemoController:
    """Service-side controller for a single live Isaac worker."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._process: mp.Process | None = None
        self._command_queue: mp.Queue | None = None
        self._snapshot_queue: mp.Queue | None = None
        self._stop_event: mp.Event | None = None
        self._reader_thread: threading.Thread | None = None
        self._latest_snapshot = empty_snapshot()
        self._catalog = initial_catalog()
        self._profile_key = DEFAULT_PROFILE_KEY
        self._headless = True
        self._width = 1600
        self._height = 900
        self._session_id = "live"

    def catalog(self) -> dict[str, Any]:
        return {"profiles": list_demo_profiles(), "default_profile_key": DEFAULT_PROFILE_KEY}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._latest_snapshot)

    def ensure_session(self, *, profile_key: str | None = None) -> None:
        with self._lock:
            current_profile = self._profile_key
            if self._process is not None and self._process.is_alive() and (
                profile_key in (None, "") or str(profile_key) == current_profile
            ):
                return
        desired_profile = current_profile if profile_key in (None, "") else str(profile_key)
        self.restart(profile_key=desired_profile)

    def restart(self, *, profile_key: str | None = None) -> None:
        desired_profile = DEFAULT_PROFILE_KEY if profile_key in (None, "") else str(profile_key)
        self.stop()
        process = self._ctx.Process
        command_queue = self._ctx.Queue(maxsize=64)
        snapshot_queue = self._ctx.Queue(maxsize=2)
        stop_event = self._ctx.Event()
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        worker = process(
            target=demo_worker_main,
            kwargs={
                "command_queue": command_queue,
                "snapshot_queue": snapshot_queue,
                "stop_event": stop_event,
                "profile_key": desired_profile,
                "headless": self._headless,
                "width": self._width,
                "height": self._height,
                "session_id": session_id,
            },
            daemon=True,
        )
        worker.start()
        with self._lock:
            self._process = worker
            self._command_queue = command_queue
            self._snapshot_queue = snapshot_queue
            self._stop_event = stop_event
            self._profile_key = desired_profile
            self._session_id = session_id
            self._latest_snapshot = empty_snapshot(profile_key=desired_profile)
            self._latest_snapshot["session"]["state"] = "starting"
            self._latest_snapshot["estimation"]["summary"] = "Booting Isaac worker."
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def stop(self) -> None:
        process = None
        command_queue = None
        stop_event = None
        with self._lock:
            process = self._process
            command_queue = self._command_queue
            stop_event = self._stop_event
            self._process = None
            self._command_queue = None
            self._snapshot_queue = None
            self._stop_event = None
        if command_queue is not None:
            try:
                command_queue.put_nowait({"type": "stop"})
            except Exception:
                pass
        if stop_event is not None:
            stop_event.set()
        if process is not None and process.is_alive():
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

    def send_command(self, payload: Mapping[str, Any]) -> None:
        if not payload:
            return
        command_type = str(payload.get("type", "")).strip().lower()
        if command_type == "switch_profile":
            self.restart(profile_key=str(payload.get("profile_key") or DEFAULT_PROFILE_KEY))
            return
        if command_type == "reset_session":
            self.restart(profile_key=self._profile_key)
            return
        self.ensure_session(profile_key=self._profile_key)
        with self._lock:
            command_queue = self._command_queue
        if command_queue is None:
            return
        try:
            command_queue.put_nowait(dict(payload))
        except Exception:
            pass

    def _reader_loop(self) -> None:
        while True:
            with self._lock:
                process = self._process
                snapshot_queue = self._snapshot_queue
            if process is None or snapshot_queue is None:
                return
            try:
                snapshot = snapshot_queue.get(timeout=0.5)
            except queue.Empty:
                if not process.is_alive():
                    with self._lock:
                        current = deepcopy(self._latest_snapshot)
                        if current["session"]["state"] != "error":
                            self._latest_snapshot = snapshot_with_error(
                                "Isaac worker process exited unexpectedly.",
                                profile_key=self._profile_key,
                            )
                    return
                continue
            if isinstance(snapshot, dict):
                with self._lock:
                    self._latest_snapshot = snapshot


__all__ = ["IsaacDemoController", "demo_worker_main"]
