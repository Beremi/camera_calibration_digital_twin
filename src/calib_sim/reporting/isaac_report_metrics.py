"""Metric extraction for Isaac runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calib_sim.isaac.runtime.replay import load_replay_bundle


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _is_fallback_backend(backend: Any) -> bool:
    lowered = str(backend or "").strip().lower()
    return (
        "fallback" in lowered
        or "bright_quad_match" in lowered
        or "rejected_candidate_match" in lowered
    )


def _load_config_snapshot(run_dir: Path, name: str) -> dict[str, Any]:
    path = run_dir / "config_snapshot" / f"{name}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _timestamp_values(bundle: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for row in bundle.get("camera_frames", []):
        values.append(float(row["timestamp_s"]))
    for row in bundle.get("imu", []):
        timestamp = _float_or_none(row.get("timestamp_s"))
        if timestamp is not None:
            values.append(timestamp)
    for row in bundle.get("commands", []):
        timestamp = _float_or_none(row.get("timestamp_s"))
        if timestamp is not None:
            values.append(timestamp)
    return values


def _vector_or_none(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


def _matrix3_or_none(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    rows: list[list[float]] = []
    for row in value[:3]:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            return None
        rows.append([float(row[0]), float(row[1]), float(row[2])])
    return np.asarray(rows, dtype=np.float64)


def _csv_vector_or_none(row: dict[str, Any], x_key: str, y_key: str, z_key: str) -> np.ndarray | None:
    if x_key not in row or y_key not in row or z_key not in row:
        return None
    x = _float_or_none(row.get(x_key))
    y = _float_or_none(row.get(y_key))
    z = _float_or_none(row.get(z_key))
    if x is None or y is None or z is None:
        return None
    return np.asarray([x, y, z], dtype=np.float64)


def _pipe_vector_or_none(value: Any, *, expected_len: int = 3) -> np.ndarray | None:
    if value in ("", None):
        return None
    parts = str(value).split("|")
    if len(parts) < expected_len:
        return None
    usable = []
    for part in parts[:expected_len]:
        numeric = _float_or_none(part)
        if numeric is None:
            return None
        usable.append(float(numeric))
    return np.asarray(usable, dtype=np.float64)


def _position_from_gt_row(row: dict[str, Any]) -> np.ndarray | None:
    position = _csv_vector_or_none(row, "px", "py", "pz")
    if position is not None:
        return position
    return _csv_vector_or_none(row, "x", "y", "z")


def _position_from_tag_gt(row: dict[str, Any]) -> np.ndarray | None:
    position = _vector_or_none(row.get("position_world_m"))
    if position is not None:
        return position
    position = _vector_or_none(row.get("translation_world_m"))
    if position is not None:
        return position
    pose = row.get("pose_world")
    if isinstance(pose, list) and len(pose) >= 4 and isinstance(pose[0], list) and len(pose[0]) >= 4:
        return np.asarray([float(pose[0][3]), float(pose[1][3]), float(pose[2][3])], dtype=np.float64)
    return None


def _rotation_from_tag_gt(row: dict[str, Any]) -> np.ndarray | None:
    rotation = _matrix3_or_none(row.get("rotation_wt"))
    if rotation is not None:
        return rotation
    pose = row.get("pose_world")
    if isinstance(pose, list) and len(pose) >= 3:
        return _matrix3_or_none([pose[0][:3], pose[1][:3], pose[2][:3]])
    return None


def _tag_map_error(gt_payload: dict[str, Any] | None, smoother_rows: list[dict[str, Any]]) -> float | None:
    if not gt_payload:
        return None
    gt_tags = {
        int(tag["tag_id"]): _position_from_tag_gt(tag)
        for tag in gt_payload.get("tags", [])
        if isinstance(tag, dict) and "tag_id" in tag
    }
    errors: list[float] = []
    for row in smoother_rows:
        active_tag_poses = row.get("active_tag_poses", {})
        if not isinstance(active_tag_poses, dict):
            continue
        for tag_id_text, pose in active_tag_poses.items():
            gt_position = gt_tags.get(int(tag_id_text))
            if gt_position is None or not isinstance(pose, list) or len(pose) < 4:
                continue
            estimate = np.asarray([float(pose[0][3]), float(pose[1][3]), float(pose[2][3])], dtype=np.float64)
            errors.append(float(np.linalg.norm(estimate - gt_position)))
    return None if not errors else float(np.mean(errors))


def _auxiliary_tag_position_error_stats(
    gt_payload: dict[str, Any] | None,
    smoother_rows: list[dict[str, Any]],
    *,
    anchor_tag_id: int,
) -> dict[str, float | None]:
    if not gt_payload:
        return {
            "mean_auxiliary_tag_position_error_m": None,
            "p95_auxiliary_tag_position_error_m": None,
            "auxiliary_tag_count": 0.0,
        }
    gt_tags = {
        int(tag["tag_id"]): _position_from_tag_gt(tag)
        for tag in gt_payload.get("tags", [])
        if isinstance(tag, dict) and "tag_id" in tag and int(tag["tag_id"]) != int(anchor_tag_id)
    }
    errors: list[float] = []
    observed_auxiliary_tag_ids: set[int] = set()
    for row in smoother_rows:
        active_tag_poses = row.get("active_tag_poses", {})
        if not isinstance(active_tag_poses, dict):
            continue
        for tag_id_text, pose in active_tag_poses.items():
            tag_id = int(tag_id_text)
            gt_position = gt_tags.get(tag_id)
            if gt_position is None or not isinstance(pose, list) or len(pose) < 4:
                continue
            estimate = np.asarray([float(pose[0][3]), float(pose[1][3]), float(pose[2][3])], dtype=np.float64)
            errors.append(float(np.linalg.norm(estimate - gt_position)))
            observed_auxiliary_tag_ids.add(tag_id)
    return {
        "mean_auxiliary_tag_position_error_m": None if not errors else float(np.mean(errors)),
        "p95_auxiliary_tag_position_error_m": _safe_percentile(errors, 95.0),
        "auxiliary_tag_count": float(len(observed_auxiliary_tag_ids)),
    }


def _reprojection_error_stats(
    camera_frames: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
    gt_payload: dict[str, Any] | None,
) -> dict[str, float | None]:
    if not camera_frames or not detections or not filter_rows or not gt_payload:
        return {
            "mean_reprojection_rmse_px": None,
            "p95_reprojection_rmse_px": None,
            "reprojection_sample_count": 0.0,
        }
    gt_tag_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for tag in gt_payload.get("tags", []):
        if not isinstance(tag, dict) or "tag_id" not in tag:
            continue
        position = _position_from_tag_gt(tag)
        rotation = _rotation_from_tag_gt(tag)
        if position is None or rotation is None:
            continue
        gt_tag_poses[int(tag["tag_id"])] = (position, rotation)
    frame_by_index = {
        int(row["frame_index"]): row for row in camera_frames if "frame_index" in row and "intrinsics_snapshot" in row
    }
    filter_samples: list[tuple[float, np.ndarray, np.ndarray]] = []
    for row in filter_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _vector_or_none(row.get("position_world_m"))
        rotation = _matrix3_or_none(row.get("rotation_wi"))
        if timestamp is None or position is None or rotation is None:
            continue
        filter_samples.append((timestamp, position, rotation))
    if not filter_samples:
        return {
            "mean_reprojection_rmse_px": None,
            "p95_reprojection_rmse_px": None,
            "reprojection_sample_count": 0.0,
        }
    filter_times = np.asarray([sample[0] for sample in filter_samples], dtype=np.float64)
    rmse_values_px: list[float] = []
    for detection in detections:
        frame = frame_by_index.get(int(detection.get("frame_index", -1)))
        tag_pose = gt_tag_poses.get(int(detection.get("tag_id", -1)))
        timestamp = _float_or_none(detection.get("timestamp_s"))
        local_tag_points_m = detection.get("local_tag_points_m")
        corners_xy = detection.get("corners_xy")
        if frame is None or tag_pose is None or timestamp is None:
            continue
        local_points = np.asarray(local_tag_points_m, dtype=np.float64)
        observed_corners = np.asarray(corners_xy, dtype=np.float64)
        if local_points.shape != (4, 3) or observed_corners.shape != (4, 2):
            continue
        intrinsics = frame.get("intrinsics_snapshot", {})
        fx = _float_or_none(intrinsics.get("fx_px"))
        fy = _float_or_none(intrinsics.get("fy_px"))
        cx = _float_or_none(intrinsics.get("cx_px"))
        cy = _float_or_none(intrinsics.get("cy_px"))
        if fx is None or fy is None or cx is None or cy is None:
            continue
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        distortion = intrinsics.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0])
        dist_coeffs = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
        filter_index = int(np.argmin(np.abs(filter_times - timestamp)))
        _, camera_position_world_m, rotation_wc = filter_samples[filter_index]
        # `rotation_wi` in the first-pass filter logs is the world-from-camera
        # rotation in the OpenCV camera frame used by solvePnP detections.
        rotation_cw = rotation_wc.T
        translation_cw = -rotation_cw @ camera_position_world_m.reshape(3)
        rvec_cw, _ = cv2.Rodrigues(rotation_cw)
        tag_position_world_m, rotation_wt = tag_pose
        world_points = (rotation_wt @ local_points.T).T + tag_position_world_m.reshape(1, 3)
        projected_points, _ = cv2.projectPoints(
            world_points,
            rvec_cw,
            translation_cw.reshape(3, 1),
            camera_matrix,
            dist_coeffs,
        )
        projected_corners = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
        rmse_values_px.append(float(np.sqrt(np.mean(np.sum((projected_corners - observed_corners) ** 2, axis=1)))))
    return {
        "mean_reprojection_rmse_px": None if not rmse_values_px else float(np.mean(rmse_values_px)),
        "p95_reprojection_rmse_px": _safe_percentile(rmse_values_px, 95.0),
        "reprojection_sample_count": float(len(rmse_values_px)),
    }


def _nearest_position_matches(
    filter_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    uncertainty_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gt_samples: list[tuple[float, np.ndarray]] = []
    for row in gt_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _position_from_gt_row(row)
        if timestamp is not None and position is not None:
            gt_samples.append((timestamp, position))
    if not gt_samples:
        return []

    gt_times = np.asarray([item[0] for item in gt_samples], dtype=np.float64)
    radii_by_timestamp = {
        float(row["timestamp_s"]): float(row["position_radius_95_m"])
        for row in uncertainty_rows
        if "timestamp_s" in row and "position_radius_95_m" in row
    }
    matches: list[dict[str, Any]] = []
    for row in filter_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _vector_or_none(row.get("position_world_m"))
        if timestamp is None or position is None:
            continue
        gt_index = int(np.argmin(np.abs(gt_times - timestamp)))
        gt_timestamp, gt_position = gt_samples[gt_index]
        covariance = np.asarray(row.get("covariance", []), dtype=np.float64)
        error_vector = position - gt_position
        position_nees: float | None = None
        if covariance.shape[0] >= 6 and covariance.shape[1] >= 6:
            position_cov = covariance[3:6, 3:6]
            try:
                position_nees = float(error_vector.T @ np.linalg.solve(position_cov, error_vector))
            except np.linalg.LinAlgError:
                position_nees = None
        matches.append(
            {
                "timestamp_s": float(timestamp),
                "gt_timestamp_s": float(gt_timestamp),
                "position_error_m": float(np.linalg.norm(error_vector)),
                "position_radius_95_m": radii_by_timestamp.get(float(timestamp)),
                "position_nees": position_nees,
            }
        )
    return matches


def _waypoint_summary(control_config: dict[str, Any], filter_rows: list[dict[str, Any]], start_time_s: float | None) -> dict[str, float | None]:
    positions: list[tuple[float, np.ndarray]] = []
    for row in filter_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _vector_or_none(row.get("position_world_m"))
        if timestamp is not None and position is not None:
            positions.append((timestamp, position))
    waypoints = control_config.get("waypoints", []) if isinstance(control_config, dict) else []
    if not positions or not isinstance(waypoints, list) or not waypoints:
        return {
            "mean_waypoint_error_m": None,
            "p95_waypoint_error_m": None,
            "completion_fraction": None,
            "mean_completion_time_s": None,
            "failure_rate_percent": None,
        }

    waypoint_errors: list[float] = []
    completion_times: list[float] = []
    for waypoint in waypoints:
        if not isinstance(waypoint, dict):
            continue
        target = _vector_or_none(waypoint.get("position_world_m"))
        tolerance = _float_or_none(waypoint.get("tolerance_m"))
        if target is None:
            continue
        distances = [float(np.linalg.norm(position - target)) for _, position in positions]
        min_index = int(np.argmin(distances))
        min_distance = distances[min_index]
        waypoint_errors.append(min_distance)
        if tolerance is not None and min_distance <= tolerance:
            reached_timestamp = positions[min_index][0]
            if start_time_s is None:
                completion_times.append(float(reached_timestamp))
            else:
                completion_times.append(float(reached_timestamp - start_time_s))
    completion_fraction = 0.0 if not waypoint_errors else float(len(completion_times) / len(waypoint_errors))
    return {
        "mean_waypoint_error_m": None if not waypoint_errors else float(np.mean(waypoint_errors)),
        "p95_waypoint_error_m": None if not waypoint_errors else float(np.percentile(waypoint_errors, 95.0)),
        "completion_fraction": completion_fraction,
        "mean_completion_time_s": None if not completion_times else float(np.mean(completion_times)),
        "failure_rate_percent": float((1.0 - completion_fraction) * 100.0),
    }


def _ground_truth_waypoint_summary(
    control_config: dict[str, Any],
    camera_gt_rows: list[dict[str, Any]],
    start_time_s: float | None,
) -> dict[str, float | None]:
    positions: list[tuple[float, np.ndarray]] = []
    for row in camera_gt_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _position_from_gt_row(row)
        if timestamp is not None and position is not None:
            positions.append((timestamp, position))
    if not positions:
        return _waypoint_summary(control_config, [], start_time_s)
    return _waypoint_summary(
        control_config,
        [{"timestamp_s": timestamp, "position_world_m": position.tolist()} for timestamp, position in positions],
        start_time_s,
    )


def _bool_from_csv(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _median_dt(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    deltas = [max(0.0, float(right) - float(left)) for left, right in zip(times[:-1], times[1:])]
    usable = [delta for delta in deltas if delta > 0.0]
    return 0.0 if not usable else float(np.median(np.asarray(usable, dtype=np.float64)))


def _series_within_window(
    samples: list[tuple[float, np.ndarray]],
    *,
    start_s: float,
    end_s: float,
) -> list[tuple[float, np.ndarray]]:
    return [
        (timestamp, value)
        for timestamp, value in samples
        if float(start_s) <= float(timestamp) <= float(end_s)
    ]


def _controller_waypoint_windows(
    controller_rows: list[dict[str, Any]],
    *,
    waypoint_count: int,
    default_end_time_s: float | None,
) -> list[dict[str, float | int | None]]:
    if waypoint_count <= 0 or not controller_rows:
        return []
    sorted_rows = sorted(
        controller_rows,
        key=lambda row: _float_or_none(row.get("timestamp_s")) if _float_or_none(row.get("timestamp_s")) is not None else 1e18,
    )
    rows_by_waypoint: dict[int, list[float]] = {index: [] for index in range(waypoint_count)}
    for row in sorted_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        waypoint_index = int(row.get("waypoint_index", -1) or -1)
        if timestamp is None or waypoint_index < 0 or waypoint_index >= waypoint_count:
            continue
        rows_by_waypoint.setdefault(waypoint_index, []).append(float(timestamp))
    last_timestamp = default_end_time_s
    if last_timestamp is None:
        timestamps = [_float_or_none(row.get("timestamp_s")) for row in sorted_rows]
        usable = [float(value) for value in timestamps if value is not None]
        last_timestamp = None if not usable else float(max(usable))
    windows: list[dict[str, float | int | None]] = []
    for waypoint_index in range(waypoint_count):
        timestamps = rows_by_waypoint.get(waypoint_index, [])
        if not timestamps:
            windows.append({"waypoint_index": waypoint_index, "start_s": None, "end_s": None})
            continue
        start_s = float(min(timestamps))
        next_start_candidates = [
            float(min(candidate_times))
            for next_index, candidate_times in rows_by_waypoint.items()
            if next_index > waypoint_index and candidate_times
        ]
        end_s = float(min(next_start_candidates)) if next_start_candidates else last_timestamp
        windows.append({"waypoint_index": waypoint_index, "start_s": start_s, "end_s": end_s})
    return windows


def _window_first_hit_time_s(
    samples: list[tuple[float, np.ndarray]],
    target: np.ndarray,
    *,
    threshold_m: float,
) -> float | None:
    for timestamp, position in samples:
        if float(np.linalg.norm(position - target)) <= float(threshold_m):
            return float(timestamp)
    return None


def _window_dwell_time_s(
    samples: list[tuple[float, np.ndarray]],
    target: np.ndarray,
    *,
    threshold_m: float,
) -> float:
    if len(samples) < 2:
        return 0.0
    dwell = 0.0
    for (left_t, left_position), (right_t, _) in zip(samples[:-1], samples[1:]):
        if float(np.linalg.norm(left_position - target)) <= float(threshold_m):
            dwell += max(0.0, float(right_t) - float(left_t))
    return float(dwell)


def _controller_desired_series(controller_rows: list[dict[str, Any]]) -> list[tuple[float, np.ndarray, bool]]:
    samples: list[tuple[float, np.ndarray, bool]] = []
    for row in controller_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        desired_position = _pipe_vector_or_none(row.get("desired_position_world_m"))
        if timestamp is None or desired_position is None:
            continue
        samples.append((float(timestamp), desired_position, _bool_from_csv(row.get("dropped_command"))))
    return samples


def _realized_ee_series(realized_rows: list[dict[str, Any]]) -> list[tuple[float, np.ndarray]]:
    samples: list[tuple[float, np.ndarray]] = []
    for row in realized_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _pipe_vector_or_none(row.get("end_effector_position_world_m"))
        if timestamp is None or position is None:
            continue
        samples.append((float(timestamp), position))
    return samples


def _commanded_realized_path_deviation(
    controller_rows: list[dict[str, Any]],
    realized_rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    desired_samples = _controller_desired_series(controller_rows)
    realized_samples = _realized_ee_series(realized_rows)
    if not desired_samples or not realized_samples:
        return {
            "mean_commanded_realized_path_deviation_m": None,
            "p95_commanded_realized_path_deviation_m": None,
            "time_integrated_commanded_realized_path_deviation_m_s": None,
        }
    realized_times = np.asarray([timestamp for timestamp, _ in realized_samples], dtype=np.float64)
    deviations: list[float] = []
    sample_times: list[float] = []
    for timestamp, desired_position, _ in desired_samples:
        realized_index = int(np.argmin(np.abs(realized_times - float(timestamp))))
        realized_position = realized_samples[realized_index][1]
        deviations.append(float(np.linalg.norm(desired_position - realized_position)))
        sample_times.append(float(timestamp))
    if not deviations:
        return {
            "mean_commanded_realized_path_deviation_m": None,
            "p95_commanded_realized_path_deviation_m": None,
            "time_integrated_commanded_realized_path_deviation_m_s": None,
        }
    integral = 0.0
    if len(deviations) >= 2:
        for left_time, right_time, left_deviation in zip(sample_times[:-1], sample_times[1:], deviations[:-1]):
            integral += float(left_deviation) * max(0.0, float(right_time) - float(left_time))
    return {
        "mean_commanded_realized_path_deviation_m": float(np.mean(np.asarray(deviations, dtype=np.float64))),
        "p95_commanded_realized_path_deviation_m": _safe_percentile(deviations, 95.0),
        "time_integrated_commanded_realized_path_deviation_m_s": float(integral),
    }


def _dropped_command_stats(controller_rows: list[dict[str, Any]]) -> dict[str, float]:
    timestamps = [
        float(value)
        for value in (_float_or_none(row.get("timestamp_s")) for row in controller_rows)
        if value is not None
    ]
    median_dt = _median_dt(timestamps)
    event_count = 0
    total_duration_s = 0.0
    in_event = False
    sorted_rows = sorted(
        controller_rows,
        key=lambda row: _float_or_none(row.get("timestamp_s")) if _float_or_none(row.get("timestamp_s")) is not None else 1e18,
    )
    for index, row in enumerate(sorted_rows):
        dropped = _bool_from_csv(row.get("dropped_command"))
        if dropped and not in_event:
            event_count += 1
            in_event = True
        if not dropped:
            in_event = False
            continue
        current_time = _float_or_none(row.get("timestamp_s"))
        if current_time is None:
            continue
        if index + 1 < len(sorted_rows):
            next_time = _float_or_none(sorted_rows[index + 1].get("timestamp_s"))
            if next_time is not None:
                total_duration_s += max(0.0, float(next_time) - float(current_time))
                continue
        total_duration_s += float(median_dt)
    return {
        "dropped_command_event_count": float(event_count),
        "dropped_command_duration_s": float(total_duration_s),
    }


def _control_success_summary(
    control_config: dict[str, Any],
    camera_gt_rows: list[dict[str, Any]],
    controller_rows: list[dict[str, Any]],
    realized_rows: list[dict[str, Any]],
    *,
    start_time_s: float | None,
    end_time_s: float | None,
) -> dict[str, float | None]:
    waypoints = control_config.get("waypoints", []) if isinstance(control_config, dict) else []
    gt_samples = [
        (float(timestamp), position)
        for row in camera_gt_rows
        if (timestamp := _float_or_none(row.get("timestamp_s"))) is not None
        and (position := _position_from_gt_row(row)) is not None
    ]
    if not waypoints or not gt_samples:
        return {
            "waypoint_success_fraction_1cm": None,
            "waypoint_success_fraction_2cm": None,
            "waypoint_success_fraction_5cm": None,
            "mean_waypoint_dwell_time_s": None,
            "p95_waypoint_dwell_time_s": None,
            "final_completion_time_s": None,
            "mean_commanded_realized_path_deviation_m": None,
            "p95_commanded_realized_path_deviation_m": None,
            "time_integrated_commanded_realized_path_deviation_m_s": None,
            "dropped_command_event_count": None,
            "dropped_command_duration_s": None,
        }

    windows = _controller_waypoint_windows(
        controller_rows,
        waypoint_count=len(waypoints),
        default_end_time_s=end_time_s,
    )
    success_counts = {0.01: 0, 0.02: 0, 0.05: 0}
    dwell_times_s: list[float] = []
    final_completion_time_s: float | None = None
    for waypoint_index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, dict):
            continue
        target = _vector_or_none(waypoint.get("position_world_m"))
        tolerance_m = _float_or_none(waypoint.get("tolerance_m"))
        if target is None:
            continue
        window = windows[waypoint_index] if waypoint_index < len(windows) else {}
        window_start_s = _float_or_none(window.get("start_s"))
        window_end_s = _float_or_none(window.get("end_s"))
        if window_start_s is None or window_end_s is None:
            window_samples: list[tuple[float, np.ndarray]] = []
        else:
            window_samples = _series_within_window(gt_samples, start_s=window_start_s, end_s=window_end_s)
        for threshold_m in success_counts:
            if _window_first_hit_time_s(window_samples, target, threshold_m=float(threshold_m)) is not None:
                success_counts[threshold_m] += 1
        if tolerance_m is not None:
            dwell_times_s.append(_window_dwell_time_s(window_samples, target, threshold_m=float(tolerance_m)))
            hit_time = _window_first_hit_time_s(window_samples, target, threshold_m=float(tolerance_m))
            if waypoint_index == len(waypoints) - 1 and hit_time is not None:
                final_completion_time_s = float(hit_time) if start_time_s is None else float(hit_time - start_time_s)

    waypoint_count = max(len(waypoints), 1)
    path_deviation = _commanded_realized_path_deviation(controller_rows, realized_rows)
    dropped_command = _dropped_command_stats(controller_rows)
    return {
        "waypoint_success_fraction_1cm": float(success_counts[0.01] / waypoint_count),
        "waypoint_success_fraction_2cm": float(success_counts[0.02] / waypoint_count),
        "waypoint_success_fraction_5cm": float(success_counts[0.05] / waypoint_count),
        "mean_waypoint_dwell_time_s": None if not dwell_times_s else float(np.mean(np.asarray(dwell_times_s, dtype=np.float64))),
        "p95_waypoint_dwell_time_s": _safe_percentile(dwell_times_s, 95.0),
        "final_completion_time_s": final_completion_time_s,
        "mean_commanded_realized_path_deviation_m": path_deviation["mean_commanded_realized_path_deviation_m"],
        "p95_commanded_realized_path_deviation_m": path_deviation["p95_commanded_realized_path_deviation_m"],
        "time_integrated_commanded_realized_path_deviation_m_s": path_deviation[
            "time_integrated_commanded_realized_path_deviation_m_s"
        ],
        "dropped_command_event_count": dropped_command["dropped_command_event_count"],
        "dropped_command_duration_s": dropped_command["dropped_command_duration_s"],
    }


def _first_realized_position(row: dict[str, Any]) -> float | None:
    raw = row.get("positions")
    if raw in ("", None):
        return None
    first = str(raw).split("|")[0]
    return _float_or_none(first)


def _first_command_value(row: dict[str, Any]) -> float | None:
    raw_value = _float_or_none(row.get("command_value"))
    if raw_value is not None:
        return raw_value
    for key in ("desired_positions", "effective_positions"):
        raw = row.get(key)
        if raw in ("", None):
            continue
        first = str(raw).split("|")[0]
        value = _float_or_none(first)
        if value is not None:
            return value
    return None


def _actuator_tracking_error(command_rows: list[dict[str, Any]], realized_rows: list[dict[str, Any]]) -> float | None:
    realized_samples = []
    for row in realized_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _first_realized_position(row)
        if timestamp is not None and position is not None:
            realized_samples.append((timestamp, position))
    if not realized_samples:
        return None
    realized_times = np.asarray([item[0] for item in realized_samples], dtype=np.float64)
    errors: list[float] = []
    for row in command_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        command_value = _first_command_value(row)
        if timestamp is None or command_value is None:
            continue
        realized_index = int(np.argmin(np.abs(realized_times - timestamp)))
        realized_value = realized_samples[realized_index][1]
        errors.append(abs(command_value - realized_value))
    return None if not errors else float(np.mean(errors))


def _safe_percentile(values: list[float], percentile: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _safe_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compute_isaac_run_metrics(run_dir: str | Path) -> dict[str, Any]:
    resolved = Path(run_dir).resolve()
    bundle = load_replay_bundle(resolved, include_gt=True)
    scene_config = _load_config_snapshot(resolved, "scene")
    camera_config = _load_config_snapshot(resolved, "camera")
    imu_config = _load_config_snapshot(resolved, "imu")
    actuation_config = _load_config_snapshot(resolved, "actuation")
    visibility_config = _load_config_snapshot(resolved, "visibility")

    timestamps = _timestamp_values(bundle.raw)
    duration_s = 0.0 if not timestamps else float(max(timestamps) - min(timestamps))
    start_time_s = None if not timestamps else float(min(timestamps))

    detected_tag_ids = sorted({int(row["tag_id"]) for row in bundle.raw["detections"]})
    auxiliary_tag_ids = [tag_id for tag_id in detected_tag_ids if tag_id != bundle.manifest.anchor_tag_id]
    uncertainty_rows = bundle.estimates["uncertainty"]
    filter_rows = bundle.estimates["filter_state"]
    smoother_rows = bundle.estimates["smoother_state"]
    command_rows = bundle.raw["commands"]
    controller_diagnostics_rows = bundle.raw["controller_diagnostics"]
    camera_gt_rows = bundle.gt.get("camera_gt", []) if bundle.gt else []
    reprojection_stats = _reprojection_error_stats(
        bundle.raw["camera_frames"],
        bundle.raw["detections"],
        filter_rows,
        bundle.gt.get("tag_gt", {}) if bundle.gt else {},
    )
    auxiliary_tag_position_stats = _auxiliary_tag_position_error_stats(
        bundle.gt.get("tag_gt", {}) if bundle.gt else {},
        smoother_rows,
        anchor_tag_id=int(bundle.manifest.anchor_tag_id),
    )

    command_values = [abs(value) for row in command_rows if (value := _first_command_value(row)) is not None]
    max_command = max(command_values) if command_values else 0.0
    saturation_fraction = (
        0.0
        if not command_values or max_command <= 0.0
        else float(sum(value >= 0.95 * max_command for value in command_values) / len(command_values))
    )

    position_radii = [float(row["position_radius_95_m"]) for row in uncertainty_rows if "position_radius_95_m" in row]
    innovation_norms = [
        float(row.get("innovation_diagnostics", {}).get("last_innovation_norm", 0.0))
        for row in filter_rows
    ]
    gyro_bias_norms = [
        float(np.linalg.norm(np.asarray(row.get("gyro_bias_rps", []), dtype=np.float64)))
        for row in filter_rows
        if row.get("gyro_bias_rps")
    ]
    accel_bias_norms = [
        float(np.linalg.norm(np.asarray(row.get("accel_bias_mps2", []), dtype=np.float64)))
        for row in filter_rows
        if row.get("accel_bias_mps2")
    ]

    position_matches = _nearest_position_matches(filter_rows, bundle.gt.get("camera_gt", []) if bundle.gt else [], uncertainty_rows)
    position_errors = [match["position_error_m"] for match in position_matches]
    matched_radii = [
        float(match["position_radius_95_m"])
        for match in position_matches
        if match.get("position_radius_95_m") is not None
    ]
    matched_error_subset = [
        float(match["position_error_m"])
        for match in position_matches
        if match.get("position_radius_95_m") is not None
    ]
    position_nees_values = [
        float(match["position_nees"]) for match in position_matches if match.get("position_nees") is not None
    ]
    waypoint_summary = _ground_truth_waypoint_summary(bundle.manifest.controller_config, camera_gt_rows, start_time_s)
    control_success_summary = _control_success_summary(
        bundle.manifest.controller_config,
        camera_gt_rows,
        controller_diagnostics_rows,
        bundle.raw["realized_joints"],
        start_time_s=start_time_s,
        end_time_s=None if not timestamps else float(max(timestamps)),
    )

    coverage_hits = [
        float(match["position_error_m"]) <= float(match["position_radius_95_m"])
        for match in position_matches
        if match.get("position_radius_95_m") is not None
    ]
    frame_pack_rows = [
        row
        for row in bundle.raw["estimator_input"]
        if str(row.get("kind", "")) == "frame_pack"
    ]
    anchor_visible_flags = [bool(row.get("anchor_visible", False)) for row in frame_pack_rows]
    anchor_visible_raw_flags = [
        bool(row.get("anchor_visible_raw", row.get("anchor_visible", False))) for row in frame_pack_rows
    ]
    anchor_visible_effective_flags = [
        bool(row.get("anchor_visible_effective", row.get("anchor_visible", False))) for row in frame_pack_rows
    ]
    anchor_update_suppressed_flags = [
        bool(row.get("anchor_update_suppressed", False)) for row in frame_pack_rows
    ]
    anchor_visible_fraction = (
        None if not anchor_visible_flags else float(sum(anchor_visible_flags) / len(anchor_visible_flags))
    )
    anchor_visible_raw_fraction = (
        None if not anchor_visible_raw_flags else float(sum(anchor_visible_raw_flags) / len(anchor_visible_raw_flags))
    )
    anchor_visible_effective_fraction = (
        None
        if not anchor_visible_effective_flags
        else float(sum(anchor_visible_effective_flags) / len(anchor_visible_effective_flags))
    )
    anchor_update_suppressed_fraction = (
        None
        if not anchor_update_suppressed_flags
        else float(sum(anchor_update_suppressed_flags) / len(anchor_update_suppressed_flags))
    )
    anchor_innovation_norms = [
        float(row.get("innovation_diagnostics", {}).get("last_innovation_norm", 0.0))
        for row in filter_rows
        if bool(row.get("anchor_visible", False))
    ]
    anchor_relocalization_count = max(
        [0.0]
        + [
            float(row.get("innovation_diagnostics", {}).get("anchor_relocalizations", 0.0))
            for row in filter_rows
        ]
    )
    auxiliary_update_count = max(
        [0.0]
        + [
            float(row.get("innovation_diagnostics", {}).get("auxiliary_update_count", 0.0))
            for row in filter_rows
        ]
    )
    auxiliary_rejection_count = max(
        [0.0]
        + [
            float(row.get("innovation_diagnostics", {}).get("auxiliary_rejection_count", 0.0))
            for row in filter_rows
        ]
    )
    detections_by_frame: dict[int, list[dict[str, Any]]] = {}
    for detection in bundle.raw["detections"]:
        detections_by_frame.setdefault(int(detection.get("frame_index", -1)), []).append(detection)
    anchor_pnp_success_frames = 0
    fallback_only_frames = 0
    for frame in bundle.raw["camera_frames"]:
        frame_index = int(frame.get("frame_index", -1))
        frame_detections = detections_by_frame.get(frame_index, [])
        if any(
            int(detection.get("tag_id", -1)) == int(bundle.manifest.anchor_tag_id)
            and detection.get("pose_camera_tvec_m") is not None
            and not _is_fallback_backend(detection.get("detector_backend", ""))
            for detection in frame_detections
        ):
            anchor_pnp_success_frames += 1
        if frame_detections and all(
            detection.get("pose_camera_tvec_m") is None or _is_fallback_backend(detection.get("detector_backend", ""))
            for detection in frame_detections
        ):
            fallback_only_frames += 1
    total_frames = max(len(bundle.raw["camera_frames"]), 1)
    controller_ik_failures = [
        not str(row.get("ik_success", "")).lower() in {"true", "1"}
        for row in controller_diagnostics_rows
    ]
    controller_failure_fraction = (
        None if not controller_ik_failures else float(sum(controller_ik_failures) / len(controller_ik_failures))
    )
    safety_reason_counts: dict[str, int] = {}
    for row in controller_diagnostics_rows:
        reason = str(row.get("safety_reason", "") or "")
        safety_reason_counts[reason] = safety_reason_counts.get(reason, 0) + 1
    dominant_safety_reason = None if not safety_reason_counts else max(safety_reason_counts.items(), key=lambda item: item[1])[0]
    current_position_source_counts: dict[str, int] = {}
    for row in controller_diagnostics_rows:
        source = str(row.get("current_position_source", "") or "")
        current_position_source_counts[source] = current_position_source_counts.get(source, 0) + 1
    metrics = {
        "schema_version": 2,
        "run_dir": str(resolved),
        "run_id": bundle.manifest.run_id,
        "manifest": bundle.manifest.summary(),
        "config": {
            "physics_rate_hz": _float_or_none(scene_config.get("physics_rate_hz")),
            "imu_rate_hz": _float_or_none(imu_config.get("rate_hz") or scene_config.get("imu_rate_hz")),
            "camera_rate_hz": _float_or_none(camera_config.get("rate_hz") or scene_config.get("camera_rate_hz")),
            "filter_rate_hz": _float_or_none(scene_config.get("filter_rate_hz")),
            "smoother_rate_hz": _float_or_none(scene_config.get("smoother_rate_hz")),
            "controller_rate_hz": _float_or_none(scene_config.get("controller_rate_hz")),
            "observed_auxiliary_tag_ids": auxiliary_tag_ids,
            "noise_preset": bundle.manifest.noise_presets.get("imu") or actuation_config.get("name"),
            "actuation_preset": bundle.manifest.noise_presets.get("actuation"),
            "estimator_mode": bundle.manifest.estimator_mode,
            "controller_mode": bundle.manifest.controller_mode,
            "bootstrap_control_policy": bundle.manifest.bootstrap_control_policy,
            "visibility_preset": visibility_config.get("name"),
        },
        "counts": {
            "camera_frames": len(bundle.raw["camera_frames"]),
            "detections": len(bundle.raw["detections"]),
            "imu_packets": len(bundle.raw["imu"]),
            "commands": len(bundle.raw["commands"]),
            "controller_diagnostics": len(bundle.raw["controller_diagnostics"]),
            "realized_joints": len(bundle.raw["realized_joints"]),
            "filter_states": len(bundle.estimates["filter_state"]),
            "smoother_states": len(bundle.estimates["smoother_state"]),
            "uncertainty_states": len(bundle.estimates["uncertainty"]),
            "unique_detected_tags": len(detected_tag_ids),
            "unique_detected_auxiliary_tags": len(auxiliary_tag_ids),
        },
        "timing": {
            "duration_s": float(duration_s),
            "first_timestamp_s": None if not timestamps else float(min(timestamps)),
            "last_timestamp_s": None if not timestamps else float(max(timestamps)),
        },
        "estimation": {
            "detected_tag_ids": detected_tag_ids,
            "mean_position_radius_95_m": None if not position_radii else float(np.mean(position_radii)),
            "max_position_radius_95_m": None if not position_radii else float(np.max(position_radii)),
            "mean_innovation_norm": None if not innovation_norms else float(np.mean(innovation_norms)),
            "anchor_visible_fraction": anchor_visible_fraction,
            "anchor_visible_raw_fraction": anchor_visible_raw_fraction,
            "anchor_visible_effective_fraction": anchor_visible_effective_fraction,
            "anchor_update_suppressed_fraction": anchor_update_suppressed_fraction,
            "anchor_relocalization_count": float(anchor_relocalization_count),
            "mean_anchor_innovation_norm": None
            if not anchor_innovation_norms
            else float(np.mean(anchor_innovation_norms)),
            "auxiliary_update_count": float(auxiliary_update_count),
            "auxiliary_rejection_count": float(auxiliary_rejection_count),
            "anchor_pnp_success_fraction": float(anchor_pnp_success_frames / total_frames),
            "fallback_only_frame_fraction": float(fallback_only_frames / total_frames),
        },
        "trajectory": {
            "mean_position_error_m": None if not position_errors else float(np.mean(position_errors)),
            "p95_position_error_m": _safe_percentile(position_errors, 95.0),
            "max_position_error_m": None if not position_errors else float(np.max(position_errors)),
            "mean_rotation_error_deg": None,
            "mean_reprojection_error_px": reprojection_stats["mean_reprojection_rmse_px"],
            "p95_reprojection_error_px": reprojection_stats["p95_reprojection_rmse_px"],
            "map_error_m": _tag_map_error(bundle.gt.get("tag_gt", {}) if bundle.gt else {}, smoother_rows),
        },
        "parameters": {
            "mean_gyro_bias_norm_rps": None if not gyro_bias_norms else float(np.mean(gyro_bias_norms)),
            "mean_accel_bias_norm_mps2": None if not accel_bias_norms else float(np.mean(accel_bias_norms)),
        },
        "residuals": {
            "mean_visual_residual_sq": None,
            "mean_imu_residual_sq": None,
            "mean_reprojection_rmse_px": reprojection_stats["mean_reprojection_rmse_px"],
            "p95_reprojection_rmse_px": reprojection_stats["p95_reprojection_rmse_px"],
            "reprojection_sample_count": reprojection_stats["reprojection_sample_count"],
            "mean_anchor_innovation_norm": None
            if not anchor_innovation_norms
            else float(np.mean(anchor_innovation_norms)),
            "mean_innovation_norm": None if not innovation_norms else float(np.mean(innovation_norms)),
            "anchor_pnp_success_fraction": float(anchor_pnp_success_frames / total_frames),
            "fallback_only_frame_fraction": float(fallback_only_frames / total_frames),
        },
        "uncertainty_calibration": {
            "mean_position_radius_95_m": None if not position_radii else float(np.mean(position_radii)),
            "empirical_95_coverage_percent": None
            if not coverage_hits
            else float(sum(coverage_hits) / len(coverage_hits) * 100.0),
            "pose_nees": None if not position_nees_values else float(np.mean(position_nees_values)),
            "velocity_nees": None,
            "sigma_error_correlation": _safe_correlation(matched_radii, matched_error_subset),
        },
        "map_quality": {
            "mean_auxiliary_tag_position_error_m": auxiliary_tag_position_stats["mean_auxiliary_tag_position_error_m"],
            "p95_auxiliary_tag_position_error_m": auxiliary_tag_position_stats["p95_auxiliary_tag_position_error_m"],
            "auxiliary_tag_count": auxiliary_tag_position_stats["auxiliary_tag_count"],
            "anchor_relocalization_count": float(anchor_relocalization_count),
            "anchor_visible_fraction": anchor_visible_fraction,
            "anchor_visible_raw_fraction": anchor_visible_raw_fraction,
            "anchor_visible_effective_fraction": anchor_visible_effective_fraction,
        },
        "control": {
            "saturation_fraction": float(saturation_fraction),
            "max_command_abs": float(max_command),
            "mean_waypoint_error_m": waypoint_summary["mean_waypoint_error_m"],
            "p95_waypoint_error_m": waypoint_summary["p95_waypoint_error_m"],
            "completion_fraction": waypoint_summary["completion_fraction"],
            "mean_completion_time_s": waypoint_summary["mean_completion_time_s"],
            "failure_rate_percent": waypoint_summary["failure_rate_percent"],
            "mean_actuator_tracking_error": _actuator_tracking_error(command_rows, bundle.raw["realized_joints"]),
            "ik_failure_fraction": controller_failure_fraction,
            "dominant_safety_reason": dominant_safety_reason,
            "state_source_counts": current_position_source_counts,
        },
        "control_success": control_success_summary,
        "ground_truth_available": {
            "camera_gt": bool(bundle.gt and bundle.gt.get("camera_gt")),
            "imu_gt": bool(bundle.gt and bundle.gt.get("imu_gt")),
            "joint_gt": bool(bundle.gt and bundle.gt.get("joint_gt")),
            "tag_gt": bool(bundle.gt and bundle.gt.get("tag_gt")),
        },
    }
    analysis_dir = resolved / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _nearest_filter_sample(
    filter_rows: list[dict[str, Any]],
    *,
    timestamp_s: float,
) -> dict[str, Any] | None:
    usable_rows = [
        row
        for row in filter_rows
        if _float_or_none(row.get("timestamp_s")) is not None
    ]
    if not usable_rows:
        return None
    times = np.asarray([float(row["timestamp_s"]) for row in usable_rows], dtype=np.float64)
    index = int(np.argmin(np.abs(times - float(timestamp_s))))
    return usable_rows[index]


def _detection_reprojection_rows(
    bundle: Any,
) -> list[dict[str, Any]]:
    if not bundle.gt or not bundle.gt.get("tag_gt"):
        return []
    gt_payload = bundle.gt.get("tag_gt", {})
    gt_tag_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for tag in gt_payload.get("tags", []):
        if not isinstance(tag, dict) or "tag_id" not in tag:
            continue
        position = _position_from_tag_gt(tag)
        rotation = _rotation_from_tag_gt(tag)
        if position is None or rotation is None:
            continue
        gt_tag_poses[int(tag["tag_id"])] = (position, rotation)
    frame_by_index = {
        int(row["frame_index"]): row
        for row in bundle.raw.get("camera_frames", [])
        if "frame_index" in row and "intrinsics_snapshot" in row
    }
    frame_pack_by_index = {
        int(row["frame_index"]): row
        for row in bundle.raw.get("estimator_input", [])
        if str(row.get("kind", "")) == "frame_pack" and "frame_index" in row
    }
    filter_rows = bundle.estimates.get("filter_state", [])
    rows: list[dict[str, Any]] = []
    for detection in bundle.raw.get("detections", []):
        frame = frame_by_index.get(int(detection.get("frame_index", -1)))
        if frame is None:
            continue
        gt_pose = gt_tag_poses.get(int(detection.get("tag_id", -1)))
        if gt_pose is None:
            continue
        filter_row = _nearest_filter_sample(filter_rows, timestamp_s=float(detection.get("timestamp_s", 0.0) or 0.0))
        if filter_row is None:
            continue
        position_world_m = _vector_or_none(filter_row.get("position_world_m"))
        rotation_wi = _matrix3_or_none(filter_row.get("rotation_wi"))
        local_tag_points_m = np.asarray(detection.get("local_tag_points_m", []), dtype=np.float64)
        corners_xy = np.asarray(detection.get("corners_xy", []), dtype=np.float64)
        intrinsics = frame.get("intrinsics_snapshot", {})
        if (
            position_world_m is None
            or rotation_wi is None
            or local_tag_points_m.shape != (4, 3)
            or corners_xy.shape != (4, 2)
        ):
            continue
        fx = _float_or_none(intrinsics.get("fx_px"))
        fy = _float_or_none(intrinsics.get("fy_px"))
        cx = _float_or_none(intrinsics.get("cx_px"))
        cy = _float_or_none(intrinsics.get("cy_px"))
        if fx is None or fy is None or cx is None or cy is None:
            continue
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        distortion = intrinsics.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0])
        dist_coeffs = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
        tag_position_world_m, rotation_wt = gt_pose
        world_points = (rotation_wt @ local_tag_points_m.T).T + tag_position_world_m.reshape(1, 3)
        rotation_wc = rotation_wi
        rotation_cw = rotation_wc.T
        translation_cw = -rotation_cw @ position_world_m.reshape(3)
        rvec_cw, _ = cv2.Rodrigues(rotation_cw)
        projected_points, _ = cv2.projectPoints(
            world_points,
            rvec_cw,
            translation_cw.reshape(3, 1),
            camera_matrix,
            dist_coeffs,
        )
        projected_corners = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
        rmse_px = float(np.sqrt(np.mean(np.sum((projected_corners - corners_xy) ** 2, axis=1))))
        frame_pack = frame_pack_by_index.get(int(detection.get("frame_index", -1)), {})
        innovation_diagnostics = filter_row.get("innovation_diagnostics", {})
        anchor_relocalizations = int(float(innovation_diagnostics.get("anchor_relocalizations", 0.0) or 0.0))
        detector_backend = str(detection.get("detector_backend", ""))
        rows.append(
            {
                "timestamp_s": float(detection.get("timestamp_s", 0.0)),
                "frame_index": int(detection.get("frame_index", -1)),
                "tag_id": int(detection.get("tag_id", -1)),
                "is_anchor": bool(detection.get("is_anchor", False)),
                "detector_backend": detector_backend,
                "native_backend": not _is_fallback_backend(detector_backend),
                "reprojection_rmse_px": rmse_px,
                "anchor_visible": bool(frame_pack.get("anchor_visible", False)),
                "anchor_relocalizations": anchor_relocalizations,
                "before_anchor_relocalization": bool(anchor_relocalizations <= 0),
                "after_anchor_relocalization": bool(anchor_relocalizations > 0),
            }
        )
    return rows


def _mean_and_p95(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.percentile(array, 95.0))


def compute_isaac_estimator_quality(run_dir: str | Path) -> dict[str, Any]:
    resolved = Path(run_dir).resolve()
    input_bundle = load_replay_bundle(resolved, include_gt=True)
    metrics = compute_isaac_run_metrics(resolved)
    detection_rows = _detection_reprojection_rows(input_bundle)
    frame_pack_rows = [
        row
        for row in input_bundle.raw.get("estimator_input", [])
        if str(row.get("kind", "")) == "frame_pack"
    ]
    smoother_rows = input_bundle.estimates.get("smoother_state", [])

    anchor_residuals = [float(row["reprojection_rmse_px"]) for row in detection_rows if bool(row["is_anchor"])]
    auxiliary_residuals = [float(row["reprojection_rmse_px"]) for row in detection_rows if not bool(row["is_anchor"])]
    native_residuals = [float(row["reprojection_rmse_px"]) for row in detection_rows if bool(row["native_backend"])]
    fallback_residuals = [float(row["reprojection_rmse_px"]) for row in detection_rows if not bool(row["native_backend"])]
    pre_relocalization = [
        float(row["reprojection_rmse_px"]) for row in detection_rows if bool(row["before_anchor_relocalization"])
    ]
    post_relocalization = [
        float(row["reprojection_rmse_px"]) for row in detection_rows if bool(row["after_anchor_relocalization"])
    ]

    residuals_by_tag: dict[int, list[float]] = {}
    for row in detection_rows:
        residuals_by_tag.setdefault(int(row["tag_id"]), []).append(float(row["reprojection_rmse_px"]))

    tag_update_breakdown: dict[int, dict[str, Any]] = {}
    total_accepted_aux_updates = 0
    total_rejected_aux_updates = 0
    total_auxiliary_contributor_count = 0
    anchor_visible_flags: list[bool] = []
    for row in frame_pack_rows:
        anchor_visible_flags.append(bool(row.get("anchor_visible", False)))
        total_auxiliary_contributor_count += int(row.get("accepted_auxiliary_update_count", 0) or 0)
        for decision in row.get("auxiliary_update_decisions", []) or []:
            if not isinstance(decision, dict):
                continue
            tag_id = int(decision.get("tag_id", -1))
            entry = tag_update_breakdown.setdefault(
                tag_id,
                {
                    "tag_id": tag_id,
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "fallback_only": 0,
                    "corner_margin_fail": 0,
                    "reproj_fail": 0,
                    "innovation_fail": 0,
                    "visibility_streak_fail": 0,
                    "covariance_fail": 0,
                },
            )
            accepted = bool(decision.get("accepted", False))
            reason = str(decision.get("reason", ""))
            if accepted:
                entry["accepted_count"] += 1
                total_accepted_aux_updates += 1
            else:
                entry["rejected_count"] += 1
                total_rejected_aux_updates += 1
                if reason in entry:
                    entry[reason] += 1
    if not tag_update_breakdown:
        for row in detection_rows:
            if bool(row.get("is_anchor", False)):
                continue
            tag_id = int(row["tag_id"])
            entry = tag_update_breakdown.setdefault(
                tag_id,
                {
                    "tag_id": tag_id,
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "fallback_only": 0,
                    "corner_margin_fail": 0,
                    "covariance_fail": 0,
                    "innovation_fail": 0,
                    "reproj_fail": 0,
                    "visibility_streak_fail": 0,
                },
            )
            if bool(row.get("native_backend", False)):
                entry["accepted_count"] += 1
            else:
                entry["rejected_count"] += 1
                entry["fallback_only"] += 1
        if input_bundle.estimates.get("filter_state"):
            total_accepted_aux_updates = int(
                max(
                    float(row.get("innovation_diagnostics", {}).get("auxiliary_update_count", 0.0))
                    for row in input_bundle.estimates.get("filter_state", [])
                )
            )
            total_rejected_aux_updates = int(
                max(
                    float(row.get("innovation_diagnostics", {}).get("auxiliary_rejection_count", 0.0))
                    for row in input_bundle.estimates.get("filter_state", [])
                )
            )
    if frame_pack_rows and total_auxiliary_contributor_count == 0:
        auxiliary_detections_by_frame = {}
        for row in detection_rows:
            if bool(row.get("is_anchor", False)):
                continue
            frame_index = int(row["frame_index"])
            auxiliary_detections_by_frame[frame_index] = auxiliary_detections_by_frame.get(frame_index, 0) + 1
        total_auxiliary_contributor_count = int(sum(auxiliary_detections_by_frame.values()))
    if not frame_pack_rows:
        auxiliary_detections_by_frame: dict[int, int] = {}
        anchor_visible_by_frame: dict[int, bool] = {}
        for row in detection_rows:
            frame_index = int(row["frame_index"])
            if bool(row.get("is_anchor", False)):
                anchor_visible_by_frame[frame_index] = True
            else:
                auxiliary_detections_by_frame[frame_index] = auxiliary_detections_by_frame.get(frame_index, 0) + 1
        if auxiliary_detections_by_frame:
            total_auxiliary_contributor_count = int(sum(auxiliary_detections_by_frame.values()))
            anchor_visible_flags = [
                bool(anchor_visible_by_frame.get(frame_index, False))
                for frame_index in sorted(auxiliary_detections_by_frame)
            ]
            frame_pack_rows = [{"frame_index": frame_index} for frame_index in sorted(auxiliary_detections_by_frame)]

    smoother_timeline = []
    for row in smoother_rows:
        diagnostics = row.get("diagnostics", {})
        smoother_timeline.append(
            {
                "timestamp_s": float(row.get("timestamp_s", 0.0) or 0.0),
                "feedback_correction_norm_m": _float_or_none(diagnostics.get("feedback_correction_norm_m")),
                "trusted_feedback_count": _float_or_none(diagnostics.get("trusted_feedback_count")),
                "total_feedback_candidates": _float_or_none(diagnostics.get("total_feedback_candidates")),
            }
        )

    anchor_mean, anchor_p95 = _mean_and_p95(anchor_residuals)
    aux_mean, aux_p95 = _mean_and_p95(auxiliary_residuals)
    native_mean, native_p95 = _mean_and_p95(native_residuals)
    fallback_mean, fallback_p95 = _mean_and_p95(fallback_residuals)
    pre_mean, pre_p95 = _mean_and_p95(pre_relocalization)
    post_mean, post_p95 = _mean_and_p95(post_relocalization)
    smoother_norms = [
        float(row["feedback_correction_norm_m"])
        for row in smoother_timeline
        if row.get("feedback_correction_norm_m") is not None
    ]
    smoother_mean, smoother_p95 = _mean_and_p95(smoother_norms)
    tag_rows = [
        {
            "tag_id": int(tag_id),
            "mean_reprojection_rmse_px": float(np.mean(values)),
            "p95_reprojection_rmse_px": float(np.percentile(np.asarray(values, dtype=np.float64), 95.0)),
            "sample_count": int(len(values)),
        }
        for tag_id, values in sorted(residuals_by_tag.items())
    ]
    quality = {
        "run_id": metrics["run_id"],
        "summary": {
            "anchor_mean_reprojection_rmse_px": anchor_mean,
            "anchor_p95_reprojection_rmse_px": anchor_p95,
            "auxiliary_mean_reprojection_rmse_px": aux_mean,
            "auxiliary_p95_reprojection_rmse_px": aux_p95,
            "native_mean_reprojection_rmse_px": native_mean,
            "native_p95_reprojection_rmse_px": native_p95,
            "fallback_mean_reprojection_rmse_px": fallback_mean,
            "fallback_p95_reprojection_rmse_px": fallback_p95,
            "pre_relocalization_mean_reprojection_rmse_px": pre_mean,
            "pre_relocalization_p95_reprojection_rmse_px": pre_p95,
            "post_relocalization_mean_reprojection_rmse_px": post_mean,
            "post_relocalization_p95_reprojection_rmse_px": post_p95,
            "mean_smoother_correction_norm_m": smoother_mean,
            "p95_smoother_correction_norm_m": smoother_p95,
            "accepted_auxiliary_updates": int(total_accepted_aux_updates),
            "rejected_auxiliary_updates": int(total_rejected_aux_updates),
            "anchor_visible_fraction": None
            if not anchor_visible_flags
            else float(sum(anchor_visible_flags) / len(anchor_visible_flags)),
            "mean_auxiliary_contributor_count": None
            if not frame_pack_rows
            else float(total_auxiliary_contributor_count / len(frame_pack_rows)),
        },
        "uncertainty": dict(metrics.get("uncertainty_calibration", {})),
        "map_quality": dict(metrics.get("map_quality", {})),
        "tag_residuals": tag_rows,
        "tag_update_breakdown": [value for _tag_id, value in sorted(tag_update_breakdown.items())],
        "smoother_timeline": smoother_timeline,
        "detection_residuals": detection_rows,
    }
    return quality
