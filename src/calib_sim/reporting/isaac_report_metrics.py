"""Metric extraction for Isaac runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from calib_sim.isaac.runtime.replay import load_replay_bundle


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


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


def _csv_vector_or_none(row: dict[str, Any], x_key: str, y_key: str, z_key: str) -> np.ndarray | None:
    if x_key not in row or y_key not in row or z_key not in row:
        return None
    x = _float_or_none(row.get(x_key))
    y = _float_or_none(row.get(y_key))
    z = _float_or_none(row.get(z_key))
    if x is None or y is None or z is None:
        return None
    return np.asarray([x, y, z], dtype=np.float64)


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

    timestamps = _timestamp_values(bundle.raw)
    duration_s = 0.0 if not timestamps else float(max(timestamps) - min(timestamps))
    start_time_s = None if not timestamps else float(min(timestamps))

    detected_tag_ids = sorted({int(row["tag_id"]) for row in bundle.raw["detections"]})
    auxiliary_tag_ids = [tag_id for tag_id in detected_tag_ids if tag_id != bundle.manifest.anchor_tag_id]
    uncertainty_rows = bundle.estimates["uncertainty"]
    filter_rows = bundle.estimates["filter_state"]
    smoother_rows = bundle.estimates["smoother_state"]
    command_rows = bundle.raw["commands"]

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
    waypoint_summary = _waypoint_summary(bundle.manifest.controller_config, filter_rows, start_time_s)

    coverage_hits = [
        float(match["position_error_m"]) <= float(match["position_radius_95_m"])
        for match in position_matches
        if match.get("position_radius_95_m") is not None
    ]
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
        },
        "counts": {
            "camera_frames": len(bundle.raw["camera_frames"]),
            "detections": len(bundle.raw["detections"]),
            "imu_packets": len(bundle.raw["imu"]),
            "commands": len(bundle.raw["commands"]),
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
        },
        "trajectory": {
            "mean_position_error_m": None if not position_errors else float(np.mean(position_errors)),
            "p95_position_error_m": _safe_percentile(position_errors, 95.0),
            "max_position_error_m": None if not position_errors else float(np.max(position_errors)),
            "mean_rotation_error_deg": None,
            "mean_reprojection_error_px": None,
            "map_error_m": _tag_map_error(bundle.gt.get("tag_gt", {}) if bundle.gt else {}, smoother_rows),
        },
        "parameters": {
            "mean_gyro_bias_norm_rps": None if not gyro_bias_norms else float(np.mean(gyro_bias_norms)),
            "mean_accel_bias_norm_mps2": None if not accel_bias_norms else float(np.mean(accel_bias_norms)),
        },
        "residuals": {
            "mean_visual_residual_sq": None,
            "mean_imu_residual_sq": None,
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
        "control": {
            "saturation_fraction": float(saturation_fraction),
            "max_command_abs": float(max_command),
            "mean_waypoint_error_m": waypoint_summary["mean_waypoint_error_m"],
            "p95_waypoint_error_m": waypoint_summary["p95_waypoint_error_m"],
            "completion_fraction": waypoint_summary["completion_fraction"],
            "mean_completion_time_s": waypoint_summary["mean_completion_time_s"],
            "failure_rate_percent": waypoint_summary["failure_rate_percent"],
            "mean_actuator_tracking_error": _actuator_tracking_error(command_rows, bundle.raw["realized_joints"]),
        },
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
