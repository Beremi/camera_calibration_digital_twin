"""Slim tabletop-only localization exports for the checked-in dataset workflow."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calib_sim.estimation._geometry import relative_rotation_error_deg, rotation_matrix_from_rvec
from calib_sim.isaac.estimation.uncertainty import radius_95_from_covariance, sanitize_covariance
from calib_sim.isaac.logging.schemas import IsaacCameraFramePacket, IsaacTagDetectionPacket
from calib_sim.isaac.runtime.replay import load_replay_bundle
from calib_sim.isaac.stage_builder import stage_spec_from_config
from calib_sim.isaac.tag_builder import TagPoseSpec


MEASUREMENT_SCHEMA_VERSION = 1
_MEASUREMENT_COVARIANCE_ORDER = ("rx_rad", "ry_rad", "rz_rad", "x_m", "y_m", "z_m")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _vector_or_none(value: Any, *, expected_len: int = 3) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) < expected_len:
        return None
    return np.asarray([float(value[index]) for index in range(expected_len)], dtype=np.float64)


def _safe_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(percentile)))


def _rotation_matrix_from_quaternion_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
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


def _quaternion_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        return np.array(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    diagonal = np.diag(matrix)
    if diagonal[0] > diagonal[1] and diagonal[0] > diagonal[2]:
        s = 2.0 * math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        return np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / s,
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    if diagonal[1] > diagonal[2]:
        s = 2.0 * math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        return np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    s = 2.0 * math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
    return np.array(
        [
            (matrix[1, 0] - matrix[0, 1]) / s,
            (matrix[0, 2] + matrix[2, 0]) / s,
            (matrix[1, 2] + matrix[2, 1]) / s,
            0.25 * s,
        ],
        dtype=np.float64,
    )


def _rotation_average(rotations: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if len(rotations) == 1:
        return np.asarray(rotations[0], dtype=np.float64).reshape(3, 3)
    accumulator = np.zeros((4, 4), dtype=np.float64)
    reference = _quaternion_wxyz_from_rotation(rotations[0])
    for rotation, weight in zip(rotations, weights):
        quaternion = _quaternion_wxyz_from_rotation(rotation)
        if float(np.dot(quaternion, reference)) < 0.0:
            quaternion = -quaternion
        accumulator += float(weight) * np.outer(quaternion, quaternion)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    quaternion = eigenvectors[:, int(np.argmax(eigenvalues))]
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    return _rotation_matrix_from_quaternion_wxyz(quaternion)


def _camera_frame_packet_from_row(row: dict[str, Any]) -> IsaacCameraFramePacket:
    return IsaacCameraFramePacket(
        frame_index=int(row["frame_index"]),
        timestamp_s=float(row["timestamp_s"]),
        sim_time_s=float(row.get("sim_time_s", row["timestamp_s"])),
        sensor_time_s=float(row.get("sensor_time_s", row["timestamp_s"])),
        host_time_s=float(row.get("host_time_s", row["timestamp_s"])),
        rgb_path=str(row.get("rgb_path", "")),
        intrinsics_snapshot=dict(row.get("intrinsics_snapshot", {})),
        extrinsics_snapshot=dict(row.get("extrinsics_snapshot", {})),
        image_width_px=int(row.get("image_width_px", 0) or 0),
        image_height_px=int(row.get("image_height_px", 0) or 0),
        visible_gt_tag_ids=tuple(int(tag_id) for tag_id in row.get("visible_gt_tag_ids", [])),
    )


def _tag_detection_packet_from_row(row: dict[str, Any]) -> IsaacTagDetectionPacket:
    return IsaacTagDetectionPacket(
        timestamp_s=float(row["timestamp_s"]),
        sim_time_s=float(row.get("sim_time_s", row["timestamp_s"])),
        frame_index=int(row["frame_index"]),
        tag_id=int(row["tag_id"]),
        family=str(row.get("family", "apriltag36h11")),
        tag_size_m=float(row.get("tag_size_m", 0.0) or 0.0),
        pnp_tag_size_m=None if row.get("pnp_tag_size_m") in (None, "") else float(row["pnp_tag_size_m"]),
        corners_xy=tuple((float(point[0]), float(point[1])) for point in row.get("corners_xy", [])),
        local_tag_points_m=tuple(
            (float(point[0]), float(point[1]), float(point[2]))
            for point in row.get("local_tag_points_m", [])
        ),
        corner_order=str(row.get("corner_order", "clockwise_top_left_first")),
        score=float(row.get("score", 0.0) or 0.0),
        is_anchor=bool(row.get("is_anchor", False)),
        pose_camera_rvec=None if row.get("pose_camera_rvec") is None else tuple(float(value) for value in row["pose_camera_rvec"]),
        pose_camera_tvec_m=None
        if row.get("pose_camera_tvec_m") is None
        else tuple(float(value) for value in row["pose_camera_tvec_m"]),
        detector_backend=str(row.get("detector_backend", "")),
        visibility_flags={str(key): bool(value) for key, value in dict(row.get("visibility_flags", {})).items()},
    )


def _analysis_dir(run_dir: Path) -> Path:
    path = run_dir / "analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _measurement_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "anchor_only_jsonl": output_dir / "camera_pose_measurements_anchor_only.jsonl",
        "multitag_jsonl": output_dir / "camera_pose_measurements_multitag.jsonl",
        "reprojection_summary_csv": output_dir / "camera_pose_measurement_reprojection_summary.csv",
    }


def _tag_pose_map_from_run(run_dir: Path) -> dict[int, TagPoseSpec]:
    scene_config = _load_json(run_dir / "config_snapshot" / "scene.json")
    stage_spec = stage_spec_from_config(scene_config)
    return {int(tag.tag_id): tag for tag in stage_spec.tag_pose_specs}


def _load_camera_gt_rows(run_dir: Path) -> list[dict[str, str]]:
    projection_gt_path = run_dir / "gt" / "camera_projection_gt.csv"
    if projection_gt_path.exists():
        return _load_csv(projection_gt_path)
    return _load_csv(run_dir / "gt" / "camera_gt.csv")


def _match_gt_row(
    gt_rows: list[dict[str, str]],
    *,
    frame_index: int | None,
    timestamp_s: float,
) -> dict[str, str] | None:
    if not gt_rows:
        return None
    if frame_index is not None:
        matching_rows = [
            row
            for row in gt_rows
            if row.get("frame_index") not in ("", None) and int(row["frame_index"]) == int(frame_index)
        ]
        if matching_rows:
            return matching_rows[0]
    return min(
        gt_rows,
        key=lambda row: abs(float(row.get("timestamp_s", row.get("sim_time_s", "0.0"))) - float(timestamp_s)),
    )


def _gt_position_and_rotation(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    position = _vector_or_none([row.get("px"), row.get("py"), row.get("pz")])
    quaternion = _vector_or_none([row.get("qw"), row.get("qx"), row.get("qy"), row.get("qz")], expected_len=4)
    if position is None or quaternion is None:
        return None, None
    return position, _rotation_matrix_from_quaternion_wxyz(quaternion)


def _camera_pose_from_detection(tag_detection: IsaacTagDetectionPacket, tag_pose: TagPoseSpec) -> tuple[np.ndarray, np.ndarray]:
    if tag_detection.pose_camera_rvec is None or tag_detection.pose_camera_tvec_m is None:
        raise ValueError("Tag detection is missing solvePnP pose.")
    rotation_ct = rotation_matrix_from_rvec(np.asarray(tag_detection.pose_camera_rvec, dtype=np.float64))
    rotation_tc = rotation_ct.T
    translation_ct = np.asarray(tag_detection.pose_camera_tvec_m, dtype=np.float64).reshape(3)
    rotation_wt = np.asarray(tag_pose.rotation_wt, dtype=np.float64).reshape(3, 3)
    position_wt = np.asarray(tag_pose.position_world_m, dtype=np.float64).reshape(3)
    position_world = position_wt + rotation_wt @ (-rotation_tc @ translation_ct)
    rotation_wi = rotation_wt @ rotation_tc
    return position_world.astype(np.float64), rotation_wi.astype(np.float64)


def _world_points_for_tag(detection: IsaacTagDetectionPacket, tag_pose: TagPoseSpec) -> np.ndarray | None:
    local_points = np.asarray(detection.local_tag_points_m, dtype=np.float64)
    if local_points.shape != (4, 3):
        return None
    rotation_wt = np.asarray(tag_pose.rotation_wt, dtype=np.float64).reshape(3, 3)
    position_wt = np.asarray(tag_pose.position_world_m, dtype=np.float64).reshape(3)
    return (rotation_wt @ local_points.T).T + position_wt.reshape(1, 3)


def _reprojection_rmse_px(
    *,
    position_world_m: np.ndarray,
    rotation_wi: np.ndarray,
    detection: IsaacTagDetectionPacket,
    tag_pose: TagPoseSpec,
    intrinsics_snapshot: dict[str, object] | None,
) -> float | None:
    if not isinstance(intrinsics_snapshot, dict):
        return None
    fx = _float_or_none(intrinsics_snapshot.get("fx_px"))
    fy = _float_or_none(intrinsics_snapshot.get("fy_px"))
    cx = _float_or_none(intrinsics_snapshot.get("cx_px"))
    cy = _float_or_none(intrinsics_snapshot.get("cy_px"))
    if fx is None or fy is None or cx is None or cy is None:
        return None
    world_points = _world_points_for_tag(detection, tag_pose)
    observed_corners = np.asarray(detection.corners_xy, dtype=np.float64)
    if world_points is None or observed_corners.shape != (4, 2):
        return None
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.asarray(
        intrinsics_snapshot.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(-1, 1)
    rotation_wc = np.asarray(rotation_wi, dtype=np.float64).reshape(3, 3)
    rotation_cw = rotation_wc.T
    translation_cw = -rotation_cw @ np.asarray(position_world_m, dtype=np.float64).reshape(3)
    rvec_cw, _ = cv2.Rodrigues(rotation_cw)
    projected_points, _ = cv2.projectPoints(
        world_points,
        rvec_cw,
        translation_cw.reshape(3, 1),
        camera_matrix,
        dist_coeffs,
    )
    projected_corners = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((projected_corners - observed_corners) ** 2, axis=1))))


def _native_pose_ready(detection: IsaacTagDetectionPacket) -> bool:
    flags = dict(detection.visibility_flags)
    return bool(flags.get("native_backend", False) and flags.get("pose_ready", False))


def _measurement_uncertainty_from_detection(
    detection: IsaacTagDetectionPacket,
    *,
    reprojection_rmse_px: float | None,
) -> tuple[float, float]:
    corners = np.asarray(detection.corners_xy, dtype=np.float64)
    if corners.shape != (4, 2):
        return 0.01, np.deg2rad(2.5)
    edge_lengths_px = [
        float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
        for index in range(4)
    ]
    mean_edge_length_px = max(float(np.mean(np.asarray(edge_lengths_px, dtype=np.float64))), 1.0)
    pixel_error = max(float(reprojection_rmse_px or 0.75), 0.25)
    tag_size_m = float(detection.tag_size_m or detection.pnp_tag_size_m or 0.2)
    meters_per_pixel = tag_size_m / mean_edge_length_px
    position_std_m = max(pixel_error * meters_per_pixel * 1.5, 0.0025)
    rotation_std_rad = max(pixel_error / mean_edge_length_px * 4.0, np.deg2rad(0.5))
    return float(position_std_m), float(rotation_std_rad)


def _pose_covariance(rotation_std_rad: float, position_std_m: float) -> np.ndarray:
    covariance = np.eye(6, dtype=np.float64)
    covariance[0:3, 0:3] *= float(rotation_std_rad) ** 2
    covariance[3:6, 3:6] *= float(position_std_m) ** 2
    return sanitize_covariance(covariance)


def export_isaac_tag_pose_measurements(run_dir: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    resolved = Path(run_dir).resolve()
    bundle = load_replay_bundle(resolved, include_gt=False)
    output_path = _analysis_dir(resolved) if output_dir is None else Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    paths = _measurement_output_paths(output_path)
    manifest_path = output_path / "camera_pose_measurements_manifest.json"
    if manifest_path.exists() and paths["anchor_only_jsonl"].exists() and paths["multitag_jsonl"].exists():
        payload = _load_json(manifest_path)
        payload["manifest_path"] = str(manifest_path.resolve())
        return payload
    tag_pose_map = _tag_pose_map_from_run(resolved)
    gt_rows = _load_camera_gt_rows(resolved)

    frames = [_camera_frame_packet_from_row(row) for row in bundle.raw.get("camera_frames", [])]
    detections = [_tag_detection_packet_from_row(row) for row in bundle.raw.get("detections", [])]
    detections_by_frame: dict[int, list[IsaacTagDetectionPacket]] = {}
    for detection in detections:
        detections_by_frame.setdefault(int(detection.frame_index), []).append(detection)

    packets_by_mode: dict[str, list[dict[str, Any]]] = {"anchor_only": [], "multitag": []}
    summary_rows: list[dict[str, Any]] = []

    for frame in frames:
        frame_detections = detections_by_frame.get(int(frame.frame_index), [])
        gt_row = _match_gt_row(
            gt_rows,
            frame_index=int(frame.frame_index),
            timestamp_s=float(frame.timestamp_s),
        )
        gt_position, _ = (None, None) if gt_row is None else _gt_position_and_rotation(gt_row)
        candidates: list[dict[str, Any]] = []
        for detection in frame_detections:
            if int(detection.tag_id) not in tag_pose_map or not _native_pose_ready(detection):
                continue
            try:
                position_world_m, rotation_wi = _camera_pose_from_detection(detection, tag_pose_map[int(detection.tag_id)])
            except ValueError:
                continue
            reprojection_rmse_px = _reprojection_rmse_px(
                position_world_m=position_world_m,
                rotation_wi=rotation_wi,
                detection=detection,
                tag_pose=tag_pose_map[int(detection.tag_id)],
                intrinsics_snapshot=frame.intrinsics_snapshot,
            )
            position_std_m, rotation_std_rad = _measurement_uncertainty_from_detection(
                detection,
                reprojection_rmse_px=reprojection_rmse_px,
            )
            candidates.append(
                {
                    "tag_id": int(detection.tag_id),
                    "is_anchor": bool(detection.is_anchor),
                    "position_world_m": position_world_m,
                    "rotation_wi": rotation_wi,
                    "reprojection_rmse_px": reprojection_rmse_px,
                    "covariance": _pose_covariance(rotation_std_rad, position_std_m),
                    "weight": 1.0 / max(float(position_std_m) ** 2, 1e-9),
                }
            )

        anchor_candidates = [candidate for candidate in candidates if bool(candidate["is_anchor"])]
        for mode in ("anchor_only", "multitag"):
            selected = anchor_candidates[:1] if mode == "anchor_only" else candidates
            if not selected:
                continue
            weights = [float(candidate["weight"]) for candidate in selected]
            positions = [np.asarray(candidate["position_world_m"], dtype=np.float64) for candidate in selected]
            rotations = [np.asarray(candidate["rotation_wi"], dtype=np.float64) for candidate in selected]
            position_world_m = np.average(np.asarray(positions, dtype=np.float64), axis=0, weights=np.asarray(weights, dtype=np.float64))
            rotation_wi = _rotation_average(rotations, weights)
            if len(selected) == 1:
                covariance = np.asarray(selected[0]["covariance"], dtype=np.float64)
            else:
                mean_position = np.asarray(position_world_m, dtype=np.float64)
                position_covariance = np.zeros((3, 3), dtype=np.float64)
                rotation_covariance = np.zeros((3, 3), dtype=np.float64)
                weight_sum = max(float(sum(weights)), 1e-9)
                for candidate, weight in zip(selected, weights):
                    delta_position = np.asarray(candidate["position_world_m"], dtype=np.float64) - mean_position
                    position_covariance += float(weight) * np.outer(delta_position, delta_position)
                    delta_rotation = rotation_wi.T @ np.asarray(candidate["rotation_wi"], dtype=np.float64)
                    rotvec, _ = cv2.Rodrigues(delta_rotation)
                    delta_rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
                    rotation_covariance += float(weight) * np.outer(delta_rotvec, delta_rotvec)
                position_covariance = sanitize_covariance(position_covariance / weight_sum + np.eye(3) * 1e-5)
                rotation_covariance = sanitize_covariance(rotation_covariance / weight_sum + np.eye(3) * 1e-5)
                covariance = np.zeros((6, 6), dtype=np.float64)
                covariance[0:3, 0:3] = rotation_covariance
                covariance[3:6, 3:6] = position_covariance
                covariance = sanitize_covariance(covariance)

            reprojection_values = [
                float(candidate["reprojection_rmse_px"])
                for candidate in selected
                if candidate["reprojection_rmse_px"] is not None
            ]
            packet = {
                "schema_version": MEASUREMENT_SCHEMA_VERSION,
                "timestamp_s": float(frame.timestamp_s),
                "frame_index": int(frame.frame_index),
                "mode": mode,
                "pose_covariance_order": list(_MEASUREMENT_COVARIANCE_ORDER),
                "position_world_m": [float(value) for value in position_world_m.tolist()],
                "rotation_wi": [[float(value) for value in row] for row in np.asarray(rotation_wi, dtype=np.float64).tolist()],
                "pose_covariance_6x6": [[float(value) for value in row] for row in covariance.tolist()],
                "position_radius_95_m": float(radius_95_from_covariance(covariance[3:6, 3:6])),
                "rotation_radius_95_rad": float(radius_95_from_covariance(covariance[0:3, 0:3])),
                "tag_ids_used": [int(candidate["tag_id"]) for candidate in selected],
                "tag_contributor_count": int(len(selected)),
                "reprojection_rmse_px": None if not reprojection_values else float(np.mean(np.asarray(reprojection_values, dtype=np.float64))),
                "gating": {
                    "native_pose_ready_count": int(len(candidates)),
                    "anchor_pose_ready_count": int(len(anchor_candidates)),
                    "accepted_tag_ids": [int(candidate["tag_id"]) for candidate in selected],
                    "rejected_tag_ids": [
                        int(candidate["tag_id"])
                        for candidate in candidates
                        if int(candidate["tag_id"]) not in {int(item["tag_id"]) for item in selected}
                    ],
                    "detected_tag_ids": [int(detection.tag_id) for detection in frame_detections],
                },
            }
            packets_by_mode[mode].append(packet)
            position_error_m = None if gt_position is None else float(np.linalg.norm(position_world_m - gt_position))
            summary_rows.append(
                {
                    "frame_index": int(frame.frame_index),
                    "timestamp_s": float(frame.timestamp_s),
                    "mode": mode,
                    "tag_contributor_count": int(len(selected)),
                    "tag_ids_used": "|".join(str(int(candidate["tag_id"])) for candidate in selected),
                    "reprojection_rmse_px": packet["reprojection_rmse_px"],
                    "position_radius_95_m": packet["position_radius_95_m"],
                    "position_error_m": position_error_m,
                }
            )

    _write_jsonl(paths["anchor_only_jsonl"], packets_by_mode["anchor_only"])
    _write_jsonl(paths["multitag_jsonl"], packets_by_mode["multitag"])
    _write_csv(
        paths["reprojection_summary_csv"],
        ["frame_index", "timestamp_s", "mode", "tag_contributor_count", "tag_ids_used", "reprojection_rmse_px", "position_radius_95_m", "position_error_m"],
        summary_rows,
    )
    payload = {
        "run_dir": str(resolved),
        "output_dir": str(output_path),
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "paths": {key: str(value) for key, value in paths.items()},
        "counts": {
            "anchor_only": int(len(packets_by_mode["anchor_only"])),
            "multitag": int(len(packets_by_mode["multitag"])),
            "frames": int(len(frames)),
        },
    }
    _write_json(manifest_path, payload)
    payload["manifest_path"] = str(manifest_path.resolve())
    return payload


def calibrate_tag_measurement_noise(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    measurement_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = Path(run_dir).resolve()
    analysis_dir = _analysis_dir(resolved) if output_dir is None else Path(output_dir).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    calibration_json_path = analysis_dir / "camera_pose_noise_calibration.json"
    calibration_table_path = analysis_dir / "camera_pose_noise_calibration_table.csv"
    if calibration_json_path.exists() and calibration_table_path.exists():
        payload = _load_json(calibration_json_path)
        payload["calibration_json_path"] = str(calibration_json_path.resolve())
        payload["calibration_table_path"] = str(calibration_table_path.resolve())
        return payload
    measurement_manifest = (
        export_isaac_tag_pose_measurements(resolved, output_dir=analysis_dir)
        if measurement_manifest is None
        else dict(measurement_manifest)
    )
    gt_rows = _load_camera_gt_rows(resolved)
    mode_results: dict[str, dict[str, Any]] = {}
    calibration_rows: list[dict[str, Any]] = []
    for mode, path_key in (("anchor_only", "anchor_only_jsonl"), ("multitag", "multitag_jsonl")):
        packets = []
        path = Path(measurement_manifest["paths"][path_key])
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        packets.append(json.loads(line))
        estimated_rotations: list[np.ndarray] = []
        gt_rotations: list[np.ndarray] = []
        for packet in packets:
            gt_row = _match_gt_row(
                gt_rows,
                frame_index=None if packet.get("frame_index") in (None, "") else int(packet["frame_index"]),
                timestamp_s=float(packet["timestamp_s"]),
            )
            if gt_row is None:
                continue
            _gt_position, gt_rotation = _gt_position_and_rotation(gt_row)
            if gt_rotation is None:
                continue
            estimated_rotations.append(np.asarray(packet["rotation_wi"], dtype=np.float64))
            gt_rotations.append(np.asarray(gt_rotation, dtype=np.float64))
        rotation_alignment = _rotation_average(
            [
                np.asarray(estimated_rotation, dtype=np.float64).reshape(3, 3).T
                @ np.asarray(gt_rotation, dtype=np.float64).reshape(3, 3)
                for estimated_rotation, gt_rotation in zip(estimated_rotations, gt_rotations)
            ],
            [1.0 for _ in estimated_rotations],
        ) if estimated_rotations and len(estimated_rotations) == len(gt_rotations) else np.eye(3, dtype=np.float64)
        residual_vectors: list[np.ndarray] = []
        position_errors_m: list[float] = []
        rotation_errors_deg: list[float] = []
        for packet in packets:
            gt_row = _match_gt_row(
                gt_rows,
                frame_index=None if packet.get("frame_index") in (None, "") else int(packet["frame_index"]),
                timestamp_s=float(packet["timestamp_s"]),
            )
            if gt_row is None:
                continue
            gt_position, gt_rotation = _gt_position_and_rotation(gt_row)
            if gt_position is None or gt_rotation is None:
                continue
            estimated_position = np.asarray(packet["position_world_m"], dtype=np.float64)
            estimated_rotation = np.asarray(packet["rotation_wi"], dtype=np.float64) @ rotation_alignment
            position_residual = estimated_position - gt_position
            delta_rotation = gt_rotation.T @ estimated_rotation
            rotvec, _ = cv2.Rodrigues(delta_rotation)
            rotation_residual = np.asarray(rotvec, dtype=np.float64).reshape(3)
            residual_vectors.append(np.concatenate((rotation_residual, position_residual), axis=0))
            position_errors_m.append(float(np.linalg.norm(position_residual)))
            rotation_errors_deg.append(relative_rotation_error_deg(estimated_rotation, gt_rotation))
        if residual_vectors:
            residual_array = np.asarray(residual_vectors, dtype=np.float64)
            residual_covariance = sanitize_covariance(np.cov(residual_array.T))
            recommended_rotation_std_deg = [
                float(np.degrees(math.sqrt(max(residual_covariance[index, index], 0.0))))
                for index in range(3)
            ]
            recommended_position_std_m = [
                float(math.sqrt(max(residual_covariance[3 + index, 3 + index], 0.0)))
                for index in range(3)
            ]
        else:
            residual_covariance = np.eye(6, dtype=np.float64) * 1e-4
            recommended_rotation_std_deg = [0.0, 0.0, 0.0]
            recommended_position_std_m = [0.0, 0.0, 0.0]
        mode_results[mode] = {
            "sample_count": int(len(position_errors_m)),
            "mean_position_error_m": None if not position_errors_m else float(np.mean(np.asarray(position_errors_m, dtype=np.float64))),
            "p95_position_error_m": _safe_percentile(position_errors_m, 95.0),
            "mean_rotation_error_deg": None if not rotation_errors_deg else float(np.mean(np.asarray(rotation_errors_deg, dtype=np.float64))),
            "p95_rotation_error_deg": _safe_percentile(rotation_errors_deg, 95.0),
            "recommended_rotation_std_deg": recommended_rotation_std_deg,
            "recommended_position_std_m": recommended_position_std_m,
            "gt_alignment_rotation_wi": [[float(value) for value in row] for row in rotation_alignment.tolist()],
            "gt_alignment_angle_deg": relative_rotation_error_deg(rotation_alignment, np.eye(3, dtype=np.float64)),
            "residual_covariance_6x6": [[float(value) for value in row] for row in residual_covariance.tolist()],
        }
        calibration_rows.append(
            {
                "mode": mode,
                "sample_count": mode_results[mode]["sample_count"],
                "mean_position_error_m": mode_results[mode]["mean_position_error_m"],
                "p95_position_error_m": mode_results[mode]["p95_position_error_m"],
                "mean_rotation_error_deg": mode_results[mode]["mean_rotation_error_deg"],
                "p95_rotation_error_deg": mode_results[mode]["p95_rotation_error_deg"],
            }
        )
    payload = {
        "run_dir": str(resolved),
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "measurement_manifest_path": str(Path(measurement_manifest["manifest_path"]).resolve()),
        "by_mode": mode_results,
    }
    _write_json(calibration_json_path, payload)
    _write_csv(
        calibration_table_path,
        ["mode", "sample_count", "mean_position_error_m", "p95_position_error_m", "mean_rotation_error_deg", "p95_rotation_error_deg"],
        calibration_rows,
    )
    payload["calibration_json_path"] = str(calibration_json_path.resolve())
    payload["calibration_table_path"] = str(calibration_table_path.resolve())
    return payload


__all__ = [
    "_camera_frame_packet_from_row",
    "calibrate_tag_measurement_noise",
    "export_isaac_tag_pose_measurements",
]
