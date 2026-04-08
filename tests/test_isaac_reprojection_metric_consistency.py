"""Direct reprojection consistency regression for estimator-quality reporting."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from calib_sim.reporting import compute_isaac_estimator_quality

from tests._isaac_test_helpers import make_estimator_quality_isaac_run


def test_reprojection_metric_matches_direct_pixel_computation(tmp_path: Path) -> None:
    run_dir = make_estimator_quality_isaac_run(tmp_path)
    quality = compute_isaac_estimator_quality(run_dir)

    camera_frames = [
        json.loads(line)
        for line in (run_dir / "raw" / "camera_frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    detections = [
        json.loads(line)
        for line in (run_dir / "raw" / "detections.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    filter_rows = [
        json.loads(line)
        for line in (run_dir / "estimates" / "filter_state.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tag_gt = json.loads((run_dir / "gt" / "tag_gt.json").read_text(encoding="utf-8"))

    anchor_detection = next(row for row in detections if int(row["tag_id"]) == 0)
    frame = next(row for row in camera_frames if int(row["frame_index"]) == int(anchor_detection["frame_index"]))
    filter_row = next(row for row in filter_rows if abs(float(row["timestamp_s"]) - float(anchor_detection["timestamp_s"])) < 1e-9)
    anchor_gt = next(tag for tag in tag_gt["tags"] if int(tag["tag_id"]) == 0)

    local_points = np.asarray(anchor_detection["local_tag_points_m"], dtype=np.float64)
    observed_corners = np.asarray(anchor_detection["corners_xy"], dtype=np.float64)
    rotation_wt = np.asarray(anchor_gt["rotation_wt"], dtype=np.float64)
    position_wt = np.asarray(anchor_gt["position_world_m"], dtype=np.float64).reshape(3)
    world_points = (rotation_wt @ local_points.T).T + position_wt.reshape(1, 3)

    intrinsics = frame["intrinsics_snapshot"]
    camera_matrix = np.array(
        [
            [float(intrinsics["fx_px"]), 0.0, float(intrinsics["cx_px"])],
            [0.0, float(intrinsics["fy_px"]), float(intrinsics["cy_px"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.asarray(intrinsics["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    rotation_wc = np.asarray(filter_row["rotation_wi"], dtype=np.float64)
    rotation_cw = rotation_wc.T
    translation_cw = -rotation_cw @ np.asarray(filter_row["position_world_m"], dtype=np.float64).reshape(3)
    rvec_cw, _ = cv2.Rodrigues(rotation_cw)
    projected_points, _ = cv2.projectPoints(
        world_points,
        rvec_cw,
        translation_cw.reshape(3, 1),
        camera_matrix,
        dist_coeffs,
    )
    projected_corners = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
    direct_rmse_px = float(np.sqrt(np.mean(np.sum((projected_corners - observed_corners) ** 2, axis=1))))

    assert direct_rmse_px < 1e-6
    assert quality["summary"]["anchor_mean_reprojection_rmse_px"] is not None
    assert abs(float(quality["summary"]["anchor_mean_reprojection_rmse_px"]) - direct_rmse_px) < 1e-6

