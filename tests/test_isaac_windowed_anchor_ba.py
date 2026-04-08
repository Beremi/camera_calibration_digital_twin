"""Windowed anchored BA backend regressions."""

from __future__ import annotations

import cv2
import numpy as np

from calib_sim.isaac.estimation.fixed_lag_smoother import FixedLagSmoother
from calib_sim.isaac.estimation.state_defs import FilterStateSnapshot
from calib_sim.isaac.estimation.windowed_anchor_ba import WindowedAnchorBASmoother
from calib_sim.isaac.logging.schemas import IsaacTagDetectionPacket
from calib_sim.isaac.tag_builder import TagPoseSpec


def _projected_corners(tag_position_world_m: np.ndarray, camera_position_world_m: np.ndarray) -> tuple[tuple[float, float], ...]:
    camera_matrix = np.array([[160.0, 0.0, 80.0], [0.0, 160.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    local_points = np.array(
        [
            [-0.05, 0.05, 0.0],
            [0.05, 0.05, 0.0],
            [0.05, -0.05, 0.0],
            [-0.05, -0.05, 0.0],
        ],
        dtype=np.float64,
    )
    world_points = local_points + np.asarray(tag_position_world_m, dtype=np.float64).reshape(1, 3)
    translation_cw = -np.asarray(camera_position_world_m, dtype=np.float64).reshape(3, 1)
    projected, _ = cv2.projectPoints(
        world_points,
        np.zeros((3, 1), dtype=np.float64),
        translation_cw,
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
    )
    return tuple((float(point[0]), float(point[1])) for point in np.asarray(projected, dtype=np.float64).reshape(-1, 2))


def _synthetic_detection(frame_index: int, timestamp_s: float, tag_id: int, *, is_anchor: bool, tag_position_world_m: np.ndarray, camera_position_world_m: np.ndarray) -> IsaacTagDetectionPacket:
    local_points = (
        (-0.05, 0.05, 0.0),
        (0.05, 0.05, 0.0),
        (0.05, -0.05, 0.0),
        (-0.05, -0.05, 0.0),
    )
    tag_translation = np.asarray(tag_position_world_m, dtype=np.float64) - np.asarray(camera_position_world_m, dtype=np.float64)
    return IsaacTagDetectionPacket(
        timestamp_s=timestamp_s,
        sim_time_s=timestamp_s,
        frame_index=frame_index,
        tag_id=tag_id,
        family="apriltag36h11",
        tag_size_m=0.10,
        pnp_tag_size_m=0.10,
        corners_xy=_projected_corners(tag_position_world_m, camera_position_world_m),
        local_tag_points_m=local_points,
        corner_order="clockwise_top_left_first",
        score=1.0,
        is_anchor=is_anchor,
        pose_camera_rvec=(0.0, 0.0, 0.0),
        pose_camera_tvec_m=tuple(float(value) for value in tag_translation.tolist()),
        detector_backend="opencv_aruco_apriltag36h11",
        visibility_flags={"detected": True, "native_backend": True, "pose_ready": True},
    )


def test_windowed_anchor_ba_improves_auxiliary_tag_estimate_over_lightweight() -> None:
    anchor_pose = TagPoseSpec(
        tag_id=0,
        size_m=0.10,
        position_world_m=(0.0, 0.0, 1.0),
        rotation_wt=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        is_anchor=True,
    )
    aux_initial = TagPoseSpec(
        tag_id=42,
        size_m=0.10,
        position_world_m=(0.50, 0.0, 1.0),
        rotation_wt=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        is_anchor=False,
    )
    tag_map = {0: anchor_pose, 42: aux_initial}
    true_aux_position = np.array([0.2, 0.0, 1.0], dtype=np.float64)
    intrinsics_snapshot = {
        "fx_px": 160.0,
        "fy_px": 160.0,
        "cx_px": 80.0,
        "cy_px": 60.0,
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    true_camera_positions = [np.array([0.0, 0.0, 0.0], dtype=np.float64), np.array([0.1, 0.0, 0.0], dtype=np.float64)]
    biased_camera_positions = [position + np.array([0.3, 0.0, 0.0], dtype=np.float64) for position in true_camera_positions]

    lightweight = FixedLagSmoother(lag_size=4, use_auxiliary_tags=True)
    windowed = WindowedAnchorBASmoother(lag_size=4, use_auxiliary_tags=True)
    for frame_index, (true_position, biased_position) in enumerate(zip(true_camera_positions, biased_camera_positions)):
        snapshot = FilterStateSnapshot(
            timestamp_s=0.1 * (frame_index + 1),
            sim_time_s=0.1 * (frame_index + 1),
            rotation_wi=np.eye(3, dtype=np.float64),
            position_world_m=biased_position.copy(),
            velocity_world_mps=np.zeros(3, dtype=np.float64),
            gyro_bias_rps=np.zeros(3, dtype=np.float64),
            accel_bias_mps2=np.zeros(3, dtype=np.float64),
            covariance=np.eye(15, dtype=np.float64) * 1e-3,
            mode="visual",
        )
        detections = (
            _synthetic_detection(frame_index, snapshot.timestamp_s, 0, is_anchor=True, tag_position_world_m=np.array(anchor_pose.position_world_m), camera_position_world_m=true_position),
            _synthetic_detection(frame_index, snapshot.timestamp_s, 42, is_anchor=False, tag_position_world_m=true_aux_position, camera_position_world_m=true_position),
        )
        lightweight.push_snapshot(snapshot)
        lightweight.observe_auxiliary_detections(detections=detections, current_state=snapshot, anchor_pose_map=tag_map)
        windowed.push_snapshot(snapshot)
        windowed.observe_imu_interval(imu_packets=())
        windowed.observe_auxiliary_detections(
            detections=detections,
            current_state=snapshot,
            anchor_pose_map=tag_map,
            intrinsics_snapshot=intrinsics_snapshot,
        )

    lightweight_estimate = lightweight.active_tag_estimates(min_observation_count=1)[42]
    windowed_snapshot = windowed.solve()
    windowed_estimate = windowed.active_tag_estimates(min_observation_count=1)[42]
    lightweight_error = float(np.linalg.norm(lightweight_estimate.pose_wt[:3, 3] - true_aux_position))
    windowed_error = float(np.linalg.norm(windowed_estimate.pose_wt[:3, 3] - true_aux_position))

    assert windowed_snapshot.diagnostics["backend"] == "windowed_ba"
    assert windowed_snapshot.diagnostics["anchored_window"] is True
    assert windowed_error < lightweight_error
