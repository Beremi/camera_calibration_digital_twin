"""Auxiliary-tag quality gates for the second-pass filter."""

from __future__ import annotations

import numpy as np

from calib_sim.isaac.estimation.online_filter import AnchoredOnlineFilter
from calib_sim.isaac.logging.schemas import IsaacTagDetectionPacket
from calib_sim.isaac.tag_builder import TagPoseSpec


def _tag_pose(tag_id: int = 42) -> TagPoseSpec:
    return TagPoseSpec(
        tag_id=tag_id,
        size_m=0.10,
        position_world_m=(0.0, 0.0, 1.0),
        rotation_wt=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        is_anchor=False,
    )


def _native_detection(
    *,
    tag_id: int = 42,
    corners_xy: tuple[tuple[float, float], ...] = ((40.0, 50.0), (60.0, 50.0), (60.0, 70.0), (40.0, 70.0)),
    pose_camera_tvec_m: tuple[float, float, float] = (0.0, 0.0, 1.0),
    detector_backend: str = "opencv_aruco_apriltag36h11",
    pose_ready: bool = True,
    native_backend: bool = True,
) -> IsaacTagDetectionPacket:
    return IsaacTagDetectionPacket(
        timestamp_s=0.10,
        sim_time_s=0.10,
        frame_index=0,
        tag_id=tag_id,
        family="apriltag36h11",
        tag_size_m=0.10,
        pnp_tag_size_m=0.10,
        corners_xy=corners_xy,
        local_tag_points_m=(
            (-0.05, 0.05, 0.0),
            (0.05, 0.05, 0.0),
            (0.05, -0.05, 0.0),
            (-0.05, -0.05, 0.0),
        ),
        corner_order="clockwise_top_left_first",
        score=0.9,
        is_anchor=False,
        pose_camera_rvec=(0.0, 0.0, 0.0) if pose_ready else None,
        pose_camera_tvec_m=pose_camera_tvec_m if pose_ready else None,
        detector_backend=detector_backend,
        visibility_flags={"detected": True, "native_backend": native_backend, "pose_ready": pose_ready},
    )


def _intrinsics() -> dict[str, object]:
    return {
        "fx_px": 100.0,
        "fy_px": 100.0,
        "cx_px": 50.0,
        "cy_px": 60.0,
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
    }


def test_auxiliary_gating_reason_codes_cover_expected_failures() -> None:
    mapped_tag_poses = {42: _tag_pose()}

    fallback_filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    fallback_summary = fallback_filter.update_aux_tags(
        tag_detections=(_native_detection(detector_backend="opencv_aruco_apriltag36h11_bright_quad_match", pose_ready=False, native_backend=False),),
        mapped_tag_poses=mapped_tag_poses,
        intrinsics_snapshot=_intrinsics(),
        image_width_px=100,
        image_height_px=120,
    )
    assert fallback_summary.rejection_reason_counts["fallback_only"] == 1

    margin_filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    margin_filter.auxiliary_visibility_streaks[42] = 3
    margin_summary = margin_filter.update_aux_tags(
        tag_detections=(_native_detection(corners_xy=((1.0, 1.0), (21.0, 1.0), (21.0, 21.0), (1.0, 21.0))),),
        mapped_tag_poses=mapped_tag_poses,
        intrinsics_snapshot=_intrinsics(),
        image_width_px=100,
        image_height_px=120,
        min_corner_margin_px=12.0,
    )
    assert margin_summary.rejection_reason_counts["corner_margin_fail"] == 1

    streak_filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    streak_summary = streak_filter.update_aux_tags(
        tag_detections=(_native_detection(),),
        mapped_tag_poses=mapped_tag_poses,
        intrinsics_snapshot=_intrinsics(),
        image_width_px=100,
        image_height_px=120,
        min_visibility_streak=2,
    )
    assert streak_summary.rejection_reason_counts["visibility_streak_fail"] == 1

    reproj_filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    reproj_filter.auxiliary_visibility_streaks[42] = 3
    reproj_summary = reproj_filter.update_aux_tags(
        tag_detections=(_native_detection(corners_xy=((80.0, 80.0), (90.0, 80.0), (90.0, 90.0), (80.0, 90.0))),),
        mapped_tag_poses=mapped_tag_poses,
        intrinsics_snapshot=_intrinsics(),
        image_width_px=100,
        image_height_px=120,
        max_reprojection_error_px=2.0,
    )
    assert reproj_summary.rejection_reason_counts["reproj_fail"] == 1

    innovation_filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    innovation_filter.auxiliary_visibility_streaks[42] = 3
    innovation_filter.state.position_world_m = np.zeros(3, dtype=np.float64)
    innovation_summary = innovation_filter.update_aux_tags(
        tag_detections=(_native_detection(pose_camera_tvec_m=(2.0, 0.0, 1.0)),),
        mapped_tag_poses=mapped_tag_poses,
        intrinsics_snapshot=_intrinsics(),
        image_width_px=100,
        image_height_px=120,
        max_innovation_norm=0.1,
    )
    assert innovation_summary.rejection_reason_counts["innovation_fail"] == 1

    covariance_filter = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    covariance_filter.auxiliary_visibility_streaks[42] = 3
    covariance_summary = covariance_filter.update_aux_tags(
        tag_detections=(_native_detection(),),
        mapped_tag_poses=mapped_tag_poses,
        intrinsics_snapshot=_intrinsics(),
        image_width_px=100,
        image_height_px=120,
        tag_feedback_diagnostics={42: {"trusted": False, "feedback_correction_norm_m": 0.5}},
    )
    assert covariance_summary.rejection_reason_counts["covariance_fail"] == 1


def test_auxiliary_disable_switch_removes_auxiliary_updates() -> None:
    filter_model = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    filter_model.use_auxiliary_tag_updates = False
    summary = filter_model.update_aux_tags(
        tag_detections=(_native_detection(),),
        mapped_tag_poses={42: _tag_pose()},
        intrinsics_snapshot=_intrinsics(),
        image_width_px=100,
        image_height_px=120,
    )
    assert summary.visible_count == 1
    assert summary.accepted_count == 0
    assert filter_model.auxiliary_update_count == 0
