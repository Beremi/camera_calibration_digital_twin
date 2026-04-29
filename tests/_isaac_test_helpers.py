"""Small Isaac run fixtures for the slim tabletop test suite."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from calib_sim.isaac.estimation.state_defs import FilterStateSnapshot, SmootherStateSnapshot, UncertaintySnapshot
from calib_sim.isaac.logging.run_manifest import build_run_manifest
from calib_sim.isaac.logging.schemas import (
    IsaacCameraFramePacket,
    IsaacControllerDiagnosticPacket,
    IsaacImuPacket,
    IsaacJointCommandPacket,
    IsaacRealizedJointPacket,
    IsaacTagDetectionPacket,
)
from calib_sim.isaac.logging.writer import IsaacRunWriter


def _write_placeholder_png(path: Path) -> None:
    image = np.full((24, 24, 3), 245, dtype=np.uint8)
    cv2.putText(image, "T", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def make_minimal_isaac_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "isaac_runs" / "test_run"
    writer = IsaacRunWriter(run_dir)
    writer.write_config_snapshot("scene", {"stage_path": "assets/isaac/anchor_room.usd", "anchor_tag_id": 0})
    manifest = build_run_manifest(
        repo_root=Path(__file__).resolve().parents[1],
        run_id="test_run",
        isaac_sim_version="test",
        stage_usd_path="assets/isaac/anchor_room.usd",
        robot_preset="ur5e_phone_head",
        anchor_tag_id=0,
        estimator_mode="fused",
        controller_mode="closed-loop",
        bootstrap_control_policy="hold_until_first_detection",
        noise_presets={"imu": "phone_nominal", "actuation": "servo_nominal"},
        random_seed=7,
        controller_config={
            "name": "path_tracking",
            "waypoints": [{"position_world_m": [0.0, 0.0, 0.0], "tolerance_m": 0.05}],
        },
        estimator_config={"name": "anchored_vio"},
        ros2_bridge_used=False,
    )
    writer.write_manifest(manifest)
    writer.append_camera_frame(
        IsaacCameraFramePacket(
            frame_index=0,
            timestamp_s=0.1,
            sim_time_s=0.1,
            sensor_time_s=0.1,
            host_time_s=1.0,
            rgb_path="raw/rgb/frame_0000.png",
            intrinsics_snapshot={"fx_px": 100.0, "fy_px": 100.0, "cx_px": 50.0, "cy_px": 60.0},
            extrinsics_snapshot={"frame": "C"},
            visible_gt_tag_ids=(0, 42),
        )
    )
    writer.append_detection(
        IsaacTagDetectionPacket(
            timestamp_s=0.1,
            sim_time_s=0.1,
            frame_index=0,
            tag_id=0,
            family="apriltag36h11",
            tag_size_m=0.10,
            corners_xy=((45.0, 55.0), (55.0, 55.0), (55.0, 65.0), (45.0, 65.0)),
            local_tag_points_m=(
                (-0.05, -0.05, 0.0),
                (0.05, -0.05, 0.0),
                (0.05, 0.05, 0.0),
                (-0.05, 0.05, 0.0),
            ),
            corner_order="clockwise_top_left_first",
            score=1.0,
            is_anchor=True,
            visibility_flags={"detected": True},
        )
    )
    writer.append_estimator_input({"timestamp_s": 0.1, "kind": "frame_pack", "frame_index": 0})
    for packet_index, timestamp_s in enumerate((0.10, 0.11)):
        writer.append_imu(
            IsaacImuPacket(
                packet_index=packet_index,
                timestamp_s=timestamp_s,
                sim_time_s=timestamp_s,
                dt_s=0.01,
                wx=0.0,
                wy=0.0,
                wz=0.0,
                ax=0.0,
                ay=0.0,
                az=9.81,
                imu_frame="I",
                imu_semantics="specific_force",
                noise_preset="phone_nominal",
            )
        )
    writer.append_command(
        IsaacJointCommandPacket(
            timestamp_s=0.10,
            sim_time_s=0.10,
            joint_names=("joint0",),
            desired_positions=(0.2,),
            effective_positions=(0.19,),
            estimator_mode="fused",
            controller_mode="closed-loop",
            waypoint_index=0,
            safety_reason="nominal",
            tracking_error_world_m=(0.0, 0.0, 0.0),
            dropped_command=False,
        )
    )
    writer.append_controller_diagnostic(
        IsaacControllerDiagnosticPacket(
            timestamp_s=0.10,
            sim_time_s=0.10,
            estimator_mode="fused",
            controller_mode="closed-loop",
            waypoint_index=0,
            current_position_source="estimate",
            tracking_error_norm_m=0.0,
            command_delta_norm_m=0.01,
            desired_position_world_m=(0.0, 0.0, 0.0),
            safety_reason="nominal",
            position_radius_95_m=0.01,
            innovation_norm=0.0,
            ik_success=True,
            ik_retry_alpha=1.0,
            joint_target_delta_norm=0.01,
            min_joint_limit_margin=0.5,
            dropped_command=False,
        )
    )
    writer.append_realized_joint(
        IsaacRealizedJointPacket(
            timestamp_s=0.10,
            sim_time_s=0.10,
            joint_names=("joint0",),
            positions=(0.18,),
            velocities=(0.05,),
            end_effector_position_world_m=(0.0, 0.0, 0.0),
            end_effector_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            servo_internal_state={"joint0": {"effective_command": 0.19}},
        )
    )
    writer.append_gt_camera({"timestamp_s": 0.10, "sim_time_s": 0.10, "px": 0.0, "py": 0.0, "pz": 0.0})
    writer.append_gt_imu({"timestamp_s": 0.10, "sim_time_s": 0.10, "ax": 0.0, "ay": 0.0, "az": 9.81})
    writer.append_gt_joint({"timestamp_s": 0.10, "sim_time_s": 0.10, "joint0": 0.18})
    writer.write_tag_gt({"anchor_tag_id": 0, "tags": [{"tag_id": 0, "is_anchor": True}, {"tag_id": 42, "is_anchor": False}]})
    writer.append_filter_state(
        FilterStateSnapshot(
            timestamp_s=0.11,
            sim_time_s=0.11,
            rotation_wi=np.eye(3, dtype=np.float64),
            position_world_m=np.zeros(3, dtype=np.float64),
            velocity_world_mps=np.zeros(3, dtype=np.float64),
            gyro_bias_rps=np.zeros(3, dtype=np.float64),
            accel_bias_mps2=np.zeros(3, dtype=np.float64),
            covariance=np.eye(15, dtype=np.float64) * 1e-3,
            innovation_diagnostics={"last_innovation_norm": 0.0},
        )
    )
    writer.append_smoother_state(
        SmootherStateSnapshot(
            timestamp_s=0.11,
            sim_time_s=0.11,
            active_tag_poses={42: np.eye(4, dtype=np.float64)},
            cloned_positions_world_m=(np.zeros(3, dtype=np.float64),),
            covariance=np.eye(6, dtype=np.float64) * 1e-3,
        )
    )
    writer.append_uncertainty(
        UncertaintySnapshot(
            timestamp_s=0.11,
            sim_time_s=0.11,
            pose_covariance=np.eye(3, dtype=np.float64) * 1e-3,
            velocity_covariance=np.eye(3, dtype=np.float64) * 1e-2,
            bias_covariance=np.eye(6, dtype=np.float64) * 1e-4,
            position_radius_95_m=0.01,
            velocity_radius_95_mps=0.1,
            diagnostics={"condition_number": 1.0},
        )
    )
    return run_dir


def make_estimator_quality_isaac_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "isaac_runs" / "quality_run"
    writer = IsaacRunWriter(run_dir)
    writer.write_config_snapshot(
        "scene",
        {
            "name": "anchor_room_test_scene",
            "stage_path": "programmatic:anchor_room",
            "anchor_tag_id": 0,
            "robot_prim_path": "/World/Robot",
            "camera_prim_path": "/World/Robot/phone_camera",
            "imu_prim_path": "/World/Robot/phone_imu",
            "physics_rate_hz": 240.0,
            "tags": [
                {"tag_id": 0, "prim_path": "/World/Tags/tag_0", "size_m": 0.10, "position_world_m": [0.0, 0.0, 1.0], "orientation_rpy_deg": [0.0, 0.0, 0.0], "is_anchor": True},
                {"tag_id": 42, "prim_path": "/World/Tags/tag_42", "size_m": 0.10, "position_world_m": [0.2, 0.0, 1.0], "orientation_rpy_deg": [0.0, 0.0, 0.0], "is_anchor": False},
                {"tag_id": 43, "prim_path": "/World/Tags/tag_43", "size_m": 0.10, "position_world_m": [-0.2, 0.0, 1.0], "orientation_rpy_deg": [0.0, 0.0, 0.0], "is_anchor": False},
            ],
        },
    )
    writer.write_config_snapshot(
        "camera",
        {
            "name": "phone_main_test",
            "source_toml_path": "config/camera/pixel_9a_main.toml",
            "width_px": 100,
            "height_px": 120,
            "rate_hz": 10.0,
            "frame_id": "C",
            "intrinsics": {"fx_px": 100.0, "fy_px": 100.0, "cx_px": 50.0, "cy_px": 60.0},
            "distortion_model": "plumb_bob",
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )
    writer.write_config_snapshot(
        "imu",
        {
            "name": "phone_nominal_test",
            "rate_hz": 100.0,
            "frame_id": "I",
            "imu_semantics": "specific_force",
            "noise_preset": "imu_nominal_phone",
            "local_translation_m": [0.0, 0.0, 0.0],
            "local_orientation_rpy_deg": [0.0, 0.0, 0.0],
        },
    )
    writer.write_config_snapshot("actuation", {"name": "servo_nominal"})
    writer.write_config_snapshot(
        "estimation",
        {
            "filter": {
                "use_aux_tags_in_filter": True,
                "vision_covariance_scale": 1.0,
                "imu_process_covariance_scale": 1.0,
                "post_relocalization_covariance_scale": 1.0,
            },
            "smoother": {
                "use_aux_tags_in_smoother": True,
                "min_feedback_observation_count": 2,
                "max_feedback_correction_norm_m": 0.10,
            },
        },
    )
    writer.write_config_snapshot("control", {"use_aux_map_for_control": True, "waypoints": []})
    writer.write_manifest(
        build_run_manifest(
            repo_root=Path(__file__).resolve().parents[1],
            run_id="quality_run",
            isaac_sim_version="test",
            stage_usd_path="programmatic:anchor_room",
            robot_preset="franka_phone_head",
            anchor_tag_id=0,
            estimator_mode="fused",
            controller_mode="closed-loop",
            bootstrap_control_policy="hold_until_first_detection",
            noise_presets={"imu": "phone_nominal", "actuation": "servo_nominal"},
            random_seed=7,
            controller_config={"name": "path_tracking", "waypoints": []},
            estimator_config={"name": "anchored_vio"},
            ros2_bridge_used=False,
        )
    )

    rgb_dir = run_dir / "raw" / "rgb"
    _write_placeholder_png(rgb_dir / "frame_000000.png")
    _write_placeholder_png(rgb_dir / "frame_000001.png")

    camera_matrix = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    local_points = np.array(
        [[-0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.05, -0.05, 0.0], [-0.05, -0.05, 0.0]],
        dtype=np.float64,
    )
    tag_poses = {
        0: np.array([0.0, 0.0, 1.0], dtype=np.float64),
        42: np.array([0.2, 0.0, 1.0], dtype=np.float64),
        43: np.array([-0.2, 0.0, 1.0], dtype=np.float64),
    }

    def projected_corners(tag_position_world_m: np.ndarray) -> tuple[tuple[float, float], ...]:
        world_points = local_points + tag_position_world_m.reshape(1, 3)
        projected, _ = cv2.projectPoints(
            world_points,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            camera_matrix,
            dist_coeffs,
        )
        return tuple((float(point[0]), float(point[1])) for point in np.asarray(projected, dtype=np.float64).reshape(-1, 2))

    for frame_packet in (
        IsaacCameraFramePacket(
            frame_index=0,
            timestamp_s=0.10,
            sim_time_s=0.10,
            sensor_time_s=0.10,
            host_time_s=1.0,
            rgb_path="raw/rgb/frame_000000.png",
            intrinsics_snapshot={"fx_px": 100.0, "fy_px": 100.0, "cx_px": 50.0, "cy_px": 60.0, "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0]},
            extrinsics_snapshot={"position_world_m": [0.0, 0.0, 0.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            image_width_px=100,
            image_height_px=120,
            visible_gt_tag_ids=(0, 42, 43),
        ),
        IsaacCameraFramePacket(
            frame_index=1,
            timestamp_s=0.20,
            sim_time_s=0.20,
            sensor_time_s=0.20,
            host_time_s=1.1,
            rgb_path="raw/rgb/frame_000001.png",
            intrinsics_snapshot={"fx_px": 100.0, "fy_px": 100.0, "cx_px": 50.0, "cy_px": 60.0, "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0]},
            extrinsics_snapshot={"position_world_m": [0.0, 0.0, 0.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            image_width_px=100,
            image_height_px=120,
            visible_gt_tag_ids=(0, 42),
        ),
    ):
        writer.write_camera_frame(frame_packet)

    for frame_index, timestamp_s in ((0, 0.10), (1, 0.20)):
        writer.write_detection(
            IsaacTagDetectionPacket(
                timestamp_s=timestamp_s,
                sim_time_s=timestamp_s,
                frame_index=frame_index,
                tag_id=0,
                family="apriltag36h11",
                tag_size_m=0.10,
                pnp_tag_size_m=0.10,
                corners_xy=projected_corners(tag_poses[0]),
                local_tag_points_m=tuple(tuple(float(value) for value in row) for row in local_points.tolist()),
                corner_order="clockwise_top_left_first",
                score=1.0,
                is_anchor=True,
                pose_camera_rvec=(0.0, 0.0, 0.0),
                pose_camera_tvec_m=tuple(float(value) for value in tag_poses[0].tolist()),
                detector_backend="pupil_apriltags_apriltag36h11",
                visibility_flags={"detected": True, "native_backend": True, "pose_ready": True},
            )
        )
        writer.write_detection(
            IsaacTagDetectionPacket(
                timestamp_s=timestamp_s,
                sim_time_s=timestamp_s,
                frame_index=frame_index,
                tag_id=42,
                family="apriltag36h11",
                tag_size_m=0.10,
                pnp_tag_size_m=0.10,
                corners_xy=projected_corners(tag_poses[42]),
                local_tag_points_m=tuple(tuple(float(value) for value in row) for row in local_points.tolist()),
                corner_order="clockwise_top_left_first",
                score=0.9,
                is_anchor=False,
                pose_camera_rvec=(0.0, 0.0, 0.0),
                pose_camera_tvec_m=tuple(float(value) for value in tag_poses[42].tolist()),
                detector_backend="pupil_apriltags_apriltag36h11",
                visibility_flags={"detected": True, "native_backend": True, "pose_ready": True},
            )
        )
    writer.write_detection(
        IsaacTagDetectionPacket(
            timestamp_s=0.10,
            sim_time_s=0.10,
            frame_index=0,
            tag_id=43,
            family="apriltag36h11",
            tag_size_m=0.10,
            corners_xy=projected_corners(tag_poses[43]),
            local_tag_points_m=tuple(tuple(float(value) for value in row) for row in local_points.tolist()),
            corner_order="clockwise_top_left_first",
            score=0.4,
            is_anchor=False,
            detector_backend="pupil_apriltags_temporal_track",
            visibility_flags={"detected": True, "native_backend": False, "pose_ready": False},
        )
    )
    writer.write_estimator_input(
        {
            "timestamp_s": 0.10,
            "kind": "frame_pack",
            "frame_index": 0,
            "anchor_visible": True,
            "detected_tag_ids": [0, 42, 43],
            "auxiliary_visible_count": 2,
            "native_auxiliary_pose_ready_count": 1,
            "accepted_auxiliary_update_count": 1,
            "rejected_auxiliary_update_count": 1,
            "accepted_auxiliary_tag_ids": [42],
            "rejected_auxiliary_tag_ids": [43],
            "auxiliary_rejection_reasons": {"fallback_only": 1},
            "auxiliary_update_decisions": [
                {"tag_id": 42, "accepted": True, "reason": "accepted"},
                {"tag_id": 43, "accepted": False, "reason": "fallback_only"},
            ],
        }
    )
    writer.write_estimator_input(
        {
            "timestamp_s": 0.20,
            "kind": "frame_pack",
            "frame_index": 1,
            "anchor_visible": True,
            "detected_tag_ids": [0, 42],
            "auxiliary_visible_count": 1,
            "native_auxiliary_pose_ready_count": 1,
            "accepted_auxiliary_update_count": 1,
            "rejected_auxiliary_update_count": 0,
            "accepted_auxiliary_tag_ids": [42],
            "rejected_auxiliary_tag_ids": [],
            "auxiliary_rejection_reasons": {},
            "auxiliary_update_decisions": [{"tag_id": 42, "accepted": True, "reason": "accepted"}],
        }
    )
    for packet_index, timestamp_s in enumerate((0.10, 0.11, 0.20, 0.21)):
        writer.write_imu_packet(
            IsaacImuPacket(
                packet_index=packet_index,
                timestamp_s=timestamp_s,
                sim_time_s=timestamp_s,
                dt_s=0.01,
                wx=0.0,
                wy=0.0,
                wz=0.0,
                ax=0.0,
                ay=0.0,
                az=9.81,
                imu_frame="I",
                imu_semantics="specific_force",
                noise_preset="phone_nominal",
            )
        )
    writer.write_command(
        IsaacJointCommandPacket(
            timestamp_s=0.10,
            sim_time_s=0.10,
            joint_names=("joint0",),
            desired_positions=(0.2,),
            effective_positions=(0.19,),
            estimator_mode="fused",
            controller_mode="closed-loop",
            waypoint_index=0,
            safety_reason="nominal",
            tracking_error_world_m=(0.0, 0.0, 0.0),
            dropped_command=False,
        )
    )
    writer.write_controller_diagnostic(
        IsaacControllerDiagnosticPacket(
            timestamp_s=0.10,
            sim_time_s=0.10,
            estimator_mode="fused",
            controller_mode="closed-loop",
            waypoint_index=0,
            current_position_source="estimate",
            tracking_error_norm_m=0.0,
            command_delta_norm_m=0.01,
            desired_position_world_m=(0.0, 0.0, 0.0),
            safety_reason="nominal",
            position_radius_95_m=0.02,
            innovation_norm=0.05,
            ik_success=True,
            ik_retry_alpha=1.0,
            joint_target_delta_norm=0.01,
            min_joint_limit_margin=0.5,
            dropped_command=False,
        )
    )
    writer.write_realized_state(
        IsaacRealizedJointPacket(
            timestamp_s=0.10,
            sim_time_s=0.10,
            joint_names=("joint0",),
            positions=(0.18,),
            velocities=(0.05,),
            end_effector_position_world_m=(0.0, 0.0, 0.0),
            end_effector_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            servo_internal_state={"joint0": {"effective_command": 0.19}},
        )
    )
    for timestamp_s in (0.10, 0.20):
        writer.write_gt(
            namespace="camera",
            payload={"timestamp_s": timestamp_s, "sim_time_s": timestamp_s, "px": 0.0, "py": 0.0, "pz": 0.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
        )
    writer.write_gt(namespace="imu", payload={"timestamp_s": 0.10, "sim_time_s": 0.10, "ax": 0.0, "ay": 0.0, "az": 9.81})
    writer.write_gt(namespace="joint", payload={"timestamp_s": 0.10, "sim_time_s": 0.10, "joint0": 0.18})
    writer.write_tag_gt(
        {
            "anchor_tag_id": 0,
            "tags": [
                {"tag_id": 0, "is_anchor": True, "size_m": 0.10, "position_world_m": [0.0, 0.0, 1.0], "rotation_wt": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
                {"tag_id": 42, "is_anchor": False, "size_m": 0.10, "position_world_m": [0.2, 0.0, 1.0], "rotation_wt": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
                {"tag_id": 43, "is_anchor": False, "size_m": 0.10, "position_world_m": [-0.2, 0.0, 1.0], "rotation_wt": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
            ],
        }
    )
    covariance = np.eye(15, dtype=np.float64) * 1e-3
    covariance[3:6, 3:6] = np.eye(3, dtype=np.float64) * 1e-4
    for timestamp_s, innovation_norm, relocs in ((0.10, 0.05, 0.0), (0.20, 0.02, 1.0)):
        writer.write_filter_state(
            FilterStateSnapshot(
                timestamp_s=timestamp_s,
                sim_time_s=timestamp_s,
                rotation_wi=np.eye(3, dtype=np.float64),
                position_world_m=np.zeros(3, dtype=np.float64),
                velocity_world_mps=np.zeros(3, dtype=np.float64),
                gyro_bias_rps=np.zeros(3, dtype=np.float64),
                accel_bias_mps2=np.zeros(3, dtype=np.float64),
                covariance=covariance.copy(),
                innovation_diagnostics={
                    "last_innovation_norm": innovation_norm,
                    "anchor_relocalizations": relocs,
                    "auxiliary_update_count": 2.0 if timestamp_s > 0.1 else 1.0,
                    "auxiliary_rejection_count": 1.0,
                },
                auxiliary_rejection_reason_counts={"fallback_only": 1},
            )
        )
    for timestamp_s, correction_norm in ((0.10, 0.02), (0.20, 0.01)):
        writer.write_smoother_state(
            SmootherStateSnapshot(
                timestamp_s=timestamp_s,
                sim_time_s=timestamp_s,
                active_tag_poses={42: np.eye(4, dtype=np.float64)},
                cloned_positions_world_m=(np.zeros(3, dtype=np.float64),),
                covariance=np.eye(6, dtype=np.float64) * 1e-3,
                diagnostics={"feedback_correction_norm_m": correction_norm, "trusted_feedback_count": 1.0, "total_feedback_candidates": 1.0},
            )
        )
    for timestamp_s in (0.10, 0.20):
        writer.write_uncertainty(
            UncertaintySnapshot(
                timestamp_s=timestamp_s,
                sim_time_s=timestamp_s,
                pose_covariance=np.eye(3, dtype=np.float64) * 1e-4,
                velocity_covariance=np.eye(3, dtype=np.float64) * 1e-3,
                bias_covariance=np.eye(6, dtype=np.float64) * 1e-4,
                position_radius_95_m=0.03,
                velocity_radius_95_mps=0.1,
                diagnostics={"condition_number": 1.0},
            )
        )
    return run_dir


__all__ = ["make_estimator_quality_isaac_run", "make_minimal_isaac_run"]
