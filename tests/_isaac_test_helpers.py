"""Helpers for Isaac scaffold tests."""

from __future__ import annotations

import json
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


def first_pass_run_id(estimator_mode: str, controller_mode: str, actuation_name: str, seed: int) -> str:
    return f"first_pass_{estimator_mode}_{controller_mode}_{actuation_name}_seed_{seed:03d}"


def _write_placeholder_png(path: Path) -> None:
    image = np.full((24, 24, 3), 245, dtype=np.uint8)
    cv2.putText(image, "T", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def make_first_pass_suite_lock(docs_dir: Path) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    lock_path = docs_dir / "first_pass_suite_lock.json"
    lock_payload = {
        "branch_name": "feature/isaac-runtime-first-pass",
        "canonical_run_id": first_pass_run_id("fused", "closed-loop", "servo_nominal", 7),
        "expected_actuation_run_ids": [
            first_pass_run_id("fused", "closed-loop", "none", 7),
            first_pass_run_id("fused", "closed-loop", "servo_nominal", 7),
        ],
        "expected_matrix_run_ids": [
            first_pass_run_id("visual", "open-loop", "servo_nominal", 7),
            first_pass_run_id("visual", "closed-loop", "servo_nominal", 7),
            first_pass_run_id("fused", "open-loop", "servo_nominal", 7),
            first_pass_run_id("fused", "closed-loop", "servo_nominal", 7),
        ],
        "expected_reproducibility": {
            "visual_closed_loop": {
                "run_ids": [first_pass_run_id("visual", "closed-loop", "servo_nominal", seed) for seed in (11, 17, 23, 31, 47)],
                "seeds": [11, 17, 23, 31, 47],
            },
            "fused_closed_loop": {
                "run_ids": [first_pass_run_id("fused", "closed-loop", "servo_nominal", seed) for seed in (11, 17, 23, 31, 47)],
                "seeds": [11, 17, 23, 31, 47],
            },
        },
        "frozen_date": "2026-04-08",
        "matrix_seed": 7,
        "reproducibility_seeds": [11, 17, 23, 31, 47],
    }
    lock_path.write_text(json.dumps(lock_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock_path


def make_fake_first_pass_suite(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = (11, 17, 23, 31, 47)

    def metrics_payload(run_id: str, *, estimator_mode: str, controller_mode: str, actuation_name: str, seed: int) -> dict:
        completion_fraction = 0.98 if controller_mode == "closed-loop" else 0.60
        mean_position_error = 0.02 if estimator_mode == "fused" else 0.04
        mean_waypoint_error = 0.03 if controller_mode == "closed-loop" else 0.08
        ik_failure_fraction = 0.05 if actuation_name == "servo_nominal" else 0.01
        return {
            "run_id": run_id,
            "manifest": {
                "run_id": run_id,
                "stage_usd_path": "programmatic:anchor_room",
                "robot_preset": "franka_phone_head",
                "anchor_tag_id": 0,
                "noise_presets": {"imu": "phone_nominal", "actuation": actuation_name},
                "random_seed": seed,
                "controller_config": {"waypoints": []},
                "estimator_config": {"name": "anchored_vio"},
                "ros2_bridge_used": False,
                "estimator_mode": estimator_mode,
                "controller_mode": controller_mode,
                "bootstrap_control_policy": "hold_until_first_detection",
            },
            "config": {
                "physics_rate_hz": 240.0,
                "imu_rate_hz": 200.0,
                "camera_rate_hz": 30.0,
                "filter_rate_hz": 60.0,
                "smoother_rate_hz": 10.0,
                "controller_rate_hz": 50.0,
                "noise_preset": "phone_nominal",
                "actuation_preset": actuation_name,
                "estimator_mode": estimator_mode,
                "controller_mode": controller_mode,
                "bootstrap_control_policy": "hold_until_first_detection",
            },
            "counts": {
                "camera_frames": 240,
                "detections": 480,
                "imu_packets": 1600,
                "commands": 400,
                "controller_diagnostics": 400,
                "realized_joints": 400,
                "filter_states": 480,
                "smoother_states": 80,
                "uncertainty_states": 480,
                "unique_detected_tags": 3,
                "unique_detected_auxiliary_tags": 2,
            },
            "timing": {"duration_s": 8.0},
            "trajectory": {
                "mean_position_error_m": mean_position_error,
                "p95_position_error_m": mean_position_error * 1.4,
                "max_position_error_m": mean_position_error * 2.0,
                "mean_reprojection_error_px": 0.12 if estimator_mode == "fused" else 0.18,
                "p95_reprojection_error_px": 0.19 if estimator_mode == "fused" else 0.26,
                "map_error_m": 0.003,
            },
            "estimation": {
                "anchor_visible_fraction": 0.95,
                "anchor_relocalization_count": 1.0,
                "mean_anchor_innovation_norm": 0.03,
                "mean_innovation_norm": 0.05,
                "anchor_pnp_success_fraction": 0.97,
                "fallback_only_frame_fraction": 0.01,
            },
            "uncertainty_calibration": {
                "mean_position_radius_95_m": 0.025,
                "empirical_95_coverage_percent": 92.0,
                "pose_nees": 2.6,
                "sigma_error_correlation": 0.71,
            },
            "residuals": {
                "mean_reprojection_rmse_px": 0.12 if estimator_mode == "fused" else 0.18,
                "p95_reprojection_rmse_px": 0.19 if estimator_mode == "fused" else 0.26,
                "mean_anchor_innovation_norm": 0.03,
                "mean_innovation_norm": 0.05,
                "anchor_pnp_success_fraction": 0.97,
                "fallback_only_frame_fraction": 0.01,
            },
            "map_quality": {
                "mean_auxiliary_tag_position_error_m": 0.004,
                "p95_auxiliary_tag_position_error_m": 0.007,
                "auxiliary_tag_count": 2.0,
                "anchor_relocalization_count": 1.0,
                "anchor_visible_fraction": 0.95,
            },
            "control": {
                "mean_waypoint_error_m": mean_waypoint_error,
                "completion_fraction": completion_fraction,
                "ik_failure_fraction": ik_failure_fraction,
                "dominant_safety_reason": "nominal",
                "mean_actuator_tracking_error": 0.004 if actuation_name == "servo_nominal" else 0.0,
            },
        }

    specs = [
        ("visual", "open-loop", "servo_nominal", 7),
        ("visual", "closed-loop", "servo_nominal", 7),
        ("fused", "open-loop", "servo_nominal", 7),
        ("fused", "closed-loop", "servo_nominal", 7),
        *[(estimator_mode, "closed-loop", "servo_nominal", seed) for estimator_mode in ("visual", "fused") for seed in seeds],
        ("fused", "closed-loop", "none", 7),
    ]
    for estimator_mode, controller_mode, actuation_name, seed in specs:
        run_id = first_pass_run_id(estimator_mode, controller_mode, actuation_name, seed)
        run_dir = output_root / run_id
        report_dir = run_dir / "analysis" / "report_data"
        report_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "analysis" / "metrics.json").write_text(
            json.dumps(
                metrics_payload(
                    run_id,
                    estimator_mode=estimator_mode,
                    controller_mode=controller_mode,
                    actuation_name=actuation_name,
                    seed=seed,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for figure_name in (
            "system_architecture.png",
            "timing_timeline.png",
            "trajectory_path.png",
            "path_tracking.png",
            "uncertainty_calibration.png",
            "residual_histogram.png",
            "actuator_command_vs_realized.png",
            "smoother_convergence.png",
        ):
            _write_placeholder_png(report_dir / figure_name)
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "controller_diagnostics.csv").write_text(
            "timestamp_s,ik_success,safety_reason\n0.0,True,nominal\n0.1,False,ik_hold\n",
            encoding="utf-8",
        )
    return output_root


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
        controller_config={"name": "path_tracking"},
        estimator_config={"name": "anchored_vio"},
        ros2_bridge_used=False,
    )
    writer.write_manifest(manifest)
    camera_frame = IsaacCameraFramePacket(
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
    writer.append_camera_frame(camera_frame)
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
    writer.append_imu(
        IsaacImuPacket(
            packet_index=0,
            timestamp_s=0.10,
            sim_time_s=0.10,
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
    writer.append_imu(
        IsaacImuPacket(
            packet_index=1,
            timestamp_s=0.11,
            sim_time_s=0.11,
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
    writer.write_config_snapshot("scene", {"stage_path": "programmatic:anchor_room", "anchor_tag_id": 0})
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
    manifest = build_run_manifest(
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
    writer.write_manifest(manifest)

    camera_matrix = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    local_points = np.array(
        [
            [-0.05, 0.05, 0.0],
            [0.05, 0.05, 0.0],
            [0.05, -0.05, 0.0],
            [-0.05, -0.05, 0.0],
        ],
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

    frame_packets = [
        IsaacCameraFramePacket(
            frame_index=0,
            timestamp_s=0.10,
            sim_time_s=0.10,
            sensor_time_s=0.10,
            host_time_s=1.0,
            rgb_path="raw/rgb/frame_000000.png",
            intrinsics_snapshot={
                "fx_px": 100.0,
                "fy_px": 100.0,
                "cx_px": 50.0,
                "cy_px": 60.0,
                "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            extrinsics_snapshot={
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
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
            intrinsics_snapshot={
                "fx_px": 100.0,
                "fy_px": 100.0,
                "cx_px": 50.0,
                "cy_px": 60.0,
                "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            extrinsics_snapshot={
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            image_width_px=100,
            image_height_px=120,
            visible_gt_tag_ids=(0, 42),
        ),
    ]
    for frame_packet in frame_packets:
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
                detector_backend="opencv_aruco_apriltag36h11",
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
                detector_backend="opencv_aruco_apriltag36h11",
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
            detector_backend="opencv_aruco_apriltag36h11_bright_quad_match",
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
            "auxiliary_update_decisions": [
                {"tag_id": 42, "accepted": True, "reason": "accepted"},
            ],
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
        writer.write_gt(namespace="camera", payload={"timestamp_s": timestamp_s, "sim_time_s": timestamp_s, "px": 0.0, "py": 0.0, "pz": 0.0})
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
    writer.write_filter_state(
        FilterStateSnapshot(
            timestamp_s=0.10,
            sim_time_s=0.10,
            rotation_wi=np.eye(3, dtype=np.float64),
            position_world_m=np.zeros(3, dtype=np.float64),
            velocity_world_mps=np.zeros(3, dtype=np.float64),
            gyro_bias_rps=np.zeros(3, dtype=np.float64),
            accel_bias_mps2=np.zeros(3, dtype=np.float64),
            covariance=covariance.copy(),
            innovation_diagnostics={
                "last_innovation_norm": 0.05,
                "anchor_relocalizations": 0.0,
                "auxiliary_update_count": 1.0,
                "auxiliary_rejection_count": 1.0,
            },
            auxiliary_rejection_reason_counts={"fallback_only": 1},
        )
    )
    writer.write_filter_state(
        FilterStateSnapshot(
            timestamp_s=0.20,
            sim_time_s=0.20,
            rotation_wi=np.eye(3, dtype=np.float64),
            position_world_m=np.zeros(3, dtype=np.float64),
            velocity_world_mps=np.zeros(3, dtype=np.float64),
            gyro_bias_rps=np.zeros(3, dtype=np.float64),
            accel_bias_mps2=np.zeros(3, dtype=np.float64),
            covariance=covariance.copy(),
            innovation_diagnostics={
                "last_innovation_norm": 0.02,
                "anchor_relocalizations": 1.0,
                "auxiliary_update_count": 2.0,
                "auxiliary_rejection_count": 1.0,
            },
            auxiliary_rejection_reason_counts={"fallback_only": 1},
        )
    )
    writer.write_smoother_state(
        SmootherStateSnapshot(
            timestamp_s=0.10,
            sim_time_s=0.10,
            active_tag_poses={42: np.eye(4, dtype=np.float64)},
            cloned_positions_world_m=(np.zeros(3, dtype=np.float64),),
            covariance=np.eye(6, dtype=np.float64) * 1e-3,
            diagnostics={"feedback_correction_norm_m": 0.02, "trusted_feedback_count": 1.0, "total_feedback_candidates": 1.0},
        )
    )
    writer.write_smoother_state(
        SmootherStateSnapshot(
            timestamp_s=0.20,
            sim_time_s=0.20,
            active_tag_poses={42: np.eye(4, dtype=np.float64)},
            cloned_positions_world_m=(np.zeros(3, dtype=np.float64),),
            covariance=np.eye(6, dtype=np.float64) * 1e-3,
            diagnostics={"feedback_correction_norm_m": 0.01, "trusted_feedback_count": 1.0, "total_feedback_candidates": 1.0},
        )
    )
    writer.write_uncertainty(
        UncertaintySnapshot(
            timestamp_s=0.10,
            sim_time_s=0.10,
            pose_covariance=np.eye(3, dtype=np.float64) * 1e-4,
            velocity_covariance=np.eye(3, dtype=np.float64) * 1e-3,
            bias_covariance=np.eye(6, dtype=np.float64) * 1e-4,
            position_radius_95_m=0.03,
            velocity_radius_95_mps=0.1,
            diagnostics={"condition_number": 1.0},
        )
    )
    writer.write_uncertainty(
        UncertaintySnapshot(
            timestamp_s=0.20,
            sim_time_s=0.20,
            pose_covariance=np.eye(3, dtype=np.float64) * 1e-4,
            velocity_covariance=np.eye(3, dtype=np.float64) * 1e-3,
            bias_covariance=np.eye(6, dtype=np.float64) * 1e-4,
            position_radius_95_m=0.03,
            velocity_radius_95_mps=0.1,
            diagnostics={"condition_number": 1.0},
        )
    )
    return run_dir
