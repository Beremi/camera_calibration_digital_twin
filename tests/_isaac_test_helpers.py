"""Helpers for Isaac scaffold tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calib_sim.isaac.estimation.state_defs import FilterStateSnapshot, SmootherStateSnapshot, UncertaintySnapshot
from calib_sim.isaac.logging.run_manifest import build_run_manifest
from calib_sim.isaac.logging.schemas import (
    IsaacCameraFramePacket,
    IsaacImuPacket,
    IsaacJointCommandPacket,
    IsaacRealizedJointPacket,
    IsaacTagDetectionPacket,
)
from calib_sim.isaac.logging.writer import IsaacRunWriter


def make_minimal_isaac_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "isaac_run"
    writer = IsaacRunWriter(run_dir)
    writer.write_config_snapshot("scene", {"stage_path": "assets/isaac/anchor_room.usd", "anchor_tag_id": 0})
    manifest = build_run_manifest(
        repo_root=Path(__file__).resolve().parents[1],
        run_id="test_run",
        isaac_sim_version="test",
        stage_usd_path="assets/isaac/anchor_room.usd",
        robot_preset="ur5e_phone_head",
        anchor_tag_id=0,
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
            corners_xy=((45.0, 55.0), (55.0, 55.0), (55.0, 65.0), (45.0, 65.0)),
            corner_order="clockwise_top_left_first",
            score=1.0,
            is_anchor=True,
            visibility_flags={"detected": True},
        )
    )
    writer.append_estimator_input({"timestamp_s": 0.1, "kind": "frame_pack", "frame_index": 0})
    writer.append_imu(IsaacImuPacket(timestamp_s=0.10, sim_time_s=0.10, wx=0.0, wy=0.0, wz=0.0, ax=0.0, ay=0.0, az=9.81, imu_frame="I", noise_preset="phone_nominal"))
    writer.append_imu(IsaacImuPacket(timestamp_s=0.11, sim_time_s=0.11, wx=0.0, wy=0.0, wz=0.0, ax=0.0, ay=0.0, az=9.81, imu_frame="I", noise_preset="phone_nominal"))
    writer.append_command(IsaacJointCommandPacket(timestamp_s=0.10, sim_time_s=0.10, joint_id="joint0", command_type="position", command_value=0.2, controller_mode="closed_loop"))
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
