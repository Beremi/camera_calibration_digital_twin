#!/usr/bin/env python3
"""Check first-pass waypoint reachability using the runtime's IK path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from calib_sim.isaac.app import IsaacAppBootstrapConfig, create_runtime


def _default_config_paths() -> dict[str, str]:
    return {
        "scene": "config/isaac/scene/anchor_room.yaml",
        "robot": "config/isaac/robot/franka_phone_head.yaml",
        "camera": "config/isaac/camera/phone_main.yaml",
        "imu": "config/isaac/imu/phone_nominal.yaml",
        "actuation": "config/isaac/actuation/servo_nominal.yaml",
        "estimation": "config/isaac/estimation/anchored_vio.yaml",
        "control": "config/isaac/control/path_tracking.yaml",
    }


def parse_args() -> argparse.Namespace:
    defaults = _default_config_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", default=defaults["scene"])
    parser.add_argument("--robot-config", default=defaults["robot"])
    parser.add_argument("--camera-config", default=defaults["camera"])
    parser.add_argument("--imu-config", default=defaults["imu"])
    parser.add_argument("--actuation-config", default=defaults["actuation"])
    parser.add_argument("--estimation-config", default=defaults["estimation"])
    parser.add_argument("--control-config", default=defaults["control"])
    parser.add_argument("--output-root", default="output/isaac_runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--neighborhood-step-m", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("reachability_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / "_probe" / run_id
    runtime = create_runtime(
        IsaacAppBootstrapConfig(
            scene_config_path=args.scene_config,
            robot_config_path=args.robot_config,
            camera_config_path=args.camera_config,
            imu_config_path=args.imu_config,
            actuation_config_path=args.actuation_config,
            estimation_config_path=args.estimation_config,
            control_config_path=args.control_config,
            run_id=run_id,
            run_dir=str(run_dir.resolve()),
            headless=bool(args.headless),
            seed=int(args.seed),
            duration_s=0.1,
            max_steps=1,
            estimator_mode="fused",
            controller_mode="closed-loop",
        )
    )
    runtime.start()
    try:
        if runtime._robot_binding is None or runtime._tracker is None or runtime._camera_binding is None:
            raise RuntimeError("Runtime failed to bind the robot, camera, or tracker.")
        current_position, current_orientation = runtime._robot_binding.get_end_effector_pose()
        current_control_position, _ = runtime._camera_binding.camera.get_world_pose()
        current_control_position = np.asarray(current_control_position, dtype=np.float64).reshape(3)
        current_joint_positions = runtime._robot_binding.get_joint_positions()
        offsets = [
            np.zeros(3, dtype=np.float64),
            np.array([args.neighborhood_step_m, 0.0, 0.0], dtype=np.float64),
            np.array([-args.neighborhood_step_m, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, args.neighborhood_step_m, 0.0], dtype=np.float64),
            np.array([0.0, -args.neighborhood_step_m, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, args.neighborhood_step_m], dtype=np.float64),
            np.array([0.0, 0.0, -args.neighborhood_step_m], dtype=np.float64),
        ]
        waypoint_payloads: list[dict[str, object]] = []
        success_count = 0
        total_samples = 0
        min_joint_limit_margin: float | None = None
        for index, waypoint in enumerate(runtime._tracker.waypoint_manager.waypoints):
            target = np.asarray(waypoint.position_world_m, dtype=np.float64)
            sample_payloads: list[dict[str, object]] = []
            for offset in offsets:
                sampled_target = target + offset
                ik_target = runtime._control_target_to_ik_target(
                    current_control_position_world_m=current_control_position,
                    desired_control_position_world_m=sampled_target,
                    current_ee_position_world_m=current_position,
                )
                joint_targets, success, alpha, orientation_policy, candidate_position = runtime._solve_control_ik(
                    current_position_world_m=current_position,
                    current_joint_positions=current_joint_positions,
                    desired_position_world_m=ik_target,
                    current_orientation_wxyz=current_orientation,
                )
                margin = runtime._robot_binding.joint_limit_margin(joint_targets)
                if margin is not None:
                    min_joint_limit_margin = margin if min_joint_limit_margin is None else min(min_joint_limit_margin, margin)
                sample_payloads.append(
                    {
                        "offset_world_m": [float(value) for value in offset.tolist()],
                        "target_world_m": [float(value) for value in sampled_target.tolist()],
                        "ik_target_world_m": [float(value) for value in np.asarray(ik_target, dtype=np.float64).tolist()],
                        "success": bool(success),
                        "ik_retry_alpha": float(alpha),
                        "orientation_policy": orientation_policy,
                        "candidate_position_world_m": [float(value) for value in candidate_position.tolist()],
                        "joint_limit_margin": margin,
                    }
                )
                success_count += int(bool(success))
                total_samples += 1
            waypoint_payloads.append(
                {
                    "waypoint_index": int(index),
                    "waypoint_world_m": [float(value) for value in target.tolist()],
                    "samples": sample_payloads,
                }
            )
        success_fraction = 0.0 if total_samples == 0 else float(success_count / total_samples)
        payload = {
            "run_id": run_id,
            "robot_preset": runtime.config.robot_preset,
            "control_config_path": args.control_config,
            "success_fraction": success_fraction,
            "minimum_joint_limit_margin": min_joint_limit_margin,
            "maximum_local_reachable_step_size_m": float(args.neighborhood_step_m) if success_fraction > 0.0 else 0.0,
            "fixed_orientation_primary": True,
            "waypoints": waypoint_payloads,
        }
    finally:
        runtime.shutdown()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
