#!/usr/bin/env python3
"""Execute or summarize the seed-007 fused-dropout debug pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from calib_sim.reporting.isaac_second_pass_dropout_debug import (
    DEFAULT_SECOND_PASS_DROPOUT_DEBUG_DOC,
    DEFAULT_SECOND_PASS_DROPOUT_DEBUG_RUN_SPECS,
    build_second_pass_dropout_debug_bundle,
    generate_second_pass_dropout_debug_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output/isaac_runs")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--scene-config", default="config/isaac/scene/anchor_room.yaml")
    parser.add_argument("--robot-config", default="config/isaac/robot/franka_phone_head.yaml")
    parser.add_argument("--camera-config", default="config/isaac/camera/phone_main.yaml")
    parser.add_argument("--imu-config", default="config/isaac/imu/phone_nominal.yaml")
    parser.add_argument("--actuation-config", default="config/isaac/actuation/servo_nominal.yaml")
    parser.add_argument("--estimation-config", default="config/isaac/estimation/anchored_vio.yaml")
    parser.add_argument("--control-config", default="config/isaac/control/path_tracking.yaml")
    parser.add_argument("--visibility-config", default="config/isaac/visibility/anchor_dropout_nominal.yaml")
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--execute-missing", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gyro-process-covariance-scale", type=float, default=None)
    parser.add_argument("--accel-process-covariance-scale", type=float, default=None)
    parser.add_argument("--suppression-imu-specific-force-gate-mps2", type=float, default=None)
    parser.add_argument("--docs-path", default=str(DEFAULT_SECOND_PASS_DROPOUT_DEBUG_DOC))
    return parser.parse_args()


def _run_command(args: argparse.Namespace, spec: dict[str, Any]) -> list[str]:
    command = [
        str(args.python_executable),
        "scripts/run_isaac_anchor_vio.py",
        "--scene-config",
        args.scene_config,
        "--robot-config",
        args.robot_config,
        "--camera-config",
        args.camera_config,
        "--imu-config",
        args.imu_config,
        "--actuation-config",
        args.actuation_config,
        "--estimation-config",
        args.estimation_config,
        "--control-config",
        args.control_config,
        "--visibility-config",
        args.visibility_config,
        "--output-root",
        args.output_root,
        "--run-id",
        str(spec["run_id"]),
        "--seed",
        str(args.seed),
        "--duration-s",
        str(args.duration_s),
        "--estimator-mode",
        str(spec["estimator_mode"]),
        "--controller-mode",
        "closed-loop",
        "--bootstrap-control-policy",
        "hold_until_first_detection",
        "--smoother-backend",
        "lightweight",
        "--vision-covariance-scale",
        "4.0",
        "--imu-process-covariance-scale",
        "8.0",
        "--post-relocalization-covariance-scale",
        "1.0",
        "--no-use-aux-tags-in-filter",
        "--no-use-aux-tags-in-smoother",
        "--no-use-aux-map-for-control",
        "--no-promote-global-latest",
    ]
    debug_mode = str(spec["debug_mode"])
    if str(spec["estimator_mode"]) == "fused":
        if args.gyro_process_covariance_scale is not None:
            command.extend(["--gyro-process-covariance-scale", str(args.gyro_process_covariance_scale)])
        if args.accel_process_covariance_scale is not None:
            command.extend(["--accel-process-covariance-scale", str(args.accel_process_covariance_scale)])
        if args.suppression_imu_specific_force_gate_mps2 is not None:
            command.extend(
                [
                    "--suppression-imu-specific-force-gate-mps2",
                    str(args.suppression_imu_specific_force_gate_mps2),
                ]
            )
    if debug_mode == "fused_no_reacquisition":
        command.append("--no-allow-anchor-reacquisition-after-first-lock")
    elif debug_mode == "fused_no_imu_during_suppression":
        command.append("--disable-imu-prediction-while-anchor-suppressed")
    elif debug_mode == "fused_reacquisition_covariance_inflation":
        command.extend(["--dropout-post-reacquisition-covariance-scale", "8.0"])
    elif debug_mode == "fused_reacquisition_correction_clipping":
        command.extend(
            [
                "--dropout-max-reacquisition-position-correction-m",
                "0.05",
                "--dropout-max-reacquisition-rotation-correction-deg",
                "5.0",
                "--dropout-max-reacquisition-velocity-correction-mps",
                "0.20",
            ]
        )
    if args.headless:
        command.append("--headless")
    return command


def _analyze_command(args: argparse.Namespace, run_id: str) -> list[str]:
    return [
        str(args.python_executable),
        "scripts/analyze_isaac_estimator_quality.py",
        str(Path(args.output_root) / run_id),
    ]


def run_debug_pack(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    planned_runs: list[dict[str, Any]] = []
    executed_runs: list[str] = []
    for spec in DEFAULT_SECOND_PASS_DROPOUT_DEBUG_RUN_SPECS:
        run_id = str(spec["run_id"])
        run_dir = output_root / run_id
        planned_runs.append(
            {
                "run_id": run_id,
                "debug_mode": str(spec["debug_mode"]),
                "estimator_mode": str(spec["estimator_mode"]),
                "command": _run_command(args, spec),
            }
        )
        metrics_path = run_dir / "analysis" / "metrics.json"
        quality_path = run_dir / "analysis" / "estimator_quality.json"
        debug_summary_path = run_dir / "analysis" / "dropout_debug_summary.json"
        if args.execute_missing and (not metrics_path.exists() or not quality_path.exists()):
            subprocess.run(_run_command(args, spec), cwd=str(REPO_ROOT), check=True)
            subprocess.run(_analyze_command(args, run_id), cwd=str(REPO_ROOT), check=True)
            executed_runs.append(run_id)
        if debug_summary_path.exists() or (run_dir / "raw" / "dropout_debug_frames.csv").exists():
            generate_second_pass_dropout_debug_artifacts(run_dir)
    bundle_payload = build_second_pass_dropout_debug_bundle(output_root, docs_path=args.docs_path)
    return {
        "planned_runs": planned_runs,
        "executed_runs": executed_runs,
        "bundle": bundle_payload,
    }


def main() -> int:
    args = parse_args()
    if args.dry_run:
        payload = {
            "planned_runs": [
                {
                    "run_id": str(spec["run_id"]),
                    "debug_mode": str(spec["debug_mode"]),
                    "estimator_mode": str(spec["estimator_mode"]),
                    "command": _run_command(args, spec),
                }
                for spec in DEFAULT_SECOND_PASS_DROPOUT_DEBUG_RUN_SPECS
            ]
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    payload = run_debug_pack(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
