#!/usr/bin/env python3
"""Execute or summarize the minimal second-pass publication draft suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from calib_sim.reporting.isaac_second_pass_suite import (
    DEFAULT_SECOND_PASS_CONDITIONS,
    DEFAULT_SECOND_PASS_DRAFT_LOCK,
    DEFAULT_SECOND_PASS_DRAFT_SEEDS,
    generate_second_pass_suite_artifacts,
    second_pass_draft_run_id,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output/isaac_runs")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--lock-path", default=str(DEFAULT_SECOND_PASS_DRAFT_LOCK))
    parser.add_argument("--scene-config", default="config/isaac/scene/anchor_room.yaml")
    parser.add_argument("--robot-config", default="config/isaac/robot/franka_phone_head.yaml")
    parser.add_argument("--camera-config", default="config/isaac/camera/phone_main.yaml")
    parser.add_argument("--imu-config", default="config/isaac/imu/phone_nominal.yaml")
    parser.add_argument("--estimation-config", default="config/isaac/estimation/anchored_vio.yaml")
    parser.add_argument("--control-config", default="config/isaac/control/path_tracking.yaml")
    parser.add_argument("--visibility-config", default="config/isaac/visibility/anchor_dropout_nominal.yaml")
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SECOND_PASS_DRAFT_SEEDS))
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_SECOND_PASS_CONDITIONS))
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--execute-missing", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-lock", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _condition_overrides(condition: str, args: argparse.Namespace) -> dict[str, str | None]:
    if condition == "nominal_full_anchor":
        return {
            "actuation_config": "config/isaac/actuation/servo_nominal.yaml",
            "visibility_config": None,
        }
    if condition == "intermittent_anchor":
        return {
            "actuation_config": "config/isaac/actuation/servo_nominal.yaml",
            "visibility_config": args.visibility_config,
        }
    if condition == "servo_stress":
        return {
            "actuation_config": "config/isaac/actuation/servo_stress.yaml",
            "visibility_config": None,
        }
    raise ValueError(f"Unsupported second-pass draft condition: {condition}")


def _fused_selection(lock_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    draft_selection = dict(lock_payload.get("draft_selection", {}))
    backend = str(draft_selection.get("fused_nominal_backend", "lightweight"))
    covariance_scales = dict(draft_selection.get("fused_nominal_covariance_scales", {}))
    return backend, covariance_scales


def _run_command(
    args: argparse.Namespace,
    *,
    condition: str,
    estimator_mode: str,
    seed: int,
    smoother_backend: str,
    covariance_scales: dict[str, Any],
) -> list[str]:
    overrides = _condition_overrides(condition, args)
    run_id = second_pass_draft_run_id(condition, estimator_mode, seed)
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
        str(overrides["actuation_config"]),
        "--estimation-config",
        args.estimation_config,
        "--control-config",
        args.control_config,
        "--output-root",
        args.output_root,
        "--run-id",
        run_id,
        "--seed",
        str(seed),
        "--duration-s",
        str(args.duration_s),
        "--estimator-mode",
        estimator_mode,
        "--controller-mode",
        "closed-loop",
        "--bootstrap-control-policy",
        "hold_until_first_detection",
        "--use-aux-tags-in-filter",
        "--use-aux-tags-in-smoother",
        "--use-aux-map-for-control",
        "--smoother-backend",
        smoother_backend,
        "--no-promote-global-latest",
    ]
    visibility_config = overrides["visibility_config"]
    if visibility_config not in (None, ""):
        command.extend(["--visibility-config", str(visibility_config)])
    for flag, key in (
        ("--vision-covariance-scale", "vision_covariance_scale"),
        ("--anchor-vision-covariance-scale", "anchor_vision_covariance_scale"),
        ("--aux-vision-covariance-scale", "aux_vision_covariance_scale"),
        ("--imu-process-covariance-scale", "imu_process_covariance_scale"),
        ("--gyro-process-covariance-scale", "gyro_process_covariance_scale"),
        ("--accel-process-covariance-scale", "accel_process_covariance_scale"),
        ("--post-relocalization-covariance-scale", "post_relocalization_covariance_scale"),
    ):
        value = covariance_scales.get(key)
        if value in (None, ""):
            continue
        command.extend([flag, str(value)])
    if args.headless:
        command.append("--headless")
    return command


def _analyze_command(args: argparse.Namespace, *, condition: str, estimator_mode: str, seed: int) -> list[str]:
    return [
        str(args.python_executable),
        "scripts/analyze_isaac_estimator_quality.py",
        str(Path(args.output_root) / second_pass_draft_run_id(condition, estimator_mode, seed)),
    ]


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    lock_path = Path(args.lock_path).resolve()
    lock_payload = _load_json(lock_path)
    fused_backend, fused_scales = _fused_selection(lock_payload)
    planned_runs = []
    executed_runs = []
    for condition in args.conditions:
        for estimator_mode in ("visual", "fused"):
            smoother_backend = fused_backend if estimator_mode == "fused" else "lightweight"
            covariance_scales = fused_scales if estimator_mode == "fused" else {}
            for seed in args.seeds:
                run_id = second_pass_draft_run_id(condition, estimator_mode, seed)
                run_dir = output_root / run_id
                planned_runs.append(
                    {
                        "run_id": run_id,
                        "condition": condition,
                        "estimator_mode": estimator_mode,
                        "seed": int(seed),
                        "smoother_backend": smoother_backend,
                    }
                )
                metrics_path = run_dir / "analysis" / "metrics.json"
                quality_path = run_dir / "analysis" / "estimator_quality.json"
                if args.execute_missing and (not metrics_path.exists() or not quality_path.exists()):
                    subprocess.run(
                        _run_command(
                            args,
                            condition=condition,
                            estimator_mode=estimator_mode,
                            seed=int(seed),
                            smoother_backend=smoother_backend,
                            covariance_scales=covariance_scales,
                        ),
                        cwd=str(REPO_ROOT),
                        check=True,
                    )
                    subprocess.run(
                        _analyze_command(args, condition=condition, estimator_mode=estimator_mode, seed=int(seed)),
                        cwd=str(REPO_ROOT),
                        check=True,
                    )
                    executed_runs.append(run_id)
    suite_payload = generate_second_pass_suite_artifacts(
        output_root,
        lock_path=lock_path,
        regenerate_run_artifacts=False,
        artifact_source="latest_second_pass_suite",
    )
    if bool(args.update_lock):
        draft_selection = lock_payload.setdefault("draft_selection", {})
        draft_selection["suite_run_ids"] = [str(item["run_id"]) for item in planned_runs]
        _write_json(lock_path, lock_payload)
    return {
        "planned_runs": planned_runs,
        "executed_runs": executed_runs,
        "suite": suite_payload,
    }


def main() -> int:
    args = parse_args()
    if args.dry_run:
        lock_payload = _load_json(Path(args.lock_path).resolve())
        fused_backend, _ = _fused_selection(lock_payload)
        payload = {
            "planned_runs": [
                {
                    "run_id": second_pass_draft_run_id(condition, estimator_mode, seed),
                    "condition": condition,
                    "estimator_mode": estimator_mode,
                    "seed": int(seed),
                    "smoother_backend": fused_backend if estimator_mode == "fused" else "lightweight",
                }
                for condition in args.conditions
                for estimator_mode in ("visual", "fused")
                for seed in args.seeds
            ]
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    payload = run_suite(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
