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
    second_pass_fused_filter_overrides,
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


def _selection_settings(
    lock_payload: dict[str, Any],
    *,
    estimator_mode: str,
) -> tuple[str, dict[str, Any], dict[str, bool], dict[str, Any]]:
    draft_selection = dict(lock_payload.get("draft_selection", {}))
    if estimator_mode == "fused":
        backend = str(draft_selection.get("fused_nominal_backend", "lightweight"))
        covariance_scales = dict(draft_selection.get("fused_nominal_covariance_scales", {}))
        runtime_switches = dict(
            draft_selection.get(
                "fused_nominal_runtime_switches",
                {
                    "use_aux_tags_in_filter": True,
                    "use_aux_tags_in_smoother": True,
                    "use_aux_map_for_control": True,
                },
            )
        )
        return backend, covariance_scales, runtime_switches, second_pass_fused_filter_overrides(lock_payload)
    runtime_switches = dict(
        draft_selection.get(
            "visual_runtime_switches",
            {
                "use_aux_tags_in_filter": True,
                "use_aux_tags_in_smoother": True,
                "use_aux_map_for_control": True,
            },
        )
    )
    return "lightweight", {}, runtime_switches, {}


def _run_command(
    args: argparse.Namespace,
    *,
    condition: str,
    estimator_mode: str,
    seed: int,
    smoother_backend: str,
    covariance_scales: dict[str, Any],
    runtime_switches: dict[str, bool],
    filter_overrides: dict[str, Any],
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
        "--smoother-backend",
        smoother_backend,
        "--no-promote-global-latest",
    ]
    command.append(
        "--use-aux-tags-in-filter"
        if bool(runtime_switches.get("use_aux_tags_in_filter", True))
        else "--no-use-aux-tags-in-filter"
    )
    command.append(
        "--use-aux-tags-in-smoother"
        if bool(runtime_switches.get("use_aux_tags_in_smoother", True))
        else "--no-use-aux-tags-in-smoother"
    )
    command.append(
        "--use-aux-map-for-control"
        if bool(runtime_switches.get("use_aux_map_for_control", True))
        else "--no-use-aux-map-for-control"
    )
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
    if estimator_mode == "fused":
        command.append(
            "--allow-anchor-reacquisition-after-first-lock"
            if bool(filter_overrides.get("allow_anchor_reacquisition_after_first_lock", True))
            else "--no-allow-anchor-reacquisition-after-first-lock"
        )
        command.append(
            "--disable-imu-prediction-while-anchor-suppressed"
            if bool(filter_overrides.get("disable_imu_prediction_while_anchor_suppressed", False))
            else "--no-disable-imu-prediction-while-anchor-suppressed"
        )
        mode = filter_overrides.get("suppression_propagation_mode")
        if mode not in (None, ""):
            command.extend(["--suppression-propagation-mode", str(mode)])
        gate = filter_overrides.get("suppression_imu_specific_force_gate_mps2")
        if gate not in (None, ""):
            command.extend(["--suppression-imu-specific-force-gate-mps2", str(gate)])
        covinfl = filter_overrides.get("dropout_post_reacquisition_covariance_scale")
        if covinfl not in (None, ""):
            command.extend(["--dropout-post-reacquisition-covariance-scale", str(covinfl)])
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
    planned_runs = []
    executed_runs = []
    for condition in args.conditions:
        for estimator_mode in ("visual", "fused"):
            smoother_backend, covariance_scales, runtime_switches, filter_overrides = _selection_settings(
                lock_payload,
                estimator_mode=estimator_mode,
            )
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
                        "runtime_switches": runtime_switches,
                        "filter_overrides": filter_overrides,
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
                            runtime_switches=runtime_switches,
                            filter_overrides=filter_overrides,
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
    requested_conditions = {str(condition) for condition in args.conditions}
    requested_seeds = {int(seed) for seed in args.seeds}
    full_suite_requested = requested_conditions == set(DEFAULT_SECOND_PASS_CONDITIONS) and requested_seeds == set(
        int(seed) for seed in DEFAULT_SECOND_PASS_DRAFT_SEEDS
    )
    suite_payload: dict[str, Any]
    suite_summary: dict[str, Any]
    if full_suite_requested:
        suite_payload = generate_second_pass_suite_artifacts(
            output_root,
            lock_path=lock_path,
            regenerate_run_artifacts=False,
            artifact_source="latest_second_pass_suite",
        )
        suite_summary = _load_json(Path(suite_payload["summary_json"]).resolve())
    else:
        suite_payload = {
            "skipped": True,
            "reason": "partial_selection",
            "requested_conditions": sorted(requested_conditions),
            "requested_seeds": sorted(requested_seeds),
        }
        suite_summary = {}
    if bool(args.update_lock) and full_suite_requested:
        draft_selection = lock_payload.setdefault("draft_selection", {})
        draft_selection["suite_run_ids"] = [str(item["run_id"]) for item in planned_runs]
        draft_selection["representative_run_ids"] = dict(suite_summary.get("representative_runs", {}))
        draft_selection["suite_artifacts_stale"] = False
        draft_selection["media_artifacts_stale"] = True
        draft_selection["draft_ready"] = False
        draft_selection["draft_suite_status"] = "fresh_suite_rerun_pending_publication_refresh"
        draft_selection[
            "draft_suite_status_reason"
        ] = "The 18-run suite has been rerun from the promoted fused lock, but the second-pass publication draft and media bundle have not yet been regenerated from these refreshed artifacts."
        _write_json(lock_path, lock_payload)
    return {
        "planned_runs": planned_runs,
        "executed_runs": executed_runs,
        "suite": suite_payload,
        "full_suite_requested": full_suite_requested,
    }


def main() -> int:
    args = parse_args()
    if args.dry_run:
        lock_payload = _load_json(Path(args.lock_path).resolve())
        payload = {
            "planned_runs": [
                {
                    "run_id": second_pass_draft_run_id(condition, estimator_mode, seed),
                    "condition": condition,
                    "estimator_mode": estimator_mode,
                    "seed": int(seed),
                    "smoother_backend": _selection_settings(lock_payload, estimator_mode=estimator_mode)[0],
                    "runtime_switches": _selection_settings(lock_payload, estimator_mode=estimator_mode)[2],
                    "filter_overrides": _selection_settings(lock_payload, estimator_mode=estimator_mode)[3],
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
