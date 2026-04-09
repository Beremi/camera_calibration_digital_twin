#!/usr/bin/env python3
"""Execute or summarize the minimal fused second-pass tuning sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from calib_sim.reporting.isaac_second_pass_suite import DEFAULT_SECOND_PASS_DRAFT_LOCK
from calib_sim.reporting.isaac_second_pass_suite import second_pass_fused_filter_overrides


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
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--execute-missing", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-lock", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("anchor_only", "aux_for_control", "aux_estimation_only"),
        default=["anchor_only", "aux_for_control"],
    )
    parser.add_argument("--imu-process-covariance-scales", nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--vision-covariance-scales", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--post-relocalization-covariance-scales", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--gyro-process-covariance-scales", nargs="*", type=float, default=[])
    parser.add_argument("--accel-process-covariance-scales", nargs="*", type=float, default=[])
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _scale_token(name: str, value: float | None) -> str:
    if value is None:
        return f"{name}_default"
    numeric = str(value).replace(".", "p")
    return f"{name}_{numeric}"


def _suppression_profile_suffix(filter_overrides: dict[str, Any]) -> str:
    tokens: list[str] = []
    mode = str(filter_overrides.get("suppression_propagation_mode", "full_imu") or "full_imu")
    gate = _float_or_none(filter_overrides.get("suppression_imu_specific_force_gate_mps2"))
    covinfl = _float_or_none(filter_overrides.get("dropout_post_reacquisition_covariance_scale"))
    allow_reacq = bool(filter_overrides.get("allow_anchor_reacquisition_after_first_lock", True))
    disable_imu = bool(filter_overrides.get("disable_imu_prediction_while_anchor_suppressed", False))
    if mode != "full_imu" or gate is not None or covinfl is not None or not allow_reacq or disable_imu:
        tokens.append("supp")
    if mode != "full_imu":
        tokens.append(mode)
    if gate is not None:
        tokens.append(f"gate_{str(gate).replace('.', 'p')}")
    if covinfl is not None:
        tokens.append(f"covinfl_{str(covinfl).replace('.', 'p')}")
    if not allow_reacq:
        tokens.append("no_reacq")
    if disable_imu:
        tokens.append("no_imu")
    return "_".join(tokens)


def _candidate_run_id(
    *,
    phase: str,
    backend: str,
    seed: int,
    imu_scale: float,
    vision_scale: float,
    post_scale: float,
    gyro_scale: float | None,
    accel_scale: float | None,
    suppression_suffix: str,
) -> str:
    tokens = [
        "second_pass_tuning",
        phase,
        backend,
        f"seed_{int(seed):03d}",
        _scale_token("imu", imu_scale),
        _scale_token("vision", vision_scale),
        _scale_token("post", post_scale),
        _scale_token("gyro", gyro_scale),
        _scale_token("accel", accel_scale),
    ]
    if suppression_suffix:
        tokens.append(suppression_suffix)
    return "_".join(tokens)


def _run_command(
    args: argparse.Namespace,
    *,
    run_id: str,
    backend: str,
    aux_enabled: bool,
    imu_scale: float,
    vision_scale: float,
    post_scale: float,
    gyro_scale: float | None,
    accel_scale: float | None,
    filter_overrides: dict[str, Any],
) -> list[str]:
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
        "config/isaac/actuation/servo_nominal.yaml",
        "--estimation-config",
        args.estimation_config,
        "--control-config",
        args.control_config,
        "--output-root",
        args.output_root,
        "--run-id",
        run_id,
        "--seed",
        str(args.seed),
        "--duration-s",
        str(args.duration_s),
        "--estimator-mode",
        "fused",
        "--controller-mode",
        "closed-loop",
        "--bootstrap-control-policy",
        "hold_until_first_detection",
        "--smoother-backend",
        backend,
        "--vision-covariance-scale",
        str(vision_scale),
        "--imu-process-covariance-scale",
        str(imu_scale),
        "--post-relocalization-covariance-scale",
        str(post_scale),
        "--no-promote-global-latest",
    ]
    command.append("--use-aux-tags-in-filter" if aux_enabled else "--no-use-aux-tags-in-filter")
    command.append("--use-aux-tags-in-smoother" if aux_enabled else "--no-use-aux-tags-in-smoother")
    command.append("--use-aux-map-for-control" if aux_enabled else "--no-use-aux-map-for-control")
    if gyro_scale is not None:
        command.extend(["--gyro-process-covariance-scale", str(gyro_scale)])
    if accel_scale is not None:
        command.extend(["--accel-process-covariance-scale", str(accel_scale)])
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


def _analyze_command(args: argparse.Namespace, run_id: str) -> list[str]:
    return [
        str(args.python_executable),
        "scripts/analyze_isaac_estimator_quality.py",
        str(Path(args.output_root) / run_id),
    ]


def _ensure_candidate(
    args: argparse.Namespace,
    *,
    phase: str,
    backend: str,
    aux_enabled: bool,
    imu_scale: float,
    vision_scale: float,
    post_scale: float,
    gyro_scale: float | None,
    accel_scale: float | None,
    filter_overrides: dict[str, Any],
) -> dict[str, Any]:
    suppression_suffix = _suppression_profile_suffix(filter_overrides)
    run_id = _candidate_run_id(
        phase=phase,
        backend=backend,
        seed=int(args.seed),
        imu_scale=imu_scale,
        vision_scale=vision_scale,
        post_scale=post_scale,
        gyro_scale=gyro_scale,
        accel_scale=accel_scale,
        suppression_suffix=suppression_suffix,
    )
    run_dir = Path(args.output_root) / run_id
    metrics_path = run_dir / "analysis" / "metrics.json"
    quality_path = run_dir / "analysis" / "estimator_quality.json"
    if args.execute_missing and (not metrics_path.exists() or not quality_path.exists()):
        subprocess.run(
            _run_command(
                args,
                run_id=run_id,
                backend=backend,
                aux_enabled=aux_enabled,
                imu_scale=imu_scale,
                vision_scale=vision_scale,
                post_scale=post_scale,
                gyro_scale=gyro_scale,
                accel_scale=accel_scale,
                filter_overrides=filter_overrides,
            ),
            cwd=str(REPO_ROOT),
            check=True,
        )
        subprocess.run(_analyze_command(args, run_id), cwd=str(REPO_ROOT), check=True)
    if not metrics_path.exists() or not quality_path.exists():
        return {
            "run_id": run_id,
            "phase": phase,
            "aux_enabled": aux_enabled,
            "smoother_backend": backend,
            "imu_process_covariance_scale": imu_scale,
            "vision_covariance_scale": vision_scale,
            "post_relocalization_covariance_scale": post_scale,
            "gyro_process_covariance_scale": gyro_scale,
        "accel_process_covariance_scale": accel_scale,
        "suppression_profile_suffix": suppression_suffix,
        "suppression_propagation_mode": str(filter_overrides.get("suppression_propagation_mode", "full_imu")),
        "suppression_imu_specific_force_gate_mps2": _float_or_none(
            filter_overrides.get("suppression_imu_specific_force_gate_mps2")
        ),
        "dropout_post_reacquisition_covariance_scale": _float_or_none(
            filter_overrides.get("dropout_post_reacquisition_covariance_scale")
        ),
        "allow_anchor_reacquisition_after_first_lock": bool(
            filter_overrides.get("allow_anchor_reacquisition_after_first_lock", True)
        ),
        "disable_imu_prediction_while_anchor_suppressed": bool(
            filter_overrides.get("disable_imu_prediction_while_anchor_suppressed", False)
        ),
        "missing": True,
    }
    metrics = _load_json(metrics_path)
    quality = _load_json(quality_path)
    summary = dict(quality.get("summary", {}))
    return {
        "run_id": run_id,
        "phase": phase,
        "aux_enabled": aux_enabled,
        "smoother_backend": backend,
        "imu_process_covariance_scale": imu_scale,
        "vision_covariance_scale": vision_scale,
        "post_relocalization_covariance_scale": post_scale,
        "gyro_process_covariance_scale": gyro_scale,
        "accel_process_covariance_scale": accel_scale,
        "suppression_profile_suffix": suppression_suffix,
        "suppression_propagation_mode": str(filter_overrides.get("suppression_propagation_mode", "full_imu")),
        "suppression_imu_specific_force_gate_mps2": _float_or_none(
            filter_overrides.get("suppression_imu_specific_force_gate_mps2")
        ),
        "dropout_post_reacquisition_covariance_scale": _float_or_none(
            filter_overrides.get("dropout_post_reacquisition_covariance_scale")
        ),
        "allow_anchor_reacquisition_after_first_lock": bool(
            filter_overrides.get("allow_anchor_reacquisition_after_first_lock", True)
        ),
        "disable_imu_prediction_while_anchor_suppressed": bool(
            filter_overrides.get("disable_imu_prediction_while_anchor_suppressed", False)
        ),
        "missing": False,
        "mean_position_error_m": _float_or_none(metrics.get("trajectory", {}).get("mean_position_error_m")),
        "mean_waypoint_error_m": _float_or_none(metrics.get("control", {}).get("mean_waypoint_error_m")),
        "completion_fraction": _float_or_none(metrics.get("control", {}).get("completion_fraction")),
        "anchor_rmse_px": _float_or_none(summary.get("anchor_mean_reprojection_rmse_px")),
        "aux_rmse_px": _float_or_none(summary.get("auxiliary_mean_reprojection_rmse_px")),
        "mean_position_radius_95_m": _float_or_none(metrics.get("uncertainty_calibration", {}).get("mean_position_radius_95_m")),
        "empirical_95_coverage_percent": _float_or_none(metrics.get("uncertainty_calibration", {}).get("empirical_95_coverage_percent")),
        "pose_nees": _float_or_none(metrics.get("uncertainty_calibration", {}).get("pose_nees")),
        "mean_smoother_feedback_norm_m": _float_or_none(summary.get("mean_smoother_correction_norm_m")),
        "mean_actuator_tracking_error": _float_or_none(metrics.get("control", {}).get("mean_actuator_tracking_error")),
    }


def _load_visual_reference(lock_payload: dict[str, Any], output_root: Path) -> dict[str, float | None]:
    run_id = str(lock_payload.get("draft_selection", {}).get("visual_nominal_reference_run_id", "") or "")
    if not run_id:
        return {"mean_position_error_m": None, "mean_waypoint_error_m": None}
    metrics_path = output_root / run_id / "analysis" / "metrics.json"
    if not metrics_path.exists():
        return {"mean_position_error_m": None, "mean_waypoint_error_m": None}
    metrics = _load_json(metrics_path)
    return {
        "mean_position_error_m": _float_or_none(metrics.get("trajectory", {}).get("mean_position_error_m")),
        "mean_waypoint_error_m": _float_or_none(metrics.get("control", {}).get("mean_waypoint_error_m")),
    }


def _candidate_priority(row: dict[str, Any], visual_reference: dict[str, float | None]) -> tuple[float, ...]:
    completion = _float_or_none(row.get("completion_fraction"))
    position_error = _float_or_none(row.get("mean_position_error_m"))
    waypoint_error = _float_or_none(row.get("mean_waypoint_error_m"))
    coverage = _float_or_none(row.get("empirical_95_coverage_percent"))
    pose_nees = _float_or_none(row.get("pose_nees"))
    feedback = _float_or_none(row.get("mean_smoother_feedback_norm_m"))
    visual_position = _float_or_none(visual_reference.get("mean_position_error_m"))
    visual_waypoint = _float_or_none(visual_reference.get("mean_waypoint_error_m"))
    completion_penalty = 0.0 if completion is not None and abs(completion - 1.0) < 1e-9 else 1.0
    waypoint_term = 1e6 if waypoint_error is None else waypoint_error
    position_ratio_penalty = 0.0
    if visual_position not in (None, 0.0) and position_error is not None and position_error > visual_position * 1.10:
        position_ratio_penalty = position_error - visual_position * 1.10
    coverage_penalty = 1.0
    if coverage is not None and 90.0 <= coverage <= 100.0:
        coverage_penalty = 0.0
    elif coverage is not None:
        coverage_penalty = abs(coverage - 95.0) / 100.0
    nees_term = 1e6 if pose_nees is None else pose_nees
    feedback_term = 1e6 if feedback is None else feedback
    position_term = 1e6 if position_error is None else position_error
    return (
        completion_penalty,
        position_ratio_penalty,
        waypoint_term,
        coverage_penalty,
        nees_term,
        feedback_term,
        position_term,
    )


def _build_anchor_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for imu_scale in args.imu_process_covariance_scales:
        for vision_scale in args.vision_covariance_scales:
            for post_scale in args.post_relocalization_covariance_scales:
                rows.append(
                    {
                        "phase": "anchor_only",
                        "aux_enabled": False,
                        "smoother_backend": "lightweight",
                        "imu_process_covariance_scale": float(imu_scale),
                        "vision_covariance_scale": float(vision_scale),
                        "post_relocalization_covariance_scale": float(post_scale),
                        "gyro_process_covariance_scale": None,
                        "accel_process_covariance_scale": None,
                    }
                )
    return rows


def _build_split_candidates(
    args: argparse.Namespace,
    *,
    base_anchor_row: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    gyro_scales = [None] if not args.gyro_process_covariance_scales else [None] + list(args.gyro_process_covariance_scales)
    accel_scales = [None] if not args.accel_process_covariance_scales else [None] + list(args.accel_process_covariance_scales)
    if len(gyro_scales) == 1 and len(accel_scales) == 1:
        return []
    if base_anchor_row is None:
        return []
    rows: list[dict[str, Any]] = []
    for gyro_scale in gyro_scales:
        for accel_scale in accel_scales:
            if gyro_scale is None and accel_scale is None:
                continue
            rows.append(
                {
                    "phase": "anchor_only",
                    "aux_enabled": False,
                    "smoother_backend": "lightweight",
                    "imu_process_covariance_scale": float(base_anchor_row["imu_process_covariance_scale"]),
                    "vision_covariance_scale": float(base_anchor_row["vision_covariance_scale"]),
                    "post_relocalization_covariance_scale": float(base_anchor_row["post_relocalization_covariance_scale"]),
                    "gyro_process_covariance_scale": gyro_scale,
                    "accel_process_covariance_scale": accel_scale,
                }
            )
    return rows


def _phase_runtime_switches(phase: str) -> dict[str, bool]:
    if phase == "anchor_only":
        return {
            "use_aux_tags_in_filter": False,
            "use_aux_tags_in_smoother": False,
            "use_aux_map_for_control": False,
        }
    if phase == "aux_for_control":
        return {
            "use_aux_tags_in_filter": True,
            "use_aux_tags_in_smoother": True,
            "use_aux_map_for_control": True,
        }
    if phase == "aux_estimation_only":
        return {
            "use_aux_tags_in_filter": True,
            "use_aux_tags_in_smoother": True,
            "use_aux_map_for_control": False,
        }
    raise ValueError(f"Unsupported tuning phase: {phase}")


def run_tuning(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    lock_path = Path(args.lock_path).resolve()
    lock_payload = _load_json(lock_path)
    visual_reference = _load_visual_reference(lock_payload, output_root)
    filter_overrides = second_pass_fused_filter_overrides(lock_payload)
    tuning_dir = output_root / "latest_second_pass_tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    base_candidates = _build_anchor_candidates(args) if "anchor_only" in set(args.phases) else []
    for candidate in base_candidates:
        candidate_rows.append(
            _ensure_candidate(
                args,
                phase=str(candidate["phase"]),
                backend=str(candidate["smoother_backend"]),
                aux_enabled=bool(candidate["aux_enabled"]),
                imu_scale=float(candidate["imu_process_covariance_scale"]),
                vision_scale=float(candidate["vision_covariance_scale"]),
                post_scale=float(candidate["post_relocalization_covariance_scale"]),
                gyro_scale=_float_or_none(candidate.get("gyro_process_covariance_scale")),
                accel_scale=_float_or_none(candidate.get("accel_process_covariance_scale")),
                filter_overrides=filter_overrides,
            )
        )
    completed_anchor_rows = [row for row in candidate_rows if not bool(row.get("missing"))]
    best_anchor_global = None if not completed_anchor_rows else min(
        completed_anchor_rows,
        key=lambda row: _candidate_priority(row, visual_reference),
    )
    split_candidates = _build_split_candidates(args, base_anchor_row=best_anchor_global)
    for candidate in split_candidates:
        candidate_rows.append(
            _ensure_candidate(
                args,
                phase=str(candidate["phase"]),
                backend=str(candidate["smoother_backend"]),
                aux_enabled=bool(candidate["aux_enabled"]),
                imu_scale=float(candidate["imu_process_covariance_scale"]),
                vision_scale=float(candidate["vision_covariance_scale"]),
                post_scale=float(candidate["post_relocalization_covariance_scale"]),
                gyro_scale=_float_or_none(candidate.get("gyro_process_covariance_scale")),
                accel_scale=_float_or_none(candidate.get("accel_process_covariance_scale")),
                filter_overrides=filter_overrides,
            )
        )
    completed_anchor_rows = [
        row
        for row in candidate_rows
        if not bool(row.get("missing")) and str(row.get("phase", "")) == "anchor_only"
    ]
    phase_set = set(args.phases)
    if completed_anchor_rows and ("aux_for_control" in phase_set or "aux_estimation_only" in phase_set):
        best_anchor = min(completed_anchor_rows, key=lambda row: _candidate_priority(row, visual_reference))
        if "aux_for_control" in phase_set:
            for backend in ("lightweight", "windowed_ba"):
                candidate_rows.append(
                    _ensure_candidate(
                        args,
                        phase="aux_for_control",
                        backend=backend,
                        aux_enabled=True,
                        imu_scale=float(best_anchor["imu_process_covariance_scale"]),
                        vision_scale=float(best_anchor["vision_covariance_scale"]),
                        post_scale=float(best_anchor["post_relocalization_covariance_scale"]),
                        gyro_scale=_float_or_none(best_anchor.get("gyro_process_covariance_scale")),
                        accel_scale=_float_or_none(best_anchor.get("accel_process_covariance_scale")),
                        filter_overrides=filter_overrides,
                    )
                )
        if "aux_estimation_only" in phase_set:
            candidate_rows.append(
                _ensure_candidate(
                    args,
                    phase="aux_estimation_only",
                    backend="lightweight",
                    aux_enabled=True,
                    imu_scale=float(best_anchor["imu_process_covariance_scale"]),
                    vision_scale=float(best_anchor["vision_covariance_scale"]),
                    post_scale=float(best_anchor["post_relocalization_covariance_scale"]),
                    gyro_scale=_float_or_none(best_anchor.get("gyro_process_covariance_scale")),
                    accel_scale=_float_or_none(best_anchor.get("accel_process_covariance_scale")),
                    filter_overrides=filter_overrides,
                )
            )
    completed_rows = [row for row in candidate_rows if not bool(row.get("missing"))]
    completed_non_anchor_rows = [
        row
        for row in completed_rows
        if str(row.get("phase", "")) in {"aux_for_control", "aux_estimation_only"}
    ]
    selection_pool = completed_non_anchor_rows if completed_non_anchor_rows else completed_anchor_rows
    selected = None if not selection_pool else min(selection_pool, key=lambda row: _candidate_priority(row, visual_reference))
    ranked_candidates = sorted(selection_pool, key=lambda row: _candidate_priority(row, visual_reference))
    top_candidates = ranked_candidates[:3]

    summary = {
        "rows": candidate_rows,
        "visual_reference": visual_reference,
        "selected_run_id": None if selected is None else str(selected["run_id"]),
        "selected": selected,
        "selection_pool_phase": "non_anchor" if completed_non_anchor_rows else "anchor_only",
        "selected_backend": None if selected is None else str(selected["smoother_backend"]),
        "selected_filter_overrides": filter_overrides,
        "selected_covariance_scales": None
        if selected is None
        else {
            "vision_covariance_scale": float(selected["vision_covariance_scale"]),
            "anchor_vision_covariance_scale": None,
            "aux_vision_covariance_scale": None,
            "imu_process_covariance_scale": float(selected["imu_process_covariance_scale"]),
            "gyro_process_covariance_scale": _float_or_none(selected.get("gyro_process_covariance_scale")),
            "accel_process_covariance_scale": _float_or_none(selected.get("accel_process_covariance_scale")),
            "post_relocalization_covariance_scale": float(selected["post_relocalization_covariance_scale"]),
        },
        "top_candidates": top_candidates,
    }
    _write_csv(tuning_dir / "summary.csv", candidate_rows)
    _write_json(tuning_dir / "summary.json", summary)
    _write_csv(tuning_dir / "top_fused_candidates.csv", top_candidates)

    if bool(args.update_lock) and selected is not None:
        draft_selection = lock_payload.setdefault("draft_selection", {})
        draft_selection["fused_nominal_backend"] = str(selected["smoother_backend"])
        draft_selection["fused_nominal_reference_run_id"] = str(selected["run_id"])
        draft_selection["fused_nominal_covariance_scales"] = {
            "vision_covariance_scale": float(selected["vision_covariance_scale"]),
            "anchor_vision_covariance_scale": None,
            "aux_vision_covariance_scale": None,
            "imu_process_covariance_scale": float(selected["imu_process_covariance_scale"]),
            "gyro_process_covariance_scale": _float_or_none(selected.get("gyro_process_covariance_scale")),
            "accel_process_covariance_scale": _float_or_none(selected.get("accel_process_covariance_scale")),
            "post_relocalization_covariance_scale": float(selected["post_relocalization_covariance_scale"]),
        }
        draft_selection["fused_nominal_runtime_switches"] = _phase_runtime_switches(str(selected["phase"]))
        draft_selection["fused_nominal_filter_overrides"] = filter_overrides
        draft_selection["tuning_summary_path"] = str((tuning_dir / "summary.json").resolve())
        draft_selection["latest_nominal_recheck_run_id"] = str(selected["run_id"])
        _write_json(lock_path, lock_payload)

    return {
        "summary_json": str((tuning_dir / "summary.json").resolve()),
        "summary_csv": str((tuning_dir / "summary.csv").resolve()),
        "top_candidates_csv": str((tuning_dir / "top_fused_candidates.csv").resolve()),
        "selected_run_id": None if selected is None else str(selected["run_id"]),
        "selected": selected,
        "completed_run_count": len(completed_rows),
        "planned_run_count": len(candidate_rows),
    }


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "planned_candidates": _build_anchor_candidates(args)
                    if "anchor_only" in set(args.phases)
                    else [],
                    "phases": list(args.phases),
                    "filter_overrides": second_pass_fused_filter_overrides(
                        _load_json(Path(args.lock_path).resolve())
                    ),
                    "execute_missing": bool(args.execute_missing),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    payload = run_tuning(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
