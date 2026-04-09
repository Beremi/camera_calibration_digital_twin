#!/usr/bin/env python3
"""Run the first-pass Isaac anchored VIO experiment."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

from calib_sim.isaac.app import IsaacAppBootstrapConfig, create_runtime, load_isaac_yaml
from calib_sim.isaac.logging.run_manifest import build_run_manifest
from calib_sim.reporting import generate_isaac_report_artifacts


def _default_config_paths() -> dict[str, str]:
    return {
        "scene": "config/isaac/scene/anchor_room.yaml",
        "robot": "config/isaac/robot/franka_phone_head.yaml",
        "camera": "config/isaac/camera/phone_main.yaml",
        "imu": "config/isaac/imu/phone_nominal.yaml",
        "actuation": "config/isaac/actuation/servo_nominal.yaml",
        "estimation": "config/isaac/estimation/anchored_vio.yaml",
        "control": "config/isaac/control/path_tracking.yaml",
        "visibility": "config/isaac/visibility/anchor_dropout_nominal.yaml",
        "media": "config/isaac/media/hero_capture.yaml",
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
    parser.add_argument("--visibility-config", default=None)
    parser.add_argument("--media-config", default=None)
    parser.add_argument("--output-root", default="output/isaac_runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--estimator-mode", choices=("visual", "fused"), default="fused")
    parser.add_argument("--controller-mode", choices=("open-loop", "closed-loop"), default="closed-loop")
    parser.add_argument("--mode", choices=("visual", "fused", "open-loop", "closed-loop"), default=None)
    parser.add_argument(
        "--bootstrap-control-policy",
        choices=("hold_until_first_detection", "gt_until_first_detection"),
        default="hold_until_first_detection",
    )
    parser.add_argument("--promote-latest-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and manifest wiring without starting Isaac.")
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--allow-incomplete-report", action="store_true")
    parser.add_argument("--allow-gt-debug-control", action="store_true")
    parser.add_argument("--promote-global-latest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-aux-tags-in-filter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-aux-tags-in-smoother", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-aux-map-for-control", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--smoother-backend", choices=("lightweight", "windowed_ba"), default=None)
    parser.add_argument("--vision-covariance-scale", type=float, default=None)
    parser.add_argument("--anchor-vision-covariance-scale", type=float, default=None)
    parser.add_argument("--aux-vision-covariance-scale", type=float, default=None)
    parser.add_argument("--imu-process-covariance-scale", type=float, default=None)
    parser.add_argument("--gyro-process-covariance-scale", type=float, default=None)
    parser.add_argument("--accel-process-covariance-scale", type=float, default=None)
    parser.add_argument("--post-relocalization-covariance-scale", type=float, default=None)
    parser.add_argument(
        "--allow-anchor-reacquisition-after-first-lock",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--disable-imu-prediction-while-anchor-suppressed",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--dropout-post-reacquisition-covariance-scale", type=float, default=None)
    parser.add_argument("--dropout-max-reacquisition-position-correction-m", type=float, default=None)
    parser.add_argument("--dropout-max-reacquisition-rotation-correction-deg", type=float, default=None)
    parser.add_argument("--dropout-max-reacquisition-velocity-correction-mps", type=float, default=None)
    parser.add_argument("--anchor-reacquisition-max-innovation-norm", type=float, default=None)
    parser.add_argument("--anchor-reacquisition-max-nis", type=float, default=None)
    parser.add_argument("--suppression-imu-specific-force-gate-mps2", type=float, default=None)
    parser.add_argument(
        "--suppression-propagation-mode",
        choices=("full_imu", "gyro_only", "constant_velocity", "freeze"),
        default=None,
    )
    return parser.parse_args()


def _resolve_mode_args(args: argparse.Namespace) -> tuple[str, str]:
    if args.mode is None:
        return str(args.estimator_mode), str(args.controller_mode)
    legacy_mapping = {
        "visual": ("visual", "closed-loop"),
        "fused": ("fused", "closed-loop"),
        "open-loop": ("fused", "open-loop"),
        "closed-loop": ("fused", "closed-loop"),
    }
    return legacy_mapping[str(args.mode)]


def _isaac_version() -> str:
    try:
        return importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable_in_current_env"


def _apply_second_pass_overrides(config_payloads: dict[str, dict[str, object]], args: argparse.Namespace) -> None:
    estimation = config_payloads.setdefault("estimation", {})
    filter_config = estimation.setdefault("filter", {})
    smoother_config = estimation.setdefault("smoother", {})
    control_config = config_payloads.setdefault("control", {})
    if args.use_aux_tags_in_filter is not None:
        filter_config["use_aux_tags_in_filter"] = bool(args.use_aux_tags_in_filter)
    if args.use_aux_tags_in_smoother is not None:
        smoother_config["use_aux_tags_in_smoother"] = bool(args.use_aux_tags_in_smoother)
    if args.use_aux_map_for_control is not None:
        control_config["use_aux_map_for_control"] = bool(args.use_aux_map_for_control)
    if args.smoother_backend is not None:
        smoother_config["backend"] = str(args.smoother_backend)
    if args.vision_covariance_scale is not None:
        filter_config["vision_covariance_scale"] = float(args.vision_covariance_scale)
    if args.anchor_vision_covariance_scale is not None:
        filter_config["anchor_vision_covariance_scale"] = float(args.anchor_vision_covariance_scale)
    if args.aux_vision_covariance_scale is not None:
        filter_config["aux_vision_covariance_scale"] = float(args.aux_vision_covariance_scale)
    if args.imu_process_covariance_scale is not None:
        filter_config["imu_process_covariance_scale"] = float(args.imu_process_covariance_scale)
    if args.gyro_process_covariance_scale is not None:
        filter_config["gyro_process_covariance_scale"] = float(args.gyro_process_covariance_scale)
    if args.accel_process_covariance_scale is not None:
        filter_config["accel_process_covariance_scale"] = float(args.accel_process_covariance_scale)
    if args.post_relocalization_covariance_scale is not None:
        filter_config["post_relocalization_covariance_scale"] = float(args.post_relocalization_covariance_scale)
    if args.allow_anchor_reacquisition_after_first_lock is not None:
        filter_config["allow_anchor_reacquisition_after_first_lock"] = bool(
            args.allow_anchor_reacquisition_after_first_lock
        )
    if args.disable_imu_prediction_while_anchor_suppressed is not None:
        filter_config["disable_imu_prediction_while_anchor_suppressed"] = bool(
            args.disable_imu_prediction_while_anchor_suppressed
        )
    if args.dropout_post_reacquisition_covariance_scale is not None:
        filter_config["dropout_post_reacquisition_covariance_scale"] = float(
            args.dropout_post_reacquisition_covariance_scale
        )
    if args.dropout_max_reacquisition_position_correction_m is not None:
        filter_config["dropout_max_reacquisition_position_correction_m"] = float(
            args.dropout_max_reacquisition_position_correction_m
        )
    if args.dropout_max_reacquisition_rotation_correction_deg is not None:
        filter_config["dropout_max_reacquisition_rotation_correction_deg"] = float(
            args.dropout_max_reacquisition_rotation_correction_deg
        )
    if args.dropout_max_reacquisition_velocity_correction_mps is not None:
        filter_config["dropout_max_reacquisition_velocity_correction_mps"] = float(
            args.dropout_max_reacquisition_velocity_correction_mps
        )
    if args.anchor_reacquisition_max_innovation_norm is not None:
        filter_config["anchor_reacquisition_max_innovation_norm"] = float(
            args.anchor_reacquisition_max_innovation_norm
        )
    if args.anchor_reacquisition_max_nis is not None:
        filter_config["anchor_reacquisition_max_nis"] = float(args.anchor_reacquisition_max_nis)
    if args.suppression_imu_specific_force_gate_mps2 is not None:
        filter_config["suppression_imu_specific_force_gate_mps2"] = float(
            args.suppression_imu_specific_force_gate_mps2
        )
    if args.suppression_propagation_mode is not None:
        filter_config["suppression_propagation_mode"] = str(args.suppression_propagation_mode)


def _inject_optional_config(
    runtime: object,
    *,
    name: str,
    path: str | None,
) -> None:
    if path in (None, ""):
        return
    payload = load_isaac_yaml(path)
    runtime.config.config_payloads[str(name)] = payload
    runtime.config.config_paths[str(name)] = str(path)


def main() -> int:
    args = parse_args()
    estimator_mode, controller_mode = _resolve_mode_args(args)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    bootstrap = IsaacAppBootstrapConfig(
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
        duration_s=float(args.duration_s),
        max_steps=None if args.max_steps is None else int(args.max_steps),
        estimator_mode=estimator_mode,
        controller_mode=controller_mode,
        bootstrap_control_policy=str(args.bootstrap_control_policy),
        mode=args.mode,
        promote_latest_complete=bool(args.promote_latest_complete),
        allow_gt_debug_control=bool(args.allow_gt_debug_control),
    )
    runtime = create_runtime(bootstrap)
    _inject_optional_config(runtime, name="visibility", path=args.visibility_config)
    _inject_optional_config(runtime, name="media", path=args.media_config)
    _apply_second_pass_overrides(runtime.config.config_payloads, args)
    writer = runtime.writer
    for name, payload in runtime.config.config_payloads.items():
        writer.write_config_snapshot(name, payload)
    manifest = build_run_manifest(
        repo_root=Path(__file__).resolve().parents[1],
        run_id=run_id,
        isaac_sim_version=_isaac_version(),
        stage_usd_path=runtime.config.stage_path if Path(runtime.config.stage_path).exists() else "programmatic:anchor_room",
        robot_preset=runtime.config.robot_preset,
        anchor_tag_id=runtime.config.anchor_tag_id,
        estimator_mode=runtime.config.estimator_mode,
        controller_mode=runtime.config.controller_mode,
        bootstrap_control_policy=runtime.config.bootstrap_control_policy,
        noise_presets={
            "imu": str(runtime.config.config_payloads["imu"].get("noise_preset", Path(args.imu_config).stem)),
            "actuation": str(runtime.config.config_payloads["actuation"].get("name", Path(args.actuation_config).stem)),
        },
        random_seed=int(args.seed),
        controller_config=copy.deepcopy(runtime.config.config_payloads["control"]),
        estimator_config=copy.deepcopy(runtime.config.config_payloads["estimation"]),
        ros2_bridge_used=False,
    )
    writer.write_manifest(manifest)
    summary_payload = {
        "run_dir": str(run_dir.resolve()),
        "runtime": runtime.config.summary(),
        "manifest": manifest.summary(),
        "config_snapshots": runtime.config.config_payloads,
    }
    if args.dry_run or args.validate_config_only:
        print(json.dumps(summary_payload, indent=2, sort_keys=True))
        return 0
    try:
        runtime.start()
        summary = runtime.run_until_done()
        report_payload = generate_isaac_report_artifacts(
            run_dir,
            allow_incomplete=bool(args.allow_incomplete_report),
            promote_links=bool(args.promote_global_latest),
        )
        summary.latest_any_promoted = bool(report_payload.get("latest_any_link"))
        summary.latest_complete_promoted = bool(report_payload.get("latest_complete_link"))
        summary.complete = bool(report_payload.get("complete", False))
        runtime.persist_summary(summary)
        summary_payload["summary"] = summary.as_json()
        summary_payload["report"] = {
            "complete_for_publication": bool(report_payload.get("complete", False)),
            "promote_links": bool(report_payload.get("promote_links", False)),
            "latest_any_link": report_payload.get("latest_any_link"),
            "latest_complete_link": report_payload.get("latest_complete_link"),
        }
        if not report_payload.get("complete", False) and not bool(args.allow_incomplete_report):
            print(json.dumps(summary_payload, indent=2, sort_keys=True))
            return 3
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        summary_payload["summary"] = runtime.summary().as_json() if runtime.started else summary_payload.get("summary", {})
        print(json.dumps(summary_payload, indent=2, sort_keys=True))
        return 3
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        print(json.dumps(summary_payload, indent=2, sort_keys=True))
        return 2
    finally:
        runtime.shutdown()
    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
