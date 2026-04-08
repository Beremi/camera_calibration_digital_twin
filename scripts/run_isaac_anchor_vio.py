#!/usr/bin/env python3
"""Run the first-pass Isaac anchored VIO experiment."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--mode", choices=("visual", "fused", "open-loop", "closed-loop"), default="closed-loop")
    parser.add_argument("--promote-latest-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and manifest wiring without starting Isaac.")
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--allow-gt-debug-control", action="store_true")
    return parser.parse_args()


def _isaac_version() -> str:
    try:
        return importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable_in_current_env"


def main() -> int:
    args = parse_args()
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
        mode=str(args.mode),
        promote_latest_complete=bool(args.promote_latest_complete),
        allow_gt_debug_control=bool(args.allow_gt_debug_control),
    )
    runtime = create_runtime(bootstrap)
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
        noise_presets={
            "imu": str(runtime.config.config_payloads["imu"].get("noise_preset", Path(args.imu_config).stem)),
            "actuation": str(runtime.config.config_payloads["actuation"].get("name", Path(args.actuation_config).stem)),
        },
        random_seed=int(args.seed),
        controller_config=load_isaac_yaml(args.control_config),
        estimator_config=load_isaac_yaml(args.estimation_config),
        ros2_bridge_used=False,
    )
    writer.write_manifest(manifest)
    summary_payload = {
        "run_dir": str(run_dir.resolve()),
        "runtime": runtime.config.summary(),
        "manifest": manifest.summary(),
    }
    if args.dry_run or args.validate_config_only:
        print(json.dumps(summary_payload, indent=2, sort_keys=True))
        return 0
    try:
        runtime.start()
        summary = runtime.run_until_done()
        report_payload = generate_isaac_report_artifacts(run_dir, allow_incomplete=True)
        summary.latest_any_promoted = bool(report_payload.get("latest_any_link"))
        summary.latest_complete_promoted = bool(report_payload.get("latest_complete_link"))
        runtime.persist_summary(summary)
        summary_payload["summary"] = summary.as_json()
        summary_payload["report"] = {
            "complete_for_publication": bool(report_payload.get("complete", False)),
            "latest_any_link": report_payload.get("latest_any_link"),
            "latest_complete_link": report_payload.get("latest_complete_link"),
        }
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
