#!/usr/bin/env python3
"""Execute the first-pass Isaac benchmark suite and aggregate paper artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from calib_sim.reporting import generate_isaac_report_artifacts, is_complete_run
from calib_sim.reporting.isaac_first_pass_suite import generate_first_pass_suite_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output/isaac_runs")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--scene-config", default="config/isaac/scene/anchor_room.yaml")
    parser.add_argument("--robot-config", default="config/isaac/robot/franka_phone_head.yaml")
    parser.add_argument("--camera-config", default="config/isaac/camera/phone_main.yaml")
    parser.add_argument("--imu-config", default="config/isaac/imu/phone_nominal.yaml")
    parser.add_argument("--estimation-config", default="config/isaac/estimation/anchored_vio.yaml")
    parser.add_argument("--control-config", default="config/isaac/control/path_tracking.yaml")
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--matrix-seed", type=int, default=7)
    parser.add_argument("--reproducibility-seeds", nargs="+", type=int, default=[11, 17, 23, 31, 47])
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run_id(estimator_mode: str, controller_mode: str, actuation_name: str, seed: int) -> str:
    return f"first_pass_{estimator_mode}_{controller_mode}_{actuation_name}_seed_{seed:03d}"


def _actuation_config_path(actuation_name: str) -> str:
    return {
        "none": "config/isaac/actuation/none.yaml",
        "servo_nominal": "config/isaac/actuation/servo_nominal.yaml",
    }[actuation_name]


def _suite_specs(matrix_seed: int, reproducibility_seeds: list[int]) -> list[tuple[str, str, str, int]]:
    matrix = [
        ("visual", "open-loop", "servo_nominal", matrix_seed),
        ("visual", "closed-loop", "servo_nominal", matrix_seed),
        ("fused", "open-loop", "servo_nominal", matrix_seed),
        ("fused", "closed-loop", "servo_nominal", matrix_seed),
    ]
    reproducibility = [
        (estimator_mode, "closed-loop", "servo_nominal", int(seed))
        for estimator_mode in ("visual", "fused")
        for seed in reproducibility_seeds
    ]
    actuation = [
        ("fused", "closed-loop", "none", matrix_seed),
        ("fused", "closed-loop", "servo_nominal", matrix_seed),
    ]
    return list(dict.fromkeys(matrix + reproducibility + actuation))


def _run_complete(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    try:
        payload = generate_isaac_report_artifacts(run_dir, allow_incomplete=True)
    except Exception:
        return False
    return bool(payload.get("complete", False) and is_complete_run(payload["metrics"], figure_count=payload.get("generated_figure_count")))


def _execute_run(args: argparse.Namespace, *, estimator_mode: str, controller_mode: str, actuation_name: str, seed: int) -> dict[str, object]:
    run_id = _run_id(estimator_mode, controller_mode, actuation_name, seed)
    run_dir = Path(args.output_root) / run_id
    if _run_complete(run_dir):
        return {"run_id": run_id, "status": "skipped_complete", "run_dir": str(run_dir)}
    if run_dir.exists():
        raise RuntimeError(
            f"Run directory {run_dir} already exists but is not complete. "
            "Move it away or delete it before rerunning this exact suite cell."
        )
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
        _actuation_config_path(actuation_name),
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
        controller_mode,
        "--bootstrap-control-policy",
        "hold_until_first_detection",
        "--promote-latest-complete",
    ]
    if args.headless:
        command.append("--headless")
    subprocess.run(command, check=True)
    return {"run_id": run_id, "status": "executed", "run_dir": str(run_dir)}


def main() -> int:
    args = parse_args()
    specs = _suite_specs(args.matrix_seed, args.reproducibility_seeds)
    if args.dry_run:
        payload = {
            "planned_runs": [
                {
                    "run_id": _run_id(estimator_mode, controller_mode, actuation_name, seed),
                    "estimator_mode": estimator_mode,
                    "controller_mode": controller_mode,
                    "actuation": actuation_name,
                    "seed": int(seed),
                }
                for estimator_mode, controller_mode, actuation_name, seed in specs
            ]
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    run_results = [
        _execute_run(
            args,
            estimator_mode=estimator_mode,
            controller_mode=controller_mode,
            actuation_name=actuation_name,
            seed=seed,
        )
        for estimator_mode, controller_mode, actuation_name, seed in specs
    ]
    suite_payload = generate_first_pass_suite_artifacts(
        args.output_root,
        matrix_seed=int(args.matrix_seed),
        reproducibility_seeds=tuple(int(seed) for seed in args.reproducibility_seeds),
    )
    print(json.dumps({"runs": run_results, "suite": suite_payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
