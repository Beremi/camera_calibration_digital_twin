"""Suite-level aggregation checks for the first-pass Isaac benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from calib_sim.reporting.isaac_first_pass_suite import generate_first_pass_suite_artifacts


def _run_id(estimator_mode: str, controller_mode: str, actuation_name: str, seed: int) -> str:
    return f"first_pass_{estimator_mode}_{controller_mode}_{actuation_name}_seed_{seed:03d}"


def _metrics_payload(run_id: str, *, estimator_mode: str, controller_mode: str, actuation_name: str, seed: int) -> dict:
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
            "map_error_m": 0.003,
        },
        "estimation": {
            "anchor_visible_fraction": 0.95,
            "anchor_relocalization_count": 1.0,
            "mean_anchor_innovation_norm": 0.03,
        },
        "control": {
            "mean_waypoint_error_m": mean_waypoint_error,
            "completion_fraction": completion_fraction,
            "ik_failure_fraction": ik_failure_fraction,
            "dominant_safety_reason": "nominal",
            "mean_actuator_tracking_error": 0.004 if actuation_name == "servo_nominal" else 0.0,
        },
    }


def test_first_pass_suite_generation_writes_tables_and_macros(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "isaac_runs"
    seeds = (11, 17, 23, 31, 47)
    specs = [
        ("visual", "open-loop", "servo_nominal", 7),
        ("visual", "closed-loop", "servo_nominal", 7),
        ("fused", "open-loop", "servo_nominal", 7),
        ("fused", "closed-loop", "servo_nominal", 7),
        *[(estimator_mode, "closed-loop", "servo_nominal", seed) for estimator_mode in ("visual", "fused") for seed in seeds],
        ("fused", "closed-loop", "none", 7),
    ]
    for estimator_mode, controller_mode, actuation_name, seed in specs:
        run_id = _run_id(estimator_mode, controller_mode, actuation_name, seed)
        run_dir = output_root / run_id
        report_dir = run_dir / "analysis" / "report_data"
        report_dir.mkdir(parents=True, exist_ok=True)
        metrics = _metrics_payload(
            run_id,
            estimator_mode=estimator_mode,
            controller_mode=controller_mode,
            actuation_name=actuation_name,
            seed=seed,
        )
        (run_dir / "analysis" / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
            (report_dir / figure_name).write_bytes(b"placeholder")
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "controller_diagnostics.csv").write_text(
            "timestamp_s,ik_success,safety_reason\n0.0,True,nominal\n0.1,False,ik_hold\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "calib_sim.reporting.isaac_first_pass_suite.generate_isaac_report_artifacts",
        lambda run_dir, allow_incomplete=False: {"metrics": json.loads((Path(run_dir) / "analysis" / "metrics.json").read_text(encoding="utf-8")), "complete": True},
    )

    payload = generate_first_pass_suite_artifacts(output_root)

    suite_dir = output_root / "latest_first_pass_suite"
    assert payload["canonical_run_id"] == _run_id("fused", "closed-loop", "servo_nominal", 7)
    assert (suite_dir / "analysis" / "suite_summary.json").exists()
    tex = (suite_dir / "analysis" / "report_data" / "paper_artifacts.tex").read_text(encoding="utf-8")
    assert "IsaacFirstPassMatrixRows" in tex
    assert "IsaacFirstPassReproducibilityRows" in tex
    assert r"servo\_nominal" in tex
    assert r"\ArtifactPending{}" not in tex
    assert (suite_dir / "analysis" / "first_pass_matrix_table.csv").exists()
    assert (suite_dir / "analysis" / "report_data" / "ik_failure_timeline.png").exists()
